"""Proceso de captura: la UNICA lectura de la fuente, publicada en el ring compartido.

Con un video, lo reproduce a ritmo de reloj (`pace_fps`, por defecto los FPS nativos) para que se
comporte como una camara: los frames llegan cuando llegan, no cuando el modelo los pide. Con una
camara (webcam o iPhone), el ritmo lo pone el dispositivo.
"""
from __future__ import annotations

import queue
import sys
import time

MESSAGE_EVERY_S = 0.5
CAMERA_READ_RETRIES = 30
PHONE_NOMINAL_FPS = 20.0   # objetivo de envio de static/phone.js; fija el ritmo del MP4 de salida
PHONE_WAIT_S = 30.0        # espera al primer frame del celular
PHONE_TIMEOUT_S = 10.0     # sin frames por este tiempo = el celular se fue


def watch_parent(stop_event, period_s: float = 1.0):
    """Si el servidor muere de golpe (sin pasar por su lifespan), los hijos `spawn` NO mueren
    con el en Windows. Este hilo activa la parada ordenada en cuanto el padre desaparece.

    Devuelve un Event `orphaned`: si esta activo, el hijo debe llamar a `release_queues` antes
    de salir; si no, se quedaria colgado para siempre vaciando sus colas hacia nadie.
    """
    import multiprocessing as mp
    import threading

    orphaned = threading.Event()
    parent = mp.parent_process()
    if parent is None:
        return orphaned

    def run():
        while True:
            # tras una parada normal se sigue vigilando: el padre puede morir mientras el hijo
            # todavia escribe sus resultados
            time.sleep(period_s)
            if not parent.is_alive():
                orphaned.set()
                stop_event.set()
                return

    threading.Thread(target=run, daemon=True).start()
    return orphaned


def release_queues(orphaned, *queues) -> None:
    """Al salir, un proceso hijo espera a que sus mp.Queue terminen de escribir en el pipe. Sin
    lector (el padre murio) ese pipe se llena y la espera no termina nunca. Solo si quedamos
    huerfanos se renuncia a esa espera; en una salida normal no, o se perderia el mensaje `done`."""
    import multiprocessing as mp

    parent = mp.parent_process()
    if orphaned.is_set() or (parent is not None and not parent.is_alive()):
        for q in queues:
            if q is not None:
                q.cancel_join_thread()


def _fit(frame, cv2):
    from webapp.shared_frames import MAX_H, MAX_W

    h, w = frame.shape[:2]
    if h <= MAX_H and w <= MAX_W:
        return frame
    scale = min(MAX_H / h, MAX_W / w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _letterbox(frame, width, height, cv2):
    """Encaja el frame en (width, height) sin deformarlo. Si el celular rota a mitad de sesion,
    el tamano cambia; el video de salida (ffmpeg) y los workers esperan un tamano fijo."""
    import numpy as np

    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _capture_phone(ring, channel, stop_event, go_event, phone_q, cv2) -> int:
    """Fuente "iPhone": los JPEG llegan del relay HTTPS (webapp/phone.py) por `phone_q` como
    (t_llegada_ns, jpeg). El ritmo lo pone el celular; aqui solo se decodifica y se publica.

    La latencia E2E se mide desde la LLEGADA al PC: el reloj del iPhone no esta sincronizado con
    el del PC, asi que la red y la codificacion en el telefono quedan fuera de la medicion.
    """
    import numpy as np

    def decode(jpeg):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return None if frame is None else _fit(frame, cv2)

    t0, first = time.monotonic(), None
    while first is None:
        if stop_event.is_set():
            return 0
        if time.monotonic() - t0 > PHONE_WAIT_S:
            raise RuntimeError("El celular no esta transmitiendo. Escanea el QR, pulsa Empezar en el "
                               "iPhone y vuelve a ejecutar.")
        try:
            _, jpeg = phone_q.get(timeout=0.5)
        except queue.Empty:
            continue
        first = decode(jpeg)

    h, w = first.shape[:2]
    ring.set_ready(PHONE_NOMINAL_FPS, -1, w, h)
    channel.put({"type": "status", "proc": "capture", "state": "running",
                 "message": f"iPhone conectado: {w}x{h}", "width": w, "height": h,
                 "fps": PHONE_NOMINAL_FPS, "total": 0})
    while not go_event.wait(0.1):  # mientras cargan los modelos, se descartan los frames viejos
        if stop_event.is_set():
            return 0
        while True:
            try:
                phone_q.get_nowait()
            except queue.Empty:
                break

    frame_idx, t_msg, n_since, t_last = 0, time.perf_counter(), 0, time.monotonic()
    while not stop_event.is_set():
        try:
            t_ns, jpeg = phone_q.get(timeout=0.5)
        except queue.Empty:
            if time.monotonic() - t_last > PHONE_TIMEOUT_S:
                raise RuntimeError("El celular dejo de transmitir (se bloqueo la pantalla, se cerro "
                                   "Safari o salio de la red).")
            continue
        frame = decode(jpeg)
        if frame is None:
            continue
        t_last = time.monotonic()
        ring.publish(_letterbox(frame, w, h, cv2), frame_idx, t_ns)
        frame_idx += 1
        n_since += 1
        now = time.perf_counter()
        if now - t_msg >= MESSAGE_EVERY_S:
            channel.put({"type": "capture", "proc": "capture", "frame_idx": frame_idx, "loop": 1,
                         "fps": n_since / (now - t_msg)})
            t_msg, n_since = now, 0
    return frame_idx


def capture_main(cfg: dict, shm_name: str, channel, stop_event, go_event, phone_q=None) -> None:
    """Punto de entrada del subproceso (importable a nivel de modulo, requisito de spawn).

    Abre la fuente y la anuncia (los workers ya pueden leer fps/tamano), pero no publica frames
    hasta `go_event`: el servidor lo activa cuando los modelos terminaron de cargar, para que
    ninguno se pierda el arranque del video mientras inicializa CUDA o MediaPipe.

    Fuentes: "video" (archivo, a ritmo de reloj), "camera" (webcam del PC) y "phone" (iPhone por
    la red local; los frames llegan por `phone_q` desde el relay del servidor).
    """
    import cv2

    from webapp.shared_frames import STATE_EOF, STATE_ERROR, FrameRing

    orphaned = watch_parent(stop_event)
    ring = FrameRing.attach(shm_name)
    cap = None
    try:
        source = cfg["source"]
        if source["kind"] == "phone":
            frames = _capture_phone(ring, channel, stop_event, go_event, phone_q, cv2)
            ring.set_state(STATE_EOF)
            channel.put({"type": "done", "proc": "capture", "frames": frames})
            return
        is_camera = source["kind"] == "camera"
        if is_camera:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(source["index"]), backend)
        else:
            cap = cv2.VideoCapture(source["path"])
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source.get('path', source.get('index'))}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not native_fps or native_fps != native_fps or native_fps <= 0:  # 0 o NaN
            native_fps = 30.0
        frame_count = -1 if is_camera else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        loops = 0 if is_camera else int(cfg.get("loops", 1))  # 0 = sin fin
        pace_fps = None if is_camera else float(cfg.get("pace_fps") or native_fps)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("La fuente no entrego ningun frame.")
        frame = _fit(frame, cv2)
        h, w = frame.shape[:2]
        ring.set_ready(pace_fps or native_fps, frame_count, w, h)
        channel.put({"type": "status", "proc": "capture", "state": "running",
                     "message": f"Fuente abierta: {w}x{h} a {pace_fps or native_fps:.1f} FPS",
                     "width": w, "height": h, "fps": pace_fps or native_fps,
                     "total": frame_count * loops if loops > 0 else 0})
        while not go_event.wait(0.1):
            if stop_event.is_set():
                break

        frame_idx, loop_i, misses = 0, 0, 0
        period = 1.0 / pace_fps if pace_fps else 0.0
        next_t = time.perf_counter()
        t_msg, n_since = time.perf_counter(), 0
        while not stop_event.is_set():
            ring.publish(frame, frame_idx, time.perf_counter_ns())
            frame_idx += 1
            n_since += 1

            now = time.perf_counter()
            if now - t_msg >= MESSAGE_EVERY_S:
                channel.put({"type": "capture", "proc": "capture", "frame_idx": frame_idx,
                             "loop": loop_i + 1, "fps": n_since / (now - t_msg)})
                t_msg, n_since = now, 0

            if period:  # ritmo de reloj: si vamos atrasados, no se acumula deuda
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    misses += 1
                    if misses > CAMERA_READ_RETRIES:
                        raise RuntimeError("La camara dejo de entregar frames.")
                    time.sleep(0.01)
                    continue
                loop_i += 1
                if loops and loop_i >= loops:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            misses = 0
            frame = _fit(frame, cv2)

        ring.set_state(STATE_EOF)
        channel.put({"type": "done", "proc": "capture", "frames": frame_idx})
    except BaseException as e:  # noqa: BLE001 - cualquier fallo se reporta a la UI
        import traceback

        ring.set_state(STATE_ERROR)
        channel.put({"type": "error", "proc": "capture",
                     "error": {"code": type(e).__name__, "user_message": str(e) or "La captura fallo.",
                               "technical_detail": traceback.format_exc()}})
    finally:
        if cap is not None:
            cap.release()
        ring.close()
        channel.put({"type": "eof", "proc": "capture"})
        release_queues(orphaned, channel, phone_q)
