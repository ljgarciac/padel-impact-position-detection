"""Pipeline de vision en tiempo real: version sincronica (baseline) vs. version concurrente
(threading + queue) sobre el mismo detector YOLO-pose + ByteTrack de run_inference_video.py.

Arquitectura concurrente (modo "threaded"):

    [hilo captura] --frame--> [queue A, maxsize=4] --> [hilo inferencia] --resultado--> [queue B, maxsize=4] --> [hilo principal: render+display+write]

Un solo hilo de inferencia (no un pool): el estado de tracking de ByteTrack vive dentro del objeto
`model` (persist=True), no es thread-local, asi que llamar a model.track() desde mas de un hilo a la
vez lo correlacionaria. Restringir todas las llamadas a .track() a un unico hilo dedicado es lo que
elimina la condicion de carrera y preserva el orden temporal de las predicciones.

cv2.VideoCapture.read() (decode en C) y los kernels CUDA que dispara model.track() liberan el GIL
durante la llamada bloqueante, por eso captura e inferencia se solapan en tiempo real de reloj, no solo
se intercalan por el GIL. El contexto CUDA de PyTorch es por-proceso, no por-hilo, asi que no hay
problema de afinidad al llamar siempre desde el mismo hilo de inferencia.

El detector es intercambiable (`detector=`): por defecto el YOLO-pose de 5 keypoints, y
`mediapipe_detector.MediaPipePoseDetector` como referencia. La app web de dos pantallas (webapp/)
reutiliza run_sync/run_threaded con esos ganchos opcionales (detector, cancel_token, on_frame);
sin pasarlos, el comportamiento es exactamente el del benchmark del notebook.

Uso:
    python realtime_pipeline.py --mode compare --loops 15
    python realtime_pipeline.py --mode threaded --detector mediapipe
"""
import argparse
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import psutil

from run_inference_video import KPT_NAMES, SKELETON, TRACK_COLORS, load_weights_path

ROOT = Path(__file__).resolve().parent
WARMUP_LOOPS = 1  # se descarta el primer loop completo de cada corrida de las estadisticas

# Fuente de frames: video local (con loop) o camara en vivo, mismo objeto
class FrameSource:
    def __init__(self, video_path=None, camera_index=None, loops=1, sim_capture_delay_s=0.0,
                 display=False):
        # sim_capture_delay_s: retardo sintetico tras cada read(), para experimentos controlados
        # que simulan una fuente de captura mas lenta que un mp4 local (p.ej. el frame-grab + la
        # conversion de color inicial de una camara en vivo real, tipicamente 15-30ms). 0.0 = sin
        # simular nada (comportamiento real del archivo local).
        self.sim_capture_delay_s = sim_capture_delay_s
        self._display = display  # solo con ventana tiene sentido leer el teclado ('q')
        if camera_index is not None:
            # en Windows, DirectShow abre la camara en ~1 s; el backend por defecto (MSMF) tarda mucho mas
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera_index, backend)
            self.loops = 1  # una camara en vivo no "repite"; es un stream continuo
            self._is_camera = True
        else:
            self.cap = cv2.VideoCapture(str(video_path))
            self.loops = loops
            self._is_camera = False
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir la fuente de video: {video_path or camera_index}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and not math.isnan(fps) and fps > 0 else (30.0 if self._is_camera else 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        frame_idx = 0
        for loop_i in range(self.loops):
            if loop_i > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                if self.sim_capture_delay_s > 0:
                    time.sleep(self.sim_capture_delay_s)
                yield frame_idx, frame
                frame_idx += 1
                if self._is_camera and self._display and cv2.waitKey(1) & 0xFF == ord("q"):
                    return

    def release(self):
        self.cap.release()

# Muestreo de CPU/GPU en segundo plano (no bloquea el pipeline principal)
class ResourceSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []  # lista de (t, cpu_pct, gpu_pct, gpu_mem_mb)
        self._stop = threading.Event()
        self._thread = None
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None

    def _run(self, t0):
        psutil.cpu_percent(interval=None)  # primera llamada "descartable" (referencia)
        while not self._stop.is_set():
            cpu_pct = psutil.cpu_percent(interval=None)
            gpu_pct, gpu_mem_mb = None, None
            if self._nvml_handle is not None:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_pct, gpu_mem_mb = util.gpu, mem.used / 1024 / 1024
            self.samples.append((time.perf_counter() - t0, cpu_pct, gpu_pct, gpu_mem_mb))
            time.sleep(self.interval)

    def start(self):
        t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, args=(t0,), daemon=True)
        self._thread.start()
        return t0

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

# Modelo y dibujo (reutiliza KPT_NAMES/SKELETON/TRACK_COLORS de run_inference_video.py)
def load_model(weights_path=None):
    from ultralytics import YOLO
    return YOLO(str(load_weights_path(weights_path)))


# Subconjuntos de keypoints a dibujar (indices de KPT_NAMES: hombro, codo, muneca, raqueta, cabeza)
KPT_SUBSETS = {
    "brazo": (0, 1, 2),                 # hombro - codo - muneca
    "brazo_raqueta": (0, 1, 2, 3),      # + raqueta
    "todos": (0, 1, 2, 3, 4),
}


def draw_annotations(frame, result, kpt_conf_min=0.3, show=None):
    """show: indices de keypoints a dibujar (None = los 5, el comportamiento de siempre)."""
    show = set(range(len(KPT_NAMES))) if show is None else set(show)
    skeleton = [(a, b) for a, b in SKELETON if a in show and b in show]
    n_detections = 0
    if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_conf = (result.keypoints.conf.cpu().numpy()
                     if result.keypoints.conf is not None else None)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = (result.boxes.id.int().cpu().tolist()
                     if result.boxes.id is not None else [None] * len(boxes_xyxy))
        n_detections = len(boxes_xyxy)

        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            track_id = track_ids[i]
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)] if track_id is not None else (200, 200, 200)
            pts = kpts_xy[i]
            pconf = kpts_conf[i] if kpts_conf is not None else [1.0] * len(pts)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"jugador {track_id}" if track_id is not None else "sin ID"
            cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            for a, b in skeleton:
                if pconf[a] > kpt_conf_min and pconf[b] > kpt_conf_min:
                    cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2)
            for j, (px, py) in enumerate(pts):
                if j in show and pconf[j] > kpt_conf_min:
                    cv2.circle(frame, (int(px), int(py)), 3, color, -1)
    return frame, n_detections


def extract_keypoints_summary(result):
    """Vector plano de keypoints (para el chequeo de paridad sync vs threaded). None si no hay detecciones."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    return result.keypoints.xy.cpu().numpy().round(2).flatten().tolist()


# ---------------------------------------------------------------------------
# Detectores intercambiables: misma interfaz para nuestro YOLO-pose y para MediaPipe
#   infer(frame) -> result | draw(frame, result, kpt_conf_min) -> (frame, n_det)
#   keypoints_summary(result) -> list|None | mean_kpt_conf(result) -> float | close()
# ---------------------------------------------------------------------------
def _torch_device_label():
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu"


class YoloPoseDetector:
    """Envuelve el model.track(persist=True) + draw_annotations de siempre, sin cambiar logica."""
    name = "yolo"

    def __init__(self, model, imgsz=384, conf=0.25, tracker="bytetrack.yaml", keypoints="todos"):
        self.model = model
        self.imgsz, self.conf, self.tracker = imgsz, conf, tracker
        self.show = KPT_SUBSETS[keypoints]  # solo cambia lo que se dibuja y se promedia, no la inferencia
        self.device = _torch_device_label()

    def infer(self, frame):
        return self.model.track(frame, imgsz=self.imgsz, conf=self.conf, persist=True,
                                tracker=self.tracker, verbose=False)[0]

    def draw(self, frame, result, kpt_conf_min=0.3):
        return draw_annotations(frame, result, kpt_conf_min, show=self.show)

    def keypoints_summary(self, result):
        return extract_keypoints_summary(result)

    def mean_kpt_conf(self, result):
        """Confianza media de los keypoints que se muestran (p.ej. solo los del brazo)."""
        if result.keypoints is None or result.keypoints.conf is None or len(result.keypoints.conf) == 0:
            return 0.0
        return float(result.keypoints.conf[:, list(self.show)].mean().item())

    def warmup(self, size=(640, 640), n=3):
        """Inicializa CUDA/kernels antes de ver frames reales (lo usa la app en vivo; el benchmark
        del notebook descarta en cambio el primer loop). Usa predict y no track, para no dejar
        estado en ByteTrack."""
        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf, verbose=False)

    def close(self):
        pass


def make_detector(name="yolo", imgsz=384, conf=0.25, weights_path=None, keypoints="todos"):
    if name == "yolo":
        return YoloPoseDetector(load_model(weights_path), imgsz=imgsz, conf=conf, keypoints=keypoints)
    if name == "mediapipe":
        from mediapipe_detector import MediaPipePoseDetector  # perezoso: no carga mediapipe si no se usa
        return MediaPipePoseDetector(keypoints=keypoints)
    raise ValueError(f"Detector desconocido: {name}")


def _unpack(item):
    """Las fuentes entregan (frame_idx, frame) o (frame_idx, frame, t_captura_perf_counter).
    La segunda forma la usa la fuente compartida entre procesos de la app web, para medir la
    latencia E2E desde el instante real de captura (perf_counter es monotonico en todo el sistema)."""
    if len(item) == 3:
        return item
    return item[0], item[1], None


def _cancelled(cancel_token):
    return cancel_token is not None and cancel_token.is_set()


@dataclass
class FrameRecord:
    frame_idx: int
    t_capture_start: float
    t_capture_end: float
    t_inference_start: float
    t_inference_end: float
    t_render_end: float
    n_detections: int
    keypoints: list = field(default_factory=list)
    queue_wait: float = 0.0  # tiempo que el frame esperó en una queue antes de ser consumido (solo threaded)
    mean_kpt_conf: float = 0.0  # confianza media de los keypoints detectados en el frame


# ---------------------------------------------------------------------------
# Modo sincronico (baseline): un solo hilo, secuencial
# ---------------------------------------------------------------------------
def run_sync(model, source: FrameSource, imgsz=384, conf=0.25, tracker="bytetrack.yaml",
             display=False, writer=None, kpt_conf_min=0.3, playback_fps=None,
             detector=None, cancel_token=None, on_frame=None):
    # playback_fps: si se da, pausa cada frame el tiempo necesario para que la ventana se vea a un
    # ritmo "humano" (el modelo puede correr a 50+ FPS, demasiado rapido para percibir nada en un
    # clip de pocos segundos). None = sin pausa artificial (ritmo maximo, usado para benchmarking).
    # detector=None -> el YOLO de siempre sobre `model`; cancel_token (con .is_set()) corta el bucle
    # igual que la tecla 'q'; on_frame(frame, record) se llama DESPUES de medir t_render_end, asi que
    # lo que haga (p.ej. la vista previa en vivo) no entra en la latencia medida.
    detector = detector or YoloPoseDetector(model, imgsz=imgsz, conf=conf, tracker=tracker)
    wait_ms = max(1, int(1000 / playback_fps)) if playback_fps else 1
    records = []
    t_start = time.perf_counter()
    for item in source:
        if _cancelled(cancel_token):
            break
        frame_idx, frame, t_src = _unpack(item)
        t_cap_start = t_src if t_src is not None else time.perf_counter()
        t_cap_end = time.perf_counter()  # ya lo tenemos en mano; el costo de read() se mide aparte abajo

        t_inf_start = time.perf_counter()
        result = detector.infer(frame)
        kpts = detector.keypoints_summary(result)
        t_inf_end = time.perf_counter()
        mean_conf = detector.mean_kpt_conf(result)

        frame, n_det = detector.draw(frame, result, kpt_conf_min)
        if writer is not None:
            writer.write(frame)
        quit_requested = False
        if display:
            cv2.imshow("sync", frame)
            quit_requested = cv2.waitKey(wait_ms) & 0xFF == ord("q")
        t_render_end = time.perf_counter()

        record = FrameRecord(frame_idx, t_cap_start, t_cap_end, t_inf_start, t_inf_end,
                             t_render_end, n_det, kpts or [], mean_kpt_conf=mean_conf)
        records.append(record)
        if on_frame is not None:
            on_frame(frame, record)
        if quit_requested:
            break
    t_total = time.perf_counter() - t_start
    if display:
        cv2.destroyWindow("sync")
    return records, t_total


# Modo concurrente: captura || inferencia || render, con 2 queues acotadas
class PipelineError(Exception):
    pass


def run_threaded(model, source: FrameSource, imgsz=384, conf=0.25, tracker="bytetrack.yaml",
                  display=False, writer=None, kpt_conf_min=0.3, queue_maxsize=4, playback_fps=None,
                  detector=None, cancel_token=None, on_frame=None):
    # ver nota de playback_fps en run_sync; aqui solo pausa el hilo principal (render), no afecta
    # la seguridad de la arquitectura -- captura/inferencia siguen desacopladas en sus propios hilos.
    # detector / cancel_token / on_frame: mismos ganchos opcionales que run_sync. Todas las llamadas
    # a detector.infer() siguen ocurriendo en el unico hilo de inferencia.
    detector = detector or YoloPoseDetector(model, imgsz=imgsz, conf=conf, tracker=tracker)
    wait_ms = max(1, int(1000 / playback_fps)) if playback_fps else 1
    queue_a = queue.Queue(maxsize=queue_maxsize)  # capturado -> por inferir
    queue_b = queue.Queue(maxsize=queue_maxsize)  # inferido -> por renderizar
    stop_event = threading.Event()
    errors = []

    def capture_worker():
        try:
            for item in source:
                if stop_event.is_set() or _cancelled(cancel_token):
                    break
                frame_idx, frame, t_src = _unpack(item)
                t_cap_start = t_src if t_src is not None else time.perf_counter()
                t_cap_end = time.perf_counter()
                queue_a.put((frame_idx, frame, t_cap_start, t_cap_end, time.perf_counter()))
            queue_a.put(None)  # centinela: no hay mas frames
        except Exception as e:  # pragma: no cover - ruta de error defensiva
            errors.append(("capture", e))
            queue_a.put(None)

    def inference_worker():
        try:
            while True:
                item = queue_a.get()
                if item is None or stop_event.is_set():
                    queue_b.put(None)
                    break
                frame_idx, frame, t_cap_start, t_cap_end, t_enqueued_a = item
                queue_wait_a = time.perf_counter() - t_enqueued_a

                t_inf_start = time.perf_counter()
                result = detector.infer(frame)
                kpts = detector.keypoints_summary(result)
                t_inf_end = time.perf_counter()
                mean_conf = detector.mean_kpt_conf(result)

                queue_b.put((frame_idx, frame, result, t_cap_start, t_cap_end, t_inf_start,
                             t_inf_end, kpts, queue_wait_a, mean_conf, time.perf_counter()))
        except Exception as e:  # pragma: no cover
            errors.append(("inference", e))
            queue_b.put(None)

    cap_thread = threading.Thread(target=capture_worker, daemon=True)
    inf_thread = threading.Thread(target=inference_worker, daemon=True)

    records = []
    t_start = time.perf_counter()
    cap_thread.start()
    inf_thread.start()

    while True:
        try:
            # get con timeout (y no bloqueante para siempre) para atender una cancelacion aunque la
            # fuente no entregue frames (p.ej. una camara que se colgo)
            item = queue_b.get(timeout=0.2)
        except queue.Empty:
            if _cancelled(cancel_token):
                break
            continue
        if item is None:
            break
        (frame_idx, frame, result, t_cap_start, t_cap_end, t_inf_start, t_inf_end,
         kpts, queue_wait_a, mean_conf, t_enqueued_b) = item
        queue_wait_b = time.perf_counter() - t_enqueued_b

        frame, n_det = detector.draw(frame, result, kpt_conf_min)
        if writer is not None:
            writer.write(frame)
        quit_requested = False
        if display:
            cv2.imshow("threaded", frame)
            quit_requested = cv2.waitKey(wait_ms) & 0xFF == ord("q")
        t_render_end = time.perf_counter()

        record = FrameRecord(frame_idx, t_cap_start, t_cap_end, t_inf_start, t_inf_end,
                             t_render_end, n_det, kpts or [], queue_wait_a + queue_wait_b,
                             mean_kpt_conf=mean_conf)
        records.append(record)
        if on_frame is not None:
            on_frame(frame, record)
        if quit_requested or _cancelled(cancel_token):
            break

    stop_event.set()
    # si se salio temprano (p.ej. el usuario presiono 'q'), drenar las colas: de lo contrario un
    # hilo productor podria quedar bloqueado para siempre en un put() a una cola llena que ya nadie
    # consume, y el join() de abajo esperaria el timeout completo en vano.
    for q in (queue_a, queue_b):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
    cap_thread.join(timeout=5)
    inf_thread.join(timeout=5)
    t_total = time.perf_counter() - t_start
    if display:
        cv2.destroyWindow("threaded")

    if errors:
        stage, exc = errors[0]
        raise PipelineError(f"Fallo en el hilo de {stage}: {exc}") from exc

    records.sort(key=lambda r: r.frame_idx)
    return records, t_total

# Resumen de metricas a partir de los records de una corrida
def summarize(records, t_total, warmup_frames):
    if not records:
        return {}
    # con camara, frame_count es -1 (stream sin fin): el warm-up no puede ser negativo
    warmup_frames = max(0, warmup_frames)
    steady = [r for r in records if r.frame_idx >= warmup_frames]
    if not steady:
        # la corrida fue mas corta que el warm-up esperado (p.ej. --loops 1 con WARMUP_LOOPS=1):
        # no hay nada que descartar, se usan todos los frames disponibles en vez de fallar
        steady = records
    inf_latency_ms = np.array([(r.t_inference_end - r.t_inference_start) * 1000 for r in steady])
    e2e_latency_ms = np.array([(r.t_render_end - r.t_capture_start) * 1000 for r in steady])
    # FPS de pipeline end-to-end: frames en estado estable / tiempo de reloj real que tomo
    # procesarlos (desde el primer t_capture_start hasta el ultimo t_render_end). Usar esto en vez
    # de dividir por t_total (que incluye el loop de warmup descartado) evita inflar/deflar el FPS.
    steady_start = min(r.t_capture_start for r in steady)
    steady_end = max(r.t_render_end for r in steady)
    steady_duration_s = steady_end - steady_start
    queue_wait_ms = np.array([r.queue_wait * 1000 for r in steady])
    # estabilidad temporal: dispersion del intervalo entre frames entregados (jitter)
    render_ends = np.sort(np.array([r.t_render_end for r in steady]))
    frame_intervals_ms = np.diff(render_ends) * 1000 if len(render_ends) > 1 else np.zeros(1)
    return dict(
        n_frames_total=len(records),
        n_frames_steady=len(steady),
        inf_latency_ms_mean=float(inf_latency_ms.mean()),
        inf_latency_ms_p50=float(np.percentile(inf_latency_ms, 50)),
        inf_latency_ms_p95=float(np.percentile(inf_latency_ms, 95)),
        inf_latency_ms_p99=float(np.percentile(inf_latency_ms, 99)),
        inf_latency_ms_std=float(inf_latency_ms.std()),
        e2e_latency_ms_mean=float(e2e_latency_ms.mean()),
        e2e_latency_ms_p50=float(np.percentile(e2e_latency_ms, 50)),
        e2e_latency_ms_p95=float(np.percentile(e2e_latency_ms, 95)),
        e2e_latency_ms_std=float(e2e_latency_ms.std()),
        inference_fps=float(len(steady) / (inf_latency_ms.sum() / 1000)),
        pipeline_fps=float(len(steady) / steady_duration_s) if steady_duration_s > 0 else 0.0,
        steady_duration_s=float(steady_duration_s),
        wall_time_s_total_incl_warmup=float(t_total),
        queue_wait_ms_mean=float(queue_wait_ms.mean()),
        queue_wait_ms_p95=float(np.percentile(queue_wait_ms, 95)),
        jitter_ms=float(frame_intervals_ms.std()),
        mean_detections=float(np.mean([r.n_detections for r in steady])),
        mean_kpt_conf=float(np.mean([r.mean_kpt_conf for r in steady])),
    )


def check_parity(records_a, records_b, tol=1.0):
    by_idx_a = {r.frame_idx: r for r in records_a}
    by_idx_b = {r.frame_idx: r for r in records_b}
    common = sorted(set(by_idx_a) & set(by_idx_b))
    matches = 0
    for idx in common:
        ra, rb = by_idx_a[idx], by_idx_b[idx]
        if ra.n_detections != rb.n_detections:
            continue
        if not ra.keypoints and not rb.keypoints:
            matches += 1
            continue
        if ra.keypoints and rb.keypoints and len(ra.keypoints) == len(rb.keypoints):
            if np.allclose(ra.keypoints, rb.keypoints, atol=tol):
                matches += 1
    pct = 100.0 * matches / len(common) if common else 0.0
    return dict(n_common=len(common), n_matches=matches, pct_match=pct)


def _make_writer(path, src):
    if path is None:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, src.fps, (src.width, src.height))


def run_pipeline(mode, video=None, camera_index=None, loops=15, display=False, output=None,
                  imgsz=384, conf=0.25, sim_capture_delay_s=0.0, playback_fps=None,
                  detector_name="yolo", source=None, cancel_token=None, on_frame=None,
                  writer_factory=None, sampler=None, warmup_frames=None, detector=None):
    """Ejecuta el pipeline completo en modo 'sync' o 'threaded' y devuelve
    (records, t_total, resource_samples, warmup_frames, summary).

    Se puede usar desde la línea de comandos (`main()`) o desde el notebook, sin pasar por
    subprocess ni leer CSV.

    Carga un modelo nuevo en cada corrida, para que el historial de ByteTrack (persist=True)
    no afecte la siguiente y la comparación sync vs. threaded sea justa.

    `playback_fps` solo aplica con display=True (demo visual), no en benchmarking.

    Parámetros opcionales (los usa la app web; por defecto no cambian nada):
      detector_name   "yolo" (modelo propio de 5 keypoints) o "mediapipe" (referencia, 33 landmarks).
      source          Fuente de video ya creada (por ejemplo, compartida entre procesos); no se cierra aquí.
      cancel_token    Objeto con .is_set() para detener la corrida de forma ordenada (como la tecla 'q').
      on_frame        Callback(frame_anotado, FrameRecord), ejecutado fuera de la latencia medida.
      writer_factory  Callback(source) que devuelve un escritor con write()/release() (p. ej. H.264).
      sampler         ResourceSampler externo, para leer sus muestras durante la corrida.
      warmup_frames   Frames iniciales que se descartan de las estadísticas (por defecto, WARMUP_LOOPS loops).
      detector        Detector ya construido; tiene prioridad sobre detector_name y se cierra al terminar.
    """
    detector = detector or make_detector(detector_name, imgsz=imgsz, conf=conf)
    own_source = source is None
    if own_source:
        source = FrameSource(video, camera_index, loops, sim_capture_delay_s=sim_capture_delay_s,
                             display=display)
    writer = writer_factory(source) if writer_factory is not None else _make_writer(output, source)
    sampler = sampler or ResourceSampler()
    sampler.start()
    fn = run_sync if mode == "sync" else run_threaded
    try:
        records, t_total = fn(getattr(detector, "model", None), source, imgsz=imgsz, conf=conf,
                               display=display, writer=writer, playback_fps=playback_fps,
                               detector=detector, cancel_token=cancel_token, on_frame=on_frame)
    finally:
        sampler.stop()
        if own_source:
            source.release()
        if writer is not None:
            writer.release()
        detector.close()
    warmup = warmup_frames if warmup_frames is not None else source.frame_count * WARMUP_LOOPS
    summary = summarize(records, t_total, warmup)
    return records, t_total, sampler.samples, warmup, summary


def records_to_dataframe(records_by_mode):
    """records_by_mode: dict {mode_name: [FrameRecord, ...]} -> un DataFrame largo para graficar/guardar."""
    import pandas as pd
    rows = []
    for mode, recs in records_by_mode.items():
        for r in recs:
            rows.append(dict(mode=mode, frame_idx=r.frame_idx,
                              inf_latency_ms=(r.t_inference_end - r.t_inference_start) * 1000,
                              e2e_latency_ms=(r.t_render_end - r.t_capture_start) * 1000,
                              queue_wait_ms=r.queue_wait * 1000,
                              n_detections=r.n_detections,
                              mean_kpt_conf=r.mean_kpt_conf))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sync", "threaded", "compare"], default="compare")
    parser.add_argument("--detector", choices=["yolo", "mediapipe"], default="yolo",
                         help="yolo = nuestro modelo de 5 keypoints; mediapipe = MediaPipe Pose (referencia)")
    parser.add_argument("--video", default=str(ROOT / "data" / "processed" / "video_demo" / "Punto_1-4_mp4_demo.mp4"))
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--loops", type=int, default=15)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--playback-fps", type=float, default=None,
                         help="Limita la velocidad de la ventana en vivo a este FPS (solo con --display; sin esto corre al ritmo maximo del modelo, ~50 FPS, demasiado rapido para ver en un clip corto)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--results-csv", default=str(ROOT / "report" / "benchmark_results.csv"))
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sim-capture-delay-ms", type=float, default=0.0,
                         help="Retardo sintetico tras cada frame capturado (simula una fuente mas lenta, ej. camara en vivo)")
    args = parser.parse_args()

    kwargs = dict(video=args.video, camera_index=args.camera_index, loops=args.loops,
                  display=args.display, output=args.output, imgsz=args.imgsz, conf=args.conf,
                  sim_capture_delay_s=args.sim_capture_delay_ms / 1000.0, playback_fps=args.playback_fps,
                  detector_name=args.detector)

    if args.mode in ("sync", "threaded"):
        records, t_total, samples, warmup, summary = run_pipeline(args.mode, **kwargs)
        print(json.dumps(summary, indent=2))
        records_to_dataframe({args.mode: records}).to_csv(args.results_csv, index=False)
        print(f"Resultados guardados en {args.results_csv}")
    else:
        rec_sync, t_sync, samples_sync, warmup, summary_sync = run_pipeline("sync", **kwargs)
        rec_thr, t_thr, samples_thr, _, summary_thr = run_pipeline("threaded", **kwargs)
        parity = check_parity(rec_sync, rec_thr)
        print("=== referencia (sync) ===")
        print(json.dumps(summary_sync, indent=2))
        print("=== threaded ===")
        print(json.dumps(summary_thr, indent=2))
        print("=== paridad de calidad predictiva ===")
        print(json.dumps(parity, indent=2))
        records_to_dataframe({"sync": rec_sync, "threaded": rec_thr}).to_csv(args.results_csv, index=False)
        print(f"Resultados guardados en {args.results_csv}")


if __name__ == "__main__":
    main()
