"""Proceso de modelo (uno por pantalla): corre el pipeline de realtime_pipeline.py sobre el ring.

Adaptado de `_worker` en live_camera (feat/local-app, core/jobs.py): los imports pesados viven
dentro de la funcion (el servidor web nunca carga torch ni mediapipe) y por el canal de mensajes
solo viajan diccionarios pequenos. Los frames de la vista previa van por un canal aparte y acotado.

Dentro del proceso, la arquitectura es la del notebook (CP1): con mode="threaded", hilo de
captura (lee del ring) -> cola A -> hilo de inferencia -> cola B -> hilo de render. Se le suman
dos hilos auxiliares, ambos FUERA de la latencia medida (on_frame corre despues de t_render_end):
  - vista previa: reduce y codifica a JPEG el frame mas nuevo (cv2.imencode libera el GIL)
  - video: escribe el MP4 H.264 (AsyncVideoWriter)
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

METRICS_EVERY_S = 0.5
WINDOW_FRAMES = 30          # ventana para percentiles en vivo
CAMERA_WARMUP_FRAMES = 30   # con camara no hay "primer loop" que descartar
PREVIEW_MAX_W = 960


class _NullSampler:
    """Los recursos se miden desde el proceso servidor (por PID); aqui no se muestrea nada."""
    samples: list = []

    def start(self):
        return time.perf_counter()

    def stop(self):
        pass


class PreviewPublisher:
    """Hilo que codifica a JPEG el frame mas nuevo y lo manda al servidor sin bloquear nunca."""

    def __init__(self, frames_q, max_fps: float = 20.0, quality: int = 75):
        self.frames_q = frames_q
        self.min_period = 1.0 / max_fps
        self.quality = quality
        self._latest: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_t = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        now = time.perf_counter()
        if now - self._last_t < self.min_period:
            return
        self._last_t = now
        try:
            self._latest.get_nowait()  # se descarta el pendiente: solo importa el mas nuevo
        except queue.Empty:
            pass
        try:
            self._latest.put_nowait(frame)
        except queue.Full:
            pass

    def _run(self) -> None:
        import cv2

        while not self._stop.is_set():
            try:
                frame = self._latest.get(timeout=0.2)
            except queue.Empty:
                continue
            h, w = frame.shape[:2]
            if w > PREVIEW_MAX_W:
                frame = cv2.resize(frame, (PREVIEW_MAX_W, int(h * PREVIEW_MAX_W / w)),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                continue
            try:
                self.frames_q.put_nowait(buf.tobytes())
            except queue.Full:
                pass  # el servidor va atrasado: este frame se pierde, el siguiente llegara

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class LiveMetrics:
    """Metricas de desempeno en ventana deslizante, calculadas con los FrameRecord del pipeline."""

    def __init__(self):
        self.frames = 0
        self.inf_ms = deque(maxlen=WINDOW_FRAMES)
        self.e2e_ms = deque(maxlen=WINDOW_FRAMES)
        self.queue_ms = deque(maxlen=WINDOW_FRAMES)
        self.n_det = deque(maxlen=WINDOW_FRAMES)
        self.conf = deque(maxlen=WINDOW_FRAMES)
        self.render_t = deque(maxlen=240)
        self._last_emit = 0.0

    def add(self, rec) -> None:
        self.frames += 1
        self.inf_ms.append((rec.t_inference_end - rec.t_inference_start) * 1000)
        self.e2e_ms.append((rec.t_render_end - rec.t_capture_start) * 1000)
        self.queue_ms.append(rec.queue_wait * 1000)
        self.n_det.append(rec.n_detections)
        self.conf.append(rec.mean_kpt_conf)
        self.render_t.append(rec.t_render_end)

    def due(self) -> bool:
        now = time.perf_counter()
        if now - self._last_emit >= METRICS_EVERY_S:
            self._last_emit = now
            return True
        return False

    def snapshot(self) -> dict:
        now = time.perf_counter()
        recent = [t for t in self.render_t if now - t <= 1.0]
        intervals = np.diff(np.array(self.render_t)[-WINDOW_FRAMES:]) * 1000 if len(self.render_t) > 2 else []

        def pct(values, q):
            return float(np.percentile(values, q)) if len(values) else None

        return {
            "frames_processed": self.frames,
            "fps": float(len(recent)),
            "inference_fps": 1000.0 / float(np.mean(self.inf_ms)) if self.inf_ms else None,
            "inf_ms_p50": pct(self.inf_ms, 50), "inf_ms_p95": pct(self.inf_ms, 95),
            "e2e_ms_p50": pct(self.e2e_ms, 50), "e2e_ms_p95": pct(self.e2e_ms, 95),
            "queue_ms_mean": float(np.mean(self.queue_ms)) if self.queue_ms else None,
            "jitter_ms": float(np.std(intervals)) if len(intervals) else None,
            "detections": float(np.mean(self.n_det)) if self.n_det else None,
            "kpt_conf": float(np.mean(self.conf)) if self.conf else None,
        }


def _torch_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return {"vram_alloc_mb": torch.cuda.memory_allocated() / 2**20,
                    "vram_reserved_mb": torch.cuda.memory_reserved() / 2**20}
    except Exception:
        pass
    return {}


def build_detector(name: str, cfg: dict):
    from realtime_pipeline import make_detector

    keypoints = cfg.get("keypoints", "brazo_raqueta")
    if name == "mediapipe":
        from mediapipe_detector import MediaPipePoseDetector
        return MediaPipePoseDetector(variant=cfg.get("mp_variant", "full"), keypoints=keypoints)
    return make_detector("yolo", imgsz=int(cfg.get("imgsz", 384)), conf=float(cfg.get("conf", 0.25)),
                         keypoints=keypoints)


def worker_main(name: str, cfg: dict, shm_name: str, channel, frames_q, stop_event, out_dir: str) -> None:
    """Punto de entrada del subproceso de un modelo ('yolo' o 'mediapipe')."""
    from webapp.capture import release_queues, watch_parent

    orphaned = watch_parent(stop_event)
    preview = writer = ring = None
    try:
        from realtime_pipeline import records_to_dataframe, run_pipeline
        from webapp.shared_frames import H_HEIGHT, H_WIDTH, FrameRing, SharedFrameSource
        from webapp.video_out import AsyncVideoWriter

        channel.put({"type": "status", "proc": name, "state": "loading", "message": "Cargando modelo"})
        detector = build_detector(name, cfg)

        ring = FrameRing.attach(shm_name)
        source = SharedFrameSource(ring, cancel_token=stop_event)
        source.wait_ready()
        # precalentar con el tamano real de la fuente: la primera inferencia (kernels CUDA, grafo de
        # MediaPipe) tarda segundos y, en vivo, se comeria el arranque del video
        channel.put({"type": "status", "proc": name, "state": "loading", "device": detector.device,
                     "message": "Precalentando"})
        if hasattr(detector, "warmup"):
            detector.warmup(size=(int(ring.header[H_WIDTH]) or 640, int(ring.header[H_HEIGHT]) or 640))
        channel.put({"type": "status", "proc": name, "state": "waiting", "device": detector.device,
                     "message": "Esperando la fuente"})

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        writer = AsyncVideoWriter(out / "anotado.mp4", fps=source.fps)
        preview = PreviewPublisher(frames_q, max_fps=float(cfg.get("preview_fps", 20))) if cfg.get("preview", True) else None
        live = LiveMetrics()
        extra_gpu = name == "yolo"
        channel.put({"type": "status", "proc": name, "state": "running", "device": detector.device,
                     "message": "Procesando"})

        def on_frame(frame, rec):
            live.add(rec)
            writer.submit(frame, rec.frame_idx)
            if preview is not None:
                preview.submit(frame)
            if live.due():
                metrics = live.snapshot()
                metrics["frames_dropped"] = source.dropped
                metrics["frame_idx"] = rec.frame_idx
                if extra_gpu:
                    metrics.update(_torch_vram_mb())
                channel.put({"type": "metrics", "proc": name, "metrics": metrics})

        warmup = source.frame_count if source.frame_count > 0 else CAMERA_WARMUP_FRAMES
        records, t_total, _, warmup, summary = run_pipeline(
            cfg.get("mode", "threaded"), detector=detector, source=source, cancel_token=stop_event,
            on_frame=on_frame, sampler=_NullSampler(), warmup_frames=warmup)

        if preview is not None:
            preview.close()
            preview = None
        video_path = writer.close()
        video_error = writer.error
        writer = None

        summary.update(
            model=name, mode=cfg.get("mode", "threaded"), device=detector.device,
            frames_delivered=source.delivered, frames_dropped=source.dropped,
            drop_pct=100.0 * source.dropped / max(1, source.delivered + source.dropped),
            warmup_frames=warmup,
        )
        csv_path = out / "resultados.csv"
        records_to_dataframe({name: records}).to_csv(csv_path, index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        channel.put({"type": "done", "proc": name, "summary": summary,
                     "files": {"csv": str(csv_path), "video": str(video_path) if video_path else None},
                     "video_error": video_error})
    except BaseException as e:  # noqa: BLE001 - hay que reportar cualquier fallo
        import traceback

        channel.put({"type": "error", "proc": name,
                     "error": {"code": type(e).__name__, "user_message": str(e) or "El modelo fallo.",
                               "technical_detail": traceback.format_exc()}})
    finally:
        if preview is not None:
            preview.close()
        if writer is not None:
            writer.close()
        if ring is not None:
            ring.close()
        channel.put({"type": "eof", "proc": name})
        release_queues(orphaned, channel, frames_q)
