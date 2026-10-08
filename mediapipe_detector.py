"""Detector de referencia MediaPipe Pose (33 landmarks) con la misma interfaz `Detector` que
`YoloPoseDetector` de realtime_pipeline.py, para compararlo en vivo contra nuestro modelo de 5
keypoints sobre los mismos frames (app web de dos pantallas, ver webapp/).

Usa el Tasks API (`PoseLandmarker`), unico disponible en mediapipe>=1.0, en modo VIDEO (aprovecha
el seguimiento entre frames) y con num_poses=4 porque en una cancha de padel hay 4 jugadores; el
modo legacy `mp.solutions.pose` solo detectaba una persona.

MediaPipe en Python corre en CPU (no hay delegado GPU en Windows), mientras el modelo YOLO corre en
CUDA: la comparacion es tambien CPU vs. GPU, y la app lo deja visible en las metricas.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
MIN_MODEL_BYTES = 1_000_000  # un .task real pesa 5-30 MB; menos que esto es una descarga rota

# Equivalencias con nuestros 5 keypoints (hombro, codo, muneca, raqueta, cabeza). La raqueta no
# tiene equivalente en MediaPipe: es la ventaja propia del modelo afinado.
EQUIVALENT_LANDMARKS = {
    0: "cabeza",  # nariz
    11: "hombro", 12: "hombro",
    13: "codo", 14: "codo",
    15: "muneca", 16: "muneca",
}
# Mismos subconjuntos que KPT_SUBSETS de realtime_pipeline.py, en indices de MediaPipe. MediaPipe
# no tiene raqueta, asi que "brazo_raqueta" muestra lo mismo que "brazo"; y como no sabe cual es el
# brazo que golpea, se muestran los dos.
MP_SUBSETS = {
    "brazo": (11, 12, 13, 14, 15, 16),
    "brazo_raqueta": (11, 12, 13, 14, 15, 16),
    "todos": tuple(range(33)),
}
POSE_COLORS = [(60, 180, 75), (220, 80, 60), (60, 140, 230), (220, 180, 40)]
HIGHLIGHT = (0, 255, 255)


def ensure_model(variant="full"):
    """Ruta local del .task; lo descarga una sola vez a models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if path.exists() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
    if tmp.stat().st_size < MIN_MODEL_BYTES:
        tmp.unlink()
        raise RuntimeError(f"La descarga del modelo MediaPipe '{variant}' salio incompleta.")
    tmp.replace(path)
    return path


class MediaPipePoseDetector:
    name = "mediapipe"
    device = "cpu"

    def __init__(self, variant="full", num_poses=4, min_conf=0.5, keypoints="todos"):
        from mediapipe.tasks.python import BaseOptions, vision

        self._vision = vision
        self.show = set(MP_SUBSETS[keypoints])
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(variant))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        self._last_ts_ms = -1
        self._t0 = time.perf_counter()

    def infer(self, frame):
        import mediapipe as mp

        # el modo VIDEO exige timestamps estrictamente crecientes; se usa el reloj real (la fuente
        # puede saltar frames), forzando +1 ms si dos llamadas caen en el mismo milisegundo
        ts_ms = int((time.perf_counter() - self._t0) * 1000)
        ts_ms = max(ts_ms, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        # la conversion de color y el empaquetado cuentan como parte de la inferencia (YOLO tambien
        # preprocesa dentro de model.track), para comparar en igualdad de condiciones
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, ts_ms)

    def _poses_px(self, result, width, height):
        poses = []
        for landmarks in result.pose_landmarks or []:
            pts = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)
            vis = np.array([lm.visibility if lm.visibility is not None else 1.0 for lm in landmarks],
                           dtype=np.float32)
            poses.append((pts, vis))
        return poses

    def draw(self, frame, result, kpt_conf_min=0.3):
        h, w = frame.shape[:2]
        poses = self._poses_px(result, w, h)
        for i, (pts, vis) in enumerate(poses):
            color = POSE_COLORS[i % len(POSE_COLORS)]
            for a, b in self._connections:
                if a in self.show and b in self.show and vis[a] > kpt_conf_min and vis[b] > kpt_conf_min:
                    cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, 2)
            for j, (px, py) in enumerate(pts):
                if j not in self.show or vis[j] <= kpt_conf_min:
                    continue
                if j in EQUIVALENT_LANDMARKS:  # los equivalentes a nuestros 5 puntos, resaltados
                    cv2.circle(frame, (int(px), int(py)), 5, HIGHLIGHT, -1)
                else:
                    cv2.circle(frame, (int(px), int(py)), 2, color, -1)
            visible = pts[vis > kpt_conf_min]
            if len(visible):
                x1, y1 = visible.min(axis=0)
                cv2.putText(frame, f"pose {i + 1}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return frame, len(poses)

    def keypoints_summary(self, result):
        if not result.pose_landmarks:
            return None
        return [round(v, 4) for lms in result.pose_landmarks for lm in lms for v in (lm.x, lm.y)]

    def mean_kpt_conf(self, result):
        """Visibilidad media de los landmarks que se muestran (p.ej. solo los del brazo)."""
        vis = [lm.visibility for lms in (result.pose_landmarks or []) for j, lm in enumerate(lms)
               if j in self.show and lm.visibility is not None]
        return float(np.mean(vis)) if vis else 0.0

    def warmup(self, size=(640, 640), n=2):
        """Inicializa el grafo de MediaPipe antes de ver frames reales (simetrico a YOLO)."""
        dummy = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)

    def close(self):
        self._landmarker.close()
