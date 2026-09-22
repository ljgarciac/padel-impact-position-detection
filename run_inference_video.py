"""Framework de ejecucion: procesa un video de padel frame por frame con el detector YOLO-pose
entrenado en notebooks/actividad_3b_yolo_pose_deteccion.ipynb, trackeando a cada jugador con una
identidad persistente (ByteTrack, via ultralytics model.track()) y dibujando sus 5 keypoints
(cabeza-hombro-codo-muneca-raqueta) cuadro a cuadro.

A diferencia de correr el detector de forma independiente por frame (model.predict()), el tracking
asocia las detecciones de un mismo jugador a lo largo del video, dandole un ID estable aunque se mueva,
se ocluya brevemente o cambie de tamano en pantalla.

Uso:
    python run_inference_video.py --video ruta/al/video.mp4 --output salida.mp4

Por defecto busca los pesos entrenados en data/processed/pose_model_artifact.json (generado por
notebooks/actividad_3b_yolo_pose_deteccion.ipynb).
"""
import argparse
from pathlib import Path
import json

import cv2

ROOT = Path(__file__).resolve().parent
# Orden real de los 5 keypoints en el modelo (verificado visualmente en notebooks/01_pipeline_cnn_impacto.ipynb,
# Checkpoint 0.6): NO es cabeza-hombro-codo-muneca-raqueta secuencial, esta desplazado en uno.
KPT_NAMES = ["hombro", "codo", "muneca", "raqueta", "cabeza"]
SKELETON = [(4, 0), (0, 1), (1, 2), (2, 3)]  # cabeza(4)-hombro(0)-codo(1)-muneca(2)-raqueta(3)

# un color estable por ID de track (se repite ciclicamente si hay mas de 8 jugadores simultaneos)
TRACK_COLORS = [
    (60, 180, 75), (220, 80, 60), (60, 140, 230), (220, 180, 40),
    (200, 80, 200), (0, 200, 200), (230, 130, 40), (140, 60, 220),
]


def load_weights_path(weights_path=None):
    if weights_path is not None:
        return weights_path
    with open(ROOT / "data" / "processed" / "pose_model_artifact.json") as f:
        artifact = json.load(f)
    return artifact["weights_path"]


def process_video(video_path, output_path, weights_path=None, imgsz=384, conf=0.25,
                   kpt_conf_min=0.3, tracker="bytetrack.yaml", progress_every=30):
    from ultralytics import YOLO

    weights_path = load_weights_path(weights_path)
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    seen_ids = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = model.track(frame, imgsz=imgsz, conf=conf, persist=True,
                              tracker=tracker, verbose=False)[0]

        if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = (result.keypoints.conf.cpu().numpy()
                         if result.keypoints.conf is not None else None)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = (result.boxes.id.int().cpu().tolist()
                         if result.boxes.id is not None else [None] * len(boxes_xyxy))

            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                track_id = track_ids[i]
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)] if track_id is not None else (200, 200, 200)
                if track_id is not None:
                    seen_ids.add(track_id)

                pts = kpts_xy[i]
                pconf = kpts_conf[i] if kpts_conf is not None else [1.0] * len(pts)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"jugador {track_id}" if track_id is not None else "sin ID (track perdido)"
                cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                for a, b in SKELETON:
                    if pconf[a] > kpt_conf_min and pconf[b] > kpt_conf_min:
                        pa = tuple(pts[a].astype(int))
                        pb = tuple(pts[b].astype(int))
                        cv2.line(frame, pa, pb, color, 2)
                for j, (px, py) in enumerate(pts):
                    if pconf[j] > kpt_conf_min:
                        cv2.circle(frame, (int(px), int(py)), 3, color, -1)

        writer.write(frame)
        if progress_every and frame_idx % progress_every == 0:
            print(f"frame {frame_idx}/{n_frames or '?'}  jugadores vistos hasta ahora: {len(seen_ids)}")

    cap.release()
    writer.release()
    print(f"Video anotado escrito en: {output_path}")
    print(f"Jugadores distintos trackeados en el clip: {len(seen_ids)} (IDs: {sorted(seen_ids)})")
    return seen_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Ruta al video de entrada (mp4)")
    parser.add_argument("--output", required=True, help="Ruta al video de salida anotado")
    parser.add_argument("--weights", default=None, help="Pesos YOLO-pose (.pt); por defecto usa el artefacto del notebook 2")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza mínima de detección de jugador")
    parser.add_argument("--kpt-conf-min", type=float, default=0.3, help="Confianza mínima por keypoint para dibujarlo")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Config de tracker de ultralytics (bytetrack.yaml o botsort.yaml)")
    args = parser.parse_args()

    process_video(args.video, args.output, weights_path=args.weights, imgsz=args.imgsz,
                  conf=args.conf, kpt_conf_min=args.kpt_conf_min, tracker=args.tracker)


if __name__ == "__main__":
    main()
