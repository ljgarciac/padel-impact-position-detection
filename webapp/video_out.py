"""Escritura de video reproducible en el navegador.

Siempre H.264 (libx264, yuv420p) en MP4, que es lo único que Chrome, Firefox y Safari
reproducen todos. El ffmpeg sale del sistema si trae libx264 y, si no, del que viene
dentro del paquete imageio-ffmpeg. Antes, sin ffmpeg en el PATH (lo normal en
Windows), se caía a OpenCV con mp4v: el archivo existía, pero Chrome se quedaba en 0:00.
Ahora, sin un codificador H.264, falla con un mensaje claro en vez de entregar un
video que no se puede ver.

Portado de live_camera (rama feat/local-app, core/video_out.py). Se agrega `AsyncVideoWriter`,
que escribe desde un hilo propio para no frenar el render del pipeline en vivo.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

CODECS_H264 = {"h264", "avc1", "avc3"}


def _tiene_libx264(ffmpeg: str) -> bool:
    try:
        salida = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "libx264" in salida


def ffmpeg_disponible() -> str | None:
    """Un ffmpeg capaz de escribir H.264, o None."""
    sistema = shutil.which("ffmpeg")
    if sistema and _tiene_libx264(sistema):
        return sistema
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def codec_de_video(path: Path | str) -> str | None:
    """El códec real de la pista de video ('h264', 'mp4v', ...), no la extensión."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        valor = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    if not valor:
        return None
    return valor.to_bytes(4, "little").decode("latin-1").strip("\x00 ").lower()


def es_reproducible(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".mp4" and codec_de_video(path) in CODECS_H264


class VideoWriterMP4:
    """Escribe frames BGR a un MP4 con H.264. Mismo uso que cv2.VideoWriter.

    El atributo `path` dice dónde acabó el archivo: siempre con extensión .mp4.
    """

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int]):
        self.path = Path(path).with_suffix(".mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.size = size

        ffmpeg = ffmpeg_disponible()
        if ffmpeg is None:
            raise ValueError(
                "No hay un codificador H.264 para escribir el video anotado. "
                "Corre `uv sync` para que se instale imageio-ffmpeg, o instala "
                "ffmpeg. El CSV del análisis sí se guardó."
            )
        ancho, alto = size
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{ancho}x{alto}",
                "-r",
                f"{self.fps:g}",
                "-i",
                "-",
                # H.264 por CPU. NO h264_nvenc: tiene mínimos de resolución y límites
                # de sesión, y el cuello de botella es dibujar, no codificar.
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",  # sin esto, Safari, QuickTime y a veces Chrome no lo abren
                "-movflags",
                "+faststart",  # permite reproducir sin descargar entero
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            # ffmpeg murió. release() recogerá su stderr y lo reportará.
            pass

    def release(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        error = b""
        if self._proc.stderr is not None:
            error = self._proc.stderr.read() or b""
        codigo = self._proc.wait(timeout=180)
        self._proc = None
        if codigo != 0:
            detalle = error.decode("utf-8", "replace").strip().splitlines()
            ultima = detalle[-1] if detalle else "sin detalle"
            raise ValueError(
                f"ffmpeg no pudo escribir {self.path} (código {codigo}): {ultima}"
            )
        codec = codec_de_video(self.path)
        if codec not in CODECS_H264:
            raise ValueError(
                f"El video {self.path.name} salió con el códec {codec or 'desconocido'}, "
                f"que el navegador no reproduce; se esperaba H.264."
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
        return False


class AsyncVideoWriter:
    """Escribe el video anotado desde un hilo aparte, abriendo el MP4 con el primer frame.

    En vivo el worker no procesa todos los frames de la fuente (toma siempre el mas nuevo).
    Para que el video dure lo mismo que la fuente, cada frame procesado se repite tantas veces
    como frames de la fuente cubrio (`frame_idx` actual menos el anterior): se ve a la tasa
    efectiva del modelo, tironeado si salta frames, que es justo lo que hay que mostrar.

    La cola esta acotada: si el codificador no da abasto, `submit` descarta el frame en vez de
    bloquear el render (y lo cuenta en `skipped`).
    """

    MAX_REPEAT = 30  # tope de repeticiones por frame (p.ej. tras una pausa larga de la fuente)

    def __init__(self, path: Path | str, fps: float, maxsize: int = 120):
        self.path = Path(path).with_suffix(".mp4")
        self.fps = fps
        self.error: str | None = None
        self.skipped = 0
        self.written = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._writer: VideoWriterMP4 | None = None
        self._last_idx: int | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, frame_idx: int) -> None:
        try:
            self._q.put_nowait((frame, frame_idx))
        except queue.Full:
            self.skipped += 1

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if self.error is not None:
                continue
            frame, frame_idx = item
            try:
                if self._writer is None:
                    h, w = frame.shape[:2]
                    self._writer = VideoWriterMP4(self.path, self.fps, (w, h))
                repeat = 1 if self._last_idx is None else frame_idx - self._last_idx
                self._last_idx = frame_idx
                for _ in range(max(1, min(repeat, self.MAX_REPEAT))):
                    self._writer.write(frame)
                    self.written += 1
            except Exception as e:  # noqa: BLE001 - el video es secundario: se reporta, no se cae
                self.error = str(e)

    def close(self) -> Path | None:
        """Vacia la cola, cierra ffmpeg y devuelve la ruta del MP4 (None si no se escribio)."""
        self._q.put(None)
        self._thread.join(timeout=120)
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
        if self._writer is None or self.error is not None:
            return None
        return self.path
