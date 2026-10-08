"""Una sola captura, varios consumidores: ring de frames en memoria compartida entre procesos.

Por que no mandar los frames por una multiprocessing.Queue: cada frame se serializaria (pickle) y
pasaria por un pipe. A 1080p son ~6 MB por frame; a 30 FPS y dos consumidores, ~370 MB/s de copias
que se comerian la CPU que queremos medir. Y por que no abrir la fuente dos veces: una camara en
Windows no se puede abrir desde dos procesos a la vez.

El proceso de captura escribe en un ring de N slots dentro de un bloque `SharedMemory`; cada
worker copia el slot MAS NUEVO (semantica de camara real: si el modelo es lento, salta frames y
se cuentan como descartados). La consistencia se asegura con un seqlock por slot:

    escritor: slot_seq = -1  -> copia pixeles + metadatos -> slot_seq = seq -> latest = seq
    lector:   lee slot_seq   -> copia                     -> relee slot_seq; si cambio, reintenta

Las escrituras de enteros de 64 bits alineados son atomicas en x86-64/ARM64, que es lo que el
seqlock necesita. El tiempo de captura viaja como `perf_counter_ns()`, monotonico a nivel de
sistema (QPC en Windows, CLOCK_MONOTONIC en Linux): comparable entre procesos para la latencia E2E.
"""
from __future__ import annotations

import os
import time
from multiprocessing import shared_memory

import numpy as np

MAX_H, MAX_W = 1080, 1920  # frames mayores se reducen (manteniendo aspecto) antes de publicarse
N_SLOTS = 4

# cabecera (int64): estado global + metadatos por slot
H_LATEST, H_STATE, H_FPS_MILLI, H_FRAME_COUNT, H_WIDTH, H_HEIGHT = range(6)
GLOBAL_FIELDS = 8
S_SEQ, S_IDX, S_TNS, S_H, S_W = range(5)
SLOT_FIELDS = 5
HEADER_LEN = GLOBAL_FIELDS + N_SLOTS * SLOT_FIELDS

STATE_STARTING, STATE_READY, STATE_EOF, STATE_ERROR = 0, 1, 2, 3


def _nbytes():
    return HEADER_LEN * 8 + N_SLOTS * MAX_H * MAX_W * 3


class FrameRing:
    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self.shm = shm
        self.owner = owner
        self.header = np.ndarray((HEADER_LEN,), dtype=np.int64, buffer=shm.buf)
        self.pixels = np.ndarray((N_SLOTS, MAX_H, MAX_W, 3), dtype=np.uint8,
                                 buffer=shm.buf, offset=HEADER_LEN * 8)

    # -- ciclo de vida ------------------------------------------------------------------
    @classmethod
    def create(cls) -> "FrameRing":
        ring = cls(shared_memory.SharedMemory(create=True, size=_nbytes()), owner=True)
        ring.header[:] = 0
        ring.header[H_LATEST] = -1
        for s in range(N_SLOTS):
            ring.header[GLOBAL_FIELDS + s * SLOT_FIELDS + S_SEQ] = -1
        return ring

    @classmethod
    def attach(cls, name: str) -> "FrameRing":
        shm = shared_memory.SharedMemory(name=name)
        if os.name != "nt":
            # en POSIX el resource_tracker del hijo borraria el bloque al salir; el dueno es el padre
            from multiprocessing import resource_tracker
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        return cls(shm, owner=False)

    @property
    def name(self) -> str:
        return self.shm.name

    def close(self) -> None:
        # soltar las vistas numpy antes de cerrar: si no, SharedMemory.close() falla (BufferError)
        self.header = None
        self.pixels = None
        self.shm.close()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    # -- estado global ------------------------------------------------------------------
    def set_ready(self, fps: float, frame_count: int, width: int, height: int) -> None:
        self.header[H_FPS_MILLI] = int(round(fps * 1000))
        self.header[H_FRAME_COUNT] = frame_count
        self.header[H_WIDTH], self.header[H_HEIGHT] = width, height
        self.header[H_STATE] = STATE_READY

    def set_state(self, state: int) -> None:
        self.header[H_STATE] = state

    @property
    def state(self) -> int:
        return int(self.header[H_STATE])

    @property
    def fps(self) -> float:
        return self.header[H_FPS_MILLI] / 1000.0

    @property
    def frame_count(self) -> int:
        return int(self.header[H_FRAME_COUNT])

    @property
    def latest_seq(self) -> int:
        return int(self.header[H_LATEST])

    # -- escritor (proceso de captura) ----------------------------------------------------
    def publish(self, frame: np.ndarray, frame_idx: int, t_ns: int) -> int:
        h, w = frame.shape[:2]
        if h > MAX_H or w > MAX_W:
            raise ValueError(f"Frame de {w}x{h} excede el maximo {MAX_W}x{MAX_H}; reducirlo antes")
        seq = self.latest_seq + 1
        slot = seq % N_SLOTS
        base = GLOBAL_FIELDS + slot * SLOT_FIELDS
        self.header[base + S_SEQ] = -1  # "escribiendo": un lector que llegue ahora reintenta
        self.pixels[slot, :h, :w] = frame
        self.header[base + S_IDX] = frame_idx
        self.header[base + S_TNS] = t_ns
        self.header[base + S_H], self.header[base + S_W] = h, w
        self.header[base + S_SEQ] = seq
        self.header[H_LATEST] = seq
        return seq

    # -- lector (workers) -----------------------------------------------------------------
    def read_latest(self, after_seq: int):
        """(seq, frame_idx, t_ns, frame_copia) del frame mas nuevo con seq > after_seq, o None."""
        for _ in range(8):  # reintentos si el escritor pisa el slot mientras copiamos
            seq = self.latest_seq
            if seq <= after_seq:
                return None
            base = GLOBAL_FIELDS + (seq % N_SLOTS) * SLOT_FIELDS
            if self.header[base + S_SEQ] != seq:
                continue
            frame_idx = int(self.header[base + S_IDX])
            t_ns = int(self.header[base + S_TNS])
            h, w = int(self.header[base + S_H]), int(self.header[base + S_W])
            frame = self.pixels[seq % N_SLOTS, :h, :w].copy()
            if self.header[base + S_SEQ] == seq:
                return seq, frame_idx, t_ns, frame
        return None


class SharedFrameSource:
    """Iterable compatible con `FrameSource` de realtime_pipeline.py, alimentado por el ring.

    Entrega (frame_idx, frame, t_captura_s) con t_captura en la escala de time.perf_counter(),
    para que run_sync/run_threaded midan la latencia E2E desde la captura real.
    """

    POLL_S = 0.001

    def __init__(self, ring: FrameRing, cancel_token=None):
        self.ring = ring
        self.cancel_token = cancel_token
        self.delivered = 0
        self.dropped = 0  # frames de la fuente que este consumidor no alcanzo a procesar
        self._last_seq = -1

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        t0 = time.monotonic()
        while self.ring.state == STATE_STARTING:
            if self._cancelled():
                return
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError("La fuente de video no arranco a tiempo.")
            time.sleep(0.01)
        if self.ring.state == STATE_ERROR:
            raise RuntimeError("La captura fallo al abrir la fuente.")

    # atributos que usan run_pipeline / summarize / el escritor
    @property
    def fps(self) -> float:
        return self.ring.fps or 30.0

    @property
    def frame_count(self) -> int:
        return self.ring.frame_count

    def _cancelled(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_set()

    def __iter__(self):
        while not self._cancelled():
            got = self.ring.read_latest(self._last_seq)
            if got is None:
                if self.ring.state in (STATE_EOF, STATE_ERROR) and self.ring.latest_seq <= self._last_seq:
                    return
                time.sleep(self.POLL_S)
                continue
            seq, frame_idx, t_ns, frame = got
            if self._last_seq >= 0:
                self.dropped += seq - self._last_seq - 1
            self._last_seq = seq
            self.delivered += 1
            yield frame_idx, frame, t_ns / 1e9

    def release(self) -> None:
        pass
