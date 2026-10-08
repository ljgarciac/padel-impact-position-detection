"""Sesiones en vivo: 1 proceso de captura + 1 proceso por modelo, con progreso, parada y limpieza.

Port de `JobManager` de live_camera (feat/local-app, core/jobs.py), adaptado de "un analisis = un
subproceso" a "una sesion = varios subprocesos". Se conservan sus decisiones:
  - subprocesos y no hilos: la VRAM se libera de verdad al terminar, un OOM de CUDA o un segfault no
    tumba el servidor, y `model.track(persist=True)` arranca limpio en cada sesion;
  - `spawn` explicito (lo unico que existe en Windows y macOS);
  - una sesion a la vez; las demas esperan en cola;
  - parada cooperativa primero (Event), y si no responde: terminate -> kill;
  - un proceso que muere sin avisar se detecta por el silencio del canal + is_alive().
Lo nuevo: un proceso por modelo (para medir CPU/RAM por pantalla), un canal de frames JPEG por
modelo para la vista en vivo (MJPEG) y el muestreo de recursos por PID desde el servidor.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "report" / "live_runs"
MODELS = ("yolo", "mediapipe")
FINAL_PROC_STATES = {"done", "error", "died"}


class SessionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_final(self) -> bool:
        return self in (SessionState.SUCCEEDED, SessionState.FAILED, SessionState.CANCELLED)


@dataclass
class Session:
    id: str
    config: dict
    state: SessionState = SessionState.QUEUED
    message: str = "En cola"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    procs: dict = field(default_factory=dict)       # nombre -> estado, metricas, resumen, error...
    capture: dict = field(default_factory=dict)     # frame_idx, fps, total, tamano
    resources: dict = field(default_factory=dict)   # ultima muestra de ResourceMonitor
    out_dir: str = ""

    def snapshot(self) -> dict:
        total = self.capture.get("total") or 0
        idx = self.capture.get("frame_idx") or 0
        return {
            "id": self.id,
            "config": self.config,
            "state": self.state.value,
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 1),
            "percent": round(min(100.0, 100.0 * idx / total), 1) if total else None,
            "capture": self.capture,
            "procs": self.procs,
            "resources": self.resources,
        }


# --- arranque real de los procesos ---------------------------------------------------------


class _Handle:
    """Lo que el gestor necesita de una sesion lanzada. Las pruebas inyectan uno falso."""

    def __init__(self, procs, channel, frame_queues, stop_event, go_event, ring, phone_q=None):
        self.procs = procs
        self.channel = channel
        self.frame_queues = frame_queues
        self.stop_event = stop_event
        self.go_event = go_event
        self._ring = ring
        self._phone_q = phone_q

    def close(self) -> None:
        if self._phone_q is not None:
            from webapp.phone import hub

            hub.detach(self._phone_q)  # el relay deja de reenviar frames a esta sesion
            self._phone_q = None
        if self._ring is not None:
            self._ring.close()
            self._ring = None


def spawn_real(config: dict, out_dir: Path) -> _Handle:
    from webapp.capture import capture_main
    from webapp.shared_frames import FrameRing
    from webapp.worker import worker_main

    ctx = mp.get_context("spawn")
    ring = FrameRing.create()
    channel = ctx.Queue()
    stop_event, go_event = ctx.Event(), ctx.Event()
    procs, frame_queues = {}, {}
    for name in config["models"]:
        frame_queues[name] = ctx.Queue(maxsize=3)
        procs[name] = ctx.Process(
            target=worker_main, name=f"modelo-{name}", daemon=True,
            args=(name, config, ring.name, channel, frame_queues[name], stop_event, str(out_dir / name)))
    phone_q = None
    if config["source"]["kind"] == "phone":
        from webapp.phone import hub

        # cola corta: si la captura va atrasada, el relay descarta y gana el frame mas nuevo
        phone_q = ctx.Queue(maxsize=2)
        hub.attach(phone_q)
    procs["capture"] = ctx.Process(target=capture_main, name="captura", daemon=True,
                                   args=(config, ring.name, channel, stop_event, go_event, phone_q))
    for p in procs.values():
        p.start()
    return _Handle(procs, channel, frame_queues, stop_event, go_event, ring, phone_q)


# --- gestor en el proceso servidor ----------------------------------------------------------


class SessionManager:
    TIMEOUT_TERMINATE_S = 10.0
    TIMEOUT_KILL_S = 5.0
    MAX_HISTORIAL = 50

    def __init__(self, spawn: Callable | None = None, monitor_factory: Callable | None = None,
                 runs_dir: Path = RUNS_DIR):
        self._sessions: dict[str, Session] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._pending: list[str] = []
        self._active: str | None = None
        self._handle: _Handle | None = None
        self._monitor = None
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._frames: dict[tuple[str, str], tuple[int, bytes]] = {}
        self._frames_cond = threading.Condition()
        self._spawn = spawn or spawn_real
        self._monitor_factory = monitor_factory or self._real_monitor
        self.runs_dir = runs_dir

    # -- API publica ------------------------------------------------------------------------

    def submit(self, config: dict) -> Session:
        with self._lock:
            session = Session(id=uuid.uuid4().hex[:12], config=config)
            session.procs = {name: {"state": "starting", "message": "En cola", "metrics": {},
                                    "summary": None, "files": None, "error": None, "device": ""}
                             for name in config["models"]}
            session.procs["capture"] = {"state": "starting", "message": "En cola", "error": None}
            self._sessions[session.id] = session
            self._order.append(session.id)
            self._pending.append(session.id)
            self._prune()
            self._start_next()
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [self._sessions[i].snapshot() for i in reversed(self._order)]

    def stop(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.state.is_final:
                return False
            if session.state == SessionState.QUEUED:
                self._pending.remove(session_id)
                self._finalize(session, SessionState.CANCELLED, "Cancelada antes de empezar")
                return True
            if session.state == SessionState.STOPPING:
                return True
            session.state = SessionState.STOPPING
            session.message = "Deteniendo"
            handle = self._handle
            self._emit(session)
        if handle is not None:
            handle.stop_event.set()  # cooperativa primero
            threading.Thread(target=self._escalate, args=(session_id, handle), daemon=True).start()
        return True

    def subscribe(self, session_id: str) -> queue.Queue:
        """Cola de eventos para SSE. El primero es siempre el estado actual."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                q.put(session.snapshot())
                if session.state.is_final:
                    q.put(None)
                    return q
            self._subscribers.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(session_id, [])
            if q in subs:
                subs.remove(q)

    def latest_frame(self, session_id: str, name: str, after_seq: int, timeout: float = 1.0):
        """(seq, jpeg) del frame mas nuevo con seq > after_seq; None si no llego ninguno a tiempo."""
        key = (session_id, name)
        with self._frames_cond:
            self._frames_cond.wait_for(lambda: self._frames.get(key, (-1, b""))[0] > after_seq,
                                       timeout=timeout)
            item = self._frames.get(key)
        if item is None or item[0] <= after_seq:
            return None
        return item

    def shutdown(self) -> None:
        """No dejar procesos huerfanos al parar el servidor."""
        with self._lock:
            handle = self._handle
        if handle is None:
            return
        handle.stop_event.set()
        handle.go_event.set()
        deadline = time.monotonic() + 3.0
        for p in handle.procs.values():
            p.join(timeout=max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()

    # -- internos ---------------------------------------------------------------------------

    def _real_monitor(self, session: Session):
        from webapp.resources import ResourceMonitor

        def on_sample(sample):
            with self._lock:
                if session.state.is_final:
                    return
                session.resources = sample
                self._emit(session)

        return ResourceMonitor(on_sample)

    def _prune(self) -> None:
        extra = len(self._order) - self.MAX_HISTORIAL
        for sid in list(self._order):
            if extra <= 0:
                break
            s = self._sessions.get(sid)
            if s is None or not s.state.is_final:
                continue
            self._sessions.pop(sid, None)
            self._order.remove(sid)
            self._subscribers.pop(sid, None)
            with self._frames_cond:
                for key in [k for k in self._frames if k[0] == sid]:
                    self._frames.pop(key, None)
            extra -= 1

    def _start_next(self) -> None:
        if self._active is not None or not self._pending:
            return
        sid = self._pending.pop(0)
        session = self._sessions[sid]
        session.state = SessionState.RUNNING
        session.started_at = time.time()
        session.message = "Arrancando procesos"
        out_dir = self.runs_dir / sid
        session.out_dir = str(out_dir)
        self._active = sid
        try:
            self._handle = self._spawn(session.config, out_dir)
        except Exception as e:  # noqa: BLE001
            self._active = None
            self._finalize(session, SessionState.FAILED, f"No se pudieron lanzar los procesos: {e}")
            return
        for name, p in self._handle.procs.items():
            session.procs[name]["pid"] = getattr(p, "pid", None)
        self._monitor = self._monitor_factory(session)
        if self._monitor is not None:
            for name, p in self._handle.procs.items():
                if getattr(p, "pid", None):
                    self._monitor.track(name, p.pid)
            self._monitor.start()
        self._emit(session)
        threading.Thread(target=self._pump, args=(sid, self._handle), daemon=True).start()
        for name, fq in self._handle.frame_queues.items():
            threading.Thread(target=self._pump_frames, args=(sid, name, fq), daemon=True).start()

    def _pump_frames(self, sid: str, name: str, fq) -> None:
        seq = 0
        while True:
            try:
                jpeg = fq.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    s = self._sessions.get(sid)
                    if s is None or s.state.is_final:
                        return
                continue
            except (EOFError, OSError):
                return
            seq += 1
            with self._frames_cond:
                self._frames[(sid, name)] = (seq, jpeg)
                self._frames_cond.notify_all()

    def _pump(self, sid: str, handle: _Handle) -> None:
        """Lee el canal de los hijos hasta que todos terminan o mueren."""
        eofs: set[str] = set()
        dead_since: dict[str, float] = {}
        while True:
            try:
                msg = handle.channel.get(timeout=1.0)
            except (queue.Empty, EOFError, OSError):
                msg = None
            with self._lock:
                session = self._sessions.get(sid)
                if session is None:
                    return
                if msg is not None:
                    self._handle_message(session, handle, msg, eofs)
                # Procesos que murieron sin mandar eof (OOM, segfault, kill externo). Se da un
                # margen: uno que salio bien puede tener su eof todavia en el canal, detras de
                # las metricas del otro modelo.
                now = time.monotonic()
                for name, p in handle.procs.items():
                    if name in eofs or p.is_alive():
                        continue
                    if now - dead_since.setdefault(name, now) < 2.0:
                        continue
                    info = session.procs[name]
                    if info["state"] not in FINAL_PROC_STATES:
                        info["state"] = "died"
                        info["error"] = {"code": "process_died",
                                         "user_message": "El proceso murio inesperadamente. "
                                                         "Lo mas habitual es quedarse sin memoria (RAM o GPU)."}
                    eofs.add(name)
                    self._emit(session)
                self._advance(session, handle, eofs)
                finished = len(eofs) == len(handle.procs)
            if finished:
                # fuera del lock: joins y la parada del monitor pueden tardar, y el callback del
                # monitor tambien toma el lock
                self._close_session(session, handle)
                return

    def _handle_message(self, session: Session, handle: _Handle, msg: dict, eofs: set) -> None:
        name, kind = msg.get("proc"), msg.get("type")
        info = session.procs.get(name)
        if info is None:
            return
        if kind == "status":
            info["state"] = msg.get("state", info["state"])
            info["message"] = msg.get("message", "")
            if msg.get("device"):
                info["device"] = msg["device"]
            if name == "capture":
                session.capture.update({k: msg[k] for k in ("width", "height", "fps", "total") if k in msg})
        elif kind == "metrics":
            info["metrics"] = msg["metrics"]
        elif kind == "capture":
            session.capture.update({"frame_idx": msg["frame_idx"], "loop": msg["loop"],
                                    "capture_fps": msg["fps"]})
        elif kind == "done":
            info["state"] = "done"
            info["message"] = "Terminado"
            if name != "capture":
                info["summary"] = msg.get("summary")
                info["files"] = msg.get("files")
                info["video_error"] = msg.get("video_error")
        elif kind == "error":
            info["state"] = "error"
            info["error"] = msg.get("error")
            info["message"] = (msg.get("error") or {}).get("user_message", "Error")
        elif kind == "eof":
            eofs.add(name)
            if info["state"] not in FINAL_PROC_STATES:
                info["state"] = "died"
                info["error"] = {"code": "worker_eof", "user_message": "El proceso termino sin resultado."}
        self._emit(session)

    def _advance(self, session: Session, handle: _Handle, eofs: set) -> None:
        workers = [n for n in session.procs if n != "capture"]
        # todos los modelos cargados (o caidos): arranca la fuente
        if not handle.go_event.is_set() and all(
                session.procs[n]["state"] in ("waiting", "running") or session.procs[n]["state"] in FINAL_PROC_STATES
                for n in workers):
            handle.go_event.set()
            session.message = "En vivo"
            self._emit(session)
        # todos los modelos terminaron (p.ej. ambos fallaron): no tiene sentido seguir capturando
        if all(session.procs[n]["state"] in FINAL_PROC_STATES for n in workers) and not handle.stop_event.is_set():
            handle.stop_event.set()
            handle.go_event.set()
        # la captura fallo: los modelos no van a recibir nada
        if session.procs["capture"]["state"] in ("error", "died") and not handle.stop_event.is_set():
            handle.stop_event.set()

    def _escalate(self, sid: str, handle: _Handle) -> None:
        """Si la parada cooperativa no basta: terminate, y despues kill."""
        handle.go_event.set()
        deadline = time.monotonic() + self.TIMEOUT_TERMINATE_S
        for p in handle.procs.values():
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [p for p in handle.procs.values() if p.is_alive()]
        for p in alive:
            p.terminate()
        for p in alive:
            p.join(timeout=self.TIMEOUT_KILL_S)
            if p.is_alive() and hasattr(p, "kill"):
                p.kill()

    def _close_session(self, session: Session, handle: _Handle) -> None:
        for p in handle.procs.values():
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        handle.close()
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            monitor.stop()
            rows = getattr(monitor, "rows", [])
            if rows and session.out_dir:
                self._write_resources(Path(session.out_dir) / "recursos.csv", rows)
        with self._lock:
            workers = [n for n in session.procs if n != "capture"]
            if any(session.procs[n]["state"] == "done" for n in workers):
                stopped = session.state == SessionState.STOPPING
                self._finalize(session, SessionState.SUCCEEDED, "Detenida" if stopped else "Completada")
            else:
                self._finalize(session, SessionState.FAILED, "Ningun modelo termino correctamente")
            self._handle = None
            self._active = None
            self._start_next()

    @staticmethod
    def _write_resources(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "t_s", k != "proc", k))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def _finalize(self, session: Session, state: SessionState, message: str) -> None:
        session.state = state
        session.message = message
        session.finished_at = time.time()
        self._emit(session, final=True)
        with self._frames_cond:
            self._frames_cond.notify_all()

    def _emit(self, session: Session, final: bool = False) -> None:
        snap = session.snapshot()
        for q in self._subscribers.get(session.id, []):
            q.put(snap)
            if final:
                q.put(None)
        if final:
            self._subscribers.pop(session.id, None)


_manager: SessionManager | None = None


def get_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
