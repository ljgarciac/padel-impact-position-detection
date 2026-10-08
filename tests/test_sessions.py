"""Ciclo de vida de SessionManager sin lanzar procesos reales (spawn inyectado).

Port de la idea de live_camera/tests/test_jobs.py: un `spawn` falso devuelve colas y procesos de
mentira que la prueba controla mensaje a mensaje.
"""
import queue
import threading
import time

import pytest

from webapp.sessions import SessionManager, SessionState


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False


class FakeHandle:
    def __init__(self, models):
        self.procs = {name: FakeProc(100 + i) for i, name in enumerate([*models, "capture"])}
        self.channel = queue.Queue()
        self.frame_queues = {m: queue.Queue(maxsize=3) for m in models}
        self.stop_event = threading.Event()
        self.go_event = threading.Event()
        self.closed = False

    def close(self):
        self.closed = True

    def send(self, **msg):
        self.channel.put(msg)

    def finish(self, name, kind="done", **extra):
        self.send(type=kind, proc=name, **extra)
        self.send(type="eof", proc=name)
        self.procs[name].alive = False


@pytest.fixture
def manager(tmp_path):
    handles = []

    def spawn(config, out_dir):
        h = FakeHandle(config["models"])
        handles.append(h)
        return h

    m = SessionManager(spawn=spawn, monitor_factory=lambda s: None, runs_dir=tmp_path)
    m.handles = handles
    return m


def config(models=("yolo", "mediapipe")):
    return {"models": list(models), "mode": "threaded", "source": {"kind": "video", "path": "x.mp4"}}


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("la condicion no se cumplio a tiempo")
        time.sleep(0.02)


def test_one_session_at_a_time(manager):
    a = manager.submit(config())
    b = manager.submit(config())
    assert a.state == SessionState.RUNNING
    assert b.state == SessionState.QUEUED
    assert len(manager.handles) == 1

    h = manager.handles[0]
    for name in ("yolo", "mediapipe", "capture"):
        h.finish(name, summary={"pipeline_fps": 30.0}, files={})
    wait_for(lambda: a.state.is_final)
    assert a.state == SessionState.SUCCEEDED
    assert h.closed
    wait_for(lambda: b.state == SessionState.RUNNING)
    assert len(manager.handles) == 2


def test_go_event_waits_for_every_model(manager):
    manager.submit(config())
    h = manager.handles[0]
    h.send(type="status", proc="yolo", state="waiting")
    time.sleep(0.2)
    assert not h.go_event.is_set()  # MediaPipe todavia cargando: la fuente no arranca
    h.send(type="status", proc="mediapipe", state="waiting")
    wait_for(h.go_event.is_set)


def test_stop_is_graceful(manager):
    s = manager.submit(config(["yolo"]))
    h = manager.handles[0]
    assert manager.stop(s.id)
    assert h.stop_event.is_set()
    assert s.state == SessionState.STOPPING
    h.finish("yolo", summary={"pipeline_fps": 10.0}, files={})
    h.finish("capture")
    wait_for(lambda: s.state.is_final)
    assert s.state == SessionState.SUCCEEDED
    assert s.message == "Detenida"


def test_cancel_queued(manager):
    manager.submit(config())
    queued = manager.submit(config())
    assert manager.stop(queued.id)
    assert queued.state == SessionState.CANCELLED


def test_dead_worker_does_not_kill_the_other(manager):
    s = manager.submit(config())
    h = manager.handles[0]
    h.procs["mediapipe"].alive = False  # murio sin mandar eof (p.ej. taskkill)
    wait_for(lambda: s.procs["mediapipe"]["state"] == "died", timeout=6)
    assert s.state == SessionState.RUNNING
    h.finish("yolo", summary={"pipeline_fps": 30.0}, files={})
    h.finish("capture")
    wait_for(lambda: s.state.is_final)
    assert s.state == SessionState.SUCCEEDED
    assert s.procs["yolo"]["state"] == "done"


def test_all_workers_failed(manager):
    s = manager.submit(config())
    h = manager.handles[0]
    h.finish("yolo", kind="error", error={"user_message": "OOM"})
    h.finish("mediapipe", kind="error", error={"user_message": "x"})
    wait_for(h.stop_event.is_set)  # sin modelos vivos no tiene sentido seguir capturando
    h.finish("capture")
    wait_for(lambda: s.state.is_final)
    assert s.state == SessionState.FAILED


def test_sse_first_snapshot_then_none(manager):
    s = manager.submit(config(["yolo"]))
    q = manager.subscribe(s.id)
    assert q.get(timeout=1)["id"] == s.id  # el primero es siempre el estado actual
    h = manager.handles[0]
    h.finish("yolo", summary={}, files={})
    h.finish("capture")
    items = []
    while True:
        item = q.get(timeout=5)
        if item is None:
            break
        items.append(item)
    assert items[-1]["state"] == "succeeded"
    # quien se suscribe tarde a una sesion terminada recibe el estado y el fin
    late = manager.subscribe(s.id)
    assert late.get(timeout=1)["state"] == "succeeded"
    assert late.get(timeout=1) is None


def test_latest_frame_keeps_only_newest(manager):
    s = manager.submit(config(["yolo"]))
    h = manager.handles[0]
    for i in range(3):
        h.frame_queues["yolo"].put(f"jpeg{i}".encode())
    wait_for(lambda: (manager.latest_frame(s.id, "yolo", 0, timeout=0.1) or (0, b""))[1] == b"jpeg2")
    seq, jpeg = manager.latest_frame(s.id, "yolo", 0)
    assert manager.latest_frame(s.id, "yolo", seq, timeout=0.1) is None  # nada nuevo despues de seq
