"""Los ganchos opcionales de realtime_pipeline.py (detector, cancel_token, on_frame) sin GPU."""
import threading
import time

import numpy as np
import pytest

import realtime_pipeline as rp


class FakeSource:
    frame_count = -1
    fps = 30.0

    def __init__(self, n=None):
        self.n = n  # None = stream sin fin, como una camara

    def __iter__(self):
        i = 0
        while self.n is None or i < self.n:
            yield i, np.zeros((32, 32, 3), np.uint8)
            i += 1
            time.sleep(0.001)


class FakeDetector:
    name = "fake"
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def infer(self, frame):
        self.calls += 1
        time.sleep(0.002)
        return self.calls

    def draw(self, frame, result, kpt_conf_min=0.3):
        return frame, 1

    def keypoints_summary(self, result):
        return [float(result)]

    def mean_kpt_conf(self, result):
        return 0.5

    def close(self):
        pass


@pytest.mark.parametrize("fn", [rp.run_sync, rp.run_threaded])
def test_cancel_token_stops_an_endless_source(fn):
    cancel = threading.Event()
    seen = []

    def on_frame(frame, rec):
        seen.append(rec)
        if len(seen) == 20:
            cancel.set()

    before = threading.active_count()
    records, _ = fn(None, FakeSource(), detector=FakeDetector(), cancel_token=cancel, on_frame=on_frame)
    assert 20 <= len(records) <= 25
    assert len(seen) == len(records)
    assert all(r.mean_kpt_conf == 0.5 for r in records)
    deadline = time.monotonic() + 6
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() == before, "quedaron hilos colgados tras cancelar"


@pytest.mark.parametrize("fn", [rp.run_sync, rp.run_threaded])
def test_on_frame_runs_after_render_timestamp(fn):
    stamps = []

    def on_frame(frame, rec):
        stamps.append((rec.t_render_end, time.perf_counter()))

    fn(None, FakeSource(10), detector=FakeDetector(), on_frame=on_frame)
    assert len(stamps) == 10
    assert all(t_render <= t_cb for t_render, t_cb in stamps)  # el callback queda fuera de la latencia


def test_source_capture_time_is_used_for_e2e():
    class TimedSource(FakeSource):
        def __iter__(self):
            for i in range(5):
                yield i, np.zeros((8, 8, 3), np.uint8), time.perf_counter() - 0.5  # capturado hace 0.5 s

    records, _ = rp.run_sync(None, TimedSource(), detector=FakeDetector())
    assert all(r.t_render_end - r.t_capture_start >= 0.5 for r in records)


def test_default_detector_wraps_model_track():
    """Sin `detector`, run_sync usa model.track(persist=True) como siempre."""
    class Result:
        keypoints = None
        boxes = None

    class Model:
        def __init__(self):
            self.kwargs = []

        def track(self, frame, **kw):
            self.kwargs.append(kw)
            return [Result()]

    model = Model()
    records, _ = rp.run_sync(model, FakeSource(3), imgsz=384, conf=0.25)
    assert len(records) == 3
    assert all(kw["persist"] is True and kw["imgsz"] == 384 for kw in model.kwargs)
    assert all(r.n_detections == 0 and r.keypoints == [] for r in records)


def test_summarize_camera_and_new_fields():
    records, t = rp.run_sync(None, FakeSource(12), detector=FakeDetector())
    s = rp.summarize(records, t, warmup_frames=-1 * rp.WARMUP_LOOPS)  # camara: frame_count = -1
    assert s["n_frames_steady"] == 12
    assert s["jitter_ms"] >= 0 and s["mean_kpt_conf"] == 0.5 and s["mean_detections"] == 1
    assert rp.summarize([], 0, 0) == {}
