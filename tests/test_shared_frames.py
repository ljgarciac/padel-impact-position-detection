"""El ring compartido entre procesos: sin frames corruptos, siempre el mas nuevo, descartes contados."""
import multiprocessing as mp
import time

import numpy as np

from webapp.shared_frames import STATE_EOF, FrameRing, SharedFrameSource

H, W = 480, 640


def _writer(name, n_frames, period_s):
    ring = FrameRing.attach(name)
    ring.set_ready(30.0, n_frames, W, H)
    for i in range(n_frames):
        # cada frame es uniforme con el valor i % 256: una copia a medias se notaria
        ring.publish(np.full((H, W, 3), i % 256, dtype=np.uint8), i, time.perf_counter_ns())
        if period_s:
            time.sleep(period_s)
    ring.set_state(STATE_EOF)
    ring.close()


def _run(n_frames, period_s, consumer_delay_s):
    ring = FrameRing.create()
    try:
        p = mp.get_context("spawn").Process(target=_writer, args=(ring.name, n_frames, period_s))
        p.start()
        source = SharedFrameSource(ring)
        source.wait_ready(timeout_s=30)
        seen = []
        for idx, frame, t_cap in source:
            assert frame.shape == (H, W, 3)
            assert frame.min() == frame.max() == idx % 256, "frame corrupto (seqlock roto)"
            assert t_cap <= time.perf_counter()  # el tiempo de captura es comparable entre procesos
            seen.append(idx)
            if consumer_delay_s:
                time.sleep(consumer_delay_s)
        p.join(timeout=10)
        return seen, source
    finally:
        ring.close()


def test_fast_consumer_sees_every_frame():
    seen, source = _run(n_frames=60, period_s=0.005, consumer_delay_s=0)
    assert seen == sorted(seen)
    assert source.dropped == 0
    assert len(seen) == 60


def test_slow_consumer_drops_and_counts():
    seen, source = _run(n_frames=120, period_s=0.002, consumer_delay_s=0.02)
    assert seen == sorted(seen) and len(set(seen)) == len(seen)
    assert source.dropped > 0
    # todo lo que no se entrego desde el primer frame visto queda contado como descartado
    assert source.delivered + source.dropped == seen[-1] - seen[0] + 1
