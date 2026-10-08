"""Camara del celular: relay (token, pagina, WebSocket, POST), certificado, QR y la fuente "phone"."""
import queue
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from webapp import phone
from webapp.capture import capture_main
from webapp.shared_frames import STATE_EOF, STATE_READY, FrameRing


def jpeg(value=128, w=64, h=48):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    return buf.tobytes()


@pytest.fixture
def hub(monkeypatch):
    fresh = phone.PhoneHub()
    monkeypatch.setattr(phone, "hub", fresh)
    return fresh


def test_token_and_rotation(hub):
    old = hub.token
    assert hub.matches(old) and not hub.matches("x" * len(old)) and not hub.matches("corto")
    gen = hub.connect()
    hub.rotate()
    assert not hub.matches(old)
    assert not hub.publish(jpeg(), gen), "un celular con el enlace viejo no debe poder publicar"


def test_publish_validates_and_forwards_newest(hub):
    gen = hub.connect()
    assert not hub.publish(b"no es un jpeg", gen)
    sink = queue.Queue(maxsize=2)
    hub.attach(sink)
    for v in (10, 20, 30):
        assert hub.publish(jpeg(v), gen)
    assert sink.qsize() == 2 and hub.dropped_to_sink == 1  # la cola llena descarta, no bloquea
    t_ns, data = sink.get_nowait()
    assert isinstance(t_ns, int) and data.startswith(b"\xff\xd8")
    hub.detach(sink)
    hub.publish(jpeg(40), gen)
    assert sink.qsize() == 1  # ya no se reenvia a la sesion
    assert hub.status()["streaming"]


def test_relay_routes(hub):
    client = TestClient(phone.create_relay_app())
    assert client.get("/" + "x" * len(hub.token)).status_code == 404
    page = client.get("/" + hub.token)
    assert page.status_code == 200 and "phone.js" in page.text
    assert client.post(f"/{hub.token}/frame", content=jpeg()).json() == {"ok": True}
    assert client.post("/malo/frame", content=jpeg()).status_code == 404

    with client.websocket_connect(f"/{hub.token}/ws") as ws:
        ws.send_text("size 64 48")
        ws.send_bytes(jpeg(77))
        for _ in range(50):
            if hub.status()["size"] == (64, 48) and hub.latest() == jpeg(77):
                break
            time.sleep(0.02)
    assert hub.status()["size"] == (64, 48)
    assert hub.latest() == jpeg(77)


def test_bad_token_websocket_is_rejected(hub):
    client = TestClient(phone.create_relay_app())
    with pytest.raises(Exception):
        with client.websocket_connect("/" + "x" * len(hub.token) + "/ws") as ws:
            ws.receive_text()


def test_certificate_has_ip_and_is_reused(tmp_path, monkeypatch):
    from cryptography import x509

    monkeypatch.setattr(phone, "CERT_DIR", tmp_path)
    cert, key = phone.ensure_certificate("192.168.1.50")
    parsed = x509.load_pem_x509_certificate(cert.read_bytes())
    san = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert [str(ip) for ip in san.get_values_for_type(x509.IPAddress)] == ["192.168.1.50"]
    mtime = cert.stat().st_mtime_ns
    phone.ensure_certificate("192.168.1.50")
    assert cert.stat().st_mtime_ns == mtime  # misma IP: no se regenera
    phone.ensure_certificate("10.0.0.7")
    assert cert.stat().st_mtime_ns != mtime  # cambio la IP: certificado nuevo


def test_qr_svg_scales():
    svg = phone.qr_svg("https://192.168.0.10:8443/abc")
    assert svg.startswith("<svg") and "viewBox" in svg and 'width="' not in svg.split(">")[0]


def test_capture_from_phone_queue_publishes_and_letterboxes():
    ring = FrameRing.create()
    try:
        channel, phone_q = queue.Queue(), queue.Queue(maxsize=2)
        stop, go = threading.Event(), threading.Event()
        cfg = {"source": {"kind": "phone"}}
        t = threading.Thread(target=capture_main, args=(cfg, ring.name, channel, stop, go, phone_q), daemon=True)
        t.start()
        phone_q.put((time.perf_counter_ns(), jpeg(100, 64, 48)))
        deadline = time.monotonic() + 5
        while ring.state != STATE_READY and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ring.state == STATE_READY and ring.frame_count == -1
        go.set()
        phone_q.put((time.perf_counter_ns(), jpeg(200, 48, 64)))  # el celular roto: otro tamano
        deadline = time.monotonic() + 5
        while ring.latest_seq < 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        seq, idx, _, frame = ring.read_latest(-1)
        assert frame.shape == (48, 64, 3), "el frame rotado se encaja en el tamano anunciado"
        stop.set()
        t.join(timeout=5)
        assert ring.state == STATE_EOF
        kinds = [m["type"] for m in iter(lambda: channel.get_nowait() if not channel.empty() else None, None)]
        assert "done" in kinds and kinds[-1] == "eof"
    finally:
        ring.close()
