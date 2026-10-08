"""Camara de un iPhone (o cualquier celular) en la misma red Wi-Fi, como fuente en vivo.

Adaptado de la propuesta de live_camera (rama live_camara_dual, web/en_vivo.py):
  - La app sigue escuchando SOLO en 127.0.0.1. El celular no puede usar esa direccion (es la de la
    propia maquina), asi que se abre un segundo servidor, el "relay", en la IP de la red local.
  - El relay va por HTTPS: Safari solo deja usar la camara (getUserMedia) en paginas seguras. El
    certificado lo genera esta app (autofirmado, con la IP de la maquina) y no sale de la red; el
    iPhone avisa la primera vez ("Esta conexion no es privada" -> Mostrar detalles -> visitar).
  - Solo entra quien tenga el enlace: lleva un token aleatorio que se muestra como QR en el PC y se
    puede renovar. El relay no expone nada mas de la app (ni sesiones, ni archivos, ni control).

Diferencia con la propuesta: alla el celular calcula MediaPipe en el navegador. Aqui el celular es
solo la camara: manda JPEG y TODOS los modelos (nuestro YOLO de 5 keypoints, que no corre en el
navegador, y MediaPipe) corren en el PC, medidos con las mismas metricas. Los frames llegan al
proceso de captura de la sesion (webapp/capture.py) por una cola, y de ahi al ring compartido.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import queue
import secrets
import socket
import threading
import time
from pathlib import Path

# a nivel de modulo, no dentro de create_relay_app: con `from __future__ import annotations`
# FastAPI resuelve las anotaciones de las rutas en el espacio global del modulo
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
CERT_DIR = ROOT / ".tools" / "phone-cert"
FIRST_PORT = 8443
MAX_JPEG = 2_000_000


class PhoneError(Exception):
    """No se puede recibir la camara del celular. El mensaje se muestra tal cual en la UI."""


# --- estado compartido entre el relay y la app --------------------------------------------


class PhoneHub:
    """Token del enlace, ultimo frame (para la vista previa del PC) y la cola hacia la sesion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = secrets.token_urlsafe(12)
        self._generation = 0
        self._connected = 0
        self._jpeg: bytes | None = None
        self._jpeg_t = 0.0
        self._size: tuple[int, int] | None = None
        self._arrivals: list[float] = []
        self._sink = None  # mp.Queue del proceso de captura mientras hay una sesion con iPhone
        self.dropped_to_sink = 0

    @property
    def token(self) -> str:
        with self._lock:
            return self._token

    def matches(self, received: str) -> bool:
        actual = self.token
        return len(received) == len(actual) and secrets.compare_digest(received, actual)

    def rotate(self) -> str:
        """Invalida el enlace anterior: el celular que estaba dentro se corta."""
        with self._lock:
            self._generation += 1
            self._token = secrets.token_urlsafe(12)
            self._jpeg = None
            return self._token

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def connect(self) -> int:
        with self._lock:
            self._connected += 1
            return self._generation

    def disconnect(self) -> None:
        with self._lock:
            self._connected = max(0, self._connected - 1)

    def publish(self, jpeg: bytes, generation: int) -> bool:
        if len(jpeg) < 4 or len(jpeg) > MAX_JPEG or not jpeg.startswith(b"\xff\xd8"):
            return False
        now = time.perf_counter()
        t_ns = time.perf_counter_ns()  # llegada al PC: reloj comparable con los workers
        with self._lock:
            if generation != self._generation:
                return False
            self._jpeg, self._jpeg_t = jpeg, now
            self._arrivals = [t for t in self._arrivals if now - t <= 1.0] + [now]
            sink = self._sink
        if sink is not None:
            try:
                sink.put_nowait((t_ns, jpeg))
            except queue.Full:
                self.dropped_to_sink += 1  # el proceso de captura va atrasado: gana el mas nuevo
        return True

    def set_size(self, width: int, height: int) -> None:
        with self._lock:
            self._size = (width, height)

    def attach(self, sink) -> None:
        with self._lock:
            self._sink = sink

    def detach(self, sink=None) -> None:
        with self._lock:
            if sink is None or self._sink is sink:
                self._sink = None

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            recent = [t for t in self._arrivals if now - t <= 1.0]
            return {
                "connected": self._connected > 0,
                "streaming": self._jpeg is not None and now - self._jpeg_t < 2.0,
                "fps": float(len(recent)),
                "size": self._size,
                "in_session": self._sink is not None,
            }


hub = PhoneHub()


# --- red, certificado y QR ------------------------------------------------------------------


def lan_ip() -> str | None:
    """La IP por la que el celular, en la misma Wi-Fi, llega a esta maquina (la de la ruta por
    defecto; las de VPN, maquinas virtuales o compartir internet no sirven)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # no envia paquetes: solo pregunta que interfaz usaria
        ip = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if ip.startswith("127.") else ip


def ensure_certificate(ip: str) -> tuple[Path, Path]:
    """Certificado autofirmado con la IP en el SAN. Se regenera solo si la IP cambia."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path, key_path, mark = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "ip.txt"
    if cert_path.is_file() and key_path.is_file() and mark.is_file() and mark.read_text() == ip:
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Padel Pose en vivo (local)")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=825))  # tope que aceptan iOS/macOS
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    mark.write_text(ip)
    return cert_path, key_path


def qr_svg(text: str) -> str:
    import io

    import segno

    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="svg", scale=1, border=2, dark="#161616", light="#ffffff",
                                    xmldecl=False, svgns=True, nl=False, omitsize=True)  # viewBox: escala sin pixelarse
    return buf.getvalue().decode()


def _free_port() -> int:
    for port in range(FIRST_PORT, FIRST_PORT + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise PhoneError("No hay un puerto libre entre 8443 y 8462 para recibir la camara del celular.")


# --- el relay (servidor HTTPS en la red local) ----------------------------------------------


def create_relay_app():
    """App minima que ve el celular: su pagina, sus estaticos y la entrada de frames."""
    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/{token}", response_class=HTMLResponse)
    def phone_page(request: Request, token: str):
        if not hub.matches(token):
            return HTMLResponse("Este enlace ya no sirve. Escanea el QR nuevo en la computadora.", status_code=404)
        return templates.TemplateResponse(request, "phone.html", {"token": token},
                                          headers={"Cache-Control": "no-store"})

    @app.websocket("/{token}/ws")
    async def camera(ws: WebSocket, token: str):
        if not hub.matches(token):
            await ws.close(code=1008)
            return
        await ws.accept()
        generation = hub.connect()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect" or hub.generation() != generation:
                    break
                if msg.get("bytes"):
                    hub.publish(msg["bytes"], generation)
                elif msg.get("text", "").startswith("size "):
                    parts = msg["text"].split()
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        hub.set_size(int(parts[1]), int(parts[2]))
        except Exception:
            pass  # el celular cerro la pagina o se fue de la red
        finally:
            hub.disconnect()
            try:
                await ws.close()
            except Exception:
                pass

    @app.post("/{token}/frame")
    async def frame_post(request: Request, token: str):
        """Plan B si el WebSocket seguro no abre (algunas versiones de Safari con certificados
        autofirmados): el celular manda cada JPEG con un POST normal."""
        if not hub.matches(token):
            return JSONResponse({"ok": False}, status_code=404)
        generation = hub.connect()
        try:
            ok = hub.publish(await request.body(), generation)
        finally:
            hub.disconnect()
        return {"ok": ok}

    return app


class Relay:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server = None
        self._ip: str | None = None
        self._port: int | None = None

    def ensure(self) -> str:
        """Arranca (una vez) el relay y devuelve la URL que va en el QR."""
        ip = lan_ip()
        if ip is None:
            raise PhoneError("Esta computadora no tiene direccion en la red local. "
                             "Conectala al mismo Wi-Fi que el celular.")
        with self._lock:
            s = self._server
            if s is not None and s.started and not s.should_exit and self._ip == ip:
                return self.url
            if s is not None:
                s.should_exit = True
            cert, key = ensure_certificate(ip)
            port = _free_port()
            import uvicorn

            config = uvicorn.Config(create_relay_app(), host="0.0.0.0", port=port, ssl_certfile=str(cert),
                                    ssl_keyfile=str(key), log_level="warning", access_log=False,
                                    ws_max_size=MAX_JPEG + 1024)
            server = uvicorn.Server(config)
            threading.Thread(target=server.run, name="relay-celular", daemon=True).start()
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            else:
                server.should_exit = True
                raise PhoneError("No arranco la recepcion de la camara. Si Windows pregunta si Python "
                                 "puede aceptar conexiones, pulsa Permitir (redes privadas) y recarga.")
            self._server, self._ip, self._port = server, ip, port
            return self.url

    @property
    def url(self) -> str:
        return f"https://{self._ip}:{self._port}/{hub.token}"

    def stop(self) -> None:
        with self._lock:
            if self._server is not None:
                self._server.should_exit = True
                self._server = None


relay = Relay()
