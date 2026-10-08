"""Servidor web local (FastAPI). Regla heredada de live_camera: ni una linea de logica cientifica.

Rutas:
    GET  /                                        pagina de dos pantallas
    GET  /api/sources                             videos disponibles
    POST /api/sessions                            lanzar una sesion en vivo
    GET  /api/sessions                            historial
    GET  /api/sessions/{id}                       estado actual
    POST /api/sessions/{id}/stop                  detener (parada ordenada)
    GET  /api/sessions/{id}/events                progreso y metricas en vivo (SSE)
    GET  /api/sessions/{id}/stream/{modelo}.mjpg  video anotado en vivo (MJPEG)
    GET  /api/sessions/{id}/files/{modelo}/{tipo} video H.264 o CSV al terminar
    GET  /api/sessions/{id}/files/recursos        serie temporal de CPU/RAM/GPU
    GET  /api/phone                               estado del iPhone + enlace (arranca el relay)
    GET  /api/phone/qr.svg                        QR del enlace para escanear con el iPhone
    POST /api/phone/rotate                        enlace nuevo (el anterior deja de servir)
    GET  /api/phone/frame                         ultimo JPEG recibido (vista previa)

El celular NO entra a este servidor (solo escucha en 127.0.0.1): entra al relay HTTPS de
webapp/phone.py, en la IP de la red local, que solo sirve su pagina y recibe frames.
"""
from __future__ import annotations

import json
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from webapp.sessions import MODELS, RUNS_DIR, get_manager

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
VIDEO_DIRS = [ROOT / "data" / "processed" / "video_demo"]
HEARTBEAT_S = 15.0  # sin esto, un cliente colgado mantendria viva la conexion para siempre
NO_CACHE = {"Cache-Control": "no-cache"}

templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def list_videos() -> list[dict]:
    videos = []
    for d in VIDEO_DIRS:
        for p in sorted(d.glob("*.mp4")):
            if p.name.endswith("_anotado.mp4"):
                continue  # salidas de run_inference_video.py, no fuentes
            videos.append({"name": p.name, "path": str(p)})
    return videos


class SourceSpec(BaseModel):
    kind: Literal["video", "camera", "phone"]
    name: str | None = None      # nombre de un video de /api/sources
    index: int = Field(0, ge=0, le=9)


class SessionRequest(BaseModel):
    source: SourceSpec
    models: list[Literal["yolo", "mediapipe"]] = Field(default_factory=lambda: list(MODELS))
    mode: Literal["sync", "threaded"] = "threaded"
    loops: int = Field(15, ge=0, le=500)            # 0 = repetir hasta detener
    pace_fps: float = Field(30.0, ge=0, le=240)      # 0 = FPS nativos del video
    imgsz: int = Field(384, ge=160, le=1280, multiple_of=32)
    conf: float = Field(0.25, gt=0, lt=1)
    mp_variant: Literal["lite", "full", "heavy"] = "full"
    # que puntos se dibujan (y promedian en la confianza); la inferencia es la misma
    keypoints: Literal["brazo", "brazo_raqueta", "todos"] = "brazo_raqueta"
    preview: bool = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    get_manager().shutdown()  # no dejar subprocesos huerfanos al cerrar el servidor
    from webapp.phone import relay

    relay.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Padel Pose en vivo", lifespan=lifespan, docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html", headers=NO_CACHE)

    @app.get("/api/sources")
    def sources() -> dict:
        return {"videos": [{"name": v["name"]} for v in list_videos()]}

    @app.post("/api/sessions")
    def create_session(req: SessionRequest) -> dict:
        if not req.models:
            raise HTTPException(422, "Elige al menos un modelo.")
        source = {"kind": req.source.kind}
        if req.source.kind == "video":
            match = [v for v in list_videos() if v["name"] == req.source.name]
            if not match:
                raise HTTPException(422, f"Video no disponible: {req.source.name}")
            source["path"] = match[0]["path"]
            source["name"] = match[0]["name"]
        elif req.source.kind == "camera":
            source["index"] = req.source.index
        config = req.model_dump()
        config["source"] = source
        config["models"] = [m for m in MODELS if m in req.models]  # orden estable: yolo, mediapipe
        return get_manager().submit(config).snapshot()

    # --- iPhone por la red local -----------------------------------------------------------

    @app.get("/api/phone")
    def phone_status() -> dict:
        from webapp.phone import PhoneError, hub, relay

        try:
            url, error = relay.ensure(), None
        except PhoneError as e:
            url, error = None, str(e)
        return {"url": url, "error": error, **hub.status()}

    @app.get("/api/phone/qr.svg")
    def phone_qr() -> Response:
        from webapp.phone import PhoneError, qr_svg, relay

        try:
            url = relay.ensure()
        except PhoneError as e:
            return Response(str(e), status_code=503, media_type="text/plain")
        return Response(qr_svg(url), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    @app.post("/api/phone/rotate")
    def phone_rotate() -> dict:
        from webapp.phone import hub

        hub.rotate()
        return phone_status()

    @app.get("/api/phone/frame")
    def phone_frame() -> Response:
        from webapp.phone import hub

        jpeg = hub.latest()
        if jpeg is None:
            return Response(status_code=204)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/sessions")
    def list_sessions() -> list[dict]:
        return get_manager().list()

    def _session(session_id: str):
        session = get_manager().get(session_id)
        if session is None:
            raise HTTPException(404, f"No existe la sesion {session_id}")
        return session

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        return _session(session_id).snapshot()

    @app.post("/api/sessions/{session_id}/stop")
    def stop_session(session_id: str) -> dict:
        _session(session_id)
        return {"stopping": get_manager().stop(session_id)}

    @app.get("/api/sessions/{session_id}/events")
    def session_events(session_id: str) -> StreamingResponse:
        _session(session_id)
        manager = get_manager()

        def generate():
            q = manager.subscribe(session_id)
            try:
                while True:
                    try:
                        snap = q.get(timeout=HEARTBEAT_S)
                    except queue.Empty:
                        yield ": latido\n\n"  # comentario SSE: mantiene viva la conexion
                        continue
                    if snap is None:
                        yield "event: end\ndata: {}\n\n"
                        return
                    yield f"data: {json.dumps(snap)}\n\n"
            finally:
                manager.unsubscribe(session_id, q)

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={**NO_CACHE, "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    @app.get("/api/sessions/{session_id}/stream/{model}.mjpg")
    def stream(session_id: str, model: Literal["yolo", "mediapipe"]) -> StreamingResponse:
        """MJPEG: el navegador lo pinta en un <img> sin JavaScript ni codecs. Se cierra solo cuando
        la sesion termina (el <img> se queda con el ultimo frame)."""
        _session(session_id)
        manager = get_manager()

        def generate():
            seq = 0
            while True:
                got = manager.latest_frame(session_id, model, seq, timeout=1.0)
                if got is None:
                    session = manager.get(session_id)
                    if session is None or session.state.is_final:
                        return
                    continue
                seq, jpeg = got
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers=NO_CACHE)

    def _inside_runs(path: str | None) -> Path:
        if not path:
            raise HTTPException(404, "El archivo no existe.")
        p = Path(path).resolve()
        if RUNS_DIR.resolve() not in p.parents or not p.is_file():
            raise HTTPException(404, "El archivo no existe.")
        return p

    @app.get("/api/sessions/{session_id}/files/recursos")
    def resources_file(session_id: str) -> FileResponse:
        session = _session(session_id)
        p = _inside_runs(str(Path(session.out_dir) / "recursos.csv") if session.out_dir else None)
        return FileResponse(p, media_type="text/csv", filename=f"recursos_{session_id}.csv")

    @app.get("/api/sessions/{session_id}/files/{model}/{kind}")
    def model_file(session_id: str, model: Literal["yolo", "mediapipe"],
                   kind: Literal["video", "csv"]) -> FileResponse:
        session = _session(session_id)
        files = (session.procs.get(model) or {}).get("files") or {}
        p = _inside_runs(files.get(kind))
        if kind == "video":
            return FileResponse(p, media_type="video/mp4")
        return FileResponse(p, media_type="text/csv", filename=f"{model}_{session_id}.csv")

    return app


app = create_app()
