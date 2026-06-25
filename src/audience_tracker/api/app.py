"""FastAPI application exposing the audience state.

Endpoints (per spec):
  REST   GET /api/audience            active audience (GID + visible + center)
         GET /api/audience/{gid}      one member (detail)
         GET /api/stats               active_people / total_people_seen
         GET /api/snapshot            full snapshot
         GET /metrics                 runtime metrics
  WS     /ws                          live audience-state change stream
  Extra  GET /video                   MJPEG overlay stream
         GET /health                  liveness

The WebSocket is the primary integration interface for external systems. Only
GIDs are ever exposed — tracker IDs stay internal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import Config
from ..statestore import InMemoryStateStore

log = logging.getLogger("audience_tracker.api")


def create_app(cfg: Config | None = None, store: InMemoryStateStore | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    store = store or InMemoryStateStore()

    # -------------------------------------------------------------- #
    # Lifecycle: optionally run the tracking pipeline in this process.
    # -------------------------------------------------------------- #
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        store.attach_loop(asyncio.get_running_loop())
        if cfg.pipeline.run_pipeline:
            from ..factory import build_pipeline, open_source

            built = build_pipeline(cfg, store=store, num_people=cfg.pipeline.mock_people)
            pipeline = built["pipeline"]
            camera = open_source(cfg, simulator=built["simulator"])
            pipeline.start_background(camera)
            app.state.pipeline = pipeline
            log.info("Pipeline running (%s backend)", built["backend"])
        try:
            yield
        finally:
            if app.state.pipeline is not None:
                app.state.pipeline.stop()

    app = FastAPI(title="Audience Tracking System", version="1.0.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.pipeline = None

    # -------------------------------------------------------------- #
    # REST
    # -------------------------------------------------------------- #
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "pipeline": app.state.pipeline is not None}

    @app.get("/api/audience")
    async def get_audience() -> list:
        return store.get_active()

    @app.get("/api/audience/{gid}")
    async def get_audience_member(gid: int) -> dict:
        member = store.get_member(gid)
        if member is None:
            raise HTTPException(status_code=404, detail=f"GID {gid} not found")
        return member

    @app.get("/api/stats")
    async def get_stats() -> dict:
        return store.get_stats()

    @app.get("/api/snapshot")
    async def get_snapshot() -> dict:
        return store.get_snapshot()

    @app.get("/metrics")
    async def get_metrics() -> JSONResponse:
        return JSONResponse(store.get_metrics())

    # -------------------------------------------------------------- #
    # WebSocket — primary integration interface
    # -------------------------------------------------------------- #
    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = store.subscribe()
        try:
            # Prime the client with the current full snapshot.
            await websocket.send_json({"type": "snapshot", "data": store.get_snapshot()})
            while True:
                message = await queue.get()
                await websocket.send_json(message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover
            log.debug("WebSocket closed: %s", exc)
        finally:
            store.unsubscribe(queue)

    # -------------------------------------------------------------- #
    # Overlay video stream (MJPEG)
    # -------------------------------------------------------------- #
    @app.get("/video")
    async def video() -> StreamingResponse:
        boundary = "frame"

        async def gen():
            while True:
                jpeg = store.get_frame()
                if jpeg:
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                await asyncio.sleep(1 / 25)

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}"
        )

    @app.get("/")
    async def index() -> dict:
        return {
            "service": "Audience Tracking System",
            "version": "1.0.0",
            "endpoints": [
                "/api/audience",
                "/api/audience/{gid}",
                "/api/stats",
                "/api/snapshot",
                "/metrics",
                "/ws",
                "/video",
                "/health",
            ],
        }

    return app


# Importable ASGI app for `uvicorn audience_tracker.api.app:app`, configured
# from defaults + env. The CLI builds its own app with an explicit Config.
app = create_app()
