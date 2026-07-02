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

from ..config import Config, ZonesConfig
from ..statestore import InMemoryStateStore
from ..zones import ZoneError, ZoneMap

log = logging.getLogger("audience_tracker.api")


def _build_zone_map(cfg: Config) -> ZoneMap:
    try:
        return ZoneMap(cfg.zones)
    except ZoneError as exc:
        log.warning("Zone API disabled: %s", exc)
        return ZoneMap(ZonesConfig(enabled=False))


def create_app(cfg: Config | None = None, store: InMemoryStateStore | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    store = store or InMemoryStateStore()
    zones = _build_zone_map(cfg)

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
            source = open_source(
                cfg, simulator=built["simulator"], ingest_source=app.state.ingest_source
            )
            pipeline.start_background(source)
            app.state.pipeline = pipeline
            log.info(
                "Pipeline running (%s backend, source=%s)",
                built["backend"],
                cfg.pipeline.source,
            )
        try:
            yield
        finally:
            if app.state.pipeline is not None:
                app.state.pipeline.stop()

    app = FastAPI(title="Audience Tracking System", version="1.0.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.zones = zones
    app.state.pipeline = None
    app.state.ingest_source = None

    # When the frame source is the ingestion queue, create it and expose the
    # /ingest WebSocket so the venue Capture Agent can stream frames in.
    if cfg.pipeline.source == "ingest":
        from ..ingestion.frame_source import QueueFrameSource
        from ..ingestion.server import register_ingest_route

        app.state.ingest_source = QueueFrameSource(maxsize=cfg.ingest.queue_size)
        register_ingest_route(app, app.state.ingest_source, token=cfg.ingest.token)

    # -------------------------------------------------------------- #
    # REST
    # -------------------------------------------------------------- #
    @app.get("/health")
    async def health() -> dict:
        pipeline = app.state.pipeline
        return {
            "status": "ok",
            "pipeline": pipeline is not None,
            # False after the run loop gives up on a persistent fault — the
            # operator's signal that tracking died while the API stayed up.
            "pipeline_running": bool(pipeline is not None and pipeline.running),
        }

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

    @app.get("/api/zones")
    async def get_zones() -> dict:
        return app.state.zones.as_dict()

    @app.get("/api/zones/counts")
    async def get_zone_counts() -> dict:
        return store.get_zone_counts()

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
                if cfg.api.ws_heartbeat_interval_s > 0:
                    try:
                        message = await asyncio.wait_for(
                            queue.get(), timeout=cfg.api.ws_heartbeat_interval_s
                        )
                    except asyncio.TimeoutError:
                        await websocket.send_json(
                            {"type": "snapshot", "data": store.get_snapshot()}
                        )
                        continue
                else:
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
        endpoints = [
            "/api/audience",
            "/api/audience/{gid}",
            "/api/stats",
            "/api/zones",
            "/api/zones/counts",
            "/api/snapshot",
            "/metrics",
            "/ws",
            "/video",
            "/health",
        ]
        if app.state.ingest_source is not None:
            endpoints += ["/ingest", "/ingest/stats"]
        return {"service": "Audience Tracking System", "version": "1.0.0", "endpoints": endpoints}

    return app


# Importable ASGI app for `uvicorn audience_tracker.api.app:app`, configured
# from defaults + env. The CLI builds its own app with an explicit Config.
app = create_app()
