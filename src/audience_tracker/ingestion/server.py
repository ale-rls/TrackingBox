"""Modal Ingestion Service: the WebSocket endpoint that receives the Capture
Agent's frame stream and feeds the bounded :class:`QueueFrameSource`.

Responsibilities (spec "Modal Ingestion Service"): accept the incoming stream,
authenticate it, validate + decode packets, keep frames ordered, buffer a small
number, and forward them to the Tracking Pipeline via the queue. It performs no
computer-vision inference.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .frame_source import QueueFrameSource
from .packet import InvalidPacket, decode_packet
from ..models import Frame

log = logging.getLogger("audience_tracker.ingest")

# A decoder turns raw JPEG bytes into a decoded BGR image (numpy array) or None.
Decoder = Callable[[bytes], Optional[Any]]


def default_decoder(jpeg_bytes: bytes) -> Optional[Any]:
    """Decode JPEG bytes to a BGR image with OpenCV (Modal/venue runtime)."""
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def _authorized(websocket: WebSocket, token: str) -> bool:
    if not token:
        return True  # auth disabled (dev only)
    header = websocket.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header[len("Bearer ") :].strip() == token
    # Fallback for clients that cannot set headers (e.g. browsers): ?token=...
    return websocket.query_params.get("token") == token


def register_ingest_route(
    app: FastAPI,
    source: QueueFrameSource,
    token: str = "",
    decode: Optional[Decoder] = None,
) -> None:
    """Attach the ``/ingest`` WebSocket and ``/ingest/stats`` to ``app``."""
    decode = decode or default_decoder

    @app.websocket("/ingest")
    async def ingest(websocket: WebSocket) -> None:
        if not _authorized(websocket, token):
            # Reject the handshake (HTTP 403) before accepting — unauthenticated
            # clients are rejected (spec "Security").
            await websocket.close(code=1008)
            log.warning("Rejected unauthenticated ingest connection")
            return
        await websocket.accept()
        # New connection => new frame-id sequence (the agent may have restarted).
        source.begin_session()
        log.info("Capture Agent connected to /ingest")
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None:
                    continue  # ignore text / keepalive frames
                try:
                    pkt = decode_packet(data)
                    image = decode(pkt["jpeg_bytes"])
                    if image is None:
                        raise InvalidPacket("JPEG failed to decode")
                    source.push(
                        Frame(
                            image=image,
                            timestamp=pkt["timestamp"],
                            frame_id=pkt["frame_id"],
                            width=pkt["width"],
                            height=pkt["height"],
                        )
                    )
                except InvalidPacket as exc:
                    log.debug("Dropping invalid packet: %s", exc)
                    continue
                except Exception as exc:  # one bad frame must not kill the stream
                    log.debug("Dropping frame after error: %s", exc)
                    continue
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Ingest connection closed: %s", exc)
        finally:
            log.info("Capture Agent disconnected from /ingest")

    @app.get("/ingest/stats")
    async def ingest_stats() -> dict:
        return source.status()
