"""Ingestion WebSocket: auth + receive/validate/enqueue.

A fake decoder is injected so the test needs no OpenCV (the real service decodes
JPEG with cv2). Skipped if FastAPI is absent.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from audience_tracker.ingestion.frame_source import QueueFrameSource  # noqa: E402
from audience_tracker.ingestion.packet import encode_packet  # noqa: E402
from audience_tracker.ingestion.server import _authorized, register_ingest_route  # noqa: E402


class _FakeWS:
    def __init__(self, headers=None, query=None):
        self.headers = headers or {}
        self.query_params = query or {}


def test_authorized_logic():
    assert _authorized(_FakeWS({"authorization": "Bearer secret"}), "secret") is True
    assert _authorized(_FakeWS({"authorization": "Bearer nope"}), "secret") is False
    assert _authorized(_FakeWS(), "secret") is False
    assert _authorized(_FakeWS(query={"token": "secret"}), "secret") is True
    assert _authorized(_FakeWS(), "") is True  # auth disabled


def _app(token="secret"):
    app = FastAPI()
    source = QueueFrameSource(maxsize=8)
    register_ingest_route(app, source, token=token, decode=lambda b: [[0, 0, 0]])
    return app, source


def test_unauthenticated_connection_rejected():
    app, source = _app()
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ingest?token=wrong"):
            pass
    assert source.status()["received"] == 0


def test_authenticated_frame_is_decoded_and_enqueued():
    app, source = _app()
    client = TestClient(app)
    pkt = encode_packet(7, time.time(), b"\xff\xd8jpeg\xff\xd9", 320, 240)
    with client.websocket_connect("/ingest?token=secret") as ws:
        ws.send_bytes(pkt)
        deadline = time.time() + 2.0
        while time.time() < deadline and source.status()["received"] == 0:
            time.sleep(0.02)
    st = source.status()
    assert st["received"] == 1
    assert st["last_frame_id"] == 7
    frame = source.next_frame(timeout=0)
    assert frame is not None and frame.frame_id == 7 and frame.timestamp > 0

    stats = client.get("/ingest/stats").json()
    assert stats["capacity"] == 8


if __name__ == "__main__":
    test_authorized_logic()
    print("ok")
