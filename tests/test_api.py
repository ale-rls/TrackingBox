"""API integration test: boots the app with the mock pipeline and hits every
documented endpoint, including the WebSocket. Skipped if FastAPI is absent.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

pytest.importorskip("fastapi")
pytest.importorskip("numpy")
from fastapi.testclient import TestClient  # noqa: E402

from audience_tracker.api.app import create_app  # noqa: E402
from audience_tracker.config import Config  # noqa: E402


def _client() -> TestClient:
    cfg = Config()
    cfg.pipeline.backend = "mock"
    cfg.pipeline.mock_people = 6
    cfg.pipeline.run_pipeline = True
    cfg.pipeline.stream_overlay = False
    cfg.logging.enabled = False
    return TestClient(create_app(cfg))


def _wait_for_people(client: TestClient, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    stats = {}
    while time.time() < deadline:
        stats = client.get("/api/stats").json()
        if stats.get("total_people_seen", 0) >= 1:
            return stats
        time.sleep(0.1)
    return stats


def test_endpoints_and_websocket():
    with _client() as client:
        assert client.get("/health").json()["pipeline"] is True

        stats = _wait_for_people(client)
        assert stats["total_people_seen"] >= 1

        audience = client.get("/api/audience").json()
        assert isinstance(audience, list)
        for entry in audience:  # GID-only contract
            assert set(entry.keys()) == {
                "gid",
                "visible",
                "center",
                "bbox",
                "floor",
                "floor_valid",
                "zone",
            }

        snap = client.get("/api/snapshot").json()
        assert "timestamp" in snap and "people" in snap

        metrics = client.get("/metrics").json()
        assert "fps" in metrics and "latency_ms" in metrics

        zones = client.get("/api/zones").json()
        assert set(zones.keys()) == {"enabled", "default_zone", "zones"}
        assert client.get("/api/zones/counts").json() == {}

        # Unknown GID -> 404.
        assert client.get("/api/audience/99999999").status_code == 404

        # Known GID -> detail.
        if snap["people"]:
            gid = snap["people"][0]["gid"]
            member = client.get(f"/api/audience/{gid}").json()
            assert member["gid"] == gid
            assert "duration_seen_seconds" in member

        # WebSocket primes with a snapshot message.
        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["type"] == "snapshot"
            assert "people" in first["data"]


def test_zone_endpoints_return_configured_definitions():
    cfg = Config()
    cfg.pipeline.run_pipeline = False
    cfg.zones.enabled = True
    cfg.zones.default_zone = "outside"
    cfg.zones.zones = [{"id": "stage", "label": "Stage", "rect": [0, 0, 1, 1]}]

    with TestClient(create_app(cfg)) as client:
        zones = client.get("/api/zones").json()

    assert zones["enabled"] is True
    assert zones["default_zone"] == "outside"
    assert zones["zones"] == [
        {
            "id": "stage",
            "label": "Stage",
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    ]


if __name__ == "__main__":
    test_endpoints_and_websocket()
    print("ok")
