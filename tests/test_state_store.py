"""Tests for the shared state store read shapes and change diffing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker.models import AudienceState  # noqa: E402
from audience_tracker.statestore import InMemoryStateStore  # noqa: E402


def _state(gid, visible=True, x=10):
    return AudienceState(
        gid=gid, visible=visible, bbox=(x, 0, x + 10, 20), center=(x + 5, 10),
        confidence=0.9, first_seen=0.0, last_seen=1.0, duration_seen_seconds=1.0,
    )


def test_read_shapes_match_spec():
    store = InMemoryStateStore()
    store.publish([_state(17)], {"active_people": 1, "total_people_seen": 1})

    active = store.get_active()
    assert active == [
        {
            "gid": 17,
            "visible": True,
            "center": [15, 10],
            "bbox": [10, 0, 20, 20],
            "floor": None,
            "floor_valid": False,
        }
    ]

    member = store.get_member(17)
    assert member["gid"] == 17 and "duration_seen_seconds" in member
    assert "track_id" not in member  # GID-only, never tracker ids

    assert store.get_stats() == {"active_people": 1, "total_people_seen": 1}

    snap = store.get_snapshot()
    assert snap["active_people"] == 1 and len(snap["people"]) == 1 and "timestamp" in snap

    assert store.get_member(999) is None


def test_publish_is_idempotent_for_reads():
    store = InMemoryStateStore()
    store.publish([_state(1), _state(2, x=50)], {"active_people": 2, "total_people_seen": 2})
    store.publish([_state(1), _state(2, x=50)], {"active_people": 2, "total_people_seen": 2})
    assert len(store.get_active()) == 2


if __name__ == "__main__":
    test_read_shapes_match_spec()
    test_publish_is_idempotent_for_reads()
    print("ok")
