"""Tests for the Global Identity Manager — the system's source of truth.

Pure stdlib (no numpy / no ML deps). Runnable via ``pytest`` or directly:
``python tests/test_identity_manager.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker.config import IdentityConfig, ReIDConfig  # noqa: E402
from audience_tracker.identity import IdentityManager  # noqa: E402
from audience_tracker.models import Track  # noqa: E402

# Orthogonal appearance embeddings -> cosine 1.0 with self, 0.0 with each other.
EMB_A = [1.0, 0.0, 0.0, 0.0]
EMB_B = [0.0, 1.0, 0.0, 0.0]
EMB_A_NOISY = [0.95, 0.10, 0.05, 0.0]  # still very similar to A


def make_manager(**identity_overrides) -> IdentityManager:
    return IdentityManager(IdentityConfig(**identity_overrides), ReIDConfig())


def track(track_id: int, x: float = 0.0) -> Track:
    return Track(track_id=track_id, bbox=(x, 0.0, x + 10.0, 20.0), confidence=0.9)


# --------------------------------------------------------------------- #
def test_rule1_new_member_gets_unique_gid():
    mgr = make_manager()
    states = mgr.update([track(1)], {1: EMB_A}, now=0.0)
    assert len(states) == 1
    assert states[0].gid == 1
    assert states[0].visible is True


def test_two_members_get_distinct_gids():
    mgr = make_manager()
    states = mgr.update([track(1, x=0), track(2, x=100)], {1: EMB_A, 2: EMB_B}, now=0.0)
    gids = sorted(s.gid for s in states)
    assert gids == [1, 2]


def test_rule2_reid_recovers_gid_after_occlusion():
    mgr = make_manager()
    mgr.update([track(1)], {1: EMB_A}, now=0.0)        # GID 1 appears
    mgr.update([], {}, now=1.0)                          # occluded -> lost
    # Re-enters under a *different* tracker id, similar appearance.
    states = mgr.update([track(9)], {9: EMB_A_NOISY}, now=2.0)
    assert len(states) == 1
    assert states[0].gid == 1, "ReID should recover the original GID"
    assert mgr.counters()["recoveries"] == 1


def test_rule3_dissimilar_appearance_creates_new_gid():
    mgr = make_manager()
    mgr.update([track(1)], {1: EMB_A}, now=0.0)
    mgr.update([], {}, now=1.0)
    states = mgr.update([track(2)], {2: EMB_B}, now=2.0)  # different person
    assert states[0].gid == 2
    assert mgr.stats()["total_people_seen"] == 2


def test_rule4_gids_never_reused():
    mgr = make_manager(forget_timeout_seconds=0.0)  # forget immediately when lost
    mgr.update([track(1)], {1: EMB_A}, now=0.0)
    mgr.update([], {}, now=1.0)        # lost
    mgr.update([], {}, now=2.0)        # GC removes the forgotten identity
    assert mgr.get(1) is None          # dropped from memory
    # A new, identical-looking person still gets a fresh GID, not #1.
    states = mgr.update([track(5)], {5: EMB_A}, now=3.0)
    assert states[0].gid == 2
    assert mgr.stats()["total_people_seen"] == 2


def test_rule5_track_id_changes_gid_stable():
    mgr = make_manager()
    mgr.update([track(1)], {1: EMB_A}, now=0.0)
    mgr.update([], {}, now=0.5)                    # track 1 vanishes
    s2 = mgr.update([track(2)], {2: EMB_A}, now=1.0)  # ByteTrack relabel
    mgr.update([], {}, now=1.5)
    s3 = mgr.update([track(3)], {3: EMB_A}, now=2.0)  # relabel again
    assert s2[0].gid == 1 and s3[0].gid == 1


def test_recovery_window_expires():
    mgr = make_manager(lost_timeout_seconds=2.0)
    mgr.update([track(1)], {1: EMB_A}, now=0.0)
    mgr.update([], {}, now=1.0)                     # lost at t=1
    # Re-enters after the recovery window -> brand new GID.
    states = mgr.update([track(2)], {2: EMB_A}, now=10.0)
    assert states[0].gid == 2


def test_duration_accumulates_only_while_visible():
    mgr = make_manager()
    mgr.update([track(1)], {1: EMB_A}, now=0.0)
    mgr.update([track(1)], {1: EMB_A}, now=1.0)
    mgr.update([track(1)], {1: EMB_A}, now=2.0)
    st = mgr.get(1, now=2.0)
    assert abs(st.duration_seen_seconds - 2.0) < 1e-6


def test_stats_and_counters():
    mgr = make_manager()
    mgr.update([track(1, x=0), track(2, x=50)], {1: EMB_A, 2: EMB_B}, now=0.0)
    stats = mgr.stats()
    counters = mgr.counters()
    assert stats == {"active_people": 2, "total_people_seen": 2}
    assert counters["active_tracks"] == 2
    # One leaves -> active drops, total stays.
    mgr.update([track(1)], {1: EMB_A}, now=1.0)
    assert mgr.stats() == {"active_people": 1, "total_people_seen": 2}


def test_track_ids_never_in_public_state():
    mgr = make_manager()
    states = mgr.update([track(1)], {1: EMB_A}, now=0.0)
    public = states[0].detail()
    assert "track_id" not in public and "active_track_id" not in public


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:  # pragma: no cover - manual runner
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(funcs) - failures}/{len(funcs)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
