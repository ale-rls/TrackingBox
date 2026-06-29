"""Bounded, drop-oldest ingestion queue (Video Ingestion spec "Frame Queue")."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker.ingestion.frame_source import QueueFrameSource  # noqa: E402
from audience_tracker.models import Frame  # noqa: E402


def _frame(fid: int) -> Frame:
    return Frame(image=fid, timestamp=float(fid), frame_id=fid, width=1, height=1)


def test_fifo_ordering():
    q = QueueFrameSource(maxsize=5)
    for i in range(1, 4):
        assert q.push(_frame(i))
    assert [q.next_frame(timeout=0).frame_id for _ in range(3)] == [1, 2, 3]


def test_drops_oldest_when_full():
    q = QueueFrameSource(maxsize=2)
    for i in range(1, 5):  # 1,2,3,4 into a depth-2 queue
        q.push(_frame(i))
    # Oldest (1,2) dropped; newest two (3,4) remain, still ordered.
    assert [q.next_frame(timeout=0).frame_id for _ in range(2)] == [3, 4]
    st = q.status()
    assert st["received"] == 4
    assert st["dropped_overflow"] == 2
    assert st["depth"] == 0


def test_out_of_order_and_duplicates_rejected():
    q = QueueFrameSource(maxsize=5)
    assert q.push(_frame(10))
    assert not q.push(_frame(10))   # duplicate id
    assert not q.push(_frame(5))    # older id
    assert q.push(_frame(11))       # newer id accepted
    st = q.status()
    assert st["dropped_out_of_order"] == 2
    assert st["last_frame_id"] == 11
    assert [q.next_frame(timeout=0).frame_id for _ in range(2)] == [10, 11]


def test_begin_session_resets_ordering_for_restarted_agent():
    q = QueueFrameSource(maxsize=5)
    for i in range(1, 11):  # previous agent session reached frame_id 10
        q.push(_frame(i))
    assert q.status()["last_frame_id"] == 10
    # Agent restarts -> frame_id sequence starts over at 1. Without a session
    # reset these would all be dropped as out-of-order; with it they flow again.
    q.begin_session()
    assert q.status()["last_frame_id"] == -1
    assert q.status()["depth"] == 0          # stale buffered frames cleared
    assert q.push(_frame(1))
    assert q.push(_frame(2))
    assert [q.next_frame(timeout=0).frame_id for _ in range(2)] == [1, 2]


def test_timeout_returns_none_when_empty():
    q = QueueFrameSource(maxsize=3)
    assert q.next_frame(timeout=0.01) is None
    assert q.exhausted is False  # empty != finished for a live source


def test_release_marks_exhausted_after_drain():
    q = QueueFrameSource(maxsize=3)
    q.push(_frame(1))
    q.release()
    assert q.exhausted is False          # still has a buffered frame
    assert q.next_frame(timeout=0).frame_id == 1
    assert q.exhausted is True           # closed and drained
    assert q.next_frame(timeout=0) is None


if __name__ == "__main__":
    test_fifo_ordering()
    test_drops_oldest_when_full()
    test_out_of_order_and_duplicates_rejected()
    test_timeout_returns_none_when_empty()
    test_release_marks_exhausted_after_drain()
    print("ok")
