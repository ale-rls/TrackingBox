# TouchDesigner integration scripts

Drop-in Python for wiring TouchDesigner to a local `audience-tracker serve`
process. Full setup is in the runbook:
[Run with TouchDesigner](../docs/touchdesigner.md).

These run **inside TouchDesigner's Python** (they reference `op()`), so they are
templates/components — they are intentionally not imported or unit-tested by the
package.

| File | Mode | Purpose |
|---|---|---|
| `td_receive_state.py` | A & B | Callbacks DAT for a WebSocket DAT on `/ws`. Keeps a Table DAT `audience` (`gid, visible, cx, cy, x1…y2`) in sync with live GIDs. |
| `td_send_frames.py` | B only | Encodes a TOP to JPEG and streams it to `/ingest`. Needs `opencv-python` in TD's Python. Not needed if the service opens the camera itself (Mode A). |

**Mode A (recommended):** the service owns the camera; TouchDesigner only consumes
output — a **Video Stream In TOP** on `http://localhost:8000/video` for the overlay
and a **WebSocket DAT** on `ws://localhost:8000/ws` (with `td_receive_state.py`) for
the GID data. No frame sending required.

**Mode B:** TouchDesigner sends frames (`td_send_frames.py` → `/ingest`); run the
service with `--source ingest`.

Only GIDs cross the boundary — tracker ids never leave the service.
