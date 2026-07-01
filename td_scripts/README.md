# TouchDesigner integration scripts

Drop-in Python for wiring TouchDesigner to a local `audience-tracker serve`
process. Full setup is in the runbook:
[Run with TouchDesigner](../docs/touchdesigner.md).

These run inside TouchDesigner's Python, so they are templates/components and are
not imported by the Python package.

| File | Purpose |
|---|---|
| `td_launch_tracker.py` | Reads a preset Table DAT and launches/stops the local tracker service. |
| `td_receive_state.py` | Callbacks DAT for a WebSocket DAT on `/ws`. Keeps a Table DAT `audience` (`gid, visible, cx, cy, x1...y2, floor_x, floor_y, floor_valid`) in sync with live GIDs. |

The service owns the camera; TouchDesigner consumes output with:

* **Video Stream In TOP**: `http://localhost:8000/video`
* **WebSocket DAT**: `ws://localhost:8000/ws`
* **Callbacks DAT**: `td_receive_state.py`

`td_launch_tracker.py` expects a Table DAT named `tracker_presets`. Add one row
per camera/server preset:

```text
name,source,backend,device,port,reid,confidence,image_size,debug
HDMI USB Camera,0,real,cuda,8000,0,0.15,1280,0
Mock Test,0,mock,auto,8000,0,0.30,960,0
```

Optional helper DATs:

* `selected_preset` - first cell contains the preset name to launch. If omitted,
  the first preset row is used.
* `tracker_status` - receives one-line launch/stop status messages.

Button callbacks:

```python
mod('td_launch_tracker').launch_selected()
mod('td_launch_tracker').stop()
mod('td_launch_tracker').restart_selected()
```

Only GIDs cross the boundary; tracker ids never leave the service.

For top-down floor positions, calibrate the venue camera and use `floor_x` /
`floor_y` instead of `cx` / `cy`. See
[Floor Projection Calibration](../docs/floor_projection.md).
