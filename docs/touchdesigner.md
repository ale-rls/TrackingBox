# Run with TouchDesigner

The local deployment runs the whole tracking pipeline on the Windows theater PC.
The service opens the USB camera directly and produces an annotated MJPEG stream;
TouchDesigner consumes the video and audience state over localhost.

ReID (OSNet) is part of the venue setup: it recovers a person's GID after a
full occlusion and appearance-checks the tracker when a track id reappears
after a miss. If `torchreid` won't build on a given machine, `--no-reid` still
runs detection + tracking with GIDs persisting across short misses — only
occlusion recovery is lost.

> Looking for the cloud GPU path instead? See [Run on Modal](modal.md).

## Prerequisites

* An NVIDIA GPU with a current driver (`nvidia-smi` to confirm).
* TouchDesigner installed.
* Python 3.10-3.12 installed separately from TouchDesigner. The service runs in
  its own venv, not inside TouchDesigner's Python.

## Install

From the repo root in a terminal, or by double-clicking the batch file:

```bat
scripts\install_windows.bat
```

Sanity-check anytime:

```bat
.venv\Scripts\audience-tracker doctor
```

## Start Manually

```bat
scripts\run_windows.bat
:: = .venv\Scripts\audience-tracker serve --backend real --device cuda --source 0 --port 8000
:: pass-through args work, e.g.: scripts\run_windows.bat --source 1 --no-reid --port 9000
```

In TouchDesigner:

* **Overlay video**: Video Stream In TOP, URL `http://localhost:8000/video`
* **Audience data**: WebSocket DAT, network address `localhost`, port `8000`,
  request `/ws`
* **Callbacks DAT**: `td_scripts/td_receive_state.py`
* **Output table**: Table DAT named `audience`

Enable the WebSocket DAT's auto-reconnect option if it is available in your
TouchDesigner build. The server sends a fresh snapshot on connect and refreshes
that snapshot periodically while the scene is quiet, so reconnects should heal
the table automatically.

The `audience` table columns are:

```text
gid visible cx cy x1 y1 x2 y2 floor_x floor_y floor_valid zone
```

`cx`, `cy`, and `bbox` are camera pixels. `floor_x` and `floor_y` are top-down
floor coordinates when calibration is enabled. `zone` is a configured floor
region when zones are enabled.

Sanity check:

```bat
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/audience
curl http://localhost:8000/api/zones/counts
```

## Launch From TouchDesigner

Create a Table DAT named `tracker_presets`:

```text
name,source,backend,device,port,reid,confidence,image_size,debug
HDMI USB Camera,0,real,cuda,8000,0,0.15,1280,0
Mock Test,0,mock,auto,8000,0,0.30,960,0
```

Add `td_scripts/td_launch_tracker.py` to a Text DAT named
`td_launch_tracker`. Then wire Button COMPs to:

```python
mod('td_launch_tracker').launch_selected()
mod('td_launch_tracker').stop()
mod('td_launch_tracker').restart_selected()
```

Optional DATs:

* `selected_preset` - first cell contains the preset name to launch. If omitted,
  the first preset row is used.
* `tracker_status` - receives one-line launch/stop status messages.

The launcher reads these columns:

| Column | Effect |
|---|---|
| `name` | Human-readable preset name. |
| `source` | Camera index or stream URL, for example `0`, `1`, or `rtsp://...`. |
| `backend` | `real`, `mock`, or `auto`. |
| `device` | `cuda`, `cpu`, or `auto`. |
| `port` | API/video/WebSocket port. |
| `reid` | `1` enables ReID; `0` adds `--no-reid`. |
| `confidence` | Sets `AT_DETECTOR_CONFIDENCE_THRESHOLD`. |
| `image_size` | Sets `AT_DETECTOR_IMAGE_SIZE`. |
| `debug` | Sets `AT_OVERLAY_DEBUG`. |

## Floor Projection

For top-down instancing, zones, and game logic, use `floor_x` and `floor_y`
instead of the camera pixel center. The service computes those coordinates from
the bottom-center of each bounding box, with optional fisheye correction.

Run the tracker with a local calibrated config:

```bat
.venv\Scripts\audience-tracker serve --config config.json
```

When calibration is disabled, the table still includes the floor columns, but
`floor_valid` is `0` and both floor values are `0`. After calibration,
`floor_valid` becomes `1`.

Calibration runbook: [Floor Projection Calibration](floor_projection.md).

## Floor Zones

Zones are named regions in the calibrated floor plane. They are configured in
`config.json`, appear per person as `zone`, and are counted at:

```bat
curl http://localhost:8000/api/zones/counts
```

In TouchDesigner, use:

* `td_receive_state.py` for the per-person `zone` column.
* `td_fetch_zone_counts.py` to fill a Table DAT named `zone_counts`.

Zone runbook: [Floor Zones](zones.md).

## Useful Options

| Flag / env | Effect |
|---|---|
| `--no-reid` | Detection + tracking only (fallback when torchreid won't install). |
| `--device cuda` | Force GPU. |
| `--source 0` / `--source rtsp://...` | Camera index or stream. |
| `AT_DETECTOR_IMAGE_SIZE=960` | Lower means more FPS; higher helps small/distant people. |
| `AT_DETECTOR_CONFIDENCE_THRESHOLD=0.35` | Detection confidence. |
| `AT_REID_SIMILARITY_THRESHOLD=0.6` | Min similarity to recover a GID after occlusion. |
| `AT_REID_REBIND_VETO_THRESHOLD=0.35` | Below this, a returning track id is treated as a different person. |
| `AT_OVERLAY_DEBUG=true` | Draw confidence under each GID. |

Installing ReID (if not done by the installer):

```bat
pip install -e ".[reid]"
```

Then set the preset `reid` column to `1`.

## Troubleshooting

Run `audience-tracker doctor` first; it pinpoints most setup issues.

| Symptom | Likely cause | Fix |
|---|---|---|
| `doctor`: "PyTorch installed but CUDA NOT available" | CPU-only torch wheel | Re-run `scripts\install_windows.bat`, or reinstall torch from the CUDA index. |
| Low FPS, GPU idle in Task Manager | Running on CPU torch | Confirm with `doctor`; expect CUDA to be available. |
| Installer: "No suitable Python found" | Python missing or 3.13+ only | Install Python 3.11 and tick "Add to PATH". |
| `Could not open video source: '0'` | Wrong camera index or camera in use | Try `--source 1`, `2`, etc.; close other apps using the camera. |
| Overlay freezes for a moment, then resumes | Camera hiccup | Nothing to do — the service retries and reopens the device automatically (watch the logs). |
| API responds but tracking is frozen | Pipeline thread died after a persistent fault | `curl http://localhost:8000/health` — `pipeline_running: false` confirms it; check the service logs and restart. |
| Video Stream In TOP stays black | Service down or wrong URL | Check `http://localhost:8000/health`; URL must be `http://localhost:8000/video`. |
| WebSocket DAT will not connect | Wrong address/port or firewall | Use `localhost` / `8000` / `/ws`; allow Python through Windows Firewall. |
| Audience table stops refreshing | WebSocket DAT dropped and did not reconnect | Enable auto-reconnect on the WebSocket DAT, then pulse Active off/on. The server resends a full snapshot after reconnect. |
| `pip install .[reid]` fails | `torchreid` build issue on Windows | Keep `reid` set to `0`, or install `deep-person-reid` from source. |
| Small/distant people missed | Detection input too low-res | Raise `image_size` to `1280`; lower `confidence`. |
| Need more FPS | Model/input too heavy | Lower `image_size`, keep ReID off, ensure CUDA is active. |

TouchDesigner-side script details: [td_scripts/README.md](../td_scripts/README.md).
