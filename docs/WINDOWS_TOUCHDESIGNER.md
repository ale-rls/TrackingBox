# Running on Windows with TouchDesigner (local, no Modal)

This is the **all-local** deployment: the whole pipeline runs on the Windows
theater PC, TouchDesigner is the camera front-end and/or the renderer, and they
talk over WebSocket on `localhost`. No cloud, no Modal, no network round-trip.

It mirrors the architecture of
[`torinmb/yolo-touchdesigner`](https://github.com/torinmb/yolo-touchdesigner):
**YOLO runs as an independent local service; TouchDesigner bridges to it over
WebSocket.** Here that service is `audience-tracker serve`, which adds ByteTrack,
the Global Identity Manager (persistent anonymous GIDs), and the REST/WS APIs.

ReID is **optional** and **off** in this guide (`--no-reid`) so you don't have to
fight the `torchreid` build on Windows. With ReID off you still get persistent
GIDs while a person is continuously tracked — you only lose identity *recovery*
after a full occlusion. Add it later (see the end).

---

## 1. Prerequisites

* An NVIDIA GPU with a current driver. (If the `yolo-touchdesigner` plugin already
  runs on this machine, CUDA + drivers are good to go.)
* TouchDesigner installed.
* **Python 3.10 or 3.11 installed separately** from TouchDesigner (the service runs
  in its own venv, *not* inside TD's Python). Get it from python.org.

Check the driver/GPU from a terminal:

```bat
nvidia-smi
```

## 2. Install the service

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip

:: CUDA build of PyTorch — pick the cuXXX that matches your driver (cu124 shown).
:: See https://pytorch.org/get-started/locally/ for the exact line.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: Detection + tracking only (no ReID). YOLO11 weights auto-download on first run.
pip install -e ".[detect]"
```

Confirm the GPU is visible to PyTorch (this is the one thing that silently bites
on Windows — a CPU-only torch wheel will "work" but be far too slow):

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
:: -> True NVIDIA GeForce RTX ...
```

## 3. Pick an integration mode

### Mode A — the service owns the camera (recommended, simplest)

The service opens the USB camera directly and produces an annotated MJPEG stream;
TouchDesigner just *consumes* results. Nothing to encode in TD.

```bat
audience-tracker serve --backend real --device cuda --source 0 --no-reid --port 8000
```

In TouchDesigner:

* **Overlay video** → a **Video Stream In TOP**, URL `http://localhost:8000/video`
  (MJPEG). This shows the live frame with `#GID` labels drawn on it.
* **Audience data** → a **WebSocket DAT**, Network Address `localhost`, Port `8000`,
  the request `/ws`. Attach `td_scripts/td_receive_state.py` as its callbacks DAT.
  It maintains a table `audience` of `gid, visible, cx, cy, x1, y1, x2, y2` you can
  use to drive instances/labels.

Sanity check from a terminal:

```bat
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/audience
```

### Mode B — TouchDesigner feeds the frames

Use this when YOLO should run on a *composited/processed* TD source rather than
the raw camera. TD encodes each frame to JPEG and streams it to `/ingest`.

```bat
audience-tracker serve --backend real --device cuda --source ingest --no-reid --port 8000
```

In TouchDesigner:

* Install OpenCV into **TD's** Python once (TD ships its own interpreter):
  `"<TouchDesigner>/bin/python" -m pip install opencv-python`
* **Send frames** → a **WebSocket DAT** to `/ingest` (use the token below if set),
  driven by `td_scripts/td_send_frames.py` from an Execute DAT (`onFrameEnd`).
* **Output** is the same as Mode A: Video Stream In TOP on `/video`, WebSocket DAT
  on `/ws`.

If you set `AT_INGEST_TOKEN`, connect to `/ingest?token=<token>`; otherwise auth is
off (fine on `localhost`).

## 4. Useful options

| Flag / env | Effect |
|---|---|
| `--no-reid` | Detection + tracking only (no torchreid needed). |
| `--device cuda` | Force GPU. |
| `--source 0` / `--source rtsp://...` | Camera index or stream (Mode A). |
| `--source ingest` | Receive frames over `/ingest` (Mode B). |
| `AT_DETECTOR_IMAGE_SIZE=960` | Lower for more FPS, higher for small/distant people. |
| `AT_DETECTOR_CONFIDENCE_THRESHOLD=0.35` | Detection confidence. |
| `AT_OVERLAY_DEBUG=true` | Draw `conf:` under each GID. |

All config is also available via `--config config.json` and `AT_*` env vars.

## 5. Adding ReID later (optional)

```bat
pip install -e ".[reid]"
:: If torchreid's PyPI build misbehaves on Windows, install the maintained source:
:: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Then drop `--no-reid` and run with `--backend real --device cuda`. ReID enables
GID recovery after a person is briefly occluded or leaves and re-enters frame.
