# Run with TouchDesigner (local, no Modal)

The **all-local** deployment: the whole pipeline runs on the Windows theater PC,
TouchDesigner is the camera front-end and/or renderer, and they talk over
WebSocket on `localhost` — no cloud, no network round-trip. This mirrors
[`torinmb/yolo-touchdesigner`](https://github.com/torinmb/yolo-touchdesigner): YOLO
runs as an independent local service (here, `audience-tracker serve`, which adds
ByteTrack, the Identity Manager, and the REST/WS APIs) that TouchDesigner bridges to.

ReID is **optional** and **off** here (`--no-reid`) so you don't have to fight the
`torchreid` build on Windows. With it off you still get persistent GIDs while a person
is continuously tracked — you only lose identity *recovery* after a full occlusion.

> Looking for the cloud GPU path instead? See [Run on Modal](modal.md).

## Prerequisites

* An NVIDIA GPU with a current driver (`nvidia-smi` to confirm). If the
  `yolo-touchdesigner` plugin already runs on this machine, CUDA + drivers are fine.
* TouchDesigner installed.
* **Python 3.10–3.12 installed separately** from TouchDesigner (the service runs in
  its own venv, *not* inside TD's Python). From python.org; tick *Add to PATH*.

## Install — one command

From the repo root in a terminal (or double-click `scripts\install_windows.bat`):

```bat
scripts\install_windows.bat
```

It handles the fiddly parts: finds a usable Python (3.10–3.12) or tells you what to
install; creates/reuses the `.venv`; runs `nvidia-smi`, reads your CUDA version, and
installs the **matching** CUDA PyTorch wheel (or CPU with a clear warning if there's
no GPU); installs the detection stack (`.[detect]`); and verifies with
`audience-tracker doctor`. Options: `-Reid` (also install ReID), `-Cpu` (force CPU),
`-CudaTag cu121` (override the CUDA wheel). Sanity-check anytime:

```bat
.venv\Scripts\audience-tracker doctor
```

<details><summary>Manual install (alternative)</summary>

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
:: CUDA build of PyTorch — pick the cuXXX matching your driver (see pytorch.org).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
:: Detection + tracking only (no ReID). YOLO11 weights auto-download on first run.
pip install -e ".[detect]"
audience-tracker doctor --require detect --require-cuda
```
</details>

## Mode A — the service owns the camera (recommended)

The service opens the USB camera directly and produces an annotated MJPEG stream;
TouchDesigner just *consumes* results. Nothing to encode in TD.

```bat
scripts\run_windows.bat
:: = .venv\Scripts\audience-tracker serve --backend real --device cuda --source 0 --no-reid --port 8000
:: pass-through args work, e.g.:  scripts\run_windows.bat --source 1 --port 9000
```

In TouchDesigner:

* **Overlay video** → a **Video Stream In TOP**, URL `http://localhost:8000/video`
  (MJPEG) — the live frame with `#GID` labels drawn on it.
* **Audience data** → a **WebSocket DAT** (Network Address `localhost`, Port `8000`,
  request `/ws`) with `td_scripts/td_receive_state.py` as its callbacks DAT. It keeps
  a table `audience` of `gid, visible, cx, cy, x1, y1, x2, y2` to drive instances/labels.

Sanity check: `curl http://localhost:8000/api/stats` and `.../api/audience`.

## Mode B — TouchDesigner feeds the frames

Use when YOLO should run on a *composited/processed* TD source rather than the raw
camera. TD encodes each frame to JPEG and streams it to `/ingest`.

```bat
audience-tracker serve --backend real --device cuda --source ingest --no-reid --port 8000
```

* Install OpenCV into **TD's** Python once: `"<TouchDesigner>\bin\python" -m pip install opencv-python`.
* **Send frames** → a **WebSocket DAT** to `/ingest`, driven by
  `td_scripts/td_send_frames.py` from an Execute DAT (`onFrameEnd`).
* **Output** is the same as Mode A (`/video` + `/ws`).
* If `AT_INGEST_TOKEN` is set, connect to `/ingest?token=<token>`; otherwise auth is
  off (fine on `localhost`).

## Useful options

| Flag / env | Effect |
|---|---|
| `--no-reid` | Detection + tracking only (no torchreid needed). |
| `--device cuda` | Force GPU. |
| `--source 0` / `--source rtsp://...` | Camera index or stream (Mode A). |
| `--source ingest` | Receive frames over `/ingest` (Mode B). |
| `AT_DETECTOR_IMAGE_SIZE=960` | Lower → more FPS; higher → small/distant people. |
| `AT_DETECTOR_CONFIDENCE_THRESHOLD=0.35` | Detection confidence. |
| `AT_OVERLAY_DEBUG=true` | Draw `conf:` under each GID. |

Adding ReID later: `pip install -e ".[reid]"` (if its PyPI build misbehaves on Windows,
`pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`), then drop
`--no-reid`.

## Troubleshooting

Run `audience-tracker doctor` first — it pinpoints most of these.

| Symptom | Likely cause | Fix |
|---|---|---|
| `doctor`: *"PyTorch installed but CUDA NOT available"* | CPU-only torch wheel | re-run `scripts\install_windows.bat` (auto-detects), or reinstall torch from the CUDA index. |
| Low FPS, GPU idle in Task Manager | running on CPU torch | confirm with `doctor`; expect `[OK] CUDA GPU: ...`. |
| Installer: *"No suitable Python found"* | Python missing or 3.13+ only | install Python 3.11 (tick *Add to PATH*); torch has no 3.13+ wheels yet. |
| `Could not open video source: '0'` | wrong camera index or in use | try `--source 1`, `2`…; close other apps using the webcam. |
| TD **Video Stream In TOP** stays black | service down / wrong URL | `curl http://localhost:8000/health`; URL must be `http://localhost:8000/video`. |
| TD **WebSocket DAT** won't connect | wrong address/port or firewall | `localhost` / `8000` / `/ws`; allow Python through Windows Firewall. |
| `pip install .[reid]` fails | torchreid build on Windows | stay on `--no-reid`, or install deep-person-reid from source (above). |
| Mode B: nothing arrives at `/ingest` | OpenCV missing in **TD's** Python | `"<TouchDesigner>\bin\python" -m pip install opencv-python`. |
| Small/distant people missed | detection input too low-res | raise `AT_DETECTOR_IMAGE_SIZE` (e.g. 1280); lower the confidence threshold. |
| Need more FPS | model/input too heavy | lower `AT_DETECTOR_IMAGE_SIZE` (e.g. 768), keep `--no-reid`, ensure CUDA is active. |

TouchDesigner-side script details: [`td_scripts/README.md`](../td_scripts/README.md).
