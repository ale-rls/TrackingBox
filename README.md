# Audience Tracking System (V1)

Real-time, **anonymous** audience tracking for an interactive theater installation.

A single live camera feed is processed into persistent, anonymous **Global IDs
(GIDs)** — one per audience member — rendered as an overlay and exposed over
REST + WebSocket. Identity continuity through short occlusions and crowd
clustering is maintained with appearance-based ReID.

> Anonymous by design. The system assigns opaque numeric GIDs (`#17`). It does
> **not** perform face identification, name lookup, demographics, or emotion
> analysis.

```
  camera / TouchDesigner / RTSP / file / ingestion queue   (any FrameSource)
        │
        ▼
   YOLO11m ─▶ ByteTrack ─▶ [OSNet ReID] ─▶ Identity Manager ─┬─▶ Overlay video (/video)
   (detect)   (track)      (optional)       (GIDs, truth)    ├─▶ REST API  (/api/*)
                                                              └─▶ WebSocket (/ws)
```

**Two ways to run it:**

* **Local on Windows with TouchDesigner** — the whole pipeline runs on the venue
  PC; TouchDesigner is the camera/renderer, bridged over WebSocket on `localhost`.
  No cloud. → [Run on Windows with TouchDesigner](#run-on-windows-with-touchdesigner-local-no-modal).
* **Modal (cloud GPU)** — a venue Capture Agent streams frames to a Modal worker.
  → [Video ingestion](#video-ingestion-capture-agent--modal) + [Modal deployment](#modal-deployment-cloud-gpu).

Either way the pipeline consumes frames through a `FrameSource` and is agnostic to
the transport (camera / TouchDesigner / RTSP / file / ingestion queue / simulator).

## Why it runs anywhere

The heavy ML stack (YOLO11m, ByteTrack, OSNet) lives behind small adapter
classes. A **synthetic backend** (a colour-coded crowd simulator + colour
detector + colour ReID) implements the same interfaces, so the *entire*
pipeline — identity logic, APIs, overlay, metrics, logging, benchmark — runs and
is testable on a laptop with **no GPU and no camera**.

* `backend = real` → YOLO11m + ByteTrack + OSNet (needs the `ml` extra + GPU)
* `backend = mock` → synthetic simulator (CPU only)
* `backend = auto` → real if importable, else mock (default)

## Install

```bash
# Core serving layer (API + state logic + mock backend demo)
pip install -e .

# Detection + tracking only (YOLO11 + ByteTrack), no ReID — the Windows/
# TouchDesigner-friendly install (pair with a CUDA torch wheel):
pip install -e ".[detect]"

# Full real stack (GPU worker): YOLO11m + ByteTrack + OSNet ReID
pip install -e ".[ml]"

# Add ReID on top of [detect] later (torchreid is awkward on Windows):
pip install -e ".[reid]"

# Dev/test tooling
pip install -e ".[dev]"
```

For the **all-local Windows + TouchDesigner** deployment (one-command installer, TD
wiring, troubleshooting) jump to
[Run on Windows with TouchDesigner](#run-on-windows-with-touchdesigner-local-no-modal).

Check any environment (Windows/Mac/Linux) with `audience-tracker doctor` — it reports
which dependencies and GPU are present and whether `serve` / `detect` / `reid` are ready.

Python ≥ 3.10. The core/identity logic is pure stdlib; frame processing needs
numpy (+ OpenCV for the overlay video stream).

## Quick start

```bash
# 1) No-GPU sanity run — synthetic crowd, stats to stdout
audience-tracker demo --people 24 --frames 200

# 2) Serve the API (auto-runs the pipeline; mock backend if no GPU)
audience-tracker serve --backend mock --port 8000
#   REST:  curl localhost:8000/api/audience | jq
#          curl localhost:8000/api/stats
#          curl localhost:8000/api/snapshot
#   WS:    websocat ws://localhost:8000/ws
#   Video: open http://localhost:8000/video   (MJPEG overlay)

# 3) Process a real source on a GPU box
audience-tracker serve --backend real --device cuda --source rtsp://camera/stream

# 4) Annotated output file from a recording
audience-tracker run --source rehearsal.mp4 --output annotated.mp4

# 5) Benchmark all scenarios -> JSON report
audience-tracker benchmark --frames 300 --output benchmarks/report.json
```

Configuration comes from defaults → optional JSON file (`--config`) → `AT_*`
environment variables (highest priority). See `config.example.json`. Example:
`AT_REID_SIMILARITY_THRESHOLD=0.7 AT_PIPELINE_SOURCE=rtsp://cam/stream`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/audience` | Active audience: `[{gid, visible, center, bbox}]` |
| GET | `/api/audience/{gid}` | One member: `{gid, visible, duration_seen_seconds, ...}` |
| GET | `/api/stats` | `{active_people, total_people_seen}` |
| GET | `/api/snapshot` | `{timestamp, active_people, people:[...]}` |
| GET | `/metrics` | `{fps, latency_ms, gpu_utilization, active_people, active_tracks, reid_inference_time_ms, id_switches}` |
| WS | `/ws` | Primes with a snapshot, then streams `{gid, visible, center, bbox}` on every change |
| GET | `/video` | MJPEG overlay stream |
| GET | `/health` | Liveness |

**Tracker IDs are internal only** and never appear in any response — external
consumers use GIDs exclusively.

## Identity model (source of truth)

The `IdentityManager` reconciles tracker output into stable GIDs:

1. New audience member → unique GID (monotonic counter).
2. New track matching a lost identity above the cosine-similarity threshold →
   **reuse** that GID (ReID recovery).
3. No match → new GID.
4. GIDs are **never reused** within a session.
5. Tracker IDs may churn; GIDs stay stable.

ReID embeddings are refreshed every 5 frames for *established* tracks and
immediately for *new/unmatched* tracks (so a re-entering person is recovered
without delay). A running (EMA) appearance average per identity smooths matching.

## Video ingestion (Capture Agent → Modal)

This is the **Modal (cloud)** path — skip it if you're running locally with
TouchDesigner (above). Modal has no physical camera, so a separate **Capture
Agent** runs on the venue computer and streams frames to the cloud. Acquisition/
transport only — no CV inference happens locally. (The same `/ingest` endpoint
also backs TouchDesigner "Mode B".)

```
USB camera ─▶ Capture Agent ──(JPEG over one long-lived WSS)──▶ /ingest ─▶ bounded queue ─▶ pipeline
```

* **Capture Agent** (`ingestion/capture_agent.py`): OpenCV capture, timestamp,
  optional resize, JPEG-compress (quality configurable, default 85%), and stream
  over a single persistent WebSocket. Never blocks on the network — it always
  sends the *newest* frame and drops stale ones. Auto-reconnects with exponential
  backoff; reopens the camera on unplug — no restart. Exposes `GET /status`
  (`connected`, `fps_capture`, `fps_sent`, `latency_ms`, `frames_dropped`,
  `uptime_seconds`).
* **Ingestion Service** (`ingestion/server.py`): the `/ingest` WebSocket on the
  Modal app. Bearer-token authenticated (WSS), validates + decodes each packet,
  and pushes it onto a **bounded, drop-oldest queue** (`QueueFrameSource`,
  default depth 3) so latency never grows. Frames are kept strictly ordered by
  `frame_id`. `GET /ingest/stats` reports queue depth/received/dropped.
* **Packet** (`ingestion/packet.py`): a compact binary frame
  `MAGIC | header_len | {frame_id, timestamp, width, height} | jpeg_bytes`
  (no base64 overhead).
* **FrameSource** (`base.py`): the pipeline consumes `next_frame()` and never
  learns the transport — USB / RTSP / file / queue / simulator are
  interchangeable.

Run it:

```bash
# On the venue machine (after: pip install -e ".[agent]")
audience-tracker capture-agent \
  --server-url wss://<your-app>.modal.run/ingest \
  --token "$INGEST_TOKEN" --source 0
curl localhost:9000/status        # live capture/transport metrics
```

Set the matching token on Modal via `AT_INGEST_TOKEN` (a Modal secret). If unset,
auth is disabled (development only). On Modal the pipeline is configured with
`pipeline.source = "ingest"`.

## Run on Windows with TouchDesigner (local, no Modal)

The **all-local** alternative to the Modal path: the whole pipeline runs on the
Windows theater PC, TouchDesigner is the camera front-end and/or renderer, and they
talk over WebSocket on `localhost` — no cloud, no network round-trip. This mirrors
[`torinmb/yolo-touchdesigner`](https://github.com/torinmb/yolo-touchdesigner): YOLO
runs as an independent local service (here, `audience-tracker serve`, which adds
ByteTrack, the Identity Manager, and the REST/WS APIs) that TouchDesigner bridges to.

ReID is **optional** and **off** here (`--no-reid`) so you don't have to fight the
`torchreid` build on Windows. With it off you still get persistent GIDs while a person
is continuously tracked — you only lose identity *recovery* after a full occlusion.

### Prerequisites

* An NVIDIA GPU with a current driver (`nvidia-smi` to confirm). If the
  `yolo-touchdesigner` plugin already runs on this machine, CUDA + drivers are fine.
* TouchDesigner installed.
* **Python 3.10–3.12 installed separately** from TouchDesigner (the service runs in
  its own venv, *not* inside TD's Python). From python.org; tick *Add to PATH*.

### Install — one command

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

### Mode A — the service owns the camera (recommended)

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

### Mode B — TouchDesigner feeds the frames

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

### Useful options

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

### Troubleshooting

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

TouchDesigner-side script details: [`td_scripts/README.md`](td_scripts/README.md).

## Project layout

```
src/audience_tracker/
  config.py        Layered configuration (defaults / JSON / env)
  models.py        Detection, Track, Identity, AudienceState, Frame
  base.py          Detector / Tracker / ReIDExtractor / FrameSource protocols
  identity.py      IdentityManager  ← source of truth (Identity Rules 1–5)
  vecmath.py       Pure-Python cosine / EMA for embeddings
  detection.py     YOLODetector  | MockDetector
  tracking.py      ByteTrackTracker | IoUTracker (fallback)
  reid.py          OSNetExtractor | MockReID | NullReID
  overlay.py       OverlayRenderer (#GID labels; debug conf)
  metrics.py       MetricsCollector (fps/latency/gpu/reid time)
  framelog.py      Replayable JSONL frame logger
  statestore.py    Shared state + WebSocket fan-out (Redis = future)
  pipeline.py      TrackingPipeline (detect→track→reid→identity→publish)
  factory.py       Backend resolution + component wiring
  simulator.py     Synthetic crowd (no-GPU demo/test/benchmark)
  benchmark.py     Scenario runner → JSON report
  cli.py           serve / run / benchmark / capture-agent / demo
  api/app.py       FastAPI REST + WebSocket + video + /ingest
  ingestion/       Capture Agent, /ingest WebSocket, packet, FrameSources
deploy/modal_app.py  Modal deployment (pipeline.source = "ingest")
tests/               Identity, state store, pipeline, API, packet, queue, ingest
```

## Modal deployment (cloud GPU)

The **alternative** to running locally on Windows (above) — host the pipeline on a
cloud GPU and feed it from the venue Capture Agent.

```bash
pip install -e ".[deploy]"
modal deploy deploy/modal_app.py        # API + GPU pipeline (in-memory state)
modal run    deploy/modal_app.py        # run the benchmark on a GPU
```

V1 co-locates the GPU tracking worker and the API in one container with in-memory
shared state. The documented next step is splitting them into a GPU worker + a
CPU API container backed by Redis — see `RedisStateStore` in `statestore.py`.

## Performance

Targets: ≥15 FPS (preferred 20–30), end-to-end latency < 500 ms on 1080p with up
to ~70 people, stable for multi-hour shows. These are met by the **real** stack
on a CUDA GPU. The included benchmark also runs on the CPU mock backend for
plumbing validation — its FPS reflects the unoptimized synthetic detector, not
the GPU pipeline.

## Tests

```bash
pytest            # identity rules, state store, end-to-end mock, API
```

The Identity Manager (the system's correctness core) is covered by deterministic
unit tests for all five Identity Rules, ReID recovery, recovery-window expiry,
duration accounting, and the GID-only public contract.

## Acceptance criteria mapping

| # | Criterion | Where |
|---|---|---|
| 1 | Live feed processed | `pipeline.py`, `factory.open_source` (camera/RTSP/file) |
| 2 | Persistent GIDs | `identity.py` (Rules 1–5) |
| 3 | GIDs rendered on video | `overlay.py`, `/video` |
| 4 | State via REST | `api/app.py` `/api/*` |
| 5 | State via WebSocket | `api/app.py` `/ws` |
| 6 | 15+ FPS on 1080p | real stack on GPU; `benchmark.py` verifies |
| 7 | ReID recovery after occlusion | `identity._match_lost`, ReID scheduling |
| 8 | Metrics exposed + recorded | `metrics.py` `/metrics`, `framelog.py` |
| 9 | Stable ~70-person audience | benchmark scenarios; bounded-memory GC |
```
