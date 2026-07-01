# Audience Tracking System

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

## Runs anywhere

The heavy ML stack (YOLO11m, ByteTrack, OSNet) lives behind small adapter
classes, and a **synthetic backend** (a colour-coded crowd simulator) implements
the same interfaces. So the *entire* pipeline — identity logic, APIs, overlay,
metrics, benchmark — runs and is testable on a laptop with **no GPU and no camera**.

* `backend = real` → YOLO11m + ByteTrack + OSNet (needs the `ml` extra + GPU)
* `backend = mock` → synthetic simulator (CPU only)
* `backend = auto` → real if importable, else mock (default)

## Install

```bash
pip install -e .            # core serving layer + mock backend demo
```

| Extra | Installs | For |
|---|---|---|
| `.[detect]` | YOLO11 + ByteTrack (no ReID) | Windows/TouchDesigner box (pair with a CUDA torch wheel) |
| `.[reid]`   | torchreid | add OSNet ReID on top of `[detect]` |
| `.[ml]`     | full real stack: YOLO11m + ByteTrack + OSNet | GPU worker |
| `.[agent]`  | OpenCV capture deps | venue Capture Agent (Modal path) |
| `.[deploy]` | Modal SDK | deploying to Modal |
| `.[dev]`    | pytest, ruff, httpx | development / tests |

Python ≥ 3.10. The core/identity logic is pure stdlib; frame processing needs
numpy (+ OpenCV for the overlay video stream). Check any environment with
`audience-tracker doctor` — it reports which dependencies and GPU are present and
whether `serve` / `detect` / `reid` are ready.

## Quick start (laptop, no GPU)

```bash
# 1) No-GPU sanity run — synthetic crowd, stats to stdout
audience-tracker demo --people 24 --frames 200

# 2) Serve the API (auto-runs the pipeline; mock backend if no GPU)
audience-tracker serve --backend mock --port 8000
#   REST:  curl localhost:8000/api/audience | jq
#   WS:    websocat ws://localhost:8000/ws
#   Video: open http://localhost:8000/video   (MJPEG overlay)

# 3) Annotated output file from a recording
audience-tracker run --source rehearsal.mp4 --output annotated.mp4
```

Configuration comes from defaults → optional JSON file (`--config`) → `AT_*`
environment variables (highest priority). See `config.example.json`.

## Running it for real

Two deployment paths, both consuming frames through the same transport-agnostic
`FrameSource`:

* **[Run with TouchDesigner](docs/touchdesigner.md)** *(local, no cloud)* — the
  whole pipeline runs on the venue PC; TouchDesigner is the camera/renderer,
  bridged over WebSocket on `localhost`. The recommended default.
* **[Run on Modal](docs/modal.md)** *(cloud GPU)* — a venue Capture Agent streams
  frames to a Modal worker over a long-lived WebSocket.

## API at a glance

REST + WebSocket, **GIDs only** — tracker IDs never cross the boundary. Endpoints:
`/api/audience`, `/api/stats`, `/api/snapshot`, `/metrics`, `/ws`, `/video`,
`/health`. Full table and payloads in [docs/architecture.md](docs/architecture.md#api).

## Docs

* [docs/touchdesigner.md](docs/touchdesigner.md) — local Windows + TouchDesigner runbook, presets, launcher, troubleshooting
* [docs/floor_projection.md](docs/floor_projection.md) — fisheye/floor calibration and top-down `floor` coordinates
* [docs/modal.md](docs/modal.md) — cloud path: Capture Agent + `/ingest` + Modal deploy
* [docs/architecture.md](docs/architecture.md) — identity model, API reference, project layout, performance, tests

## Tests

```bash
pytest            # identity rules, state store, end-to-end mock, API
```
