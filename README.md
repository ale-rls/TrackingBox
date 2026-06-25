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
Camera ─▶ YOLO11m ─▶ ByteTrack ─▶ OSNet ReID ─▶ Identity Manager ─┬─▶ Overlay video (/video)
        (detect)    (track)       (embed/5f)     (GIDs, truth)     ├─▶ REST API  (/api/*)
                                                                    └─▶ WebSocket (/ws)
```

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

# Real ML stack (GPU worker): YOLO11m + ByteTrack + OSNet
pip install -e ".[ml]"

# Dev/test tooling
pip install -e ".[dev]"
```

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

## Project layout

```
src/audience_tracker/
  config.py        Layered configuration (defaults / JSON / env)
  models.py        Detection, Track, Identity, AudienceState
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
  cli.py           serve / run / benchmark / demo
  api/app.py       FastAPI REST + WebSocket + video
deploy/modal_app.py  Modal deployment
tests/               Identity, state store, pipeline, API tests
```

## Deployment (Modal)

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
