# Architecture & reference

How the system is wired, the public API surface, and the identity guarantees that
make GIDs trustworthy. For deployment, see [TouchDesigner](touchdesigner.md) or
[Modal](modal.md). For spatial calibration, see
[Floor Projection Calibration](floor_projection.md).

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

**Tracker IDs are internal only** and never appear in any response — external
consumers use GIDs exclusively.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/audience` | Active audience with GID, bbox, floor, and zone |
| GET | `/api/audience/{gid}` | One member: `{gid, visible, duration_seen_seconds, ...}` |
| GET | `/api/stats` | `{active_people, total_people_seen}` |
| GET | `/api/zones` | Configured floor-space zones |
| GET | `/api/zones/counts` | Live visible-person counts by zone |
| GET | `/api/snapshot` | `{timestamp, active_people, people:[...]}` |
| GET | `/metrics` | `{fps, latency_ms, gpu_utilization, active_people, active_tracks, reid_inference_time_ms, id_switches}` |
| WS | `/ws` | Primes with a snapshot, then streams changed audience entries |
| GET | `/video` | MJPEG overlay stream |
| GET | `/health` | Liveness |

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
  projection.py    Fisheye-aware image-to-floor projection
  zones.py         Floor-space zone definitions, matching, and counts
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
