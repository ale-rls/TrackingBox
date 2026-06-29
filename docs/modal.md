# Run on Modal (cloud GPU)

The **cloud** deployment: host the pipeline on a Modal GPU and feed it from the
venue. Modal has no physical camera, so a separate **Capture Agent** runs on the
venue computer and streams frames to the cloud over a single long-lived WebSocket.
Acquisition/transport only — no CV inference happens locally.

> Want to run everything on the venue PC instead, with no cloud? See
> [Run with TouchDesigner](touchdesigner.md). (Its "Mode B" reuses the same
> `/ingest` endpoint described here.)

```
USB camera ─▶ Capture Agent ──(JPEG over one long-lived WSS)──▶ /ingest ─▶ bounded queue ─▶ pipeline
```

## Ingestion subsystem

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

## Run the Capture Agent (venue machine)

```bash
# After: pip install -e ".[agent]"
audience-tracker capture-agent \
  --server-url wss://<your-app>.modal.run/ingest \
  --token "$INGEST_TOKEN" --source 0
curl localhost:9000/status        # live capture/transport metrics
```

Set the matching token on Modal via `AT_INGEST_TOKEN` (a Modal secret). If unset,
auth is disabled (development only). On Modal the pipeline is configured with
`pipeline.source = "ingest"`.

## Deploy the worker (Modal)

```bash
pip install -e ".[deploy]"
modal deploy deploy/modal_app.py        # API + GPU pipeline (in-memory state)
modal run    deploy/modal_app.py        # run the benchmark on a GPU
```

V1 co-locates the GPU tracking worker and the API in one container with in-memory
shared state. The documented next step is splitting them into a GPU worker + a
CPU API container backed by Redis — see `RedisStateStore` in `statestore.py`.
