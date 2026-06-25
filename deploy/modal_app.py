"""Modal deployment for the Audience Tracking System.

Deploy:
    modal deploy deploy/modal_app.py

This V1 deployment co-locates the GPU Tracking Worker and the API Service in a
single GPU container, sharing in-memory state (per the spec's "Initial
implementation: In-memory state"). The pipeline reads from the source given by
``AT_PIPELINE_SOURCE`` (an RTSP/HTTP camera URL reachable from Modal).

Future split (spec "Future option: Redis"): run the worker as a separate
``@app.cls`` GPU container that writes audience state to Redis, and a CPU-only
API container that reads from Redis + Pub/Sub. See ``RedisStateStore`` in
``audience_tracker.statestore`` for the extension point.
"""

from __future__ import annotations

import modal

# --------------------------------------------------------------------- #
# Image: core + ML stack, with YOLO11m weights baked in.
# --------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # OpenCV runtime libs
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "websockets>=12",
        "numpy>=1.24",
        "opencv-python-headless>=4.9",
        "ultralytics>=8.3",
        "supervision>=0.20",
        "torch>=2.2",
        "torchvision>=0.17",
        "torchreid>=0.2.5",
    )
    # Pre-download YOLO11m so cold starts don't pay for it.
    .run_commands("python -c \"from ultralytics import YOLO; YOLO('yolo11m.pt')\"")
    # Ship our package.
    .add_local_python_source("audience_tracker")
)

app = modal.App("audience-tracker")


# --------------------------------------------------------------------- #
# API + pipeline in one GPU container (V1).
# --------------------------------------------------------------------- #
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,  # multi-hour performances
    min_containers=1,     # keep warm during a show
    max_containers=1,     # single-camera => single worker (in-memory state)
)
@modal.concurrent(max_inputs=100)  # many API/WS clients, one pipeline
@modal.asgi_app()
def api():
    from audience_tracker.api.app import create_app
    from audience_tracker.config import Config

    cfg = Config.load()  # AT_* env vars apply (e.g. AT_PIPELINE_SOURCE)
    cfg.pipeline.backend = "real"
    cfg.pipeline.device = "cuda"
    cfg.pipeline.run_pipeline = True
    return create_app(cfg)


# --------------------------------------------------------------------- #
# Optional: batch-process recorded rehearsal footage for benchmarking.
# --------------------------------------------------------------------- #
@app.function(image=image, gpu="A10G", timeout=60 * 60)
def benchmark(frames_per_scenario: int = 300) -> dict:
    from audience_tracker.benchmark import run_benchmark
    from audience_tracker.config import Config

    cfg = Config.load()
    cfg.pipeline.backend = "real"
    cfg.pipeline.device = "cuda"
    return run_benchmark(cfg, frames_per_scenario=frames_per_scenario, output_path=None)


@app.local_entrypoint()
def main():
    """`modal run deploy/modal_app.py` -> run the benchmark on a GPU."""
    import json

    print(json.dumps(benchmark.remote(frames_per_scenario=120)["summary"], indent=2))
