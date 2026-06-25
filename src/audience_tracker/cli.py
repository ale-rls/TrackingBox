"""Command-line entry point.

Subcommands:
  serve       Run the REST/WebSocket API (and, by default, the tracking pipeline).
  run         Process a video source headlessly, writing an annotated output.
  benchmark   Run the benchmark scenarios and emit a JSON report.
  demo        Quick mock-backend run (no GPU/camera) with stats to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from .config import Config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="Path to a JSON config file")
    p.add_argument("--source", help="Camera index, file path or RTSP/HTTP URL")
    p.add_argument("--backend", choices=["auto", "real", "mock"], help="Override backend")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Inference device")


def _load_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "source", None):
        cfg.pipeline.source = args.source
    if getattr(args, "backend", None):
        cfg.pipeline.backend = args.backend
    if getattr(args, "device", None):
        cfg.pipeline.device = args.device
    return cfg


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import create_app

    cfg = _load_cfg(args)
    cfg.pipeline.run_pipeline = not args.no_pipeline
    app = create_app(cfg)
    uvicorn.run(app, host=args.host or cfg.api.host, port=args.port or cfg.api.port)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .factory import build_pipeline, open_source

    cfg = _load_cfg(args)
    cfg.pipeline.run_pipeline = False
    cfg.pipeline.output_path = args.output or cfg.pipeline.output_path
    built = build_pipeline(cfg, num_people=cfg.pipeline.mock_people)
    pipeline = built["pipeline"]
    camera = open_source(cfg, simulator=built["simulator"], max_frames=args.frames)

    writer = None
    if cfg.pipeline.output_path:
        writer = _open_writer(cfg.pipeline.output_path, camera)
    print(f"Processing ({built['backend']} backend)...")
    n = pipeline.run(camera, writer=writer, max_frames=args.frames)
    print(json.dumps({"frames": n, **pipeline.metrics.snapshot(), **pipeline.identity.stats()}, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    cfg = _load_cfg(args)
    report = run_benchmark(cfg, frames_per_scenario=args.frames, output_path=args.output)
    print(json.dumps(report["summary"], indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .factory import build_pipeline, open_source

    cfg = _load_cfg(args)
    cfg.pipeline.backend = args.backend or "mock"
    cfg.pipeline.run_pipeline = False
    built = build_pipeline(cfg, num_people=args.people)
    pipeline = built["pipeline"]
    camera = open_source(cfg, simulator=built["simulator"], max_frames=args.frames)
    print(f"Demo: {built['backend']} backend, {args.people} simulated people\n")
    t0 = time.time()
    for i in range(args.frames):
        ok, frame = camera.read()
        if not ok:
            break
        pipeline.process_frame(frame, i)
        if i % args.report_every == 0:
            print(f"frame {i:4d}  " + json.dumps(pipeline.metrics.snapshot()))
    dt = time.time() - t0
    print("\n--- result ---")
    print(json.dumps({**pipeline.identity.stats(), **pipeline.identity.counters(),
                      "wall_seconds": round(dt, 2)}, indent=2))
    return 0


def _open_writer(path: str, camera):
    import cv2

    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("Empty source")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    return writer


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="audience-tracker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the REST/WebSocket API")
    _add_common(p_serve)
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.add_argument("--no-pipeline", action="store_true", help="API only (state from elsewhere)")
    p_serve.set_defaults(func=cmd_serve)

    p_run = sub.add_parser("run", help="Process a source to an annotated video")
    _add_common(p_run)
    p_run.add_argument("--output", help="Output annotated video path (.mp4)")
    p_run.add_argument("--frames", type=int, default=None, help="Max frames to process")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("benchmark", help="Run benchmark scenarios -> JSON report")
    _add_common(p_bench)
    p_bench.add_argument("--frames", type=int, default=300, help="Frames per scenario")
    p_bench.add_argument("--output", default="benchmarks/report.json")
    p_bench.set_defaults(func=cmd_benchmark)

    p_demo = sub.add_parser("demo", help="Quick mock run, stats to stdout")
    _add_common(p_demo)
    p_demo.add_argument("--people", type=int, default=24)
    p_demo.add_argument("--frames", type=int, default=200)
    p_demo.add_argument("--report-every", type=int, default=25)
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
