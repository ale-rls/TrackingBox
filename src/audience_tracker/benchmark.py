"""Benchmark harness.

Runs the spec's scenarios (sparse / medium / dense crowds, cluster formation,
occlusions, people entering & leaving) and emits a JSON report with FPS,
latency, active tracks, active GIDs and ID switches.

By default it drives the synthetic simulator (no GPU needed). To benchmark the
real stack on recorded rehearsal footage, pass ``--backend real`` and a
``scenarios`` mapping of name -> video path via the config file.
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone

from .config import Config
from .factory import build_pipeline, open_source

# (name, simulated_people) — crowd density / behaviour is emergent from motion.
DEFAULT_SCENARIOS = [
    ("sparse_crowd", 5),
    ("medium_crowd", 20),
    ("dense_crowd", 60),
    ("cluster_formation", 40),
    ("occlusions", 30),
    ("entering_leaving", 25),
]

TARGET_FPS = 15.0
TARGET_LATENCY_MS = 500.0


def _bench_config(cfg: Config) -> Config:
    c = copy.deepcopy(cfg)
    # Measure the core pipeline; overlay/stream/logging are I/O, not tracking.
    c.pipeline.render_overlay = False
    c.pipeline.stream_overlay = False
    c.logging.enabled = False
    return c


def run_scenario(cfg: Config, name: str, people: int, frames: int) -> dict:
    built = build_pipeline(cfg, num_people=people)
    pipeline = built["pipeline"]
    camera = open_source(cfg, simulator=built["simulator"], max_frames=frames)

    latencies: list[float] = []
    peak_tracks = 0
    peak_people = 0
    t0 = time.perf_counter()
    processed = 0
    for i in range(frames):
        ok, frame = camera.read()
        if not ok:
            if getattr(camera, "exhausted", True):
                break
            continue  # transient live-read failure — the source is retrying
        pipeline.process_frame(frame, i)
        snap = pipeline.metrics.snapshot()
        latencies.append(snap["latency_ms"])
        peak_tracks = max(peak_tracks, snap["active_tracks"])
        peak_people = max(peak_people, snap["active_people"])
        processed += 1
    wall = time.perf_counter() - t0
    camera.release()

    counters = pipeline.identity.counters()
    fps = processed / wall if wall > 0 else 0.0
    latencies.sort()
    return {
        "scenario": name,
        "simulated_people": people,
        "frames": processed,
        "avg_fps": round(fps, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1) if latencies else 0.0,
        "peak_active_tracks": peak_tracks,
        "peak_active_people": peak_people,
        "total_gids": counters["total_people_seen"],
        "id_switches": counters["id_switches"],
        "recoveries": counters["recoveries"],
        "meets_fps_target": fps >= TARGET_FPS,
    }


def run_benchmark(
    cfg: Config,
    frames_per_scenario: int = 300,
    output_path: str | None = "benchmarks/report.json",
    scenarios=None,
) -> dict:
    bench_cfg = _bench_config(cfg)
    scenarios = scenarios or DEFAULT_SCENARIOS
    from .factory import resolve_backend

    results = []
    for name, people in scenarios:
        print(f"  scenario: {name} ({people} people)...")
        results.append(run_scenario(bench_cfg, name, people, frames_per_scenario))

    fps_vals = [r["avg_fps"] for r in results]
    summary = {
        "overall_avg_fps": round(sum(fps_vals) / len(fps_vals), 2) if fps_vals else 0.0,
        "min_scenario_fps": round(min(fps_vals), 2) if fps_vals else 0.0,
        "max_latency_ms": max((r["p95_latency_ms"] for r in results), default=0.0),
        "all_scenarios_meet_15fps": all(r["meets_fps_target"] for r in results),
        "latency_under_500ms": all(r["p95_latency_ms"] < TARGET_LATENCY_MS for r in results),
        "total_id_switches": sum(r["id_switches"] for r in results),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": resolve_backend(bench_cfg),
        "targets": {"min_fps": TARGET_FPS, "max_latency_ms": TARGET_LATENCY_MS},
        "config": {
            "detector_model": cfg.detector.model_path,
            "reid_model": cfg.reid.model_name,
            "reid_update_every": cfg.reid.update_every,
            "similarity_threshold": cfg.reid.similarity_threshold,
        },
        "scenarios": results,
        "summary": summary,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  report written to {output_path}")
    return report
