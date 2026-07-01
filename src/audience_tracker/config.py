"""Central configuration for the audience tracking system.

Configuration is a tree of dataclasses with sensible defaults. Values can be
overridden from a JSON file (``Config.load(path)``) and/or environment
variables (prefix ``AT_``). Environment variables win over the file, which wins
over the defaults.

Kept dependency-free on purpose (stdlib only) so the config can be imported by
every component, including the lightweight API service.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class DetectorConfig:
    model_path: str = "yolo11m.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.70
    person_class_id: int = 0  # COCO "person"
    image_size: int = 1280


@dataclass
class TrackerConfig:
    # supervision.ByteTrack parameters.
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.80
    frame_rate: int = 30


@dataclass
class ReIDConfig:
    enabled: bool = True
    model_name: str = "osnet_x1_0"
    # Re-embed *already identified* tracks at most this often (frames). New /
    # unmatched tracks are always embedded immediately (see pipeline).
    update_every: int = 5
    # Cosine-similarity threshold for reusing an existing GID (Identity Rule 2).
    similarity_threshold: float = 0.60
    # EMA weight for the running average appearance embedding.
    embedding_alpha: float = 0.10


@dataclass
class IdentityConfig:
    # How long a disappeared identity remains eligible for ReID re-matching.
    lost_timeout_seconds: float = 8.0
    # How long an identity is kept in memory for stats/API after going lost.
    # GIDs are NEVER reused regardless of this value (Identity Rule 4).
    forget_timeout_seconds: float = 300.0


@dataclass
class OverlayConfig:
    debug: bool = False
    box_thickness: int = 2
    font_scale: float = 0.6
    box_color: tuple[int, int, int] = (0, 220, 120)   # BGR
    text_color: tuple[int, int, int] = (255, 255, 255)
    lost_color: tuple[int, int, int] = (90, 90, 90)


@dataclass
class LoggingConfig:
    enabled: bool = True
    # JSONL frame log (one record per processed frame). Replayable.
    frame_log_path: str = "logs/frames.jsonl"


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Application-level keepalive for clients that do not expose WebSocket pings
    # cleanly (for example TouchDesigner WebSocket DAT).
    ws_heartbeat_interval_s: float = 10.0


@dataclass
class CameraConfig:
    """Physical camera settings for the local Capture Agent (Video Ingestion spec)."""
    device_index: int = 0      # OpenCV device index; agent also accepts file/RTSP via source
    width: int = 1920
    height: int = 1080
    fps: int = 30


@dataclass
class LensCalibrationConfig:
    """Optional camera-lens calibration used before floor projection.

    Fisheye correction is applied to points, not necessarily to full frames, so
    detection can keep running against the raw camera image.
    """

    enabled: bool = False
    model: str = "fisheye"  # fisheye | pinhole
    image_size: tuple[int, int] = (0, 0)
    camera_matrix: list[list[float]] = field(default_factory=list)
    distortion_coeffs: list[float] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    """Image-to-floor calibration for top-down 2D coordinates."""

    enabled: bool = False
    anchor: str = "bottom_center"  # bottom_center | center
    floor_space: str = "normalized"
    image_points: list[list[float]] = field(default_factory=list)
    floor_points: list[list[float]] = field(default_factory=list)
    smoothing_alpha: float = 0.35


@dataclass
class ZonesConfig:
    """Named regions in calibrated floor space."""

    enabled: bool = False
    default_zone: str | None = None
    zones: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CaptureAgentConfig:
    """Local Capture Agent: reads the camera and streams JPEG frames to Modal."""
    # Modal ingestion WebSocket URL, e.g. wss://<app>.modal.run/ingest
    server_url: str = "ws://localhost:8000/ingest"
    # Bearer token presented on connect; must match the ingestion service token.
    token: str = ""
    jpeg_quality: int = 85         # 1-100; spec default 85%
    # Optional downscale before encoding (0 = send at capture resolution). Kept as
    # int (not None) so AT_* env overrides coerce to int rather than passing a str.
    resize_width: int = 0
    resize_height: int = 0
    # Local status endpoint (GET /status).
    status_host: str = "127.0.0.1"
    status_port: int = 9000
    # Reconnect backoff bounds (seconds), exponential between them.
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 30.0


@dataclass
class IngestConfig:
    """Modal Ingestion Service: receives the Capture Agent's frame stream."""
    # Bearer token required from clients. Empty string disables auth (dev only).
    token: str = ""
    # Bounded frame queue depth (spec recommends 2-3); oldest dropped when full.
    queue_size: int = 3
    # How long the pipeline waits for a frame before looping (keeps it responsive
    # to shutdown and lets tracking idle while the agent is disconnected).
    frame_timeout_s: float = 1.0


@dataclass
class PipelineConfig:
    # Frame source: camera index ("0"), file path, RTSP/HTTP URL, or the literal
    # "ingest" to consume the Modal ingestion queue fed by the Capture Agent.
    source: str = "0"
    # "auto": use the real ML stack if importable, otherwise fall back to mocks.
    # "real": force the ML stack. "mock": force the synthetic stack (no GPU).
    backend: str = "auto"
    device: str = "auto"  # auto | cpu | cuda
    # Optional path to write an annotated output video.
    output_path: str | None = None
    # Throttle processing to at most this FPS (None = as fast as possible).
    max_fps: float | None = None
    # Render the overlay (costs CPU). Disable for headless metric runs.
    render_overlay: bool = True
    # Publish annotated frames as an MJPEG stream at GET /video (needs OpenCV).
    stream_overlay: bool = True
    # Start the tracking pipeline alongside the API service.
    run_pipeline: bool = True
    # Number of simulated people when running the mock backend.
    mock_people: int = 24


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    lens_calibration: LensCalibrationConfig = field(default_factory=LensCalibrationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    capture_agent: CaptureAgentConfig = field(default_factory=CaptureAgentConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Build a Config from defaults, an optional JSON file, then env vars."""
        cfg = cls()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                _merge(cfg, json.load(fh))
        cfg.apply_env()
        return cfg

    def apply_env(self, environ: dict[str, str] | None = None) -> "Config":
        """Override fields from ``AT_<SECTION>_<FIELD>`` environment variables.

        Example: ``AT_REID_SIMILARITY_THRESHOLD=0.7``,
        ``AT_PIPELINE_SOURCE=rtsp://cam/stream``.
        """
        environ = environ if environ is not None else dict(os.environ)
        for section_name in _section_names(self):
            section = getattr(self, section_name)
            for f in fields(section):
                env_key = f"AT_{section_name.upper()}_{f.name.upper()}"
                if env_key in environ:
                    setattr(section, f.name, _coerce(getattr(section, f.name), environ[env_key]))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _section_names(cfg: Config) -> list[str]:
    return [f.name for f in fields(cfg) if is_dataclass(getattr(cfg, f.name))]


def _merge(cfg: Config, data: dict[str, Any]) -> None:
    for section_name, values in data.items():
        if not hasattr(cfg, section_name):
            continue
        section = getattr(cfg, section_name)
        if not is_dataclass(section) or not isinstance(values, dict):
            continue
        valid = {f.name for f in fields(section)}
        for key, val in values.items():
            if key in valid:
                setattr(section, key, val)


def _coerce(current: Any, raw: str) -> Any:
    """Coerce an env string to the type of the current value."""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if current is None:
        return raw
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        parts = [p.strip() for p in raw.split(",")]
        return tuple(int(p) for p in parts)
    return raw
