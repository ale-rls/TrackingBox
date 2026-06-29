"""Video ingestion & transport subsystem (Video Ingestion spec V1).

Delivers live camera frames from a local venue to the cloud Tracking Pipeline:

    Capture Agent (venue)  --WSS-->  Ingestion Service (Modal)  -->  FrameSource

This package owns *acquisition and transport only* — no detection, tracking,
ReID, overlay or identity logic lives here.
"""

from .frame_source import OpenCVFrameSource, QueueFrameSource, SimulatorFrameSource
from .packet import InvalidPacket, MAGIC, decode_packet, encode_packet

__all__ = [
    "OpenCVFrameSource",
    "QueueFrameSource",
    "SimulatorFrameSource",
    "InvalidPacket",
    "MAGIC",
    "decode_packet",
    "encode_packet",
]
