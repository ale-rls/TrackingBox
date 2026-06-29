"""Wire format for frame packets streamed Capture Agent -> Ingestion Service.

Each packet (spec "Packet Format") carries::

    { frame_id, timestamp, jpeg_bytes, width, height }

sent as a single **binary** WebSocket message. We avoid base64/JSON-wrapping the
JPEG (which would add ~33% overhead per frame) by using a compact framing:

    | MAGIC (4B) | header_len (uint32 BE, 4B) | header_json | jpeg_bytes |

``header_json`` is UTF-8 ``{"frame_id", "timestamp", "width", "height"}``. The
JPEG payload follows verbatim. Decoding validates every field before the frame
is admitted to the queue (spec: "Validate packets").
"""

from __future__ import annotations

import json
import struct
from typing import Any

MAGIC = b"ATF1"  # Audience Tracker Frame, v1
_HEADER_STRUCT = struct.Struct(">I")  # uint32 big-endian header length
_MAX_HEADER = 4096  # generous cap; the header is tiny JSON
_PREFIX_LEN = len(MAGIC) + _HEADER_STRUCT.size


class InvalidPacket(ValueError):
    """Raised when a received packet fails validation."""


def encode_packet(
    frame_id: int,
    timestamp: float,
    jpeg_bytes: bytes,
    width: int,
    height: int,
) -> bytes:
    """Serialize one frame packet to bytes for a binary WebSocket send."""
    if not jpeg_bytes:
        raise InvalidPacket("jpeg_bytes is empty")
    header = json.dumps(
        {
            "frame_id": int(frame_id),
            "timestamp": float(timestamp),
            "width": int(width),
            "height": int(height),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return MAGIC + _HEADER_STRUCT.pack(len(header)) + header + bytes(jpeg_bytes)


def decode_packet(data: bytes) -> dict[str, Any]:
    """Parse and validate a packet, returning its fields.

    Returns ``{frame_id:int, timestamp:float, width:int, height:int,
    jpeg_bytes:bytes}``. Raises :class:`InvalidPacket` on any malformation.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidPacket(f"expected bytes, got {type(data).__name__}")
    if len(data) < _PREFIX_LEN:
        raise InvalidPacket("packet shorter than header prefix")
    if data[: len(MAGIC)] != MAGIC:
        raise InvalidPacket("bad magic")

    (header_len,) = _HEADER_STRUCT.unpack(data[len(MAGIC) : _PREFIX_LEN])
    if header_len == 0 or header_len > _MAX_HEADER:
        raise InvalidPacket(f"implausible header length {header_len}")
    header_end = _PREFIX_LEN + header_len
    if header_end > len(data):
        raise InvalidPacket("truncated header")

    try:
        header = json.loads(data[_PREFIX_LEN:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidPacket(f"header not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise InvalidPacket("header is not an object")

    jpeg = data[header_end:]
    if not jpeg:
        raise InvalidPacket("missing jpeg payload")

    try:
        frame_id = int(header["frame_id"])
        timestamp = float(header["timestamp"])
        width = int(header["width"])
        height = int(header["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidPacket(f"bad header fields: {exc}") from exc
    if frame_id < 0:
        raise InvalidPacket("frame_id must be non-negative")
    if timestamp <= 0:
        raise InvalidPacket("timestamp must be positive")

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "width": width,
        "height": height,
        "jpeg_bytes": bytes(jpeg),
    }
