"""Wire-packet encode/decode and validation (Video Ingestion spec)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from audience_tracker.ingestion.packet import (  # noqa: E402
    InvalidPacket,
    MAGIC,
    decode_packet,
    encode_packet,
)


def test_round_trip_preserves_fields():
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
    data = encode_packet(frame_id=42, timestamp=1718000000.5, jpeg_bytes=jpeg, width=1920, height=1080)
    out = decode_packet(data)
    assert out["frame_id"] == 42
    assert out["timestamp"] == pytest.approx(1718000000.5)
    assert out["width"] == 1920 and out["height"] == 1080
    assert out["jpeg_bytes"] == jpeg


def test_empty_jpeg_rejected_on_encode():
    with pytest.raises(InvalidPacket):
        encode_packet(1, 1.0, b"", 10, 10)


@pytest.mark.parametrize(
    "bad",
    [
        b"",                        # empty
        b"XXXX" + b"\x00" * 8,      # bad magic
        MAGIC + b"\x00\x00\x00\x05",  # header length but no header/payload
    ],
)
def test_malformed_packets_rejected(bad):
    with pytest.raises(InvalidPacket):
        decode_packet(bad)


def test_missing_payload_rejected():
    # Valid magic + header, but zero JPEG bytes following.
    good = encode_packet(1, 1.0, b"x", 1, 1)
    header_only = good[: len(good) - 1]  # strip the single payload byte
    with pytest.raises(InvalidPacket):
        decode_packet(header_only)


def test_non_bytes_rejected():
    with pytest.raises(InvalidPacket):
        decode_packet("not-bytes")  # type: ignore[arg-type]


if __name__ == "__main__":
    test_round_trip_preserves_fields()
    test_empty_jpeg_rejected_on_encode()
    test_missing_payload_rejected()
    test_non_bytes_rejected()
    print("ok")
