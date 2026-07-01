"""Click floor reference points and write image-to-floor calibration config.

Example:
    python scripts/calibrate_floor.py --config config.json \
      --source 0 --floor-points "0,0;1,0;1,1;0,1"

Click the visible floor points in the same order as --floor-points. The clicked
image points are stored in raw camera/image coordinates; the runtime projector
applies lens undistortion before the homography when lens calibration is enabled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calibration_common import parse_points, update_section


def main() -> int:
    args = _parse_args()
    floor_points = parse_points(args.floor_points)
    frame = _capture_frame(args.source)
    image_points = _collect_clicks(frame, len(floor_points))

    update_section(
        args.config,
        "calibration",
        {
            "enabled": True,
            "anchor": args.anchor,
            "floor_space": args.floor_space,
            "image_points": image_points,
            "floor_points": floor_points,
            "smoothing_alpha": args.smoothing_alpha,
        },
    )
    print(f"Wrote {len(image_points)} floor calibration points to {args.config}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Config JSON to update")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, image/video file, or stream URL used for the calibration still",
    )
    parser.add_argument(
        "--floor-points",
        required=True,
        help="Known floor points as 'x,y;x,y;...' in the same order you will click",
    )
    parser.add_argument(
        "--anchor",
        default="bottom_center",
        choices=["bottom_center", "center"],
        help="BBox anchor used later by runtime projection",
    )
    parser.add_argument("--floor-space", default="normalized", help="Label for floor units")
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    return parser.parse_args()


def _capture_frame(source: str):
    import cv2

    path = Path(source)
    if path.exists():
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image file: {source}")
        return frame

    capture_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open calibration source: {source}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not capture frame from source: {source}")
        return frame
    finally:
        cap.release()


def _collect_clicks(frame, count: int) -> list[list[float]]:
    import cv2

    window = "floor calibration"
    clicks: list[list[float]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < count:
            clicks.append([float(x), float(y)])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while len(clicks) < count:
        preview = frame.copy()
        for index, (x, y) in enumerate(clicks, start=1):
            center = (int(round(x)), int(round(y)))
            cv2.circle(preview, center, 6, (0, 255, 0), -1)
            cv2.putText(
                preview,
                str(index),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        message = f"Click point {len(clicks) + 1}/{count}; r=reset, q/esc=quit"
        cv2.putText(preview, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            raise RuntimeError("Floor calibration cancelled")
        if key == ord("r"):
            clicks.clear()
    cv2.destroyWindow(window)
    return clicks


if __name__ == "__main__":
    raise SystemExit(main())
