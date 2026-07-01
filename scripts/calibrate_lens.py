"""Estimate fisheye lens calibration from checkerboard samples.

Examples:
    python scripts/calibrate_lens.py --config config.json --source 0
    python scripts/calibrate_lens.py --config config.json --images "calib/*.jpg"

The board dimensions are inner-corner counts, not printed square counts.
"""

from __future__ import annotations

import argparse
import glob

from calibration_common import open_capture, update_section


def main() -> int:
    args = _parse_args()
    frames = _load_image_frames(args.images) if args.images else _capture_live_frames(args)
    rms, usable_samples, image_size, camera_matrix, distortion_coeffs = _calibrate_fisheye(
        frames, args
    )

    update_section(
        args.config,
        "lens_calibration",
        {
            "enabled": True,
            "model": "fisheye",
            "image_size": list(image_size),
            "camera_matrix": camera_matrix,
            "distortion_coeffs": distortion_coeffs,
        },
    )
    print(f"Wrote fisheye calibration to {args.config}")
    print(f"Usable samples: {usable_samples}  RMS reprojection error: {rms:.4f}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Config JSON to update")
    parser.add_argument("--source", default="0", help="Camera index or stream URL")
    parser.add_argument("--images", help="Optional glob of checkerboard still images")
    parser.add_argument("--board-cols", type=int, default=9, help="Checkerboard inner columns")
    parser.add_argument("--board-rows", type=int, default=6, help="Checkerboard inner rows")
    parser.add_argument("--square-size", type=float, default=1.0, help="Physical square size")
    parser.add_argument("--samples", type=int, default=15, help="Live samples to collect")
    return parser.parse_args()


def _load_image_frames(pattern: str):
    import cv2

    frames = []
    for name in sorted(glob.glob(pattern)):
        frame = cv2.imread(name)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No readable calibration images matched: {pattern}")
    return frames


def _capture_live_frames(args: argparse.Namespace):
    import cv2

    cap = open_capture(args.source, args.config)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open calibration source: {args.source}")

    frames = []
    window = "lens calibration"
    pattern_size = (args.board_cols, args.board_rows)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while len(frames) < args.samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not capture frame from source: {args.source}")
            found, corners = _find_corners(frame, pattern_size)
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, pattern_size, corners, found)
            status = "found" if found else "not found"
            message = (
                f"{len(frames)}/{args.samples} saved, board {status}; "
                "space=save, q/esc=quit"
            )
            cv2.putText(
                preview,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Lens calibration cancelled")
            if key == ord(" ") and found:
                frames.append(frame.copy())
    finally:
        cap.release()
        cv2.destroyWindow(window)
    return frames


def _calibrate_fisheye(frames, args: argparse.Namespace):
    import cv2
    import numpy as np

    pattern_size = (args.board_cols, args.board_rows)
    objp = np.zeros((1, args.board_cols * args.board_rows, 3), np.float64)
    grid = np.mgrid[0 : args.board_cols, 0 : args.board_rows].T.reshape(-1, 2)
    objp[0, :, :2] = grid * args.square_size

    objpoints = []
    imgpoints = []
    image_size = None
    for frame in frames:
        found, corners = _find_corners(frame, pattern_size)
        if not found:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif image_size != size:
            raise RuntimeError("All calibration frames must have the same size")
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))

    if image_size is None or len(objpoints) < 3:
        raise RuntimeError("Need at least 3 usable checkerboard samples")

    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    rms, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
        flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    return (
        float(rms),
        len(objpoints),
        image_size,
        camera_matrix.tolist(),
        distortion.reshape(-1).tolist(),
    )


def _find_corners(frame, pattern_size):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
    return True, corners


if __name__ == "__main__":
    raise SystemExit(main())
