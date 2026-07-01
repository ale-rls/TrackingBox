# Floor Projection Calibration

Floor projection converts each tracked person's camera-space bounding box into a
top-down floor coordinate for TouchDesigner, phones, and later zone/game logic.

Runtime mapping:

```text
bbox -> bottom-center foot point -> optional fisheye undistortion -> homography -> floor
```

The public API keeps the original image-space fields and adds floor-space fields:

```json
{
  "gid": 17,
  "visible": true,
  "center": [812, 521],
  "bbox": [760, 230, 864, 824],
  "floor": [0.43, 0.71],
  "floor_valid": true,
  "zone": "answer_a"
}
```

If calibration is disabled or incomplete, `floor` is `null` and `floor_valid` is
`false`.

## Coordinate System

Use normalized floor coordinates unless the installation needs real-world units:

```text
[0,0] bottom/left corner of the calibrated floor plane
[1,0] bottom/right
[1,1] top/right
[0,1] top/left
```

TouchDesigner can scale `floor_x` and `floor_y` to any visual plane.

## Config

Use a local `config.json` for venue-specific settings. Keep it out of source
control; `config.example.json` documents the shared shape.

```json
{
  "lens_calibration": {
    "enabled": true,
    "model": "fisheye",
    "image_size": [1920, 1080],
    "camera_matrix": [[...], [...], [...]],
    "distortion_coeffs": [...]
  },
  "calibration": {
    "enabled": true,
    "anchor": "bottom_center",
    "floor_space": "normalized",
    "image_points": [[120, 820], [1750, 790], [1450, 360], [520, 350]],
    "floor_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "smoothing_alpha": 0.35
  }
}
```

`image_points` are raw camera-image points. At runtime, the projector applies
fisheye correction to those points before computing/applying the homography.

## Lens Calibration

Generate an A4 checkerboard target:

```bat
python scripts\generate_checkerboard.py --output outputs\checkerboard_a4.pdf
```

Print it at 100% / actual size. The default target is `9x6` inner corners,
which means the printed pattern has `10x7` squares.

Run lens calibration with a checkerboard visible to the fisheye camera:

```bat
python scripts\calibrate_lens.py --config config.json --source 12
```

Alternative, from saved checkerboard images:

```bat
python scripts\calibrate_lens.py --config config.json --images "calib\*.jpg"
```

The board size arguments are inner-corner counts:

```bat
python scripts\calibrate_lens.py --config config.json --source 12 --board-cols 9 --board-rows 6
```

Use at least several good samples across the full frame, especially near the
edges, because fisheye error is strongest there.

## Floor Calibration

Mark known floor points in the real space, then click them in the same order:

```bat
python scripts\calibrate_floor.py --config config.json --source 12 --floor-points "0,0;1,0;1,1;0,1"
```

Four floor points are the minimum. Use six to ten points when possible for a more
stable least-squares homography.

Recommended click targets:

* taped floor corners
* stage/platform corners
* measured grid intersections
* any point that is fixed, visible, and repeatable

## Running With Calibration

Start the service with the calibrated config:

```bat
.venv\Scripts\audience-tracker serve --config config.json
```

Check that projection is active:

```bat
curl http://localhost:8000/api/audience
```

Expected result:

```json
{
  "floor": [0.42, 0.68],
  "floor_valid": true
}
```

In TouchDesigner, `td_receive_state.py` writes:

```text
gid visible cx cy x1 y1 x2 y2 floor_x floor_y floor_valid zone
```

Use `floor_x` and `floor_y` for top-down instancing. Keep `floor_valid` as a gate
so uncalibrated or invalid people do not drive zone/game state.

Named regions and counts are documented in [Floor Zones](zones.md).

## Validation Checklist

1. Stand on each known floor marker and verify the reported `floor` coordinate.
2. Walk slowly across the floor and check for smooth, monotonic motion.
3. Check the edges of the camera frame; fisheye error usually appears there first.
4. Confirm the TouchDesigner `audience` table updates `floor_x` and `floor_y`.
5. Leave calibration disabled only for camera/tracker testing, not spatial work.
