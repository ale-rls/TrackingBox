# Floor Zones

Zones are named regions in calibrated floor space. They depend on Phase 1 floor
projection, so define them in `floor` coordinates rather than camera pixels.

Runtime mapping:

```text
bbox -> floor -> zone -> zone counts
```

When zones are enabled, each audience entry includes `zone`:

```json
{
  "gid": 17,
  "floor": [0.43, 0.71],
  "floor_valid": true,
  "zone": "answer_a"
}
```

If floor projection is invalid, `zone` is `null`. If a valid floor point is
outside all configured zones, `default_zone` is used when set.

## Config

Add a `zones` section to the same local `config.json` used for calibration:

```json
{
  "zones": {
    "enabled": true,
    "default_zone": "outside",
    "zones": [
      {"id": "answer_a", "label": "Answer A", "rect": [0, 0, 0.5, 1]},
      {"id": "answer_b", "label": "Answer B", "rect": [0.5, 0, 1, 1]}
    ]
  }
}
```

Supported shapes:

```json
{"id": "left", "rect": [0, 0, 0.5, 1]}
{"id": "right", "x": 0.5, "y": 0, "width": 0.5, "height": 1}
{"id": "triangle", "points": [[0, 0], [1, 0], [0, 1]]}
```

The first matching enabled zone wins. Put small special-purpose zones before
larger fallback zones when regions overlap.

## API

```bat
curl http://localhost:8000/api/zones
curl http://localhost:8000/api/zones/counts
```

`/api/zones` returns configured zone polygons after rectangles are normalized to
points. `/api/zones/counts` returns one integer count per configured zone, plus
`default_zone` when configured.

## TouchDesigner

`td_scripts/td_receive_state.py` writes a `zone` column to the `audience` table:

```text
gid visible cx cy x1 y1 x2 y2 floor_x floor_y floor_valid zone
```

For compact counts, use `td_scripts/td_fetch_zone_counts.py` with a Table DAT
named `zone_counts` and call:

```python
mod('td_fetch_zone_counts').poll()
```

from a Timer CHOP, Execute DAT, or button callback.

## Validation Checklist

1. Confirm `floor_valid` is `1` for people standing on the playable area.
2. Stand in each marked zone and verify `/api/audience` reports the expected `zone`.
3. Check `/api/zones/counts` while moving people between zones.
4. Confirm TouchDesigner receives both the per-person `zone` and count table.
