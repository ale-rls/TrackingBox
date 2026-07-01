# TouchDesigner helper for polling GET /api/zones/counts into a Table DAT.
#
# Put this in a Text DAT named 'td_fetch_zone_counts', create a Table DAT named
# 'zone_counts', then call:
#
#   mod('td_fetch_zone_counts').poll()
#
# from a Timer CHOP, Execute DAT, or button callback. Optional: create a Text DAT
# named 'zone_counts_url' whose first cell overrides the URL.

import json
from urllib.request import urlopen

TABLE = 'zone_counts'
URL = 'http://localhost:8000/api/zones/counts'
HEADERS = ['zone', 'count']


def _table():
    t = op(TABLE)
    if t is None:
        return None
    if t.numRows == 0:
        t.appendRow(HEADERS)
    return t


def _url():
    override = op('zone_counts_url')
    if override is not None and override.numRows > 0 and override.numCols > 0:
        value = str(override[0, 0]).strip()
        if value:
            return value
    return URL


def _replace(counts):
    t = _table()
    if t is None:
        return
    t.clear()
    t.appendRow(HEADERS)
    for zone_id in sorted(counts):
        t.appendRow([zone_id, counts[zone_id]])


def poll():
    try:
        with urlopen(_url(), timeout=0.5) as response:
            counts = json.loads(response.read().decode('utf-8'))
    except Exception:
        return
    if isinstance(counts, dict):
        _replace(counts)
    return
