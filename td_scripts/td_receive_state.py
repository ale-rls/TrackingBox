# TouchDesigner — WebSocket DAT callbacks for the audience /ws stream.
#
# Attach this as the *Callbacks DAT* of a WebSocket DAT pointed at
#   ws://localhost:8000/ws
# It keeps a table DAT named 'audience' in sync with the live audience state so
# you can drive instancing, labels, particles, etc. from persistent GIDs.
#
# The server sends:
#   * once on connect:  {"type":"snapshot","data":{"people":[{gid,center,bbox,floor,...}]}}
#   * then per change:  {"gid":17,"visible":true,"center":[cx,cy],"bbox":[...],"floor":[fx,fy]}
#
# Only GIDs are ever exposed — there are no tracker ids here, by design.
#
# Setup: create a Table DAT called 'audience'. This script writes these columns:
#   gid  visible  cx  cy  x1  y1  x2  y2  floor_x  floor_y  floor_valid
#
# This file runs inside TouchDesigner's Python (it references op()/parent()); it
# is not part of the Python package and is not unit-tested here.

import json

TABLE = 'audience'
HEADERS = [
    'gid',
    'visible',
    'cx',
    'cy',
    'x1',
    'y1',
    'x2',
    'y2',
    'floor_x',
    'floor_y',
    'floor_valid',
]


def _table():
    t = op(TABLE)
    if t is None:
        return None
    if t.numRows == 0:
        t.appendRow(HEADERS)
    return t


def _row_values(entry):
    gid = entry.get('gid')
    center = entry.get('center') or [0, 0]
    bbox = entry.get('bbox') or [0, 0, 0, 0]
    floor = entry.get('floor') or [0, 0]
    visible = 1 if entry.get('visible') else 0
    floor_valid = 1 if entry.get('floor_valid') else 0
    return [
        gid,
        visible,
        center[0],
        center[1],
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        floor[0],
        floor[1],
        floor_valid,
    ]


def _upsert(entry):
    t = _table()
    if t is None or entry.get('gid') is None:
        return
    gid = str(entry['gid'])
    vals = _row_values(entry)
    cell = t.findCell(gid, cols=['gid'])
    if not entry.get('visible'):
        # Person no longer visible -> drop their row.
        if cell is not None:
            t.deleteRow(cell.row)
        return
    if cell is None:
        t.appendRow(vals)
    else:
        for col, v in zip(HEADERS, vals):
            t[cell.row, col] = v


def _replace_all(people):
    t = _table()
    if t is None:
        return
    t.clear()
    t.appendRow(HEADERS)
    for entry in people:
        if entry.get('visible', True):
            t.appendRow(_row_values(entry))


def onReceiveText(dat, rowIndex, message):
    try:
        msg = json.loads(message)
    except Exception:
        return
    if isinstance(msg, dict) and msg.get('type') == 'snapshot':
        _replace_all(msg.get('data', {}).get('people', []))
    elif isinstance(msg, dict) and 'gid' in msg:
        _upsert(msg)
    return


# Optional: clear the table if the connection drops.
def onDisconnect(dat):
    t = op(TABLE)
    if t is not None:
        t.clear()
        t.appendRow(HEADERS)
    return
