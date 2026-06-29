# TouchDesigner — stream frames from a TOP to the audience tracker /ingest WS.
#
# Use this for Mode B (TouchDesigner feeds the frames). For Mode A (the service
# opens the camera itself) you do NOT need this file.
#
# Wiring:
#   1. A WebSocket DAT (call it 'ingest_ws') pointed at:
#         ws://localhost:8000/ingest          (or /ingest?token=<token>)
#   2. A source TOP to analyze (camera, composite, etc.) — referenced below.
#   3. An Execute DAT with this file, calling send_frame() from onFrameEnd.
#
# Requires OpenCV + numpy in TouchDesigner's Python:
#   "<TouchDesigner>/bin/python" -m pip install opencv-python numpy
#
# Each frame is sent as JSON text the server understands:
#   {"frame_id": N, "timestamp": t, "width": w, "height": h, "jpeg_b64": "..."}
#
# This file runs inside TouchDesigner's Python; it is not part of the package
# and is not unit-tested here.

import base64
import json
import time

import cv2
import numpy as np

SOURCE_TOP = 'camera'     # name of the TOP to analyze
INGEST_DAT = 'ingest_ws'  # name of the WebSocket DAT connected to /ingest
JPEG_QUALITY = 85
SEND_EVERY = 1            # send every Nth frame (raise to throttle bandwidth)

_state = {'frame_id': 0}


def _top_to_bgr(top):
    # numpyArray() returns RGBA float32 in [0, 1], origin bottom-left.
    arr = top.numpyArray(delayed=False)
    if arr is None:
        return None
    rgb = (arr[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.flipud(rgb)                 # flip to top-left origin
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def send_frame():
    ws = op(INGEST_DAT)
    src = op(SOURCE_TOP)
    if ws is None or src is None:
        return
    _state['frame_id'] += 1
    if _state['frame_id'] % SEND_EVERY != 0:
        return

    img = _top_to_bgr(src)
    if img is None:
        return
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return

    h, w = img.shape[:2]
    msg = json.dumps({
        'frame_id': _state['frame_id'],
        'timestamp': time.time(),
        'width': w,
        'height': h,
        'jpeg_b64': base64.b64encode(buf.tobytes()).decode('ascii'),
    })
    ws.sendText(msg)


# In an Execute DAT, call send_frame() from onFrameEnd:
#
# def onFrameEnd(frame):
#     mod('td_send_frames').send_frame()
#     return
