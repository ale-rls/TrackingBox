# TouchDesigner - launch the local audience-tracker service from a preset table.
#
# Put this in a Text DAT named td_launch_tracker, or keep it external and
# reference it with mod(). It runs inside TouchDesigner, but launches the repo's
# separate .venv executable.
#
# Expected Table DAT:
#   tracker_presets
#
# Header row can use these columns:
#   name, source, backend, device, port, reid, confidence, image_size, debug
#
# Useful example row:
#   HDMI USB Camera, 0, real, cuda, 8000, 0, 0.15, 1280, 0
#
# Button callbacks:
#   mod('td_launch_tracker').launch_selected()
#   mod('td_launch_tracker').stop()
#   mod('td_launch_tracker').restart_selected()

import os
import subprocess

PRESETS_TABLE = 'tracker_presets'
STATUS_DAT = 'tracker_status'
SELECTED_PRESET = 'selected_preset'

REPO_ROOT = r'C:\Users\Interrobang\Documents\Enter The Blackbox\TrackingBox'
TRACKER_EXE = os.path.join(REPO_ROOT, '.venv', 'Scripts', 'audience-tracker.exe')

_state = {'process': None}


def _status(message):
    dat = op(STATUS_DAT)
    if dat is not None:
        dat.clear()
        dat.appendRow([message])
    print('[audience-tracker]', message)


def _text(value, default=''):
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _truthy(value):
    return _text(value).lower() in ('1', 'true', 'yes', 'on', 'enabled')


def _table_rows(table):
    if table is None or table.numRows < 2:
        return []
    headers = [_text(table[0, c]).lower() for c in range(table.numCols)]
    rows = []
    for r in range(1, table.numRows):
        row = {}
        for c, header in enumerate(headers):
            if header:
                row[header] = _text(table[r, c])
        rows.append(row)
    return rows


def _selected_name():
    selector = op(SELECTED_PRESET)
    if selector is None or selector.numRows == 0 or selector.numCols == 0:
        return ''
    return _text(selector[0, 0])


def _preset_by_name(name=''):
    rows = _table_rows(op(PRESETS_TABLE))
    if not rows:
        raise RuntimeError('Preset table is empty or missing: {}'.format(PRESETS_TABLE))

    name = _text(name) or _selected_name()
    if name:
        for row in rows:
            if row.get('name', '').lower() == name.lower():
                return row
        raise RuntimeError('Preset not found: {}'.format(name))
    return rows[0]


def _env_with_preset(preset):
    env = os.environ.copy()
    confidence = preset.get('confidence') or preset.get('conf')
    image_size = preset.get('image_size') or preset.get('imagesize')
    debug = preset.get('debug') or preset.get('overlay_debug')

    if confidence:
        env['AT_DETECTOR_CONFIDENCE_THRESHOLD'] = confidence
    if image_size:
        env['AT_DETECTOR_IMAGE_SIZE'] = image_size
    if debug:
        env['AT_OVERLAY_DEBUG'] = 'true' if _truthy(debug) else 'false'
    return env


def _command_from_preset(preset):
    source = preset.get('source') or preset.get('camera') or preset.get('index') or '0'
    backend = preset.get('backend') or 'real'
    device = preset.get('device') or 'cuda'
    port = preset.get('port') or '8000'

    cmd = [
        TRACKER_EXE,
        'serve',
        '--backend', backend,
        '--device', device,
        '--source', source,
        '--port', port,
    ]

    if not _truthy(preset.get('reid')):
        cmd.append('--no-reid')

    return cmd


def is_running():
    process = _state.get('process')
    return process is not None and process.poll() is None


def launch_selected(name=''):
    if is_running():
        _status('Already running')
        return

    if not os.path.exists(TRACKER_EXE):
        raise RuntimeError('Missing tracker executable. Run scripts\\install_windows.bat first.')

    preset = _preset_by_name(name)
    cmd = _command_from_preset(preset)
    env = _env_with_preset(preset)

    creationflags = 0
    if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        creationflags=creationflags,
    )
    _state['process'] = process
    _status('Launched {} on source {} port {}'.format(
        preset.get('name', 'preset'),
        preset.get('source', '0'),
        preset.get('port', '8000'),
    ))


def stop():
    process = _state.get('process')
    if process is None or process.poll() is not None:
        _state['process'] = None
        _status('Not running')
        return
    process.terminate()
    _state['process'] = None
    _status('Stopped')


def restart_selected(name=''):
    stop()
    launch_selected(name)
