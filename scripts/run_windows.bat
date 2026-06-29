@echo off
REM Launch the tracking service on Windows (Mode A: the service opens the camera).
REM Extra args are passed through, e.g.:  run_windows.bat --source 1 --port 9000
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%..\.venv\Scripts"

if not exist "%VENV%\audience-tracker.exe" (
  echo Could not find the virtual environment.
  echo Run scripts\install_windows.bat first.
  pause
  exit /b 1
)

REM Defaults: real backend, GPU, local camera 0, ReID off. Override via args.
"%VENV%\audience-tracker.exe" serve --backend real --device cuda --source 0 --no-reid --port 8000 %*
