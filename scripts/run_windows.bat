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

REM Defaults: real backend, GPU, local camera 0, ReID on. Override via args
REM (pass --no-reid if torchreid is not installed on this machine).
"%VENV%\audience-tracker.exe" serve --backend real --device cuda --source 0 --port 8000 %*
if errorlevel 1 (
  echo.
  echo The tracker exited with an error. Run "%VENV%\audience-tracker.exe" doctor
  echo to check this machine. If ReID/torchreid is not installed here, retry with:
  echo    run_windows.bat --no-reid
  pause
)
