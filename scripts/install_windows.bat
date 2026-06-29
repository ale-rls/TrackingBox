@echo off
REM Double-click / cmd-friendly wrapper around install_windows.ps1.
REM Passes through any args, e.g.:  install_windows.bat -Reid
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Install reported a problem ^(exit %ERRORLEVEL%^). See messages above.
  pause
)
