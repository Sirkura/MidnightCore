@echo off
REM Portable Emergency Stop for MidnightCore - Auto-detects location
echo ========================================
echo MIDNIGHT CORE EMERGENCY STOP (PORTABLE)
echo ========================================
echo.

REM Get the directory where this script is located and navigate to MidnightCore root
set "SCRIPT_DIR=%~dp0"
set "MIDNIGHT_CORE_ROOT=%SCRIPT_DIR%..\..\.."

echo Script location: %SCRIPT_DIR%
echo MidnightCore root: %MIDNIGHT_CORE_ROOT%
echo.

echo Killing all Python processes...
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM pythonw.exe /F >nul 2>&1
echo Python processes terminated.
echo.

echo Sending OSC reset commands...
REM Use portable Python path
"%MIDNIGHT_CORE_ROOT%\Core\Common\.venv\Scripts\python.exe" -c "from pythonosc import udp_client; import time; client = udp_client.SimpleUDPClient('127.0.0.1', 9000); client.send_message('/input/Vertical', 0.0); client.send_message('/input/Horizontal', 0.0); client.send_message('/input/LookVertical', 0.0); client.send_message('/input/LookLeft', 0); client.send_message('/input/LookRight', 0); client.send_message('/input/Jump', 0); client.send_message('/input/Run', 0); print('SUCCESS: All OSC controls reset to safe state')" 2>nul || echo WARNING: OSC reset failed - VRChat may not be running

echo.
echo Clearing command files...
REM Clear any pending OSC commands
if exist "%MIDNIGHT_CORE_ROOT%\beta_commands.json" (
    echo {"action": "", "tick_id": 0, "frame_id": 0, "timestamp": 0} > "%MIDNIGHT_CORE_ROOT%\beta_commands.json"
    echo Command files cleared
) else (
    echo No command files found to clear
)

echo.
echo ========================================
echo EMERGENCY STOP COMPLETE (PORTABLE)
echo All processes killed and controls reset
echo ========================================
pause