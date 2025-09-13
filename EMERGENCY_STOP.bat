@echo off
echo ========================================
echo MIDNIGHT CORE EMERGENCY STOP
echo ========================================
echo.
echo Killing all Python processes...
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM pythonw.exe /F >nul 2>&1
echo Python processes terminated.
echo.
echo Sending OSC reset commands...
"G:\Experimental\Production\MidnightCore\Core\Common\.venv\Scripts\python.exe" -c "from pythonosc import udp_client; import time; client = udp_client.SimpleUDPClient('127.0.0.1', 9000); client.send_message('/input/Vertical', 0.0); client.send_message('/input/Horizontal', 0.0); client.send_message('/input/LookVertical', 0.0); client.send_message('/input/LookLeft', 0); client.send_message('/input/LookRight', 0); client.send_message('/input/Jump', 0); client.send_message('/input/Run', 0); print('SUCCESS: All OSC controls reset to safe state')" 2>nul || echo WARNING: OSC reset failed - VRChat may not be running
echo.
echo ========================================
echo EMERGENCY STOP COMPLETE
echo All processes killed and controls reset
echo ========================================
pause