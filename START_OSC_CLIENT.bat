@echo off
REM Beta OSC Client Launcher - External command execution for Beta brain
REM This client reads beta_commands.json and executes VRChat OSC commands
REM providing complete separation between analysis and execution phases

echo Starting Beta OSC Client...
echo.
echo This client will:
echo - Monitor beta_commands.json for new commands
echo - Execute validated OSC commands to VRChat
echo - Provide safety validation and rate limiting
echo - Log all command execution
echo.
echo Press Ctrl+C to stop the client
echo.

cd /d "G:\Experimental\Production\MidnightCore"

REM Launch the OSC client with the Python interpreter
"G:\Experimental\Production\MidnightCore\Core\Common\.venv\Scripts\python.exe" Core\Engine\osc_client.py

echo.
echo Beta OSC Client stopped.
pause