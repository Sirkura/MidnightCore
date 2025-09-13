@echo off
echo =====================================================
echo  MIDNIGHTCORE BETA - VRChat AI Brain System (Session 21+)
echo =====================================================
echo.
echo Starting Beta with Hybrid Navigation Integration...
echo - Unified NDJSON logging: ALWAYS ACTIVE
echo - Hybrid navigation safety: ENABLED
echo - Two-tier analysis: Depth + Florence-2
echo - Real-time threat detection: ENABLED
echo - Movement safety checks: ACTIVE
echo.
echo Emergency Stop: Run EMERGENCY_STOP.bat or type "qwen stop" in VRChat
echo.
echo =====================================================
echo              SETUP PREPARATION
echo =====================================================
echo.
echo Before starting Beta, please:
echo 1. Move any panels or windows out of Beta's view
echo 2. Position VRChat window for optimal visibility
echo 3. Ensure VRChat is focused and ready
echo.
echo Press any key when ready to start Beta's vision system...
pause
echo.
echo Starting Beta in 3 seconds...
timeout /t 3 /nobreak >nul
echo.

REM Set unified logging mode (always active)
set ENGINE_LOG=enabled
set DEEP_TRACE=disabled
set FILE_MAP=disabled
set MAX_TICKS=999999

REM Navigate to the correct directory
cd /d "G:\Experimental\Production\MidnightCore"

REM Launch the brain with hybrid navigation
echo [BETA] Initializing Hybrid Navigation Brain System...
echo [BETA] Unified logs: Core\Engine\Logging\events.ndjson
echo [BETA] Legacy logs: Core\Engine\Logging\.engine_log\engine.log
echo [BETA] Hybrid navigation: TWO-TIER ANALYSIS ACTIVE
echo.

"G:\Experimental\Production\MidnightCore\Core\Common\.venv\Scripts\python.exe" Core\Engine\Qwen_Brain_ActiveTesting.py

echo.
echo =====================================================
echo            BETA SESSION ENDED (Session 21+)
echo =====================================================
echo.
echo LOGS GENERATED:
echo - Unified events: Core\Engine\Logging\events.ndjson
echo - Navigation decisions: Real-time in events.ndjson
echo - Legacy engine: Core\Engine\Logging\.engine_log\engine.log
echo - Chat history: Core\Engine\Logging\Chat-Log.md
echo.
echo HYBRID NAVIGATION STATUS:
echo - Two-tier depth + semantic analysis: ACTIVE
echo - Regional percentiles + temporal smoothing: ACTIVE
echo - Movement safety checks: INTEGRATED
echo - Florence-2 verification: ON-DEMAND
echo.
pause