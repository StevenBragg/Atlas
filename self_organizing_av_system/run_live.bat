@echo off
echo ===============================================
echo Self-Organizing Audio-Visual Learning System
echo "Live Demo (Webcam & Microphone)"
echo ===============================================
echo.

REM Install required dependencies for visualization
echo Installing dependencies...
pip install PyQt5 matplotlib

REM Set Python path to include the current directory
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Run the live demo in a way that ensures the window stays open
echo Starting live demo...
python -c "import sys; sys.path.append('%CD%'); from examples import run_live_demo; run_live_demo.main()"

echo.
pause 