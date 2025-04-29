@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo Self-Organizing Audio-Visual Learning System
echo Installation Script
echo ===============================================
echo.

:: Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first.
    echo Visit: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Check if environment exists
conda env list | findstr /C:"self_learn" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Creating new conda environment 'self_learn'...
    conda create -y -n self_learn python=3.10
) else (
    echo Conda environment 'self_learn' already exists.
)

:: Activate environment and install packages
echo.
echo Installing required packages...
echo This may take a few minutes...
echo.

:: Use call to prevent the script from exiting after conda activate
call conda activate self_learn

:: Install the package and dependencies using the public PyPI repository
pip install -e . --no-deps

:: Install dependencies directly from PyPI
pip install -r requirements.txt --index-url https://pypi.org/simple

:: Reset the pip index URL to default
pip config unset global.index-url >nul 2>&1

echo.
echo ===============================================
echo Installation complete!
echo.
echo To run the program:
echo.
echo 1. Open Command Prompt or PowerShell
echo 2. Run: conda activate self_learn
echo 3. Choose one of the following commands:
echo.
echo    For live demo (webcam and microphone):
echo    python examples/run_live_demo.py
echo.
echo    For processing a video file:
echo    python examples/run_file_demo.py path/to/your/video.mp4
echo.
echo ===============================================

pause 