@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo Self-Organizing Audio-Visual Learning System
echo Video File Processing
echo ===============================================
echo.

:: Check if a video file was provided
if "%~1"=="" (
    echo Please drag and drop a video file onto this batch file.
    echo.
    echo For example:
    echo 1. Find your video file in File Explorer
    echo 2. Drag the video file and drop it on this batch file icon
    pause
    exit /b 1
)

:: Activate the conda environment
call conda activate self_learn

:: Set the Python path to include the parent directory
set PYTHONPATH=%~dp0..

:: Run the file demo with the provided video file
python "%~dp0examples/run_file_demo.py" "%~1"

pause 