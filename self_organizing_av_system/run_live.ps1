Write-Host "==============================================="
Write-Host "Self-Organizing Audio-Visual Learning System"
Write-Host "Live Demo (Webcam & Microphone)"
Write-Host "==============================================="
Write-Host ""

# Install required dependencies for visualization
Write-Host "Installing dependencies..."
pip install PyQt5 matplotlib

# Set Python path to include the current directory
$pwd_escaped = $pwd -replace '\\', '\\\\'
$env:PYTHONPATH += ";$pwd"

# Run the live demo directly
Write-Host "Starting live demo..."
python examples/run_live_demo.py

Write-Host ""
Write-Host "Press any key to exit..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null 