#!/bin/bash
# Atlas Continuous Teacher - Safe Launcher v3
# Ensures only one instance runs at a time

ATLAS_DIR="/root/.openclaw/workspace/Atlas"
PID_FILE="$ATLAS_DIR/teacher_state/continuous_teacher.pid"
LOG_DIR="$ATLAS_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] Continuous teacher is already running (PID: $OLD_PID)"
        echo "[INFO] To restart, run: pkill -f continuous_teacher_v3.py"
        exit 1
    else
        echo "[INFO] Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

echo "[INFO] Starting Atlas Continuous Teacher v3..."
echo "[INFO] Logs: $LOG_DIR/continuous_teacher.log"

# Run in a loop with proper error handling
cd "$ATLAS_DIR" || exit 1
source venv/bin/activate

while true; do
    echo "[INFO] Starting teaching session at $(date)"
    
    # Run one session - capture exit code properly
    python3 continuous_teacher_v3.py >> "$LOG_DIR/continuous_teacher.log" 2>&1
    EXIT_CODE=$?
    
    # Show last few lines of output
    tail -15 "$LOG_DIR/continuous_teacher.log"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[INFO] Session completed successfully at $(date)"
    else
        echo "[WARNING] Session exited with code $EXIT_CODE at $(date)"
    fi
    
    # Wait before next session
    echo "[INFO] Sleeping for 60 seconds before next session..."
    sleep 60
done
