#!/bin/bash
# ============================================
# Atlas + OpenClaw Entrypoint for Salad Cloud
# ============================================
# Starts OpenClaw gateway in the background,
# then runs the Atlas service in the foreground.
# ============================================

set -e

echo "============================================"
echo " ATLAS + OpenClaw on Salad Cloud"
echo "============================================"

# Start OpenClaw gateway in background
OPENCLAW_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
echo "Starting OpenClaw gateway on port ${OPENCLAW_PORT}..."

openclaw gateway \
  --port "${OPENCLAW_PORT}" \
  --allow-unconfigured \
  --bind lan \
  > /data/logs/openclaw.log 2>&1 &

OPENCLAW_PID=$!
echo "OpenClaw gateway started (PID: ${OPENCLAW_PID})"

# Graceful shutdown: forward signals to both processes
cleanup() {
  echo "Shutting down..."
  kill "${OPENCLAW_PID}" 2>/dev/null || true
  wait "${OPENCLAW_PID}" 2>/dev/null || true
  exit 0
}
trap cleanup SIGTERM SIGINT

# Start Atlas service in foreground
echo "Starting Atlas service..."
exec python3 -m cloud.salad_service
