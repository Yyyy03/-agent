#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/Users/tanglei/workspace/EverLoop/tl/最新金融智能体"
BACKEND_DIR="$APP_DIR/backend"
PYTHON_BIN="$APP_DIR/backend/venv/bin/python"
ENV_FILE="$APP_DIR/.env.local_mintcu"
BRIDGE_PORT="4182"
FRONTEND_PORT="3000"

read_env_value() {
  local key="$1"
  python3 - "$ENV_FILE" "$key" <<"PY2"
from pathlib import Path
import sys
path = Path(sys.argv[1])
key = sys.argv[2]
for line in path.read_text().splitlines():
    if line.startswith(key + "="):
        print(line.split("=", 1)[1].strip())
        break
PY2
}

BRIDGE_TOKEN="$(read_env_value FIRE_AGENT_BRIDGE_TOKEN)"
if [ -z "$BRIDGE_TOKEN" ]; then
  echo "missing FIRE_AGENT_BRIDGE_TOKEN in $ENV_FILE" >&2
  exit 1
fi

for port in "$BRIDGE_PORT" "$FRONTEND_PORT"; do
  pids="$(lsof -tiTCP:$port -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
  fi
done
sleep 1

cd "$BACKEND_DIR"
FIRE_AGENT_BRIDGE_TOKEN="$BRIDGE_TOKEN" nohup "$PYTHON_BIN" bridge/fire_agent_bridge.py \
  --host 127.0.0.1 \
  --port "$BRIDGE_PORT" \
  --env-file "$ENV_FILE" \
  > "$APP_DIR/latest-finagent-bridge.log" 2>&1 &

sleep 3
cd "$APP_DIR"
nohup /opt/homebrew/bin/node --env-file=.env.local_mintcu node_modules/vinext/dist/cli.js dev \
  > "$APP_DIR/latest-finagent-dev.log" 2>&1 &

sleep 3
lsof -nP -iTCP:$BRIDGE_PORT -sTCP:LISTEN 2>/dev/null || true
lsof -nP -iTCP:$FRONTEND_PORT -sTCP:LISTEN 2>/dev/null || true
