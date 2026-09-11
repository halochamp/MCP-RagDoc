#!/bin/zsh
# Double-click in Finder to start the bundled MCP-RagDoc local HTML UI.
set -u

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${MCP_RAG_PYTHON:-$APP_DIR/.venv/bin/python}"
PORT="${RAGDOC_UI_PORT:-8769}"
URL="http://127.0.0.1:${PORT}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Error: Python interpreter not found: $PYTHON"
  echo "Create a project-local .venv or set MCP_RAG_PYTHON."
  read -r "?Press Return to close..."
  exit 1
fi

cd "$APP_DIR" || exit 1
if curl --silent --fail --max-time 1 "$URL/" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Error: port $PORT is already in use by another process."
  read -r "?Press Return to close..."
  exit 1
fi
(
  for _ in {1..40}; do
    if curl --silent --fail --max-time 1 "$URL/" >/dev/null 2>&1; then open "$URL"; exit 0; fi
    sleep 0.25
  done
) &
WATCHER_PID=$!
cleanup() { kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$PYTHON" "$APP_DIR/main.py" --host 127.0.0.1 --port "$PORT"
