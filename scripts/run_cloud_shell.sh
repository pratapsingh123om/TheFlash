#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${STATE_ROOT:-$HOME/india-lightning-cloudshell}"
VENV_DIR="${VENV_DIR:-$STATE_ROOT/.venv}"
ASSET_ROOT="${ASSET_ROOT:-$STATE_ROOT/assets}"
OUTPUT_DIR="${OUTPUT_DIR:-$STATE_ROOT/runtime}"
LOG_FILE="${LOG_FILE:-$STATE_ROOT/server.log}"
PID_FILE="${PID_FILE:-$STATE_ROOT/server.pid}"
MODEL_BUNDLE_URI="${MODEL_BUNDLE_URI:-gs://nuclei-lenses/flash-model/india_lightning_24h_demo_ed5dec2770f7.zip}"
MODEL_BUNDLE_SHA256="${MODEL_BUNDLE_SHA256:-0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
PACKAGE_ROOT="$ASSET_ROOT/releases/$MODEL_BUNDLE_SHA256/india_lightning_24h_demo"

mkdir -p "$STATE_ROOT" "$ASSET_ROOT" "$OUTPUT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating persistent Python environment..."
  python3 -m venv "$VENV_DIR"
fi

REQUIREMENTS_SHA="$(sha256sum "$REPO_ROOT/requirements.txt" | awk '{print $1}')"
INSTALL_MARKER="$VENV_DIR/.requirements-$REQUIREMENTS_SHA"
if [[ ! -f "$INSTALL_MARKER" ]]; then
  echo "Installing backend dependencies..."
  "$VENV_DIR/bin/python" -m pip install \
    --disable-pip-version-check \
    --quiet \
    -r "$REPO_ROOT/requirements.txt"
  find "$VENV_DIR" -maxdepth 1 -name '.requirements-*' -type f -delete
  touch "$INSTALL_MARKER"
fi

echo "Downloading/verifying frozen bucket bundle (cached after first success)..."
(
  cd "$REPO_ROOT"
  MODEL_BUNDLE_URI="$MODEL_BUNDLE_URI" \
  MODEL_BUNDLE_SHA256="$MODEL_BUNDLE_SHA256" \
  ASSET_ROOT="$ASSET_ROOT" \
    "$VENV_DIR/bin/python" -m backend.bootstrap_assets
)

MODEL_DIR="$PACKAGE_ROOT/models"
INPUT_PARQUET="$PACKAGE_ROOT/data/next_24h_features.parquet"
if [[ ! -d "$MODEL_DIR" || ! -f "$INPUT_PARQUET" ]]; then
  echo "Verified bundle did not expose the expected model and replay paths." >&2
  exit 2
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    OLD_COMMAND="$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null || true)"
    if [[ "$OLD_COMMAND" == *"backend.serve"* && "$OLD_COMMAND" == *"$VENV_DIR"* ]]; then
      echo "Stopping previous demo process $OLD_PID..."
      kill "$OLD_PID"
      for _ in {1..20}; do
        kill -0 "$OLD_PID" 2>/dev/null || break
        sleep 0.25
      done
    fi
  fi
fi

if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "Port $PORT is already serving another process." >&2
  echo "Stop that process or run this launcher with a different PORT." >&2
  exit 2
fi

echo "Starting backend and same-origin frontend..."
(
  cd "$REPO_ROOT"
  nohup env \
    APP_ROOT="$REPO_ROOT" \
    MODEL_DIR="$MODEL_DIR" \
    INPUT_PARQUET="$INPUT_PARQUET" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    RECURSIVE_MODE=1 \
    FORECAST_HORIZONS=4 \
    DEFAULT_FORECAST_MODE="${DEFAULT_FORECAST_MODE:-lightning_only_today}" \
    LIGHTNING_HISTORY_PARQUET="${LIGHTNING_HISTORY_PARQUET:-}" \
    LIGHTNING_HISTORY_URI="${LIGHTNING_HISTORY_URI:-}" \
    AUTO_RUN_FORECAST=1 \
    HOST="$HOST" \
    PORT="$PORT" \
    "$VENV_DIR/bin/python" -m backend.serve \
    >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
)

SERVER_PID="$(cat "$PID_FILE")"
echo "Waiting for the service..."
sleep 1
for _ in {1..90}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server stopped before becoming healthy." >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >"$STATE_ROOT/health.json" \
    && grep -q '"status":"ready"' "$STATE_ROOT/health.json" \
    && grep -q '"model_loaded":true' "$STATE_ROOT/health.json" \
    && grep -q '"input_exists":true' "$STATE_ROOT/health.json"; then
    echo
    echo "INDIA LIGHTNING DEMO IS READY"
    echo "PID: $SERVER_PID"
    echo "Local URL: http://127.0.0.1:$PORT"
    echo "Log: $LOG_FILE"
    cat "$STATE_ROOT/health.json"
    echo
    echo "Cloud Shell menu: Web preview -> Preview on port $PORT"
    if command -v cloudshell >/dev/null 2>&1; then
      echo "Or run: cloudshell launch-web-preview --port $PORT"
    fi
    exit 0
  fi
  sleep 2
done

echo "Service did not become healthy within 180 seconds." >&2
tail -n 120 "$LOG_FILE" >&2 || true
exit 1
