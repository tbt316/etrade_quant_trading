#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./deploy/sync_to_pi.sh [--state]

Options:
  --state     Also sync private runtime state such as config.ini and OAuth/settings JSON files.
  --restart   Rejected while live deployment is suspended.

Environment:
  PI_TARGET       SSH target. Defaults to pi@raspberrypi.local.
  PI_DIR          Destination repo directory on the Pi. Defaults to /home/pi/etrade_python_client.
  SYNC_DELETE     Set to 1 to delete remote files that no longer exist locally.
USAGE
}

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_TARGET="${PI_TARGET:-pi@raspberrypi.local}"
PI_DIR="${PI_DIR:-/home/pi/etrade_python_client}"
SYNC_STATE=0
RESTART=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state)
      SYNC_STATE=1
      shift
      ;;
    --restart)
      RESTART=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$RESTART" == "1" ]]; then
  echo "Remote restart is suspended; no files were synced and no remote command ran." >&2
  exit 78
fi

DELETE_ARG=""
if [[ "${SYNC_DELETE:-0}" == "1" ]]; then
  DELETE_ARG="--delete"
fi

ssh "$PI_TARGET" "mkdir -p '$PI_DIR'"

rsync -az ${DELETE_ARG:+"$DELETE_ARG"} \
  --include 'live_trading/dashboard_template.html' \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'daily_log/' \
  --exclude 'executed_order_tracker/' \
  --exclude '_option_data_legacy_backup/' \
  --exclude 'backtest_cache/' \
  --exclude 'backtest_logs/' \
  --exclude 'audit_plots/' \
  --exclude 's_and_p_data/' \
  --exclude 'regime_review_2025/' \
  --exclude 'option_value_plot/' \
  --exclude 'python_client.log*' \
  --exclude 'server_log.txt' \
  --exclude 'ngrok.log' \
  --exclude 'dashboard_requests.log' \
  --exclude '.etrade_oauth' \
  --exclude 'etrade_session.json' \
  --exclude 'config.ini' \
  --exclude '.env' \
  --exclude '*.env' \
  --exclude '*.db' \
  --exclude '*.sqlite' \
  --exclude '*.json' \
  --exclude '*.csv' \
  --exclude '*.png' \
  --exclude '*.html' \
  --exclude '*.log' \
  --exclude 'backtesting/reports/' \
  --exclude 'backtesting/.engine_snapshots/' \
  "$APP_DIR/" "$PI_TARGET:$PI_DIR/"

if [[ "$SYNC_STATE" == "1" ]]; then
  STATE_FILES=(
    config.ini
    .env
    .etrade_oauth
    live_trading_settings.json
    trade_status.json
    manual_order_status.json
    spy_tracking_data.json
    spy_vix_price_cache.json
    nudge_history.json
    nudge_state.json
  )
  STATE_PATHS=()
  for state_file in "${STATE_FILES[@]}"; do
    if [[ -e "$APP_DIR/$state_file" ]]; then
      STATE_PATHS+=("$APP_DIR/$state_file")
    fi
  done

  if [[ "${#STATE_PATHS[@]}" -gt 0 ]]; then
    rsync -az "${STATE_PATHS[@]}" "$PI_TARGET:$PI_DIR/"
  else
    echo "No private runtime state files found to sync."
  fi
fi
