#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-etrade-live}"
APP_USER="${APP_USER:-$(id -un)}"
PYTHON_BIN="${PYTHON_BIN:-${APP_DIR}/.venv/bin/python}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended to run on the Raspberry Pi/Linux host."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found at $PYTHON_BIN"
  echo "Run deploy/pi_bootstrap.sh first, or set PYTHON_BIN=/path/to/python."
  exit 1
fi

if [[ ! -f "${APP_DIR}/config.ini" ]]; then
  echo "Missing ${APP_DIR}/config.ini"
  echo "Copy private runtime state before starting the service."
fi

sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=E*TRADE live trading dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} -m live_trading.etrade_cover_call_new --no-sandbox --trade
Restart=on-failure
RestartSec=30
KillSignal=SIGINT
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "Installed and enabled ${SERVICE_NAME}.service"
echo "Start it with: sudo systemctl start ${SERVICE_NAME}"
echo "View logs with: journalctl -u ${SERVICE_NAME} -f"

if [[ "${START_NOW:-0}" == "1" ]]; then
  sudo systemctl start "$SERVICE_NAME"
fi
