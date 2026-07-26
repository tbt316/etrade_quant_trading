#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${APP_DIR}/.venv}"

cd "$APP_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This bootstrap is intended to run on the Raspberry Pi/Linux host."
  exit 1
fi

echo "Bootstrapping live E*TRADE runtime in: $APP_DIR"

PYTHON_VERSION_OK="$("$PYTHON_BIN" - <<'PY'
import sys
print("1" if sys.version_info >= (3, 10) else "0")
PY
)"

if [[ "$PYTHON_VERSION_OK" != "1" ]]; then
  echo "Python 3.10+ is required for the pinned live runtime dependencies."
  echo "Found: $("$PYTHON_BIN" --version 2>&1)"
  echo "For Raspberry Pi 4, use Raspberry Pi OS 64-bit Bookworm, or install a newer Python and set PYTHON_BIN."
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    pkg-config

  if [[ "${INSTALL_BROWSER_DEPS:-1}" == "1" ]]; then
    sudo apt-get install -y chromium chromium-driver || \
      sudo apt-get install -y chromium-browser chromium-chromedriver || true
  fi
else
  echo "apt-get not found; install Python 3, venv, build tools, Chromium, and chromedriver manually."
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "${APP_DIR}/deploy/requirements_live.txt"

mkdir -p "$APP_DIR/logs"

echo "Bootstrap complete."
echo "Next: copy private state, then install the systemd service."
