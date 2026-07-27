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

if [[ ! "$PI_TARGET" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "PI_TARGET must be an explicit user@host SSH target." >&2
  exit 78
fi
if [[
  ! "$PI_DIR" =~ ^/[A-Za-z0-9._/-]+$
  || "$PI_DIR" == *"//"*
  || "$PI_DIR" == *"/../"*
  || "$PI_DIR" == */..
  || "$PI_DIR" == *"/./"*
  || "$PI_DIR" == */.
  || "$PI_DIR" != */etrade_python_client
]]; then
  echo "PI_DIR must be a normalized absolute path ending in /etrade_python_client." >&2
  exit 78
fi
if [[ "${SYNC_DELETE:-0}" != "0" && "${SYNC_DELETE:-0}" != "1" ]]; then
  echo "SYNC_DELETE must be exactly 0 or 1." >&2
  exit 78
fi

UNSAFE_GIT_ENVIRONMENT_VARIABLES=(
  GIT_ALTERNATE_OBJECT_DIRECTORIES
  GIT_COMMON_DIR
  GIT_DIR
  GIT_INDEX_FILE
  GIT_NAMESPACE
  GIT_OBJECT_DIRECTORY
  GIT_REPLACE_REF_BASE
  GIT_WORK_TREE
)
for git_environment_name in "${UNSAFE_GIT_ENVIRONMENT_VARIABLES[@]}"; do
  if printenv "$git_environment_name" >/dev/null 2>&1; then
    echo "Unsafe Git repository override is set: $git_environment_name" >&2
    exit 78
  fi
done
export GIT_NO_REPLACE_OBJECTS=1

APP_GIT_DIR="$(git -C "$APP_DIR" rev-parse --absolute-git-dir)"
GIT_ROOT="$(git -C "$APP_DIR" rev-parse --show-toplevel)"
ROOT_GIT_DIR="$(git -C "$GIT_ROOT" rev-parse --absolute-git-dir)"
if [[ "$ROOT_GIT_DIR" != "$APP_GIT_DIR" ]]; then
  echo "Git repository identity changed while resolving the application root." >&2
  exit 78
fi
APP_REPO_PATH="$(git -C "$APP_DIR" rev-parse --show-prefix)"
APP_REPO_PATH="${APP_REPO_PATH%/}"
if [[ -z "$APP_REPO_PATH" ]]; then
  echo "Application directory must be a tracked subdirectory of the repository." >&2
  exit 78
fi

DEPLOY_COMMIT="$(git -C "$GIT_ROOT" rev-parse --verify 'HEAD^{commit}')"
if ! git -C "$GIT_ROOT" diff --quiet "$DEPLOY_COMMIT" --; then
  echo "Tracked changes are present; commit them before syncing code." >&2
  exit 78
fi

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
STATE_SOURCE_PATHS=()
if [[ "$SYNC_STATE" == "1" ]]; then
  for state_file in "${STATE_FILES[@]}"; do
    state_path="$APP_DIR/$state_file"
    if [[ -L "$state_path" ]]; then
      echo "Refusing symbolic-link state file: $state_file" >&2
      exit 78
    fi
    if [[ -e "$state_path" && ! -f "$state_path" ]]; then
      echo "Refusing non-regular state path: $state_file" >&2
      exit 78
    fi
    if [[ -f "$state_path" ]]; then
      STATE_SOURCE_PATHS+=("$state_path")
    fi
  done
fi

SNAPSHOT_ROOT="$(
  mktemp -d "${TMPDIR:-/tmp}/etrade-code-snapshot.XXXXXX"
)"
cleanup_snapshot() {
  case "${SNAPSHOT_ROOT:-}" in
    */etrade-code-snapshot.*)
      rm -rf -- "$SNAPSHOT_ROOT"
      ;;
  esac
}
trap cleanup_snapshot EXIT
chmod 700 "$SNAPSHOT_ROOT"
SYNC_SOURCE="$SNAPSHOT_ROOT/code"
mkdir -m 700 "$SYNC_SOURCE"

git -C "$GIT_ROOT" archive --format=tar "$DEPLOY_COMMIT:$APP_REPO_PATH" \
  | tar -xf - -C "$SYNC_SOURCE"
if [[ ! -f "$SYNC_SOURCE/deploy/sync_to_pi.sh" ]]; then
  echo "Committed deployment snapshot is incomplete; no remote command ran." >&2
  exit 1
fi
snapshot_symlink="$(find "$SYNC_SOURCE" -type l -print -quit)"
if [[ -n "$snapshot_symlink" ]]; then
  echo "Committed deployment snapshot contains a symbolic link; no remote command ran." >&2
  exit 78
fi
TRACKED_MANIFEST="$SNAPSHOT_ROOT/tracked-paths"
git -C "$GIT_ROOT" ls-tree -r -z --name-only \
  "$DEPLOY_COMMIT:$APP_REPO_PATH" > "$TRACKED_MANIFEST"
while IFS= read -r -d '' tracked_path; do
  snapshot_path="$SYNC_SOURCE/$tracked_path"
  if [[ ! -f "$snapshot_path" ]]; then
    echo "Committed deployment snapshot omitted or changed a tracked path; no remote command ran." >&2
    exit 78
  fi
done < "$TRACKED_MANIFEST"

STATE_PATHS=()
if [[ "$SYNC_STATE" == "1" && "${#STATE_SOURCE_PATHS[@]}" -gt 0 ]]; then
  STATE_SNAPSHOT="$SNAPSHOT_ROOT/state"
  mkdir -m 700 "$STATE_SNAPSHOT"
  for state_source in "${STATE_SOURCE_PATHS[@]}"; do
    state_name="${state_source##*/}"
    cp -pP -- "$state_source" "$STATE_SNAPSHOT/$state_name"
    state_snapshot_path="$STATE_SNAPSHOT/$state_name"
    if [[ -L "$state_snapshot_path" || ! -f "$state_snapshot_path" ]]; then
      echo "Private state changed while its snapshot was created." >&2
      exit 78
    fi
    chmod 600 "$state_snapshot_path"
    STATE_PATHS+=("$state_snapshot_path")
  done
fi

DELETE_ARG=""
if [[ "${SYNC_DELETE:-0}" == "1" ]]; then
  DELETE_ARG="--delete"
fi

echo "Syncing immutable commit $DEPLOY_COMMIT"
ssh -- "$PI_TARGET" "mkdir -p '$PI_DIR'"

rsync -az ${DELETE_ARG:+"$DELETE_ARG"} \
  --include 'live_trading/dashboard_template.html' \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '.direnv/' \
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
  --exclude '.etrade_oauth*' \
  --exclude 'etrade_session.json*' \
  --exclude '.etrade_session.json*' \
  --exclude 'config.ini*' \
  --exclude 'credentials.json*' \
  --exclude '.netrc' \
  --exclude '.pypirc' \
  --exclude '.aws/' \
  --exclude '.ssh/' \
  --exclude '*.key*' \
  --exclude '*.pem*' \
  --exclude '*.p12*' \
  --exclude '*.pfx*' \
  --exclude '.env*' \
  --exclude '*.env*' \
  --exclude '*.db*' \
  --exclude '*.sqlite*' \
  --exclude '*.pkl*' \
  --exclude '*.pickle*' \
  --exclude '*.numbers*' \
  --exclude '*.parquet*' \
  --exclude '*.json*' \
  --exclude '*.csv*' \
  --exclude '*.png' \
  --exclude '*.html' \
  --exclude '*.log*' \
  --exclude '*.bak*' \
  --exclude '*.backup' \
  --exclude '*.old' \
  --exclude '*.tmp*' \
  --exclude '*.swp*' \
  --exclude '*.swo*' \
  --exclude '*.save*' \
  --exclude '*~' \
  --exclude 'backtesting/reports/' \
  --exclude 'backtesting/.engine_snapshots/' \
  -- \
  "$SYNC_SOURCE/" "$PI_TARGET:$PI_DIR/"

if [[ "$SYNC_STATE" == "1" ]]; then
  if [[ "${#STATE_PATHS[@]}" -gt 0 ]]; then
    rsync -az -- "${STATE_PATHS[@]}" "$PI_TARGET:$PI_DIR/"
  else
    echo "No private runtime state files found to sync."
  fi
fi
