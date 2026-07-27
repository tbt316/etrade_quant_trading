#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./deploy/sync_to_pi.sh [--state]
  ./deploy/sync_to_pi.sh --rollback RELEASE_ID

Options:
  --state                Rejected until atomic state-generation migration exists.
  --rollback RELEASE_ID  Atomically select an already verified release.
  --restart              Rejected while live deployment is suspended.

Environment:
  PI_TARGET    SSH target. Defaults to pi@raspberrypi.local.
  PI_DIR       Deployment root on the Pi. Defaults to /home/pi/etrade_python_client.
  SYNC_DELETE  Deprecated compatibility setting; only exact 0 or 1 is accepted.
USAGE
}

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_TARGET="${PI_TARGET:-pi@raspberrypi.local}"
PI_DIR="${PI_DIR:-/home/pi/etrade_python_client}"
SYNC_STATE=0
RESTART=0
ROLLBACK_RELEASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state)
      SYNC_STATE=1
      shift
      ;;
    --rollback)
      if [[ $# -lt 2 ]]; then
        echo "--rollback requires an exact release ID." >&2
        exit 78
      fi
      ROLLBACK_RELEASE="$2"
      shift 2
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
if [[ "$SYNC_STATE" == "1" && -n "$ROLLBACK_RELEASE" ]]; then
  echo "--state and --rollback cannot be combined." >&2
  exit 78
fi
if [[ "$SYNC_STATE" == "1" ]]; then
  echo "Private state sync is suspended until an atomic state-generation consumer exists; no remote command ran." >&2
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
if [[
  -n "$ROLLBACK_RELEASE"
  && ! "$ROLLBACK_RELEASE" =~ ^[0-9a-f]{40}-[0-9a-f]{64}$
  && ! "$ROLLBACK_RELEASE" =~ ^[0-9a-f]{64}-[0-9a-f]{64}$
]]; then
  echo "--rollback requires a canonical commit-and-archive release ID." >&2
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
if [[
  ! "$DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$
  && ! "$DEPLOY_COMMIT" =~ ^[0-9a-f]{64}$
]]; then
  echo "Resolved HEAD is not a canonical Git commit identity." >&2
  exit 78
fi
if ! git -C "$GIT_ROOT" diff --quiet "$DEPLOY_COMMIT" --; then
  echo "Tracked changes are present; commit them before syncing code." >&2
  exit 78
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
CODE_ARCHIVE="$SNAPSHOT_ROOT/release.tar"

git -C "$GIT_ROOT" archive --format=tar \
  --output="$CODE_ARCHIVE" \
  "$DEPLOY_COMMIT:$APP_REPO_PATH"
chmod 600 "$CODE_ARCHIVE"
tar -xf "$CODE_ARCHIVE" -C "$SYNC_SOURCE"
if [[
  ! -f "$SYNC_SOURCE/deploy/sync_to_pi.sh"
  || ! -f "$SYNC_SOURCE/deploy/pi_release.sh"
  || ! -f "$SYNC_SOURCE/scripts/check_repo_hygiene.py"
]]; then
  echo "Committed deployment snapshot is incomplete; no remote command ran." >&2
  exit 1
fi
snapshot_unsafe_path="$(
  find "$SYNC_SOURCE" -mindepth 1 \
    ! -type d ! -type f \
    -print -quit
)"
if [[ -n "$snapshot_unsafe_path" ]]; then
  echo "Committed deployment snapshot contains an unsafe file type; no remote command ran." >&2
  exit 78
fi
TRACKED_MANIFEST="$SNAPSHOT_ROOT/tracked-paths"
git -C "$GIT_ROOT" ls-tree -r -z \
  "$DEPLOY_COMMIT:$APP_REPO_PATH" > "$TRACKED_MANIFEST"
TRACKED_BLOB="$SNAPSHOT_ROOT/tracked-blob"
TRACKED_ATTRIBUTE="$SNAPSHOT_ROOT/tracked-attribute"
TRACKED_FILE_COUNT=0
while IFS= read -r -d '' tracked_entry; do
  ((TRACKED_FILE_COUNT += 1))
  if [[ "$tracked_entry" != *$'\t'* ]]; then
    echo "Committed tree manifest is malformed; no remote command ran." >&2
    exit 78
  fi
  tracked_identity="${tracked_entry%%$'\t'*}"
  tracked_path="${tracked_entry#*$'\t'}"
  read -r tracked_mode tracked_type tracked_oid <<< "$tracked_identity"
  if [[
    "$tracked_type" != "blob"
    || ( "$tracked_mode" != "100644" && "$tracked_mode" != "100755" )
    || ( ! "$tracked_oid" =~ ^[0-9a-f]{40}$ && ! "$tracked_oid" =~ ^[0-9a-f]{64}$ )
  ]]; then
    echo "Committed deployment tree contains an unsupported entry; no remote command ran." >&2
    exit 78
  fi
  if [[
    "$tracked_path" == /* || "$tracked_path" == ../*
    || "$tracked_path" == */../* || "$tracked_path" == */..
    || "$tracked_path" == *$'\n'* || "$tracked_path" == *$'\r'*
  ]]; then
    echo "Committed deployment snapshot contains an unsafe tracked path." >&2
    exit 78
  fi
  repository_tracked_path="$APP_REPO_PATH/$tracked_path"
  git -C "$GIT_ROOT" check-attr -z export-subst -- \
    "$repository_tracked_path" > "$TRACKED_ATTRIBUTE"
  {
    IFS= read -r -d '' attribute_path
    IFS= read -r -d '' attribute_name
    IFS= read -r -d '' attribute_value
  } < "$TRACKED_ATTRIBUTE"
  if [[
    "$attribute_path" != "$repository_tracked_path"
    || "$attribute_name" != "export-subst"
  ]]; then
    echo "Git attribute proof was malformed; no remote command ran." >&2
    exit 78
  fi
  if [[ "$attribute_value" != "unspecified" && "$attribute_value" != "unset" ]]; then
    echo "Refusing an export-subst deployment path; no remote command ran." >&2
    exit 78
  fi
  snapshot_path="$SYNC_SOURCE/$tracked_path"
  if [[ ! -f "$snapshot_path" ]]; then
    echo "Committed deployment snapshot omitted or changed a tracked path; no remote command ran." >&2
    exit 78
  fi
  if [[
    ( "$tracked_mode" == "100755" && ! -x "$snapshot_path" )
    || ( "$tracked_mode" == "100644" && -x "$snapshot_path" )
  ]]; then
    echo "Committed deployment snapshot changed a tracked executable mode; no remote command ran." >&2
    exit 78
  fi
  git -C "$GIT_ROOT" cat-file blob "$tracked_oid" > "$TRACKED_BLOB"
  if ! cmp -s "$TRACKED_BLOB" "$snapshot_path"; then
    echo "Committed deployment snapshot differs from an exact Git blob; no remote command ran." >&2
    exit 78
  fi
done < "$TRACKED_MANIFEST"
SNAPSHOT_FILE_COUNT=0
while IFS= read -r -d '' snapshot_file; do
  ((SNAPSHOT_FILE_COUNT += 1))
done < <(find "$SYNC_SOURCE" -type f -print0)
if [[ "$SNAPSHOT_FILE_COUNT" -ne "$TRACKED_FILE_COUNT" ]]; then
  echo "Committed deployment snapshot contains an untracked file; no remote command ran." >&2
  exit 78
fi
if ! python3 -I "$SYNC_SOURCE/scripts/check_repo_hygiene.py" \
  --start "$GIT_ROOT" \
  --tree "$DEPLOY_COMMIT" \
  --tree-prefix "$APP_REPO_PATH" \
  --redact-paths; then
  echo "Resolved release tree failed deployment hygiene; no remote command ran." >&2
  exit 78
fi
if ! bash -n "$SYNC_SOURCE/deploy/sync_to_pi.sh"; then
  echo "Committed sync script failed its syntax precheck; no remote command ran." >&2
  exit 78
fi
if ! bash -n "$SYNC_SOURCE/deploy/pi_release.sh"; then
  echo "Committed release helper failed its syntax precheck; no remote command ran." >&2
  exit 78
fi

ARCHIVE_SHA256="$(shasum -a 256 "$CODE_ARCHIVE" | awk '{print $1}')"
if [[ ! "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Could not resolve a canonical SHA-256 release archive identity." >&2
  exit 78
fi
RELEASE_ID="$DEPLOY_COMMIT-$ARCHIVE_SHA256"
RELEASE_HELPER="$SYNC_SOURCE/deploy/pi_release.sh"
SNAPSHOT_TOKEN="${SNAPSHOT_ROOT##*.}"
UPLOAD_TOKEN="$DEPLOY_COMMIT.$$.$SNAPSHOT_TOKEN"
if [[
  ! "$UPLOAD_TOKEN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$
  || "$UPLOAD_TOKEN" == *".."*
]]; then
  echo "Could not create a safe release operation token." >&2
  exit 78
fi

if [[ "${SYNC_DELETE:-0}" == "1" ]]; then
  echo "SYNC_DELETE=1 is obsolete for clean versioned releases and has no effect." >&2
fi

run_remote_release_action() {
  local remote_command="bash -s --"
  local remote_argument
  local quoted_argument
  for remote_argument in "$@"; do
    if [[ ! "$remote_argument" =~ ^[A-Za-z0-9._/-]+$ ]]; then
      echo "Unsafe remote release argument was rejected." >&2
      exit 78
    fi
    printf -v quoted_argument '%q' "$remote_argument"
    remote_command+=" $quoted_argument"
  done
  ssh -- "$PI_TARGET" "$remote_command" < "$RELEASE_HELPER"
}

if [[ -n "$ROLLBACK_RELEASE" ]]; then
  echo "Requesting verified rollback to $ROLLBACK_RELEASE"
  run_remote_release_action \
    rollback "$PI_DIR" "$ROLLBACK_RELEASE" "$UPLOAD_TOKEN"
  echo "Rollback selected release $ROLLBACK_RELEASE; no service was restarted."
  exit 0
fi

echo "Preparing owner-read-only, reverified release $RELEASE_ID"
EXPECTED_GENERATION="$(
  run_remote_release_action prepare "$PI_DIR" "$UPLOAD_TOKEN"
)"
if [[
  "$EXPECTED_GENERATION" != "missing"
  && ! "$EXPECTED_GENERATION" =~ ^[0-9a-f]{64}$
]]; then
  echo "Remote prepare did not return a canonical current generation." >&2
  exit 78
fi
rsync -az -- \
  "$CODE_ARCHIVE" \
  "$PI_TARGET:$PI_DIR/incoming/$UPLOAD_TOKEN.tar"
run_remote_release_action \
  install "$PI_DIR" "$RELEASE_ID" "$DEPLOY_COMMIT" \
  "$ARCHIVE_SHA256" "$UPLOAD_TOKEN"

run_remote_release_action \
  activate "$PI_DIR" "$RELEASE_ID" "$EXPECTED_GENERATION" "$UPLOAD_TOKEN"
echo "Activated release $RELEASE_ID; no service was restarted."
