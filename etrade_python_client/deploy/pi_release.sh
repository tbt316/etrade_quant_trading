#!/usr/bin/env bash
set -euo pipefail

# This helper is streamed from an exact committed snapshot by sync_to_pi.sh.
# It never installs dependencies, starts a service, or contacts E*TRADE.

fail() {
  echo "Release operation rejected: $*" >&2
  exit 78
}

BASE=""
RELEASES=""
ARTIFACTS=""
VERIFIED=""
INCOMING=""
SELECTIONS=""
CLEANUP_PATHS=()
LOCK_PATH=""
LOCK_OWNER_FILE=""
LOCK_OWNER_ID=""
LOCK_HELD=0

cleanup() {
  local cleanup_path
  if [[ "${#CLEANUP_PATHS[@]}" -gt 0 ]]; then
    for cleanup_path in "${CLEANUP_PATHS[@]}"; do
      case "$cleanup_path" in
        "$INCOMING"/*.tar|"$RELEASES"/.staging-*|"$RELEASES"/.verify-*|"$VERIFIED"/.metadata-*|"$BASE"/.current-*)
          if [[ -d "$cleanup_path" && ! -L "$cleanup_path" ]]; then
            chmod -R u+w "$cleanup_path" 2>/dev/null || true
            rm -rf -- "$cleanup_path"
          else
            rm -f -- "$cleanup_path"
          fi
          ;;
      esac
    done
  fi
  if [[ "$LOCK_HELD" == "1" ]]; then
    if [[
      -d "$LOCK_PATH" && ! -L "$LOCK_PATH"
      && -f "$LOCK_OWNER_FILE" && ! -L "$LOCK_OWNER_FILE"
      && "$(sed -n '1p' "$LOCK_OWNER_FILE")" == "$LOCK_OWNER_ID"
    ]]; then
      rm -f -- "$LOCK_OWNER_FILE"
      rmdir -- "$LOCK_PATH" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

validate_base() {
  local candidate="$1"
  if [[
    ! "$candidate" =~ ^/[A-Za-z0-9._/-]+$
    || "$candidate" == *"//"*
    || "$candidate" == *"/../"*
    || "$candidate" == */..
    || "$candidate" == *"/./"*
    || "$candidate" == */.
    || "$candidate" != */etrade_python_client
  ]]; then
    fail "base path must be normalized, absolute, and end in /etrade_python_client"
  fi
}

validate_commit() {
  local candidate="$1"
  if [[
    ! "$candidate" =~ ^[0-9a-f]{40}$
    && ! "$candidate" =~ ^[0-9a-f]{64}$
  ]]; then
    fail "invalid commit identity"
  fi
}

validate_digest() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256 digest"
}

validate_release_id() {
  local candidate="$1"
  if [[
    ! "$candidate" =~ ^[0-9a-f]{40}-[0-9a-f]{64}$
    && ! "$candidate" =~ ^[0-9a-f]{64}-[0-9a-f]{64}$
  ]]; then
    fail "invalid release identity"
  fi
}

validate_current_generation() {
  local candidate="$1"
  [[ "$candidate" == "missing" || "$candidate" =~ ^[0-9a-f]{64}$ ]] \
    || fail "invalid current generation"
}

validate_token() {
  local candidate="$1"
  if [[
    ! "$candidate" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$
    || "$candidate" == *".."*
  ]]; then
    fail "invalid operation token"
  fi
}

reject_symlink_components() {
  local candidate="$1"
  local component
  local current=""
  local remainder="${candidate#/}"
  local components=()
  IFS='/' read -r -a components <<< "$remainder"
  for component in "${components[@]}"; do
    current="$current/$component"
    [[ ! -L "$current" ]] || fail "symbolic-link path component: $current"
  done
}

path_mode() {
  local candidate="$1"
  local mode
  if mode="$(stat -c '%a' "$candidate" 2>/dev/null)"; then
    :
  else
    mode="$(stat -f '%Lp' "$candidate")"
  fi
  [[ "$mode" =~ ^[0-7]+$ ]] || fail "could not verify deployment permissions"
  PATH_MODE_VALUE="$((8#$mode))"
}

path_link_count() {
  local candidate="$1"
  local link_count
  if link_count="$(stat -c '%h' "$candidate" 2>/dev/null)"; then
    :
  else
    link_count="$(stat -f '%l' "$candidate")"
  fi
  [[ "$link_count" =~ ^[1-9][0-9]*$ ]] \
    || fail "could not verify deployment file link count"
  PATH_LINK_COUNT_VALUE="$link_count"
}

require_single_link_regular_file() {
  local candidate="$1"
  [[ -f "$candidate" && ! -L "$candidate" ]] \
    || fail "deployment file is missing or not regular"
  path_link_count "$candidate"
  [[ "$PATH_LINK_COUNT_VALUE" -eq 1 ]] \
    || fail "multi-link deployment file is not allowed"
}

require_owner_only() {
  local candidate="$1"
  [[ -O "$candidate" ]] || fail "deployment path is not owned by the release user"
  path_mode "$candidate"
  (( (PATH_MODE_VALUE & 077) == 0 )) \
    || fail "deployment path grants group or other access"
}

ensure_directory() {
  local directory="$1"
  reject_symlink_components "$directory"
  if [[ -e "$directory" && ! -d "$directory" ]]; then
    fail "non-directory deployment path: $directory"
  fi
  mkdir -p -- "$directory"
  reject_symlink_components "$directory"
  [[ -d "$directory" && ! -L "$directory" ]] \
    || fail "unsafe deployment directory: $directory"
  chmod 700 "$directory"
  require_owner_only "$directory"
}

initialize_base() {
  BASE="$1"
  validate_base "$BASE"
  RELEASES="$BASE/releases"
  ARTIFACTS="$BASE/artifacts"
  VERIFIED="$BASE/verified"
  INCOMING="$BASE/incoming"
  SELECTIONS="$BASE/selections"

  ensure_directory "$BASE"
}

initialize_layout() {
  ensure_directory "$RELEASES"
  ensure_directory "$ARTIFACTS"
  ensure_directory "$VERIFIED"
  ensure_directory "$INCOMING"
  ensure_directory "$SELECTIONS"
}

acquire_operation_lock() {
  LOCK_PATH="$BASE/.deploy-lock"
  LOCK_OWNER_FILE="$LOCK_PATH/owner"
  LOCK_OWNER_ID="$$.$RANDOM.$RANDOM"
  if [[ -L "$LOCK_PATH" || ( -e "$LOCK_PATH" && ! -d "$LOCK_PATH" ) ]]; then
    fail "deployment lock path is unsafe"
  fi
  if ! mkdir -m 700 -- "$LOCK_PATH" 2>/dev/null; then
    fail "another deployment operation holds the remote lock"
  fi
  LOCK_HELD=1
  [[ -d "$LOCK_PATH" && ! -L "$LOCK_PATH" ]] \
    || fail "deployment lock identity changed"
  printf '%s\n' "$LOCK_OWNER_ID" > "$LOCK_OWNER_FILE"
  chmod 600 "$LOCK_OWNER_FILE"
  [[ -f "$LOCK_OWNER_FILE" && ! -L "$LOCK_OWNER_FILE" ]] \
    || fail "deployment lock owner marker is unsafe"
  require_owner_only "$LOCK_PATH"
  require_owner_only "$LOCK_OWNER_FILE"
}

release_parts() {
  local release_id="$1"
  RELEASE_COMMIT="${release_id%%-*}"
  RELEASE_DIGEST="${release_id#*-}"
  validate_commit "$RELEASE_COMMIT"
  validate_digest "$RELEASE_DIGEST"
  [[ "$release_id" == "$RELEASE_COMMIT-$RELEASE_DIGEST" ]] \
    || fail "release identity is not canonical"
}

reject_unsafe_tree() {
  local tree="$1"
  local unsafe_path
  local regular_file
  unsafe_path="$(
    find "$tree" -mindepth 1 \
      ! -type d ! -type f \
      -print -quit
  )"
  [[ -z "$unsafe_path" ]] || fail "release tree contains an unsafe file type"
  while IFS= read -r -d '' regular_file; do
    require_single_link_regular_file "$regular_file"
  done < <(find "$tree" -type f -print0)
}

health_precheck() {
  local tree="$1"
  reject_unsafe_tree "$tree"
  [[ -f "$tree/deploy/sync_to_pi.sh" && ! -L "$tree/deploy/sync_to_pi.sh" ]] \
    || fail "release is missing deploy/sync_to_pi.sh"
  [[ -f "$tree/deploy/pi_release.sh" && ! -L "$tree/deploy/pi_release.sh" ]] \
    || fail "release is missing deploy/pi_release.sh"
  bash -n "$tree/deploy/sync_to_pi.sh" \
    || fail "release sync script failed its syntax precheck"
  bash -n "$tree/deploy/pi_release.sh" \
    || fail "release helper failed its syntax precheck"
}

archive_digest() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "no SHA-256 utility is available"
  fi
}

verify_archive() {
  local archive="$1"
  local expected_digest="$2"
  require_single_link_regular_file "$archive"
  [[ "$(archive_digest "$archive")" == "$expected_digest" ]] \
    || fail "canonical release archive digest mismatch"
}

verify_read_only_release_permissions() {
  local tree="$1"
  local candidate
  while IFS= read -r -d '' candidate; do
    if [[ -f "$candidate" ]]; then
      require_single_link_regular_file "$candidate"
    fi
    require_owner_only "$candidate"
    (( (PATH_MODE_VALUE & 0200) == 0 )) \
      || fail "verified release is owner-writable"
  done < <(find "$tree" -print0)
}

verify_release_modes() {
  local canonical_tree="$1"
  local release_tree="$2"
  local canonical_path
  local release_path
  local relative_path
  local expected_mode
  local release_mode
  while IFS= read -r -d '' canonical_path; do
    relative_path="${canonical_path#"$canonical_tree"}"
    release_path="$release_tree$relative_path"
    [[
      ( -d "$canonical_path" && -d "$release_path" )
      || ( -f "$canonical_path" && -f "$release_path" )
    ]] || fail "release tree changed a canonical file type"
    path_mode "$canonical_path"
    expected_mode="$((PATH_MODE_VALUE & 0500))"
    path_mode "$release_path"
    release_mode="$PATH_MODE_VALUE"
    [[ "$release_mode" -eq "$expected_mode" ]] \
      || fail "release tree mode differs from its canonical archive"
  done < <(find "$canonical_tree" -print0)
}

metadata_text() {
  local release_id="$1"
  local commit="$2"
  local digest="$3"
  printf 'version=1\nrelease_id=%s\ncommit=%s\narchive_sha256=%s\n' \
    "$release_id" "$commit" "$digest"
}

verify_metadata() {
  local metadata="$1"
  local release_id="$2"
  local commit="$3"
  local digest="$4"
  local actual_metadata
  local expected_metadata

  [[ -f "$metadata" && ! -L "$metadata" ]] \
    || fail "verified-release marker is missing or unsafe"
  require_single_link_regular_file "$metadata"
  require_owner_only "$metadata"
  [[ "$PATH_MODE_VALUE" -eq 0400 ]] \
    || fail "verified-release marker mode is not owner-read-only"
  actual_metadata="$(<"$metadata")"
  expected_metadata="$(metadata_text "$release_id" "$commit" "$digest")"
  [[ "$actual_metadata" == "$expected_metadata" ]] \
    || fail "verified-release marker does not match its release identity"
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  mkdir -m 700 -- "$destination"
  CLEANUP_PATHS+=("$destination")
  tar -xf "$archive" -C "$destination"
  health_precheck "$destination"
}

verify_release() {
  local release_id="$1"
  local token="$2"
  local release="$RELEASES/$release_id"
  local archive="$ARTIFACTS/$release_id.tar"
  local metadata="$VERIFIED/$release_id"
  local verification_tree="$RELEASES/.verify-$release_id-$token"

  release_parts "$release_id"
  [[ -d "$release" && ! -L "$release" ]] \
    || fail "verified release directory is missing or unsafe"
  health_precheck "$release"
  verify_metadata \
    "$metadata" "$release_id" "$RELEASE_COMMIT" "$RELEASE_DIGEST"
  require_owner_only "$archive"
  [[ "$PATH_MODE_VALUE" -eq 0400 ]] \
    || fail "canonical release archive mode is not owner-read-only"
  verify_archive "$archive" "$RELEASE_DIGEST"
  verify_read_only_release_permissions "$release"
  [[ ! -e "$verification_tree" && ! -L "$verification_tree" ]] \
    || fail "verification workspace already exists"
  extract_archive "$archive" "$verification_tree"
  if ! diff -qr "$verification_tree" "$release" >/dev/null; then
    fail "release tree differs from its canonical archive"
  fi
  verify_release_modes "$verification_tree" "$release"
  chmod -R u+w "$verification_tree"
  rm -rf -- "$verification_tree"
}

current_release_id() {
  local current="$BASE/current"
  local target
  local selection
  local selection_target
  if [[ ! -e "$current" && ! -L "$current" ]]; then
    return 1
  fi
  [[ -L "$current" ]] || fail "current must be a managed symbolic link"
  target="$(readlink "$current")"
  case "$target" in
    selections/*)
      CURRENT_GENERATION="${target#selections/}"
      validate_current_generation "$CURRENT_GENERATION"
      [[ "$target" == "selections/$CURRENT_GENERATION" ]] \
        || fail "current generation target is not canonical"
      selection="$SELECTIONS/$CURRENT_GENERATION"
      [[ -L "$selection" ]] \
        || fail "current selection is missing or unsafe"
      selection_target="$(readlink "$selection")"
      case "$selection_target" in
        ../releases/*)
          CURRENT_RELEASE_ID="${selection_target#../releases/}"
          ;;
        *)
          fail "current selection points outside the managed release directory"
          ;;
      esac
      validate_release_id "$CURRENT_RELEASE_ID"
      [[ "$selection_target" == "../releases/$CURRENT_RELEASE_ID" ]] \
        || fail "current release selection is not canonical"
      ;;
    *)
      fail "current points outside the managed selection directory"
      ;;
  esac
}

new_selection_generation() {
  local release_id="$1"
  local token="$2"
  local seed="$release_id:$token:$LOCK_OWNER_ID"
  local generation
  if command -v sha256sum >/dev/null 2>&1; then
    generation="$(printf '%s' "$seed" | sha256sum | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    generation="$(printf '%s' "$seed" | shasum -a 256 | awk '{print $1}')"
  else
    fail "no SHA-256 utility is available"
  fi
  validate_current_generation "$generation"
  NEW_SELECTION_GENERATION="$generation"
}

switch_current() {
  local release_id="$1"
  local token="$2"
  local expected_generation="$3"
  local temporary_link="$BASE/.current-$token"
  local selection
  [[ ! -e "$temporary_link" && ! -L "$temporary_link" ]] \
    || fail "temporary current pointer already exists"
  if [[ "$expected_generation" == "missing" ]]; then
    if current_release_id; then
      fail "current changed before activation"
    fi
  else
    current_release_id || fail "current disappeared before activation"
    [[ "$CURRENT_GENERATION" == "$expected_generation" ]] \
      || fail "current changed before activation"
  fi
  new_selection_generation "$release_id" "$token"
  selection="$SELECTIONS/$NEW_SELECTION_GENERATION"
  [[ ! -e "$selection" && ! -L "$selection" ]] \
    || fail "new selection generation already exists"
  ln -s -- "../releases/$release_id" "$selection"
  [[ -L "$selection" ]] || fail "new selection generation is unsafe"
  ln -s -- "selections/$NEW_SELECTION_GENERATION" "$temporary_link"
  CLEANUP_PATHS+=("$temporary_link")
  if [[ "$expected_generation" == "missing" ]]; then
    if current_release_id; then
      fail "current changed before activation"
    fi
  else
    current_release_id || fail "current disappeared before activation"
    [[ "$CURRENT_GENERATION" == "$expected_generation" ]] \
      || fail "current changed before activation"
  fi
  python3 -I - "$temporary_link" "$BASE/current" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
  current_release_id || fail "current disappeared after activation"
  [[
    "$CURRENT_RELEASE_ID" == "$release_id"
    && "$CURRENT_GENERATION" == "$NEW_SELECTION_GENERATION"
  ]] \
    || fail "current does not identify the activated release"
}

prepare_upload() {
  local token="$1"
  local upload="$INCOMING/$token.tar"
  local expected_generation="missing"
  validate_token "$token"
  [[ ! -e "$upload" && ! -L "$upload" ]] \
    || fail "upload path already exists"
  if current_release_id; then
    expected_generation="$CURRENT_GENERATION"
  fi
  printf '%s\n' "$expected_generation"
}

install_release() {
  local release_id="$1"
  local commit="$2"
  local digest="$3"
  local token="$4"
  local upload="$INCOMING/$token.tar"
  local artifact="$ARTIFACTS/$release_id.tar"
  local release="$RELEASES/$release_id"
  local metadata="$VERIFIED/$release_id"
  local staging="$RELEASES/.staging-$release_id-$token"
  local metadata_temp="$VERIFIED/.metadata-$token"

  validate_release_id "$release_id"
  validate_commit "$commit"
  validate_digest "$digest"
  validate_token "$token"
  [[ "$release_id" == "$commit-$digest" ]] \
    || fail "release identity does not bind the requested commit and archive"
  verify_archive "$upload" "$digest"
  CLEANUP_PATHS+=("$upload")

  if [[ -e "$metadata" || -L "$metadata" ]]; then
    verify_release "$release_id" "$token"
    rm -f -- "$upload"
    printf 'Release %s was already verified.\n' "$release_id"
    return
  fi

  [[ ! -e "$staging" && ! -L "$staging" ]] \
    || fail "release staging path already exists"
  extract_archive "$upload" "$staging"

  if [[ -e "$artifact" || -L "$artifact" ]]; then
    verify_archive "$artifact" "$digest"
  fi

  if [[ -e "$release" || -L "$release" ]]; then
    [[ -d "$release" && ! -L "$release" ]] \
      || fail "release destination is not a safe directory"
    health_precheck "$release"
    if ! diff -qr "$staging" "$release" >/dev/null; then
      fail "unverified release destination differs from the canonical archive"
    fi
  fi

  if [[ ! -e "$artifact" && ! -L "$artifact" ]]; then
    mv -- "$upload" "$artifact"
  fi
  verify_archive "$artifact" "$digest"
  chmod 400 "$artifact"
  require_owner_only "$artifact"

  if [[ -e "$release" || -L "$release" ]]; then
    chmod -R u+w "$staging"
    rm -rf -- "$staging"
  else
    mv -- "$staging" "$release"
  fi

  health_precheck "$release"
  verify_archive "$artifact" "$digest"
  chmod -R go-rwx "$release"
  chmod -R u-w "$release"

  [[ ! -e "$metadata_temp" && ! -L "$metadata_temp" ]] \
    || fail "temporary verified marker already exists"
  CLEANUP_PATHS+=("$metadata_temp")
  metadata_text "$release_id" "$commit" "$digest" > "$metadata_temp"
  require_single_link_regular_file "$metadata_temp"
  chmod 400 "$metadata_temp"
  mv -- "$metadata_temp" "$metadata"
  verify_release "$release_id" "$token"
  rm -f -- "$upload"
  printf 'Verified release %s.\n' "$release_id"
}

activate_release() {
  local release_id="$1"
  local expected_generation="$2"
  local token="$3"
  validate_release_id "$release_id"
  validate_current_generation "$expected_generation"
  validate_token "$token"
  verify_release "$release_id" "$token"

  if current_release_id; then
    [[ "$CURRENT_GENERATION" == "$expected_generation" ]] \
      || fail "current generation changed after prepare"
    if [[ "$CURRENT_RELEASE_ID" == "$release_id" ]]; then
      printf 'Release %s is already current.\n' "$release_id"
      return
    fi
  else
    [[ "$expected_generation" == "missing" ]] \
      || fail "current generation disappeared after prepare"
  fi

  switch_current "$release_id" "$token" "$expected_generation"
  printf 'Activated release %s.\n' "$release_id"
}

rollback_release() {
  local release_id="$1"
  local token="$2"
  validate_release_id "$release_id"
  validate_token "$token"
  current_release_id \
    || fail "rollback requires an existing managed current release"
  local expected_generation="$CURRENT_GENERATION"
  [[ "$CURRENT_RELEASE_ID" != "$release_id" ]] \
    || fail "rollback target is already current"
  verify_release "$release_id" "$token-rollback"
  switch_current "$release_id" "$token" "$expected_generation"
  printf 'Rolled back to verified release %s.\n' "$release_id"
}

if [[ $# -lt 2 ]]; then
  fail "expected an action and deployment base"
fi

ACTION="$1"
initialize_base "$2"
acquire_operation_lock
initialize_layout
shift 2

case "$ACTION" in
  prepare)
    [[ $# -eq 1 ]] || fail "prepare expects one token"
    prepare_upload "$1"
    ;;
  install)
    [[ $# -eq 4 ]] || fail "install expects release, commit, digest, and token"
    install_release "$1" "$2" "$3" "$4"
    ;;
  activate)
    [[ $# -eq 3 ]] \
      || fail "activate expects release, expected generation, and token"
    activate_release "$1" "$2" "$3"
    ;;
  rollback)
    [[ $# -eq 2 ]] || fail "rollback expects release and token"
    rollback_release "$1" "$2"
    ;;
  *)
    fail "unknown release action"
    ;;
esac
