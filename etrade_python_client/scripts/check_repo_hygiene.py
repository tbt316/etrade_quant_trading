#!/usr/bin/env python3
"""Fail when generated, runtime, or local-secret state is tracked by Git.

The checker examines only the repository index. It deliberately does not walk
the working tree or inspect file contents, so ignored local credentials and
runtime artifacts neither leak into diagnostics nor make the result depend on
workstation state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


FORBIDDEN_ENVIRONMENT_DIRECTORIES = frozenset(
    {"venv", ".venv", ".direnv"}
)
FORBIDDEN_GENERATED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".engine_snapshots",
        "audit_plots",
        "backtest_cache",
        "backtest_logs",
        "build",
        "daily_log",
        "debug_traces",
        "dist",
        "executed_order_tracker",
        "htmlcov",
        "option_value_plot",
        "plot_result_overlay",
        "polygon_api_option_data",
        "training_results",
    }
)
FORBIDDEN_PATH_PREFIXES = (
    "yfinance/prices/",
    "etrade_python_client/yfinance/",
)
FORBIDDEN_LOCAL_STATE_NAMES = frozenset(
    {
        ".etrade_oauth",
        ".etrade_session.json",
        ".netrc",
        ".pypirc",
        "config.ini",
        "credentials.json",
        "etrade_session.json",
        "latest_snapshot.json",
        "live_trading_settings.json",
        "manual_order_status.json",
        "nudge_history.json",
        "nudge_state.json",
        "production-arm.json",
        "production_arm.json",
        "runtime_config.json",
        "runtime_secrets.json",
        "secrets.json",
        "spy_tracking_data.json",
        "spy_vix_price_cache.json",
        "trade_status.json",
    }
)
FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pkl",
    ".pickle",
    ".numbers",
    ".parquet",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".db-journal",
    ".db-shm",
    ".db-wal",
    ".sqlite-journal",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite3-journal",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".tmp",
    ".swp",
    ".swo",
    ".save",
    ".orig",
    ".rej",
)
FORBIDDEN_AUTH_SUFFIXES = (".key", ".pem", ".p12", ".pfx")
UNSAFE_GIT_ENVIRONMENT_VARIABLES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)
BACKUP_NAME = re.compile(r"\.bak(?:\d+|[._-].*)?$", re.IGNORECASE)
ROTATED_LOG_NAME = re.compile(r"\.log(?:\..*)?$", re.IGNORECASE)
CACHE_DATA_NAME = re.compile(
    r"(?:^|_)cache\.(?:csv|json|db|sqlite|sqlite3)(?:$|[.~_-])",
    re.IGNORECASE,
)


class RepositoryConfigurationError(RuntimeError):
    """Git could not provide a deterministic view of the repository index."""


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One deterministic repository-index violation."""

    path: str
    code: str
    message: str

    def render(self) -> str:
        encoded_path = json.dumps(self.path, ensure_ascii=True)
        return f"{self.code} {encoded_path}: {self.message}"


@dataclass(frozen=True)
class ScanResult:
    """Repository root, tracked-path count, and sorted diagnostics."""

    root: Path
    tracked_path_count: int
    diagnostics: tuple[Diagnostic, ...]


def _run_git(arguments: Sequence[str], *, cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        command = " ".join(arguments)
        raise RepositoryConfigurationError(
            f"git {command} failed for {cwd}: {error or 'unknown error'}"
        )
    return completed.stdout


def resolve_git_root(start: Path) -> Path:
    """Resolve the containing Git root using Git rather than path assumptions."""

    start = start.resolve()
    if start.is_file():
        start = start.parent
    raw_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start)
    try:
        root_text = raw_root.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise RepositoryConfigurationError(
            "Git repository root is not valid UTF-8"
        ) from exc
    if not root_text:
        raise RepositoryConfigurationError("Git returned an empty repository root")
    root = Path(root_text)
    if not root.is_absolute():
        root = start / root
    return root.resolve()


def _decode_git_paths(payload: bytes, *, description: str) -> tuple[str, ...]:
    paths: list[str] = []
    for raw_path in payload.split(b"\0"):
        if not raw_path:
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryConfigurationError(
                f"{description} contains a path that is not valid UTF-8"
            ) from exc
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise RepositoryConfigurationError(
                f"{description} contains an unsafe path: {path!r}"
            )
        paths.append(path)
    return tuple(sorted(set(paths)))


def tracked_paths(root: Path) -> tuple[str, ...]:
    """Return every indexed path using Git's unambiguous NUL format."""

    payload = _run_git(["ls-files", "-z"], cwd=root)
    return _decode_git_paths(payload, description="Git index")


def tracked_ignored_paths(root: Path) -> tuple[str, ...]:
    """Return indexed paths that also match Git's standard ignore rules."""

    payload = _run_git(
        ["ls-files", "-z", "-ci", "--exclude-standard"],
        cwd=root,
    )
    return _decode_git_paths(payload, description="ignored Git index")


def tracked_index_violations(
    root: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Return non-regular modes and unmerged stages from the Git index."""

    payload = _run_git(["ls-files", "-z", "--stage"], cwd=root)
    unsafe_modes: dict[str, set[str]] = {}
    unmerged_stages: dict[str, set[str]] = {}
    for raw_entry in payload.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            mode_bytes, _object_id, stage_bytes = header.split(b" ")
            mode = mode_bytes.decode("ascii")
            stage = stage_bytes.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryConfigurationError(
                "staged Git index contains an invalid mode record"
            ) from exc
        (path,) = _decode_git_paths(
            raw_path + b"\0",
            description="staged Git index",
        )
        if mode not in {"100644", "100755"}:
            unsafe_modes.setdefault(path, set()).add(mode)
        if stage != "0":
            unmerged_stages.setdefault(path, set()).add(stage)
    normalized_modes = {
        path: tuple(sorted(modes))
        for path, modes in sorted(unsafe_modes.items())
    }
    normalized_stages = {
        path: tuple(sorted(stages))
        for path, stages in sorted(unmerged_stages.items())
    }
    return normalized_modes, normalized_stages


def _validate_git_environment() -> None:
    overridden = sorted(
        name for name in UNSAFE_GIT_ENVIRONMENT_VARIABLES if name in os.environ
    )
    if overridden:
        raise RepositoryConfigurationError(
            "unsafe Git repository override(s) are set: "
            + ", ".join(overridden)
        )


def _has_forbidden_suffix_variant(
    name: str,
    suffixes: Sequence[str],
) -> bool:
    for suffix in suffixes:
        position = name.rfind(suffix)
        if position < 0:
            continue
        remainder = name[position + len(suffix) :]
        if not remainder or remainder == "~" or remainder.startswith(
            (".", "_", "-")
        ):
            return True
    return False


def semantic_forbidden_reason(path: str) -> str | None:
    """Return a stable reason when *path* is forbidden independent of ignores."""

    pure_path = PurePosixPath(path)
    folded_parts = tuple(part.casefold() for part in pure_path.parts)
    folded_path = "/".join(folded_parts)

    if any(
        part in FORBIDDEN_ENVIRONMENT_DIRECTORIES for part in folded_parts
    ):
        return "virtual-environment content must not be tracked"

    if any(part.endswith(".egg-info") for part in folded_parts):
        return "generated Python package metadata must not be tracked"

    if any(part in FORBIDDEN_GENERATED_DIRECTORIES for part in folded_parts):
        return "generated or runtime output directory must not be tracked"

    if any(folded_path.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
        return "generated market-data cache must not be tracked"

    name = folded_parts[-1] if folded_parts else ""
    if name == ".env.example":
        return None
    if name == ".ds_store":
        return "operating-system metadata must not be tracked"
    if name == ".coverage" or name.startswith(".coverage."):
        return "generated coverage data must not be tracked"
    if (
        name == ".env"
        or name == ".envrc"
        or name.startswith(".env.")
        or ".env." in name
        or name.endswith(".env")
    ):
        return "local environment or credential file must not be tracked"
    if (
        folded_path == ".aws/credentials"
        or folded_path.endswith("/.aws/credentials")
        or (
            ".ssh" in folded_parts
            and name.startswith("id_")
        )
        or _has_forbidden_suffix_variant(name, FORBIDDEN_AUTH_SUFFIXES)
    ):
        return "private authentication material must not be tracked"
    if any(name.startswith(base) for base in FORBIDDEN_LOCAL_STATE_NAMES):
        return "local authentication, session, configuration, or state file must not be tracked"
    if name.startswith("order_audit_log.") or name.startswith(
        "order_audit_log_"
    ):
        return "runtime order-audit state must not be tracked"
    if name.startswith("experiments_log.jsonl"):
        return "generated experiment registry must not be tracked"
    if name in {"last_1000_logs.txt", "server_log.txt"}:
        return "runtime log output must not be tracked"
    if ROTATED_LOG_NAME.search(name):
        return "runtime log output must not be tracked"
    if CACHE_DATA_NAME.search(name):
        return "generated cache data must not be tracked"
    if name.endswith(("~", ".backup", ".old")):
        return "editor, compressed, or rotated backup must not be tracked"
    if _has_forbidden_suffix_variant(name, FORBIDDEN_ARTIFACT_SUFFIXES):
        return "generated cache, database, bytecode, or document artifact must not be tracked"
    if BACKUP_NAME.search(name):
        return "editor or migration backup must not be tracked"
    if folded_path == "polygonio/fix_patch.patch":
        return "obsolete local patch artifact must not be tracked"
    return None


def scan_repository(start: Path) -> ScanResult:
    """Scan the index for ignored and semantically forbidden tracked paths."""

    _validate_git_environment()
    root = resolve_git_root(start)
    indexed_paths = tracked_paths(root)
    ignored_paths = set(tracked_ignored_paths(root))
    unsafe_modes, unmerged_stages = tracked_index_violations(root)
    diagnostics: list[Diagnostic] = []

    for path in indexed_paths:
        semantic_reason = semantic_forbidden_reason(path)
        if path in ignored_paths:
            message = "tracked path also matches Git ignore rules"
            if semantic_reason is not None:
                message = f"{message}; {semantic_reason}"
            diagnostics.append(Diagnostic(path, "TRACKED_IGNORED", message))
        elif path in unmerged_stages:
            stages = ", ".join(unmerged_stages[path])
            diagnostics.append(
                Diagnostic(
                    path,
                    "UNMERGED_INDEX",
                    f"Git index contains unresolved stage(s) {stages}",
                )
            )
        elif path in unsafe_modes:
            modes = ", ".join(unsafe_modes[path])
            diagnostics.append(
                Diagnostic(
                    path,
                    "UNSAFE_MODE",
                    f"Git mode(s) {modes} are not regular files",
                )
            )
        elif semantic_reason is not None:
            diagnostics.append(
                Diagnostic(path, "FORBIDDEN_PATH", semantic_reason)
            )

    return ScanResult(
        root=root,
        tracked_path_count=len(indexed_paths),
        diagnostics=tuple(sorted(diagnostics)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when the Git index contains ignored, generated, runtime, "
            "cache, backup, or local-secret state."
        )
    )
    parser.add_argument(
        "--start",
        type=Path,
        default=Path.cwd(),
        help="path inside the repository (default: current working directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = scan_repository(args.start)
    except RepositoryConfigurationError as exc:
        print(f"repository hygiene configuration error: {exc}", file=sys.stderr)
        return 2

    if result.diagnostics:
        for diagnostic in result.diagnostics:
            print(diagnostic.render(), file=sys.stderr)
        print(
            (
                "repository index hygiene failed with "
                f"{len(result.diagnostics)} violation(s)."
            ),
            file=sys.stderr,
        )
        return 1

    print(
        (
            "repository index hygiene passed for "
            f"{result.tracked_path_count} tracked path(s)."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
