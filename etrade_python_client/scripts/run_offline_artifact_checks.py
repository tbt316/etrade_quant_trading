#!/usr/bin/env python3
"""Run installed-artifact checks inside CI's non-root network namespace."""

from __future__ import annotations

import argparse
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class OfflineCheckError(RuntimeError):
    """The operating-system containment contract is not active."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--runtime-smoke", type=Path, required=True)
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    parser.add_argument("--offline-home", type=Path, required=True)
    parser.add_argument("--offline-temp", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser


def _contained_environment(
    primary_python: Path,
    secondary_python: Path,
    offline_home: Path,
    offline_temp: Path,
) -> dict[str, str]:
    return {
        "ETRADE_TEST_ARTIFACT": "1",
        "ETRADE_TEST_NETWORK": "deny",
        "ETRADE_TEST_NETWORK_NAMESPACE": "1",
        "HOME": str(offline_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": os.environ.get("LOGNAME", "runner"),
        "MASSIVE_OFFLINE_ONLY": "1",
        "MPLCONFIGDIR": str(offline_home / "matplotlib"),
        "PATH": os.pathsep.join(
            (
                str(primary_python.parent),
                str(secondary_python.parent),
                "/usr/bin",
                "/bin",
            )
        ),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(offline_temp),
        "TZ": "UTC",
        "USER": os.environ.get("USER", "runner"),
        "XDG_CACHE_HOME": str(offline_home / "cache"),
    }


def _owner_only_empty_directory(path: Path, label: str) -> None:
    metadata = path.stat()
    if (
        not path.is_dir()
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(path.iterdir())
    ):
        raise OfflineCheckError(
            f"{label} must be empty, owner-only, and runner-owned"
        )


def _verify_containment(
    work_dir: Path,
    offline_home: Path,
    offline_temp: Path,
) -> None:
    if os.geteuid() == 0:
        raise OfflineCheckError("artifact checks must not run as root")
    interfaces = {name for _index, name in socket.if_nameindex()}
    if interfaces != {"lo"}:
        raise OfflineCheckError(
            f"network namespace must expose only loopback: {sorted(interfaces)}"
        )
    _owner_only_empty_directory(work_dir, "runtime smoke directory")
    _owner_only_empty_directory(offline_home, "offline home")
    _owner_only_empty_directory(offline_temp, "offline temp")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runtime_python = Path(os.path.abspath(args.runtime_python))
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise OfflineCheckError(
            f"runtime Python is not an executable file: {runtime_python}"
        )
    runtime_smoke = args.runtime_smoke.resolve(strict=True)
    runtime_work_dir = args.runtime_work_dir.resolve(strict=True)
    offline_home = args.offline_home.resolve(strict=True)
    offline_temp = args.offline_temp.resolve(strict=True)
    repository_root = args.repository_root.resolve(strict=True)
    _verify_containment(runtime_work_dir, offline_home, offline_temp)
    test_python = Path(sys.executable).absolute()
    runtime_environment = _contained_environment(
        runtime_python,
        test_python,
        offline_home,
        offline_temp,
    )
    test_environment = _contained_environment(
        test_python,
        runtime_python,
        offline_home,
        offline_temp,
    )

    subprocess.run(
        [str(runtime_python), str(runtime_smoke)],
        cwd=runtime_work_dir,
        env=runtime_environment,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not integration",
            "etrade_python_client/tests",
            "--ignore=etrade_python_client/tests/test_etrade_mutation_boundary.py",
            "--ignore=etrade_python_client/tests/test_repo_hygiene.py",
        ],
        cwd=repository_root,
        env=test_environment,
        check=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OfflineCheckError, OSError, subprocess.CalledProcessError) as error:
        print(f"offline artifact checks failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
