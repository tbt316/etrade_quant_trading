from __future__ import annotations

import os
import shlex
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "deploy" / "install_pi_service.sh"
SYNC = REPO_ROOT / "deploy" / "sync_to_pi.sh"


def _write_probe(bin_dir: Path, name: str, marker: Path) -> None:
    probe = bin_dir / name
    probe.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' {shlex.quote(name)} >> {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)


def _probe_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    marker = tmp_path / "remote-or-privileged-command.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("rsync", "ssh", "sudo", "systemctl", "tee"):
        _write_probe(bin_dir, command, marker)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{os.defpath}"
    return environment, marker


def test_installer_fails_before_any_privileged_or_service_action(
    tmp_path: Path,
) -> None:
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "installation is suspended" in completed.stderr
    assert not marker.exists()
    source = INSTALLER.read_text(encoding="utf-8")
    for forbidden in ("sudo ", "systemctl ", "ExecStart=", "START_NOW"):
        assert forbidden not in source


def test_restart_fails_before_ssh_rsync_or_service_action(
    tmp_path: Path,
) -> None:
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SYNC), "--restart"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "no files were synced" in completed.stderr
    assert not marker.exists()


def test_sync_only_path_is_retained_without_restart(tmp_path: Path) -> None:
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SYNC)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = marker.read_text(encoding="utf-8").splitlines()
    assert commands == ["ssh", "rsync"]


def test_deployment_scripts_are_valid_bash() -> None:
    for script in (INSTALLER, SYNC):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
