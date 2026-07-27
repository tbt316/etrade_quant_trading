from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "deploy" / "install_pi_service.sh"
BOOTSTRAP = REPO_ROOT / "deploy" / "pi_bootstrap.sh"
SYNC = REPO_ROOT / "deploy" / "sync_to_pi.sh"
RELEASE_HELPER = REPO_ROOT / "deploy" / "pi_release.sh"
HYGIENE_CHECKER = REPO_ROOT / "scripts" / "check_repo_hygiene.py"


def _write_probe(bin_dir: Path, name: str, marker: Path) -> None:
    probe = bin_dir / name
    prepare_result = ""
    if name == "ssh":
        prepare_result = (
            "if [[ \"$*\" == *\" prepare \"* ]]; then\n"
            "  printf '%s\\n' missing\n"
            "fi\n"
        )
    probe.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' {shlex.quote(name)} >> {shlex.quote(str(marker))}\n"
        f"{prepare_result}",
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


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def _git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _clean_sync_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    application = repository / "etrade_python_client"
    script = application / "deploy" / "sync_to_pi.sh"
    script.parent.mkdir(parents=True)
    script.write_text(SYNC.read_text(encoding="utf-8"), encoding="utf-8")
    release_helper = application / "deploy" / "pi_release.sh"
    release_helper.write_text(
        RELEASE_HELPER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release_helper.chmod(release_helper.stat().st_mode | stat.S_IXUSR)
    hygiene_checker = application / "scripts" / "check_repo_hygiene.py"
    hygiene_checker.parent.mkdir()
    hygiene_checker.write_text(
        HYGIENE_CHECKER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (application / "safe.py").write_text("committed = True\n", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "etrade_python_client/.env\n",
        encoding="utf-8",
    )
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "containment@example.invalid")
    _git(repository, "config", "user.name", "Containment Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return application, script


def _write_snapshot_rsync_probe(
    bin_dir: Path,
    marker: Path,
    *,
    required_name: str,
    forbidden_names: tuple[str, ...],
    expected_delete_count: int = 0,
) -> None:
    probe = bin_dir / "rsync"
    forbidden_checks = "\n".join(
        (
            f"if tar -tf \"$source_archive\" | grep -Fqx {shlex.quote(name)}; then "
            f"echo forbidden:{shlex.quote(name)} >> {shlex.quote(str(marker))}; "
            "exit 91; fi"
        )
        for name in forbidden_names
    )
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "args=(\"$@\")\n"
        "delete_count=0\n"
        "separator_count=0\n"
        "for argument in \"${args[@]}\"; do\n"
        "  if [[ \"$argument\" == \"--delete\" ]]; then "
        "delete_count=$((delete_count + 1)); fi\n"
        "  if [[ \"$argument\" == \"--\" ]]; then "
        "separator_count=$((separator_count + 1)); fi\n"
        "done\n"
        f"test \"$delete_count\" -eq {expected_delete_count}\n"
        "test \"$separator_count\" -eq 1\n"
        "source_archive=\"${args[${#args[@]}-2]}\"\n"
        "test -f \"$source_archive\"\n"
        f"tar -tf \"$source_archive\" | grep -Fqx {shlex.quote(required_name)}\n"
        f"{forbidden_checks}\n"
        f"printf '%s\\n' rsync >> {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    probe.chmod(probe.stat().st_mode | stat.S_IXUSR)


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


def test_bootstrap_fails_before_any_install_or_privileged_action(
    tmp_path: Path,
) -> None:
    environment, marker = _probe_environment(tmp_path)
    bin_dir = tmp_path / "bin"
    for command in ("apt-get", "uname", "python3", "mkdir"):
        _write_probe(bin_dir, command, marker)

    completed = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "bootstrap is suspended" in completed.stderr
    assert not marker.exists()
    source = BOOTSTRAP.read_text(encoding="utf-8")
    for forbidden in ("sudo ", "apt-get", "-m pip", "-m venv", "mkdir "):
        assert forbidden not in source


def test_restart_fails_before_ssh_rsync_or_service_action(
    tmp_path: Path,
) -> None:
    environment, marker = _probe_environment(tmp_path)
    bin_dir = tmp_path / "bin"
    for command in ("git", "tar", "mktemp"):
        _write_probe(bin_dir, command, marker)

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


def test_sync_uses_only_the_clean_committed_snapshot(tmp_path: Path) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    (application / ".env").write_text("ignored secret\n", encoding="utf-8")
    rogue = application / "scratch" / "rogue.py"
    rogue.parent.mkdir()
    rogue.write_text("untracked_mutation_capability = True\n", encoding="utf-8")
    environment, marker = _probe_environment(tmp_path)
    snapshot_tmp = tmp_path / "snapshots"
    snapshot_tmp.mkdir()
    environment["TMPDIR"] = str(snapshot_tmp)
    _write_snapshot_rsync_probe(
        tmp_path / "bin",
        marker,
        required_name="safe.py",
        forbidden_names=(".env", "scratch/rogue.py"),
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    commands = marker.read_text(encoding="utf-8").splitlines()
    assert commands == ["ssh", "rsync", "ssh", "ssh"]
    assert "Preparing owner-read-only, reverified release" in completed.stdout
    assert "no service was restarted" in completed.stdout
    assert list(snapshot_tmp.glob("etrade-code-snapshot.*")) == []


def test_sync_rejects_a_tracked_secret_before_remote_actions_without_naming_it(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    secret_path = application / ".env"
    secret_value = "API_SECRET=must-not-leak"
    secret_path.write_text(f"{secret_value}\n", encoding="utf-8")
    _git(
        application.parent,
        "add",
        "--force",
        "etrade_python_client/.env",
    )
    _git(application.parent, "commit", "--quiet", "-m", "secret fixture")
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    combined_output = completed.stdout + completed.stderr
    assert completed.returncode == 78
    assert "failed deployment hygiene" in completed.stderr
    assert "violating paths are redacted" in completed.stderr
    assert secret_path.name not in combined_output
    assert secret_value not in combined_output
    assert not marker.exists()


def test_sync_allows_safe_tracked_json_even_when_ignore_rules_match(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    safe_json = application / "docs" / "review_protocol.json"
    safe_json.parent.mkdir()
    safe_json.write_text('{"status":"safe-static-contract"}\n', encoding="utf-8")
    gitignore = application.parent / ".gitignore"
    gitignore.write_text(
        gitignore.read_text(encoding="utf-8")
        + "etrade_python_client/docs/*.json\n",
        encoding="utf-8",
    )
    _git(
        application.parent,
        "add",
        "--force",
        ".gitignore",
        "etrade_python_client/docs/review_protocol.json",
    )
    _git(application.parent, "commit", "--quiet", "-m", "safe JSON fixture")
    environment, marker = _probe_environment(tmp_path)
    _write_snapshot_rsync_probe(
        tmp_path / "bin",
        marker,
        required_name="docs/review_protocol.json",
        forbidden_names=(),
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "ssh",
        "rsync",
        "ssh",
        "ssh",
    ]


def test_sync_delete_compatibility_flag_cannot_mutate_a_clean_release(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    environment, marker = _probe_environment(tmp_path)
    environment["SYNC_DELETE"] = "1"
    _write_snapshot_rsync_probe(
        tmp_path / "bin",
        marker,
        required_name="safe.py",
        forbidden_names=(),
        expected_delete_count=0,
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "ssh",
        "rsync",
        "ssh",
        "ssh",
    ]
    assert "obsolete" in completed.stderr


@pytest.mark.parametrize("staged", [False, True])
def test_sync_rejects_tracked_changes_before_remote_or_snapshot_actions(
    tmp_path: Path,
    staged: bool,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    (application / "safe.py").write_text("committed = False\n", encoding="utf-8")
    if staged:
        _git(application.parent, "add", "etrade_python_client/safe.py")
    environment, marker = _probe_environment(tmp_path)
    snapshot_tmp = tmp_path / "snapshots"
    snapshot_tmp.mkdir()
    environment["TMPDIR"] = str(snapshot_tmp)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "commit them before syncing" in completed.stderr
    assert not marker.exists()
    assert list(snapshot_tmp.glob("etrade-code-snapshot.*")) == []


def test_sync_ignores_local_git_replacement_objects(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    repository = application.parent
    original_commit = _git_output(repository, "rev-parse", "HEAD")
    (application / "safe.py").write_text(
        "unreviewed_replacement = True\n",
        encoding="utf-8",
    )
    _git(repository, "add", "etrade_python_client/safe.py")
    replacement_tree = _git_output(repository, "write-tree")
    replacement_commit = _git_output(
        repository,
        "commit-tree",
        replacement_tree,
        "-p",
        original_commit,
        "-m",
        "unreviewed replacement",
    )
    _git(
        repository,
        "update-index",
        "--skip-worktree",
        "etrade_python_client/safe.py",
    )
    _git(repository, "replace", original_commit, replacement_commit)
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "commit them before syncing" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_repo_configured_worktree_redirection(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "ancestor"
    repository = ancestor / "nested"
    application = repository / "etrade_python_client"
    script = application / "deploy" / "sync_to_pi.sh"
    script.parent.mkdir(parents=True)
    script.write_text(SYNC.read_text(encoding="utf-8"), encoding="utf-8")
    safe_path = application / "safe.py"
    safe_path.write_text("outer_unreviewed = True\n", encoding="utf-8")

    _git(ancestor, "init", "--quiet")
    _git(ancestor, "config", "user.email", "containment@example.invalid")
    _git(ancestor, "config", "user.name", "Containment Test")
    _git(ancestor, "add", ".")
    _git(ancestor, "commit", "--quiet", "-m", "outer fixture")

    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "containment@example.invalid")
    _git(repository, "config", "user.name", "Containment Test")
    safe_path.write_text("inner_reviewed = True\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "inner fixture")
    safe_path.write_text("outer_unreviewed = True\n", encoding="utf-8")
    _git(
        repository,
        "update-index",
        "--skip-worktree",
        "etrade_python_client/safe.py",
    )
    _git(repository, "config", "core.worktree", str(ancestor))
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "repository identity changed" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_symbolic_link_state_before_remote_actions(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    (application / "config.ini").symlink_to(application / "safe.py")
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "--state"],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "state sync is suspended" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_non_regular_state_before_remote_actions(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    (application / "config.ini").mkdir()
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script), "--state"],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "state sync is suspended" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_committed_code_symlink_before_remote_actions(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    link = application / "runtime_extension.so"
    link.symlink_to(tmp_path / "outside-runtime-extension.so")
    _git(application.parent, "add", "etrade_python_client/runtime_extension.so")
    _git(application.parent, "commit", "--quiet", "-m", "add symlink")
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "contains an unsafe file type" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_git_attribute_archive_omission_before_remote_actions(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    attributes = application.parent / ".git" / "info" / "attributes"
    attributes.write_text(
        "safe.py export-ignore\n",
        encoding="utf-8",
    )
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "omitted or changed a tracked path" in completed.stderr
    assert not marker.exists()


def test_sync_rejects_git_attribute_export_substitution_before_remote_actions(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    (application / "safe.py").write_text(
        "commit = '$Format:%H$'\n",
        encoding="utf-8",
    )
    _git(application.parent, "add", "etrade_python_client/safe.py")
    _git(application.parent, "commit", "--quiet", "-m", "add format marker")
    attributes = application.parent / ".git" / "info" / "attributes"
    attributes.write_text(
        "etrade_python_client/safe.py export-subst\n",
        encoding="utf-8",
    )
    environment, marker = _probe_environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "export-subst deployment path" in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mutation_command", "expected_message"),
    [
        (
            "printf '%s\\n' 'archive_mutation = True' "
            '> "$mutation_root/safe.py"',
            "differs from an exact Git blob",
        ),
        (
            'chmod 755 "$mutation_root/safe.py"',
            "changed a tracked executable mode",
        ),
        (
            "printf '%s\\n' 'ambient = True' "
            '> "$mutation_root/ambient.py"',
            "contains an untracked file",
        ),
    ],
)
def test_sync_rejects_an_archive_tree_that_differs_from_exact_git(
    tmp_path: Path,
    mutation_command: str,
    expected_message: str,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    environment, marker = _probe_environment(tmp_path)
    actual_git = shutil.which("git")
    assert actual_git is not None
    git_probe = tmp_path / "bin" / "git"
    git_probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "is_archive=0\n"
        "archive_path=''\n"
        "for argument in \"$@\"; do\n"
        "  if [[ \"$argument\" == archive ]]; then is_archive=1; fi\n"
        "  if [[ \"$argument\" == --output=* ]]; then\n"
        "    archive_path=\"${argument#--output=}\"\n"
        "  fi\n"
        "done\n"
        f"{shlex.quote(actual_git)} \"$@\"\n"
        "if [[ \"$is_archive\" == 1 ]]; then\n"
        "  mutation_root=\"$(mktemp -d)\"\n"
        "  trap 'rm -rf -- \"$mutation_root\"' EXIT\n"
        "  tar -xf \"$archive_path\" -C \"$mutation_root\"\n"
        f"  {mutation_command}\n"
        "  tar -cf \"$archive_path\" -C \"$mutation_root\" .\n"
        "fi\n",
        encoding="utf-8",
    )
    git_probe.chmod(git_probe.stat().st_mode | stat.S_IXUSR)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert expected_message in completed.stderr
    assert not marker.exists()


def test_sync_state_is_suspended_before_git_snapshot_or_remote_actions(
    tmp_path: Path,
) -> None:
    environment, marker = _probe_environment(tmp_path)
    for command in ("git", "tar", "mktemp"):
        _write_probe(tmp_path / "bin", command, marker)

    completed = subprocess.run(
        ["bash", str(SYNC), "--state"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "state sync is suspended" in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("name", "value", "expected_message"),
    [
        ("PI_TARGET", "-oProxyCommand=malicious", "PI_TARGET"),
        ("PI_TARGET", "pi@host;malicious", "PI_TARGET"),
        ("PI_DIR", "/", "PI_DIR"),
        ("PI_DIR", "/home", "PI_DIR"),
        ("PI_DIR", "/home/pi/app';malicious", "PI_DIR"),
        ("PI_DIR", "/home/pi/../root", "PI_DIR"),
        ("SYNC_DELETE", "yes", "SYNC_DELETE"),
        ("GIT_DIR", "/tmp/alternate.git", "Unsafe Git"),
        ("GIT_INDEX_FILE", "/tmp/alternate.index", "Unsafe Git"),
        ("GIT_WORK_TREE", "/tmp/alternate-tree", "Unsafe Git"),
        ("GIT_COMMON_DIR", "/tmp/alternate-common", "Unsafe Git"),
        (
            "GIT_OBJECT_DIRECTORY",
            "/tmp/alternate-objects",
            "Unsafe Git",
        ),
        (
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "/tmp/alternate-objects",
            "Unsafe Git",
        ),
        ("GIT_NAMESPACE", "alternate", "Unsafe Git"),
        ("GIT_REPLACE_REF_BASE", "refs/alternate/", "Unsafe Git"),
    ],
)
def test_sync_rejects_unsafe_environment_before_local_or_remote_actions(
    tmp_path: Path,
    name: str,
    value: str,
    expected_message: str,
) -> None:
    environment, marker = _probe_environment(tmp_path)
    for command in ("git", "tar", "mktemp"):
        _write_probe(tmp_path / "bin", command, marker)
    environment[name] = value

    completed = subprocess.run(
        ["bash", str(SYNC)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert expected_message in completed.stderr
    assert not marker.exists()


def test_archive_failure_cleans_snapshot_before_any_remote_action(
    tmp_path: Path,
) -> None:
    application, script = _clean_sync_fixture(tmp_path)
    environment, marker = _probe_environment(tmp_path)
    snapshot_tmp = tmp_path / "snapshots"
    snapshot_tmp.mkdir()
    environment["TMPDIR"] = str(snapshot_tmp)
    actual_git = shutil.which("git")
    assert actual_git is not None
    git_probe = tmp_path / "bin" / "git"
    git_probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "for argument in \"$@\"; do\n"
        "  if [[ \"$argument\" == \"archive\" ]]; then exit 92; fi\n"
        "done\n"
        f"exec {shlex.quote(actual_git)} \"$@\"\n",
        encoding="utf-8",
    )
    git_probe.chmod(git_probe.stat().st_mode | stat.S_IXUSR)

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=application,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not marker.exists()
    assert list(snapshot_tmp.glob("etrade-code-snapshot.*")) == []


def test_deployment_scripts_are_valid_bash() -> None:
    for script in (INSTALLER, BOOTSTRAP, SYNC, RELEASE_HELPER):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
