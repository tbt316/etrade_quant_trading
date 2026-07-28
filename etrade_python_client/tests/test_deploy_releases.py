from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC = REPO_ROOT / "deploy" / "sync_to_pi.sh"
RELEASE_HELPER = REPO_ROOT / "deploy" / "pi_release.sh"
HYGIENE_CHECKER = REPO_ROOT / "scripts" / "check_repo_hygiene.py"
SECRET_CHECKER = REPO_ROOT / "scripts" / "check_secret_content.py"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_helper(
    base: Path,
    action: str,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "bash",
            str(RELEASE_HELPER),
            action,
            str(base),
            *arguments,
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(
    tmp_path: Path,
    name: str,
    *,
    payload: str,
    valid_health_scripts: bool = True,
) -> Path:
    source = tmp_path / f"{name}-source"
    deploy = source / "deploy"
    deploy.mkdir(parents=True)
    sync_source = "#!/usr/bin/env bash\nset -euo pipefail\n"
    helper_source = "#!/usr/bin/env bash\nset -euo pipefail\n"
    if not valid_health_scripts:
        helper_source = "#!/usr/bin/env bash\nif then\n"
    (deploy / "sync_to_pi.sh").write_text(sync_source, encoding="utf-8")
    (deploy / "pi_release.sh").write_text(helper_source, encoding="utf-8")
    (source / "safe.py").write_text(payload, encoding="utf-8")
    archive = tmp_path / f"{name}.tar"
    completed = _run(
        ["tar", "-cf", str(archive), "-C", str(source), "."],
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    return archive


def _install(
    base: Path,
    archive: Path,
    commit: str,
    token: str,
) -> str:
    digest = _sha256(archive)
    release_id = f"{commit}-{digest}"
    prepared = _run_helper(base, "prepare", token)
    assert prepared.returncode == 0, prepared.stderr
    upload = base / "incoming" / f"{token}.tar"
    shutil.copyfile(archive, upload)
    installed = _run_helper(
        base,
        "install",
        release_id,
        commit,
        digest,
        token,
    )
    assert installed.returncode == 0, installed.stderr
    return release_id


def _activate(
    base: Path,
    release_id: str,
    token: str,
    *,
    expected_generation: str | None = None,
) -> None:
    if expected_generation is None:
        current = base / "current"
        expected_generation = (
            _generation(base) if current.is_symlink() else "missing"
        )
    activated = _run_helper(
        base,
        "activate",
        release_id,
        expected_generation,
        token,
    )
    assert activated.returncode == 0, activated.stderr


def _current(base: Path) -> str:
    generation = _generation(base)
    return os.readlink(base / "selections" / generation).removeprefix(
        "../releases/"
    )


def _generation(base: Path) -> str:
    return os.readlink(base / "current").removeprefix("selections/")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_install_creates_content_addressed_release_and_atomic_current(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    base.mkdir(parents=True)
    rogue = base / "rogue.py"
    rogue.write_text("ambient = True\n", encoding="utf-8")
    archive = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )

    release_id = _install(base, archive, COMMIT_A, "install-a")
    _activate(base, release_id, "activate-a")

    assert _current(base) == release_id
    assert (base / "current" / "safe.py").read_text(encoding="utf-8") == (
        "release = 'a'\n"
    )
    assert rogue.read_text(encoding="utf-8") == "ambient = True\n"
    assert not (base / "current" / "rogue.py").exists()
    assert _sha256(base / "artifacts" / f"{release_id}.tar") == (
        release_id.split("-", 1)[1]
    )
    metadata = (base / "verified" / release_id).read_text(encoding="utf-8")
    assert f"release_id={release_id}\n" in metadata
    for private_directory in (
        base,
        base / "releases",
        base / "artifacts",
        base / "verified",
        base / "incoming",
        base / "selections",
    ):
        assert _mode(private_directory) == 0o700
    assert _mode(base / "artifacts" / f"{release_id}.tar") == 0o400
    assert _mode(base / "verified" / release_id) == 0o400
    assert not (base / ".deploy-lock").exists()


@pytest.mark.parametrize("failure", ["checksum", "extract", "health"])
def test_failed_install_never_changes_current(
    tmp_path: Path,
    failure: str,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    _activate(base, release_a, "activate-a")

    if failure == "extract":
        archive_b = tmp_path / "not-a-tar"
        archive_b.write_bytes(b"not a tar archive")
    else:
        archive_b = _make_archive(
            tmp_path,
            "release-b",
            payload="release = 'b'\n",
            valid_health_scripts=failure != "health",
        )
    digest_b = _sha256(archive_b)
    release_b = f"{COMMIT_B}-{digest_b}"
    token = f"install-b-{failure}"
    prepared = _run_helper(base, "prepare", token)
    assert prepared.returncode == 0, prepared.stderr
    upload = base / "incoming" / f"{token}.tar"
    shutil.copyfile(archive_b, upload)
    expected_digest = digest_b
    if failure == "checksum":
        expected_digest = "c" * 64
        release_b = f"{COMMIT_B}-{expected_digest}"

    installed = _run_helper(
        base,
        "install",
        release_b,
        COMMIT_B,
        expected_digest,
        token,
    )

    assert installed.returncode != 0
    assert _current(base) == release_a
    assert not (base / "verified" / release_b).exists()
    assert not (base / ".deploy-lock").exists()


@pytest.mark.parametrize(
    ("tamper_kind", "expected_message"),
    [
        ("bytes", "differs from its canonical archive"),
        ("mode", "mode differs from its canonical archive"),
    ],
)
def test_rollback_refuses_a_tampered_release(
    tmp_path: Path,
    tamper_kind: str,
    expected_message: str,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    archive_b = _make_archive(
        tmp_path,
        "release-b",
        payload="release = 'b'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    _activate(base, release_a, "activate-a")
    release_b = _install(base, archive_b, COMMIT_B, "install-b")
    _activate(base, release_b, "activate-b")
    tampered = base / "releases" / release_a / "safe.py"
    if tamper_kind == "bytes":
        tampered.chmod(0o600)
        tampered.write_text("tampered = True\n", encoding="utf-8")
        tampered.chmod(0o400)
    else:
        tampered.chmod(0o500)

    rolled_back = _run_helper(base, "rollback", release_a, "rollback-a")

    assert rolled_back.returncode == 78
    assert expected_message in rolled_back.stderr
    assert _current(base) == release_b


def _write_atomic_replace_probe(
    bin_dir: Path,
    *,
    move_then_fail: bool,
) -> None:
    actual_python = shutil.which("python3")
    assert actual_python is not None
    probe = bin_dir / "python3"
    body = "#!/usr/bin/env bash\nset -euo pipefail\n"
    if move_then_fail:
        body += f"{shlex.quote(actual_python)} \"$@\"\n"
    body += "exit 93\n"
    probe.write_text(body, encoding="utf-8")
    probe.chmod(0o700)


@pytest.mark.parametrize("move_then_fail", [False, True])
def test_pointer_rename_failure_is_always_recoverable(
    tmp_path: Path,
    move_then_fail: bool,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    archive_b = _make_archive(
        tmp_path,
        "release-b",
        payload="release = 'b'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    _activate(base, release_a, "activate-a")
    release_b = _install(base, archive_b, COMMIT_B, "install-b")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_atomic_replace_probe(bin_dir, move_then_fail=move_then_fail)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{os.defpath}"
    expected_generation = _generation(base)

    activated = _run_helper(
        base,
        "activate",
        release_b,
        expected_generation,
        "activate-b-fail",
        env=environment,
    )

    assert activated.returncode == 93
    expected_current = release_b if move_then_fail else release_a
    assert _current(base) == expected_current
    assert not (base / ".deploy-lock").exists()
    if move_then_fail:
        rolled_back = _run_helper(
            base,
            "rollback",
            release_a,
            "recover-a",
        )
        assert rolled_back.returncode == 0, rolled_back.stderr
        assert _current(base) == release_a


@pytest.mark.parametrize("lock_kind", ["directory", "symlink"])
def test_remote_operation_lock_fails_closed(
    tmp_path: Path,
    lock_kind: str,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    prepared = _run_helper(base, "prepare", "first")
    assert prepared.returncode == 0, prepared.stderr
    lock = base / ".deploy-lock"
    if lock_kind == "directory":
        lock.mkdir(mode=0o700)
        owner = lock / "owner"
        owner.write_text("another-operation\n", encoding="utf-8")
        owner.chmod(0o600)
    else:
        outside = tmp_path / "outside-lock"
        outside.mkdir()
        lock.symlink_to(outside, target_is_directory=True)

    blocked = _run_helper(base, "prepare", "second")

    assert blocked.returncode == 78
    assert "lock" in blocked.stderr
    assert not (base / "incoming" / "second.tar").exists()


def test_remote_base_rejects_a_symbolic_link_path_component(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "remote-alias"
    alias.symlink_to(outside, target_is_directory=True)
    base = alias / "etrade_python_client"

    prepared = _run_helper(base, "prepare", "unsafe-base")

    assert prepared.returncode == 78
    assert "symbolic-link path component" in prepared.stderr
    assert not (outside / "etrade_python_client").exists()


def test_install_rejects_an_outside_hard_link_before_permission_changes(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive = _make_archive(
        tmp_path,
        "release-hard-link",
        payload="release = 'aliased'\n",
    )
    digest = _sha256(archive)
    release_id = f"{COMMIT_A}-{digest}"
    token = "hard-link"
    prepared = _run_helper(base, "prepare", token)
    assert prepared.returncode == 0, prepared.stderr
    shutil.copyfile(archive, base / "incoming" / f"{token}.tar")
    release = base / "releases" / release_id
    release.mkdir()
    extracted = _run(
        ["tar", "-xf", str(archive), "-C", str(release)],
        cwd=tmp_path,
    )
    assert extracted.returncode == 0, extracted.stderr
    outside = tmp_path / "outside-runtime-state"
    outside.write_text("release = 'aliased'\n", encoding="utf-8")
    before_mode = _mode(outside)
    (release / "safe.py").unlink()
    os.link(outside, release / "safe.py")

    installed = _run_helper(
        base,
        "install",
        release_id,
        COMMIT_A,
        digest,
        token,
    )

    assert installed.returncode == 78
    assert "multi-link deployment file" in installed.stderr
    assert _mode(outside) == before_mode
    assert outside.stat().st_ino == (release / "safe.py").stat().st_ino
    assert not (base / "verified" / release_id).exists()
    assert not (base / "current").exists()


def test_archive_and_verified_metadata_reject_multiple_links(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive = _make_archive(
        tmp_path,
        "release-multi-link",
        payload="release = 'a'\n",
    )
    digest = _sha256(archive)
    release_id = f"{COMMIT_A}-{digest}"
    prepared = _run_helper(base, "prepare", "linked-upload")
    assert prepared.returncode == 0, prepared.stderr
    outside_archive = tmp_path / "outside-release.tar"
    shutil.copyfile(archive, outside_archive)
    os.link(
        outside_archive,
        base / "incoming" / "linked-upload.tar",
    )

    rejected_upload = _run_helper(
        base,
        "install",
        release_id,
        COMMIT_A,
        digest,
        "linked-upload",
    )

    assert rejected_upload.returncode == 78
    assert "multi-link deployment file" in rejected_upload.stderr

    release_id = _install(base, archive, COMMIT_A, "install-clean")
    metadata = base / "verified" / release_id
    metadata.chmod(0o600)
    outside_metadata = tmp_path / "outside-metadata"
    os.link(metadata, outside_metadata)
    metadata.chmod(0o400)

    rejected_metadata = _run_helper(
        base,
        "activate",
        release_id,
        "missing",
        "activate-linked-metadata",
    )

    assert rejected_metadata.returncode == 78
    assert "multi-link deployment file" in rejected_metadata.stderr
    assert not (base / "current").exists()


def test_stale_publish_cannot_replace_a_newer_activation(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    archive_b = _make_archive(
        tmp_path,
        "release-b",
        payload="release = 'b'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    release_b = _install(base, archive_b, COMMIT_B, "install-b")
    prepare_a = _run_helper(base, "prepare", "publish-a")
    prepare_b = _run_helper(base, "prepare", "publish-b")
    assert prepare_a.stdout.strip() == "missing"
    assert prepare_b.stdout.strip() == "missing"

    _activate(
        base,
        release_b,
        "activate-b",
        expected_generation=prepare_b.stdout.strip(),
    )
    stale = _run_helper(
        base,
        "activate",
        release_a,
        prepare_a.stdout.strip(),
        "activate-a-stale",
    )

    assert stale.returncode == 78
    assert "current generation changed after prepare" in stale.stderr
    assert _current(base) == release_b


def test_activation_generation_blocks_release_id_aba(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    archive_b = _make_archive(
        tmp_path,
        "release-b",
        payload="release = 'b'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    release_b = _install(base, archive_b, COMMIT_B, "install-b")
    _activate(base, release_a, "activate-a", expected_generation="missing")
    first_a_generation = _generation(base)
    stale_prepare = _run_helper(base, "prepare", "stale-b")
    current_prepare = _run_helper(base, "prepare", "current-b")
    assert stale_prepare.stdout.strip() == first_a_generation
    assert current_prepare.stdout.strip() == first_a_generation

    _activate(
        base,
        release_b,
        "activate-b",
        expected_generation=current_prepare.stdout.strip(),
    )
    rolled_back = _run_helper(base, "rollback", release_a, "rollback-a")
    assert rolled_back.returncode == 0, rolled_back.stderr
    second_a_generation = _generation(base)
    assert second_a_generation != first_a_generation
    assert _current(base) == release_a

    stale = _run_helper(
        base,
        "activate",
        release_b,
        stale_prepare.stdout.strip(),
        "activate-b-stale",
    )

    assert stale.returncode == 78
    assert "current generation changed after prepare" in stale.stderr
    assert _current(base) == release_a
    assert _generation(base) == second_a_generation


def test_current_is_rechecked_immediately_before_pointer_replace(
    tmp_path: Path,
) -> None:
    base = tmp_path / "remote" / "etrade_python_client"
    archive_a = _make_archive(
        tmp_path,
        "release-a",
        payload="release = 'a'\n",
    )
    archive_b = _make_archive(
        tmp_path,
        "release-b",
        payload="release = 'b'\n",
    )
    release_a = _install(base, archive_a, COMMIT_A, "install-a")
    _activate(base, release_a, "activate-a")
    release_b = _install(base, archive_b, COMMIT_B, "install-b")
    bin_dir = tmp_path / "bin-current-race"
    bin_dir.mkdir()
    race_generation = "c" * 64
    (base / "selections" / race_generation).symlink_to(
        f"../releases/{release_b}"
    )
    actual_ln = shutil.which("ln")
    assert actual_ln is not None
    probe = bin_dir / "ln"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{shlex.quote(actual_ln)} \"$@\"\n"
        f"rm -f -- {shlex.quote(str(base / 'current'))}\n"
        f"{shlex.quote(actual_ln)} -s "
        f"{shlex.quote(f'selections/{race_generation}')} "
        f"{shlex.quote(str(base / 'current'))}\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{os.defpath}"
    expected_generation = _generation(base)

    activated = _run_helper(
        base,
        "activate",
        release_b,
        expected_generation,
        "activate-b-race",
        env=environment,
    )

    assert activated.returncode == 78
    assert "current changed before activation" in activated.stderr
    assert _current(base) == release_b
    assert not (base / ".deploy-lock").exists()


def _git(repository: Path, *arguments: str) -> None:
    completed = _run(
        ["git", "-C", str(repository), *arguments],
        cwd=repository,
    )
    assert completed.returncode == 0, completed.stderr


def _sync_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    application = repository / "etrade_python_client"
    deploy = application / "deploy"
    deploy.mkdir(parents=True)
    script = deploy / "sync_to_pi.sh"
    script.write_text(SYNC.read_text(encoding="utf-8"), encoding="utf-8")
    helper = deploy / "pi_release.sh"
    helper.write_text(
        RELEASE_HELPER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    helper.chmod(0o755)
    hygiene_checker = application / "scripts" / "check_repo_hygiene.py"
    hygiene_checker.parent.mkdir()
    hygiene_checker.write_text(
        HYGIENE_CHECKER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    secret_checker = application / "scripts" / "check_secret_content.py"
    secret_checker.write_text(
        SECRET_CHECKER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (application / "safe.py").write_text("release = 1\n", encoding="utf-8")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "release@example.invalid")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "release one")
    return application, script


def _write_remote_stubs(
    bin_dir: Path,
    marker: Path,
    *,
    fail_upload: bool = False,
) -> None:
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test \"$1\" = --\n"
        "shift\n"
        "target=\"$1\"\n"
        "shift\n"
        f"printf 'ssh:%s\\n' \"$*\" >> {shlex.quote(str(marker))}\n"
        "bash -c \"$1\"\n",
        encoding="utf-8",
    )
    ssh.chmod(0o700)
    rsync = bin_dir / "rsync"
    if fail_upload:
        rsync_source = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\n' rsync-failed >> {shlex.quote(str(marker))}\n"
            "exit 91\n"
        )
    else:
        rsync_source = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "args=(\"$@\")\n"
            "source_file=\"${args[${#args[@]}-2]}\"\n"
            "destination=\"${args[${#args[@]}-1]}\"\n"
            "destination_path=\"${destination#*:}\"\n"
            "mkdir -p -- \"${destination_path%/*}\"\n"
            "cp -- \"$source_file\" \"$destination_path\"\n"
            f"printf '%s\\n' rsync >> {shlex.quote(str(marker))}\n"
        )
    rsync.write_text(rsync_source, encoding="utf-8")
    rsync.chmod(0o700)
    for forbidden in ("sudo", "systemctl"):
        probe = bin_dir / forbidden
        probe.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' {forbidden} >> {shlex.quote(str(marker))}\n"
            "exit 97\n",
            encoding="utf-8",
        )
        probe.chmod(0o700)


def _sync_environment(
    tmp_path: Path,
    remote_base: Path,
    *,
    fail_upload: bool = False,
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir(exist_ok=True)
    marker = tmp_path / "remote-operations.log"
    _write_remote_stubs(bin_dir, marker, fail_upload=fail_upload)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}{os.pathsep}{os.defpath}"
    environment["TMPDIR"] = str(snapshot_dir)
    environment["PI_TARGET"] = "pi@test-host"
    environment["PI_DIR"] = str(remote_base)
    return environment, marker


def _release_id(stdout: str) -> str:
    match = re.search(r"Activated release ([0-9a-f]{40}-[0-9a-f]{64})", stdout)
    assert match is not None, stdout
    return match.group(1)


def test_sync_and_explicit_rollback_use_only_stubbed_remote_operations(
    tmp_path: Path,
) -> None:
    application, script = _sync_fixture(tmp_path)
    remote_base = tmp_path / "remote" / "etrade_python_client"
    environment, marker = _sync_environment(tmp_path, remote_base)

    first = _run(["bash", str(script)], cwd=application, env=environment)
    assert first.returncode == 0, first.stderr
    release_a = _release_id(first.stdout)
    assert _current(remote_base) == release_a

    (application / "safe.py").write_text("release = 2\n", encoding="utf-8")
    _git(application.parent, "add", "etrade_python_client/safe.py")
    _git(application.parent, "commit", "--quiet", "-m", "release two")
    second = _run(["bash", str(script)], cwd=application, env=environment)
    assert second.returncode == 0, second.stderr
    release_b = _release_id(second.stdout)
    assert release_b != release_a
    assert _current(remote_base) == release_b

    rolled_back = _run(
        ["bash", str(script), "--rollback", release_a],
        cwd=application,
        env=environment,
    )
    assert rolled_back.returncode == 0, rolled_back.stderr
    assert _current(remote_base) == release_a
    operations = marker.read_text(encoding="utf-8").splitlines()
    assert sum(operation == "rsync" for operation in operations) == 2
    assert any(" prepare " in operation for operation in operations)
    assert any(" install " in operation for operation in operations)
    assert any(" activate " in operation for operation in operations)
    assert any(" rollback " in operation for operation in operations)
    assert "sudo" not in operations
    assert "systemctl" not in operations


def test_stubbed_upload_failure_leaves_current_untouched(
    tmp_path: Path,
) -> None:
    application, script = _sync_fixture(tmp_path)
    remote_base = tmp_path / "remote" / "etrade_python_client"
    environment, _ = _sync_environment(tmp_path, remote_base)
    first = _run(["bash", str(script)], cwd=application, env=environment)
    assert first.returncode == 0, first.stderr
    release_a = _release_id(first.stdout)

    (application / "safe.py").write_text("release = 2\n", encoding="utf-8")
    _git(application.parent, "add", "etrade_python_client/safe.py")
    _git(application.parent, "commit", "--quiet", "-m", "release two")
    failed_environment, marker = _sync_environment(
        tmp_path,
        remote_base,
        fail_upload=True,
    )
    failed = _run(
        ["bash", str(script)],
        cwd=application,
        env=failed_environment,
    )

    assert failed.returncode == 91
    assert _current(remote_base) == release_a
    assert marker.read_text(encoding="utf-8").splitlines()[-1] == (
        "rsync-failed"
    )
