from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import check_release_artifacts as release
from scripts import check_repo_hygiene as hygiene
from scripts import check_secret_content as gate


@pytest.fixture(autouse=True)
def _isolate_git_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in gate.UNSAFE_GIT_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "secret-gate@example.invalid")
    _git(repo, "config", "user.name", "Secret Gate Test")
    return repo


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _credential_value() -> str:
    return "".join(("R7", "-", "Actual", "-", "Value", "-", "839201"))


def _credential_assignment() -> str:
    return "CoNsUmEr_SeCrEt = " + repr(_credential_value()) + "\n"


def _rule_ids(report: gate.SecretScanReport) -> set[str]:
    return {finding.rule_id for finding in report.findings}


def _zip_member(
    archive: zipfile.ZipFile,
    name: str,
    content: bytes,
) -> None:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (0o100644 << 16)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content)


def _write_zip(path: Path, members: tuple[tuple[str, bytes], ...]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            _zip_member(archive, name, content)


def _write_tar(path: Path, members: tuple[tuple[str, bytes], ...]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_index_scan_uses_staged_blob_not_mutable_worktree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "src/config.py", "artifact_sha256 = '" + ("a" * 64) + "'\n")
    _git(repo, "add", "src/config.py")

    _write(repo, "src/config.py", _credential_assignment())
    assert gate.scan_git_index(repo).findings == ()

    _git(repo, "add", "src/config.py")
    _write(repo, "src/config.py", "safe = True\n")
    report = gate.scan_git_index(repo)

    assert _rule_ids(report) == {"LITERAL_CREDENTIAL_ASSIGNMENT"}


def test_tree_scan_is_bound_to_exact_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "src/config.py", "safe = True\n")
    _git(repo, "add", "src/config.py")
    _git(repo, "commit", "--quiet", "-m", "safe")
    safe_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write(repo, "src/config.py", _credential_assignment())
    _git(repo, "add", "src/config.py")

    assert gate.scan_git_tree(repo, safe_commit).findings == ()
    assert _rule_ids(gate.scan_git_index(repo)) == {
        "LITERAL_CREDENTIAL_ASSIGNMENT"
    }


def test_tree_prefix_scans_the_literal_subtree(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "application/config.py", _credential_assignment())
    _write(repo, "outside.py", "safe = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "prefixed fixture")
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    report = gate.scan_git_tree(
        repo,
        commit,
        prefix="application",
    )

    assert report.scanned_items == 1
    assert _rule_ids(report) == {"LITERAL_CREDENTIAL_ASSIGNMENT"}


@pytest.mark.parametrize(
    ("content_factory", "expected_rule"),
    [
        (
            lambda: (
                "-----BeGiN "
                + "OpEnSsH PrIvAtE KeY-----\n"
                + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n"
                + "-----EnD OpEnSsH PrIvAtE KeY-----\n"
            ).encode(),
            "PRIVATE_KEY_PEM",
        ),
        (
            lambda: ("gh" + "p_" + ("A1" * 18)).encode(),
            "TOKEN_GITHUB_CLASSIC",
        ),
        (
            lambda: ("AK" + "IA" + ("A1" * 8)).encode(),
            "TOKEN_AWS_ACCESS_KEY_ID",
        ),
        (
            lambda: _credential_assignment().encode(),
            "LITERAL_CREDENTIAL_ASSIGNMENT",
        ),
        (
            lambda: (
                "https://operator:"
                + _credential_value()
                + "@example.invalid/status"
            ).encode(),
            "URL_USERINFO_CREDENTIAL",
        ),
        (
            lambda: (
                "https://example.invalid/status?Api_Key="
                + _credential_value()
            ).encode(),
            "URL_QUERY_CREDENTIAL",
        ),
        (
            lambda: ("Authorization: Bearer " + ("Ab9_" * 8)).encode(),
            "TOKEN_BEARER_LITERAL",
        ),
    ],
)
def test_credible_secret_rules_are_selective(
    content_factory,
    expected_rule: str,
) -> None:
    findings = gate.scan_bytes("src/safe_name.py", content_factory())

    assert expected_rule in {finding.rule_id for finding in findings}


def test_explicit_placeholders_expressions_and_hashes_pass() -> None:
    normal_digest = "d" * 64
    content = "\n".join(
        (
            "api_key = os.getenv('PROVIDER_API_KEY')",
            "access_token = config.access_token",
            "pass" + "word: str",
            "consumer_secret = 'TEST-PLACEHOLDER'",
            "dashboard_password = '${DASHBOARD_PASSWORD}'",
            "artifact_sha256 = '" + normal_digest + "'",
            "tree_object_id = '" + ("a" * 40) + "'",
            "url = 'https://example.invalid/?token={TOKEN}'",
            "-----BEGIN " + "PRIVATE KEY-----",
            "EXPLICIT-PLACEHOLDER",
            "-----END " + "PRIVATE KEY-----",
        )
    ).encode()

    assert gate.scan_bytes("src/settings.py", content) == ()


def test_delimited_test_word_is_not_a_global_placeholder_exemption() -> None:
    risky_value = "alpha-test-" + ("A9" * 10)
    risky_assignment = "api_key = " + repr(risky_value)
    explicit_placeholder = "api_key = " + repr(
        "TEST-PLACEHOLDER-" + ("A9" * 10)
    )

    assert {
        finding.rule_id
        for finding in gate.scan_bytes(
            "src/config.py",
            risky_assignment.encode(),
        )
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}
    assert gate.scan_bytes(
        "src/config.py",
        explicit_placeholder.encode(),
    ) == ()


@pytest.mark.parametrize("embedded_word", ["fixture", "mock", "synthetic"])
def test_placeholder_like_word_does_not_exempt_credential(
    embedded_word: str,
) -> None:
    value = "actual-" + embedded_word + "-" + ("A9" * 10)
    assignment = "api_key = " + repr(value)

    assert {
        finding.rule_id
        for finding in gate.scan_bytes(
            "src/config.py",
            assignment.encode(),
        )
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}


def test_placeholder_comment_does_not_exempt_real_looking_pem() -> None:
    begin = "-----BEGIN " + "PRIVATE KEY-----"
    end = "-----END " + "PRIVATE KEY-----"
    content = "\n".join(
        (
            begin,
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj",
            "# explicit placeholder comment",
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
            end,
        )
    ).encode()

    assert {
        finding.rule_id
        for finding in gate.scan_bytes("src/key.txt", content)
    } == {"PRIVATE_KEY_PEM"}


def test_reviewed_fixture_exception_binds_exact_path_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _credential_value()
    assignment = ("api_key = " + repr(value)).encode()
    reviewed_path = "tests/reviewed_fixture.py"
    monkeypatch.setattr(
        gate,
        "_REVIEWED_PLACEHOLDER_SHA256",
        frozenset(
            {
                (
                    reviewed_path,
                    hashlib.sha256(value.encode()).hexdigest(),
                )
            }
        ),
    )

    assert gate.scan_bytes(reviewed_path, assignment) == ()
    assert {
        finding.rule_id
        for finding in gate.scan_bytes("tests/other_fixture.py", assignment)
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}


def test_inline_json_and_python_quoted_literals_are_detected() -> None:
    value = _credential_value()
    minified_json = (
        "{"
        + '"consumer_secret":"'
        + value
        + '"}'
    ).encode()
    inline_python = (
        "Client("
        + "api_key="
        + repr(value)
        + ")"
    ).encode()
    escaped_json = (
        "{"
        + '"api_key":"Prefix\\\\\\"'
        + value
        + '"}'
    ).encode()

    assert {
        finding.rule_id
        for finding in gate.scan_bytes("src/minified.json", minified_json)
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}
    assert {
        finding.rule_id
        for finding in gate.scan_bytes("src/client.py", inline_python)
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}
    assert {
        finding.rule_id
        for finding in gate.scan_bytes("src/escaped.json", escaped_json)
    } == {"LITERAL_CREDENTIAL_ASSIGNMENT"}


def test_inline_expressions_and_explicit_placeholders_pass() -> None:
    content = (
        "{"
        + '"consumer_secret":os.getenv("CONSUMER_SECRET"),'
        + '"api_key":"EXPLICIT-PLACEHOLDER"'
        + "}\nClient("
        + "access_token=config.access_token"
        + ")"
    ).encode()

    assert gate.scan_bytes("src/config.py", content) == ()


def test_cli_never_prints_secret_or_path_without_review_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    relative_path = "src/innocent_name.py"
    _write(repo, relative_path, _credential_assignment())
    _git(repo, "add", relative_path)

    assert gate.main(["--start", str(repo)]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert _credential_value() not in captured.err
    assert relative_path not in captured.err
    assert "LITERAL_CREDENTIAL_ASSIGNMENT" in captured.err
    assert "path_fingerprint=" in captured.err
    assert "match_fingerprint=" in captured.err


def test_review_mode_maps_fingerprint_to_path_but_not_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    relative_path = "src/review_target.py"
    _write(repo, relative_path, _credential_assignment())
    _git(repo, "add", relative_path)

    assert gate.main(["--start", str(repo), "--review-paths"]) == 1
    captured = capsys.readouterr()

    assert f'path="{relative_path}"' in captured.err
    assert _credential_value() not in captured.err


def test_wheel_and_sdist_regular_members_are_scanned(tmp_path: Path) -> None:
    content = _credential_assignment().encode()
    wheel = tmp_path / "fixture.whl"
    sdist = tmp_path / "fixture.tar.gz"
    _write_zip(wheel, (("package/config.py", content),))
    _write_tar(sdist, (("package-1.0/package/config.py", content),))

    assert _rule_ids(gate.scan_archive(wheel)) == {
        "LITERAL_CREDENTIAL_ASSIGNMENT"
    }
    assert _rule_ids(gate.scan_archive(sdist)) == {
        "LITERAL_CREDENTIAL_ASSIGNMENT"
    }


@pytest.mark.parametrize("archive_kind", ["zip", "tar"])
def test_archive_path_traversal_fails_closed(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    if archive_kind == "zip":
        path = tmp_path / "unsafe.whl"
        _write_zip(path, (("../escape.py", b"safe = True\n"),))
    else:
        path = tmp_path / "unsafe.tar.gz"
        _write_tar(path, (("../escape.py", b"safe = True\n"),))

    assert "ARCHIVE_UNSAFE_PATH" in _rule_ids(gate.scan_archive(path))


def test_archive_nonregular_member_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "link.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("package/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        archive.addfile(info)

    assert "ARCHIVE_NONREGULAR_MEMBER" in _rule_ids(
        gate.scan_archive(path)
    )


def test_archive_member_and_count_bounds_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.whl"
    too_many = tmp_path / "too-many.whl"
    _write_zip(oversized, (("package/data.bin", b"x" * 65),))
    _write_zip(
        too_many,
        (
            ("package/a.py", b"safe = True\n"),
            ("package/b.py", b"safe = True\n"),
        ),
    )

    monkeypatch.setattr(gate, "MAX_ARCHIVE_MEMBER_BYTES", 64)
    assert "ARCHIVE_MEMBER_SIZE_LIMIT" in _rule_ids(
        gate.scan_archive(oversized)
    )
    monkeypatch.setattr(gate, "MAX_ARCHIVE_MEMBERS", 1)
    assert "ARCHIVE_MEMBER_COUNT_LIMIT" in _rule_ids(
        gate.scan_archive(too_many)
    )


def test_archive_normal_hash_does_not_trigger(tmp_path: Path) -> None:
    wheel = tmp_path / "hashes.whl"
    _write_zip(
        wheel,
        (
            (
                "package/manifest.json",
                ('{"sha256":"' + ("e" * 64) + '"}\n').encode(),
            ),
        ),
    )

    assert gate.scan_archive(wheel).findings == ()


def test_hygiene_stops_before_path_policy_when_content_gate_rejects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    relative_path = ".env"
    _write(repo, relative_path, _credential_assignment())
    _git(repo, "add", "--force", relative_path)

    assert hygiene.main(["--start", str(repo)]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "LITERAL_CREDENTIAL_ASSIGNMENT" in captured.err
    assert "SECRET_GATE_BLOCKED_REPOSITORY_HYGIENE" in captured.err
    assert "FORBIDDEN_PATH" not in captured.err
    assert "TRACKED_IGNORED" not in captured.err
    assert relative_path not in captured.err
    assert _credential_value() not in captured.err


def test_release_gate_rejects_exact_tree_before_payload_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, "safe_name.py", _credential_assignment())
    _git(repo, "add", "safe_name.py")
    _git(repo, "commit", "--quiet", "-m", "secret fixture")
    payload_inspected = False

    def _unexpected_payload_inspection(*_args, **_kwargs):
        nonlocal payload_inspected
        payload_inspected = True
        raise AssertionError("payload inspection must not run")

    monkeypatch.setattr(
        release,
        "_expected_git_payload",
        _unexpected_payload_inspection,
    )

    with pytest.raises(release.ArtifactContractError) as captured:
        release.inspect_release(tmp_path / "dist", repo, "HEAD")

    assert not payload_inspected
    assert "secret content gate rejected exact content" in str(captured.value)
    assert _credential_value() not in str(captured.value)
    assert "safe_name.py" not in str(captured.value)


def test_release_archive_gate_rejects_secret_without_printing_it(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    wheel = tmp_path / "fixture.whl"
    _write_zip(
        wheel,
        (("package/config.py", _credential_assignment().encode()),),
    )

    with pytest.raises(release.ArtifactContractError) as captured:
        release._run_secret_content_gate(repo, archives=(wheel,))

    assert "LITERAL_CREDENTIAL_ASSIGNMENT" in str(captured.value)
    assert _credential_value() not in str(captured.value)
    assert "package/config.py" not in str(captured.value)
