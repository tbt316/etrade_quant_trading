"""Offline contracts for the canonical package and dependency inputs."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import os
import re
import socket
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import pytest
import tomli


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "etrade_python_client"
EXPECTED_PACKAGES = {
    "accounts",
    "ai_agents",
    "backtesting",
    "core_api",
    "data_and_research",
    "live_trading",
    "market",
    "order",
    "polygonio",
    "strategies",
}


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _exact_pins(requirements: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for requirement in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", requirement)
        assert match, f"dependency must be an exact pin: {requirement}"
        pins[_normalized_name(match.group(1))] = match.group(2)
    return pins


def _lock_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if match:
            pins[_normalized_name(match.group(1))] = match.group(2)
    return pins


def _pyproject() -> dict:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return tomli.load(handle)


def _artifact_checker():
    checker_path = SOURCE_ROOT / "scripts" / "check_release_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "_release_artifact_checker",
        checker_path,
    )
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def _sdist_normalizer():
    normalizer_path = SOURCE_ROOT / "scripts" / "normalize_sdist.py"
    spec = importlib.util.spec_from_file_location(
        "_sdist_normalizer",
        normalizer_path,
    )
    assert spec is not None and spec.loader is not None
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    return normalizer


def _offline_artifact_runner():
    runner_path = (
        SOURCE_ROOT / "scripts" / "run_offline_artifact_checks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_offline_artifact_runner",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_pyproject_owns_one_explicit_source_layout() -> None:
    project = _pyproject()
    setuptools = project["tool"]["setuptools"]

    assert setuptools["package-dir"] == {"": "etrade_python_client"}
    assert set(setuptools["packages"]) == EXPECTED_PACKAGES
    assert setuptools["include-package-data"] is False
    assert project["project"]["requires-python"] == ">=3.10,<3.11"
    assert not (REPOSITORY_ROOT / "setup.py").exists()
    assert not (REPOSITORY_ROOT / "requirements.txt").exists()
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prune tests" in manifest
    assert "prune etrade_python_client/tests" in manifest
    assert "prune etrade_python_client/scratch" in manifest

    for package in EXPECTED_PACKAGES:
        assert (SOURCE_ROOT / package / "__init__.py").is_file()
        assert not (SOURCE_ROOT / package / "_init_.py").exists()

    assert not (
        SOURCE_ROOT / "ai_agents" / "integrate_browser_use.py"
    ).exists()
    assert "from . import polygonio_config" in (
        SOURCE_ROOT / "data_and_research" / "vol_plot.py"
    ).read_text(encoding="utf-8")
    stock_trade_source = (
        SOURCE_ROOT / "core_api" / "stock_trade_class.py"
    ).read_text(encoding="utf-8")
    assert "/Users/btian/pairs_trading" not in stock_trade_source
    assert "sys.path.append" not in stock_trade_source


def test_package_data_contract_is_explicit_and_complete() -> None:
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]

    assert package_data == {
        "backtesting": ["strategies/*.yaml"],
        "live_trading": ["dashboard_template.html"],
    }
    assert (SOURCE_ROOT / "live_trading" / "dashboard_template.html").is_file()
    assert sorted(
        path.name
        for path in (SOURCE_ROOT / "backtesting" / "strategies").glob("*.yaml")
    ) == [
        "baseline_put_spread.yaml",
        "baseline_put_spread_catchup_refill.yaml",
        "put_call_credit_spread.yaml",
    ]


def test_local_packages_do_not_shadow_third_party_yfinance() -> None:
    assert not (REPOSITORY_ROOT / "yfinance" / "__init__.py").exists()
    assert not (SOURCE_ROOT / "yfinance" / "__init__.py").exists()

    import yfinance

    assert callable(yfinance.download)
    assert callable(yfinance.Ticker)
    assert "site-packages" in Path(yfinance.__file__).parts


def test_lock_files_are_hash_pinned_without_local_or_vcs_inputs() -> None:
    for name in ("build.lock", "runtime.lock", "test.lock"):
        text = (REPOSITORY_ROOT / "requirements" / name).read_text(
            encoding="utf-8"
        )
        assert "--hash=sha256:" in text
        assert not re.search(
            r"(?im)^\s*(?:-e|--editable|file:|git\+|https?://)",
            text,
        )

        requirement_lines = [
            line
            for line in text.splitlines()
            if line and not line.startswith((" ", "#"))
        ]
        assert requirement_lines
        assert all("==" in line for line in requirement_lines)


def test_direct_runtime_pins_match_both_resolved_environments() -> None:
    project = _pyproject()
    direct = _exact_pins(project["project"]["dependencies"])

    for name in ("runtime.lock", "test.lock"):
        resolved = _lock_pins(REPOSITORY_ROOT / "requirements" / name)
        assert {
            package: resolved.get(package)
            for package in direct
        } == direct, f"direct dependency drift in {name}"

    development = _exact_pins(
        project["project"]["optional-dependencies"]["dev"]
    )
    test_resolved = _lock_pins(
        REPOSITORY_ROOT / "requirements" / "test.lock"
    )
    assert {
        package: test_resolved.get(package)
        for package in development
    } == development, "development dependency drift in test.lock"

    build = _exact_pins(project["build-system"]["requires"])
    build_resolved = _lock_pins(
        REPOSITORY_ROOT / "requirements" / "build.lock"
    )
    assert {
        package: build_resolved.get(package)
        for package in build
    } == build, "build dependency drift in build.lock"


def test_ci_builds_and_tests_an_immutable_offline_artifact() -> None:
    workflow = (
        REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert (
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
        in workflow
    )
    assert (
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
        in workflow
    )
    assert "python -m build --sdist --wheel --no-isolation" in workflow
    assert workflow.count("git archive --format=tar HEAD") == 2
    assert workflow.count("umask 022") == 2
    assert "${{ runner.temp }}" not in workflow
    assert "MPLCONFIGDIR=$RUNNER_TEMP/matplotlib" in workflow
    assert "XDG_CACHE_HOME=$RUNNER_TEMP/cache" in workflow
    assert "SOURCE_DATE_EPOCH=" in workflow
    assert "scripts/check_release_artifacts.py" in workflow
    assert workflow.count("scripts/normalize_sdist.py") == 2
    assert '--repository-root "$GITHUB_WORKSPACE"' in workflow
    assert "--git-revision HEAD" in workflow
    assert "cmp dist/*.whl reproducible-dist/*.whl" in workflow
    assert "cmp dist/*.tar.gz reproducible-dist/*.tar.gz" in workflow
    assert '--strip-components=1' in workflow
    assert 'cmp dist/*.whl "$RUNNER_TEMP/sdist-dist"/*.whl' in workflow
    assert (
        "python -m pip install --no-deps --no-build-isolation dist/*.whl"
        in workflow
    )
    assert "python -m pip check" in workflow
    assert "requirements/runtime.lock" in workflow
    assert "scripts/check_installed_runtime.py" in workflow
    assert '"$RUNNER_TEMP/runtime-venv/bin/python" -m pip uninstall' in workflow
    assert "--offline-home" in workflow
    assert "--offline-temp" in workflow
    assert "sudo --preserve-env" not in workflow
    assert "ETRADE_TEST_ARTIFACT: '1'" in workflow
    assert "ETRADE_TEST_NETWORK: deny" in workflow
    assert "ETRADE_TEST_NETWORK_NAMESPACE: '1'" in workflow
    assert "unshare --net -- bash -c" in workflow
    assert "ip link set lo up" in workflow
    assert '--reuid="$SUDO_UID"' in workflow
    assert '--regid="$SUDO_GID"' in workflow
    assert "--no-new-privs" in workflow
    assert "env -u PYTHONPATH" in workflow
    assert "scripts/run_offline_artifact_checks.py" in workflow
    assert '-m "not integration"' in workflow
    assert "PYTHONPATH: ." not in workflow.split(
        "- name: Test release artifacts without network",
        maxsplit=1,
    )[1]


def test_polygon_client_import_does_not_require_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polygonio.poly_client import _resolve_api_key

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing Polygon API key"):
        _resolve_api_key(None, "")
    assert _resolve_api_key(" explicit ", "") == "explicit"


def test_legacy_pairs_surfaces_are_portable_and_fail_closed() -> None:
    from core_api.stock_trade_class import (
        BackEndAgent,
        import_ticker_from_csv,
    )
    from live_trading.runtime_safety import LegacyExecutionDisabled

    with pytest.raises(TypeError, match="csv_directory"):
        import_ticker_from_csv("2024-01-01")
    with pytest.raises(LegacyExecutionDisabled):
        BackEndAgent(
            start_date="2024-01-01",
            LiveTradeAgent_id="test",
            stock_universe="sp500",
            sector_key="all",
            use_existing_file=False,
            metric_selected="sharpe",
            dynamic_trade_setting=False,
        )


def test_http_diagnostics_redact_credentials_and_exception_text() -> None:
    from polygonio.http_safety import (
        redacted_request_url,
        safe_exception_summary,
    )

    sentinel = "polygon-secret-sentinel"
    label = redacted_request_url(
        f"https://api.polygon.io/v3/quotes?token={sentinel}",
        {
            "apiKey": sentinel,
            "ticker": "SPY",
        },
    )
    assert sentinel not in label
    assert label.count("%3Credacted%3E") == 2
    assert "ticker=SPY" in label

    summary = safe_exception_summary(RuntimeError(sentinel))
    assert summary == "RuntimeError"
    assert sentinel not in summary


def test_artifact_mode_resolves_runtime_imports_from_installed_wheel() -> None:
    if os.environ.get("ETRADE_TEST_ARTIFACT") != "1":
        pytest.skip("installed-wheel assertion is enabled only in artifact CI")

    import accounts
    import ai_agents
    import backtesting
    import core_api
    import data_and_research
    import live_trading
    import market
    import order
    import polygonio
    import strategies

    for module in (
        accounts,
        ai_agents,
        backtesting,
        core_api,
        data_and_research,
        live_trading,
        market,
        order,
        polygonio,
        strategies,
    ):
        module_path = Path(module.__file__).resolve()
        assert SOURCE_ROOT not in module_path.parents
        assert "site-packages" in module_path.parts


def test_network_guard_blocks_non_loopback_resolution() -> None:
    if os.environ.get("ETRADE_TEST_NETWORK") != "deny":
        pytest.skip("network guard assertion is enabled only in offline CI")

    with pytest.raises(RuntimeError, match="outbound network is disabled"):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match="outbound network is disabled"):
        socket.gethostbyname("example.com")
    datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(RuntimeError, match="outbound network is disabled"):
            datagram.sendto(b"blocked", ("192.0.2.1", 9))
        if hasattr(datagram, "sendmsg"):
            with pytest.raises(
                RuntimeError,
                match="outbound network is disabled",
            ):
                datagram.sendmsg(
                    [b"blocked"],
                    [],
                    0,
                    ("192.0.2.1", 9),
                )
    finally:
        datagram.close()
    assert socket.getaddrinfo("localhost", 0)
    if os.environ.get("ETRADE_TEST_NETWORK_NAMESPACE") == "1":
        assert os.geteuid() != 0
        assert {name for _index, name in socket.if_nameindex()} == {"lo"}


def test_release_artifact_checker_rejects_unsafe_paths() -> None:
    checker = _artifact_checker()

    for path in (
        "../credential",
        "/absolute/path",
        "package//module.py",
        "package/./module.py",
        "package/\nmodule.py",
        r"..\windows-credential",
    ):
        with pytest.raises(checker.ArtifactContractError):
            checker._safe_path(path)

    truncated = zipfile.ZipInfo("accounts/__init__.py")
    truncated.orig_filename = "accounts/__init__.py\0hidden"
    with pytest.raises(checker.ArtifactContractError, match="truncated"):
        checker._safe_zip_path(truncated)

    assert checker._forbidden_path(
        PurePosixPath("live_trading/tests/secret.py")
    )
    assert checker._forbidden_path(
        PurePosixPath("live_trading/session.json")
    )


def test_release_artifact_checker_uses_git_blobs_and_verifies_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _artifact_checker()
    expected_dailytrade = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "show",
            "HEAD:etrade_python_client/backtesting/polygonio_dailytrade.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    monkeypatch.setenv("GIT_DIR", "/untrusted/git-dir")
    monkeypatch.setenv("GIT_WORK_TREE", "/untrusted/worktree")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/untrusted/objects")
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/untrusted")
    _commit, payload, tracked_sdist = checker._expected_git_payload(
        REPOSITORY_ROOT,
        "HEAD",
    )

    assert "backtesting/polygonio_dailytrade.py" in payload
    assert set(tracked_sdist) == {"MANIFEST.in", "README.md", "pyproject.toml"}
    assert payload["backtesting/polygonio_dailytrade.py"] == expected_dailytrade

    data = b"reviewed payload"
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()
    ).rstrip(b"=").decode("ascii")
    record_name = "etrade_quant_trading-0.1.0.dist-info/RECORD"
    record = (
        f"accounts/__init__.py,sha256={digest},{len(data)}\n"
        f"{record_name},,\n"
    ).encode("utf-8")
    contents = {
        "accounts/__init__.py": data,
        record_name: record,
    }
    checker._verify_record(contents, record_name)
    contents["accounts/__init__.py"] = b"tampered payload"
    with pytest.raises(checker.ArtifactContractError, match="integrity"):
        checker._verify_record(contents, record_name)


def test_sdist_normalization_is_deterministic_and_rejects_links(
    tmp_path: Path,
) -> None:
    normalizer = _sdist_normalizer()

    def write_sdist(path: Path, mtime: int, *, link: bool = False) -> None:
        with tarfile.open(path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            root = tarfile.TarInfo("release-1.0")
            root.type = tarfile.DIRTYPE
            root.mode = 0o755
            root.mtime = mtime
            archive.addfile(root)
            member = tarfile.TarInfo("release-1.0/module.py")
            member.mtime = mtime
            if link:
                member.type = tarfile.SYMTYPE
                member.mode = 0o777
                member.linkname = "../secret"
                archive.addfile(member)
            else:
                payload = b"VALUE = 1\n"
                member.mode = 0o644
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    write_sdist(first, 1)
    write_sdist(second, 2)
    normalizer.normalize_sdist(first, 123456789)
    normalizer.normalize_sdist(second, 123456789)
    assert first.read_bytes() == second.read_bytes()

    malicious = tmp_path / "malicious.tar.gz"
    write_sdist(malicious, 3, link=True)
    with pytest.raises(
        normalizer.SdistNormalizationError,
        match="non-regular",
    ):
        normalizer.normalize_sdist(malicious, 123456789)


def test_offline_runtime_environment_preserves_the_venv_launcher(
    tmp_path: Path,
) -> None:
    runner = _offline_artifact_runner()
    runtime_python = tmp_path / "runtime-venv" / "bin" / "python"
    test_python = tmp_path / "test-venv" / "bin" / "python"
    environment = runner._contained_environment(
        runtime_python,
        test_python,
        tmp_path / "home",
        tmp_path / "temp",
    )

    assert environment["PATH"].split(os.pathsep)[:2] == [
        str(runtime_python.parent),
        str(test_python.parent),
    ]
    assert "PYTHONPATH" not in environment
    assert "GIT_DIR" not in environment


def test_packaged_backtest_imports_do_not_create_runtime_directories(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    if os.environ.get("ETRADE_TEST_ARTIFACT") == "1":
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import backtesting.polygonio_dailytrade; "
                "import backtesting.polygon_multi; "
                "import data_and_research.polygonio_improvequery"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "monthly_backtest_data").exists()
    assert not (tmp_path / "option_test_log").exists()
