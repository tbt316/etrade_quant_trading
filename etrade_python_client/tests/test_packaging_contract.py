"""Offline contracts for the canonical package and dependency inputs."""

from __future__ import annotations

import re
from pathlib import Path

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


def test_polygon_client_import_does_not_require_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polygonio.poly_client import _resolve_api_key

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing Polygon API key"):
        _resolve_api_key(None, "")
    assert _resolve_api_key(" explicit ", "") == "explicit"


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
