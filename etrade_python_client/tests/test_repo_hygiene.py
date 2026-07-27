from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts import check_repo_hygiene as hygiene


@pytest.fixture(autouse=True)
def _isolate_git_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in hygiene.UNSAFE_GIT_ENVIRONMENT_VARIABLES:
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
    return repo


def _write(repo: Path, relative_path: str, content: str = "fixture\n") -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "is_forbidden"),
    [
        (".env.example", False),
        ("config.example.ini", False),
        ("docs/regime_v2_calibration_plan.json", False),
        ("live_trading/dashboard_template.html", False),
        ("live_trading/runtime_config.example.json", False),
        ("research_reports/regime_v2_calibration_artifact.json", False),
        ("yfinance/__init__.py", False),
        ("venv/bin/python", True),
        ("src/.venv/pyvenv.cfg", True),
        (".direnv/allow", True),
        ("src/.direnv/secret", True),
        ("package.egg-info/PKG-INFO", True),
        ("src/__pycache__/module.pyc", True),
        ("build/lib/package.py", True),
        ("htmlcov/index.html", True),
        (".coverage.local", True),
        ("etrade_python_client/audit_plots/chart.png", True),
        ("etrade_python_client/executed_order_tracker/orders.numbers", True),
        ("etrade_python_client/backtesting/.engine_snapshots/a/runner.py", True),
        ("polygon_api_option_data/SPY.pkl", True),
        ("yfinance/prices/SPY_prices.csv", True),
        (".etrade_oauth", True),
        (".etrade_oauth.backup", True),
        (".etrade_oauth~", True),
        ("etrade_python_client/.env.local", True),
        ("etrade_python_client/.ENVRC", True),
        ("etrade_python_client/gemini_api_key.env", True),
        (".netrc", True),
        (".pypirc", True),
        (".aws/credentials", True),
        ("ops/.aws/credentials", True),
        (".ssh/id_rsa", True),
        ("ops/.ssh/id_ed25519.pub", True),
        ("certificates/client.key", True),
        ("certificates/client.pem", True),
        ("certificates/client.p12", True),
        ("certificates/client.pfx", True),
        ("certificates/client.key.gz", True),
        ("certificates/client.pem.enc", True),
        ("certificates/client.p12.gz", True),
        ("certificates/client.pfx.copy", True),
        ("etrade_python_client/config.ini", True),
        ("etrade_python_client/config.ini.backup", True),
        ("etrade_python_client/credentials.json.old", True),
        ("etrade_python_client/etrade_session.json", True),
        ("etrade_python_client/etrade_session.json.1", True),
        ("etrade_python_client/trade_status.json", True),
        ("etrade_python_client/trade_status.jsonl", True),
        ("etrade_python_client/nudge_history.json", True),
        ("etrade_python_client/spy_vix_price_cache.json", True),
        ("etrade_python_client/production-arm.json", True),
        ("etrade_python_client/runtime_config.json", True),
        ("etrade_python_client/runtime_secrets.json", True),
        ("etrade_python_client/secrets.json", True),
        ("etrade_python_client/.DS_Store", True),
        ("backtesting/experiments_log.jsonl", True),
        ("data/options.sqlite3", True),
        ("data/orders.sqlite3.backup", True),
        ("data/ledger.db.gz", True),
        ("data/SPY.parquet", True),
        ("spy_gains_cache.json", True),
        ("cache.json.gz", True),
        ("polygonio/recursive_backtest.py.bak2", True),
        ("polygonio/recursive_backtest.py.bak.previous", True),
        ("polygonio/cache_io.py.bak_pathwrap", True),
        ("logs/service.log.gz", True),
        ("scratch/source.tmp", True),
        ("scratch/source.swp", True),
        ("scratch/source.swo", True),
        ("scratch/source.save", True),
        ("polygonio/fix_patch.patch", True),
    ],
)
def test_semantic_forbidden_policy(path: str, is_forbidden: bool) -> None:
    reason = hygiene.semantic_forbidden_reason(path)
    assert (reason is not None) is is_forbidden


def test_clean_temp_repository_passes_and_resolves_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    safe_paths = (
        ".env.example",
        "docs/regime_v2_calibration_plan.json",
        "live_trading/dashboard_template.html",
        "research_reports/regime_v2_calibration_artifact.json",
        "src/file with spaces.py",
        "yfinance/__init__.py",
    )
    for path in safe_paths:
        _write(repo, path)
    _git(repo, "add", "--", *safe_paths)

    nested = repo / "src"
    result = hygiene.scan_repository(nested)

    assert result.root == repo.resolve()
    assert result.tracked_path_count == len(safe_paths)
    assert result.diagnostics == ()
    assert hygiene.main(["--start", str(nested)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        f"repository index hygiene passed for {len(safe_paths)} "
        "tracked path(s).\n"
    )


def test_temp_repository_reports_sorted_index_violations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _init_repo(tmp_path)
    _write(repo, ".gitignore", "*.tmp\nvenv/\n")
    _write(repo, "safe.py")
    forbidden_paths = (
        "notes.tmp",
        "venv/bin/python",
        "src/.venv/pyvenv.cfg",
        "etrade_quant_trading.egg-info/PKG-INFO",
        "etrade_python_client/audit_plots/chart.png",
        "etrade_python_client/backtesting/.engine_snapshots/abc/runner.py",
        "etrade_python_client/backtesting/experiments_log.jsonl",
        "etrade_python_client/.etrade_oauth",
        "etrade_python_client/config.ini",
        "etrade_python_client/etrade_session.json",
        "etrade_python_client/etrade_session.json.1",
        "models/regime.pkl",
        "models/regime.pkl.backup",
        "cache/options.sqlite3",
        "cache/options.sqlite3.old",
        "polygonio/recursive_backtest.py.bak_dbg",
        "polygonio/fix_patch.patch",
        "yfinance/prices/SPY_prices.csv",
    )
    for path in forbidden_paths:
        _write(repo, path)
    _git(repo, "add", "--force", ".")

    result = hygiene.scan_repository(repo)
    diagnostic_paths = tuple(item.path for item in result.diagnostics)

    assert diagnostic_paths == tuple(sorted(forbidden_paths))
    assert len(diagnostic_paths) == len(set(diagnostic_paths))
    codes_by_path = {item.path: item.code for item in result.diagnostics}
    assert codes_by_path["notes.tmp"] == "TRACKED_IGNORED"
    assert codes_by_path["venv/bin/python"] == "TRACKED_IGNORED"
    assert codes_by_path["models/regime.pkl"] == "FORBIDDEN_PATH"

    assert hygiene.main(["--start", str(repo)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    expected_lines = [item.render() for item in result.diagnostics]
    expected_lines.append(
        "repository index hygiene failed with "
        f"{len(result.diagnostics)} violation(s)."
    )
    assert captured.err.splitlines() == expected_lines


def test_symbolic_link_index_mode_is_rejected(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    target = tmp_path / "outside-runtime-extension.so"
    link = repo / "runtime_extension.so"
    link.symlink_to(target)
    _git(repo, "add", "runtime_extension.so")

    result = hygiene.scan_repository(repo)

    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == "runtime_extension.so"
    assert result.diagnostics[0].code == "UNSAFE_MODE"
    assert "120000" in result.diagnostics[0].message


def test_unmerged_index_stages_are_rejected(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    object_ids = []
    for content in ("base\n", "ours\n", "theirs\n"):
        completed = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=content,
            check=True,
            capture_output=True,
            text=True,
        )
        object_ids.append(completed.stdout.strip())
    index_entries = "".join(
        f"100644 {object_id} {stage}\tconflict.py\n"
        for stage, object_id in enumerate(object_ids, start=1)
    )
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--index-info"],
        input=index_entries,
        check=True,
        capture_output=True,
        text=True,
    )

    result = hygiene.scan_repository(repo)

    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].path == "conflict.py"
    assert result.diagnostics[0].code == "UNMERGED_INDEX"
    assert "1, 2, 3" in result.diagnostics[0].message


def test_non_repository_is_a_configuration_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside_repo = tmp_path / "outside"
    outside_repo.mkdir()

    assert hygiene.main(["--start", str(outside_repo)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err.startswith("repository hygiene configuration error:")


@pytest.mark.parametrize(
    "variable",
    sorted(hygiene.UNSAFE_GIT_ENVIRONMENT_VARIABLES),
)
def test_git_repository_environment_overrides_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variable: str,
) -> None:
    repo = _init_repo(tmp_path)
    alternate = tmp_path / "alternate"
    monkeypatch.setenv(variable, str(alternate))

    assert hygiene.main(["--start", str(repo)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert variable in captured.err
    assert "unsafe Git repository override" in captured.err
