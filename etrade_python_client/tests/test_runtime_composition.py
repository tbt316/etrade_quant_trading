from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from live_trading.runtime_composition import (
    DashboardAuth,
    MAX_SESSION_SECONDS,
    ReadOnlyRuntimeContext,
    load_read_only_runtime,
)
from live_trading.runtime_config import DashboardSecrets, RuntimeConfigError


def _document(mode: str = "paper") -> dict[str, object]:
    execution: dict[str, object] = {
        "selected_account_id_key": None,
        "account_allowlist": [],
        "broker_mutations_enabled": False,
    }
    if mode != "paper":
        execution = {
            "selected_account_id_key": "account-key-private",
            "account_allowlist": [
                {
                    "account_id": "12345678",
                    "account_id_key": "account-key-private",
                    "institution_type": "BROKERAGE",
                }
            ],
            "broker_mutations_enabled": False,
        }
    return {
        "schema_version": 1,
        "mode": mode,
        "runtime_root": "runtime",
        "strategy": {
            "enabled": False,
            "strategy_id": "disabled",
            "symbols": [],
        },
        "data": {
            "require_complete_snapshots": True,
            "max_snapshot_age_seconds": 300,
        },
        "model": {
            "enabled": False,
            "required_for_entry": False,
            "max_signal_age_seconds": 86_400,
        },
        "execution": execution,
        "risk": {
            "max_order_contracts": 0,
            "max_order_loss_cents": 0,
            "max_account_open_risk_cents": 0,
            "max_daily_loss_cents": 0,
            "max_quote_age_seconds": 30,
        },
    }


def _environment(mode: str = "paper") -> dict[str, str]:
    values = {
        "ETRADE_DASHBOARD_USER": "runtime-operator",
        "ETRADE_DASHBOARD_PASSWORD": "correct-horse-battery-staple",
        "ETRADE_DASHBOARD_PIN": "A9~strong",
        "ETRADE_DASHBOARD_SESSION_SECRET": (
            "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
        ),
    }
    if mode == "sandbox":
        values.update(
            {
                "ETRADE_SANDBOX_CONSUMER_KEY": "sandbox-consumer-key",
                "ETRADE_SANDBOX_CONSUMER_SECRET": "sandbox-consumer-secret",
                "ETRADE_USER": "sandbox-user",
                "ETRADE_PASS": "sandbox-password",
            }
        )
    elif mode in {"shadow", "live"}:
        values.update(
            {
                "ETRADE_LIVE_CONSUMER_KEY": "production-consumer-key",
                "ETRADE_LIVE_CONSUMER_SECRET": "production-consumer-secret",
                "ETRADE_USER": "production-user",
                "ETRADE_PASS": "production-password",
            }
        )
    return values


def _write_runtime(
    parent: Path,
    *,
    mode: str = "paper",
    create_directories: bool = True,
) -> Path:
    config_path = parent / "runtime-config.json"
    config_path.write_text(
        json.dumps(_document(mode)),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)
    if create_directories:
        root = parent / "runtime"
        for relative in (
            "",
            "state",
            "cache",
            "logs",
            "artifacts",
            "execution",
            "data",
            "model",
        ):
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
    return config_path


def test_load_read_only_runtime_binds_exact_paths_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    config_path = _write_runtime(private, mode="shadow")
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir(mode=0o700)
    monkeypatch.chdir(empty_cwd)

    context = load_read_only_runtime(
        config_path,
        environ=_environment("shadow"),
    )

    assert type(context) is ReadOnlyRuntimeContext
    assert context.mode == "shadow"
    assert context.broker_environment == "production"
    assert context.broker_reads_enabled is True
    assert context.broker_mutations_enabled is False
    assert context.paths.positions_artifact_file == (
        private / "runtime" / "artifacts" / "positions.html"
    )
    assert context.regime_shadow_reader.path == (
        private / "runtime" / "model" / "regime-v2-shadow.json"
    )
    assert not hasattr(context.regime_shadow_reader, "publish")
    assert not hasattr(context, "config")
    assert not hasattr(context, "selected_account")
    rendered = repr(context)
    for private_value in (
        "12345678",
        "account-key-private",
        "correct-horse-battery-staple",
        "A9~strong",
        "production-consumer-key",
    ):
        assert private_value not in rendered
    assert list(empty_cwd.iterdir()) == []


def test_dashboard_auth_sessions_are_bounded_and_tamper_evident() -> None:
    secrets = DashboardSecrets(
        username="runtime-operator",
        password="correct-horse-battery-staple",
        session_secret="test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R",
        source="environment",
    )
    auth = DashboardAuth.from_dashboard_secrets(
        secrets,
        audience="a" * 64,
    )
    now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

    session = auth.issue_session(now=now, lifetime_seconds=600)

    assert auth.verify_login(
        "runtime-operator",
        "correct-horse-battery-staple",
    )
    assert not auth.verify_login(
        "runtime-operator",
        "wrong-password",
    )
    assert auth.verify_session(session, now=now + timedelta(seconds=599))
    assert not auth.verify_session(session, now=now + timedelta(seconds=600))
    assert not auth.verify_session(session + "0", now=now)
    rotated_password = DashboardAuth.from_dashboard_secrets(
        DashboardSecrets(
            username="runtime-operator",
            password="different-correct-horse-battery-staple",
            session_secret=(
                "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
            ),
            source="environment",
        ),
        audience="a" * 64,
    )
    assert not rotated_password.verify_session(session, now=now)
    different_audience = DashboardAuth.from_dashboard_secrets(
        secrets,
        audience="b" * 64,
    )
    assert not different_audience.verify_session(session, now=now)
    assert repr(auth) == "DashboardAuth([REDACTED])"
    assert "correct-horse" not in repr(auth)

    with pytest.raises(ValueError, match="session lifetime"):
        auth.issue_session(now=now, lifetime_seconds=59)
    with pytest.raises(ValueError, match="session lifetime"):
        auth.issue_session(
            now=now,
            lifetime_seconds=MAX_SESSION_SECONDS + 1,
        )


def test_directory_validation_precedes_any_secret_resolution(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    config_path = _write_runtime(
        private,
        mode="paper",
        create_directories=False,
    )

    class ExplodingEnvironment(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:
            raise AssertionError(f"secret environment was read: {key}")

    with pytest.raises(RuntimeConfigError, match="runtime directory"):
        load_read_only_runtime(
            config_path,
            environ=ExplodingEnvironment(),
        )


def test_live_context_is_unarmed_and_read_only(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    context = load_read_only_runtime(
        _write_runtime(private, mode="live"),
        environ=_environment("live"),
    )

    assert context.mode == "live"
    assert context.broker_reads_enabled is True
    assert context.broker_mutations_enabled is False
    assert not hasattr(context, "config")
    assert not hasattr(context.settings, "execution")


def test_read_only_runtime_never_reads_or_retains_broker_secrets(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    accessed: list[str] = []

    class RecordingEnvironment(dict[str, str]):
        def get(
            self,
            key: str,
            default: str | None = None,
        ) -> str | None:
            accessed.append(key)
            if key.startswith("ETRADE_LIVE_") or key in {
                "ETRADE_USER",
                "ETRADE_PASS",
                "ETRADE_DASHBOARD_PIN",
            }:
                raise AssertionError(f"least-privilege violation: {key}")
            return super().get(key, default)

    context = load_read_only_runtime(
        _write_runtime(private, mode="live"),
        environ=RecordingEnvironment(
            {
                "ETRADE_DASHBOARD_USER": "runtime-operator",
                "ETRADE_DASHBOARD_PASSWORD": (
                    "correct-horse-battery-staple"
                ),
                "ETRADE_DASHBOARD_SESSION_SECRET": (
                    "test-session-secret-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
                ),
            }
        ),
    )

    assert context.mode == "live"
    assert accessed == [
        "ETRADE_DASHBOARD_USER",
        "ETRADE_DASHBOARD_PASSWORD",
        "ETRADE_DASHBOARD_SESSION_SECRET",
    ]
    assert not hasattr(context, "secrets")


def test_runtime_instance_audience_is_path_bound(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    first_context = load_read_only_runtime(
        _write_runtime(first),
        environ=_environment(),
    )
    second_context = load_read_only_runtime(
        _write_runtime(second),
        environ=_environment(),
    )

    assert first_context.source_sha256 == second_context.source_sha256
    assert (
        first_context.settings.instance_id
        != second_context.settings.instance_id
    )
    assert (
        first_context.session_cookie_name
        != second_context.session_cookie_name
    )
