from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from live_trading.positions_artifact import (
    PositionsArtifactReader,
    PositionsArtifactSigningKey,
)
from live_trading.positions_artifact_publisher import (
    PositionsArtifactPublisher,
)

SIGNING_KEY = PositionsArtifactSigningKey.from_text(
    "test-integration-artifact-key-4Vf7q2Zw9Lm5Nx3Bc6Hd0P8R"
)
RUNTIME_BINDING = "c" * 64


def _position(*, symbol: str = "SPY") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        security_type="Option",
        quantity=-1,
        last_price=1.25,
        price_paid=2.50,
        market_value=-125.00,
        total_gain=125.00,
        call_put="CALL",
        expiration_date="2026-08-21",
        strike_price=650,
        underlying_last_price=640.25,
        osi_key="SPY---260821C00650000",
        option_multiplier=100,
        options_adjusted_flag=False,
        option_deliverables="100 shares",
        position_id="must-not-be-published",
        lots_url="https://broker.invalid/private",
    )


def _configure_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path]:
    module = importlib.import_module(
        "live_trading.etrade_cover_call_new"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    path = artifacts / "positions.html"
    monkeypatch.setattr(
        module,
        "READ_ONLY_POSITIONS_PUBLISHER",
        PositionsArtifactPublisher(
            path,
            max_source_age_seconds=300,
            signing_key=SIGNING_KEY,
            broker_environment="production",
            runtime_binding=RUNTIME_BINDING,
        ),
    )
    monkeypatch.setattr(
        module,
        "READ_ONLY_POSITIONS_BROKER_ENVIRONMENT",
        "production",
    )
    return module, path


def test_legacy_cycle_publishes_only_sanitized_static_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, path = _configure_publisher(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)

    receipt = module._publish_confirmed_positions(
        [_position()],
        observed_at=now,
    )
    observed = PositionsArtifactReader(
        path,
        max_age_seconds=300,
        signing_key=SIGNING_KEY,
        expected_broker_environment="production",
        expected_runtime_binding=RUNTIME_BINDING,
    ).read(now=now)

    assert receipt.state == "committed_durable"
    assert observed.available is True
    assert path.stat().st_mode & 0o777 == 0o600
    payload = path.read_bytes()
    assert b"SPY" in payload
    assert b"must-not-be-published" not in payload
    assert b"broker.invalid" not in payload
    assert b"<script" not in payload


def test_invalid_cycle_preserves_last_confirmed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, path = _configure_publisher(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    module._publish_confirmed_positions(
        [_position()],
        observed_at=now,
    )
    confirmed = path.read_bytes()

    with pytest.raises(RuntimeError, match="position_symbol_invalid"):
        module._publish_confirmed_positions(
            [_position(symbol='SPY"><script>')],
            observed_at=now,
        )

    assert path.read_bytes() == confirmed
    assert not [
        value
        for value in os.listdir(path.parent)
        if value.endswith(".tmp")
    ]


def test_legacy_cycle_rejects_a_stale_completed_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, path = _configure_publisher(tmp_path, monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="positions_snapshot_stale",
    ):
        module._publish_confirmed_positions(
            [_position()],
            observed_at=(
                datetime.now(timezone.utc) - timedelta(seconds=301)
            ),
        )

    assert not path.exists()


def test_stable_portfolio_loader_requires_matching_consecutive_reads() -> None:
    module = importlib.import_module(
        "live_trading.etrade_cover_call_new"
    )

    class StableAccounts:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def portfolio(self, **kwargs: object) -> list[SimpleNamespace]:
            self.calls.append(kwargs)
            position = _position()
            position.last_price = 1.00 + len(self.calls)
            return [position]

    accounts = StableAccounts()

    before = datetime.now(timezone.utc)
    positions, source_as_of = module._load_stable_positions(accounts)
    after = datetime.now(timezone.utc)

    assert len(positions) == 1
    assert before <= source_as_of <= after
    assert accounts.calls == [
        {
            "print_enable": False,
            "minimal": True,
            "require_success": True,
        },
        {
            "print_enable": False,
            "require_success": True,
        },
    ]


def test_unstable_portfolio_loader_preserves_publication_boundary() -> None:
    module = importlib.import_module(
        "live_trading.etrade_cover_call_new"
    )

    class MovingAccounts:
        def __init__(self) -> None:
            self.calls = 0

        def portfolio(self, **_kwargs: object) -> list[SimpleNamespace]:
            self.calls += 1
            position = _position()
            position.quantity = -self.calls
            return [position]

    with pytest.raises(
        RuntimeError,
        match="changed between confirmation reads",
    ):
        module._load_stable_positions(MovingAccounts())


def test_portfolio_loader_detects_a_contract_identity_change() -> None:
    module = importlib.import_module(
        "live_trading.etrade_cover_call_new"
    )

    class MovingContractAccounts:
        def __init__(self) -> None:
            self.calls = 0

        def portfolio(self, **_kwargs: object) -> list[SimpleNamespace]:
            self.calls += 1
            position = _position()
            if self.calls == 2:
                position.osi_key = "SPYW--260821C00650000"
            return [position]

    with pytest.raises(
        RuntimeError,
        match="changed between confirmation reads",
    ):
        module._load_stable_positions(MovingContractAccounts())
