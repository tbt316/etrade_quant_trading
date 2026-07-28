from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from rauth import OAuth1Session

from live_trading.execution_runtime import build_manual_open_service
from live_trading.manual_open import ManualOpenService
from live_trading.runtime_config import load_runtime_config
from live_trading.runtime_safety import RuntimeSafetyBoundary


def _runtime_config(tmp_path: Path):
    document = {
        "schema_version": 2,
        "mode": "sandbox",
        "runtime_root": "runtime",
        "strategy": {
            "enabled": True,
            "strategy_id": "manual-credit-spread.v1",
            "symbols": ["SPY", "SPX"],
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
        "execution": {
            "selected_account_id_key": "account-key",
            "account_allowlist": [
                {
                    "account_id": "12345678",
                    "account_id_key": "account-key",
                    "institution_type": "BROKERAGE",
                }
            ],
            "broker_mutations_enabled": True,
        },
        "risk": {
            "max_order_contracts": 4,
            "max_order_loss_cents": 200_000,
            "max_account_open_risk_cents": 500_000,
            "max_daily_loss_cents": 300_000,
            "max_quote_age_seconds": 30,
        },
    }
    path = tmp_path / "runtime-config.json"
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return load_runtime_config(path)


def test_composition_reconciles_before_returning_narrow_service(
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    safety = RuntimeSafetyBoundary(
        "sandbox",
        "12345678",
        "account-key",
        "BROKERAGE",
    )
    session = OAuth1Session(
        "consumer-key",
        "consumer-secret",
        "access-token",
        "access-secret",
    )

    with (
        patch(
            "live_trading.execution_runtime.validate_runtime_directories"
        ) as validate_directories,
        patch(
            "live_trading.execution_runtime.OrderIntentLedger"
        ) as ledger_type,
        patch(
            "live_trading.execution_runtime.ETradeBrokerTransport"
        ) as transport_type,
        patch(
            "live_trading.execution_runtime.ETradeBrokerReader"
        ) as reader_type,
        patch(
            "live_trading.execution_runtime.EtradeOrderGateway"
        ) as gateway_type,
    ):
        gateway = gateway_type.return_value
        gateway.submit_opening.return_value = None
        gateway.execution_ready = True

        service = build_manual_open_service(
            session=session,
            runtime_config=config,
            runtime_safety=safety,
            proposal_secret="x" * 64,
        )

    assert type(service) is ManualOpenService
    validate_directories.assert_called_once_with(config.paths)
    ledger_type.assert_called_once()
    transport_type.assert_called_once()
    reader_type.assert_called_once()
    gateway.start.assert_called_once_with()
    assert gateway_type.call_args.kwargs["runtime_safety"] is safety
    assert (
        gateway_type.call_args.kwargs["opening_risk_budget"]
        == Decimal("5000")
    )
    assert (
        gateway_type.call_args.kwargs["daily_opening_risk_budget"]
        == Decimal("3000")
    )
    selected = transport_type.call_args.kwargs["selected_account"]
    assert selected.account_id == "12345678"
    assert selected.account_id_key == "account-key"
    assert selected.institution_type == "BROKERAGE"
    assert (
        reader_type.call_args.kwargs["selected_account"]
        == selected
    )
