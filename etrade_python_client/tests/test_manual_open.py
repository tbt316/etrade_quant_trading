from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from live_trading.etrade_order_gateway import (
    GatewayMutationResult,
    GatewayValidationError,
)
from live_trading.etrade_broker_reader import OpeningQuoteRead
from live_trading.manual_open import (
    ManualOpenService,
    ManualOpenUnavailable,
    ManualOpenValidationError,
    ManualSpreadPreview,
)
from live_trading.opening_risk_lineage import OpeningQuoteReceiptRef
from live_trading.order_intent_ledger import IntentRecord, OrderIntent
from live_trading.pretrade_risk import (
    ContractQuote,
    QuoteSnapshotEvidence,
)
from live_trading.runtime_config import load_runtime_config
from live_trading.runtime_safety import RuntimeSafetyBoundary


PROPOSAL_SECRET = "manual-open-test-secret-with-at-least-32-bytes"
ACCOUNT_ID = "12345678"
ACCOUNT_KEY = "opaque-account-key"
INSTITUTION_TYPE = "BROKERAGE"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 27, 17, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class RecordingGateway:
    """Small durable-gateway double with exact idempotency semantics."""

    def __init__(self) -> None:
        self.execution_ready = True
        self.commands = []
        self._requests = {}
        self.recent_records = ()
        self.recent_calls = []

    def recent_opening_intents(self, *, idempotency_scope, limit):
        self.recent_calls.append((idempotency_scope, limit))
        return self.recent_records[:limit]

    def submit_opening(self, command):
        self.commands.append(command)
        economic_request = (
            command.decision_id,
            command.payload_bytes,
            command.max_loss_amount,
            command.collateral_amount,
            command.quote_observed_at,
            command.quote_valid_until,
            command.quote_digest,
        )
        existing = self._requests.get(command.idempotency_key)
        if existing is not None:
            prior_request, prior_result = existing
            if prior_request != economic_request:
                raise GatewayValidationError(
                    "idempotency key was reused with different economics"
                )
            return GatewayMutationResult(
                intent_id=prior_result.intent_id,
                client_order_id=prior_result.client_order_id,
                state=prior_result.state,
                created=False,
                broker_order_id=prior_result.broker_order_id,
                preview_id=prior_result.preview_id,
                reason_code="IDEMPOTENT_REPLAY",
            )
        result = GatewayMutationResult(
            intent_id=f"intent-{len(self._requests) + 1}",
            client_order_id=f"client-{len(self._requests) + 1}",
            state="SUBMITTED",
            created=True,
            broker_order_id=f"broker-{len(self._requests) + 1}",
            preview_id=f"preview-{len(self._requests) + 1}",
            reason_code=None,
        )
        self._requests[command.idempotency_key] = (
            economic_request,
            result,
        )
        return result


class RecordingQuoteReader:
    """Exact two-leg quote double; candidate dashboard prices are irrelevant."""

    def __init__(
        self,
        clock: Clock,
        *,
        credit: str = "1.25",
        sell_age_seconds: int = 0,
        buy_age_seconds: int = 0,
    ) -> None:
        self.clock = clock
        self.credit = Decimal(credit)
        self.sell_age_seconds = sell_age_seconds
        self.buy_age_seconds = buy_age_seconds
        self.calls = []

    def read_opening_quotes(self, account, contracts):
        self.calls.append((account, contracts))
        credit_cents = self.credit * Decimal("100")
        assert credit_cents == credit_cents.to_integral_value()
        buy_bid_cents = 40
        buy_ask_cents = 60
        sell_total = (
            buy_bid_cents
            + buy_ask_cents
            + 2 * int(credit_cents)
        )
        sell_bid_cents = sell_total // 2
        sell_ask_cents = sell_total - sell_bid_cents
        sell_contract, buy_contract = contracts
        return OpeningQuoteRead(
            receipt=OpeningQuoteReceiptRef("3" * 64),
            snapshot=QuoteSnapshotEvidence(
                complete=True,
                snapshot_sha256="4" * 64,
                quotes=(
                    ContractQuote(
                        contract=sell_contract,
                        bid_cents=sell_bid_cents,
                        ask_cents=sell_ask_cents,
                        delta=Decimal("-0.20"),
                        open_interest=500,
                        volume=50,
                        observed_at=self.clock.now
                        - timedelta(seconds=self.sell_age_seconds),
                        source_sha256="5" * 64,
                    ),
                    ContractQuote(
                        contract=buy_contract,
                        bid_cents=buy_bid_cents,
                        ask_cents=buy_ask_cents,
                        delta=Decimal("-0.10"),
                        open_interest=500,
                        volume=50,
                        observed_at=self.clock.now
                        - timedelta(seconds=self.buy_age_seconds),
                        source_sha256="5" * 64,
                    ),
                ),
            ),
        )


def _config_document(
    *,
    runtime_root: str = "runtime",
    account_id: str = ACCOUNT_ID,
    account_key: str = ACCOUNT_KEY,
    symbols: list[str] | None = None,
    max_order_contracts: int = 5,
    max_order_loss_cents: int = 250_000,
    max_quote_age_seconds: int = 30,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "mode": "sandbox",
        "runtime_root": runtime_root,
        "strategy": {
            "enabled": True,
            "strategy_id": "manual-credit-spread.v1",
            "symbols": ["SPY", "SPX"] if symbols is None else symbols,
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
            "selected_account_id_key": account_key,
            "account_allowlist": [
                {
                    "account_id": account_id,
                    "account_id_key": account_key,
                    "institution_type": INSTITUTION_TYPE,
                }
            ],
            "broker_mutations_enabled": True,
        },
        "risk": {
            "max_order_contracts": max_order_contracts,
            "max_order_loss_cents": max_order_loss_cents,
            "max_account_open_risk_cents": 500_000,
            "max_daily_loss_cents": 250_000,
            "max_quote_age_seconds": max_quote_age_seconds,
        },
    }


def _write_config(
    directory: Path,
    document: dict[str, object],
    *,
    name: str = "runtime-config.json",
    indent: int | None = None,
) -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            document,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
            indent=indent,
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def _service(
    tmp_path: Path,
    *,
    clock: Clock,
    gateway: RecordingGateway | None = None,
    document: dict[str, object] | None = None,
    name: str = "runtime-config.json",
    indent: int | None = None,
    account_id: str = ACCOUNT_ID,
    account_key: str = ACCOUNT_KEY,
    secret: str = PROPOSAL_SECRET,
    quote_credit: str = "1.25",
    quote_reader: RecordingQuoteReader | None = None,
) -> tuple[ManualOpenService, RecordingGateway, object]:
    config = load_runtime_config(
        _write_config(
            tmp_path,
            _config_document() if document is None else document,
            name=name,
            indent=indent,
        )
    )
    recording_gateway = gateway or RecordingGateway()
    recording_quote_reader = quote_reader or RecordingQuoteReader(
        clock, credit=quote_credit
    )
    service = ManualOpenService(
        gateway=recording_gateway,
        quote_reader=recording_quote_reader,
        runtime_config=config,
        runtime_safety=RuntimeSafetyBoundary(
            "sandbox",
            account_id,
            account_key,
            INSTITUTION_TYPE,
        ),
        proposal_secret=secret,
        clock=clock,
    )
    return service, recording_gateway, config


def _preview(
    clock: Clock,
    *,
    ticker: str = "SPY",
    broker_symbol: str = "SPY",
    side: str = "PUT",
    sell_strike: str = "505",
    buy_strike: str = "500",
    sell_osi_key: str | None = None,
    buy_osi_key: str | None = None,
    limit_credit: str = "1.25",
) -> ManualSpreadPreview:
    expiration = date(2026, 8, 21)

    def osi_key(strike: str) -> str:
        scaled_strike = Decimal(strike) * Decimal("1000")
        assert scaled_strike == scaled_strike.to_integral_value()
        return (
            f"{broker_symbol.ljust(6, '-')}"
            f"{expiration:%y%m%d}{side[0]}"
            f"{int(scaled_strike):08d}"
        )

    return ManualSpreadPreview(
        ticker=ticker,
        broker_symbol=broker_symbol,
        side=side,
        expiration=expiration,
        sell_strike=Decimal(sell_strike),
        buy_strike=Decimal(buy_strike),
        sell_osi_key=(
            osi_key(sell_strike)
            if sell_osi_key is None
            else sell_osi_key
        ),
        buy_osi_key=(
            osi_key(buy_strike)
            if buy_osi_key is None
            else buy_osi_key
        ),
        limit_credit=Decimal(limit_credit),
        observed_at=clock.now,
    )


def _decode_token(token: str) -> tuple[dict[str, object], str]:
    payload_part, signature_part = token.split(".")
    padding = "=" * (-len(payload_part) % 4)
    payload = base64.urlsafe_b64decode(payload_part + padding)
    return json.loads(payload), signature_part


def _signed_token(material: dict[str, object], secret: str) -> str:
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    signature = hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).digest()
    return ".".join(
        (
            base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii"),
            base64.urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii"),
        )
    )


def _intent_record(
    command,
    *,
    state: str,
    proposal_id: str | None = None,
    intent_id: str = "intent-status-1",
    account_id: str = ACCOUNT_ID,
    environment: str = "sandbox",
    strategy_id: str | None = None,
    broker_order_id: str | None = None,
    created_at: datetime,
    updated_at: datetime,
) -> IntentRecord:
    envelope = OrderIntent.build(
        account_id=account_id,
        environment=environment,
        strategy_id=strategy_id or command.strategy_id,
        decision_id=command.decision_id,
        idempotency_scope=command.idempotency_scope,
        idempotency_key=proposal_id or command.idempotency_key,
        intent_kind="OPENING",
        order_payload=json.loads(command.payload_bytes),
    )
    return IntentRecord(
        intent_id=intent_id,
        envelope=envelope,
        client_order_id="client-order-id-must-remain-redacted",
        state=state,
        broker_order_id=broker_order_id,
        submission_fence=0,
        submission_lease_owner=None,
        submission_lease_expires_at=None,
        pending_operation=None,
        pending_fence=None,
        last_reconciled_run=None,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_signed_proposal_issuance_is_bounded_and_redacted(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, _gateway, _config = _service(tmp_path, clock=clock)

    proposal = service.issue_proposal(_preview(clock))
    material, _signature = _decode_token(proposal.proposal_token)

    assert service.account_id == "12345678"
    assert service.environment == "sandbox"
    assert service.runtime_config_sha256 == _config.source_sha256
    assert uuid.UUID(proposal.proposal_id).version == 4
    assert str(uuid.UUID(proposal.proposal_id)) == proposal.proposal_id
    assert proposal.expires_at == clock.now + timedelta(seconds=30)
    assert proposal.max_quantity == 5
    assert material["proposal_id"] == proposal.proposal_id
    assert material["account_id"] == "12345678"
    assert material["ticker"] == "SPY"
    assert material["broker_symbol"] == "SPY"
    assert material["side"] == "PUT"
    assert material["sell_osi_key"] == "SPY---260821P00505000"
    assert material["buy_osi_key"] == "SPY---260821P00500000"
    assert material["quote_receipt_sha256"] == "3" * 64
    assert material["quote_snapshot_sha256"] == "4" * 64
    assert material["sell_bid_cents"] == 175
    assert material["sell_ask_cents"] == 175
    assert material["buy_bid_cents"] == 40
    assert material["buy_ask_cents"] == 60
    assert proposal.sell_osi_key == material["sell_osi_key"]
    assert proposal.buy_osi_key == material["buy_osi_key"]
    assert material["observed_at"] == clock.now.isoformat()
    assert material["expires_at"] == proposal.expires_at.isoformat()
    assert proposal.proposal_token not in repr(proposal)
    assert proposal.dashboard_payload() == {
        "proposal_token": proposal.proposal_token,
        "proposal_id": proposal.proposal_id,
        "proposal_expires_at": proposal.expires_at.isoformat(),
        "execution_account_id": "12345678",
        "execution_environment": "sandbox",
        "runtime_config_sha256": _config.source_sha256,
        "ticker": "SPY",
        "broker_symbol": "SPY",
        "side": "PUT",
        "expiration": "2026-08-21",
        "sell_strike": "505",
        "buy_strike": "500",
        "sell_osi_key": "SPY---260821P00505000",
        "buy_osi_key": "SPY---260821P00500000",
        "premium": "1.25",
        "limit_credit": "1.25",
        "quote_receipt_sha256": "3" * 64,
        "quote_snapshot_sha256": "4" * 64,
        "sell_quote_observed_at": clock.now.isoformat(),
        "buy_quote_observed_at": clock.now.isoformat(),
        "quote_observed_at": clock.now.isoformat(),
        "max_quantity": 5,
        "execution_enabled": True,
    }


def test_candidate_price_cannot_override_exact_broker_quote(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(
        tmp_path, clock=clock, quote_credit="1.25"
    )

    proposal = service.issue_proposal(
        _preview(clock, limit_credit="4.99")
    )
    material, _signature = _decode_token(proposal.proposal_token)
    submission = service.submit(
        proposal_token=proposal.proposal_token,
        quantity=1,
        request_id=str(uuid.uuid4()),
    )

    assert proposal.limit_credit == Decimal("1.25")
    assert proposal.dashboard_payload()["premium"] == "1.25"
    assert proposal.dashboard_payload()["limit_credit"] == "1.25"
    assert material["limit_credit"] == "1.25"
    assert json.loads(gateway.commands[0].payload_bytes)["limitPrice"] == 1.25
    assert gateway.commands[0].quote_digest == "3" * 64
    assert submission.state == "SUBMITTED"


@pytest.mark.parametrize(
    ("sell_age_seconds", "buy_age_seconds"),
    ((31, 31), (6, 0)),
)
def test_stale_or_skewed_broker_quotes_fail_before_gateway(
    tmp_path: Path,
    sell_age_seconds: int,
    buy_age_seconds: int,
) -> None:
    clock = Clock()
    quote_reader = RecordingQuoteReader(
        clock,
        sell_age_seconds=sell_age_seconds,
        buy_age_seconds=buy_age_seconds,
    )
    service, gateway, _config = _service(
        tmp_path,
        clock=clock,
        quote_reader=quote_reader,
    )

    with pytest.raises(
        ManualOpenValidationError,
        match="stale, future, or time-skewed",
    ):
        service.issue_proposal(_preview(clock))

    assert gateway.commands == []


def test_closed_regular_session_disables_execution_before_quote_read(
    tmp_path: Path,
) -> None:
    clock = Clock()
    clock.now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    quote_reader = RecordingQuoteReader(clock)
    service, gateway, _config = _service(
        tmp_path,
        clock=clock,
        quote_reader=quote_reader,
    )

    assert service.execution_enabled is False
    with pytest.raises(
        ManualOpenUnavailable, match="open NYSE regular session"
    ):
        service.issue_proposal(_preview(clock))

    assert quote_reader.calls == []
    assert gateway.commands == []


def test_proposal_expiry_is_capped_before_nyse_close(
    tmp_path: Path,
) -> None:
    clock = Clock()
    clock.now = datetime(
        2026, 7, 27, 19, 59, 40, tzinfo=timezone.utc
    )
    service, gateway, _config = _service(tmp_path, clock=clock)

    proposal = service.issue_proposal(_preview(clock))

    assert proposal.expires_at == datetime(
        2026, 7, 27, 19, 59, 45, tzinfo=timezone.utc
    )
    clock.now = datetime(
        2026, 7, 27, 19, 59, 46, tzinfo=timezone.utc
    )
    with pytest.raises(
        ManualOpenValidationError, match="proposal is stale"
    ):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )

    assert gateway.commands == []


def test_session_close_buffer_blocks_quote_read(
    tmp_path: Path,
) -> None:
    clock = Clock()
    clock.now = datetime(
        2026, 7, 27, 19, 59, 45, tzinfo=timezone.utc
    )
    quote_reader = RecordingQuoteReader(clock)
    service, gateway, _config = _service(
        tmp_path,
        clock=clock,
        quote_reader=quote_reader,
    )

    with pytest.raises(
        ManualOpenUnavailable, match="too close to the NYSE session close"
    ):
        service.issue_proposal(_preview(clock))

    assert quote_reader.calls == []
    assert gateway.commands == []


def test_signed_quote_economics_must_remain_internally_consistent(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    material, _signature = _decode_token(proposal.proposal_token)
    material["buy_bid_cents"] += 2

    with pytest.raises(
        ManualOpenValidationError,
        match="exact two-leg quote midpoint",
    ):
        service.submit(
            proposal_token=_signed_token(material, PROPOSAL_SECRET),
            quantity=1,
            request_id=str(uuid.uuid4()),
        )

    assert gateway.commands == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"sell_osi_key": "SPY1--260821P00505000"},
            "standard unadjusted SPY contract",
        ),
        (
            {"sell_osi_key": "SPY---260821P0050500"},
            "OSI identity is malformed",
        ),
        (
            {"buy_osi_key": "SPY---260821C00500000"},
            "does not match preview economics",
        ),
        (
            {"buy_osi_key": "SPY---260821P00499000"},
            "does not match preview economics",
        ),
    ),
)
def test_adjusted_malformed_or_mismatched_osi_identity_is_rejected(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)

    with pytest.raises(ManualOpenValidationError, match=message):
        service.issue_proposal(_preview(clock, **overrides))

    assert gateway.commands == []


def test_proposal_payload_tampering_and_wrong_signing_key_are_rejected(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    material, signature = _decode_token(proposal.proposal_token)
    material["limit_credit"] = "1.5"
    encoded = base64.urlsafe_b64encode(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).rstrip(b"=").decode("ascii")

    with pytest.raises(
        ManualOpenValidationError, match="signature is invalid"
    ):
        service.submit(
            proposal_token=f"{encoded}.{signature}",
            quantity=1,
            request_id=str(uuid.uuid4()),
        )

    wrong_key_service, _other_gateway, _config = _service(
        tmp_path,
        clock=clock,
        name="wrong-key.json",
        secret="different-manual-open-secret-with-at-least-32-bytes",
    )
    with pytest.raises(
        ManualOpenValidationError, match="signature is invalid"
    ):
        wrong_key_service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert gateway.commands == []


def test_proposal_expiry_is_enforced_before_gateway_submission(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    clock.now = proposal.expires_at

    with pytest.raises(ManualOpenValidationError, match="proposal is stale"):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert gateway.commands == []


def test_expired_production_arm_blocks_submission_before_gateway(
    tmp_path: Path,
) -> None:
    clock = Clock()
    document = _config_document()
    document["mode"] = "live"
    config = load_runtime_config(
        _write_config(tmp_path, document, name="live-config.json")
    )
    gateway = RecordingGateway()
    service = ManualOpenService(
        gateway=gateway,
        quote_reader=RecordingQuoteReader(clock),
        runtime_config=config,
        runtime_safety=RuntimeSafetyBoundary(
            "production",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            arm_issued_at=clock.now - timedelta(seconds=1),
            arm_expires_at=clock.now + timedelta(seconds=5),
        ),
        proposal_secret=PROPOSAL_SECRET,
        clock=clock,
    )
    proposal = service.issue_proposal(_preview(clock))
    assert service.execution_enabled is True
    clock.now += timedelta(seconds=5)
    assert service.execution_enabled is False

    with pytest.raises(ManualOpenUnavailable, match="arm is not current"):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert gateway.commands == []


def test_execution_enabled_requires_started_gateway(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)

    assert service.execution_enabled is True
    gateway.execution_ready = False
    assert service.execution_enabled is False


def test_unreadable_durable_history_blocks_proposal_and_submission(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))

    def unreadable_history(**_kwargs):
        raise RuntimeError("ledger integrity failure")

    gateway.recent_opening_intents = unreadable_history

    assert service.execution_enabled is False
    with pytest.raises(
        ManualOpenUnavailable, match="status is unavailable"
    ):
        service.issue_proposal(_preview(clock))
    with pytest.raises(
        ManualOpenUnavailable, match="status is unavailable"
    ):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert gateway.commands == []


def test_proposal_is_bound_to_exact_runtime_config_bytes(
    tmp_path: Path,
) -> None:
    clock = Clock()
    document = _config_document(runtime_root="runtime-a")
    issuer, _issuer_gateway, first_config = _service(
        tmp_path,
        clock=clock,
        document=document,
        name="config-a.json",
    )
    verifier, verifier_gateway, second_config = _service(
        tmp_path,
        clock=clock,
        document=document,
        name="config-b.json",
        indent=2,
    )
    proposal = issuer.issue_proposal(_preview(clock))

    assert first_config.source_sha256 != second_config.source_sha256
    with pytest.raises(
        ManualOpenValidationError, match="bound to another runtime"
    ):
        verifier.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert verifier_gateway.commands == []


def test_proposal_is_bound_to_exact_selected_account(
    tmp_path: Path,
) -> None:
    clock = Clock()
    issuer, _issuer_gateway, _issuer_config = _service(
        tmp_path,
        clock=clock,
        document=_config_document(runtime_root="runtime-a"),
        name="account-a.json",
    )
    verifier, verifier_gateway, verifier_config = _service(
        tmp_path,
        clock=clock,
        document=_config_document(
            runtime_root="runtime-b",
            account_id="87654321",
            account_key="other-account-key",
        ),
        name="account-b.json",
        account_id="87654321",
        account_key="other-account-key",
    )
    proposal = issuer.issue_proposal(_preview(clock))
    material, _signature = _decode_token(proposal.proposal_token)
    material["runtime_config_sha256"] = verifier_config.source_sha256
    token_with_matching_config = _signed_token(material, PROPOSAL_SECRET)

    with pytest.raises(
        ManualOpenValidationError, match="bound to another runtime"
    ):
        verifier.submit(
            proposal_token=token_with_matching_config,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert verifier_gateway.commands == []


@pytest.mark.parametrize(
    (
        "ticker",
        "broker_symbol",
        "side",
        "sell_strike",
        "buy_strike",
        "credit",
        "quantity",
        "expected_max_loss",
        "expected_collateral",
    ),
    (
        (
            "SPY",
            "SPY",
            "PUT",
            "505",
            "500",
            "1.25",
            2,
            Decimal("750"),
            Decimal("750"),
        ),
        (
            "SPY",
            "SPY",
            "CALL",
            "500",
            "505",
            "0.75",
            1,
            Decimal("425"),
            Decimal("425"),
        ),
        (
            "SPX",
            "SPXW",
            "PUT",
            "5500",
            "5495",
            "1.5",
            1,
            Decimal("350"),
            Decimal("350"),
        ),
    ),
)
def test_exact_vertical_payload_and_risk_economics(
    tmp_path: Path,
    ticker: str,
    broker_symbol: str,
    side: str,
    sell_strike: str,
    buy_strike: str,
    credit: str,
    quantity: int,
    expected_max_loss: Decimal,
    expected_collateral: Decimal,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(
        tmp_path, clock=clock, quote_credit=credit
    )
    preview = _preview(
        clock,
        ticker=ticker,
        broker_symbol=broker_symbol,
        side=side,
        sell_strike=sell_strike,
        buy_strike=buy_strike,
        limit_credit=credit,
    )
    proposal = service.issue_proposal(preview)
    request_id = str(uuid.uuid4())

    submission = service.submit(
        proposal_token=proposal.proposal_token,
        quantity=quantity,
        request_id=request_id,
    )

    assert submission.request_id == request_id
    assert len(gateway.commands) == 1
    command = gateway.commands[0]
    payload = json.loads(command.payload_bytes)
    assert payload == {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": float(credit),
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": broker_symbol,
                "callPut": side,
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "quantity": quantity,
                "strikePrice": float(sell_strike),
                "orderAction": "SELL_OPEN",
            },
            {
                "symbol": broker_symbol,
                "callPut": side,
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "quantity": quantity,
                "strikePrice": float(buy_strike),
                "orderAction": "BUY_OPEN",
            },
        ],
    }
    assert command.max_loss_amount == expected_max_loss
    assert command.collateral_amount == expected_collateral
    assert command.quote_valid_until == proposal.expires_at
    assert command.idempotency_key == proposal.proposal_id


def test_quantity_and_order_loss_ceilings_fail_before_gateway(
    tmp_path: Path,
) -> None:
    clock = Clock()
    quantity_document = _config_document(
        max_order_contracts=2,
        max_order_loss_cents=250_000,
    )
    quantity_service, quantity_gateway, _config = _service(
        tmp_path,
        clock=clock,
        document=quantity_document,
        name="quantity.json",
    )
    quantity_proposal = quantity_service.issue_proposal(_preview(clock))

    with pytest.raises(
        ManualOpenValidationError, match="per-order limit"
    ):
        quantity_service.submit(
            proposal_token=quantity_proposal.proposal_token,
            quantity=3,
            request_id=str(uuid.uuid4()),
        )
    assert quantity_gateway.commands == []

    loss_document = _config_document(
        runtime_root="runtime-loss",
        max_order_loss_cents=39_999,
    )
    loss_service, loss_gateway, _config = _service(
        tmp_path,
        clock=clock,
        document=loss_document,
        name="loss.json",
        quote_credit="1",
    )
    loss_proposal = loss_service.issue_proposal(
        _preview(clock, limit_credit="1")
    )
    with pytest.raises(
        ManualOpenValidationError, match="maximum loss"
    ):
        loss_service.submit(
            proposal_token=loss_proposal.proposal_token,
            quantity=1,
            request_id=str(uuid.uuid4()),
        )
    assert loss_gateway.commands == []


@pytest.mark.parametrize(
    "request_id",
    (
        "not-a-uuid",
        str(uuid.uuid1()),
        "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        uuid.uuid4(),
        True,
    ),
)
def test_request_id_must_be_a_canonical_uuidv4(
    tmp_path: Path,
    request_id: object,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))

    with pytest.raises(ManualOpenValidationError, match="canonical UUID"):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=1,
            request_id=request_id,
        )
    assert gateway.commands == []


def test_proposal_is_single_durable_intent_across_http_request_ids(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    first_request_id = str(uuid.uuid4())
    second_request_id = str(uuid.uuid4())

    first = service.submit(
        proposal_token=proposal.proposal_token,
        quantity=2,
        request_id=first_request_id,
    )
    replay = service.submit(
        proposal_token=proposal.proposal_token,
        quantity=2,
        request_id=second_request_id,
    )

    assert first.intent_id == replay.intent_id
    assert first.created is True
    assert replay.created is False
    assert first.request_id == first_request_id
    assert replay.request_id == second_request_id
    assert [command.idempotency_key for command in gateway.commands] == [
        proposal.proposal_id,
        proposal.proposal_id,
    ]
    assert gateway.commands[0].decision_id == gateway.commands[1].decision_id
    assert gateway.commands[0].payload_bytes == gateway.commands[1].payload_bytes


def test_reusing_proposal_with_changed_quantity_fails_closed(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    service.submit(
        proposal_token=proposal.proposal_token,
        quantity=1,
        request_id=str(uuid.uuid4()),
    )

    with pytest.raises(
        GatewayValidationError, match="different economics"
    ):
        service.submit(
            proposal_token=proposal.proposal_token,
            quantity=2,
            request_id=str(uuid.uuid4()),
        )
    assert [command.idempotency_key for command in gateway.commands] == [
        proposal.proposal_id,
        proposal.proposal_id,
    ]


def test_recent_submissions_maps_exact_durable_states_and_redacts(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    service.submit(
        proposal_token=proposal.proposal_token,
        quantity=2,
        request_id=str(uuid.uuid4()),
    )
    command = gateway.commands[0]
    expected_statuses = {
        "INTENT": "DO_NOT_RETRY",
        "CLAIMED": "DO_NOT_RETRY",
        "SUBMISSION_UNKNOWN": "DO_NOT_RETRY",
        "SUBMITTED": "BROKER_ACKNOWLEDGED",
        "FILLED": "FILLED",
        "CANCELLED": "CANCELLED",
        "REJECTED": "REJECTED",
        "EXPIRED": "EXPIRED",
        "FAILED": "NOT_SENT",
    }
    records = []
    proposal_ids = []
    for index, state in enumerate(expected_statuses):
        proposal_id = str(uuid.uuid4())
        proposal_ids.append(proposal_id)
        records.append(
            _intent_record(
                command,
                state=state,
                proposal_id=proposal_id,
                intent_id=f"intent-status-{index}",
                broker_order_id=(
                    None
                    if state
                    in {
                        "INTENT",
                        "CLAIMED",
                        "SUBMISSION_UNKNOWN",
                        "FAILED",
                    }
                    else f"broker-{index}"
                ),
                created_at=clock.now - timedelta(minutes=1),
                updated_at=clock.now,
            )
        )
    gateway.recent_records = tuple(records)
    gateway.recent_calls.clear()

    statuses = service.recent_submissions(limit=25)

    assert gateway.recent_calls == [
        ("dashboard-manual-open.v1", 25)
    ]
    assert [status.proposal_id for status in statuses] == proposal_ids
    for status, durable_state in zip(statuses, expected_statuses):
        assert status.status == expected_statuses[durable_state]
        assert status.durable_state == durable_state
        payload = status.dashboard_payload()
        assert payload == {
            "proposal_id": status.proposal_id,
            "intent_id": status.intent_id,
            "status": expected_statuses[durable_state],
            "durable_state": durable_state,
            "broker_order_id": status.broker_order_id,
            "reason_code": None,
            "created_at": (
                clock.now - timedelta(minutes=1)
            ).isoformat(),
            "updated_at": clock.now.isoformat(),
            "ticker": "SPY",
            "side": "PUT",
            "expiration": "2026-08-21",
            "sell_strike": "505",
            "buy_strike": "500",
            "limit_credit": "1.25",
            "quantity": 2,
        }
        serialized = json.dumps(payload, sort_keys=True)
        assert "client-order-id-must-remain-redacted" not in serialized
        assert proposal.proposal_token not in serialized
        assert ACCOUNT_KEY not in serialized
        assert command.payload_bytes.decode("ascii") not in serialized


def test_recent_submissions_rejects_other_runtime_and_bounds_limit(
    tmp_path: Path,
) -> None:
    clock = Clock()
    service, gateway, _config = _service(tmp_path, clock=clock)
    proposal = service.issue_proposal(_preview(clock))
    service.submit(
        proposal_token=proposal.proposal_token,
        quantity=1,
        request_id=str(uuid.uuid4()),
    )
    command = gateway.commands[0]
    foreign = _intent_record(
        command,
        state="SUBMITTED",
        proposal_id=str(uuid.uuid4()),
        intent_id="foreign-intent",
        account_id="87654321",
        broker_order_id="foreign-broker-order",
        created_at=clock.now,
        updated_at=clock.now,
    )
    current = _intent_record(
        command,
        state="SUBMITTED",
        proposal_id=str(uuid.uuid4()),
        intent_id="current-intent",
        broker_order_id="current-broker-order",
        created_at=clock.now,
        updated_at=clock.now,
    )
    gateway.recent_records = (foreign, current)

    with pytest.raises(
        ManualOpenUnavailable, match="out-of-scope status"
    ):
        service.recent_submissions(limit=2)

    renamed_strategy = _intent_record(
        command,
        state="SUBMITTED",
        proposal_id=str(uuid.uuid4()),
        intent_id="renamed-strategy-intent",
        strategy_id="manual-credit-spread.v2",
        broker_order_id="renamed-strategy-broker-order",
        created_at=clock.now,
        updated_at=clock.now,
    )
    gateway.recent_records = (renamed_strategy, current)

    statuses = service.recent_submissions(limit=2)

    assert [status.intent_id for status in statuses] == [
        "renamed-strategy-intent",
        "current-intent",
    ]

    for invalid_limit in (0, 26, True):
        with pytest.raises(
            ManualOpenValidationError, match="between 1 and 25"
        ):
            service.recent_submissions(limit=invalid_limit)
