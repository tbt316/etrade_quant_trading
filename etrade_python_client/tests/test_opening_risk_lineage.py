import json
import importlib
import os
import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from live_trading.opening_risk_lineage import (
    OpeningQuoteResponseEvidence,
    OpeningRiskLineageError,
    expected_quote_route,
    parse_opening_quote_response,
)
from live_trading.order_intent_ledger import (
    OrderIntentIntegrityError,
    OrderIntentLedger,
    SCHEMA_VERSION,
)
from live_trading.pretrade_risk import (
    OpeningRiskAuthorization,
    evaluate_pretrade,
)

_gateway_tests = importlib.import_module(
    "etrade_python_client.tests.test_etrade_order_gateway"
)


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _raw_quote_response(authorization, *, omit_last=False):
    rows = []
    for item in authorization.quotes.quotes:
        contract = item.contract
        rows.append(
            {
                "dateTimeUTC": str(int(item.observed_at.timestamp())),
                "quoteStatus": "REALTIME",
                "Product": {
                    "symbol": contract.symbol,
                    "securityType": "OPTN",
                    "callPut": contract.call_put,
                    "expiryYear": str(contract.expiry.year),
                    "expiryMonth": str(contract.expiry.month),
                    "expiryDay": str(contract.expiry.day),
                    "strikePrice": str(contract.strike),
                },
                "All": {
                    "adjustedFlag": False,
                    "bid": str(item.bid_cents / 100),
                    "ask": str(item.ask_cents / 100),
                    "openInterest": str(item.open_interest),
                    "totalVolume": str(item.volume),
                },
                "Option": {
                    "osiKey": contract.osi_key,
                    "optionMultiplier": "100",
                    "optionGreeks": {"delta": str(item.delta)},
                },
            }
        )
    if omit_last:
        rows.pop()
    return _canonical_json(
        {"QuoteResponse": {"QuoteData": rows}}
    ).encode("utf-8")


def _capture(case, authorization, raw):
    contracts = tuple(
        leg.contract for leg in authorization.spread.legs
    )
    return OpeningQuoteResponseEvidence(
        account_id=authorization.spread.account_id,
        account_id_key=authorization.spread.account_id_key,
        institution_type=authorization.spread.institution_type,
        environment=authorization.spread.environment,
        origin="https://apisb.etrade.com",
        route=expected_quote_route(contracts),
        query_json=_canonical_json(
            [
                ["detailFlag", "ALL"],
                ["overrideSymbolCount", "false"],
                ["skipMiniOptionsCheck", "true"],
            ]
        ),
        authorization_sha256=authorization.authority_sha256,
        request_started_at=case.clock.now - timedelta(milliseconds=10),
        response_completed_at=case.clock.now,
        http_status=200,
        raw_response_bytes=raw,
    )


def _case_with_replayed_quote():
    case = _gateway_tests.EtradeOrderGatewayTests(
        "test_pure_risk_prerequisite_is_durable_but_not_send_authority"
    )
    case.setUp()
    envelope, original, capacity = (
        case.opening_risk_prerequisite_inputs()
    )
    capture = _capture(case, original, _raw_quote_response(original))
    quotes, _canonical = parse_opening_quote_response(capture)
    decision = evaluate_pretrade(
        original.spread,
        original.limits,
        original.authority,
        quotes,
        original.portfolio,
        original.overlays,
        evaluated_at=case.clock.now,
    )
    authorization = OpeningRiskAuthorization(
        original.spread,
        original.limits,
        original.authority,
        quotes,
        original.portfolio,
        original.overlays,
        decision,
    )
    case.ledger.create_opening_risk_prerequisite(
        envelope,
        authorization,
        capacity.decision_sha256,
        intent_id="opening-lineage-intent",
    )
    return case, authorization, replace(
        capture, authorization_sha256=authorization.authority_sha256
    )


def test_quote_bytes_and_missing_aggregates_survive_restart():
    case, _authorization, capture = _case_with_replayed_quote()
    try:
        case.clock.advance(1)
        receipt = case.ledger.record_opening_quote_response(capture)
        lineage = case.ledger.record_opening_risk_lineage(
            "opening-lineage-intent", receipt
        )
        assert lineage.status == "INDEPENDENT_EVIDENCE_PENDING"
        assert lineage.missing_evidence_reasons == (
            "BROKER_OPEN_ORDER_OPEN_RISK_NOT_REPLAYABLE",
            "BROKER_POSITION_OPEN_RISK_NOT_REPLAYABLE",
            "DAILY_PNL_NOT_REPLAYABLE",
            "DAILY_SESSION_BOUNDARY_NOT_DURABLE",
            "PORTFOLIO_DELTA_NOT_REPLAYABLE",
            "QUOTE_ACQUISITION_CHANNEL_NOT_COMPOSED",
            "QUOTE_MARKET_DATA_ENTITLEMENT_NOT_DURABLE",
            "SYMBOL_DELTA_NOT_REPLAYABLE",
        )
        with sqlite3.connect(case.path) as connection:
            row = connection.execute(
                """
                SELECT raw_response_bytes, parser_schema,
                       parser_code_sha256, parser_config_sha256,
                       response_completed_at, recorded_at
                FROM opening_quote_receipts
                """
            ).fetchone()
            assert row[0] == capture.raw_response_bytes
            assert row[1] == "etrade-opening-quote.v1"
            assert len(row[2]) == len(row[3]) == 64
            assert row[5] > row[4]
            derivation = json.loads(
                connection.execute(
                    """
                    SELECT canonical_derivation_json
                    FROM opening_risk_lineages
                    """
                ).fetchone()[0]
            )
        assert derivation["buying_power_cents"] == 200_000
        assert derivation["requested_position_conflicts"] == []
        assert derivation["requested_active_order_conflicts"] == []
        assert derivation["aggregate_completeness"] == {
            "account_open_risk": "PARTIAL_DURABLE_ONLY",
            "active_order_contract_conflicts": "COMPLETE",
            "buying_power": "COMPLETE",
            "daily_new_risk": "PARTIAL_UTC_DIAGNOSTIC_ONLY",
            "daily_order_count": "PARTIAL_UTC_DIAGNOSTIC_ONLY",
            "daily_pnl": "MISSING",
            "portfolio_delta": "MISSING",
            "position_contract_conflicts": "COMPLETE",
            "symbol_delta": "MISSING",
            "symbol_open_risk": "PARTIAL_DURABLE_ONLY",
        }
        restarted = OrderIntentLedger(
            case.path, clock=case.clock, run_id="lineage-restart"
        )
        assert restarted.get_opening_risk_lineage(
            "opening-lineage-intent"
        ) == lineage
        with pytest.raises(Exception):
            restarted.claim_submission(
                "opening-lineage-intent", "worker", lease_seconds=30
            )
    finally:
        case.tearDown()


def test_quote_page_completeness_and_freshness_fail_closed():
    case, authorization, capture = _case_with_replayed_quote()
    try:
        incomplete = replace(
            capture,
            raw_response_bytes=_raw_quote_response(
                authorization, omit_last=True
            ),
        )
        with pytest.raises(
            OrderIntentIntegrityError,
            match="failed independent replay",
        ):
            case.ledger.record_opening_quote_response(incomplete)

        future_document = json.loads(capture.raw_response_bytes)
        future_timestamp = str(
            int(capture.response_completed_at.timestamp()) + 1
        )
        for quote_row in future_document["QuoteResponse"]["QuoteData"]:
            quote_row["dateTimeUTC"] = future_timestamp
        with pytest.raises(
            OrderIntentIntegrityError,
            match="observation postdates",
        ):
            case.ledger.record_opening_quote_response(
                replace(
                    capture,
                    raw_response_bytes=_canonical_json(
                        future_document
                    ).encode("utf-8"),
                )
            )
        with sqlite3.connect(case.path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM opening_quote_receipts"
                ).fetchone()[0]
                == 0
            )

        receipt = case.ledger.record_opening_quote_response(capture)
        case.clock.advance(61)
        lineage = case.ledger.record_opening_risk_lineage(
            "opening-lineage-intent", receipt
        )
        assert (
            "PREREQUISITE_STALE_AT_LINEAGE_RECORD"
            in lineage.missing_evidence_reasons
        )
        assert lineage.status == "INDEPENDENT_EVIDENCE_PENDING"
    finally:
        case.tearDown()


def test_quote_and_lineage_tamper_are_detected_after_restart():
    case, _authorization, capture = _case_with_replayed_quote()
    try:
        receipt = case.ledger.record_opening_quote_response(capture)
        case.ledger.record_opening_risk_lineage(
            "opening-lineage-intent", receipt
        )
        with sqlite3.connect(case.path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    UPDATE opening_quote_receipts
                    SET raw_response_bytes = X'7B7D'
                    """
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM opening_risk_lineages"
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO opening_quote_receipts
                    SELECT * FROM opening_quote_receipts
                    """
                )
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO opening_risk_lineages
                    SELECT * FROM opening_risk_lineages
                    """
                )
            connection.execute(
                "DROP TRIGGER prevent_opening_quote_receipt_update"
            )
            connection.execute(
                """
                UPDATE opening_quote_receipts
                SET raw_response_bytes = ?
                """,
                (
                    bytes([capture.raw_response_bytes[0] ^ 1])
                    + capture.raw_response_bytes[1:],
                ),
            )
            connection.execute(
                """
                CREATE TRIGGER prevent_opening_quote_receipt_update
                BEFORE UPDATE ON opening_quote_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'opening quote receipts are append-only');
                END
                """
            )
            connection.commit()
        restarted = OrderIntentLedger(
            case.path, clock=case.clock, run_id="lineage-tamper"
        )
        with pytest.raises(
            OrderIntentIntegrityError,
            match="does not replay",
        ):
            restarted.get_opening_risk_lineage(
                "opening-lineage-intent"
            )
    finally:
        case.tearDown()


def test_quote_parser_rejects_route_and_provider_message_ambiguity():
    case, authorization, capture = _case_with_replayed_quote()
    try:
        with pytest.raises(OpeningRiskLineageError):
            parse_opening_quote_response(
                replace(
                    capture,
                    raw_response_bytes=_canonical_json(
                        {
                            "QuoteResponse": {
                                "Messages": {
                                    "Message": [
                                        {
                                            "code": "1",
                                            "description": "invalid",
                                            "type": "ERROR",
                                        }
                                    ]
                                },
                                "QuoteData": [],
                            }
                        }
                    ).encode("utf-8"),
                )
            )
        with pytest.raises(
            OrderIntentIntegrityError,
            match="route does not match",
        ):
            case.ledger.record_opening_quote_response(
                replace(capture, route="/v1/market/quote/SPY")
            )
    finally:
        case.tearDown()


def test_genuine_schema_15_shape_migrates_to_non_authorizing_lineage():
    case, _authorization, _capture = _case_with_replayed_quote()
    try:
        with sqlite3.connect(case.path) as connection:
            for trigger in (
                "prevent_opening_quote_receipt_update",
                "prevent_opening_quote_receipt_delete",
                "prevent_opening_quote_receipt_replace",
                "prevent_opening_risk_lineage_update",
                "prevent_opening_risk_lineage_delete",
                "prevent_opening_risk_lineage_replace",
            ):
                connection.execute(f"DROP TRIGGER {trigger}")
            connection.execute("DROP TABLE opening_risk_lineages")
            connection.execute("DROP TABLE opening_quote_receipts")
            connection.execute(
                """
                UPDATE ledger_metadata SET schema_version = 15
                WHERE singleton = 1
                """
            )
        OrderIntentLedger(
            case.path, clock=case.clock, run_id="schema-15-lineage-migration"
        )
        with sqlite3.connect(case.path) as connection:
            assert connection.execute(
                "SELECT schema_version FROM ledger_metadata"
            ).fetchone()[0] == SCHEMA_VERSION
            tables = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master WHERE type = 'table'
                    """
                )
            }
            assert {
                "opening_quote_receipts",
                "opening_risk_lineages",
            }.issubset(tables)
    finally:
        case.tearDown()
