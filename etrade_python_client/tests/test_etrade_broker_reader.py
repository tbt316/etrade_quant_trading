from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from rauth import OAuth1Session

from live_trading.etrade_broker_reader import (
    ETradeBrokerReader,
    ETradeBrokerReaderIntegrityError,
    ETradeBrokerReaderUnavailable,
    _PARSER_CODE_SHA256,
    _PARSER_CONFIG_SHA256,
    _PARSER_SCHEMA,
    _canonical_fill_summary_legs,
    _reparse_broker_read_response,
)
from live_trading.etrade_broker_transport import (
    SelectedBrokerAccount,
    _ExchangeResult,
)
from live_trading.order_intent_ledger import (
    BrokerReadResponseEvidence,
    OrderIntentLedger,
    canonical_order_payload_hash,
)
from live_trading.runtime_safety import RuntimeSafetyBoundary


ACCOUNT_ID = "842468410"
ACCOUNT_KEY = "account/key"
INSTITUTION_TYPE = "BROKERAGE"
ORIGIN = "https://api.etrade.com"
ORDER_ID = "94"


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(milliseconds=1)
        return result


class FixedClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


@dataclass(frozen=True)
class Reply:
    status: int
    raw: bytes

    @classmethod
    def json(cls, body: dict, *, status: int = 200) -> "Reply":
        return cls(
            status,
            json.dumps(
                body, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )


class ExchangeHarness:
    def __init__(self, outcomes: list[Reply]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[object, float]] = []

    def exchange(
        self,
        prepared,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
    ):
        if max_response_bytes != 2 * 1024 * 1024:
            raise AssertionError(
                "reader must pass its exact response-byte ceiling"
            )
        self.calls.append((prepared, timeout_seconds))
        if not self.outcomes:
            raise AssertionError("unexpected broker GET")
        reply = self.outcomes.pop(0)
        return _ExchangeResult(
            "RESPONSE",
            http_status=reply.status,
            raw_response=reply.raw,
        )


@dataclass
class ReaderCase:
    temporary: tempfile.TemporaryDirectory
    database: Path
    ledger: OrderIntentLedger
    reader: ETradeBrokerReader
    account: SelectedBrokerAccount
    adapter: ExchangeHarness
    patcher: object


def account(
    *,
    account_id: str = ACCOUNT_ID,
    status: str = "ACTIVE",
    mode: str = "MARGIN",
) -> dict:
    return {
        "accountId": account_id,
        "accountIdKey": ACCOUNT_KEY,
        "institutionType": INSTITUTION_TYPE,
        "accountStatus": status,
        "accountMode": mode,
        "accountType": "INDIVIDUAL",
    }


def account_list(*accounts: dict) -> Reply:
    return Reply.json(
        {
            "AccountListResponse": {
                "Accounts": {"Account": list(accounts)}
            }
        }
    )


def balance(
    buying_power: str,
    *,
    as_of_date: str | None = "1785168000000",
) -> Reply:
    response = {
        "accountId": ACCOUNT_ID,
        "institutionType": INSTITUTION_TYPE,
        "Computed": {"marginBuyingPower": buying_power},
    }
    if as_of_date is not None:
        response["asOfDate"] = as_of_date
    return Reply.json(
        {"BalanceResponse": response}
    )


def position_lot(
    position_id: str,
    *,
    lot_id: str,
    order_no: str = ORDER_ID,
    leg_no: str = "1",
    original_quantity: str = "1",
    remaining_quantity: str = "1",
    available_quantity: str = "1",
    acquired_date: str = "1785167100000",
) -> dict:
    return {
        "positionId": position_id,
        "positionLotId": lot_id,
        "orderNo": order_no,
        "legNo": leg_no,
        "originalQty": original_quantity,
        "remainingQty": remaining_quantity,
        "availableQty": available_quantity,
        "acquiredDate": acquired_date,
    }


def position(
    position_id: str,
    strike: str,
    *,
    lots: list[dict] | None = None,
) -> dict:
    if lots is None:
        lots = [
            position_lot(
                position_id,
                lot_id=str(1_000 + int(position_id)),
                leg_no="1" if strike == "620" else "2",
            )
        ]
    return {
        "positionId": position_id,
        "accountId": ACCOUNT_ID,
        "quantity": "1",
        "positionType": "LONG",
        "positionIndicator": "TYPE1",
        "osiKey": f"SPY---260821P00{strike}",
        "Product": {
            "symbol": "SPY",
            "securityType": "OPTN",
            "callPut": "PUT",
            "expiryYear": "2026",
            "expiryMonth": "8",
            "expiryDay": "21",
            "strikePrice": strike,
        },
        "PositionLot": lots,
    }


def portfolio_page(
    page_number: int,
    total_pages: int,
    positions: list[dict],
    *,
    next_page: int | None = None,
    total_field: str = "totalNoOfPages",
) -> Reply:
    portfolio = {
        "accountId": ACCOUNT_ID,
        total_field: str(total_pages),
        "Position": positions,
    }
    if next_page is not None:
        portfolio["nextPageNo"] = str(next_page)
    return Reply.json(
        {
            "PortfolioResponse": {
                "AccountPortfolio": [portfolio]
            }
        }
    )


def no_content() -> Reply:
    return Reply(204, b"")


def orders_page(
    *, marker: str | None, orders: list[dict] | None = None
) -> Reply:
    response: dict[str, object] = {"Order": orders or []}
    if marker is not None:
        response["marker"] = marker
    return Reply.json({"OrdersResponse": response})


def active_order_with_detail_statuses(*statuses: str) -> dict:
    return {
        "orderId": ORDER_ID,
        "orderType": "SPREADS",
        "OrderDetail": [
            {
                "accountId": ACCOUNT_ID,
                "status": status,
                "Instrument": [
                    option_leg(
                        "SELL_OPEN",
                        "620",
                        filled_quantity="0",
                    )
                ],
            }
            for status in statuses
        ],
    }


def option_leg(
    action: str,
    strike: str,
    *,
    filled_quantity: str,
    cancel_quantity: str = "0",
) -> dict:
    return {
        "Product": {
            "symbol": "SPY",
            "securityType": "OPTN",
            "callPut": "PUT",
            "expiryYear": "2026",
            "expiryMonth": "8",
            "expiryDay": "21",
            "strikePrice": strike,
        },
        "orderAction": action,
        "quantityType": "QUANTITY",
        "orderedQuantity": "1",
        "filledQuantity": filled_quantity,
        "cancelQuantity": cancel_quantity,
    }


def known_order(
    status: str,
    *,
    filled_quantity: str = "0",
    second_filled_quantity: str | None = None,
    cancel_quantity: str | None = None,
    second_cancel_quantity: str | None = None,
    order_id: str = ORDER_ID,
    replaces_order_id: str | None = None,
    replaced_by_order_id: str | None = None,
) -> Reply:
    second_filled = (
        filled_quantity
        if second_filled_quantity is None
        else second_filled_quantity
    )
    terminal_zero = status in {"CANCELLED", "REJECTED", "EXPIRED"}
    first_cancel = (
        ("1" if terminal_zero and filled_quantity == "0" else "0")
        if cancel_quantity is None
        else cancel_quantity
    )
    second_cancel = (
        ("1" if terminal_zero and second_filled == "0" else "0")
        if second_cancel_quantity is None
        else second_cancel_quantity
    )
    detail = {
        "accountId": ACCOUNT_ID,
        "orderNumber": order_id,
        "placedTime": "1785167000000",
        "status": status,
        "priceType": "NET_CREDIT",
        "limitPrice": "1.25",
        "orderTerm": "GOOD_FOR_DAY",
        "marketSession": "REGULAR",
        "allOrNone": False,
        "stopPrice": "0",
        "Instrument": [
            option_leg(
                "SELL_OPEN",
                "620",
                filled_quantity=filled_quantity,
                cancel_quantity=first_cancel,
            ),
            option_leg(
                "BUY_OPEN",
                "615",
                filled_quantity=second_filled,
                cancel_quantity=second_cancel,
            ),
        ],
    }
    if status == "EXECUTED":
        detail["executedTime"] = "1785167100000"
    outer_order = {
        "orderId": order_id,
        "orderType": "SPREADS",
        "OrderDetail": [detail],
    }
    if replaces_order_id is not None:
        outer_order["replacesOrderId"] = replaces_order_id
    if replaced_by_order_id is not None:
        outer_order["replacedByOrderId"] = replaced_by_order_id
    return Reply.json(
        {
            "OrdersResponse": {
                "Order": [outer_order]
            }
        }
    )


def reply_document(reply: Reply) -> dict:
    return json.loads(reply.raw)


def expected_vertical_payload() -> dict:
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": Decimal("1.25"),
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": Decimal("620"),
                "orderAction": "SELL_OPEN",
                "quantity": 1,
            },
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2026,
                "expiryMonth": 8,
                "expiryDay": 21,
                "strikePrice": Decimal("615"),
                "orderAction": "BUY_OPEN",
                "quantity": 1,
            },
        ],
    }


def one_page_scan(buying_power: str) -> list[Reply]:
    return [
        balance(buying_power),
        portfolio_page(1, 1, []),
        no_content(),
        no_content(),
        no_content(),
    ]


class ETradeBrokerReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases: list[ReaderCase] = []

    def tearDown(self) -> None:
        for case in reversed(self.cases):
            case.patcher.stop()
            case.temporary.cleanup()

    def case(
        self,
        outcomes: list[Reply],
        *,
        clock: Clock | FixedClock | None = None,
    ) -> ReaderCase:
        temporary = tempfile.TemporaryDirectory()
        os.chmod(temporary.name, 0o700)
        database = Path(temporary.name) / "runtime" / "orders.sqlite3"
        clock = clock or Clock()
        ledger = OrderIntentLedger(
            database,
            clock=clock,
            run_id="broker-reader-test",
        )
        runtime = RuntimeSafetyBoundary(
            environment="production",
            expected_account_id=ACCOUNT_ID,
            expected_account_id_key=ACCOUNT_KEY,
            expected_institution_type=INSTITUTION_TYPE,
            arm_issued_at=clock.value - timedelta(seconds=1),
            arm_expires_at=clock.value + timedelta(minutes=10),
        )
        selected = SelectedBrokerAccount(
            ACCOUNT_ID, ACCOUNT_KEY, INSTITUTION_TYPE
        )
        reader = ETradeBrokerReader(
            session=OAuth1Session(
                "consumer-key",
                "consumer-secret",
                access_token="access-token",
                access_token_secret="access-secret",
            ),
            ledger=ledger,
            runtime_safety=runtime,
            selected_account=selected,
            clock=clock,
        )
        adapter = ExchangeHarness(outcomes)
        patcher = patch(
            "live_trading.etrade_broker_reader._isolated_exchange",
            side_effect=adapter.exchange,
        )
        patcher.start()
        result = ReaderCase(
            temporary,
            database,
            ledger,
            reader,
            selected,
            adapter,
            patcher,
        )
        self.cases.append(result)
        return result

    @staticmethod
    def rows(
        case: ReaderCase, sql: str, parameters: tuple = ()
    ) -> list[sqlite3.Row]:
        with sqlite3.connect(case.database) as connection:
            connection.row_factory = sqlite3.Row
            return connection.execute(sql, parameters).fetchall()

    def manifest_result(
        self, case: ReaderCase, evidence_sha256: str
    ) -> dict:
        rows = self.rows(
            case,
            """
            SELECT canonical_result_json
            FROM broker_read_manifests
            WHERE evidence_sha256 = ?
            """,
            (evidence_sha256,),
        )
        self.assertEqual(len(rows), 1)
        return json.loads(rows[0]["canonical_result_json"])

    def test_get_is_origin_pinned_and_an_ineligible_response_is_not_retried(
        self,
    ) -> None:
        case = self.case(
            [
                account_list(account()),
                Reply.json(
                    {"Error": {"message": "temporarily unavailable"}},
                    status=503,
                ),
            ]
        )

        with self.assertRaises(ETradeBrokerReaderUnavailable):
            case.reader.query_order(case.account, ORDER_ID)

        self.assertEqual(case.adapter.outcomes, [])
        self.assertEqual(len(case.adapter.calls), 2)
        for prepared, timeout_seconds in case.adapter.calls:
            parsed = urlsplit(prepared.url)
            self.assertEqual(prepared.method, "GET")
            self.assertIsNone(prepared.body)
            self.assertEqual(
                f"{parsed.scheme}://{parsed.netloc}", ORIGIN
            )
            self.assertEqual(timeout_seconds, 15.0)
            self.assertNotIn("Cookie", prepared.headers)
        self.assertEqual(
            case.adapter.calls[1][0].url,
            f"{ORIGIN}/v1/accounts/account%2Fkey/orders/{ORDER_ID}.json",
        )
        rows = self.rows(
            case,
            """
            SELECT http_status, completeness, http_method, origin
            FROM broker_read_receipts
            ORDER BY request_started_at
            """,
        )
        self.assertEqual(
            [
                (
                    row["http_status"],
                    row["completeness"],
                    row["http_method"],
                    row["origin"],
                )
                for row in rows
            ],
            [
                (200, "COMPLETE", "GET", ORIGIN),
                (503, "INELIGIBLE", "GET", ORIGIN),
            ],
        )

    def test_account_binding_requires_one_active_margin_match(self) -> None:
        variants = (
            (
                [account(), account()],
                ETradeBrokerReaderIntegrityError,
            ),
            (
                [account(status="CLOSED")],
                ETradeBrokerReaderUnavailable,
            ),
            (
                [account(account_id="842468411")],
                ETradeBrokerReaderIntegrityError,
            ),
        )
        for accounts, expected_error in variants:
            with self.subTest(
                accounts=accounts, expected_error=expected_error
            ):
                case = self.case([account_list(*accounts)])
                with self.assertRaises(expected_error):
                    case.reader.query_order(case.account, ORDER_ID)
                self.assertEqual(len(case.adapter.calls), 1)
                rows = self.rows(
                    case,
                    """
                    SELECT read_kind, completeness, canonical_parsed_json
                    FROM broker_read_receipts
                    """,
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["read_kind"], "ACCOUNT_LIST")
                self.assertEqual(rows[0]["completeness"], "INELIGIBLE")
                self.assertEqual(rows[0]["canonical_parsed_json"], "null")

    def test_capacity_uses_declared_portfolio_pages_and_persists_lineage(
        self,
    ) -> None:
        pages = [
            portfolio_page(
                1,
                2,
                [position("12", "620")],
                next_page=2,
            ),
            portfolio_page(2, 2, [position("11", "615")]),
        ]
        scan = [
            balance("1000"),
            *pages,
            no_content(),
            no_content(),
            no_content(),
        ]
        case = self.case(
            [
                account_list(account()),
                *scan,
                *scan,
                account_list(account()),
            ]
        )

        evidence = case.reader.read_capacity(case.account)

        self.assertEqual(evidence.evidence_kind, "CAPACITY")
        self.assertEqual(case.adapter.outcomes, [])
        self.assertEqual(len(case.adapter.calls), 14)
        portfolio_calls = [
            prepared
            for prepared, _timeout in case.adapter.calls
            if urlsplit(prepared.url).path.endswith("/portfolio.json")
        ]
        self.assertEqual(len(portfolio_calls), 4)
        self.assertTrue(
            all(
                parse_qs(urlsplit(prepared.url).query)[
                    "lotsRequired"
                ]
                == ["true"]
                for prepared in portfolio_calls
            )
        )
        self.assertEqual(
            [
                parse_qs(urlsplit(prepared.url).query)["pageNumber"][0]
                for prepared in portfolio_calls
            ],
            ["1", "2", "1", "2"],
        )
        result = self.manifest_result(
            case, evidence.evidence_sha256
        )
        self.assertEqual(result["schema"], "etrade-capacity.v2")
        self.assertEqual(result["broker_buying_power"], "1000")
        self.assertEqual(
            [item["position_id"] for item in result["positions"]],
            ["11", "12"],
        )
        self.assertEqual(result["open_orders"], [])
        self.assertEqual(
            result["positions"][0]["lots"][0],
            {
                "position_id": "11",
                "position_lot_id": "1011",
                "order_no": ORDER_ID,
                "leg_no": "2",
                "original_quantity": "1",
                "remaining_quantity": "1",
                "available_quantity": "1",
                "acquired_date_epoch_ms": "1785167100000",
            },
        )
        self.assertEqual(len(result["state_sha256"]), 64)
        receipt_rows = self.rows(
            case,
            "SELECT * FROM broker_read_receipts ORDER BY request_started_at",
        )
        self.assertEqual(len(receipt_rows), 14)
        for row in receipt_rows:
            self.assertEqual(row["http_method"], "GET")
            self.assertEqual(row["origin"], ORIGIN)
            self.assertEqual(
                row["raw_byte_length"], len(row["raw_response_bytes"])
            )
            self.assertEqual(
                row["raw_response_sha256"],
                hashlib.sha256(row["raw_response_bytes"]).hexdigest(),
            )
            self.assertIn(
                row["completeness"], {"COMPLETE", "HAS_NEXT"}
            )
        member_rows = self.rows(
            case,
            """
            SELECT member_role
            FROM broker_read_manifest_members
            WHERE evidence_sha256 = ?
            ORDER BY member_ordinal
            """,
            (evidence.evidence_sha256,),
        )
        self.assertEqual(len(member_rows), 14)
        self.assertEqual(member_rows[0]["member_role"], "binding.start")
        self.assertEqual(member_rows[-1]["member_role"], "binding.end")
        self.assertIn(
            "scan_a.portfolio.0002",
            {row["member_role"] for row in member_rows},
        )
        self.assertIn(
            "scan_b.orders.INDIVIDUAL_FILLS.0000",
            {row["member_role"] for row in member_rows},
        )

    def test_order_marker_cycle_fails_closed_after_durable_pages(self) -> None:
        case = self.case(
            [
                account_list(account()),
                balance("1000"),
                no_content(),
                orders_page(marker="M1"),
                orders_page(marker="M1"),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderIntegrityError,
            "marker pagination repeated or cycled",
        ):
            case.reader.read_capacity(case.account)

        self.assertEqual(case.adapter.outcomes, [])
        self.assertEqual(len(case.adapter.calls), 5)
        second_order_url = case.adapter.calls[-1][0].url
        query = parse_qs(urlsplit(second_order_url).query)
        self.assertEqual(query["status"], ["OPEN"])
        self.assertEqual(query["marker"], ["M1"])
        order_rows = self.rows(
            case,
            """
            SELECT completeness, canonical_parsed_json
            FROM broker_read_receipts
            WHERE read_kind = 'OPEN_ORDERS_PAGE'
            ORDER BY request_started_at
            """,
        )
        self.assertEqual(
            [row["completeness"] for row in order_rows],
            ["HAS_NEXT", "HAS_NEXT"],
        )
        self.assertEqual(
            [
                json.loads(row["canonical_parsed_json"])["marker"]
                for row in order_rows
            ],
            ["M1", "M1"],
        )
        self.assertEqual(
            self.rows(case, "SELECT * FROM broker_read_manifests"), []
        )

    def test_capacity_requires_semantically_identical_complete_scans(
        self,
    ) -> None:
        case = self.case(
            [
                account_list(account()),
                *one_page_scan("1000"),
                *one_page_scan("999"),
                account_list(account()),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderUnavailable,
            "capacity state changed between complete scans",
        ):
            case.reader.read_capacity(case.account)

        self.assertEqual(case.adapter.outcomes, [])
        self.assertEqual(len(case.adapter.calls), 12)
        self.assertEqual(
            self.rows(case, "SELECT * FROM broker_read_manifests"), []
        )
        self.assertEqual(
            len(
                self.rows(
                    case, "SELECT * FROM broker_read_receipts"
                )
            ),
            12,
        )

    def test_capacity_stability_includes_exact_position_lots(self) -> None:
        first_position = position(
            "12",
            "620",
            lots=[
                position_lot(
                    "12",
                    lot_id="1012",
                    remaining_quantity="1",
                )
            ],
        )
        second_position = position(
            "12",
            "620",
            lots=[
                position_lot(
                    "12",
                    lot_id="1012",
                    remaining_quantity="0",
                )
            ],
        )
        first_scan = [
            balance("1000"),
            portfolio_page(1, 1, [first_position]),
            no_content(),
            no_content(),
            no_content(),
        ]
        second_scan = [
            balance("1000"),
            portfolio_page(1, 1, [second_position]),
            no_content(),
            no_content(),
            no_content(),
        ]
        case = self.case(
            [
                account_list(account()),
                *first_scan,
                *second_scan,
                account_list(account()),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderUnavailable,
            "capacity state changed between complete scans",
        ):
            case.reader.read_capacity(case.account)

    def test_position_lots_require_valid_parent_and_unique_ids(self) -> None:
        malformed_positions = []
        malformed_positions.append(
            position(
                "12",
                "620",
                lots=[
                    position_lot("13", lot_id="1012")
                ],
            )
        )
        malformed_positions.append(
            position(
                "12",
                "620",
                lots=[
                    position_lot("12", lot_id="1012"),
                    position_lot("12", lot_id="1012"),
                ],
            )
        )
        for malformed in malformed_positions:
            with self.subTest(position=malformed):
                case = self.case(
                    [
                        account_list(account()),
                        balance("1000"),
                        portfolio_page(1, 1, [malformed]),
                    ]
                )
                with self.assertRaises(ETradeBrokerReaderIntegrityError):
                    case.reader.read_capacity(case.account)
                receipt = self.rows(
                    case,
                    """
                    SELECT completeness, canonical_parsed_json
                    FROM broker_read_receipts
                    WHERE read_kind = 'PORTFOLIO_PAGE'
                    """,
                )
                self.assertEqual(len(receipt), 1)
                self.assertEqual(receipt[0]["completeness"], "INELIGIBLE")
                self.assertEqual(
                    receipt[0]["canonical_parsed_json"], "null"
                )

    def test_position_account_must_match_the_selected_account(self) -> None:
        cross_account = position("12", "620")
        cross_account["accountId"] = "999999999"
        case = self.case(
            [
                account_list(account()),
                balance("1000"),
                portfolio_page(1, 1, [cross_account]),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderIntegrityError,
            "position account id does not match configured account",
        ):
            case.reader.read_capacity(case.account)

        receipt = self.rows(
            case,
            """
            SELECT completeness, canonical_parsed_json
            FROM broker_read_receipts
            WHERE read_kind = 'PORTFOLIO_PAGE'
            """,
        )
        self.assertEqual(len(receipt), 1)
        self.assertEqual(receipt[0]["completeness"], "INELIGIBLE")
        self.assertEqual(receipt[0]["canonical_parsed_json"], "null")

    def test_missing_or_empty_option_lots_normalize_to_empty(self) -> None:
        missing = position("12", "620")
        missing.pop("PositionLot")
        empty = position("12", "620", lots=[])
        case = self.case([])
        for raw_position in (missing, empty):
            with self.subTest(raw_position=raw_position):
                parsed, completeness = (
                    case.reader._parse_portfolio_page(
                        200,
                        portfolio_page(1, 1, [raw_position]).raw,
                        1,
                        lots_required=True,
                    )
                )
                self.assertEqual(completeness, "COMPLETE")
                self.assertEqual(
                    parsed["positions"][0]["lots"], []
                )

    def test_unrelated_lot_order_and_leg_sentinels_are_preserved(
        self,
    ) -> None:
        missing_links = position_lot("12", lot_id="1013")
        missing_links.pop("orderNo")
        missing_links.pop("legNo")
        raw_position = position(
            "12",
            "620",
            lots=[
                position_lot(
                    "12",
                    lot_id="1012",
                    order_no="0",
                    leg_no="0",
                ),
                missing_links,
            ],
        )
        case = self.case([])

        parsed, completeness = case.reader._parse_portfolio_page(
            200,
            portfolio_page(1, 1, [raw_position]).raw,
            1,
            lots_required=True,
        )

        self.assertEqual(completeness, "COMPLETE")
        links = {
            lot["position_lot_id"]: (
                lot["order_no"],
                lot["leg_no"],
            )
            for lot in parsed["positions"][0]["lots"]
        }
        self.assertEqual(
            links, {"1012": ("0", "0"), "1013": (None, None)}
        )

    def test_short_option_lot_quantities_preserve_their_sign(self) -> None:
        short_lot = position_lot(
            "12",
            lot_id="1012",
            original_quantity="-1",
            remaining_quantity="-1",
            available_quantity="-1",
        )
        raw_position = position("12", "620", lots=[short_lot])
        raw_position["quantity"] = "-1"
        raw_position["positionType"] = "SHORT"
        case = self.case([])

        parsed, completeness = case.reader._parse_portfolio_page(
            200,
            portfolio_page(1, 1, [raw_position]).raw,
            1,
            lots_required=True,
        )

        self.assertEqual(completeness, "COMPLETE")
        normalized = parsed["positions"][0]
        self.assertEqual(normalized["quantity"], "-1")
        self.assertEqual(normalized["position_type"], "SHORT")
        self.assertEqual(
            normalized["lots"][0]["original_quantity"], "-1"
        )
        self.assertEqual(
            normalized["lots"][0]["remaining_quantity"], "-1"
        )
        self.assertEqual(
            normalized["lots"][0]["available_quantity"], "-1"
        )

    def test_position_lots_are_canonically_sorted(self) -> None:
        case = self.case([])
        raw_position = position(
            "12",
            "620",
            lots=[
                position_lot("12", lot_id="1014", leg_no="2"),
                position_lot(
                    "12",
                    lot_id="1013",
                    leg_no="1",
                    acquired_date="-2208988800000",
                ),
            ],
        )

        parsed, completeness = case.reader._parse_portfolio_page(
            200,
            portfolio_page(1, 1, [raw_position]).raw,
            1,
            lots_required=True,
        )

        self.assertEqual(completeness, "COMPLETE")
        self.assertEqual(
            [
                lot["position_lot_id"]
                for lot in parsed["positions"][0]["lots"]
            ],
            ["1013", "1014"],
        )
        self.assertEqual(
            parsed["positions"][0]["lots"][0][
                "acquired_date_epoch_ms"
            ],
            "-2208988800000",
        )

    def test_position_lot_ids_are_unique_across_the_full_scan(self) -> None:
        duplicate_lot_id = "1012"
        case = self.case(
            [
                account_list(account()),
                balance("1000"),
                portfolio_page(
                    1,
                    1,
                    [
                        position(
                            "12",
                            "620",
                            lots=[
                                position_lot(
                                    "12", lot_id=duplicate_lot_id
                                )
                            ],
                        ),
                        position(
                            "11",
                            "615",
                            lots=[
                                position_lot(
                                    "11", lot_id=duplicate_lot_id
                                )
                            ],
                        ),
                    ],
                ),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderIntegrityError,
            "duplicate position lot id",
        ):
            case.reader.read_capacity(case.account)

    def test_reparse_preserves_legacy_lots_required_false_shape(
        self,
    ) -> None:
        legacy_position = position("12", "620")
        legacy_position.pop("PositionLot")
        raw = portfolio_page(1, 1, [legacy_position]).raw
        completed_at = datetime(
            2026, 7, 27, 16, 0, tzinfo=timezone.utc
        )
        evidence = BrokerReadResponseEvidence(
            read_kind="PORTFOLIO_PAGE",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            environment="production",
            origin=ORIGIN,
            route=f"/v1/accounts/{ACCOUNT_KEY}/portfolio.json",
            query_json=json.dumps(
                [
                    ["lotsRequired", "false"],
                    ["pageNumber", "1"],
                ],
                separators=(",", ":"),
            ),
            authorization_sha256="a" * 64,
            target_broker_order_id=None,
            request_started_at=completed_at - timedelta(milliseconds=1),
            response_completed_at=completed_at,
            http_status=200,
            raw_response_bytes=raw,
            parser_schema=_PARSER_SCHEMA,
            parser_code_sha256=_PARSER_CODE_SHA256,
            parser_config_sha256=_PARSER_CONFIG_SHA256,
            canonical_parsed_json="null",
            completeness="INELIGIBLE",
        )

        parsed_json, completeness = _reparse_broker_read_response(
            evidence
        )

        self.assertEqual(completeness, "COMPLETE")
        parsed = json.loads(parsed_json)
        self.assertNotIn("lots", parsed["positions"][0])

    def test_balance_requires_a_present_and_fresh_as_of_date(self) -> None:
        missing_scan = [
            balance("1000", as_of_date=None),
            portfolio_page(1, 1, []),
            no_content(),
            no_content(),
            no_content(),
        ]
        missing = self.case(
            [
                account_list(account()),
                *missing_scan,
                *missing_scan,
                account_list(account()),
            ]
        )
        with self.assertRaises(ETradeBrokerReaderIntegrityError):
            missing.reader.read_capacity(missing.account)
        missing_receipt = self.rows(
            missing,
            """
            SELECT completeness, canonical_parsed_json
            FROM broker_read_receipts
            WHERE read_kind = 'BALANCE'
            """,
        )
        self.assertEqual(len(missing_receipt), 1)
        self.assertEqual(
            missing_receipt[0]["completeness"], "INELIGIBLE"
        )
        self.assertEqual(
            missing_receipt[0]["canonical_parsed_json"], "null"
        )

        stale_scan = [
            balance(
                "1000",
                as_of_date="1785153600000",
            ),
            portfolio_page(1, 1, []),
            no_content(),
            no_content(),
            no_content(),
        ]
        stale = self.case(
            [
                account_list(account()),
                *stale_scan,
                *stale_scan,
                account_list(account()),
            ]
        )
        with self.assertRaises(ETradeBrokerReaderUnavailable):
            stale.reader.read_capacity(stale.account)
        stale_receipt = self.rows(
            stale,
            """
            SELECT completeness, canonical_parsed_json
            FROM broker_read_receipts
            WHERE read_kind = 'BALANCE'
            """,
        )
        self.assertEqual(len(stale_receipt), 1)
        self.assertEqual(
            stale_receipt[0]["completeness"], "INELIGIBLE"
        )
        self.assertEqual(
            stale_receipt[0]["canonical_parsed_json"], "null"
        )

    def test_stable_capacity_has_unambiguous_lineage_with_a_fixed_clock(
        self,
    ) -> None:
        case = self.case(
            [
                account_list(account()),
                *one_page_scan("1000"),
                *one_page_scan("1000"),
                account_list(account()),
            ],
            clock=FixedClock(),
        )

        evidence = case.reader.read_capacity(case.account)

        self.assertEqual(evidence.evidence_kind, "CAPACITY")
        receipt_rows = self.rows(
            case,
            """
            SELECT receipt_sha256
            FROM broker_read_receipts
            ORDER BY rowid
            """,
        )
        member_rows = self.rows(
            case,
            """
            SELECT receipt_sha256
            FROM broker_read_manifest_members
            WHERE evidence_sha256 = ?
            ORDER BY member_ordinal
            """,
            (evidence.evidence_sha256,),
        )
        self.assertEqual(len(receipt_rows), 12)
        self.assertEqual(len(member_rows), 12)
        self.assertEqual(
            len({row["receipt_sha256"] for row in receipt_rows}), 12
        )
        self.assertEqual(
            len({row["receipt_sha256"] for row in member_rows}), 12
        )

    def test_active_order_first_detail_must_match_its_status_lane(
        self,
    ) -> None:
        case = self.case(
            [
                account_list(account()),
                balance("1000"),
                no_content(),
                orders_page(
                    marker=None,
                    orders=[
                        active_order_with_detail_statuses(
                            "CANCEL_REQUESTED", "OPEN"
                        )
                    ],
                ),
            ]
        )

        with self.assertRaisesRegex(
            ETradeBrokerReaderIntegrityError,
            "active order status does not match its requested lane",
        ):
            case.reader.read_capacity(case.account)

        self.assertEqual(case.adapter.outcomes, [])
        self.assertEqual(len(case.adapter.calls), 4)
        order_receipts = self.rows(
            case,
            """
            SELECT completeness, canonical_parsed_json
            FROM broker_read_receipts
            WHERE read_kind = 'OPEN_ORDERS_PAGE'
            """,
        )
        self.assertEqual(len(order_receipts), 1)
        self.assertEqual(
            order_receipts[0]["completeness"], "INELIGIBLE"
        )
        self.assertEqual(
            order_receipts[0]["canonical_parsed_json"], "null"
        )
        self.assertEqual(
            self.rows(case, "SELECT * FROM broker_read_manifests"), []
        )

    def test_active_order_quantities_never_default_missing_fields(
        self,
    ) -> None:
        for missing_field in ("filledQuantity", "cancelQuantity"):
            with self.subTest(missing_field=missing_field):
                active = active_order_with_detail_statuses("OPEN")
                del active["OrderDetail"][0]["Instrument"][0][
                    missing_field
                ]
                case = self.case([])
                with self.assertRaises(
                    ETradeBrokerReaderIntegrityError
                ):
                    case.reader._parse_orders_page(
                        200,
                        orders_page(
                            marker=None, orders=[active]
                        ).raw,
                        "OPEN",
                        True,
                    )

    def test_order_error_message_cannot_become_complete_empty_capacity(
        self,
    ) -> None:
        for message_key in ("Messages", "messages"):
            with self.subTest(message_key=message_key):
                case = self.case(
                    [
                        account_list(account()),
                        balance("1000"),
                        no_content(),
                        Reply.json(
                            {
                                "OrdersResponse": {
                                    "Order": [],
                                    message_key: {
                                        "Message": [
                                            {
                                                "type": "ERROR",
                                                "code": "100",
                                                "description": "unavailable",
                                            }
                                        ]
                                    },
                                }
                            }
                        ),
                    ]
                )

                with self.assertRaisesRegex(
                    ETradeBrokerReaderIntegrityError,
                    "broker messages",
                ):
                    case.reader.read_capacity(case.account)

                order_receipts = self.rows(
                    case,
                    """
                    SELECT completeness, canonical_parsed_json
                    FROM broker_read_receipts
                    WHERE read_kind = 'OPEN_ORDERS_PAGE'
                    """,
                )
                self.assertEqual(len(order_receipts), 1)
                self.assertEqual(
                    order_receipts[0]["completeness"], "INELIGIBLE"
                )
                self.assertEqual(
                    order_receipts[0]["canonical_parsed_json"], "null"
                )
                self.assertEqual(
                    self.rows(
                        case, "SELECT * FROM broker_read_manifests"
                    ),
                    [],
                )

    def test_known_order_statuses_map_to_fill_classification_and_payload_hash(
        self,
    ) -> None:
        expected_hash = canonical_order_payload_hash(
            expected_vertical_payload()
        )
        variants = (
            ("OPEN", "0", "OPEN", "OPEN"),
            ("EXECUTED", "1", "FILLED", "FULL_FILL"),
            (
                "CANCELLED",
                "0",
                "CANCELLED",
                "ZERO_FILL_TERMINAL",
            ),
            (
                "REJECTED",
                "0",
                "REJECTED",
                "ZERO_FILL_TERMINAL",
            ),
            (
                "EXPIRED",
                "0",
                "EXPIRED",
                "ZERO_FILL_TERMINAL",
            ),
            (
                "CANCEL_REQUESTED",
                "0",
                "UNRESOLVED",
                "UNRESOLVED",
            ),
        )
        for (
            status,
            filled_quantity,
            expected_outcome,
            expected_classification,
        ) in variants:
            with self.subTest(status=status):
                case = self.case(
                    [
                        account_list(account()),
                        known_order(
                            status,
                            filled_quantity=filled_quantity,
                        ),
                        account_list(account()),
                    ]
                )
                evidence = case.reader.query_order(
                    case.account, ORDER_ID
                )
                result = self.manifest_result(
                    case, evidence.evidence_sha256
                )
                self.assertEqual(
                    result["schema"], "etrade-order-query.v2"
                )
                self.assertEqual(result["raw_status"], status)
                self.assertEqual(result["outcome"], expected_outcome)
                summary = result["fill_summary"]
                self.assertEqual(
                    summary["classification"],
                    expected_classification,
                )
                self.assertEqual(
                    summary["placed_time_epoch_ms"],
                    "1785167000000",
                )
                self.assertEqual(
                    summary["executed_time_epoch_ms"],
                    (
                        "1785167100000"
                        if status == "EXECUTED"
                        else None
                    ),
                )
                self.assertEqual(len(summary["legs"]), 2)
                self.assertEqual(
                    {
                        leg["order_action"]
                        for leg in summary["legs"]
                    },
                    {"BUY_OPEN", "SELL_OPEN"},
                )
                for leg in summary["legs"]:
                    self.assertEqual(
                        set(leg),
                        {
                            "product",
                            "leg_number",
                            "order_action",
                            "ordered_quantity",
                            "filled_quantity",
                            "cancel_quantity",
                        },
                    )
                self.assertFalse(result["not_found"])
                self.assertIn(
                    expected_hash, result["order_payload_hashes"]
                )
                self.assertEqual(
                    result["raw_response_digest"],
                    self.rows(
                        case,
                        """
                        SELECT raw_response_sha256
                        FROM broker_read_receipts
                        WHERE read_kind = 'ORDER_DETAIL'
                        """,
                    )[0]["raw_response_sha256"],
                )

    def test_partial_terminal_and_replacement_link_stay_unresolved(
        self,
    ) -> None:
        variants = (
            known_order(
                "CANCELLED",
                filled_quantity="1",
                second_filled_quantity="0",
            ),
            known_order("OPEN", replaced_by_order_id="95"),
            known_order("EXECUTED", filled_quantity="1", replaces_order_id="93"),
        )
        for reply in variants:
            with self.subTest(raw=reply.raw):
                case = self.case(
                    [
                        account_list(account()),
                        reply,
                        account_list(account()),
                    ]
                )
                evidence = case.reader.query_order(
                    case.account, ORDER_ID
                )
                result = self.manifest_result(
                    case, evidence.evidence_sha256
                )
                self.assertEqual(result["outcome"], "UNRESOLVED")

    def test_fill_summary_preserves_raw_leg_numbers_before_sorting(
        self,
    ) -> None:
        reply = known_order("EXECUTED", filled_quantity="1")
        document = reply_document(reply)
        instruments = document["OrdersResponse"]["Order"][0][
            "OrderDetail"
        ][0]["Instrument"]
        instruments.reverse()
        case = self.case([])

        parsed, completeness = case.reader._parse_order_detail(
            200,
            Reply.json(document).raw,
            ORDER_ID,
        )

        self.assertEqual(completeness, "COMPLETE")
        by_action = {
            leg["order_action"]: leg["leg_number"]
            for leg in parsed["fill_summary"]["legs"]
        }
        self.assertEqual(
            by_action, {"BUY_OPEN": 1, "SELL_OPEN": 2}
        )
        normalized_legs = parsed["normalized_order"]["legs"]
        with self.assertRaises(ETradeBrokerReaderIntegrityError):
            _canonical_fill_summary_legs(normalized_legs[:1])
        duplicate = [dict(leg) for leg in normalized_legs]
        duplicate[1]["legNumber"] = duplicate[0]["legNumber"]
        with self.assertRaises(ETradeBrokerReaderIntegrityError):
            _canonical_fill_summary_legs(duplicate)

    def test_missing_fill_or_cancel_quantity_never_defaults_to_zero(
        self,
    ) -> None:
        for missing_field in ("filledQuantity", "cancelQuantity"):
            with self.subTest(missing_field=missing_field):
                document = reply_document(known_order("OPEN"))
                del document["OrdersResponse"]["Order"][0][
                    "OrderDetail"
                ][0]["Instrument"][0][missing_field]
                case = self.case(
                    [
                        account_list(account()),
                        Reply.json(document),
                    ]
                )

                with self.assertRaises(ETradeBrokerReaderIntegrityError):
                    case.reader.query_order(case.account, ORDER_ID)

                detail = self.rows(
                    case,
                    """
                    SELECT completeness, canonical_parsed_json
                    FROM broker_read_receipts
                    WHERE read_kind = 'ORDER_DETAIL'
                    """,
                )
                self.assertEqual(len(detail), 1)
                self.assertEqual(detail[0]["completeness"], "INELIGIBLE")
                self.assertEqual(
                    detail[0]["canonical_parsed_json"], "null"
                )

    def test_full_fill_requires_execution_time_balance_and_zero_cancel(
        self,
    ) -> None:
        variants: list[dict] = []
        missing_execution = reply_document(
            known_order("EXECUTED", filled_quantity="1")
        )
        del missing_execution["OrdersResponse"]["Order"][0][
            "OrderDetail"
        ][0]["executedTime"]
        variants.append(missing_execution)

        nonzero_cancel = reply_document(
            known_order("EXECUTED", filled_quantity="0")
        )
        nonzero_cancel["OrdersResponse"]["Order"][0]["OrderDetail"][0][
            "Instrument"
        ][0]["cancelQuantity"] = "1"
        variants.append(nonzero_cancel)

        unbalanced = reply_document(
            known_order("EXECUTED", filled_quantity="1")
        )
        second_leg = unbalanced["OrdersResponse"]["Order"][0][
            "OrderDetail"
        ][0]["Instrument"][1]
        second_leg["orderedQuantity"] = "2"
        second_leg["filledQuantity"] = "2"
        variants.append(unbalanced)

        case = self.case([])
        for document in variants:
            with self.subTest(document=document):
                parsed, completeness = case.reader._parse_order_detail(
                    200, Reply.json(document).raw, ORDER_ID
                )
                self.assertEqual(completeness, "COMPLETE")
                self.assertEqual(parsed["outcome"], "UNRESOLVED")
                self.assertEqual(
                    parsed["fill_summary"]["classification"],
                    "UNRESOLVED",
                )

    def test_zero_fill_terminal_requires_complete_cancel_arithmetic(
        self,
    ) -> None:
        document = reply_document(
            known_order(
                "CANCELLED",
                filled_quantity="0",
                cancel_quantity="0",
                second_cancel_quantity="0",
            )
        )
        case = self.case([])

        parsed, completeness = case.reader._parse_order_detail(
            200, Reply.json(document).raw, ORDER_ID
        )

        self.assertEqual(completeness, "COMPLETE")
        self.assertEqual(parsed["outcome"], "UNRESOLVED")
        self.assertEqual(
            parsed["fill_summary"]["classification"], "UNRESOLVED"
        )

    def test_zero_fill_terminal_rejects_an_execution_timestamp(
        self,
    ) -> None:
        document = reply_document(known_order("CANCELLED"))
        document["OrdersResponse"]["Order"][0]["OrderDetail"][0][
            "executedTime"
        ] = "1785167100000"
        case = self.case([])

        parsed, completeness = case.reader._parse_order_detail(
            200, Reply.json(document).raw, ORDER_ID
        )

        self.assertEqual(completeness, "COMPLETE")
        self.assertEqual(parsed["outcome"], "UNRESOLVED")
        self.assertEqual(
            parsed["fill_summary"]["classification"], "UNRESOLVED"
        )

    def test_order_epoch_milliseconds_are_exact_and_chronological(
        self,
    ) -> None:
        invalid_documents: list[dict] = []
        short_epoch = reply_document(
            known_order("EXECUTED", filled_quantity="1")
        )
        short_epoch["OrdersResponse"]["Order"][0]["OrderDetail"][0][
            "executedTime"
        ] = "123"
        invalid_documents.append(short_epoch)
        reversed_times = reply_document(
            known_order("EXECUTED", filled_quantity="1")
        )
        reversed_times["OrdersResponse"]["Order"][0]["OrderDetail"][0][
            "executedTime"
        ] = "1785166000000"
        invalid_documents.append(reversed_times)
        case = self.case([])
        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(
                    ETradeBrokerReaderIntegrityError
                ):
                    case.reader._parse_order_detail(
                        200, Reply.json(document).raw, ORDER_ID
                    )

    def test_order_404_is_complete_durable_but_remains_a_blocker(self) -> None:
        case = self.case(
            [
                account_list(account()),
                Reply(404, b""),
                account_list(account()),
            ]
        )

        evidence = case.reader.query_order(case.account, ORDER_ID)

        result = self.manifest_result(
            case, evidence.evidence_sha256
        )
        self.assertTrue(result["not_found"])
        self.assertEqual(result["outcome"], "UNRESOLVED")
        self.assertEqual(result["raw_status"], "NOT_FOUND")
        self.assertEqual(
            result["fill_summary"],
            {
                "classification": "UNRESOLVED",
                "placed_time_epoch_ms": None,
                "executed_time_epoch_ms": None,
                "legs": [],
            },
        )
        self.assertEqual(result["order_payload_hashes"], [])
        detail = self.rows(
            case,
            """
            SELECT http_status, completeness, target_broker_order_id,
                   raw_byte_length, raw_response_sha256
            FROM broker_read_receipts
            WHERE read_kind = 'ORDER_DETAIL'
            """,
        )
        self.assertEqual(len(detail), 1)
        self.assertEqual(detail[0]["http_status"], 404)
        self.assertEqual(detail[0]["completeness"], "COMPLETE")
        self.assertEqual(
            detail[0]["target_broker_order_id"], ORDER_ID
        )
        self.assertEqual(detail[0]["raw_byte_length"], 0)
        self.assertEqual(
            detail[0]["raw_response_sha256"],
            hashlib.sha256(b"").hexdigest(),
        )
        self.assertEqual(
            len(
                self.rows(
                    case,
                    """
                    SELECT *
                    FROM broker_read_manifest_members
                    WHERE evidence_sha256 = ?
                    """,
                    (evidence.evidence_sha256,),
                )
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
