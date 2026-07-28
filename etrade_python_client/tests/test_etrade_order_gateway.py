from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from unittest.mock import patch

from rauth import OAuth1Session

import live_trading.etrade_order_gateway as gateway_module
from live_trading.etrade_broker_reader import (
    ETradeBrokerReader as ConcreteETradeBrokerReader,
    ETradeBrokerReaderUnavailable,
    _PARSER_CODE_SHA256,
    _PARSER_CONFIG_SHA256,
    _PARSER_SCHEMA,
    _reparse_broker_read_response,
)
from live_trading.etrade_broker_transport import (
    ETradeBrokerTransport,
    SelectedBrokerAccount,
    _ExchangeResult,
)
from live_trading.etrade_order_gateway import (
    CancelClosingCommand,
    CancelOpeningCommand,
    EtradeOrderGateway,
    GatewayReconciliationRequired,
    GatewayValidationError,
    RepriceOpeningCommand,
    SubmitClosingCommand,
    SubmitOpeningCommand,
)
from live_trading.order_intent_ledger import (
    SCHEMA_VERSION,
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    BrokerReadResponseEvidence,
    OrderIntent,
    OrderIntentIntegrityError,
    OrderIntentLedger,
    OrderIntentLedgerError,
    OrderIntentReconciliationRequired,
    OrderIntentReservationError,
    OrderIntentValidationError,
    RiskEvidence,
    canonical_order_payload_hash,
)
from live_trading.order_domain import OptionContractId
from live_trading.pretrade_risk import (
    AuthorityEvidence,
    ContractQuote,
    OpeningLeg,
    OpeningRiskAuthorization,
    OpeningSpreadRequest,
    PortfolioRiskEvidence,
    QuoteSnapshotEvidence,
    RiskLimits,
    evaluate_pretrade,
)
from live_trading.runtime_safety import (
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
)


ACCOUNT_ID = "842468410"
ACCOUNT_KEY = "account/key"
INSTITUTION_TYPE = "BROKERAGE"
OWNER = "gateway-worker"


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    account: SelectedBrokerAccount
    environment: Literal["sandbox", "production"]
    broker_order_id: str = field(repr=False)
    outcome: Literal[
        "OPEN",
        "FILLED",
        "CANCELLED",
        "REJECTED",
        "EXPIRED",
        "UNRESOLVED",
    ]
    observed_at: datetime
    http_status: int
    raw_response_digest: str
    order_payload_hash: str
    complete: bool


@dataclass(frozen=True)
class BrokerOrderNotFound:
    account: SelectedBrokerAccount
    environment: Literal["sandbox", "production"]
    broker_order_id: str = field(repr=False)
    observed_at: datetime
    http_status: int
    raw_response_digest: str
    complete: bool


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _domain_json_sha256(domain, value):
    return hashlib.sha256(
        domain + _canonical_json(value).encode("utf-8")
    ).hexdigest()


class Clock:
    def __init__(self):
        self.now = datetime(2026, 7, 27, 18, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def vertical_payload(limit_price=1.25):
    return {
        "securityType": "OPTN",
        "orderAction": "SPREAD",
        "priceType": "NET_CREDIT",
        "limitPrice": limit_price,
        "orderTerm": "GOOD_FOR_DAY",
        "spreadType": "VERTICAL",
        "legs": [
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2027,
                "expiryMonth": 1,
                "expiryDay": 15,
                "strikePrice": 620,
                "orderAction": "SELL_OPEN",
                "quantity": 1,
            },
            {
                "symbol": "SPY",
                "callPut": "PUT",
                "expiryYear": 2027,
                "expiryMonth": 1,
                "expiryDay": 15,
                "strikePrice": 615,
                "orderAction": "BUY_OPEN",
                "quantity": 1,
            },
        ],
    }


def payload_bytes(limit_price=1.25):
    return json.dumps(
        vertical_payload(limit_price),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def closing_vertical_payload(limit_price=0.75):
    payload = vertical_payload(limit_price)
    payload["priceType"] = "NET_DEBIT"
    payload["legs"][0]["orderAction"] = "BUY_CLOSE"
    payload["legs"][1]["orderAction"] = "SELL_CLOSE"
    return payload


def closing_payload_bytes(limit_price=0.75):
    return json.dumps(
        closing_vertical_payload(limit_price),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def order_payload_hash(limit_price=1.25):
    return canonical_order_payload_hash(
        vertical_payload(limit_price)
    )


def preview_result(
    *,
    preview_id="1020563279",
    status="OPEN",
    messages=None,
):
    order = {"status": status}
    if messages is not None:
        order["messages"] = {"Message": messages}
    body = {
        "PreviewOrderResponse": {
            "accountId": ACCOUNT_ID,
            "orderType": "SPREADS",
            "PreviewIds": [{"previewId": preview_id}],
            "Order": [order],
        }
    }
    return _ExchangeResult(
        "RESPONSE",
        http_status=200,
        raw_response=json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


def place_result(*, order_id="94", status="OPEN", messages=None):
    order = {"status": status}
    if messages is not None:
        order["messages"] = {"Message": messages}
    body = {
        "PlaceOrderResponse": {
            "accountId": ACCOUNT_ID,
            "orderType": "SPREADS",
            "OrderIds": [{"orderId": order_id}],
            "Order": [order],
        }
    }
    return _ExchangeResult(
        "RESPONSE",
        http_status=200,
        raw_response=json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


def cancel_result(*, order_id="94"):
    body = {
        "CancelOrderResponse": {
            "accountId": ACCOUNT_ID,
            "orderId": order_id,
            "cancelTime": 1785171600000,
            "messages": {
                "Message": {
                    "code": 5011,
                    "description": (
                        "200|Your request to cancel your order is "
                        "being processed."
                    ),
                    "type": "WARNING",
                }
            },
        }
    }
    return _ExchangeResult(
        "RESPONSE",
        http_status=200,
        raw_response=json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    )


class ExchangeHarness:
    def __init__(self):
        self.outcomes = []
        self.calls = []

    def exchange(
        self,
        prepared,
        *,
        timeout_seconds,
        max_response_bytes=64 * 1024,
    ):
        self.calls.append((prepared, timeout_seconds))
        if not self.outcomes:
            raise AssertionError("unexpected broker mutation")
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def add(self, *outcomes):
        self.outcomes.extend(outcomes)

    def count(self, suffix):
        return sum(call[0].url.endswith(suffix) for call in self.calls)


class FakeReader:
    def __init__(self, clock, account, environment="sandbox"):
        self.clock = clock
        self.account = account
        self.environment = environment
        self.capacity_digest = "a" * 64
        self.capacity_complete = True
        self.capacity_order_ids = ()
        self.capacity_advance_seconds = 0
        self.capacity_position_hook = None
        self.capacity_calls = []
        self.selected_hook = None
        self.query_behaviors = {}
        self.query_calls = []
        self.authorization_counter = 0
        self.ledger = None
        self.runtime_safety = None
        self.parsed_receipts = {}

    def assert_gateway_binding(self, ledger, runtime_safety):
        if self.environment != runtime_safety.environment:
            raise RuntimeSafetyError("reader environment mismatch")
        if self.ledger is None:
            self.ledger = ledger
            self.runtime_safety = runtime_safety
        elif (
            self.ledger is not ledger
            or self.runtime_safety is not runtime_safety
        ):
            raise RuntimeSafetyError("reader durable binding mismatch")

    def selected_account(self):
        if self.selected_hook is not None:
            hook = self.selected_hook
            self.selected_hook = None
            hook()
        return self.account

    def read_capacity(self, account):
        self.capacity_calls.append(account)
        if account != self.account:
            raise RuntimeSafetyError("reader account mismatch")
        if not self.capacity_complete:
            receipt = self._record_response(
                read_kind="ACCOUNT_LIST",
                route="/v1/accounts/list.json",
                raw=b'{"AccountListResponse":{"incomplete":true}}',
                parsed=self._binding(),
                final_response=True,
            )
            return self.ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="CAPACITY",
                    account_id=self.account.account_id,
                    account_id_key=self.account.account_id_key,
                    institution_type=self.account.institution_type,
                    environment=self.environment,
                    origin=self._origin,
                    target_broker_order_id=None,
                    observed_at=self.clock.now,
                    completeness="INCOMPLETE",
                    canonical_result_json=_canonical_json({}),
                ),
                (
                    BrokerReadManifestMember(
                        role="binding.start",
                        receipt_sha256=receipt.receipt_sha256,
                    ),
                ),
            )

        if self.capacity_order_ids or self.capacity_advance_seconds:
            self.clock.advance(max(1, self.capacity_advance_seconds))
        binding = self._binding()
        encoded_account = quote(self.account.account_id_key, safe="")
        as_of = str(int(self.clock.now.timestamp() * 1_000))
        positions = self._filled_positions()
        if self.capacity_position_hook is not None:
            positions = self.capacity_position_hook(positions)
        sources = [
            (
                "binding.start",
                "ACCOUNT_LIST",
                "/v1/accounts/list.json",
                (),
                b'{"AccountListResponse":{"binding":"start"}}',
                binding,
            )
        ]
        for scan in ("a", "b"):
            sources.extend(
                (
                    (
                        f"scan_{scan}.balance",
                        "BALANCE",
                        f"/v1/accounts/{encoded_account}/balance.json",
                        (
                            ("instType", self.account.institution_type),
                            ("realTimeNAV", "true"),
                        ),
                        (
                            '{"BalanceResponse":{"Computed":'
                            '{"marginBuyingPower":2000}}}'
                        ).encode("ascii"),
                        {
                            "account_id": self.account.account_id,
                            "institution_type": self.account.institution_type,
                            "margin_buying_power": "2000",
                            "as_of_date": as_of,
                        },
                    ),
                    (
                        f"scan_{scan}.portfolio.0001",
                        "PORTFOLIO_PAGE",
                        f"/v1/accounts/{encoded_account}/portfolio.json",
                        (
                            ("count", "50"),
                            ("lotsRequired", "true"),
                            ("marketSession", "REGULAR"),
                            ("pageNumber", "1"),
                            ("sortBy", "SYMBOL"),
                            ("sortOrder", "ASC"),
                            ("totalsRequired", "false"),
                            ("view", "COMPLETE"),
                        ),
                        b'{"PortfolioResponse":{"page":1}}',
                        {
                            "page_number": 1,
                            "total_pages": 1,
                            "metadata_field": "totalNoOfPages",
                            "next_page": None,
                            "positions": positions,
                        },
                    ),
                )
            )
            for lane in ("OPEN", "CANCEL_REQUESTED", "INDIVIDUAL_FILLS"):
                sources.append(
                    (
                        f"scan_{scan}.orders.{lane}.0000",
                        "OPEN_ORDERS_PAGE",
                        f"/v1/accounts/{encoded_account}/orders.json",
                        (("count", "100"), ("status", lane)),
                        b'{"OrdersResponse":{"Order":[]}}',
                        {
                            "status_lane": lane,
                            "orders": [],
                            "marker": None,
                        },
                    )
                )
        sources.append(
            (
                "binding.end",
                "ACCOUNT_LIST",
                "/v1/accounts/list.json",
                (),
                b'{"AccountListResponse":{"binding":"end"}}',
                binding,
            )
        )
        members = []
        for role, kind, route, query, raw, parsed in sources:
            receipt = self._record_response(
                read_kind=kind,
                route=route,
                query=query,
                raw=raw,
                parsed=parsed,
                final_response=role == "binding.end",
            )
            members.append(
                BrokerReadManifestMember(
                    role=role, receipt_sha256=receipt.receipt_sha256
                )
            )
        economic_state = {
            "schema": "etrade-capacity.v3",
            "account_status": "ACTIVE",
            "account_mode": "MARGIN",
            "account_type": "INDIVIDUAL",
            "broker_buying_power": "2000",
            "positions": positions,
            "open_orders": [],
        }
        result = dict(economic_state)
        result["broker_buying_power_as_of"] = as_of
        result["state_sha256"] = _domain_json_sha256(
            b"etrade-capacity-state.v3\0", economic_state
        )
        return self.ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="CAPACITY",
                account_id=self.account.account_id,
                account_id_key=self.account.account_id_key,
                institution_type=self.account.institution_type,
                environment=self.environment,
                origin=self._origin,
                target_broker_order_id=None,
                observed_at=self.clock.now,
                completeness="COMPLETE",
                canonical_result_json=_canonical_json(result),
            ),
            tuple(members),
        )

    def _filled_positions(self):
        positions = []
        acquired = str(int(self.clock.now.timestamp() * 1_000) - 1)
        for offset, broker_order_id in enumerate(
            self.capacity_order_ids, start=1
        ):
            for leg_no, (strike, quantity, position_type) in enumerate(
                (("620", "-1", "SHORT"), ("615", "1", "LONG")),
                start=1,
            ):
                position_id = str(
                    int(broker_order_id) * 100 + offset * 10 + leg_no
                )
                positions.append(
                    {
                        "position_id": position_id,
                        "account_id": self.account.account_id,
                        "product": {
                            "symbol": "SPY",
                            "security_type": "OPTN",
                            "call_put": "PUT",
                            "expiry_year": "2027",
                            "expiry_month": "1",
                            "expiry_day": "15",
                            "strike_price": strike,
                            "product_id": None,
                        },
                        "quantity": quantity,
                        "position_type": position_type,
                        "position_indicator": "TYPE1",
                        "osi_key": (
                            "SPY---270115P00620000"
                            if strike == "620"
                            else "SPY---270115P00615000"
                        ),
                        "option_multiplier": "100",
                        "options_adjusted_flag": False,
                        "deliverables": "100 shares of SPY",
                        "lots": [
                            {
                                "position_id": position_id,
                                "position_lot_id": str(
                                    int(position_id) * 100 + 1
                                ),
                                "order_no": broker_order_id,
                                "leg_no": str(leg_no),
                                "original_quantity": quantity,
                                "remaining_quantity": quantity,
                                "available_quantity": quantity,
                                "acquired_date_epoch_ms": acquired,
                            }
                        ],
                    }
                )
        positions.sort(key=lambda item: item["position_id"])
        return positions

    def query_order(self, account, broker_order_id):
        self.query_calls.append((account, broker_order_id))
        behavior = self.query_behaviors.get(broker_order_id)
        if isinstance(behavior, BaseException):
            raise behavior
        if callable(behavior):
            return behavior()
        if behavior is None:
            return self._order_manifest(
                broker_order_id=broker_order_id,
                outcome="UNRESOLVED",
                payload_hashes=(),
                not_found=True,
            )
        if type(behavior) is BrokerOrderNotFound:
            return self._order_manifest(
                broker_order_id=behavior.broker_order_id,
                outcome="UNRESOLVED",
                payload_hashes=(),
                not_found=True,
            )
        if type(behavior) is not BrokerOrderSnapshot:
            return behavior
        return self._order_manifest(
            broker_order_id=behavior.broker_order_id,
            outcome=behavior.outcome,
            payload_hashes=(behavior.order_payload_hash,),
            not_found=False,
        )

    @property
    def _origin(self):
        return (
            "https://api.etrade.com"
            if self.environment == "production"
            else "https://apisb.etrade.com"
        )

    def _binding(self):
        return {
            "account_id": self.account.account_id,
            "account_id_key": self.account.account_id_key,
            "institution_type": self.account.institution_type,
            "account_status": "ACTIVE",
            "account_mode": "MARGIN",
            "account_type": "INDIVIDUAL",
        }

    def _record_response(
        self,
        *,
        read_kind,
        route,
        raw,
        parsed,
        query=(),
        target_broker_order_id=None,
        http_status=200,
        final_response=False,
    ):
        self.authorization_counter += 1
        completed_at = (
            self.clock.now
            if final_response
            else self.clock.now
            - timedelta(milliseconds=50)
            + timedelta(milliseconds=self.authorization_counter)
        )
        if read_kind == "ACCOUNT_LIST":
            raw = _canonical_json(
                {
                    "AccountListResponse": {
                        "Accounts": {
                            "Account": [
                                {
                                    "accountId": self.account.account_id,
                                    "accountIdKey": self.account.account_id_key,
                                    "institutionType": self.account.institution_type,
                                    "accountStatus": "ACTIVE",
                                    "accountMode": "MARGIN",
                                    "accountType": "INDIVIDUAL",
                                }
                            ]
                        }
                    }
                }
            ).encode("ascii")
        elif read_kind == "BALANCE":
            raw = _canonical_json(
                {
                    "BalanceResponse": {
                        "accountId": self.account.account_id,
                        "institutionType": self.account.institution_type,
                        "asOfDate": parsed["as_of_date"],
                        "Computed": {
                            "marginBuyingPower": parsed[
                                "margin_buying_power"
                            ]
                        },
                    }
                }
            ).encode("ascii")
        elif read_kind == "PORTFOLIO_PAGE":
            raw = _canonical_json(
                {
                    "PortfolioResponse": {
                        "AccountPortfolio": [
                            {
                                "accountId": self.account.account_id,
                                "totalNoOfPages": "1",
                                "Position": [
                                    self._raw_position(position)
                                    for position in parsed["positions"]
                                ],
                            }
                        ]
                    }
                }
            ).encode("ascii")
        elif read_kind == "OPEN_ORDERS_PAGE":
            raw = b""
            http_status = 204
        evidence = BrokerReadResponseEvidence(
            read_kind=read_kind,
            account_id=self.account.account_id,
            account_id_key=self.account.account_id_key,
            institution_type=self.account.institution_type,
            environment=self.environment,
            origin=self._origin,
            route=route,
            query_json=_canonical_json(
                [list(pair) for pair in sorted(query)]
            ),
            authorization_sha256=hashlib.sha256(
                (
                    "gateway-reader-authorization:"
                    f"{self.authorization_counter}"
                ).encode("ascii")
            ).hexdigest(),
            target_broker_order_id=target_broker_order_id,
            request_started_at=completed_at - timedelta(microseconds=500),
            response_completed_at=completed_at,
            http_status=http_status,
            raw_response_bytes=raw,
            parser_schema=_PARSER_SCHEMA,
            parser_code_sha256=_PARSER_CODE_SHA256,
            parser_config_sha256=_PARSER_CONFIG_SHA256,
            canonical_parsed_json="null",
            completeness="INELIGIBLE",
        )
        canonical_parsed_json, completeness = (
            _reparse_broker_read_response(evidence)
        )
        evidence = replace(
            evidence,
            canonical_parsed_json=canonical_parsed_json,
            completeness=completeness,
        )
        receipt = self.ledger.record_broker_read_response(evidence)
        self.parsed_receipts[receipt.receipt_sha256] = (
            canonical_parsed_json
        )
        return receipt

    @staticmethod
    def _raw_position(position):
        product = position["product"]
        raw_product = {
            "symbol": product["symbol"],
            "securityType": product["security_type"],
        }
        for raw_name, normalized_name in (
            ("callPut", "call_put"),
            ("expiryYear", "expiry_year"),
            ("expiryMonth", "expiry_month"),
            ("expiryDay", "expiry_day"),
            ("strikePrice", "strike_price"),
        ):
            value = product[normalized_name]
            if value is not None:
                raw_product[raw_name] = value
        return {
            "positionId": position["position_id"],
            "accountId": position["account_id"],
            "Product": raw_product,
            "quantity": position["quantity"],
            "positionType": position["position_type"],
            "positionIndicator": position["position_indicator"],
            "osiKey": position["osi_key"],
            "optionMultiplier": position["option_multiplier"],
            "optionsAdjustedFlag":
                position["options_adjusted_flag"],
            "deliverablesStr": position["deliverables"],
            "PositionLot": [
                {
                    "positionId": lot["position_id"],
                    "positionLotId": lot["position_lot_id"],
                    "orderNo": lot["order_no"],
                    "legNo": lot["leg_no"],
                    "originalQty": lot["original_quantity"],
                    "remainingQty": lot["remaining_quantity"],
                    "availableQty": lot["available_quantity"],
                    "acquiredDate": lot["acquired_date_epoch_ms"],
                }
                for lot in position["lots"]
            ],
        }

    def _order_manifest(
        self,
        *,
        broker_order_id,
        outcome,
        payload_hashes,
        not_found,
    ):
        binding = self._binding()
        encoded_account = quote(self.account.account_id_key, safe="")
        encoded_order = quote(broker_order_id, safe="")
        binding_start = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=b'{"AccountListResponse":{"binding":"query-start"}}',
            parsed=binding,
        )
        terminal = outcome in {
            "FILLED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
        }
        placed_time = str(
            int((self.clock.now - timedelta(seconds=2)).timestamp() * 1_000)
        )
        executed_time = str(
            int((self.clock.now - timedelta(seconds=1)).timestamp() * 1_000)
        )
        is_closing = canonical_order_payload_hash(
            closing_vertical_payload()
        ) in payload_hashes
        raw = (
            b""
            if not_found
            else _canonical_json(
                {
                    "OrdersResponse": {
                        "Order": [
                            {
                                "orderId": broker_order_id,
                                "orderType": "SPREADS",
                                "OrderDetail": [
                                    {
                                        "accountId": self.account.account_id,
                                        "orderNumber": broker_order_id,
                                        "status": (
                                            "EXECUTED"
                                            if outcome == "FILLED"
                                            else outcome
                                        ),
                                        "placedTime": placed_time,
                                        **(
                                            {"executedTime": executed_time}
                                            if outcome == "FILLED"
                                            else {}
                                        ),
                                        "priceType": (
                                            "NET_DEBIT"
                                            if is_closing
                                            else "NET_CREDIT"
                                        ),
                                        "limitPrice": (
                                            "0.75"
                                            if is_closing
                                            else (
                                                "1.50"
                                                if order_payload_hash(1.50)
                                                in payload_hashes
                                                else "1.25"
                                            )
                                        ),
                                        "orderTerm": "GOOD_FOR_DAY",
                                        "marketSession": "REGULAR",
                                        "allOrNone": False,
                                        "stopPrice": "0",
                                        "Instrument": [
                                            self._known_leg(
                                                (
                                                    "BUY_CLOSE"
                                                    if is_closing
                                                    else "SELL_OPEN"
                                                ),
                                                "620",
                                                filled=outcome == "FILLED",
                                                cancelled=(
                                                    terminal
                                                    and outcome != "FILLED"
                                                ),
                                            ),
                                            self._known_leg(
                                                (
                                                    "SELL_CLOSE"
                                                    if is_closing
                                                    else "BUY_OPEN"
                                                ),
                                                "615",
                                                filled=outcome == "FILLED",
                                                cancelled=(
                                                    terminal
                                                    and outcome != "FILLED"
                                                ),
                                            ),
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                }
            ).encode("ascii")
        )
        detail = self._record_response(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{encoded_account}/orders/"
                f"{encoded_order}.json"
            ),
            raw=raw,
            parsed={},
            target_broker_order_id=broker_order_id,
            http_status=404 if not_found else 200,
        )
        parsed = json.loads(
            self._receipt_parsed_json(detail.receipt_sha256)
        )
        binding_end = self._record_response(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            raw=b'{"AccountListResponse":{"binding":"query-end"}}',
            parsed=binding,
            final_response=True,
        )
        result = {
            "schema": "etrade-order-query.v2",
            "broker_order_id": broker_order_id,
            "raw_status": parsed["raw_status"],
            "outcome": parsed["outcome"],
            "fill_summary": parsed["fill_summary"],
            "order_payload_hashes": parsed["order_payload_hashes"],
            "http_status": 404 if not_found else 200,
            "raw_response_digest": hashlib.sha256(raw).hexdigest(),
            "not_found": parsed["not_found"],
            "replacement_links": parsed["replacement_links"],
        }
        return self.ledger.record_broker_read_manifest(
            BrokerReadManifestEvidence(
                evidence_kind="ORDER_QUERY",
                account_id=self.account.account_id,
                account_id_key=self.account.account_id_key,
                institution_type=self.account.institution_type,
                environment=self.environment,
                origin=self._origin,
                target_broker_order_id=broker_order_id,
                observed_at=self.clock.now,
                completeness="COMPLETE",
                canonical_result_json=_canonical_json(result),
            ),
            (
                BrokerReadManifestMember(
                    role="binding.start",
                    receipt_sha256=binding_start.receipt_sha256,
                ),
                BrokerReadManifestMember(
                    role="order.detail",
                    receipt_sha256=detail.receipt_sha256,
                ),
                BrokerReadManifestMember(
                    role="binding.end",
                    receipt_sha256=binding_end.receipt_sha256,
                ),
            ),
        )

    def _known_leg(self, action, strike, *, filled, cancelled=False):
        return {
            "Product": {
                "symbol": "SPY",
                "securityType": "OPTN",
                "callPut": "PUT",
                "expiryYear": "2027",
                "expiryMonth": "1",
                "expiryDay": "15",
                "strikePrice": strike,
            },
            "orderAction": action,
            "quantityType": "QUANTITY",
            "orderedQuantity": "1",
            "filledQuantity": "1" if filled else "0",
            "cancelQuantity": "1" if cancelled else "0",
        }

    def _receipt_parsed_json(self, receipt_sha256):
        return self.parsed_receipts[receipt_sha256]


class EtradeOrderGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        os.chmod(self.temporary.name, 0o700)
        self.path = (
            Path(self.temporary.name) / "runtime" / "orders.sqlite3"
        )
        self.clock = Clock()
        self.account = SelectedBrokerAccount(
            ACCOUNT_ID, ACCOUNT_KEY, INSTITUTION_TYPE
        )
        self.boundary = RuntimeSafetyBoundary(
            "sandbox",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(minutes=10),
        )
        self.ledger = OrderIntentLedger(
            self.path, clock=self.clock, run_id="gateway-run-a"
        )
        self.harness = ExchangeHarness()
        self.patcher = patch(
            "live_trading.etrade_broker_transport._isolated_mutation_exchange",
            side_effect=self.harness.exchange,
        )
        self.patcher.start()
        self.reader_type_patcher = patch.object(
            gateway_module, "ETradeBrokerReader", FakeReader
        )
        self.reader_type_patcher.start()
        self.reader = FakeReader(self.clock, self.account)
        self.transport = self.make_transport(
            self.ledger, self.boundary
        )
        self.gateway = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=self.ledger,
            transport=self.transport,
            reader=self.reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        self.gateway.start()

    def tearDown(self):
        self.reader_type_patcher.stop()
        self.patcher.stop()
        self.temporary.cleanup()

    def make_transport(self, ledger, boundary):
        return ETradeBrokerTransport(
            session=OAuth1Session(
                "consumer-key",
                "consumer-secret",
                access_token="access-token",
                access_token_secret="access-secret",
            ),
            ledger=ledger,
            runtime_safety=boundary,
            selected_account=self.account,
            clock=self.clock,
        )

    def command(
        self,
        *,
        key="order-1",
        decision="decision-1",
        lease_seconds=30,
    ):
        return SubmitOpeningCommand(
            strategy_id="put-credit-spread",
            decision_id=decision,
            idempotency_scope="decision",
            idempotency_key=key,
            payload_bytes=payload_bytes(),
            max_loss_amount=Decimal("400"),
            collateral_amount=Decimal("500"),
            quote_observed_at=self.clock.now,
            quote_valid_until=self.clock.now + timedelta(seconds=30),
            quote_digest="f" * 64,
            owner=OWNER,
            lease_seconds=lease_seconds,
        )

    def opening_risk_prerequisite_inputs(
        self,
        *,
        key="risk-prerequisite-1",
        decision_id="risk-decision-1",
    ):
        capacity_ref = self.reader.read_capacity(self.account)
        capacity = self.ledger.set_reservation_cap_from_read(
            capacity_ref, risk_budget=Decimal("2000")
        )

        def contract(strike):
            return OptionContractId(
                symbol="SPY",
                expiry=date(2027, 1, 15),
                call_put="PUT",
                strike=Decimal(strike),
                osi_key=(
                    f"{'SPY':<6}270115P"
                    f"{int(strike) * 1000:08d}"
                ),
                multiplier=Decimal("100"),
                adjusted=False,
                deliverables=None,
            )

        short_contract = contract(620)
        long_contract = contract(615)
        spread = OpeningSpreadRequest(
            strategy_decision_id=decision_id,
            strategy_id="put-credit-spread",
            environment="sandbox",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            trade_session=self.clock.now.date(),
            legs=(
                OpeningLeg(short_contract, "SELL_OPEN"),
                OpeningLeg(long_contract, "BUY_OPEN"),
            ),
            quantity=1,
            price_type="NET_CREDIT",
            limit_price_cents=125,
        )
        limits = RiskLimits(
            allowed_environments=("sandbox",),
            allowed_strategies=("put-credit-spread",),
            allowed_symbols=("SPY",),
            max_order_contracts=5,
            max_order_notional_cents=10_000_000,
            max_order_loss_cents=100_000,
            max_order_collateral_cents=100_000,
            max_account_open_risk_cents=200_000,
            max_symbol_open_risk_cents=200_000,
            min_remaining_buying_power_cents=0,
            max_abs_portfolio_delta=Decimal("1000"),
            max_abs_symbol_delta=Decimal("1000"),
            max_daily_loss_cents=100_000,
            max_daily_orders=100,
            max_daily_new_risk_cents=200_000,
            max_quote_age_seconds=60,
            max_portfolio_age_seconds=60,
            max_authority_age_seconds=60,
            max_overlay_age_seconds=300,
            max_option_ask_cents=1_000,
            max_bid_ask_width_cents=10,
            max_fee_cents_per_contract_per_leg=1,
            min_open_interest=100,
            min_volume=10,
            require_model_provenance=False,
            require_regime_provenance=False,
        )
        authority = AuthorityEvidence(
            environment="sandbox",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            session_date=self.clock.now.date(),
            observed_at=self.clock.now,
            arm_issued_at=self.clock.now - timedelta(seconds=1),
            arm_expires_at=self.clock.now + timedelta(minutes=5),
            session_opens_at=self.clock.now - timedelta(hours=1),
            session_closes_at=self.clock.now + timedelta(hours=1),
            complete=True,
            mutations_enabled=True,
            operator_armed=True,
            kill_switch_clear=True,
            session_open=True,
            runtime_config_sha256="1" * 64,
            arm_sha256="2" * 64,
            session_sha256="3" * 64,
        )
        quotes = QuoteSnapshotEvidence(
            complete=True,
            snapshot_sha256="4" * 64,
            quotes=(
                ContractQuote(
                    short_contract,
                    200,
                    205,
                    Decimal("-0.20"),
                    500,
                    50,
                    self.clock.now,
                    "5" * 64,
                ),
                ContractQuote(
                    long_contract,
                    70,
                    75,
                    Decimal("-0.10"),
                    500,
                    50,
                    self.clock.now,
                    "6" * 64,
                ),
            ),
        )
        portfolio = PortfolioRiskEvidence(
            environment="sandbox",
            account_id=ACCOUNT_ID,
            account_id_key=ACCOUNT_KEY,
            institution_type=INSTITUTION_TYPE,
            session_date=self.clock.now.date(),
            symbol="SPY",
            observed_at=capacity.observed_at,
            complete=True,
            stable=True,
            buying_power_cents=int(
                capacity.broker_buying_power * Decimal("100")
            ),
            open_risk_cents=0,
            symbol_open_risk_cents=0,
            portfolio_delta=Decimal("0"),
            symbol_delta=Decimal("0"),
            daily_pnl_cents=0,
            daily_order_count=0,
            daily_new_risk_cents=0,
            position_contracts=(),
            open_order_contracts=(),
            snapshot_sha256=capacity.portfolio_snapshot_digest,
            broker_read_evidence_sha256=capacity.evidence_sha256,
        )
        risk_decision = evaluate_pretrade(
            spread,
            limits,
            authority,
            quotes,
            portfolio,
            (),
            evaluated_at=self.clock.now,
        )
        authorization = OpeningRiskAuthorization(
            spread,
            limits,
            authority,
            quotes,
            portfolio,
            (),
            risk_decision,
        )
        envelope = OrderIntent.build(
            account_id=ACCOUNT_ID,
            environment="sandbox",
            strategy_id=spread.strategy_id,
            decision_id=decision_id,
            idempotency_scope="decision",
            idempotency_key=key,
            intent_kind="OPENING",
            order_payload=vertical_payload(),
        )
        return envelope, authorization, capacity

    def closing_command(
        self,
        *,
        key="close-1",
        decision="close-decision-1",
        lease_seconds=30,
    ):
        return SubmitClosingCommand(
            strategy_id="put-credit-spread-close",
            decision_id=decision,
            idempotency_scope="decision",
            idempotency_key=key,
            payload_bytes=closing_payload_bytes(),
            owner=OWNER,
            lease_seconds=lease_seconds,
        )

    def restart(self, *, reader=None):
        ledger = OrderIntentLedger(
            self.path, clock=self.clock, run_id="gateway-run-b"
        )
        reader = reader or FakeReader(self.clock, self.account)
        transport = self.make_transport(ledger, self.boundary)
        gateway = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=ledger,
            transport=transport,
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        return gateway, ledger, reader

    def submit_success(self):
        self.harness.add(preview_result(), place_result())
        return self.gateway.submit_opening(self.command())

    def submit_closing_success(self, *, broker_order_id="94"):
        self.reader.capacity_order_ids = ("93",)
        self.harness.add(
            preview_result(),
            place_result(order_id=broker_order_id),
        )
        return self.gateway.submit_closing(self.closing_command())

    def make_pending_terminal(
        self,
        *,
        key,
        decision,
        broker_order_id,
        outcome,
    ):
        submitted = self.submit_named_order(
            key=key,
            decision=decision,
            broker_order_id=broker_order_id,
        )
        self.terminalize(submitted, broker_order_id, outcome)
        return submitted

    def submit_named_order(self, *, key, decision, broker_order_id):
        self.harness.add(
            preview_result(preview_id=str(1_020_563_000 + int(broker_order_id))),
            place_result(order_id=broker_order_id),
        )
        return self.gateway.submit_opening(
            self.command(key=key, decision=decision)
        )

    def terminalize(self, submitted, broker_order_id, outcome):
        snapshot = BrokerOrderSnapshot(
            account=self.account,
            environment="sandbox",
            broker_order_id=broker_order_id,
            outcome=outcome,
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        self.reader.query_behaviors[broker_order_id] = snapshot
        read = self.reader.query_order(self.account, broker_order_id)
        terminal_evidence = self.ledger.broker_evidence_from_read(
            submitted.intent_id, read, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(terminal_evidence)
        self.ledger.reconcile_terminal(
            submitted.intent_id, outcome, terminal_evidence
        )
        self.reader.query_calls.clear()

    def terminalize_closing(self, submitted, broker_order_id, outcome):
        snapshot = BrokerOrderSnapshot(
            account=self.account,
            environment="sandbox",
            broker_order_id=broker_order_id,
            outcome=outcome,
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=canonical_order_payload_hash(
                closing_vertical_payload()
            ),
            complete=True,
        )
        self.reader.query_behaviors[broker_order_id] = snapshot
        read = self.reader.query_order(self.account, broker_order_id)
        terminal_evidence = self.ledger.broker_evidence_from_read(
            submitted.intent_id, read, operation="ORDER_QUERY"
        )
        self.assertIsNotNone(terminal_evidence)
        self.ledger.reconcile_terminal(
            submitted.intent_id, outcome, terminal_evidence
        )
        self.reader.query_calls.clear()

    def test_pure_risk_prerequisite_is_durable_but_not_send_authority(
        self,
    ):
        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs()
        )
        reserved_before = self.ledger.active_reserved_margin(
            ACCOUNT_ID, "sandbox"
        )

        created = self.ledger.create_opening_risk_prerequisite(
            envelope,
            authorization,
            capacity.decision_sha256,
            intent_id="risk-prerequisite-intent",
        )
        persisted = self.ledger.get_opening_risk_prerequisite(
            created.intent.intent_id
        )
        reservation = self.ledger.get_margin_reservation(
            created.intent.intent_id
        )

        self.assertTrue(created.created)
        self.assertEqual(
            persisted.authorization_state,
            "INDEPENDENT_EVIDENCE_PENDING",
        )
        self.assertEqual(
            persisted.decision_sha256,
            authorization.decision.decision_sha256,
        )
        self.assertEqual(
            persisted.portfolio_broker_read_evidence_sha256,
            capacity.evidence_sha256,
        )
        self.assertIsNone(reservation)
        self.assertEqual(
            persisted.collateral_amount, Decimal("500.02")
        )
        self.assertEqual(
            persisted.max_loss_amount, Decimal("375.02")
        )
        self.assertEqual(
            self.ledger.active_reserved_margin(ACCOUNT_ID, "sandbox"),
            reserved_before,
        )
        self.assertEqual(
            tuple(event.event_type for event in self.ledger.events(
                created.intent.intent_id
            )),
            ("INTENT_CREATED",),
        )
        replay = self.ledger.create_opening_risk_prerequisite(
            envelope,
            authorization,
            capacity.decision_sha256,
            intent_id="different-replay-id",
        )
        self.assertFalse(replay.created)
        self.assertEqual(replay.intent.intent_id, created.intent.intent_id)

        with self.assertRaisesRegex(
            OrderIntentReservationError,
            "independently replayable quote and portfolio-risk evidence",
        ):
            self.ledger.claim_submission(
                created.intent.intent_id, OWNER, lease_seconds=30
            )
        self.assertEqual(self.harness.calls, [])

        restarted = OrderIntentLedger(
            self.path, clock=self.clock, run_id="risk-restart"
        )
        self.assertEqual(
            restarted.get_opening_risk_prerequisite(
                created.intent.intent_id
            ),
            persisted,
        )
        with self.assertRaises(OrderIntentReservationError):
            restarted.claim_submission(
                created.intent.intent_id, OWNER, lease_seconds=30
            )
        self.assertEqual(self.harness.calls, [])

    def test_pure_risk_and_legacy_reservation_lanes_cannot_mix(self):
        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs(
                key="risk-prerequisite-lane-mixing",
                decision_id="risk-decision-lane-mixing",
            )
        )
        legacy = self.ledger.create_intent(
            envelope, intent_id="legacy-before-risk-proof"
        )
        self.ledger.reserve_margin(
            legacy.intent.intent_id,
            RiskEvidence(
                decision_id=envelope.decision_id,
                max_loss_amount=Decimal("375.02"),
                collateral_amount=Decimal("500.02"),
                quote_observed_at=self.clock.now,
                quote_digest=authorization.quote_evidence_sha256,
                portfolio_observed_at=capacity.observed_at,
                portfolio_snapshot_digest=(
                    capacity.portfolio_snapshot_digest
                ),
                capacity_decision_sha256=capacity.decision_sha256,
            ),
        )

        with self.assertRaisesRegex(
            OrderIntentIntegrityError,
            "persisted opening risk intent is incomplete",
        ):
            self.ledger.create_opening_risk_prerequisite(
                envelope,
                authorization,
                capacity.decision_sha256,
                intent_id="ignored-risk-proof-id",
            )
        self.assertIsNone(
            self.ledger.get_opening_risk_prerequisite(
                legacy.intent.intent_id
            )
        )
        self.assertIsNotNone(
            self.ledger.get_margin_reservation(legacy.intent.intent_id)
        )

    def test_pure_risk_prerequisite_fails_before_partial_creation(
        self,
    ):
        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs(
                key="risk-prerequisite-adversarial",
                decision_id="risk-decision-adversarial",
            )
        )
        mismatched_payload = OrderIntent.build(
            account_id=ACCOUNT_ID,
            environment="sandbox",
            strategy_id=envelope.strategy_id,
            decision_id=envelope.decision_id,
            idempotency_scope=envelope.idempotency_scope,
            idempotency_key=envelope.idempotency_key,
            intent_kind="OPENING",
            order_payload=vertical_payload(1.30),
        )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.create_opening_risk_prerequisite(
                mismatched_payload,
                authorization,
                capacity.decision_sha256,
                intent_id="mismatched-payload-intent",
            )

        denied_limits = replace(
            authorization.limits, allowed_symbols=("QQQ",)
        )
        denied_decision = evaluate_pretrade(
            authorization.spread,
            denied_limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            evaluated_at=authorization.decision.evaluated_at,
        )
        denied = OpeningRiskAuthorization(
            authorization.spread,
            denied_limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            denied_decision,
        )
        self.assertFalse(denied.decision.allowed)
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.create_opening_risk_prerequisite(
                envelope,
                denied,
                capacity.decision_sha256,
                intent_id="denied-risk-intent",
            )

        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.create_opening_risk_prerequisite(
                envelope,
                authorization,
                "0" * 64,
                intent_id="wrong-capacity-intent",
            )

        future_decision = evaluate_pretrade(
            authorization.spread,
            authorization.limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            evaluated_at=self.clock.now + timedelta(seconds=20),
        )
        future = OpeningRiskAuthorization(
            authorization.spread,
            authorization.limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            future_decision,
        )
        with self.assertRaisesRegex(
            OrderIntentValidationError,
            "RISK_AUTHORIZATION_FROM_FUTURE",
        ):
            self.ledger.create_opening_risk_prerequisite(
                envelope,
                future,
                capacity.decision_sha256,
                intent_id="future-risk-intent",
            )

        tight_limits = replace(
            authorization.limits,
            max_quote_age_seconds=1,
            max_portfolio_age_seconds=1,
            max_authority_age_seconds=1,
        )
        tight_decision = evaluate_pretrade(
            authorization.spread,
            tight_limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            evaluated_at=self.clock.now,
        )
        tight_authorization = OpeningRiskAuthorization(
            authorization.spread,
            tight_limits,
            authorization.authority,
            authorization.quotes,
            authorization.portfolio,
            authorization.overlays,
            tight_decision,
        )

        class DelayedTransactionLedger(OrderIntentLedger):
            @contextmanager
            def _transaction(delayed_self):
                with super()._transaction() as connection:
                    self.clock.advance(2)
                    yield connection

        delayed_ledger = DelayedTransactionLedger(
            self.path,
            clock=self.clock,
            run_id="delayed-risk-transaction",
        )
        with self.assertRaisesRegex(
            OrderIntentValidationError,
            "RISK_AUTHORIZATION_NOT_CURRENT",
        ):
            delayed_ledger.create_opening_risk_prerequisite(
                envelope,
                tight_authorization,
                capacity.decision_sha256,
                intent_id="contended-stale-risk-intent",
            )

        self.clock.advance(61)
        with self.assertRaises(OrderIntentValidationError):
            self.ledger.create_opening_risk_prerequisite(
                envelope,
                authorization,
                capacity.decision_sha256,
                intent_id="stale-risk-intent",
            )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM order_intents
                    WHERE idempotency_key = ?
                    """,
                    (envelope.idempotency_key,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM opening_risk_prerequisites
                    """
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM margin_reservations"
                ).fetchone()[0],
                0,
            )
        self.assertEqual(self.harness.calls, [])

    def test_pure_risk_prerequisite_is_append_only_and_self_verifying(
        self,
    ):
        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs(
                key="risk-prerequisite-tamper",
                decision_id="risk-decision-tamper",
            )
        )
        created = self.ledger.create_opening_risk_prerequisite(
            envelope,
            authorization,
            capacity.decision_sha256,
            intent_id="risk-tamper-intent",
        )
        unrelated = self.ledger.create_intent(
            OrderIntent.build(
                account_id=ACCOUNT_ID,
                environment="sandbox",
                strategy_id="unrelated-opening",
                decision_id="unrelated-decision",
                idempotency_scope="decision",
                idempotency_key="unrelated-risk-binding",
                intent_kind="OPENING",
                order_payload=vertical_payload(),
            ),
            intent_id="unrelated-risk-intent",
        )
        self.ledger.reserve_margin(
            unrelated.intent.intent_id,
            RiskEvidence(
                decision_id=unrelated.intent.envelope.decision_id,
                max_loss_amount=Decimal("500"),
                collateral_amount=Decimal("500"),
                quote_observed_at=self.clock.now,
                quote_digest="7" * 64,
                portfolio_observed_at=capacity.observed_at,
                portfolio_snapshot_digest=(
                    capacity.portfolio_snapshot_digest
                ),
                capacity_decision_sha256=capacity.decision_sha256,
            ),
        )
        with self.assertRaisesRegex(
            OrderIntentReservationError,
            "cannot reserve capacity",
        ):
            self.ledger.reserve_margin(
                created.intent.intent_id,
                RiskEvidence(
                    decision_id=created.intent.envelope.decision_id,
                    max_loss_amount=Decimal("500.02"),
                    collateral_amount=Decimal("500.02"),
                    quote_observed_at=self.clock.now,
                    quote_digest=authorization.quote_evidence_sha256,
                    portfolio_observed_at=capacity.observed_at,
                    portfolio_snapshot_digest=(
                        capacity.portfolio_snapshot_digest
                    ),
                    capacity_decision_sha256=capacity.decision_sha256,
                ),
            )
        with self.ledger._connection() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    UPDATE order_intents
                    SET opening_risk_prerequisite_sha256 = NULL
                    WHERE intent_id = ?
                    """,
                    (created.intent.intent_id,),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    INSERT INTO margin_reservations (
                        intent_id, account_id, environment, amount,
                        risk_decision_id, max_loss_amount,
                        quote_observed_at, quote_digest,
                        portfolio_observed_at,
                        portfolio_snapshot_digest,
                        capacity_decision_sha256, state,
                        released_reason_code, created_at, released_at
                    )
                    SELECT ?, account_id, environment, amount, ?,
                           max_loss_amount, quote_observed_at, quote_digest,
                           portfolio_observed_at,
                           portfolio_snapshot_digest,
                           capacity_decision_sha256, state,
                           released_reason_code, created_at, released_at
                    FROM margin_reservations
                    WHERE intent_id = ?
                    """,
                    (
                        created.intent.intent_id,
                        created.intent.envelope.decision_id,
                        unrelated.intent.intent_id,
                    ),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    UPDATE opening_risk_prerequisites
                    SET request_sha256 = ?
                    WHERE intent_id = ?
                    """,
                    ("0" * 64, created.intent.intent_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    """
                    DELETE FROM opening_risk_prerequisites
                    WHERE intent_id = ?
                    """,
                    (created.intent.intent_id,),
                )
            connection.execute(
                """
                DROP TRIGGER prevent_opening_risk_prerequisite_update
                """
            )
            connection.execute(
                """
                UPDATE opening_risk_prerequisites
                SET request_sha256 = ?
                WHERE intent_id = ?
                """,
                ("0" * 64, created.intent.intent_id),
            )
        with self.assertRaises(OrderIntentIntegrityError):
            self.ledger.get_opening_risk_prerequisite(
                created.intent.intent_id
            )
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                DROP TRIGGER prevent_opening_risk_prerequisite_delete
                """
            )
            connection.execute(
                """
                DELETE FROM opening_risk_prerequisites
                WHERE intent_id = ?
                """,
                (created.intent.intent_id,),
            )
        with self.assertRaisesRegex(
            OrderIntentIntegrityError,
            "lost its pure-risk prerequisite",
        ):
            self.ledger.claim_submission(
                created.intent.intent_id, OWNER, lease_seconds=30
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path, clock=self.clock, run_id="tamper-restart"
            )
        self.assertEqual(self.harness.calls, [])

    def test_pure_risk_marker_blocks_reservation_after_proof_removal(
        self,
    ):
        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs(
                key="risk-prerequisite-marker-only",
                decision_id="risk-decision-marker-only",
            )
        )
        created = self.ledger.create_opening_risk_prerequisite(
            envelope,
            authorization,
            capacity.decision_sha256,
            intent_id="risk-marker-only-intent",
        )
        unrelated = self.ledger.create_intent(
            OrderIntent.build(
                account_id=ACCOUNT_ID,
                environment="sandbox",
                strategy_id="unrelated-opening",
                decision_id="unrelated-marker-decision",
                idempotency_scope="decision",
                idempotency_key="unrelated-marker-binding",
                intent_kind="OPENING",
                order_payload=vertical_payload(),
            ),
            intent_id="unrelated-marker-intent",
        )
        self.ledger.reserve_margin(
            unrelated.intent.intent_id,
            RiskEvidence(
                decision_id=unrelated.intent.envelope.decision_id,
                max_loss_amount=Decimal("500"),
                collateral_amount=Decimal("500"),
                quote_observed_at=self.clock.now,
                quote_digest="8" * 64,
                portfolio_observed_at=capacity.observed_at,
                portfolio_snapshot_digest=(
                    capacity.portfolio_snapshot_digest
                ),
                capacity_decision_sha256=capacity.decision_sha256,
            ),
        )

        with self.ledger._connection() as connection:
            connection.execute(
                """
                DROP TRIGGER prevent_opening_risk_prerequisite_delete
                """
            )
            connection.execute(
                """
                DELETE FROM opening_risk_prerequisites
                WHERE intent_id = ?
                """,
                (created.intent.intent_id,),
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "margin reservation insert lacks exact risk provenance",
            ):
                connection.execute(
                    """
                    INSERT INTO margin_reservations (
                        intent_id, account_id, environment, amount,
                        risk_decision_id, max_loss_amount,
                        quote_observed_at, quote_digest,
                        portfolio_observed_at,
                        portfolio_snapshot_digest,
                        capacity_decision_sha256, state,
                        released_reason_code, created_at, released_at
                    )
                    SELECT ?, account_id, environment, amount, ?,
                           max_loss_amount, quote_observed_at, quote_digest,
                           portfolio_observed_at,
                           portfolio_snapshot_digest,
                           capacity_decision_sha256, state,
                           released_reason_code, created_at, released_at
                    FROM margin_reservations
                    WHERE intent_id = ?
                    """,
                    (
                        created.intent.intent_id,
                        created.intent.envelope.decision_id,
                        unrelated.intent.intent_id,
                    ),
                )
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM margin_reservations
                    WHERE intent_id = ?
                    """,
                    (created.intent.intent_id,),
                ).fetchone()[0],
                0,
            )

        with self.assertRaisesRegex(
            OrderIntentIntegrityError,
            "lost its pure-risk prerequisite",
        ):
            self.ledger.claim_submission(
                created.intent.intent_id, OWNER, lease_seconds=30
            )
        with self.assertRaises(OrderIntentLedgerError):
            OrderIntentLedger(
                self.path, clock=self.clock, run_id="marker-only-restart"
            )
        self.assertEqual(self.harness.calls, [])

    def test_genuine_schema_14_shape_migrates_to_schema_15(self):
        with sqlite3.connect(self.path) as connection:
            for trigger in (
                "prevent_opening_risk_prerequisite_update",
                "prevent_opening_risk_prerequisite_delete",
                "validate_opening_risk_prerequisite_insert",
                "prevent_opening_risk_reservation_insert",
                "prevent_intent_identity_mutation",
                "validate_margin_reservation_insert",
            ):
                connection.execute(f"DROP TRIGGER {trigger}")
            connection.execute(
                """
                DROP INDEX idx_opening_risk_prerequisites_account
                """
            )
            connection.execute(
                """
                ALTER TABLE order_intents
                DROP COLUMN opening_risk_prerequisite_sha256
                """
            )
            connection.execute(
                "DROP TABLE opening_risk_prerequisites"
            )
            connection.execute(
                """
                UPDATE ledger_metadata SET schema_version = 14
                WHERE singleton = 1
                """
            )
        migrated = OrderIntentLedger(
            self.path, clock=self.clock, run_id="schema-14-migration"
        )
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT schema_version FROM ledger_metadata
                    WHERE singleton = 1
                    """
                ).fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertIsNotNone(
                connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'opening_risk_prerequisites'
                    """
                ).fetchone()
            )
            self.assertNotIn(
                "opening_risk_binding_sha256",
                {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(margin_reservations)"
                    )
                },
            )
            self.assertIn(
                "opening_risk_prerequisite_sha256",
                {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(order_intents)"
                    )
                },
            )

        envelope, authorization, capacity = (
            self.opening_risk_prerequisite_inputs(
                key="post-schema-14-migration",
                decision_id="post-schema-14-risk",
            )
        )
        created = migrated.create_opening_risk_prerequisite(
            envelope,
            authorization,
            capacity.decision_sha256,
            intent_id="post-schema-14-intent",
        )
        self.assertIsNotNone(
            migrated.get_opening_risk_prerequisite(
                created.intent.intent_id
            )
        )
        self.assertIsNone(
            migrated.get_margin_reservation(created.intent.intent_id)
        )
        verified = OrderIntentLedger(
            self.path, clock=self.clock, run_id="schema-15-verified"
        )
        self.assertIsNotNone(
            verified.get_opening_risk_prerequisite(
                created.intent.intent_id
            )
        )
        with self.assertRaises(OrderIntentReservationError):
            verified.claim_submission(
                created.intent.intent_id, OWNER, lease_seconds=30
            )
        self.assertEqual(self.harness.calls, [])

    def test_pure_risk_proof_has_no_production_composition_and_legacy_is_outside(
        self,
    ):
        legacy = self.ledger.create_intent(
            OrderIntent.build(
                account_id=ACCOUNT_ID,
                environment="sandbox",
                strategy_id="legacy-opening",
                decision_id="legacy-decision",
                idempotency_scope="decision",
                idempotency_key="legacy-outside-proof",
                intent_kind="OPENING",
                order_payload=vertical_payload(),
            )
        )
        self.assertIsNone(
            self.ledger.get_opening_risk_prerequisite(
                legacy.intent.intent_id
            )
        )
        gateway_source = inspect.getsource(gateway_module)
        self.assertNotIn(
            "create_opening_risk_prerequisite", gateway_source
        )
        self.assertNotIn("OpeningRiskAuthorization", gateway_source)
        self.assertEqual(self.harness.calls, [])

    def test_real_transport_submission_is_idempotent_and_places_once(self):
        first = self.submit_success()
        replay = self.gateway.submit_opening(self.command())

        self.assertEqual(first.state, "SUBMITTED")
        self.assertEqual(replay.state, "SUBMITTED")
        self.assertFalse(replay.created)
        self.assertEqual(first.intent_id, replay.intent_id)
        self.assertEqual(self.harness.count("/orders/preview"), 1)
        self.assertEqual(self.harness.count("/orders/place"), 1)
        self.assertEqual(
            [
                receipt.transport_operation
                for receipt in self.ledger.transport_response_receipts(
                    first.intent_id
                )
            ],
            ["SUBMIT_PREVIEW", "SUBMIT_PLACE"],
        )
        reservation = self.ledger.get_margin_reservation(first.intent_id)
        self.assertIsNotNone(reservation.capacity_decision_sha256)
        self.assertEqual(len(reservation.capacity_decision_sha256), 64)

    def test_closing_vertical_is_reserved_and_placed_exactly_once(self):
        first = self.submit_closing_success()
        replay = self.gateway.submit_closing(self.closing_command())

        self.assertEqual(first.state, "SUBMITTED")
        self.assertEqual(replay.state, "SUBMITTED")
        self.assertFalse(replay.created)
        self.assertEqual(first.intent_id, replay.intent_id)
        self.assertEqual(len(self.reader.capacity_calls), 1)
        reservation = self.ledger.get_closing_reservation(first.intent_id)
        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.quantity, 1)
        self.assertEqual(
            self.ledger.get_intent(first.intent_id).envelope.intent_kind,
            "CLOSING",
        )
        self.assertEqual(self.harness.count("/orders/preview"), 1)
        self.assertEqual(self.harness.count("/orders/place"), 1)
        for prepared, _timeout in self.harness.calls:
            self.assertIn(
                b"<orderAction>BUY_CLOSE</orderAction>",
                prepared.body,
            )
            self.assertIn(
                b"<orderAction>SELL_CLOSE</orderAction>",
                prepared.body,
            )

    def test_overlapping_closing_reservation_is_denied_before_preview(self):
        self.submit_closing_success()

        with self.assertRaises(GatewayValidationError):
            self.gateway.submit_closing(
                self.closing_command(
                    key="close-2", decision="close-decision-2"
                )
            )

        self.assertEqual(self.harness.count("/orders/preview"), 1)
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_failed_closing_preview_voids_capacity_for_a_new_intent(self):
        self.reader.capacity_order_ids = ("93",)
        self.harness.add(
            preview_result(
                messages={
                    "description": "manual review required",
                    "code": 1042,
                    "type": "WARNING",
                }
            )
        )

        failed = self.gateway.submit_closing(self.closing_command())

        self.assertEqual(failed.state, "FAILED")
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.clock.advance(1)
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        submitted = self.gateway.submit_closing(
            self.closing_command(
                key="close-after-failure",
                decision="close-after-failure-decision",
            )
        )
        self.assertEqual(submitted.state, "SUBMITTED")
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_stale_never_claimed_close_is_abandoned_without_broker_io(self):
        self.reader.capacity_order_ids = ("93",)
        with patch.object(
            self.ledger,
            "claim_submission",
            side_effect=SystemExit("crash before claim"),
        ):
            with self.assertRaises(SystemExit):
                self.gateway.submit_closing(self.closing_command())
        self.clock.advance(301)

        result = self.gateway.submit_closing(self.closing_command())

        self.assertEqual(result.state, "FAILED")
        self.assertEqual(result.reason_code, "STALE_CAPACITY_ABANDONED")
        self.assertEqual(self.harness.calls, [])
        self.assertEqual(
            self.ledger.closing_reservation_blockers(
                ACCOUNT_ID, "sandbox"
            ),
            (),
        )
        self.harness.add(preview_result(), place_result(order_id="95"))
        fresh = self.gateway.submit_closing(
            self.closing_command(
                key="fresh-after-stale",
                decision="fresh-after-stale-decision",
            )
        )
        self.assertEqual(fresh.state, "SUBMITTED")

    def test_unknown_closing_place_blocks_restart_without_retry(self):
        self.reader.capacity_order_ids = ("93",)
        self.harness.add(preview_result(), _ExchangeResult("TIMEOUT"))

        result = self.gateway.submit_closing(self.closing_command())

        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        with self.assertRaises(GatewayReconciliationRequired):
            self.gateway.submit_closing(self.closing_command())
        restarted, _, _ = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_zero_fill_closing_terminal_releases_without_capacity_read(self):
        submitted = self.submit_closing_success()
        self.terminalize_closing(submitted, "94", "CANCELLED")
        mutation_count = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "CANCELLED",
            self.clock.now,
            200,
            "5" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        restarted, ledger, reader = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(reader.capacity_calls, [])
        self.assertEqual(
            ledger.closing_reservation_blockers(ACCOUNT_ID, "sandbox"),
            (),
        )
        self.assertEqual(len(self.harness.calls), mutation_count)

    def test_closing_cancel_is_one_shot_and_releases_only_after_read(self):
        submitted = self.submit_closing_success()
        self.reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "OPEN",
            self.clock.now,
            200,
            "5" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        self.harness.add(cancel_result())
        command = CancelClosingCommand(
            intent_id=submitted.intent_id,
            idempotency_key="cancel-close-94",
            owner=OWNER,
        )

        accepted = self.gateway.cancel_closing(command)
        replay = self.gateway.cancel_closing(command)

        self.assertEqual(accepted.state, "REQUEST_ACCEPTED")
        self.assertEqual(replay.state, "REQUEST_ACCEPTED")
        self.assertIsNotNone(
            self.ledger.get_closing_reservation(submitted.intent_id)
        )
        self.assertEqual(self.harness.count("/orders/cancel"), 1)

        self.clock.advance(1)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "CANCELLED",
            self.clock.now,
            200,
            "6" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        restarted, ledger, reader = self.restart(reader=reader)
        restarted.start()

        self.assertEqual(
            ledger.closing_reservation_blockers(ACCOUNT_ID, "sandbox"),
            (),
        )
        self.assertEqual(reader.capacity_calls, [])
        self.assertEqual(self.harness.count("/orders/cancel"), 1)

    def test_closing_cancel_fill_race_requires_position_absorption(self):
        submitted = self.submit_closing_success()
        self.reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "OPEN",
            self.clock.now,
            200,
            "5" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        self.harness.add(cancel_result())
        self.gateway.cancel_closing(
            CancelClosingCommand(
                intent_id=submitted.intent_id,
                idempotency_key="cancel-close-fill-race",
                owner=OWNER,
            )
        )
        self.clock.advance(1)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "FILLED",
            self.clock.now,
            200,
            "6" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        reader.capacity_advance_seconds = 1
        restarted, ledger, reader = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.get_intent(submitted.intent_id).state, "FILLED"
        )
        self.assertEqual(
            ledger.closing_reservation_blockers(ACCOUNT_ID, "sandbox"),
            (),
        )
        self.assertEqual(reader.capacity_calls, [self.account])
        self.assertEqual(self.harness.count("/orders/cancel"), 1)

    def test_full_fill_closing_requires_a_newer_flat_position_read(self):
        submitted = self.submit_closing_success()
        self.terminalize_closing(submitted, "94", "FILLED")
        mutation_count = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "FILLED",
            self.clock.now,
            200,
            "6" * 64,
            canonical_order_payload_hash(closing_vertical_payload()),
            True,
        )
        restarted, ledger, reader = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(
            len(
                ledger.closing_reservation_blockers(
                    ACCOUNT_ID, "sandbox"
                )
            ),
            1,
        )

        reader.capacity_advance_seconds = 1
        restarted.start()

        self.assertEqual(
            ledger.closing_reservation_blockers(ACCOUNT_ID, "sandbox"),
            (),
        )
        self.assertEqual(len(reader.capacity_calls), 2)
        self.assertEqual(len(self.harness.calls), mutation_count)

    def test_ambiguous_closing_contract_evidence_denies_before_preview(self):
        def wrong_osi(positions):
            positions[0]["osi_key"] = "SPY---270115P00621000"
            return positions

        def adjusted(positions):
            positions[0]["options_adjusted_flag"] = True
            return positions

        def wrong_multiplier(positions):
            positions[0]["option_multiplier"] = "50"
            return positions

        def missing_lots(positions):
            positions[0]["lots"] = []
            return positions

        for index, position_hook in enumerate(
            (wrong_osi, adjusted, wrong_multiplier, missing_lots), start=1
        ):
            with self.subTest(position_hook=position_hook.__name__):
                self.reader.capacity_order_ids = ("93",)
                self.reader.capacity_position_hook = position_hook
                with self.assertRaises(GatewayValidationError):
                    self.gateway.submit_closing(
                        self.closing_command(
                            key=f"invalid-close-{index}",
                            decision=f"invalid-close-decision-{index}",
                        )
                    )
        self.assertEqual(self.harness.calls, [])

    def test_cancel_ack_stays_pending_and_restart_never_resends(
        self,
    ):
        submitted = self.submit_named_order(
            key="cancel-order",
            decision="cancel-decision",
            broker_order_id="94",
        )
        self.reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "OPEN",
            self.clock.now,
            200,
            "1" * 64,
            order_payload_hash(),
            True,
        )
        self.harness.add(cancel_result())
        command = CancelOpeningCommand(
            intent_id=submitted.intent_id,
            idempotency_key="cancel-order-94",
            owner=OWNER,
        )

        result = self.gateway.cancel_opening(command)
        replay = self.gateway.cancel_opening(command)

        self.assertEqual(result.state, "REQUEST_ACCEPTED")
        self.assertEqual(replay.state, "REQUEST_ACCEPTED")
        self.assertEqual(replay.reason_code, "IDEMPOTENT_REPLAY")
        self.assertEqual(self.harness.count("/orders/cancel"), 1)
        self.assertEqual(
            self.ledger.get_margin_reservation(
                submitted.intent_id
            ).state,
            "ACTIVE",
        )

        pending_reader = FakeReader(self.clock, self.account)
        pending_reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "OPEN",
            self.clock.now,
            200,
            "2" * 64,
            order_payload_hash(),
            True,
        )
        restarted, _, _ = self.restart(reader=pending_reader)
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(self.harness.count("/orders/cancel"), 1)

        self.clock.advance(1)
        terminal_reader = FakeReader(self.clock, self.account)
        terminal_reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "CANCELLED",
            self.clock.now,
            200,
            "3" * 64,
            order_payload_hash(),
            True,
        )
        resolved, resolved_ledger, _ = self.restart(
            reader=terminal_reader
        )
        resolved.start()
        self.assertEqual(
            resolved_ledger.get_cancellation(
                submitted.intent_id
            ).state,
            "TERMINAL",
        )
        self.assertEqual(
            resolved_ledger.get_margin_reservation(
                submitted.intent_id
            ).state,
            "RELEASED",
        )
        self.assertEqual(self.harness.count("/orders/cancel"), 1)

    def test_ambiguous_cancel_send_is_durable_and_not_retried(self):
        submitted = self.submit_named_order(
            key="cancel-timeout",
            decision="cancel-timeout-decision",
            broker_order_id="95",
        )
        self.reader.query_behaviors["95"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "95",
            "OPEN",
            self.clock.now,
            200,
            "4" * 64,
            order_payload_hash(),
            True,
        )
        self.harness.add(_ExchangeResult("TIMEOUT"))
        command = CancelOpeningCommand(
            intent_id=submitted.intent_id,
            idempotency_key="cancel-timeout-95",
            owner=OWNER,
        )

        result = self.gateway.cancel_opening(command)
        replay = self.gateway.cancel_opening(command)

        self.assertEqual(result.state, "SEND_UNKNOWN")
        self.assertEqual(result.reason_code, "TIMEOUT")
        self.assertEqual(replay.state, "SEND_UNKNOWN")
        self.assertEqual(replay.reason_code, "IDEMPOTENT_REPLAY")
        self.assertEqual(self.harness.count("/orders/cancel"), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM cancel_send_attempts
                    WHERE intent_id = ?
                    """,
                    (submitted.intent_id,),
                ).fetchone()[0],
                1,
            )

    def test_unsent_cancel_lease_can_resume_once_after_restart(self):
        submitted = self.submit_named_order(
            key="cancel-pre-send",
            decision="cancel-pre-send-decision",
            broker_order_id="96",
        )
        self.reader.query_behaviors["96"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "96",
            "OPEN",
            self.clock.now,
            200,
            "5" * 64,
            order_payload_hash(),
            True,
        )
        read = self.reader.query_order(self.account, "96")
        self.ledger.authorize_cancellation(
            submitted.intent_id,
            "cancel-pre-send-96",
            OWNER,
            30,
            read,
        )
        restarted, restarted_ledger, restarted_reader = self.restart()
        restarted_reader.query_behaviors["96"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "96",
            "OPEN",
            self.clock.now,
            200,
            "6" * 64,
            order_payload_hash(),
            True,
        )
        self.harness.add(cancel_result(order_id="96"))

        result = restarted.cancel_opening(
            CancelOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="cancel-pre-send-96",
                owner=OWNER,
            )
        )

        self.assertEqual(result.state, "REQUEST_ACCEPTED")
        self.assertEqual(self.harness.count("/orders/cancel"), 1)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM cancel_send_attempts
                    WHERE intent_id = ?
                    """,
                    (submitted.intent_id,),
                ).fetchone()[0],
                1,
            )
        self.assertEqual(
            restarted_ledger.get_cancellation(
                submitted.intent_id
            ).state,
            "REQUEST_ACCEPTED",
        )

    def test_fake_reader_requires_explicit_test_patch_and_dependencies_are_immutable(
        self,
    ):
        with patch.object(
            gateway_module,
            "ETradeBrokerReader",
            ConcreteETradeBrokerReader,
        ):
            with self.assertRaises(GatewayValidationError):
                EtradeOrderGateway(
                    runtime_safety=self.boundary,
                    ledger=self.ledger,
                    transport=self.transport,
                    reader=FakeReader(self.clock, self.account),
                    opening_risk_budget=Decimal("2000"),
                    clock=self.clock,
                )

        for name, value in (
            ("reader", self.reader),
            ("transport", self.transport),
            ("ledger", self.ledger),
            ("runtime_safety", self.boundary),
            ("_reader", self.reader),
            ("_transport", self.transport),
            ("_ledger", self.ledger),
            ("_runtime_safety", self.boundary),
        ):
            with self.subTest(name=name):
                with self.assertRaises((AttributeError, TypeError)):
                    setattr(self.gateway, name, value)
        with self.assertRaises(AttributeError):
            _ = self.gateway.transport

    def test_no_id_timeout_is_durable_and_cannot_be_auto_reconciled(self):
        self.assertTrue(self.gateway.execution_ready)
        self.harness.add(preview_result(), _ExchangeResult("TIMEOUT"))

        result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(result.reason_code, "TIMEOUT")
        self.assertFalse(self.gateway.execution_ready)
        with self.assertRaises(GatewayReconciliationRequired):
            self.gateway.submit_opening(self.command())
        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_known_executed_response_reconciles_filled_but_absorption_blocks(self):
        self.harness.add(
            preview_result(), place_result(status="EXECUTED")
        )
        unknown = self.gateway.submit_opening(self.command())
        self.assertEqual(unknown.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(unknown.broker_order_id, "94")
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="FILLED",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        # A newer stable portfolio may contain unrelated lots, but it cannot
        # absorb this fill without lots bound to broker order 94.
        reader.capacity_order_ids = ("93",)
        restarted, ledger, _ = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(ledger.get_intent(unknown.intent_id).state, "FILLED")
        self.assertEqual(
            ledger.unabsorbed_filled_reservation_count(
                ACCOUNT_ID, "sandbox"
            ),
            1,
        )

    def test_start_absorbs_zero_fill_terminal_without_capacity_or_mutation(
        self,
    ):
        pending = self.make_pending_terminal(
            key="zero-fill",
            decision="zero-fill-decision",
            broker_order_id="94",
            outcome="CANCELLED",
        )
        mutation_calls = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment="sandbox",
            broker_order_id="94",
            outcome="CANCELLED",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="5" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, reader = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.get_margin_reservation(pending.intent_id).state,
            "RELEASED",
        )
        self.assertEqual(reader.query_calls, [(self.account, "94")])
        self.assertEqual(reader.capacity_calls, [])
        self.assertEqual(len(self.harness.calls), mutation_calls)

    def test_start_absorbs_full_fill_and_restart_is_idempotent(self):
        pending = self.make_pending_terminal(
            key="full-fill",
            decision="full-fill-decision",
            broker_order_id="94",
            outcome="FILLED",
        )
        mutation_calls = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment="sandbox",
            broker_order_id="94",
            outcome="FILLED",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="5" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        reader.capacity_order_ids = ("94",)
        restarted, ledger, reader = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.get_margin_reservation(pending.intent_id).state,
            "RELEASED",
        )
        self.assertEqual(reader.query_calls, [(self.account, "94")])
        self.assertEqual(reader.capacity_calls, [self.account])
        self.assertEqual(len(self.harness.calls), mutation_calls)

        replay_reader = FakeReader(self.clock, self.account)
        replay, replay_ledger, replay_reader = self.restart(
            reader=replay_reader
        )
        replay.start()
        self.assertEqual(replay_reader.query_calls, [])
        self.assertEqual(replay_reader.capacity_calls, [])
        self.assertEqual(
            replay_ledger.get_margin_reservation(pending.intent_id).state,
            "RELEASED",
        )
        self.assertEqual(len(self.harness.calls), mutation_calls)

    def test_active_opening_prevents_a_second_pending_reservation(self):
        first = self.submit_named_order(
            key="terminal-first",
            decision="terminal-first-decision",
            broker_order_id="94",
        )
        self.clock.advance(1)
        with self.assertRaises(OrderIntentReservationError):
            self.submit_named_order(
                key="terminal-second",
                decision="terminal-second-decision",
                broker_order_id="95",
            )

        self.assertEqual(
            self.ledger.get_intent(first.intent_id).state, "SUBMITTED"
        )
        self.assertEqual(
            self.ledger.active_reserved_margin(
                ACCOUNT_ID, "sandbox"
            ),
            Decimal("400"),
        )
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_full_fill_must_be_absorbed_before_next_opening(self):
        first = self.submit_named_order(
            key="full-fill-first",
            decision="full-fill-first-decision",
            broker_order_id="94",
        )
        self.clock.advance(1)
        self.terminalize(first, "94", "FILLED")
        with self.assertRaises(GatewayReconciliationRequired):
            self.gateway.submit_opening(
                self.command(
                    key="full-fill-second-blocked",
                    decision="full-fill-second-blocked-decision",
                )
            )

        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "FILLED",
            self.clock.now,
            200,
            "6" * 64,
            order_payload_hash(),
            True,
        )
        reader.capacity_order_ids = ("94",)
        restarted, ledger, reader = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.get_margin_reservation(first.intent_id).state,
            "RELEASED",
        )
        self.clock.advance(1)
        self.harness.add(
            preview_result(preview_id="1020563295"),
            place_result(order_id="95"),
        )
        second = restarted.submit_opening(
            self.command(
                key="full-fill-second",
                decision="full-fill-second-decision",
            )
        )
        self.assertEqual(second.state, "SUBMITTED")

    def test_unavailable_pending_keeps_execution_blocked(
        self,
    ):
        first = self.submit_named_order(
            key="unavailable-first",
            decision="unavailable-first-decision",
            broker_order_id="94",
        )
        self.terminalize(first, "94", "FILLED")
        mutation_calls = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = ETradeBrokerReaderUnavailable(
            "simulated read outage"
        )
        restarted, ledger, reader = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(
            reader.query_calls, [(self.account, "94")]
        )
        self.assertEqual(
            ledger.get_margin_reservation(first.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        self.assertEqual(len(self.harness.calls), mutation_calls)

    def test_full_fill_without_order_bound_lots_stays_read_only(self):
        pending = self.make_pending_terminal(
            key="missing-lots",
            decision="missing-lots-decision",
            broker_order_id="94",
            outcome="FILLED",
        )
        mutation_calls = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            self.account,
            "sandbox",
            "94",
            "FILLED",
            self.clock.now,
            200,
            "9" * 64,
            order_payload_hash(),
            True,
        )
        reader.capacity_order_ids = ("93",)
        restarted, ledger, reader = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(
            ledger.get_margin_reservation(pending.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        self.assertEqual(reader.capacity_calls, [self.account])
        self.assertEqual(len(self.harness.calls), mutation_calls)

        reader.capacity_order_ids = ("94",)
        restarted.start()
        self.assertEqual(
            ledger.get_margin_reservation(pending.intent_id).state,
            "RELEASED",
        )
        self.assertEqual(len(self.harness.calls), mutation_calls)

    def test_invalid_terminal_read_reference_fails_closed_without_mutation(
        self,
    ):
        pending = self.make_pending_terminal(
            key="invalid-reference",
            decision="invalid-reference-decision",
            broker_order_id="94",
            outcome="CANCELLED",
        )
        mutation_calls = len(self.harness.calls)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = object()
        restarted, ledger, _ = self.restart(reader=reader)

        with self.assertRaises(GatewayValidationError):
            restarted.start()

        self.assertEqual(
            ledger.get_margin_reservation(pending.intent_id).state,
            "FILLED_PENDING_ABSORPTION",
        )
        self.assertEqual(len(self.harness.calls), mutation_calls)

    def test_known_hold_reconciles_open_and_blocks_next_opening(self):
        self.harness.add(
            preview_result(),
            place_result(
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                }
            ),
        )
        unknown = self.gateway.submit_opening(self.command())
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="4" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, reader = self.restart(reader=reader)
        restarted.start()
        self.assertEqual(ledger.get_intent(unknown.intent_id).state, "SUBMITTED")

        self.clock.advance(1)
        reader.capacity_digest = "5" * 64
        with self.assertRaises(OrderIntentReservationError):
            restarted.submit_opening(
                self.command(key="order-2", decision="decision-2")
            )
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_process_crash_after_place_claim_is_never_retried(self):
        self.harness.add(
            preview_result(), SystemExit("simulated crash")
        )

        with self.assertRaises(SystemExit):
            self.gateway.submit_opening(self.command())

        blockers = self.ledger.reconciliation_blockers(
            ACCOUNT_ID, "sandbox"
        )
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_preview_warning_fails_before_place_and_releases_reservation(self):
        self.harness.add(
            preview_result(
                messages={
                    "description": "possible duplicate",
                    "code": 1042,
                    "type": "WARNING",
                }
            )
        )

        result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "FAILED")
        self.assertEqual(result.reason_code, "PREVIEW_REVIEW_REQUIRED")
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            self.ledger.get_margin_reservation(result.intent_id).state,
            "RELEASED",
        )

    def test_lease_expiry_after_preview_resets_without_place(self):
        def expire_after_preview():
            self.reader.selected_hook = lambda: self.clock.advance(31)
            return preview_result()

        self.harness.add(expire_after_preview)

        with self.assertRaises(OrderIntentReconciliationRequired):
            self.gateway.submit_opening(self.command())

        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            self.ledger.reconciliation_blockers(
                ACCOUNT_ID, "sandbox"
            ),
            (),
        )

    def test_runtime_expiry_after_preview_fails_before_place(self):
        boundary = RuntimeSafetyBoundary(
            "production",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(seconds=1),
        )
        ledger = OrderIntentLedger(
            Path(self.temporary.name) / "expiring" / "orders.sqlite3",
            clock=self.clock,
            run_id="expiring-run",
        )
        reader = FakeReader(self.clock, self.account, "production")
        gateway = EtradeOrderGateway(
            runtime_safety=boundary,
            ledger=ledger,
            transport=self.make_transport(ledger, boundary),
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        gateway.start()

        def expire_after_preview():
            reader.selected_hook = lambda: self.clock.advance(2)
            return preview_result()

        self.harness.add(expire_after_preview)
        with self.assertRaises(RuntimeSafetyError):
            gateway.submit_opening(self.command())
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            ledger.active_reserved_margin(ACCOUNT_ID, "production"),
            Decimal("0"),
        )

    def test_quote_expiry_after_preview_fails_before_place(self):
        command = replace(
            self.command(),
            quote_valid_until=self.clock.now + timedelta(seconds=1),
        )

        def expire_quote_after_preview():
            self.reader.selected_hook = lambda: self.clock.advance(2)
            return preview_result()

        self.harness.add(expire_quote_after_preview)
        with self.assertRaisesRegex(
            GatewayValidationError,
            "quote expired before broker placement",
        ):
            self.gateway.submit_opening(command)
        self.assertEqual(self.harness.count("/orders/place"), 0)
        self.assertEqual(
            self.ledger.active_reserved_margin(ACCOUNT_ID, "sandbox"),
            Decimal("0"),
        )

    def test_runtime_expiry_after_place_claim_blocks_before_socket_io(self):
        boundary = RuntimeSafetyBoundary(
            "production",
            ACCOUNT_ID,
            ACCOUNT_KEY,
            INSTITUTION_TYPE,
            self.clock.now - timedelta(seconds=1),
            self.clock.now + timedelta(seconds=1),
        )
        ledger = OrderIntentLedger(
            Path(self.temporary.name) / "post-claim" / "orders.sqlite3",
            clock=self.clock,
            run_id="post-claim-run",
        )
        reader = FakeReader(self.clock, self.account, "production")
        gateway = EtradeOrderGateway(
            runtime_safety=boundary,
            ledger=ledger,
            transport=self.make_transport(ledger, boundary),
            reader=reader,
            opening_risk_budget=Decimal("2000"),
            clock=self.clock,
        )
        gateway.start()
        original = ledger.claim_transport_send

        def claim(request, authorization):
            original(request, authorization)
            if request.transport_operation == "SUBMIT_PLACE":
                self.clock.advance(2)

        self.harness.add(preview_result())
        with patch.object(
            ledger, "claim_transport_send", side_effect=claim
        ):
            with self.assertRaises(RuntimeSafetyError):
                gateway.submit_opening(self.command())

        blockers = ledger.reconciliation_blockers(
            ACCOUNT_ID, "production"
        )
        self.assertEqual(len(blockers), 1)
        self.assertEqual(blockers[0].state, "SUBMISSION_UNKNOWN")
        self.assertEqual(self.harness.count("/orders/place"), 0)

    def test_response_persistence_failure_returns_durable_unknown(self):
        self.harness.add(preview_result(), place_result())
        original = self.ledger.record_transport_response

        def persist(request, response):
            if request.transport_operation == "SUBMIT_PLACE":
                raise RuntimeError("simulated durable write failure")
            return original(request, response)

        with patch.object(
            self.ledger,
            "record_transport_response",
            side_effect=persist,
        ):
            result = self.gateway.submit_opening(self.command())

        self.assertEqual(result.state, "SUBMISSION_UNKNOWN")
        self.assertEqual(
            result.reason_code,
            "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
        )
        self.assertEqual(
            self.ledger.get_intent(result.intent_id).pending_operation,
            "SUBMIT",
        )

    def test_amendment_response_persistence_failure_is_durable_unknown(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        original = self.ledger.record_transport_response

        def persist(request, response):
            if request.transport_operation == "AMEND_PLACE":
                raise RuntimeError("simulated durable write failure")
            return original(request, response)

        with patch.object(
            self.ledger,
            "record_transport_response",
            side_effect=persist,
        ):
            result = self.gateway.reprice_opening(
                RepriceOpeningCommand(
                    intent_id=submitted.intent_id,
                    idempotency_key="reprice-persistence-failure",
                    payload_bytes=payload_bytes(1.50),
                    owner=OWNER,
                )
            )

        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")
        self.assertEqual(
            result.reason_code,
            "TRANSPORT_RESPONSE_PERSISTENCE_ERROR",
        )
        self.assertEqual(
            self.ledger.get_intent(result.intent_id).pending_operation,
            "AMEND",
        )

    def test_successful_reprice_uses_real_change_transport(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(order_id="95"),
        )
        command = RepriceOpeningCommand(
            intent_id=submitted.intent_id,
            idempotency_key="reprice-1",
            payload_bytes=payload_bytes(1.50),
            owner=OWNER,
        )
        result = self.gateway.reprice_opening(command)
        replay = self.gateway.reprice_opening(command)

        self.assertEqual(result.state, "SUBMITTED")
        self.assertEqual(result.broker_order_id, "95")
        self.assertEqual(replay.reason_code, "IDEMPOTENT_REPLAY")
        self.assertFalse(replay.created)
        self.assertEqual(self.harness.count("/change/preview"), 1)
        self.assertEqual(self.harness.count("/change/place"), 1)
        current = self.ledger.get_intent(submitted.intent_id)
        self.assertIsNone(current.pending_operation)
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="9" * 64,
            order_payload_hash=order_payload_hash(1.50),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(
            ledger.expected_order_payload_hash(submitted.intent_id),
            order_payload_hash(1.50),
        )

    def test_no_id_amendment_timeout_remains_hard_blocker(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            _ExchangeResult("TIMEOUT"),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-timeout",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")

        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])

    def test_process_crash_after_amendment_place_claim_is_not_retried(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            SystemExit("simulated amendment crash"),
        )

        with self.assertRaises(SystemExit):
            self.gateway.reprice_opening(
                RepriceOpeningCommand(
                    intent_id=submitted.intent_id,
                    idempotency_key="reprice-crash",
                    payload_bytes=payload_bytes(1.50),
                    owner=OWNER,
                )
            )

        self.assertEqual(
            self.ledger.get_intent(submitted.intent_id).pending_operation,
            "AMEND",
        )
        restarted, _, reader = self.restart()
        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()
        self.assertEqual(reader.query_calls, [])
        self.assertEqual(self.harness.count("/change/place"), 1)

    def test_known_amendment_hold_reconciles_replacement_open(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(
                order_id="95",
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                },
            ),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-hold",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        self.assertEqual(result.state, "AMENDMENT_UNKNOWN")
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="6" * 64,
            order_payload_hash=order_payload_hash(1.50),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        current = ledger.get_intent(submitted.intent_id)
        self.assertEqual(current.broker_order_id, "95")
        self.assertIsNone(current.pending_operation)

    def test_open_order_with_old_terms_does_not_clear_pending_amendment(self):
        submitted = self.submit_success()
        self.harness.add(
            preview_result(preview_id="1020563280"),
            place_result(
                order_id="95",
                messages={
                    "description": "manual review",
                    "code": 4002,
                    "type": "INFO_HOLD",
                },
            ),
        )
        result = self.gateway.reprice_opening(
            RepriceOpeningCommand(
                intent_id=submitted.intent_id,
                idempotency_key="reprice-old-terms",
                payload_bytes=payload_bytes(1.50),
                owner=OWNER,
            )
        )
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["95"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="95",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="8" * 64,
            order_payload_hash=order_payload_hash(1.25),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(
            ledger.get_intent(result.intent_id).pending_operation,
            "AMEND",
        )

    def test_submitted_order_is_revalidated_on_restart(self):
        submitted = self.submit_success()
        reader = FakeReader(self.clock, self.account)
        reader.query_behaviors["94"] = BrokerOrderSnapshot(
            account=self.account,
            environment=reader.environment,
            broker_order_id="94",
            outcome="OPEN",
            observed_at=self.clock.now,
            http_status=200,
            raw_response_digest="7" * 64,
            order_payload_hash=order_payload_hash(),
            complete=True,
        )
        restarted, ledger, _ = self.restart(reader=reader)

        restarted.start()

        self.assertEqual(reader.query_calls[0][1], "94")
        self.assertEqual(
            ledger.get_intent(submitted.intent_id).last_reconciled_run,
            "gateway-run-b",
        )

    def test_execution_readiness_includes_amendment_leases(self):
        submitted = self.submit_success()
        self.assertTrue(self.gateway.execution_ready)
        amendment = self.ledger.acquire_amendment_lease(
            submitted.intent_id,
            "nudger",
            lease_seconds=30,
            idempotency_key="readiness-amend",
            amendment_payload=vertical_payload(1.50),
        )

        self.assertTrue(
            self.ledger.has_execution_blockers(
                ACCOUNT_ID, "sandbox"
            )
        )
        self.assertFalse(self.gateway.execution_ready)

        self.ledger.release_amendment_lease(
            submitted.intent_id,
            "nudger",
            amendment.fencing_token,
        )
        self.assertTrue(self.gateway.execution_ready)

    def test_not_found_known_order_keeps_gateway_read_only(self):
        self.submit_success()
        restarted, _, reader = self.restart()

        with self.assertRaises(GatewayReconciliationRequired):
            restarted.start()

        self.assertEqual(reader.query_calls[0][1], "94")

    def test_incomplete_capacity_blocks_before_any_preview(self):
        self.reader.capacity_complete = False
        command = self.command()

        with self.assertRaises(GatewayValidationError):
            self.gateway.submit_opening(command)

        self.assertEqual(self.harness.calls, [])
        envelope = OrderIntent.build(
            account_id=ACCOUNT_ID,
            environment="sandbox",
            strategy_id=command.strategy_id,
            decision_id=command.decision_id,
            idempotency_scope=command.idempotency_scope,
            idempotency_key=command.idempotency_key,
            intent_kind="OPENING",
            order_payload=json.loads(command.payload_bytes),
        )
        record = self.ledger.find_intent(envelope)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "FAILED")
        self.assertIsNone(
            self.ledger.get_margin_reservation(record.intent_id)
        )
        self.assertEqual(
            self.ledger.reconciliation_blockers(
                ACCOUNT_ID, "sandbox"
            ),
            (),
        )
        self.assertEqual(
            self.gateway.submit_opening(command).state,
            "FAILED",
        )
        self.assertEqual(self.harness.calls, [])

    def test_gateway_owned_risk_ceiling_cannot_be_raised_by_a_command(self):
        limited = EtradeOrderGateway(
            runtime_safety=self.boundary,
            ledger=self.ledger,
            transport=self.transport,
            reader=self.reader,
            opening_risk_budget=Decimal("600"),
            clock=self.clock,
        )
        limited.start()
        self.harness.add(preview_result(), place_result())
        limited.submit_opening(self.command())

        self.clock.advance(1)
        with self.assertRaises(OrderIntentReservationError):
            limited.submit_opening(
                self.command(key="order-2", decision="decision-2")
            )

        self.assertNotIn(
            "risk_budget",
            SubmitOpeningCommand.__dataclass_fields__,
        )
        self.assertEqual(self.harness.count("/orders/place"), 1)

    def test_strict_command_json_rejects_duplicates_and_nonfinite_values(self):
        for raw in (
            b'{"securityType":"OPTN","securityType":"OPTN"}',
            b'{"limitPrice":NaN}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(GatewayValidationError):
                    self.gateway.submit_opening(
                        replace(self.command(), payload_bytes=raw)
                    )
        self.assertEqual(self.harness.calls, [])

    def test_account_or_boundary_mismatch_is_rejected_before_mutation(self):
        other_reader = FakeReader(
            self.clock,
            SelectedBrokerAccount(
                "999999999", "other-key", INSTITUTION_TYPE
            ),
        )
        with self.assertRaises(RuntimeSafetyError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=self.ledger,
                transport=self.transport,
                reader=other_reader,
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            ).start()
        other_ledger = OrderIntentLedger(
            Path(self.temporary.name) / "other" / "orders.sqlite3",
            clock=self.clock,
            run_id="other-run",
        )
        with self.assertRaises(GatewayValidationError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=other_ledger,
                transport=self.transport,
                reader=self.reader,
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            )
        with self.assertRaises(GatewayValidationError):
            EtradeOrderGateway(
                runtime_safety=self.boundary,
                ledger=self.ledger,
                transport=self.transport,
                reader=FakeReader(
                    self.clock, self.account, "production"
                ),
                opening_risk_budget=Decimal("2000"),
                clock=self.clock,
            )
        self.assertEqual(self.harness.calls, [])

    def test_gateway_owns_no_duplicate_post_state_transitions(self):
        source = inspect.getsource(gateway_module)
        self.assertNotIn(".begin_submission(", source)
        self.assertNotIn(".begin_amendment(", source)
        self.assertNotIn(".record_post_acknowledgement(", source)
        self.assertNotIn(".record_amendment_acknowledgement(", source)

    def test_gateway_result_repr_redacts_broker_identifiers(self):
        result = self.submit_success()
        rendered = repr(result)
        self.assertNotIn("client_order_id=", rendered)
        self.assertNotIn("broker_order_id=", rendered)
        self.assertNotIn("preview_id=", rendered)


if __name__ == "__main__":
    unittest.main()
