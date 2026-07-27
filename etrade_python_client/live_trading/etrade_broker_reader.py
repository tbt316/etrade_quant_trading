"""Origin-bound, durable, read-only E*TRADE broker evidence.

Every usable result is assembled from immutable raw-response receipts in the
order ledger.  The reader exposes no generic HTTP method and never retries,
follows broker-provided links, or authorizes a mutation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import quote, urlencode, urlsplit

from rauth import OAuth1Session
from rauth.utils import CaseInsensitiveDict, OAuth1Auth
from requests import Request

from live_trading.etrade_broker_transport import (
    SelectedBrokerAccount,
    _ExchangeResult,
    _isolated_get_exchange,
)
from live_trading.order_intent_ledger import (
    BrokerReadEvidenceRef,
    BrokerReadManifestEvidence,
    BrokerReadManifestMember,
    BrokerReadReceiptRef,
    BrokerReadResponseEvidence,
    OrderIntentLedger,
    canonical_order_payload_hash,
)
from live_trading.runtime_safety import (
    RuntimeSafetyBoundary,
    RuntimeSafetyError,
)


_ETRADE_ORIGINS = {
    "sandbox": "https://apisb.etrade.com",
    "production": "https://api.etrade.com",
}
_TOTAL_EXCHANGE_TIMEOUT_SECONDS = 15.0
_MAX_RAW_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_BALANCE_AGE_SECONDS = 300
_MAX_JSON_NODES = 50_000
_MAX_JSON_DEPTH = 48
_MAX_PORTFOLIO_PAGES = 100
_MAX_POSITIONS = 5_000
_MAX_POSITION_LOTS = 20_000
_MAX_ORDER_PAGES = 100
_MAX_ORDERS = 10_000
_MAX_BROKER_INT64 = 9_223_372_036_854_775_807
_ACTIVE_ORDER_STATUSES = (
    "OPEN",
    "CANCEL_REQUESTED",
    "INDIVIDUAL_FILLS",
)
_PARSER_SCHEMA = "etrade-broker-reader.v2"
_PARSER_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PARSER_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "active_statuses": _ACTIVE_ORDER_STATUSES,
            "max_json_depth": _MAX_JSON_DEPTH,
            "max_json_nodes": _MAX_JSON_NODES,
            "max_order_pages": _MAX_ORDER_PAGES,
            "max_orders": _MAX_ORDERS,
            "max_portfolio_pages": _MAX_PORTFOLIO_PAGES,
            "max_position_lots": _MAX_POSITION_LOTS,
            "max_positions": _MAX_POSITIONS,
            "max_raw_response_bytes": _MAX_RAW_RESPONSE_BYTES,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


class ETradeBrokerReaderError(RuntimeError):
    """Base class for fail-closed broker read failures."""


class ETradeBrokerReaderUnavailable(ETradeBrokerReaderError):
    """The broker did not yield complete usable evidence."""


class ETradeBrokerReaderIntegrityError(ETradeBrokerReaderError):
    """A response or durable receipt violated the closed contract."""


@dataclass(frozen=True)
class _AccountBinding:
    account_id: str
    account_id_key: str
    institution_type: str
    account_status: Literal["ACTIVE"]
    account_mode: Literal["MARGIN"]
    account_type: str


@dataclass(frozen=True)
class _ReadResult:
    parsed: Any
    receipt: BrokerReadReceiptRef
    http_status: int
    raw_response_digest: str
    completed_at: datetime
    completeness: Literal["COMPLETE", "HAS_NEXT", "INELIGIBLE"]


@dataclass(frozen=True)
class _CapacityScan:
    result: dict[str, Any]
    state_sha256: str
    members: tuple[BrokerReadManifestMember, ...]
    completed_at: datetime


class ETradeBrokerReader:
    """Perform exact GETs and return only durable evidence references."""

    def __init__(
        self,
        *,
        session: OAuth1Session,
        ledger: OrderIntentLedger,
        runtime_safety: RuntimeSafetyBoundary,
        selected_account: SelectedBrokerAccount,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(session) is not OAuth1Session:
            raise ETradeBrokerReaderError(
                "broker reader requires an exact rauth OAuth1Session"
            )
        if type(ledger) is not OrderIntentLedger:
            raise ETradeBrokerReaderError(
                "broker reader requires the exact durable ledger"
            )
        if type(runtime_safety) is not RuntimeSafetyBoundary:
            raise ETradeBrokerReaderError(
                "broker reader requires an exact runtime safety boundary"
            )
        if type(selected_account) is not SelectedBrokerAccount:
            raise ETradeBrokerReaderError(
                "broker reader requires an exact selected account"
            )
        if (
            runtime_safety.environment not in _ETRADE_ORIGINS
            or runtime_safety.expected_account_id is None
            or runtime_safety.expected_account_id_key is None
            or runtime_safety.expected_institution_type is None
        ):
            raise ETradeBrokerReaderError(
                "broker reads require an explicitly armed account"
            )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        now = self._now()
        runtime_safety.assert_current(now)
        runtime_safety.verify_account(
            selected_account.runtime_mapping(), now=now
        )
        credentials = (
            session.consumer_key,
            session.consumer_secret,
            session.access_token,
            session.access_token_secret,
        )
        if any(type(value) is not str or not value for value in credentials):
            raise ETradeBrokerReaderError(
                "OAuth session lacks exact consumer/access credentials"
            )
        self._oauth = OAuth1Session(
            session.consumer_key,
            session.consumer_secret,
            access_token=session.access_token,
            access_token_secret=session.access_token_secret,
        )
        self._oauth.trust_env = False
        self._oauth.params = {}
        self._oauth.proxies = {}
        self._oauth.hooks = {"response": []}
        self._oauth.cookies.clear()
        self._oauth.verify = True
        self._oauth.cert = None
        self._ledger = ledger
        self._runtime_safety = runtime_safety
        self._selected_account = selected_account
        self._account_id = selected_account.account_id
        self._account_id_key = selected_account.account_id_key
        self._institution_type = selected_account.institution_type
        self._environment = runtime_safety.environment
        self._origin = _ETRADE_ORIGINS[self._environment]
        self._read_lock = threading.Lock()

    def assert_gateway_binding(
        self,
        ledger: OrderIntentLedger,
        runtime_safety: RuntimeSafetyBoundary,
    ) -> None:
        """Prove the reader shares the exact ledger and runtime boundary."""

        if ledger is not self._ledger or runtime_safety is not self._runtime_safety:
            raise ETradeBrokerReaderError(
                "gateway and reader must share exact durable/runtime boundaries"
            )
        self._assert_runtime()

    def selected_account(self) -> SelectedBrokerAccount:
        return SelectedBrokerAccount(
            self._account_id,
            self._account_id_key,
            self._institution_type,
        )

    def read_capacity(
        self, account: SelectedBrokerAccount
    ) -> BrokerReadEvidenceRef:
        """Return a complete capacity manifest after two identical scans."""

        self._require_account(account)
        with self._read_lock:
            self._assert_runtime()
            binding_start, binding_start_read = self._read_account_binding()
            scan_a = self._read_capacity_scan(binding_start, "scan_a")
            scan_b = self._read_capacity_scan(binding_start, "scan_b")
            binding_end, binding_end_read = self._read_account_binding()
            if binding_start != binding_end:
                raise ETradeBrokerReaderIntegrityError(
                    "account binding changed during capacity read"
                )
            if scan_a.state_sha256 != scan_b.state_sha256:
                raise ETradeBrokerReaderUnavailable(
                    "capacity state changed between complete scans"
                )
            if int(
                scan_b.result["broker_buying_power_as_of"]
            ) < int(scan_a.result["broker_buying_power_as_of"]):
                raise ETradeBrokerReaderIntegrityError(
                    "balance effective time moved backward between scans"
                )
            result = dict(scan_b.result)
            result["state_sha256"] = scan_b.state_sha256
            members = (
                BrokerReadManifestMember(
                    "binding.start",
                    binding_start_read.receipt.receipt_sha256,
                ),
                *scan_a.members,
                *scan_b.members,
                BrokerReadManifestMember(
                    "binding.end",
                    binding_end_read.receipt.receipt_sha256,
                ),
            )
            evidence = self._ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="CAPACITY",
                    account_id=self._account_id,
                    account_id_key=self._account_id_key,
                    institution_type=self._institution_type,
                    environment=self._environment,
                    origin=self._origin,
                    target_broker_order_id=None,
                    observed_at=binding_end_read.completed_at,
                    completeness="COMPLETE",
                    canonical_result_json=_canonical_json(result),
                ),
                tuple(members),
            )
            self._assert_runtime()
            return evidence

    def query_order(
        self,
        account: SelectedBrokerAccount,
        broker_order_id: str,
    ) -> BrokerReadEvidenceRef:
        """Query exactly one already-known broker ID with no list fallback."""

        self._require_account(account)
        target = _broker_int64_text(
            broker_order_id, "broker order id"
        )
        with self._read_lock:
            self._assert_runtime()
            binding_start, binding_start_read = self._read_account_binding()
            order_read = self._read_order_detail(target)
            binding_end, binding_end_read = self._read_account_binding()
            if binding_start != binding_end:
                raise ETradeBrokerReaderIntegrityError(
                    "account binding changed during order query"
                )
            parsed = order_read.parsed
            result = {
                "schema": "etrade-order-query.v2",
                "broker_order_id": target,
                "raw_status": parsed["raw_status"],
                "outcome": parsed["outcome"],
                "fill_summary": parsed["fill_summary"],
                "order_payload_hashes": parsed["order_payload_hashes"],
                "http_status": order_read.http_status,
                "raw_response_digest": order_read.raw_response_digest,
                "not_found": parsed["not_found"],
                "replacement_links": parsed["replacement_links"],
            }
            evidence = self._ledger.record_broker_read_manifest(
                BrokerReadManifestEvidence(
                    evidence_kind="ORDER_QUERY",
                    account_id=self._account_id,
                    account_id_key=self._account_id_key,
                    institution_type=self._institution_type,
                    environment=self._environment,
                    origin=self._origin,
                    target_broker_order_id=target,
                    observed_at=binding_end_read.completed_at,
                    completeness="COMPLETE",
                    canonical_result_json=_canonical_json(result),
                ),
                (
                    BrokerReadManifestMember(
                        "binding.start",
                        binding_start_read.receipt.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        "order.detail",
                        order_read.receipt.receipt_sha256,
                    ),
                    BrokerReadManifestMember(
                        "binding.end",
                        binding_end_read.receipt.receipt_sha256,
                    ),
                ),
            )
            self._assert_runtime()
            return evidence

    def _read_account_binding(
        self,
    ) -> tuple[_AccountBinding, _ReadResult]:
        read = self._get(
            read_kind="ACCOUNT_LIST",
            route="/v1/accounts/list.json",
            query=(),
            target_broker_order_id=None,
            allowed_statuses={200},
            parser=self._parse_account_list,
        )
        parsed = read.parsed
        binding = _AccountBinding(
            account_id=parsed["account_id"],
            account_id_key=parsed["account_id_key"],
            institution_type=parsed["institution_type"],
            account_status=parsed["account_status"],
            account_mode=parsed["account_mode"],
            account_type=parsed["account_type"],
        )
        return binding, read

    def _parse_account_list(
        self, status: int, raw: bytes
    ) -> tuple[dict[str, Any], Literal["COMPLETE"]]:
        if status != 200:
            raise ETradeBrokerReaderUnavailable(
                "account list did not return HTTP 200"
            )
        document = _strict_json(raw)
        root = _exact_object(
            document, {"AccountListResponse"}, "account list root"
        )["AccountListResponse"]
        response = _exact_object(
            root, {"Accounts"}, "AccountListResponse"
        )
        accounts_container = _exact_object(
            response["Accounts"], {"Account"}, "Accounts"
        )
        accounts = accounts_container["Account"]
        if type(accounts) is not list or not accounts:
            raise ETradeBrokerReaderIntegrityError(
                "account list must contain a non-empty Account array"
            )
        matches: list[dict[str, Any]] = []
        for value in accounts:
            account = _object(value, "Account")
            required = {
                "accountId",
                "accountIdKey",
                "institutionType",
                "accountStatus",
                "accountMode",
                "accountType",
            }
            if not required.issubset(account):
                raise ETradeBrokerReaderIntegrityError(
                    "account list entry is missing identity/status fields"
                )
            normalized = {
                "account_id": _broker_int64_text(
                    account["accountId"], "account id"
                ),
                "account_id_key": _ascii_text(
                    account["accountIdKey"], "account key"
                ),
                "institution_type": _ascii_text(
                    account["institutionType"], "institution type"
                ),
                "account_status": _ascii_text(
                    account["accountStatus"], "account status"
                ),
                "account_mode": _ascii_text(
                    account["accountMode"], "account mode"
                ),
                "account_type": _ascii_text(
                    account["accountType"], "account type"
                ),
            }
            if (
                normalized["account_id"] == self._account_id
                and normalized["account_id_key"] == self._account_id_key
                and normalized["institution_type"]
                == self._institution_type
            ):
                matches.append(normalized)
        if len(matches) != 1:
            raise ETradeBrokerReaderIntegrityError(
                "account list did not yield exactly one configured account"
            )
        match = matches[0]
        if (
            match["account_status"] != "ACTIVE"
            or match["account_mode"] != "MARGIN"
            or match["institution_type"] != "BROKERAGE"
        ):
            raise ETradeBrokerReaderUnavailable(
                "configured account is not an ACTIVE MARGIN brokerage account"
            )
        return match, "COMPLETE"

    def _read_capacity_scan(
        self, binding: _AccountBinding, label: str
    ) -> _CapacityScan:
        balance = self._read_balance()
        positions, portfolio_members, portfolio_completed = (
            self._read_portfolio()
        )
        orders, order_members, orders_completed = self._read_active_orders()
        result = {
            "schema": "etrade-capacity.v2",
            "account_status": binding.account_status,
            "account_mode": binding.account_mode,
            "account_type": binding.account_type,
            "broker_buying_power": balance.parsed[
                "margin_buying_power"
            ],
            "broker_buying_power_as_of": balance.parsed["as_of_date"],
            "positions": positions,
            "open_orders": orders,
        }
        economic_state = dict(result)
        economic_state.pop("broker_buying_power_as_of")
        state_sha256 = _domain_json_hash(
            b"etrade-capacity-state.v2\0", economic_state
        )
        members = (
            BrokerReadManifestMember(
                f"{label}.balance", balance.receipt.receipt_sha256
            ),
            *(
                BrokerReadManifestMember(
                    f"{label}.{member.role}", member.receipt_sha256
                )
                for member in portfolio_members
            ),
            *(
                BrokerReadManifestMember(
                    f"{label}.{member.role}", member.receipt_sha256
                )
                for member in order_members
            ),
        )
        return _CapacityScan(
            result=result,
            state_sha256=state_sha256,
            members=tuple(members),
            completed_at=max(
                balance.completed_at,
                portfolio_completed,
                orders_completed,
            ),
        )

    def _read_balance(self) -> _ReadResult:
        encoded_account = quote(self._account_id_key, safe="")
        return self._get(
            read_kind="BALANCE",
            route=f"/v1/accounts/{encoded_account}/balance.json",
            query=(
                ("instType", self._institution_type),
                ("realTimeNAV", "true"),
            ),
            target_broker_order_id=None,
            allowed_statuses={200},
            parser=self._parse_balance,
        )

    def _parse_balance(
        self, status: int, raw: bytes
    ) -> tuple[dict[str, Any], Literal["COMPLETE"]]:
        if status != 200:
            raise ETradeBrokerReaderUnavailable(
                "balance did not return HTTP 200"
            )
        document = _strict_json(raw)
        response = _exact_object(
            document, {"BalanceResponse"}, "balance root"
        )["BalanceResponse"]
        response = _object(response, "BalanceResponse")
        account_id = _broker_int64_text(
            response.get("accountId"), "balance account id"
        )
        if account_id != self._account_id:
            raise ETradeBrokerReaderIntegrityError(
                "balance account id does not match configured account"
            )
        institution = response.get("institutionType")
        if institution is not None and _ascii_text(
            institution, "balance institution type"
        ) != self._institution_type:
            raise ETradeBrokerReaderIntegrityError(
                "balance institution type does not match configured account"
            )
        computed = _object(response.get("Computed"), "BalanceResponse.Computed")
        buying_power = _decimal_text(
            computed.get("marginBuyingPower"),
            "margin buying power",
            nonnegative=True,
        )
        as_of = response.get("asOfDate")
        if as_of is None:
            raise ETradeBrokerReaderIntegrityError(
                "real-time balance omitted its effective timestamp"
            )
        as_of_text = _epoch_milliseconds_text(
            as_of, "balance asOfDate"
        )
        as_of_time = datetime.fromtimestamp(
            int(as_of_text) / 1_000, timezone.utc
        )
        now = self._now()
        if (
            as_of_time > now + timedelta(seconds=5)
            or now - as_of_time
            > timedelta(seconds=_MAX_BALANCE_AGE_SECONDS)
        ):
            raise ETradeBrokerReaderUnavailable(
                "real-time balance effective timestamp is stale or future"
            )
        return {
            "account_id": account_id,
            "institution_type": (
                self._institution_type if institution is not None else None
            ),
            "margin_buying_power": buying_power,
            "as_of_date": as_of_text,
        }, "COMPLETE"

    def _read_portfolio(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        tuple[BrokerReadManifestMember, ...],
        datetime,
    ]:
        encoded_account = quote(self._account_id_key, safe="")
        positions: list[dict[str, Any]] = []
        members: list[BrokerReadManifestMember] = []
        seen_position_ids: set[str] = set()
        seen_position_lot_ids: set[str] = set()
        expected_total_pages: int | None = None
        expected_metadata_field: str | None = None
        page_number = 1
        completed_at: datetime | None = None
        while True:
            if page_number > _MAX_PORTFOLIO_PAGES:
                raise ETradeBrokerReaderIntegrityError(
                    "portfolio pagination exceeded its fixed page bound"
                )
            read = self._get(
                read_kind="PORTFOLIO_PAGE",
                route=f"/v1/accounts/{encoded_account}/portfolio.json",
                query=(
                    ("count", "50"),
                    ("lotsRequired", "true"),
                    ("marketSession", "REGULAR"),
                    ("pageNumber", str(page_number)),
                    ("sortBy", "SYMBOL"),
                    ("sortOrder", "ASC"),
                    ("totalsRequired", "false"),
                    ("view", "COMPLETE"),
                ),
                target_broker_order_id=None,
                allowed_statuses={200, 204},
                parser=lambda status, raw, page=page_number: (
                    self._parse_portfolio_page(
                        status, raw, page, lots_required=True
                    )
                ),
            )
            parsed = read.parsed
            members.append(
                BrokerReadManifestMember(
                    f"portfolio.{page_number:04d}",
                    read.receipt.receipt_sha256,
                )
            )
            completed_at = read.completed_at
            total_pages = parsed["total_pages"]
            metadata_field = parsed["metadata_field"]
            if page_number == 1:
                expected_total_pages = total_pages
                expected_metadata_field = metadata_field
            elif (
                total_pages != expected_total_pages
                or metadata_field != expected_metadata_field
            ):
                raise ETradeBrokerReaderIntegrityError(
                    "portfolio pagination metadata changed between pages"
                )
            for position in parsed["positions"]:
                position_id = position["position_id"]
                if position_id in seen_position_ids:
                    raise ETradeBrokerReaderIntegrityError(
                        "portfolio contains a duplicate position id"
                    )
                seen_position_ids.add(position_id)
                for lot in position["lots"]:
                    lot_id = lot["position_lot_id"]
                    if lot_id in seen_position_lot_ids:
                        raise ETradeBrokerReaderIntegrityError(
                            "portfolio contains a duplicate position lot id"
                        )
                    seen_position_lot_ids.add(lot_id)
                    if len(seen_position_lot_ids) > _MAX_POSITION_LOTS:
                        raise ETradeBrokerReaderIntegrityError(
                            "portfolio exceeded its fixed position-lot bound"
                        )
                positions.append(position)
                if len(positions) > _MAX_POSITIONS:
                    raise ETradeBrokerReaderIntegrityError(
                        "portfolio exceeded its fixed position bound"
                    )
            if read.completeness == "COMPLETE":
                break
            next_page = parsed["next_page"]
            if next_page != page_number + 1:
                raise ETradeBrokerReaderIntegrityError(
                    "portfolio next-page sequence is not contiguous"
                )
            page_number = next_page
        if completed_at is None:
            raise ETradeBrokerReaderIntegrityError(
                "portfolio scan produced no durable page"
            )
        positions.sort(key=lambda item: item["position_id"])
        return positions, tuple(members), completed_at

    def _parse_portfolio_page(
        self,
        status: int,
        raw: bytes,
        page_number: int,
        *,
        lots_required: bool,
    ) -> tuple[
        dict[str, Any], Literal["COMPLETE", "HAS_NEXT"]
    ]:
        if status == 204:
            if page_number != 1 or raw:
                raise ETradeBrokerReaderIntegrityError(
                    "only an empty first-page 204 can represent no positions"
                )
            return {
                "page_number": 1,
                "total_pages": 0,
                "metadata_field": "HTTP_204",
                "next_page": None,
                "positions": [],
            }, "COMPLETE"
        if status != 200:
            raise ETradeBrokerReaderUnavailable(
                "portfolio page did not return HTTP 200/204"
            )
        document = _strict_json(raw)
        response = _exact_object(
            document, {"PortfolioResponse"}, "portfolio root"
        )["PortfolioResponse"]
        response = _object(response, "PortfolioResponse")
        portfolios = response.get("AccountPortfolio")
        if type(portfolios) is not list or len(portfolios) != 1:
            raise ETradeBrokerReaderIntegrityError(
                "portfolio page must contain exactly one AccountPortfolio"
            )
        account_portfolio = _object(
            portfolios[0], "AccountPortfolio"
        )
        account_id = _broker_int64_text(
            account_portfolio.get("accountId"),
            "portfolio account id",
        )
        if account_id != self._account_id:
            raise ETradeBrokerReaderIntegrityError(
                "portfolio page account id does not match"
            )
        total_fields = [
            field
            for field in ("totalNoOfPages", "totalPages")
            if field in account_portfolio
        ]
        if len(total_fields) != 1:
            raise ETradeBrokerReaderIntegrityError(
                "portfolio page must use one reviewed total-pages field"
            )
        metadata_field = total_fields[0]
        total_pages = _positive_int(
            account_portfolio[metadata_field],
            "portfolio total pages",
        )
        if total_pages > _MAX_PORTFOLIO_PAGES or page_number > total_pages:
            raise ETradeBrokerReaderIntegrityError(
                "portfolio total-pages metadata is out of bounds"
            )
        raw_positions = account_portfolio.get("Position", [])
        if type(raw_positions) is not list:
            raise ETradeBrokerReaderIntegrityError(
                "AccountPortfolio.Position must be an array"
            )
        positions = [
            self._normalize_position(
                position, lots_required=lots_required
            )
            for position in raw_positions
        ]
        raw_next_page = account_portfolio.get("nextPageNo")
        if page_number < total_pages:
            next_page = _positive_int(
                raw_next_page, "portfolio next page"
            )
            if next_page != page_number + 1:
                raise ETradeBrokerReaderIntegrityError(
                    "portfolio next page skipped or repeated a page"
                )
            completeness: Literal["COMPLETE", "HAS_NEXT"] = "HAS_NEXT"
        else:
            if raw_next_page not in {None, ""}:
                raise ETradeBrokerReaderIntegrityError(
                    "terminal portfolio page advertised another page"
                )
            next_page = None
            completeness = "COMPLETE"
        positions.sort(key=lambda item: item["position_id"])
        return {
            "page_number": page_number,
            "total_pages": total_pages,
            "metadata_field": metadata_field,
            "next_page": next_page,
            "positions": positions,
        }, completeness

    def _normalize_position(
        self, value: Any, *, lots_required: bool
    ) -> dict[str, Any]:
        position = _object(value, "Position")
        position_id = _broker_int64_text(
            position.get("positionId"), "position id"
        )
        account_id = _broker_int64_text(
            position.get("accountId"), "position account id"
        )
        if account_id != self._account_id:
            raise ETradeBrokerReaderIntegrityError(
                "position account id does not match configured account"
            )
        product = _normalize_product(
            position.get("Product"), "position product"
        )
        quantity = _decimal_text(
            position.get("quantity"),
            "position quantity",
            nonnegative=False,
        )
        position_type = _optional_ascii_text(
            position.get("positionType"), "position type"
        )
        position_indicator = _optional_ascii_text(
            position.get("positionIndicator"), "position indicator"
        )
        osi_key = _optional_ascii_text(
            position.get("osiKey"), "position osi key"
        )
        normalized = {
            "position_id": position_id,
            "account_id": account_id,
            "product": product,
            "quantity": quantity,
            "position_type": position_type,
            "position_indicator": position_indicator,
            "osi_key": osi_key,
        }
        if not lots_required:
            return normalized
        lot_fields = [
            field
            for field in ("PositionLot", "positionLot")
            if field in position
        ]
        if len(lot_fields) > 1:
            raise ETradeBrokerReaderIntegrityError(
                "position uses conflicting PositionLot fields"
            )
        raw_lots = position.get(lot_fields[0], []) if lot_fields else []
        if type(raw_lots) is not list:
            raise ETradeBrokerReaderIntegrityError(
                "PositionLot must be an array"
            )
        lots = [
            _normalize_position_lot(lot, position_id)
            for lot in raw_lots
        ]
        lot_ids = [lot["position_lot_id"] for lot in lots]
        if len(lot_ids) != len(set(lot_ids)):
            raise ETradeBrokerReaderIntegrityError(
                "position contains duplicate position lot ids"
            )
        lots.sort(key=_canonical_json)
        normalized["lots"] = lots
        return normalized

    def _read_active_orders(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        tuple[BrokerReadManifestMember, ...],
        datetime,
    ]:
        encoded_account = quote(self._account_id_key, safe="")
        all_orders: list[dict[str, Any]] = []
        members: list[BrokerReadManifestMember] = []
        seen_order_ids: set[str] = set()
        completed_at: datetime | None = None
        for lane in _ACTIVE_ORDER_STATUSES:
            marker: str | None = None
            seen_markers: set[str] = set()
            page = 0
            while True:
                if page >= _MAX_ORDER_PAGES:
                    raise ETradeBrokerReaderIntegrityError(
                        "order pagination exceeded its fixed page bound"
                    )
                query = [("count", "100"), ("status", lane)]
                if marker is not None:
                    query.append(("marker", marker))
                read = self._get(
                    read_kind="OPEN_ORDERS_PAGE",
                    route=f"/v1/accounts/{encoded_account}/orders.json",
                    query=tuple(query),
                    target_broker_order_id=None,
                    allowed_statuses={200, 204},
                    parser=lambda status, raw, status_lane=lane, first=(
                        marker is None
                    ): self._parse_orders_page(
                        status, raw, status_lane, first
                    ),
                )
                parsed = read.parsed
                members.append(
                    BrokerReadManifestMember(
                        f"orders.{lane}.{page:04d}",
                        read.receipt.receipt_sha256,
                    )
                )
                completed_at = read.completed_at
                for order in parsed["orders"]:
                    order_id = order["order_id"]
                    if order_id in seen_order_ids:
                        raise ETradeBrokerReaderIntegrityError(
                            "active-order scans contain a duplicate order id"
                        )
                    seen_order_ids.add(order_id)
                    all_orders.append(order)
                    if len(all_orders) > _MAX_ORDERS:
                        raise ETradeBrokerReaderIntegrityError(
                            "active orders exceeded the fixed row bound"
                        )
                next_marker = parsed["marker"]
                if read.completeness == "COMPLETE":
                    break
                if (
                    next_marker is None
                    or next_marker in seen_markers
                    or next_marker == marker
                ):
                    raise ETradeBrokerReaderIntegrityError(
                        "order marker pagination repeated or cycled"
                    )
                seen_markers.add(next_marker)
                marker = next_marker
                page += 1
        if completed_at is None:
            raise ETradeBrokerReaderIntegrityError(
                "active-order scan produced no durable page"
            )
        all_orders.sort(key=lambda item: item["order_id"])
        return all_orders, tuple(members), completed_at

    def _parse_orders_page(
        self,
        status: int,
        raw: bytes,
        lane: str,
        first_page: bool,
    ) -> tuple[
        dict[str, Any], Literal["COMPLETE", "HAS_NEXT"]
    ]:
        if status == 204:
            if not first_page or raw:
                raise ETradeBrokerReaderIntegrityError(
                    "only an empty first-page 204 can mean no active orders"
                )
            return {"status_lane": lane, "orders": [], "marker": None}, "COMPLETE"
        if status != 200:
            raise ETradeBrokerReaderUnavailable(
                "orders page did not return HTTP 200/204"
            )
        document = _strict_json(raw)
        response = _exact_object(
            document, {"OrdersResponse"}, "orders root"
        )["OrdersResponse"]
        response = _object(response, "OrdersResponse")
        if any(key.casefold() == "messages" for key in response):
            raise ETradeBrokerReaderIntegrityError(
                "orders page contains broker messages instead of "
                "unambiguous order evidence"
            )
        if "Order" not in response:
            raise ETradeBrokerReaderIntegrityError(
                "HTTP 200 orders page omitted the Order array"
            )
        raw_orders = response["Order"]
        if type(raw_orders) is not list:
            raise ETradeBrokerReaderIntegrityError(
                "OrdersResponse.Order must be an array"
            )
        orders = [
            self._normalize_live_order(order, lane)
            for order in raw_orders
        ]
        order_ids = [order["order_id"] for order in orders]
        if len(order_ids) != len(set(order_ids)):
            raise ETradeBrokerReaderIntegrityError(
                "orders page contains duplicate order ids"
            )
        raw_marker = response.get("marker")
        if raw_marker in {None, ""}:
            marker = None
            completeness: Literal["COMPLETE", "HAS_NEXT"] = "COMPLETE"
        else:
            marker = _ascii_text(raw_marker, "orders marker", maximum=512)
            completeness = "HAS_NEXT"
        orders.sort(key=lambda item: item["order_id"])
        return {
            "status_lane": lane,
            "orders": orders,
            "marker": marker,
        }, completeness

    def _normalize_live_order(
        self, value: Any, lane: str
    ) -> dict[str, Any]:
        order = _object(value, "Order")
        order_id = _broker_int64_text(
            order.get("orderId"), "order id"
        )
        order_type = _ascii_text(
            order.get("orderType"), "order type"
        )
        raw_details = order.get("OrderDetail")
        if type(raw_details) is not list or not raw_details:
            raise ETradeBrokerReaderIntegrityError(
                "active order must contain OrderDetail entries"
            )
        details = [
            self._normalize_generic_order_detail(detail)
            for detail in raw_details
        ]
        if {detail["status"] for detail in details} != {lane}:
            raise ETradeBrokerReaderIntegrityError(
                "active order status does not match its requested lane"
            )
        return {
            "order_id": order_id,
            "order_type": order_type,
            "replaces_order_id": _optional_broker_id(
                order.get("replacesOrderId"), "replaces order id"
            ),
            "replaced_by_order_id": _optional_broker_id(
                order.get("replacedByOrderId"), "replaced-by order id"
            ),
            "details": details,
        }

    def _normalize_generic_order_detail(
        self, value: Any
    ) -> dict[str, Any]:
        detail = _object(value, "OrderDetail")
        raw_account_id = detail.get("accountId")
        account_id = (
            self._account_id
            if raw_account_id is None
            else _broker_int64_text(
                raw_account_id, "order detail account id"
            )
        )
        if raw_account_id is not None and account_id != self._account_id:
            raise ETradeBrokerReaderIntegrityError(
                "order detail account id does not match configured account"
            )
        status = _ascii_text(
            detail.get("status"), "order detail status"
        )
        raw_instruments = detail.get("Instrument")
        if type(raw_instruments) is not list or not raw_instruments:
            raise ETradeBrokerReaderIntegrityError(
                "order detail must contain an Instrument array"
            )
        instruments = [
            _normalize_generic_instrument(instrument)
            for instrument in raw_instruments
        ]
        instruments.sort(key=_canonical_json)
        return {
            "account_id": account_id,
            "status": status,
            "price_type": _optional_ascii_text(
                detail.get("priceType"), "price type"
            ),
            "limit_price": _optional_decimal_text(
                detail.get("limitPrice"), "limit price"
            ),
            "order_term": _optional_ascii_text(
                detail.get("orderTerm"), "order term"
            ),
            "market_session": _optional_ascii_text(
                detail.get("marketSession"), "market session"
            ),
            "all_or_none": _optional_bool(
                detail.get("allOrNone"), "all or none"
            ),
            "replaces_order_id": _optional_broker_id(
                detail.get("replacesOrderId"), "detail replaces order id"
            ),
            "replaced_by_order_id": _optional_broker_id(
                detail.get("replacedByOrderId"),
                "detail replaced-by order id",
            ),
            "instruments": instruments,
        }

    def _read_order_detail(self, target: str) -> _ReadResult:
        encoded_account = quote(self._account_id_key, safe="")
        encoded_order = quote(target, safe="")
        return self._get(
            read_kind="ORDER_DETAIL",
            route=(
                f"/v1/accounts/{encoded_account}/orders/"
                f"{encoded_order}.json"
            ),
            query=(),
            target_broker_order_id=target,
            allowed_statuses={200, 404},
            parser=lambda status, raw: self._parse_order_detail(
                status, raw, target
            ),
        )

    def _parse_order_detail(
        self, status: int, raw: bytes, target: str
    ) -> tuple[dict[str, Any], Literal["COMPLETE"]]:
        if status == 404:
            if raw:
                error = _strict_json(raw)
                if type(error) is not dict:
                    raise ETradeBrokerReaderIntegrityError(
                        "order 404 body must be an object or empty"
                    )
            return {
                "not_found": True,
                "raw_status": "NOT_FOUND",
                "outcome": "UNRESOLVED",
                "fill_summary": {
                    "classification": "UNRESOLVED",
                    "placed_time_epoch_ms": None,
                    "executed_time_epoch_ms": None,
                    "legs": [],
                },
                "order_payload_hashes": [],
                "replacement_links": {},
                "normalized_order": None,
            }, "COMPLETE"
        if status != 200:
            raise ETradeBrokerReaderUnavailable(
                "order detail did not return HTTP 200/404"
            )
        document = _strict_json(raw)
        response = _exact_object(
            document, {"OrdersResponse"}, "order detail root"
        )["OrdersResponse"]
        response = _object(response, "OrdersResponse")
        raw_orders = response.get("Order")
        if type(raw_orders) is not list or len(raw_orders) != 1:
            raise ETradeBrokerReaderIntegrityError(
                "known-order response must contain exactly one Order"
            )
        order = _object(raw_orders[0], "known Order")
        order_id = _broker_int64_text(
            order.get("orderId"), "known order id"
        )
        if order_id != target:
            raise ETradeBrokerReaderIntegrityError(
                "known-order response returned a different order id"
            )
        if _ascii_text(order.get("orderType"), "known order type") != "SPREADS":
            raise ETradeBrokerReaderIntegrityError(
                "known R7 order must be a SPREADS order"
            )
        details = order.get("OrderDetail")
        if type(details) is not list or len(details) != 1:
            raise ETradeBrokerReaderIntegrityError(
                "known R7 order must contain exactly one OrderDetail"
            )
        normalized, fill_summary, hashes, replacement_links = (
            self._normalize_known_vertical(order, details[0])
        )
        classification = fill_summary["classification"]
        if classification == "OPEN":
            outcome = "OPEN"
        elif classification == "FULL_FILL":
            outcome = "FILLED"
        elif classification == "ZERO_FILL_TERMINAL":
            outcome = normalized["status"]
        else:
            outcome = "UNRESOLVED"
        return {
            "not_found": False,
            "raw_status": normalized["status"],
            "outcome": outcome,
            "fill_summary": fill_summary,
            "order_payload_hashes": hashes,
            "replacement_links": replacement_links,
            "normalized_order": normalized,
        }, "COMPLETE"

    def _normalize_known_vertical(
        self, order: dict[str, Any], raw_detail: Any
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[str],
        dict[str, str | None],
    ]:
        detail = _object(raw_detail, "known OrderDetail")
        raw_account_id = detail.get("accountId")
        account_id = (
            self._account_id
            if raw_account_id is None
            else _broker_int64_text(
                raw_account_id, "known order account id"
            )
        )
        if raw_account_id is not None and account_id != self._account_id:
            raise ETradeBrokerReaderIntegrityError(
                "known order account id does not match"
            )
        order_number = detail.get("orderNumber")
        if (
            order_number is not None
            and _broker_int64_text(
                order_number, "known order number"
            )
            != _broker_int64_text(order["orderId"], "known order id")
        ):
            raise ETradeBrokerReaderIntegrityError(
                "known order number conflicts with outer order id"
            )
        status = _ascii_text(detail.get("status"), "known order status")
        placed_time = _optional_epoch_milliseconds_text(
            detail.get("placedTime"), "known order placed time"
        )
        executed_time = _optional_epoch_milliseconds_text(
            detail.get("executedTime"), "known order executed time"
        )
        if (
            placed_time is not None
            and executed_time is not None
            and int(executed_time) < int(placed_time)
        ):
            raise ETradeBrokerReaderIntegrityError(
                "known order executed time precedes placed time"
            )
        price_type = _ascii_text(
            detail.get("priceType"), "known order price type"
        )
        if price_type not in {"NET_CREDIT", "NET_DEBIT"}:
            raise ETradeBrokerReaderIntegrityError(
                "known vertical has an unsupported price type"
            )
        limit_price = _decimal_text(
            detail.get("limitPrice"),
            "known order limit price",
            nonnegative=price_type == "NET_CREDIT",
            positive=price_type == "NET_DEBIT",
        )
        if _ascii_text(
            detail.get("orderTerm"), "known order term"
        ) != "GOOD_FOR_DAY":
            raise ETradeBrokerReaderIntegrityError(
                "known vertical order term changed"
            )
        if _ascii_text(
            detail.get("marketSession"), "known market session"
        ) != "REGULAR":
            raise ETradeBrokerReaderIntegrityError(
                "known vertical market session changed"
            )
        if _bool(detail.get("allOrNone"), "known all-or-none") is not False:
            raise ETradeBrokerReaderIntegrityError(
                "known vertical all-or-none semantics changed"
            )
        stop_price = detail.get("stopPrice")
        if (
            stop_price is not None
            and _decimal_text(
                stop_price, "known stop price", nonnegative=True
            )
            != "0"
        ):
            raise ETradeBrokerReaderIntegrityError(
                "known vertical stop price must remain zero"
            )
        raw_instruments = detail.get("Instrument")
        if type(raw_instruments) is not list or len(raw_instruments) != 2:
            raise ETradeBrokerReaderIntegrityError(
                "known vertical must contain exactly two option legs"
            )
        legs = [
            _normalize_known_leg(instrument, leg_number=index)
            for index, instrument in enumerate(raw_instruments, start=1)
        ]
        if {leg["legNumber"] for leg in legs} != {1, 2}:
            raise ETradeBrokerReaderIntegrityError(
                "known vertical leg numbers must be exactly 1 and 2"
            )
        reference = (
            legs[0]["symbol"],
            legs[0]["callPut"],
            legs[0]["expiryYear"],
            legs[0]["expiryMonth"],
            legs[0]["expiryDay"],
        )
        if any(
            (
                leg["symbol"],
                leg["callPut"],
                leg["expiryYear"],
                leg["expiryMonth"],
                leg["expiryDay"],
            )
            != reference
            for leg in legs
        ):
            raise ETradeBrokerReaderIntegrityError(
                "known vertical leg identity is inconsistent"
            )
        balanced = len({leg["quantity"] for leg in legs}) == 1
        if (
            {leg["orderAction"] for leg in legs}
            != {"BUY_OPEN", "SELL_OPEN"}
            or len({leg["strikePrice"] for leg in legs}) != 2
        ):
            raise ETradeBrokerReaderIntegrityError(
                "known vertical is not one buy-open and one sell-open"
            )
        payload_base = {
            "securityType": "OPTN",
            "orderAction": "SPREAD",
            "priceType": price_type,
            "limitPrice": Decimal(limit_price),
            "orderTerm": "GOOD_FOR_DAY",
            "spreadType": "VERTICAL",
        }
        payload_hashes = (
            sorted(
                {
                    canonical_order_payload_hash(
                        {
                            **payload_base,
                            "legs": [
                                {
                                    "symbol": leg["symbol"],
                                    "callPut": leg["callPut"],
                                    "expiryYear": int(
                                        leg["expiryYear"]
                                    ),
                                    "expiryMonth": int(
                                        leg["expiryMonth"]
                                    ),
                                    "expiryDay": int(leg["expiryDay"]),
                                    "strikePrice": Decimal(
                                        leg["strikePrice"]
                                    ),
                                    "orderAction": leg["orderAction"],
                                    "quantity": int(leg["quantity"]),
                                }
                                for leg in permutation
                            ],
                        }
                    )
                    for permutation in (
                        legs,
                        list(reversed(legs)),
                    )
                }
            )
            if balanced
            else []
        )
        fills_complete = all(
            Decimal(leg["filledQuantity"])
            == Decimal(leg["quantity"])
            and Decimal(leg["cancelQuantity"]) == 0
            for leg in legs
        )
        all_zero_fill = all(
            Decimal(leg["filledQuantity"]) == 0 for leg in legs
        )
        all_zero_cancel = all(
            Decimal(leg["cancelQuantity"]) == 0 for leg in legs
        )
        cancel_arithmetic_complete = all(
            Decimal(leg["filledQuantity"])
            + Decimal(leg["cancelQuantity"])
            == Decimal(leg["quantity"])
            for leg in legs
        )
        replacement_links = {
            "replaces_order_id": _first_optional_broker_id(
                order.get("replacesOrderId"),
                detail.get("replacesOrderId"),
                "replaces order id",
            ),
            "replaced_by_order_id": _first_optional_broker_id(
                order.get("replacedByOrderId"),
                detail.get("replacedByOrderId"),
                "replaced-by order id",
            ),
        }
        has_replacement_link = any(
            value is not None for value in replacement_links.values()
        )
        if has_replacement_link:
            classification = "UNRESOLVED"
        elif (
            status == "OPEN"
            and balanced
            and all_zero_fill
            and all_zero_cancel
            and executed_time is None
        ):
            classification = "OPEN"
        elif (
            status == "EXECUTED"
            and balanced
            and fills_complete
            and executed_time is not None
        ):
            classification = "FULL_FILL"
        elif (
            status in {"CANCELLED", "REJECTED", "EXPIRED"}
            and balanced
            and all_zero_fill
            and cancel_arithmetic_complete
            and executed_time is None
        ):
            classification = "ZERO_FILL_TERMINAL"
        else:
            classification = "UNRESOLVED"
        fill_legs = _canonical_fill_summary_legs(legs)
        fill_summary = {
            "classification": classification,
            "placed_time_epoch_ms": placed_time,
            "executed_time_epoch_ms": executed_time,
            "legs": fill_legs,
        }
        return {
            "account_id": account_id,
            "status": status,
            "price_type": price_type,
            "limit_price": limit_price,
            "order_term": "GOOD_FOR_DAY",
            "market_session": "REGULAR",
            "all_or_none": False,
            "stop_price": (
                "0"
                if stop_price is None
                else _decimal_text(
                    stop_price, "known stop price", nonnegative=True
                )
            ),
            "legs": legs,
            "placed_time_epoch_ms": placed_time,
            "executed_time_epoch_ms": executed_time,
        }, fill_summary, payload_hashes, replacement_links

    def _get(
        self,
        *,
        read_kind: Literal[
            "ACCOUNT_LIST",
            "BALANCE",
            "PORTFOLIO_PAGE",
            "OPEN_ORDERS_PAGE",
            "ORDER_DETAIL",
        ],
        route: str,
        query: tuple[tuple[str, str], ...],
        target_broker_order_id: str | None,
        allowed_statuses: set[int],
        parser: Callable[
            [int, bytes],
            tuple[
                Any,
                Literal["COMPLETE", "HAS_NEXT", "INELIGIBLE"],
            ],
        ],
    ) -> _ReadResult:
        self._assert_runtime()
        canonical_query = tuple(sorted(query))
        if (
            len({name for name, _value in canonical_query})
            != len(canonical_query)
            or any(
                type(name) is not str
                or type(value) is not str
                or not name
                for name, value in canonical_query
            )
        ):
            raise ETradeBrokerReaderIntegrityError(
                "broker read query is not a unique string mapping"
            )
        prepared, expected_url = _prepare_oauth_get(
            self._oauth,
            self._origin,
            route,
            canonical_query,
        )
        _validate_prepared_get(
            prepared,
            expected_url,
            self._origin,
            route,
            canonical_query,
        )
        authorization = prepared.headers["Authorization"]
        if type(authorization) is not str or not authorization.isascii():
            raise ETradeBrokerReaderIntegrityError(
                "prepared OAuth authorization is not bounded ASCII"
            )
        authorization_sha256 = hashlib.sha256(
            authorization.encode("ascii")
        ).hexdigest()
        started_at = self._now()
        self._assert_runtime()
        try:
            exchange = _isolated_get_exchange(
                prepared,
                timeout_seconds=_TOTAL_EXCHANGE_TIMEOUT_SECONDS,
                max_response_bytes=_MAX_RAW_RESPONSE_BYTES,
            )
        except Exception as exc:
            raise ETradeBrokerReaderUnavailable(
                "isolated broker GET failed"
            ) from exc
        completed_at = self._now()
        self._assert_runtime()
        if type(exchange) is not _ExchangeResult or exchange.kind != "RESPONSE":
            raise ETradeBrokerReaderUnavailable(
                "isolated broker GET did not return a complete response"
            )
        status = exchange.http_status
        raw = exchange.raw_response
        if (
            type(status) is not int
            or type(raw) is not bytes
            or len(raw) > _MAX_RAW_RESPONSE_BYTES
        ):
            raise ETradeBrokerReaderIntegrityError(
                "isolated broker GET returned an invalid result"
            )
        parsed: Any = None
        completeness: Literal[
            "COMPLETE", "HAS_NEXT", "INELIGIBLE"
        ] = "INELIGIBLE"
        parse_error: Exception | None = None
        if status not in allowed_statuses:
            parse_error = ETradeBrokerReaderUnavailable(
                "broker GET returned an ineligible HTTP status"
            )
        else:
            try:
                parsed, completeness = parser(status, raw)
                _canonical_json(parsed)
            except Exception as exc:
                parse_error = exc
                parsed = None
                completeness = "INELIGIBLE"
        try:
            receipt = self._ledger.record_broker_read_response(
                BrokerReadResponseEvidence(
                    read_kind=read_kind,
                    account_id=self._account_id,
                    account_id_key=self._account_id_key,
                    institution_type=self._institution_type,
                    environment=self._environment,
                    origin=self._origin,
                    route=route,
                    query_json=_canonical_json(
                        [list(pair) for pair in canonical_query]
                    ),
                    authorization_sha256=authorization_sha256,
                    target_broker_order_id=target_broker_order_id,
                    request_started_at=started_at,
                    response_completed_at=completed_at,
                    http_status=status,
                    raw_response_bytes=raw,
                    parser_schema=_PARSER_SCHEMA,
                    parser_code_sha256=_PARSER_CODE_SHA256,
                    parser_config_sha256=_PARSER_CONFIG_SHA256,
                    canonical_parsed_json=_canonical_json(parsed),
                    completeness=completeness,
                )
            )
        except Exception as exc:
            raise ETradeBrokerReaderIntegrityError(
                "broker GET response could not be persisted durably"
            ) from exc
        if parse_error is not None:
            if isinstance(parse_error, ETradeBrokerReaderError):
                raise parse_error
            raise ETradeBrokerReaderIntegrityError(
                "broker GET response violated its parser contract"
            ) from parse_error
        self._assert_runtime()
        return _ReadResult(
            parsed=parsed,
            receipt=receipt,
            http_status=status,
            raw_response_digest=hashlib.sha256(raw).hexdigest(),
            completed_at=completed_at,
            completeness=completeness,
        )

    def _require_account(self, account: SelectedBrokerAccount) -> None:
        if type(account) is not SelectedBrokerAccount:
            raise ETradeBrokerReaderError(
                "reader account must use the exact selected account type"
            )
        if account != self._selected_account:
            raise ETradeBrokerReaderError(
                "reader account does not match its immutable binding"
            )
        self._assert_runtime()

    def _assert_runtime(self) -> None:
        now = self._now()
        try:
            self._runtime_safety.assert_current(now)
            self._runtime_safety.verify_account(
                self._selected_account.runtime_mapping(), now=now
            )
        except RuntimeSafetyError as exc:
            raise ETradeBrokerReaderError(
                "broker reader runtime safety boundary is not current"
            ) from exc

    def _now(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is not timezone.utc:
            raise ETradeBrokerReaderError(
                "broker reader clock must use the exact UTC timezone"
            )
        return value


def _reparse_broker_read_response(
    evidence: BrokerReadResponseEvidence,
) -> tuple[str, Literal["COMPLETE", "HAS_NEXT", "INELIGIBLE"]]:
    """Reproduce persisted parser output from raw bytes inside the ledger."""

    if type(evidence) is not BrokerReadResponseEvidence:
        raise ETradeBrokerReaderIntegrityError(
            "durable parser requires exact broker-read evidence"
        )
    try:
        query_pairs = json.loads(evidence.query_json)
        if (
            type(query_pairs) is not list
            or any(
                type(pair) is not list
                or len(pair) != 2
                or any(type(item) is not str for item in pair)
                for pair in query_pairs
            )
        ):
            raise ETradeBrokerReaderIntegrityError(
                "durable parser query is invalid"
            )
        query = {pair[0]: pair[1] for pair in query_pairs}
        parser = object.__new__(ETradeBrokerReader)
        parser._account_id = evidence.account_id
        parser._account_id_key = evidence.account_id_key
        parser._institution_type = evidence.institution_type
        parser._clock = lambda: evidence.response_completed_at
        if evidence.read_kind == "ACCOUNT_LIST":
            parsed, completeness = parser._parse_account_list(
                evidence.http_status, evidence.raw_response_bytes
            )
        elif evidence.read_kind == "BALANCE":
            parsed, completeness = parser._parse_balance(
                evidence.http_status, evidence.raw_response_bytes
            )
        elif evidence.read_kind == "PORTFOLIO_PAGE":
            page_number = _positive_int(
                query.get("pageNumber"),
                "durable portfolio page number",
            )
            raw_lots_required = query.get("lotsRequired")
            if raw_lots_required == "true":
                lots_required = True
            elif raw_lots_required == "false":
                lots_required = False
            else:
                raise ETradeBrokerReaderIntegrityError(
                    "durable portfolio lotsRequired query is unsupported"
                )
            parsed, completeness = parser._parse_portfolio_page(
                evidence.http_status,
                evidence.raw_response_bytes,
                page_number,
                lots_required=lots_required,
            )
        elif evidence.read_kind == "OPEN_ORDERS_PAGE":
            lane = _ascii_text(
                query.get("status"), "durable order status lane"
            )
            if lane not in _ACTIVE_ORDER_STATUSES:
                raise ETradeBrokerReaderIntegrityError(
                    "durable order status lane is unsupported"
                )
            parsed, completeness = parser._parse_orders_page(
                evidence.http_status,
                evidence.raw_response_bytes,
                lane,
                "marker" not in query,
            )
        elif evidence.read_kind == "ORDER_DETAIL":
            target = _broker_int64_text(
                evidence.target_broker_order_id,
                "durable target broker order id",
            )
            parsed, completeness = parser._parse_order_detail(
                evidence.http_status,
                evidence.raw_response_bytes,
                target,
            )
        else:
            raise ETradeBrokerReaderIntegrityError(
                "durable broker-read kind is unsupported"
            )
        return _canonical_json(parsed), completeness
    except Exception:
        return _canonical_json(None), "INELIGIBLE"


def _prepare_oauth_get(
    oauth: OAuth1Session,
    origin: str,
    route: str,
    query: tuple[tuple[str, str], ...],
):
    if (
        type(oauth) is not OAuth1Session
        or oauth.trust_env is not False
        or oauth.params
        or oauth.proxies
        or oauth.hooks != {"response": []}
        or oauth.cookies
        or oauth.verify is not True
        or oauth.cert is not None
        or origin not in _ETRADE_ORIGINS.values()
        or type(route) is not str
        or not route.startswith("/v1/accounts/")
        or "?" in route
        or "#" in route
    ):
        raise ETradeBrokerReaderError(
            "private OAuth reader configuration is unsafe"
        )
    params = {name: value for name, value in query}
    base_url = origin + route
    encoded_query = urlencode(params)
    expected_url = (
        base_url if not encoded_query else f"{base_url}?{encoded_query}"
    )
    headers = CaseInsensitiveDict(
        {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "consumerKey": oauth.consumer_key,
        }
    )
    signing_kwargs = {"headers": headers, "params": params}
    oauth_params = oauth._get_oauth_params(signing_kwargs)
    oauth_params["oauth_signature"] = oauth.signature.sign(
        oauth.consumer_secret,
        oauth.access_token_secret,
        "GET",
        base_url,
        oauth_params,
        signing_kwargs,
    )
    prepared = Request(
        method="GET",
        url=base_url,
        headers=dict(headers),
        params=params,
        auth=OAuth1Auth(oauth_params, ""),
    ).prepare()
    return prepared, expected_url


def _validate_prepared_get(
    prepared: Any,
    expected_url: str,
    expected_origin: str,
    expected_route: str,
    expected_query: tuple[tuple[str, str], ...],
) -> None:
    parsed = urlsplit(getattr(prepared, "url", ""))
    headers = getattr(prepared, "headers", None)
    if not hasattr(headers, "items"):
        raise ETradeBrokerReaderError(
            "prepared broker GET headers are invalid"
        )
    normalized = {
        name.lower(): value for name, value in headers.items()
    }
    expected_headers = {
        "accept",
        "accept-encoding",
        "consumerkey",
        "authorization",
    }
    if (
        getattr(prepared, "method", None) != "GET"
        or prepared.url != expected_url
        or f"{parsed.scheme}://{parsed.netloc}" != expected_origin
        or parsed.path != expected_route
        or parsed.fragment
        or getattr(prepared, "body", None) is not None
        or set(normalized) != expected_headers
        or normalized["accept"] != "application/json"
        or normalized["accept-encoding"] != "identity"
        or not normalized["consumerkey"]
        or not normalized["authorization"].startswith("OAuth ")
        or "cookie" in normalized
    ):
        raise ETradeBrokerReaderError(
            "prepared OAuth GET changed its pinned request"
        )
    params = {name: value for name, value in expected_query}
    if parsed.query != urlencode(params):
        raise ETradeBrokerReaderError(
            "prepared OAuth GET query is not canonical"
        )


def _strict_json(raw: bytes) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > _MAX_RAW_RESPONSE_BYTES
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        raise ETradeBrokerReaderIntegrityError(
            "broker JSON bytes are empty, oversized, or BOM-prefixed"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ETradeBrokerReaderIntegrityError(
            "broker JSON is not strict UTF-8"
        ) from exc

    def strict_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise ETradeBrokerReaderIntegrityError(
                    "broker JSON contains a duplicate key"
                )
            result[key] = value
        return result

    def parse_number(token: str) -> str:
        if type(token) is not str or len(token) > 128:
            raise ETradeBrokerReaderIntegrityError(
                "broker JSON number is out of bounds"
            )
        try:
            number = Decimal(token)
        except InvalidOperation as exc:
            raise ETradeBrokerReaderIntegrityError(
                "broker JSON number is invalid"
            ) from exc
        if (
            not number.is_finite()
            or number.adjusted() > 100
            or number.adjusted() < -100
        ):
            raise ETradeBrokerReaderIntegrityError(
                "broker JSON number magnitude is out of bounds"
            )
        return token

    def reject_constant(_token: str) -> Any:
        raise ETradeBrokerReaderIntegrityError(
            "broker JSON contains a non-finite number"
        )

    try:
        document = json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_int=parse_number,
            parse_float=parse_number,
            parse_constant=reject_constant,
        )
    except ETradeBrokerReaderError:
        raise
    except Exception as exc:
        raise ETradeBrokerReaderIntegrityError(
            "broker response is not valid JSON"
        ) from exc
    _validate_json_tree(document)
    if type(document) is not dict:
        raise ETradeBrokerReaderIntegrityError(
            "broker JSON root must be an object"
        )
    return document


def _validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ETradeBrokerReaderIntegrityError(
                "broker JSON exceeds fixed complexity bounds"
            )
        item_type = type(item)
        if item is None or item_type in {str, bool}:
            if item_type is str and len(item) > _MAX_RAW_RESPONSE_BYTES:
                raise ETradeBrokerReaderIntegrityError(
                    "broker JSON string exceeds its fixed bound"
                )
            continue
        if item_type is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if item_type is dict:
            if any(type(key) is not str for key in item):
                raise ETradeBrokerReaderIntegrityError(
                    "broker JSON keys must be strings"
                )
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise ETradeBrokerReaderIntegrityError(
            "broker JSON contains an unsupported scalar"
        )


def _canonical_json(value: Any) -> str:
    _validate_canonical_tree(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _validate_canonical_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ETradeBrokerReaderIntegrityError(
                "normalized broker evidence exceeds complexity bounds"
            )
        item_type = type(item)
        if item is None or item_type in {str, bool, int}:
            continue
        if item_type is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if item_type is dict:
            if any(type(key) is not str for key in item):
                raise ETradeBrokerReaderIntegrityError(
                    "normalized broker evidence keys must be strings"
                )
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise ETradeBrokerReaderIntegrityError(
            "normalized broker evidence contains an unsupported scalar"
        )


def _domain_json_hash(domain: bytes, value: Any) -> str:
    if type(domain) is not bytes or not domain.endswith(b"\0"):
        raise ETradeBrokerReaderIntegrityError(
            "reader hash domain is invalid"
        )
    return hashlib.sha256(
        domain + _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _exact_object(
    value: Any, keys: set[str], label: str
) -> dict[str, Any]:
    result = _object(value, label)
    if set(result) != keys:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} has an unexpected shape"
        )
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be an object"
        )
    return value


def _ascii_text(
    value: Any, label: str, *, maximum: int = 256
) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be bounded printable ASCII"
        )
    return value


def _optional_ascii_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _ascii_text(value, label)


def _integer_text(
    value: Any,
    label: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> str:
    if type(value) is int:
        text = str(value)
    elif type(value) is str:
        text = value
    else:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be an exact integer"
        )
    if (
        not text
        or len(text) > 20
        or text.startswith("+")
        or (text.startswith("-") and not text[1:].isdigit())
        or (not text.startswith("-") and not text.isdigit())
        or (len(text.lstrip("-")) > 1 and text.lstrip("-").startswith("0"))
    ):
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be a canonical integer"
        )
    number = int(text)
    if positive and number <= 0:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be positive"
        )
    if nonnegative and number < 0:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be non-negative"
        )
    return str(number)


def _positive_int(value: Any, label: str) -> int:
    return int(_integer_text(value, label, positive=True))


def _integral_decimal_text(
    value: Any,
    label: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> str:
    text = _decimal_text(
        value,
        label,
        nonnegative=nonnegative,
        positive=positive,
    )
    number = Decimal(text)
    if number != number.to_integral_value():
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be an integral quantity"
        )
    return str(int(number))


def _epoch_milliseconds_text(value: Any, label: str) -> str:
    text = _integer_text(value, label, positive=True)
    if len(text) != 13:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be Unix epoch milliseconds"
        )
    return text


def _broker_int64_text(value: Any, label: str) -> str:
    text = _integer_text(value, label, positive=True)
    if int(text) > _MAX_BROKER_INT64:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} exceeds signed broker int64"
        )
    return text


def _signed_broker_int64_text(value: Any, label: str) -> str:
    text = _integer_text(value, label)
    number = int(text)
    if (
        number < -_MAX_BROKER_INT64 - 1
        or number > _MAX_BROKER_INT64
    ):
        raise ETradeBrokerReaderIntegrityError(
            f"{label} exceeds signed broker int64"
        )
    return text


def _optional_nonnegative_broker_int64_text(
    value: Any, label: str
) -> str | None:
    if value is None:
        return None
    text = _integer_text(value, label, nonnegative=True)
    if int(text) > _MAX_BROKER_INT64:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} exceeds signed broker int64"
        )
    return text


def _optional_broker_id(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _broker_int64_text(value, label)


def _optional_epoch_milliseconds_text(
    value: Any, label: str
) -> str | None:
    if value is None:
        return None
    return _epoch_milliseconds_text(value, label)


def _first_optional_broker_id(
    first: Any, second: Any, label: str
) -> str | None:
    values = [
        _optional_broker_id(value, label)
        for value in (first, second)
        if value is not None
    ]
    if not values:
        return None
    if len(set(values)) != 1:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} conflicts across order/detail"
        )
    return values[0]


def _decimal_text(
    value: Any,
    label: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> str:
    if type(value) not in {str, int, Decimal}:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} must be a decimal-compatible primitive"
        )
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ETradeBrokerReaderIntegrityError(
            f"{label} is not a valid decimal"
        ) from exc
    if (
        not number.is_finite()
        or number.adjusted() > 100
        or number.adjusted() < -100
        or positive
        and number <= 0
        or nonnegative
        and number < 0
    ):
        raise ETradeBrokerReaderIntegrityError(
            f"{label} is outside its finite bounds"
        )
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _optional_decimal_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _decimal_text(value, label)


def _bool(value: Any, label: str) -> bool:
    if type(value) is bool:
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ETradeBrokerReaderIntegrityError(
        f"{label} must be an exact boolean"
    )


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, label)


def _normalize_product(value: Any, label: str) -> dict[str, Any]:
    product = _object(value, label)
    symbol = _ascii_text(product.get("symbol"), f"{label} symbol")
    security_type = _ascii_text(
        product.get("securityType"), f"{label} security type"
    )
    product_id = product.get("ProductId", product.get("productId"))
    if product_id is not None:
        product_id_object = _object(product_id, f"{label} product id")
        normalized_product_id = {
            "symbol": _optional_ascii_text(
                product_id_object.get(
                    "symbol", product_id_object.get("Symbol")
                ),
                f"{label} product-id symbol",
            ),
            "type_code": _optional_ascii_text(
                product_id_object.get(
                    "typeCode", product_id_object.get("TypeCode")
                ),
                f"{label} product-id type",
            ),
        }
    else:
        normalized_product_id = None
    return {
        "symbol": symbol,
        "security_type": security_type,
        "call_put": _optional_ascii_text(
            product.get("callPut"), f"{label} call/put"
        ),
        "expiry_year": (
            _integer_text(
                product["expiryYear"],
                f"{label} expiry year",
                positive=True,
            )
            if "expiryYear" in product
            else None
        ),
        "expiry_month": (
            _integer_text(
                product["expiryMonth"],
                f"{label} expiry month",
                positive=True,
            )
            if "expiryMonth" in product
            else None
        ),
        "expiry_day": (
            _integer_text(
                product["expiryDay"],
                f"{label} expiry day",
                positive=True,
            )
            if "expiryDay" in product
            else None
        ),
        "strike_price": _optional_decimal_text(
            product.get("strikePrice"), f"{label} strike price"
        ),
        "product_id": normalized_product_id,
    }


def _normalize_position_lot(
    value: Any, parent_position_id: str
) -> dict[str, Any]:
    lot = _object(value, "PositionLot")
    position_id = _broker_int64_text(
        lot.get("positionId"), "position lot parent position id"
    )
    if position_id != parent_position_id:
        raise ETradeBrokerReaderIntegrityError(
            "position lot parent id does not match its position"
        )
    raw_leg_no = lot.get("legNo")
    if raw_leg_no is None:
        leg_no = None
    else:
        leg_no = _integer_text(
            raw_leg_no,
            "position lot leg number",
            nonnegative=True,
        )
        if int(leg_no) > 2_147_483_647:
            raise ETradeBrokerReaderIntegrityError(
                "position lot leg number exceeds signed int32"
            )
    return {
        "position_id": position_id,
        "position_lot_id": _broker_int64_text(
            lot.get("positionLotId"), "position lot id"
        ),
        "order_no": _optional_nonnegative_broker_int64_text(
            lot.get("orderNo"), "position lot order number"
        ),
        "leg_no": leg_no,
        "original_quantity": _decimal_text(
            lot.get("originalQty"), "position lot original quantity"
        ),
        "remaining_quantity": _decimal_text(
            lot.get("remainingQty"), "position lot remaining quantity"
        ),
        "available_quantity": _decimal_text(
            lot.get("availableQty"), "position lot available quantity"
        ),
        "acquired_date_epoch_ms": _signed_broker_int64_text(
            lot.get("acquiredDate"), "position lot acquired date"
        ),
    }


def _normalize_generic_instrument(value: Any) -> dict[str, Any]:
    instrument = _object(value, "Instrument")
    ordered = instrument.get(
        "orderedQuantity", instrument.get("quantity")
    )
    if ordered is None:
        raise ETradeBrokerReaderIntegrityError(
            "active-order instrument lacks ordered quantity"
        )
    ordered_text = _decimal_text(
        ordered, "instrument ordered quantity", nonnegative=True
    )
    if "filledQuantity" not in instrument:
        raise ETradeBrokerReaderIntegrityError(
            "active-order instrument lacks filled quantity"
        )
    if "cancelQuantity" not in instrument:
        raise ETradeBrokerReaderIntegrityError(
            "active-order instrument lacks cancel quantity"
        )
    filled_text = _decimal_text(
        instrument["filledQuantity"],
        "instrument filled quantity",
        nonnegative=True,
    )
    cancel_text = _decimal_text(
        instrument["cancelQuantity"],
        "instrument cancel quantity",
        nonnegative=True,
    )
    if Decimal(filled_text) + Decimal(cancel_text) > Decimal(ordered_text):
        raise ETradeBrokerReaderIntegrityError(
            "instrument fill/cancel quantities exceed ordered quantity"
        )
    return {
        "product": _normalize_product(
            instrument.get("Product"), "instrument product"
        ),
        "order_action": _ascii_text(
            instrument.get("orderAction"), "instrument order action"
        ),
        "quantity_type": _optional_ascii_text(
            instrument.get("quantityType"), "instrument quantity type"
        ),
        "ordered_quantity": ordered_text,
        "filled_quantity": filled_text,
        "cancel_quantity": cancel_text,
    }


def _normalize_known_leg(
    value: Any, *, leg_number: int
) -> dict[str, Any]:
    if type(leg_number) is not int or leg_number not in {1, 2}:
        raise ETradeBrokerReaderIntegrityError(
            "known vertical leg number must be 1 or 2"
        )
    instrument = _object(value, "known Instrument")
    product = _normalize_product(
        instrument.get("Product"), "known option product"
    )
    if (
        product["security_type"] != "OPTN"
        or product["call_put"] not in {"CALL", "PUT"}
        or any(
            product[field] is None
            for field in (
                "expiry_year",
                "expiry_month",
                "expiry_day",
                "strike_price",
            )
        )
    ):
        raise ETradeBrokerReaderIntegrityError(
            "known vertical leg product is incomplete"
        )
    if _ascii_text(
        instrument.get("quantityType"), "known quantity type"
    ) != "QUANTITY":
        raise ETradeBrokerReaderIntegrityError(
            "known vertical quantity type changed"
        )
    quantity = _integral_decimal_text(
        instrument.get(
            "orderedQuantity", instrument.get("quantity")
        ),
        "known ordered quantity",
        positive=True,
    )
    if "filledQuantity" not in instrument:
        raise ETradeBrokerReaderIntegrityError(
            "known vertical leg lacks filled quantity"
        )
    if "cancelQuantity" not in instrument:
        raise ETradeBrokerReaderIntegrityError(
            "known vertical leg lacks cancel quantity"
        )
    filled = _integral_decimal_text(
        instrument["filledQuantity"],
        "known filled quantity",
        nonnegative=True,
    )
    cancelled = _integral_decimal_text(
        instrument["cancelQuantity"],
        "known cancel quantity",
        nonnegative=True,
    )
    if int(filled) + int(cancelled) > int(quantity):
        raise ETradeBrokerReaderIntegrityError(
            "known filled/cancel quantities exceed ordered quantity"
        )
    action = _ascii_text(
        instrument.get("orderAction"), "known order action"
    )
    if action not in {"BUY_OPEN", "SELL_OPEN"}:
        raise ETradeBrokerReaderIntegrityError(
            "known vertical leg is not opening exposure"
        )
    return {
        "legNumber": leg_number,
        "symbol": product["symbol"],
        "securityType": product["security_type"],
        "callPut": product["call_put"],
        "expiryYear": product["expiry_year"],
        "expiryMonth": product["expiry_month"],
        "expiryDay": product["expiry_day"],
        "strikePrice": product["strike_price"],
        "productId": product["product_id"],
        "orderAction": action,
        "quantityType": "QUANTITY",
        "quantity": quantity,
        "filledQuantity": filled,
        "cancelQuantity": cancelled,
    }


def _canonical_fill_summary_legs(
    legs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        type(legs) is not list
        or len(legs) != 2
        or {leg.get("legNumber") for leg in legs} != {1, 2}
    ):
        raise ETradeBrokerReaderIntegrityError(
            "fill summary requires unique leg numbers 1 and 2"
        )
    result = [
        {
            "leg_number": leg["legNumber"],
            "product": {
                "symbol": leg["symbol"],
                "security_type": leg["securityType"],
                "call_put": leg["callPut"],
                "expiry_year": leg["expiryYear"],
                "expiry_month": leg["expiryMonth"],
                "expiry_day": leg["expiryDay"],
                "strike_price": leg["strikePrice"],
                "product_id": leg["productId"],
            },
            "order_action": leg["orderAction"],
            "ordered_quantity": leg["quantity"],
            "filled_quantity": leg["filledQuantity"],
            "cancel_quantity": leg["cancelQuantity"],
        }
        for leg in legs
    ]
    result.sort(key=_canonical_json)
    return result
