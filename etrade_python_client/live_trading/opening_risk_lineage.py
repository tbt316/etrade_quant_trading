"""Offline, independently replayable opening-risk quote evidence.

This module has no HTTP client and no execution capability.  It parses retained
E*TRADE quote-response bytes into the exact pure pre-trade quote contract.  The
durable ledger owns persistence and replays this parser before accepting a
receipt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

from live_trading.order_domain import OptionContractId, OrderDomainError
from live_trading.pretrade_risk import ContractQuote, QuoteSnapshotEvidence


OPENING_QUOTE_PARSER_SCHEMA = "etrade-opening-quote.v1"
_MAX_RAW_BYTES = 2 * 1024 * 1024
_MAX_NODES = 20_000
_MAX_DEPTH = 32
_ALLOWED_QUOTE_STATUSES = frozenset({"REALTIME"})
OPENING_QUOTE_PARSER_CODE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
OPENING_QUOTE_PARSER_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "allowed_quote_statuses": sorted(_ALLOWED_QUOTE_STATUSES),
            "detail_flag": "ALL",
            "max_depth": _MAX_DEPTH,
            "max_nodes": _MAX_NODES,
            "max_raw_bytes": _MAX_RAW_BYTES,
            "required_multiplier": "100",
            "required_unadjusted": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
).hexdigest()


class OpeningRiskLineageError(ValueError):
    """Retained evidence does not satisfy the closed replay contract."""


@dataclass(frozen=True, slots=True, repr=False)
class OpeningQuoteResponseEvidence:
    account_id: str
    account_id_key: str
    institution_type: str
    environment: str
    origin: str
    route: str
    query_json: str
    authorization_sha256: str
    request_started_at: datetime
    response_completed_at: datetime
    http_status: int
    raw_response_bytes: bytes
    parser_schema: str = OPENING_QUOTE_PARSER_SCHEMA
    parser_code_sha256: str = OPENING_QUOTE_PARSER_CODE_SHA256
    parser_config_sha256: str = OPENING_QUOTE_PARSER_CONFIG_SHA256

    def __post_init__(self) -> None:
        for name in (
            "account_id",
            "account_id_key",
            "institution_type",
            "origin",
            "route",
            "query_json",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 4096:
                raise OpeningRiskLineageError(f"invalid {name}")
        if self.environment not in {"sandbox", "production"}:
            raise OpeningRiskLineageError("invalid environment")
        expected_origin = {
            "sandbox": "https://apisb.etrade.com",
            "production": "https://api.etrade.com",
        }[self.environment]
        if self.origin != expected_origin:
            raise OpeningRiskLineageError("quote origin/environment mismatch")
        if (
            not self.route.startswith("/v1/market/quote/")
            or "?" in self.route
            or "#" in self.route
        ):
            raise OpeningRiskLineageError("invalid quote route")
        _sha256(self.authorization_sha256, "authorization")
        _utc(self.request_started_at, "request start")
        _utc(self.response_completed_at, "response completion")
        if self.response_completed_at < self.request_started_at:
            raise OpeningRiskLineageError("quote response predates request")
        if type(self.http_status) is not int or not 100 <= self.http_status <= 599:
            raise OpeningRiskLineageError("invalid quote HTTP status")
        if (
            type(self.raw_response_bytes) is not bytes
            or len(self.raw_response_bytes) > _MAX_RAW_BYTES
        ):
            raise OpeningRiskLineageError("invalid quote response bytes")
        if self.parser_schema != OPENING_QUOTE_PARSER_SCHEMA:
            raise OpeningRiskLineageError("unsupported quote parser schema")
        _sha256(self.parser_code_sha256, "parser code")
        _sha256(self.parser_config_sha256, "parser config")

    def __repr__(self) -> str:
        return "OpeningQuoteResponseEvidence([REDACTED])"


@dataclass(frozen=True, slots=True)
class OpeningQuoteReceiptRef:
    receipt_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.receipt_sha256, "quote receipt")


@dataclass(frozen=True, slots=True)
class OpeningRiskLineageRef:
    lineage_sha256: str
    intent_id: str
    status: str
    missing_evidence_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.lineage_sha256, "opening risk lineage")
        if type(self.intent_id) is not str or not self.intent_id:
            raise OpeningRiskLineageError("invalid intent id")
        if self.status != "INDEPENDENT_EVIDENCE_PENDING":
            raise OpeningRiskLineageError("lineage cannot authorize execution")
        if (
            type(self.missing_evidence_reasons) is not tuple
            or self.missing_evidence_reasons
            != tuple(sorted(set(self.missing_evidence_reasons)))
        ):
            raise OpeningRiskLineageError("invalid missing-evidence reasons")


def parse_opening_quote_response(
    evidence: OpeningQuoteResponseEvidence,
) -> tuple[QuoteSnapshotEvidence, str]:
    """Replay retained JSON bytes into the exact quote evidence contract."""

    if type(evidence) is not OpeningQuoteResponseEvidence:
        raise OpeningRiskLineageError("exact quote response evidence required")
    if (
        evidence.parser_schema != OPENING_QUOTE_PARSER_SCHEMA
        or evidence.parser_code_sha256 != OPENING_QUOTE_PARSER_CODE_SHA256
        or evidence.parser_config_sha256 != OPENING_QUOTE_PARSER_CONFIG_SHA256
    ):
        raise OpeningRiskLineageError("quote parser provenance is not installed")
    if evidence.http_status != 200:
        raise OpeningRiskLineageError("quote response is not HTTP 200")
    query = _strict_query(evidence.query_json)
    if query != {
        "detailFlag": "ALL",
        "overrideSymbolCount": "false",
        "skipMiniOptionsCheck": "true",
    }:
        raise OpeningRiskLineageError("quote request did not use the fixed query")
    document = _strict_json(evidence.raw_response_bytes)
    if set(document) != {"QuoteResponse"}:
        raise OpeningRiskLineageError("quote response root is not exact")
    response = _object(document["QuoteResponse"], "QuoteResponse")
    if "Messages" in response or "messages" in response:
        raise OpeningRiskLineageError("quote response contains broker messages")
    raw_quotes = response.get("QuoteData")
    if type(raw_quotes) is not list or len(raw_quotes) != 2:
        raise OpeningRiskLineageError("exactly two quote rows are required")
    raw_sha256 = hashlib.sha256(evidence.raw_response_bytes).hexdigest()
    quotes = tuple(
        sorted(
            (
                _normalize_quote(item, raw_sha256)
                for item in raw_quotes
            ),
            key=lambda item: item.contract.canonical_material,
        )
    )
    if quotes[0].contract == quotes[1].contract:
        raise OpeningRiskLineageError("quote response duplicated a contract")
    normalized = {
        "schema": OPENING_QUOTE_PARSER_SCHEMA,
        "raw_response_sha256": raw_sha256,
        "quotes": [
            {
                "contract": {
                    "symbol": item.contract.symbol,
                    "expiry": item.contract.expiry.isoformat(),
                    "call_put": item.contract.call_put,
                    "strike": _decimal_text(item.contract.strike),
                    "osi_key": item.contract.osi_key,
                    "multiplier": _decimal_text(item.contract.multiplier),
                    "adjusted": item.contract.adjusted,
                    "deliverables": item.contract.deliverables,
                },
                "bid_cents": item.bid_cents,
                "ask_cents": item.ask_cents,
                "delta": _decimal_text(item.delta),
                "open_interest": item.open_interest,
                "volume": item.volume,
                "observed_at": item.observed_at.isoformat(),
                "source_sha256": item.source_sha256,
            }
            for item in quotes
        ],
    }
    canonical = _canonical_json(normalized)
    snapshot_sha256 = hashlib.sha256(
        b"etrade-opening-quote-snapshot.v1\0" + canonical.encode("utf-8")
    ).hexdigest()
    return (
        QuoteSnapshotEvidence(
            complete=True,
            snapshot_sha256=snapshot_sha256,
            quotes=quotes,
        ),
        canonical,
    )


def expected_quote_route(contracts: tuple[OptionContractId, ...]) -> str:
    """Return the credential-free fixed route for exactly two contracts."""

    if (
        type(contracts) is not tuple
        or len(contracts) != 2
        or any(type(item) is not OptionContractId for item in contracts)
    ):
        raise OpeningRiskLineageError("exactly two option contracts required")
    symbols = []
    for contract in sorted(contracts, key=lambda item: item.canonical_material):
        symbols.append(
            ":".join(
                (
                    contract.symbol,
                    str(contract.expiry.year),
                    str(contract.expiry.month),
                    str(contract.expiry.day),
                    contract.call_put,
                    f"{contract.strike:.6f}",
                )
            )
        )
    return "/v1/market/quote/" + quote(",".join(symbols), safe=",:")


def _normalize_quote(value: Any, raw_sha256: str) -> ContractQuote:
    row = _object(value, "QuoteData")
    product = _object(row.get("Product"), "QuoteData.Product")
    all_details = _object(row.get("All"), "QuoteData.All")
    option = _object(row.get("Option"), "QuoteData.Option")
    greeks = _object(option.get("optionGreeks"), "Option.optionGreeks")
    if product.get("securityType") != "OPTN":
        raise OpeningRiskLineageError("quote is not an option")
    status = row.get("quoteStatus")
    if status not in _ALLOWED_QUOTE_STATUSES:
        raise OpeningRiskLineageError("quote status is unusable")
    observed = _epoch_seconds(row.get("dateTimeUTC"), "quote timestamp")
    adjusted = all_details.get("adjustedFlag")
    if type(adjusted) is not bool or adjusted:
        raise OpeningRiskLineageError("adjusted option quote is unsupported")
    multiplier = _decimal(option.get("optionMultiplier"), "option multiplier")
    if multiplier != Decimal("100"):
        raise OpeningRiskLineageError("nonstandard option multiplier")
    osi_key = option.get("osiKey")
    if type(osi_key) is not str or not osi_key or len(osi_key) > 128:
        raise OpeningRiskLineageError("quote omitted exact OSI identity")
    try:
        contract = OptionContractId(
            symbol=_ascii(product.get("symbol"), "quote symbol", 15),
            expiry=date(
                _integer(product.get("expiryYear"), "expiry year"),
                _integer(product.get("expiryMonth"), "expiry month"),
                _integer(product.get("expiryDay"), "expiry day"),
            ),
            call_put=_ascii(product.get("callPut"), "call/put", 4),
            strike=_decimal(product.get("strikePrice"), "strike"),
            osi_key=osi_key,
            multiplier=multiplier,
            adjusted=False,
            deliverables=None,
        )
        return ContractQuote(
            contract=contract,
            bid_cents=_price_cents(all_details.get("bid"), "bid"),
            ask_cents=_price_cents(all_details.get("ask"), "ask"),
            delta=_decimal(greeks.get("delta"), "delta"),
            open_interest=_integer(
                all_details.get("openInterest"), "open interest", minimum=0
            ),
            volume=_integer(
                all_details.get("totalVolume"), "volume", minimum=0
            ),
            observed_at=observed,
            source_sha256=raw_sha256,
        )
    except (OrderDomainError, ValueError) as exc:
        raise OpeningRiskLineageError(
            "quote contract identity is inconsistent"
        ) from exc


def _strict_json(raw: bytes) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > _MAX_RAW_BYTES
        or raw.startswith(b"\xef\xbb\xbf")
    ):
        raise OpeningRiskLineageError("quote JSON bytes are invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OpeningRiskLineageError("quote JSON is not UTF-8") from exc

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise OpeningRiskLineageError(
                    "quote JSON has duplicate/non-string keys"
                )
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_int=str,
            parse_float=str,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                OpeningRiskLineageError("non-finite quote number")
            ),
        )
    except OpeningRiskLineageError:
        raise
    except Exception as exc:
        raise OpeningRiskLineageError("quote response is not strict JSON") from exc
    _validate_tree(result)
    return _object(result, "quote root")


def _validate_tree(value: Any) -> None:
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise OpeningRiskLineageError("quote JSON exceeds fixed bounds")
        if item is None or type(item) in {str, bool}:
            continue
        if type(item) is list:
            stack.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
            continue
        raise OpeningRiskLineageError("unsupported quote JSON scalar")


def _strict_query(value: str) -> dict[str, str]:
    try:
        pairs = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OpeningRiskLineageError("quote query is invalid") from exc
    if (
        type(pairs) is not list
        or any(
            type(pair) is not list
            or len(pair) != 2
            or any(type(item) is not str for item in pair)
            for pair in pairs
        )
    ):
        raise OpeningRiskLineageError("quote query shape is invalid")
    if pairs != sorted(pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise OpeningRiskLineageError("quote query is not canonical")
    return {pair[0]: pair[1] for pair in pairs}


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise OpeningRiskLineageError(f"{label} must be an object")
    return value


def _ascii(value: Any, label: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise OpeningRiskLineageError(f"invalid {label}")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if type(value) not in {str, int}:
        raise OpeningRiskLineageError(f"invalid {label}")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise OpeningRiskLineageError(f"invalid {label}") from exc
    if not result.is_finite() or result.adjusted() > 20 or result.adjusted() < -20:
        raise OpeningRiskLineageError(f"invalid {label}")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _price_cents(value: Any, label: str) -> int:
    cents = _decimal(value, label) * Decimal("100")
    if cents < 0 or cents != cents.to_integral_value() or cents > 2**63 - 1:
        raise OpeningRiskLineageError(f"{label} is not exact cents")
    return int(cents)


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    number = _decimal(value, label)
    if (
        number != number.to_integral_value()
        or number < -(2**63)
        or number > 2**63 - 1
        or minimum is not None
        and number < minimum
    ):
        raise OpeningRiskLineageError(f"invalid {label}")
    return int(number)


def _epoch_seconds(value: Any, label: str) -> datetime:
    seconds = _integer(value, label, minimum=1)
    try:
        return datetime.fromtimestamp(seconds, timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise OpeningRiskLineageError(f"invalid {label}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OpeningRiskLineageError(f"invalid {label} SHA-256")


def _utc(value: Any, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is not timezone.utc:
        raise OpeningRiskLineageError(f"{label} must use exact UTC")


__all__ = [
    "OPENING_QUOTE_PARSER_CODE_SHA256",
    "OPENING_QUOTE_PARSER_CONFIG_SHA256",
    "OPENING_QUOTE_PARSER_SCHEMA",
    "OpeningQuoteReceiptRef",
    "OpeningQuoteResponseEvidence",
    "OpeningRiskLineageError",
    "OpeningRiskLineageRef",
    "expected_quote_route",
    "parse_opening_quote_response",
]
