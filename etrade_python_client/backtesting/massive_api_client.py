"""
Async HTTP client for the Massive API (formerly Polygon.io).
Cache-aware: checks OptionDataCache before making API calls.
Supports batch request aggregation with semaphore-based concurrency.
"""
import asyncio
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
import ssl
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
import certifi
import pandas_market_calendars as mcal
import pytz

from backtesting.massive_config import API_KEY, BASE_URL
from backtesting.contract_universe import (
    CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE,
    ConfirmedContractReferenceSnapshot,
)
from backtesting.option_data_cache import (
    MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT,
    ContractReferenceCacheError,
    OptionDataCache,
)

logger = logging.getLogger(__name__)

# ANSI Colors for terminal logging
CLR_YEL = "\033[93m"
CLR_RST = "\033[0m"

# Massive/Polygon documents OPRA options quote flat-file history starting on
# 2022-03-07, while options trades are available much further back. Allow an
# override for accounts/data products with different historical coverage.
OPTION_QUOTE_HISTORY_START_DATE = os.environ.get(
    "MASSIVE_OPTION_QUOTES_START_DATE",
    "2022-03-07",
)
OPTION_TRADE_HISTORY_START_DATE = os.environ.get(
    "MASSIVE_OPTION_TRADES_START_DATE",
    "2014-06-02",
)

_CONTRACT_REFERENCE_PATH = "/v3/reference/options/contracts"
_CONTRACT_REFERENCE_REQUEST_DIGEST_DOMAIN = (
    "etrade_backtest_contract_reference_request_v1"
)
_CONTRACT_REFERENCE_RESPONSE_DIGEST_DOMAIN = (
    "etrade_backtest_contract_reference_response_v1"
)
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "apikey",
        "api_key",
        "authorization",
        "access_token",
        "token",
    }
)
_NYSE = mcal.get_calendar("NYSE")


class HistoricalFillSource(str, Enum):
    """Closed source taxonomy for historical option-price evidence."""

    OBSERVED_NBBO = "OBSERVED_NBBO"
    SYNCHRONIZED_MINUTE_AGGREGATE = "SYNCHRONIZED_MINUTE_AGGREGATE"
    TRADE_PRINT = "TRADE_PRINT"
    THEORETICAL = "THEORETICAL"
    DAILY_CLOSE = "DAILY_CLOSE"


class HistoricalFillEvidenceError(ValueError):
    """Raised when historical fill evidence is malformed or contradictory."""


@lru_cache(maxsize=4096)
def _regular_session_bounds_utc(pricing_date: str) -> Optional[Tuple[datetime, datetime]]:
    if type(pricing_date) is not str:
        return None
    try:
        parsed = datetime.strptime(pricing_date, "%Y-%m-%d")
    except ValueError:
        return None
    if parsed.strftime("%Y-%m-%d") != pricing_date:
        return None

    schedule = _NYSE.schedule(start_date=pricing_date, end_date=pricing_date)
    if len(schedule.index) != 1:
        return None
    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()
    return (
        market_open.astimezone(timezone.utc),
        market_close.astimezone(timezone.utc),
    )


def _parse_explicit_utc_timestamp(value: str) -> datetime:
    if type(value) is not str or not value:
        raise HistoricalFillEvidenceError("event timestamp must be a non-empty string")
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise HistoricalFillEvidenceError("event timestamp must declare UTC explicitly")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalFillEvidenceError("event timestamp is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise HistoricalFillEvidenceError("event timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any, field_name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise HistoricalFillEvidenceError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise HistoricalFillEvidenceError(f"{field_name} must be finite")
    return normalized


def _positive_finite_minutes(value: Any, field_name: str) -> float:
    normalized = _finite_number(value, field_name)
    if normalized <= 0:
        raise HistoricalFillEvidenceError(f"{field_name} must be positive")
    return normalized


def _nonnegative_finite_minutes(value: Any, field_name: str) -> float:
    normalized = _finite_number(value, field_name)
    if normalized < 0:
        raise HistoricalFillEvidenceError(
            f"{field_name} must be non-negative"
        )
    return normalized


@dataclass(frozen=True)
class HistoricalFillEvidence(Mapping[str, Any]):
    """Immutable per-leg evidence used by the historical mark boundary.

    Parsed provider JSON is not raw-response proof. The object therefore records
    that provider bytes were not retained. Quote evidence validates a historical
    mark only; displayed size and modeled execution are not proven.
    """

    option_ticker: str
    pricing_date: str
    source: HistoricalFillSource
    mid: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    event_timestamp_utc: Optional[str] = None
    provider: str = "massive"
    provider_route: str = ""
    ticker_binding: str = "exact_requested_ticker"
    raw_provider_bytes_retained: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.option_ticker) is not str or not self.option_ticker:
            raise HistoricalFillEvidenceError("option_ticker must be a non-empty string")
        bounds = _regular_session_bounds_utc(self.pricing_date)
        if bounds is None:
            raise HistoricalFillEvidenceError("pricing_date must be an NYSE trading session")
        if type(self.source) is not HistoricalFillSource:
            raise HistoricalFillEvidenceError("source must be HistoricalFillSource")
        if type(self.provider) is not str or not self.provider:
            raise HistoricalFillEvidenceError("provider must be a non-empty string")
        if type(self.provider_route) is not str:
            raise HistoricalFillEvidenceError("provider_route must be a string")
        if self.ticker_binding != "exact_requested_ticker":
            raise HistoricalFillEvidenceError("ticker binding is not exact")
        if type(self.raw_provider_bytes_retained) is not bool:
            raise HistoricalFillEvidenceError("raw_provider_bytes_retained must be bool")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise HistoricalFillEvidenceError("unsupported fill evidence schema")

        mid = _finite_number(self.mid, "mid")
        if mid < 0:
            raise HistoricalFillEvidenceError("mid must be non-negative")
        object.__setattr__(self, "mid", mid)
        for field_name in ("bid", "ask"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = _finite_number(value, field_name)
                if normalized < 0:
                    raise HistoricalFillEvidenceError(
                        f"{field_name} must be non-negative"
                    )
                object.__setattr__(self, field_name, normalized)
        if (
            self.bid is not None
            and self.ask is not None
            and self.ask < self.bid
        ):
            raise HistoricalFillEvidenceError("bid/ask market is crossed")
        object.__setattr__(
            self,
            "bid_size",
            _finite_number(self.bid_size, "bid_size"),
        )
        object.__setattr__(
            self,
            "ask_size",
            _finite_number(self.ask_size, "ask_size"),
        )
        if self.bid_size < 0 or self.ask_size < 0:
            raise HistoricalFillEvidenceError("quote sizes must be non-negative")

        if self.source is HistoricalFillSource.OBSERVED_NBBO:
            if self.provider != "massive":
                raise HistoricalFillEvidenceError(
                    "observed NBBO provider is not Massive"
                )
            if self.bid is None or self.ask is None:
                raise HistoricalFillEvidenceError("observed NBBO requires bid and ask")
            if self.bid <= 0 or self.ask < self.bid:
                raise HistoricalFillEvidenceError("observed NBBO is zero or crossed")
            expected_mid = round((self.bid + self.ask) / 2.0, 4)
            if not math.isclose(self.mid, expected_mid, rel_tol=0.0, abs_tol=1e-9):
                raise HistoricalFillEvidenceError("observed NBBO midpoint is inconsistent")
            if self.provider_route != f"/v3/quotes/{self.option_ticker}":
                raise HistoricalFillEvidenceError(
                    "observed NBBO is not bound to the exact contract endpoint"
                )
            if self.event_timestamp_utc is None:
                raise HistoricalFillEvidenceError(
                    "observed NBBO requires an event timestamp"
                )

        if self.source in {
            HistoricalFillSource.OBSERVED_NBBO,
            HistoricalFillSource.SYNCHRONIZED_MINUTE_AGGREGATE,
            HistoricalFillSource.TRADE_PRINT,
        } and self.event_timestamp_utc is None:
            raise HistoricalFillEvidenceError(
                f"{self.source.value} requires an event timestamp"
            )

        if self.event_timestamp_utc is not None:
            event_time = _parse_explicit_utc_timestamp(self.event_timestamp_utc)
            market_open, market_close = bounds
            if event_time < market_open or event_time > market_close:
                raise HistoricalFillEvidenceError(
                    "event timestamp is outside the requested NYSE session"
                )
            object.__setattr__(
                self,
                "event_timestamp_utc",
                event_time.isoformat().replace("+00:00", "Z"),
            )

    @property
    def strict_nbbo_mark_eligible(self) -> bool:
        return self.source is HistoricalFillSource.OBSERVED_NBBO

    @property
    def strict_nbbo_eligible(self) -> bool:
        """Deprecated: historical evidence never proves executable eligibility."""

        return False

    @property
    def session_close_utc(self) -> str:
        bounds = _regular_session_bounds_utc(self.pricing_date)
        if bounds is None:  # Guarded by construction.
            raise HistoricalFillEvidenceError("pricing date has no trading session")
        return bounds[1].isoformat().replace("+00:00", "Z")

    @property
    def quote_age_at_close_seconds(self) -> Optional[float]:
        if (
            self.source is not HistoricalFillSource.OBSERVED_NBBO
            or self.event_timestamp_utc is None
        ):
            return None
        close = _parse_explicit_utc_timestamp(self.session_close_utc)
        event = _parse_explicit_utc_timestamp(self.event_timestamp_utc)
        return (close - event).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "option_ticker": self.option_ticker,
            "pricing_date": self.pricing_date,
            "source": self.source.value,
            "mid": self.mid,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "quote_timestamp": self.event_timestamp_utc,
            "timestamp": self.event_timestamp_utc,
            "event_timestamp_utc": self.event_timestamp_utc,
            "provider": self.provider,
            "provider_route": self.provider_route,
            "ticker_binding": self.ticker_binding,
            "raw_provider_bytes_retained": self.raw_provider_bytes_retained,
            "strict_nbbo_mark_eligible": self.strict_nbbo_mark_eligible,
            "strict_nbbo_eligible": False,
            "strict_nbbo_eligible_deprecated": (
                "Use strict_nbbo_mark_eligible; historical quotes do not prove "
                "an executable fill."
            ),
            "session_close_utc": self.session_close_utc,
            "quote_age_at_close_seconds": self.quote_age_at_close_seconds,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def validate_strict_nbbo_bundle(
    evidence: Sequence[HistoricalFillEvidence],
    expected_tickers: Sequence[str],
    pricing_date: str,
    max_time_delta_minutes: float,
    max_quote_age_minutes: float = 5.0,
) -> Tuple[HistoricalFillEvidence, ...]:
    """Validate exact binding, close freshness, and pairwise timing."""

    max_delta = _nonnegative_finite_minutes(
        max_time_delta_minutes,
        "max_time_delta_minutes",
    )
    max_quote_age = _positive_finite_minutes(
        max_quote_age_minutes,
        "max_quote_age_minutes",
    )
    if len(expected_tickers) not in {2, 3}:
        raise HistoricalFillEvidenceError(
            "strict mark bundle requires exactly two or three legs"
        )
    if len(evidence) != len(expected_tickers):
        raise HistoricalFillEvidenceError(
            "mark evidence does not match expected legs"
        )
    if len(set(expected_tickers)) != len(expected_tickers):
        raise HistoricalFillEvidenceError("expected option tickers must be unique")
    if any(type(ticker) is not str or not ticker for ticker in expected_tickers):
        raise HistoricalFillEvidenceError(
            "expected option tickers must be non-empty strings"
        )

    timestamps = []
    validated = []
    for item, expected_ticker in zip(evidence, expected_tickers):
        if type(item) is not HistoricalFillEvidence:
            raise HistoricalFillEvidenceError(
                "mark evidence type is not exact"
            )
        if item.option_ticker != expected_ticker:
            raise HistoricalFillEvidenceError(
                "mark evidence ticker mismatch"
            )
        if item.pricing_date != pricing_date:
            raise HistoricalFillEvidenceError(
                "mark evidence pricing-date mismatch"
            )
        if item.source is not HistoricalFillSource.OBSERVED_NBBO:
            raise HistoricalFillEvidenceError("strict mode requires observed NBBO")
        if item.provider != "massive":
            raise HistoricalFillEvidenceError("strict NBBO provider mismatch")
        if not item.strict_nbbo_mark_eligible:
            raise HistoricalFillEvidenceError(
                "evidence is not eligible for strict NBBO mark validation"
            )
        event_timestamp = _parse_explicit_utc_timestamp(
            item.event_timestamp_utc
        )
        bounds = _regular_session_bounds_utc(pricing_date)
        if bounds is None:  # Guarded by the evidence constructor.
            raise HistoricalFillEvidenceError("pricing date has no trading session")
        session_close = bounds[1]
        quote_age = session_close - event_timestamp
        if quote_age < timedelta(0):
            raise HistoricalFillEvidenceError(
                "strict observed NBBO timestamp is after the session close"
            )
        if quote_age > timedelta(minutes=max_quote_age):
            raise HistoricalFillEvidenceError(
                "strict observed NBBO is too old relative to the session close"
            )
        timestamps.append(event_timestamp)
        validated.append(item)

    if max(timestamps) - min(timestamps) > timedelta(minutes=max_delta):
        raise HistoricalFillEvidenceError(
            "multi-leg observed NBBO timestamps exceed the configured delta"
        )
    return tuple(validated)


def fill_bundle_diagnostics(
    evidence: Sequence[HistoricalFillEvidence],
    max_time_delta_minutes: float,
    max_quote_age_minutes: float = 5.0,
) -> Dict[str, Any]:
    max_delta = _nonnegative_finite_minutes(
        max_time_delta_minutes,
        "max_time_delta_minutes",
    )
    max_quote_age = _positive_finite_minutes(
        max_quote_age_minutes,
        "max_quote_age_minutes",
    )
    timestamps = [
        _parse_explicit_utc_timestamp(item.event_timestamp_utc)
        for item in evidence
        if item.event_timestamp_utc is not None
    ]
    aligned = None
    max_delta_seconds = None
    if len(timestamps) == len(evidence) and timestamps:
        max_delta_seconds = (max(timestamps) - min(timestamps)).total_seconds()
        aligned = max_delta_seconds <= max_delta * 60.0
    quote_freshness = []
    for item in evidence:
        age_seconds = item.quote_age_at_close_seconds
        quote_freshness.append(
            {
                "ticker": item.option_ticker,
                "source": item.source.value,
                "session_close_utc": item.session_close_utc,
                "event_timestamp_utc": item.event_timestamp_utc,
                "quote_age_at_close_seconds": age_seconds,
                "within_max_quote_age": (
                    age_seconds is not None
                    and 0.0 <= age_seconds <= max_quote_age * 60.0
                ),
            }
        )
    all_observed = bool(evidence) and all(
        item.source is HistoricalFillSource.OBSERVED_NBBO
        for item in evidence
    )
    tickers = [item.option_ticker for item in evidence]
    pricing_dates = [item.pricing_date for item in evidence]
    valid_leg_count = len(evidence) in {2, 3}
    unique_tickers = len(set(tickers)) == len(tickers)
    same_pricing_date = (
        bool(pricing_dates) and len(set(pricing_dates)) == 1
    )
    bundle_shape_valid = (
        valid_leg_count and unique_tickers and same_pricing_date
    )
    observed_quote_freshness = [
        item
        for item in quote_freshness
        if item["source"] == HistoricalFillSource.OBSERVED_NBBO.value
    ]
    all_quotes_close_fresh = bool(observed_quote_freshness) and all(
        item["within_max_quote_age"] for item in observed_quote_freshness
    )
    rejection_reasons = []
    if not valid_leg_count:
        rejection_reasons.append("INVALID_LEG_COUNT")
    if not unique_tickers:
        rejection_reasons.append("DUPLICATE_OPTION_TICKER")
    if not same_pricing_date:
        rejection_reasons.append("MIXED_PRICING_DATES")
    if evidence and not all_observed:
        rejection_reasons.append("NON_OBSERVED_NBBO_SOURCE")
    if observed_quote_freshness and not all_quotes_close_fresh:
        rejection_reasons.append("OBSERVED_NBBO_TOO_OLD_AT_SESSION_CLOSE")
    if aligned is False:
        rejection_reasons.append("MULTILEG_TIMESTAMP_DELTA_EXCEEDED")
    return {
        "schema_version": 1,
        "sources": [item.source.value for item in evidence],
        "timestamps_utc": [item.event_timestamp_utc for item in evidence],
        "tickers": tickers,
        "pricing_dates": pricing_dates,
        "valid_leg_count": valid_leg_count,
        "unique_tickers": unique_tickers,
        "same_pricing_date": same_pricing_date,
        "bundle_shape_valid": bundle_shape_valid,
        "temporally_synchronized": aligned,
        "max_observed_delta_seconds": max_delta_seconds,
        "configured_max_delta_minutes": max_delta,
        "configured_max_quote_age_minutes": max_quote_age,
        "quote_close_freshness": quote_freshness,
        "all_observed_quotes_close_fresh": all_quotes_close_fresh,
        "strict_rejection_reasons": rejection_reasons,
        "strict_nbbo_mark_validated": bundle_shape_valid
        and all_observed
        and aligned is True
        and all_quotes_close_fresh,
        "strict_nbbo_authorized": False,
        "strict_nbbo_authorized_deprecated": (
            "Historical NBBO evidence validates a mark only; quote size and "
            "execution are not proven."
        ),
        "execution_assumption": "HISTORICAL_MARK_NOT_EXECUTABLE_FILL",
        "raw_provider_bytes_retained": bool(evidence) and all(
            item.raw_provider_bytes_retained for item in evidence
        ),
    }


class _ContractReferenceAcquisitionError(RuntimeError):
    """Internal typed failure that is safe to persist as a code only."""

    def __init__(self, failure_code: str):
        super().__init__(failure_code)
        self.failure_code = failure_code


class MassiveAPIClient:
    """Async client for Massive/Polygon API with integrated caching."""

    def __init__(
        self,
        cache: OptionDataCache,
        api_key: str = API_KEY,
        max_concurrent: int = 50,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        requests_per_second: float = 50.0,
        offline_only: bool = False,
    ):
        self.api_key = api_key
        self.cache = cache
        self.offline_only = bool(
            offline_only
            or os.environ.get("MASSIVE_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
            or not api_key
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiter state
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._rate_lock = asyncio.Lock()

        # Stats
        self.api_calls = 0
        self.cache_hits = 0

    async def __aenter__(self):
        if self.offline_only:
            return self
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(ssl=self.ssl_ctx, limit=20),
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def print_stats(self):
        total = self.api_calls + self.cache_hits
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        print(f"  API calls: {self.api_calls:,} | Cache hits: {self.cache_hits:,} | "
              f"Hit rate: {hit_rate:.1f}%")

    @staticmethod
    def _format_timestamp(value) -> Optional[str]:
        if value is None:
            return None
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return str(value)
        if raw > 1e18:
            seconds = raw / 1e9
        elif raw > 1e15:
            seconds = raw / 1e6
        elif raw > 1e12:
            seconds = raw / 1e3
        else:
            seconds = raw
        try:
            return datetime.fromtimestamp(seconds, tz=pytz.utc).isoformat()
        except Exception:
            return str(value)

    @staticmethod
    def _date_before(value: str, boundary: str) -> bool:
        try:
            return value[:10] < boundary
        except Exception:
            return False

    def _cached_price_result(
        self,
        option_ticker: str,
        date: str,
        allow_trade_fallback: bool = True,
    ) -> Optional[HistoricalFillEvidence]:
        """Return only an unambiguous research fallback from the legacy cache.

        The legacy row has no quote/trade timestamp or source column. Its
        bid/ask/mid fields therefore cannot prove NBBO, a trade, a synchronized
        aggregate, or a theoretical mark. Only the daily OHLCV close column has
        an unambiguous class, and it is never strict-NBBO eligible.
        """
        cached = self.cache.get_ohlcv(option_ticker, date)
        if not cached:
            return None

        close_price = cached.get("close")
        try:
            close_price = _finite_number(close_price, "daily close")
        except HistoricalFillEvidenceError:
            close_price = None
        if close_price is not None and close_price > 0:
            try:
                evidence = HistoricalFillEvidence(
                    option_ticker=option_ticker,
                    pricing_date=date,
                    source=HistoricalFillSource.DAILY_CLOSE,
                    mid=close_price,
                    bid=close_price,
                    ask=close_price,
                    provider="legacy_option_price_cache",
                    provider_route="option_prices.close",
                )
            except HistoricalFillEvidenceError:
                return None
            self.cache_hits += 1
            return evidence

        return None

    async def _rate_limit(self):
        """Token-bucket style rate limiter."""
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

    async def _get(self, url: str, params: dict) -> Optional[dict]:
        """Make a GET request with retry and rate limiting."""
        if self.offline_only:
            return None

        params["apiKey"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            await self._rate_limit()
            try:
                async with self.semaphore:
                    async with self.session.get(url, params=params) as resp:
                        self.api_calls += 1
                        if resp.status == 429:
                            wait = self.backoff_factor * (2 ** attempt)
                            logger.warning(f"Rate limited, waiting {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status == 404:
                            return None
                        resp.raise_for_status()
                        return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    return None
                wait = self.backoff_factor * (2 ** (attempt - 1))
                await asyncio.sleep(wait)
        return None

    def _parse_ticker(self, ticker: str) -> dict:
        """
        Parse Polygon/Massive ticker format: O:SPY151016P00155000
        Returns {underlying, expiration, contract_type, strike}
        """
        try:
            # Remove O: prefix if present
            clean_ticker = ticker[2:] if ticker.startswith("O:") else ticker
            
            # Find the split between underlying and date (date starts with numbers)
            # Ticker format: SYMBOL YYMMDD [C/P] STRIKE
            # SYMBOL can be 1-6 chars
            for i, char in enumerate(clean_ticker):
                if char.isdigit():
                    split_idx = i
                    break
            else:
                return {}

            underlying = clean_ticker[:split_idx]
            rem = clean_ticker[split_idx:]
            
            exp_str = rem[:6] # YYMMDD
            yy = "20" + exp_str[:2]
            mm = exp_str[2:4]
            dd = exp_str[4:6]
            expiration = f"{yy}-{mm}-{dd}"
            
            if rem[6] == "C":
                contract_type = "call"
            elif rem[6] == "P":
                contract_type = "put"
            else:
                return {}
            
            strike_raw = rem[7:]
            strike = float(strike_raw) / 1000.0
            
            return {
                "underlying": underlying,
                "expiration": expiration,
                "contract_type": contract_type,
                "strike": strike
            }
        except Exception:
            return {}

    @staticmethod
    def _canonical_json_sha256(value: object, domain: str) -> str:
        try:
            encoded = json.dumps(
                {
                    "domain": domain,
                    "value": value,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError) as exc:
            raise _ContractReferenceAcquisitionError(
                "NON_CANONICAL_PROVIDER_RESPONSE"
            ) from exc
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _reference_request_identity(
        url: str,
        params: dict,
    ) -> tuple[str, str]:
        """Return a credential-free request URL and its canonical identity."""

        if type(url) is not str or not url:
            raise _ContractReferenceAcquisitionError(
                "INVALID_PAGINATION_URL"
            )
        if type(params) is not dict:
            raise _ContractReferenceAcquisitionError(
                "INVALID_REFERENCE_PARAMETERS"
            )
        parts = urlsplit(url)
        base = urlsplit(BASE_URL)
        if (
            parts.scheme.lower() != "https"
            or parts.scheme.lower() != base.scheme.lower()
            or parts.hostname != base.hostname
            or parts.port != base.port
            or parts.username is not None
            or parts.password is not None
            or parts.path != _CONTRACT_REFERENCE_PATH
            or bool(parts.fragment)
        ):
            raise _ContractReferenceAcquisitionError(
                "INVALID_PAGINATION_URL"
            )

        query_pairs = [
            (key, value)
            for key, value in parse_qsl(
                parts.query,
                keep_blank_values=True,
            )
            if key.casefold() not in _SENSITIVE_QUERY_KEYS
        ]
        parameter_pairs = []
        for key, value in params.items():
            if type(key) is not str:
                raise _ContractReferenceAcquisitionError(
                    "INVALID_REFERENCE_PARAMETERS"
                )
            if key.casefold() in _SENSITIVE_QUERY_KEYS:
                continue
            if type(value) not in {str, int, float, bool}:
                raise _ContractReferenceAcquisitionError(
                    "INVALID_REFERENCE_PARAMETERS"
                )
            parameter_pairs.append((key, str(value)))

        credential_free_url = urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path,
                urlencode(query_pairs),
                "",
            )
        )
        request_payload = {
            "method": "GET",
            "scheme": parts.scheme.lower(),
            "authority": parts.netloc.lower(),
            "path": parts.path,
            "query": sorted(query_pairs + parameter_pairs),
        }
        request_sha256 = MassiveAPIClient._canonical_json_sha256(
            request_payload,
            _CONTRACT_REFERENCE_REQUEST_DIGEST_DOMAIN,
        )
        return credential_free_url, request_sha256

    @staticmethod
    def _canonical_strike_text(value: object) -> str:
        if type(value) not in {int, float} or type(value) is bool:
            raise _ContractReferenceAcquisitionError(
                "MALFORMED_CONTRACT_ROW"
            )
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise _ContractReferenceAcquisitionError(
                "MALFORMED_CONTRACT_ROW"
            )
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise _ContractReferenceAcquisitionError(
                "MALFORMED_CONTRACT_ROW"
            ) from exc
        return format(decimal_value.normalize(), "f")

    def _strict_reference_contract(
        self,
        item: object,
        *,
        root_ticker: str,
        expiration: str,
        contract_type: str,
    ) -> dict:
        if type(item) is not dict:
            raise _ContractReferenceAcquisitionError(
                "MALFORMED_CONTRACT_ROW"
            )
        ticker = item.get("ticker")
        strike = item.get("strike_price")
        if type(ticker) is not str or not ticker:
            raise _ContractReferenceAcquisitionError(
                "MALFORMED_CONTRACT_ROW"
            )
        strike_text = self._canonical_strike_text(strike)
        parsed = self._parse_ticker(ticker)
        if (
            parsed.get("underlying") != root_ticker
            or parsed.get("expiration") != expiration
            or parsed.get("contract_type") != contract_type
            or self._canonical_strike_text(parsed.get("strike"))
            != strike_text
        ):
            raise _ContractReferenceAcquisitionError(
                "CONTRACT_ROW_BINDING_MISMATCH"
            )
        if (
            "expiration_date" in item
            and item["expiration_date"] != expiration
        ):
            raise _ContractReferenceAcquisitionError(
                "CONTRACT_ROW_BINDING_MISMATCH"
            )
        if (
            "contract_type" in item
            and item["contract_type"] != contract_type
        ):
            raise _ContractReferenceAcquisitionError(
                "CONTRACT_ROW_BINDING_MISMATCH"
            )
        return {
            "option_ticker": ticker,
            "strike": float(strike),
            "contract_type": contract_type,
        }

    def _finish_reference_failure(
        self,
        attempt_id: str,
        failure_code: str,
    ) -> None:
        try:
            self.cache.finish_contract_reference_failure(
                attempt_id,
                failure_code,
            )
        except ContractReferenceCacheError:
            logger.error(
                "Contract-reference attempt could not record failure: %s",
                attempt_id,
            )

    # ── Contracts List ──────────────────────────────────────────────

    async def fetch_contracts_snapshot(
        self,
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of: str,
    ) -> Optional[ConfirmedContractReferenceSnapshot]:
        """
        Return only a durable, complete, exact-tuple reference snapshot.

        Attempts are recorded before provider I/O. Every successful page is
        linked by credential-free request/response hashes. Empty, offline,
        interrupted, malformed, over-limit, and conflicting attempts never
        publish a cache head and therefore cannot authorize eligibility.
        """
        cached = self.cache.get_confirmed_contract_reference(
            underlying=underlying,
            expiration=expiration,
            contract_type=contract_type,
            as_of_date=as_of,
        )
        if cached is not None:
            self.cache_hits += 1
            return cached

        expected_roots = (
            ("SPX", "SPXW")
            if underlying == "SPX"
            else (underlying,)
        )
        try:
            attempt_id = self.cache.begin_contract_reference_attempt(
                underlying=underlying,
                expiration=expiration,
                contract_type=contract_type,
                as_of_date=as_of,
                expected_roots=expected_roots,
            )
        except ContractReferenceCacheError:
            logger.error(
                "Contract-reference attempt could not start for exact tuple"
            )
            return None

        if self.offline_only:
            self._finish_reference_failure(attempt_id, "OFFLINE_ONLY")
            return None

        print(
            f"{CLR_YEL}  [API FETCH] {underlying} contracts for "
            f"{expiration} (as_of {as_of}){CLR_RST}"
        )
        all_contracts: list[dict] = []
        seen_tickers: set[str] = set()

        try:
            for root_ordinal, root_ticker in enumerate(expected_roots):
                url = f"{BASE_URL}{_CONTRACT_REFERENCE_PATH}"
                params = {
                    "underlying_ticker": root_ticker,
                    "contract_type": contract_type,
                    "expiration_date": expiration,
                    "as_of": as_of,
                    "limit": CONTRACT_REFERENCE_MAX_RESULTS_PER_PAGE,
                    "order": "asc",
                    "sort": "strike_price",
                }
                page_ordinal = 1
                seen_requests: set[str] = set()

                while True:
                    if page_ordinal > MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT:
                        raise _ContractReferenceAcquisitionError(
                            "PAGE_LIMIT_EXCEEDED"
                        )
                    request_url, request_sha256 = (
                        self._reference_request_identity(url, params)
                    )
                    if request_sha256 in seen_requests:
                        raise _ContractReferenceAcquisitionError(
                            "PAGINATION_LOOP"
                        )
                    seen_requests.add(request_sha256)

                    try:
                        data = await self._get(
                            request_url,
                            dict(params),
                        )
                    except Exception as exc:
                        raise _ContractReferenceAcquisitionError(
                            "REQUEST_FAILED"
                        ) from exc
                    if data is None:
                        raise _ContractReferenceAcquisitionError(
                            "REQUEST_FAILED"
                        )
                    if type(data) is not dict:
                        raise _ContractReferenceAcquisitionError(
                            "MALFORMED_PROVIDER_RESPONSE"
                        )
                    results = data.get("results")
                    if type(results) is not list:
                        raise _ContractReferenceAcquisitionError(
                            "MALFORMED_PROVIDER_RESPONSE"
                        )

                    next_url = data.get("next_url")
                    if next_url is None:
                        next_request_sha256 = None
                        next_request_url = None
                    else:
                        if type(next_url) is not str or not next_url:
                            raise _ContractReferenceAcquisitionError(
                                "INVALID_PAGINATION_URL"
                            )
                        next_request_url, next_request_sha256 = (
                            self._reference_request_identity(next_url, {})
                        )
                    response_evidence = dict(data)
                    if next_request_url is not None:
                        response_evidence["next_url"] = next_request_url
                    response_sha256 = self._canonical_json_sha256(
                        response_evidence,
                        _CONTRACT_REFERENCE_RESPONSE_DIGEST_DOMAIN,
                    )

                    self.cache.record_contract_reference_page(
                        attempt_id=attempt_id,
                        root_ordinal=root_ordinal,
                        root_ticker=root_ticker,
                        page_ordinal=page_ordinal,
                        request_sha256=request_sha256,
                        response_sha256=response_sha256,
                        result_count=len(results),
                        next_request_sha256=next_request_sha256,
                        is_terminal=next_request_url is None,
                    )

                    for item in results:
                        contract = self._strict_reference_contract(
                            item,
                            root_ticker=root_ticker,
                            expiration=expiration,
                            contract_type=contract_type,
                        )
                        ticker = contract["option_ticker"]
                        if ticker in seen_tickers:
                            raise _ContractReferenceAcquisitionError(
                                "DUPLICATE_CONTRACT_TICKER"
                            )
                        seen_tickers.add(ticker)
                        all_contracts.append(contract)

                    if next_request_url is None:
                        break
                    if page_ordinal == MAX_CONTRACT_REFERENCE_PAGES_PER_ROOT:
                        raise _ContractReferenceAcquisitionError(
                            "PAGE_LIMIT_EXCEEDED"
                        )
                    if next_request_sha256 in seen_requests:
                        raise _ContractReferenceAcquisitionError(
                            "PAGINATION_LOOP"
                        )
                    url = next_request_url
                    params = {}
                    page_ordinal += 1

            if not all_contracts:
                self.cache.finish_contract_reference_empty(attempt_id)
                return None

            return self.cache.confirm_contract_reference_attempt(
                attempt_id,
                all_contracts,
            )
        except _ContractReferenceAcquisitionError as exc:
            self._finish_reference_failure(
                attempt_id,
                exc.failure_code,
            )
            return None
        except ContractReferenceCacheError:
            self._finish_reference_failure(
                attempt_id,
                "EVIDENCE_REJECTED",
            )
            return None
        except Exception:
            self._finish_reference_failure(
                attempt_id,
                "UNEXPECTED_ACQUISITION_ERROR",
            )
            return None

    async def fetch_contracts_list(
        self,
        underlying: str,
        expiration: Optional[str],
        contract_type: str,
        as_of: str,
    ) -> List[dict]:
        """Compatibility adapter that unwraps only confirmed snapshot rows."""

        if not expiration:
            return []
        snapshot = await self.fetch_contracts_snapshot(
            underlying,
            expiration,
            contract_type,
            as_of,
        )
        return snapshot.as_legacy_records() if snapshot is not None else []

    # ── OHLCV Daily Bars ────────────────────────────────────────────

    async def fetch_contract_daily_bars(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
        underlying: str = "",
        contract_type: str = "",
        strike: float = 0.0,
        expiration: str = "",
        force: bool = False,
    ) -> List[dict]:
        """
        Fetch daily OHLCV bars for a single contract over a date range.
        Checks cache and fetch log before calling API.
        """
        # 1. Check completed range cache. Partial cached rows are not enough:
        # a previous narrow fetch may have only one date and would otherwise make
        # the caller believe the whole requested range had been populated.
        if not force and self.cache.is_ticker_range_fetched(option_ticker, from_date, to_date):
            self.cache_hits += 1
            return self.cache.get_ohlcv_range(option_ticker, from_date, to_date)

        if self.offline_only:
            self.cache.mark_ticker_range_fetched(option_ticker, from_date, to_date, bar_count=0)
            return []

        print(f"{CLR_YEL}  [API FETCH] OHLCV for {option_ticker} from {from_date} to {to_date}{CLR_RST}")
        url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{from_date}/{to_date}"
        params = {"adjusted": "false", "sort": "asc", "limit": 5000}

        data = await self._get(url, params)
        if not data or "results" not in data:
            # Record that we tried and got nothing (negative cache)
            self.cache.mark_ticker_range_fetched(option_ticker, from_date, to_date, bar_count=0)
            return []

        now_iso = datetime.now().isoformat()
        bars = []
        cache_records = []

        _ET = pytz.timezone("America/New_York")
        for bar in data["results"]:
            ts_ms = bar.get("t", 0)
            pricing_date = datetime.fromtimestamp(ts_ms / 1000, tz=pytz.utc).astimezone(_ET).strftime("%Y-%m-%d")

            bar_dict = {
                "option_ticker": option_ticker,
                "pricing_date": pricing_date,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v", 0),
                "vwap": bar.get("vw"),
                "num_trades": bar.get("n", 0),
            }
            bars.append(bar_dict)

            cache_records.append((
                underlying, option_ticker, contract_type, strike,
                expiration, pricing_date,
                bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
                bar.get("v", 0), bar.get("vw"), bar.get("n", 0),
                now_iso,
            ))

        if cache_records:
            self.cache.bulk_upsert_ohlcv(cache_records)

        # Record successful fetch (positive cache)
        self.cache.mark_ticker_range_fetched(option_ticker, from_date, to_date, bar_count=len(bars))

        return bars

    # ── EOD Quote (Bid/Ask) ─────────────────────────────────────────

    @staticmethod
    def _quote_timestamp(quote: Mapping[str, Any]) -> Optional[str]:
        for field_name in (
            "participant_timestamp",
            "sip_timestamp",
            "timestamp",
        ):
            value = quote.get(field_name)
            if value is not None:
                return MassiveAPIClient._format_timestamp(value)
        return None

    @staticmethod
    def _evidence_from_trade_result(
        option_ticker: str,
        pricing_date: str,
        trade: Mapping[str, Any],
    ) -> Optional[HistoricalFillEvidence]:
        try:
            price = _finite_number(trade.get("price"), "trade price")
            if price <= 0:
                return None
            source = trade.get("source")
            if source == HistoricalFillSource.DAILY_CLOSE.value:
                return HistoricalFillEvidence(
                    option_ticker=option_ticker,
                    pricing_date=pricing_date,
                    source=HistoricalFillSource.DAILY_CLOSE,
                    mid=price,
                    bid=price,
                    ask=price,
                    provider="legacy_option_price_cache",
                    provider_route="option_prices.close",
                )
            if source != HistoricalFillSource.TRADE_PRINT.value:
                return None
            return HistoricalFillEvidence(
                option_ticker=option_ticker,
                pricing_date=pricing_date,
                source=HistoricalFillSource.TRADE_PRINT,
                mid=price,
                bid=price,
                ask=price,
                bid_size=trade.get("size") or 0,
                ask_size=trade.get("size") or 0,
                event_timestamp_utc=trade.get("timestamp"),
                provider="massive",
                provider_route=f"/v3/trades/{option_ticker}",
            )
        except HistoricalFillEvidenceError:
            return None

    async def fetch_observed_nbbo(
        self,
        option_ticker: str,
        date: str,
        max_quote_age_minutes: Optional[float] = 5.0,
    ) -> Optional[HistoricalFillEvidence]:
        """Fetch one exact-contract, regular-session observed NBBO.

        This intentionally bypasses legacy cache rows. Parsed JSON is validated
        in memory, but raw provider response bytes are not retained by this
        client, so the result is not a durable/replayable provider receipt.
        A finite positive ``max_quote_age_minutes`` constrains acquisition and
        acceptance to quotes near the exact exchange-calendar close. ``None``
        is research-only and may return an older in-session quote.
        """

        max_quote_age = (
            None
            if max_quote_age_minutes is None
            else _positive_finite_minutes(
                max_quote_age_minutes,
                "max_quote_age_minutes",
            )
        )
        if self.offline_only or self._date_before(
            date,
            OPTION_QUOTE_HISTORY_START_DATE,
        ):
            return None
        bounds = _regular_session_bounds_utc(date)
        if bounds is None:
            return None
        market_open, market_close = bounds
        if max_quote_age_minutes is None:
            quote_window_start = market_open
        else:
            quote_window_start = max(
                market_open,
                market_close - timedelta(minutes=max_quote_age),
            )
        url = f"{BASE_URL}/v3/quotes/{option_ticker}"
        params = {
            "timestamp.gte": quote_window_start.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "timestamp.lte": market_close.isoformat().replace("+00:00", "Z"),
            "order": "desc",
            "sort": "timestamp",
            "limit": 100,
        }
        print(
            f"{CLR_YEL}  [API FETCH] Observed NBBO for "
            f"{option_ticker} on {date}{CLR_RST}"
        )
        data = await self._get(url, params)
        if not isinstance(data, Mapping) or not isinstance(
            data.get("results"),
            list,
        ):
            return None

        for quote in data["results"]:
            if not isinstance(quote, Mapping):
                continue
            response_ticker = quote.get("ticker") or quote.get("option_ticker")
            if response_ticker is not None and response_ticker != option_ticker:
                continue
            try:
                bid = _finite_number(quote.get("bid_price"), "bid")
                ask = _finite_number(quote.get("ask_price"), "ask")
                evidence = HistoricalFillEvidence(
                    option_ticker=option_ticker,
                    pricing_date=date,
                    source=HistoricalFillSource.OBSERVED_NBBO,
                    bid=bid,
                    ask=ask,
                    mid=round((bid + ask) / 2.0, 4),
                    bid_size=quote.get("bid_size") or 0,
                    ask_size=quote.get("ask_size") or 0,
                    event_timestamp_utc=self._quote_timestamp(quote),
                    provider="massive",
                    provider_route=f"/v3/quotes/{option_ticker}",
                )
            except HistoricalFillEvidenceError:
                continue
            if max_quote_age_minutes is not None:
                age_seconds = evidence.quote_age_at_close_seconds
                if (
                    age_seconds is None
                    or age_seconds < 0
                    or age_seconds > max_quote_age * 60.0
                ):
                    continue
            return evidence
        return None

    async def fetch_eod_quote(
        self,
        option_ticker: str,
        date: str,
        allow_trade_fallback: bool = True,
        strict_nbbo: bool = False,
        max_quote_age_minutes: float = 5.0,
    ) -> Optional[HistoricalFillEvidence]:
        """Get typed historical price evidence for one option contract.

        ``strict_nbbo=True`` never reads the legacy cache and never falls back
        to a trade or daily close. Research mode may return an explicitly
        classified non-NBBO source.
        """
        if strict_nbbo:
            return await self.fetch_observed_nbbo(
                option_ticker,
                date,
                max_quote_age_minutes=max_quote_age_minutes,
            )

        cached_price = self._cached_price_result(
            option_ticker,
            date,
            allow_trade_fallback=allow_trade_fallback,
        )
        if cached_price:
            return cached_price

        if self.offline_only:
            return None

        quote_result = None
        if not self._date_before(date, OPTION_QUOTE_HISTORY_START_DATE):
            quote_result = await self.fetch_observed_nbbo(
                option_ticker,
                date,
                max_quote_age_minutes=None,
            )
        elif allow_trade_fallback:
            print(
                f"{CLR_YEL}  [API FETCH] Quote history unavailable before "
                f"{OPTION_QUOTE_HISTORY_START_DATE}; considering an explicit "
                f"trade fallback for {option_ticker} on {date}{CLR_RST}"
            )

        if not quote_result and allow_trade_fallback:
            print(
                f"{CLR_YEL}  [API FETCH] No observed NBBO for "
                f"{option_ticker} on {date}, trying an explicit trade source...{CLR_RST}"
            )
            trade = await self.fetch_latest_trade(option_ticker, date)
            if trade:
                quote_result = self._evidence_from_trade_result(
                    option_ticker,
                    date,
                    trade,
                )

        return quote_result

    async def fetch_quote_at_timestamp(
        self,
        option_ticker: str,
        timestamp,
        allow_trade_fallback: bool = True,
    ) -> Optional[HistoricalFillEvidence]:
        """
        Get the latest quote at or before a specific execution timestamp.

        The timestamp should be timezone-aware whenever possible. This is used
        for execution-quality analysis where the fill must be compared against
        the quote stream closest to the actual order timestamp.
        """
        if isinstance(timestamp, str):
            ts_text = timestamp.replace("Z", "+00:00")
            try:
                ts_dt = datetime.fromisoformat(ts_text)
            except ValueError:
                return None
        elif isinstance(timestamp, datetime):
            ts_dt = timestamp
        else:
            return None

        if ts_dt.tzinfo is None:
            return None
        ts_utc = ts_dt.astimezone(timezone.utc)
        ts_iso = ts_utc.isoformat().replace("+00:00", "Z")
        pricing_date = ts_utc.astimezone(
            pytz.timezone("America/New_York")
        ).strftime("%Y-%m-%d")
        bounds = _regular_session_bounds_utc(pricing_date)
        if bounds is None:
            return None
        market_open, market_close = bounds
        if ts_utc < market_open or ts_utc > market_close:
            return None

        print(f"{CLR_YEL}  [API FETCH] Quote at {ts_iso} for {option_ticker}{CLR_RST}")
        url = f"{BASE_URL}/v3/quotes/{option_ticker}"
        params = {
            "timestamp.gte": market_open.isoformat().replace("+00:00", "Z"),
            "timestamp.lte": ts_iso,
            "order": "desc",
            "sort": "timestamp",
            "limit": 100,
        }

        data = await self._get(url, params)
        quote_result: Optional[HistoricalFillEvidence] = None
        if data and "results" in data and data["results"]:
            for quote in data["results"]:
                response_ticker = quote.get("ticker") or quote.get("option_ticker")
                if response_ticker is not None and response_ticker != option_ticker:
                    continue
                try:
                    bid = _finite_number(quote.get("bid_price"), "bid")
                    ask = _finite_number(quote.get("ask_price"), "ask")
                    quote_result = HistoricalFillEvidence(
                        option_ticker=option_ticker,
                        pricing_date=pricing_date,
                        source=HistoricalFillSource.OBSERVED_NBBO,
                        bid=bid,
                        ask=ask,
                        mid=round((bid + ask) / 2.0, 4),
                        bid_size=quote.get("bid_size") or 0,
                        ask_size=quote.get("ask_size") or 0,
                        event_timestamp_utc=self._quote_timestamp(quote),
                        provider="massive",
                        provider_route=f"/v3/quotes/{option_ticker}",
                    )
                except HistoricalFillEvidenceError:
                    continue
                break

        if not quote_result and allow_trade_fallback:
            print(f"{CLR_YEL}  [API FETCH] No valid quote at {ts_iso} for {option_ticker}, trying TRADES...{CLR_RST}")
            trade = await self.fetch_latest_trade_at_timestamp(option_ticker, ts_iso)
            if trade:
                quote_result = self._evidence_from_trade_result(
                    option_ticker,
                    pricing_date,
                    trade,
                )

        return quote_result

    async def fetch_latest_trade(
        self,
        option_ticker: str,
        date: str,
    ) -> Optional[dict]:
        """
        Fetch the last trade for a contract on a specific date.
        """
        cached = self.cache.get_ohlcv(option_ticker, date)
        if cached:
            close_price = cached.get("close")
            if (
                type(close_price) in (int, float)
                and type(close_price) is not bool
                and math.isfinite(float(close_price))
                and close_price > 0
            ):
                self.cache_hits += 1
                return {
                    "price": close_price,
                    "size": cached.get("volume") or 0,
                    "timestamp": None,
                    "source": HistoricalFillSource.DAILY_CLOSE.value,
                }

        if self._date_before(date, OPTION_TRADE_HISTORY_START_DATE):
            return None

        if self.offline_only:
            return None

        bounds = _regular_session_bounds_utc(date)
        if bounds is None:
            return None
        market_open, market_close = bounds
        url = f"{BASE_URL}/v3/trades/{option_ticker}"
        params = {
            "timestamp.gte": market_open.isoformat().replace("+00:00", "Z"),
            "timestamp.lte": market_close.isoformat().replace("+00:00", "Z"),
            "order": "desc",
            "sort": "timestamp",
            "limit": 1,
        }
        data = await self._get(url, params)
        if data and "results" in data and data["results"]:
            t = data["results"][0]
            if not isinstance(t, Mapping):
                return None
            response_ticker = t.get("ticker") or t.get("option_ticker")
            if response_ticker is not None and response_ticker != option_ticker:
                return None
            res = {
                "price": t.get("price"),
                "size": t.get("size"),
                "timestamp": self._format_timestamp(
                    t.get("participant_timestamp") or t.get("sip_timestamp") or t.get("timestamp")
                ),
                "source": HistoricalFillSource.TRADE_PRINT.value,
            }
            return res
        return None

    async def fetch_latest_trade_at_timestamp(
        self,
        option_ticker: str,
        timestamp,
    ) -> Optional[dict]:
        """
        Fetch the latest trade at or before a specific timestamp.
        """
        if isinstance(timestamp, str):
            ts_text = timestamp.replace("Z", "+00:00")
            try:
                ts_dt = datetime.fromisoformat(ts_text)
            except ValueError:
                return None
        elif isinstance(timestamp, datetime):
            ts_dt = timestamp
        else:
            return None

        if ts_dt.tzinfo is None:
            return None
        ts_utc = ts_dt.astimezone(pytz.utc)
        ts_iso = ts_utc.isoformat().replace("+00:00", "Z")
        day_start = ts_utc.strftime("%Y-%m-%dT00:00:00Z")

        if self.offline_only:
            return None

        url = f"{BASE_URL}/v3/trades/{option_ticker}"
        params = {
            "timestamp.gte": day_start,
            "timestamp.lte": ts_iso,
            "order": "desc",
            "sort": "timestamp",
            "limit": 1,
        }
        data = await self._get(url, params)
        if data and "results" in data and data["results"]:
            t = data["results"][0]
            if not isinstance(t, Mapping):
                return None
            response_ticker = t.get("ticker") or t.get("option_ticker")
            if response_ticker is not None and response_ticker != option_ticker:
                return None
            return {
                "price": t.get("price"),
                "size": t.get("size"),
                "timestamp": self._format_timestamp(
                    t.get("participant_timestamp") or t.get("sip_timestamp") or t.get("timestamp")
                ),
                "source": HistoricalFillSource.TRADE_PRINT.value,
            }
        return None

    # ── Batch Operations ────────────────────────────────────────────

    async def fetch_chain_ohlcv_batch(
        self,
        contracts: List[dict],
        from_date: str,
        to_date: str,
        underlying: str,
        contract_type: str,
        expiration: str,
    ) -> int:
        """
        Batch fetch daily OHLCV for all contracts in a chain.
        Skips contracts already fully cached OR previously fetched with empty results.
        Returns number of API calls made.
        """
        tasks = []
        for c in contracts:
            ticker = c["option_ticker"]
            strike = c["strike"]

            # Skip only if the requested range was explicitly fetched before.
            # Existing rows alone may represent a partial range from a prior run.
            if self.cache.is_ticker_range_fetched(ticker, from_date, to_date):
                self.cache_hits += 1
                continue

            tasks.append(
                self.fetch_contract_daily_bars(
                    ticker, from_date, to_date,
                    underlying=underlying,
                    contract_type=contract_type,
                    strike=strike,
                    expiration=expiration,
                )
            )

        if tasks:
            logger.info(f"  Fetching {len(tasks)} contracts for exp={expiration} ({len(contracts)-len(tasks)} cached)")
            await asyncio.gather(*tasks, return_exceptions=True)

        return len(tasks)

    async def fetch_synchronized_minute_aggregates(
        self,
        option_tickers: Sequence[str],
        date: str,
        max_time_delta_minutes: float = 5.0,
    ) -> Optional[Tuple[HistoricalFillEvidence, ...]]:
        """Return a two-or-more-leg synchronized aggregate research fallback.

        Minute aggregate closes are explicitly not NBBO. Legacy
        ``is_synchronized`` cache flags are ignored because those rows do not
        retain event timestamps or source provenance.
        """

        tickers = tuple(option_tickers)
        if len(tickers) < 2 or len(set(tickers)) != len(tickers):
            return None
        if any(type(ticker) is not str or not ticker for ticker in tickers):
            return None
        if self.offline_only or _regular_session_bounds_utc(date) is None:
            return None
        if (
            type(max_time_delta_minutes) not in (int, float)
            or type(max_time_delta_minutes) is bool
            or not math.isfinite(float(max_time_delta_minutes))
            or float(max_time_delta_minutes) < 0
        ):
            return None

        print(
            f"{CLR_YEL}  [API FETCH] Synchronizing 1m aggregates for "
            f"{len(tickers)} legs on {date}{CLR_RST}"
        )
        urls = [
            f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
            for ticker in tickers
        ]
        responses = await asyncio.gather(
            *[
                self._get(
                    url,
                    {
                        "adjusted": "false",
                        "sort": "desc",
                        "limit": 1440,
                    },
                )
                for url in urls
            ],
            return_exceptions=True,
        )

        streams: List[List[Tuple[float, HistoricalFillEvidence]]] = []
        for ticker, route_url, response in zip(tickers, urls, responses):
            if isinstance(response, Exception) or not isinstance(response, Mapping):
                return None
            rows = response.get("results")
            if not isinstance(rows, list) or not rows:
                return None
            stream: List[Tuple[float, HistoricalFillEvidence]] = []
            provider_route = urlsplit(route_url).path
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                response_ticker = row.get("ticker") or row.get(
                    "option_ticker"
                )
                if (
                    response_ticker is not None
                    and response_ticker != ticker
                ):
                    continue
                try:
                    timestamp_ms = _finite_number(row.get("t"), "aggregate timestamp")
                    close_price = _finite_number(row.get("c"), "aggregate close")
                    if close_price <= 0:
                        continue
                    item = HistoricalFillEvidence(
                        option_ticker=ticker,
                        pricing_date=date,
                        source=HistoricalFillSource.SYNCHRONIZED_MINUTE_AGGREGATE,
                        mid=close_price,
                        bid=close_price,
                        ask=close_price,
                        event_timestamp_utc=self._format_timestamp(timestamp_ms),
                        provider="massive",
                        provider_route=provider_route,
                    )
                except HistoricalFillEvidenceError:
                    continue
                normalized_timestamp_ms = (
                    _parse_explicit_utc_timestamp(
                        item.event_timestamp_utc
                    ).timestamp()
                    * 1000.0
                )
                stream.append((normalized_timestamp_ms, item))
            if not stream:
                return None
            stream.sort(key=lambda pair: pair[0])
            streams.append(stream)

        timestamp_lists = [[pair[0] for pair in stream] for stream in streams]
        anchors = sorted(
            {timestamp for timestamps in timestamp_lists for timestamp in timestamps},
            reverse=True,
        )
        max_delta_ms = float(max_time_delta_minutes) * 60.0 * 1000.0
        for anchor in anchors:
            selected: List[Tuple[float, HistoricalFillEvidence]] = []
            for timestamps, stream in zip(timestamp_lists, streams):
                index = bisect_left(timestamps, anchor)
                candidate_indexes = [
                    candidate
                    for candidate in (index - 1, index)
                    if 0 <= candidate < len(stream)
                ]
                if not candidate_indexes:
                    selected = []
                    break
                best_index = min(
                    candidate_indexes,
                    key=lambda candidate: abs(timestamps[candidate] - anchor),
                )
                selected.append(stream[best_index])
            if not selected:
                continue
            selected_timestamps = [pair[0] for pair in selected]
            if max(selected_timestamps) - min(selected_timestamps) <= max_delta_ms:
                return tuple(pair[1] for pair in selected)
        return None

    async def fetch_synchronized_ohlcv(
        self,
        ticker_a: str,
        ticker_b: str,
        date: str,
        max_time_delta_minutes: float = 5.0,
    ) -> Optional[Tuple[HistoricalFillEvidence, HistoricalFillEvidence]]:
        """Compatibility wrapper for the typed two-leg aggregate fallback."""

        evidence = await self.fetch_synchronized_minute_aggregates(
            (ticker_a, ticker_b),
            date,
            max_time_delta_minutes=max_time_delta_minutes,
        )
        if evidence is None:
            return None
        return evidence[0], evidence[1]

    async def fetch_multileg_eod_marks(
        self,
        option_tickers: Sequence[str],
        date: str,
        *,
        strict_nbbo: bool,
        max_time_delta_minutes: float,
        max_quote_age_minutes: float = 5.0,
    ) -> Optional[Tuple[HistoricalFillEvidence, ...]]:
        """Fetch one coherent two- or three-leg historical mark bundle."""

        max_time_delta = _nonnegative_finite_minutes(
            max_time_delta_minutes,
            "max_time_delta_minutes",
        )
        max_quote_age = _positive_finite_minutes(
            max_quote_age_minutes,
            "max_quote_age_minutes",
        )
        tickers = tuple(option_tickers)
        if len(tickers) not in {2, 3} or len(set(tickers)) != len(tickers):
            return None
        if strict_nbbo:
            responses = await asyncio.gather(
                *[
                    self.fetch_observed_nbbo(
                        ticker,
                        date,
                        max_quote_age_minutes=max_quote_age,
                    )
                    for ticker in tickers
                ],
                return_exceptions=True,
            )
            if any(
                isinstance(item, Exception) or item is None for item in responses
            ):
                return None
            try:
                return validate_strict_nbbo_bundle(
                    responses,
                    tickers,
                    date,
                    max_time_delta,
                    max_quote_age,
                )
            except HistoricalFillEvidenceError:
                return None

        responses = await asyncio.gather(
            *[
                self.fetch_eod_quote(
                    ticker,
                    date,
                    allow_trade_fallback=True,
                    strict_nbbo=False,
                    max_quote_age_minutes=max_quote_age,
                )
                for ticker in tickers
            ],
            return_exceptions=True,
        )
        complete = not any(
            isinstance(item, Exception) or item is None for item in responses
        )
        if complete:
            evidence = tuple(responses)
            diagnostics = fill_bundle_diagnostics(
                evidence,
                max_time_delta,
                max_quote_age,
            )
            if (
                all(
                    item.source is HistoricalFillSource.OBSERVED_NBBO
                    for item in evidence
                )
                and diagnostics["temporally_synchronized"] is False
            ):
                synchronized = await self.fetch_synchronized_minute_aggregates(
                    tickers,
                    date,
                    max_time_delta_minutes=max_time_delta,
                )
                return synchronized or evidence
            return evidence

        return await self.fetch_synchronized_minute_aggregates(
            tickers,
            date,
            max_time_delta_minutes=max_time_delta,
        )

    async def fetch_multileg_eod_fills(
        self,
        option_tickers: Sequence[str],
        date: str,
        *,
        strict_nbbo: bool,
        max_time_delta_minutes: float,
        max_quote_age_minutes: float = 5.0,
    ) -> Optional[Tuple[HistoricalFillEvidence, ...]]:
        """Deprecated compatibility name; this returns marks, not fills."""

        return await self.fetch_multileg_eod_marks(
            option_tickers,
            date,
            strict_nbbo=strict_nbbo,
            max_time_delta_minutes=max_time_delta_minutes,
            max_quote_age_minutes=max_quote_age_minutes,
        )

    async def fetch_theoretical_price(
        self,
        target_ticker: str,
        reference_ticker: str,
        reference_price: float,
        underlying_price: float,
        date: str,
        dte_years: float,
        risk_free_rate: float,
        option_type: str,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Estimate price of target_ticker using the implied vol of reference_ticker.
        Used when the deep OTM leg (target) has no trades but the closer leg (reference) does.
        """
        if self.offline_only:
            return None

        from backtesting.greeks_calculator import (
            bs_call_price,
            bs_put_price,
            implied_volatility,
        )

        normalized_option_type = str(option_type).strip().lower()
        if normalized_option_type not in {"put", "call"}:
            raise ValueError("option_type must be 'put' or 'call'")
        expected_flag = "C" if normalized_option_type == "call" else "P"
        if any(
            len(ticker) < 9 or ticker[-9].upper() != expected_flag
            for ticker in (target_ticker, reference_ticker)
        ):
            raise ValueError(
                "option_type does not match target/reference contract symbols"
            )
        
        # 1. Parse strikes from tickers
        # O:SPY150117P00200000 -> 200.0
        try:
            ref_strike = float(reference_ticker[-8:]) / 1000.0
            tgt_strike = float(target_ticker[-8:]) / 1000.0
        except (TypeError, ValueError):
            return None

        # 2. Solve IV for reference leg
        iv = implied_volatility(
            reference_price,
            underlying_price,
            ref_strike,
            dte_years,
            risk_free_rate,
            dividend_yield,
            normalized_option_type,
        )
        if iv is None:
            return None
            
        # 3. Calculate theoretical price for target leg
        price_function = (
            bs_call_price
            if normalized_option_type == "call"
            else bs_put_price
        )
        theo_price = price_function(
            underlying_price,
            tgt_strike,
            dte_years,
            risk_free_rate,
            iv,
            dividend_yield,
        )
        theo_price = round(theo_price, 4)
        
        return theo_price

    async def fetch_chain_quotes_batch(
        self,
        contracts: List[dict],
        date: str,
        underlying: str,
        contract_type: str,
        expiration: str,
    ) -> Dict[float, dict]:
        """
        Batch fetch EOD quotes for all contracts in a chain on a date.
        Returns {strike: {"bid", "ask", "mid"}} dict.
        """
        results = {}

        async def _fetch_one(c):
            ticker = c["option_ticker"]
            strike = c["strike"]

            quote = await self.fetch_eod_quote(ticker, date)
            if quote:
                results[strike] = quote

        tasks = [_fetch_one(c) for c in contracts]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results
    async def fetch_option_snapshot(self, underlying: str) -> List[dict]:
        """
        Fetch full snapshot for all options of an underlying.
        Includes Greeks, IV, and Open Interest if available.
        """
        if self.offline_only:
            return []

        print(f"{CLR_YEL}  [API FETCH] Full Option Snapshot for {underlying}{CLR_RST}")
        url = f"{BASE_URL}/v3/snapshot/options/{underlying}"
        params = {"limit": 250, "order": "asc", "sort": "ticker"}
        
        all_results = []
        while url:
            data = await self._get(url, params)
            if not data or "results" not in data:
                break
            all_results.extend(data["results"])
            
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {}
            else:
                break
        return all_results
