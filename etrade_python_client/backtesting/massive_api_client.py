"""
Async HTTP client for the Massive API (formerly Polygon.io).
Cache-aware: checks OptionDataCache before making API calls.
Supports batch request aggregation with semaphore-based concurrency.
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import ssl
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
import certifi
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
    ) -> Optional[dict]:
        """Return cached quote, OHLCV close, or trade midpoint for this option/date."""
        cached = self.cache.get_ohlcv(option_ticker, date)
        if not cached:
            return None

        bid = cached.get("bid")
        ask = cached.get("ask")
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid != ask:
            self.cache_hits += 1
            return {
                "bid": bid,
                "ask": ask,
                "mid": cached.get("mid") or round((bid + ask) / 2.0, 4),
                "bid_size": cached.get("bid_size") or 0,
                "ask_size": cached.get("ask_size") or 0,
                "source": "quote_cache",
            }

        close_price = cached.get("close")
        if close_price is not None and close_price > 0:
            self.cache_hits += 1
            return {
                "bid": close_price,
                "ask": close_price,
                "mid": close_price,
                "bid_size": 0,
                "ask_size": 0,
                "source": "ohlcv_close_cache",
            }

        mid = cached.get("mid")
        if allow_trade_fallback and mid is not None and mid > 0:
            self.cache_hits += 1
            return {
                "bid": mid,
                "ask": mid,
                "mid": mid,
                "bid_size": cached.get("bid_size") or 0,
                "ask_size": cached.get("ask_size") or 0,
                "source": "trade_cache",
            }

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

    async def fetch_eod_quote(
        self,
        option_ticker: str,
        date: str,
        allow_trade_fallback: bool = True,
    ) -> Optional[dict]:
        """
        Get end-of-day NBBO quote for a contract on a specific date.
        Returns {"bid": float, "ask": float, "mid": float, ...} or None.
        """
        cached_price = self._cached_price_result(
            option_ticker,
            date,
            allow_trade_fallback=allow_trade_fallback,
        )
        if cached_price:
            return cached_price

        if self.offline_only:
            return None

        quote_available = not self._date_before(date, OPTION_QUOTE_HISTORY_START_DATE)

        # For historical dates before quote coverage starts, do not waste an API
        # request on /v3/quotes. Go directly to trades when fallback is allowed.
        if not quote_available:
            if not allow_trade_fallback:
                return None
            print(
                f"{CLR_YEL}  [API FETCH] Quote history unavailable before "
                f"{OPTION_QUOTE_HISTORY_START_DATE}; fetching TRADES for {option_ticker} on {date}{CLR_RST}"
            )
            trade = await self.fetch_latest_trade(option_ticker, date)
            if not trade:
                return None
            return {
                "bid": trade["price"],
                "ask": trade["price"],
                "mid": trade["price"],
                "bid_size": trade.get("size", 0),
                "ask_size": trade.get("size", 0),
                "source": "trade",
                "quote_timestamp": self._format_timestamp(trade.get("timestamp")),
            }

        # Check negative quote cache after checking actual cached prices. A prior
        # failed quote fetch may still have a cached trade midpoint for the day.
        if self.cache.is_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="quote"):
            self.cache_hits += 1
            return None

        # 3. Fetch from API
        print(f"{CLR_YEL}  [API FETCH] EOD Quote for {option_ticker} on {date}{CLR_RST}")
        url = f"{BASE_URL}/v3/quotes/{option_ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lte": f"{date}T23:59:59Z", # Search up to end of day
            "order": "desc",
            "sort": "timestamp",
            "limit": 100,
        }

        data = await self._get(url, params)
        quote_result = None
        
        if data and "results" in data and data["results"]:
            # Find first quote with valid bid and ask, filtering blown-out spreads
            for quote in data["results"]:
                bid = quote.get("bid_price", 0)
                ask = quote.get("ask_price", 0)
                if bid > 0 and ask > 0:
                    # Reject blown-out spreads (ask-bid > 200% of bid)
                    spread_ratio = (ask - bid) / bid
                    if spread_ratio > 2.0:
                        continue
                    quote_result = {
                        "bid": bid,
                        "ask": ask,
                        "mid": round((bid + ask) / 2.0, 4),
                        "bid_size": quote.get("bid_size", 0),
                        "ask_size": quote.get("ask_size", 0),
                        "source": "quote",
                        "quote_timestamp": self._format_timestamp(
                            quote.get("participant_timestamp")
                            or quote.get("sip_timestamp")
                            or quote.get("timestamp")
                        ),
                    }
                    break

        # 4. Fallback to Actual Trades if Quote is missing or bad
        if not quote_result and allow_trade_fallback:
            print(f"{CLR_YEL}  [API FETCH] No valid quote for {option_ticker} on {date}, trying TRADES...{CLR_RST}")
            trade = await self.fetch_latest_trade(option_ticker, date)
            if trade:
                print(f"  [TRADE FALLBACK] Found trade for {option_ticker} at ${trade['price']}")
                quote_result = {
                    "bid": trade["price"],
                    "ask": trade["price"],
                    "mid": trade["price"],
                    "bid_size": trade.get("size", 0),
                    "ask_size": trade.get("size", 0),
                    "source": "trade",
                    "quote_timestamp": self._format_timestamp(trade.get("timestamp")),
                }

        # 5. Save to cache if found
        if quote_result:
            meta = self._parse_ticker(option_ticker)
            record = {
                "option_ticker": option_ticker,
                "pricing_date": date,
                "mid": quote_result["mid"],
                "bid_size": quote_result.get("bid_size"),
                "ask_size": quote_result.get("ask_size"),
                "fetched_at": datetime.now().isoformat(),
                **meta
            }
            if quote_result.get("source") == "quote":
                record["bid"] = quote_result["bid"]
                record["ask"] = quote_result["ask"]
            self.cache.upsert_full_record(record)
        else:
            # Mark as empty result so we don't try again
            self.cache.mark_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="quote")

        return quote_result

    async def fetch_quote_at_timestamp(
        self,
        option_ticker: str,
        timestamp,
        allow_trade_fallback: bool = True,
    ) -> Optional[dict]:
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
            ts_dt = ts_dt.replace(tzinfo=pytz.utc)
        ts_utc = ts_dt.astimezone(pytz.utc)
        ts_iso = ts_utc.isoformat().replace("+00:00", "Z")
        day_start = ts_utc.strftime("%Y-%m-%dT00:00:00Z")

        print(f"{CLR_YEL}  [API FETCH] Quote at {ts_iso} for {option_ticker}{CLR_RST}")
        url = f"{BASE_URL}/v3/quotes/{option_ticker}"
        params = {
            "timestamp.gte": day_start,
            "timestamp.lte": ts_iso,
            "order": "desc",
            "sort": "timestamp",
            "limit": 100,
        }

        data = await self._get(url, params)
        quote_result = None
        if data and "results" in data and data["results"]:
            for quote in data["results"]:
                bid = quote.get("bid_price", 0)
                ask = quote.get("ask_price", 0)
                if bid > 0 and ask > 0:
                    spread_ratio = (ask - bid) / bid if bid else float("inf")
                    if spread_ratio > 2.0:
                        continue
                    quote_result = {
                        "bid": bid,
                        "ask": ask,
                        "mid": round((bid + ask) / 2.0, 4),
                        "bid_size": quote.get("bid_size", 0),
                        "ask_size": quote.get("ask_size", 0),
                        "source": "quote",
                        "quote_timestamp": self._format_timestamp(
                            quote.get("participant_timestamp")
                            or quote.get("sip_timestamp")
                            or quote.get("timestamp")
                        ),
                    }
                    break

        if not quote_result and allow_trade_fallback:
            print(f"{CLR_YEL}  [API FETCH] No valid quote at {ts_iso} for {option_ticker}, trying TRADES...{CLR_RST}")
            trade = await self.fetch_latest_trade_at_timestamp(option_ticker, ts_iso)
            if trade:
                quote_result = {
                    "bid": trade["price"],
                    "ask": trade["price"],
                    "mid": trade["price"],
                    "bid_size": trade.get("size", 0),
                    "ask_size": trade.get("size", 0),
                    "source": "trade",
                    "quote_timestamp": self._format_timestamp(trade.get("timestamp")),
                }

        return quote_result

    async def fetch_latest_trade(
        self,
        option_ticker: str,
        date: str,
    ) -> Optional[dict]:
        """
        Fetch the last trade for a contract on a specific date.
        """
        trade_fetch_complete = self.cache.is_fetch_complete(
            underlying="",
            pricing_date=date,
            expiration=option_ticker,
            data_type="trade",
        )
        cached = self.cache.get_ohlcv(option_ticker, date)
        if cached:
            close_price = cached.get("close")
            if close_price is not None and close_price > 0:
                self.cache_hits += 1
                return {
                    "price": close_price,
                    "size": cached.get("volume") or 0,
                    "timestamp": None,
                    "source": "ohlcv_close_cache",
                }
            mid = cached.get("mid")
            if trade_fetch_complete and mid is not None and mid > 0:
                self.cache_hits += 1
                return {
                    "price": mid,
                    "size": cached.get("bid_size") or cached.get("ask_size") or 0,
                    "timestamp": None,
                    "source": "trade_cache",
                }

        if self._date_before(date, OPTION_TRADE_HISTORY_START_DATE):
            return None

        if trade_fetch_complete:
            self.cache_hits += 1
            return None

        if self.offline_only:
            self.cache.mark_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="trade")
            return None

        url = f"{BASE_URL}/v3/trades/{option_ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lte": f"{date}T23:59:59Z",
            "order": "desc",
            "sort": "timestamp",
            "limit": 1,
        }
        data = await self._get(url, params)
        if data and "results" in data and data["results"]:
            t = data["results"][0]
            res = {
                "price": t.get("price"),
                "size": t.get("size"),
                "timestamp": self._format_timestamp(
                    t.get("participant_timestamp") or t.get("sip_timestamp") or t.get("timestamp")
                ),
            }
            # Cache this as the latest trade midpoint for EOD fallback reuse.
            meta = self._parse_ticker(option_ticker)
            self.cache.upsert_full_record({
                "option_ticker": option_ticker,
                "pricing_date": date,
                "mid": res["price"],
                "fetched_at": datetime.now().isoformat(),
                **meta
            })
            self.cache.mark_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="trade")
            return res
        self.cache.mark_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="trade")
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
            ts_dt = ts_dt.replace(tzinfo=pytz.utc)
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
            return {
                "price": t.get("price"),
                "size": t.get("size"),
                "timestamp": self._format_timestamp(
                    t.get("participant_timestamp") or t.get("sip_timestamp") or t.get("timestamp")
                ),
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

    async def fetch_synchronized_ohlcv(
        self,
        ticker_a: str,
        ticker_b: str,
        date: str,
        max_time_delta_minutes: float = 5.0,
    ) -> Optional[Tuple[dict, dict]]:
        """
        Fetch 1-minute aggregates for both legs and find the latest minute where both traded
        within max_time_delta_minutes of each other.
        Returns (quote_a, quote_b) where each is {"bid", "ask", "mid", "timestamp"} or None.
        """
        # Check cache first for both tickers on this date
        cached_a = self.cache.get_ohlcv(ticker_a, date)
        cached_b = self.cache.get_ohlcv(ticker_b, date)

        # If either ticker was already queried and found to have no daily bar (empty day),
        # or if the cached record has no daily close and no bid/ask quote (not traded),
        # we can't possibly have synchronized 1-minute bars, so skip API calls entirely.
        if (cached_a is None and self.cache.is_ticker_range_fetched(ticker_a, date, date)) or \
           (cached_a is not None and (cached_a.get("close") is None or cached_a.get("close") == 0) and cached_a.get("bid") is None):
            return None
        if (cached_b is None and self.cache.is_ticker_range_fetched(ticker_b, date, date)) or \
           (cached_b is not None and (cached_b.get("close") is None or cached_b.get("close") == 0) and cached_b.get("bid") is None):
            return None

        if self.offline_only:
            return None

        def is_sync(rec):
            if not rec:
                return False
            # Check explicit flag
            if rec.get("is_synchronized") == 1:
                return True
            # Check if valid EOD quote
            bid = rec.get("bid")
            ask = rec.get("ask")
            if bid is not None and ask is not None and bid > 0 and ask > 0 and bid != ask:
                return True
            # Check if valid daily OHLCV close
            if rec.get("close") is not None and rec.get("close") > 0:
                return True
            return False

        if cached_a and cached_b and is_sync(cached_a) and is_sync(cached_b):
            self.cache_hits += 1
            mid_a = cached_a.get("mid") or cached_a.get("close")
            mid_b = cached_b.get("mid") or cached_b.get("close")
            return (
                {
                    "bid": cached_a.get("bid") or mid_a,
                    "ask": cached_a.get("ask") or mid_a,
                    "mid": mid_a,
                    "timestamp": 0,
                },
                {
                    "bid": cached_b.get("bid") or mid_b,
                    "ask": cached_b.get("ask") or mid_b,
                    "mid": mid_b,
                    "timestamp": 0,
                }
            )

        print(f"{CLR_YEL}  [API FETCH] Synchronizing 1m bars for {ticker_a} & {ticker_b} on {date}{CLR_RST}")
        
        # 1. Fetch both 1m aggregate streams. A cached daily mid, quote, or
        # theoretical price is not enough to prove the two legs are synchronized.
        url_a = f"{BASE_URL}/v2/aggs/ticker/{ticker_a}/range/1/minute/{date}/{date}"
        url_b = f"{BASE_URL}/v2/aggs/ticker/{ticker_b}/range/1/minute/{date}/{date}"
        params = {"adjusted": "false", "sort": "desc", "limit": 1440}

        data_a, data_b = await asyncio.gather(
            self._get(url_a, params),
            self._get(url_b, params),
            return_exceptions=True,
        )
        if isinstance(data_a, Exception) or not data_a:
            results_a = []
        else:
            results_a = data_a.get("results", [])
        if isinstance(data_b, Exception) or not data_b:
            results_b = []
        else:
            results_b = data_b.get("results", [])

        if not results_a or not results_b:
            return None

        # 2. Find valid pairs within max_time_delta_minutes.
        max_delta_ms = max_time_delta_minutes * 60.0 * 1000.0
        valid_pairs = []
        for bar_a in results_a:
            ts_a = bar_a["t"]
            for bar_b in results_b:
                ts_b = bar_b["t"]
                delta_ms = abs(ts_a - ts_b)
                if delta_ms <= max_delta_ms:
                    valid_pairs.append((bar_a, bar_b, ts_a, ts_b))

        if not valid_pairs:
            return None

        # Sort valid pairs by average timestamp descending (latest time in day / closest to close)
        valid_pairs.sort(key=lambda x: (x[2] + x[3]) / 2.0, reverse=True)
        best_pair = valid_pairs[0]
        bar_a, bar_b, ts_a, ts_b = best_pair

        price_a = bar_a["c"]
        price_b = bar_b["c"]
        
        # Treat OHLCV close as mid for both
        quote_a = {"bid": price_a, "ask": price_a, "mid": price_a, "timestamp": ts_a}
        quote_b = {"bid": price_b, "ask": price_b, "mid": price_b, "timestamp": ts_b}
        
        # Cache them (upsert style)
        meta_a = self._parse_ticker(ticker_a)
        meta_b = self._parse_ticker(ticker_b)
        
        self.cache.upsert_full_record({
            "option_ticker": ticker_a, 
            "pricing_date": date, 
            "mid": price_a, 
            "is_synchronized": 1,
            "fetched_at": datetime.now().isoformat(),
            **meta_a
        })
        self.cache.upsert_full_record({
            "option_ticker": ticker_b, 
            "pricing_date": date, 
            "mid": price_b, 
            "is_synchronized": 1,
            "fetched_at": datetime.now().isoformat(),
            **meta_b
        })

        return (quote_a, quote_b)

    async def fetch_theoretical_price(
        self,
        target_ticker: str,
        reference_ticker: str,
        reference_price: float,
        underlying_price: float,
        date: str,
        dte_years: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Estimate price of target_ticker using the implied vol of reference_ticker.
        Used when the deep OTM leg (target) has no trades but the closer leg (reference) does.
        """
        # Check cache first
        cached = self.cache.get_ohlcv(target_ticker, date)
        if cached and cached.get("mid") is not None:
            self.cache_hits += 1
            return cached["mid"]

        if self.offline_only:
            return None

        from backtesting.greeks_calculator import implied_volatility, bs_put_price
        
        # 1. Parse strikes from tickers
        # O:SPY150117P00200000 -> 200.0
        try:
            ref_strike = float(reference_ticker[-8:]) / 1000.0
            tgt_strike = float(target_ticker[-8:]) / 1000.0
        except:
            return None

        # 2. Solve IV for reference leg
        iv = implied_volatility(reference_price, underlying_price, ref_strike, dte_years, risk_free_rate, dividend_yield, "put")
        if iv is None:
            return None
            
        # 3. Calculate theoretical price for target leg
        theo_price = bs_put_price(underlying_price, tgt_strike, dte_years, risk_free_rate, iv, dividend_yield)
        theo_price = round(theo_price, 4)
        
        # Cache it
        meta = self._parse_ticker(target_ticker)
        self.cache.upsert_full_record({
            "option_ticker": target_ticker,
            "pricing_date": date,
            "mid": theo_price,
            "fetched_at": datetime.now().isoformat(),
            **meta
        })

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
        update_rows = []

        async def _fetch_one(c):
            ticker = c["option_ticker"]
            strike = c["strike"]

            # Check cache first (allows mid-point from fallbacks)
            cached = self.cache.get_ohlcv(ticker, date)
            if cached and cached.get("mid") is not None:
                self.cache_hits += 1
                results[strike] = {
                    "bid": cached.get("bid"),
                    "ask": cached.get("ask"),
                    "mid": cached["mid"],
                }
                return

            quote = await self.fetch_eod_quote(ticker, date)
            if quote:
                results[strike] = quote
                update_rows.append(
                    (
                        quote["bid"],
                        quote["ask"],
                        quote["mid"],
                        quote.get("bid_size"),
                        quote.get("ask_size"),
                        ticker,
                        date,
                    )
                )

        tasks = [_fetch_one(c) for c in contracts]
        await asyncio.gather(*tasks, return_exceptions=True)
        if update_rows:
            self.cache.bulk_update_quotes(update_rows)

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
