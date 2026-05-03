"""
Async HTTP client for the Massive API (formerly Polygon.io).
Cache-aware: checks OptionDataCache before making API calls.
Supports batch request aggregation with semaphore-based concurrency.
"""
import asyncio
import logging
import ssl
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
import certifi
import pytz

from backtesting.massive_config import API_KEY, BASE_URL
from backtesting.option_data_cache import OptionDataCache

logger = logging.getLogger(__name__)

# ANSI Colors for terminal logging
CLR_YEL = "\033[93m"
CLR_RST = "\033[0m"


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
    ):
        self.api_key = api_key
        self.cache = cache
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
            
            contract_type = "call" if rem[6] == 'C' else "put"
            
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

    # ── Contracts List ──────────────────────────────────────────────

    async def fetch_contracts_list(
        self,
        underlying: str,
        expiration: str,
        contract_type: str,
        as_of: str,
    ) -> List[dict]:
        """
        Get list of option contracts for an underlying/expiration.
        Returns list of {"option_ticker": str, "strike": float}.
        """
        # Check cache
        cached = self.cache.get_cached_contracts(
            underlying, expiration, contract_type, as_of
        )
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Fetch from API with pagination
        print(f"{CLR_YEL}  [API FETCH] {underlying} contracts for {expiration} (as_of {as_of}){CLR_RST}")
        all_contracts = []
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expiration_date": expiration,
            "contract_type": contract_type,
            "as_of": as_of,
            "limit": 1000,
            "order": "asc",
            "sort": "strike_price",
        }

        while url:
            data = await self._get(url, params)
            if not data or "results" not in data:
                break

            for item in data["results"]:
                all_contracts.append({
                    "option_ticker": item["ticker"],
                    "strike": item["strike_price"],
                })

            # Handle pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {}  # next_url includes all params
            else:
                break

        # Save to cache
        if all_contracts:
            self.cache.save_contracts(
                underlying, expiration, contract_type, as_of, all_contracts
            )

        return all_contracts

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
    ) -> List[dict]:
        """
        Fetch daily OHLCV bars for a single contract over a date range.
        Checks cache and fetch log before calling API.
        """
        # 1. Check if already cached (has actual data)
        cached = self.cache.get_ohlcv_range(option_ticker, from_date, to_date)
        if cached:
            # Check if we have the full range (simple check: count days)
            # For simplicity, if we have any data, we trust it for now
            self.cache_hits += 1
            return cached

        # 2. Check negative cache
        if self.cache.is_ticker_range_fetched(option_ticker, from_date, to_date):
            self.cache_hits += 1
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
    ) -> Optional[dict]:
        """
        Get end-of-day NBBO quote for a contract on a specific date.
        Returns {"bid": float, "ask": float, "mid": float, ...} or None.
        """
        # 1. Check negative cache first (did we already try and find nothing?)
        if self.cache.is_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="quote"):
            self.cache_hits += 1
            return None

        # 2. Check cache for actual data
        cached = self.cache.get_ohlcv(option_ticker, date)
        if cached and cached.get("mid") is not None:
            self.cache_hits += 1
            return {
                "bid": cached.get("bid"),
                "ask": cached.get("ask"),
                "mid": cached["mid"],
                "bid_size": cached.get("bid_size", 0),
                "ask_size": cached.get("ask_size", 0),
            }

        # 3. Fetch from API
        print(f"{CLR_YEL}  [API FETCH] EOD Quote for {option_ticker} on {date}{CLR_RST}")
        url = f"{BASE_URL}/v3/quotes/{option_ticker}"
        params = {
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
                    }
                    break

        # 4. Fallback to Actual Trades if Quote is missing or bad
        if not quote_result:
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
                }

        # 5. Save to cache if found
        if quote_result:
            meta = self._parse_ticker(option_ticker)
            self.cache.upsert_full_record({
                "option_ticker": option_ticker,
                "pricing_date": date,
                "bid": quote_result["bid"],
                "ask": quote_result["ask"],
                "mid": quote_result["mid"],
                "bid_size": quote_result.get("bid_size"),
                "ask_size": quote_result.get("ask_size"),
                "fetched_at": datetime.now().isoformat(),
                **meta
            })
        else:
            # Mark as empty result so we don't try again
            self.cache.mark_fetch_complete(underlying="", pricing_date=date, expiration=option_ticker, data_type="quote")

        return quote_result

    async def fetch_latest_trade(
        self,
        option_ticker: str,
        date: str,
    ) -> Optional[dict]:
        """
        Fetch the last trade for a contract on a specific date.
        """
        # Check cache first
        cached = self.cache.get_ohlcv(option_ticker, date)
        if cached and cached.get("mid") is not None:
            self.cache_hits += 1
            return {"price": cached["mid"], "timestamp": cached.get("fetched_at")}

        url = f"{BASE_URL}/v3/trades/{option_ticker}"
        params = {
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
                "timestamp": t.get("participant_timestamp"),
            }
            # Cache this as a mid-point quote
            meta = self._parse_ticker(option_ticker)
            self.cache.upsert_full_record({
                "option_ticker": option_ticker,
                "pricing_date": date,
                "mid": res["price"],
                "fetched_at": datetime.now().isoformat(),
                **meta
            })
            return res
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

            # Check if already cached for this range (has actual data)
            cached = self.cache.get_ohlcv_range(ticker, from_date, to_date)
            if cached:
                self.cache_hits += 1
                continue

            # Check negative cache: already fetched but API returned empty
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
    ) -> Optional[Tuple[dict, dict]]:
        """
        Fetch 1-minute aggregates for both legs and find the latest minute where both traded.
        Returns (quote_a, quote_b) where each is {"bid", "ask", "mid", "timestamp"} or None.
        """
        # Check cache for both legs first
        cached_a = self.cache.get_ohlcv(ticker_a, date)
        cached_b = self.cache.get_ohlcv(ticker_b, date)
        
        if cached_a and cached_a.get("mid") is not None and cached_b and cached_b.get("mid") is not None:
            self.cache_hits += 1
            return (
                {"bid": None, "ask": None, "mid": cached_a["mid"], "timestamp": cached_a.get("fetched_at")},
                {"bid": None, "ask": None, "mid": cached_b["mid"], "timestamp": cached_b.get("fetched_at")}
            )

        print(f"{CLR_YEL}  [API FETCH] Synchronizing 1m bars for {ticker_a} & {ticker_b} on {date}{CLR_RST}")
        
        # 1. Fetch 1m aggs for missing legs
        url_a = f"{BASE_URL}/v2/aggs/ticker/{ticker_a}/range/1/minute/{date}/{date}"
        url_b = f"{BASE_URL}/v2/aggs/ticker/{ticker_b}/range/1/minute/{date}/{date}"
        params = {"adjusted": "false", "sort": "desc", "limit": 1440}

        tasks = []
        if cached_a and cached_a.get("mid") is not None:
            tasks.append(asyncio.sleep(0)) # Placeholder
            results_a = [{"t": int(time.time()*1000), "c": cached_a["mid"]}] # Simulated single bar
        else:
            tasks.append(self._get(url_a, params))
            
        if cached_b and cached_b.get("mid") is not None:
            tasks.append(asyncio.sleep(0))
            results_b = [{"t": int(time.time()*1000), "c": cached_b["mid"]}]
        else:
            tasks.append(self._get(url_b, params))

        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update results from API if needed
        if not (cached_a and cached_a.get("mid") is not None):
            data_a = api_results[0]
            if isinstance(data_a, Exception) or not data_a: results_a = []
            else: results_a = data_a.get("results", [])
            
        if not (cached_b and cached_b.get("mid") is not None):
            # If both were missing, api_results[1] is data_b. 
            # If only B was missing, api_results[1] is data_b.
            # If only A was missing, api_results[1] is sleep(0).
            data_b = api_results[1]
            if isinstance(data_b, Exception) or not data_b: results_b = []
            else: results_b = data_b.get("results", [])

        if not results_a or not results_b:
            return None

        # 2. Find latest common minute
        # Map timestamp -> close price
        if not results_a or not results_b:
            return None
            
        # If one leg was simulated from cache (length 1), use it as the master price
        if len(results_a) == 1 and not (cached_a and cached_a.get("mid") is not None):
             # This case shouldn't happen with current logic, but for safety:
             pass
        
        # Standard matching logic
        map_b = {bar["t"]: bar["c"] for bar in results_b}
        
        # Special case: if one results list has only 1 entry (cached simulated), 
        # we can't match timestamps, so we just take the latest from the other leg.
        if len(results_a) == 1 and cached_a:
            price_a = results_a[0]["c"]
            price_b = results_b[0]["c"] # results_b is already sorted desc
            return ({"bid": price_a, "ask": price_a, "mid": price_a, "timestamp": date},
                    {"bid": price_b, "ask": price_b, "mid": price_b, "timestamp": date})
        
        if len(results_b) == 1 and cached_b:
            price_b = results_b[0]["c"]
            price_a = results_a[0]["c"]
            return ({"bid": price_a, "ask": price_a, "mid": price_a, "timestamp": date},
                    {"bid": price_b, "ask": price_b, "mid": price_b, "timestamp": date})

        for bar_a in results_a:
            ts = bar_a["t"]
            if ts in map_b:
                # Found a match!
                price_a = bar_a["c"]
                price_b = map_b[ts]
                
                # Treat OHLCV close as mid for both
                quote_a = {"bid": price_a, "ask": price_a, "mid": price_a, "timestamp": ts}
                quote_b = {"bid": price_b, "ask": price_b, "mid": price_b, "timestamp": ts}
                
                # Cache them (upsert style)
                meta_a = self._parse_ticker(ticker_a)
                meta_b = self._parse_ticker(ticker_b)
                
                self.cache.upsert_full_record({
                    "option_ticker": ticker_a, 
                    "pricing_date": date, 
                    "mid": price_a, 
                    "fetched_at": datetime.now().isoformat(),
                    **meta_a
                })
                self.cache.upsert_full_record({
                    "option_ticker": ticker_b, 
                    "pricing_date": date, 
                    "mid": price_b, 
                    "fetched_at": datetime.now().isoformat(),
                    **meta_b
                })

                return (quote_a, quote_b)

        return None

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
                # Update cache with quote data
                self.cache.conn.execute(
                    """UPDATE option_prices SET bid=?, ask=?, mid=?, bid_size=?, ask_size=?
                       WHERE option_ticker=? AND pricing_date=?""",
                    (quote["bid"], quote["ask"], quote["mid"],
                     quote.get("bid_size"), quote.get("ask_size"),
                     ticker, date),
                )

        tasks = [_fetch_one(c) for c in contracts]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.cache.conn.commit()

        return results
    async def fetch_option_snapshot(self, underlying: str) -> List[dict]:
        """
        Fetch full snapshot for all options of an underlying.
        Includes Greeks, IV, and Open Interest if available.
        """
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
