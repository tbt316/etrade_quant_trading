"""
Historical option backtest runner using Massive API data.

Supports:
  - Put credit spreads
  - Call credit spreads
  - Iron condors (future)

Strategy workflow:
  1. For each trading day, find target expiration (~42 DTE Friday)
  2. Fetch option chain and compute BS deltas
  3. Select short leg at target delta, long leg at spread_width below
  4. Use EOD NBBO midpoint as execution price
  5. Close at close_dte days remaining
"""
import asyncio
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas_market_calendars as mcal

from backtesting.option_data_cache import OptionDataCache
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.greeks_calculator import compute_chain_deltas, bs_put_delta, implied_volatility


# ANSI Colors for terminal logging
CLR_RED = "\033[91m"
CLR_YEL = "\033[93m"
CLR_RST = "\033[0m"


NYSE = mcal.get_calendar("NYSE")


@dataclass
class SpreadTrade:
    """Represents a single credit spread trade."""
    entry_date: str
    exit_date: str = ""
    expiration: str = ""
    short_strike: float = 0.0
    long_strike: float = 0.0
    short_entry_mid: float = 0.0
    long_entry_mid: float = 0.0
    net_credit: float = 0.0
    short_exit_mid: float = 0.0
    long_exit_mid: float = 0.0
    net_debit_close: float = 0.0
    pnl_per_share: float = 0.0
    pnl_per_contract: float = 0.0
    entry_delta: float = 0.0
    entry_dte: int = 0
    exit_dte: int = 0
    status: str = "open"     # "open", "closed", "expired", "error"
    num_contracts: int = 1
    margin_required: float = 0.0
    exit_reason: str = ""    # "scheduled", "early_profit", "expired"
    hold_to_expiration: bool = False # If ITM at close_dte, hold until expiration
    entry_regime: int = -1
    entry_regime_name: str = ""
    profit_target: float = 0.70 # Default to early_profit_pct
    short_ticker: str = ""
    long_ticker: str = ""
    entry_price: float = 0.0
    
    _exp_dt: Optional[datetime] = field(default=None, init=False, repr=False)
    
    @property
    def expiration_dt(self) -> datetime:
        if self._exp_dt is None:
            self._exp_dt = datetime.strptime(self.expiration, "%Y-%m-%d")
        return self._exp_dt


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    trades: List[SpreadTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    avg_credit: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    underlying_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    capital_history: List[Tuple[str, float]] = field(default_factory=list)
    cash_history: List[Tuple[str, float]] = field(default_factory=list)
    nvl_history: List[Tuple[str, float]] = field(default_factory=list)
    margin_history: List[Tuple[str, float]] = field(default_factory=list)
    vix_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    regime_history: List[Tuple[str, int]] = field(default_factory=list)
    regime_labels: Dict[int, str] = field(default_factory=dict)
    daily_leg_premiums: List[Tuple[str, float, float]] = field(default_factory=list)
    daily_scatter_data: List[dict] = field(default_factory=list)
    daily_opened_dte: List[Tuple[str, int]] = field(default_factory=list)
    daily_opened_spread: List[Tuple[str, float]] = field(default_factory=list)
    initial_capital: float = 100000.0
    margin_limit_pct: float = 0.5
    data_gap_count: int = 0
    critical_gap_count: int = 0
    abnormalities: List[Dict[str, Any]] = field(default_factory=list)


class BacktestPathLogger:
    """
    Handles structured logging of the backtest path for later replay/review.
    Saves daily snapshots including spot price, regime, NLV, cash, and active trades.
    """
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.log_data = {
            "backtest_config": {},
            "path": [],
            "final_results": {}
        }

    def set_config(self, config: dict):
        self.log_data["backtest_config"] = config

    def log_day(self, date_str: str, spot: float, regime: int, regime_name: str, nlv: float, cash: float, margin: float, active_trades: list, events: list, abnormalities: list = None):
        day_entry = {
            "date": date_str,
            "spot_price": float(spot),
            "regime": int(regime),
            "regime_name": regime_name,
            "nlv": float(nlv),
            "cash": float(cash),
            "margin": float(margin),
            "active_trades": [asdict(t) for t in active_trades],
            "events": events,
            "abnormalities": abnormalities or []
        }
        self.log_data["path"].append(day_entry)

    def _sanitize(self, obj):
        """Recursively sanitize objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def save(self, final_results: dict = None):
        if final_results:
            self.log_data["final_results"] = final_results
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            sanitized_data = self._sanitize(self.log_data)
            with open(self.output_path, 'w') as f:
                json.dump(sanitized_data, f, indent=2)
            print(f"\n  ✓ Backtest path log saved to: {self.output_path}")
        except Exception as e:
            print(f"\n  ❌ Failed to save backtest log: {e}")


def get_trading_dates(start: str, end: str) -> List[str]:
    """Get NYSE trading dates in range."""
    schedule = NYSE.schedule(start_date=start, end_date=end)
    return [d.strftime("%Y-%m-%d") for d in schedule.index]


def find_target_expiration_friday(trade_date: str, target_dte: int = 42) -> List[str]:
    """
    Find candidate Friday expirations near target_dte.
    Returns a list of Fridays to try, closest first, then alternating out.
    """
    td = datetime.strptime(trade_date, "%Y-%m-%d")
    target = td + timedelta(days=target_dte)

    # Find the closest Friday
    days_to_friday = (4 - target.weekday()) % 7
    if days_to_friday == 0 and target.weekday() != 4:
        days_to_friday = 7
    closest_friday = target + timedelta(days=days_to_friday)
    if target.weekday() == 4:
        closest_friday = target

    # Generate candidates: closest, then +1 week, -1 week, +2, -2, etc.
    candidates = [closest_friday]
    for offset in range(1, 5):
        candidates.append(closest_friday + timedelta(weeks=offset))
        candidates.append(closest_friday - timedelta(weeks=offset))

    # Filter out past dates
    return [c.strftime("%Y-%m-%d") for c in candidates if c > td]


def find_close_date(entry_date: str, expiration: str, close_dte: int = 21) -> Optional[str]:
    """
    Find the trading date closest to close_dte days before expiration.
    """
    exp = datetime.strptime(expiration, "%Y-%m-%d")
    target_close = exp - timedelta(days=close_dte)

    # Get trading dates around the target
    search_start = (target_close - timedelta(days=5)).strftime("%Y-%m-%d")
    search_end = (target_close + timedelta(days=5)).strftime("%Y-%m-%d")
    trading_dates = get_trading_dates(search_start, search_end)

    if not trading_dates:
        return None

    # Find closest trading date to target
    target_str = target_close.strftime("%Y-%m-%d")
    closest = min(trading_dates, key=lambda d: abs(
        (datetime.strptime(d, "%Y-%m-%d") - target_close).days
    ))

    # Don't close before entry
    if closest <= entry_date:
        return None

    return closest


async def run_put_credit_spread_backtest(
    underlying: str = "SPY",
    start_date: str = "2025-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
    target_dte: int = 42,
    close_dte: int = 21,
    target_short_delta: float = -0.15,
    spread_width: float = 20.0,
    risk_free_rate: float = 0.05,
    db_path: str = None,
    plot: bool = False,
    initial_capital: float = 100000.0,
    margin_limit_pct: float = 0.5,
    early_profit_pct: float = 0.7,
    backtest_qty: int = 0, # 0 means size to max allowed margin
    regimes: Optional[dict] = None, # date string -> state int
    underlying_prices: Optional[pd.Series] = None,  # Pre-fetched prices to avoid redundant download
    vix_prices: Optional[pd.Series] = None,        # Pre-fetched VIX prices
    enable_logging: bool = False,
    output_log_path: Optional[str] = None,
    hold_itm_to_expiration: bool = False,
    regime_labels: Optional[dict] = None, # state int -> name string
    dynamic_delta_variant: bool = True,
    panic_delta_multiplier: float = 4.0,
    panic_width_multiplier: float = 2.0,
    panic_qty_multiplier: float = 0.5,
    panic_dte_target: int = 63,
    panic_swap_enabled: bool = True,
    dividend_yield: float = 0.0,
) -> BacktestResult:
    """
    Run a put credit spread backtest with capital management and advanced tracking.
    """
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "backtest_cache", "option_data.db",
        )

    cache = OptionDataCache(db_path)
    result = BacktestResult()
    result.initial_capital = initial_capital
    result.margin_limit_pct = margin_limit_pct
    result.regime_labels = regime_labels if regime_labels else {}

    print("=" * 70)
    print(f"  PUT CREDIT SPREAD BACKTEST (ADVANCED)")
    print(f"  {underlying} | {start_date} → {end_date}")
    print(f"  Target DTE: {target_dte} | Close DTE: {close_dte}")
    print(f"  Short Delta: {target_short_delta} | Width: ${spread_width}")
    print(f"  Initial Capital: ${initial_capital:,.0f} | Margin Limit: {margin_limit_pct:.0%}")
    print("=" * 70)

    trading_dates = get_trading_dates(start_date, end_date)
    print(f"\n  Trading days: {len(trading_dates)}")
    print(f"  Debug: trading_dates sample: {trading_dates[:5]} ... {trading_dates[-5:]}")
    feb_dates = [d for d in trading_dates if "2026-02" in d]
    print(f"  Debug: Feb trading dates count: {len(feb_dates)}")

    # Use pre-fetched prices if provided, otherwise fetch
    if underlying_prices is not None and not underlying_prices.empty:
        print("  Using pre-fetched underlying price data.", flush=True)
    else:
        print("  Fetching underlying price history...", flush=True)
        underlying_prices = _fetch_underlying_prices(underlying, start_date, end_date)
    if underlying_prices.empty:
        print("  ERROR: No underlying price data.")
        cache.close()
        return result
    result.underlying_prices = underlying_prices

    # Use pre-fetched VIX prices if provided, otherwise fetch
    if vix_prices is not None and not vix_prices.empty:
        print("  Using pre-fetched VIX price data.", flush=True)
    else:
        print("  Fetching VIX price history...", flush=True)
        vix_prices = _fetch_underlying_prices("^VIX", start_date, end_date)
    result.vix_prices = vix_prices

    # Initialize Path Logger
    path_logger = None
    if enable_logging:
        if output_log_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_log_path = f"backtest_logs/backtest_path_{underlying}_{ts}.json"
        path_logger = BacktestPathLogger(output_log_path)
        path_logger.set_config({
            "underlying": underlying,
            "start_date": start_date,
            "end_date": end_date,
            "target_dte": target_dte,
            "close_dte": close_dte,
            "target_short_delta": target_short_delta,
            "spread_width": spread_width,
            "initial_capital": initial_capital,
            "margin_limit_pct": margin_limit_pct,
            "early_profit_pct": early_profit_pct,
            "hold_itm_to_expiration": hold_itm_to_expiration,
            "dividend_yield": dividend_yield
        })

    async with MassiveAPIClient(cache=cache) as client:
        # ── Phase 1: Identify all needed expirations ────────────
        print("\n─── Phase 1: Identifying target expirations ───")
        trade_plan = []  # (trade_date, candidates)
        first_targeted_date = {} # exp -> earliest td

        for td in trading_dates:
            current_regime = regimes.get(td, -1) if regimes else -1
            current_regime_name = result.regime_labels.get(current_regime, "")
            is_panic = "Panic / Crisis" in current_regime_name
            
            eff_target_dte = target_dte
            if dynamic_delta_variant and is_panic:
                eff_target_dte = 63
                
            candidates = find_target_expiration_friday(td, eff_target_dte)
            trade_plan.append((td, candidates))

            for exp in candidates[:3]:
                if exp not in first_targeted_date or td < first_targeted_date[exp]:
                    first_targeted_date[exp] = td

        # ── Phase 2: Fetch contracts lists (batch async) ────────
        print(f"\n─── Phase 2: Fetching contracts lists (batch async) ── {len(first_targeted_date)} expirations")
        
        sorted_exps = sorted(first_targeted_date.keys())
        tasks = []
        for exp in sorted_exps:
            # Strictly anchor polling to the first day we consider trading this expiration
            as_of = first_targeted_date[exp]
            tasks.append(client.fetch_contracts_list(underlying, exp, "put", as_of))
            
        results = await asyncio.gather(*tasks)
        
        contracts_by_exp = {}
        for exp, contracts in zip(sorted_exps, results):
            if contracts:
                contracts_by_exp[exp] = contracts
            else:
                print(f"  WARNING: No contracts found for expiration {exp}")

        # ── Phase 3: Batch fetch OHLCV for all contracts (global batch) ────────
        print(f"\n─── Phase 3: Fetching OHLCV bars (global batch async) ── {len(contracts_by_exp)} expirations")
        
        tasks = []
        for exp, contracts in contracts_by_exp.items():
            relevant_dates = [td for td, cands in trade_plan if exp in cands[:3]]
            if not relevant_dates: continue
            fetch_from = min(relevant_dates)
            fetch_to = exp

            # Strike Filtering Optimization
            # Only fetch strikes within a reasonable buffer of the spot price range during this window.
            # This significantly reduces API calls for deep OTM/ITM strikes we will never trade.
            spot_window = underlying_prices[(underlying_prices.index >= fetch_from) & (underlying_prices.index <= fetch_to)]
            if not spot_window.empty:
                min_s = spot_window.min()
                max_s = spot_window.max()
                # Buffer: 20% below min spot to 40% above max spot
                # (Upper bound expanded to 40% to catch deep ITM rolls)
                filtered = [c for c in contracts if (min_s * 0.8) <= c["strike"] <= (max_s * 1.4)]
                print(f"  [FILTER] {exp}: {len(contracts)} -> {len(filtered)} strikes")
                contracts = filtered

            tasks.append(client.fetch_chain_ohlcv_batch(
                contracts, fetch_from, fetch_to,
                underlying=underlying, contract_type="put", expiration=exp,
            ))
            
        results = await asyncio.gather(*tasks)
        total_api_calls_p3 = sum(results)
        
        print(f"  Phase 3 complete: {total_api_calls_p3} API calls | {client.cache_hits} cache hits")

        print("\n─── Phase 4: Executing strategy sequentially ───")
        print(f"  Debug: Phase 4 starting. contracts_by_exp keys: {sorted(contracts_by_exp.keys())}")
        current_cash = initial_capital
        active_trades: List[SpreadTrade] = []
        previously_panic = False
        
        for td, candidates in trade_plan:
            if td not in underlying_prices.index:
                continue
            if td == trading_dates[0] or td == trading_dates[-1] or "2026-02" in td:
                print(f"  Debug: Processing {td}")
            spot = underlying_prices.loc[td]
            daily_events = []
            
            # 0. Track Regime
            current_regime = regimes.get(td, -1) if regimes else -1
            result.regime_history.append((td, current_regime))
            
            current_regime_name = result.regime_labels.get(current_regime, "")
            is_panic = "Panic / Crisis" in current_regime_name
            
            # Calculate daily margin budget based on MORNING cash.
            # This ensures that rolls/replacements are not penalized by the realized loss
            # of the trade they are replacing.
            daily_allowed_margin = current_cash * margin_limit_pct

            # Update NLV placeholder (will be finalized at end of daily loop)
            current_nlv = current_cash 
            closed_today = []
            replacement_requests = []  # Unified list for rolls and panic swap replacements
            still_active = []

            # 1. Update active trades (Close if needed)
            daily_total_unrealized = 0.0
            
            # Pre-fetch all needed quotes for active trades in parallel
            unique_tickers = set()
            for trade in active_trades:
                unique_tickers.add(trade.short_ticker)
                unique_tickers.add(trade.long_ticker)
            
            # Pre-fetch all needed quotes AND bars for active trades in parallel
            quotes_map = {}
            bars_map = {}
            if unique_tickers:
                print(f"  {td}: [FETCH] Pre-fetching data for {len(unique_tickers)} tickers...", flush=True)
                # 1. NBBO Quote tasks
                quote_tasks = {ticker: client.fetch_eod_quote(ticker, td) for ticker in unique_tickers}
                
                # 2. Daily bar tasks (for those not already in cache)
                bar_tasks = {}
                for ticker in unique_tickers:
                    cached = cache.get_ohlcv(ticker, td)
                    if cached:
                        bars_map[ticker] = cached
                    else:
                        bar_tasks[ticker] = client.fetch_contract_daily_bars(ticker, td, td)
                
                # Combine all async tasks
                all_tasks = list(quote_tasks.values()) + list(bar_tasks.values())
                all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                
                # Split results
                n_q = len(quote_tasks)
                quote_results = all_results[:n_q]
                bar_results = all_results[n_q:]
                
                quotes_map = dict(zip(quote_tasks.keys(), quote_results))
                
                # Map new bars to our bars_map
                for ticker, res in zip(bar_tasks.keys(), bar_results):
                    if res and isinstance(res, list) and len(res) > 0:
                        bars_map[ticker] = res[0]
                    else:
                        bars_map[ticker] = None
                
                print(f"  {td}: [FETCH] Data pre-fetched.", flush=True)

            curr_dt = datetime.strptime(td, "%Y-%m-%d")
            for trade in active_trades:
                exp_dt = trade.expiration_dt
                current_dte = (exp_dt - curr_dt).days
                
                # Use stored tickers
                short_ticker = trade.short_ticker
                long_ticker = trade.long_ticker
                
                short_cached = bars_map.get(short_ticker)
                long_cached = bars_map.get(long_ticker)
                
                exit_triggered = False
                exit_reason = ""
                
                gain_pct = 0.0
                short_quote = quotes_map.get(short_ticker)
                long_quote = quotes_map.get(long_ticker)
                
                # Handle potential exceptions from gather
                if isinstance(short_quote, Exception): short_quote = None
                if isinstance(long_quote, Exception): long_quote = None

                if short_quote and long_quote:
                    current_short_mid = short_quote["mid"]
                    current_long_mid = long_quote["mid"]
                else:
                    # ABNORMALITY: Missing Bid/Ask Quote
                    # Try synchronized 1-minute aggregates as the primary fallback for pairs
                    sync_quotes = await client.fetch_synchronized_ohlcv(short_ticker, long_ticker, td)
                    if sync_quotes:
                        current_short_mid = sync_quotes[0]["mid"]
                        current_long_mid = sync_quotes[1]["mid"]
                        print(f"  [SYNC FALLBACK] Found synchronized 1m bars for {td} (timestamp: {sync_quotes[0]['timestamp']})")
                    else:
                        # NEW: Theoretical Extrapolation Fallback
                        # If we can't find synchronized bars, use the Short Leg's latest trade 
                        # and extrapolate the Long Leg's price using its IV.
                        short_trade = await client.fetch_latest_trade(short_ticker, td)
                        if short_trade:
                            short_p = short_trade["price"]
                            theo_long = await client.fetch_theoretical_price(
                                long_ticker, short_ticker, short_p, spot, td, dte_years, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield
                            )
                            if theo_long is not None:
                                current_short_mid = short_p
                                current_long_mid = theo_long
                                print(f"  [THEO FALLBACK] Extrapolated {long_ticker} from {short_ticker} trade (${short_p}) on {td}")
                            else:
                                # Final fallbacks
                                current_short_mid = short_p
                                current_long_mid = long_cached.get("close", 0) if long_cached else 0
                        else:
                            current_short_mid = short_cached.get("close", 0) if short_cached else 0
                            current_long_mid = long_cached.get("close", 0) if long_cached else 0

                        if not current_short_mid or not current_long_mid:
                            # ABNORMALITY: Missing Bid/Ask Quote & No Sync 1m Bars
                            result.data_gap_count += 1
                            reason = "Missing NBBO Quote & No Sync 1m Bars & No Theo Fallback"
                            
                            abnormality = {
                                "date": td,
                                "type": "MISSING_NBBO_QUOTE",
                                "reason": reason,
                                "short_ticker": short_ticker,
                                "long_ticker": long_ticker,
                                "fallback": "OHLCV Close (Unsynchronized)"
                            }
                            result.abnormalities.append(abnormality)
                            daily_events.append({"type": "abnormality", **abnormality})
                            
                            print(f"{CLR_RED}  [DATA GAP] {reason} on {td}. Falling back to unsynchronized OHLCV Close.{CLR_RST}")
                            # Prices already set to cached above
                        
                        if not current_short_mid or not current_long_mid:
                            # CRITICAL: Total Data Gap
                            result.critical_gap_count += 1
                            reason_crit = "Total Data Gap (Quote + OHLCV + Theo)"
                            
                            abnormality_crit = {
                                "date": td,
                                "type": "CRITICAL_DATA_GAP",
                                "reason": reason_crit,
                                "short_ticker": short_ticker,
                                "long_ticker": long_ticker,
                                "fallback": "Entry Mid"
                            }
                            result.abnormalities.append(abnormality_crit)
                            daily_events.append({"type": "abnormality", **abnormality_crit})
                            
                            print(f"{CLR_RED}  [CRITICAL GAP] {reason_crit} on {td}. Falling back to Entry Mid.{CLR_RST}")
                            current_short_mid = trade.short_entry_mid
                            current_long_mid = trade.long_entry_mid
                
                current_debit = current_short_mid - current_long_mid
                gain_pct = (trade.net_credit - current_debit) / trade.net_credit if trade.net_credit > 0 else 0
                    
                if not trade.hold_to_expiration:
                    # Early profit exit (only if price data is available)
                    if short_cached and long_cached and gain_pct >= trade.profit_target:
                        exit_triggered = True
                        exit_reason = "early_profit"
                
                    # Scheduled exit (based on DTE)
                    is_panic_trade = "Panic / Crisis" in trade.entry_regime_name
                    if not is_panic_trade and current_dte <= close_dte and not exit_triggered:
                        # NEW: Only exit at half-DTE if the position is in profit.
                        # If in loss, skip and follow profit target (or hold to expiration).
                        if gain_pct > 0:
                            if hold_itm_to_expiration and spot <= trade.short_strike:
                                trade.hold_to_expiration = True
                                print(f"  {td}: Position ITM ({spot:.2f} <= {trade.short_strike}). Holding {trade.num_contracts}x to expiration.")
                            else:
                                exit_triggered = True
                                exit_reason = "scheduled"
                
                # Rolling Trigger: ITM and < 7 DTE
                if current_dte <= 7 and spot <= trade.short_strike and not exit_triggered:
                    # Calculate current delta for the roll
                    curr_iv = implied_volatility(current_short_mid, spot, trade.short_strike, current_dte/365.0, risk_free_rate, dividend_yield, "put") if current_dte > 0 else None
                    curr_delta = bs_put_delta(spot, trade.short_strike, current_dte/365.0, risk_free_rate, curr_iv, dividend_yield) if curr_iv else -1.0
                    
                    # Determine target delta/strike based on rules
                    t_delta = curr_delta
                    if dynamic_delta_variant and is_panic:
                        t_delta = curr_delta * panic_delta_multiplier
                    
                    t_strike = None
                    # User Rule: if close to -1, same strike. Else if in (-1, -0.5], target -1.0 delta.
                    if t_delta <= -0.95:
                        t_strike = trade.short_strike
                    elif -1.0 < t_delta <= -0.5:
                        t_delta = -1.0
                    
                    print(f"  {td}: [ROLL REQUEST (STANDARD)] Orig Strike={trade.short_strike} -> Target Delta={t_delta:.2f}, Target Strike={t_strike}")
                    
                    replacement_requests.append({
                        "target_delta": t_delta,
                        "target_strike": t_strike,
                        "num_contracts": trade.num_contracts,
                        "reason": "roll"
                    })
                    
                    exit_triggered = True
                    exit_reason = "roll"

                # Expiration exit (always check)
                if current_dte == 0:
                    # On expiration day, the value is simply the intrinsic value.
                    # This prevents false "max loss" results due to missing EOD quotes.
                    intrinsic_short = max(0, trade.short_strike - spot)
                    intrinsic_long = max(0, trade.long_strike - spot)
                    
                    current_short_mid = intrinsic_short
                    current_long_mid = intrinsic_long
                    current_debit = intrinsic_short - intrinsic_long
                    
                    if not exit_triggered:
                        exit_triggered = True
                        exit_reason = "expired"
                    
                if exit_triggered:
                    trade.exit_date = td
                    trade.short_exit_mid = current_short_mid
                    trade.long_exit_mid = current_long_mid
                    trade.net_debit_close = current_debit
                    trade.pnl_per_share = trade.net_credit - trade.net_debit_close
                    trade.pnl_per_contract = trade.pnl_per_share * 100
                    trade.exit_dte = current_dte
                    trade.status = "closed"
                    trade.exit_reason = exit_reason
                    
                    # Cash flow on EXIT
                    current_cash -= trade.net_debit_close * 100 * trade.num_contracts
                    closed_today.append(trade)
                    print(f"  {td}: CLOSE {trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p | PnL=${trade.pnl_per_contract*trade.num_contracts:,.0f}")
                    
                    if path_logger:
                        daily_events.append({
                            "type": "close",
                            "reason": exit_reason,
                            "trade": asdict(trade)
                        })
                else:
                    still_active.append(trade)
                    daily_total_unrealized += current_debit * 100 * trade.num_contracts
                    
                # Track for scatter plot (capture state regardless of exit today)
                if short_cached and long_cached:
                    itm = spot < trade.short_strike
                    result.daily_scatter_data.append({
                        'date': td,
                        'val': current_debit * 100,
                        'itm': itm,
                        'expiration': trade.expiration,
                        'is_exit': exit_triggered
                    })
            
            active_trades = still_active
            # Each trade is already in result.trades from its entry day.
            # We don't extend it again here to avoid duplicates.
            
            # 2. Update Portfolio Stats
            # NOTE: Margin from positions closed or expired today is released here, 
            # making it available for new entries in the next step (Phase 3).
            current_margin_usage = sum(t.margin_required for t in active_trades)
            current_nlv = current_cash - daily_total_unrealized
            
            # (History recording moved to the end of the day to capture entries)
            
            # 3. Entry Phase: Process replacements and scheduled entries
            # We use the daily_allowed_margin calculated at the start of the day.
            allowed_margin = daily_allowed_margin
            
            # --- DYNAMIC DELTA VARIANT: PARAMETER ADJUSTMENT ---
            eff_short_delta = target_short_delta
            eff_spread_width = spread_width
            eff_qty_multiplier = 1.0
            eff_profit_target = early_profit_pct
            
            if dynamic_delta_variant and is_panic:
                eff_short_delta = target_short_delta * panic_delta_multiplier
                # Ensure delta is within sensible bounds (e.g., -0.95 to -0.01)
                eff_short_delta = max(-0.95, min(-0.01, eff_short_delta))
                eff_spread_width = spread_width * panic_width_multiplier
                eff_qty_multiplier = panic_qty_multiplier
                print(f"  {td}: [PANIC MODE] Targets adjusted: Delta={eff_short_delta:.2f}, Width=${eff_spread_width:.1f}, Qty x{eff_qty_multiplier}", flush=True)

            # Loop to allow multiple entries if we just performed a panic swap or have rolls
            # prioritize replacement_requests, then the scheduled entry
            entries_to_attempt = len(replacement_requests) + 1
            daily_delta_cache = {} # (expiration) -> delta_chain
            daily_chain_cache = {} # (expiration) -> chain_data
            
            opened_today = False
            last_dte = 0
            last_spread = 0.0
            last_short_mid = 0.0
            last_long_mid = 0.0
            
            for i in range(entries_to_attempt):
                current_margin_usage = sum(t.margin_required for t in active_trades)
                can_enter = current_margin_usage < allowed_margin
                
                if not can_enter:
                    break
                
                if entries_to_attempt > 1:
                    print(f"  {td}: [ENTRY] Attempt {i+1}/{entries_to_attempt} (Margin: ${current_margin_usage:,.0f}/${allowed_margin:,.0f})...", flush=True)

                eff_target_dte = target_dte if not (dynamic_delta_variant and is_panic) else panic_dte_target
                selected_exp = None
                chain_data = None
                
                possible_exps = find_target_expiration_friday(td, eff_target_dte)
                for exp in possible_exps:
                    if exp in contracts_by_exp:
                        if exp in daily_chain_cache:
                            chain_data = daily_chain_cache[exp]
                        else:
                            chain_data = cache.get_chain_for_date(underlying, exp, "put", td)
                            daily_chain_cache[exp] = chain_data
                            
                        if chain_data and len(chain_data) >= 10:
                            selected_exp = exp
                            break
                        elif chain_data:
                            print(f"  {td}: Skipping {exp} (only {len(chain_data)} strikes found)")
                
                if selected_exp:
                    exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
                    trade_dt = datetime.strptime(td, "%Y-%m-%d")
                    dte_days = (exp_dt - trade_dt).days
                    dte_years = dte_days / 365.0

                    strikes = [r["strike"] for r in chain_data]
                    close_prices = [r["close"] for r in chain_data]
                    valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]
                    
                    if len(valid) >= 5:
                        if selected_exp not in daily_delta_cache:
                            v_strikes, v_prices = zip(*valid)
                            daily_delta_cache[selected_exp] = compute_chain_deltas(spot, list(v_strikes), list(v_prices), dte_years, risk_free_rate, dividend_yield, "put")
                        
                        delta_chain = daily_delta_cache[selected_exp]
                        
                        if delta_chain:
                            # Handle target selection (replacement vs scheduled)
                            req = None
                            if i < len(replacement_requests):
                                req = replacement_requests[i]
                            
                            if req and req.get("target_strike"):
                                t_strike = req["target_strike"]
                                short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][0] - t_strike))
                            else:
                                t_delta = req["target_delta"] if req else eff_short_delta
                                short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][1] - t_delta))
                            
                            short_strike = delta_chain[short_idx][0]
                            short_delta = delta_chain[short_idx][1]
                            long_strike = short_strike - eff_spread_width
                            
                            # Find long strike mid
                            long_mid = next((r["close"] for r in chain_data if abs(r["strike"] - long_strike) < 0.01 and r["close"] is not None), 0.0)
                            if not long_mid:
                                nearest_long = min(strikes, key=lambda s: abs(s - long_strike))
                                long_mid = next((r["close"] for r in chain_data if abs(r["strike"] - nearest_long) < 0.01 and r["close"] is not None), 0.0)
                                long_strike = nearest_long
                            
                            short_mid = delta_chain[short_idx][2] or 0.0
                            net_credit = short_mid - long_mid
                            
                            # NEW: Synchronize Entry Pricing
                            # Re-fetch both legs using 1-minute bars to ensure they are temporally aligned.
                            # This prevents picking a spread where one leg traded at 10am and another at 4pm.
                            short_ticker = next(r["option_ticker"] for r in chain_data if abs(r["strike"] - short_strike) < 0.01)
                            long_ticker = next(r["option_ticker"] for r in chain_data if abs(r["strike"] - long_strike) < 0.01)
                            
                            sync_quotes = await client.fetch_synchronized_ohlcv(short_ticker, long_ticker, td)
                            if sync_quotes:
                                short_mid = sync_quotes[0]["mid"] or 0.0
                                long_mid = sync_quotes[1]["mid"] or 0.0
                                net_credit = short_mid - long_mid
                                print(f"  [ENTRY SYNC] Synchronized pricing found for {td}: Short=${short_mid}, Long=${long_mid}, Net=${net_credit:.2f}")
                            else:
                                # THEO FALLBACK for Entry
                                short_trade = await client.fetch_latest_trade(short_ticker, td)
                                if short_trade:
                                    short_p = short_trade["price"]
                                    theo_long = await client.fetch_theoretical_price(
                                        long_ticker, short_ticker, short_p, spot, td, dte_years, risk_free_rate=risk_free_rate
                                    )
                                    if theo_long is not None:
                                        short_mid = short_p
                                        long_mid = theo_long
                                        net_credit = short_mid - long_mid
                                        print(f"  [ENTRY THEO] Extrapolated {long_ticker} from {short_ticker} trade (${short_p}) on {td}: Net=${net_credit:.2f}")
                                    else:
                                        print(f"  [ENTRY WARNING] No synchronized 1m bars or theo fallback for {short_ticker}/{long_ticker} on {td}. Using unsynchronized EOD Close.")
                                else:
                                    print(f"  [ENTRY WARNING] No synchronized 1m bars or short trade for {short_ticker}/{long_ticker} on {td}. Using unsynchronized EOD Close.")

                            if net_credit > 0:
                                margin_per_lot = (short_strike - long_strike) * 100
                                if req:
                                    num_contracts = req["num_contracts"]
                                elif backtest_qty > 0:
                                    num_contracts = int(backtest_qty * eff_qty_multiplier) or 1
                                else:
                                    num_contracts = int(((allowed_margin - current_margin_usage) // margin_per_lot) * eff_qty_multiplier)
                                
                                if num_contracts > 0:
                                    trade = SpreadTrade(
                                        short_ticker=next(r["option_ticker"] for r in chain_data if abs(r["strike"] - short_strike) < 0.01),
                                        long_ticker=next(r["option_ticker"] for r in chain_data if abs(r["strike"] - long_strike) < 0.01),
                                        short_strike=short_strike,
                                        long_strike=long_strike,
                                        expiration=selected_exp,
                                        entry_date=td,
                                        entry_price=spot,
                                        net_credit=net_credit,
                                        num_contracts=num_contracts,
                                        margin_required=margin_per_lot * num_contracts,
                                        status="open",
                                        entry_regime=current_regime,
                                        entry_regime_name=current_regime_name,
                                        entry_dte=dte_days,
                                        short_entry_mid=short_mid,
                                        long_entry_mid=long_mid,
                                        profit_target=eff_profit_target,
                                        entry_delta=short_delta
                                    )
                                    
                                    active_trades.append(trade)
                                    result.trades.append(trade)
                                    current_cash += net_credit * 100 * num_contracts
                                    
                                    opened_today = True
                                    last_dte = dte_days
                                    short_row = next((r for r in chain_data if abs(r["strike"] - short_strike) < 0.01), {})
                                    long_row = next((r for r in chain_data if abs(r["strike"] - long_strike) < 0.01), {})
                                    s_spread = (short_row.get("ask", 0) - short_row.get("bid", 0)) if short_row.get("bid") and short_row.get("ask") else 0
                                    l_spread = (long_row.get("ask", 0) - long_row.get("bid", 0)) if long_row.get("bid") and long_row.get("ask") else 0
                                    last_spread = s_spread + l_spread
                                    last_short_mid = short_mid
                                    last_long_mid = long_mid

                                    print(f"  {td}: OPEN {num_contracts}x {short_strike}/{long_strike}p | Credit=${net_credit:.2f} | Margin=${trade.margin_required:,.0f}")

                                    if path_logger:
                                        daily_events.append({
                                            "type": "entry",
                                            "expiration": selected_exp,
                                            "short_strike": short_strike,
                                            "long_strike": long_strike,
                                            "net_credit": float(net_credit),
                                            "num_contracts": num_contracts
                                        })
                                    
            
            # Finalize daily summary stats (once per day)
            if opened_today:
                result.daily_opened_dte.append((td, last_dte))
                result.daily_opened_spread.append((td, last_spread))
                result.daily_leg_premiums.append((td, last_short_mid, last_long_mid))
            else:
                result.daily_leg_premiums.append((td, 0.0, 0.0))
            
            # --- PHASE 4: RECORD HISTORY ---
            # Update margin and NLV one last time for the day to capture new entries
            current_margin_usage = sum(t.margin_required for t in active_trades)
            # Recalculate unrealized if needed? Actually current_cash was updated in entry.
            # But unrealized of new trades is 0 at entry.
            result.cash_history.append((td, current_cash))
            result.nvl_history.append((td, current_cash - daily_total_unrealized))
            result.margin_history.append((td, current_margin_usage))


            # End of daily loop: Log Day
            if path_logger:
                path_logger.log_day(
                    date_str=td,
                    spot=spot,
                    regime=current_regime,
                    regime_name=regime_labels.get(current_regime, f"Regime {current_regime}") if regime_labels else f"Regime {current_regime}",
                    nlv=current_nlv,
                    cash=current_cash,
                    margin=current_margin_usage,
                    active_trades=active_trades,
                    events=daily_events,
                    abnormalities=[a for a in result.abnormalities if a["date"] == td]
                )
            
            previously_panic = is_panic

        print(f"\n  Backtest loop complete. Processed {len(trading_dates)} days.")
        print(f"  Total trades recorded: {len(result.trades)}")
        
        if path_logger:
            summary = {
                "total_pnl": float(result.total_pnl),
                "win_count": int(result.win_count),
                "loss_count": int(result.loss_count),
                "total_trades": int(result.total_trades),
                "max_drawdown": float(result.max_drawdown),
                "abnormalities": result.abnormalities
            }
            path_logger.save(final_results=summary)
        
        # ── Results ─────────────────────────────────────────────
        result.api_calls = client.api_calls
        result.cache_hits = client.cache_hits

    cache.close()

    # Compute summary stats
    closed = [t for t in result.trades if t.status == "closed"]
    result.total_trades = len(closed)
    result.total_pnl = sum(t.pnl_per_contract * t.num_contracts for t in closed)
    result.win_count = sum(1 for t in closed if t.pnl_per_contract > 0)
    result.loss_count = sum(1 for t in closed if t.pnl_per_contract <= 0)
    result.avg_credit = (
        np.mean([t.net_credit for t in closed]) if closed else 0
    )
    result.avg_pnl = (
        np.mean([t.pnl_per_contract * t.num_contracts for t in closed]) if closed else 0
    )

    # Max drawdown
    if closed:
        cumulative = np.cumsum([t.pnl_per_contract for t in closed])
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        result.max_drawdown = float(np.min(drawdown))

    _print_results(result)
    
    if plot:
        plot_results_interactive(result, underlying)
        
    return result


def plot_results(result: BacktestResult, ticker: str):
    """
    Optimized 3x2 multi-panel plotting for backtest results with VIX overlay.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    if not result.cash_history:
        print("  No history data to plot.")
        return

    # 1. Prepare Shared X-Axis (Dates)
    underlying_dates = [datetime.strptime(d, "%Y-%m-%d") for d in result.underlying_prices.index]
    underlying_vals = result.underlying_prices.values
    hist_dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in result.cash_history]

    # Create figure with 3 rows and 2 columns, sharing X axis
    fig, axes = plt.subplots(3, 2, figsize=(22, 16), sharex=True, constrained_layout=True)
    
    # Flatten axes for easier indexing
    axs = axes.flatten()
    ax1, ax2, ax3, ax4, ax5, ax6 = axs[0], axs[1], axs[2], axs[3], axs[4], axs[5]

    # --- Add HMM Backgrounds to ALL subplots ---
    if result.regime_history:
        reg_dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in result.regime_history]
        reg_vals = [v for _, v in result.regime_history]
        colors_list = plt.cm.Set3.colors
        regime_df = pd.DataFrame({'date': reg_dates, 'regime': reg_vals}).set_index('date')
        state_changes = regime_df['regime'].ne(regime_df['regime'].shift()).cumsum()
        groups = regime_df.groupby(state_changes)
        
        for ax in axs:
            added_to_legend = set()
            for _, group in groups:
                state = group['regime'].iloc[0]
                if state == -1: continue 
                color = colors_list[state % len(colors_list)]
                label = result.regime_labels.get(state, f"Regime {state}")
                
                # Only add label to legend for ax3 to avoid duplicates everywhere
                if ax == ax3:
                    if label not in added_to_legend:
                        ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15, label=label)
                        added_to_legend.add(label)
                    else:
                        ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15)
                else:
                    ax.axvspan(group.index[0], group.index[-1], color=color, alpha=0.15)

    # --- Subplot 1: Underlying Price & Trades ---
    ax1.plot(underlying_dates, underlying_vals, color='navy', alpha=0.3, label=f"{ticker} Price", linewidth=1.5)
    ax1.set_title(f"Underlying Price & Trades with VIX Overlay", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)", fontsize=12, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(True, alpha=0.2)
    
    # ax1.plot(underlying_dates, underlying_vals, color='navy', alpha=0.3, label=f"{ticker} Price", linewidth=1.5)
    # Move VIX later to ax3

    closed_trades = [t for t in result.trades if t.status == "closed"]
    
    # Legend tracking
    entry_plotted = False
    exit_profit_plotted = False
    exit_loss_otm_plotted = False
    exit_loss_itm_plotted = False

    for t in closed_trades:
        entry_dt = datetime.strptime(t.entry_date, "%Y-%m-%d")
        exit_dt = datetime.strptime(t.exit_date, "%Y-%m-%d")
        
        # Entry: Blue dot
        ax1.scatter(entry_dt, t.short_strike, color='blue', marker='o', s=3, alpha=0.8, 
                    label="Entry" if not entry_plotted else "")
        entry_plotted = True
        
        # Exit Logic
        pnl = t.pnl_per_contract
        spot_at_exit = result.underlying_prices.get(t.exit_date, t.short_strike)
        
        if pnl > 0:
            exit_color = 'green'
            exit_label = "Exit (Profit)" if not exit_profit_plotted else ""
            exit_profit_plotted = True
        elif spot_at_exit > t.short_strike:
            exit_color = 'orange'
            exit_label = "Exit (Loss OTM)" if not exit_loss_otm_plotted else ""
            exit_loss_otm_plotted = True
        else:
            exit_color = 'red'
            exit_label = "Exit (ITM)" if not exit_loss_itm_plotted else ""
            exit_loss_itm_plotted = True
            
        ax1.scatter(exit_dt, t.short_strike, color=exit_color, marker='o', s=3, alpha=0.8, label=exit_label)
        ax1.plot([entry_dt, exit_dt], [t.short_strike, t.short_strike], color='blue', alpha=0.05, linestyle='-')
    
    ax1.legend(loc="upper left", fontsize=9, ncol=2)

    # --- Subplot 2: Capital & Portfolio Value (NVL) ---
    cash_vals = [v for _, v in result.cash_history]
    nvl_vals = [v for _, v in result.nvl_history]
    margin_vals = [v for _, v in result.margin_history]
    
    ax2.plot(hist_dates, cash_vals, label="Cash", color='tab:green', linewidth=1.2, linestyle='--')
    ax2.plot(hist_dates, nvl_vals, label="NLV", color='tab:blue', linewidth=2)
    ax2.set_title("Capital & Portfolio Value (NVL)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Account Value ($)", fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(hist_dates, margin_vals, 0, step="post", alpha=0.1, color='orange', label="Margin")
    ax2_twin.set_ylabel("Margin ($)", color='orange', fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Red dots for hitting margin cap
    margin_cap_dates = []
    margin_cap_vals = []
    for i, (dt, nvl) in enumerate(result.nvl_history):
        allowed = nvl * result.margin_limit_pct
        if margin_vals[i] >= allowed * 0.99: # 99% threshold for "hitting" cap
            margin_cap_dates.append(hist_dates[i])
            margin_cap_vals.append(margin_vals[i])
    
    if margin_cap_dates:
        ax2_twin.scatter(margin_cap_dates, margin_cap_vals, color='red', s=15, zorder=5, label="Margin Cap Hit")
        
    ax2_twin.set_ylim(0, max(margin_vals) * 2 if margin_vals else 1000)
    
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    # --- Subplot 3: Market Regimes ---
    if result.regime_history:
        reg_vals = [v for _, v in result.regime_history]
        
        # ax3.step(reg_dates, reg_vals, where='post', color='black', linewidth=1, alpha=0.8)
        ax3.set_title("Market Regime History & VIX Overlay", fontsize=14, fontweight='bold')
        ax3.set_ylabel("HMM State", fontsize=12)
        ax3.set_yticks(sorted(list(set(reg_vals))))
        ax3.grid(True, alpha=0.1)
        
        # Add VIX to Regime Plot
        ax3_vix = ax3.twinx()
        if not result.vix_prices.empty:
            vix_dates = [datetime.strptime(d, "%Y-%m-%d") for d in result.vix_prices.index]
            vix_vals = pd.to_numeric(result.vix_prices, errors='coerce').values
            ax3_vix.plot(vix_dates, vix_vals, color='darkred', alpha=0.3, label="VIX", linewidth=1)
            ax3_vix.set_ylabel("VIX Index", color='darkred', fontsize=12)
            ax3_vix.tick_params(axis='y', labelcolor='darkred')
            
        ax3.legend(loc="upper left", fontsize=9, frameon=True, shadow=True)

    # --- Subplot 4: Leg Premiums ---
    if result.daily_leg_premiums:
        prem_dates = [datetime.strptime(d, "%Y-%m-%d") for d, _, _ in result.daily_leg_premiums]
        short_prems = [s for _, s, _ in result.daily_leg_premiums]
        long_prems = [l for _, _, l in result.daily_leg_premiums]
        
        entry_mask = np.array(short_prems) > 0
        if any(entry_mask):
            ax4.scatter(np.array(prem_dates)[entry_mask], np.array(short_prems)[entry_mask], color='red', label="Short", s=12, alpha=0.7)
            ax4.scatter(np.array(prem_dates)[entry_mask], np.array(long_prems)[entry_mask], color='blue', label="Long", s=7, alpha=0.7)
            
            # Draw line for net premium
            net_prems = np.array(short_prems) - np.array(long_prems)
            ax4.plot(np.array(prem_dates)[entry_mask], net_prems[entry_mask], color='purple', alpha=0.5, label="Net Premium", linewidth=1)
            
            ax4.set_title("Entry Leg Premiums", fontsize=14, fontweight='bold')
            ax4.set_ylabel("Premium ($)", fontsize=12)
            ax4.legend(loc="upper left", fontsize=9)
            ax4.grid(True, alpha=0.2)

    # --- Subplot 5: Position Distribution (ITM/OTM) ---
    if result.daily_scatter_data:
        otm_dates, otm_vals = [], []
        itm_open_dates, itm_open_vals = [], []
        itm_exp_dates, itm_exp_vals = [], []

        for d in result.daily_scatter_data:
            dt = datetime.strptime(d['date'], "%Y-%m-%d")
            val = d['val']
            itm = d['itm']
            exp = d.get('expiration', '')
            
            if not itm:
                otm_dates.append(dt)
                otm_vals.append(val)
            else:
                # Color red if it's an exit day (for consistency with top plot) OR actual expiration day
                is_exit = d.get('is_exit', False)
                is_exp = pd.to_datetime(d['date']) == pd.to_datetime(exp) if exp else False
                
                if is_exit or is_exp:
                    itm_exp_dates.append(dt)
                    itm_exp_vals.append(val)
                else:
                    itm_open_dates.append(dt)
                    itm_open_vals.append(val)
        
        if otm_dates:
            ax5.scatter(otm_dates, otm_vals, color='tab:green', s=4, alpha=0.4, label="OTM")
        
        if itm_open_dates:
            ax5.scatter(itm_open_dates, itm_open_vals, color='orange', s=6, alpha=0.7, marker='o', label="ITM (Open)")
            
        if itm_exp_dates:
            ax5.scatter(itm_exp_dates, itm_exp_vals, color='red', s=6, alpha=0.9, marker='o', label="ITM (Exit/Exp)")
            
        ax5.set_title("Position Distribution", fontsize=14, fontweight='bold')
        ax5.set_ylabel("Pair Value ($)", fontsize=12)
        ax5.legend(loc="upper left", fontsize=9)
        ax5.grid(True, alpha=0.2)
        
    # --- Subplot 6: DTE and Bid-Ask Spread ---
    if result.daily_opened_dte:
        dte_dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in result.daily_opened_dte]
        dte_vals = [v for _, v in result.daily_opened_dte]
        
        ax6.plot(dte_dates, dte_vals, color='tab:blue', marker='o', markersize=4, linestyle='-', label="DTE", alpha=0.8)
        ax6.set_title("Entry DTE & Bid-Ask Spread", fontsize=14, fontweight='bold')
        ax6.set_ylabel("DTE (Days)", fontsize=12, color='tab:blue')
        ax6.tick_params(axis='y', labelcolor='tab:blue')
        ax6.grid(True, alpha=0.2)
        
        if result.daily_opened_spread:
            ax6_twin = ax6.twinx()
            spread_vals = [v for _, v in result.daily_opened_spread]
            ax6_twin.plot(dte_dates, spread_vals, color='tab:red', marker='x', markersize=4, linestyle='--', label="Bid-Ask Spread", alpha=0.6)
            ax6_twin.set_ylabel("Combined Bid-Ask Spread ($)", color='tab:red', fontsize=12)
            ax6_twin.tick_params(axis='y', labelcolor='tab:red')
            
            # Combine legends
            h1, l1 = ax6.get_legend_handles_labels()
            h2, l2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
        else:
            ax6.legend(loc="upper left", fontsize=9)

    # Final Formatting
    for ax in axs[:6]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Add text summary
    summary_text = (
        f"Total PnL: \\${result.total_pnl:,.0f} | "
        f"Win Rate: {result.win_count/max(1, result.total_trades)*100:.1f}% | "
        f"Max Drawdown: \\${result.max_drawdown:,.0f}"
    )
    fig.text(0.5, 0.015, summary_text, fontsize=16, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plot_path = f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\n  Plot saved to: {plot_path}")
    # plt.show() # Commented out to avoid blocking


def plot_results_interactive(result: BacktestResult, ticker: str):
    """
    Interactive 3x2 multi-panel plotting for backtest results using Plotly.
    Saves to ev_plots_backtest.html.
    """
    if not result.cash_history:
        print("  No history data to plot.")
        return

    # 1. Prepare Data
    underlying_dates = result.underlying_prices.index.tolist()
    underlying_vals = result.underlying_prices.values.tolist()
    hist_dates = [d for d, _ in result.cash_history]
    cash_vals = [v for _, v in result.cash_history]
    nvl_vals = [v for _, v in result.nvl_history]
    margin_vals = [v for _, v in result.margin_history]

    # Create figure with 3 rows and 2 columns
    fig = make_subplots(
        rows=3, cols=2,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            f"{ticker} Price & Trade Entry/Exit", 
            "Capital & Portfolio Value (NLV)", 
            "Market Regimes (HMM) & VIX",
            "Entry Leg Premiums",
            "Position Distribution (ITM/OTM)",
            "Entry DTE & Bid-Ask Spread"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )

    # --- Subplot 1: Underlying Price & Trades ---
    fig.add_trace(
        go.Scatter(x=underlying_dates, y=underlying_vals, name=f"{ticker} Price",
                   line=dict(color='navy', width=1), opacity=0.3, legend='legend'),
        row=1, col=1
    )

    closed_trades = [t for t in result.trades if t.status == "closed"]
    entry_x, entry_y = [], []
    exit_profit_x, exit_profit_y = [], []
    exit_loss_otm_x, exit_loss_otm_y = [], []
    exit_loss_itm_x, exit_loss_itm_y = [], []

    for t in closed_trades:
        entry_x.append(t.entry_date)
        entry_y.append(t.short_strike)
        
        pnl = t.pnl_per_contract
        spot_at_exit = result.underlying_prices.get(t.exit_date, t.short_strike)
        
        if pnl > 0:
            exit_profit_x.append(t.exit_date)
            exit_profit_y.append(t.short_strike)
        elif spot_at_exit > t.short_strike:
            exit_loss_otm_x.append(t.exit_date)
            exit_loss_otm_y.append(t.short_strike)
        else:
            exit_loss_itm_x.append(t.exit_date)
            exit_loss_itm_y.append(t.short_strike)
        
        # Add connection line
        fig.add_trace(
            go.Scatter(x=[t.entry_date, t.exit_date], y=[t.short_strike, t.short_strike],
                       mode='lines', line=dict(color='blue', width=1), opacity=0.05, 
                       showlegend=False, hoverinfo='skip'),
            row=1, col=1
        )

    fig.add_trace(go.Scatter(x=entry_x, y=entry_y, mode='markers', name='Entry', marker=dict(color='blue', size=4), legend='legend'), row=1, col=1)
    fig.add_trace(go.Scatter(x=exit_profit_x, y=exit_profit_y, mode='markers', name='Exit (Profit)', marker=dict(color='green', size=4), legend='legend'), row=1, col=1)
    fig.add_trace(go.Scatter(x=exit_loss_otm_x, y=exit_loss_otm_y, mode='markers', name='Exit (Loss OTM)', marker=dict(color='orange', size=4), legend='legend'), row=1, col=1)
    fig.add_trace(go.Scatter(x=exit_loss_itm_x, y=exit_loss_itm_y, mode='markers', name='Exit (ITM)', marker=dict(color='red', size=4), legend='legend'), row=1, col=1)

    # --- Subplot 2: Capital & NLV ---
    fig.add_trace(go.Scatter(x=hist_dates, y=cash_vals, name='Cash', line=dict(color='green', dash='dash', width=1), legend='legend2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=hist_dates, y=nvl_vals, name='NLV', line=dict(color='blue', width=2), legend='legend2'), row=1, col=2)
    
    fig.add_trace(
        go.Scatter(x=hist_dates, y=margin_vals, name='Margin', fill='tozeroy', 
                   line=dict(color='orange', width=0.5), opacity=0.08, legend='legend2'),
        row=1, col=2, secondary_y=True
    )

    # Margin Cap Hits
    margin_cap_dates, margin_cap_vals = [], []
    for i, (dt, nvl) in enumerate(result.nvl_history):
        allowed = nvl * result.margin_limit_pct
        if margin_vals[i] >= allowed * 0.99:
            margin_cap_dates.append(dt)
            margin_cap_vals.append(margin_vals[i])
    
    if margin_cap_dates:
        fig.add_trace(
            go.Scatter(x=margin_cap_dates, y=margin_cap_vals, name='Margin Cap Hit', 
                       mode='markers', marker=dict(color='red', size=2.5), legend='legend2'),
            row=1, col=2, secondary_y=True
        )

    # --- Subplot 3: Regimes & VIX ---
    if result.regime_history:
        reg_df = pd.DataFrame(result.regime_history, columns=['date', 'state']).dropna()
        reg_dates = reg_df['date'].tolist()
        reg_vals = reg_df['state'].tolist()
        
        # (Background loop moved to end of function to ensure subplot initialization)
        
        # Add HMM State line and VIX
        fig.add_trace(go.Scatter(x=reg_dates, y=reg_vals, name='HMM State', line=dict(color='black', width=1.5), legend='legend3'), row=2, col=1)

    if not result.vix_prices.empty:
        vix_clean = result.vix_prices.dropna()
        fig.add_trace(
            go.Scatter(x=vix_clean.index.tolist(), y=vix_clean.values.tolist(), 
                       name='VIX', line=dict(color='darkred', width=1), opacity=0.4, legend='legend3'),
            row=2, col=1, secondary_y=True
        )

    # --- Subplot 4: Leg Premiums ---
    if result.daily_leg_premiums:
        prem_dates = [d for d, _, _ in result.daily_leg_premiums]
        short_prems = [s for _, s, _ in result.daily_leg_premiums]
        long_prems = [l for _, _, l in result.daily_leg_premiums]
        
        entry_mask = np.array(short_prems) > 0
        if any(entry_mask):
            p_dates = np.array(prem_dates)[entry_mask]
            s_prems = np.array(short_prems)[entry_mask]
            l_prems = np.array(long_prems)[entry_mask]
            net_prems = s_prems - l_prems
            
            fig.add_trace(go.Scatter(x=p_dates, y=s_prems, mode='markers', name='Short Prem', marker=dict(color='red', size=5), legend='legend4'), row=2, col=2)
            fig.add_trace(go.Scatter(x=p_dates, y=l_prems, mode='markers', name='Long Prem', marker=dict(color='blue', size=4), legend='legend4'), row=2, col=2)
            fig.add_trace(go.Scatter(x=p_dates, y=net_prems, name='Net Premium', line=dict(color='purple', width=1), opacity=0.5, legend='legend4'), row=2, col=2)

    # --- Subplot 5: Position Distribution ---
    if result.daily_scatter_data:
        otm_x, otm_y = [], []
        itm_open_x, itm_open_y = [], []
        itm_exp_x, itm_exp_y = [], []

        for d in result.daily_scatter_data:
            x = d['date']
            y = d['val']
            itm = d['itm']
            exp = d.get('expiration', '')
            is_exit = d.get('is_exit', False)
            is_exp = x == exp if exp else False
            
            if not itm:
                otm_x.append(x)
                otm_y.append(y)
            elif is_exit or is_exp:
                itm_exp_x.append(x)
                itm_exp_y.append(y)
            else:
                itm_open_x.append(x)
                itm_open_y.append(y)
        
        fig.add_trace(go.Scatter(x=otm_x, y=otm_y, mode='markers', name='OTM', marker=dict(color='green', size=3, opacity=0.4), legend='legend5'), row=3, col=1)
        fig.add_trace(go.Scatter(x=itm_open_x, y=itm_open_y, mode='markers', name='ITM (Open)', marker=dict(color='orange', size=5), legend='legend5'), row=3, col=1)
        fig.add_trace(go.Scatter(x=itm_exp_x, y=itm_exp_y, mode='markers', name='ITM (Exit/Exp)', marker=dict(color='red', size=5), legend='legend5'), row=3, col=1)

    # --- Subplot 6: DTE & Spread ---
    if result.daily_opened_dte:
        dte_dates = [d for d, _ in result.daily_opened_dte]
        dte_vals = [v for _, v in result.daily_opened_dte]
        fig.add_trace(go.Scatter(x=dte_dates, y=dte_vals, name='DTE', line=dict(color='blue', width=1), mode='lines+markers', marker=dict(size=4), legend='legend6'), row=3, col=2)
        
        if result.daily_opened_spread:
            spread_vals = [v for _, v in result.daily_opened_spread]
            fig.add_trace(go.Scatter(x=dte_dates, y=spread_vals, name='Bid-Ask Spread', line=dict(color='red', dash='dash', width=1), mode='lines+markers', marker=dict(size=4, symbol='x'), legend='legend6'), row=3, col=2, secondary_y=True)

    # Layout Updates
    fig.update_layout(
        height=1200,
        autosize=True,
        title_text=f"Backtest Analysis: {ticker} Strategy",
        template="plotly_white",
        hovermode="x",
        spikedistance=-1,
        hoverdistance=-1,
        showlegend=True,
        dragmode="zoom",
        # Configure Multiple Legends
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        legend2=dict(x=0.51, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        legend3=dict(x=0.01, y=0.64, bgcolor="rgba(255,255,255,0.7)"),
        legend4=dict(x=0.51, y=0.64, bgcolor="rgba(255,255,255,0.7)"),
        legend5=dict(x=0.01, y=0.31, bgcolor="rgba(255,255,255,0.7)"),
        legend6=dict(x=0.51, y=0.31, bgcolor="rgba(255,255,255,0.7)"),
    )
    
    # Synchronize X-axes and hover spikes across subplots
    # NOTE: We do NOT use shared_xaxes=True because Plotly's internal `matches`
    # mechanism creates circular references in complex 3x2 layouts with secondary_y,
    # causing NaN rendering errors. Instead, all synchronization is done in JavaScript.
    fig.update_xaxes(
        type='date',
        showspikes=True,
        spikemode='across',
        spikedash='dash',
        spikecolor='#666666',
        spikethickness=1,
        spikesnap='cursor'
    )

    # Enable Y-axis autorange for dynamic scaling
    fig.update_yaxes(autorange=True, fixedrange=False)

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="NLV ($)", row=1, col=2)
    fig.update_yaxes(title_text="Margin ($)", secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text="HMM State", row=2, col=1)
    fig.update_yaxes(title_text="VIX", secondary_y=True, row=2, col=1)
    fig.update_yaxes(title_text="Premium ($)", row=2, col=2)
    fig.update_yaxes(title_text="Pair Value ($)", row=3, col=1)
    fig.update_yaxes(title_text="DTE (Days)", row=3, col=2)
    fig.update_yaxes(title_text="Spread ($)", secondary_y=True, row=3, col=2)

    # --- Apply Regime Backgrounds to ALL Subplots ---
    if result.regime_history and 'reg_df' in locals():
        regime_colors = {
            "Panic": "#ff0000",       # Pure Red
            "Crisis": "#ff0000",      # Pure Red
            "High Vol": "#ff9933",    # Orange
            "Stress": "#ff9933",      # Orange
            "Bearish": "#ffcccc",     # Light Pink-Red
            "Bullish": "#ccffcc",     # Light Green
            "Quiet": "#e6ffe6",       # Very Light Green
            "Choppy": "#fff5e6",      # Very Light Orange
            "Neutral": "#f2f2f2",     # Very Light Grey
        }
        state_changes = reg_df['state'].ne(reg_df['state'].shift()).cumsum()
        for _, group in reg_df.groupby(state_changes):
            state = group['state'].iloc[0]
            if state == -1: continue
            label = result.regime_labels.get(state, f"Regime {state}")
            color = "#f2f2f2"
            for key in ["Panic", "Crisis", "High Vol", "Stress", "Bearish", "Bullish", "Quiet", "Choppy", "Neutral"]:
                if key in label:
                    color = regime_colors[key]
                    break
            # Add to all 6 subplots explicitly using row/col
            for r in [1, 2, 3]:
                for c in [1, 2]:
                    fig.add_vrect(
                        x0=group['date'].iloc[0], x1=group['date'].iloc[-1],
                        fillcolor=color, opacity=0.35, layer="below", line_width=0,
                        row=r, col=c
                    )

    # Sync X-axes using Plotly's native 'matches' property
    # This ensures all subplots stay in sync even after 'Reset Axes' or 'Autoscale'
    for i in range(2, 7):
        ax_name = f'xaxis{i}'
        if ax_name in fig.layout:
            fig.layout[ax_name].matches = 'x'

    output_path = "ev_plots_backtest.html"
    
    # JavaScript to synchronize X-axes and auto-rescale Y-axes when zooming.
    # Since we can't use Plotly's built-in shared_xaxes (broken with secondary_y),
    # we handle all cross-subplot synchronization in JS.
    auto_rescale_js = """
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    var _rescaleTimer = null;
    var _isRescaling = false;
    
    gd.on('plotly_relayout', function(eventData) {
        if (_isRescaling) return;

        // Detect x-axis change (sync is handled natively by 'matches')
        var xKey = null;
        for (var key in eventData) {
            if (key.indexOf('xaxis') === 0) {
                xKey = key.split('.')[0];
                break;
            }
        }
        if (!xKey) return;
        
        // Handle Reset
        if (eventData[xKey + '.autorange'] === true) {
            _isRescaling = true;
            var update = {};
            for (var key in gd.layout) {
                if (key.match(/^yaxis\\d*$/)) update[key + '.autorange'] = true;
            }
            Plotly.relayout(gd, update).then(function() { _isRescaling = false; });
            return;
        }
        
        // Handle Zoom: Auto-rescale Y axes
        var range = gd.layout[xKey].range;
        if (!range || !range[0]) return;

        if (_rescaleTimer) clearTimeout(_rescaleTimer);
        _rescaleTimer = setTimeout(function() {
            var data = gd.data;
            var update = {};
            var x0 = new Date(range[0]).getTime();
            var x1 = new Date(range[1]).getTime();
            if (isNaN(x0) || isNaN(x1)) return;

            var yRanges = {};
            for (var i = 0; i < data.length; i++) {
                var trace = data[i];
                if (!trace.x || !trace.y || trace.visible === false) continue;
                var yax = trace.yaxis || 'y';
                var yaxName = 'yaxis' + (yax === 'y' ? '' : yax.substring(1));
                
                var yMin = Infinity, yMax = -Infinity;
                for (var j = 0; j < trace.x.length; j++) {
                    var tx = new Date(trace.x[j]).getTime();
                    if (tx >= x0 && tx <= x1) {
                        var yv = trace.y[j];
                        if (yv !== null && yv !== undefined && isFinite(yv)) {
                            yMin = Math.min(yMin, yv);
                            yMax = Math.max(yMax, yv);
                        }
                    }
                }
                if (yMin !== Infinity) {
                    if (!yRanges[yaxName]) yRanges[yaxName] = {min: yMin, max: yMax};
                    else {
                        yRanges[yaxName].min = Math.min(yRanges[yaxName].min, yMin);
                        yRanges[yaxName].max = Math.max(yRanges[yaxName].max, yMax);
                    }
                }
            }

            for (var axName in yRanges) {
                var r = yRanges[axName];
                var span = r.max - r.min;
                if (span === 0) span = Math.abs(r.max) * 0.1 || 1;
                var pad = span * 0.05;
                update[axName + '.range'] = [r.min - pad, r.max + pad];
                update[axName + '.autorange'] = false;
            }

            if (Object.keys(update).length > 0) {
                _isRescaling = true;
                Plotly.relayout(gd, update).then(function() { _isRescaling = false; });
            }
        }, 100);
    });
    """
    
    fig.write_html(output_path, include_plotlyjs='cdn', post_script=auto_rescale_js)
    print(f"\n  ✓ Interactive plot saved to: {output_path}")


def _fetch_underlying_prices(underlying: str, start: str, end: str) -> pd.Series:
    """Fetch daily close prices via yfinance with local caching.
    Uses a simple Date,Price CSV format for caching to avoid column shifting.
    """
    cache_dir = "s_and_p_data"
    cache_path = os.path.join(cache_dir, f"underlying_{underlying}.csv")
    os.makedirs(cache_dir, exist_ok=True)

    # Try loading from cache first
    cached_series = None
    if os.path.exists(cache_path):
        try:
            # We expect a simple CSV with Date,Price (index_col=0)
            df = pd.read_csv(cache_path, index_col=0)
            if not df.empty:
                # Use the first column regardless of name
                s = df.iloc[:, 0]
                s.index = pd.to_datetime(s.index, errors='coerce')
                s = s[s.index.notna()].sort_index()
                s = s[~s.index.duplicated(keep='last')]
                
                if not s.empty:
                    cache_start = s.index.min()
                    cache_end = s.index.max()
                    req_start = pd.Timestamp(start)
                    req_end = pd.Timestamp(end)
                    
                    # Tolerant comparison (4 days slack)
                    if req_start >= (cache_start - pd.Timedelta(days=4)) and \
                       req_end <= (cache_end + pd.Timedelta(days=4)):
                        mask = (s.index >= start) & (s.index <= end)
                        subset = s.loc[mask]
                        if not subset.empty:
                            print(f"  Using cached {underlying} prices ({len(subset)} rows)")
                            subset.index = subset.index.strftime("%Y-%m-%d")
                            return subset
                    cached_series = s
        except Exception as e:
            print(f"  WARNING: Failed to read underlying cache {cache_path}: {e}")

    # Download from yfinance
    print(f"  Fetching {underlying} from yfinance ({start} to {end})...")
    try:
        from live_trading.ev_engine import yf_download_with_retry
        fetch_start = min(start, "2020-01-01") 
        data = yf_download_with_retry(underlying, start=fetch_start, end=end, auto_adjust=False)
        
        if data.empty:
            return pd.Series(dtype=float)

        # Robust extraction of 'Close' or first column
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close = data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            else:
                close = data.iloc[:, 0]
        else:
            close = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

        close.index = pd.to_datetime(close.index)
        
        # Merge with cache
        if cached_series is not None:
            combined = pd.concat([cached_series, close]).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            close = combined

        # Save to cache (Date,Price format)
        close.to_frame("Price").to_csv(cache_path)
        
        # Return requested range
        close.index = close.index.strftime("%Y-%m-%d")
        mask = (close.index >= start) & (close.index <= end)
        return close[mask]
        
    except Exception as e:
        print(f"  WARNING: yfinance fetch failed for {underlying}: {e}")
        return pd.Series(dtype=float)



def _print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    print(f"  API Calls:       {result.api_calls:,}")
    print(f"  Cache Hits:      {result.cache_hits:,}")
    hit_rate = (result.cache_hits / max(1, result.api_calls + result.cache_hits)) * 100
    print(f"  Cache Hit Rate:  {hit_rate:.1f}%")
    
    print(f"  ─────────────────────────────────────")
    print(f"  NBBO Data Gaps:  {result.data_gap_count} (fell back to OHLCV)")
    print(f"  Critical Gaps:   {result.critical_gap_count} (fell back to entry)")

    if not result.trades:
        print("\n  No trades recorded.")
        return

    win_rate = (result.win_count / result.total_trades * 100) if result.total_trades > 0 else 0.0

    print(f"  ─────────────────────────────────────")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winners:         {result.win_count} ({win_rate:.1f}%)")
    print(f"  Losers:          {result.loss_count} ({100-win_rate:.1f}%)")
    print(f"  Avg Credit:      ${result.avg_credit:.2f}/share")
    print(f"  Avg PnL:         ${result.avg_pnl:.0f}/contract")
    print(f"  Total PnL:       ${result.total_pnl:,.0f} (per 1-lot)")
    print(f"  Max Drawdown:    ${result.max_drawdown:,.0f}")
    print()

    # Trade detail table
    print("  Date       │ Strikes      │ Qty │ Δ      │ DTE │ Credit │ PnL/C   │ Total PnL │ Reason     │ Entry Regime")
    print("  ───────────┼──────────────┼─────┼────────┼─────┼────────┼─────────┼───────────┼────────────┼───────────────────")
    for t in result.trades:
        # For open trades, pnl_per_contract is 0.0
        status_icon = "✅" if t.pnl_per_contract > 0 else ("❌" if t.status == "closed" else "⚠️")
        total_pnl = t.pnl_per_contract * t.num_contracts
        
        # Display exit reason or status
        reason_str = t.exit_reason if t.status == "closed" else (t.status.upper())
        
        print(
            f"  {t.entry_date} │ {t.short_strike:>6.1f}/{t.long_strike:<6.1f} │ "
            f"{t.num_contracts:>3d} │ {t.entry_delta:>6.3f} │ {t.entry_dte:>3d} │ "
            f"${t.net_credit:>5.2f} │ ${t.pnl_per_contract:>6.0f}  │ ${total_pnl:>8.0f}  │ {reason_str:<10} │ {t.entry_regime_name:<18} {status_icon}"
        )
    print()


async def main():
    """CLI entry point for standalone testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Historical Option Backtest")
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--start", default="2025-03-03")
    parser.add_argument("--end", default="2025-03-14")
    parser.add_argument("--target-dte", type=int, default=42)
    parser.add_argument("--close-dte", type=int, default=21)
    parser.add_argument("--short-delta", type=float, default=-0.15)
    parser.add_argument("--spread-width", type=float, default=20.0)
    parser.add_argument("--plot", action="store_true", help="Plot backtest results")
    parser.add_argument("--log", action="store_true", help="Enable structured path logging")
    args = parser.parse_args()

    await run_put_credit_spread_backtest(
        underlying=args.underlying,
        start_date=args.start,
        end_date=args.end,
        target_dte=args.target_dte,
        close_dte=args.close_dte,
        target_short_delta=args.short_delta,
        spread_width=args.spread_width,
        plot=args.plot,
        enable_logging=args.log,
    )


if __name__ == "__main__":
    asyncio.run(main())
