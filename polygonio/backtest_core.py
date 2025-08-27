from __future__ import annotations
"""
Backtest orchestration primitives (no strategy changes).

This module wires together the previously split components (calendar,
chains, pricing, earnings, option math, settings) and exposes thin,
composable functions you can call from a CLI, notebook, or runner.

Design goals
------------
- Keep the original trading logic intact (no behavior changes)
- Provide clear seams for testing and parallelization
- Avoid hidden globals except for the pre-existing caches
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import get_settings
from .market_calendar import TradingCalendar
from .chains import pull_option_chain_data
from .pricing import interpolate_option_price, calculate_delta
from .earnings import get_earnings_dates
from .option_math import calculate_implied_volatility


# ----------------------------
# Configuration surface
# ----------------------------

@dataclass(frozen=True)
class BacktestConfig:
    ticker: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    expiry_weekday: str = "Friday"  # or "Wednesday"
    expiring_wk: int = 1            # weeks between expiries (carried from legacy)
    force_otm: bool = False         # legacy knob
    force_update: bool = False      # refresh remote quotes even if cached

    @staticmethod
    def from_settings(ticker: str, start_date: str, end_date: str) -> "BacktestConfig":
        s = get_settings()
        # Map a few legacy knobs if needed later
        return BacktestConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            expiry_weekday="Friday",
            expiring_wk=1,
            force_otm=False,
            force_update=s.option_chain_force_update,
        )


# ----------------------------
# Calendar helpers
# ----------------------------

def trading_dates_df(start_date: str, end_date: str):
    """Return a DataFrame with a single 'date' column of trading datetimes.

    Uses NYSE by default via `TradingCalendar` and mirrors the legacy
    behavior of adjusting to last trading day <= candidate weekday.
    """
    cal = TradingCalendar("NYSE")
    return cal.trading_dates_df(start_date, end_date)


def list_expiries(weekday: str, start_date: str, end_date: str, expiring_wk: int, trading_df) -> List:
    """Wrapper that delegates to the calendar module's logic.

    Kept as a thin compatibility layer in case the caller uses the same
    signature as before.
    """
    from .market_calendar import list_expiries as _list

    return _list(weekday, start_date, end_date, expiring_wk, trading_df)


# ----------------------------
# Chain + pricing snapshot
# ----------------------------

async def build_chain_snapshot(
    *,
    ticker: str,
    expiration_date: str,
    as_of_date: str,
    close_price: float,
    client,
    call_put: str = "call_put_both",  # "call", "put", or "call_put_both" (legacy accepted values)
    force_otm: bool = False,
    force_update: bool = False,
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Dict[str, Optional[float]]]],
]:
    """Fetch/assemble a single (expiry, as-of) chain slice around spot.

    This is a small adapter over `pull_option_chain_data` that preserves
    the original return shape, while allowing the caller to request only
    calls or puts by passing `call_put`.
    """
    cp = {
        "call": "call",
        "put": "put",
        "call_put_both": "call_put_both",
    }.get(call_put, "call_put_both")

    return await pull_option_chain_data(
        ticker=ticker,
        call_put=cp,
        expiration_str=expiration_date,
        as_of_str=as_of_date,
        close_price=close_price,
        client=client,
        force_otm=force_otm,
        force_update=force_update,
    )


# ----------------------------
# Premium estimation adapter
# ----------------------------

async def estimate_premium(
    *,
    ticker: str,
    close_price_today: float,
    strike: float,
    option_type: str,
    expiration_date: str,
    pricing_date: str,
    client,
) -> float:
    """Thin adapter to the interpolation function (unchanged behavior)."""
    return await interpolate_option_price(
        ticker=ticker,
        close_price_today=close_price_today,
        strike_price_to_interpolate=strike,
        option_type=option_type,
        expiration_date=expiration_date,
        pricing_date=pricing_date,
        client=client,
    )


# ----------------------------
# Earnings / IV helpers (unchanged behavior)
# ----------------------------

def earnings_dates_in_range(ticker: str, start_date: str, end_date: str):
    return get_earnings_dates(ticker, start_date, end_date)


def implied_vol(
    *,
    spot: float,
    strike: float,
    option_price: float,
    days_to_expire: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
):
    return calculate_implied_volatility(
        close_price=spot,
        strike_price=strike,
        option_price=option_price,
        days_to_expire=days_to_expire,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )


# ----------------------------
# Delta population (compat)
# ----------------------------

def populate_deltas(ticker: str, pricing_date: str, expiration_date: str, call_put: str, *, force: bool = False) -> None:
    calculate_delta(ticker, pricing_date, expiration_date, call_put, force_delta_update=force)


# ----------------------------
# Parallel monthly runner (adapter)
# ----------------------------

def run_monthly_backtest_cpu_bound(
    ticker: str,
    global_start_date: str,
    global_end_date: str,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
):
    """Adapter to the existing monthly CPU-bound runner.

    We import lazily to avoid hard dependency if you stub/replace this
    in tests. Return value and semantics are identical to your original
    `monthly_cpu_bound.run_monthly_backtest_cpu_bound`.
    """
    from monthly_cpu_bound import run_monthly_backtest_cpu_bound as _run

    kwargs = kwargs or {}
    return _run(ticker, global_start_date, global_end_date, **kwargs)

