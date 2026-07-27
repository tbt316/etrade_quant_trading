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
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple, Dict, Any, Mapping

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
from backtesting.contract_universe import (
    ContractUniverseSnapshot,
    acquisition_union_by_expiration,
    build_contract_universe_snapshot,
    filter_chain_for_snapshot,
)
from live_trading.regime_signal import (
    RegimeSignal,
    annotation_for_session,
)


# ANSI Colors for terminal logging
CLR_RED = "\033[91m"
CLR_YEL = "\033[93m"
CLR_RST = "\033[0m"


NYSE = mcal.get_calendar("NYSE")


def validate_regime_v2_annotations(
    annotations: Optional[Mapping[str, RegimeSignal]],
) -> Dict[str, RegimeSignal]:
    """Validate the exact-date, shadow-only V2 annotation boundary.

    V2 signals deliberately remain separate from legacy integer HMM regimes.
    This function accepts no numeric states, performs no forward fill, and
    returns no order-facing projection.
    """

    if annotations is None:
        return {}
    if not isinstance(annotations, Mapping):
        raise TypeError("regime_v2_annotations must be a mapping")

    validated: Dict[str, RegimeSignal] = {}
    for key, value in annotations.items():
        if not isinstance(key, str):
            raise TypeError("regime_v2_annotations keys must be ISO date strings")
        signal = annotation_for_session(annotations, key)
        if signal is None or signal is not value:
            raise ValueError("regime_v2_annotations contains an invalid entry")
        if signal.may_authorize_execution:
            raise ValueError("R4 V2 annotations cannot authorize execution")
        validated[key] = signal
    return validated


def lag_daily_regime_map(regimes: Dict[str, int], trading_dates: List[str]) -> Dict[str, int]:
    """
    Shift close_T regime labels forward one trading session for trade entry.

    The regime detector uses daily close inputs, so a state calculated for date T
    is available for the next trading session by default. This helper converts a
    close-dated trace into the map consumed by run_put_credit_spread_backtest().
    """
    if not regimes:
        return {}
    lagged = {}
    previous_state = None
    for td in trading_dates:
        if previous_state is not None:
            lagged[td] = previous_state
        if td in regimes:
            previous_state = regimes[td]
    return lagged


def _forward_return_assignment_strike(
    underlying_prices: pd.Series,
    trading_dates: List[str],
    trading_date_index: Dict[str, int],
    as_of_date: str,
    current_spot: float,
    option_horizon_days: int,
    target_assignment_prob: float,
    contract_type: str = "put",
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Select the strike implied by the causal forward-return distribution.

    The distribution is rebuilt using only outcomes that were fully resolved as
    of the trade date and is censored to the requested option horizon.
    """
    if underlying_prices is None or underlying_prices.empty:
        return None, {"reason": "missing_underlying_prices"}

    # We map the full series of underlying prices to construct a complete,
    # robust causal history rather than throwing away pre-backtest prices.
    all_dates = list(underlying_prices.index)
    all_date_index = {date: idx for idx, date in enumerate(all_dates)}

    as_of_idx_full = all_date_index.get(as_of_date)
    if as_of_idx_full is None:
        return None, {
            "reason": "as_of_date_missing_from_underlying_prices",
            "as_of_date": as_of_date,
        }

    trading_horizon = max(1, int(round(float(option_horizon_days) * 252.0 / 365.0)))
    # A sample is eligible only after its terminal close is strictly earlier
    # than the entry session. A return resolving on the entry date is not
    # decision-time evidence for that same EOD entry.
    resolved_cutoff_full = as_of_idx_full - trading_horizon - 1
    if resolved_cutoff_full < 1:
        return None, {"reason": "insufficient_history", "trading_horizon": trading_horizon}

    # Calculate returns on the full unsliced pricing series to leverage 2011+ history
    full_prices = underlying_prices.astype(float)
    forward_returns_full = full_prices.shift(-trading_horizon) / full_prices - 1.0
    
    # Causal slice up to resolved_cutoff_full
    sample = forward_returns_full.iloc[: resolved_cutoff_full + 1].dropna()
    sample = sample[np.isfinite(sample)]
    if len(sample) < 10:
        return None, {
            "reason": "insufficient_resolved_returns",
            "trading_horizon": trading_horizon,
            "n": int(len(sample)),
        }

    target_prob = float(np.clip(target_assignment_prob, 1e-6, 1.0 - 1e-6))
    if contract_type == "call":
        target_return = float(np.quantile(sample.values, 1.0 - target_prob))
    else:
        target_return = float(np.quantile(sample.values, target_prob))

    spot = float(current_spot) if np.isfinite(current_spot) else np.nan
    strike = None if not np.isfinite(target_return) or not np.isfinite(spot) else spot * (1.0 + target_return)
    diagnostics = {
        "reason": "empirical_forward_return_quantile",
        "n": int(len(sample)),
        "trading_horizon": int(trading_horizon),
        "latest_resolution_session": pd.Timestamp(
            all_dates[resolved_cutoff_full + trading_horizon]
        ).date().isoformat(),
        "outcomes_resolved_strictly_before_entry": True,
        "target_assignment_prob": target_prob,
        "target_return": target_return,
    }
    return strike, diagnostics


def _build_trade_entry_regime_inputs(feature_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build the regime map and labels used by the backtester.

    Preference order:
      1. Audit-consistent causal stress overlay from train_regime_hmm()
      2. Raw HMM state trace as a fallback

    This keeps the backtest aligned with the probability audit script, which
    uses the detected overlay as the final actionable regime and then lags it
    one trading day for entry.
    """
    if feature_df is None or feature_df.empty:
        return {}, {}

    if {"Detected_Regime_State", "Detected_Regime_Label"}.issubset(feature_df.columns):
        state_col = "Detected_Regime_State"
        label_col = "Detected_Regime_Label"
    elif {"Raw_Overlay_State", "Raw_Overlay_Label"}.issubset(feature_df.columns):
        state_col = "Raw_Overlay_State"
        label_col = "Raw_Overlay_Label"
    else:
        state_col = "HMM_State"
        label_col = "Regime_Label"

    close_regimes = {
        d.strftime("%Y-%m-%d"): int(s)
        for d, s in feature_df[state_col].dropna().to_dict().items()
    }

    regime_labels = {}
    if label_col in feature_df.columns:
        label_pairs = feature_df[[state_col, label_col]].dropna().drop_duplicates(subset=[state_col])
        for _, row in label_pairs.iterrows():
            regime_labels[int(row[state_col])] = str(row[label_col])

    return close_regimes, regime_labels


@dataclass
class SpreadTrade:
    """Represents a single credit spread trade."""
    entry_date: str
    option_type: str = "put"
    exit_date: str = ""
    expiration: str = ""
    short_strike: float = 0.0
    long_strike: float = 0.0
    far_long_ticker: str = ""
    far_long_strike: float = 0.0
    short_entry_mid: float = 0.0
    long_entry_mid: float = 0.0
    far_long_entry_mid: float = 0.0
    net_credit: float = 0.0
    short_exit_mid: float = 0.0
    long_exit_mid: float = 0.0
    far_long_exit_mid: float = 0.0
    net_debit_close: float = 0.0
    pnl_per_share: float = 0.0
    pnl_per_contract: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    total_fees: float = 0.0
    entry_delta: float = 0.0
    entry_dte: int = 0
    exit_dte: int = 0
    status: str = "open"     # "open", "closed", "expired", "error"
    num_contracts: int = 1
    margin_required: float = 0.0
    exit_reason: str = ""    # "scheduled", "early_profit", "roll_chain_breakeven", "expired"
    hold_to_expiration: bool = False # If ITM at close_dte, hold until expiration
    is_roll: bool = False # If opened from rollover
    entry_regime: int = -1
    entry_regime_name: str = ""
    entry_regime_v2: Optional[Dict[str, Any]] = None
    profit_target: float = 0.70 # Default to early_profit_pct
    short_ticker: str = ""
    long_ticker: str = ""
    entry_price: float = 0.0
    trade_id: str = ""
    root_trade_id: str = ""
    parent_trade_id: str = ""
    roll_chain_depth: int = 0
    roll_chain_realized_pnl_before_entry: float = 0.0
    
    _exp_dt: Optional[datetime] = field(default=None, init=False, repr=False)
    
    @property
    def expiration_dt(self) -> datetime:
        if self._exp_dt is None:
            self._exp_dt = datetime.strptime(self.expiration, "%Y-%m-%d")
        return self._exp_dt


def _trade_standalone_margin(trade: SpreadTrade) -> float:
    return abs(float(trade.short_strike) - float(trade.long_strike)) * 100.0 * int(trade.num_contracts)


def _portfolio_margin_requirement(trades: List[SpreadTrade]) -> float:
    """
    Risk-defined SPY put/call spreads with the same expiration are margined as
    the larger side, not the sum of both sides. Grouping by expiration prevents
    call-side entries from receiving offset credit against unrelated expiries.
    """
    expiry_sides: Dict[str, Dict[str, float]] = {}
    for trade in trades:
        if trade.status != "open":
            continue
        option_type = (trade.option_type or "put").lower()
        if option_type not in {"put", "call"}:
            continue
        expiry_sides.setdefault(trade.expiration, {"put": 0.0, "call": 0.0})
        expiry_sides[trade.expiration][option_type] += _trade_standalone_margin(trade)
    return sum(max(sides["put"], sides["call"]) for sides in expiry_sides.values())


def _option_trade_fee(num_contracts: int, num_legs: int, fee_per_contract_per_side: float) -> float:
    return max(0.0, float(fee_per_contract_per_side)) * max(0, int(num_contracts)) * max(0, int(num_legs))


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
    risk_free_rates: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    regime_history: List[Tuple[str, int]] = field(default_factory=list)
    regime_labels: Dict[int, str] = field(default_factory=dict)
    regime_v2_history: List[Tuple[str, Optional[Dict[str, Any]]]] = field(
        default_factory=list
    )
    daily_leg_premiums: List[Tuple[str, float, float]] = field(default_factory=list)
    daily_scatter_data: List[dict] = field(default_factory=list)
    daily_opened_dte: List[Tuple[str, int]] = field(default_factory=list)
    daily_opened_spread: List[Tuple[str, float]] = field(default_factory=list)
    initial_capital: float = 100000.0
    margin_limit_pct: float = 0.5
    data_gap_count: int = 0
    critical_gap_count: int = 0
    abnormalities: List[Dict[str, Any]] = field(default_factory=list)
    causal_validity: str = "UNVERIFIED"
    causal_validity_reasons: List[str] = field(
        default_factory=lambda: [
            "contract_reference_available_at_is_modeled_not_provider_observed",
            "contract_reference_pagination_completeness_unverified",
            "fill_timestamp_causality_not_certified",
        ]
    )
    contract_universe_manifest_sha256: Dict[str, str] = field(
        default_factory=dict
    )
    contract_universe_snapshot_request_count: int = 0
    contract_universe_request_amplification: float = 0.0


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

    def log_day(
        self,
        date_str: str,
        spot: float,
        regime: int,
        regime_name: str,
        regime_v2: Optional[Dict[str, Any]],
        nlv: float,
        cash: float,
        margin: float,
        active_trades: list,
        events: list,
        abnormalities: list = None,
        margin_budget_capital: Optional[float] = None,
        margin_limit: Optional[float] = None,
    ):
        day_entry = {
            "date": date_str,
            "spot_price": float(spot),
            "regime": int(regime),
            "regime_name": regime_name,
            "regime_v2": regime_v2,
            "nlv": float(nlv),
            "cash": float(cash),
            "margin": float(margin),
            "margin_budget_capital": float(margin_budget_capital) if margin_budget_capital is not None else float(cash),
            "margin_limit": float(margin_limit) if margin_limit is not None else None,
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


def _session_decision_times(trading_dates: List[str]) -> Dict[str, str]:
    """Bind this EOD simulator to each session's exact NYSE close timestamp."""

    if not trading_dates:
        return {}
    schedule = NYSE.schedule(
        start_date=min(trading_dates),
        end_date=max(trading_dates),
    )
    closes = {
        index.strftime("%Y-%m-%d"): (
            pd.Timestamp(row["market_close"])
            .tz_convert("UTC")
            .isoformat()
            .replace("+00:00", "Z")
        )
        for index, row in schedule.iterrows()
    }
    missing = sorted(set(trading_dates) - set(closes))
    if missing:
        raise ValueError(
            f"missing NYSE decision timestamps for sessions: {missing[:3]}"
        )
    return {trade_date: closes[trade_date] for trade_date in trading_dates}


async def _fetch_point_in_time_contract_universe(
    client,
    *,
    underlying: str,
    expiration: str,
    contract_type: str,
    trade_date: str,
    decision_time: str,
) -> Optional[ContractUniverseSnapshot]:
    """Fetch and seal the reference universe as of the trade decision date."""

    contracts = await client.fetch_contracts_list(
        underlying,
        expiration,
        contract_type,
        trade_date,
    )
    if not contracts:
        return None
    return build_contract_universe_snapshot(
        underlying=underlying,
        expiration=expiration,
        contract_type=contract_type,
        as_of_date=trade_date,
        # Massive documents a date-granular point-in-time `as_of` view, not an
        # observed intraday available_at. Close T is therefore a conservative
        # modeling boundary, not provider evidence; the run remains UNVERIFIED.
        available_at=decision_time,
        decision_time=decision_time,
        contracts=contracts,
    )


def _point_in_time_chain(
    cache,
    contract_universes: Mapping[
        Tuple[str, str, str],
        ContractUniverseSnapshot,
    ],
    *,
    underlying: str,
    expiration: str,
    contract_type: str,
    trade_date: str,
    decision_time: str,
) -> list:
    """Filter cached price rows through the exact decision-time universe."""

    snapshot = contract_universes.get(
        (trade_date, expiration, contract_type)
    )
    if snapshot is None:
        return []
    raw_chain = cache.get_chain_for_date(
        underlying,
        expiration,
        contract_type,
        trade_date,
    )
    return filter_chain_for_snapshot(
        raw_chain,
        snapshot,
        decision_time,
    )


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

    # Generate candidates: prioritize the week closest to target_dte, 
    # then try +/- weeks. For each week, try Friday then Thursday (holiday shift).
    candidates = []
    # Week offsets to try: 0 (closest), then 1, -1, 2, -2, etc.
    for offset in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
        week_friday = closest_friday + timedelta(weeks=offset)
        candidates.append(week_friday)
        candidates.append(week_friday - timedelta(days=1)) # Thursday shift
        # Note: Wednesday shifts are rare (e.g. Christmas), but we could add them if needed.

    # Filter out past dates and ensure uniqueness
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_str = c.strftime("%Y-%m-%d")
        if c > td and c_str not in seen:
            unique_candidates.append(c_str)
            seen.add(c_str)
            
    return unique_candidates[:12] # Return top candidates


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


def _select_put_roll_strikes_from_atm_long(
    delta_chain: list,
    spot: float,
    spread_width: float,
) -> Optional[Tuple[int, float, float, float]]:
    """
    Select a put roll by anchoring the long leg near ATM, then moving the
    requested spread width upward to find the short leg.

    Returns (short_idx, short_strike, long_strike, actual_width).
    """
    if not delta_chain:
        return None

    long_idx = min(range(len(delta_chain)), key=lambda idx: abs(delta_chain[idx][0] - spot))
    long_strike = float(delta_chain[long_idx][0])
    target_short = long_strike + spread_width
    short_idx = min(range(len(delta_chain)), key=lambda idx: abs(delta_chain[idx][0] - target_short))
    short_strike = float(delta_chain[short_idx][0])
    actual_width = short_strike - long_strike
    if actual_width <= 0:
        return None
    return short_idx, short_strike, long_strike, actual_width


def _select_put_roll_strikes_for_target_credit(
    delta_chain: list,
    chain_data: list,
    spread_width: float,
    target_net_credit: float,
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Select a put spread by keeping width fixed and choosing the short strike
    whose observed credit is closest to the requested target credit.

    This is used for the panic-transition roll variant to keep the roll as
    close to cash-neutral as the available chain allows.
    """
    if not delta_chain or not chain_data or spread_width <= 0:
        return None

    strikes = [r["strike"] for r in chain_data]
    best = None
    best_score = None
    target_net_credit = max(0.0, float(target_net_credit))

    for short_idx, (short_strike, _, short_mid_guess) in enumerate(delta_chain):
        short_strike = float(short_strike)
        long_strike = short_strike - spread_width
        if long_strike < delta_chain[0][0]:
            continue

        long_mid = next(
            (
                _row_mark_price(r)
                for r in chain_data
                if abs(r["strike"] - long_strike) < 0.01 and _row_mark_price(r) > 0
            ),
            0.0,
        )
        if not long_mid:
            nearest_long = min(strikes, key=lambda s: abs(s - long_strike))
            long_mid = next(
                (
                    _row_mark_price(r)
                    for r in chain_data
                    if abs(r["strike"] - nearest_long) < 0.01 and _row_mark_price(r) > 0
                ),
                0.0,
            )
            long_strike = float(nearest_long)

        actual_spread_width = short_strike - long_strike
        if actual_spread_width <= 0:
            continue

        short_mid = float(short_mid_guess or 0.0)
        if short_mid <= 0:
            short_mid = next(
                (
                    _row_mark_price(r)
                    for r in chain_data
                    if abs(r["strike"] - short_strike) < 0.01 and _row_mark_price(r) > 0
                ),
                0.0,
            )
        if short_mid <= 0:
            continue

        net_credit = short_mid - long_mid
        if net_credit <= 0 or net_credit >= actual_spread_width:
            continue

        score = abs(net_credit - target_net_credit)
        if best is None or score < best_score:
            best = (short_idx, short_strike, long_strike, actual_spread_width, net_credit)
            best_score = score

    return best


def _roll_setup_requires_deviation(
    chain_data: list,
    contracts_by_exp: dict,
    selected_exp: str,
    trade_date: str,
    spot: float,
    spread_width: float,
    risk_free_rate: float,
    dividend_yield: float,
    spread_width_tolerance_pct: float,
    min_credit: float,
) -> bool:
    """
    Return True when the exact requested roll setup is not priceable and the
    engine would need to shift strikes or fall back to imprecise pricing.
    """
    if not chain_data:
        return True

    exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
    trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    dte_days = (exp_dt - trade_dt).days
    if dte_days <= 0:
        return True
    dte_years = dte_days / 365.0

    strikes = [r["strike"] for r in chain_data]
    close_prices = [r["close"] for r in chain_data]
    valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]
    if len(valid) < 5:
        return True

    v_strikes, v_prices = zip(*valid)
    delta_chain = compute_chain_deltas(
        spot,
        list(v_strikes),
        list(v_prices),
        dte_years,
        risk_free_rate,
        dividend_yield,
        "put",
    )
    exact_roll = _select_put_roll_strikes_from_atm_long(delta_chain, spot, spread_width)
    if not exact_roll:
        return True

    short_idx, short_strike, long_strike, actual_spread_width = exact_roll
    if not _spread_width_within_tolerance(actual_spread_width, spread_width, spread_width_tolerance_pct):
        return True

    long_mid = next((_row_mark_price(r) for r in chain_data if abs(r["strike"] - long_strike) < 0.01 and _row_mark_price(r) > 0), 0.0)
    if not long_mid:
        return True

    short_mid = delta_chain[short_idx][2] or 0.0
    net_credit = short_mid - long_mid
    return not (net_credit >= min_credit and net_credit < actual_spread_width)


def _pick_monthly_alternative(
    possible_exps: list,
    contract_universes: Mapping[
        Tuple[str, str, str],
        ContractUniverseSnapshot,
    ],
    daily_chain_cache: dict,
    precomputed_chain_data: dict,
    cache,
    underlying: str,
    trade_date: str,
    decision_time: str,
    min_chain_strikes: int,
) -> Tuple[Optional[str], Optional[list]]:
    """
    Return the first monthly expiration in candidate order that has usable chain data.
    """
    for exp in possible_exps:
        if (
            not _is_third_friday(exp)
            or (trade_date, exp, "put") not in contract_universes
        ):
            continue
        if exp in daily_chain_cache:
            chain_data = daily_chain_cache[exp]
        else:
            precomputed_key = (trade_date, exp)
            if precomputed_key in precomputed_chain_data:
                chain_data = precomputed_chain_data[precomputed_key]
            else:
                chain_data = _point_in_time_chain(
                    cache,
                    contract_universes,
                    underlying=underlying,
                    expiration=exp,
                    contract_type="put",
                    trade_date=trade_date,
                    decision_time=decision_time,
                )
                daily_chain_cache[exp] = chain_data
        if not chain_data:
            continue

        strikes = [r["strike"] for r in chain_data]
        close_prices = [r["close"] for r in chain_data]
        valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]
        if len(valid) < min_chain_strikes:
            continue

        if chain_data:
            return exp, chain_data
    return None, None


def _build_roll_request(
    trade: "SpreadTrade",
    current_trade_pnl_dollars: float,
    reason: str,
    spot: float,
    roll_strike_behavior: str,
    roll_dte_multiplier: float,
    num_contracts: int,
    spread_width_override: float,
    target_net_credit: Optional[float] = None,
) -> dict:
    """
    Build a replacement request for a roll while preserving chain provenance.
    """
    return {
        "target_delta": None,
        "target_long_strike": spot if roll_strike_behavior == "follow_underlying" else None,
        "target_strike": trade.short_strike if roll_strike_behavior != "follow_underlying" else None,
        "num_contracts": num_contracts,
        "spread_width_override": spread_width_override,
        "released_margin": trade.margin_required,
        "target_roll_margin": num_contracts * spread_width_override * 100,
        "target_net_credit": target_net_credit,
        "reason": reason,
        "target_dte_override": int(trade.entry_dte * roll_dte_multiplier) if roll_dte_multiplier != 1.0 else None,
        "root_trade_id": trade.root_trade_id or trade.trade_id,
        "parent_trade_id": trade.trade_id,
        "roll_chain_depth": trade.roll_chain_depth + 1,
        "roll_chain_realized_pnl_before_entry": trade.roll_chain_realized_pnl_before_entry + current_trade_pnl_dollars,
    }


def _spread_width_within_tolerance(actual_width: float, target_width: float, tolerance_pct: float) -> bool:
    if target_width <= 0:
        return False
    return abs(actual_width - target_width) / target_width <= tolerance_pct


def _delta_deviation_pct(actual_delta: float, target_delta: float) -> float:
    """
    Return the relative deviation between actual and target delta.

    The comparison is done on magnitude so that -0.19 vs -0.15 is treated as
    a 26.7% deviation from target.
    """
    target_abs = abs(target_delta)
    if target_abs <= 1e-12:
        return 0.0 if abs(actual_delta) <= 1e-12 else float("inf")
    return abs(abs(actual_delta) - target_abs) / target_abs


def _row_mark_price(row: dict) -> float:
    return row.get("close") or row.get("mid") or 0.0


def _adjust_entry_credit(short_mid: float, short_bid: float, short_ask: float,
                          long_mid: float, long_bid: float, long_ask: float,
                          model: str) -> float:
    if model == "none":
        return short_mid - long_mid
    
    # Under worst-case slippage, we get bid for short and ask for long
    if model == "worst_case":
        s_price = short_bid if (short_bid is not None and short_bid > 0) else short_mid
        l_price = long_ask if (long_ask is not None and long_ask > 0) else long_mid
        return s_price - l_price
    elif model == "steer_50":
        short_spread = (short_ask - short_bid) if (short_bid and short_ask) else 0.0
        long_spread = (long_ask - long_bid) if (long_bid and long_ask) else 0.0
        s_price = short_mid - 0.25 * short_spread
        l_price = long_mid + 0.25 * long_spread
        return s_price - l_price
    return short_mid - long_mid


def _adjust_exit_debit(short_mid: float, short_bid: float, short_ask: float,
                         long_mid: float, long_bid: float, long_ask: float,
                         model: str) -> float:
    if model == "none":
        return short_mid - long_mid
    
    # Under worst-case slippage, we pay ask for short and get bid for long
    if model == "worst_case":
        s_price = short_ask if (short_ask is not None and short_ask > 0) else short_mid
        l_price = long_bid if (long_bid is not None and long_bid > 0) else long_mid
        return s_price - l_price
    elif model == "steer_50":
        short_spread = (short_ask - short_bid) if (short_bid and short_ask) else 0.0
        long_spread = (long_ask - long_bid) if (long_bid and long_ask) else 0.0
        s_price = short_mid + 0.25 * short_spread
        l_price = long_mid - 0.25 * long_spread
        return s_price - l_price
    return short_mid - long_mid


def _pricing_source_label(*quotes: Optional[dict]) -> str:
    sources = {
        str(q.get("source", "")).lower()
        for q in quotes
        if q and not isinstance(q, Exception)
    }
    if not sources:
        return "UNKNOWN"
    if all(src in {"quote", "quote_cache"} for src in sources):
        return "NBBO"
    if all(src in {"trade", "trade_cache"} for src in sources):
        return "TRADE"
    if all(src in {"ohlcv_close", "ohlcv_close_cache"} for src in sources):
        return "OHLCV"
    return "MIXED"


def _ticker_matches_contract_type(option_ticker: str, contract_type: str) -> bool:
    flag = "P" if contract_type == "put" else "C"
    try:
        clean = option_ticker[2:] if option_ticker.startswith("O:") else option_ticker
        first_digit = next(idx for idx, char in enumerate(clean) if char.isdigit())
        return clean[first_digit + 6] == flag
    except (StopIteration, IndexError, TypeError):
        return False


def _row_matches_contract_type(row: dict, contract_type: str) -> bool:
    row_type = row.get("contract_type")
    if row_type:
        return row_type == contract_type and _ticker_matches_contract_type(row.get("option_ticker", ""), contract_type)
    return _ticker_matches_contract_type(row.get("option_ticker", ""), contract_type)


def _find_contract_for_strike(contracts: list, strike: float, contract_type: str = "put") -> Optional[dict]:
    return next(
        (
            c for c in contracts
            if abs(c["strike"] - strike) < 0.01
            and _row_matches_contract_type(c, contract_type)
        ),
        None,
    )


def _find_ticker_for_strike(chain_data: list, strike: float, contract_type: str = "put") -> Optional[str]:
    row = next(
        (
            r for r in chain_data
            if abs(r["strike"] - strike) < 0.01
            and _row_matches_contract_type(r, contract_type)
        ),
        None,
    )
    return row["option_ticker"] if row else None


def _is_third_friday(expiration: str) -> bool:
    exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
    return exp_dt.weekday() == 4 and 15 <= exp_dt.day <= 21


def _compute_delta_chain_worker(payload: Dict[str, Any]) -> Tuple[str, str, str, list, Optional[str]]:
    """
    Process-pool worker for CPU-bound option-chain delta/IV computation.

    Keep this function top-level and pass only primitive data so it is safe with
    the macOS spawn start method and does not share SQLite/cache state.
    """
    try:
        delta_chain = compute_chain_deltas(
            payload["underlying_price"],
            payload["strikes"],
            payload["mid_prices"],
            payload["dte_years"],
            payload["risk_free_rate"],
            payload["dividend_yield"],
            payload["option_type"],
        )
        return payload["trade_date"], payload["expiration"], payload["option_type"], delta_chain, None
    except Exception as exc:
        return payload["trade_date"], payload["expiration"], payload.get("option_type", ""), [], str(exc)


async def run_put_credit_spread_backtest(
    underlying: str = "SPY",
    start_date: str = "2025-01-01",
    end_date: str = datetime.now().strftime("%Y-%m-%d"),
    target_dte: int = 42,
    close_dte: Any = 21,
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
    panic_exit_enabled: bool = True,
    dividend_yield: float = 0.0,
    strategy_config: dict = None,
    daily_pacing_slots: int = 0,
    roll_spread_width_multiplier: float = 1.0,
    slippage_model: str = "none",
    max_time_delta_minutes: float = 5.0,
    regime_probabilities: Optional[pd.DataFrame] = None,
    regime_v2_annotations: Optional[Mapping[str, RegimeSignal]] = None,
    offline_only: bool = False,
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
    result.regime_probabilities = regime_probabilities
    regime_v2_annotations = validate_regime_v2_annotations(
        regime_v2_annotations
    )
    
    # Extract strategy-level configurations
    entry_config = (strategy_config or {}).get("entry", {})
    regime_dynamic_delta = bool(entry_config.get("regime_dynamic_delta", False))
    regime_aware = bool((strategy_config or {}).get("regime_aware", False)) or regime_dynamic_delta
    
    # New Delta & Margin Configs
    dynamic_delta_method = entry_config.get("dynamic_delta_method", None)
    vix_delta_scale_factor = entry_config.get("vix_delta_scale_factor", None)
    target_assignment_prob = float(entry_config.get("target_assignment_prob", 0.05))
    far_long_leg_enabled = bool(entry_config.get("far_long_leg_enabled", False))
    far_long_leg_multiplier = float(entry_config.get("far_long_leg_multiplier", 2.0))
    far_long_leg_vix_below = entry_config.get("far_long_leg_vix_below", None)
    far_long_leg_vix_below = float(far_long_leg_vix_below) if far_long_leg_vix_below is not None else None

    sizing_config = (strategy_config or {}).get("sizing", {})
    dynamic_margin_method = sizing_config.get("dynamic_margin_method", None)
    vix_margin_bounds = sizing_config.get("vix_margin_bounds", None)

    exit_config = (strategy_config or {}).get("exit", {})
    conditional_half_dte_exit = exit_config.get("conditional_half_dte_exit", False)
    max_daily_profit_exit_margin_budget_multiple = float(
        exit_config.get("max_daily_profit_exit_margin_budget_multiple", 0.0) or 0.0
    )

    filters_config = (strategy_config or {}).get("filters", {})
    call_side_config = (strategy_config or {}).get("call_side", {})
    execution_config = (strategy_config or {}).get("execution", {})
    fee_per_contract_per_side = float(
        execution_config.get(
            "fee_per_contract_per_side",
            execution_config.get("option_fee_per_contract_per_side", 0.0),
        )
    )
    require_entry_nbbo = bool(execution_config.get("require_entry_nbbo", False))
    call_side_enabled = bool(call_side_config.get("enabled", False))
    call_side_short_delta = float(call_side_config.get("short_delta", abs(target_short_delta) / 2.0))
    call_side_spread_width = float(call_side_config.get("spread_width", spread_width))
    call_side_vix_skip_above = call_side_config.get("skip_when_vix_above", None)
    call_side_vix_reentry_delay_days = int(call_side_config.get("vix_reentry_delay_days", 5))
    call_side_profit_target = float(call_side_config.get("profit_target", early_profit_pct))
    call_side_hold_to_expiration = bool(call_side_config.get("hold_to_expiration", False))
    rolling_config = (strategy_config or {}).get("rolling", {})
    rolling_enabled = bool(rolling_config.get("enabled", True))
    min_chain_strikes = int(filters_config.get("min_chain_strikes", 10))
    min_credit = float(filters_config.get("min_credit", 0.0))
    call_side_min_credit = float(call_side_config.get("min_credit", min_credit))
    spread_width_tolerance_pct = float(filters_config.get("strike_tolerance_pct", 0.30))
    delta_tolerance_pct = float(filters_config.get("delta_tolerance_pct", 0.15))
    repeat_roll_itm_at_expiration = bool(rolling_config.get("repeat_itm_at_expiration", False))
    prefer_monthly_on_deviation = bool(rolling_config.get("prefer_monthly_on_deviation", False))
    stop_when_chain_breakeven = bool(rolling_config.get("stop_when_chain_breakeven", False))
    roll_dte_multiplier = float(rolling_config.get("roll_dte_multiplier", 1.0))
    roll_strike_behavior = rolling_config.get("roll_strike_behavior", "same_short")
    panic_transition_roll_enabled = bool(rolling_config.get("panic_transition_roll_enabled", False))
    panic_transition_from_states = rolling_config.get("panic_transition_from_states", [0, 1])
    if not isinstance(panic_transition_from_states, (list, tuple, set)):
        panic_transition_from_states = [0, 1]
    panic_transition_from_states = {int(s) for s in panic_transition_from_states}
    panic_transition_to_state = int(rolling_config.get("panic_transition_to_state", 2))
    panic_transition_min_remaining_dte_fraction = float(
        rolling_config.get("panic_transition_min_remaining_dte_fraction", 0.5)
    )
    weekly_sparse_min_valid_rows = int(
        filters_config.get(
            "weekly_sparse_valid_rows",
            max(30, int(filters_config.get("min_chain_strikes", 10)) * 3),
        )
    )

    print("=" * 70)
    print(f"  PUT CREDIT SPREAD BACKTEST (ADVANCED)")
    print(f"  {underlying} | {start_date} → {end_date}")
    print(f"  Target DTE: {target_dte} | Close DTE: {close_dte}")
    print(f"  Short Delta: {target_short_delta} | Width: ${spread_width}")
    if dynamic_delta_method == "forward_assignment_prob":
        print(f"  Assignment Prob Target: {target_assignment_prob:.1%} from causal forward return distribution")
    print(f"  Min Credit: ${min_credit:.2f}")
    print(f"  Spread Width Tolerance: {spread_width_tolerance_pct:.0%}")
    print(f"  Delta Tolerance: {delta_tolerance_pct:.0%}")
    print(f"  Prefer Monthly On Deviation: {prefer_monthly_on_deviation}")
    print(f"  Weekly Sparse Cutoff: {weekly_sparse_min_valid_rows} priced rows")
    print(f"  Initial Capital: ${initial_capital:,.0f} | Margin Limit: {margin_limit_pct:.0%}")
    if fee_per_contract_per_side:
        print(f"  Option Fee: ${fee_per_contract_per_side:.2f}/contract/side")
    if require_entry_nbbo:
        print("  Entry Pricing: strict NBBO required")
    if call_side_enabled:
        print(
            f"  Call Side: enabled | Short Delta: {call_side_short_delta:+.2f} | "
            f"Width: ${call_side_spread_width:g} | Min Credit: ${call_side_min_credit:.2f} | "
            f"VIX Skip Above: {call_side_vix_skip_above} | Re-entry Delay: {call_side_vix_reentry_delay_days} trading days | "
            f"Profit Target: {call_side_profit_target:.0%} | "
            f"Hold To Expiration: {call_side_hold_to_expiration}"
        )
    if panic_transition_roll_enabled:
        print(
            f"  Panic Transition Roll: enabled | From states: {sorted(panic_transition_from_states)} "
            f"-> {panic_transition_to_state} | Min remaining DTE: {panic_transition_min_remaining_dte_fraction:.0%}"
        )
    if far_long_leg_enabled:
        print(
            f"  Far Long Leg: enabled | Multiplier: {far_long_leg_multiplier:.1f}x | "
            f"VIX Below: {far_long_leg_vix_below}"
        )
    print("=" * 70)

    trading_dates = get_trading_dates(start_date, end_date)
    trading_date_index = {date: idx for idx, date in enumerate(trading_dates)}
    decision_times = _session_decision_times(trading_dates)
    print(f"\n  Trading days: {len(trading_dates)}")
    
    # If the strategy is regime-aware and no regimes are provided, auto-train walk-forward HMM
    if regime_aware and regimes is None:
        print("  [Regime Detection] Strategy is regime-aware. Training HMM walk-forward...")
        try:
            from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels
            df_hist = fetch_historical_data()
            if not df_hist.empty:
                cal_start_date = pd.to_datetime(start_date) - timedelta(days=365 * 5)
                df_cal = df_hist[
                    (df_hist.index >= cal_start_date) &
                    (df_hist.index <= pd.to_datetime(end_date))
                ].copy()
                print(f"  [Walk-Forward] Training causal HMM trace from {cal_start_date.strftime('%Y-%m-%d')} to {end_date}...")
                best_hmm, k, feature_df = train_regime_hmm(df_cal, n_components=3, expanding_window=True)
                if best_hmm is not None and not feature_df.empty:
                    close_regimes, detected_labels = _build_trade_entry_regime_inputs(feature_df)
                    regime_labels = detected_labels or get_regime_labels(best_hmm, feature_df)
                    regimes = lag_daily_regime_map(close_regimes, trading_dates)
                    result.regime_labels = regime_labels
                    # Keep the daily probabilities for plotting
                    prob_cols = [f'prob_state_{i}' for i in range(k)]
                    if all(col in feature_df.columns for col in prob_cols):
                        result.regime_probabilities = (
                            feature_df[prob_cols]
                            .reindex(trading_dates)
                            .ffill()
                        )
                    print("  [Regime Timing] Using one-trading-day-lagged close_T overlay regimes for trade entry.")
                else:
                    print("  WARNING: Could not build causal regime trace.")
            else:
                print("  WARNING: Historical data for regimes is empty.")
        except Exception as e:
            print(f"  WARNING: Failed to auto-train regimes: {e}")
    print(f"  Debug: trading_dates sample: {trading_dates[:5]} ... {trading_dates[-5:]}")
    feb_dates = [d for d in trading_dates if "2026-02" in d]
    print(f"  Debug: Feb trading dates count: {len(feb_dates)}")

    # Use pre-fetched prices if provided, otherwise fetch
    if underlying_prices is not None and not underlying_prices.empty:
        print("  Using pre-fetched underlying price data.", flush=True)
        # Ensure it has sufficient history; if it doesn't go back far enough, fetch the full history
        if underlying_prices.index.min() > "2012-01-01":
            print("  Pre-fetched prices do not cover 2011. Re-fetching full history for robust causal analysis...")
            underlying_prices = _fetch_underlying_prices(underlying, "2011-05-03", end_date)
    else:
        print("  Fetching underlying price history...", flush=True)
        # Fetch starting from the warm-up date to ensure historical return distributions are robust
        pricing_start = "2011-05-03"
        underlying_prices = _fetch_underlying_prices(underlying, pricing_start, end_date)
    if underlying_prices.empty:
        print("  ERROR: No underlying price data.")
        cache.close()
        return result
    # Slice the result series to the backtest range so that post-backtest charts and metrics are not stretched or distorted
    result.underlying_prices = underlying_prices[underlying_prices.index >= start_date]

    # Use pre-fetched VIX prices if provided, otherwise fetch
    if vix_prices is not None and not vix_prices.empty:
        print("  Using pre-fetched VIX price data.", flush=True)
    else:
        print("  Fetching VIX price history...", flush=True)
        vix_prices = _fetch_underlying_prices("^VIX", start_date, end_date)
    result.vix_prices = vix_prices

    # Fetch dynamic historical risk-free rate (^IRX)
    print("  Fetching dynamic risk-free rate history (^IRX)...", flush=True)
    irx_prices = _fetch_underlying_prices("^IRX", start_date, end_date)
    if irx_prices.empty:
        print("  WARNING: ^IRX prices empty, falling back to constant 0.05")
        risk_free_rates = pd.Series(0.05, index=trading_dates)
    else:
        # Convert ^IRX percentage yield (e.g. 5.25) to decimal rate (0.0525), reindex/ffill/bfill on trading_dates
        risk_free_rates = (irx_prices / 100.0).reindex(trading_dates).ffill().bfill()
        risk_free_rates = risk_free_rates.fillna(0.05)
    result.risk_free_rates = risk_free_rates
    print(f"  Dynamic Risk-Free Rates loaded: {risk_free_rates.min()*100:.2f}% to {risk_free_rates.max()*100:.2f}%")

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

    async with MassiveAPIClient(cache=cache, offline_only=offline_only) as client:
        # ── Phase 1: Identify all needed expirations ────────────
        print("\n─── Phase 1: Identifying target expirations ───")
        trade_plan = []  # (trade_date, candidates)
        universe_requests = set()  # (trade_date, expiration, contract_type)

        for td in trading_dates:
            current_regime = regimes.get(td, -1) if regimes else -1
            current_regime_name = result.regime_labels.get(current_regime, "")
            is_panic = "Panic / Crisis" in current_regime_name
            
            eff_target_dte = target_dte
            if dynamic_delta_variant and is_panic:
                eff_target_dte = panic_dte_target
                
            candidates = find_target_expiration_friday(td, eff_target_dte)
            trade_plan.append((td, candidates))

            for exp in candidates[:8]:
                universe_requests.add((td, exp, "put"))
                if call_side_enabled:
                    universe_requests.add((td, exp, "call"))

            if rolling_enabled and roll_dte_multiplier != 1.0:
                roll_candidates = find_target_expiration_friday(td, int(eff_target_dte * roll_dte_multiplier))
                for exp in roll_candidates[:8]:
                    universe_requests.add((td, exp, "put"))
                    if call_side_enabled:
                        universe_requests.add((td, exp, "call"))

        # ── Phase 2: Fetch exact decision-time contract universes ────────
        print(
            "\n─── Phase 2: Fetching point-in-time contract universes "
            f"(batch async) ── {len(universe_requests)} snapshots"
        )

        task_meta = sorted(universe_requests)
        expiration_request_keys = {
            (expiration, contract_type)
            for _, expiration, contract_type in task_meta
        }
        result.contract_universe_snapshot_request_count = len(task_meta)
        result.contract_universe_request_amplification = (
            len(task_meta) / len(expiration_request_keys)
            if expiration_request_keys
            else 0.0
        )
        tasks = [
            _fetch_point_in_time_contract_universe(
                client,
                underlying=underlying,
                expiration=exp,
                contract_type=contract_type,
                trade_date=td,
                decision_time=decision_times[td],
            )
            for td, exp, contract_type in task_meta
        ]
        snapshots = await asyncio.gather(*tasks)

        contract_universes: Dict[
            Tuple[str, str, str],
            ContractUniverseSnapshot,
        ] = {}
        for key, snapshot in zip(task_meta, snapshots):
            if snapshot is not None:
                contract_universes[key] = snapshot
        result.contract_universe_manifest_sha256 = {
            "|".join(key): snapshot.snapshot_sha256
            for key, snapshot in sorted(contract_universes.items())
        }

        missing_snapshot_count = len(task_meta) - len(contract_universes)
        if missing_snapshot_count:
            print(
                "  WARNING: "
                f"{missing_snapshot_count} point-in-time contract snapshots "
                "were unavailable and cannot authorize eligibility"
            )

        # These unions are acquisition-only over-fetch plans. Eligibility is
        # always re-applied from contract_universes for the exact trade date.
        contracts_by_exp = acquisition_union_by_expiration(
            contract_universes.values(),
            "put",
        )
        call_contracts_by_exp = acquisition_union_by_expiration(
            contract_universes.values(),
            "call",
        )

        # ── Phase 3: Batch fetch OHLCV for all contracts (global batch) ────────
        print(f"\n─── Phase 3: Fetching OHLCV bars (global batch async) ── {len(contracts_by_exp)} expirations")
        
        tasks = []
        for exp, contracts in contracts_by_exp.items():
            relevant_dates = [
                td
                for td, candidate_exp, contract_type in contract_universes
                if candidate_exp == exp and contract_type == "put"
            ]
            if not relevant_dates:
                continue
            fetch_from = min(relevant_dates)
            fetch_to = min(exp, end_date)

            tasks.append(client.fetch_chain_ohlcv_batch(
                contracts, fetch_from, fetch_to,
                underlying=underlying, contract_type="put", expiration=exp,
            ))

        if call_side_enabled:
            for exp, contracts in call_contracts_by_exp.items():
                relevant_dates = [
                    td
                    for td, candidate_exp, contract_type in contract_universes
                    if candidate_exp == exp and contract_type == "call"
                ]
                if not relevant_dates:
                    continue
                fetch_from = min(relevant_dates)
                fetch_to = min(exp, end_date)
                tasks.append(client.fetch_chain_ohlcv_batch(
                    contracts, fetch_from, fetch_to,
                    underlying=underlying, contract_type="call", expiration=exp,
                ))
            
        results = await asyncio.gather(*tasks)
        total_api_calls_p3 = sum(results)
        
        print(f"  Phase 3 complete: {total_api_calls_p3} API calls | {client.cache_hits} cache hits")

        # ── Phase 3.5: Precompute delta chains across CPU cores ─────
        print("\n─── Phase 3.5: Precomputing delta chains (process pool) ───")
        precomputed_chain_data: Dict[Tuple[str, str], list] = {}
        precomputed_delta_chains: Dict[Tuple[str, str], list] = {}
        precomputed_call_chain_data: Dict[Tuple[str, str], list] = {}
        precomputed_call_delta_chains: Dict[Tuple[str, str], list] = {}
        delta_payloads = []

        for td, candidates in trade_plan:
            if td not in underlying_prices.index:
                continue

            current_regime = regimes.get(td, -1) if regimes else -1
            current_regime_name = result.regime_labels.get(current_regime, "")
            is_panic = "Panic / Crisis" in current_regime_name
            eff_target_dte = target_dte if not (dynamic_delta_variant and is_panic) else panic_dte_target
            possible_exps = find_target_expiration_friday(td, eff_target_dte)

            selected_exp = None
            selected_chain = None
            for exp in possible_exps:
                if (td, exp, "put") not in contract_universes:
                    continue
                cache_key = (td, exp)
                chain_data = _point_in_time_chain(
                    cache,
                    contract_universes,
                    underlying=underlying,
                    expiration=exp,
                    contract_type="put",
                    trade_date=td,
                    decision_time=decision_times[td],
                )
                precomputed_chain_data[cache_key] = chain_data

                if chain_data and len(chain_data) >= 10:
                    selected_exp = exp
                    selected_chain = chain_data
                    break

            if rolling_enabled and roll_dte_multiplier != 1.0:
                roll_exps = find_target_expiration_friday(td, int(eff_target_dte * roll_dte_multiplier))
                for rexp in roll_exps:
                    if (td, rexp, "put") in contract_universes:
                        r_cache_key = (td, rexp)
                        if r_cache_key not in precomputed_chain_data:
                            precomputed_chain_data[r_cache_key] = _point_in_time_chain(
                                cache,
                                contract_universes,
                                underlying=underlying,
                                expiration=rexp,
                                contract_type="put",
                                trade_date=td,
                                decision_time=decision_times[td],
                            )
            if call_side_enabled and rolling_enabled and roll_dte_multiplier != 1.0:
                roll_exps = find_target_expiration_friday(td, int(eff_target_dte * roll_dte_multiplier))
                for rexp in roll_exps:
                    if (td, rexp, "call") in contract_universes:
                        r_cache_key = (td, rexp)
                        if r_cache_key not in precomputed_call_chain_data:
                            precomputed_call_chain_data[r_cache_key] = _point_in_time_chain(
                                cache,
                                contract_universes,
                                underlying=underlying,
                                expiration=rexp,
                                contract_type="call",
                                trade_date=td,
                                decision_time=decision_times[td],
                            )

            if not selected_exp or not selected_chain:
                continue

            exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
            trade_dt = datetime.strptime(td, "%Y-%m-%d")
            dte_days = (exp_dt - trade_dt).days
            dte_years = dte_days / 365.0

            strikes = [r["strike"] for r in selected_chain]
            close_prices = [r["close"] for r in selected_chain]
            valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]
            if len(valid) < 5:
                continue

            v_strikes, v_prices = zip(*valid)
            delta_payloads.append({
                "trade_date": td,
                "expiration": selected_exp,
                "underlying_price": float(underlying_prices.loc[td]),
                "strikes": list(v_strikes),
                "mid_prices": list(v_prices),
                "dte_years": float(dte_years),
                "risk_free_rate": float(risk_free_rates.get(td, risk_free_rate)),
                "dividend_yield": float(dividend_yield),
                "option_type": "put",
            })

            if (
                call_side_enabled
                and (td, selected_exp, "call") in contract_universes
            ):
                call_chain_data = _point_in_time_chain(
                    cache,
                    contract_universes,
                    underlying=underlying,
                    expiration=selected_exp,
                    contract_type="call",
                    trade_date=td,
                    decision_time=decision_times[td],
                )
                precomputed_call_chain_data[(td, selected_exp)] = call_chain_data
                if call_chain_data and len(call_chain_data) >= 10:
                    call_strikes = [r["strike"] for r in call_chain_data]
                    call_prices = [r["close"] for r in call_chain_data]
                    call_valid = [(s, p) for s, p in zip(call_strikes, call_prices) if p and p > 0]
                    if len(call_valid) >= 5:
                        v_call_strikes, v_call_prices = zip(*call_valid)
                        delta_payloads.append({
                            "trade_date": td,
                            "expiration": selected_exp,
                            "underlying_price": float(underlying_prices.loc[td]),
                            "strikes": list(v_call_strikes),
                            "mid_prices": list(v_call_prices),
                            "dte_years": float(dte_years),
                            "risk_free_rate": float(risk_free_rates.get(td, risk_free_rate)),
                            "dividend_yield": float(dividend_yield),
                            "option_type": "call",
                        })

        if delta_payloads:
            max_workers = max(1, os.cpu_count() or 1)
            worker_count = min(max_workers, len(delta_payloads))
            print(f"  Precomputing {len(delta_payloads)} chain/date delta sets using {worker_count} CPU workers")
            failures = 0
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = lambda x, **kwargs: x

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_compute_delta_chain_worker, payload) for payload in delta_payloads]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Delta Precompute", smoothing=0.1):
                    td, exp, option_type, delta_chain, error = future.result()
                    if error:
                        failures += 1
                        print(f"  WARNING: Delta precompute failed for {td} {exp}: {error}")
                        continue
                    if option_type == "call":
                        precomputed_call_delta_chains[(td, exp)] = delta_chain
                    else:
                        precomputed_delta_chains[(td, exp)] = delta_chain
            print(f"  Phase 3.5 complete: {len(precomputed_delta_chains)} ready | {failures} failed")
        else:
            print("  Phase 3.5 skipped: no valid chain/date pairs found")

        print("\n─── Phase 4: Executing strategy sequentially ───")
        print(f"  Debug: Phase 4 starting. contracts_by_exp keys: {sorted(contracts_by_exp.keys())}")
        current_cash = initial_capital
        realized_capital = initial_capital
        active_trades: List[SpreadTrade] = []
        call_side_vix_cooldown_start_idx: Optional[int] = None
        call_side_vix_last_seen_above = False
        sync_quote_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        latest_trade_tasks: Dict[str, asyncio.Task] = {}

        def _finalize_trade_close(
            trade: SpreadTrade,
            td: str,
            current_short_mid: float,
            current_long_mid: float,
            current_far_long_mid: float,
            current_debit: float,
            current_dte: int,
            exit_reason: str,
            short_quote: Optional[dict] = None,
            long_quote: Optional[dict] = None,
            far_long_quote: Optional[dict] = None,
            sync_quotes: Optional[list] = None,
        ) -> None:
            nonlocal current_cash, realized_capital, closed_today, daily_events
            trade.exit_date = td
            trade.short_exit_mid = current_short_mid
            trade.long_exit_mid = current_long_mid
            trade.far_long_exit_mid = current_far_long_mid
            trade.net_debit_close = current_debit
            trade.exit_fee = _option_trade_fee(
                trade.num_contracts,
                3 if trade.far_long_ticker else 2,
                fee_per_contract_per_side,
            )
            trade.total_fees = trade.entry_fee + trade.exit_fee
            gross_trade_pnl = (trade.net_credit - trade.net_debit_close) * 100 * trade.num_contracts
            net_trade_pnl = gross_trade_pnl - trade.total_fees
            trade.pnl_per_contract = net_trade_pnl / trade.num_contracts
            trade.pnl_per_share = trade.pnl_per_contract / 100
            trade.exit_dte = current_dte
            trade.status = "closed"
            trade.exit_reason = exit_reason

            current_cash -= trade.net_debit_close * 100 * trade.num_contracts + trade.exit_fee
            realized_capital += gross_trade_pnl - trade.exit_fee
            closed_today.append(trade)
            option_suffix = "c" if (trade.option_type or "put").lower() == "call" else "p"
            strikes_label = f"{trade.short_strike}/{trade.long_strike}"
            if trade.far_long_ticker:
                strikes_label = f"{trade.short_strike}/{trade.long_strike}/{trade.far_long_strike}"
            print(
                f"  {td}: CLOSE {trade.num_contracts}x {strikes_label}{option_suffix} "
                f"| PnL=${trade.pnl_per_contract * trade.num_contracts:,.0f}"
            )

            if path_logger:
                daily_events.append({
                    "type": "close",
                    "reason": exit_reason,
                    "trade": asdict(trade)
                })

        async def _resolve_trade_market_context(
            trade: SpreadTrade,
            td: str,
            spot: float,
            current_dte: int,
            dte_years: float,
            risk_free_rate: float,
            dividend_yield: float,
        ) -> Dict[str, Any]:
            short_ticker = trade.short_ticker
            long_ticker = trade.long_ticker
            far_long_ticker = trade.far_long_ticker

            short_cached = bars_map.get(short_ticker)
            long_cached = bars_map.get(long_ticker)
            far_long_cached = bars_map.get(far_long_ticker) if far_long_ticker else None

            short_quote = quotes_map.get(short_ticker)
            long_quote = quotes_map.get(long_ticker)
            far_long_quote = quotes_map.get(far_long_ticker) if far_long_ticker else None

            if isinstance(short_quote, Exception):
                short_quote = None
            if isinstance(long_quote, Exception):
                long_quote = None
            if isinstance(far_long_quote, Exception):
                far_long_quote = None

            abnormalities = []
            sync_quotes = None
            current_short_mid = 0.0
            current_long_mid = 0.0

            if short_quote and long_quote:
                current_short_mid = short_quote["mid"]
                current_long_mid = long_quote["mid"]
            else:
                sync_key = (short_ticker, long_ticker)
                sync_task = sync_quote_tasks.get(sync_key)
                if sync_task is None:
                    sync_task = asyncio.create_task(
                        client.fetch_synchronized_ohlcv(
                            short_ticker,
                            long_ticker,
                            td,
                            max_time_delta_minutes=max_time_delta_minutes,
                        )
                    )
                    sync_quote_tasks[sync_key] = sync_task
                sync_quotes = await sync_task
                if sync_quotes:
                    current_short_mid = sync_quotes[0]["mid"]
                    current_long_mid = sync_quotes[1]["mid"]
                    print(
                        f"  [SYNC FALLBACK] Found synchronized 1m bars for {td} "
                        f"(timestamp: {sync_quotes[0]['timestamp']})"
                    )
                else:
                    trade_task = latest_trade_tasks.get(short_ticker)
                    if trade_task is None:
                        trade_task = asyncio.create_task(client.fetch_latest_trade(short_ticker, td))
                        latest_trade_tasks[short_ticker] = trade_task
                    short_trade = await trade_task
                    if short_trade:
                        short_p = short_trade["price"]
                        theo_long = await client.fetch_theoretical_price(
                            long_ticker,
                            short_ticker,
                            short_p,
                            spot,
                            td,
                            dte_years,
                            risk_free_rate=risk_free_rate,
                            dividend_yield=dividend_yield,
                        )
                        if theo_long is not None:
                            current_short_mid = short_p
                            current_long_mid = theo_long
                            print(f"  [THEO FALLBACK] Extrapolated {long_ticker} from {short_ticker} trade (${short_p}) on {td}")
                        else:
                            current_short_mid = short_p
                            current_long_mid = long_cached.get("close", 0) if long_cached else 0
                    else:
                        current_short_mid = short_cached.get("close", 0) if short_cached else 0
                        current_long_mid = long_cached.get("close", 0) if long_cached else 0

                    if not current_short_mid or not current_long_mid:
                        reason = "Missing NBBO Quote & No Sync 1m Bars & No Theo Fallback"
                        abnormalities.append({
                            "date": td,
                            "type": "MISSING_NBBO_QUOTE",
                            "reason": reason,
                            "short_ticker": short_ticker,
                            "long_ticker": long_ticker,
                            "fallback": "OHLCV Close (Unsynchronized)",
                        })
                        print(f"{CLR_RED}  [DATA GAP] {reason} on {td}. Falling back to unsynchronized OHLCV Close.{CLR_RST}")

                    if not current_short_mid or not current_long_mid:
                        reason_crit = "Total Data Gap (Quote + OHLCV + Theo)"
                        abnormalities.append({
                            "date": td,
                            "type": "CRITICAL_DATA_GAP",
                            "reason": reason_crit,
                            "short_ticker": short_ticker,
                            "long_ticker": long_ticker,
                            "fallback": "Entry Mid",
                        })
                        print(f"{CLR_RED}  [CRITICAL GAP] {reason_crit} on {td}. Falling back to Entry Mid.{CLR_RST}")
                        current_short_mid = trade.short_entry_mid
                        current_long_mid = trade.long_entry_mid

            if far_long_ticker:
                if far_long_quote:
                    current_far_long_mid = far_long_quote.get("mid", 0.0) or 0.0
                elif far_long_cached:
                    current_far_long_mid = far_long_cached.get("close", 0.0) or 0.0
                else:
                    current_far_long_mid = trade.far_long_entry_mid
            else:
                current_far_long_mid = 0.0

            if short_quote and long_quote:
                s_mid = current_short_mid
                s_bid = short_quote.get("bid", s_mid)
                s_ask = short_quote.get("ask", s_mid)
                l_mid = current_long_mid
                l_bid = long_quote.get("bid", l_mid)
                l_ask = long_quote.get("ask", l_mid)
                current_debit = _adjust_exit_debit(s_mid, s_bid, s_ask, l_mid, l_bid, l_ask, slippage_model)
            elif sync_quotes:
                s_mid = current_short_mid
                s_bid = sync_quotes[0].get("bid", s_mid)
                s_ask = sync_quotes[0].get("ask", s_mid)
                l_mid = current_long_mid
                l_bid = sync_quotes[1].get("bid", l_mid)
                l_ask = sync_quotes[1].get("ask", l_mid)
                current_debit = _adjust_exit_debit(s_mid, s_bid, s_ask, l_mid, l_bid, l_ask, slippage_model)
            else:
                current_debit = current_short_mid - current_long_mid
            if far_long_ticker:
                current_debit -= current_far_long_mid

            gain_pct = (trade.net_credit - current_debit) / trade.net_credit if trade.net_credit > 0 else 0
            current_trade_pnl_dollars = (trade.net_credit - current_debit) * 100 * trade.num_contracts
            pair_key = (trade.entry_date, trade.expiration)

            return {
                "trade": trade,
                "short_ticker": short_ticker,
                "long_ticker": long_ticker,
                "far_long_ticker": far_long_ticker,
                "short_cached": short_cached,
                "long_cached": long_cached,
                "far_long_cached": far_long_cached,
                "short_quote": short_quote,
                "long_quote": long_quote,
                "far_long_quote": far_long_quote,
                "sync_quotes": sync_quotes,
                "current_short_mid": current_short_mid,
                "current_long_mid": current_long_mid,
                "current_far_long_mid": current_far_long_mid,
                "current_debit": current_debit,
                "gain_pct": gain_pct,
                "current_trade_pnl_dollars": current_trade_pnl_dollars,
                "pair_key": pair_key,
                "current_dte": current_dte,
                "abnormalities": abnormalities,
            }

        try:
            from tqdm import tqdm
            iter_plan = tqdm(trade_plan, desc="Daily Backtest Loop", smoothing=0.1)
        except ImportError:
            iter_plan = trade_plan

        for td, candidates in iter_plan:
            if td not in underlying_prices.index:
                continue
            if td == trading_dates[0] or td == trading_dates[-1] or "2026-02" in td:
                print(f"  Debug: Processing {td}")
            spot = underlying_prices.loc[td]
            # Use dynamic risk-free rate for the day
            risk_free_rate = float(risk_free_rates.get(td, risk_free_rate))
            daily_events = []
            
            # 0. Track Regime (for plotting only, not used in strategy logic)
            current_regime = regimes.get(td, -1) if regimes else -1
            current_regime_name = result.regime_labels.get(current_regime, "")
            result.regime_history.append((td, current_regime))
            current_regime_v2 = annotation_for_session(
                regime_v2_annotations,
                td,
            )
            current_regime_v2_payload = (
                current_regime_v2.to_envelope()
                if current_regime_v2 is not None
                else None
            )
            result.regime_v2_history.append(
                (td, current_regime_v2_payload)
            )
            td_idx = trading_date_index.get(td, -1)
            prev_regime = regimes.get(trading_dates[td_idx - 1], -1) if regimes and td_idx > 0 else -1
            regime_switched_to_panic = (
                panic_transition_roll_enabled
                and current_regime == panic_transition_to_state
                and prev_regime in panic_transition_from_states
            )
            
            # Retrieve lagged panic probability (State 2) from prev trading day to avoid look-ahead bias
            lagged_panic_prob = 0.0
            if td_idx > 0 and result.regime_probabilities is not None:
                prev_td = trading_dates[td_idx - 1]
                if prev_td in result.regime_probabilities.index:
                    lagged_panic_prob = float(result.regime_probabilities.loc[prev_td, "prob_state_2"])
            
            # Update NLV placeholder (will be finalized at end of daily loop)
            current_nlv = current_cash 
            closed_today = []
            replacement_requests = []  # Unified list for rolls and panic swap replacements
            pending_profit_exit_candidates = []
            still_active = []

            # 1. Update active trades (Close if needed)
            daily_total_unrealized = 0.0
            
            # Pre-fetch all needed quotes for active trades in parallel
            unique_tickers = set()
            for trade in active_trades:
                unique_tickers.add(trade.short_ticker)
                unique_tickers.add(trade.long_ticker)
                if trade.far_long_ticker:
                    unique_tickers.add(trade.far_long_ticker)
            
            # Pre-fetch all needed quotes AND bars for active trades sequentially/parallelized to prevent race conditions on cache
            quotes_map = {}
            bars_map = {}
            if unique_tickers:
                print(f"  {td}: [FETCH] Pre-fetching data for {len(unique_tickers)} tickers...", flush=True)
                
                # Pre-load SQLite cache records into memory for this date
                cache.set_daily_memory_cache(td, list(unique_tickers))
                
                # Phase A: Fetch all daily bars first (and write them to the cache)
                bar_tasks = {}
                for ticker in unique_tickers:
                    cached = cache.get_ohlcv(ticker, td)
                    if cached:
                        bars_map[ticker] = cached
                    else:
                        bar_tasks[ticker] = client.fetch_contract_daily_bars(ticker, td, td)
                
                if bar_tasks:
                    bar_results = await asyncio.gather(*bar_tasks.values(), return_exceptions=True)
                    for ticker, res in zip(bar_tasks.keys(), bar_results):
                        if res and isinstance(res, list) and len(res) > 0:
                            bars_map[ticker] = res[0]
                        else:
                            bars_map[ticker] = None
                    
                    # Update memory cache to include newly fetched bars
                    cache.set_daily_memory_cache(td, list(unique_tickers))
                
                # Phase B: Now fetch EOD quotes (which can hit the newly cached daily bars instantaneously!)
                quote_tasks = {ticker: client.fetch_eod_quote(ticker, td) for ticker in unique_tickers}
                quote_results = await asyncio.gather(*quote_tasks.values(), return_exceptions=True)
                quotes_map = dict(zip(quote_tasks.keys(), quote_results))
                
            print(f"  {td}: [FETCH] Data pre-fetched.", flush=True)

            sync_quote_tasks = {}
            latest_trade_tasks = {}
            curr_dt = datetime.strptime(td, "%Y-%m-%d")
            trade_context_specs = []
            for trade in active_trades:
                exp_dt = trade.expiration_dt
                current_dte = (exp_dt - curr_dt).days
                dte_years = max(current_dte, 1) / 365.0
                trade_context_specs.append((trade, current_dte, dte_years))

            trade_context_tasks = [
                _resolve_trade_market_context(
                    trade=trade,
                    td=td,
                    spot=spot,
                    current_dte=current_dte,
                    dte_years=dte_years,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                )
                for trade, current_dte, dte_years in trade_context_specs
            ]
            trade_context_results = (
                await asyncio.gather(*trade_context_tasks, return_exceptions=True)
                if trade_context_tasks
                else []
            )

            for (trade, current_dte, dte_years), trade_context in zip(trade_context_specs, trade_context_results):
                exit_triggered = False
                exit_reason = ""
                profit_exit_candidate = False
                profit_exit_snapshot = None

                if isinstance(trade_context, Exception):
                    print(f"  {td}: [CONTEXT ERROR] {trade_context}")
                    trade_context = {
                        "trade": trade,
                        "short_ticker": trade.short_ticker,
                        "long_ticker": trade.long_ticker,
                        "far_long_ticker": trade.far_long_ticker,
                        "short_cached": bars_map.get(trade.short_ticker),
                        "long_cached": bars_map.get(trade.long_ticker),
                        "far_long_cached": bars_map.get(trade.far_long_ticker) if trade.far_long_ticker else None,
                        "short_quote": quotes_map.get(trade.short_ticker),
                        "long_quote": quotes_map.get(trade.long_ticker),
                        "far_long_quote": quotes_map.get(trade.far_long_ticker) if trade.far_long_ticker else None,
                        "sync_quotes": None,
                        "current_short_mid": trade.short_entry_mid,
                        "current_long_mid": trade.long_entry_mid,
                        "current_far_long_mid": trade.far_long_entry_mid if trade.far_long_ticker else 0.0,
                        "current_debit": trade.short_entry_mid - trade.long_entry_mid - (trade.far_long_entry_mid if trade.far_long_ticker else 0.0),
                        "gain_pct": 0.0,
                        "current_trade_pnl_dollars": 0.0,
                        "pair_key": (trade.entry_date, trade.expiration),
                        "abnormalities": [{
                            "date": td,
                            "type": "CRITICAL_DATA_GAP",
                            "reason": f"Trade context resolution failed: {trade_context}",
                            "short_ticker": trade.short_ticker,
                            "long_ticker": trade.long_ticker,
                            "fallback": "Entry Mid",
                        }],
                    }

                short_ticker = trade_context["short_ticker"]
                long_ticker = trade_context["long_ticker"]
                far_long_ticker = trade_context["far_long_ticker"]
                short_cached = trade_context["short_cached"]
                long_cached = trade_context["long_cached"]
                far_long_cached = trade_context["far_long_cached"]
                short_quote = trade_context["short_quote"]
                long_quote = trade_context["long_quote"]
                far_long_quote = trade_context["far_long_quote"]
                sync_quotes = trade_context["sync_quotes"]
                current_short_mid = trade_context["current_short_mid"]
                current_long_mid = trade_context["current_long_mid"]
                current_far_long_mid = trade_context["current_far_long_mid"]
                current_debit = trade_context["current_debit"]
                gain_pct = trade_context["gain_pct"]
                current_trade_pnl_dollars = trade_context["current_trade_pnl_dollars"]
                pair_key = trade_context["pair_key"]

                for abnormality in trade_context.get("abnormalities", []):
                    result.abnormalities.append(abnormality)
                    daily_events.append({"type": "abnormality", **abnormality})
                    if abnormality.get("type") == "CRITICAL_DATA_GAP":
                        result.critical_gap_count += 1
                    else:
                        result.data_gap_count += 1

                if (
                    call_side_enabled
                    and (trade.option_type or "put").lower() == "call"
                    and pair_key in paired_call_exit_requests
                    and gain_pct >= 0
                ):
                    exit_triggered = True
                    exit_reason = "paired_put_exit"
                if not exit_triggered:
                    if panic_transition_roll_enabled and regime_switched_to_panic and 0 < current_dte < (trade.entry_dte * panic_transition_min_remaining_dte_fraction) and (trade.option_type or "put").lower() == "put":
                        roll_qty = trade.num_contracts
                        roll_width = trade.short_strike - trade.long_strike
                        print(
                            f"  {td}: [PANIC TRANSITION ROLL] "
                            f"{trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p -> "
                            f"roll to {roll_qty}x at {trade.entry_dte} DTE target, width=${roll_width:.0f}, "
                            f"target even credit=${current_debit:.2f}"
                        )
                        replacement_requests.append(
                            _build_roll_request(
                                trade=trade,
                                current_trade_pnl_dollars=current_trade_pnl_dollars,
                                reason="panic_transition_roll",
                                spot=spot,
                                roll_strike_behavior=roll_strike_behavior,
                                roll_dte_multiplier=1.0,
                                num_contracts=roll_qty,
                                spread_width_override=roll_width,
                                target_net_credit=current_debit,
                            )
                        )
                        exit_triggered = True
                        exit_reason = "panic_transition_roll"
                    elif getattr(trade, "is_roll", False):
                        # Rolled positions stay open until the chain has fully recovered.
                        if stop_when_chain_breakeven:
                            chain_start_pnl = float(getattr(trade, "roll_chain_realized_pnl_before_entry", 0.0))
                            chain_total_if_closed_now = chain_start_pnl + current_trade_pnl_dollars
                            if chain_total_if_closed_now >= 0:
                                print(
                                    f"  {td}: [ROLL CHAIN CLOSE] root={trade.root_trade_id or trade.trade_id} "
                                    f"depth={trade.roll_chain_depth} chain_pnl=${chain_total_if_closed_now:,.0f}"
                                )
                                exit_triggered = True
                                exit_reason = "roll_chain_breakeven"
                        elif short_cached and long_cached and current_debit < 0.50:
                            exit_triggered = True
                            exit_reason = "early_profit"
                    else:
                        if not trade.hold_to_expiration:
                            # Early profit exit (only if price data is available)
                            if short_cached and long_cached and gain_pct >= trade.profit_target:
                                if max_daily_profit_exit_margin_budget_multiple > 0:
                                    profit_exit_candidate = True
                                    profit_exit_snapshot = {
                                        "trade": trade,
                                        "current_short_mid": current_short_mid,
                                        "current_long_mid": current_long_mid,
                                        "current_far_long_mid": current_far_long_mid,
                                        "current_debit": current_debit,
                                        "current_dte": current_dte,
                                        "current_trade_pnl_dollars": current_trade_pnl_dollars,
                                    }
                                else:
                                    exit_triggered = True
                                    exit_reason = "early_profit"
                        
                            # Scheduled exit at <= close_dte (can be dynamically set to half of entry_dte)
                            effective_close_dte = close_dte
                            if isinstance(close_dte, str) and close_dte == "half_entry":
                                effective_close_dte = trade.entry_dte // 2

                            if effective_close_dte > 0 and current_dte <= effective_close_dte and not exit_triggered:
                                # Apply conditional bypass for losing trades if configured
                                skip_scheduled_exit = False
                                if conditional_half_dte_exit:
                                    if conditional_half_dte_exit == "profitable_only" and gain_pct < 0:
                                        skip_scheduled_exit = True
                                    elif conditional_half_dte_exit is True and gain_pct < 0:
                                        skip_scheduled_exit = True

                                if (trade.option_type or "put").lower() == "call":
                                    is_itm = spot >= trade.short_strike
                                else:
                                    is_itm = spot <= trade.short_strike
                                
                                if is_itm and (trade.option_type or "put").lower() == "put":
                                    if rolling_enabled:
                                        # ITM at <= close_dte: ROLL instead of closing for a loss
                                        import math
                                        roll_qty = max(1, math.ceil(trade.num_contracts / 2))
                                        roll_width = spread_width * roll_spread_width_multiplier
                                        
                                        print(f"  {td}: [ROLL REQUEST (ITM at DTE={current_dte})] "
                                              f"{trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p -> "
                                              f"Roll to {roll_qty}x with ATM Long Leg, width=${roll_width:.0f}")
                                        
                                        replacement_requests.append(
                                            _build_roll_request(
                                                trade=trade,
                                                current_trade_pnl_dollars=current_trade_pnl_dollars,
                                                reason="roll_itm",
                                                spot=spot,
                                                roll_strike_behavior=roll_strike_behavior,
                                                roll_dte_multiplier=roll_dte_multiplier,
                                                num_contracts=roll_qty,
                                                spread_width_override=roll_width,
                                            )
                                        )
                                        
                                        exit_triggered = True
                                        exit_reason = "roll_itm"
                                    else:
                                        # OTM/ITM close if rolling is disabled
                                        if not skip_scheduled_exit:
                                            exit_triggered = True
                                            exit_reason = "scheduled"
                                elif not skip_scheduled_exit:
                                    # OTM at <= close_dte: close regardless of P/L
                                    exit_triggered = True
                                    exit_reason = "scheduled"
                        
                        # Legacy Rolling Trigger: ITM and < 7 DTE (safety net)
                        if 0 < current_dte <= 7 and isinstance(close_dte, int) and close_dte > 0 and (trade.option_type or "put").lower() == "put" and spot <= trade.short_strike and not exit_triggered and rolling_enabled:
                            import math
                            roll_qty = max(1, math.ceil(trade.num_contracts / 2))
                            roll_width = spread_width * roll_spread_width_multiplier
                            
                            print(f"  {td}: [ROLL REQUEST (ITM EXPIRY SAFETY)] "
                                  f"{trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p -> "
                                  f"Roll to {roll_qty}x with ATM Long Leg, width=${roll_width:.0f}")
                            
                            replacement_requests.append(
                                _build_roll_request(
                                    trade=trade,
                                    current_trade_pnl_dollars=current_trade_pnl_dollars,
                                    reason="roll",
                                    spot=spot,
                                    roll_strike_behavior=roll_strike_behavior,
                                    roll_dte_multiplier=roll_dte_multiplier,
                                    num_contracts=roll_qty,
                                    spread_width_override=roll_width,
                                )
                            )
                            
                            exit_triggered = True
                            exit_reason = "roll"

                    # HMM Panic Exit Rule: Close if current DTE < half of entry DTE when lagged panic probability >= 50%
                    if panic_exit_enabled and not exit_triggered and lagged_panic_prob >= 0.50 and current_dte < (trade.entry_dte / 2.0):
                        exit_triggered = True
                        exit_reason = "panic_exit"

                # Expiration exit (always check)
                if current_dte == 0:
                    # On expiration day, the value is simply the intrinsic value.
                    # This prevents false "max loss" results due to missing EOD quotes.
                    if (trade.option_type or "put").lower() == "call":
                        intrinsic_short = max(0, spot - trade.short_strike)
                        intrinsic_long = max(0, spot - trade.long_strike)
                    else:
                        intrinsic_short = max(0, trade.short_strike - spot)
                        intrinsic_long = max(0, trade.long_strike - spot)
                    
                    current_short_mid = intrinsic_short
                    current_long_mid = intrinsic_long
                    current_debit = intrinsic_short - intrinsic_long
                    
                    if not exit_triggered:
                        exit_triggered = True
                        exit_reason = "expired"
                        if (
                            rolling_enabled
                            and (
                                (getattr(trade, "is_roll", False) and repeat_roll_itm_at_expiration)
                                or (not getattr(trade, "is_roll", False) and rolling_config.get("roll_at_expiration", False))
                            )
                            and (
                                spot >= trade.short_strike
                                if (trade.option_type or "put").lower() == "call"
                                else spot <= trade.short_strike
                            )
                        ):
                            import math
                            roll_qty = max(1, math.ceil(trade.num_contracts / 2))
                            if roll_strike_behavior == "follow_underlying":
                                roll_width = spread_width * roll_spread_width_multiplier
                            else:
                                roll_width = (trade.short_strike - trade.long_strike) * roll_spread_width_multiplier
                            print(
                                f"  {td}: [ROLL REQUEST (ITM at expiration)] "
                                f"{trade.num_contracts}x {trade.short_strike}/{trade.long_strike}p -> "
                                f"Roll to {roll_qty}x with ATM Long Leg, width=${roll_width:.0f}"
                            )
                            replacement_requests.append(
                                _build_roll_request(
                                    trade=trade,
                                    current_trade_pnl_dollars=current_trade_pnl_dollars,
                                    reason="roll_itm_expired",
                                    spot=spot,
                                    roll_strike_behavior=roll_strike_behavior,
                                    roll_dte_multiplier=roll_dte_multiplier,
                                    num_contracts=roll_qty,
                                    spread_width_override=roll_width,
                                )
                            )
                            exit_reason = "roll_itm_expired"
                    
                if exit_triggered:
                    if call_side_enabled and (trade.option_type or "put").lower() == "put":
                        paired_call = next(
                            (
                                t
                                for t in active_trades
                                if t.status == "open"
                                and (t.option_type or "put").lower() == "call"
                                and t.entry_date == trade.entry_date
                                and t.expiration == trade.expiration
                            ),
                            None,
                        )
                        if paired_call is not None:
                            paired_call_exit_requests[(paired_call.entry_date, paired_call.expiration)] = exit_reason
                    _finalize_trade_close(
                        trade=trade,
                        td=td,
                        current_short_mid=current_short_mid,
                        current_long_mid=current_long_mid,
                        current_far_long_mid=current_far_long_mid,
                        current_debit=current_debit,
                        current_dte=current_dte,
                        exit_reason=exit_reason,
                        short_quote=short_quote,
                        long_quote=long_quote,
                        far_long_quote=far_long_quote,
                        sync_quotes=sync_quotes,
                    )
                    if (trade.option_type or "put").lower() == "call":
                        paired_call_exit_requests.pop(pair_key, None)
                else:
                    if profit_exit_candidate and profit_exit_snapshot is not None:
                        pending_profit_exit_candidates.append(profit_exit_snapshot)
                    still_active.append(trade)
                    daily_total_unrealized += current_debit * 100 * trade.num_contracts
                    
                # Track for scatter plot (capture state regardless of exit today)
                if short_cached and long_cached:
                    itm = spot > trade.short_strike if (trade.option_type or "put").lower() == "call" else spot < trade.short_strike
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
            current_margin_usage = _portfolio_margin_requirement(active_trades)
            current_nlv = current_cash - daily_total_unrealized

            # --- VIX Lookback ---
            prev_vix = None
            if vix_prices is not None and not vix_prices.empty:
                vix_history = vix_prices[vix_prices.index < td]
                if not vix_history.empty:
                    prev_vix = float(vix_history.iloc[-1])

            call_side_vix_block_reason = None
            if call_side_enabled and call_side_vix_skip_above is not None and prev_vix is not None:
                vix_skip_above = float(call_side_vix_skip_above)
                td_idx = trading_date_index.get(td)
                if td_idx is not None:
                    if prev_vix > vix_skip_above:
                        call_side_vix_last_seen_above = True
                        call_side_vix_cooldown_start_idx = None
                        call_side_vix_block_reason = (
                            f"VIX {prev_vix:.2f} above {vix_skip_above:.2f}"
                        )
                    elif call_side_vix_last_seen_above:
                        if call_side_vix_cooldown_start_idx is None:
                            call_side_vix_cooldown_start_idx = td_idx
                        cooldown_elapsed = td_idx - call_side_vix_cooldown_start_idx
                        if cooldown_elapsed < call_side_vix_reentry_delay_days:
                            remaining_days = call_side_vix_reentry_delay_days - cooldown_elapsed
                            call_side_vix_block_reason = (
                                f"VIX cooldown: {remaining_days} trading days remaining "
                                f"after falling back below {vix_skip_above:.2f}"
                            )

            # --- SIZING CALCULATION ---
            eff_margin_limit_pct = margin_limit_pct
            if dynamic_margin_method == "vix_scaled" and prev_vix is not None:
                # Simplify step function: 20% for VIX < 25, 25% for VIX >= 25
                eff_margin_limit_pct = 0.20 if prev_vix < 25.0 else 0.25
            elif dynamic_margin_method == "vix_scaled_15_25" and prev_vix is not None:
                # Variant 4: 15% for VIX < 25, 25% for VIX >= 25
                eff_margin_limit_pct = 0.15 if prev_vix < 25.0 else 0.25

            total_margin_budget = realized_capital * eff_margin_limit_pct
            sizing_config = (strategy_config or {}).get("sizing", {})
            dynamic_sizing = sizing_config.get("dynamic_sizing", False)
            if not dynamic_sizing:
                # Also check top-level for backward compatibility or simple overrides.
                dynamic_sizing = (strategy_config or {}).get("dynamic_sizing", False)
            catch_up_utilization_trigger = float(sizing_config.get("catch_up_utilization_trigger", 0.0) or 0.0)
            catch_up_utilization_target = float(sizing_config.get("catch_up_utilization_target", 0.0) or 0.0)
            catch_up_max_daily_slots = float(sizing_config.get("catch_up_max_daily_slots", 1.0) or 1.0)
            catch_up_min_profit_exit_slots = float(sizing_config.get("catch_up_min_profit_exit_slots", 1.0) or 1.0)
            new_entry_margin_budget = total_margin_budget
            if daily_pacing_slots > 0:
                base_daily_margin_budget = total_margin_budget / daily_pacing_slots
                profit_exit_margin_released = sum(
                    float(t.margin_required)
                    for t in closed_today
                    if t.exit_reason == "early_profit"
                )
                if dynamic_sizing:
                    remaining_budget = total_margin_budget - current_margin_usage
                    daily_new_margin_budget = max(0, remaining_budget / daily_pacing_slots)
                else:
                    daily_new_margin_budget = base_daily_margin_budget

                if (
                    catch_up_utilization_trigger > 0
                    and catch_up_utilization_target > catch_up_utilization_trigger
                    and total_margin_budget > 0
                    and current_margin_usage / total_margin_budget < catch_up_utilization_trigger
                    and profit_exit_margin_released >= base_daily_margin_budget * catch_up_min_profit_exit_slots
                ):
                    target_margin = total_margin_budget * min(catch_up_utilization_target, 1.0)
                    catch_up_gap = max(0.0, target_margin - current_margin_usage)
                    catch_up_cap = base_daily_margin_budget * max(1.0, catch_up_max_daily_slots)
                    daily_new_margin_budget = max(
                        daily_new_margin_budget,
                        min(catch_up_gap, catch_up_cap),
                    )

                new_entry_margin_budget = daily_new_margin_budget
                daily_allowed_margin = min(current_margin_usage + daily_new_margin_budget, total_margin_budget)
            else:
                daily_allowed_margin = total_margin_budget

            if pending_profit_exit_candidates and max_daily_profit_exit_margin_budget_multiple > 0 and daily_pacing_slots > 0:
                selected_candidates = []
                daily_exit_budget = (
                    max_daily_profit_exit_margin_budget_multiple
                    * (total_margin_budget / daily_pacing_slots)
                )
                candidate_margins = [
                    float(candidate["trade"].margin_required)
                    for candidate in pending_profit_exit_candidates
                    if float(candidate["trade"].margin_required) > 0
                ]
                representative_margin = (
                    float(sum(candidate_margins) / len(candidate_margins))
                    if candidate_margins
                    else 0.0
                )
                max_profit_exit_count = (
                    int(daily_exit_budget // representative_margin)
                    if representative_margin > 0 and daily_exit_budget > 0
                    else 0
                )
                selected_margin = 0.0
                for candidate in sorted(
                    pending_profit_exit_candidates,
                    key=lambda item: item["current_trade_pnl_dollars"],
                    reverse=True,
                ):
                    if len(selected_candidates) >= max_profit_exit_count:
                        break
                    candidate_margin = float(candidate["trade"].margin_required)
                    if selected_margin + candidate_margin > daily_exit_budget + 1e-6:
                        continue
                    selected_candidates.append(candidate)
                    selected_margin += candidate_margin

                if selected_candidates:
                    selected_ids = {id(item["trade"]) for item in selected_candidates}
                    still_active = [t for t in still_active if id(t) not in selected_ids]
                    for candidate in selected_candidates:
                        daily_total_unrealized -= float(candidate["current_debit"]) * 100.0 * int(candidate["trade"].num_contracts)
                        _finalize_trade_close(
                            trade=candidate["trade"],
                            td=td,
                            current_short_mid=candidate["current_short_mid"],
                            current_long_mid=candidate["current_long_mid"],
                            current_far_long_mid=candidate["current_far_long_mid"],
                            current_debit=candidate["current_debit"],
                            current_dte=candidate["current_dte"],
                            exit_reason="early_profit",
                        )
                print(
                    f"  {td}: [EXIT CAP] Closed {len(selected_candidates)} high-PnL profit candidates "
                    f"using ${selected_margin:,.0f} of ${daily_exit_budget:,.0f} budget "
                    f"(rep margin ${representative_margin:,.0f} -> max {max_profit_exit_count} exits)"
                )
                selected_ids = {id(item["trade"]) for item in selected_candidates}
                pending_profit_exit_candidates = [
                    c for c in pending_profit_exit_candidates if id(c["trade"]) not in selected_ids
                ]

                current_margin_usage = _portfolio_margin_requirement(still_active)
                current_nlv = current_cash - daily_total_unrealized
                daily_allowed_margin = min(current_margin_usage + new_entry_margin_budget, total_margin_budget)
                active_trades = still_active
            
            # (History recording moved to the end of the day to capture entries)
            
            # 3. Entry Phase: Process replacements and scheduled entries
            # We use the realized-capital margin budget after same-day exits.
            allowed_margin = daily_allowed_margin
            
            # --- ENTRY PARAMETERS ---
            eff_short_delta = target_short_delta
            eff_target_strike = None
            
            if dynamic_delta_method == "vix_scaled" and vix_delta_scale_factor and prev_vix is not None:
                min_delta = vix_delta_scale_factor.get("min_delta", target_short_delta)
                max_delta = vix_delta_scale_factor.get("max_delta", -0.20)
                vix_min = vix_delta_scale_factor.get("vix_min", 15.0)
                vix_max = vix_delta_scale_factor.get("vix_max", 35.0)
                if prev_vix <= vix_min:
                    eff_short_delta = min_delta
                elif prev_vix >= vix_max:
                    eff_short_delta = max_delta
                else:
                    ratio = (prev_vix - vix_min) / (vix_max - vix_min)
                    eff_short_delta = min_delta + ratio * (max_delta - min_delta)

            elif dynamic_delta_method == "vix_double_above_25" and prev_vix is not None:
                if prev_vix > 25.0:
                    eff_short_delta = target_short_delta * 2.0
                else:
                    eff_short_delta = target_short_delta

            elif regime_dynamic_delta:
                # Legacy compatibility only. V2 never enters this path.
                is_crisis = current_regime_name and any(
                    term in current_regime_name
                    for term in ["Panic", "Crisis", "Turmoil"]
                )
                eff_short_delta = -0.20 if is_crisis else -0.10
            eff_spread_width = spread_width
            if (
                dynamic_delta_method == "vix_double_above_25"
                and prev_vix is not None
                and prev_vix > 25.0
            ):
                eff_spread_width = spread_width * 2.0
            eff_qty_multiplier = 1.0
            eff_profit_target = early_profit_pct

            # Loop to allow multiple entries if we just performed a panic swap or have rolls
            # prioritize replacement_requests, then the scheduled entry
            entries_to_attempt = len(replacement_requests) + 1
            daily_delta_cache = {} # (expiration) -> delta_chain
            daily_chain_cache = {} # (expiration) -> chain_data
            daily_call_delta_cache = {} # (expiration) -> call delta_chain
            daily_call_chain_cache = {} # (expiration) -> call chain_data
            paired_call_exit_requests: Dict[Tuple[str, str], str] = {}
            
            opened_today = False
            last_dte = 0
            last_spread = 0.0
            last_short_mid = 0.0
            last_long_mid = 0.0
            
            for i in range(entries_to_attempt):
                current_margin_usage = _portfolio_margin_requirement(active_trades)
                is_roll = i < len(replacement_requests)  # Rolls come first in the queue
                req = replacement_requests[i] if is_roll else None
                
                # Rolls bypass the daily pacing gate because they replace existing risk.
                # The hard cap is still checked below, with a narrow margin-neutral waiver
                # for the small overages caused by ceil(qty / 2), doubled width, and strike availability.
                if not is_roll:
                    can_enter = current_margin_usage < allowed_margin
                    if not can_enter:
                        break
                
                if entries_to_attempt > 1:
                    label = "ROLL" if is_roll else "NEW"
                    print(f"  {td}: [ENTRY] Attempt {i+1}/{entries_to_attempt} ({label} | Margin: ${current_margin_usage:,.0f}/${allowed_margin:,.0f})...", flush=True)

                eff_target_dte = (
                    int(req["target_dte_override"])
                    if is_roll and req and req.get("target_dte_override") is not None
                    else target_dte
                )
                selected_exp = None
                chain_data = None
                
                possible_exps = find_target_expiration_friday(td, eff_target_dte)
                for exp in possible_exps:
                    if (td, exp, "put") in contract_universes:
                        if exp in daily_chain_cache:
                            chain_data = daily_chain_cache[exp]
                        else:
                            precomputed_key = (td, exp)
                            if precomputed_key in precomputed_chain_data:
                                chain_data = precomputed_chain_data[precomputed_key]
                            else:
                                chain_data = _point_in_time_chain(
                                    cache,
                                    contract_universes,
                                    underlying=underlying,
                                    expiration=exp,
                                    contract_type="put",
                                    trade_date=td,
                                    decision_time=decision_times[td],
                                )
                            daily_chain_cache[exp] = chain_data
                        if not chain_data:
                            print(f"  {td}: Skipping {exp} (empty)")
                            continue

                        strikes = [r["strike"] for r in chain_data]
                        close_prices = [r["close"] for r in chain_data]
                        valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]

                        if len(valid) < 10:
                            print(f"  {td}: Skipping {exp} (only {len(valid)} priced rows)")
                            continue

                        if not _is_third_friday(exp) and len(valid) < weekly_sparse_min_valid_rows:
                            print(
                                f"  {td}: Skipping {exp} (sparse weekly chain: "
                                f"only {len(valid)} priced rows)"
                            )
                            continue

                        selected_exp = exp
                        break
                
                if not selected_exp:
                    print(f"  {td}: [ENTRY FAILED] No valid expiration found for Attempt {i+1}")
                    continue

                if (
                    is_roll
                    and prefer_monthly_on_deviation
                    and not _is_third_friday(selected_exp)
                ):
                    target_entry_spread_width = (
                        float(req.get("spread_width_override"))
                        if req and req.get("spread_width_override")
                        else spread_width * roll_spread_width_multiplier
                    )
                    roll_needs_deviation = _roll_setup_requires_deviation(
                        chain_data,
                        contracts_by_exp,
                        selected_exp,
                        td,
                        spot,
                        target_entry_spread_width,
                        risk_free_rate,
                        dividend_yield,
                        spread_width_tolerance_pct,
                        min_credit,
                    )
                    if roll_needs_deviation:
                        alt_exp, alt_chain = _pick_monthly_alternative(
                            possible_exps,
                            contract_universes,
                            daily_chain_cache,
                            precomputed_chain_data,
                            cache,
                            underlying,
                            td,
                            decision_times[td],
                            min_chain_strikes,
                        )
                        if alt_exp and alt_chain:
                            print(
                                f"  {td}: [MONTHLY PREFERENCE] Weekly expiration {selected_exp} would require strike/pricing fallback; "
                                f"switching to monthly expiration {alt_exp}."
                            )
                            selected_exp = alt_exp
                            chain_data = alt_chain
                
                if selected_exp:
                    selection_retry = True
                    while selection_retry and selected_exp:
                        exp_dt = datetime.strptime(selected_exp, "%Y-%m-%d")
                        trade_dt = datetime.strptime(td, "%Y-%m-%d")
                        dte_days = (exp_dt - trade_dt).days
                        dte_years = dte_days / 365.0
                        eff_target_strike = None

                        if dynamic_delta_method == "regime_assignment_prob":
                            from live_trading.ev_engine import build_regime_return_arrays, fit_gmm, query_gmm
                            import scipy.optimize as opt

                            if not hasattr(result, "gmm_cache"):
                                result.gmm_cache = {}

                            cache_key = (td, current_regime, selected_exp)
                            if cache_key not in result.gmm_cache:
                                regime_dict, _, _ = build_regime_return_arrays(
                                    "dummy",
                                    horizon=dte_days,
                                    n_components=3,
                                    as_of_date=td,
                                )
                                bucket = regime_dict.get(f"State_{current_regime}", np.array([0.0]))
                                if len(bucket) >= 10:
                                    gmm_model = fit_gmm(bucket, regime_label=f"State_{current_regime}")
                                else:
                                    gmm_model = None
                                result.gmm_cache[cache_key] = gmm_model

                            gmm_model = result.gmm_cache[cache_key]
                            if gmm_model is not None:
                                def obj(r):
                                    return query_gmm(gmm_model, 1.0, np.exp(r)) - target_assignment_prob

                                try:
                                    target_log_return = opt.brentq(obj, -1.0, 0.5)
                                    eff_target_strike = spot * np.exp(target_log_return)
                                except Exception:
                                    eff_short_delta = target_short_delta
                        elif dynamic_delta_method == "forward_assignment_prob":
                            eff_target_strike, strike_diag = _forward_return_assignment_strike(
                                underlying_prices=underlying_prices,
                                trading_dates=trading_dates,
                                trading_date_index=trading_date_index,
                                as_of_date=td,
                                current_spot=spot,
                                option_horizon_days=dte_days,
                                target_assignment_prob=target_assignment_prob,
                                contract_type="put",
                            )
                            if eff_target_strike is None:
                                print(
                                    f"  {td}: [ASSIGNMENT PROB FALLBACK] "
                                    f"{strike_diag.get('reason', 'unknown')} -> reverting to delta target."
                                )
                                eff_short_delta = target_short_delta

                        strikes = [r["strike"] for r in chain_data]
                        close_prices = [r["close"] for r in chain_data]
                        valid = [(s, p) for s, p in zip(strikes, close_prices) if p and p > 0]

                        if len(valid) < 5:
                            break

                        if selected_exp not in daily_delta_cache:
                            precomputed_key = (td, selected_exp)
                            if precomputed_key in precomputed_delta_chains:
                                daily_delta_cache[selected_exp] = precomputed_delta_chains[precomputed_key]
                            else:
                                v_strikes, v_prices = zip(*valid)
                                daily_delta_cache[selected_exp] = compute_chain_deltas(
                                    spot,
                                    list(v_strikes),
                                    list(v_prices),
                                    dte_years,
                                    risk_free_rate,
                                    dividend_yield,
                                    "put",
                                )

                        delta_chain = daily_delta_cache[selected_exp]

                        if not delta_chain:
                            break

                        # Handle target selection (replacement vs scheduled)
                        req = None
                        if i < len(replacement_requests):
                            req = replacement_requests[i]

                        entry_spread_width = eff_spread_width
                        if req and req.get("spread_width_override"):
                            entry_spread_width = req["spread_width_override"]
                        target_entry_spread_width = entry_spread_width

                        if req and req.get("target_net_credit") is not None:
                            even_roll_selection = _select_put_roll_strikes_for_target_credit(
                                delta_chain,
                                chain_data,
                                target_entry_spread_width,
                                float(req.get("target_net_credit") or 0.0),
                            )
                            if not even_roll_selection:
                                print(
                                    f"  {td}: [ENTRY FAILED] Could not select even-credit roll strikes for Attempt {i+1}"
                                )
                                break
                            short_idx, short_strike, long_strike, actual_spread_width, target_roll_credit = even_roll_selection
                        elif req and req.get("target_long_strike"):
                            roll_selection = _select_put_roll_strikes_from_atm_long(
                                delta_chain,
                                req["target_long_strike"],
                                target_entry_spread_width,
                            )
                            if not roll_selection:
                                print(f"  {td}: [ENTRY FAILED] Could not select ATM-long roll strikes for Attempt {i+1}")
                                break
                            short_idx, short_strike, long_strike, actual_spread_width = roll_selection
                        elif req and req.get("target_strike"):
                            t_strike = req["target_strike"]
                            short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][0] - t_strike))
                            short_strike = delta_chain[short_idx][0]
                            long_strike = short_strike - target_entry_spread_width
                        elif not req and dynamic_delta_method == "atm_long_spread":
                            selection = _select_put_roll_strikes_from_atm_long(
                                delta_chain,
                                spot,
                                target_entry_spread_width,
                            )
                            if not selection:
                                print(f"  {td}: [ENTRY FAILED] Could not select ATM-long strikes for Attempt {i+1}")
                                break
                            short_idx, short_strike, long_strike, actual_spread_width = selection
                        elif not req and eff_target_strike is not None:
                            t_strike = eff_target_strike
                            short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][0] - t_strike))
                            short_strike = delta_chain[short_idx][0]
                            long_strike = short_strike - target_entry_spread_width
                        else:
                            t_delta = req["target_delta"] if req else eff_short_delta
                            short_idx = min(range(len(delta_chain)), key=lambda i: abs(delta_chain[i][1] - t_delta))
                            short_strike = delta_chain[short_idx][0]
                            long_strike = short_strike - target_entry_spread_width

                        short_delta = delta_chain[short_idx][1]

                        if (
                            prefer_monthly_on_deviation
                            and not _is_third_friday(selected_exp)
                            and not (req and (req.get("target_long_strike") or req.get("target_strike") or req.get("target_net_credit") is not None))
                            and not (not req and dynamic_delta_method == "atm_long_spread")
                            and not (not req and eff_target_strike is not None)
                        ):
                            target_delta = req["target_delta"] if req else eff_short_delta
                            delta_deviation_pct = _delta_deviation_pct(short_delta, target_delta)
                            width_deviation_pct = abs(short_strike - long_strike - target_entry_spread_width) / target_entry_spread_width
                            if delta_deviation_pct > delta_tolerance_pct or width_deviation_pct > 1e-9:
                                alt_exp, alt_chain = _pick_monthly_alternative(
                                    possible_exps,
                                    contract_universes,
                                    daily_chain_cache,
                                    precomputed_chain_data,
                                    cache,
                                    underlying,
                                    td,
                                    decision_times[td],
                                    min_chain_strikes,
                                )
                                if alt_exp and alt_chain:
                                    print(
                                        f"  {td}: [MONTHLY PREFERENCE] Weekly expiration {selected_exp} "
                                        f"missed target delta by {delta_deviation_pct:.0%}"
                                        f"{' and' if width_deviation_pct > 1e-9 else ''}"
                                        f"{' target width deviated' if width_deviation_pct > 1e-9 else ''}; switching to "
                                        f"monthly expiration {alt_exp}."
                                    )
                                    selected_exp = alt_exp
                                    chain_data = alt_chain
                                    continue

                        # Find long strike mid and handle search fallback
                        found_valid_strike_pair = False
                        base_short_idx = short_idx
                        base_short_strike = short_strike
                        base_long_strike = long_strike

                        # Let's perform a search for a valid strike pair starting from the target strikes
                        # We allow searching up to 15 strikes lower for rolls to find a positive credit spread
                        max_search_steps = 15 if is_roll else 1

                        for offset in range(max_search_steps):
                                if offset > 0:
                                    new_short_idx = base_short_idx - offset
                                    if new_short_idx < 0:
                                        break
                                    
                                    short_strike = delta_chain[new_short_idx][0]
                                    short_idx = new_short_idx
                                    long_strike = short_strike - target_entry_spread_width
                                    
                                    # Ensure long strike is within the chain
                                    if long_strike < delta_chain[0][0]:
                                        break
                                
                                short_delta = delta_chain[short_idx][1]
                                
                                # Find long strike mid
                                long_mid = next((_row_mark_price(r) for r in chain_data if abs(r["strike"] - long_strike) < 0.01 and _row_mark_price(r) > 0), 0.0)
                                if not long_mid:
                                    nearest_long = min(strikes, key=lambda s: abs(s - long_strike))
                                    long_mid = next((_row_mark_price(r) for r in chain_data if abs(r["strike"] - nearest_long) < 0.01 and _row_mark_price(r) > 0), 0.0)
                                    long_strike = nearest_long

                                actual_spread_width = short_strike - long_strike
                                if not _spread_width_within_tolerance(
                                    actual_spread_width,
                                    target_entry_spread_width,
                                    spread_width_tolerance_pct,
                                ):
                                    continue
                                
                                short_mid = delta_chain[short_idx][2] or 0.0
                                net_credit = short_mid - long_mid
                                
                                # NEW: Synchronize Entry Pricing
                                # Re-fetch both legs using 1-minute bars to ensure they are temporally aligned.
                                short_ticker = _find_ticker_for_strike(chain_data, short_strike, "put")
                                long_ticker = _find_ticker_for_strike(chain_data, long_strike, "put")
                                far_long_requested = (
                                    (not req)
                                    and far_long_leg_enabled
                                    and far_long_leg_vix_below is not None
                                    and prev_vix is not None
                                    and prev_vix < far_long_leg_vix_below
                                )
                                far_long_strike = 0.0
                                far_long_ticker = ""
                                far_long_mid = 0.0
                                far_long_entry_quote = None
                                if far_long_requested:
                                    far_long_strike = short_strike - (target_entry_spread_width * far_long_leg_multiplier)
                                    far_long_mid = next(
                                        (
                                            _row_mark_price(r)
                                            for r in chain_data
                                            if abs(r["strike"] - far_long_strike) < 0.01
                                            and _row_matches_contract_type(r, "put")
                                            and _row_mark_price(r) > 0
                                        ),
                                        0.0,
                                    )
                                    if not far_long_mid:
                                        nearest_far_long = min(strikes, key=lambda s: abs(s - far_long_strike))
                                        far_long_mid = next(
                                            (
                                                _row_mark_price(r)
                                                for r in chain_data
                                                if abs(r["strike"] - nearest_far_long) < 0.01
                                                and _row_matches_contract_type(r, "put")
                                                and _row_mark_price(r) > 0
                                            ),
                                            0.0,
                                        )
                                        far_long_strike = nearest_far_long
                                    if far_long_mid:
                                        far_long_ticker = _find_ticker_for_strike(chain_data, far_long_strike, "put") or ""
                                    if not far_long_ticker or not far_long_mid:
                                        far_long_requested = False
                                        far_long_ticker = ""
                                        far_long_strike = 0.0
                                        far_long_mid = 0.0
                                
                                if not short_ticker or not long_ticker:
                                    continue

                                quote_tasks = [
                                    client.fetch_eod_quote(short_ticker, td, allow_trade_fallback=not require_entry_nbbo),
                                    client.fetch_eod_quote(long_ticker, td, allow_trade_fallback=not require_entry_nbbo),
                                ]
                                if far_long_requested and far_long_ticker:
                                    quote_tasks.append(
                                        client.fetch_eod_quote(far_long_ticker, td, allow_trade_fallback=not require_entry_nbbo)
                                    )
                                quote_results = await asyncio.gather(*quote_tasks, return_exceptions=True)
                                short_entry_quote = quote_results[0]
                                long_entry_quote = quote_results[1]
                                far_long_entry_quote = quote_results[2] if len(quote_results) > 2 else None
                                if isinstance(short_entry_quote, Exception): short_entry_quote = None
                                if isinstance(long_entry_quote, Exception): long_entry_quote = None
                                if isinstance(far_long_entry_quote, Exception): far_long_entry_quote = None
                                sync_quotes = None
                                entry_priced = False
                                if short_entry_quote and long_entry_quote:
                                    short_mid = short_entry_quote["mid"]
                                    short_bid = short_entry_quote.get("bid", short_mid)
                                    short_ask = short_entry_quote.get("ask", short_mid)
                                    long_mid = long_entry_quote["mid"]
                                    long_bid = long_entry_quote.get("bid", long_mid)
                                    long_ask = long_entry_quote.get("ask", long_mid)
                                    net_credit = _adjust_entry_credit(short_mid, short_bid, short_ask, long_mid, long_bid, long_ask, slippage_model)
                                    entry_priced = True
                                    source_label = _pricing_source_label(short_entry_quote, long_entry_quote)
                                    if offset > 0:
                                        print(f"  [ENTRY {source_label} FALLBACK] {source_label} pricing found for fallback strike {short_strike}/{long_strike} on {td}: Short=${short_mid}, Long=${long_mid}, Net=${net_credit:.2f}")
                                    else:
                                        print(f"  [ENTRY {source_label}] {source_label} pricing found for {td}: Short=${short_mid}, Long=${long_mid}, Net=${net_credit:.2f}")
                                elif require_entry_nbbo:
                                    print(f"  {td}: [ENTRY FAILED] Missing strict NBBO quote for {short_strike}/{long_strike}p")
                                    continue
                                else:
                                    sync_quotes = await client.fetch_synchronized_ohlcv(short_ticker, long_ticker, td, max_time_delta_minutes=max_time_delta_minutes)
                                
                                if sync_quotes:
                                    short_mid = sync_quotes[0]["mid"]
                                    short_bid = sync_quotes[0].get("bid", short_mid)
                                    short_ask = sync_quotes[0].get("ask", short_mid)
                                    long_mid = sync_quotes[1]["mid"]
                                    long_bid = sync_quotes[1].get("bid", long_mid)
                                    long_ask = sync_quotes[1].get("ask", long_mid)
                                    net_credit = _adjust_entry_credit(short_mid, short_bid, short_ask, long_mid, long_bid, long_ask, slippage_model)
                                    entry_priced = True
                                    if offset > 0:
                                        print(f"  [ENTRY SYNC FALLBACK] Synchronized pricing found for fallback strike {short_strike}/{long_strike} on {td}: Short=${short_mid}, Long=${long_mid}, Net=${net_credit:.2f}")
                                    else:
                                        print(f"  [ENTRY SYNC] Synchronized pricing found for {td}: Short=${short_mid}, Long=${long_mid}, Net=${net_credit:.2f}")
                                if not entry_priced:
                                    # THEO FALLBACK for Entry
                                    short_trade = await client.fetch_latest_trade(short_ticker, td)
                                    if short_trade:
                                        short_p = short_trade["price"]
                                        theo_long = await client.fetch_theoretical_price(
                                            long_ticker, short_ticker, short_p, spot, td, dte_years, risk_free_rate=risk_free_rate
                                        )
                                        long_mid = theo_long if theo_long is not None else (long_cached.get("close", 0) if long_cached else 0)
                                        if not long_mid:
                                            print(
                                                f"  [ENTRY THEO FAILED] Unable to derive a fallback price for {long_ticker} "
                                                f"from {short_ticker} on {td}; skipping {short_strike}/{long_strike}."
                                            )
                                            continue
                                        short_mid = short_p
                                        net_credit = short_mid - long_mid
                                        if theo_long is not None:
                                            if offset > 0:
                                                print(f"  [ENTRY THEO FALLBACK] Extrapolated {long_ticker} from {short_ticker} trade (${short_p}) on {td} at fallback strike {short_strike}/{long_strike}: Net=${net_credit:.2f}")
                                            else:
                                                print(f"  [ENTRY THEO] Extrapolated {long_ticker} from {short_ticker} trade (${short_p}) on {td}: Net=${net_credit:.2f}")
                                        else:
                                            print(
                                                f"  [ENTRY THEO FALLBACK] Used {short_ticker} trade (${short_p}) and cached "
                                                f"close for {long_ticker} on {td}: Net=${net_credit:.2f}"
                                            )
                                    else:
                                        print(
                                            f"  [ENTRY THEO FAILED] No latest trade for {short_ticker} on {td}; "
                                            f"skipping {short_strike}/{long_strike}."
                                        )
                                        continue

                                if far_long_requested:
                                    if far_long_entry_quote:
                                        far_long_mid = far_long_entry_quote["mid"]
                                        if not entry_priced:
                                            entry_priced = True
                                        print(
                                            f"  [ENTRY FAR LONG] Added 2x-width far long leg {far_long_strike}p "
                                            f"for VIX {prev_vix:.2f}; mid=${far_long_mid:.2f}"
                                        )
                                    elif require_entry_nbbo:
                                        print(f"  {td}: [ENTRY FAILED] Missing strict NBBO quote for far long {far_long_strike}p")
                                        continue
                                    net_credit -= far_long_mid

                                if net_credit >= min_credit and net_credit < actual_spread_width:
                                    found_valid_strike_pair = True
                                    break
                                else:
                                    if is_roll:
                                        print(f"  {td}: [ROLL WARNING] Primary target strike {short_strike}/{long_strike} yielded negative or invalid credit (${net_credit:.2f}). Stepping down...")
                        if not found_valid_strike_pair:
                            if req:
                                print(f"  {td}: [ENTRY FAILED] All fallback strikes failed for ATM-long roll starting at {base_short_strike}/{base_long_strike}")
                                break
                            else:
                                print(f"  {td}: [ENTRY FAILED] Negative or zero credit for Attempt {i+1}")
                                break

                        if net_credit < min_credit:
                            print(
                                f"  {td}: [ENTRY FAILED] Credit (${net_credit:.4f}) below "
                                f"minimum (${min_credit:.4f}) for {short_strike}/{long_strike}p"
                            )
                            break

                        if net_credit >= (short_strike - long_strike):
                            print(f"  {td}: [ENTRY FAILED] Credit (${net_credit:.2f}) exceeds spread width (${short_strike - long_strike:.2f}) for {short_strike}/{long_strike}")
                            break

                        if net_credit > 0:
                            margin_per_lot = (short_strike - long_strike) * 100
                            if req:
                                num_contracts = req["num_contracts"]
                            elif backtest_qty > 0:
                                num_contracts = int(backtest_qty * eff_qty_multiplier) or 1
                            elif daily_pacing_slots > 0:
                                # Daily pacing: allocate only the daily margin budget for NEW entries
                                daily_budget = new_entry_margin_budget
                                remaining_allowed = max(0.0, allowed_margin - current_margin_usage)
                                num_contracts = int(min(daily_budget, remaining_allowed) // margin_per_lot)
                                
                                # Floor to 1 if budget limits it but total capital allows it
                                if num_contracts == 0 and margin_per_lot > daily_budget:
                                    total_remaining_room = total_margin_budget - current_margin_usage
                                    if total_remaining_room >= margin_per_lot:
                                        num_contracts = 1
                                        allowed_margin = max(allowed_margin, current_margin_usage + margin_per_lot)
                                        print(
                                            f"  {td}: [SIZING FLOOR] Floored contracts to 1 since "
                                            f"margin per lot (${margin_per_lot:,.0f}) exceeds daily budget "
                                            f"(${daily_budget:,.0f}) but total remaining room "
                                            f"(${total_remaining_room:,.0f}) allows it."
                                        )
                            else:
                                num_contracts = int(((allowed_margin - current_margin_usage) // margin_per_lot) * eff_qty_multiplier)

                            if num_contracts <= 0:
                                print(
                                    f"  {td}: [ENTRY SKIPPED] Sized to 0 contracts for "
                                    f"{short_strike}/{long_strike}p "
                                    f"(margin per lot ${margin_per_lot:,.0f}, remaining allowed "
                                    f"${max(0.0, allowed_margin - current_margin_usage):,.0f}, "
                                    f"daily budget ${daily_budget if daily_pacing_slots > 0 else allowed_margin:,.0f})"
                                )
                                break

                            if num_contracts > 0:
                                margin_required = margin_per_lot * num_contracts
                                entry_fee = _option_trade_fee(
                                    num_contracts,
                                    3 if far_long_requested else 2,
                                    fee_per_contract_per_side,
                                )
                                entry_margin_limit = total_margin_budget if req else allowed_margin
                                candidate_put_margin = SpreadTrade(
                                    entry_date=td,
                                    option_type="put",
                                    expiration=selected_exp,
                                    short_strike=short_strike,
                                    long_strike=long_strike,
                                    far_long_ticker=far_long_ticker if far_long_requested else "",
                                    far_long_strike=far_long_strike if far_long_requested else 0.0,
                                    num_contracts=num_contracts,
                                    margin_required=margin_required,
                                )
                                portfolio_margin_after_entry = _portfolio_margin_requirement(active_trades + [candidate_put_margin])
                                incremental_portfolio_margin = portfolio_margin_after_entry - current_margin_usage
                                if portfolio_margin_after_entry > entry_margin_limit + 1e-6:
                                    margin_limit_exceeded = True
                                    if req:
                                        req_reason = str(req.get("reason") or "")
                                        if req_reason.startswith("roll"):
                                            released_margin = float(req.get("released_margin") or 0.0)
                                            target_roll_margin = float(req.get("target_roll_margin") or 0.0)
                                            margin_neutral_allowance = max(released_margin, target_roll_margin)
                                            margin_neutral_allowance *= (1.0 + spread_width_tolerance_pct)
                                            if margin_required <= margin_neutral_allowance + 1e-6:
                                                margin_limit_exceeded = False
                                                print(
                                                    f"  {td}: [ROLL MARGIN WAIVER] Allowing margin-neutral roll "
                                                    f"{num_contracts}x {short_strike}/{long_strike}p "
                                                    f"(${portfolio_margin_after_entry:,.0f}/"
                                                    f"${entry_margin_limit:,.0f}; released ${released_margin:,.0f})"
                                                )

                                    if margin_limit_exceeded:
                                        limit_label = "hard margin cap" if req else "allowed daily margin"
                                        print(
                                            f"  {td}: [ENTRY FAILED] {limit_label} exceeded by "
                                            f"{num_contracts}x {short_strike}/{long_strike}p "
                                            f"(${portfolio_margin_after_entry:,.0f}/${entry_margin_limit:,.0f}; "
                                            f"incremental ${incremental_portfolio_margin:,.0f})"
                                        )
                                        break

                                trade = SpreadTrade(
                                    short_ticker=short_ticker,
                                    long_ticker=long_ticker,
                                    short_strike=short_strike,
                                    long_strike=long_strike,
                                    far_long_ticker=far_long_ticker if far_long_requested else "",
                                    far_long_strike=far_long_strike if far_long_requested else 0.0,
                                    expiration=selected_exp,
                                    entry_date=td,
                                    entry_price=spot,
                                    net_credit=net_credit,
                                    entry_fee=entry_fee,
                                    total_fees=entry_fee,
                                    num_contracts=num_contracts,
                                    margin_required=margin_required,
                                    status="open",
                                    entry_regime=current_regime,
                                    entry_regime_name=current_regime_name,
                                    entry_regime_v2=current_regime_v2_payload,
                                    entry_dte=dte_days,
                                    short_entry_mid=short_mid,
                                    long_entry_mid=long_mid,
                                    far_long_entry_mid=far_long_mid if far_long_requested else 0.0,
                                    profit_target=eff_profit_target,
                                    entry_delta=short_delta,
                                    is_roll=is_roll,
                                    trade_id=uuid.uuid4().hex[:8],
                                    root_trade_id=(req.get("root_trade_id") if req else None) or "",
                                    parent_trade_id=(req.get("parent_trade_id") if req else None) or "",
                                    roll_chain_depth=int(req.get("roll_chain_depth", 0)) if req else 0,
                                    roll_chain_realized_pnl_before_entry=float(
                                        req.get("roll_chain_realized_pnl_before_entry", 0.0)
                                    ) if req else 0.0,
                                )

                                if not trade.root_trade_id:
                                    trade.root_trade_id = trade.trade_id

                                active_trades.append(trade)
                                result.trades.append(trade)
                                current_cash += net_credit * 100 * num_contracts - entry_fee
                                realized_capital -= entry_fee

                                opened_today = True
                                last_dte = dte_days
                                short_row = next((r for r in chain_data if abs(r["strike"] - short_strike) < 0.01 and _row_matches_contract_type(r, "put")), {})
                                long_row = next((r for r in chain_data if abs(r["strike"] - long_strike) < 0.01 and _row_matches_contract_type(r, "put")), {})
                                s_spread = (short_row.get("ask", 0) - short_row.get("bid", 0)) if short_row.get("bid") and short_row.get("ask") else 0
                                l_spread = (long_row.get("ask", 0) - long_row.get("bid", 0)) if long_row.get("bid") and long_row.get("ask") else 0
                                last_spread = s_spread + l_spread
                                last_short_mid = short_mid
                                last_long_mid = long_mid
                                open_strikes = f"{short_strike}/{long_strike}"
                                if far_long_requested:
                                    open_strikes = f"{short_strike}/{long_strike}/{far_long_strike}"

                                print(f"  {td}: OPEN {num_contracts}x {open_strikes}p | Credit=${net_credit:.2f} | Fee=${entry_fee:.2f} | Margin=${trade.margin_required:,.0f}")
                                if is_roll:
                                    print(
                                        f"       chain root={trade.root_trade_id} parent={trade.parent_trade_id} "
                                        f"depth={trade.roll_chain_depth} chain_pnl_before_entry=${trade.roll_chain_realized_pnl_before_entry:,.0f}"
                                    )

                                if path_logger:
                                    daily_events.append({
                                        "type": "entry",
                                        "reason": req.get("reason") if req else "new",
                                        "expiration": selected_exp,
                                        "short_strike": short_strike,
                                        "long_strike": long_strike,
                                        "far_long_strike": far_long_strike if far_long_requested else None,
                                        "net_credit": float(net_credit),
                                        "entry_fee": float(entry_fee),
                                        "num_contracts": num_contracts,
                                        "trade_id": trade.trade_id,
                                        "root_trade_id": trade.root_trade_id,
                                        "parent_trade_id": trade.parent_trade_id,
                                        "roll_chain_depth": trade.roll_chain_depth,
                                        "roll_chain_realized_pnl_before_entry": trade.roll_chain_realized_pnl_before_entry,
                                    })

                                if call_side_enabled and not is_roll:
                                    if call_side_vix_block_reason is not None:
                                        print(f"  {td}: [CALL ENTRY SKIPPED] {call_side_vix_block_reason}")
                                        continue

                                    call_chain = daily_call_chain_cache.get(selected_exp)
                                    if call_chain is None:
                                        call_key = (td, selected_exp)
                                        if call_key in precomputed_call_chain_data:
                                            call_chain = precomputed_call_chain_data[call_key]
                                        else:
                                            call_chain = _point_in_time_chain(
                                                cache,
                                                contract_universes,
                                                underlying=underlying,
                                                expiration=selected_exp,
                                                contract_type="call",
                                                trade_date=td,
                                                decision_time=decision_times[td],
                                            )
                                        daily_call_chain_cache[selected_exp] = call_chain

                                    if not call_chain or len(call_chain) < min_chain_strikes:
                                        print(f"  {td}: [CALL ENTRY SKIPPED] No usable call chain for {selected_exp}")
                                    else:
                                        call_valid = [
                                            (r["strike"], _row_mark_price(r))
                                            for r in call_chain
                                            if _row_matches_contract_type(r, "call") and _row_mark_price(r) > 0
                                        ]
                                        if len(call_valid) < 5:
                                            print(f"  {td}: [CALL ENTRY SKIPPED] Too few priced call rows for {selected_exp}")
                                        else:
                                            if selected_exp not in daily_call_delta_cache:
                                                call_key = (td, selected_exp)
                                                if call_key in precomputed_call_delta_chains:
                                                    daily_call_delta_cache[selected_exp] = precomputed_call_delta_chains[call_key]
                                                else:
                                                    c_strikes, c_prices = zip(*call_valid)
                                                    daily_call_delta_cache[selected_exp] = compute_chain_deltas(
                                                        spot,
                                                        list(c_strikes),
                                                        list(c_prices),
                                                        dte_years,
                                                        risk_free_rate,
                                                        dividend_yield,
                                                        "call",
                                                    )

                                            call_delta_chain = daily_call_delta_cache.get(selected_exp, [])
                                            if call_delta_chain:
                                                call_short_idx = min(
                                                    range(len(call_delta_chain)),
                                                    key=lambda j: abs(call_delta_chain[j][1] - call_side_short_delta),
                                                )
                                                call_short_strike = call_delta_chain[call_short_idx][0]
                                                call_long_strike = call_short_strike + call_side_spread_width
                                                call_short_delta = call_delta_chain[call_short_idx][1]
                                                call_long_mid = next(
                                                    (
                                                        _row_mark_price(r)
                                                        for r in call_chain
                                                        if abs(r["strike"] - call_long_strike) < 0.01
                                                        and _row_matches_contract_type(r, "call")
                                                        and _row_mark_price(r) > 0
                                                    ),
                                                    0.0,
                                                )
                                                if not call_long_mid:
                                                    call_strikes = [
                                                        r["strike"]
                                                        for r in call_chain
                                                        if _row_matches_contract_type(r, "call")
                                                        and _row_mark_price(r) > 0
                                                    ]
                                                    if call_strikes:
                                                        nearest_call_long = min(call_strikes, key=lambda s: abs(s - call_long_strike))
                                                        call_long_mid = next(
                                                            (
                                                                _row_mark_price(r)
                                                                for r in call_chain
                                                                if abs(r["strike"] - nearest_call_long) < 0.01
                                                                and _row_matches_contract_type(r, "call")
                                                                and _row_mark_price(r) > 0
                                                            ),
                                                            0.0,
                                                        )
                                                        call_long_strike = nearest_call_long

                                                actual_call_width = call_long_strike - call_short_strike
                                                call_short_mid = call_delta_chain[call_short_idx][2] or 0.0
                                                call_net_credit = call_short_mid - call_long_mid
                                                call_short_ticker = _find_ticker_for_strike(call_chain, call_short_strike, "call")
                                                call_long_ticker = _find_ticker_for_strike(call_chain, call_long_strike, "call")

                                                if call_short_ticker and call_long_ticker:
                                                    sync_quotes = await client.fetch_synchronized_ohlcv(call_short_ticker, call_long_ticker, td, max_time_delta_minutes=max_time_delta_minutes)
                                                    if sync_quotes:
                                                        call_short_mid = sync_quotes[0]["mid"] or 0.0
                                                        call_long_mid = sync_quotes[1]["mid"] or 0.0
                                                        call_short_bid = sync_quotes[0].get("bid", call_short_mid)
                                                        call_short_ask = sync_quotes[0].get("ask", call_short_mid)
                                                        call_long_bid = sync_quotes[1].get("bid", call_long_mid)
                                                        call_long_ask = sync_quotes[1].get("ask", call_long_mid)
                                                        call_net_credit = _adjust_entry_credit(
                                                            call_short_mid, call_short_bid, call_short_ask,
                                                            call_long_mid, call_long_bid, call_long_ask,
                                                            slippage_model
                                                        )
                                                        print(
                                                            f"  [CALL ENTRY SYNC] Synchronized pricing found for {td}: "
                                                            f"Short=${call_short_mid}, Long=${call_long_mid}, Net=${call_net_credit:.2f}"
                                                        )

                                                call_width_ok = _spread_width_within_tolerance(
                                                    actual_call_width,
                                                    call_side_spread_width,
                                                    spread_width_tolerance_pct,
                                                )
                                                if not call_short_ticker or not call_long_ticker:
                                                    print(f"  {td}: [CALL ENTRY SKIPPED] Missing call tickers for {call_short_strike}/{call_long_strike}c")
                                                elif not call_width_ok:
                                                    print(
                                                        f"  {td}: [CALL ENTRY SKIPPED] Width ${actual_call_width:.2f} outside "
                                                        f"target ${call_side_spread_width:.2f}"
                                                    )
                                                elif call_net_credit < call_side_min_credit or call_net_credit >= actual_call_width:
                                                    print(
                                                        f"  {td}: [CALL ENTRY SKIPPED] Invalid credit ${call_net_credit:.2f} "
                                                        f"for {call_short_strike}/{call_long_strike}c"
                                                    )
                                                else:
                                                    call_margin_required = actual_call_width * 100 * num_contracts
                                                    call_entry_fee = _option_trade_fee(
                                                        num_contracts,
                                                        2,
                                                        fee_per_contract_per_side,
                                                    )
                                                    candidate_call = SpreadTrade(
                                                        short_ticker=call_short_ticker,
                                                        long_ticker=call_long_ticker,
                                                        short_strike=call_short_strike,
                                                        long_strike=call_long_strike,
                                                        expiration=selected_exp,
                                                        entry_date=td,
                                                        option_type="call",
                                                        entry_price=spot,
                                                        net_credit=call_net_credit,
                                                        entry_fee=call_entry_fee,
                                                        total_fees=call_entry_fee,
                                                        num_contracts=num_contracts,
                                                        margin_required=call_margin_required,
                                                        status="open",
                                                        entry_regime=current_regime,
                                                        entry_regime_name=current_regime_name,
                                                        entry_regime_v2=current_regime_v2_payload,
                                                        entry_dte=dte_days,
                                                        short_entry_mid=call_short_mid,
                                                        long_entry_mid=call_long_mid,
                                                        profit_target=call_side_profit_target,
                                                        entry_delta=call_short_delta,
                                                        is_roll=False,
                                                        hold_to_expiration=call_side_hold_to_expiration,
                                                        trade_id=uuid.uuid4().hex[:8],
                                                    )
                                                    candidate_call.root_trade_id = candidate_call.trade_id
                                                    before_margin = _portfolio_margin_requirement(active_trades)
                                                    after_margin = _portfolio_margin_requirement(active_trades + [candidate_call])
                                                    incremental_margin = after_margin - before_margin
                                                    if after_margin > allowed_margin + 1e-6:
                                                        print(
                                                            f"  {td}: [CALL ENTRY SKIPPED] Portfolio margin cap exceeded "
                                                            f"after call side (${after_margin:,.0f}/${allowed_margin:,.0f}; "
                                                            f"incremental ${incremental_margin:,.0f})"
                                                        )
                                                    else:
                                                        active_trades.append(candidate_call)
                                                        result.trades.append(candidate_call)
                                                        current_cash += call_net_credit * 100 * num_contracts - call_entry_fee
                                                        realized_capital -= call_entry_fee
                                                        print(
                                                            f"  {td}: OPEN {num_contracts}x {call_short_strike}/{call_long_strike}c "
                                                            f"| Credit=${call_net_credit:.2f} | Fee=${call_entry_fee:.2f} | Standalone Margin=${call_margin_required:,.0f} "
                                                            f"| Incremental Portfolio Margin=${incremental_margin:,.0f}"
                                                        )
                                                        if path_logger:
                                                            daily_events.append({
                                                                "type": "entry",
                                                                "reason": "call_side",
                                                                "option_type": "call",
                                                                "expiration": selected_exp,
                                                                "short_strike": call_short_strike,
                                                                "long_strike": call_long_strike,
                                                                "net_credit": float(call_net_credit),
                                                                "entry_fee": float(call_entry_fee),
                                                                "num_contracts": num_contracts,
                                                                "trade_id": candidate_call.trade_id,
                                                                "incremental_margin": float(incremental_margin),
                                                            })
                                selection_retry = False
                                    
            
            # Finalize daily summary stats (once per day)
            if opened_today:
                result.daily_opened_dte.append((td, last_dte))
                result.daily_opened_spread.append((td, last_spread))
                result.daily_leg_premiums.append((td, last_short_mid, last_long_mid))
            else:
                result.daily_leg_premiums.append((td, 0.0, 0.0))
            
            # --- PHASE 4: RECORD HISTORY ---
            # Update margin and NLV one last time for the day to capture new entries
            current_margin_usage = _portfolio_margin_requirement(active_trades)
            # Recalculate unrealized if needed? Actually current_cash was updated in entry.
            # But unrealized of new trades is 0 at entry.
            current_nlv = current_cash - daily_total_unrealized
            result.capital_history.append((td, realized_capital))
            result.cash_history.append((td, current_cash))
            result.nvl_history.append((td, current_nlv))
            result.margin_history.append((td, current_margin_usage))


            # End of daily loop: Log Day
            if path_logger:
                path_logger.log_day(
                    date_str=td,
                    spot=spot,
                    regime=current_regime,
                    regime_name="",
                    regime_v2=current_regime_v2_payload,
                    nlv=current_nlv,
                    cash=current_cash,
                    margin=current_margin_usage,
                    active_trades=active_trades,
                    events=daily_events,
                    abnormalities=[a for a in result.abnormalities if a["date"] == td],
                    margin_budget_capital=realized_capital,
                    margin_limit=realized_capital * margin_limit_pct,
                )
            
            # Clear daily memory cache at the end of each date iteration
            cache.clear_daily_memory_cache()
            
        print(f"\n  Backtest loop complete. Processed {len(trading_dates)} days.")
        print(f"  Total trades recorded: {len(result.trades)}")
        
        if path_logger:
            summary = {
                "total_pnl": float(result.total_pnl),
                "win_count": int(result.win_count),
                "loss_count": int(result.loss_count),
                "total_trades": int(result.total_trades),
                "max_drawdown": float(result.max_drawdown),
                "abnormalities": result.abnormalities,
                "causal_validity": result.causal_validity,
                "causal_validity_reasons": result.causal_validity_reasons,
                "contract_universe_manifest_sha256": (
                    result.contract_universe_manifest_sha256
                ),
                "contract_universe_snapshot_request_count": (
                    result.contract_universe_snapshot_request_count
                ),
                "contract_universe_request_amplification": (
                    result.contract_universe_request_amplification
                ),
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
    
    if plot or strategy_config:
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_id = str(uuid.uuid4())[:8]
        report_filename = f"report_{report_id}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        plot_results_interactive(result, underlying, output_path=report_path, strategy_config=strategy_config)
        
    if strategy_config:
        try:
            from backtesting.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker()
            tracker.record_experiment(
                strategy_config=strategy_config,
                start_date=start_date,
                end_date=end_date,
                result=result,
                log_path=output_log_path if enable_logging else "",
                report_path=f"reports/{report_filename}" if 'report_filename' in locals() else ""
            )
        except Exception as e:
            print(f"  WARNING: Failed to record experiment: {e}")
        
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
    capital_vals = [v for _, v in result.capital_history] if result.capital_history else nvl_vals
    
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
    for i, capital in enumerate(capital_vals):
        allowed = capital * result.margin_limit_pct
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


def _compute_regime_assignment_data(result: BacktestResult):
    """
    Post-hoc computation of regime-conditioned assignment probabilities and
    terminal outcomes for each non-roll trade in the backtest.

    For each non-roll trade:
      1. Use the entry-date regime state (already lagged one day by the backtester)
      2. Collect all causal forward returns (horizon = trade DTE in trading days)
         that belong to the same regime and whose outcomes resolved before entry
      3. Fit a BIC-selected GMM to those returns
      4. Query P(fwd_return < cutoff) where cutoff = ln(strike/spot)
      5. Record entry_delta as the market-implied assignment probability

    Returns a dict with keys:
      'dates'            - list of entry date strings
      'entry_deltas'     - list of abs(entry_delta) per trade
      'regime_gmm_probs' - list of regime GMM P(assignment) per trade
      'regime_labels'    - list of regime label strings
      'terminal_dates'   - list of exit/expiration dates for resolved trades
      'terminal_dists'   - list of (spot_at_exit - short_strike) values
    """
    from live_trading.ev_engine import fit_gmm, query_gmm

    trades = result.trades
    underlying = result.underlying_prices

    if not trades or underlying.empty:
        return None

    regime_hist = result.regime_history   # list of (date_str, state_int)
    regime_labels_map = dict(result.regime_labels)  # {state_int: label_str}

    # Check if regime data is meaningful (not all -1, which means regime_aware=false)
    has_meaningful_regimes = regime_hist and any(s >= 0 for _, s in regime_hist)

    # If the backtester didn't populate regime data (regime_aware=false),
    # skip the slow post-hoc walk-forward HMM/GMM training, as it is not relevant
    # for non-regime strategies and adds significant execution overhead.
    if not has_meaningful_regimes:
        print("    [Audit] Non-regime strategy. Skipping post-hoc HMM walk-forward training for audit plots.")
        return None

    # Build a date -> regime_state lookup from regime_history
    regime_by_date = {d: s for d, s in regime_hist}

    # Build numeric price series indexed by date string for forward returns
    price_dates = list(underlying.index)
    price_vals = np.array(underlying.values, dtype=float)
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    # Default trading horizon ≈ 42 calendar days → ~30 trading days
    default_trading_horizon = 30

    # Pre-compute regime-labelled forward returns for the default horizon.
    # For trades with different DTEs, we'll recompute as needed.
    def _get_forward_returns_by_regime(trading_horizon, as_of_idx):
        """
        Collect all regime-labelled forward returns using data whose outcomes
        have fully resolved before as_of_idx.

        Returns: dict {regime_state: np.array of fractional returns}
        """
        regime_returns = {}
        # An entry at index i has its outcome at i + trading_horizon.
        # The outcome must be known before as_of_idx (strictly causal).
        max_entry_idx = as_of_idx - trading_horizon
        if max_entry_idx <= 0:
            return regime_returns

        for i in range(max_entry_idx):
            d = price_dates[i]
            state = regime_by_date.get(d)
            if state is None:
                continue
            end_idx = i + trading_horizon
            if end_idx >= len(price_vals):
                continue
            fwd_ret = (price_vals[end_idx] / price_vals[i]) - 1.0
            if not np.isfinite(fwd_ret):
                continue
            if state not in regime_returns:
                regime_returns[state] = []
            regime_returns[state].append(fwd_ret)

        return {s: np.array(v) for s, v in regime_returns.items()}

    # GMM cache keyed by (regime_state, trading_horizon, as_of_date_bucket)
    # We bucket as_of_date by month to avoid refitting GMM for every single trade day
    _gmm_cache = {}

    def _get_gmm_model(regime_state, trading_horizon, as_of_idx):
        """Get or compute a cached GMM model for the given regime and horizon."""
        # Bucket by 21-trading-day intervals to balance freshness vs compute
        bucket = as_of_idx // 21
        cache_key = (regime_state, trading_horizon, bucket)
        if cache_key in _gmm_cache:
            return _gmm_cache[cache_key]

        regime_returns = _get_forward_returns_by_regime(trading_horizon, as_of_idx)
        bucket_returns = regime_returns.get(regime_state)
        if bucket_returns is None or len(bucket_returns) < 10:
            _gmm_cache[cache_key] = None
            return None

        model = fit_gmm(bucket_returns, regime_label=f"State_{regime_state}")
        _gmm_cache[cache_key] = model
        return model

    # Compute per-trade probabilities
    dates = []
    entry_deltas = []
    regime_gmm_probs = []
    regime_label_strs = []
    terminal_dates = []
    terminal_dists = []

    for t in trades:
        # Skip rolls — only newly opened trades
        if t.is_roll:
            continue
        # The assignment-probability audit is currently calibrated for put
        # spreads. Call-side trades are tracked in PnL/margin, but excluded
        # from this put-specific probability panel.
        if (t.option_type or "put").lower() != "put":
            continue

        entry_date = t.entry_date
        if entry_date not in date_to_idx:
            continue

        entry_idx = date_to_idx[entry_date]
        spot = price_vals[entry_idx]
        strike = t.short_strike

        # Use the trade's stored regime if available, otherwise look up from
        # the (possibly auto-trained) regime map
        regime_state = t.entry_regime
        if regime_state < 0:
            regime_state = regime_by_date.get(entry_date, -1)

        if spot <= 0 or strike <= 0 or regime_state < 0:
            continue

        # Determine trading horizon for this trade's DTE
        dte = t.entry_dte
        if dte and dte > 0:
            # Rough calendar-to-trading conversion: dte * 5/7
            trading_horizon = max(5, int(round(dte * 5.0 / 7.0)))
        else:
            trading_horizon = default_trading_horizon

        # Get GMM model for this regime, causally
        gmm_model = _get_gmm_model(regime_state, trading_horizon, entry_idx)

        if gmm_model is not None:
            # query_gmm returns P(log_return < ln(strike/spot))
            # For puts: assignment = spot falls below strike = P(return < cutoff)
            gmm_prob = query_gmm(gmm_model, spot, strike)
        else:
            gmm_prob = float('nan')

        dates.append(entry_date)
        entry_deltas.append(abs(t.entry_delta))
        regime_gmm_probs.append(gmm_prob)
        regime_label_strs.append(
            regime_labels_map.get(regime_state, f"Regime {regime_state}")
        )

        # Terminal outcome for resolved trades
        if t.status == "closed" and t.exit_date:
            exit_date = t.exit_date
            # Use the actual exit/expiration date spot price
            spot_at_exit = underlying.get(exit_date)
            if spot_at_exit is not None and np.isfinite(float(spot_at_exit)):
                terminal_dates.append(entry_date)
                terminal_dists.append(float(spot_at_exit) - strike)
            else:
                terminal_dates.append(entry_date)
                terminal_dists.append(float('nan'))
        else:
            terminal_dates.append(entry_date)
            terminal_dists.append(float('nan'))

    if not dates:
        print("    [Audit] No valid trades processed for assignment probability plots.")
        return None

    valid_gmm = sum(1 for p in regime_gmm_probs if not np.isnan(p))
    valid_terminal = sum(1 for v in terminal_dists if not np.isnan(v))
    print(f"    [Audit] Processed {len(dates)} trades: {valid_gmm} with GMM prob, {valid_terminal} with terminal outcomes")

    return {
        'dates': dates,
        'entry_deltas': entry_deltas,
        'regime_gmm_probs': regime_gmm_probs,
        'regime_labels': regime_label_strs,
        'terminal_dates': terminal_dates,
        'terminal_dists': terminal_dists,
    }


def plot_results_interactive(result: BacktestResult, ticker: str, output_path: str = "ev_plots_backtest.html", strategy_config: dict = None):
    """
    Interactive 3x2 multi-panel plotting for backtest results using Plotly.
    Saves to the specified output_path and embeds strategy description if provided.
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
    capital_vals = [v for _, v in result.capital_history] if result.capital_history else nvl_vals

    # Compute regime assignment data for the new audit subplots
    print("  Computing regime assignment probabilities (causal GMM)...", flush=True)
    try:
        audit_data = _compute_regime_assignment_data(result)
    except Exception as e:
        print(f"  WARNING: Failed to compute regime assignment data: {e}")
        audit_data = None

    # Create figure with 4 rows and 2 columns
    fig = make_subplots(
        rows=4, cols=2,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=(
            f"{ticker} Price & Trade Entry/Exit", 
            "Capital & Portfolio Value (NLV)", 
            "Market Regimes (HMM) & VIX",
            "Entry Leg Premiums",
            "Position Distribution (ITM/OTM)",
            "Entry DTE & Bid-Ask Spread",
            "Probability of Assignment",
            "Terminal Price \u2212 Strike"
        ),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
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
    for i, capital in enumerate(capital_vals):
        allowed = capital * result.margin_limit_pct
        if margin_vals[i] >= allowed * 0.99:
            margin_cap_dates.append(hist_dates[i])
            margin_cap_vals.append(margin_vals[i])
    
    if margin_cap_dates:
        fig.add_trace(
            go.Scatter(x=margin_cap_dates, y=margin_cap_vals, name='Margin Cap Hit', 
                       mode='markers', marker=dict(color='red', size=2.5), legend='legend2'),
            row=1, col=2, secondary_y=True
        )

    # --- Subplot 3: Regimes & VIX ---
    has_probs = False
    if hasattr(result, "regime_probabilities") and result.regime_probabilities is not None and not result.regime_probabilities.empty:
        prob_df = result.regime_probabilities
        prob_cols = [c for c in prob_df.columns if c.startswith('prob_state_')]
        if prob_cols:
            if hist_dates:
                plot_start = pd.to_datetime(hist_dates[0])
                plot_end = pd.to_datetime(hist_dates[-1])
                prob_index = pd.to_datetime(prob_df.index)
                prob_df = prob_df.loc[(prob_index >= plot_start) & (prob_index <= plot_end)]
            # Color map matching HMM states: State 0 (Robust Expansion) = Blue, State 1 (Cautious Decline) = Amber/Orange, State 2 (Panic / Crisis) = Red
            colors = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#ec4899']
            for col in sorted(prob_cols):
                state_id = int(col.split('_')[-1])
                label = result.regime_labels.get(state_id, f"State {state_id}")
                color = colors[state_id % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=prob_df.index.tolist(), 
                        y=prob_df[col].values.tolist(),
                        name=f"P({label})", 
                        line=dict(color=color, width=1.5), 
                        legend='legend3'
                    ), 
                    row=2, col=1
                )
            has_probs = True

    if not has_probs and result.regime_history:
        reg_df = pd.DataFrame(result.regime_history, columns=['date', 'state']).dropna()
        reg_dates = reg_df['date'].tolist()
        reg_vals = reg_df['state'].tolist()
        
        # (Background loop moved to end of function to ensure subplot initialization)
        
        # Add HMM State line and VIX
        fig.add_trace(go.Scatter(x=reg_dates, y=reg_vals, name='HMM State', line=dict(color='black', width=1.5), legend='legend3'), row=2, col=1)

    if not result.vix_prices.empty:
        vix_clean = result.vix_prices.dropna()
        if hist_dates:
            plot_start = pd.to_datetime(hist_dates[0])
            plot_end = pd.to_datetime(hist_dates[-1])
            vix_index = pd.to_datetime(vix_clean.index)
            vix_clean = vix_clean.loc[(vix_index >= plot_start) & (vix_index <= plot_end)]
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

    # --- Subplot 7: Probability of Assignment (Regime GMM vs Delta) ---
    if audit_data is not None:
        # Regime color palette for scatter markers
        regime_color_map = {}
        regime_palette = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899']
        seen_labels = []
        for lbl in audit_data['regime_labels']:
            if lbl not in seen_labels:
                seen_labels.append(lbl)
        for i, lbl in enumerate(seen_labels):
            regime_color_map[lbl] = regime_palette[i % len(regime_palette)]

        marker_colors = [regime_color_map.get(lbl, '#64748b') for lbl in audit_data['regime_labels']]

        # Entry delta line (market-implied)
        fig.add_trace(
            go.Scatter(
                x=audit_data['dates'], y=audit_data['entry_deltas'],
                name='Entry |Δ| (Market)',
                mode='lines+markers',
                line=dict(color='rgba(100,116,139,0.4)', width=1),
                marker=dict(size=3, color='#64748b'),
                legend='legend7'
            ),
            row=4, col=1
        )

        # Regime GMM assignment probability
        fig.add_trace(
            go.Scatter(
                x=audit_data['dates'], y=audit_data['regime_gmm_probs'],
                name='Regime GMM P(Assign)',
                mode='markers',
                marker=dict(size=5, color=marker_colors, line=dict(width=0.5, color='white')),
                text=[f"{lbl}<br>Δ={d:.3f} GMM={g:.3f}" for lbl, d, g in zip(audit_data['regime_labels'], audit_data['entry_deltas'], audit_data['regime_gmm_probs'])],
                hovertemplate='%{x}<br>%{text}<extra></extra>',
                legend='legend7'
            ),
            row=4, col=1
        )

    # --- Subplot 8: Terminal Price − Strike ---
    if audit_data is not None:
        td = audit_data['terminal_dates']
        tv = audit_data['terminal_dists']
        # Separate into safe (OTM at exit) and assigned (ITM at exit)
        safe_x, safe_y = [], []
        assign_x, assign_y = [], []
        for i in range(len(td)):
            if np.isnan(tv[i]):
                continue
            if tv[i] >= 0:  # For puts: spot >= strike → OTM → safe
                safe_x.append(td[i])
                safe_y.append(tv[i])
            else:
                assign_x.append(td[i])
                assign_y.append(tv[i])

        if safe_x:
            fig.add_trace(
                go.Bar(x=safe_x, y=safe_y, name='Safe (OTM)',
                       marker_color='rgba(16,185,129,0.6)', legend='legend8'),
                row=4, col=2
            )
        if assign_x:
            fig.add_trace(
                go.Bar(x=assign_x, y=assign_y, name='Assigned (ITM)',
                       marker_color='rgba(239,68,68,0.6)', legend='legend8'),
                row=4, col=2
            )
        # Zero reference line
        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=4, col=2)

    # Layout Updates
    fig.update_layout(
        height=1600,
        autosize=True,
        title_text=f"Backtest Analysis: {ticker} Strategy",
        template="plotly_white",
        hovermode="x",
        spikedistance=-1,
        hoverdistance=-1,
        showlegend=True,
        dragmode="zoom",
        # Configure Multiple Legends
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        legend2=dict(x=0.51, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        legend3=dict(x=0.01, y=0.74, bgcolor="rgba(255,255,255,0.7)"),
        legend4=dict(x=0.51, y=0.74, bgcolor="rgba(255,255,255,0.7)"),
        legend5=dict(x=0.01, y=0.49, bgcolor="rgba(255,255,255,0.7)"),
        legend6=dict(x=0.51, y=0.49, bgcolor="rgba(255,255,255,0.7)"),
        legend7=dict(x=0.01, y=0.24, bgcolor="rgba(255,255,255,0.7)"),
        legend8=dict(x=0.51, y=0.24, bgcolor="rgba(255,255,255,0.7)"),
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
    fig.update_yaxes(title_text="Probability", row=4, col=1)
    fig.update_yaxes(title_text="Spot − Strike ($)", row=4, col=2)

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
            # Add to all 8 subplots explicitly using row/col
            for r in [1, 2, 3, 4]:
                for c in [1, 2]:
                    fig.add_vrect(
                        x0=group['date'].iloc[0], x1=group['date'].iloc[-1],
                        fillcolor=color, opacity=0.35, layer="below", line_width=0,
                        row=r, col=c
                    )

    # Sync X-axes using Plotly's native 'matches' property
    # This ensures all subplots stay in sync even after 'Reset Axes' or 'Autoscale'
    for i in range(2, 9):
        ax_name = f'xaxis{i}'
        if ax_name in fig.layout:
            fig.layout[ax_name].matches = 'x'

    
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
    
    # Wrap Plotly's output with strategy description
    config_html = ""
    if strategy_config:
        import yaml
        config_yaml = yaml.dump(strategy_config, default_flow_style=False)
        desc = strategy_config.get('description', 'No description provided.')
        config_html = f"""
        <div style="padding: 20px; background-color: #f8fafc; border-bottom: 1px solid #e2e8f0; font-family: 'Inter', sans-serif;">
            <h1 style="margin: 0 0 10px 0; color: #1e293b;">Strategy Report: {strategy_config.get('name', strategy_config.get('id', 'Unknown'))}</h1>
            <p style="margin: 0 0 20px 0; color: #64748b; line-height: 1.5; max-width: 800px;">{desc}</p>
            <details style="cursor: pointer;">
                <summary style="font-weight: 600; color: #3b82f6;">View Strategy YAML</summary>
                <pre style="background: #1e293b; color: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 10px; overflow-x: auto;">{config_yaml}</pre>
            </details>
        </div>
        """

    # Get the raw plotly html
    plotly_html = fig.to_html(include_plotlyjs='cdn', post_script=auto_rescale_js, full_html=False)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Backtest Report - {ticker}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Inter', sans-serif; background-color: #f1f5f9; }}
        </style>
    </head>
    <body>
        {config_html}
        <div style="background: white; margin: 20px; padding: 10px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
            {plotly_html}
        </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n  ✓ Detailed strategy report saved to: {output_path}")


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
        yf_ticker = "^SPX" if underlying == "SPX" else underlying
        data = yf_download_with_retry(yf_ticker, start=fetch_start, end=end, auto_adjust=False)
        
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
    from backtesting.strategy_loader import load_strategy
    
    parser = argparse.ArgumentParser(description="Historical Option Backtest")
    parser.add_argument("--strategy", type=str, default="baseline_put_spread", help="Strategy ID to load from strategies/ directory")
    parser.add_argument("--underlying", default=None)
    parser.add_argument("--start", default="2025-03-03")
    parser.add_argument("--end", default="2025-03-14")
    parser.add_argument("--plot", action="store_true", help="Plot backtest results")
    parser.add_argument("--log", action="store_true", help="Enable structured path logging")
    parser.add_argument("--strategy_config", type=str, help="Path to JSON file with strategy configuration")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite option-data cache path")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use cached option data only and never attempt Massive/Polygon API calls.",
    )
    parser.add_argument("--slippage_model", type=str, choices=["none", "worst_case", "steer_50"], default="none", help="Slippage model to use")
    parser.add_argument("--max_time_delta_minutes", type=float, default=5.0, help="Constraint on the allowed time delta between option legs in minutes")
    args = parser.parse_args()

    # Load base config
    config = load_strategy(args.strategy)
    
    # Merge with custom config if provided
    if args.strategy_config:
        with open(args.strategy_config, "r") as f:
            custom_cfg = json.load(f)
            if "entry" not in config: config["entry"] = {}
            if "exit" not in config: config["exit"] = {}
            if "sizing" not in config: config["sizing"] = {}
            if "rolling" not in config: config["rolling"] = {}
            if "execution" not in config: config["execution"] = {}
            
            # Simple top-level key mapping for legacy support
            key_map = {
                "target_dte": ("entry", "target_dte"),
                "close_dte": ("exit", "close_dte"),
                "short_delta": ("entry", "short_delta"),
                "spread_width": ("entry", "spread_width"),
                "initial_capital": ("sizing", "initial_capital"),
                "margin_limit_pct": ("sizing", "margin_limit_pct"),
                "early_profit_pct": ("exit", "early_profit_pct"),
                "daily_pacing_slots": ("sizing", "daily_pacing_slots"),
                "dynamic_sizing": ("sizing", "dynamic_sizing"),
                "regime_dynamic_delta": ("entry", "regime_dynamic_delta"),
            }
            for k, (sec, field) in key_map.items():
                if k in custom_cfg:
                    config[sec][field] = custom_cfg[k]

            # Deep merge nested config blocks
            for sec in ["entry", "exit", "sizing", "rolling", "filters", "call_side", "execution"]:
                if sec in custom_cfg and isinstance(custom_cfg[sec], dict):
                    if sec not in config: config[sec] = {}
                    for k, v in custom_cfg[sec].items():
                        config[sec][k] = v

            for flag in ["regime_aware", "dynamic_delta_variant", "panic_exit_enabled", "panic_swap_enabled"]:
                if flag in custom_cfg:
                    config[flag] = custom_cfg[flag]

            # Preserve top-level keys like name, id, underlying, and description if defined
            if "name" in custom_cfg:
                config["name"] = custom_cfg["name"]
            if "id" in custom_cfg:
                config["id"] = custom_cfg["id"]
            if "underlying" in custom_cfg:
                config["underlying"] = custom_cfg["underlying"]
            if "description" in custom_cfg:
                config["description"] = custom_cfg["description"]

    entry_cfg = config.get("entry", {})
    exit_cfg = config.get("exit", {})
    sizing_cfg = config.get("sizing", {})
    rolling_cfg = config.get("rolling", {})

    max_time_delta = args.max_time_delta_minutes
    if "entry" in config and "max_time_delta_minutes" in config["entry"]:
        max_time_delta = float(config["entry"]["max_time_delta_minutes"])

    # Determine underlying: CLI argument overrides, otherwise fall back to strategy config
    underlying = args.underlying or config.get("underlying", "SPY")

    await run_put_credit_spread_backtest(
        underlying=underlying,
        start_date=args.start,
        end_date=args.end,
        target_dte=entry_cfg.get("target_dte", 42),
        close_dte=exit_cfg.get("close_dte", 21),
        target_short_delta=entry_cfg.get("short_delta", -0.15),
        spread_width=entry_cfg.get("spread_width", 20.0),
        initial_capital=sizing_cfg.get("initial_capital", 100000.0),
        margin_limit_pct=sizing_cfg.get("margin_limit_pct", 0.5),
        early_profit_pct=exit_cfg.get("early_profit_pct", 0.7),
        dynamic_delta_variant=bool(config.get("dynamic_delta_variant", True)),
        panic_exit_enabled=bool(config.get("panic_exit_enabled", True)),
        daily_pacing_slots=sizing_cfg.get("daily_pacing_slots", 0),
        backtest_qty=sizing_cfg.get("backtest_qty", 0),
        db_path=args.db_path,
        roll_spread_width_multiplier=rolling_cfg.get("spread_width_multiplier", 1.0),
        slippage_model=args.slippage_model,
        max_time_delta_minutes=max_time_delta,
        offline_only=args.offline_only or os.environ.get("MASSIVE_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes", "on"},
        plot=args.plot,
        enable_logging=args.log,
        strategy_config=config,
    )


if __name__ == "__main__":
    asyncio.run(main())
