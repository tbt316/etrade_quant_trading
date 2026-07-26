import json
import os
import sqlite3
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MASSIVE_OFFLINE_ONLY", "1")

from backtesting.backtest_runner import (
    _build_trade_entry_regime_inputs,
    _fetch_underlying_prices,
    get_trading_dates,
    lag_daily_regime_map,
    run_put_credit_spread_backtest,
)
from live_trading.ev_engine import fetch_historical_data, get_regime_labels, train_regime_hmm


START_DATE = "2016-01-01"
END_DATE = date.today().isoformat()
STRATEGY_ID = "baseline_put_spread"
DEFAULT_CACHE_DB = PROJECT_ROOT / "backtest_cache" / "option_data.db"
CONFIG_DIR = PROJECT_ROOT / "backtest_cache" / "generated_strategy_configs"
AUDIT_TRACE_PATHS = [
    PROJECT_ROOT / "backtest_cache" / "regime_audit_results_put_d012_20190101_20260519.json",
    PROJECT_ROOT / "backtest_cache" / "regime_audit_results.json",
]


BASE_CONFIG = {
    "id": STRATEGY_ID,
    "status": "implemented",
    "regime_aware": True,
    "dynamic_delta_variant": False,
    "panic_swap_enabled": False,
    "panic_exit_enabled": False,
    "instrument": "put_credit_spread",
    "underlying": "SPX",
    "entry": {
        "method": "fixed_delta",
        "target_dte": 42,
        "short_delta": -0.12,
        "spread_width": 150.0,
        "risk_free_rate": 0.05,
        "dynamic_delta_method": None,
    },
    "exit": {
        "method": "dynamic_dte + profit_target",
        "close_dte": 0,
        "early_profit_pct": 0.50,
        "conditional_half_dte_exit": False,
        "hold_itm_to_expiration": True,
    },
    "sizing": {
        "method": "daily_pacing",
        "initial_capital": 1000000.0,
        "margin_limit_pct": 0.25,
        "daily_pacing_slots": 15,
        "dynamic_margin_method": None,
    },
    "rolling": {
        "trigger": "itm_at_expiration",
        "enabled": True,
        "roll_at_expiration": True,
        "target_dte": 42,
        "roll_strike_behavior": "follow_underlying",
        "roll_dte_multiplier": 2.0,
        "spread_width_multiplier": 2.0,
        "qty_reduction": "half",
        "repeat_itm_at_expiration": False,
        "prefer_monthly_on_deviation": True,
        "stop_when_chain_breakeven": False,
    },
    "filters": {
        "min_chain_strikes": 10,
        "min_credit": 0.01,
        "strike_tolerance_pct": 0.30,
        "delta_tolerance_pct": 0.15,
    },
    "execution": {},
}


VARIANTS = [
    {
        "name": "Control: Baseline SPX Put Spread",
        "description": "Baseline SPX put spread control with the same regime trace attached, but no panic-transition roll and no panic exit.",
        "overrides": {},
    },
    {
        "name": "Variant: Panic Transition Even Roll",
        "description": "When the regime switches from 0/1 to 2, roll open SPX put spreads that are below half of their opening DTE to the target DTE using a short strike chosen to keep the roll approximately cash-even.",
        "overrides": {
            "rolling": {
                "panic_transition_roll_enabled": True,
                "panic_transition_from_states": [0, 1],
                "panic_transition_to_state": 2,
                "panic_transition_min_remaining_dte_fraction": 0.5,
            },
        },
    },
]


def _merge_dict(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "variant"


def _build_config(variant: dict, start_date: str, end_date: str) -> dict:
    config = deepcopy(BASE_CONFIG)
    config["name"] = variant["name"]
    config["description"] = f"SPX comparison from {start_date} to {end_date}: {variant['description']}"
    _merge_dict(config, variant.get("overrides", {}))
    return config


def _write_config(config: dict, variant: dict, start_date: str, end_date: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{_slug(variant['name'])}_{start_date}_{end_date}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return config_path


def _cache_row_count(cache_db: Path) -> int:
    if not cache_db.exists():
        return 0
    try:
        with sqlite3.connect(cache_db) as conn:
            row = conn.execute("SELECT COUNT(*) FROM contracts_cache").fetchone()
            return int(row[0] or 0)
    except Exception:
        return 0


def _extract_regime_probabilities(feature_df: pd.DataFrame, trading_dates: list[str]) -> pd.DataFrame | None:
    prob_cols = [col for col in feature_df.columns if col.startswith("prob_state_")]
    if not prob_cols:
        return None
    try:
        prob_cols.sort(key=lambda col: int(col.rsplit("_", 1)[-1]))
    except Exception:
        prob_cols.sort()
    return feature_df[prob_cols].reindex(trading_dates).ffill().bfill()


def _load_regime_context(start_date: str, end_date: str):
    trading_dates = get_trading_dates(start_date, end_date)
    regimes = {}
    regime_labels = {}
    regime_probabilities = None
    cached_trace_used = False

    for audit_path in AUDIT_TRACE_PATHS:
        if not audit_path.exists():
            continue
        try:
            with audit_path.open("r", encoding="utf-8") as f:
                audit_data = json.load(f)
            for row in audit_data.get("results", []):
                td = row.get("date")
                if not td or td < start_date or td > end_date:
                    continue
                state = row.get("regime_state")
                if state is None:
                    continue
                state = int(state)
                regimes[td] = state
                label = row.get("regime_label")
                if label:
                    regime_labels.setdefault(state, str(label))
            if regimes:
                cached_trace_used = True
                print(
                    f"  [Regime Trace] Loaded cached audit trace from {audit_path.name} "
                    f"({len(regimes)} dates in range)."
                )
                break
        except Exception as exc:
            print(f"  [Regime Trace] Failed to load {audit_path.name}: {exc}")

    if not cached_trace_used:
        df_hist = fetch_historical_data()
        if df_hist.empty:
            raise RuntimeError("Historical regime data is empty.")

        cal_start_date = pd.to_datetime(start_date) - timedelta(days=365 * 5)
        end_ts = pd.to_datetime(end_date)
        df_cal = df_hist[(df_hist.index >= cal_start_date) & (df_hist.index <= end_ts)].copy()
        if df_cal.empty:
            raise RuntimeError("Calibration window for regime detection is empty.")

        print(
            f"  [Walk-Forward] Training causal HMM trace from {cal_start_date.strftime('%Y-%m-%d')} to {end_date}..."
        )
        best_hmm, k, feature_df = train_regime_hmm(df_cal, n_components=3, expanding_window=True)
        if best_hmm is None or feature_df.empty:
            raise RuntimeError("Failed to train causal regime model.")

        close_regimes, detected_labels = _build_trade_entry_regime_inputs(feature_df)
        regime_labels = detected_labels or get_regime_labels(best_hmm, feature_df)
        regimes = lag_daily_regime_map(close_regimes, trading_dates)
        regime_probabilities = _extract_regime_probabilities(feature_df, trading_dates)

    underlying_prices = _fetch_underlying_prices("SPX", "2011-05-03", end_date)
    if underlying_prices.empty:
        raise RuntimeError("SPX underlying history is empty.")
    vix_prices = _fetch_underlying_prices("^VIX", start_date, end_date)

    state_count = len(regime_labels) or (max(regimes.values()) + 1 if regimes else 0)
    print(
        f"  [Walk-Forward] Loaded {len(trading_dates)} trading dates | HMM states: {state_count} | "
        f"Regime trace rows: {len(regimes)}"
    )

    return regimes, regime_labels, regime_probabilities, underlying_prices, vix_prices


def _metric_value(result, key: str):
    value = getattr(result, key, None)
    return float(value) if isinstance(value, (int, float)) else value


def _win_rate_pct(result) -> float:
    if not getattr(result, "total_trades", 0):
        return 0.0
    return float(result.win_count) / float(result.total_trades) * 100.0


def _print_comparison(results: list[tuple[dict, object]]) -> None:
    baseline_name, baseline = results[0][0]["name"], results[0][1]
    print("\nComparison vs baseline:")
    print("  Metric              Baseline             Variant              Delta")
    print("  ------------------  -------------------  -------------------  -------------------")

    fields = [
        ("total_trades", "Trades", 0),
        ("total_pnl", "Total PnL", 0.0),
        ("avg_pnl", "Avg PnL", 0.0),
        ("avg_credit", "Avg Credit", 0.0),
        ("max_drawdown", "Max Drawdown", 0.0),
        ("api_calls", "API Calls", 0),
        ("cache_hits", "Cache Hits", 0),
        ("data_gap_count", "Data Gaps", 0),
        ("critical_gap_count", "Critical Gaps", 0),
    ]

    for attr, label, _ in fields:
        base_val = _metric_value(baseline, attr)
        var_val = _metric_value(results[1][1], attr)
        if base_val is None or var_val is None:
            continue
        delta = var_val - base_val
        print(f"  {label:<18} {base_val:>19,.2f}  {var_val:>19,.2f}  {delta:>19,.2f}")

    base_wr = _win_rate_pct(baseline)
    var_wr = _win_rate_pct(results[1][1])
    print(f"  {'Win Rate %':<18} {base_wr:>19,.2f}  {var_wr:>19,.2f}  {var_wr - base_wr:>19,.2f}")
    print(f"\n  Baseline: {baseline_name}")
    print(f"  Variant:   {results[1][0]['name']}")


async def run_variant(
    variant: dict,
    start_date: str,
    end_date: str,
    regimes,
    regime_labels,
    regime_probabilities,
    underlying_prices,
    vix_prices,
    cache_db: Path,
):
    config = _build_config(variant, start_date, end_date)
    _write_config(config, variant, start_date, end_date)

    print(f"\n=== {variant['name']} ===")
    result = await run_put_credit_spread_backtest(
        underlying="SPX",
        start_date=start_date,
        end_date=end_date,
        target_dte=config["entry"].get("target_dte", 42),
        close_dte=config["exit"].get("close_dte", 0),
        target_short_delta=config["entry"].get("short_delta", -0.12),
        spread_width=config["entry"].get("spread_width", 150.0),
        initial_capital=config["sizing"].get("initial_capital", 1_000_000.0),
        margin_limit_pct=config["sizing"].get("margin_limit_pct", 0.25),
        early_profit_pct=config["exit"].get("early_profit_pct", 0.50),
        regimes=regimes,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        dynamic_delta_variant=bool(config.get("dynamic_delta_variant", False)),
        panic_swap_enabled=bool(config.get("panic_swap_enabled", False)),
        panic_exit_enabled=bool(config.get("panic_exit_enabled", False)),
        underlying_prices=underlying_prices,
        vix_prices=vix_prices,
        enable_logging=True,
        output_log_path=str(
            PROJECT_ROOT / "backtest_logs" / f"backtest_path_SPX_{_slug(variant['name'])}_{start_date}_{end_date}.json"
        ),
        strategy_config=config,
        daily_pacing_slots=config["sizing"].get("daily_pacing_slots", 0),
        roll_spread_width_multiplier=config["rolling"].get("spread_width_multiplier", 1.0),
        db_path=str(cache_db),
        offline_only=True,
    )
    return result


async def main_async(args) -> None:
    cache_db = Path(args.db_path).expanduser()
    if not cache_db.is_absolute():
        cache_db = PROJECT_ROOT / cache_db
    cache_db.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {len(VARIANTS)} SPX width variants from {args.start} to {args.end}. "
        "Existing experiment log entries will be preserved."
    )
    print(f"Option API cache: {cache_db} ({_cache_row_count(cache_db):,} cached contract rows before run)")

    regimes, regime_labels, regime_probabilities, underlying_prices, vix_prices = _load_regime_context(
        args.start,
        args.end,
    )

    results = []
    for variant in VARIANTS:
        result = await run_variant(
            variant,
            args.start,
            args.end,
            regimes,
            regime_labels,
            regime_probabilities,
            underlying_prices,
            vix_prices,
            cache_db,
        )
        results.append((variant, result))

    _print_comparison(results)
    print(f"Option API cache contract rows after run: {_cache_row_count(cache_db):,}")


def main() -> None:
    parser = ArgumentParser(description="Run SPX width comparison backtests.")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Unused; comparison runs sequentially to share one precomputed regime trace.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_CACHE_DB),
        help="Repo-local SQLite cache for option data.",
    )
    args = parser.parse_args()
    import asyncio

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
