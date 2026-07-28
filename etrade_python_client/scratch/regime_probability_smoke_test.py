import os
import sys
import asyncio
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import (
    fetch_historical_data,
    train_regime_hmm,
    calendar_days_to_trading_days,
)
from live_trading.market_sessions import latest_available_session_before


def build_smoke_test_frame(
    train_start="2022-01-01",
    train_end="2024-08-31",
    audit_start="2024-05-01",
    audit_end="2024-06-28",
    horizon_calendar_days=14,
    target_put_delta=0.15,
    n_components=2,
):
    trading_horizon = calendar_days_to_trading_days(horizon_calendar_days)

    df = fetch_historical_data()
    if df.empty:
        raise RuntimeError("Historical data is empty.")

    df = df.loc[train_start:train_end].copy()
    df = df.dropna(subset=["SPY_Close", "VIX_Close"]).copy()
    if df.empty:
        raise RuntimeError("No data available in requested smoke-test window.")

    df["Log_Return"] = np.log(df["SPY_Close"] / df["SPY_Close"].shift(1))
    df["future_return"] = df["SPY_Close"].shift(-trading_horizon) / df["SPY_Close"] - 1
    fit_end = latest_available_session_before(
        df.index,
        audit_start,
    )

    best_hmm, best_k, feature_df = train_regime_hmm(
        df,
        n_components=n_components,
        expanding_window=False,
        fit_end=fit_end,
    )
    if best_hmm is None or feature_df.empty:
        raise RuntimeError("HMM training did not produce a usable feature frame.")

    audit_dates = feature_df.loc[audit_start:audit_end].index
    rows = []
    z_15 = abs(norm.ppf(target_put_delta))

    for d in audit_dates:
        if d not in df.index:
            continue

        current_pos = df.index.get_loc(d)
        resolved_cutoff_pos = current_pos - trading_horizon
        if resolved_cutoff_pos < 0:
            continue

        spot = float(df.loc[d, "SPY_Close"])
        vol = float(df.loc[:d, "Log_Return"].tail(20).std() * np.sqrt(252))
        if not np.isfinite(vol) or vol <= 0:
            continue

        strike = spot * (1 - z_15 * vol * np.sqrt(horizon_calendar_days / 365.0))
        strike_return = (strike / spot) - 1.0

        resolved_index = df.index[:resolved_cutoff_pos + 1]
        hist_sample = df.loc[resolved_index, "future_return"].dropna()
        if hist_sample.empty:
            continue

        state = int(feature_df.loc[d, "HMM_State"])
        regime_label = feature_df.loc[d, "Regime_Label"]
        eligible_feature_df = feature_df.loc[feature_df.index.intersection(resolved_index)]
        regime_dates = eligible_feature_df.index[eligible_feature_df["HMM_State"] == state]
        regime_sample = df.loc[regime_dates, "future_return"].dropna()
        if regime_sample.empty:
            continue

        terminal_pos = current_pos + trading_horizon
        if terminal_pos >= len(df):
            continue

        realized_return = float(df["future_return"].iloc[current_pos])
        terminal_spot = float(df["SPY_Close"].iloc[terminal_pos])
        rows.append(
            {
                "date": d,
                "spot": spot,
                "strike": strike,
                "strike_return": strike_return,
                "hist_prob": float(np.mean(hist_sample.values <= strike_return)),
                "regime_prob": float(np.mean(regime_sample.values <= strike_return)),
                "realized_return": realized_return,
                "breach": int(realized_return <= strike_return),
                "regime_state": state,
                "regime_label": regime_label,
                "terminal_spot": terminal_spot,
                "effective_trading_horizon": trading_horizon,
                "best_k": best_k,
                "model_calibration_end": fit_end,
            }
        )

    if not rows:
        raise RuntimeError("Smoke test produced no audit rows; widen the date window or reduce warmup requirements.")

    audit_df = pd.DataFrame(rows).set_index("date").sort_index()
    return audit_df, best_k


def summarize_smoke_test(audit_df):
    overall = {
        "rows": int(len(audit_df)),
        "mean_hist_prob": float(audit_df["hist_prob"].mean()),
        "mean_regime_prob": float(audit_df["regime_prob"].mean()),
        "realized_breach_rate": float(audit_df["breach"].mean()),
        "mean_realized_return": float(audit_df["realized_return"].mean()),
        "hist_brier": float(np.mean((audit_df["hist_prob"] - audit_df["breach"]) ** 2)),
        "regime_brier": float(np.mean((audit_df["regime_prob"] - audit_df["breach"]) ** 2)),
    }

    by_regime = (
        audit_df.groupby("regime_label")
        .agg(
            n=("breach", "size"),
            avg_hist_prob=("hist_prob", "mean"),
            avg_regime_prob=("regime_prob", "mean"),
            breach_rate=("breach", "mean"),
            avg_realized_return=("realized_return", "mean"),
            avg_strike_return=("strike_return", "mean"),
        )
        .sort_values(["breach_rate", "avg_realized_return"])
    )

    sample_rows = audit_df[
        [
            "regime_label",
            "hist_prob",
            "regime_prob",
            "strike_return",
            "realized_return",
            "breach",
        ]
    ].head(12)

    return overall, by_regime, sample_rows


async def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("live_trading.pca_fusion").setLevel(logging.ERROR)

    audit_df, best_k = build_smoke_test_frame()
    overall, by_regime, sample_rows = summarize_smoke_test(audit_df)

    print("=" * 72)
    print("REGIME PROBABILITY SMOKE TEST")
    print("=" * 72)
    print("Window: 2022-01-01 -> 2024-08-31")
    print(
        "Calibration end: "
        f"{audit_df['model_calibration_end'].iloc[0]}"
    )
    print("Audit:  2024-05-01 -> 2024-06-28")
    print("Horizon: 14 calendar days")
    print(f"Trading horizon used: {int(audit_df['effective_trading_horizon'].iloc[0])} rows")
    print(f"HMM K: {best_k}")
    print()
    print("Overall")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    print("By Regime")
    print(by_regime.round(4).to_string())
    print()
    print("Sample Rows")
    print(sample_rows.round(4).to_string())


if __name__ == "__main__":
    asyncio.run(main())
