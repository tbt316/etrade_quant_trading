"""Read-only audit for the causal two-timescale regime detector.

The script combines the longer local SPY/VIX Parquet history with the fresher
dashboard cache, preferring the dashboard value on overlapping dates.  It never
downloads or writes data.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_trading.regime_detector_v2 import RegimeDetectorConfig, detect_regimes

NYSE = mcal.get_calendar("NYSE")


def _load_prices() -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    spy_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "SPY.parquet"
    vix_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "INDEX_VIX.parquet"
    live_path = PROJECT_ROOT / "spy_vix_price_cache.json"

    spy = pd.read_parquet(spy_path)["Close"].rename("SPY_Close")
    vix = pd.read_parquet(vix_path)["Close"].rename("VIX_Close")
    historical = pd.concat([spy, vix], axis=1)

    with live_path.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    live = pd.concat(
        [
            pd.Series(cache["SPY"], dtype=float, name="SPY_Close"),
            pd.Series(cache["VIX"], dtype=float, name="VIX_Close"),
        ],
        axis=1,
    )
    live.index = pd.to_datetime(live.index)

    combined = pd.concat([historical, live]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    observed_rows = combined.dropna(how="all")
    full_schedule = NYSE.schedule(
        start_date=observed_rows.index.min(),
        end_date=observed_rows.index.max(),
    )
    full_sessions = pd.DatetimeIndex(full_schedule.index).tz_localize(None).normalize()
    unexpected = observed_rows.index.difference(full_sessions)
    quarantined = observed_rows.loc[unexpected].copy()

    common_start = max(combined[column].first_valid_index() for column in combined.columns)
    common_end = max(combined[column].last_valid_index() for column in combined.columns)
    common_schedule = NYSE.schedule(start_date=common_start, end_date=common_end)
    common_sessions = pd.DatetimeIndex(common_schedule.index).tz_localize(None).normalize()
    combined = combined.reindex(common_sessions)

    file_modified_at = pd.Timestamp(live_path.stat().st_mtime, unit="s", tz="UTC")
    return combined, file_modified_at, quarantined


def _legacy_overlay(prices: pd.DataFrame) -> pd.Series:
    spy = prices["SPY_Close"]
    vix = prices["VIX_Close"]
    log_return = np.log(spy / spy.shift(1))
    return_5d = np.log(spy / spy.shift(5))
    drawdown_21d = spy / spy.rolling(21, min_periods=5).max() - 1.0
    panic = (
        (vix >= 35.0)
        | ((vix >= 30.0) & (drawdown_21d <= -0.08))
        | (log_return <= -0.045)
        | (return_5d <= -0.075)
    )
    decline = (
        ~panic
        & ((vix >= 25.0) | (drawdown_21d <= -0.06) | (return_5d <= -0.04))
    )
    return pd.Series(
        np.select(
            [panic, decline],
            ["panic_crisis", "cautious_decline"],
            default="expansion",
        ),
        index=prices.index,
        name="Legacy_Overlay",
    )


def _term_structure_cache_status() -> str:
    vix3m_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "INDEX_VIX3M.parquet"
    vvix_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "INDEX_VVIX.parquet"
    vix3m = pd.read_parquet(vix3m_path)["Close"].rename("VIX3M")
    vvix = pd.read_parquet(vvix_path)["Close"].rename("VVIX")
    overlap = pd.concat([vix3m, vvix], axis=1, join="inner").dropna()
    recent = overlap.loc["2025-06-02":"2026-04-29"]
    identical = len(recent) >= 20 and recent["VIX3M"].equals(recent["VVIX"])
    if identical:
        return (
            f"INVALID: VIX3M exactly equals VVIX on {len(recent)} rows; "
            f"VIX3M ends {vix3m.index.max().date()}"
        )
    return f"not_identically_copied; VIX3M ends {vix3m.index.max().date()}"


def _availability_proxies(
    prices: pd.DataFrame,
    file_modified_at: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    schedule = NYSE.schedule(start_date=prices.index.min(), end_date=prices.index.max())
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    spy_available_at = pd.Series(
        pd.DatetimeIndex(schedule["market_close"]) + pd.Timedelta(minutes=1),
        index=sessions,
    )
    vix_available_at = pd.Series(
        (
            sessions.tz_localize("America/New_York")
            + pd.Timedelta(hours=16, minutes=16)
        ).tz_convert("UTC"),
        index=sessions,
    )
    spy_available_at.iloc[-1] = max(spy_available_at.iloc[-1], file_modified_at)
    vix_available_at.iloc[-1] = max(vix_available_at.iloc[-1], file_modified_at)
    return spy_available_at, vix_available_at


def main():
    prices, file_modified_at, quarantined = _load_prices()
    config = RegimeDetectorConfig()
    spy_available_at, vix_available_at = _availability_proxies(
        prices,
        file_modified_at,
    )
    result = detect_regimes(
        prices,
        config,
        as_of=pd.Timestamp.now(tz="UTC"),
        spy_available_at=spy_available_at,
        vix_available_at=vix_available_at,
        source_provenance_verified=False,
    )
    result["Legacy_Overlay"] = _legacy_overlay(prices)

    print("CAUSAL SHADOW AUDIT")
    print(f"Calibration: trailing {config.calibration_window} sessions, prior rows only")
    print(
        "Minimum calibration history: "
        f"{config.min_calibration_history} prior valid feature observations"
    )
    print(f"Replay range: {prices.index.min().date()} through {prices.index.max().date()}")
    print("Inference: prefix-causal transparent score/state machine")
    print(f"Detector version: {result.attrs['detector_version']}")
    print(f"Configuration hash: {result.attrs['config_hash']}")
    print(f"Threshold status: {result.attrs['threshold_status']}")
    print(f"Signal time: {result.attrs['regime_signal_timestamp']}")
    print(f"As-of: {result.attrs['as_of']}")
    print(f"SPY available-at proxy: {result.attrs['spy_available_at']}")
    print(f"VIX available-at proxy: {result.attrs['vix_available_at']}")
    print(
        "Source provenance: UNVERIFIED "
        "(nominal historical cutoffs; latest filesystem mtime proxy)"
    )
    print(
        "Latest jointly finalized session: "
        f"{result.attrs['latest_jointly_finalized_session']}"
    )
    print(
        "Quarantined non-NYSE source rows: "
        f"{quarantined.to_dict(orient='index') if len(quarantined) else 'none'}"
    )
    print("Regime lag for trade entry: one trading session required")
    print("Return buckets: not used by this detector")
    print(f"Term-structure cache: {_term_structure_cache_status()}")

    periods = {
        "persistent conflict stress": ("2026-03-20", "2026-04-07"),
        "full April transition": ("2026-04-01", "2026-04-30"),
        "recent six weeks": ("2026-06-15", "2026-07-24"),
        "recent July": ("2026-07-09", "2026-07-24"),
    }
    for name, (start, end) in periods.items():
        period = result.loc[start:end]
        print(f"\n{name}: {start} through {end} ({len(period)} sessions)")
        print(f"  VIX median/max: {period['VIX_Close'].median():.2f}/{period['VIX_Close'].max():.2f}")
        print(f"  Background score mean: {period['Background_Score'].mean():.3f}")
        print(f"  Background states: {period['Background_State'].value_counts().to_dict()}")
        print(f"  Shock states: {period['Shock_State'].value_counts().to_dict()}")
        print(f"  Legacy overlay: {period['Legacy_Overlay'].value_counts().to_dict()}")

    columns = [
        "SPY_Close",
        "VIX_Close",
        "VIX_Daily_Return",
        "Background_Score",
        "Background_State",
        "Shock_State",
        "Composite_Regime",
        "Reason_Codes",
    ]
    recent_shocks = result.loc["2026-06-15":"2026-07-24"]
    recent_shocks = recent_shocks[recent_shocks["Shock_State"] == "active"]
    print("\nRecent active shocks")
    print(recent_shocks[columns].round(4).to_string())

    print("\nLatest signal")
    latest_columns = columns + [
        "Data_Quality",
        "Signal_Available_At",
        "Tradable_Session",
        "Regime_Signal_Timestamp",
    ]
    print(result[latest_columns].tail(1).round(4).to_string())


if __name__ == "__main__":
    main()
