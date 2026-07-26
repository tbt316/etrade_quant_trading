"""Read-only audit for the causal two-timescale regime detector.

The source-free legacy caches are wrapped in an explicitly unverified snapshot.
That preserves their usefulness for research replay without allowing normalized
local artifacts or assumed timestamps to masquerade as production provenance.
The script never downloads or writes data.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_trading.regime_detector_v2 import (
    RegimeDetectorConfig,
    detect_regimes_from_snapshot,
)
from live_trading.regime_market_data import (
    AVAILABILITY_HISTORICAL_ASSUMPTION,
    PAYLOAD_NORMALIZED_ONLY,
    MarketObservation,
    RegimeMarketDataSnapshot,
    SourceIdentity,
    regime_market_schedule,
)

SPY_LEGACY_IDENTITY = SourceIdentity(
    provider="local_artifact",
    dataset="derived_cache",
    provider_symbol="SPY",
    canonical_instrument="SPY",
    field="close",
    adjustment="unknown",
    unit="usd",
)
VIX_LEGACY_IDENTITY = SourceIdentity(
    provider="local_artifact",
    dataset="derived_cache",
    provider_symbol="VIX",
    canonical_instrument="VIX",
    field="close",
    adjustment="unknown",
    unit="index_points",
)


def _load_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spy_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "SPY.parquet"
    vix_path = PROJECT_ROOT / "s_and_p_data" / "api_cache" / "INDEX_VIX.parquet"
    live_path = PROJECT_ROOT / "spy_vix_price_cache.json"

    spy = pd.read_parquet(spy_path)["Close"].rename("SPY_Close")
    vix = pd.read_parquet(vix_path)["Close"].rename("VIX_Close")
    historical = pd.concat([spy, vix], axis=1).sort_index()

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

    conflicts = []
    for column in ("SPY_Close", "VIX_Close"):
        overlap = pd.concat(
            [
                historical[column].rename("parquet"),
                live[column].rename("dashboard_cache"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        mismatch = overlap[
            ~np.isclose(
                overlap["parquet"],
                overlap["dashboard_cache"],
                rtol=0.0,
                atol=0.005,
            )
        ].copy()
        mismatch = mismatch.reset_index(names="session")
        mismatch["instrument"] = column.removesuffix("_Close")
        mismatch["absolute_difference"] = (
            mismatch["dashboard_cache"] - mismatch["parquet"]
        ).abs()
        conflicts.append(mismatch)
    conflicts = (
        pd.concat(conflicts, ignore_index=True).sort_values(["session", "instrument"])
        if conflicts
        else pd.DataFrame()
    )

    # This precedence reproduces the old audit for comparison only.  The
    # resulting snapshot remains explicitly unverified and execution-ineligible.
    combined = pd.concat([historical, live]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    observed_rows = combined.dropna(how="all")
    full_schedule = regime_market_schedule(
        observed_rows.index.min().date(),
        observed_rows.index.max().date(),
    )
    full_sessions = pd.DatetimeIndex(full_schedule.index).tz_localize(None).normalize()
    unexpected = observed_rows.index.difference(full_sessions)
    quarantined = observed_rows.loc[unexpected].copy()

    common_start = max(combined[column].first_valid_index() for column in combined.columns)
    common_end = min(combined[column].last_valid_index() for column in combined.columns)
    common_schedule = regime_market_schedule(
        common_start.date(),
        common_end.date(),
    )
    common_sessions = pd.DatetimeIndex(common_schedule.index).tz_localize(None).normalize()
    combined = combined.reindex(common_sessions)
    if combined.isna().any().any():
        missing = combined[combined.isna().any(axis=1)].index
        raise ValueError(
            "Legacy replay has missing SPY/VIX sessions: "
            f"{[item.date().isoformat() for item in missing[:5]]}"
        )

    return combined, quarantined, conflicts


def _legacy_snapshot(prices: pd.DataFrame) -> RegimeMarketDataSnapshot:
    schedule = regime_market_schedule(
        prices.index.min().date(),
        prices.index.max().date(),
    )
    observations = []
    for session, row in prices.iterrows():
        session_date = session.date()
        clocks = schedule.loc[session]
        for identity, close, event_column in (
            (SPY_LEGACY_IDENTITY, row["SPY_Close"], "spy_event_at"),
            (VIX_LEGACY_IDENTITY, row["VIX_Close"], "vix_event_at"),
        ):
            event_at = pd.Timestamp(clocks[event_column])
            normalized_payload = json.dumps(
                {
                    "instrument": identity.canonical_instrument,
                    "session": session_date.isoformat(),
                    "selected_close": float(close),
                    "source_class": "legacy_derived_artifact",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            observations.append(
                MarketObservation(
                    session=session_date,
                    identity=identity,
                    close=close,
                    event_at=event_at,
                    available_at=event_at,
                    ingested_at=event_at,
                    request_id=(
                        f"legacy-replay:{identity.canonical_instrument.lower()}:"
                        f"{session_date.isoformat()}"
                    ),
                    raw_payload_sha256=hashlib.sha256(
                        normalized_payload
                    ).hexdigest(),
                    payload_kind=PAYLOAD_NORMALIZED_ONLY,
                    availability_basis=AVAILABILITY_HISTORICAL_ASSUMPTION,
                    is_final=True,
                )
            )
    as_of = pd.Timestamp(
        schedule.loc[prices.index.max(), "joint_finalization_at"]
    ) + pd.Timedelta(minutes=1)
    return RegimeMarketDataSnapshot(
        as_of=as_of,
        observations=tuple(observations),
    )


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


def main():
    prices, quarantined, conflicts = _load_prices()
    config = RegimeDetectorConfig()
    snapshot = _legacy_snapshot(prices)
    result = detect_regimes_from_snapshot(snapshot, config)
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
    print(f"Input snapshot SHA-256: {snapshot.snapshot_sha256}")
    print(f"Calendar schedule SHA-256: {snapshot.schedule_sha256}")
    print(f"Input schema: {snapshot.schema_version}")
    print(
        "Source policy: "
        f"{snapshot.source_policy_version} / {snapshot.source_policy_sha256}"
    )
    print(
        "Source provenance: UNVERIFIED "
        "(normalized legacy artifacts; historical finalization assumptions)"
    )
    print(
        "Provenance failure codes: "
        f"{result.attrs['input_provenance_failure_codes']}"
    )
    print(
        "Provenance metadata complete: "
        f"{result.attrs['input_provenance_evidence_complete']}"
    )
    print(f"Execution eligible: {result.attrs['execution_eligible']}")
    print(
        "Latest jointly finalized session: "
        f"{result.attrs['latest_jointly_finalized_session']}"
    )
    print(
        "Quarantined non-NYSE source rows: "
        f"{quarantined.to_dict(orient='index') if len(quarantined) else 'none'}"
    )
    print(
        "Conflicting overlapping legacy values: "
        f"{conflicts.to_dict(orient='records') if len(conflicts) else 'none'}"
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
