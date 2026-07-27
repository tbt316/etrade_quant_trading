"""Causal two-timescale market-regime detector.

This module is intentionally independent from the legacy HMM pipeline.  It is a
transparent shadow-mode baseline for distinguishing a persistent volatility
climate from short-lived market shocks.

All features for date T use observations available through the close of T.  The
result is therefore usable no earlier than the next trading session.
"""

import hashlib
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import live_trading.regime_market_data as regime_market_data_module
from live_trading.regime_evidence_store import (
    RegimeEvidenceStore,
    SnapshotEvidenceReport,
)
from live_trading.regime_market_data import (
    CALENDAR_POLICY_VERSION,
    RegimeMarketDataSnapshot,
    regime_market_schedule,
)


SIGNAL_TIMESTAMP = "after_spy_vix_finalization_T_for_next_session"
DETECTOR_VERSION = "regime_v2_shadow_0.3.0"
BACKGROUND_UNAVAILABLE = "unavailable"
BACKGROUND_CALM = "calm"
BACKGROUND_ELEVATED = "elevated"
BACKGROUND_STRESS = "persistent_stress"
SHOCK_NONE = "none"
SHOCK_ACTIVE = "active"
SHOCK_AFTERSHOCK = "aftershock"
SHOCK_UNAVAILABLE = "unavailable"

NYSE = mcal.get_calendar("NYSE")


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Configuration for the transparent shadow detector.

    The defaults are research baselines, not production-calibrated trading
    thresholds.  They must be frozen inside each walk-forward training fold
    before the detector is allowed to influence orders.
    """

    calibration_window: int = 756
    min_calibration_history: int = 252
    vix_slow_window: int = 10
    realized_vol_window: int = 10
    drawdown_window: int = 21
    score_halflife: float = 2.0
    vix_slow_weight: float = 0.45
    realized_vol_weight: float = 0.35
    drawdown_weight: float = 0.20

    stress_entry_score: float = 0.75
    stress_exit_score: float = 0.65
    calm_entry_score: float = 0.45
    calm_exit_score: float = 0.55
    stress_entry_days: int = 3
    stress_exit_days: int = 5
    calm_entry_days: int = 3
    calm_exit_days: int = 3

    stress_vix_floor: float = 22.0
    stress_realized_vol_floor: float = 0.20
    stress_drawdown_floor: float = 0.06

    vix_shock_return: float = 0.10
    spy_shock_log_return: float = -0.015
    aftershock_days: int = 2

    def __post_init__(self):
        positive_ints = {
            "calibration_window": self.calibration_window,
            "min_calibration_history": self.min_calibration_history,
            "vix_slow_window": self.vix_slow_window,
            "realized_vol_window": self.realized_vol_window,
            "drawdown_window": self.drawdown_window,
            "stress_entry_days": self.stress_entry_days,
            "stress_exit_days": self.stress_exit_days,
            "calm_entry_days": self.calm_entry_days,
            "calm_exit_days": self.calm_exit_days,
        }
        invalid = [
            name
            for name, value in positive_ints.items()
            if isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ]
        if invalid:
            raise ValueError(
                "Configuration values must be positive integers: "
                f"{invalid}"
            )
        if (
            isinstance(self.aftershock_days, bool)
            or not isinstance(self.aftershock_days, int)
            or self.aftershock_days < 0
        ):
            raise ValueError(
                "aftershock_days must be a nonnegative integer"
            )
        finite_values = {
            "score_halflife": self.score_halflife,
            "vix_slow_weight": self.vix_slow_weight,
            "realized_vol_weight": self.realized_vol_weight,
            "drawdown_weight": self.drawdown_weight,
            "stress_entry_score": self.stress_entry_score,
            "stress_exit_score": self.stress_exit_score,
            "calm_entry_score": self.calm_entry_score,
            "calm_exit_score": self.calm_exit_score,
            "stress_vix_floor": self.stress_vix_floor,
            "stress_realized_vol_floor": self.stress_realized_vol_floor,
            "stress_drawdown_floor": self.stress_drawdown_floor,
            "vix_shock_return": self.vix_shock_return,
            "spy_shock_log_return": self.spy_shock_log_return,
        }
        invalid_finite = [
            name
            for name, value in finite_values.items()
            if isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ]
        if invalid_finite:
            raise ValueError(
                "Configuration values must be finite numbers: "
                f"{invalid_finite}"
            )
        if self.min_calibration_history > self.calibration_window:
            raise ValueError("min_calibration_history cannot exceed calibration_window")
        if not 0.0 <= self.calm_entry_score < self.calm_exit_score:
            raise ValueError("calm score thresholds are inconsistent")
        if not self.calm_exit_score < self.stress_exit_score < self.stress_entry_score <= 1.0:
            raise ValueError("stress score thresholds are inconsistent")
        if self.score_halflife <= 0:
            raise ValueError("score_halflife must be positive")
        score_weights = (
            self.vix_slow_weight,
            self.realized_vol_weight,
            self.drawdown_weight,
        )
        if (
            not all(math.isfinite(value) and value >= 0 for value in score_weights)
            or not math.isclose(sum(score_weights), 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "background score weights must be finite, nonnegative, "
                "and sum to one"
            )
        if min(
            self.stress_vix_floor,
            self.stress_realized_vol_floor,
            self.stress_drawdown_floor,
            self.vix_shock_return,
        ) < 0:
            raise ValueError("stress floors and VIX shock threshold cannot be negative")
        if self.spy_shock_log_return >= 0:
            raise ValueError("spy_shock_log_return must be negative")


def regime_detector_config_sha256(config: RegimeDetectorConfig) -> str:
    """Return the canonical identity of one immutable detector configuration."""

    if not isinstance(config, RegimeDetectorConfig):
        raise TypeError("config must be a RegimeDetectorConfig")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def regime_detector_code_sha256() -> str:
    """Return the portable identity of the detector and market-clock source.

    The algorithm identity must be stable across machines running the same
    committed source. Runtime package versions are recorded separately so a
    Python patch release cannot silently invalidate a frozen research plan.
    """

    components = {
        "detector_source_sha256": hashlib.sha256(
            inspect.getsource(sys.modules[__name__]).encode("utf-8")
        ).hexdigest(),
        "market_data_source_sha256": hashlib.sha256(
            inspect.getsource(regime_market_data_module).encode("utf-8")
        ).hexdigest(),
    }
    payload = json.dumps(
        components,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def regime_detector_runtime_fingerprint() -> str:
    """Return the exact runtime identity for diagnostics and promotion gates."""

    components = {
        "numpy_version": np.__version__,
        "pandas_market_calendars_version": getattr(
            mcal,
            "__version__",
            "unknown",
        ),
        "pandas_version": pd.__version__,
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
    }
    payload = json.dumps(
        components,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    required = ["SPY_Close", "VIX_Close"]
    missing = [column for column in required if column not in prices.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    frame = prices[required].copy()
    if frame.empty:
        raise ValueError("Price history is empty")
    if pd.api.types.is_numeric_dtype(frame.index.dtype):
        raise ValueError("Price index must contain session dates, not numeric row identifiers")
    try:
        normalized_index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    except Exception as exc:
        raise ValueError("Price index must contain valid timestamps") from exc

    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_localize(None)
    normalized_index = normalized_index.normalize()
    if normalized_index.has_duplicates:
        raise ValueError("Price index contains duplicate session dates")
    frame.index = normalized_index
    frame = frame.sort_index()

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Required SPY/VIX observations contain missing or non-finite values")
    if (frame[required] <= 0).any().any():
        raise ValueError("Required SPY/VIX observations must be strictly positive")

    schedule = NYSE.schedule(start_date=frame.index.min(), end_date=frame.index.max())
    expected_sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    missing_sessions = expected_sessions.difference(frame.index)
    unexpected_sessions = frame.index.difference(expected_sessions)
    if len(missing_sessions) or len(unexpected_sessions):
        missing_preview = [date.strftime("%Y-%m-%d") for date in missing_sessions[:5]]
        unexpected_preview = [date.strftime("%Y-%m-%d") for date in unexpected_sessions[:5]]
        raise ValueError(
            "Price index must exactly match contiguous NYSE sessions; "
            f"missing={missing_preview}, unexpected={unexpected_preview}"
        )
    return frame


def _as_aware_utc(value, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _vix_finalization_times(session_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the versioned Cboe Index Options close for each NYSE session."""

    dates = pd.DatetimeIndex(session_dates)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    dates = dates.normalize()
    if dates.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns, UTC]")
    schedule = regime_market_schedule(
        dates.min().date(),
        dates.max().date(),
    )
    missing = dates.difference(schedule.index)
    if len(missing):
        raise ValueError(
            "Could not resolve Cboe finalization times for sessions: "
            f"{[item.date().isoformat() for item in missing[:5]]}"
        )
    return pd.DatetimeIndex(schedule.loc[dates, "vix_event_at"])


def _normalize_availability(
    values: pd.Series,
    expected_index: pd.DatetimeIndex,
    field_name: str,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{field_name} must be a pandas Series indexed by session")

    series = values.copy()
    try:
        normalized_index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="raise"))
    except Exception as exc:
        raise ValueError(f"{field_name} index must contain valid session dates") from exc
    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_localize(None)
    normalized_index = normalized_index.normalize()
    if normalized_index.has_duplicates:
        raise ValueError(f"{field_name} index contains duplicate session dates")
    series.index = normalized_index
    if not series.index.equals(expected_index):
        missing = expected_index.difference(series.index)
        unexpected = series.index.difference(expected_index)
        raise ValueError(
            f"{field_name} must align exactly with price sessions; "
            f"missing={list(missing[:5])}, unexpected={list(unexpected[:5])}"
        )

    converted = [
        _as_aware_utc(value, f"{field_name}[{session.date()}]")
        for session, value in series.items()
    ]
    return pd.Series(pd.DatetimeIndex(converted), index=expected_index, name=field_name)


def _validate_latest_session(
    frame: pd.DataFrame,
    as_of,
    spy_available_at,
    vix_available_at,
    market_closes: pd.Series,
    vix_finalizations: pd.Series,
) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
    """Prove that the latest row is final for both SPY and VIX."""

    as_of_utc = _as_aware_utc(as_of, "as_of")
    spy_available_utc = _normalize_availability(
        spy_available_at,
        frame.index,
        "spy_available_at",
    )
    vix_available_utc = _normalize_availability(
        vix_available_at,
        frame.index,
        "vix_available_at",
    )
    if (spy_available_utc > as_of_utc).any():
        raise ValueError("spy_available_at cannot be later than as_of")
    if (vix_available_utc > as_of_utc).any():
        raise ValueError("vix_available_at cannot be later than as_of")

    as_of_eastern = as_of_utc.tz_convert("America/New_York")
    start_date = (as_of_eastern.normalize() - pd.Timedelta(days=31)).date()
    end_date = as_of_eastern.normalize().date()
    recent_schedule = regime_market_schedule(start_date, end_date)
    completed = recent_schedule[
        recent_schedule["joint_finalization_at"] <= as_of_utc
    ]
    if completed.empty:
        raise ValueError("No jointly finalized SPY/VIX session is available at as_of")

    expected_latest = pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()
    actual_latest = frame.index.max()
    if actual_latest != expected_latest:
        raise ValueError(
            "Price history is stale or premature: "
            f"latest row={actual_latest.date()}, "
            f"latest jointly finalized SPY/VIX session={expected_latest.date()}"
        )

    early_spy = spy_available_utc < market_closes
    early_vix = vix_available_utc < vix_finalizations
    if early_spy.any():
        first_session = early_spy[early_spy].index[0]
        raise ValueError(
            "SPY observation was available before the official NYSE close: "
            f"{first_session.date()}"
        )
    if early_vix.any():
        first_session = early_vix[early_vix].index[0]
        raise ValueError(
            "VIX observation was available before the official Cboe close: "
            f"{first_session.date()}"
        )
    return as_of_utc, spy_available_utc, vix_available_utc


def _session_finalization_metadata(
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    extended_end = index.max() + pd.Timedelta(days=31)
    schedule = regime_market_schedule(
        index.min().date(),
        extended_end.date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    positions = sessions.get_indexer(index)
    if np.any(positions < 0):
        raise ValueError("Could not resolve exact NYSE finalization metadata")

    closes = pd.Series(
        pd.DatetimeIndex(schedule.iloc[positions]["spy_event_at"]),
        index=index,
    )
    vix_finalizations = pd.Series(
        pd.DatetimeIndex(schedule.iloc[positions]["vix_event_at"]),
        index=index,
    )
    return closes, vix_finalizations


def _tradable_sessions(
    index: pd.DatetimeIndex,
    signal_available_at: pd.Series,
) -> pd.Series:
    latest_arrival = signal_available_at.max().tz_convert("America/New_York")
    extended_end = max(
        index.max(),
        latest_arrival.tz_localize(None).normalize(),
    ) + pd.Timedelta(days=31)
    schedule = NYSE.schedule(start_date=index.min(), end_date=extended_end)
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    opens = pd.DatetimeIndex(schedule["market_open"]).tz_convert("UTC")

    after_session = np.searchsorted(
        sessions.asi8,
        index.asi8,
        side="right",
    )
    after_arrival = np.searchsorted(
        opens.asi8,
        pd.DatetimeIndex(signal_available_at).asi8,
        side="right",
    )
    positions = np.maximum(after_session, after_arrival)
    if np.any(positions >= len(sessions)):
        raise ValueError("Could not resolve a tradable NYSE session after signal availability")
    return pd.Series(sessions[positions], index=index, dtype="datetime64[ns]")


def _causal_percentile(
    series: pd.Series,
    window: int,
    min_history: int,
) -> pd.Series:
    """Rank each observation against prior observations only."""

    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    for position, value in enumerate(values):
        if not np.isfinite(value):
            continue
        history = values[max(0, position - window) : position]
        history = history[np.isfinite(history)]
        if len(history) < min_history:
            continue
        below = np.count_nonzero(history < value)
        equal = np.count_nonzero(history == value)
        result[position] = (below + 0.5 * equal) / len(history)
    return pd.Series(result, index=series.index, dtype=float)


def _background_states(
    score: pd.Series,
    stress_evidence: pd.Series,
    config: RegimeDetectorConfig,
) -> tuple[list[str], list[int]]:
    state = BACKGROUND_ELEVATED
    state_age = 0
    stress_entry_count = 0
    stress_exit_count = 0
    calm_entry_count = 0
    calm_exit_count = 0
    states: list[str] = []
    ages: list[int] = []

    for timestamp, value in score.items():
        if not np.isfinite(value):
            stress_entry_count = 0
            stress_exit_count = 0
            calm_entry_count = 0
            calm_exit_count = 0
            states.append(BACKGROUND_UNAVAILABLE)
            ages.append(0)
            continue

        previous_state = state
        has_stress_evidence = bool(stress_evidence.loc[timestamp])
        is_stress_candidate = value >= config.stress_entry_score and has_stress_evidence
        if state != BACKGROUND_STRESS:
            stress_entry_count = stress_entry_count + 1 if is_stress_candidate else 0
        else:
            stress_entry_count = 0

        if state == BACKGROUND_STRESS:
            stress_exit_count = stress_exit_count + 1 if value < config.stress_exit_score else 0
            if stress_exit_count >= config.stress_exit_days:
                state = BACKGROUND_ELEVATED
                stress_exit_count = 0
        elif stress_entry_count >= config.stress_entry_days:
            state = BACKGROUND_STRESS
            stress_entry_count = 0
        elif state == BACKGROUND_CALM:
            calm_exit_count = calm_exit_count + 1 if value > config.calm_exit_score else 0
            if has_stress_evidence:
                state = BACKGROUND_ELEVATED
                calm_exit_count = 0
            elif calm_exit_count >= config.calm_exit_days:
                state = BACKGROUND_ELEVATED
                calm_exit_count = 0
        else:
            is_calm_candidate = value <= config.calm_entry_score and not has_stress_evidence
            calm_entry_count = calm_entry_count + 1 if is_calm_candidate else 0

            if calm_entry_count >= config.calm_entry_days:
                state = BACKGROUND_CALM
                calm_entry_count = 0

        if state == BACKGROUND_CALM and has_stress_evidence:
            state = BACKGROUND_ELEVATED

        if state != previous_state:
            state_age = 1
            stress_exit_count = 0
            calm_entry_count = 0
            calm_exit_count = 0
        else:
            state_age += 1

        states.append(state)
        ages.append(state_age)

    return states, ages


def _shock_states(
    active_shock: pd.Series,
    aftershock_days: int,
) -> tuple[list[str], list[int]]:
    remaining_aftershock_days = 0
    shock_age = 0
    states: list[str] = []
    ages: list[int] = []

    for is_active in active_shock:
        if pd.isna(is_active):
            state = SHOCK_UNAVAILABLE
            shock_age = 0
            remaining_aftershock_days = 0
        elif bool(is_active):
            state = SHOCK_ACTIVE
            shock_age = 1
            remaining_aftershock_days = aftershock_days
        elif remaining_aftershock_days > 0:
            state = SHOCK_AFTERSHOCK
            shock_age += 1
            remaining_aftershock_days -= 1
        else:
            state = SHOCK_NONE
            shock_age = 0

        states.append(state)
        ages.append(shock_age)

    return states, ages


def _reason_codes(row: pd.Series, config: RegimeDetectorConfig) -> str:
    reasons = []
    if row["Shock_State"] == SHOCK_UNAVAILABLE:
        reasons.append("shock_evidence_unavailable")
    if row["VIX_Daily_Return"] >= config.vix_shock_return:
        reasons.append("vix_daily_change_extreme")
    if row["SPY_Log_Return"] <= config.spy_shock_log_return:
        reasons.append("spy_downside_tail_move")
    if row["Shock_State"] == SHOCK_AFTERSHOCK:
        reasons.append("shock_decay_window")
    if row["VIX_Median_Slow"] >= config.stress_vix_floor:
        reasons.append("slow_vix_absolute_stress")
    if row["SPY_Realized_Vol"] >= config.stress_realized_vol_floor:
        reasons.append("realized_vol_absolute_stress")
    if row["SPY_Drawdown_Severity"] >= config.stress_drawdown_floor:
        reasons.append("drawdown_absolute_stress")
    if row["Background_Score"] >= config.stress_entry_score:
        reasons.append("background_score_high")
    elif row["Background_Score"] > config.calm_exit_score:
        reasons.append("background_score_elevated")
    if row["Background_State"] == BACKGROUND_STRESS:
        reasons.append("persistent_score_confirmed")
    return "|".join(reasons) if reasons else "no_stress_evidence"


def _detect_regimes_from_arrays(
    prices: pd.DataFrame,
    config: RegimeDetectorConfig | None = None,
    *,
    as_of,
    spy_available_at,
    vix_available_at,
    source_provenance_verified: bool,
) -> pd.DataFrame:
    """Low-level detector core for already validated array inputs.

    Required input columns are ``SPY_Close`` and ``VIX_Close``. Source
    availability must be supplied as timezone-aware Series aligned one-to-one
    with price sessions. Invalid, stale, premature, or non-session data raises
    instead of silently defaulting to a calm regime.
    """

    config = config or RegimeDetectorConfig()
    config_hash = regime_detector_config_sha256(config)
    if not isinstance(source_provenance_verified, (bool, np.bool_)):
        raise TypeError("source_provenance_verified must be a boolean")
    frame = _validate_prices(prices)
    market_closes, vix_finalizations = _session_finalization_metadata(frame.index)
    as_of_utc, spy_available_utc, vix_available_utc = _validate_latest_session(
        frame,
        as_of=as_of,
        spy_available_at=spy_available_at,
        vix_available_at=vix_available_at,
        market_closes=market_closes,
        vix_finalizations=vix_finalizations,
    )
    row_inputs_available_at = pd.concat(
        [spy_available_utc, vix_available_utc],
        axis=1,
    ).max(axis=1)
    signal_available_at = row_inputs_available_at.cummax()
    tradable_sessions = _tradable_sessions(frame.index, signal_available_at)
    result = frame.copy()

    result["SPY_Log_Return"] = np.log(result["SPY_Close"] / result["SPY_Close"].shift(1))
    result["VIX_Daily_Return"] = result["VIX_Close"].pct_change()
    result["VIX_Median_Slow"] = result["VIX_Close"].rolling(
        config.vix_slow_window,
        min_periods=config.vix_slow_window,
    ).median()
    result["SPY_Realized_Vol"] = (
        result["SPY_Log_Return"]
        .rolling(config.realized_vol_window, min_periods=config.realized_vol_window)
        .std()
        * np.sqrt(252.0)
    )
    rolling_high = result["SPY_Close"].rolling(
        config.drawdown_window,
        min_periods=config.drawdown_window,
    ).max()
    result["SPY_Drawdown_Severity"] = (1.0 - result["SPY_Close"] / rolling_high).clip(lower=0.0)

    percentile_inputs = {
        "VIX_Slow_Percentile": result["VIX_Median_Slow"],
        "Realized_Vol_Percentile": result["SPY_Realized_Vol"],
        "Drawdown_Percentile": result["SPY_Drawdown_Severity"],
    }
    for output_column, input_series in percentile_inputs.items():
        result[output_column] = _causal_percentile(
            input_series,
            window=config.calibration_window,
            min_history=config.min_calibration_history,
        )

    result["Background_Score_Raw"] = (
        config.vix_slow_weight * result["VIX_Slow_Percentile"]
        + config.realized_vol_weight * result["Realized_Vol_Percentile"]
        + config.drawdown_weight * result["Drawdown_Percentile"]
    )
    result["Background_Score"] = result["Background_Score_Raw"].ewm(
        halflife=config.score_halflife,
        adjust=False,
    ).mean()

    result["Absolute_Stress_Evidence"] = (
        (result["VIX_Median_Slow"] >= config.stress_vix_floor)
        | (result["SPY_Realized_Vol"] >= config.stress_realized_vol_floor)
        | (result["SPY_Drawdown_Severity"] >= config.stress_drawdown_floor)
    )
    background_states, background_ages = _background_states(
        result["Background_Score"],
        result["Absolute_Stress_Evidence"],
        config,
    )
    result["Background_State"] = background_states
    result["Background_Regime_Age"] = background_ages

    shock_evidence_available = result[
        ["VIX_Daily_Return", "SPY_Log_Return"]
    ].notna().all(axis=1)
    active_shock = (
        (result["VIX_Daily_Return"] >= config.vix_shock_return)
        | (result["SPY_Log_Return"] <= config.spy_shock_log_return)
    )
    result["Shock_Active"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[shock_evidence_available, "Shock_Active"] = active_shock.loc[
        shock_evidence_available
    ]
    shock_states, shock_ages = _shock_states(result["Shock_Active"], config.aftershock_days)
    result["Shock_State"] = shock_states
    result["Shock_Age"] = shock_ages

    result["Composite_Regime"] = result["Background_State"] + "+" + result["Shock_State"]
    result["Data_Quality"] = pd.Series(
        np.where(
            result["Background_State"] == BACKGROUND_UNAVAILABLE,
            "insufficient_calibration_history",
            "exchange_sessions_and_source_times_valid",
        ),
        index=result.index,
        dtype="object",
    )
    if not source_provenance_verified:
        result["Data_Quality"] = (
            result["Data_Quality"] + "|source_provenance_unverified"
        )
    result["Market_Close_At"] = market_closes
    result["VIX_Finalization_At"] = vix_finalizations
    result["Signal_Available_At"] = signal_available_at
    result["Tradable_Session"] = tradable_sessions
    result["Detector_Version"] = DETECTOR_VERSION
    result["Config_Hash"] = config_hash
    result["Regime_Signal_Timestamp"] = SIGNAL_TIMESTAMP
    result["Reason_Codes"] = result.apply(_reason_codes, axis=1, config=config)
    result.attrs["regime_signal_timestamp"] = SIGNAL_TIMESTAMP
    result.attrs["detector_version"] = DETECTOR_VERSION
    result.attrs["config_hash"] = config_hash
    result.attrs["detector_code_sha256"] = regime_detector_code_sha256()
    result.attrs["runtime_fingerprint_sha256"] = (
        regime_detector_runtime_fingerprint()
    )
    result.attrs["threshold_status"] = "research_baseline_unverified"
    result.attrs["inference_mode"] = "causal_transparent_shadow"
    result.attrs["freshness_assessed"] = bool(source_provenance_verified)
    result.attrs["source_time_validation_passed"] = True
    result.attrs["source_provenance_verified"] = bool(source_provenance_verified)
    result.attrs["as_of"] = as_of_utc.isoformat()
    result.attrs["spy_available_at"] = spy_available_utc.iloc[-1].isoformat()
    result.attrs["vix_available_at"] = vix_available_utc.iloc[-1].isoformat()
    result.attrs["signal_available_at"] = signal_available_at.iloc[-1].isoformat()
    result.attrs["latest_jointly_finalized_session"] = (
        frame.index.max().date().isoformat()
    )
    result.attrs["exchange_calendars"] = ("NYSE", "CBOE_Index_Options")
    result.attrs["calendar_policy_version"] = CALENDAR_POLICY_VERSION
    return result


def detect_regimes(
    prices: pd.DataFrame,
    config: RegimeDetectorConfig | None = None,
    *,
    as_of,
    spy_available_at,
    vix_available_at,
    source_provenance_verified: bool = False,
) -> pd.DataFrame:
    """Research-only compatibility interface for loose array inputs.

    Loose DataFrames cannot prove source identity or raw lineage.  They may be
    used for diagnostics with explicitly unverified provenance, but callers
    cannot promote them by passing a trusted Boolean.  Production/shadow
    ingestion must use :func:`detect_regimes_from_snapshot`.
    """

    if source_provenance_verified:
        raise ValueError(
            "Verified provenance requires RegimeMarketDataSnapshot; "
            "loose array inputs are research-only"
        )
    result = _detect_regimes_from_arrays(
        prices,
        config,
        as_of=as_of,
        spy_available_at=spy_available_at,
        vix_available_at=vix_available_at,
        source_provenance_verified=False,
    )
    result["Detector_Stage"] = "research_raw_inputs"
    result["Execution_Eligible"] = False
    result.attrs["detector_stage"] = "research_raw_inputs"
    result.attrs["execution_eligible"] = False
    result.attrs["input_contract"] = "loose_arrays_unverified"
    return result


def _detect_regimes_from_snapshot(
    snapshot: RegimeMarketDataSnapshot,
    config: RegimeDetectorConfig | None,
    evidence_report: SnapshotEvidenceReport | None,
) -> pd.DataFrame:
    if not isinstance(snapshot, RegimeMarketDataSnapshot):
        raise TypeError("snapshot must be a RegimeMarketDataSnapshot")
    if evidence_report is not None:
        if not evidence_report.verified:
            raise ValueError("Snapshot evidence report is not verified")
        if evidence_report.snapshot_sha256 != snapshot.snapshot_sha256:
            raise ValueError("Snapshot evidence report does not match the snapshot")
        if (
            evidence_report.evidence_sha256 is None
            or evidence_report.verification_kind is None
            or evidence_report.verified_at is None
            or evidence_report.failures
        ):
            raise ValueError("Snapshot evidence report is incomplete")

    inputs = snapshot.detector_inputs()
    inputs["source_provenance_verified"] = evidence_report is not None
    result = _detect_regimes_from_arrays(
        inputs.pop("prices"),
        config,
        **inputs,
    )
    original_attrs = result.attrs.copy()
    metadata = snapshot.source_metadata_frame()
    if not metadata.index.equals(result.index):
        raise ValueError("Snapshot source metadata does not align with detector output")
    result = result.join(metadata)
    result.attrs.update(original_attrs)

    provenance_failure_codes = sorted(
        {
            failure.rsplit(":", 1)[-1]
            for failure in snapshot.provenance_failures
        }
    )
    result["Input_Snapshot_SHA256"] = snapshot.snapshot_sha256
    result["Input_Schema_Version"] = snapshot.schema_version
    result["Calendar_Policy_Version"] = CALENDAR_POLICY_VERSION
    result["Calendar_Schedule_SHA256"] = snapshot.schedule_sha256
    result["Source_Policy_Version"] = snapshot.source_policy_version
    result["Source_Policy_SHA256"] = snapshot.source_policy_sha256
    if evidence_report is None:
        result["Input_Provenance_Status"] = "unverified"
        result["Input_Provenance_Evidence"] = (
            "complete_but_not_durably_verified"
            if snapshot.provenance_evidence_complete
            else "incomplete"
        )
        result["Evidence_Manifest_SHA256"] = None
        result["Evidence_Verification_Kind"] = None
        result["Evidence_Verified_At"] = None
        result["Evidence_Decision_Time_Eligible"] = False
    else:
        result["Input_Provenance_Status"] = "verified"
        result["Input_Provenance_Evidence"] = (
            "durable_raw_payload_and_parser_receipts_verified"
        )
        result["Evidence_Manifest_SHA256"] = evidence_report.evidence_sha256
        result["Evidence_Verification_Kind"] = (
            evidence_report.verification_kind
        )
        result["Evidence_Verified_At"] = evidence_report.verified_at
        decision_time_evidence = (
            evidence_report.verification_kind == "decision_time"
        )
        result["Evidence_Decision_Time_Eligible"] = decision_time_evidence
        if not decision_time_evidence:
            result["Data_Quality"] = (
                result["Data_Quality"]
                + "|evidence_verified_replay_not_decision_time"
            )
    result["Detector_Stage"] = "shadow"
    result["Execution_Eligible"] = False

    result.attrs["detector_stage"] = "shadow"
    result.attrs["execution_eligible"] = False
    result.attrs["input_contract"] = snapshot.schema_version
    result.attrs["input_snapshot_sha256"] = snapshot.snapshot_sha256
    result.attrs["calendar_policy_version"] = CALENDAR_POLICY_VERSION
    result.attrs["calendar_schedule_sha256"] = snapshot.schedule_sha256
    result.attrs["source_policy_version"] = snapshot.source_policy_version
    result.attrs["source_policy_sha256"] = snapshot.source_policy_sha256
    result.attrs["input_provenance_evidence_complete"] = (
        snapshot.provenance_evidence_complete
    )
    result.attrs["input_provenance_failure_codes"] = (
        [] if evidence_report is not None else provenance_failure_codes
    )
    result.attrs["evidence_manifest_sha256"] = (
        evidence_report.evidence_sha256
        if evidence_report is not None
        else None
    )
    result.attrs["evidence_verification_kind"] = (
        evidence_report.verification_kind
        if evidence_report is not None
        else None
    )
    result.attrs["evidence_verified_at"] = (
        evidence_report.verified_at
        if evidence_report is not None
        else None
    )
    result.attrs["evidence_decision_time_eligible"] = bool(
        evidence_report is not None
        and evidence_report.verification_kind == "decision_time"
    )
    if evidence_report is not None:
        result.attrs["freshness_assessed"] = bool(
            evidence_report.verification_kind == "decision_time"
        )
    return result


def detect_regimes_from_snapshot(
    snapshot: RegimeMarketDataSnapshot,
    config: RegimeDetectorConfig | None = None,
) -> pd.DataFrame:
    """Run the shadow detector without claiming durable provenance.

    Snapshot self-descriptions cannot prove that retained provider bytes still
    exist or deterministically reproduce every observation.  Call
    :func:`detect_regimes_from_verified_snapshot` when a durable evidence store
    is available.
    """

    return _detect_regimes_from_snapshot(snapshot, config, None)


def detect_regimes_from_verified_snapshot(
    snapshot: RegimeMarketDataSnapshot,
    evidence_store: RegimeEvidenceStore,
    config: RegimeDetectorConfig | None = None,
) -> pd.DataFrame:
    """Verify retained evidence in the store, then run the shadow detector."""

    if not isinstance(evidence_store, RegimeEvidenceStore):
        raise TypeError("evidence_store must be a RegimeEvidenceStore")
    report = evidence_store.verify_snapshot(snapshot)
    if not report.verified:
        failure_codes = sorted(
            {failure.split(":", 1)[0] for failure in report.failures}
        )
        raise ValueError(
            "Snapshot does not have complete durable provider evidence; "
            f"failure_codes={failure_codes}"
        )
    return _detect_regimes_from_snapshot(snapshot, config, report)
