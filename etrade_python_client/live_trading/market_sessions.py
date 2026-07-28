"""Small, side-effect-free NYSE session date helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd
from pandas_market_calendars import get_calendar


_CANONICAL_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


class MarketSessionUnavailable(RuntimeError):
    """An exact completed or prior NYSE session could not be established."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _nyse_schedule(*, start_date, end_date):
    try:
        schedule = get_calendar("NYSE").schedule(
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        raise MarketSessionUnavailable(
            "NYSE_CALENDAR_UNAVAILABLE"
        ) from exc
    if schedule.empty:
        raise MarketSessionUnavailable(
            "NYSE_SESSION_UNAVAILABLE"
        )
    return schedule


def filter_to_nyse_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered frame containing exact NYSE sessions only."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketSessionUnavailable(
            "INVALID_NYSE_MODELING_FRAME"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_MODELING_INDEX"
        )
    index = frame.index
    if (
        index.tz is not None
        or not index.is_monotonic_increasing
        or not index.is_unique
        or not index.equals(index.normalize())
    ):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_MODELING_INDEX"
        )
    schedule = _nyse_schedule(
        start_date=index[0].date(),
        end_date=index[-1].date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(
        None
    ).normalize()
    filtered = frame.loc[index.isin(sessions)].copy()
    if filtered.empty:
        raise MarketSessionUnavailable(
            "NYSE_MODELING_SESSIONS_UNAVAILABLE"
        )
    filtered.attrs.update(frame.attrs)
    filtered.attrs["nyse_modeling_calendar"] = "NYSE"
    filtered.attrs["nyse_source_row_count"] = int(len(frame))
    filtered.attrs["nyse_modeling_row_count"] = int(len(filtered))
    filtered.attrs["nyse_non_session_rows_removed"] = int(
        len(frame) - len(filtered)
    )
    return filtered


def require_nyse_session_index(index) -> None:
    """Fail unless every index row is an exact NYSE session."""

    probe = pd.DataFrame(index=index, data={"_probe": 1.0})
    filtered = filter_to_nyse_sessions(probe)
    if not filtered.index.equals(probe.index):
        raise MarketSessionUnavailable(
            "NON_NYSE_MODELING_SESSION_PRESENT"
        )


def require_open_nyse_session(check_datetime: datetime) -> datetime:
    """Return the exact close of the containing NYSE regular session."""

    if (
        type(check_datetime) is not datetime
        or check_datetime.tzinfo is not timezone.utc
    ):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_TIMESTAMP"
        )
    try:
        schedule = _nyse_schedule(
            start_date=check_datetime.date(),
            end_date=check_datetime.date(),
        )
    except MarketSessionUnavailable as exc:
        if exc.code == "NYSE_SESSION_UNAVAILABLE":
            raise MarketSessionUnavailable(
                "NYSE_REGULAR_SESSION_CLOSED"
            ) from exc
        raise
    if len(schedule.index) != 1:
        raise MarketSessionUnavailable(
            "NYSE_REGULAR_SESSION_CLOSED"
        )
    market_open = schedule.iloc[0][
        "market_open"
    ].to_pydatetime().astimezone(timezone.utc)
    market_close = schedule.iloc[0][
        "market_close"
    ].to_pydatetime().astimezone(timezone.utc)
    if not market_open <= check_datetime < market_close:
        raise MarketSessionUnavailable(
            "NYSE_REGULAR_SESSION_CLOSED"
        )
    return market_close


def latest_nyse_session_before(cutoff_date) -> str:
    """Return the final NYSE session strictly before a calendar date."""

    try:
        cutoff = pd.Timestamp(cutoff_date)
    except (TypeError, ValueError) as exc:
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_CUTOFF"
        ) from exc
    if (
        pd.isna(cutoff)
        or cutoff.tzinfo is not None
        or cutoff != cutoff.normalize()
    ):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_CUTOFF"
        )
    schedule = _nyse_schedule(
        start_date=(cutoff - pd.Timedelta(days=31)).date(),
        end_date=cutoff.date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    prior = sessions[sessions < cutoff]
    if len(prior) == 0:
        raise MarketSessionUnavailable(
            "PRIOR_NYSE_SESSION_UNAVAILABLE"
        )
    return prior[-1].strftime("%Y-%m-%d")


def latest_available_session_before(index, first_test_date) -> str:
    """Return the final available NYSE session strictly before an OOS start."""

    try:
        sessions = pd.DatetimeIndex(index)
        first_test = pd.Timestamp(first_test_date)
    except (TypeError, ValueError) as exc:
        raise MarketSessionUnavailable(
            "INVALID_CAUSAL_REVIEW_WINDOW"
        ) from exc
    if (
        len(sessions) == 0
        or sessions.tz is not None
        or not sessions.is_monotonic_increasing
        or not sessions.is_unique
        or pd.isna(first_test)
        or first_test.tzinfo is not None
        or first_test != first_test.normalize()
    ):
        raise MarketSessionUnavailable(
            "INVALID_CAUSAL_REVIEW_WINDOW"
        )
    normalized = sessions.normalize()
    if (
        not normalized.equals(sessions)
        or not normalized.is_unique
    ):
        raise MarketSessionUnavailable(
            "INVALID_CAUSAL_REVIEW_WINDOW"
        )
    schedule = _nyse_schedule(
        start_date=(first_test - pd.Timedelta(days=31)).date(),
        end_date=first_test.date(),
    )
    nyse_sessions = pd.DatetimeIndex(schedule.index).tz_localize(
        None
    ).normalize()
    candidates = normalized[
        (normalized < first_test)
        & normalized.isin(nyse_sessions)
    ]
    if len(candidates) == 0:
        raise MarketSessionUnavailable(
            "CAUSAL_CALIBRATION_PREFIX_UNAVAILABLE"
        )
    return candidates[-1].strftime("%Y-%m-%d")


def prior_nyse_session_map(session_dates) -> dict[str, str]:
    """Map each requested NYSE session to its exact prior NYSE session."""

    if (
        not isinstance(session_dates, (list, tuple))
        or not session_dates
        or any(
            not isinstance(value, str)
            or _CANONICAL_ISO_DATE.fullmatch(value) is None
            for value in session_dates
        )
    ):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_SEQUENCE"
        )
    try:
        requested = pd.DatetimeIndex(session_dates)
    except (TypeError, ValueError) as exc:
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_SEQUENCE"
        ) from exc
    if (
        requested.tz is not None
        or any(
            parsed.strftime("%Y-%m-%d") != original
            for original, parsed in zip(session_dates, requested)
        )
        or not requested.is_monotonic_increasing
        or not requested.is_unique
    ):
        raise MarketSessionUnavailable(
            "INVALID_NYSE_SESSION_SEQUENCE"
        )
    schedule = _nyse_schedule(
        start_date=(requested[0] - pd.Timedelta(days=31)).date(),
        end_date=requested[-1].date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    session_positions = {
        session: position
        for position, session in enumerate(sessions)
    }
    result = {}
    for requested_key, requested_session in zip(
        session_dates,
        requested,
    ):
        position = session_positions.get(requested_session)
        if position is None or position == 0:
            raise MarketSessionUnavailable(
                "PRIOR_NYSE_SESSION_UNAVAILABLE"
            )
        result[requested_key] = sessions[
            position - 1
        ].strftime(
            "%Y-%m-%d"
        )
    return result


def latest_completed_nyse_session(check_datetime=None) -> str:
    """Return the latest NYSE session whose regular close is in the past."""

    observed = (
        datetime.now(timezone.utc)
        if check_datetime is None
        else check_datetime
    )
    if getattr(observed, "tzinfo", None) is None:
        raise MarketSessionUnavailable(
            "LIVE_REGIME_CLOCK_MUST_BE_TIMEZONE_AWARE"
        )
    try:
        observed_at = pd.Timestamp(observed).tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise MarketSessionUnavailable(
            "INVALID_LIVE_REGIME_CLOCK"
        ) from exc
    schedule = _nyse_schedule(
        start_date=(observed_at - pd.Timedelta(days=31)).date(),
        end_date=observed_at.date(),
    )
    if "market_close" not in schedule.columns:
        raise MarketSessionUnavailable(
            "COMPLETED_NYSE_SESSION_UNAVAILABLE"
        )
    completed = schedule[
        pd.to_datetime(schedule["market_close"], utc=True) < observed_at
    ]
    if completed.empty:
        raise MarketSessionUnavailable(
            "COMPLETED_NYSE_SESSION_UNAVAILABLE"
        )
    return pd.Timestamp(completed.index[-1]).strftime("%Y-%m-%d")
