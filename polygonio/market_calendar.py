from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Literal
import bisect

import pandas as pd
import pandas_market_calendars as mcal
from .config import get_settings

# ---------------------------------------------------------
# Market calendar helpers (NYSE by default)
# ---------------------------------------------------------

@dataclass(frozen=True)
class TradingCalendar:
    name: str = "NYSE"

    def schedule(self, start: datetime, end: datetime) -> pd.DataFrame:
        cal = mcal.get_calendar(self.name)
        # pandas_market_calendars returns a schedule with market_open/market_close tz-aware timestamps
        return cal.schedule(start_date=start, end_date=end)

    def trading_dates_df(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return a DataFrame with a single 'date' column of trading datetimes.

        We use the market_close timestamp so that a "trading day" is anchored at
        end-of-day; this matches the original script's behavior that aligns expiries
        to the last trading day on or before a target weekday.
        """
        sched = self.schedule(start, end)
        # Use market_close to represent the trading day as a datetime
        df = pd.DataFrame({"date": pd.to_datetime(sched["market_close"]).to_pydatetime()})
        return df.reset_index(drop=True)


# ---------------------------------------------------------
# Expiry listing with adjustment to last trading day
# ---------------------------------------------------------

def list_expiries(
    weekday: Literal["Friday", "Wednesday"],
    start_date: datetime,
    end_date: datetime,
    trading_dates_df: pd.DataFrame,
    *,
    ticker: str | None = None,
    as_of: date | datetime | None = None,
) -> List[datetime]:
    """
    Return all target weekdays in [start_date, end_date], each adjusted to the
    **last trading day on or before** that weekday. All comparisons are done in
    date-space to avoid datetime/date comparison issues.
    """

    # --- Normalize inputs to pure date objects
    if isinstance(start_date, datetime):
        start_d = start_date.date()
    else:
        start_d = start_date  # already date
    if isinstance(end_date, datetime):
        end_d = end_date.date()
    else:
        end_d = end_date  # already date

    as_of_date = None
    if as_of is not None:
        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    else:
        as_of_date = start_d

    # --- Normalize trading days to a sorted list of date objects
    dates = pd.to_datetime(trading_dates_df["date"], errors="coerce")
    dates = dates.dt.tz_localize(None)  # strip tz if present
    trading_days: List[date] = sorted(d.date() for d in dates.dropna().to_list())

    if not trading_days:
        return []

    # --- Map weekday string to weekday index
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                   "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    if weekday not in weekday_map:
        raise ValueError(f"weekday must be 'Friday' or 'Wednesday' (got {weekday})")
    weekday_index = weekday_map[weekday]

    # --- Advance to the first target weekday within the window
    current = start_d
    while current <= end_d and current.weekday() != weekday_index:
        current += timedelta(days=1)

    # --- For each target weekday, pick the last trading day <= that weekday
    result: List[datetime] = []
    while current <= end_d:
        # bisect_right returns insertion point; -1 gives index of item <= current
        idx = bisect.bisect_right(trading_days, current) - 1
        if idx >= 0:
            last_trading_day = trading_days[idx]  # a date
            if last_trading_day >= start_d:
                # Return datetimes for backward compat (midnight)
                result.append(datetime.combine(last_trading_day, datetime.min.time()))
        current += timedelta(days=7)

    # Only emit this when calendar timing debug is enabled
    try:
        if getattr(get_settings(), "debug_calendar_timing", False):
            context_parts: List[str] = []
            if ticker:
                context_parts.append(f"ticker={ticker}")
            if as_of_date:
                context_parts.append(f"date={as_of_date.isoformat()}")
            context = f" {' '.join(context_parts)}" if context_parts else ""
            print(
                f"[DEBUG] list_expiries{context}: produced {len(result)} adjusted expiries; "
                f"first={result[0] if result else None}, last={result[-1] if result else None}"
            )
    except Exception:
        pass
    return result
