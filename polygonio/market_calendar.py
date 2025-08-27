from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Literal
import bisect

import pandas as pd
import pandas_market_calendars as mcal

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
) -> List[datetime]:
    """Return all Wednesdays/Fridays in [start_date, end_date], each adjusted to
    the **last trading day on or before** the target weekday.

    Parameters
    ----------
    weekday : {"Friday", "Wednesday"}
        Target weekly expiry weekday.
    start_date, end_date : datetime
        Inclusive range to search.
    trading_dates_df : DataFrame
        Must contain a column 'date' of trading **datetimes** (e.g., market_close).

    Notes
    -----
    Logic intentionally mirrors the legacy `get_all_weekdays` function:
    - Find the first matching weekday ≥ start_date
    - Step forward by 7 days
    - For each candidate, pick the last trading day ≤ candidate
    - Only include adjusted dates that are ≥ start_date
    """
    if weekday not in ("Friday", "Wednesday"):
        raise ValueError("weekday must be 'Friday' or 'Wednesday'")

    weekday_index = 4 if weekday == "Friday" else 2  # Mon=0 ... Sun=6

    # Ensure trading_datetimes sorted ascending
    trading_datetimes = sorted(pd.to_datetime(trading_dates_df["date"]).to_pydatetime())

    result: List[datetime] = []
    current = start_date

    # Align to the first desired weekday ≥ start_date
    while current <= end_date and current.weekday() != weekday_index:
        current += timedelta(days=1)

    # Collect adjusted expiries
    while current <= end_date:
        # bisect to find rightmost trading day ≤ current
        idx = bisect.bisect_right(trading_datetimes, current) - 1
        if idx >= 0:
            last_trading_day = trading_datetimes[idx]
            if last_trading_day >= start_date:
                result.append(last_trading_day)
        current += timedelta(days=7)

    return result

