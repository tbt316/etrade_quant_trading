#!/usr/bin/env python3
"""
vol_plots.py
- Trailing 30-day realized volatility histogram (with normal fit)
- 30-DTE ATM call/put IV time series from Polygon
"""

import argparse
from datetime import datetime, timedelta, date as _date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

# Reuse your existing plumbing
from backtesting.polygonio_dailytrade import (
    get_historical_prices,
    PolygonAPIClient,
    PREMIUM_FIELD_MAP,
    pull_option_chain_data,
    calculate_implied_volatility,
)
import polygonio_config  # your API key


# ---------- Realized 30D vol ----------

def _rolling_30d_vol(df: pd.DataFrame) -> pd.Series:
    """Trailing 30-day realized volatility (std of daily returns, not annualized)."""
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    return df['returns'].rolling(30, min_periods=30).std().dropna()


def plot_trailing_30d_vol_hist(ticker: str, start_date: str, end_date: str):
    """Histogram of trailing 30D realized vol with normal curve; annotate mean/variance."""
    df = get_historical_prices(
        ticker, start_date, end_date, vol_lookback=30, data_source='yfinance'
    )
    if df is None or df.empty:
        print(f"No price data for {ticker} in {start_date}..{end_date}")
        return

    vol30 = _rolling_30d_vol(df)
    if vol30.empty:
        print("Not enough data to compute 30-day volatility.")
        return

    mu = float(np.mean(vol30))
    var = float(np.var(vol30))
    sigma = float(np.sqrt(var))

    plt.figure(figsize=(10, 6))
    n, bins, _ = plt.hist(vol30, bins=40, density=True, alpha=0.6, edgecolor='black')

    x = np.linspace(bins[0], bins[-1], 300)
    plt.plot(x, _norm.pdf(x, loc=mu, scale=sigma), linewidth=2)

    plt.title(f"{ticker} – Trailing 30-Day Realized Vol Histogram\n{start_date} to {end_date}")
    plt.xlabel("30-day realized vol (std of daily returns)")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.annotate(
        f"mean = {mu:.4f}\nvar = {var:.6f}",
        xy=(0.70, 0.80), xycoords='axes fraction',
        bbox=dict(boxstyle='round', alpha=0.2)
    )


# ---------- 30-DTE IV timeseries ----------

async def _iv_point_for_date(
    ticker: str,
    pricing_dt: _date,
    close_price: float,
    client: PolygonAPIClient,
    risk_free_rate: float,
    dividend_yield: float,
):
    """Compute ATM call & put IV for ~30 DTE using Polygon data."""
    pricing_str = pricing_dt.strftime('%Y-%m-%d')
    expiration_dt = pricing_dt + timedelta(days=30)
    expiration_str = expiration_dt.strftime('%Y-%m-%d')
    premium_field = PREMIUM_FIELD_MAP[PREMIUM_PRICE_MODE] if 'PREMIUM_PRICE_MODE' in globals() else PREMIUM_FIELD_MAP['trade']

    # Pull both calls & puts around spot
    all_call_data, all_put_data, call_options, put_options, _ = await pull_option_chain_data(
        ticker=ticker, call_put='both', expiration_str=expiration_str, as_of_str=pricing_str,
        close_price=close_price, client=client, force_otm=False, force_update=False
    )

    def _pick_atm(option_list, data_list):
        if not option_list or not data_list:
            return None, None
        best = None
        best_idx = None
        best_dist = float('inf')
        for i, (opt, prem) in enumerate(zip(option_list, data_list)):
            if prem is None:
                continue
            price = prem.get(premium_field, 0.0)
            if price and price > 0:
                d = abs(opt['strike_price'] - close_price)
                if d < best_dist:
                    best_dist = d
                    best = price
                    best_idx = i
        if best_idx is None:
            return None, None
        return best, option_list[best_idx]['strike_price']

    call_price, call_strike = _pick_atm(call_options, all_call_data)
    put_price,  put_strike  = _pick_atm(put_options,  all_put_data)

    days_to_expire = max((expiration_dt - pricing_dt).days, 1)

    iv_call = None
    iv_put = None
    if call_price and call_strike:
        iv_call = calculate_implied_volatility(
            close_price=close_price,
            strike_price=float(call_strike),
            option_price=float(call_price),
            days_to_expire=days_to_expire,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type='call',
        )
    if put_price and put_strike:
        iv_put = calculate_implied_volatility(
            close_price=close_price,
            strike_price=float(put_strike),
            option_price=float(put_price),
            days_to_expire=days_to_expire,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type='put',
        )

    return pricing_dt, iv_call, iv_put


def plot_30dte_iv_timeseries(
    ticker: str,
    start_date: str,
    end_date: str,
    max_points: int = 60,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
):
    """
    Build a time series of ATM 30-DTE call & put IV using Polygon option data.
    Samples up to `max_points` trading days evenly to reduce API load.
    """
    df_px = get_historical_prices(ticker, start_date, end_date, vol_lookback=5, data_source='yfinance')
    if df_px is None or df_px.empty:
        print(f"No price data for {ticker} in {start_date}..{end_date}")
        return

    px = df_px.dropna(subset=['close']).copy()
    if 'date' in px.columns:
        px['date_only'] = pd.to_datetime(px['date']).dt.date
    else:
        # if the helper returned index-based dates
        px['date_only'] = px.index.date
    dates = list(px['date_only'])

    if len(dates) == 0:
        print("No trading dates available for IV sampling.")
        return

    if len(dates) > max_points:
        idxs = np.linspace(0, len(dates) - 1, max_points, dtype=int)
        samp_dates = [dates[i] for i in idxs]
    else:
        samp_dates = dates

    async def _run():
        out = []
        async with PolygonAPIClient(
            api_key=polygonio_config.API_KEY, max_concurrent_requests=6, retries=1, backoff_factor=0.4
        ) as client:
            for d in samp_dates:
                close_price = float(px.loc[px['date_only'] == d, 'close'].iloc[0])
                try:
                    res = await _iv_point_for_date(
                        ticker, d, close_price, client, risk_free_rate, dividend_yield
                    )
                except Exception as e:
                    print(f"IV fetch error {d}: {e}")
                    res = (d, None, None)
                out.append(res)
        return out

    results = __import__("asyncio").get_event_loop().run_until_complete(_run())

    iv_df = pd.DataFrame(results, columns=['date', 'iv_call', 'iv_put']).dropna(subset=['date'])
    iv_df.sort_values('date', inplace=True)

    plt.figure(figsize=(11, 6))
    plt.plot(iv_df['date'], iv_df['iv_call'], label='ATM Call IV (30 DTE)')
    plt.plot(iv_df['date'], iv_df['iv_put'], label='ATM Put IV (30 DTE)')
    plt.title(f"{ticker} – 30-DTE ATM Implied Volatility\n{start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Implied Volatility (annualized)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()


# ---------- CLI ----------

def _default_5y_window():
    end = _date.today()
    start = end - timedelta(days=5*365 + 2)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def main():
    parser = argparse.ArgumentParser(
        description="Plot 30D realized vol histogram and 30DTE IV timeseries"
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 5y ago)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--max_iv_points", type=int, default=60, help="Max points for IV sampling")
    parser.add_argument("--rf", type=float, default=0.045, help="Risk-free rate (annualized)")
    parser.add_argument("--div", type=float, default=0.0, help="Dividend yield (annualized)")

    args = parser.parse_args()
    start, end = (args.start, args.end) if args.start and args.end else _default_5y_window()

    plot_trailing_30d_vol_hist(args.ticker.upper(), start, end)
    plot_30dte_iv_timeseries(
        args.ticker.upper(), start, end,
        max_points=args.max_iv_points, risk_free_rate=args.rf, dividend_yield=args.div
    )
    plt.show()


if __name__ == "__main__":
    main()