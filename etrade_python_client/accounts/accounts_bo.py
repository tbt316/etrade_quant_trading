import json
import logging
import configparser
from turtle import position
# from order.order_bo import Order
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime,date,timedelta
import pandas as pd
import yfinance as yf
import logging
from typing import List
import random
import pytz
import os
import csv
from io import BytesIO, StringIO
import base64
import requests
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import re
from backtesting import option_limit_backtest
from data_and_research.polygonio_improvequery import load_stored_option_data  # Ensure this is importable
from live_trading.runtime_safety import (
    configure_owner_only_logger,
    redact_http_headers,
    resolve_etrade_consumer_key,
)

ROLL_IN_GL_THRESHOLD = 60
ROLL_OUT_GL_THRESHOLD = -1
ROLL_OUT_DISTANCE_THRESHOLD = 0.5
VOLATILITY_GAIN = 4
ROLL_IN_DISTANCE_THRESHOLD = VOLATILITY_GAIN * 1
HEDGE_RATIO = 2
VOLATILITY_WINDOW = 14
ROLL_IN_COST_MIN = 10
SELL_OPTION_EXPIRE_WEEK = 2
MIN_DTE = 2
CHECK_EARNING_DATE = False


CORRELATED_STOCKS={
    'semiconductor':['NVDA','AMD','QCOM']
}

EXEMPTED_STOCKS=['SQQQ','VIX']


def _redact_account_identifier(value):
    text = str(value or "")
    return f"***{text[-4:]}" if text else "[unavailable]"

# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# logger settings
logger = configure_owner_only_logger('my_logger')

ETRADE_TICKER=["VIXW","VIX","BRKB","BRK.B","SPX"]
YFINANCE_TICKER=["^VIX","^VIX","BRK-B","BRK-B","^SPX"]

def convert_ticker_name(etrade_ticker=None, yfinance_ticker=None):
    """
    Convert a ticker name between ETRADE_TICKER and YFINANCE_TICKER.
    
    Parameters:
    - etrade_ticker (str, optional): Ticker name in ETRADE format.
    - yfinance_ticker (str, optional): Ticker name in yfinance format.

    Returns:
    - str: The converted ticker name if found, or None if not found.
    """
    if etrade_ticker:
        # Check for custom option format: SYMBOL:YYYY:MM:DD:TYPE:STRIKE
        if ":" in etrade_ticker:
            parts = etrade_ticker.split(":")
            if len(parts) >= 6:
                try:
                    symbol = parts[0]
                    year = parts[1][-2:]
                    month = parts[2].zfill(2)
                    day = parts[3].zfill(2)
                    type_char = "C" if "CALL" in parts[4].upper() else "P"
                    strike = float(parts[5])
                    strike_str = f"{int(strike * 1000):08d}"
                    return f"{symbol}{year}{month}{day}{type_char}{strike_str}"
                except Exception:
                    pass # Fallback to original if parsing fails
                    
        # Find the corresponding yfinance ticker
        if etrade_ticker in ETRADE_TICKER:
            index = ETRADE_TICKER.index(etrade_ticker)
            return YFINANCE_TICKER[index]
        else:
            return etrade_ticker
    
    if yfinance_ticker:
        # Find the corresponding etrade ticker
        if yfinance_ticker in YFINANCE_TICKER:
            index = YFINANCE_TICKER.index(yfinance_ticker)
            return ETRADE_TICKER[index]
        else:
            # print(f"{yfinance_ticker} not found in YFINANCE_TICKER.")
            return yfinance_ticker

    print("Either etrade_ticker or yfinance_ticker must be provided.")
    return None


def _float_or_zero(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _option_quote_mark(quote_data):
    all_data = quote_data.get("All", {})
    quote_status = str(quote_data.get("quoteStatus") or all_data.get("quoteStatus") or "").upper()
    bid = _float_or_zero(all_data.get("bid"))
    ask = _float_or_zero(all_data.get("ask"))
    bid_size = _float_or_zero(all_data.get("bidSize"))
    ask_size = _float_or_zero(all_data.get("askSize"))
    last_trade = _float_or_zero(all_data.get("lastTrade"))

    if last_trade > 0 and (quote_status == "CLOSING" or (bid_size == 0 and ask_size == 0)):
        return last_trade
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2.0, 4)
    if last_trade > 0:
        return last_trade
    return 0.0


def _timestamp_seconds(value):
    try:
        timestamp = int(value)
        return timestamp // 1000 if timestamp > 10_000_000_000 else timestamp
    except (TypeError, ValueError):
        return None


def _market_quote_price(quote_data):
    all_data = quote_data.get("All", {})
    extended_hours = all_data.get("ehQuote") or {}
    regular_price = _float_or_zero(all_data.get("lastTrade"))
    extended_price = _float_or_zero(extended_hours.get("lastPrice"))
    regular_time = _timestamp_seconds(all_data.get("timeOfLastTrade"))
    extended_time = _timestamp_seconds(extended_hours.get("timeOfLastTrade"))

    if extended_price > 0 and (
        regular_price <= 0
        or (
            extended_time is not None
            and regular_time is not None
            and extended_time > regular_time
        )
    ):
        return extended_price
    return regular_price or extended_price


def _quote_metadata(quote_data, price, source="E*TRADE"):
    all_data = quote_data.get("All", {})
    extended_hours = all_data.get("ehQuote") or {}
    timestamp = (
        extended_hours.get("timeOfLastTrade")
        if _float_or_zero(extended_hours.get("lastPrice")) == _float_or_zero(price)
        else all_data.get("timeOfLastTrade")
    )
    timestamp = timestamp or quote_data.get("dateTimeUTC")
    timestamp = _timestamp_seconds(timestamp)

    return {
        "price": _float_or_zero(price),
        "source": source,
        "status": str(quote_data.get("quoteStatus") or "UNKNOWN").upper(),
        "timestamp": timestamp,
        "date_time": str(quote_data.get("dateTime") or ""),
    }


def is_etrade_token_expired_response(response):
    if response is None or getattr(response, "status_code", None) != 401:
        return False
    error_text = getattr(response, "text", "") or ""
    return "oauth_problem=token_expired" in error_text or "token_expired" in error_text


def convert_standard_to_nd2_delta(delta: float, iv: float, days_to_expiration: float, call_put: str) -> float:
    """
    Convert E*TRADE API's standard delta (approx N(d1)) to the risk-neutral ITM probability delta:
    - Calls: N(d2)
    - Puts: -N(-d2)
    
    Formula:
      d1 = ppf(abs(delta)) for calls, or -ppf(abs(delta)) for puts (assuming q=0)
      d2 = d1 - iv * sqrt(T)
      Calls return N(d2)
      Puts return -N(-d2)
    """
    import numpy as np
    from scipy.stats import norm
    
    # Scale IV if it is in percentage format (e.g. 25.0 instead of 0.25)
    if iv > 2.0:
        iv = iv / 100.0
    iv = max(iv, 1e-4)
    
    T = max(days_to_expiration, 1e-5) / 365.25
    
    is_call = call_put.upper() in ["CALL", "C"]
    
    # Handle edge cases where delta is 0 or +/-1
    if delta == 0:
        return 0.0
    if abs(delta) >= 1.0:
        return 1.0 if delta > 0 else -1.0
        
    if is_call:
        # Calls: delta = N(d1)
        d_clipped = max(1e-7, min(delta, 1.0 - 1e-7))
        d1 = norm.ppf(d_clipped)
        d2 = d1 - iv * np.sqrt(T)
        return float(norm.cdf(d2))
    else:
        # Puts: delta = -N(-d1) -> abs(delta) = N(-d1)
        abs_delta = abs(delta)
        d_clipped = max(1e-7, min(abs_delta, 1.0 - 1e-7))
        d1_neg = norm.ppf(d_clipped) # This is -d1
        # -d2 = -d1 + iv * sqrt(T) = d1_neg + iv * sqrt(T)
        neg_d2 = d1_neg + iv * np.sqrt(T)
        return float(-norm.cdf(neg_d2))


def _select_nearest_expiration(available_expirations, target_expiration):
    """Select the listed expiration closest to the target date, preferring later dates on ties."""
    if not available_expirations:
        return None

    today = datetime.today().date()
    parsed = [
        datetime.strptime(exp, "%Y-%m-%d").date() if isinstance(exp, str) else exp
        for exp in available_expirations
    ]
    candidates = [exp for exp in parsed if exp >= today]
    if not candidates:
        candidates = parsed
    friday_candidates = [exp for exp in candidates if exp.weekday() == 4]
    if friday_candidates:
        candidates = friday_candidates

    return min(
        candidates,
        key=lambda exp: (abs((exp - target_expiration).days), 0 if exp >= target_expiration else 1, exp)
    )


def _symbol_from_osi_key(default_symbol, osi_key):
    if not osi_key or default_symbol != "SPX":
        return default_symbol

    osi_text = str(osi_key).upper()
    if osi_text.startswith("SPXW"):
        return "SPXW"
    if osi_text.startswith("SPX-"):
        return "SPX"
    return default_symbol


def _aggregate_option_symbol(symbol):
    symbol = (symbol or "").upper()
    return "SPX" if symbol == "SPXW" else symbol


def _missing_market_close_dates(chart_dates, price_cache, today_str, market_active):
    missing_dates = []
    for chart_date in chart_dates:
        if chart_date.endswith(" (Live)"):
            continue

        date_key = chart_date.replace(" (Live)", "")
        has_missing_close = any(
            date_key not in price_cache.get(symbol, {})
            or pd.isna(price_cache.get(symbol, {}).get(date_key))
            for symbol in ("SPY", "SPX", "VIX")
        )
        if has_missing_close or (date_key == today_str and not market_active):
            missing_dates.append(date_key)
    return missing_dates


def _extract_cboe_vix_closes(csv_text, requested_dates):
    table = pd.read_csv(StringIO(csv_text), usecols=["DATE", "CLOSE"])
    table["DATE"] = pd.to_datetime(table["DATE"], format="%m/%d/%Y", errors="coerce")
    requested_dates = set(requested_dates)
    return {
        row.DATE.strftime("%Y-%m-%d"): round(float(row.CLOSE), 2)
        for row in table.itertuples(index=False)
        if not pd.isna(row.DATE)
        and not pd.isna(row.CLOSE)
        and row.DATE.strftime("%Y-%m-%d") in requested_dates
    }


def calculate_std_dev(ticker, lookback_window):
    """
    Calculate the standard deviation of daily returns (percentage) for a given ticker symbol
    over a specified number of days, including the latest market quote.
    
    Parameters:
    - ticker (str): The stock ticker symbol
    - days (int): The number of days for which to calculate the standard deviation
    
    Returns:
    - float: Standard deviation of daily returns as a percentage
    """
    # Define the time period based on the number of days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=252)
    
    # Fetch historical data for the ticker
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Calculate daily returns as a percentage
    data['Daily Return (%)'] = data['Close'].pct_change() * 100

    # Calculate standard deviation of the daily returns (ignoring the first NaN value from pct_change)
    std_dev = round(data['Daily Return (%)'][-lookback_window:].std(), 3)
    
    ####debug#########
                # print("Daily close: ", ticker, data['Adj Close'])
                # print("Daily return: ", data['Daily Return (%)'])
                # print("Standard deviation: ", std_dev)
                # breakpoint()

    return std_dev

def process_folder_and_plot_risks_combined(folder_path):
    """
    Process all CSV files in a folder and return HTML image tags for upside and downside risk plots,
    with multiple scatter lines representing each ticker symbol.

    :param folder_path: Path to the folder containing CSV files named as "YYYY-MM-DD.csv".
    :return: HTML strings for upside and downside risk plots.
    """
    daily_risks = []

    # Iterate over all files in the folder
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.csv'):
            date = datetime.strptime(file.replace('.csv', ''), "%Y-%m-%d")
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Filter for options only
            options = df[df['Type'] == 'Option']

            options = options.copy()
            options['Symbol'] = options['Symbol'].map(_aggregate_option_symbol)

            # Calculate risks by ticker
            grouped = options.groupby(['Symbol', 'Call/Put']).apply(
                lambda group: (abs(group['Delta']) * abs(group['Quantity']) * group['Asset Price']).sum()
            ).reset_index(name='Risk')

            for _, row in grouped.iterrows():
                daily_risks.append({
                    'Date': date,
                    'Symbol': row['Symbol'],
                    'Call/Put': row['Call/Put'],
                    'Risk': row['Risk']
                })

    # Create a DataFrame for all risks
    if not daily_risks:
        return "<p>No daily risk data found.</p>", "<p>No daily risk data found.</p>"

    risks_df = pd.DataFrame(daily_risks)

    # Pivot data for plotting
    upside_df = risks_df[risks_df['Call/Put'] == 'CALL'].pivot(index='Date', columns='Symbol', values='Risk').fillna(0)
    downside_df = risks_df[risks_df['Call/Put'] == 'PUT'].pivot(index='Date', columns='Symbol', values='Risk').fillna(0)

    # Generate the Upside Risk plot
    plt.figure(figsize=(10, 6))
    for symbol in upside_df.columns:
        plt.plot(upside_df.index, upside_df[symbol], label=symbol, marker='o')
    plt.title('Daily Upside Risk by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Upside Risk')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    upside_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Generate the Downside Risk plot
    plt.figure(figsize=(10, 6))
    for symbol in downside_df.columns:
        plt.plot(downside_df.index, downside_df[symbol], label=symbol, marker='o')
    plt.title('Daily Downside Risk by Ticker')
    plt.xlabel('Date')
    plt.ylabel('Downside Risk')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    downside_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Return HTML strings
    return f"<img src='data:image/png;base64,{upside_image_base64}'/>", f"<img src='data:image/png;base64,{downside_image_base64}'/>"


def _spy_margin_totals_external(screened_options):
    """
    Calculate SPY margin totals from screened options.
    
    This is an externally-callable version of the margin calculation
    logic used in render_screened_option_pairs_html.
    
    Args:
        screened_options: List of option pair dictionaries from screen_option()
        
    Returns:
        Tuple of (spy_call_margin, spy_put_margin)
    """
    from collections import defaultdict
    
    def _as_num(v, default=0.0):
        try:
            if v is None:
                return float(default)
            return float(v)
        except Exception:
            return float(default)
    
    def _naked_margin(short_leg, qty_abs):
        """Return margin requirement per short leg outside of spreads."""
        if short_leg is None or qty_abs <= 0:
            return 0.0
        cp = getattr(short_leg, "call_put", "") or ""
        cp = cp.upper()
        strike = _as_num(getattr(short_leg, "strike_price", None), 0.0)
        if strike <= 0:
            return 0.0
        if cp == "CALL":
            return strike * 100.0 * qty_abs
        if cp == "PUT":
            underlying = _as_num(getattr(short_leg, "underlying_last_price", None), strike)
            if underlying <= 0:
                underlying = strike
            premium = _as_num(getattr(short_leg, "last_price", None), 0.0)
            calc1 = 0.2 * underlying - (strike - underlying) + premium
            calc2 = 0.1 * strike + premium
            margin_per_contract = max(calc1, calc2, 0.0)
            return margin_per_contract * 100.0 * qty_abs
        return 0.0

    groups = defaultdict(lambda: {"longs": [], "shorts": []})
    for entry in screened_options:
        for leg in (entry.get("long_lot"), entry.get("short_lot")):
            if not leg:
                continue
            sym = _aggregate_option_symbol(getattr(leg, "symbol", ""))
            if sym not in ("SPY", "SPX"):
                continue
            cp = (getattr(leg, "call_put", "") or "").upper()
            if cp not in ("CALL", "PUT"):
                continue
            exp = getattr(leg, "expiration_date", None)
            qty = _as_num(getattr(leg, "quantity", 0), 0.0)
            if qty == 0:
                continue
            bucket = groups[(sym, cp, exp)]
            bucket["longs" if qty > 0 else "shorts"].append({"leg": leg, "qty": abs(qty)})

    from collections import defaultdict
    totals_per_expiry = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0})

    def _strike(rec):
        return _as_num(getattr(rec["leg"], "strike_price", None), 0.0)

    for (sym, cp, exp), parts in groups.items():
        shorts = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["shorts"]]
        if not shorts:
            continue
        longs = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["longs"]]

        reverse = True if cp == "PUT" else False
        shorts.sort(key=lambda rec: _strike(rec), reverse=reverse)
        longs.sort(key=lambda rec: _strike(rec), reverse=reverse)

        for short in shorts:
            while short["qty"] > 0 and longs:
                best_idx = min(
                    range(len(longs)),
                    key=lambda idx: abs(_strike(short) - _strike(longs[idx]))
                )
                long = longs[best_idx]
                pair_qty = min(short["qty"], long["qty"])
                if pair_qty <= 0:
                    if long["qty"] <= 0:
                        longs.pop(best_idx)
                    continue
                strike_diff = abs(_strike(short) - _strike(long))
                totals_per_expiry[(sym, exp)][cp] += strike_diff * pair_qty * 100.0
                short["qty"] -= pair_qty
                long["qty"] -= pair_qty
                if long["qty"] <= 0:
                    longs.pop(best_idx)

            if short["qty"] > 0:
                totals_per_expiry[(sym, exp)][cp] += _naked_margin(short["leg"], short["qty"])

    # Calculate total call and put sums for backward compatibility, 
    # and a corrected total margin using the max rule per expiry.
    total_call_margin = sum(v["CALL"] for v in totals_per_expiry.values())
    total_put_margin = sum(v["PUT"] for v in totals_per_expiry.values())
    total_corrected_margin = sum(max(v["CALL"], v["PUT"]) for v in totals_per_expiry.values())

    return total_corrected_margin, total_call_margin, total_put_margin


def calculate_margin(stock_positions, cover_call_list=None):
    """
    Calculate margin requirements for option positions in the portfolio.
    
    This function identifies option pairs with the same ticker, expiration date, and opposite 
    positions (long/short), then calculates the required margin as the difference in strike 
    prices times 100 times the quantity.
    
    For naked short call positions, if the ticker is in the cover_call_list, the specified
    quantity is exempt from margin calculation (covered calls) and excluded from the output.
    
    Args:
        stock_positions (list): List of StockPosition objects from the portfolio function.
        cover_call_list (dict): Dictionary mapping ticker symbols to number of exempt covered call contracts.
                              Default is None.
        
    Returns:
        tuple: (Total margin required, Dictionary of margin details by ticker)
    """
    # Initialize cover_call_list if None
    if cover_call_list is None:
        cover_call_list = {}
    
    # Track remaining covered call exemptions
    remaining_exemptions = cover_call_list.copy()
    
    # Group positions by ticker
    positions_by_ticker = {}
    for position in stock_positions:
        # Only consider SPY option positions
        if position.security_type == "Option":
            ticker = _aggregate_option_symbol(position.symbol)
            if ticker not in ("SPY", "SPX"):
                continue
            if ticker not in positions_by_ticker:
                positions_by_ticker[ticker] = []
            positions_by_ticker[ticker].append(position)
    
    total_margin = 0
    margin_details = {}
    
    # Process each ticker symbol
    for ticker, positions in positions_by_ticker.items():
        ticker_margin = 0
        ticker_pairs = []
        
        # Check if this ticker is in the cover_call_list
        is_covered_ticker = ticker in cover_call_list
        
        # Group positions by expiration date
        expiry_groups = {}
        for position in positions:
            key = position.expiration_date
            if key not in expiry_groups:
                expiry_groups[key] = []
            expiry_groups[key].append(position)
        
        # Process each expiration date group
        for expiry, expiry_positions in expiry_groups.items():
            # For each expiry, we calculate CALL and PUT margin separately, then take the max
            
            # Helper to calculate margin for a specific type (CALL or PUT) within an expiry
            def _calculate_type_margin(type_positions, option_type):
                type_margin = 0
                type_pairs = []
                
                # Separate long and short positions
                long_positions = [p for p in type_positions if p.quantity > 0]
                short_positions = [p for p in type_positions if p.quantity < 0]
                
                # Match long and short positions to form spreads
                for long_pos in long_positions:
                    for short_pos in short_positions:
                        if abs(short_pos.quantity) <= 0:
                            continue
                        
                        pair_quantity = min(long_pos.quantity, abs(short_pos.quantity))
                        if pair_quantity <= 0:
                            continue
                        
                        strike_diff = abs(long_pos.strike_price - short_pos.strike_price)
                        pair_margin = strike_diff * 100 * pair_quantity
                        type_margin += pair_margin
                        
                        type_pairs.append({
                            "ticker": ticker,
                            "expiry": expiry,
                            "type": option_type,
                            "long_strike": long_pos.strike_price,
                            "short_strike": short_pos.strike_price,
                            "quantity": pair_quantity,
                            "margin": pair_margin,
                            "is_spread": True,
                            "long_price": long_pos.last_price,
                            "short_price": short_pos.last_price
                        })
                        
                        long_pos.quantity -= pair_quantity
                        short_pos.quantity += pair_quantity
                
                # Handle remaining naked short positions
                for short_pos in short_positions:
                    if short_pos.quantity < 0:
                        naked_quantity = abs(short_pos.quantity)
                        exempt_quantity = 0
                        
                        if option_type == "CALL" and ticker in remaining_exemptions and remaining_exemptions[ticker] > 0:
                            exempt_quantity = min(naked_quantity, remaining_exemptions[ticker])
                            remaining_exemptions[ticker] -= exempt_quantity
                            
                        margin_quantity = naked_quantity - exempt_quantity
                        
                        if is_covered_ticker and option_type == "CALL" and margin_quantity == 0:
                            continue
                        
                        if margin_quantity > 0:
                            if option_type == "PUT":
                                underlying_price = getattr(short_pos, 'underlying_last_price', 0)
                                premium = abs(short_pos.price_paid) if hasattr(short_pos, 'price_paid') and short_pos.price_paid != 0 else short_pos.last_price
                                calc1 = (0.2 * underlying_price - (underlying_price - short_pos.strike_price) + premium)
                                calc2 = (0.1 * short_pos.strike_price + premium)
                                per_contract_margin = max(calc1, calc2) * 100
                                naked_margin = per_contract_margin * margin_quantity
                            else:
                                naked_margin = short_pos.strike_price * 100 * margin_quantity
                                
                            type_margin += naked_margin
                            
                            type_pairs.append({
                                "ticker": ticker,
                                "expiry": expiry,
                                "type": option_type,
                                "short_strike": short_pos.strike_price,
                                "quantity": margin_quantity,
                                "margin": naked_margin,
                                "is_spread": False,
                                "is_covered": False,
                                "per_contract": naked_margin / margin_quantity,
                                "underlying_price": getattr(short_pos, 'underlying_last_price', 0) if option_type == "PUT" else None,
                                "premium": abs(short_pos.price_paid) if hasattr(short_pos, 'price_paid') and short_pos.price_paid != 0 else short_pos.last_price if option_type == "PUT" else None,
                                "option_price": short_pos.last_price
                            })
                        
                        if exempt_quantity > 0 and not is_covered_ticker:
                            type_pairs.append({
                                "ticker": ticker,
                                "expiry": expiry,
                                "type": option_type,
                                "short_strike": short_pos.strike_price,
                                "quantity": exempt_quantity,
                                "margin": 0,
                                "is_spread": False,
                                "is_covered": True,
                                "option_price": short_pos.last_price
                            })
                return type_margin, type_pairs

            # Separate positions by type for this expiry
            call_positions = [p for p in expiry_positions if p.call_put == "CALL"]
            put_positions = [p for p in expiry_positions if p.call_put == "PUT"]

            # Calculate margins for both types
            call_m, call_p = _calculate_type_margin(call_positions, "CALL")
            put_m, put_p = _calculate_type_margin(put_positions, "PUT")
            
            # The margin for this expiry is the MAX of call and put sides
            expiry_margin = max(call_m, put_m)
            ticker_margin += expiry_margin
            ticker_pairs.extend(call_p)
            ticker_pairs.extend(put_p)
        
        # Store the ticker's margin details if there are any pairs with margin
        if ticker_pairs:
            margin_details[ticker] = {
                "total_margin": ticker_margin,
                "pairs": ticker_pairs
            }
            total_margin += ticker_margin
    
    return total_margin, margin_details


def find_highest_margin_ratios(margin_details, already_closing=None, top_n=3):
    """
    Find the top option pairs with the highest margin ratios across all tickers.
    Excludes positions that are already being closed.
    
    Margin ratio is defined as margin / (short_price * 100 * quantity), which helps
    identify positions where the margin requirement is high relative to the option premium.
    
    Args:
        margin_details (dict): Dictionary of margin details by ticker, as returned by calculate_margin.
        already_closing (set): Set of position identifiers (ticker, expiry, type, short_strike, long_strike)
                               that are already being closed. Default is None.
        top_n (int): Number of top candidates to return. Default is 10.
        
    Returns:
        list: List of tuples (ticker, pair_details, margin_ratio) sorted in descending order by margin_ratio.
              Returns an empty list if no valid pairs are found.
    """
    if already_closing is None:
        already_closing = set()
        
    candidates = []
    # Iterate through each ticker in margin_details
    for ticker, ticker_data in margin_details.items():
        pairs = ticker_data.get("pairs", [])
        
        # Examine each pair for this ticker
        for pair in pairs:
            if pair.get("is_spread", False):
                short_price = pair.get("short_price")
                long_price = pair.get("long_price")
                margin = pair.get("margin")
                quantity = pair.get("quantity", 1)
                
                # Check for valid prices and that the short price is greater or equal to the long price
                if short_price is not None and short_price > 0 and short_price < 0.03 and margin is not None and short_price >= long_price:
                    # Create a unique identifier for this position
                    position_id = (
                        ticker,
                        pair.get("expiry"),
                        pair.get("type"),
                        pair.get("short_strike"),
                        pair.get("long_strike")
                    )
                    
                    # Skip if this position is already being closed
                    if position_id in already_closing:
                        continue
                    
                    # Calculate the margin ratio per contract
                    ratio = margin / ((short_price + 0.01)* 100 * quantity)  # add $0.01 to increase chance of order filled
                    candidates.append((ticker, pair, ratio))
    
    # Sort candidates by ratio descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n]

def print_margin_report(total_margin, margin_details, cover_call_list=None):
    """
    Print a formatted report of margin requirements for the portfolio.
    
    Args:
        total_margin (float): Total margin required for the portfolio.
        margin_details (dict): Dictionary of margin details by ticker.
        cover_call_list (dict): Dictionary mapping ticker symbols to number of exempt covered call contracts.
    """
    print("\n===== PORTFOLIO MARGIN REPORT =====")
    print(f"Total Margin Required: ${total_margin:,.2f}")
    
    # Print covered call exemptions if provided
    if cover_call_list:
        print("\nCovered Call Exemptions:")
        for ticker, quantity in cover_call_list.items():
            print(f"  {ticker}: {quantity} contracts")
    
    print("\nBreakdown by Ticker:")
    
    for ticker, details in margin_details.items():
        print(f"\n{ticker}: ${details['total_margin']:,.2f}")
        
        if details['pairs']:
            spreads = [p for p in details['pairs'] if p.get('is_spread', True)]
            naked_shorts = [p for p in details['pairs'] if not p.get('is_spread', True) and not p.get('is_covered', False)]
            covered_calls = [p for p in details['pairs'] if p.get('is_covered', False)]
            
            # Print spreads
            if spreads:
                print("  Spreads:")
                for i, pair in enumerate(spreads, 1):
                    expiry_str = pair['expiry'].strftime('%Y-%m-%d')
                    print(f"  {i}. {pair['type']} Spread - Expiry: {expiry_str}")
                    print(f"     Long Strike: ${pair['long_strike']:.2f}, Short Strike: ${pair['short_strike']:.2f}")
                    print(f"     Quantity: {pair['quantity']}, Margin: ${pair['margin']:,.2f}")
            
            # Print naked shorts
            if naked_shorts:
                print("  Naked Short Positions:")
                for i, short in enumerate(naked_shorts, 1):
                    expiry_str = short['expiry'].strftime('%Y-%m-%d')
                    print(f"  {i}. Naked {short['type']} - Expiry: {expiry_str}")
                    print(f"     Strike: ${short['short_strike']:.2f}")
                    
                    # Additional info for puts
                    if short['type'] == "PUT" and short.get('underlying_price') is not None:
                        print(f"     Underlying Price: ${short['underlying_price']:.2f}, Premium: ${short['premium']:.2f}")
                        print(f"     Margin per Contract: ${short['per_contract']:.2f}")
                    
                    print(f"     Quantity: {short['quantity']}, Total Margin: ${short['margin']:,.2f}")
            
            # Print covered calls (exempt from margin)
            if covered_calls:
                print("  Covered Calls (Exempt from Margin):")
                for i, covered in enumerate(covered_calls, 1):
                    expiry_str = covered['expiry'].strftime('%Y-%m-%d')
                    print(f"  {i}. Covered Call - Expiry: {expiry_str}")
                    print(f"     Strike: ${covered['short_strike']:.2f}")
                    print(f"     Quantity: {covered['quantity']}, Margin: $0.00")

def actions_option_trade_html(stock_positions, folder_path='./daily_log', overwrite=True):
    """
    Generate an HTML log file for option trades, including daily risk plots, with clear ticker segregation.

    :param stock_positions: List of StockPosition objects (some may be invalid)
    :param folder_path: Path to the folder containing CSV files for daily risk analysis
    :param overwrite: Boolean indicating whether to overwrite the log file
    """
    # Filter stock_positions to include only valid StockPosition objects with symbol and call_put
    filtered_positions = [
        pos for pos in stock_positions 
        if isinstance(pos, StockPosition) and hasattr(pos, 'symbol') and hasattr(pos, 'call_put') and pos.call_put is not None
    ]

    # Sort filtered positions by ticker (symbol) first, then by call_put (PUT before CALL)
    sorted_stock_positions = sorted(filtered_positions, key=lambda pos: (pos.symbol, pos.call_put))

    icloud_drive_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
    os.makedirs(icloud_drive_path, exist_ok=True)
    log_filename = os.path.join(icloud_drive_path, "option_trade_log.html") if overwrite else \
        os.path.join(icloud_drive_path, f"option_trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

    try:
        upside_plot_html, downside_plot_html = process_folder_and_plot_risks_combined(folder_path)
    except Exception as e:
        print(f"Error generating risk plots: {e}")
        upside_plot_html = "<p>Error generating upside risk plot.</p>"
        downside_plot_html = "<p>Error generating downside risk plot.</p>"

    # Write the HTML log file
    with open(log_filename, "w") as log_file:
        # Write HTML header and enhanced CSS
        log_file.write("<html><head><style>")
        log_file.write(".positive { color: green; } .negative { color: red; }")
        log_file.write(".ticker-separator td { border-top: 6px solid black; }") # Apply border to cells in separator rows
        log_file.write("table { border-collapse: separate; border-spacing: 2px; width: 100%; }")  # Ensure spacing
        log_file.write("td, th { border: 1px solid gray; padding: 5px; }")  # Consistent cell borders
        log_file.write("</style></head><body><h2>Options Action Items</h2><table>")
        total_covered_asset = 0  
        total_covered_margin = 0  
        total_sold = 0
        total_itm = 0

        # First table: Options to be rolled
        log_file.write("<tr><th>Symbol</th><th>QTY</th><th>Price</th><th>Type</th><th>Gain/Loss</th><th>DTE</th><th>Strike Distance</th><th>Volatility</th><th>Strike</th><th>Expiration</th><th>Roll Cost</th><th>Strike</th><th>Expire</th><th>IV</th><th>theta chg</th></tr>")
        prev_symbol = None
        for i, pos in enumerate(sorted_stock_positions):
            if pos.target_option_roll is not None:
                # Apply ticker-separator class when ticker changes
                row_class = "ticker-separator" if prev_symbol is not None and pos.symbol != prev_symbol else ""
                prev_symbol = pos.symbol

                gain_loss_class = "positive" if pos.gain_loss_percentage >= 0 else "negative"
                distance_to_strike_class = (
                    "positive" if (pos.call_put == "PUT" and pos.distance_to_strike > 0) or 
                                 (pos.call_put == "CALL" and pos.distance_to_strike < 0)
                    else "negative"
                )
                cost_for_roll_class = "positive" if pos.target_option_roll['cost_for_roll'] >= 0 else "negative"
                theta_chg = round(pos.target_option_roll['theta'] - pos.theta, 4)
                theta_chg_class = "positive" if theta_chg < 0 else "negative"
                underlying_last_price_class = "positive" if pos.option_intrinsic < pos.last_price else "negative"
            
                log_file.write(f"<tr class='{row_class}'>")
                log_file.write(f"<td>{pos.symbol}</td>")
                log_file.write(f"<td>{pos.quantity}</td>")
                log_file.write(f"<td class='{underlying_last_price_class}'>{pos.underlying_last_price}</td>")
                log_file.write(f"<td>{pos.call_put}</td>")
                log_file.write(f"<td class='{gain_loss_class}'>{pos.gain_loss_percentage}%</td>")
                log_file.write(f"<td>{pos.days_to_expiration}</td>")
                log_file.write(f"<td class='{distance_to_strike_class}'>{pos.distance_to_strike}%</td>")
                log_file.write(f"<td>{pos.volatility}%</td>")
                log_file.write(f"<td>{pos.strike_price}</td>")
                log_file.write(f"<td>{pos.expiration_date}</td>")
                log_file.write(f"<td class='{cost_for_roll_class}'>{pos.target_option_roll['cost_for_roll'] * abs(pos.quantity)}</td>")
                log_file.write(f"<td>{pos.target_option_roll['strikePrice']}</td>")
                log_file.write(f"<td>{pos.target_option_roll['expiryDate']}</td>")
                log_file.write(f"<td>{pos.target_option_roll['implied_volatility']}%</td>")
                log_file.write(f"<td class='{theta_chg_class}'>{theta_chg}</td>")
                log_file.write("</tr>")

        log_file.write("</table><br><h2>All Available Options</h2><table>")
        
        # Second table: All available options
        log_file.write("<tr><th>Symbol</th><th>QTY</th><th>Asset Price</th><th>Price</th><th>Type</th><th>Gain/Loss</th><th>DTE</th><th>Strike Distance</th><th>Volatility</th><th>Strike</th><th>Expiration</th><th>IV</th><th>delta</th><th>delta</th><th>hedge</th><th>net_delta</th><th>gamma</th><th>gamma</th><th>net_gamma</th><th>theta</th><th>theta_pct</th></tr>")
        daily_time_decay = 0 
        prev_symbol = None
        for i, pos in enumerate(sorted_stock_positions):
            # if pos.security_type == "Option" and pos.quantity < 0:
            if pos.security_type == "Option":
                # Apply ticker-separator class when ticker changes
                row_class = "ticker-separator" if prev_symbol is not None and pos.symbol != prev_symbol else ""
                prev_symbol = pos.symbol

                if abs(pos.theta) < 999: 
                    daily_time_decay += pos.theta * abs(pos.quantity) * 100
                if pos.call_put == "CALL":
                    total_covered_asset += pos.underlying_last_price * 100 * abs(pos.quantity)
                if pos.call_put == "PUT":
                    total_covered_margin += pos.underlying_last_price * 100 * abs(pos.quantity)
                total_sold += pos.last_price * 100 * (pos.quantity)
                exclude_list = ['TQQQ','GOOGL','AAPL','ARKK','JPM','NVDA']
                today = datetime.now()
                coming_friday = today + timedelta(days=((4 - today.weekday()) % 7 ))  # Next Friday
                if pos.symbol not in exclude_list and pos.expiration_date == coming_friday.date():
                    total_itm -= max(pos.option_intrinsic,0) * 100 * pos.quantity
                if abs(pos.delta) > 1 or abs(pos.net_ticker_delta) > 1000:
                    print(f"Delta/Gamma out of range: {pos.symbol}, {pos.delta}, {pos.gamma}, {pos.net_ticker_delta}, {pos.net_ticker_gamma}")
                    pos.delta = 0
                    pos.gamma = 0
                    pos.net_ticker_delta = 0
                    pos.net_ticker_gamma = 0
                if abs(pos.implied_volatility) > 100:
                    pos.implied_volatility = 0
                if abs(pos.theta) > 1:
                    pos.theta = 0
                gain_loss_class = "positive" if pos.gain_loss_percentage >= 0 else "negative"
                distance_to_strike_class = (
                    "positive" if (pos.call_put == "PUT" and pos.distance_to_strike > 0) or 
                                 (pos.call_put == "CALL" and pos.distance_to_strike < 0)
                    else "negative"
                )
                volatility_class = "positive" if pos.volatility >= 0.7 * pos.implied_volatility else "negative"
                net_delta_class = "positive" if abs(pos.net_ticker_delta) <= 1 else "negative"
                net_gamma_class = "positive" if abs(pos.net_ticker_gamma) <= 0.1 else "negative"
                this_delta = pos.quantity * pos.delta
                this_gamma = pos.quantity * pos.gamma              
                delta_class = "positive" if abs(pos.delta) <= 0.1 else "negative"
                gamma_class = "positive" if abs(pos.gamma) <= 0.1 else "negative"

                if abs(pos.theta) < 999: 
                    if pos.underlying_last_price == 0:
                        print(f"Underlying asset: {pos.symbol}, {pos.underlying_last_price}")
                        theta_pct = 0
                    else:
                        theta_pct = round(pos.theta / pos.underlying_last_price * 10000, 2)
                else:
                    theta_pct = 0

                theta_pct_class = "positive" if theta_pct < -2 else "negative"
                underlying_last_price_class = "positive" if pos.option_intrinsic * 1.2 < pos.last_price else "negative"
                in_the_money_class = "positive" if pos.option_intrinsic < 0 else "negative"

                log_file.write(f"<tr class='{row_class}'>")
                log_file.write(f"<td>{pos.symbol}</td>")
                log_file.write(f"<td>{pos.quantity}</td>")
                log_file.write(f"<td class='{underlying_last_price_class}'>{pos.underlying_last_price}</td>")
                log_file.write(f"<td class='{in_the_money_class}'>{pos.last_price}</td>")
                log_file.write(f"<td>{pos.call_put}</td>")
                log_file.write(f"<td class='{gain_loss_class}'>{pos.gain_loss_percentage}%</td>")
                log_file.write(f"<td>{pos.days_to_expiration}</td>")
                log_file.write(f"<td class='{distance_to_strike_class}'>{pos.distance_to_strike}%</td>")
                log_file.write(f"<td class='{volatility_class}'>{pos.volatility}%</td>")
                log_file.write(f"<td>{pos.strike_price}</td>")
                log_file.write(f"<td>{pos.expiration_date}</td>")
                log_file.write(f"<td>{pos.implied_volatility}%</td>")
                log_file.write(f"<td class='{delta_class}'>{pos.delta}</td>")
                log_file.write(f"<td>{this_delta:.2f}</td>")
                log_file.write(f"<td>${this_delta * 100 * pos.underlying_last_price:.1f}</td>")
                log_file.write(f"<td class='{net_delta_class}'>{pos.net_ticker_delta}</td>")
                log_file.write(f"<td class='{gamma_class}'>{pos.gamma}</td>")
                log_file.write(f"<td>{this_gamma:.2f}</td>")
                log_file.write(f"<td class='{net_gamma_class}'>{pos.net_ticker_gamma}</td>")
                log_file.write(f"<td>{pos.theta}</td>")
                log_file.write(f"<td class='{theta_pct_class}'>{theta_pct}%</td>")
                log_file.write("</tr>")

        # Close the table with totals
        log_file.write("<tr>")
        log_file.write(f"<td>Total asset</td>")
        log_file.write(f"<td>{0}</td>")
        log_file.write(f"<td>Total margin</td>")
        log_file.write(f"<td>{0}</td>")
        log_file.write(f"<td>Total option</td>")
        log_file.write(f"<td>{round(total_sold, 2)}</td>")
        log_file.write(f"<td>In the money</td>")
        log_file.write(f"<td>{round(total_itm, 2)}</td>")
        log_file.write(f"<td>Daily decay</td>")
        log_file.write(f"<td>{round(daily_time_decay, 2)}</td>")
        log_file.write("</tr>")
        log_file.write("</table>")

            # # Add risk plots and close HTML
            # log_file.write("<h2>Daily Risk Analysis</h2>")
            # log_file.write("<h3>Upside Risk</h3>")
            # log_file.write(upside_plot_html)
            # log_file.write("<h3>Downside Risk</h3>")
            # log_file.write(downside_plot_html)
            # log_file.write("</body></html>")

    print(f"HTML log file generated: {log_filename}")
    return log_filename

class StockPosition:
    """Class to represent a stock position in the portfolio."""
    def __init__(
        self,
        symbol,
        quantity,
        last_price=None,
        price_paid=None,
        total_gain=None,
        market_value=None,
        position_id=None,
        position_type=None,
        security_type=None,
        date_acquired=None,
        strike_price=None,
        call_put=None,
        expiration_date=None,
        days_to_expiration=None,
        distance_to_strike=None,
        volatility=None,
        implied_volatility=None,
        theta=None,
        rho=None,
        vega=None,
        delta=None,
        gamma=None,
        net_ticker_delta=None,
        ticker_hedging_cost=None,
        target_option_roll=None,
        target_hedge_option_roll=None,
        underlying_last_price=None,
        option_intrinsic=None,
        iv_skew=None,
        osi_key=None,
        option_multiplier=None,
        options_adjusted_flag=None,
        option_deliverables=None,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.last_price = last_price
        self.price_paid = price_paid
        self.total_gain = total_gain
        self.market_value = market_value
        self.position_id = position_id
        self.position_type = position_type
        self.security_type = security_type
        self.position_covered = 0
        self.date_acquired = date_acquired
        self.days_to_earnings = None
        self.next_earning_date = None
        # Additional attributes for options
        self.call_put = call_put
        self.expiration_date = expiration_date
        self.days_to_expiration = days_to_expiration
        self.strike_price = strike_price
        self.gain_loss_percentage = None
        self.distance_to_strike = distance_to_strike
        self.volatility = None
        self.implied_volatility = None
        self.theta = None
        self.rho = None
        self.vega = None
        self.delta = delta
        self.gamma = None
        self.net_ticker_delta = None
        self.ticker_hedging_cost = None
        self.target_option_roll = None
        self.target_hedge_option_roll = None
        self.underlying_last_price = None
        self.option_intrinsic = None
        self.iv_skew = None
        self.lots_url = None  # URL for lots information, if applicable
        self.osi_key = osi_key
        self.option_multiplier = option_multiplier
        self.options_adjusted_flag = options_adjusted_flag
        self.option_deliverables = option_deliverables


    def __str__(self):
        """Return a formatted string representation of the stock position."""
        base_info = (f"Symbol: {self.symbol} | Quantity #: {self.quantity} | Acquired: {self.date_acquired} | Last Price: ${self.last_price:.2f} | "
                     f"Price Paid: ${self.price_paid:.2f} | Total Gain: ${self.total_gain:.2f} | "
                     f"Value: ${self.market_value:.2f} | Position ID: {self.position_id} | "
                     f"Position Type: {self.position_type} | Security Type: {self.security_type}")
        
        # Add option-specific information if applicable
        if self.security_type == "Option":
            option_info = (f" | Option Type: {self.call_put} | Expiration Date: {self.expiration_date} | "
                           f"Strike Price: ${self.strike_price:.2f}")
            return base_info + option_info
        return base_info


class Accounts:
    def __init__(self, session, base_url, use_sandbox=False, consumer_key=None):
        """
        Initialize Accounts object with session and account information

        :param session: authenticated session
        """
        self.session = session
        self.account = {}
        self.base_url = base_url
        self.use_sandbox = use_sandbox
        config_key = "SANDBOX_CONSUMER_KEY" if self.use_sandbox else "PROD_CONSUMER_KEY"
        self.consumer_key = resolve_etrade_consumer_key(
            self.use_sandbox,
            consumer_key=consumer_key,
            config_value=config["DEFAULT"].get(config_key),
            required=False,
        )

    def _consumer_key_headers(self):
        if not self.consumer_key:
            environment = "sandbox" if self.use_sandbox else "production"
            config_key = "SANDBOX_CONSUMER_KEY" if self.use_sandbox else "PROD_CONSUMER_KEY"
            raise RuntimeError(
                f"Missing E*TRADE {environment} consumer key; pass consumer_key "
                f"to Accounts or configure {config_key}."
            )
        return {"consumerKey": self.consumer_key}

    def _refresh_auth_session_if_possible(self, reason):
        callback = getattr(self, "auth_refresh_callback", None)
        if not callable(callback):
            return False
        refreshed = callback(reason)
        if not refreshed:
            return False
        self.session, self.base_url = refreshed
        return True

    def account_list(
        self,
        selected_account_id=1,
        *,
        expected_account_id_key=None,
        expected_account_id=None,
        expected_institution_type=None,
    ):
        """
        Calls account list API to retrieve a list of the user's E*TRADE accounts

        :param self: Passes in parameter authenticated session
        :return: List of tuples containing account_id, account_desc, and institution_type
        """

        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/list.json"

        # Make API call for GET request
        response = self.session.get(url, header_auth=True)
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))

        # List to store account information
        account_info = []
        account_dict = {}

        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Account list response received")

            data = response.json()
            if data is not None and "AccountListResponse" in data and "Accounts" in data["AccountListResponse"] \
                    and "Account" in data["AccountListResponse"]["Accounts"]:
                accounts = data["AccountListResponse"]["Accounts"]["Account"]
                # Display account list
                count = 1
                print("\nBrokerage Account List:")
                accounts[:] = [d for d in accounts if d.get('accountStatus') != 'CLOSED']
                for account in accounts:
                    account_id = account.get("accountId", "")
                    account_id_key = account.get("accountIdKey", "")
                    institution_type = account.get("institutionType", "")                    
                    account_info.append((account_id, "", institution_type, account_id_key))
                    
                    print_str = f"{count-1})\t{_redact_account_identifier(account_id)}, {institution_type}"
                    print(print_str)
                    count += 1
                if expected_account_id_key is not None:
                    matches = [
                        account for account in accounts
                        if account.get("accountIdKey") == expected_account_id_key
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            "Expected E*TRADE account key did not resolve to exactly one open account"
                        )
                    selected = matches[0]
                    if (
                        selected.get("accountId") != expected_account_id
                        or selected.get("institutionType") != expected_institution_type
                    ):
                        raise RuntimeError("Expected E*TRADE account identity did not match")
                    self.account = selected
                elif selected_account_id is not None:
                    self.account = accounts[selected_account_id]
                else:
                    raise RuntimeError("An exact E*TRADE account identity is required")
                # Index selection is retained only for legacy sandbox callers.
                return account_info
            else:
                # Handle errors
                logger.debug("Response Body: %s", response.text)
                if response is not None and response.headers['Content-Type'] == 'application/json' \
                        and "Error" in response.json() and "message" in response.json()["Error"] \
                        and response.json()["Error"]["message"] is not None:
                    print("Error: " + data["Error"]["message"])
                else:
                    print("Error: AccountList API service error")
        else:
            # Handle errors
            logger.debug("Response Body: %s", response.text)
            if response is not None and response.headers['Content-Type'] == 'application/json' \
                    and "Error" in response.json() and "message" in response.json()["Error"] \
                    and response.json()["Error"]["message"] is not None:
                print("Error: " + response.json()["Error"]["message"])
            else:
                print("Error: AccountList API service error")

    def get_stock_prices(self, tickers, include_metadata=False):
        """
        Retrieve the current stock prices for multiple tickers in a single API call.
        
        Args:
            tickers (str or list): A single ticker symbol or a list of ticker symbols.
            
        Returns:
            dict or float: If a list of tickers is provided, returns a dictionary mapping tickers to prices.
                        If a single ticker is provided, returns the price as a float.
        """
        # Handle single ticker case for backward compatibility
        single_ticker = False
        if isinstance(tickers, str):
            tickers = [tickers]
            single_ticker = True
        
        # Convert tickers if needed
        converted_tickers = []
        ticker_map = {}  # Map to track original ticker to converted ticker
        
        for ticker in tickers:
            if ticker == "BRKB":
                converted_tickers.append("BRK.B")
                ticker_map["BRK.B"] = "BRKB"
            elif ticker == "VIXW" or ticker == "^VIX":
                converted_tickers.append("VIX")
                ticker_map["VIX"] = ticker
            else:
                converted_tickers.append(ticker)
                ticker_map[ticker] = ticker
        
        # E*TRADE API limits market/quote to 25 symbols per request
        result = {}
        metadata = {}
        for i in range(0, len(converted_tickers), 25):
            chunk = converted_tickers[i:i+25]
            symbols = ",".join(chunk)
            url = f"{self.base_url}/v1/market/quote/{symbols}.json"
            params = {"detailFlag": "ALL"}
            
            try:
                response = self.session.get(url, params=params)
                if is_etrade_token_expired_response(response) and self._refresh_auth_session_if_possible("quote fetch"):
                    response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    quote_data_list = data.get("QuoteResponse", {}).get("QuoteData", [])
                    if not isinstance(quote_data_list, list):
                        quote_data_list = [quote_data_list]
                        
                    for quote_data in quote_data_list:
                        prod = quote_data.get("Product", {})
                        symbol = prod.get("symbol")
                        
                        # Special handling for options: E*TRADE returns symbol as underlier,
                        # so we must match against the full OSI string used in the request.
                        mapped_ticker = None
                        if prod.get("securityType") == "OPTN":
                            cp = prod.get("callPut")
                            exp_y = prod.get("expiryYear")
                            exp_m = prod.get("expiryMonth")
                            exp_d = prod.get("expiryDay")
                            strike = float(prod.get("strikePrice", 0))
                            
                            for ticker_key in tickers:
                                if ":" in ticker_key:
                                    parts = ticker_key.split(":")
                                    if len(parts) == 6:
                                        # OSI Format: underlying:year:month:day:type:strike
                                        t_sym, t_y, t_m, t_d, t_cp, t_strike = parts
                                        try:
                                            if (t_sym == symbol and 
                                                int(t_y) == exp_y and 
                                                int(t_m) == exp_m and 
                                                int(t_d) == exp_d and 
                                                t_cp == cp and 
                                                abs(float(t_strike) - strike) < 0.001):
                                                mapped_ticker = ticker_key
                                                break
                                        except (ValueError, TypeError):
                                            continue
                        
                        if mapped_ticker is None:
                            mapped_ticker = ticker_map.get(symbol, symbol)
                            
                        all_data = quote_data.get("All", {})
                        if prod.get("securityType") == "OPTN":
                            price = _option_quote_mark(quote_data)
                        else:
                            price = _market_quote_price(quote_data)
                        result[mapped_ticker] = price
                        metadata[mapped_ticker] = _quote_metadata(quote_data, price)
                else:
                    print(f"E*TRADE quote error {response.status_code} for chunk: {chunk}")
            except Exception as e:
                print(f"Exception during E*TRADE quote fetch: {e}")
        
        # Identify missing tickers
        missing_tickers = [t for t in tickers if _float_or_zero(result.get(t)) <= 0]
        
        if missing_tickers:
            print(f"E*TRADE failed to fetch prices for {missing_tickers}. Attempting with yfinance...")
            for ticker in missing_tickers:
                # Do not attempt to look up E*TRADE OSI option symbols in yfinance
                if ":" in ticker:
                    print(f"Skipping yfinance fallback for option symbol {ticker}")
                    result[ticker] = None
                    continue
                
                try:
                    yf_ticker = convert_ticker_name(etrade_ticker=ticker)
                    ticker_obj = yf.Ticker(yf_ticker)
                    data = ticker_obj.history(period="1d")
                    if not data.empty:
                        result[ticker] = float(data['Close'].iloc[-1])
                        quote_timestamp = data.index[-1]
                        metadata[ticker] = {
                            "price": result[ticker],
                            "source": "Yahoo Finance fallback",
                            "status": "FALLBACK",
                            "timestamp": int(quote_timestamp.timestamp()) if hasattr(quote_timestamp, "timestamp") else None,
                            "date_time": str(quote_timestamp),
                        }
                    else:
                        result[ticker] = None
                except Exception as e:
                    print(f"Error fetching {ticker} from yfinance: {e}")
                    result[ticker] = None
        
        if single_ticker:
            price = result.get(tickers[0], None)
            return (price, metadata.get(tickers[0], {})) if include_metadata else price
        return (result, metadata) if include_metadata else result

    # For backward compatibility, keep the original method but make it use the new one
    def get_stock_price(self, ticker):
        """
        Retrieve the current stock price for the specified ticker.
        """
        return self.get_stock_prices(ticker)

    def get_available_expirations(self, ticker: str):
        """
        Retrieve available option expiration dates for the specified ticker using E*TRADE API by default.
        If E*TRADE fails, fallback to using yfinance.

        Parameters:
        - ticker (str): The stock ticker symbol.

        Returns:
        - list: A list of expiration dates (YYYY-MM-DD) if available, or an empty list if not.
        """
        # Attempt to fetch expiration dates using E*TRADE API
        url = f"{self.base_url}/v1/market/optionexpiredate"
        if ticker == "BRKB":
            ticker = "BRK.B"
        if ticker == "BRK-B":
            ticker = "BRK.B"
        params = {"symbol": ticker, "expiryType": "ALL"}
        response = self.session.get(url, params=params, auth=self.session.auth)
        if is_etrade_token_expired_response(response) and self._refresh_auth_session_if_possible(f"{ticker} expiration fetch"):
            response = self.session.get(url, params=params, auth=self.session.auth)
        
        if response.status_code == 200:
            try:
                # Parse XML response from E*TRADE
                root = ET.fromstring(response.content)
                expiration_dates = []
                for expiration in root.findall("ExpirationDate"):
                    year = expiration.find("year").text
                    month = expiration.find("month").text.zfill(2)  # Ensure two-digit month format
                    day = expiration.find("day").text.zfill(2)      # Ensure two-digit day format
                    expiration_dates.append(f"{year}-{month}-{day}")
                return expiration_dates
            except ET.ParseError as e:
                print("Error parsing E*TRADE XML response:", e)
                print("Response content:", response.text)
        
        # If E*TRADE fails, fallback to yfinance
        print(f"E*TRADE failed to fetch expiration dates for {ticker}. Attempting with yfinance...")
        try:
            ticker_obj = yf.Ticker(convert_ticker_name(etrade_ticker=ticker))
            expiration_dates = ticker_obj.options  # Fetch expiration dates
            if expiration_dates:
                print(f"{ticker} expiration dates from yfinance is {expiration_dates}")
                return expiration_dates
            else:
                print(f"No options available for {ticker} in yfinance.")
                return []
        except Exception as e:
            print(f"Error fetching option expirations from yfinance for {ticker}: {e}")
            return []

    def get_option_price(self, symbol: str, call_put: str, expiration_date: datetime, strike_price: float):
        """
        Retrieve the option price from the E*TRADE API for a specific option.

        Parameters:
        - symbol (str): The underlying stock symbol.
        - call_put (str): Option type, 'CALL' or 'PUT'.
        - expiration_date (datetime): The expiration date as a datetime object.
        - strike_price (float): The strike price of the option.

        Returns:
        - float: The option price (bid) if found, or None if not.
        """
        self._last_option_price_symbol = symbol
        url = f"{self.base_url}/v1/market/optionchains.json"
        params = {
            "symbol": symbol,
            "expiryYear": expiration_date.year,
            "expiryMonth": expiration_date.month,
            "expiryDay": expiration_date.day,
            "includeWeekly": True,
            "skipAdjusted": False,
            "optionCategory": "ALL",
            "chainType": call_put,
            "strikePriceNear": strike_price,
            "noOfStrikes": 20,
        }
        response = self.session.get(url, params=params)

        if response.status_code != 200:
            print("Failed to retrieve option chain:", response.status_code, response.text)
            print(f"symbol: {symbol}, call_put: {call_put}, expiration_date: {expiration_date}, strike_price: {strike_price}")
            return None

        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

        for option_pair in option_pairs:
            option = option_pair.get(call_put.capitalize())
            if option and abs(float(option.get("strikePrice", 0)) - float(strike_price)) < 1e-6:
                self._last_option_price_symbol = _symbol_from_osi_key(symbol, option.get("osiKey"))
                return option.get("bid")  # return the bid price as the option price

        print("Option with specified parameters not found.")
        return None

    def manual_option_input(self, symbol: str, call_put: str, expiration_date: str, strike_price: float):
        """
        Manually input option details, poll the option price from E*TRADE API, and return a StockPosition object.

        Parameters:
        - symbol (str): The underlying stock symbol.
        - call_put (str): Type of option ('CALL' or 'PUT').
        - expiration_date (str): Expiration date in 'YYYY-MM-DD' format.
        - strike_price (float): Strike price of the option.

        Returns:
        - StockPosition: An object representing the option's position with price details.
        """

        # Convert expiration_date to datetime
        try:
            expiry_date_obj = datetime.strptime(expiration_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid expiration date format. Use 'YYYY-MM-DD'.")
            return None

        # Poll E*TRADE API for option price
        option_price = self.get_option_price(symbol, call_put, expiry_date_obj, strike_price)
        if option_price is None:
            print("Failed to retrieve option price from E*TRADE API.")
            return None

        contract_symbol = getattr(self, "_last_option_price_symbol", symbol)

        # Create and return a StockPosition object
        stock_position = StockPosition(
            symbol=contract_symbol,
            quantity=1,
            last_price=option_price,
            price_paid=option_price,
            total_gain=0.0,
            market_value=option_price,
            position_id="manual_input",
            position_type=call_put,
            security_type="Option"
        )
        stock_position.call_put = call_put
        stock_position.expiration_date = expiration_date
        stock_position.strike_price = strike_price

        return stock_position

    def generate_target_strike(self,stock_position: StockPosition,days_to_expire=0,multiplier=1,earning_alert=False,select_mode="HEDGE",optimizer="theta"):
        VOLATILITY_GAIN_TEMP = VOLATILITY_GAIN
        if earning_alert:
            target_option_price_call = round( stock_position.underlying_last_price * (1 + 2 * (days_to_expire**0.5) * stock_position.volatility / 100),2)
            target_option_price_put = round( stock_position.underlying_last_price * (1 - 2 * (days_to_expire**0.5) * stock_position.volatility / 100),2)   
            target_hedge_option_price_call = round( stock_position.underlying_last_price * (1 + 2 * HEDGE_RATIO * (days_to_expire**0.5) * stock_position.volatility / 100),2)
            target_hedge_option_price_put = round( stock_position.underlying_last_price * (1 - 2 * HEDGE_RATIO * (days_to_expire**0.5) * stock_position.volatility / 100),2)
        else:
            if select_mode == "ROLL_OUT":
                if optimizer == "theta":
                    VOLATILITY_GAIN_TEMP = 0
                    target_option_price_call = stock_position.strike_price
                    target_option_price_put = stock_position.strike_price
                    target_hedge_option_price_call = target_option_price_call + 10
                    target_hedge_option_price_put = target_option_price_put - 10

                elif optimizer == "DTE":
                    if stock_position.option_intrinsic > 0: #in the money
                        VOLATILITY_GAIN_TEMP = 0
                        target_option_price_call = round( stock_position.underlying_last_price * (1 + multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                        target_option_price_put = round( stock_position.underlying_last_price * (1 - multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                        target_hedge_option_price_call = target_option_price_call + 10
                        target_hedge_option_price_put = target_option_price_put - 10
                    elif stock_position.option_intrinsic < 0: #out of the money
                        target_option_price_call = round( stock_position.underlying_last_price * (1 + multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                        target_option_price_put = round( stock_position.underlying_last_price * (1 - multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                        target_hedge_option_price_call = target_option_price_call + 10
                        target_hedge_option_price_put = target_option_price_put - 10
            elif select_mode == "ROLL_IN" or select_mode == "HEDGE":
                VOLATILITY_GAIN_TEMP = VOLATILITY_GAIN
                target_option_price_call = round( stock_position.underlying_last_price * (1 + multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                target_option_price_put = round( stock_position.underlying_last_price * (1 - multiplier * (days_to_expire**0.5) * stock_position.volatility / 100),2)
                target_hedge_option_price_call = target_option_price_call + 10
                target_hedge_option_price_put = target_option_price_put - 10
        
        if stock_position.call_put == "CALL":
            print(f"Generating price targe CALL: {stock_position.symbol},{stock_position.call_put},{target_option_price_call},DTE: {days_to_expire}")
            return target_option_price_call,target_hedge_option_price_call
        elif stock_position.call_put == "PUT":
            print(f"Generating price targe PUT: {stock_position.symbol},{stock_position.call_put},{target_option_price_put},DTE: {days_to_expire}")
            return target_option_price_put,target_hedge_option_price_put

    def get_option_chain(self, stock_position: StockPosition, call_put, select_mode=None, multiplier=1, earning_alert=False, optimizer='theta', show_options=True):
        """
        Retrieve and filter the option chain for the specified ticker based on criteria.
        Parameters:
        - stock_position (StockPosition): The current stock position to roll
        - call_put (str): The option type, either 'CALL' or 'PUT'
        - select_mode (str, optional): 'ROLL_OUT' for options with later expiration dates and positive roll cost, 
                                        'ROLL_IN' for earlier expiration dates, or None for closest to 2 weeks from now
                                        'HEDGE' for same expiration date but with further strike
        - optimizer: "theta"/"distance-to-strike"
            "DTE": roll to the option that has strike above the target price
        """
        stock_price = self.get_stock_price(stock_position.symbol)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None,None

        today = datetime.today()
        two_weeks_from_now = today + timedelta(weeks=SELL_OPTION_EXPIRE_WEEK)

        # Step 1: Get available expiration dates and convert them to datetime.date objects
        # available_expirations = self.get_available_expirations(convert_ticker_name(yfinance_ticker=stock_position.symbol))
        available_expirations = self.get_available_expirations(stock_position.symbol)
        if not available_expirations:
            print("No available expiration dates for options.",stock_position.symbol)
            return None,None
        available_expirations = [datetime.strptime(exp, "%Y-%m-%d").date() if isinstance(exp, str) else exp for exp in available_expirations]

        # Step 2: Determine the expiration filtering based on select_mode
        current_expiration = stock_position.expiration_date if isinstance(stock_position.expiration_date, date) else datetime.strptime(stock_position.expiration_date, "%Y-%m-%d").date()
        if select_mode == "ROLL_OUT":
            if optimizer == 'DTE':
                filtered_expirations = [exp for exp in available_expirations if exp >= current_expiration]
            else:
                filtered_expirations = available_expirations
            
        elif select_mode == "ROLL_IN":
            if optimizer == 'DTE':
                if stock_position.days_to_expiration > MIN_DTE:
                    filtered_expirations = [exp for exp in available_expirations if exp <= current_expiration]
                else:
                    filtered_expirations = available_expirations[1:]
            else:
                filtered_expirations = available_expirations[1:]

        elif select_mode == "HEDGE":
            filtered_expirations = [exp for exp in available_expirations if exp == current_expiration]

        else:
            # If select_mode is None, find expirations close to two weeks from today
            filtered_expirations = available_expirations
        # breakpoint()
        # Step 3: Loop through filtered expirations until a valid option is found

        # lowest_theta = stock_position.theta
        lowest_theta = 10
        min_theta_option = None
        MAX_RETRIES = 10  # Maximum retries for abnormal data
        THETA_THRESHOLD = 100
        IMPLIED_VOLATILITY_THRESHOLD = 100
        abnormal_detected = False

        for expiration in filtered_expirations:

            today = pd.Timestamp.now().normalize()
            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            business_days = pd.date_range(start=today, end=expiration, freq=us_bd)
            days_to_expire = len(business_days)
            # print(f"expire: {expiration}, days to expire: {days_to_expire}")
            if days_to_expire > 100:
                continue
            url = f"{self.base_url}/v1/market/optionchains.json"
            if stock_position.symbol == "BRKB":
                symbol_converted = "BRK.B"
            else:
                symbol_converted = stock_position.symbol
            # if call_put == "CALL":
            quote_target_price,hedge_target = self.generate_target_strike(stock_position,days_to_expire,multiplier,earning_alert,select_mode,optimizer)
            if stock_position.call_put == "CALL":
                target_option_price_call = quote_target_price
                target_hedge_option_price_call = hedge_target
            elif stock_position.call_put == "PUT":
                target_option_price_put = quote_target_price
                target_hedge_option_price_put = hedge_target

            # print(f"Generated strike: {stock_position.symbol}: {quote_target_price},hedge target: {hedge_target}")
            # breakpoint()
            # else:
            #     quote_target_price = target_option_price_put
            params = {
                "symbol": symbol_converted,
                "expiryYear": expiration.year,
                "expiryMonth": expiration.month,
                "expiryDay": expiration.day,
                "includeWeekly": True,
                "skipAdjusted": False,
                "optionCategory": "ALL",
                "strikePriceNear": quote_target_price,
                "chainType": call_put,
                "noOfStrikes": 40
            }

            response = self.session.get(url, params=params, auth=self.session.auth)

            if response.status_code != 200:
                print("Error fetching option chain:", symbol_converted, response.status_code, response.text)
                return None,None

            data = response.json()
            option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

            closest_option = None
            closest_diff = 100
            options = None

            for option_pair in option_pairs:
                if call_put == "CALL":
                    options = option_pair.get("Call")
                if call_put == "PUT":
                    options = option_pair.get("Put")

                # Process call option
                # if call_option:
                # Retrieve theta and ensure it's interpreted as a float
                theta = float(options.get("OptionGreeks", {}).get("theta", 100))
                implied_volatility = float(options.get("OptionGreeks", {}).get("iv", 100))
                rho = float(options.get("OptionGreeks", {}).get("rho", 100))
                vega = float(options.get("OptionGreeks", {}).get("vega", 100))
                delta = float(options.get("OptionGreeks", {}).get("delta", 100))
                gamma = float(options.get("OptionGreeks", {}).get("gamma", 100))

                osi_key = options.get("osiKey")
                if osi_key:
                    expiry_date_str = osi_key[6:12]
                    expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()

                    # Check if roll direction conditions are met
                    if select_mode == "ROLL_OUT" and expiry_date <= current_expiration:
                        continue
                    # elif select_mode == "ROLL_IN" and expiry_date > current_expiration:
                    #     continue

                    option_price = options.get("bid")
                    strike_price = options.get("strikePrice")
                    option_id = options.get("optionId")
                    roll_cost = (option_price - stock_position.last_price) * 100 # Roll cost for calls
                    
                    if option_price is None:
                        continue
                    if ( select_mode == "ROLL_OUT" and roll_cost > -20 ) or ( select_mode == "ROLL_IN" and roll_cost > 10 ) or ( select_mode == "HEDGE") :
                        if select_mode in [ "ROLL_OUT", "ROLL_IN"]:
                            diff = abs(strike_price - quote_target_price)
                        elif select_mode == "HEDGE":
                            diff = abs(strike_price - hedge_target)
                        # print(f"diff: {diff}, theta: {theta}, current theta: {stock_position.theta}")
                        if diff < closest_diff:
                            closest_diff = diff
                            if stock_position.theta >= theta or optimizer!="theta":
                                closest_option = {
                                    "optionId": option_id,
                                    "type": "CALL",
                                    "strikePrice": strike_price,
                                    "optionPrice": option_price,
                                    "expiryDate": expiry_date.strftime("%Y-%m-%d"),
                                    "return_rate": round(option_price / stock_price * 100, 2),
                                    "cost_for_roll": round(roll_cost, 2),
                                    "implied_volatility": round(implied_volatility/(252**0.5)*100,2),
                                    "theta": round(theta,4)
                                }
                                
            # Return the closest option if found for the specified type
            if closest_option:
                # if optimizer == "DTE" or select_mode == "ROLL_OUT":
                if optimizer == "DTE":
                    return closest_option,closest_option
                # elif optimizer == "theta" and select_mode == "ROLL_IN":
                elif optimizer == "theta":
                    if show_options == True:
                        print(f"strike: {closest_option['strikePrice']} price: {closest_option['optionPrice']},theta: {closest_option['theta']}")
                    if closest_option['theta'] <= lowest_theta:
                        if abs(closest_option['theta']) < 999:
                            lowest_theta = closest_option['theta']
                            min_theta_option = closest_option
                            if show_options == True and min_theta_option is not None:
                                print(f"min_theta_option updated: {min_theta_option}")
                        else:
                            print(f"Invalid option greeks received...")

        if optimizer == "DTE": 
            # If no valid option was found, return None
            print("No target DTE option found: ", stock_position.symbol,stock_position.call_put,select_mode)
            return None,None
        if optimizer == "theta" and abnormal_detected == True:
            print("No target option found due to invalid option greek: ", stock_position.symbol,stock_position.call_put,select_mode)
            return None,None
        if min_theta_option is None and min_theta_option is None:
            print("No target minimum theta option found: ", stock_position.symbol,stock_position.call_put,select_mode)
            return None,None
        print(stock_position.symbol,min_theta_option['expiryDate'],min_theta_option['strikePrice'],min_theta_option['optionPrice'],min_theta_option['theta'])
        if optimizer == "theta" and stock_position.theta < min_theta_option['theta'] and stock_position.days_to_expiration > MIN_DTE:
            print(f"ABORT ROLL OUT due to theta degradation, current theta: {stock_position.theta}, new theta: {min_theta_option['theta']}, days to expiration: {stock_position.days_to_expiration} ")
            return None,None
        else:
            return min_theta_option,min_theta_option

    def get_option_chain_simple(self, ticker):
        stock_price = self.get_stock_price(ticker)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None, None

        today = datetime.today()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        available_expirations = self.get_available_expirations(ticker)
        if not available_expirations:
            print("No available expiration dates for options.", ticker)
            return None, None
        available_expirations = [datetime.strptime(exp, "%Y-%m-%d").date() if isinstance(exp, str) else exp for exp in available_expirations]
        available_expirations = available_expirations[:4]
        # Create a list to store all option data
        all_options_data = []

        for expiration in available_expirations:
            today = pd.Timestamp.now().normalize()
            us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            business_days = pd.date_range(start=today, end=expiration, freq=us_bd)
            days_to_expire = len(business_days)

            if days_to_expire > 100:
                continue

            url = f"{self.base_url}/v1/market/optionchains.json"
            symbol_converted = "BRK.B" if ticker == "BRKB" else ticker

            quote_target_price = stock_price
            params = {
                "symbol": symbol_converted,
                "expiryYear": expiration.year,
                "expiryMonth": expiration.month,
                "expiryDay": expiration.day,
                "includeWeekly": True,
                "skipAdjusted": False,
                "optionCategory": "ALL",
                "strikePriceNear": quote_target_price,
                "noOfStrikes": 100
            }

            response = self.session.get(url, params=params, auth=self.session.auth)

            if response.status_code != 200:
                print("Error fetching option chain:", symbol_converted, response.status_code, response.text)
                return None, None

            data = response.json()
            option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

            # Create the history_option directory if it doesn't exist
            os.makedirs("history_option", exist_ok=True)

            for option_type in ["Call", "Put"]:
                for option_pair in option_pairs:
                    options = option_pair.get(option_type)

                    # Get option Greeks
                    greeks = options.get("OptionGreeks", {})
                    theta = float(greeks.get("theta", 100))
                    implied_volatility = float(greeks.get("iv", 100))
                    rho = float(greeks.get("rho", 100))
                    vega = float(greeks.get("vega", 100))
                    delta = float(greeks.get("delta", 100))
                    gamma = float(greeks.get("gamma", 100))

                    osi_key = options.get("osiKey")
                    if osi_key:
                        expiry_date_str = osi_key[6:12]
                        expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()
                        
                        # Create a dictionary for each option
                        option_data = {
                            "Ticker": ticker,
                            "underly_price": stock_price,
                            "Strike_Price": options.get("strikePrice"),
                            "Option_Type": option_type,
                            "Expiry_Date": expiry_date,
                            "Ask_Price": options.get("ask"),
                            "Bid_Price": options.get("bid"),
                            "Ask_Size": options.get("askSize"),
                            "Bid_Size": options.get("bidSize"),
                            "Open_Interest": options.get("openInterest"),
                            "Volume": options.get("volume"),
                            "Timestamp": current_time,
                            "Theta": theta,
                            "Implied_Volatility": implied_volatility,
                            "Rho": rho,
                            "Vega": vega,
                            "Delta": delta,
                            "Gamma": gamma
                        }
                        
                        all_options_data.append(option_data)

            # Convert to DataFrame and save to pickle file
            df = pd.DataFrame(all_options_data)
            pickle_filename = f"history_option/{ticker}_{datetime.now().strftime('%Y-%m-%d')}.pkl"
            # Read existing data if file exists
            existing_df = None
            if os.path.exists(pickle_filename):
                try:
                    existing_df = pd.read_pickle(pickle_filename)
                except Exception as e:
                    print(f"Error reading existing pickle file: {e}")
            
            # Convert new data to DataFrame
            new_df = pd.DataFrame(all_options_data)
            
            # Combine existing and new data if there was existing data
            if existing_df is not None and not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates based on all columns except Timestamp
                columns_for_dedup = [col for col in combined_df.columns if col != 'Timestamp']
                combined_df = combined_df.drop_duplicates(subset=columns_for_dedup, keep='last')
            else:
                combined_df = new_df
            
            # Save the combined DataFrame
            combined_df.to_pickle(pickle_filename)
            
        
    def print_stocks_with_negative_options(self, stock_positions):
        """
        Print stock positions that have associated options with negative quantity,
        where the option's symbol exactly matches the stock symbol.
        
        :param stock_positions: List of StockPosition objects in the portfolio.
        """
        # Separate stock and option positions
        stock_dict = {pos.symbol: pos for pos in stock_positions if pos.security_type == "Stock"}
        option_positions = [pos for pos in stock_positions if pos.security_type == "Option"]

        # print("\nStock positions with associated options that have negative quantities:")
        found = False

        cover_option_positions = []

        # Check each option position for negative quantity and exact matching stock symbol
        for option in option_positions:
            if option.quantity < 0:  # Check if the option has a negative quantity
                # Check if there is an exact match between the option's symbol and any stock symbol
                cover_option_positions.append(option)
                stock_position = stock_dict.get(option.symbol)
                if stock_position:
                    # If found, print the stock position and associated option details
                    # print(f"\nStock Position:\n{stock_position}")
                    # print(f"Associated Option with Negative Quantity:\nSymbol: {option.symbol} | "
                    #     f"Quantity: {option.quantity} | Description: {option.symbol} | Type: {option.position_type}")
                    found = True

        if not found:
            print("No stock positions with associated options that have negative quantities.")
        return cover_option_positions

    def portfolio(self, print_enable=False, minimal=False, require_success=False):
        """
        Call portfolio API to retrieve a list of positions held in the specified account.
        
        Args:
            print_enable (bool): Whether to print position details. Defaults to False.
            minimal (bool): If True, skips time-consuming calculations and returns only basic position data
                        needed for position matching. Defaults to False.
            require_success (bool): Retained for caller compatibility. Incomplete
                        or unsuccessful portfolio responses always raise so no
                        caller can mistake partial data for a complete snapshot.
        
        Returns:
            list: List of StockPosition objects representing positions in the portfolio.
        """
        # URL for the API endpoint
        url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/portfolio.json"
        
        # Initialize pagination parameters and storage
        all_positions = []
        offset = 1
        count = 50
        max_pages = 100
        expected_total_pages = None
        expected_total_field = None
        seen_position_ids = set()

        def reject_incomplete_snapshot(message, *, cause=None):
            logger.error("E*TRADE portfolio response rejected: %s", message)
            error = RuntimeError(
                f"E*TRADE portfolio snapshot is incomplete: {message}"
            )
            if cause is not None:
                raise error from cause
            raise error
        
        # Fetch all positions with pagination
        while True:
            if offset > max_pages:
                reject_incomplete_snapshot(
                    "pagination exceeded the fixed 100-page bound"
                )
                break
            params = {
                "view": "COMPLETE",
                "count": count,
                "pageNumber": offset,
                "sortBy": "SYMBOL",
                "sortOrder": "ASC",
                "marketSession": "REGULAR",
                "totalsRequired": "false",
                "lotsRequired": "false",
            }
            response = self.session.get(url, params=params, header_auth=True)
            if is_etrade_token_expired_response(response) and self._refresh_auth_session_if_possible("portfolio fetch"):
                url = f"{self.base_url}/v1/accounts/{self.account['accountIdKey']}/portfolio.json"
                response = self.session.get(url, params=params, header_auth=True)
            logger.debug("Request Header: %s", redact_http_headers(response.request.headers))
            
            if response.status_code == 204:
                response_body = getattr(response, "content", None)
                if response_body is None:
                    response_body = (
                        getattr(response, "text", "") or ""
                    ).encode("utf-8")
                if offset != 1 or response_body:
                    reject_incomplete_snapshot(
                        "only an empty first-page HTTP 204 can confirm "
                        "a portfolio with no positions"
                    )
                break

            # Check if API call was successful
            if response.status_code != 200:
                logger.error("API request failed with status code: %s", response.status_code)
                raise RuntimeError(
                    f"E*TRADE portfolio request failed on page {offset} "
                    f"with status code {response.status_code}"
                )
            
            try:
                data = response.json()
            except Exception as exc:
                reject_incomplete_snapshot(
                    f"page {offset} is not valid JSON",
                    cause=exc,
                )
                break
            logger.debug("Response Body: %s", json.dumps(data, indent=4, sort_keys=True))
            
            if not isinstance(data, dict):
                reject_incomplete_snapshot(
                    f"page {offset} root is not an object"
                )
                break
            portfolio_response = data.get("PortfolioResponse")
            if not isinstance(portfolio_response, dict):
                reject_incomplete_snapshot(
                    f"page {offset} omitted PortfolioResponse"
                )
                break
            account_portfolios = portfolio_response.get("AccountPortfolio")
            if not isinstance(account_portfolios, list):
                reject_incomplete_snapshot(
                    f"page {offset} AccountPortfolio is not an array"
                )
                break
            if not account_portfolios:
                reject_incomplete_snapshot(
                    "HTTP 200 omitted the account-level portfolio proof"
                )
                break
            if len(account_portfolios) != 1:
                reject_incomplete_snapshot(
                    f"page {offset} did not contain exactly one account"
                )
                break

            acctPortfolio = account_portfolios[0]
            if not isinstance(acctPortfolio, dict):
                reject_incomplete_snapshot(
                    f"page {offset} account portfolio is not an object"
                )
                break
            response_account_id = acctPortfolio.get("accountId")
            expected_account_id = self.account.get("accountId")
            if (
                expected_account_id is not None
                and str(response_account_id) != str(expected_account_id)
            ):
                reject_incomplete_snapshot(
                    f"page {offset} account identity does not match"
                )
                break
            positions_in_response = acctPortfolio.get("Position")
            if not isinstance(positions_in_response, list):
                reject_incomplete_snapshot(
                    f"page {offset} Position is not an array"
                )
                break
            if len(positions_in_response) > count:
                reject_incomplete_snapshot(
                    f"page {offset} exceeded the requested row count"
                )
                break
            total_fields = [
                name
                for name in ("totalNoOfPages", "totalPages")
                if name in acctPortfolio
            ]
            if len(total_fields) != 1:
                reject_incomplete_snapshot(
                    f"page {offset} lacks unambiguous total-page metadata"
                )
                break
            raw_total_pages = acctPortfolio[total_fields[0]]
            if (
                type(raw_total_pages) not in {int, str}
                or isinstance(raw_total_pages, bool)
                or not str(raw_total_pages).isdigit()
            ):
                reject_incomplete_snapshot(
                    f"page {offset} total-page metadata is invalid"
                )
                break
            total_pages = int(raw_total_pages)
            if (
                not 1 <= total_pages <= max_pages
                or offset > total_pages
                or (
                    expected_total_pages is not None
                    and total_pages != expected_total_pages
                )
                or (
                    expected_total_field is not None
                    and total_fields[0] != expected_total_field
                )
            ):
                reject_incomplete_snapshot(
                    f"page {offset} total-page metadata changed or is out of bounds"
                )
                break
            if expected_total_pages is None:
                expected_total_pages = total_pages
                expected_total_field = total_fields[0]
            for position in positions_in_response:
                if not isinstance(position, dict):
                    reject_incomplete_snapshot(
                        f"page {offset} contains a non-object position"
                    )
                    break
                position_id = position.get("positionId")
                position_id_text = (
                    str(position_id)
                    if type(position_id) in {int, str}
                    and not isinstance(position_id, bool)
                    else ""
                )
                if (
                    not position_id_text.isascii()
                    or not position_id_text.isdigit()
                    or not 1 <= len(position_id_text) <= 19
                    or str(int(position_id_text)) != position_id_text
                    or not 1
                    <= int(position_id_text)
                    <= 9_223_372_036_854_775_807
                    or position_id_text in seen_position_ids
                ):
                    reject_incomplete_snapshot(
                        f"page {offset} contains an invalid or duplicate position id"
                    )
                    break
                seen_position_ids.add(position_id_text)
            
            # Add to total positions
            all_positions.extend(positions_in_response)

            if offset == total_pages:
                if acctPortfolio.get("nextPageNo") not in {None, ""}:
                    reject_incomplete_snapshot(
                        "terminal portfolio page advertised another page"
                    )
                break
            raw_next_page = acctPortfolio.get("nextPageNo")
            if (
                type(raw_next_page) not in {int, str}
                or isinstance(raw_next_page, bool)
                or not str(raw_next_page).isdigit()
                or int(raw_next_page) != offset + 1
            ):
                reject_incomplete_snapshot(
                    f"page {offset} next-page metadata is invalid"
                )
                break
            offset += 1
        
        stock_positions = []
        
        # Step 1: First pass to collect information about positions and build a list of unique tickers
        stock_price_dict = {}       # For stock prices
        option_position_data = []   # Temporary storage for all option data
        option_symbols = set()      # Set of unique option underlying tickers
        live_option_prices = {}     # For real-time option quotes
        
        for position in all_positions:
            product = position.get("Product", {})
            symbol = product.get("symbol", "N/A")
            security_type_code = product.get("securityType", "N/A")
            
            # Standardize symbols
            if symbol == "BRKB":
                symbol = "BRK.B"
            if symbol == "VIXW":
                symbol = "VIX"
            
            # Stock positions: store last price
            if security_type_code == "EQ":
                last_price = position.get("Complete", {}).get("price", 0.0)
                stock_price_dict[symbol] = last_price
            # Option positions: collect symbols and data
            elif security_type_code == "OPTN" and not minimal:
                option_symbols.add(_aggregate_option_symbol(symbol))
                option_position_data.append(position)
            
        # Fetch additional stock prices for option underlyings if needed
        symbols_to_query = [symbol for symbol in option_symbols if symbol not in stock_price_dict or stock_price_dict[symbol] == 0]
        if symbols_to_query:
            additional_prices = self.get_stock_prices(symbols_to_query)
            for symbol, price in additional_prices.items():
                stock_price_dict[symbol] = price
        
        # Step 1.5: Fetch real-time quotes for the options themselves to avoid stale portfolio data
        if option_position_data and not minimal:
            option_quote_symbols = []
            symbol_to_pos_map = {} # To handle different formats if necessary
            
            for pos in option_position_data:
                p = pos.get("Product", {})
                sym = p.get("symbol", "N/A")
                if sym == "BRKB": sym = "BRK.B"
                if sym == "VIXW": sym = "VIX"
                
                # E*TRADE Option Quote Format: ROOT:YEAR:MONTH:DAY:TYPE:STRIKE
                # month/day must be two digits
                try:
                    quote_sym = f"{sym}:{p.get('expiryYear')}:{int(p.get('expiryMonth')):02d}:{int(p.get('expiryDay')):02d}:{p.get('callPut')}:{p.get('strikePrice')}"
                    option_quote_symbols.append(quote_sym)
                except (TypeError, ValueError):
                    continue

            # Fetch in batches of 50 (API limit)
            for i in range(0, len(option_quote_symbols), 50):
                batch = option_quote_symbols[i:i+50]
                try:
                    quotes = self.get_stock_prices(batch)
                    if isinstance(quotes, dict):
                        live_option_prices.update(quotes)
                except Exception as e:
                    logger.error(f"Error fetching live option quotes: {e}")
        
        # Step 2: Process each position with complete price information
        for position in all_positions:
            product = position.get("Product", {})
            complete = position.get("Complete", {})
            security_type_code = product.get("securityType", "N/A")
            
            # Calculate detailed metrics if not in minimal mode
            if not minimal:
                theta = float(complete.get("theta", 100))
                rho = float(complete.get("rho", 100))
                vega = float(complete.get("vega", 100))
                delta = float(complete.get("delta", 100))
                gamma = float(complete.get("gamma", 100))
                iv = float(complete.get("ivPct", 100))
            
            symbol = product.get("symbol", "N/A")
            if symbol == "BRKB":
                symbol = "BRK.B"
            
            call_put = product.get("callPut", "N/A")
            
            # Handle expiration date
            expiry_year = product.get("expiryYear")
            expiry_month = product.get("expiryMonth")
            expiry_day = product.get("expiryDay")
            if expiry_year and expiry_month and expiry_day:
                expiration_date = date(expiry_year, expiry_month, expiry_day)
                if not minimal:
                    today = pd.Timestamp.now().normalize()
                    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
                    business_days = pd.date_range(start=today, end=expiration_date, freq=us_bd)
                    # days_to_expiration = len(business_days)
                    days_to_expiration = ( expiration_date - today.date() ).days
                else:
                    days_to_expiration = None
            else:
                expiration_date = None
                days_to_expiration = None
            
            strike_price = product.get("strikePrice", 0.0)
            symbol_description = position.get("symbolDescription", "N/A")
            quantity = position.get("quantity", 0)
            
            # Use live price if available, otherwise fall back to stale adjPrice
            option_last_price = position.get("Complete", {}).get("adjPrice", 0.0)
            if security_type_code == "OPTN" and not minimal:
                try:
                    # Reconstruct symbol to look up in our live quotes dictionary
                    lookup_sym = f"{symbol}:{product.get('expiryYear')}:{int(product.get('expiryMonth')):02d}:{int(product.get('expiryDay')):02d}:{product.get('callPut')}:{product.get('strikePrice')}"
                    live_val = live_option_prices.get(lookup_sym)
                    if live_val is not None and live_val > 0:
                        option_last_price = live_val
                except (TypeError, ValueError):
                    pass
            
            # Additional attributes if not minimal
            if not minimal:
                price_paid = position.get("pricePaid", 0.0)
                total_gain = position.get("totalGain", 0.0)
                date_acquired_timestamp = position.get("dateAcquired", 0.0)
                if len(str(date_acquired_timestamp)) == 13:  # Milliseconds
                    date_acquired = datetime.fromtimestamp(date_acquired_timestamp / 1000)
                else:  # Seconds
                    date_acquired = datetime.fromtimestamp(date_acquired_timestamp)
                date_acquired = date_acquired + timedelta(days=1)  # Adjust to Pacific Time
            else:
                price_paid = 0.0
                total_gain = 0.0
                date_acquired = None
            
            market_value = position.get("marketValue", 0.0)
            position_id = position.get("positionId", "N/A")
            position_type = position.get("positionType", "N/A")
            security_type_code = product.get("securityType", "N/A")
            
            # Map security type
            security_type_map = {"EQ": "Stock", "OPTN": "Option"}
            security_type = security_type_map.get(security_type_code, "Unknown")
            is_option = security_type == "Option"


            # Create StockPosition object
            stock_position = StockPosition(
                symbol=symbol,
                quantity=quantity,
                last_price=option_last_price,
                price_paid=price_paid,
                total_gain=total_gain,
                market_value=market_value,
                position_id=position_id,
                position_type=position_type,
                security_type=security_type,
                date_acquired=date_acquired,
                osi_key=position.get("osiKey") if is_option else None,
                option_multiplier=(
                    complete.get("optionMultiplier")
                    if is_option
                    else None
                ),
                options_adjusted_flag=(
                    complete.get("optionsAdjustedFlag")
                    if is_option
                    else None
                ),
                option_deliverables=(
                    complete.get("deliverablesStr")
                    if is_option
                    else None
                ),
            )
            stock_position.lots_url = position.get("lotsDetails")
            # Add option-specific attributes
            if security_type == "Option":
                stock_position.call_put = call_put
                stock_position.expiration_date = expiration_date
                stock_position.strike_price = strike_price
                aggregate_symbol = _aggregate_option_symbol(symbol)
                stock_position.underlying_last_price = stock_price_dict.get(aggregate_symbol) or stock_price_dict.get(symbol) or 0.0
                
                if not minimal:
                    stock_position.days_to_expiration = days_to_expiration
                    stock_position.theta = theta
                    stock_position.implied_volatility = round(iv / (252 ** 0.5) * 100, 2)
                    stock_position.rho = rho
                    stock_position.vega = vega
                    
                    # Convert standard delta to N(d2) risk-neutral probability
                    redefined_delta = delta
                    if delta is not None and delta != 100:
                        try:
                            redefined_delta = convert_standard_to_nd2_delta(delta, iv, days_to_expiration, call_put)
                        except Exception as e:
                            logger.error(f"Error converting delta in portfolio: {e}")
                    stock_position.delta = redefined_delta
                    stock_position.gamma = gamma
                    
                    stock_position.option_intrinsic = (
                        round(stock_position.underlying_last_price - strike_price, 2)
                        if stock_position.call_put == "CALL"
                        else round(strike_price - stock_position.underlying_last_price, 2)
                    )
                    
                    gain_loss_percentage = (
                        np.sign(quantity) * round((option_last_price - price_paid) / price_paid * 100, 2)
                        if price_paid != 0 else 0
                    )
                    distance_to_strike = (
                        round((stock_position.underlying_last_price - strike_price) / strike_price * 100, 2)
                        if strike_price != 0 else 0
                    )
                    
                    stock_position.gain_loss_percentage = gain_loss_percentage
                    stock_position.distance_to_strike = distance_to_strike
                    # stock_position.volatility = calculate_std_dev(convert_ticker_name(etrade_ticker=symbol), VOLATILITY_WINDOW)
                    stock_position.volatility = 0
                    
                    if CHECK_EARNING_DATE:
                        next_earning_date = self.check_earning_date(stock_position)
                    else:
                        next_earning_date = None
                    stock_position.next_earning_date = next_earning_date
                    
                    if next_earning_date and next_earning_date != "No earnings date available":
                        try:
                            formats = ["%Y-%m-%d", "%m/%d/%Y"]
                            earnings_date = None
                            for fmt in formats:
                                try:
                                    earnings_date = datetime.strptime(next_earning_date, fmt).date()
                                    break
                                except ValueError:
                                    continue
                            if earnings_date is None:
                                raise ValueError(f"Date format for {next_earning_date} not recognized.")
                            stock_position.days_to_earnings = (earnings_date - date.today()).days
                            print(f"{symbol}: Next earnings date is in {stock_position.days_to_earnings} days.")
                        except ValueError as e:
                            stock_position.days_to_earnings = None
                            print(f"Error parsing next earnings date for {symbol}: {e}")
                    else:
                        stock_position.days_to_earnings = None
                    
                    # Print option details
                    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
                    gain_loss_color = GREEN if gain_loss_percentage >= 0 else RED
                    distance_to_strike_color = RED if distance_to_strike >= 0 else GREEN
                    print_option_details = False
                    if quantity < 0 and print_enable and print_option_details:
                        print(f"Option Position for {symbol}:")
                        print(f"Type: {call_put} | Days to expiration: {days_to_expiration} | Strike Price: {strike_price} | Underlying Stock Price: {stock_position.underlying_last_price}")
                        print(f"Quantity: {quantity} | Option Last Price: ${option_last_price:.2f} | Price Paid: ${price_paid:.2f}")
                        print(f"Gain/Loss Percentage: {gain_loss_color}{gain_loss_percentage:.2f}%{RESET} | Distance to Strike: {distance_to_strike_color}{distance_to_strike:.2f}%{RESET} | Volatility: {stock_position.volatility}\n")
            
            stock_positions.append(stock_position)
        
        # Net delta and hedging calculations (unchanged from original)
        if not minimal:
            ticker_net_delta, ticker_net_gamma, ticker_has_short = {}, {}, {}
            ticker_stock_price, ticker_positions = {}, {}
            
            for position in stock_positions:
                if position.security_type == "Option":
                    symbol = _aggregate_option_symbol(position.symbol)
                    if symbol not in ticker_net_delta:
                        ticker_net_delta[symbol] = 0
                        ticker_net_gamma[symbol] = 0
                        ticker_has_short[symbol] = False
                        ticker_stock_price[symbol] = position.underlying_last_price
                        ticker_positions[symbol] = []
                    ticker_positions[symbol].append(position)
                    if abs(position.delta) < 10 and abs(position.gamma) < 10:
                        ticker_net_delta[symbol] += position.delta * position.quantity
                        ticker_net_gamma[symbol] += position.gamma * position.quantity
                    if position.quantity < 0:
                        ticker_has_short[symbol] = True
            
            ticker_hedging_cost, ticker_best_neutralization = {}, {}
            total_hedging_cost = 0
            
            for ticker, net_delta in ticker_net_delta.items():
                if ticker_has_short.get(ticker, False):
                    underlying_price = ticker_stock_price.get(ticker, 0)
                    hedging_cost = abs(net_delta) * underlying_price * 100
                    ticker_hedging_cost[ticker] = hedging_cost
                    total_hedging_cost += hedging_cost
                    
                    if abs(net_delta) > 1 and ticker in ticker_positions:
                        positions = ticker_positions[ticker]
                        smallest_remaining_delta = abs(net_delta)
                        best_remaining_delta = 0
                        best_position = None
                        for position in positions:
                            remaining_delta = net_delta - (position.delta * position.quantity)
                            if abs(remaining_delta) < smallest_remaining_delta:
                                smallest_remaining_delta = abs(remaining_delta)
                                best_remaining_delta = remaining_delta
                                best_position = position
                        if best_position:
                            ticker_best_neutralization[ticker] = {
                                'position': best_position,
                                'original_net_delta': net_delta,
                                'remaining_delta': best_remaining_delta
                            }
            
            for position in stock_positions:
                if position.security_type == "Option":
                    symbol = _aggregate_option_symbol(position.symbol)
                    position.net_ticker_delta = round(ticker_net_delta.get(symbol, 0), 2)
                    position.net_ticker_gamma = round(ticker_net_gamma.get(symbol, 0), 2)
                    position.ticker_hedging_cost = ticker_hedging_cost.get(symbol, 0)
            
            if print_enable:
                GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
                sorted_tickers = sorted(ticker_net_delta.items(), key=lambda x: abs(x[1]), reverse=True)
                print("\nNet Delta by Ticker (Ranked by Absolute Value):")
                for ticker, net_delta in sorted_tickers:
                    net_delta_color = GREEN if net_delta >= 0 else RED
                    underlying_price = ticker_stock_price.get(ticker, 0)
                    print(f"{ticker}: {net_delta_color}{net_delta:.2f}{RESET} (abs: {abs(net_delta):.2f}) | Underlying Price: ${underlying_price:.2f}", end="")
                    if ticker_has_short.get(ticker, False):
                        hedging_cost = ticker_hedging_cost.get(ticker, 0)
                        print(f" | Hedging Cost: ${hedging_cost:,.2f}")
                        if ticker in ticker_best_neutralization and abs(net_delta) > 1:
                            best = ticker_best_neutralization[ticker]
                            remaining_delta = best['remaining_delta']
                            remaining_delta_color = GREEN if remaining_delta >= 0 else RED
                            delta_improvement = abs(net_delta) - abs(remaining_delta)
                            best_position = best['position']
                            print(f"  ↳ Best position to neutralize: {best_position.call_put} {best_position.strike_price} exp:{best_position.expiration_date} qty:{best_position.quantity}")
                            print(f"     New net delta would be: {remaining_delta_color}{remaining_delta:.2f}{RESET} (improvement: {delta_improvement:.2f})")
                    else:
                        print(" | Long-only positions, no hedging required")
                print(f"\nTotal Delta Hedging Cost: ${total_hedging_cost:,.2f}")
        
        return stock_positions

    def screen_option(self, all_positions: List["StockPosition"], sort_output: bool = True):
        """
        Build a list used by the HTML renderer.
        Each item represents either a paired spread (two legs shown on two adjacent rows)
        or a single standout leg. Every pair is quantity‑matched (CALLs with CALLs, PUTs with PUTs),
        first by exact strike, then closest strike; any remainder becomes single‑leg entries.
        """
        from collections import defaultdict
        from copy import copy

        def _as_num(v, default=0.0):
            try:
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        def _compute_gain_loss_pct(lot):
            lp  = _as_num(getattr(lot, "last_price", None), None)
            pp  = getattr(lot, "price_paid", None)
            qty = _as_num(getattr(lot, "quantity", 0), 0.0)
            if pp in (None, 0, 0.0):
                return 0.0
            raw_pct = (lp - float(pp)) / float(pp) * 100.0
            sign    = -1.0 if qty < 0 else 1.0  # short gains are positive
            return round(sign * raw_pct, 2)

        def _compute_dist_to_strike(lot):
            v = getattr(lot, "distance_to_strike", None)
            if v not in (None, ""):
                try:
                    return round(float(v), 2)
                except Exception:
                    pass
            ul = getattr(lot, "underlying_last_price", None)
            strike = getattr(lot, "strike_price", None)
            cp = getattr(lot, "call_put", None)
            if ul is None or strike in (None, 0) or cp is None:
                return 0.0
            ul = float(ul); strike = float(strike)
            pct = ((ul - strike) / strike * 100.0) if cp == "CALL" else ((strike - ul) / strike * 100.0)
            return round(pct, 2)

        def _sanitize_leg(lot):
            # identity/contract
            lot.symbol                 = getattr(lot, "symbol", "")
            lot.quantity               = int(getattr(lot, "quantity", 0) or 0)
            lot.call_put               = getattr(lot, "call_put", None)
            lot.strike_price           = _as_num(getattr(lot, "strike_price", None), 0.0)
            lot.expiration_date        = getattr(lot, "expiration_date", None)
            # market/greeks
            lot.underlying_last_price  = _as_num(getattr(lot, "underlying_last_price", None), 0.0)
            lot.last_price             = _as_num(getattr(lot, "last_price", None), 0.0)
            lot.price_paid             = _as_num(getattr(lot, "price_paid", None), 0.0)
            lot.delta                  = _as_num(getattr(lot, "delta", None), 0.0)
            lot.gamma                  = _as_num(getattr(lot, "gamma", None), 0.0)
            lot.theta                  = _as_num(getattr(lot, "theta", None), 0.0)
            lot.net_ticker_delta       = _as_num(getattr(lot, "net_ticker_delta", None), 0.0)
            lot.net_ticker_gamma       = _as_num(getattr(lot, "net_ticker_gamma", None), 0.0)
            lot.implied_volatility     = _as_num(getattr(lot, "implied_volatility", None), 0.0)
            lot.volatility             = _as_num(getattr(lot, "volatility", None), 0.0)
            # guards like actions_option_trade_html
            if abs(lot.delta) > 1 or abs(lot.net_ticker_delta) > 1000:
                lot.delta = 0.0; lot.gamma = 0.0; lot.net_ticker_delta = 0.0; lot.net_ticker_gamma = 0.0
            if abs(lot.implied_volatility) > 100:
                lot.implied_volatility = 0.0
            if abs(lot.theta) > 1:
                lot.theta = 0.0
            # derived
            lot.distance_to_strike     = _compute_dist_to_strike(lot)
            lot.gain_loss_percentage   = _compute_gain_loss_pct(lot)
            lot.position_type          = getattr(lot, "position_type", None) or ("SHORT" if lot.quantity < 0 else "LONG")
            return lot

        # 1) Normalize legs from incoming positions (options only)
        legs = [p for p in all_positions if isinstance(p, StockPosition) and getattr(p, "security_type", "") == "Option"]
        legs = [_sanitize_leg(copy(p)) for p in legs if p.call_put is not None]

        # 2) Group by (symbol, expiration)
        #    We now group all legs of the same expiry together for pairing, 
        #    regardless of when they were opened. This prevents "orphaned" legs from rolls.
        buckets = defaultdict(list)
        # Also track total availability for informational display
        counts_by_sym_exp_cp = defaultdict(lambda: {"LONG": 0, "SHORT": 0})
        for leg in legs:
            group_symbol = _aggregate_option_symbol(leg.symbol)
            key = (group_symbol, leg.expiration_date)
            buckets[key].append(leg)
            try:
                sign = "LONG" if (getattr(leg, "quantity", 0) or 0) > 0 else "SHORT"
                k2 = (group_symbol, leg.expiration_date, leg.call_put)
                counts_by_sym_exp_cp[k2][sign] += abs(int(getattr(leg, "quantity", 0) or 0))
            except Exception:
                pass

        screened_options = []

        def _open_day(lot):
            value = getattr(lot, "date_acquired", None)
            if not value:
                return None
            try:
                return value.date() if hasattr(value, "date") else pd.Timestamp(value).date()
            except Exception:
                return None

        def _valid_vertical(short_leg, long_leg):
            cp = (getattr(short_leg, "call_put", "") or "").upper()
            s_strike = _as_num(getattr(short_leg, "strike_price", None), 0.0)
            l_strike = _as_num(getattr(long_leg, "strike_price", None), 0.0)
            if s_strike <= 0 or l_strike <= 0:
                return False
            if cp == "PUT":
                return l_strike < s_strike
            if cp == "CALL":
                return l_strike > s_strike
            return False

        def _spread_width(short_leg, long_leg):
            return abs(_as_num(getattr(short_leg, "strike_price", 0.0), 0.0) - _as_num(getattr(long_leg, "strike_price", 0.0), 0.0))

        def _take_pair(long_leg, short_leg, use_qty, pairing_note=None):
            """Create a quantity‑matched pair and append to screened_options."""
            from copy import copy
            long_part = copy(long_leg)
            short_part = copy(short_leg)
            long_part.quantity = int(abs(use_qty))
            short_part.quantity = -int(abs(use_qty))
            if pairing_note:
                long_part.pairing_note = pairing_note
                short_part.pairing_note = pairing_note

            init_diff = (
                _as_num(getattr(short_part, "price_paid", None), 0.0) -
                _as_num(getattr(long_part,  "price_paid", None), 0.0)
            )
            curr_diff = (short_part.last_price - long_part.last_price)
            pair_gl = 0.0 if init_diff == 0 else (100.0 - (curr_diff / init_diff * 100.0))

            screened_options.append({
                "is_spread": True,
                "pair_quantity": int(abs(use_qty)),
                "pair_gain_loss": round(pair_gl, 2),
                "long_lot": _sanitize_leg(long_part),
                "short_lot": _sanitize_leg(short_part),
            })

        def _add_single(lot, reason):
            lot.unpaired_reason = reason
            lot.pairing_note = reason
            screened_options.append({
                "is_spread": False,
                "pair_quantity": int(abs(lot.quantity)),
                "pair_gain_loss": _compute_gain_loss_pct(lot),
                "long_lot": _sanitize_leg(lot) if lot.quantity > 0 else None,
                "short_lot": _sanitize_leg(lot) if lot.quantity < 0 else None,
            })

        # 3) For each (symbol, expiry), pair CALLs with CALLs and PUTs with PUTs.
        # For SPY/SPX, always consume valid vertical protection before showing singles.
        pairable_symbols = {"SPY", "SPX"}
        for (sym, exp), group in buckets.items():
            for cp in ("CALL", "PUT"):
                longs  = [l for l in group if l.call_put == cp and l.quantity > 0]
                shorts = [s for s in group if s.call_put == cp and s.quantity < 0]
                had_longs_initial = len(longs) > 0
                had_shorts_initial = len(shorts) > 0

                if not longs and not shorts:
                    continue

                longs.sort(key=lambda x: _as_num(x.strike_price, 0.0))
                shorts.sort(key=lambda x: _as_num(x.strike_price, 0.0))

                if (sym or "").upper() not in pairable_symbols:
                    for l in longs:
                        _add_single(l, "non-SPY/SPX shown as single leg")
                    for s in shorts:
                        _add_single(s, "non-SPY/SPX shown as single leg")
                    continue

                while True:
                    candidates = []
                    for si, s in enumerate(shorts):
                        if s.quantity >= 0:
                            continue
                        for li, l in enumerate(longs):
                            if l.quantity <= 0 or not _valid_vertical(s, l):
                                continue
                            width = _spread_width(s, l)
                            s_day = _open_day(s)
                            l_day = _open_day(l)
                            same_day = s_day is not None and s_day == l_day
                            date_rank = 0 if same_day else (1 if s_day is None or l_day is None else 2)
                            candidates.append((date_rank, width, si, li, s, l))

                    if not candidates:
                        break

                    candidates.sort(key=lambda x: (x[0], x[1], _as_num(x[4].strike_price, 0.0), _as_num(x[5].strike_price, 0.0)))
                    date_rank, width, si, li, s, l = candidates[0]
                    note = "paired by same acquisition date" if date_rank == 0 else "paired by nearest valid protective leg"

                    use = min(l.quantity, abs(s.quantity))
                    if use <= 0:
                        break
                    _take_pair(l, s, use, pairing_note=note)
                    l.quantity -= use
                    s.quantity += use
                    longs  = [lot for lot in longs if lot.quantity > 0]
                    shorts = [lot for lot in shorts if lot.quantity < 0]

                for l in longs:
                    reason_parts = []
                    if not had_shorts_initial:
                        reason_parts.append("no SHORT legs with same symbol/expiration/type")
                        k2 = (sym, exp, cp)
                        if counts_by_sym_exp_cp.get(k2, {}).get("SHORT", 0) > 0:
                            reason_parts.append("short legs exist but no reliable vertical match")
                    else:
                        reason_parts.append("leftover long after reliable pairing")
                    _add_single(l, "; ".join(reason_parts))

                for s in shorts:
                    reason_parts = []
                    if not had_longs_initial:
                        reason_parts.append("no LONG protective legs with same symbol/expiration/type")
                    else:
                        reason_parts.append("leftover short after all valid protective legs were paired")
                    _add_single(s, "; ".join(reason_parts))

        # 4) Sort for nice HTML grouping
        if sort_output and screened_options:
            def _k(item):
                lot = item.get("short_lot") or item.get("long_lot")
                sym = getattr(lot, "symbol", "") or ""
                exp = getattr(lot, "expiration_date", None)
                cp  = getattr(lot, "call_put", "") or ""
                cp_rank = 0 if cp == "CALL" else 1
                pos_type = getattr(lot, "position_type", "")
                short_rank = 0 if pos_type == "SHORT" else 1
                strike = _as_num(getattr(lot, "strike_price", None), 0.0)
                return (sym, str(exp), cp_rank, short_rank, strike)
            screened_options.sort(key=_k)

        return screened_options
    

    def render_screened_option_pairs_html(self, screened_options, out_path="history_option/screened_option_pairs.html", title="Positions by Ticker", order_instance=None, show_refresh=True):
        """
        Render screened options to an HTML file with a SPY tracking chart.
        
        Args:
            screened_options: List of option pair dictionaries
            out_path: Output HTML file path
            title: Page title
            order_instance: Optional Order instance for fetching historical SPY gains from E*TRADE
        """
        import os, html
        from pathlib import Path

        # Portfolio option rows can carry an older underlying price even while
        # the market is open. Fetch one current quote per displayed ticker and
        # use it for both the visible tables and today's chart point.
        live_index_prices = {}
        live_index_quote_metadata = {}
        try:
            displayed_tickers = sorted({
                _aggregate_option_symbol(getattr(leg, "symbol", "") or "")
                for item in screened_options
                for leg in (item.get("short_lot"), item.get("long_lot"))
                if leg is not None and getattr(leg, "symbol", None)
            })
            fetched_index_prices, live_index_quote_metadata = self.get_stock_prices(
                displayed_tickers,
                include_metadata=True,
            )
            for ticker in displayed_tickers:
                value = fetched_index_prices.get(ticker) if fetched_index_prices else None
                if value is not None and float(value) > 0:
                    live_index_prices[ticker] = round(float(value), 2)
        except Exception as quote_err:
            print(f"[Live Price] Could not refresh underlying quotes: {quote_err}")

        if live_index_prices:
            for item in screened_options:
                for leg_key in ("short_lot", "long_lot"):
                    leg = item.get(leg_key)
                    if leg is None:
                        continue
                    ticker = _aggregate_option_symbol(getattr(leg, "symbol", "") or "")
                    if ticker in live_index_prices:
                        leg.underlying_last_price = live_index_prices[ticker]

        def _as_num(v, default=0.0):
            try:
                if v is None:
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        def _fmt_pct(v):
            try:
                return f"{float(v):.2f}%"
            except Exception:
                return ""

        def _fmt_num(v, nd=3):
            try:
                if v is None:
                    return ""
                if isinstance(v, int):
                    return str(v)
                return f"{float(v):.{nd}f}"
            except Exception:
                return ""

        def _naked_margin(short_leg, qty_abs):
            """Return margin requirement per short leg outside of spreads."""
            if short_leg is None or qty_abs <= 0:
                return 0.0
            cp = getattr(short_leg, "call_put", "") or ""
            cp = cp.upper()
            strike = _as_num(getattr(short_leg, "strike_price", None), 0.0)
            if strike <= 0:
                return 0.0
            if cp == "CALL":
                return strike * 100.0 * qty_abs
            if cp == "PUT":
                underlying = _as_num(getattr(short_leg, "underlying_last_price", None), strike)
                if underlying <= 0:
                    underlying = strike
                premium = _as_num(getattr(short_leg, "last_price", None), 0.0)
                calc1 = 0.2 * underlying - (strike - underlying) + premium
                calc2 = 0.1 * strike + premium
                margin_per_contract = max(calc1, calc2, 0.0)
                return margin_per_contract * 100.0 * qty_abs
            return 0.0

        def _fmt_money(value):
            return f"${_as_num(value):,.2f}"

        def _fmt_strike(value):
            number = _as_num(value)
            return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"

        def _dte(lot):
            expiration = getattr(lot, "expiration_date", None)
            if not expiration:
                return 0
            if isinstance(expiration, datetime):
                expiration = expiration.date()
            elif not isinstance(expiration, date):
                expiration = datetime.strptime(str(expiration)[:10], "%Y-%m-%d").date()
            return (expiration - datetime.now().date()).days

        def _gain_band(value):
            if value >= 80:
                return "close-now", "80%+"
            if value >= 70:
                return "close-watch", "70%+"
            if value < 0:
                return "loss", ""
            return "positive", ""

        def _quote_context(ticker, items):
            first_lot = next(
                (
                    item.get("short_lot") or item.get("long_lot")
                    for item in items
                    if item.get("short_lot") or item.get("long_lot")
                ),
                None,
            )
            price = live_index_prices.get(ticker)
            if not price and first_lot is not None:
                price = _as_num(getattr(first_lot, "underlying_last_price", None), 0.0)

            metadata = live_index_quote_metadata.get(ticker, {})
            source = str(metadata.get("source") or "E*TRADE portfolio")
            status = str(metadata.get("status") or "UNKNOWN").upper()
            timestamp = metadata.get("timestamp")
            if timestamp:
                quote_time = datetime.fromtimestamp(timestamp).astimezone().strftime("%b %d, %I:%M:%S %p %Z")
                quote_time = quote_time.replace(" 0", " ")
            else:
                quote_time = str(metadata.get("date_time") or "time unavailable")

            status_class = "quote-live" if status in ("REALTIME", "INDICATIVE_REALTIME", "CLOSING") else "quote-warning"
            return (
                f'<span class="underlying-price">{html.escape(ticker)} '
                f'<strong>{html.escape(_fmt_money(price))}</strong></span>'
                f'<span class="quote-status {status_class}">{html.escape(status)}</span>'
                f'<span class="quote-time">{html.escape(source)} · {html.escape(quote_time)}</span>'
            )

        from collections import defaultdict
        grouped_positions = defaultdict(list)
        for item in screened_options:
            lot = item.get("short_lot") or item.get("long_lot")
            if lot is not None:
                grouped_positions[_aggregate_option_symbol(getattr(lot, "symbol", "") or "")].append(item)

        position_groups = []
        for ticker in sorted(grouped_positions):
            items = grouped_positions[ticker]
            items.sort(key=lambda item: (
                -_as_num(item.get("pair_gain_loss"), 0.0),
                _dte(item.get("short_lot") or item.get("long_lot")),
            ))
            high_gain_count = sum(_as_num(item.get("pair_gain_loss"), 0.0) >= 70 for item in items)
            overview_class = "has-close-candidates" if high_gain_count else ""
            overview_text = (
                f"{high_gain_count} at 70%+"
                if high_gain_count
                else f"{len(items)} position{'s' if len(items) != 1 else ''}"
            )

            group_rows = []
            for item in items:
                is_spread = bool(item.get("is_spread"))
                short_lot = item.get("short_lot")
                long_lot = item.get("long_lot")
                action_lot = short_lot or long_lot
                if action_lot is None:
                    continue

                call_put = str(getattr(action_lot, "call_put", "") or "").upper()
                expiration = str(getattr(action_lot, "expiration_date", "") or "")
                signed_quantity = int(getattr(action_lot, "quantity", 0) or 0)
                quantity = int(item.get("pair_quantity") or abs(signed_quantity))
                short_strike = getattr(action_lot, "strike_price", 0)
                long_strike = getattr(long_lot, "strike_price", None) if is_spread else None
                gain_loss = _as_num(item.get("pair_gain_loss"), 0.0)
                gain_class, threshold_label = _gain_band(gain_loss)

                if is_spread and short_lot is not None and long_lot is not None:
                    entry_value = (
                        _as_num(getattr(short_lot, "price_paid", 0.0))
                        - _as_num(getattr(long_lot, "price_paid", 0.0))
                    )
                    current_value = (
                        _as_num(getattr(short_lot, "last_price", 0.0))
                        - _as_num(getattr(long_lot, "last_price", 0.0))
                    )
                    strikes = (
                        f'<span class="strike-values">{html.escape(_fmt_strike(short_strike))}'
                        f'<span aria-hidden="true"> / </span>{html.escape(_fmt_strike(long_strike))}</span>'
                        '<span class="strike-labels">Short / Long</span>'
                    )
                else:
                    entry_value = _as_num(getattr(action_lot, "price_paid", 0.0))
                    current_value = _as_num(getattr(action_lot, "last_price", 0.0))
                    strikes = (
                        f'<span class="strike-values">{html.escape(_fmt_strike(short_strike))}</span>'
                        '<span class="strike-labels">Single leg</span>'
                    )

                underlying = _as_num(getattr(action_lot, "underlying_last_price", 0.0))
                strike_number = _as_num(short_strike)
                if underlying > 0 and strike_number > 0:
                    distance = abs(underlying - strike_number) / underlying * 100.0
                    is_otm = (
                        (call_put == "CALL" and strike_number > underlying)
                        or (call_put == "PUT" and strike_number < underlying)
                    )
                    distance_text = f"{distance:.1f}% {'OTM' if is_otm else 'ITM'}"
                    distance_class = "distance-safe" if is_otm else "distance-risk"
                else:
                    distance_text = "—"
                    distance_class = ""

                threshold_html = f"<small>{threshold_label}</small>" if threshold_label else ""
                group_rows.append(f'''
                  <tr class="position-row {gain_class}">
                    <td class="expiry-cell"><strong>{_dte(action_lot)} DTE</strong><span>{html.escape(expiration)}</span></td>
                    <td class="position-cell"><span class="option-type">{html.escape(call_put)}</span>{strikes}</td>
                    <td class="quantity-cell">{quantity}</td>
                    <td class="entry-cell desktop-detail">{html.escape(_fmt_money(entry_value))}</td>
                    <td class="mark-cell desktop-detail">{html.escape(_fmt_money(current_value))}</td>
                    <td class="distance-cell desktop-detail {distance_class}">{html.escape(distance_text)}</td>
                    <td class="gain-cell"><span>{gain_loss:.1f}%</span>{threshold_html}</td>
                    <td class="read-only-cell">Read only</td>
                  </tr>
                ''')

            position_groups.append(f'''
              <section class="ticker-group" aria-labelledby="ticker-{html.escape(ticker)}">
                <header class="ticker-header">
                  <div>
                    <h2 id="ticker-{html.escape(ticker)}">{html.escape(ticker)}</h2>
                    <div class="quote-context">{_quote_context(ticker, items)}</div>
                  </div>
                  <span class="ticker-overview {overview_class}">{html.escape(overview_text)}</span>
                </header>
                <div class="ticker-table-wrap">
                  <table class="ticker-table" aria-label="{html.escape(ticker)} option positions">
                    <thead>
                      <tr>
                        <th>Expiry</th>
                        <th>Position</th>
                        <th>Qty</th>
                        <th class="desktop-detail">Entry</th>
                        <th class="desktop-detail">Pair mark</th>
                        <th class="desktop-detail">Distance</th>
                        <th>Pair P/L</th>
                        <th>Mode</th>
                      </tr>
                    </thead>
                    <tbody>{''.join(group_rows)}</tbody>
                  </table>
                </div>
              </section>
            ''')

        position_groups_html = "".join(position_groups)
        rows_html = []

        # --- Add totals row (left aligned) ---
        total_price = 0.0
        sp_total_option_price = 0.0
        index_deltas = {
            "SPY": {"CALL": 0.0, "PUT": 0.0},
            "SPX": {"CALL": 0.0, "PUT": 0.0},
        }
        itm_by_symbol = {}

        from collections import defaultdict
        type_groups = defaultdict(list)

        for item in screened_options:
            for leg in (item.get("long_lot"), item.get("short_lot")):
                if not leg:
                    continue
                sym = _aggregate_option_symbol(getattr(leg, "symbol", ""))
                qty = getattr(leg, "quantity", 0) or 0   # signed quantity
                
                # Delta aggregation for S&P instruments
                if sym in index_deltas:
                    delta = _as_num(getattr(leg, "delta", 0.0), 0.0)
                    if abs(delta) > 2.0: # 100 fallback
                        delta = 0.0
                    cp_type = (getattr(leg, "call_put", "") or "").upper()
                    pos_delta = qty * delta * 100.0
                    if cp_type in index_deltas[sym]:
                        index_deltas[sym][cp_type] += pos_delta

                price = getattr(leg, "last_price", 0.0) or 0.0
                strike = getattr(leg, "strike_price", None)
                ul = getattr(leg, "underlying_last_price", None)
                cp = getattr(leg, "call_put", None)
                exp = getattr(leg, "expiration_date", "")

                if strike is not None and ul is not None:
                    is_itm = (cp == "CALL" and ul > strike) or (cp == "PUT" and ul < strike)
                    if is_itm:
                        itm_by_symbol[sym] = itm_by_symbol.get(sym, 0.0) + qty * price * 100.0
                
                if qty != 0:
                    key = (str(sym).upper(), str(exp), str(cp).upper())
                    type_groups[key].append({
                        'qty': qty,
                        'price': price,
                        'strike': strike or 0
                    })

        for key, group in type_groups.items():
            shorts = [p for p in group if p['qty'] < 0]
            longs = [p for p in group if p['qty'] > 0]
            
            opt_type = key[2]
            if opt_type == 'PUT':
                longs.sort(key=lambda x: x['strike'], reverse=True)
                shorts.sort(key=lambda x: x['strike'], reverse=True)
            else:
                longs.sort(key=lambda x: x['strike'])
                shorts.sort(key=lambda x: x['strike'])

            for short in shorts:
                val = short['qty'] * short['price'] * 100.0
                total_price += val
                if key[0] in ('SPY', 'SPX'):
                    sp_total_option_price += val
                short_qty_abs = abs(short['qty'])
                for long in longs:
                    if long['qty'] <= 0: continue
                    matched_qty = min(short_qty_abs, long['qty'])
                    if matched_qty > 0:
                        val_long = matched_qty * long['price'] * 100.0
                        total_price += val_long
                        if key[0] in ('SPY', 'SPX'):
                            sp_total_option_price += val_long
                        short_qty_abs -= matched_qty
                        long['qty'] -= matched_qty
                    if short_qty_abs <= 0:
                        break

        # Format ITM breakdown
        itm_parts = [f"{sym}: ${val:,.2f}" for sym, val in itm_by_symbol.items()]
        itm_str = ", ".join(itm_parts) if itm_parts else "$0.00"

        def _sp_margin_totals(items):
            from collections import defaultdict

            # Group by symbol and expiry so SPY and SPX do not offset each other.
            expiry_groups = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0})
            
            # Sub-grouping logic to handle pairings within each expiry
            groups = defaultdict(lambda: {"longs": [], "shorts": []})
            for entry in items:
                for leg in (entry.get("long_lot"), entry.get("short_lot")):
                    if not leg:
                        continue
                    sym = _aggregate_option_symbol(getattr(leg, "symbol", ""))
                    if sym not in ("SPY", "SPX"):
                        continue
                    cp = (getattr(leg, "call_put", "") or "").upper()
                    if cp not in ("CALL", "PUT"):
                        continue
                    exp = getattr(leg, "expiration_date", None)
                    qty = _as_num(getattr(leg, "quantity", 0), 0.0)
                    if qty == 0:
                        continue
                    bucket = groups[(sym, cp, exp)]
                    bucket["longs" if qty > 0 else "shorts"].append({"leg": leg, "qty": abs(qty)})

            def _strike(rec):
                return _as_num(getattr(rec["leg"], "strike_price", None), 0.0)

            # Calculate call and put margins per expiry
            for (sym, cp, exp), parts in groups.items():
                shorts = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["shorts"]]
                if not shorts:
                    continue
                longs = [{"leg": rec["leg"], "qty": rec["qty"]} for rec in parts["longs"]]

                reverse = True if cp == "PUT" else False
                shorts.sort(key=lambda rec: _strike(rec), reverse=reverse)
                longs.sort(key=lambda rec: _strike(rec), reverse=reverse)

                expiry_type_margin = 0.0
                for short in shorts:
                    while short["qty"] > 0 and longs:
                        best_idx = min(
                            range(len(longs)),
                            key=lambda idx: abs(_strike(short) - _strike(longs[idx]))
                        )
                        long = longs[best_idx]
                        pair_qty = min(short["qty"], long["qty"])
                        if pair_qty <= 0:
                            if long["qty"] <= 0:
                                longs.pop(best_idx)
                            continue
                        strike_diff = abs(_strike(short) - _strike(long))
                        expiry_type_margin += strike_diff * pair_qty * 100.0
                        short["qty"] -= pair_qty
                        long["qty"] -= pair_qty
                        if long["qty"] <= 0:
                            longs.pop(best_idx)

                    if short["qty"] > 0:
                        expiry_type_margin += _naked_margin(short["leg"], short["qty"])
                
                expiry_groups[(sym, exp)][cp] += expiry_type_margin

            # Final Max Risk Calculation: sum(max(call_margin, put_margin) for each expiry)
            total_corrected = 0.0
            total_call = 0.0
            total_put = 0.0
            breakdown_lines = []
            for (sym, exp), margins in expiry_groups.items():
                risk = max(margins["CALL"], margins["PUT"])
                total_corrected += risk
                total_call += margins["CALL"]
                total_put += margins["PUT"]
                breakdown_lines.append(f"{sym} {exp}: Max(C: ${margins['CALL']:,.0f}, P: ${margins['PUT']:,.0f}) = ${risk:,.0f}")
                
            return total_corrected, total_call, total_put, breakdown_lines

        # Calculate Stock Margin (30% of Long Stock Value)
        stock_margin = 0.0
        stock_details = []
        try:
            # Re-fetch or use cached positions if possible
            all_pos = self.portfolio(minimal=True)
            for p in all_pos:
                if getattr(p, "security_type", "") == "EQ" and getattr(p, "quantity", 0) > 0:
                    val = _as_num(getattr(p, "market_value", 0), 0.0)
                    m = val * 0.30
                    stock_margin += m
                    stock_details.append(f"{p.symbol}: ${val:,.0f} * 30% = ${m:,.0f}")
        except Exception:
            pass

        spy_total_margin, spy_call_margin, spy_put_margin, margin_breakdown = _sp_margin_totals(screened_options)
        grand_total_margin = stock_margin + spy_total_margin
        spy_call_delta = index_deltas["SPY"]["CALL"]
        spy_put_delta = index_deltas["SPY"]["PUT"]
        spx_call_delta = index_deltas["SPX"]["CALL"]
        spx_put_delta = index_deltas["SPX"]["PUT"]
        total_call_delta = spy_call_delta + spx_call_delta
        total_put_delta = spy_put_delta + spx_put_delta
        
        # Try to get account-level margin from E*TRADE for comparison
        etrade_margin_info = "N/A"
        try:
            bal = self.balance()
            if bal and "Computed" in bal:
                # E*Trade uses various keys for margin requirement depending on account type
                m_margin = (bal["Computed"].get("maintenanceMargin") or 
                            bal["Computed"].get("currentMarginBalance") or 
                            bal["Computed"].get("houseMarginRequirement"))
                if m_margin is not None:
                    etrade_margin_info = f"${float(m_margin):,.2f}"
        except Exception:
            pass

        totals_row = f"""
        <tr class="totals">
          <td colspan="8" style="text-align:left;font-weight:bold;">
            Total Option Price = ${total_price:,.2f} |
            Aggregated Delta (SPY+SPX): Call = {total_call_delta:,.1f}, Put = {total_put_delta:,.1f}, Net = {(total_call_delta + total_put_delta):,.1f}
            <span style="font-weight:normal; color:#64748b;">
              (SPY Net {(spy_call_delta + spy_put_delta):,.1f}; SPX Net {(spx_call_delta + spx_put_delta):,.1f})
            </span> |
            Total ITM by Symbol → {itm_str}
          </td>
        </tr>
        """
        rows_html.append(totals_row)

        margin_row = f"""
        <tr class=\"totals\">
          <td colspan=\"8\" style=\"text-align:left;font-weight:bold;\">
            Total Calculated Margin = ${grand_total_margin:,.2f} |
            E*TRADE Account Maintenance Margin = {etrade_margin_info}
            <div style="font-size:0.8em; font-weight:normal; margin-top:5px; color:#aaa;">
                <b>Breakdown:</b><br/>
                Stock (30%): {", ".join(stock_details) if stock_details else "$0.00"}<br/>
                Options (Max Risk per Expiry): {", ".join(margin_breakdown) if margin_breakdown else "$0.00"}
            </div>
          </td>
        </tr>
        """
        rows_html.append(margin_row)

        # ===== HISTORICAL TRACKER COMPARISON SECTION =====
        # Load last frozen snapshot to show alongside live data
        tracker_summary_html = ""
        try:
            import os as os_mod
            import sys
            parent_dir = os_mod.path.dirname(os_mod.path.dirname(os_mod.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from live_trading.spy_position_tracker import _load_tracker_data
            
            tracker_data = _load_tracker_data()
            snapshots = tracker_data.get("daily_snapshots", {})
            if snapshots:
                sorted_dates = sorted(snapshots.keys())
                last_date = sorted_dates[-1]
                last_snap = snapshots[last_date]
                
                frozen_opt_price = last_snap.get("total_option_price", 0.0)
                frozen_margin = last_snap.get("total_margin", 0.0)
                frozen_put_margin = last_snap.get("spy_put_margin", 0.0)
                frozen_call_margin = last_snap.get("spy_call_margin", 0.0)
                frozen_ytd_gain = last_snap.get("ytd_realized_gain", 0.0)
                frozen_timestamp = last_snap.get("timestamp", last_date)
                
                # Calculate differences against the same SPY/SPX/SPXW scope as the tracker.
                price_diff = sp_total_option_price - frozen_opt_price
                margin_diff = spy_total_margin - frozen_margin
                
                price_diff_class = "pos" if price_diff >= 0 else "neg"
                margin_diff_class = "neg" if margin_diff > 0 else "pos"  # Lower margin is better
                
                tracker_summary_html = f'''
                <div style="margin: 15px 0; padding: 12px 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px; border-left: 4px solid #6c757d; max-width: 800px; box-sizing: border-box; overflow-x: auto; -webkit-overflow-scrolling: touch;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 14px; color: #495057;">📊 Historical Tracker (Frozen End-of-Day)</span>
                    <span style="font-size: 12px; color: #6c757d;">Last recorded: {last_date}</span>
                  </div>
                  <table style="border-collapse: collapse; width: 100%; font-size: 13px; background: white; border-radius: 4px;">
                    <tr>
                      <th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #dee2e6; background: #f8f9fa;">Metric</th>
                      <th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #dee2e6; background: #f8f9fa;">Live (Current)</th>
                      <th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #dee2e6; background: #f8f9fa;">Recorded ({last_date})</th>
                      <th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #dee2e6; background: #f8f9fa;">Diff</th>
                    </tr>
                    <tr>
                      <td style="padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">SPY/SPX/SPXW Option Price</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">${sp_total_option_price:,.2f}</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">${frozen_opt_price:,.2f}</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;" class="{price_diff_class}">{'+' if price_diff >= 0 else ''}${price_diff:,.2f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">Total Margin</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">${spy_total_margin:,.2f}</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;">${frozen_margin:,.2f}</td>
                      <td style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #f1f3f4;" class="{margin_diff_class}">{'+' if margin_diff >= 0 else ''}${margin_diff:,.2f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 6px 10px;">YTD Realized Gain</td>
                      <td style="text-align: right; padding: 6px 10px; color: #6c757d;" colspan="2">—</td>
                      <td style="text-align: right; padding: 6px 10px; font-weight: 600; color: #16811f;">${frozen_ytd_gain:,.2f}</td>
                    </tr>
                  </table>
                </div>
                '''
        except Exception as tracker_err:
            print(f"[Tracker Summary] Could not load tracker data: {tracker_err}")
            tracker_summary_html = ""

        style = """
        <style>
          body { font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 15px; background: #fff; color: #0f172a; margin: 0; }
          h1 { margin: 0 0 15px 0; font-size: 22px; color: #0f172a; font-weight: 600; }
          .read-only-banner { margin: 0 0 16px; padding: 12px 16px; border: 2px solid #b91c1c; border-radius: 10px; background: #fef2f2; color: #991b1b; font-size: 14px; font-weight: 800; letter-spacing: 0.04em; text-align: center; text-transform: uppercase; }
          td.pos { color: #10b981 !important; font-weight: 600; }
          td.neg { color: #ef4444 !important; font-weight: 600; }
          .position-groups { display: grid; gap: 18px; margin: 0 0 20px; }
          .ticker-group { overflow: hidden; border: 1px solid #dbe4ee; border-radius: 14px; background: #fff; box-shadow: 0 3px 14px rgba(15, 23, 42, 0.05); }
          .ticker-header { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 14px 16px; background: linear-gradient(135deg, #f8fafc, #f0f9ff); border-bottom: 1px solid #dbe4ee; }
          .ticker-header h2 { margin: 0; color: #0f172a; font-size: 21px; line-height: 1.1; }
          .quote-context { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 5px; color: #64748b; font-size: 12px; }
          .underlying-price strong { margin-left: 4px; color: #0f172a; font-size: 14px; font-variant-numeric: tabular-nums; }
          .quote-status { padding: 2px 6px; border-radius: 999px; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; }
          .quote-live { color: #047857; background: #d1fae5; }
          .quote-warning { color: #92400e; background: #fef3c7; }
          .ticker-overview { flex: 0 0 auto; padding: 6px 9px; border-radius: 999px; background: #e2e8f0; color: #475569; font-size: 11px; font-weight: 700; }
          .ticker-overview.has-close-candidates { background: #dcfce7; color: #047857; }
          .ticker-table-wrap { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
          .ticker-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; font-variant-numeric: tabular-nums; }
          .ticker-table th { padding: 8px 9px; background: #f8fafc; color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-align: right; text-transform: uppercase; white-space: nowrap; }
          .ticker-table th:first-child, .ticker-table th:nth-child(2) { text-align: left; }
          .ticker-table td { padding: 9px; border-top: 1px solid #eef2f7; color: #0f172a; text-align: right; vertical-align: middle; }
          .ticker-table tbody tr:first-child td { border-top: 0; }
          .ticker-table tbody tr:hover { background: #f8fafc; }
          .ticker-table .expiry-cell, .ticker-table .position-cell { text-align: left; }
          .expiry-cell strong, .expiry-cell span, .strike-values, .strike-labels { display: block; }
          .expiry-cell strong { font-size: 13px; }
          .expiry-cell span, .strike-labels { margin-top: 2px; color: #64748b; font-size: 10px; }
          .option-type { display: inline-block; min-width: 38px; margin-right: 7px; padding: 3px 5px; border-radius: 5px; background: #e0e7ff; color: #3730a3; font-size: 10px; font-weight: 800; text-align: center; }
          .strike-values { display: inline; font-size: 13px; font-weight: 700; }
          .strike-labels { margin-left: 49px; }
          .distance-safe { color: #047857 !important; }
          .distance-risk { color: #dc2626 !important; font-weight: 700; }
          .gain-cell span { display: inline-block; min-width: 58px; padding: 5px 7px; border-radius: 7px; background: #ecfdf5; color: #047857; font-size: 13px; font-weight: 800; text-align: center; }
          .gain-cell small { display: block; margin-top: 2px; color: inherit; font-size: 9px; letter-spacing: 0.04em; text-transform: uppercase; }
          .position-row.close-watch { background: #f0fdf4; }
          .position-row.close-now { background: #dcfce7; }
          .position-row.close-watch .gain-cell span { background: #bbf7d0; color: #166534; }
          .position-row.close-now .gain-cell span { background: #16a34a; color: #fff; }
          .position-row.loss .gain-cell span { background: #fef2f2; color: #dc2626; }
          .read-only-cell { color: #991b1b !important; font-size: 10px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; white-space: nowrap; }
          .portfolio-summary { width: 100%; overflow-x: auto; margin-bottom: 20px; border: 1px solid #e2e8f0; border-radius: 10px; }
          .summary-table { width: 100%; border-collapse: collapse; font-size: 12px; }
          .summary-table td { padding: 10px 12px; background: #f8fafc; border-top: 1px solid #e2e8f0; white-space: normal; }
          .summary-table tr:first-child td { border-top: 0; }
          
          @media (max-width: 768px) {
            body { padding: max(8px, env(safe-area-inset-top)) max(6px, env(safe-area-inset-right)) max(10px, env(safe-area-inset-bottom)) max(6px, env(safe-area-inset-left)); background: #f8fafc; }
            h1 { font-size: 18px; }
            h2 { font-size: 17px; }
            .diagnostic-data { display: none; }
            .desktop-analytics { display: block; min-width: 0; overflow: hidden; }
            .desktop-analytics > div { max-width: 100% !important; margin-bottom: 24px !important; }
            .desktop-analytics h2 { font-size: 17px; line-height: 1.3; }
            .desktop-analytics canvas { max-width: 100% !important; }
            .desktop-analytics button { min-height: 44px; padding: 8px 12px !important; }
            .position-groups { gap: 12px; }
            .ticker-group { border-radius: 11px; }
            .ticker-header { align-items: flex-start; padding: 11px 10px; }
            .ticker-header h2 { font-size: 19px; }
            .quote-context { gap: 5px; font-size: 10px; }
            .quote-time { flex-basis: 100%; }
            .ticker-overview { padding: 5px 7px; font-size: 10px; }
            .ticker-table { min-width: 0; table-layout: fixed; }
            .ticker-table th, .ticker-table td { padding: 7px 4px; }
            .ticker-table th { font-size: 9px; white-space: normal; line-height: 1.05; }
            .ticker-table th:nth-child(1) { width: 18%; }
            .ticker-table th:nth-child(2) { width: 28%; }
            .ticker-table th:nth-child(3) { width: 8%; }
            .ticker-table th:nth-child(7) { width: 19%; }
            .ticker-table th:nth-child(8) { width: 27%; }
            .desktop-detail { display: none; }
            .expiry-cell strong { font-size: 12px; }
            .expiry-cell span { font-size: 9px; white-space: normal; }
            .option-type { display: block; min-width: 0; width: fit-content; margin: 0 0 3px; padding: 2px 4px; font-size: 8px; }
            .strike-values { font-size: 12px; white-space: nowrap; }
            .strike-labels { margin-left: 0; font-size: 8px; }
            .gain-cell span { min-width: 0; width: 100%; padding: 5px 2px; font-size: 12px; }
            .read-only-cell { font-size: 8px; white-space: normal; }
            .portfolio-summary { display: none; }
          }
        </style>
        """

        Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
        
        # Generate SPY tracking chart data
        cash_flow_sync = {}
        try:
            import sys
            import os as os_mod
            # Add parent directory to path if needed for import
            parent_dir = os_mod.path.dirname(os_mod.path.dirname(os_mod.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Use the extended history function to get historical gains and cash flows
            from live_trading.spy_position_tracker import get_spy_tracking_history_with_gains
            spy_history = get_spy_tracking_history_with_gains(order_instance, start_date="2025-01-01")
            
            chart_dates = spy_history.get("dates", [])
            chart_prices = spy_history.get("total_option_prices", [])
            chart_margins = spy_history.get("total_margins", [])
            chart_cash_flows = spy_history.get("cash_flows", [])
            chart_realized_gains = spy_history.get("realized_gains", [])
            live_cash_flow = spy_history.get("cash_flow_current")
            live_realized_gain = spy_history.get("ytd_current")
            cash_flow_sync = spy_history.get("sync_health", {})

            series_lengths = {
                len(chart_dates),
                len(chart_prices),
                len(chart_margins),
                len(chart_cash_flows),
                len(chart_realized_gains),
            }
            if len(series_lengths) != 1:
                raise ValueError("SPY chart history series are not date-aligned")

            # ===== FILTER FOR TRADING DAYS ONLY =====
            try:
                from pandas_market_calendars import get_calendar
                import pandas as pd
                nyse = get_calendar('NYSE')
                
                if chart_dates:
                    # Clean dates (remove " (Live)" suffix) for comparison
                    clean_dates = [d.split(' ')[0] for d in chart_dates]
                    start_d = clean_dates[0]
                    end_d = clean_dates[-1]
                    
                    # Get all trading days in range
                    schedule = nyse.schedule(start_date=start_d, end_date=end_d)
                    valid_trading_days = set(schedule.index.strftime('%Y-%m-%d'))
                    
                    # Keep indices that correspond to valid trading days
                    valid_indices = [i for i, d in enumerate(clean_dates) if d in valid_trading_days]
                    
                    # Also keep the " (Live)" entry if it's today and today is a trading day
                    # (The schedule check already covers today if it's a trading day)
                    
                    chart_dates = [chart_dates[i] for i in valid_indices]
                    chart_prices = [chart_prices[i] for i in valid_indices]
                    chart_margins = [chart_margins[i] for i in valid_indices]
                    chart_cash_flows = [chart_cash_flows[i] for i in valid_indices]
                    chart_realized_gains = [chart_realized_gains[i] for i in valid_indices]
                    
                    print(f"[SPY Chart] Filtered to {len(chart_dates)} trading days (from {len(clean_dates)} total days).")
            except Exception as filter_err:
                print(f"[SPY Chart] Warning: Could not filter for trading days: {filter_err}")
            
            # ===== MANUAL EXCLUSION =====
            # Add dates here to manually exclude them from the charts
            MANUAL_EXCLUDE_DATES = ["2025-01-20"] # Example: "2025-01-20"
            
            if chart_dates:
                excluded_this_time = [d for d in chart_dates if d.split(' ')[0] in MANUAL_EXCLUDE_DATES]
                if excluded_this_time:
                    print(f"[Manual Exclude] Removing corrupted dates: {excluded_this_time}")
                
                valid_manual_indices = [i for i, d in enumerate(chart_dates) 
                                        if d.split(' ')[0] not in MANUAL_EXCLUDE_DATES]
                chart_dates = [chart_dates[i] for i in valid_manual_indices]
                chart_prices = [chart_prices[i] for i in valid_manual_indices]
                chart_margins = [chart_margins[i] for i in valid_manual_indices]
                chart_cash_flows = [chart_cash_flows[i] for i in valid_manual_indices]
                chart_realized_gains = [chart_realized_gains[i] for i in valid_manual_indices]
        except Exception as chart_err:
            print(f"[SPY Chart] Could not load tracking history: {chart_err}")
            chart_dates = []
            chart_prices = []
            chart_margins = []
            chart_cash_flows = []
            chart_realized_gains = []
            live_cash_flow = None
            live_realized_gain = None
            cash_flow_sync = {}
        
        # ===== ADD LIVE DATA POINT TO CHARTS =====
        # Append current live values so charts reflect real-time data
        from datetime import datetime as dt_mod
        today_str = dt_mod.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
        live_label = f"{today_str} (Live)"
        live_margin = spy_total_margin
        if live_cash_flow is None and chart_cash_flows:
            live_cash_flow = chart_cash_flows[-1]
        if live_realized_gain is None and chart_realized_gains:
            live_realized_gain = chart_realized_gains[-1]
        
        # Only add if we don't already have today's date, or update it if we do
        last_chart_date = chart_dates[-1].replace(' (Live)', '') if chart_dates else None
        if chart_dates and last_chart_date == today_str:
            # Update the last point with live values
            chart_dates[-1] = live_label
            chart_prices[-1] = sp_total_option_price
            chart_margins[-1] = live_margin
            if chart_cash_flows:
                chart_cash_flows[-1] = live_cash_flow if live_cash_flow is not None else chart_cash_flows[-1]
            elif live_cash_flow is not None:
                chart_cash_flows.append(live_cash_flow)
            if chart_realized_gains:
                chart_realized_gains[-1] = live_realized_gain if live_realized_gain is not None else chart_realized_gains[-1]
            elif live_realized_gain is not None:
                chart_realized_gains.append(live_realized_gain)
        else:
            # Add new live data point
            chart_dates.append(live_label)
            chart_prices.append(sp_total_option_price)
            chart_margins.append(live_margin)
            chart_cash_flows.append(live_cash_flow if live_cash_flow is not None else 0.0)
            chart_realized_gains.append(live_realized_gain if live_realized_gain is not None else 0.0)
        
        # ===== FETCH SPY, SPX & VIX CLOSING PRICES (CACHED) =====
        spy_closes = []
        spx_closes = []
        vix_closes = []
        _PRICE_CACHE_FILE = "spy_vix_price_cache.json"
        try:
            import yfinance as yf_chart

            # Strip " (Live)" suffix to get clean date strings
            clean_dates = [d.replace(' (Live)', '') for d in chart_dates]
            from datetime import datetime as _dt_cache
            today_str_cache = _dt_cache.now().strftime("%Y-%m-%d")

            # Load existing cache
            _price_cache = {"SPY": {}, "SPX": {}, "VIX": {}}
            try:
                with open(_PRICE_CACHE_FILE, 'r') as _cf:
                    _price_cache = json.load(_cf)
                    if "SPY" not in _price_cache:
                        _price_cache["SPY"] = {}
                    if "SPX" not in _price_cache:
                        _price_cache["SPX"] = {}
                    if "VIX" not in _price_cache:
                        _price_cache["VIX"] = {}
            except (FileNotFoundError, json.JSONDecodeError):
                pass

            # Determine which dates are missing from cache
            # Only query yfinance for today's price outside of market hours (before 9:00 AM or after 5:00 PM EST)
            # to prevent hitting yfinance rate limits. During market hours, we use E*TRADE's real-time fallback.
            import pytz as _pytz
            _et_tz = _pytz.timezone('US/Eastern')
            _now_et = _dt_cache.now(_et_tz)
            _is_market_active = (_now_et.weekday() < 5) and (9 <= _now_et.hour < 17)
            
            missing_dates = _missing_market_close_dates(
                chart_dates,
                _price_cache,
                today_str_cache,
                _is_market_active,
            )

            if missing_dates:
                # Fetch starting 3 days before to ensure we catch everything reliably
                start_dt_obj = _dt_cache.strptime(min(missing_dates), '%Y-%m-%d')
                safe_start = (start_dt_obj - __import__('datetime').timedelta(days=3)).strftime('%Y-%m-%d')
                
                print(f"[SPY/SPX/VIX Cache] Fetching {len(missing_dates)} date(s) from yfinance (safe_start={safe_start})...")
                spy_raw = yf_chart.download('SPY', start=safe_start, progress=False, auto_adjust=False)
                spx_raw = yf_chart.download('^SPX', start=safe_start, progress=False, auto_adjust=False)
                vix_raw = yf_chart.download('^VIX', start=safe_start, progress=False, auto_adjust=False)

                # Robust extraction: flatten MultiIndex columns, reset index, iterate rows
                def _extract_closes(df):
                    """Extract {date_str: close_price} from a yfinance DataFrame, handling MultiIndex."""
                    result = {}
                    if df is None or df.empty:
                        return result
                    
                    # Store a copy to avoid modifying original
                    temp_df = df.copy()
                    
                    # Flatten MultiIndex columns if present
                    if hasattr(temp_df.columns, 'levels') and len(temp_df.columns.levels) > 1:
                        try:
                            temp_df = temp_df.droplevel(1, axis=1)
                        except Exception:
                            # Fallback: if droplevel fails, try to find 'Close' manually
                            pass
                    
                    # Try to find 'Close' or it might be ('Close', 'SPY')
                    close_col = 'Close'
                    if close_col not in temp_df.columns:
                        # Search for any column that contains 'Close' in its name/tuple
                        for col in temp_df.columns:
                            if isinstance(col, tuple) and 'Close' in col:
                                close_col = col
                                break
                            elif isinstance(col, str) and 'Close' in col:
                                close_col = col
                                break
                    
                    if close_col not in temp_df.columns:
                        return result
                        
                    for idx in temp_df.index:
                        dt_str = str(idx.date()) if hasattr(idx, 'date') else str(idx).split(' ')[0]
                        try:
                            val = temp_df.loc[idx, close_col]
                            if hasattr(val, 'iloc'): # In case of duplicate index
                                val = val.iloc[0]
                            if pd.isna(val):
                                result[dt_str] = None
                                continue
                            result[dt_str] = round(float(val), 2)

                        except (ValueError, TypeError, IndexError):
                            continue
                    return result

                spy_new = _extract_closes(spy_raw)
                spx_new = _extract_closes(spx_raw)
                vix_new = _extract_closes(vix_raw)

                # Yahoo occasionally rate-limits the dashboard host. Use two
                # independent daily sources so missing closes do not remain
                # stale or get replaced by a live portfolio quote.
                missing_spy_dates = [d for d in missing_dates if d not in spy_new]
                if missing_spy_dates:
                    try:
                        response = requests.get(
                            "https://stockanalysis.com/etf/spy/history/",
                            timeout=15,
                        )
                        response.raise_for_status()
                        table = pd.read_html(StringIO(response.text))[0]
                        table["Date"] = pd.to_datetime(table["Date"])
                        for _, row in table.iterrows():
                            date_key = row["Date"].strftime("%Y-%m-%d")
                            if date_key in missing_spy_dates:
                                spy_new[date_key] = round(float(row["Close"]), 2)
                    except Exception as fallback_err:
                        print(f"[SPY Cache] Historical fallback failed: {fallback_err}")

                missing_spx_dates = [d for d in missing_dates if d not in spx_new]
                if missing_spx_dates:
                    try:
                        fred_url = (
                            "https://fred.stlouisfed.org/graph/fredgraph.csv"
                            f"?id=SP500&cosd={min(missing_spx_dates)}&coed={max(missing_spx_dates)}"
                        )
                        response = requests.get(fred_url, timeout=15)
                        response.raise_for_status()
                        fred_data = pd.read_csv(StringIO(response.text))
                        for _, row in fred_data.iterrows():
                            date_key = str(row["observation_date"])
                            if date_key in missing_spx_dates and not pd.isna(row["SP500"]):
                                spx_new[date_key] = round(float(row["SP500"]), 2)
                    except Exception as fallback_err:
                        print(f"[SPX Cache] FRED fallback failed: {fallback_err}")

                missing_vix_dates = [
                    d for d in missing_dates
                    if d not in vix_new or vix_new[d] is None or pd.isna(vix_new[d])
                ]
                if missing_vix_dates:
                    try:
                        response = requests.get(
                            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                            timeout=15,
                        )
                        response.raise_for_status()
                        vix_new.update(
                            _extract_cboe_vix_closes(response.text, missing_vix_dates)
                        )
                    except Exception as fallback_err:
                        print(f"[VIX Cache] Cboe historical fallback failed: {fallback_err}")

                print(f"[SPY/SPX/VIX Cache] Fetched {len(spy_new)} SPY, {len(spx_new)} SPX and {len(vix_new)} VIX prices.")
                _price_cache["SPY"].update(spy_new)
                _price_cache["SPX"].update(spx_new)
                _price_cache["VIX"].update(vix_new)

                # Save updated cache
                try:
                    with open(_PRICE_CACHE_FILE, 'w') as _cf:
                        json.dump(_price_cache, _cf)
                except Exception as _save_err:
                    print(f"[SPY/SPX/VIX Cache] Warning: could not save cache: {_save_err}")
            else:
                print(f"[SPY/SPX/VIX Cache] All {len(clean_dates)} dates served from cache.")

            # Build a date-keyed lookup; chart arrays will be aligned later to price_dates
            _spy_cache_map = _price_cache["SPY"]
            _spx_cache_map = _price_cache["SPX"]
            _vix_cache_map = _price_cache["VIX"]
            _spy_count = sum(1 for d in clean_dates if d in _spy_cache_map)
            _spx_count = sum(1 for d in clean_dates if d in _spx_cache_map)
            _vix_count = sum(1 for d in clean_dates if d in _vix_cache_map)
            print(f"[SPY/SPX/VIX Chart] Loaded {_spy_count} SPY, {_spx_count} SPX and {_vix_count} VIX data points.")
        except Exception as yf_err:
            print(f"[SPY/SPX/VIX Chart] Could not fetch closing prices: {yf_err}")
            _spy_cache_map = {}
            _spx_cache_map = {}
            _vix_cache_map = {}
            # Ensure safe fallback for today's date if exception occurred
            try:
                from datetime import datetime as _dt_exc
                today_str_cache = _dt_exc.now().strftime("%Y-%m-%d")
            except Exception:
                today_str_cache = ""

        # Build Chart.js HTML section
        raw_prices_html = ""
        if chart_dates:
            # Filter all tracking arrays to start at 2025-05-01 to avoid pre-May margin spikes
            first_valid_idx = 0
            for i, d in enumerate(chart_dates):
                clean_d = d.replace(' (Live)', '')
                if clean_d >= "2025-05-01":
                    first_valid_idx = i
                    break
                    
            chart_dates = chart_dates[first_valid_idx:]
            chart_prices = chart_prices[first_valid_idx:]
            chart_margins = chart_margins[first_valid_idx:]
            chart_cash_flows = chart_cash_flows[first_valid_idx:]
            chart_realized_gains = chart_realized_gains[first_valid_idx:]
            
            price_dates = chart_dates
            price_values = []
            last_price_value = None
            for value in chart_prices:
                if value is not None:
                    last_price_value = value
                price_values.append(last_price_value)
            # Diagnostic for raw price data table (optional, showing filtered)
            raw_prices_html = "<h3>Filtered Total Option Price Data</h3><div style='max-height: 200px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; font-family: monospace; font-size: 12px;'>"
            for d, v in zip(price_dates, price_values):
                if v is not None:
                    raw_prices_html += f"<div>{d}: ${v:,.2f}</div>"
            raw_prices_html += "</div>"

            # Fetch current portfolio positions for fallback SPY pricing if needed
            current_positions = []
            try:
                current_positions = self.portfolio(minimal=True)
            except Exception:
                pass

            # Align SPY/SPX/VIX closes to the same price_dates used for the chart x-axis
            spy_closes = []
            spx_closes = []
            vix_closes = []
            for d in price_dates:
                clean_d = d.replace(' (Live)', '')
                s_val = _spy_cache_map.get(clean_d)
                x_val = _spx_cache_map.get(clean_d)
                v_val = _vix_cache_map.get(clean_d)
                s_missing = s_val is None or pd.isna(s_val)
                x_missing = x_val is None or pd.isna(x_val)
                
                # A live portfolio quote is valid only for today's point.  Using it
                # for a missing historical close makes several days look flat.
                is_today = (clean_d == today_str_cache)
                is_live = (' (Live)' in d)
                
                if s_missing and (is_today or is_live):
                    if live_index_prices.get("SPY"):
                        s_val = live_index_prices["SPY"]
                        print(f"[SPY Align] Using live quote for {clean_d}: ${s_val}")
                    try:
                        # Search for SPY in the position list to get latest underlying price
                        if (s_val is None or pd.isna(s_val)) and current_positions:
                            for pos in current_positions:
                                symbol = getattr(pos, 'symbol', '') or ''
                                if symbol.upper() == 'SPY':
                                    fallback_val = getattr(pos, 'underlying_last_price', None)
                                    if fallback_val:
                                        s_val = round(float(fallback_val), 2)
                                        print(f"[SPY Align] Using portfolio fallback for {clean_d}: ${s_val}")

                                    break
                    except Exception:
                        pass

                if x_missing and (is_today or is_live):
                    if live_index_prices.get("SPX"):
                        x_val = live_index_prices["SPX"]
                        print(f"[SPX Align] Using live quote for {clean_d}: ${x_val}")
                    try:
                        # Search for SPX/SPXW in the position list to get latest underlying price
                        if (x_val is None or pd.isna(x_val)) and current_positions:
                            for pos in current_positions:
                                symbol = _aggregate_option_symbol(getattr(pos, 'symbol', '') or '')
                                if symbol == 'SPX':
                                    fallback_val = getattr(pos, 'underlying_last_price', None)
                                    if fallback_val:
                                        x_val = round(float(fallback_val), 2)
                                        print(f"[SPX Align] Using portfolio fallback for {clean_d}: ${x_val}")

                                    break
                    except Exception:
                        pass

                spy_closes.append(s_val)
                spx_closes.append(x_val)
                vix_closes.append(v_val)

            # ===== EXTRACT ACTIVE SPX/SPXW SPREADS & BUILD DATASETS =====
            from datetime import datetime as dt_exc, timedelta
            today_str = dt_exc.now().strftime("%Y-%m-%d")
            
            active_spx_spreads = []
            for item in screened_options:
                if not item.get("is_spread"):
                    continue
                short_lot = item.get("short_lot")
                long_lot = item.get("long_lot")
                if not short_lot or not long_lot:
                    continue
                
                sym = _aggregate_option_symbol(getattr(short_lot, "symbol", "") or "")
                if sym != "SPX":
                    continue
                
                # Extract opened date
                opened_date_dt = getattr(short_lot, "date_acquired", None)
                if not opened_date_dt:
                    opened_date_dt = getattr(long_lot, "date_acquired", None)
                
                if opened_date_dt:
                    if hasattr(opened_date_dt, "strftime"):
                        opened_date = opened_date_dt.strftime("%Y-%m-%d")
                    else:
                        opened_date = str(opened_date_dt).split(" ")[0]
                else:
                    # Fallback: 14 days ago
                    opened_date = (dt_exc.now() - timedelta(days=14)).strftime("%Y-%m-%d")
                    
                # Extract expiration date
                exp_date_dt = getattr(short_lot, "expiration_date", None)
                if exp_date_dt:
                    if hasattr(exp_date_dt, "strftime"):
                        exp_date = exp_date_dt.strftime("%Y-%m-%d")
                    else:
                        exp_date = str(exp_date_dt).split(" ")[0]
                else:
                    continue
                    
                s_strike = _as_num(getattr(short_lot, "strike_price", 0.0), 0.0)
                l_strike = _as_num(getattr(long_lot, "strike_price", 0.0), 0.0)
                qty = item.get("pair_quantity", 1)
                
                # Net Delta of the spread
                s_delta = _as_num(getattr(short_lot, "delta", 0.0), 0.0)
                l_delta = _as_num(getattr(long_lot, "delta", 0.0), 0.0)
                s_qty = int(getattr(short_lot, "quantity", 0) or 0)
                l_qty = int(getattr(long_lot, "quantity", 0) or 0)
                net_delta = (s_delta * s_qty) + (l_delta * l_qty)
                
                cp = getattr(short_lot, "call_put", "PUT")
                
                active_spx_spreads.append({
                    "opened_date": opened_date,
                    "expiration_date": exp_date,
                    "short_strike": s_strike,
                    "long_strike": l_strike,
                    "quantity": qty,
                    "net_delta": round(net_delta, 4),
                    "call_put": cp,
                    "gain_loss": item.get("pair_gain_loss", 0.0)
                })

            # Determine maximum future expiration date to extend timeline
            max_future_exp = None
            for spread in active_spx_spreads:
                exp_str = spread["expiration_date"]
                if exp_str > today_str:
                    if not max_future_exp or exp_str > max_future_exp:
                        max_future_exp = exp_str

            # Generate extended labels array
            extended_dates = list(price_dates)
            if max_future_exp and max_future_exp > today_str:
                end_date_obj = dt_exc.strptime(max_future_exp, "%Y-%m-%d")
                curr_date_obj = dt_exc.strptime(today_str, "%Y-%m-%d") + timedelta(days=1)
                while curr_date_obj <= end_date_obj:
                    fut_str = curr_date_obj.strftime("%Y-%m-%d")
                    is_trade_day = curr_date_obj.weekday() < 5  # Simple weekday check
                    if is_trade_day:
                        extended_dates.append(fut_str)
                    curr_date_obj += timedelta(days=1)

            # Re-align spx/vix/price data arrays with extended_dates (padding with None)
            extended_spx_closes = []
            extended_vix_closes = []
            extended_price_values = []
            
            hist_spx_lookup = {d.replace(' (Live)', ''): val for d, val in zip(price_dates, spx_closes)}
            hist_vix_lookup = {d.replace(' (Live)', ''): val for d, val in zip(price_dates, vix_closes)}
            hist_price_lookup = {d.replace(' (Live)', ''): val for d, val in zip(price_dates, price_values)}
            
            for d in extended_dates:
                clean_d = d.replace(' (Live)', '')
                extended_spx_closes.append(hist_spx_lookup.get(clean_d, None))
                extended_vix_closes.append(hist_vix_lookup.get(clean_d, None))
                extended_price_values.append(hist_price_lookup.get(clean_d, None))

            # ===== SCALE X-AXIS FOR SPREADS TIMELINE BASED ON OLDEST OPENED DATE =====
            oldest_open_date = today_str
            if active_spx_spreads:
                oldest_open_date = min(s["opened_date"] for s in active_spx_spreads)
                # Give a 2-day lookback buffer for nice visual padding on the left
                try:
                    buffer_dt = dt_exc.strptime(oldest_open_date, "%Y-%m-%d") - timedelta(days=2)
                    oldest_open_date = buffer_dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
                    
            filtered_extended_dates = []
            filtered_spx_closes = []
            filtered_vix_closes = []
            
            for d, spx, vix in zip(extended_dates, extended_spx_closes, extended_vix_closes):
                clean_d = d.split(' ')[0]
                if clean_d >= oldest_open_date:
                    filtered_extended_dates.append(d)
                    filtered_spx_closes.append(spx)
                    filtered_vix_closes.append(vix)
                    
            # Safe fallback if filtered array becomes empty
            if not filtered_extended_dates:
                filtered_extended_dates = extended_dates
                filtered_spx_closes = extended_spx_closes
                filtered_vix_closes = extended_vix_closes

            # Build Javascript Position Datasets (using filtered timeline!)
            js_position_datasets = []
            for spread in active_spx_spreads:
                s_strike = spread["short_strike"]
                opened = spread["opened_date"]
                expiry = spread["expiration_date"]
                qty = spread["quantity"]
                cp = spread["call_put"]
                net_delta = spread["net_delta"]
                
                # Determine color based on net delta (PUT spread positive delta -> green, CALL spread negative delta -> red)
                if net_delta >= 0:
                    color_str = "rgba(16, 185, 129, 0.85)"  # Green
                else:
                    color_str = "rgba(239, 68, 68, 0.85)"   # Red
                    
                # Thickness scales with quantity, but let's make it much thinner and cleaner
                thickness = min(3.5, 1.2 + qty * 0.15)
                
                # Build active dataset (solid horizontal line from opened_date to today_str)
                active_data = []
                for d in filtered_extended_dates:
                    clean_d = d.split(' ')[0]
                    if opened <= clean_d <= today_str:
                        active_data.append(s_strike)
                    else:
                        active_data.append(None)
                        
                active_ds = {
                    "label": f"SPX {cp} Short Strike {s_strike}",
                    "data": active_data,
                    "borderColor": color_str,
                    "borderWidth": thickness,
                    "hoverBorderWidth": thickness + 1.5,
                    "fill": False,
                    "pointRadius": 0,
                    "pointHoverRadius": 4,
                    "yAxisID": "y",  # Align with left Y-axis (SPX Price) in spySpreadsChart!
                    "spanGaps": False,
                    "positionDetails": spread
                }
                js_position_datasets.append(active_ds)
                
                # Build future extension dataset (dashed horizontal line from today_str to expiry)
                ext_data = []
                for d in filtered_extended_dates:
                    clean_d = d.split(' ')[0]
                    if today_str <= clean_d <= expiry:
                        ext_data.append(s_strike)
                    else:
                        ext_data.append(None)
                        
                ext_ds = {
                    "label": f"SPX {cp} Short Strike {s_strike} (Ext)",
                    "data": ext_data,
                    "borderColor": color_str,
                    "borderWidth": thickness,
                    "hoverBorderWidth": thickness + 1.5,
                    "borderDash": [5, 5],
                    "fill": False,
                    "pointRadius": 0,
                    "pointHoverRadius": 4,
                    "yAxisID": "y",  # Align with left Y-axis (SPX Price) in spySpreadsChart!
                    "spanGaps": False,
                    "positionDetails": spread
                }
                js_position_datasets.append(ext_ds)

            # Diagnostic logging
            if len(price_dates) > 0:
                print(f"[SPX/VIX Align] First: {price_dates[0]} -> SPX: {spx_closes[0]}, VIX: {vix_closes[0]}")
                print(f"[SPX/VIX Align] Last:  {price_dates[-1]} -> SPX: {spx_closes[-1]}, VIX: {vix_closes[-1]}")
                print(f"[SPX/VIX Align] Spreads Zoomed Range: {filtered_extended_dates[0]} to {filtered_extended_dates[-1]} (Total: {len(filtered_extended_dates)} days)")


            sync_status = cash_flow_sync.get("status")
            sync_timestamp = (
                cash_flow_sync.get("last_successful_at")
                if sync_status == "error"
                else cash_flow_sync.get("attempted_at")
            )
            if sync_timestamp:
                try:
                    sync_timestamp = (
                        datetime.fromisoformat(sync_timestamp)
                        .astimezone(pytz.timezone("US/Eastern"))
                        .strftime("%b %d, %Y %I:%M %p ET")
                    )
                except (TypeError, ValueError):
                    pass

            if sync_status == "error":
                sync_badge_color = "#b45309"
                sync_badge_background = "#fffbeb"
                sync_badge_border = "#f59e0b"
                sync_badge_text = (
                    "Cash-flow sync delayed; showing the last confirmed history"
                    + (f" from {sync_timestamp}" if sync_timestamp else "")
                    + "."
                )
            elif sync_status == "ok":
                sync_badge_color = "#166534"
                sync_badge_background = "#f0fdf4"
                sync_badge_border = "#86efac"
                sync_badge_text = (
                    "Cash-flow orders synced"
                    + (f" {sync_timestamp}" if sync_timestamp else "")
                    + f" · {cash_flow_sync.get('orders_fetched', 0)} orders"
                )
            else:
                sync_badge_color = "#475569"
                sync_badge_background = "#f8fafc"
                sync_badge_border = "#cbd5e1"
                sync_badge_text = "Cash-flow sync status unavailable"

            from html import escape as html_escape
            sync_badge_text = html_escape(sync_badge_text)

            chart_html = f'''
            <div style="margin-bottom: 30px; max-width: 1200px;">
              <h2 style="margin-bottom: 15px;">SPY Benchmark Performance</h2>
              <div style="display: flex; align-items: center; gap: 8px; margin: 0 0 12px 0; flex-wrap: wrap;">
                <span style="font-size: 13px; font-weight: 600; color: #4b5563;">Time span:</span>
                <button type="button" data-benchmark-range="1month" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">1month</button>
                <button type="button" data-benchmark-range="3month" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">3month</button>
                <button type="button" data-benchmark-range="1year" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">1year</button>
              </div>
              <div style="position: relative; height: 400px; width: 100%;">
                <canvas id="spyPriceChart"></canvas>
              </div>
            </div>
            
            <div style="margin-bottom: 30px; max-width: 1200px;">
              <h2 style="margin-bottom: 15px;">SPX Active Credit Spreads Timeline</h2>
              <div style="position: relative; height: 600px; width: 100%;">
                <canvas id="spySpreadsChart"></canvas>
              </div>
            </div>
            
            <div style="margin-bottom: 30px; max-width: 1200px;">
              <h2 style="margin-bottom: 15px;">SPY/SPX/SPXW Cash Flow, Realized Gain & Margin</h2>
              <div style="display: flex; align-items: center; gap: 8px; margin: 0 0 12px 0; flex-wrap: wrap;">
                <span style="font-size: 13px; font-weight: 600; color: #4b5563;">Time span:</span>
                <button type="button" data-cashflow-range="1month" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">1month</button>
                <button type="button" data-cashflow-range="3month" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">3month</button>
                <button type="button" data-cashflow-range="1year" style="padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #111827; cursor: pointer;">1year</button>
              </div>
              <div id="cashFlowSyncStatus" data-sync-status="{sync_status or 'unknown'}" style="display: inline-block; margin: 0 0 12px 0; padding: 6px 10px; border: 1px solid {sync_badge_border}; border-radius: 999px; background: {sync_badge_background}; color: {sync_badge_color}; font-size: 12px; font-weight: 600;">
                {sync_badge_text}
              </div>
              <div style="position: relative; height: 400px; width: 100%;">
                <canvas id="spyChart"></canvas>
              </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
              const labels = {json.dumps(chart_dates)};
              const priceLabels = {json.dumps(price_dates)};
              const extendedLabels = {json.dumps(filtered_extended_dates)};
              const benchmarkSeries = {{
                labels: priceLabels,
                optionValues: {json.dumps(price_values)},
                spyCloses: {json.dumps(spy_closes)},
                vixCloses: {json.dumps(vix_closes)}
              }};
              const cashFlowSeries = {{
                labels: labels,
                cashFlows: {json.dumps(chart_cash_flows)},
                realizedGains: {json.dumps(chart_realized_gains)},
                margins: {json.dumps(chart_margins)}
              }};
              
              // Helper to generate monthly labels for X axis
              const xAxisTickCallback = function(value, index, ticks) {{
                const label = this.getLabelForValue(value);
                const parts = label.split('-');
                if (parts.length < 2) return '';
                const yearMonth = parts[0] + '-' + parts[1];
                
                let isFirstOfMonth = true;
                for (let i = 0; i < index; i++) {{
                  const prevLabel = this.getLabelForValue(i);
                  const prevParts = prevLabel.split('-');
                  if (prevParts.length >= 2 && prevParts[0] + '-' + prevParts[1] === yearMonth) {{
                    isFirstOfMonth = false;
                    break;
                  }}
                }}
                return isFirstOfMonth ? yearMonth : '';
              }};

              const commonXAxis = {{
                ticks: {{
                  callback: xAxisTickCallback,
                  maxRotation: 45,
                  minRotation: 45,
                  autoSkip: false
                }}
              }};

              const parseBenchmarkDate = function(label) {{
                const cleanLabel = String(label || '').split(' ')[0];
                return new Date(`${{cleanLabel}}T00:00:00`);
              }};

              const filterBenchmarkSeries = function(rangeKey) {{
                if (!benchmarkSeries.labels.length) {{
                  return benchmarkSeries;
                }}

                const endDate = parseBenchmarkDate(benchmarkSeries.labels[benchmarkSeries.labels.length - 1]);
                const cutoff = new Date(endDate.getTime());

                if (rangeKey === '1month') {{
                  cutoff.setMonth(cutoff.getMonth() - 1);
                }} else if (rangeKey === '3month') {{
                  cutoff.setMonth(cutoff.getMonth() - 3);
                }} else if (rangeKey === '1year') {{
                  cutoff.setFullYear(cutoff.getFullYear() - 1);
                }}

                const filtered = {{
                  labels: [],
                  optionValues: [],
                  spyCloses: [],
                  vixCloses: []
                }};

                benchmarkSeries.labels.forEach((label, idx) => {{
                  const parsedDate = parseBenchmarkDate(label);
                  if (!Number.isNaN(parsedDate.getTime()) && parsedDate >= cutoff) {{
                    filtered.labels.push(label);
                    filtered.optionValues.push(benchmarkSeries.optionValues[idx]);
                    filtered.spyCloses.push(benchmarkSeries.spyCloses[idx]);
                    filtered.vixCloses.push(benchmarkSeries.vixCloses[idx]);
                  }}
                }});

	                return filtered.labels.length ? filtered : benchmarkSeries;
	              }};

              const filterCashFlowSeries = function(rangeKey) {{
                if (!cashFlowSeries.labels.length) {{
                  return cashFlowSeries;
                }}

                const endDate = parseBenchmarkDate(cashFlowSeries.labels[cashFlowSeries.labels.length - 1]);
                const cutoff = new Date(endDate.getTime());

                if (rangeKey === '1month') {{
                  cutoff.setMonth(cutoff.getMonth() - 1);
                }} else if (rangeKey === '3month') {{
                  cutoff.setMonth(cutoff.getMonth() - 3);
                }} else if (rangeKey === '1year') {{
                  cutoff.setFullYear(cutoff.getFullYear() - 1);
                }}

                const filtered = {{
                  labels: [],
                  cashFlows: [],
                  realizedGains: [],
                  margins: []
                }};

                cashFlowSeries.labels.forEach((label, idx) => {{
                  const parsedDate = parseBenchmarkDate(label);
                  if (!Number.isNaN(parsedDate.getTime()) && parsedDate >= cutoff) {{
                    filtered.labels.push(label);
                    filtered.cashFlows.push(cashFlowSeries.cashFlows[idx]);
                    filtered.realizedGains.push(cashFlowSeries.realizedGains[idx]);
                    filtered.margins.push(cashFlowSeries.margins[idx]);
                  }}
                }});

                return filtered.labels.length ? filtered : cashFlowSeries;
              }};

              const defaultBenchmarkRange = '1month';
              const initialBenchmarkSeries = filterBenchmarkSeries(defaultBenchmarkRange);
              const defaultCashFlowRange = '1year';
              const initialCashFlowSeries = filterCashFlowSeries(defaultCashFlowRange);

              // --- Chart 1: Total Option Price ---
              const priceCtx = document.getElementById('spyPriceChart').getContext('2d');
              const benchmarkChart = new Chart(priceCtx, {{
                type: 'line',
                data: {{
                  labels: initialBenchmarkSeries.labels,
                  datasets: [{{
                    label: 'SPY + SPX Option Value ($)',
                    data: initialBenchmarkSeries.optionValues,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.2,
                    yAxisID: 'y'
                  }},
                  {{
                    label: 'SPY Close ($)',
                    data: initialBenchmarkSeries.spyCloses,
                    borderColor: 'rgb(34, 139, 34)',
                    backgroundColor: 'rgba(34, 139, 34, 0.05)',
                    fill: false,
                    tension: 0.2,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    spanGaps: true,
                    yAxisID: 'y1'
                  }},
                  {{
                    label: 'VIX Close',
                    data: initialBenchmarkSeries.vixCloses,
                    borderColor: 'rgb(255, 140, 0)',
                    backgroundColor: 'rgba(255, 140, 0, 0.05)',
                    fill: false,
                    tension: 0.2,
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    pointRadius: 0,
                    spanGaps: true,
                    yAxisID: 'y2'
                  }}]
                }},
                options: {{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {{ mode: 'index', intersect: false }},
                  plugins: {{
                    title: {{ display: true, text: 'SPY + SPX Options: Total Portfolio Value' }},
                    tooltip: {{
                      callbacks: {{
                        label: function(ctx) {{
                          const val = ctx.parsed.y;
                          if (val == null) return null;
                          if (ctx.dataset.yAxisID === 'y2') return ctx.dataset.label + ': ' + val.toFixed(2);
                          return ctx.dataset.label + ': $' + val.toLocaleString('en-US', {{minimumFractionDigits: 2}});
                        }}
                      }}
                    }}
                  }},
                  scales: {{
                    x: commonXAxis,
                    y: {{
                      position: 'left',
                      beginAtZero: false,
                      grace: '5%',
                      title: {{ display: true, text: 'Option Portfolio Value ($)', color: 'rgb(54, 162, 235)' }},
                      ticks: {{ callback: (v) => '$' + v.toLocaleString(), color: 'rgb(54, 162, 235)' }}
                    }},
                    y1: {{
                      position: 'right',
                      beginAtZero: false,
                      grace: '5%',
                      title: {{ display: true, text: 'SPY Price ($)', color: 'rgb(34, 139, 34)' }},
                      ticks: {{ callback: (v) => '$' + v.toLocaleString(), color: 'rgb(34, 139, 34)' }},
                      grid: {{ drawOnChartArea: false }}
                    }},
                    y2: {{
                      position: 'right',
                      beginAtZero: false,
                      grace: '5%',
                      title: {{ display: true, text: 'VIX Level', color: 'rgb(255, 140, 0)' }},
                      ticks: {{ color: 'rgb(255, 140, 0)' }},
                      grid: {{ drawOnChartArea: false }}
                    }}
                  }}
                }}
              }});

              const benchmarkRangeButtons = document.querySelectorAll('[data-benchmark-range]');
              const setBenchmarkRange = function(rangeKey) {{
                const filtered = filterBenchmarkSeries(rangeKey);
                benchmarkChart.data.labels = filtered.labels;
                benchmarkChart.data.datasets[0].data = filtered.optionValues;
                benchmarkChart.data.datasets[1].data = filtered.spyCloses;
                benchmarkChart.data.datasets[2].data = filtered.vixCloses;
                benchmarkChart.update();
                benchmarkRangeButtons.forEach((button) => {{
                  const isActive = button.dataset.benchmarkRange === rangeKey;
                  button.style.background = isActive ? '#1d4ed8' : '#f8fafc';
                  button.style.borderColor = isActive ? '#1d4ed8' : '#cbd5e1';
                  button.style.color = isActive ? '#ffffff' : '#111827';
                }});
              }};

              benchmarkRangeButtons.forEach((button) => {{
                button.addEventListener('click', () => setBenchmarkRange(button.dataset.benchmarkRange));
              }});
              setBenchmarkRange(defaultBenchmarkRange);

              // --- Chart 2 (NEW): SPX Active Credit Spreads Timeline ---
              const spreadsCtx = document.getElementById('spySpreadsChart').getContext('2d');
              new Chart(spreadsCtx, {{
                type: 'line',
                data: {{
                  labels: extendedLabels,
                  datasets: [{{
                    label: 'SPX Close ($)',
                    data: {json.dumps(filtered_spx_closes)},
                    borderColor: 'rgb(34, 139, 34)',
                    backgroundColor: 'rgba(34, 139, 34, 0.05)',
                    fill: false,
                    tension: 0.2,
                    borderWidth: 2,
                    pointRadius: 0,
                    spanGaps: true,
                    yAxisID: 'y'
                  }},
                  {{
                    label: 'VIX Close',
                    data: {json.dumps(filtered_vix_closes)},
                    borderColor: 'rgb(255, 140, 0)',
                    backgroundColor: 'rgba(255, 140, 0, 0.05)',
                    fill: false,
                    tension: 0.2,
                    borderWidth: 1.5,
                    borderDash: [5, 3],
                    pointRadius: 0,
                    spanGaps: true,
                    yAxisID: 'y1'
                  }}].concat({json.dumps(js_position_datasets)})
                }},
                options: {{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {{ mode: 'nearest', intersect: false, axis: 'xy' }},
                  plugins: {{
                    legend: {{
                      labels: {{
                        filter: function(item, chart) {{
                          return item.text === 'SPX Close ($)' || item.text === 'VIX Close';
                        }}
                      }}
                    }},
                    title: {{ display: true, text: 'Active SPX Credit Spreads vs Spot Price' }},
                    tooltip: {{
                      callbacks: {{
                        label: function(ctx) {{
                          const dataset = ctx.dataset;
                          const val = ctx.parsed.y;
                          if (val == null) return null;
                          if (dataset.positionDetails) {{
                            const details = dataset.positionDetails;
                            return [
                              dataset.label.replace(' (Ext)', ''),
                              `  Short Strike: $${{details.short_strike}}`,
                              `  Long Strike: $${{details.long_strike}}`,
                              `  Quantity: ${{details.quantity}} pairs`,
                              `  Net Delta: ${{details.net_delta}}`,
                              `  Opened Date: ${{details.opened_date}}`,
                              `  Expiration: ${{details.expiration_date}}`,
                              `  Gain/Loss: ${{details.gain_loss}}%`
                            ];
                          }}
                          if (ctx.dataset.yAxisID === 'y1') return ctx.dataset.label + ': ' + val.toFixed(2);
                          return ctx.dataset.label + ': $' + val.toLocaleString('en-US', {{minimumFractionDigits: 2}});
                        }}
                      }}
                    }}
                  }},
                  scales: {{
                    x: commonXAxis,
                    y: {{
                      position: 'left',
                      beginAtZero: false,
                      grace: '5%',
                      title: {{ display: true, text: 'SPX Price ($)', color: 'rgb(34, 139, 34)' }},
                      ticks: {{ callback: (v) => '$' + v.toLocaleString(), color: 'rgb(34, 139, 34)' }}
                    }},
                    y1: {{
                      position: 'right',
                      beginAtZero: false,
                      grace: '5%',
                      title: {{ display: true, text: 'VIX Level', color: 'rgb(255, 140, 0)' }},
                      ticks: {{ color: 'rgb(255, 140, 0)' }},
                      grid: {{ drawOnChartArea: false }}
                    }}
                  }}
                }}
              }});

	              // --- Chart 2: Cash Flow, Gains & Margin ---
	              const mainCtx = document.getElementById('spyChart').getContext('2d');
	              const cashFlowChart = new Chart(mainCtx, {{
	                type: 'line',
	                data: {{
	                  labels: initialCashFlowSeries.labels,
	                  datasets: [
	                    {{
	                      label: 'SPY/SPX/SPXW Cumulative Cash Flow ($)',
	                      data: initialCashFlowSeries.cashFlows,
                      borderColor: 'rgb(75, 192, 92)',
                      backgroundColor: 'rgba(75, 192, 92, 0.1)',
                      fill: false,
                      tension: 0.2,
                      yAxisID: 'y'
                    }},
	                    {{
	                      label: 'SPY/SPX/SPXW Realized Gain ($)',
	                      data: initialCashFlowSeries.realizedGains,
                      borderColor: 'rgb(255, 159, 64)',
                      backgroundColor: 'rgba(255, 159, 64, 0.1)',
                      fill: false,
                      tension: 0.2,
                      borderDash: [5, 5],
                      yAxisID: 'y'
                    }},
	                    {{
	                      label: 'SPY/SPX/SPXW Total Margin Required ($)',
	                      data: initialCashFlowSeries.margins,
                      borderColor: 'rgb(255, 99, 132)',
                      backgroundColor: 'rgba(255, 99, 132, 0.1)',
                      fill: false,
                      tension: 0.2,
                      spanGaps: false,
                      yAxisID: 'y1'
                    }}
                  ]
                }},
                options: {{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {{ mode: 'index', intersect: false }},
                  plugins: {{
                    title: {{ display: true, text: 'SPY/SPX/SPXW Options: Performance & Risk' }},
                    tooltip: {{
                      callbacks: {{
                        label: (ctx) => ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2}})
                      }}
                    }}
                  }},
                  scales: {{
                    x: commonXAxis,
                    y: {{
                      position: 'left',
                      title: {{ display: true, text: 'SPY/SPX/SPXW Cash Flow / Gains ($)' }},
                      ticks: {{ callback: (v) => '$' + v.toLocaleString() }}
                    }},
                    y1: {{
                      position: 'right',
                      title: {{ display: true, text: 'SPY/SPX/SPXW Margin Required ($)' }},
                      ticks: {{ callback: (v) => '$' + v.toLocaleString() }},
                      grid: {{ drawOnChartArea: false }}
                    }}
	                  }}
	                }}
	              }});

              const cashFlowRangeButtons = document.querySelectorAll('[data-cashflow-range]');
              const setCashFlowRange = function(rangeKey) {{
                const filtered = filterCashFlowSeries(rangeKey);
                cashFlowChart.data.labels = filtered.labels;
                cashFlowChart.data.datasets[0].data = filtered.cashFlows;
                cashFlowChart.data.datasets[1].data = filtered.realizedGains;
                cashFlowChart.data.datasets[2].data = filtered.margins;
                cashFlowChart.update();
                cashFlowRangeButtons.forEach((button) => {{
                  const isActive = button.dataset.cashflowRange === rangeKey;
                  button.style.background = isActive ? '#1d4ed8' : '#f8fafc';
                  button.style.borderColor = isActive ? '#1d4ed8' : '#cbd5e1';
                  button.style.color = isActive ? '#ffffff' : '#111827';
                }});
              }};

              cashFlowRangeButtons.forEach((button) => {{
                button.addEventListener('click', () => setCashFlowRange(button.dataset.cashflowRange));
              }});
              setCashFlowRange(defaultCashFlowRange);
	            </script>
            '''
        else:
            chart_html = '<p style="color: #666; margin-bottom: 20px;"><em>No SPY tracking history available yet. Data will appear after the first portfolio refresh.</em></p>'
        
        import json as json_mod
        # Add refresh button styling and script
        refresh_button_html = '''
            <div style="margin-bottom: 20px;">
              <button id="refreshBtn" onclick="refreshPricing()" style="
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                color: white;
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                border: none;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.2s;
              " onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.3)';" 
                 onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.2)';">
                🔄 Refresh Pricing
              </button>
              <span id="refreshStatus" style="margin-left: 12px; font-size: 13px; color: #666;"></span>
            </div>
            <script>
              async function refreshPricing() {
                const btn = document.getElementById('refreshBtn');
                const status = document.getElementById('refreshStatus');
                btn.disabled = true;
                btn.innerHTML = '⏳ Refreshing...';
                btn.style.background = 'linear-gradient(135deg, #888 0%, #666 100%)';
                status.textContent = 'Fetching data from E*TRADE...';
                status.style.color = '#666';
                
                try {
                  // Use relative path to avoid localhost issues on mobile/external access, but fallback for local files
                  const endpoint = window.location.protocol === 'file:' ? 'http://localhost:8765/refresh' : '/refresh';
                  const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                  });
                  
                  if (response.ok) {
                    const data = await response.json();
                    status.textContent = '✅ ' + (data.message || 'Refresh complete!');
                    status.style.color = '#10b981';
                    setTimeout(() => { window.location.reload(); }, 1000);
                  } else {
                    throw new Error('Server returned ' + response.status);
                  }
                } catch (err) {
                  status.textContent = '❌ Error: ' + err.message;
                  status.style.color = '#ef4444';
                  btn.innerHTML = '🔄 Refresh Pricing';
                  btn.style.background = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)';
                  btn.disabled = false;
                }
              }

            </script>
        '''

        market_close_auto_reload_html = '''
            <script>
              (function scheduleMarketCloseReload() {
                const storageKey = 'screened-option-pairs-market-close-reload';

                function easternParts(now) {
                  const parts = new Intl.DateTimeFormat('en-US', {
                    timeZone: 'America/New_York',
                    weekday: 'short',
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false
                  }).formatToParts(now);
                  return Object.fromEntries(parts.map((part) => [part.type, part.value]));
                }

                function maybeReloadForClose() {
                  const parts = easternParts(new Date());
                  if (parts.weekday === 'Sat' || parts.weekday === 'Sun') return;
                  const hour = Number(parts.hour);
                  const minute = Number(parts.minute);
                  const inPostCloseWindow = hour === 16 && minute >= 6 && minute <= 45;
                  if (!inPostCloseWindow) return;

                  const bucket = Math.floor(minute / 5);
                  const reloadToken = `${parts.year}-${parts.month}-${parts.day}-${hour}-${bucket}`;
                  if (localStorage.getItem(storageKey) === reloadToken) return;
                  localStorage.setItem(storageKey, reloadToken);
                  window.location.reload();
                }

                maybeReloadForClose();
                window.setInterval(maybeReloadForClose, 60000);
              })();
            </script>
        '''

        html_doc = f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
            <title>{html.escape(title)}</title>
            {style}
          </head>
          <body>
            <div class="read-only-banner" role="status">Read only — all E*TRADE order actions are disabled</div>
            <h1>{html.escape(title)}</h1>
            {refresh_button_html if show_refresh else ""}
            {market_close_auto_reload_html}
            <div class="position-groups" aria-label="Option positions grouped by ticker">
              {position_groups_html}
            </div>
            <div class="portfolio-summary">
              <table class="summary-table" aria-label="Portfolio totals">
                <tbody>{''.join(rows_html)}</tbody>
              </table>
            </div>
            <div class="desktop-analytics">{chart_html}</div>
            <div class="diagnostic-data">{raw_prices_html}</div>
            {tracker_summary_html}
          </body>
        </html>
        """
        Path(out_path).write_text(html_doc, encoding="utf-8")
        return out_path
    
    def get_option_trade(self, stock_positions:List[StockPosition], print_enable=False, find_roll = False):

        positions_need_actions = []

        for pos in stock_positions:
            if find_roll == False:
                break
            # For roll out, loss must be big enough and strike distance must be small enough 
            # For roll in, if gain is big enough or strike distance big enough. This is to keep the expiration date close and have higher deacy
            # if Earning is close, DO NOT ROLL IN. And set ROLL_OUT threshold lower
            if pos.symbol in EXEMPTED_STOCKS:
                continue

            # skew_check = self.analyze_option_chain(pos)
            # if skew_check is not None:
            #     pos.iv_skew = skew_check['average_iv_skew']
            # print(f"{pos.symbol} iv skew: {pos.iv_skew}")

            correlated_group = None
            for group_name, group_stocks in CORRELATED_STOCKS.items():
                if pos.symbol in group_stocks:
                    correlated_group = group_stocks
                    # print(f"{pos.symbol} group correlated: {correlated_group}")
                    break

            # Determine if any stock in the correlated group has earnings close
            earnings_close = False
            earning_triggered_stock = None
            days_to_earnings = None
            if correlated_group:
                for sub_pos in stock_positions:
                    if sub_pos.symbol in correlated_group and sub_pos.days_to_earnings is not None and 0 <= sub_pos.days_to_earnings < 7:
                        earnings_close = True
                        earning_triggered_stock = sub_pos.symbol
                        days_to_earnings = sub_pos.days_to_earnings
                        break
                    
            # if pos.days_to_earnings is not None and 0 < pos.days_to_earnings < 7:
            if earnings_close:
                print(f"{pos.symbol} is alerted because {earning_triggered_stock} is {days_to_earnings} days from earnings")
                # THIS_ROLL_OUT_GL_THRESHOLD = ROLL_OUT_GL_THRESHOLD / 4
                THIS_ROLL_OUT_GL_THRESHOLD = 100
                THIS_ROLL_OUT_DISTANCE_THRESHOLD = 4 * ROLL_OUT_DISTANCE_THRESHOLD
                THIS_ROLL_IN_GL_THRESHOLD = ROLL_IN_GL_THRESHOLD * 4
                THIS_ROLL_IN_DISTANCE_THRESHOLD = ROLL_IN_DISTANCE_THRESHOLD * 4
            else:
                THIS_ROLL_OUT_GL_THRESHOLD = ROLL_OUT_GL_THRESHOLD
                THIS_ROLL_OUT_DISTANCE_THRESHOLD = ROLL_OUT_DISTANCE_THRESHOLD
                THIS_ROLL_IN_GL_THRESHOLD = ROLL_IN_GL_THRESHOLD
                THIS_ROLL_IN_DISTANCE_THRESHOLD = ROLL_IN_DISTANCE_THRESHOLD
            
            #debug######
                                # print(f"Processing {pos.symbol} {pos.quantity} to check whether there is need for option roll")
            # if pos.distance_to_strike is not None and pos.quantity < 0 and pos.call_put == "CALL" and (( pos.gain_loss_percentage < THIS_ROLL_OUT_GL_THRESHOLD and pos.distance_to_strike > -THIS_ROLL_OUT_DISTANCE_THRESHOLD * pos.volatility) or pos.last_price < 1.2 * pos.option_intrinsic ) :
            need_action_flag = False
            if pos.distance_to_strike is not None and pos.quantity <0:
                # hedge_position = self.get_option_chain(pos,pos.call_put,"HEDGE",earnings_close,optimizer="DTE")
                # print("position: ",pos)
                # print("hedge: ",hedge_position)
                # breakpoint()
                if pos.days_to_expiration < MIN_DTE:
                    print(f"Rolling out {pos.symbol}, {pos.quantity}, {pos.call_put} due to close to expiration, intrinsic value: {pos.option_intrinsic}, price: {pos.last_price}")
                    # print(self.get_option_chain(pos,pos.call_put,"ROLL_OUT",earnings_close,optimizer="DTE"))
                    # breakpoint()
                    pos.target_option_roll, pos.target_hedge_option_roll = self.get_option_chain(pos,pos.call_put,"ROLL_OUT",earnings_close,optimizer="DTE")
                    need_action_flag = True
                elif pos.call_put == "CALL" and pos.last_price < 1.2 * pos.option_intrinsic:
                        print(f"Rolling out due to assignment risk {pos.symbol}, {pos.quantity},{pos.call_put} intrinsic value: {pos.option_intrinsic}, price: {pos.last_price}")
                        pos.target_option_roll, pos.target_hedge_option_roll = self.get_option_chain(pos,"CALL","ROLL_OUT",earnings_close,optimizer="DTE")
                        need_action_flag = True
                                # elif pos.call_put == "CALL" and (pos.distance_to_strike < -(pos.days_to_expiration**0.5) * pos.volatility) and pos.days_to_expiration >= MIN_DTE:
                                #         print("Rolling in for strike price distance",pos.symbol,pos.quantity,pos.call_put)
                                #         pos.target_option_roll, pos.target_hedge_option_roll = self.get_option_chain(pos,"CALL","ROLL_IN",earnings_close)
                                #         need_action_flag = True
                # if pos.distance_to_strike is not None and pos.quantity < 0 and pos.call_put == "PUT" and ((pos.gain_loss_percentage < THIS_ROLL_OUT_GL_THRESHOLD and pos.distance_to_strike < THIS_ROLL_OUT_DISTANCE_THRESHOLD * pos.volatility ) or pos.last_price < 1.2 * pos.option_intrinsic) :
                elif pos.call_put == "PUT" and pos.last_price < 1.2 * pos.option_intrinsic:
                        print(f"Rolling out due to assignment risk {pos.symbol}, {pos.quantity}, {pos.call_put} intrinsic value: {pos.option_intrinsic}, price: {pos.last_price}")
                        pos.target_option_roll, pos.target_hedge_option_roll = self.get_option_chain(pos,"PUT","ROLL_OUT",earnings_close,optimizer="DTE")
                        need_action_flag = True
                                # elif pos.call_put == "PUT" and (pos.distance_to_strike > (pos.days_to_expiration**0.5) * pos.volatility) and pos.days_to_expiration >= MIN_DTE:
                                #         print("Rolling in for strike price distance",pos.symbol,pos.quantity,pos.call_put)
                                #         pos.target_option_roll, pos.target_hedge_option_roll = self.get_option_chain(pos,"PUT","ROLL_IN",earnings_close)
                                #         need_action_flag = True
                if need_action_flag == True:
                    if pos.target_option_roll is not None and ( pos.target_option_roll['strikePrice'] != pos.strike_price or pos.target_option_roll['expiryDate'] !=pos.expiration_date ):
                        print(f"Appending sell option to positions_need_actions: {pos.symbol}, {pos.target_option_roll}")
                        print(f"Appending buy option to positions_need_actions: {pos.symbol}, {pos.target_hedge_option_roll}")
                        positions_need_actions.append(pos)
                    else:
                        print(f"{pos.symbol} target roll option is not generated: target_option_roll: {pos.target_option_roll},target_hedge_option_roll: {pos.target_hedge_option_roll}")

        actions_option_trade_html(stock_positions)
        cover_option_position = self.print_stocks_with_negative_options(stock_positions)
        positions_over_100 = [position for position in stock_positions if position.quantity >= 100]

        # Calculate covered shares, ensuring only valid covered calls are counted
        covered_shares_summary = {}
        for stock_position in positions_over_100:
            symbol = stock_position.symbol
            # Filter to include only covered call options for this stock symbol
            covered_options = [
                opt for opt in cover_option_position
                if opt.symbol == symbol and opt.call_put == "CALL" and opt.security_type == "Option"
            ]

            # Calculate total covered shares (1 option contract typically covers 100 shares)
            total_covered_shares = sum(100 * abs(opt.quantity) for opt in covered_options)
            uncovered_shares = max(0, stock_position.quantity - total_covered_shares)

            # Store the coverage information for this symbol
            covered_shares_summary[symbol] = {
                "total_shares": stock_position.quantity,
                "covered_shares": total_covered_shares,
                "uncovered_shares": uncovered_shares,
            }

        # Print summary
        if print_enable == True:
            print("\nSummary of Positions with >= 100 uncovered Shares:")
            for symbol, summary in covered_shares_summary.items():
                if summary['uncovered_shares'] >=100:
                    print(f"Symbol: {symbol}")
                    print(f"Total Shares: {summary['total_shares']}")
                    print(f"Covered Shares: {summary['covered_shares']}")
                    print(f"Uncovered Shares: {summary['uncovered_shares']}\n")

        return positions_need_actions

    def manual_option_input(self, symbol: str, call_put: str, expiration_date: str, strike_price: float):
        """
        Manually input option details, poll the option price from E*TRADE API, and return a StockPosition object.

        Parameters:
        - symbol (str): The underlying stock symbol.
        - call_put (str): Type of option ('CALL' or 'PUT').
        - expiration_date (str): Expiration date in 'YYYY-MM-DD' format.
        - strike_price (float): Strike price of the option.

        Returns:
        - StockPosition: An object representing the option's position with price details.
        """

        # Step 1: Convert expiration_date to datetime
        try:
            expiry_date_obj = datetime.strptime(expiration_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid expiration date format. Use 'YYYY-MM-DD'.")
            return None

        # Step 2: Poll E*TRADE API for option price
        option_price = self.get_option_price(symbol, call_put, expiry_date_obj, strike_price)
        if option_price is None:
            print("Failed to retrieve option price from E*TRADE API.")
            return None

        contract_symbol = getattr(self, "_last_option_price_symbol", symbol)

        # Step 3: Create and return a StockPosition object with the option details
        stock_position = StockPosition(
            symbol=contract_symbol,
            quantity=1,  # Set default quantity as 1, or adjust based on requirements
            last_price=option_price,
            price_paid=option_price,
            total_gain=0.0,
            market_value=option_price,
            position_id="manual_input",
            position_type=call_put,
            security_type="Option",
            date_acquired="None"
        )

        # Additional attributes specific to the option
        stock_position.call_put = call_put
        stock_position.expiration_date = expiration_date
        stock_position.strike_price = strike_price

        return stock_position

    def balance(self):
        """
        Calls account balance API to retrieve the current balance and related details for a specified account

        :param self: Pass in parameters authenticated session and information on selected account
        """
        # URL for the API endpoint
        url = self.base_url + "/v1/accounts/" + self.account["accountIdKey"] + "/balance.json"

        # Add parameters and header information
        params = {"instType": self.account["institutionType"], "realTimeNAV": "true"}
        # headers = {"consumerkey": config["DEFAULT"]["CONSUMER_KEY"]}

        # Make API call for GET request
        # response = self.session.get(url, header_auth=True, params=params, headers=headers)
        response = self.session.get(url, header_auth=True, params=params)
        logger.debug("Account balance request issued")
        logger.debug("Request Header: %s", redact_http_headers(response.request.headers))

        # Handle and parse response
        if response is not None and response.status_code == 200:
            parsed = json.loads(response.text)
            logger.debug("Response Body: %s", json.dumps(parsed, indent=4, sort_keys=True))
            data = response.json()
            if data is not None and "BalanceResponse" in data:
                balance_data = data["BalanceResponse"]
                if balance_data is not None and "accountId" in balance_data:
                    print("\n\nBalance for " + _redact_account_identifier(balance_data["accountId"]) + ":")
                else:
                    print("\n\nBalance:")
                # Display balance information
                if balance_data is not None and "accountDescription" in balance_data:
                    print("Account description withheld")
                if balance_data is not None and "Computed" in balance_data \
                        and "RealTimeValues" in balance_data["Computed"] \
                        and "totalAccountValue" in balance_data["Computed"]["RealTimeValues"]:
                    print("Net Account Value: "
                          + str('${:,.2f}'.format(balance_data["Computed"]["RealTimeValues"]["totalAccountValue"])))
                if balance_data is not None and "Computed" in balance_data:
                    comp = balance_data["Computed"]
                    print(f"DEBUG: Computed keys: {list(comp.keys())}")
                    m_margin = comp.get("maintenanceMargin") or comp.get("currentMarginBalance") or comp.get("houseMarginRequirement")
                    if m_margin is not None:
                        print(f"Maintenance Margin: ${float(m_margin):,.2f}")
                return balance_data
            else:
                # Handle errors
                logger.debug("Response Body: %s", response.text)
                if response is not None and response.headers['Content-Type'] == 'application/json' \
                        and "Error" in response.json() and "message" in response.json()["Error"] \
                        and response.json()["Error"]["message"] is not None:
                    print("Error: " + response.json()["Error"]["message"])
                else:
                    print("Error: Balance API service error")
        else:
            # Handle errors
            logger.debug("Response Body: %s", response.text)
            if response is not None and response.headers['Content-Type'] == 'application/json' \
                    and "Error" in response.json() and "message" in response.json()["Error"] \
                    and response.json()["Error"]["message"] is not None:
                print("Error: " + response.json()["Error"]["message"])
            else:
                print("Error: Balance API service error")

    def account_menu(self):
        """
        Provides read-only balance and portfolio options for the sample application.

        :param self: Pass in authenticated session and information on selected account
        """

        if self.account["institutionType"] == "BROKERAGE":
            menu_items = {"1": "Balance",
                          "2": "Portfolio",
                          "3": "Go Back"}

            while True:
                print("")
                options = menu_items.keys()
                for entry in options:
                    print(entry + ")\t" + menu_items[entry])

                selection = input("Please select an option: ")
                if selection == "1":
                    self.balance()
                elif selection == "2":
                    self.portfolio()
                elif selection == "3":
                    break
                else:
                    print("Unknown Option Selected!")
        elif self.account["institutionType"] == "BANK":
            menu_items = {"1": "Balance",
                          "2": "Go Back"}

            while True:
                print("\n")
                options = menu_items.keys()
                for entry in options:
                    print(entry + ")\t" + menu_items[entry])

                selection = input("Please select an option: ")
                if selection == "1":
                    self.balance()
                elif selection == "2":
                    break
                else:
                    print("Unknown Option Selected!")
        else:
            menu_items = {"1": "Go Back"}

            while True:
                print("")
                options = menu_items.keys()
                for entry in options:
                    print(entry + ")\t" + menu_items[entry])

                selection = input("Please select an option: ")
                if selection == "1":
                    break
                else:
                    print("Unknown Option Selected!")

    def generate_option_order(self, single_leg_stock_position: StockPosition, action: str, custom_order_id=0, spread_sell_option: StockPosition = None, spread_buy_option: StockPosition = None,
        priceType = {'priceType': 'MARKET','limitPrice': 0.1}):
        """
        Generate orders for E*TRADE based on the provided action and stock positions.

        Parameters:
        - single_leg_stock_position (StockPosition): The main stock position for the order.
        - action (str): The action to perform (e.g., "BUY_CLOSE", "SELL_OPEN", "SPREAD").
        - custom_order_id (int): Optional custom order ID.
        - spread_sell_option (StockPosition): The option to sell (required for "SPREAD" if spread_buy_option is None).
        - spread_buy_option (StockPosition): The option to buy (required for "SPREAD" if spread_sell_option is None).
        - priceType (dict): Price type information containing 'priceType' and 'limitPrice'.

        Returns:
        - list: An array of dictionaries representing the order details for E*TRADE.
        """
        client_id = custom_order_id if custom_order_id else random.randint(1000000000, 9999999999)
        orders = []

        if action in ["BUY_CLOSE", "BUY_OPEN"]:
            sell_or_buy = "buy"
        elif action in ["SELL_CLOSE", "SELL_OPEN"]:
            sell_or_buy = "sell"

        if single_leg_stock_position is not None:
            overlap_qty_single_leg = self.check_conflict_position(single_leg_stock_position, sell_or_buy, single_leg_stock_position.quantity)
        
        overlap_qty_spread_sell = 0
        if spread_sell_option is not None:
            overlap_qty_spread_sell = self.check_conflict_position(spread_sell_option, "sell", spread_sell_option.quantity)
        
        overlap_qty_spread_buy = 0
        if spread_buy_option is not None:   
            overlap_qty_spread_buy = self.check_conflict_position(spread_buy_option, "buy", spread_buy_option.quantity)

        if action in ["BUY_CLOSE", "SELL_OPEN", "SELL_CLOSE", "BUY_OPEN"]:
            # Single-leg option order
            if action == "BUY_CLOSE":
                strike_price = single_leg_stock_position.strike_price
                expiry_date_obj = single_leg_stock_position.expiration_date
                
                # For BUY_CLOSE, handle case where the position is already negative (short)
                if isinstance(overlap_qty_single_leg, int) and overlap_qty_single_leg < 0:
                    # We can directly close the short position
                    order_data = {
                        'client_order_id': client_id,
                        'symbol': single_leg_stock_position.symbol,
                        'quantity': min(abs(overlap_qty_single_leg), abs(single_leg_stock_position.quantity)),
                        'securityType': "OPTN",
                        'orderType': priceType['priceType'],
                        'priceType': priceType['priceType'],
                        'orderTerm': 'GOOD_FOR_DAY',
                        'limitPrice': priceType['limitPrice'],
                        'orderAction': "BUY_CLOSE",
                        'callPut': single_leg_stock_position.call_put,
                        'expiryYear': expiry_date_obj.year,
                        'expiryMonth': expiry_date_obj.month,
                        'expiryDay': expiry_date_obj.day,
                        'strikePrice': strike_price,
                        'required_margin': 0  # Buying options doesn't require margin
                    }
                    orders.append(order_data)
                else:
                    # Normal BUY_CLOSE order
                    order_data = {
                        'client_order_id': client_id,
                        'symbol': single_leg_stock_position.symbol,
                        'quantity': single_leg_stock_position.quantity,
                        'securityType': "OPTN",
                        'orderType': priceType['priceType'],
                        'priceType': priceType['priceType'],
                        'orderTerm': 'GOOD_FOR_DAY',
                        'limitPrice': priceType['limitPrice'],
                        'orderAction': "BUY_CLOSE",
                        'callPut': single_leg_stock_position.call_put,
                        'expiryYear': expiry_date_obj.year,
                        'expiryMonth': expiry_date_obj.month,
                        'expiryDay': expiry_date_obj.day,
                        'strikePrice': strike_price,
                        'required_margin': 0  # Buying options doesn't require margin
                    }
                    orders.append(order_data)
                    
            elif action in ["SELL_OPEN","SELL_CLOSE"]:
                strike_price = single_leg_stock_position.strike_price
                expiry_date_obj = single_leg_stock_position.expiration_date
                
                # For SELL_OPEN, check if there's an existing long position to close first
                if isinstance(overlap_qty_single_leg, int) and overlap_qty_single_leg > 0:
                    # First create a SELL_CLOSE order for the existing long position
                    close_order = {
                        'client_order_id': client_id,
                        'symbol': single_leg_stock_position.symbol,
                        'quantity': min(overlap_qty_single_leg, single_leg_stock_position.quantity),
                        'securityType': "OPTN",
                        'orderType': priceType['priceType'],
                        'priceType': priceType['priceType'],
                        'orderTerm': 'GOOD_FOR_DAY',
                        'limitPrice': priceType['limitPrice'],
                        'orderAction': "SELL_CLOSE",
                        'callPut': single_leg_stock_position.call_put,
                        'expiryYear': expiry_date_obj.year,
                        'expiryMonth': expiry_date_obj.month,
                        'expiryDay': expiry_date_obj.day,
                        'strikePrice': strike_price,
                        'required_margin': 0  # Selling existing long position doesn't require margin
                    }
                    orders.append(close_order)
                    
                    # Generate new client_id for the next order
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # If there's remaining quantity to sell, create a SELL_OPEN order
                    remaining_qty = abs(single_leg_stock_position.quantity) - abs(overlap_qty_single_leg)
                    if remaining_qty > 0:
                        # Calculate margin for selling options
                        if single_leg_stock_position.call_put == "CALL":
                            # For naked calls: Margin = Strike Price × 100 × Quantity
                            required_margin = strike_price * 100 * remaining_qty
                        else:  # PUT
                            # Try to get underlying price and premium
                            underlying_price = getattr(single_leg_stock_position, 'underlying_last_price', 0)
                            if not underlying_price:
                                # If not available, try to get it
                                underlying_price = self.get_stock_price(single_leg_stock_position.symbol)
                                
                            premium = single_leg_stock_position.last_price
                            
                            # For naked puts: Margin = max([20% × Stock Price − (Strike − Stock Price)] + Premium, 10% × Strike + Premium) × 100
                            calc1 = (0.2 * underlying_price - (strike_price - underlying_price) + premium)
                            calc2 = (0.1 * strike_price + premium)
                            required_margin = max(calc1, calc2) * 100 * remaining_qty
                        
                        open_order = {
                            'client_order_id': client_id,
                            'symbol': single_leg_stock_position.symbol,
                            'quantity': remaining_qty,
                            'securityType': "OPTN",
                            'orderType': priceType['priceType'],
                            'priceType': priceType['priceType'],
                            'orderTerm': 'GOOD_FOR_DAY',
                            'limitPrice': priceType['limitPrice'],
                            'orderAction': "SELL_OPEN",
                            'callPut': single_leg_stock_position.call_put,
                            'expiryYear': expiry_date_obj.year,
                            'expiryMonth': expiry_date_obj.month,
                            'expiryDay': expiry_date_obj.day,
                            'strikePrice': strike_price,
                            'required_margin': required_margin
                        }
                        orders.append(open_order)
                else:
                    # Calculate margin for selling options
                    if single_leg_stock_position.call_put == "CALL":
                        # For naked calls: Margin = Strike Price × 100 × Quantity
                        required_margin = strike_price * 100 * single_leg_stock_position.quantity
                    else:  # PUT
                        # Try to get underlying price and premium
                        underlying_price = getattr(single_leg_stock_position, 'underlying_last_price', 0)
                        if not underlying_price:
                            # If not available, try to get it
                            underlying_price = self.get_stock_price(single_leg_stock_position.symbol)
                            
                        premium = single_leg_stock_position.last_price
                        
                        # For naked puts: Margin = max([20% × Stock Price − (Strike − Stock Price)] + Premium, 10% × Strike + Premium) × 100
                        calc1 = (0.2 * underlying_price - (strike_price - underlying_price) + premium)
                        calc2 = (0.1 * strike_price + premium)
                        required_margin = max(calc1, calc2) * 100 * single_leg_stock_position.quantity
                    
                    # Regular SELL_OPEN without existing position
                    order_data = {
                        'client_order_id': client_id,
                        'symbol': single_leg_stock_position.symbol,
                        'quantity': single_leg_stock_position.quantity,
                        'securityType': "OPTN",
                        'orderType': priceType['priceType'],
                        'priceType': priceType['priceType'],
                        'orderTerm': 'GOOD_FOR_DAY',
                        'limitPrice': priceType['limitPrice'],
                        'orderAction': "SELL_OPEN",
                        'callPut': single_leg_stock_position.call_put,
                        'expiryYear': expiry_date_obj.year,
                        'expiryMonth': expiry_date_obj.month,
                        'expiryDay': expiry_date_obj.day,
                        'strikePrice': strike_price,
                        'required_margin': required_margin
                    }
                    orders.append(order_data)

        elif action == "SPREAD":
            # Modified to handle case where only one leg is provided
            if not spread_sell_option and not spread_buy_option:
                raise ValueError("At least one of sell or buy options must be provided for SPREAD orders")
            
            # Handle single-leg spread (only sell option)
            if spread_sell_option and not spread_buy_option:
                # Check if we have an existing long position to close first
                if isinstance(overlap_qty_spread_sell, int) and overlap_qty_spread_sell > 0:
                    # Create a SELL_CLOSE order for the existing long position
                    closing_qty = min(overlap_qty_spread_sell, spread_sell_option.quantity)
                    
                    # Single leg - SELL_CLOSE order
                    close_legs = [
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_CLOSE",
                            'quantity': closing_qty,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        }
                    ]
                    
                    # No margin required for closing positions
                    required_margin = 0
                    
                    close_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "SINGLE",
                        'orderType': 'SPREADS',
                        'priceType': "LIMIT",
                        'limitPrice': priceType['limitPrice'],
                        'legs': close_legs,
                        'required_margin': required_margin
                    }
                    orders.append(close_order)
                    
                    # Generate new client_id for the next order if needed
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # If there's remaining quantity to sell, create a SELL_OPEN order
                    remaining_qty = spread_sell_option.quantity - closing_qty
                    if remaining_qty > 0:
                        # Calculate margin for selling options
                        if spread_sell_option.call_put == "CALL":
                            # For naked calls: Margin = Strike Price × 100 × Quantity
                            required_margin = spread_sell_option.strike_price * 100 * remaining_qty
                        else:  # PUT
                            # Try to get underlying price and premium
                            underlying_price = getattr(spread_sell_option, 'underlying_last_price', 0)
                            if not underlying_price:
                                # If not available, try to get it
                                underlying_price = self.get_stock_price(spread_sell_option.symbol)
                                
                            premium = spread_sell_option.last_price
                            
                            # For naked puts: Margin = max([20% × Stock Price − (Strike − Stock Price)] + Premium, 10% × Strike + Premium) × 100
                            calc1 = (0.2 * underlying_price - (spread_sell_option.strike_price - underlying_price) + premium)
                            calc2 = (0.1 * spread_sell_option.strike_price + premium)
                            required_margin = max(calc1, calc2) * 100 * remaining_qty
                        
                        # Single leg - SELL_OPEN order
                        open_legs = [
                            {
                                'symbol': spread_sell_option.symbol,
                                'orderAction': "SELL_OPEN",
                                'quantity': remaining_qty,
                                'callPut': spread_sell_option.call_put,
                                'expiryYear': spread_sell_option.expiration_date.year,
                                'expiryMonth': spread_sell_option.expiration_date.month,
                                'expiryDay': spread_sell_option.expiration_date.day,
                                'strikePrice': spread_sell_option.strike_price,
                            }
                        ]
                        
                        open_order = {
                            'client_order_id': client_id,
                            'securityType': "OPTN",
                            'orderTerm': 'GOOD_FOR_DAY',
                            'orderAction': "SPREAD",
                            'spreadType': "SINGLE",
                            'orderType': 'SPREADS',
                            'priceType': "LIMIT",
                            'limitPrice': priceType['limitPrice'],
                            'legs': open_legs,
                            'required_margin': required_margin
                        }
                        orders.append(open_order)
                else:
                    # Calculate margin for selling options
                    if spread_sell_option.call_put == "CALL":
                        # For naked calls: Margin = Strike Price × 100 × Quantity
                        required_margin = spread_sell_option.strike_price * 100 * spread_sell_option.quantity
                    else:  # PUT
                        # Try to get underlying price and premium
                        underlying_price = getattr(spread_sell_option, 'underlying_last_price', 0)
                        if not underlying_price:
                            # If not available, try to get it
                            underlying_price = self.get_stock_price(spread_sell_option.symbol)
                            
                        premium = spread_sell_option.last_price
                        
                        # For naked puts: Margin = max([20% × Stock Price − (Strike − Stock Price)] + Premium, 10% × Strike + Premium) × 100
                        calc1 = (0.2 * underlying_price - (spread_sell_option.strike_price - underlying_price) + premium)
                        calc2 = (0.1 * spread_sell_option.strike_price + premium)
                        required_margin = max(calc1, calc2) * 100 * spread_sell_option.quantity
                    
                    # Direct SELL_OPEN order without existing position
                    open_legs = [
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_OPEN",
                            'quantity': spread_sell_option.quantity,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        }
                    ]
                    
                    order_data = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "SINGLE",
                        'orderType': 'SPREADS',
                        'priceType': "LIMIT",
                        'limitPrice': priceType['limitPrice'],
                        'legs': open_legs,
                        'required_margin': required_margin
                    }
                    orders.append(order_data)
            
            # Handle single-leg spread (only buy option)
            elif spread_buy_option and not spread_sell_option:
                # Check if we have an existing short position to close first
                if isinstance(overlap_qty_spread_buy, int) and overlap_qty_spread_buy < 0:
                    # Create a BUY_CLOSE order for the existing short position
                    closing_qty = abs(min(0, overlap_qty_spread_buy))
                    closing_qty = min(closing_qty, spread_buy_option.quantity)
                    
                    # Single leg - BUY_CLOSE order
                    close_legs = [
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_CLOSE",
                            'quantity': closing_qty,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        }
                    ]
                    
                    # No margin required for closing positions
                    required_margin = 0
                    
                    close_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "SINGLE",
                        'orderType': 'SPREADS',
                        'priceType': "LIMIT",
                        'limitPrice': priceType['limitPrice'],
                        'legs': close_legs,
                        'required_margin': required_margin
                    }
                    orders.append(close_order)
                    
                    # Generate new client_id for the next order if needed
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # If there's remaining quantity to buy, create a BUY_OPEN order
                    remaining_qty = spread_buy_option.quantity - closing_qty
                    if remaining_qty > 0:
                        # No margin required for buying options
                        required_margin = 0
                        
                        # Single leg - BUY_OPEN order
                        open_legs = [
                            {
                                'symbol': spread_buy_option.symbol,
                                'orderAction': "BUY_OPEN",
                                'quantity': remaining_qty,
                                'callPut': spread_buy_option.call_put,
                                'expiryYear': spread_buy_option.expiration_date.year,
                                'expiryMonth': spread_buy_option.expiration_date.month,
                                'expiryDay': spread_buy_option.expiration_date.day,
                                'strikePrice': spread_buy_option.strike_price,
                            }
                        ]
                        
                        open_order = {
                            'client_order_id': client_id,
                            'securityType': "OPTN",
                            'orderTerm': 'GOOD_FOR_DAY',
                            'orderAction': "SPREAD",
                            'spreadType': "SINGLE",
                            'orderType': 'SPREADS',
                            'priceType': "LIMIT",
                            'limitPrice': priceType['limitPrice'],
                            'legs': open_legs,
                            'required_margin': required_margin
                        }
                        orders.append(open_order)
                else:
                    # Direct BUY_OPEN order without existing position
                    # No margin required for buying options
                    required_margin = 0
                    
                    open_legs = [
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_OPEN",
                            'quantity': spread_buy_option.quantity,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        }
                    ]
                    
                    order_data = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "SINGLE",
                        'orderType': 'SPREADS',
                        'priceType': "LIMIT",
                        'limitPrice': priceType['limitPrice'],
                        'legs': open_legs,
                        'required_margin': required_margin
                    }
                    orders.append(order_data)
            
            # Handle true spread (both buy and sell legs)
            else:
                # Calculate the number of contracts we can close from existing positions
                closing_sell_qty = min(max(0, overlap_qty_spread_sell), abs(spread_sell_option.quantity)) if overlap_qty_spread_sell > 0 else 0
                closing_buy_qty = min(abs(min(0, overlap_qty_spread_buy)), abs(spread_buy_option.quantity)) if overlap_qty_spread_buy < 0 else 0
                
                # Calculate the number of new contracts we need to open
                opening_sell_qty = abs(spread_sell_option.quantity) - closing_sell_qty if spread_sell_option is not None else 0
                opening_buy_qty = abs(spread_buy_option.quantity) - closing_buy_qty if spread_buy_option is not None else 0
                
                # Determine the maximum number of spread orders we can create
                # For a valid spread, we need equal quantities of buy and sell
                max_spread_qty = min(spread_sell_option.quantity, spread_buy_option.quantity) if spread_sell_option and spread_buy_option else 0
                
                # Create closing spread order if needed
                if closing_sell_qty > 0 and closing_buy_qty > 0:
                    # Use the minimum of the two quantities for the spread
                    close_spread_qty = min(closing_sell_qty, closing_buy_qty)
                    
                    close_legs = [
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_CLOSE",
                            'quantity': close_spread_qty,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        },
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_CLOSE",
                            'quantity': close_spread_qty,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        }
                    ]
                    
                    # Calculate margin for closing spread (usually no margin required for closing)
                    required_margin = 0
                    
                    close_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "VERTICAL",
                        'orderType': 'SPREADS',
                        'priceType': priceType['priceType'],
                        'limitPrice': priceType['limitPrice'],
                        'legs': close_legs,
                        'required_margin': required_margin
                    }
                    orders.append(close_order)
                    
                    # Generate new client_id for the next order
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # Update remaining quantities after closing
                    closing_sell_qty -= close_spread_qty
                    closing_buy_qty -= close_spread_qty
                
                # Handle any remaining closing positions with separate spread orders
                # For single-sided closing positions, we need to create matching opening positions
                
                # Handle remaining sell close positions
                if closing_sell_qty > 0:
                    # Need to create a matching buy open to make a spread
                    # Cap opening quantity at the target quantity from parameters
                    opening_qty = min(closing_sell_qty, spread_buy_option.quantity)
                    
                    close_sell_legs = [
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_CLOSE",
                            'quantity': closing_sell_qty,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        },
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_OPEN",
                            'quantity': opening_qty,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        }
                    ]
                    
                    # Calculate margin for spread (difference in strikes times quantity times 100)
                    strike_diff = abs(spread_sell_option.strike_price - spread_buy_option.strike_price)
                    required_margin = strike_diff * 100 * opening_qty
                    
                    sell_close_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "VERTICAL",
                        'orderType': 'SPREADS',
                        'priceType': priceType['priceType'],
                        'limitPrice': priceType['limitPrice'],
                        'legs': close_sell_legs,
                        'required_margin': required_margin
                    }
                    orders.append(sell_close_order)
                    
                    # Generate new client_id for the next order
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # Adjust opening buy quantity since we've already opened some
                    opening_buy_qty += closing_sell_qty
                
                # Handle remaining buy close positions
                if closing_buy_qty > 0:
                    # Need to create a matching sell open to make a spread
                    # Cap opening quantity at the target quantity from parameters
                    opening_qty = min(closing_buy_qty, spread_sell_option.quantity)
                    
                    close_buy_legs = [
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_CLOSE",
                            'quantity': closing_buy_qty,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        },
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_OPEN",
                            'quantity': opening_qty,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        }
                    ]
                    
                    # Calculate margin for spread (difference in strikes times quantity times 100)
                    strike_diff = abs(spread_sell_option.strike_price - spread_buy_option.strike_price)
                    required_margin = strike_diff * 100 * opening_qty
                    
                    buy_close_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': "SPREAD",
                        'spreadType': "VERTICAL",
                        'orderType': 'SPREADS',
                        'priceType': priceType['priceType'],
                        'limitPrice': priceType['limitPrice'],
                        'legs': close_buy_legs,
                        'required_margin': required_margin
                    }
                    orders.append(buy_close_order)
                    
                    # Generate new client_id for the next order
                    client_id = random.randint(1000000000, 9999999999)
                    
                    # Adjust opening sell quantity since we've already opened some
                    opening_sell_qty += closing_buy_qty
                
                # Finally create open spread order for any remaining quantities
                # Use the minimum of the remaining quantities to ensure equal buy/sell
                open_spread_qty = min(opening_sell_qty, opening_buy_qty)
                
                if open_spread_qty > 0:
                    open_legs = [
                        {
                            'symbol': spread_sell_option.symbol,
                            'orderAction': "SELL_OPEN",
                            'quantity': open_spread_qty,
                            'callPut': spread_sell_option.call_put,
                            'expiryYear': spread_sell_option.expiration_date.year,
                            'expiryMonth': spread_sell_option.expiration_date.month,
                            'expiryDay': spread_sell_option.expiration_date.day,
                            'strikePrice': spread_sell_option.strike_price,
                        },
                        {
                            'symbol': spread_buy_option.symbol,
                            'orderAction': "BUY_OPEN",
                            'quantity': open_spread_qty,
                            'callPut': spread_buy_option.call_put,
                            'expiryYear': spread_buy_option.expiration_date.year,
                            'expiryMonth': spread_buy_option.expiration_date.month,
                            'expiryDay': spread_buy_option.expiration_date.day,
                            'strikePrice': spread_buy_option.strike_price,
                        }
                    ]
                    
                    # Calculate margin for spread (difference in strikes times quantity times 100)
                    strike_diff = abs(spread_sell_option.strike_price - spread_buy_option.strike_price)
                    required_margin = strike_diff * 100 * open_spread_qty
                    
                    spreadType = "VERTICAL" if len(open_legs) > 1 else "SINGLE"
                    orderType = "SPREAD"# if len(open_legs) > 1 else "OPTN"
                    priceType_adapt = priceType['priceType'] if len(open_legs) > 1 else "LIMIT"
                    
                    open_order = {
                        'client_order_id': client_id,
                        'securityType': "OPTN",
                        'orderTerm': 'GOOD_FOR_DAY',
                        'orderAction': orderType,
                        'spreadType': spreadType,
                        'orderType': 'SPREADS',
                        'priceType': priceType_adapt,
                        'limitPrice': priceType['limitPrice'],
                        'legs': open_legs,
                        'required_margin': required_margin
                    }
                    orders.append(open_order)
        else:
            raise ValueError(f"Unsupported action: {action}")

        return orders
    
    def debug_function(self, ticker):
        """
        Check the next earnings date for the underlying stock of a single stock option.

        :param stock_option: StockOption object with a `symbol` attribute representing the underlying stock.
        :return: The next earnings date as a string in "YYYY-MM-DD" format or "No earnings date available" if not found.
        """
        # Base URL for the API
        base_url = f"{self.base_url}/v1/market/quote"
        headers = self._consumer_key_headers()

        # Prepare the symbol
        symbol = "NVDA:2024:12:20:PUT:139"

        # Build the request URL
        url = f"{base_url}/{symbol}"
        params = {"requireEarningsDate": "true", "detailFlag": "OPTIONS"}

        # Make API request
        response = self.session.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch earnings date for symbol: {symbol}. Status Code: {response.status_code}")
            return "No earnings date available"

        # Parse XML response
        print(response.text)
        breakpoint()
        root = ET.fromstring(response.text)
        earnings_date = None

        # Navigate the XML structure to extract the earnings date
        for quote_data in root.findall(".//QuoteData"):
            all_data = quote_data.find("All")
            if all_data is not None:
                earnings_date = all_data.find("nextEarningDate").text
                break

        if earnings_date:
            # print(f"{symbol}: Next earnings date is {earnings_date}")
            return earnings_date
        else:
            # print(f"{symbol}: No earnings date available.")
            return "No earnings date available"

    def check_earning_date(self, stock_option):
        """
        Check the next earnings date for the underlying stock of a single stock option.

        :param stock_option: StockOption object with a `symbol` attribute representing the underlying stock.
        :return: The next earnings date as a string in "YYYY-MM-DD" format or "No earnings date available" if not found.
        """
        # Base URL for the API
        base_url = f"{self.base_url}/v1/market/quote"
        headers = self._consumer_key_headers()

        # Prepare the symbol
        symbol = stock_option.symbol

        # Build the request URL
        url = f"{base_url}/{symbol}"
        params = {"requireEarningsDate": "true", "detailFlag": "ALL"}

        # Make API request
        response = self.session.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch earnings date for symbol: {symbol}. Status Code: {response.status_code}")
            return "No earnings date available"

        root = ET.fromstring(response.text)
        earnings_date = None

        # Navigate the XML structure to extract the earnings date
        for quote_data in root.findall(".//QuoteData"):
            all_data = quote_data.find("All")
            if all_data is not None:
                earnings_date = all_data.find("nextEarningDate").text
                break

        if earnings_date:
            # print(f"{symbol}: Next earnings date is {earnings_date}")
            return earnings_date
        else:
            # print(f"{symbol}: No earnings date available.")
            return "No earnings date available"

    def get_protect_spread(self, symbol: str, quantity: int, days_to_expiration: int, target_delta: float, long_call_gap: float):
        """
        Create a protective spread by finding options based on the target delta and pricing criteria.

        Parameters:
        - symbol (str): The ticker symbol of the stock.
        - days_to_expiration (int): Desired number of days until the option's expiration.
        - target_delta (float): Target delta for the sell call option.

        Returns:
        A dictionary containing StockPosition objects for sell_call_option, buy_call_option_low, and buy_call_option_high.
        """
        # Step 1: Retrieve the stock's current price
        stock_price = self.get_stock_price(symbol)
        if stock_price is None:
            print(f"Failed to retrieve stock price for {symbol}.")
            return

        # Step 2: Get available expirations
        available_expirations = self.get_available_expirations(symbol)
        if not available_expirations:
            print(f"No available expirations for {symbol}.")
            return

        # Find the expiration closest to the target days_to_expiration
        today = datetime.today().date()
        target_expiration = min(
            available_expirations,
            key=lambda exp: abs((datetime.strptime(exp, "%Y-%m-%d").date() - today).days - days_to_expiration),
        )
        expiration_date = datetime.strptime(target_expiration, "%Y-%m-%d").date()

        # Step 3: Retrieve the option chain for the selected expiration
        url = f"{self.base_url}/v1/market/optionchains.json"
        if symbol == "BRKB":
            symbol_converted = "BRK.B"
        else:
            symbol_converted = symbol
            
        params = {
            "symbol": symbol_converted,
            "expiryYear": expiration_date.year,
            "expiryMonth": expiration_date.month,
            "expiryDay": expiration_date.day,
            "includeWeekly": True,
            "skipAdjusted": True,
            "optionCategory": "STANDARD",
            "strikePriceNear": stock_price,
            "noOfStrikes": 10,
        }
        response = self.session.get(url, params=params, auth=self.session.auth)

        if response.status_code != 200:
            print(f"Error fetching option chain for {symbol_converted}: {response.status_code}, {response.text}")
            return

        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

        # Step 4: Find the sell call option (closest to target delta)
        sell_call_option = None
        buy_call_option_low = None
        buy_call_option_high = None
        sell_delta_diff = float("inf")

        for option_pair in option_pairs:
            call_option = option_pair.get("Call")
            if not call_option:
                continue

            delta = call_option.get("OptionGreeks", {}).get("delta")
            if delta is None:
                continue

            # Find the call option closest to the target delta
            delta_diff = abs(delta - target_delta)
            if delta_diff < sell_delta_diff:
                sell_delta_diff = delta_diff
                sell_call_option = call_option

        if not sell_call_option:
            print(f"No call option found near target delta for {symbol}.")
            return

        # Step 5: Find buy call options
        sell_option_price = sell_call_option.get("bid", 0)
        sell_strike_price = sell_call_option.get("strikePrice", 0)

        for option_pair in option_pairs:
            call_option = option_pair.get("Call")
            if not call_option:
                continue

            strike_price = call_option.get("strikePrice", 0)
            option_price = call_option.get("bid", 0)

            # Find buy_call_option_low: higher strike and approximately half the sell option price
            if strike_price > sell_strike_price and abs(option_price - sell_option_price / 2) < abs(
                (buy_call_option_low or {}).get("bid", 0) - sell_option_price / 2
            ):
                buy_call_option_low = call_option
        print("buy_call_option_low: ", buy_call_option_low.get("strikePrice", 0))

        if buy_call_option_low:
            while buy_call_option_high is None:
                for option_pair in option_pairs:
                    call_option = option_pair.get("Call")
                    if not call_option:
                        continue
                    strike_price = call_option.get("strikePrice", 0)
                    option_price = call_option.get("bid", 0)
                    # Find buy_call_option_high: strike price ~$1 higher than buy_call_option_low
                    if (
                        strike_price > buy_call_option_low.get("strikePrice", 0)
                        and abs(strike_price - buy_call_option_low.get("strikePrice", 0)) < long_call_gap
                    ):
                        buy_call_option_high = call_option
                    # else:
                    #     print(f"Searching for buy_call_option_high: {strike_price}")
                long_call_gap += 0.5

        if not buy_call_option_low or not buy_call_option_high:
            print(f"Could not find appropriate buy call options for {symbol}. buy_call_option_low: {buy_call_option_low}, buy_call_option_high: {buy_call_option_high}")
            return
            
        # Step 6: Convert options to StockPosition objects
        sell_position = StockPosition(
            symbol=symbol,
            quantity=-quantity,  # Selling 1 option contract
            last_price=round(0.5*(sell_call_option.get("bid",0)+sell_call_option.get("bid",0)),2)
        )
        sell_position.strike_price = sell_call_option["strikePrice"]
        sell_position.expiration_date = expiration_date
        sell_position.call_put = "CALL"

        buy_low_position = StockPosition(
            symbol=symbol,
            quantity=quantity,  # Buying 1 option contract
            last_price=round(0.5*(buy_call_option_low.get("bid",0)+buy_call_option_low.get("bid",0)),2)
        )
        buy_low_position.strike_price = buy_call_option_low["strikePrice"]
        buy_low_position.expiration_date = expiration_date
        buy_low_position.call_put = "CALL"

        buy_high_position = StockPosition(
            symbol=symbol,
            quantity=quantity,  # Buying 1 option contract
            last_price=round(0.5*(buy_call_option_high.get("bid",0)+buy_call_option_high.get("ask",0)),2)
        )
        buy_high_position.strike_price = buy_call_option_high["strikePrice"]
        buy_high_position.expiration_date = expiration_date
        buy_high_position.call_put = "CALL"

        print(f"sell_call_option: strike: {sell_position.strike_price}, {sell_position.last_price}")
        print(f"buy_call_option: strike: {buy_low_position.strike_price}, {buy_low_position.last_price}")
        print(f"buy_call_option: strike: {buy_high_position.strike_price}, {buy_high_position.last_price}")
        # Step 7: Return the positions
        return {
            "sell_call_option": sell_position,
            "buy_call_option_low": buy_low_position,
            "buy_call_option_high": buy_high_position,
        }

    def get_historical_prices(self, symbol: str, start_date: str, end_date: str):
        """
        Retrieve historical price data for a given symbol using yfinance.
        
        Parameters:
        - symbol (str): The stock ticker or index.
        - start_date (str): The start date in 'YYYY-MM-DD' format.
        - end_date (str): The end date in 'YYYY-MM-DD' format.
        
        Returns:
        - pd.DataFrame: A DataFrame containing date and close price.
        """
        try:
            # Fetch historical data using yfinance
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                print(f"No data found for {symbol}.")
                return None
            
            # Keep only the 'Close' column and reset the index
            data = data[["Close"]].rename(columns={"Close": "close"})
            data.index.name = "date"
            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
                    
    def get_downside_exposure(self, ticker: str, daily_gain_threshold: float):
        """
        Calculate the downside exposure (correlation) of a stock with the VIX index,
        using only days where the stock's daily return is below the specified threshold.

        Parameters:
        - ticker (str): The stock ticker symbol.
        - daily_gain_threshold (float): The threshold for daily returns (e.g., -0.05 for a 5% drop).

        Returns:
        - float: The downside correlation coefficient with VIX, or None if insufficient data.
        """

        # Step 1: Fetch historical prices for the ticker and VIX
        today = datetime.today()
        start_date = (today - timedelta(days=900)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')

        stock_data = self.get_historical_prices(ticker, start_date, end_date)
        vix_data = self.get_historical_prices("^VIX", start_date, end_date)

        if stock_data is None or vix_data is None:
            print("Unable to retrieve necessary data for correlation calculation.")
            return None

        # Step 2: Merge the two datasets on their date indices
        merged_data = pd.merge(stock_data, vix_data, left_index=True, right_index=True, suffixes=(f"_{ticker}", "_VIX"))
        merged_data.columns = ["stock_close", "vix_close"]

        # Step 3: Calculate daily returns
        merged_data["stock_return"] = merged_data["stock_close"].pct_change()
        merged_data["vix_return"] = merged_data["vix_close"].pct_change()

        # Step 4: Filter for days with stock returns below the threshold
        filtered_data = merged_data[merged_data["stock_return"] < daily_gain_threshold].dropna()

        if filtered_data.empty:
            print(f"No data points with stock returns below {daily_gain_threshold:.2%}.")
            return None
        else:
            print(f"Number of days when {ticker} has gain lower than {daily_gain_threshold*100}%: {len(filtered_data)}")

        # Step 5: Calculate correlation between stock returns and VIX returns
        correlation = filtered_data["stock_return"].corr(filtered_data["vix_return"])

        print(f"Downside exposure (correlation) of {ticker} with VIX for daily returns below {daily_gain_threshold:.2%}: {correlation:.2f}")
        return correlation

    def log_daily_summary(self, stock_positions: List[StockPosition]):
        """
        Log daily summary of stock positions and option Greeks to a CSV file.

        - Creates a folder named 'daily log' if it does not exist.
        - Logs stock and option data at the start and end of the trading session.
        - Ensures no overwriting of start-of-session logs.
        - Sorts positions with options listed before stocks.
        - Overwrites session_phase 'end' if it already exists.
        - Adds a timestamp column for when the log is updated.

        :param stock_positions: List of StockPosition objects from the portfolio.
        """
        # Define folder and file path
        log_folder = "daily_log"
        os.makedirs(log_folder, exist_ok=True)  # Create folder if it doesn't exist
        today_date = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(log_folder, f"{today_date}.csv")

        # Determine session phase
        session_phase = "start" if not os.path.exists(file_path) else "end"

        # Current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log header and content
        log_header = [
            "Symbol", "Type", "Quantity", "Last Price",
            "Price Paid", "Total Gain", "Market Value", "Date Acquired",
            "Strike Price", "Asset Price", "Expiration Date", "Days to Expiration",
            "Call/Put", "Theta", "Implied Volatility", "Rho", "Vega", "Delta", "Gamma",
            "Session Phase", "Timestamp"
        ]

        # Sort stock_positions with options listed before stocks
        sorted_positions = sorted(
            stock_positions,
            key=lambda pos: 0 if pos.security_type == "Option" else 1
        )

        log_rows = []

        # Populate log rows
        for position in sorted_positions:
            base_data = [
                position.symbol,
                position.security_type,
                position.quantity,
                position.last_price,
                position.price_paid,
                position.total_gain,
                position.market_value,
                (position.date_acquired.strftime("%Y-%m-%d") if hasattr(position.date_acquired, 'strftime') else position.date_acquired) if position.date_acquired else "N/A",
            ]

            if position.security_type == "Option":
                option_data = [
                    position.strike_price,
                    position.underlying_last_price,
                    (position.expiration_date.strftime("%Y-%m-%d") if hasattr(position.expiration_date, 'strftime') else position.expiration_date) if position.expiration_date else "N/A",
                    position.days_to_expiration,
                    position.call_put,
                    position.theta,
                    position.implied_volatility,
                    position.rho,
                    position.vega,
                    position.delta,
                    position.gamma,
                ]
            else:
                option_data = ["N/A"] * 11

            log_rows.append(base_data + option_data + [session_phase, current_timestamp])

        # If file exists, read its content and replace rows with 'end'
        if os.path.exists(file_path):
            updated_rows = []
            with open(file_path, mode='r', newline='') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)  # Read header
                for row in reader:
                    # Keep all rows except those with 'end'
                    if row[-2] != "end":
                        updated_rows.append(row)

            # Add the new 'end' rows
            updated_rows.extend(log_rows)

            # Write back to the file
            with open(file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log_header)  # Write header
                writer.writerows(updated_rows)  # Write updated rows

        else:
            # If file does not exist, create it and write rows
            with open(file_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log_header)  # Write header
                writer.writerows(log_rows)  # Write rows

        print(f"Daily log updated for {session_phase} of trading session: {file_path}")


    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime, timedelta

    def analyze_option_chain(self, stock_position, time_horizon_days=30):
        """
        Retrieves an option chain for a given symbol, calculates IV skew at each expiration date, 
        and plots IV against strike price while maintaining the distance skew boundary logic.
        """
        try:
            stock_price = self.get_stock_price(stock_position.symbol)  # Assume this method is defined
            if stock_price is None:
                print("Failed to retrieve stock price.")
                return None

            symbol_converted = "BRK.B" if stock_position.symbol == "BRKB" else stock_position.symbol

            available_expirations = self.get_available_expirations(stock_position.symbol)
            if not available_expirations:
                print("No available expiration dates for options.", stock_position.symbol)
                return None

            time_horizon_end = datetime.now() + timedelta(days=time_horizon_days)

            iv_skews = {}  # Store IV skew for each expiration
            all_iv_data = []  # Store IV data for plotting

            for expiration_str in available_expirations:
                try:
                    expiration = datetime.strptime(expiration_str, "%Y-%m-%d").date()
                except ValueError:
                    print(f"Invalid expiration format: {expiration_str}")
                    continue

                if expiration <= time_horizon_end.date():  # Filter by time horizon
                    url = f"{self.base_url}/v1/market/optionchains.json"
                    params = {
                        "symbol": symbol_converted,
                        "expiryYear": expiration.year,
                        "expiryMonth": expiration.month,
                        "expiryDay": expiration.day,
                        "strikePriceNear": stock_price*0.9,
                        "includeWeekly": True,
                        "skipAdjusted": False,
                        "optionCategory": "ALL",
                        "noOfStrikes": 100
                    }

                    response = self.session.get(url, params=params, auth=self.session.auth)

                    if response.status_code != 200:
                        print(f"Error fetching option chain for {symbol_converted} on {expiration}: {response.status_code} {response.text}")
                        continue

                    option_chain = response.json()
                    if not option_chain or not option_chain.get('OptionChainResponse') or not option_chain['OptionChainResponse'].get('OptionPair'):
                        print(f"No option data for {symbol_converted} on {expiration}")
                        continue

                    options_data = []
                    for option_pair in option_chain['OptionChainResponse']['OptionPair']:
                        for option_type in ['Call', 'Put']:
                            option = option_pair.get(option_type)
                            if option:
                                option_greeks = option.get('OptionGreeks', {})
                                if abs(option_greeks.get('iv', 0)) < 100:  # Filter invalid IV
                                    options_data.append({
                                        'optionType': option['optionType'],
                                        'strikePrice': float(option['strikePrice']),
                                        'impliedVolatility': float(option_greeks.get('iv', 0)),
                                        'expiryDate': expiration
                                    })

                    df = pd.DataFrame(options_data)
                    if df.empty:
                        print(f"No options data for expiration {expiration}")
                        continue

                    # Original logic for distance skew
                    distance_skew = 1
                    scale_factor_put = 1
                    scale_factor_call = 1
                    while abs(distance_skew) > 0.05:
                        df_filter = df[(df['strikePrice'] > stock_price * 0.85 * scale_factor_put)]
                        df_filter = df_filter[(df_filter['strikePrice'] < stock_price * 1.15 * scale_factor_call)]
                        if df_filter.empty:
                            print(f"No options found within the specified time horizon for {stock_position.symbol}")
                            return None

                        otm_puts = df_filter[(df_filter['optionType'] == 'PUT') & (df_filter['strikePrice'] < stock_price)]
                        otm_calls = df_filter[(df_filter['optionType'] == 'CALL') & (df_filter['strikePrice'] > stock_price)]
                        avg_put_distance = round(otm_puts['strikePrice'].mean() - stock_price, 3)
                        avg_call_distance = round(stock_price - otm_calls['strikePrice'].mean(), 3)
                        distance_skew = (avg_put_distance - avg_call_distance) / avg_put_distance
                        if distance_skew > 0.05:
                            scale_factor_call *= 1.001
                        elif distance_skew < -0.05:
                            scale_factor_put *= 0.999

                    if otm_puts.empty or otm_calls.empty:
                        print(f"Insufficient OTM options for expiration {expiration}")
                        continue

                    avg_put_iv = otm_puts['impliedVolatility'].mean()
                    avg_call_iv = otm_calls['impliedVolatility'].mean()
                    iv_skew = round(avg_put_iv / avg_call_iv, 2)
                    iv_skews[expiration] = iv_skew

                    # Collect data for plotting
                    all_iv_data.append(df_filter)


                # Ensure the folder exists
                output_folder = "iv_plot"
                os.makedirs(output_folder, exist_ok=True)

                # Filter out expirations that expire today
                filtered_iv_data = [
                    df for df in all_iv_data if df['expiryDate'].iloc[0] > datetime.now().date()
                ]

                # Plot IV vs. Strike Price for all expirations (excluding today's)
                plt.figure(figsize=(12, 8))
                for df in filtered_iv_data:
                    expiration = df['expiryDate'].iloc[0]
                    plt.plot(df['strikePrice'], df['impliedVolatility'], label=f"Exp {expiration}")

                # Add a vertical line at stock_price
                plt.axvline(x=stock_price, color='red', linestyle='--', label=f"Stock Price: {stock_price:.2f}")

                # Add labels, legend, and grid
                plt.title(f"Implied Volatility vs. Strike Price for {stock_position.symbol}")
                plt.xlabel("Strike Price")
                plt.ylabel("Implied Volatility (IV)")
                plt.legend()
                plt.grid()

                # Save the plot to the specified folder
                plot_path = os.path.join(output_folder, f"{stock_position.symbol}_iv_plot_all_expirations.png")
                plt.savefig(plot_path)
                plt.close()  # Close the figure to avoid memory issues
            # Return IV skew data and summary
            summary = f"Calculated IV skew for {len(iv_skews)} expirations within {time_horizon_days} days."
            return {
                "iv_skews": iv_skews,
                "average_iv_skew": round(np.mean(list(iv_skews.values())),2),
                "summary": summary
            }

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_option_spread(self, ticker, call_put, days_to_expire, multiplier=1, hedge=10, show_options=True):

        stock_price = self.get_stock_price(ticker)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None

        today = datetime.today().date()
        # Calculate the next Friday
        days_to_friday = (4 - today.weekday()) % 7 + round(days_to_expire/7,0)
        target_expire = pd.Timestamp(today + pd.Timedelta(days=days_to_friday))

        # Step 1: Get available expiration dates and convert them to datetime.date objects
        # available_expirations = self.get_available_expirations(convert_ticker_name(yfinance_ticker=stock_position.symbol))
        available_expirations = self.get_available_expirations(ticker)
        if not available_expirations:
            print("No available expiration dates for options.",ticker)
            return None
        available_expirations = [datetime.strptime(exp, "%Y-%m-%d").date() if isinstance(exp, str) else exp for exp in available_expirations]
        filtered_expirations = [exp for exp in available_expirations if exp >= target_expire.date()]

        target_expiration = filtered_expirations[0]
        days_to_expire = (target_expiration-today).days

        stock_position_temp = StockPosition(
            symbol=ticker,
            quantity=1,
            last_price=stock_price
        )

        stock_position_temp.underlying_last_price = stock_price
        stock_position_temp.call_put = call_put
        stock_position_temp.volatility = calculate_std_dev(convert_ticker_name(etrade_ticker=stock_position_temp.symbol),VOLATILITY_WINDOW)
        next_earning_date = self.check_earning_date(stock_position_temp)    
        stock_position_temp.next_earning_date = next_earning_date

        # Calculate the distance in days to the next earnings date
        if next_earning_date and next_earning_date != "No earnings date available":
            try:
                # Try parsing with multiple date formats
                formats = ["%Y-%m-%d", "%m/%d/%Y"]
                earnings_date = None
                for fmt in formats:
                    try:
                        earnings_date = datetime.strptime(next_earning_date, fmt).date()
                        break  # Stop if a format works
                    except ValueError:
                        continue  # Try the next format

                if earnings_date is None:
                    raise ValueError(f"Date format for {next_earning_date} not recognized.")

                # Calculate days to the earnings date
                today = date.today()
                days_to_earnings = (earnings_date - today).days
                stock_position_temp.days_to_earnings = days_to_earnings
                print(f"{stock_position_temp.symbol}: Next earnings date is in {days_to_earnings} days.")
            except ValueError as e:
                # Handle invalid date format
                stock_position_temp.days_to_earnings = None
                print(f"Error parsing next earnings date for {stock_position_temp.symbol}: {e}")
        else:
            stock_position_temp.days_to_earnings = None
        
        if stock_position_temp.days_to_earnings is not None:
            if stock_position_temp.days_to_earnings < days_to_expire:
                print("{stock_position_temp.days_to_earnings} days to earning and {days_to_expire} days to target expiration, SKIP TRADING.....")

        lowest_theta = 10
        min_theta_option = None
        MAX_RETRIES = 10  # Maximum retries for abnormal data
        THETA_THRESHOLD = 100
        IMPLIED_VOLATILITY_THRESHOLD = 100
        abnormal_detected = False

        url = f"{self.base_url}/v1/market/optionchains.json"
        if stock_position_temp.symbol == "BRKB":
            symbol_converted = "BRK.B"
        else:
            symbol_converted = stock_position_temp.symbol

        price_target,dummy_hedge = self.generate_target_strike(stock_position_temp,days_to_expire=days_to_expire,multiplier=multiplier,earning_alert=False)

        # print(f"Generated strike: {stock_position.symbol}: {quote_target_price},hedge target: {hedge_target}")
        # breakpoint()
        # else:
        #     quote_target_price = target_option_price_put
        params = {
            "symbol": symbol_converted,
            "expiryYear": target_expiration.year,
            "expiryMonth": target_expiration.month,
            "expiryDay": target_expiration.day,
            "includeWeekly": True,
            "skipAdjusted": False,
            "optionCategory": "ALL",
            "strikePriceNear": price_target,
            "chainType": call_put,
            "noOfStrikes": 20
        }

        response = self.session.get(url, params=params, auth=self.session.auth)

        if response.status_code != 200:
            print("Error fetching option chain:", symbol_converted, response.status_code, response.text)
            return None

        def get_option_by_strike(response,price_target,call_put):
            print("getting price target: ", price_target)
            data = response.json()
            option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

            closest_option = None
            closest_diff = 100
            options = None

            for option_pair in option_pairs:
                if call_put == "CALL":
                    options = option_pair.get("Call")
                if call_put == "PUT":
                    options = option_pair.get("Put")

                # Process call option
                # if call_option:
                # Retrieve theta and ensure it's interpreted as a float
                theta = float(options.get("OptionGreeks", {}).get("theta", 100))
                implied_volatility = float(options.get("OptionGreeks", {}).get("iv", 100))
                rho = float(options.get("OptionGreeks", {}).get("rho", 100))
                vega = float(options.get("OptionGreeks", {}).get("vega", 100))
                delta = float(options.get("OptionGreeks", {}).get("delta", 100))
                gamma = float(options.get("OptionGreeks", {}).get("gamma", 100))

                osi_key = options.get("osiKey")
                if osi_key:
                    expiry_date_str = osi_key[6:12]
                    expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()

                    option_price = round((options.get("bid") + options.get("ask"))/2,3)
                    strike_price = options.get("strikePrice")
                    option_id = options.get("optionId")
                    
                    if option_price is None:
                        continue
                    
                    diff = abs(strike_price - price_target)
                    # print(f"diff: {diff}, theta: {theta}, current theta: {stock_position.theta}")
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_option = {
                            "optionId": option_id,
                            "type": "CALL",
                            "strikePrice": strike_price,
                            "optionPrice": option_price,
                            "expiryDate": expiry_date,
                            "return_rate": None,
                            "cost_for_roll": None,
                            "implied_volatility": round(implied_volatility/(252**0.5)*100,2),
                            "theta": round(theta,4)
                        }
            return closest_option
        
        option_short = get_option_by_strike(response,price_target,call_put)
        
        if call_put == "CALL":
            option_long = get_option_by_strike(response,option_short['strikePrice']+hedge,call_put)
        elif call_put == "PUT":
            option_long = get_option_by_strike(response,option_short['strikePrice']-hedge,call_put)
    
        print(f"Sell {call_put} option for {stock_position_temp.symbol} strike {option_short['strikePrice']} price {option_short['optionPrice']} expire {option_short['expiryDate']}")
        print(f"Buy {call_put} option for {stock_position_temp.symbol} strike {option_long['strikePrice']} price {option_long['optionPrice']} expire {option_long['expiryDate']}")
        print(f"Profit: {round(option_short['optionPrice']-option_long['optionPrice'],4)} ")

        # Convert options to StockPosition objects
        sell_position = StockPosition(
            symbol=stock_position_temp.symbol,
            quantity=-1,  # Selling 1 option contract
            last_price=option_short['optionPrice']
        )
        sell_position.strike_price = option_short["strikePrice"]
        sell_position.expiration_date = option_short['expiryDate']
        sell_position.call_put = call_put

        buy_position = StockPosition(
            symbol=stock_position_temp.symbol,
            quantity=1,  # buying 1 option contract
            last_price=option_long['optionPrice']
        )
        buy_position.strike_price = option_long["strikePrice"]
        buy_position.expiration_date = option_long['expiryDate']
        buy_position.call_put = call_put
        
        return {
            "sell_option": sell_position,
            "buy_option": buy_position,
        }

    def check_conflict_position(self, stock_position, sell_or_buy, quantity):
        """
        Check the portfolio for an option position matching the input stock_position and return its quantity.
        
        Args:
            stock_position (StockPosition): A StockPosition object representing an option with attributes
                                            symbol, call_put, strike_price, and expiration_date.
            quantity (int): An integer parameter (purpose unspecified, unused in this implementation).
        
        Returns:
            int: The quantity of the matching option position in the portfolio, or 0 if no match is found.
        """
        # Retrieve the list of positions from the portfolio
        portfolio_positions = self.portfolio(minimal=True)
        
        # Search for a matching option position
        for position in portfolio_positions:
            if (position.security_type == "Option" and
                position.symbol == stock_position.symbol and
                position.call_put.upper() == stock_position.call_put.upper() and
                position.strike_price == stock_position.strike_price and
                position.expiration_date and
                (position.expiration_date.strftime("%Y-%m-%d") if hasattr(position.expiration_date, 'strftime') else position.expiration_date) == 
                (stock_position.expiration_date.strftime("%Y-%m-%d") if hasattr(stock_position.expiration_date, 'strftime') else stock_position.expiration_date)):
                if ( sell_or_buy == "sell" and position.quantity > 0 ) or ( sell_or_buy == "buy" and position.quantity < 0 ): 
                    print(f"Checked for conflict position: {stock_position.symbol}, {stock_position.call_put}, {stock_position.strike_price}, {stock_position.expiration_date}, {position.quantity}")
                    return position.quantity
        
        # Return 0 if no matching position is found
        return 0

    def get_option_spread_by_price(self, ticker, call_put, days_to_expire, target_premium, hedge_ratio, hedge_spread = None, qty=1, target_delta = None, target_expiration = None):
        stock_price = self.get_stock_price(ticker)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None
        if hedge_ratio == 0:
            print("Hedge ratio is 0, no spread to create.")
            return None
        if target_premium == 0 and target_delta is None:
            print("Target premium is 0 and no target delta provided, no spread to create.")
            return None
        
        today = datetime.today().date()
        if target_expiration is not None:
            if isinstance(target_expiration, str):
                target_expiration = datetime.strptime(target_expiration, "%Y-%m-%d").date()
        else:
            target_expire = pd.Timestamp(today + pd.Timedelta(days=days_to_expire))

            available_expirations = self.get_available_expirations(ticker)
            if not available_expirations:
                print("No available expiration dates for options.", ticker)
                return None
            
            target_expiration = _select_nearest_expiration(available_expirations, target_expire.date())
            if target_expiration is None:
                print("No available future expiration dates for options.", ticker)
                return None

        url = f"{self.base_url}/v1/market/optionchains.json"

        params = {
            "symbol": ticker,
            "expiryYear": target_expiration.year,
            "expiryMonth": target_expiration.month,
            "expiryDay": target_expiration.day,
            "includeWeekly": True,
            "skipAdjusted": True,
            "optionCategory": "STANDARD",
            "chainType": call_put,
            "noOfStrikes": 400
        }

        response = self.session.get(url, params=params, auth=self.session.auth)

        if response.status_code != 200:
            print("Error fetching option chain:", ticker, response.status_code, response.text)
            return None

        data = response.json()

        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

        # Ensure consistency in option contract types for SPX (SPX vs SPXW)
        has_spxw = False
        if ticker == "SPX" and option_pairs:
            has_spxw = any(
                pair.get(call_put, {}).get("osiKey", "").startswith("SPXW")
                for pair in option_pairs if pair.get(call_put)
            )
            if has_spxw:
                option_pairs = [
                    pair for pair in option_pairs
                    if pair.get(call_put) and pair.get(call_put, {}).get("osiKey", "").startswith("SPXW")
                ]
            else:
                option_pairs = [
                    pair for pair in option_pairs
                    if pair.get(call_put) and pair.get(call_put, {}).get("osiKey", "").startswith("SPX-")
                ]
                
        def get_option_by_price(target_price, call_put, buy_sell, strike_overwrite = None, target_delta = None):
            # Sort pairs first
            if call_put == "Put":
                # Descending strikes
                sorted_option_pairs = sorted(
                    option_pairs, key=lambda x: x.get(call_put, {}).get("strikePrice", 0), reverse=True
                )
            if call_put == 'Call':
                # ascending strikes
                sorted_option_pairs = sorted(
                    option_pairs, key=lambda x: x.get(call_put, {}).get("strikePrice", float("inf"))
                )

            min_price_diff = float("inf")
            candidate = None
            previous_strike = None

            for option_pair in sorted_option_pairs:
                options = option_pair.get(call_put)
                if not options:
                    continue
        
                # Compute mid-price for each option
                # if options.get("bid",0) == 0 or options.get("ask",0) == 0:
                #     option_price = 0
                # else:
                option_price = (options.get("bid", 0) + options.get("ask", 0)) / 2
                strike = options.get("strikePrice")
                volume = options.get("volume", 0) or 0
                open_interest = options.get("openInterest", 0) or 0
                ask_bid_spread = round(options.get("ask", 0) - options.get("bid", 0), 4)
                delta = options.get("OptionGreeks", {}).get("delta", 0)
                gamma = options.get("OptionGreeks", {}).get("gamma", 0)
                osi_key = options.get("osiKey")
                contract_symbol = _symbol_from_osi_key(ticker, osi_key)

                if osi_key:
                    expiry_date_str = osi_key[6:12]
                    expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()

                # Convert E*TRADE standard delta to N(d2) risk-neutral probability
                redefined_delta = delta
                if delta != 0:
                    try:
                        iv = options.get("OptionGreeks", {}).get("iv", 0)
                        dte_days = (expiry_date - today).days
                        redefined_delta = convert_standard_to_nd2_delta(delta, iv, dte_days, call_put)
                    except Exception as e:
                        redefined_delta = delta

                if target_delta is not None:
                    # Target delta search (e.g. 0.15)
                    # Automatically handle signs: Calls are positive, Puts are negative
                    effective_target = abs(target_delta) if call_put == "Call" else -abs(target_delta)
                    price_diff = abs(redefined_delta - effective_target)
                else:
                    # Target price search
                    price_diff = abs(option_price - target_price)
                
                if price_diff < min_price_diff and (open_interest > 0 or volume > 0) and options.get("bid",0) > 0 and options.get("ask",0) > 0:
                    if ( strike > stock_price and call_put == "Call" ) or ( strike < stock_price and call_put == "Put" ):
                        min_price_diff = price_diff
                        candidate = {
                            "symbol": contract_symbol,
                            "osiKey": osi_key,
                            "strikePrice": strike,
                            "optionPrice": option_price,
                            "volume": volume,
                            "openInterest": open_interest,
                            "ask_bid_spread": ask_bid_spread,
                            "expiryDate": expiry_date,
                            "distance": round((strike - stock_price)/stock_price * 100,2),
                            "delta": round(redefined_delta, 4),
                            "gamma": round(gamma, 4)
                        }
                if strike_overwrite is not None:
                    if previous_strike and ( ( call_put == 'Put' and previous_strike >= strike_overwrite >= strike ) or ( call_put == 'Call' and previous_strike <= strike_overwrite <= strike ) ):
                        candidate = {
                            "symbol": contract_symbol,
                            "osiKey": osi_key,
                            "strikePrice": strike,
                            "optionPrice": option_price,
                            "volume": volume,
                            "ask_bid_spread": ask_bid_spread,
                            "expiryDate": expiry_date,
                            "distance": round((strike - stock_price)/stock_price * 100,2),
                            "delta": round(redefined_delta, 4),
                            "gamma": round(gamma, 4)
                        }
                        break
                    previous_strike = strike
            if candidate is None:
                search_val = f"delta {target_delta}" if target_delta is not None else f"price {target_price}"
                print(f"No suitable option found for {search_val} in {ticker} {call_put}.")
                return None
            if candidate['volume'] == 0:
                print(f"Candidate volume is 0, check the option chain:{ticker} {call_put} {stock_price} {candidate}")
                for option_pair in sorted_option_pairs:
                    options = option_pair.get(call_put)           
                    # Compute mid-price for each option
                    option_price = (options.get("bid", 0) + options.get("ask", 0)) / 2
                    strike = options.get("strikePrice")
                    volume = options.get("volume", 0) or 0
                    ask_bid_spread = round(options.get("ask", 0) - options.get("bid", 0), 4)
                    
                    osi_key = options.get("osiKey")
                    if osi_key:
                        expiry_date_str = osi_key[6:12]
                        expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()
                    # print(f"strike: {strike}, option_price: {option_price}, target: {target_price:.2f} volume: {volume}, ask_bid_spread: {ask_bid_spread},expire:{expiry_date}")
            return candidate

        sell_option = get_option_by_price(target_premium, call_put, "sell", target_delta=target_delta)
        
        # Check if the selected sell option is at the very edge of the returned strikes, 
        # which indicates the target strike was cut off by E*TRADE's strike limit.
        if sell_option is not None and target_delta is not None:
            strikes = [p.get(call_put, {}).get("strikePrice") for p in option_pairs if p.get(call_put)]
            if strikes:
                min_strike = min(strikes)
                max_strike = max(strikes)
                
                is_boundary = (call_put == "Put" and sell_option["strikePrice"] == min_strike) or \
                              (call_put == "Call" and sell_option["strikePrice"] == max_strike)
                
                effective_target = abs(target_delta) if call_put == "Call" else -abs(target_delta)
                deviates_significantly = abs(sell_option["delta"] - effective_target) > 0.015
                
                if is_boundary and deviates_significantly:
                    print(f"Sell option strike {sell_option['strikePrice']} is at the boundary ({min_strike}/{max_strike}) and deviates from target delta {target_delta} (actual: {sell_option['delta']}). Re-fetching centered further out...")
                    
                    # Shift center by 3% to keep it centered close to the boundary
                    estimated_center = sell_option["strikePrice"] * 0.97 if call_put == "Put" else sell_option["strikePrice"] * 1.03
                    
                    sec_params = params.copy()
                    sec_params["strikePriceNear"] = estimated_center
                    sec_params["noOfStrikes"] = 200
                    sec_response = self.session.get(url, params=sec_params, auth=self.session.auth)
                    if sec_response.status_code == 200:
                        sec_data = sec_response.json()
                        sec_pairs = sec_data.get("OptionChainResponse", {}).get("OptionPair", [])
                        
                        if ticker == "SPX" and sec_pairs:
                            if has_spxw:
                                sec_pairs = [p for p in sec_pairs if p.get(call_put) and p.get(call_put, {}).get("osiKey", "").startswith("SPXW")]
                            else:
                                sec_pairs = [p for p in sec_pairs if p.get(call_put) and p.get(call_put, {}).get("osiKey", "").startswith("SPX-")]
                        
                        if sec_pairs:
                            original_option_pairs = option_pairs
                            option_pairs = sec_pairs
                            better_sell_option = get_option_by_price(target_premium, call_put, "sell", target_delta=target_delta)
                            if better_sell_option is not None:
                                print(f"Found better sell option after boundary shift: {better_sell_option['strikePrice']} (delta: {better_sell_option['delta']})")
                                sell_option = better_sell_option
                            else:
                                option_pairs = original_option_pairs
        if sell_option is not None:
            print(f"Sell {call_put} option for {ticker} price now: ${stock_price} strike {sell_option['strikePrice']} price {sell_option['optionPrice']:.4f}, volume {sell_option['volume']}, delta {sell_option['delta']} ask_bid_spread {sell_option['ask_bid_spread']:.4f} expire {sell_option['expiryDate']}")
            if target_delta is None and (1.5 < sell_option['optionPrice'] / target_premium < 0.5):
                print(f"Sell option price {sell_option['optionPrice']} is not within the target premium range {target_premium}, skip trading.")
                return None
        else:
            return None
	        
        sell_position = StockPosition(
            symbol=sell_option.get("symbol", ticker),
            quantity=qty,
            last_price=sell_option['optionPrice'],
            osi_key=sell_option["osiKey"],
        )
        sell_position.strike_price = sell_option["strikePrice"]
        sell_position.expiration_date = sell_option['expiryDate']
        sell_position.call_put = call_put
        sell_position.distance_to_strike = sell_option['distance']
        sell_position.delta = sell_option.get('delta')

        if hedge_ratio == 1 and hedge_spread is None:
            print(f"No hedge planned")
            buy_option = None
            buy_position_target = None
            buy_position = None
            expect_profit = round(sell_option['optionPrice'], 2)
        else:
            if hedge_spread is not None:
                if call_put == "Call":
                    buy_strike_target = sell_option['strikePrice'] + hedge_spread
                if call_put == "Put":
                    buy_strike_target = sell_option['strikePrice'] - hedge_spread
                
                # Check if target strike is likely out of range of current option_pairs
                non_empty_strikes = [p.get(call_put, {}).get("strikePrice") for p in option_pairs if p.get(call_put)]
                if non_empty_strikes:
                    min_strike = min(non_empty_strikes)
                    max_strike = max(non_empty_strikes)
                else:
                    min_strike, max_strike = 0, float("inf")
                
                if (call_put == "Put" and buy_strike_target < min_strike) or (call_put == "Call" and buy_strike_target > max_strike):
                    print(f"Target strike {buy_strike_target} is out of current option chain range ({min_strike} to {max_strike}). Fetching new chain centered near target...")
                    sec_params = params.copy()
                    sec_params["strikePriceNear"] = buy_strike_target
                    sec_params["noOfStrikes"] = 100
                    sec_response = self.session.get(url, params=sec_params, auth=self.session.auth)
                    if sec_response.status_code == 200:
                        sec_data = sec_response.json()
                        sec_pairs = sec_data.get("OptionChainResponse", {}).get("OptionPair", [])
                        
                        # Consistently filter the new chain
                        if ticker == "SPX" and sec_pairs:
                            if has_spxw:
                                sec_pairs = [p for p in sec_pairs if p.get(call_put) and p.get(call_put, {}).get("osiKey", "").startswith("SPXW")]
                            else:
                                sec_pairs = [p for p in sec_pairs if p.get(call_put) and p.get(call_put, {}).get("osiKey", "").startswith("SPX-")]
                        
                        # Search in the new chain
                        original_option_pairs = option_pairs
                        option_pairs = sec_pairs
                        buy_option = get_option_by_price(target_premium / hedge_ratio, call_put, "buy", strike_overwrite=buy_strike_target)
                        option_pairs = original_option_pairs
                    else:
                        print(f"Error fetching option chain near target: {sec_response.status_code}")
                        buy_option = None
                else:
                    buy_option = get_option_by_price(target_premium / hedge_ratio, call_put, "buy", strike_overwrite=buy_strike_target)
            else:
                buy_option = get_option_by_price(target_premium / hedge_ratio, call_put, "buy")

            print(f"Buy {call_put} option for {ticker} price now: ${stock_price} strike {buy_option['strikePrice']} price {buy_option['optionPrice']:.4f}, volume {buy_option['volume']}, ask_bid_spread {buy_option['ask_bid_spread']:.4f} expire {buy_option['expiryDate']}")
            expect_profit = round(sell_option['optionPrice'] - buy_option['optionPrice'], 2)
            buy_position_target = {
                "ticker": buy_option.get("symbol", ticker),
                "call_put": call_put,
                "strike": buy_option['strikePrice'],
                "price": buy_option['optionPrice'],
                "volume": buy_option['volume'],
                "spread": buy_option['ask_bid_spread'],
                "expire": buy_option['expiryDate'],
                "action": "buy"
            }
            buy_position = StockPosition(
                symbol=buy_option.get("symbol", ticker),
                quantity=qty,
                last_price=buy_option['optionPrice'],
                osi_key=buy_option["osiKey"],
            )
            buy_position.strike_price = buy_option["strikePrice"] 
            buy_position.expiration_date = buy_option['expiryDate']
            buy_position.call_put = call_put
            buy_position.distance_to_strike = buy_option['distance']
            buy_position.delta = buy_option.get('delta')
            
            if sell_position.last_price - buy_position.last_price <=0:
                return None
        print(f"Expect profit: {expect_profit} ")

        sell_position_target = {
            "ticker": sell_option.get("symbol", ticker),
            "call_put": call_put,
            "strike": sell_option['strikePrice'],
            "price": sell_option['optionPrice'],
            "volume": sell_option['volume'],
            "spread": sell_option['ask_bid_spread'],
            "expire": sell_option['expiryDate'],
            "action": "sell"
        }

        return {
            "sell_option": sell_position,
            "buy_option": buy_position,
            "sell_position_target": sell_position_target,
            "buy_position_target": buy_position_target,
            "profit": expect_profit
        }

    def get_option_spread_by_credit_target(self, ticker, call_put, days_to_expire, target_credit, hedge_spread, qty=1):
        stock_price = self.get_stock_price(ticker)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None
        if target_credit is None or target_credit <= 0:
            print("Target credit must be positive.")
            return None
        if hedge_spread is None or hedge_spread <= 0:
            print("Hedge spread must be positive.")
            return None

        today = datetime.today().date()
        target_expire = pd.Timestamp(today + pd.Timedelta(days=days_to_expire))

        available_expirations = self.get_available_expirations(ticker)
        if not available_expirations:
            print("No available expiration dates for options.", ticker)
            return None

        target_expiration = _select_nearest_expiration(available_expirations, target_expire.date())
        if target_expiration is None:
            return None

        url = f"{self.base_url}/v1/market/optionchains.json"
        params = {
            "symbol": ticker,
            "expiryYear": target_expiration.year,
            "expiryMonth": target_expiration.month,
            "expiryDay": target_expiration.day,
            "includeWeekly": True,
            "skipAdjusted": False,
            "optionCategory": "ALL",
            "chainType": call_put,
            "noOfStrikes": 400
        }
        response = self.session.get(url, params=params, auth=self.session.auth)
        if response.status_code != 200:
            print("Error fetching option chain:", ticker, response.status_code, response.text)
            return None

        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])

        options_by_strike = {}
        for option_pair in option_pairs:
            option = option_pair.get(call_put)
            if not option:
                continue
            strike = float(option.get("strikePrice"))
            bid = float(option.get("bid", 0) or 0)
            ask = float(option.get("ask", 0) or 0)
            if bid <= 0 or ask <= 0:
                continue

            osi_key = option.get("osiKey")
            expiry_date = target_expiration
            if osi_key:
                expiry_date = datetime.strptime(osi_key[6:12], "%y%m%d").date()

            raw_delta = float(option.get("OptionGreeks", {}).get("delta", 0))
            iv = float(option.get("OptionGreeks", {}).get("iv", 0))
            dte_days = (expiry_date - today).days
            redefined_delta = raw_delta
            if raw_delta != 0:
                try:
                    redefined_delta = convert_standard_to_nd2_delta(raw_delta, iv, dte_days, call_put)
                except Exception:
                    pass

            options_by_strike[strike] = {
                "symbol": _symbol_from_osi_key(ticker, osi_key),
                "strikePrice": strike,
                "optionPrice": (bid + ask) / 2,
                "volume": option.get("volume", 0) or 0,
                "openInterest": option.get("openInterest", 0) or 0,
                "ask_bid_spread": round(ask - bid, 4),
                "expiryDate": expiry_date,
                "distance": round((strike - stock_price) / stock_price * 100, 2),
                "delta": round(redefined_delta, 4),
                "gamma": round(option.get("OptionGreeks", {}).get("gamma", 0), 4)
            }

        if not options_by_strike:
            print(f"No liquid option quotes found for {ticker} {call_put} {target_expiration}.")
            return None

        strikes = sorted(options_by_strike)
        candidates = []
        for short_strike, short_option in options_by_strike.items():
            if short_option["openInterest"] <= 0 and short_option["volume"] <= 0:
                continue
            if call_put == "Call" and short_strike <= stock_price:
                continue
            if call_put == "Put" and short_strike >= stock_price:
                continue

            long_strike_target = short_strike + hedge_spread if call_put == "Call" else short_strike - hedge_spread
            if call_put == "Call":
                long_candidates = [strike for strike in strikes if strike > short_strike]
            else:
                long_candidates = [strike for strike in strikes if strike < short_strike]
            if not long_candidates:
                continue

            long_strike = min(long_candidates, key=lambda strike: abs(strike - long_strike_target))
            long_option = options_by_strike.get(long_strike)
            if not long_option or (long_option["openInterest"] <= 0 and long_option["volume"] <= 0):
                continue

            credit = short_option["optionPrice"] - long_option["optionPrice"]
            if credit <= 0:
                continue
            candidates.append((credit, short_option, long_option))

        if not candidates:
            print(f"No suitable credit-target spread found for {target_credit:.2f} in {ticker} {call_put}.")
            return None

        credit, sell_option, buy_option = min(
            candidates,
            key=lambda item: (0, item[0] - target_credit) if item[0] >= target_credit else (1, target_credit - item[0])
        )

        print(
            f"Sell {call_put} option for {ticker} by target credit ${target_credit:.2f}: "
            f"strike {sell_option['strikePrice']} price {sell_option['optionPrice']:.4f}, "
            f"delta {sell_option['delta']} expire {sell_option['expiryDate']}"
        )
        print(
            f"Buy {call_put} option for {ticker} by target credit ${target_credit:.2f}: "
            f"strike {buy_option['strikePrice']} price {buy_option['optionPrice']:.4f}, "
            f"expire {buy_option['expiryDate']}"
        )
        expect_profit = round(credit, 2)
        print(f"Expect profit: {expect_profit} ")

        sell_position = StockPosition(symbol=sell_option.get("symbol", ticker), quantity=qty, last_price=sell_option['optionPrice'])
        sell_position.strike_price = sell_option["strikePrice"]
        sell_position.expiration_date = sell_option['expiryDate']
        sell_position.call_put = call_put
        sell_position.distance_to_strike = sell_option['distance']
        sell_position.delta = sell_option.get('delta')

        buy_position = StockPosition(symbol=buy_option.get("symbol", ticker), quantity=qty, last_price=buy_option['optionPrice'])
        buy_position.strike_price = buy_option["strikePrice"]
        buy_position.expiration_date = buy_option['expiryDate']
        buy_position.call_put = call_put
        buy_position.distance_to_strike = buy_option['distance']
        buy_position.delta = buy_option.get('delta')

        return {
            "sell_option": sell_position,
            "buy_option": buy_position,
            "sell_position_target": {
                "ticker": sell_option.get("symbol", ticker),
                "call_put": call_put,
                "strike": sell_option['strikePrice'],
                "price": sell_option['optionPrice'],
                "volume": sell_option['volume'],
                "spread": sell_option['ask_bid_spread'],
                "expire": sell_option['expiryDate'],
                "action": "sell"
            },
            "buy_position_target": {
                "ticker": buy_option.get("symbol", ticker),
                "call_put": call_put,
                "strike": buy_option['strikePrice'],
                "price": buy_option['optionPrice'],
                "volume": buy_option['volume'],
                "spread": buy_option['ask_bid_spread'],
                "expire": buy_option['expiryDate'],
                "action": "buy"
            },
            "profit": expect_profit,
            "target_credit": round(target_credit, 2)
        }

    def get_option_spread_by_exact_strikes(self, ticker, call_put, days_to_expire, short_strike, long_strike, qty=1):
        stock_price = self.get_stock_price(ticker)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None
        
        today = datetime.today().date()
        target_expire = pd.Timestamp(today + pd.Timedelta(days=days_to_expire))

        available_expirations = self.get_available_expirations(ticker)
        if not available_expirations:
            print("No available expiration dates for options.", ticker)
            return None
        
        target_expiration = _select_nearest_expiration(available_expirations, target_expire.date())
        if target_expiration is None:
            return None

        url = f"{self.base_url}/v1/market/optionchains.json"
        params = {
            "symbol": ticker,
            "expiryYear": target_expiration.year,
            "expiryMonth": target_expiration.month,
            "expiryDay": target_expiration.day,
            "includeWeekly": True,
            "skipAdjusted": False,
            "optionCategory": "ALL",
            "chainType": call_put,
            "noOfStrikes": 400
        }
        response = self.session.get(url, params=params, auth=self.session.auth)
        if response.status_code != 200:
            print("Error fetching option chain:", ticker, response.status_code, response.text)
            return None

        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
                
        def get_option_by_exact_strike(target_strike):
            for option_pair in option_pairs:
                options = option_pair.get(call_put)
                if not options:
                    continue
                
                strike = options.get("strikePrice")
                if abs(strike - target_strike) < 0.01:
                    option_price = (options.get("bid", 0) + options.get("ask", 0)) / 2
                    volume = options.get("volume", 0) or 0
                    ask_bid_spread = round(options.get("ask", 0) - options.get("bid", 0), 4)
                    delta = options.get("OptionGreeks", {}).get("delta", 0)
                    gamma = options.get("OptionGreeks", {}).get("gamma", 0)
                    osi_key = options.get("osiKey")
                    expiry_date = None
                    if osi_key:
                        expiry_date_str = osi_key[6:12]
                        expiry_date = datetime.strptime(expiry_date_str, "%y%m%d").date()
                    
                    raw_delta = float(delta)
                    iv = float(options.get("OptionGreeks", {}).get("iv", 0))
                    dte_days = ((expiry_date or target_expiration) - today).days
                    redefined_delta = raw_delta
                    if raw_delta != 0:
                        try:
                            redefined_delta = convert_standard_to_nd2_delta(raw_delta, iv, dte_days, call_put)
                        except Exception:
                            pass

                    return {
                        "symbol": _symbol_from_osi_key(ticker, osi_key),
                        "strikePrice": strike,
                        "optionPrice": option_price,
                        "volume": volume,
                        "ask_bid_spread": ask_bid_spread,
                        "expiryDate": expiry_date or target_expiration,
                        "distance": round((strike - stock_price)/stock_price * 100, 2),
                        "delta": round(redefined_delta, 4),
                        "gamma": round(gamma, 4)
                    }
            return None

        sell_option = get_option_by_exact_strike(short_strike)
        buy_option = get_option_by_exact_strike(long_strike)

        if sell_option is None or buy_option is None:
            print(f"Could not find exact strikes: Short {short_strike}, Long {long_strike}")
            return None
        
        sell_position = StockPosition(
            symbol=sell_option.get("symbol", ticker),
            quantity=qty,
            last_price=sell_option['optionPrice']
        )
        sell_position.strike_price = sell_option["strikePrice"]
        sell_position.expiration_date = sell_option['expiryDate']
        sell_position.call_put = call_put
        sell_position.distance_to_strike = sell_option['distance']
        sell_position.delta = sell_option.get('delta')

        buy_position = StockPosition(
            symbol=buy_option.get("symbol", ticker),
            quantity=qty,
            last_price=buy_option['optionPrice']
        )
        buy_position.strike_price = buy_option["strikePrice"] 
        buy_position.expiration_date = buy_option['expiryDate']
        buy_position.call_put = call_put
        buy_position.distance_to_strike = buy_option['distance']
        buy_position.delta = buy_option.get('delta')

        expect_profit = round(sell_option['optionPrice'] - buy_option['optionPrice'], 2)

        sell_position_target = {
            "ticker": sell_option.get("symbol", ticker),
            "call_put": call_put,
            "strike": sell_option['strikePrice'],
            "price": sell_option['optionPrice'],
            "volume": sell_option['volume'],
            "spread": sell_option['ask_bid_spread'],
            "expire": sell_option['expiryDate'],
            "action": "sell"
        }
        buy_position_target = {
            "ticker": buy_option.get("symbol", ticker),
            "call_put": call_put,
            "strike": buy_option['strikePrice'],
            "price": buy_option['optionPrice'],
            "volume": buy_option['volume'],
            "spread": buy_option['ask_bid_spread'],
            "expire": buy_option['expiryDate'],
            "action": "buy"
        }

        return {
            "sell_option": sell_position,
            "buy_option": buy_position,
            "sell_position_target": sell_position_target,
            "buy_position_target": buy_position_target,
            "profit": expect_profit
        }
        
    def update_csv_order_statuses(self, executed_orders, current_date=None):
        """
        Reads the CSV file for the given date, compares each row's order_id and strike price against
        executed_orders (list of dicts returned by get_executed_orders), and updates the "order_status" to "EXECUTED". 
        Also updates "executed_price" and "executed_quantity" from the executed orders.
        
        :param executed_orders: List of executed orders (each a dict with key "order_id" and "strike_price")
        :param current_date: Optional date string ("YYYY-MM-DD"); defaults to today.
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
        folder_name = "executed_order_tracker"
        file_name = f"{current_date}.csv"
        file_path = os.path.join(folder_name, file_name)
        
        # Update the fieldnames to include executed order fields
        fieldnames = [
            "ticker", "call_put", "strike", "price", "volume", "spread", "expire", "action", 
            "order_id", "order_status", "executed_price", "executed_quantity"
        ]
        
        if not os.path.exists(file_path):
            print(f"CSV file not found: {file_path}")
            return
        
        # Build a mapping from (order_id, strike) to executed order details
        executed_order_map = {}
        for order in executed_orders:
            oid = str(order.get("order_id"))
            strike = order.get("strike_price")
            if oid and strike is not None:
                try:
                    key = (oid, float(strike))
                except (ValueError, TypeError):
                    key = (oid, strike)
                executed_order_map[key] = order  # executed order info
            
        # Read all rows from the CSV file
        with open(file_path, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            all_rows = list(reader)
        
        if not all_rows:
            print("CSV file is empty.")
            return
        
        # Assume the first row is the header; the rest are data rows
        header = all_rows[0]
        data_rows = all_rows[1:]
        
        # Update order_status and insert executed details when both order_id and strike match
        updated_rows = []
        for row in data_rows:
            row_order_id = row.get("order_id")
            row_strike = row.get("strike")
            match_found = False
            if row_order_id and row_strike:
                try:
                    row_strike_float = float(row_strike)
                except (ValueError, TypeError):
                    row_strike_float = row_strike
                # Iterate over the mapping to find a matching executed order
                for (exec_oid, exec_strike) in executed_order_map:
                    if row_order_id == exec_oid and row_strike_float == exec_strike:
                        executed_info = executed_order_map[(exec_oid, exec_strike)]
                        row["order_status"] = "EXECUTED"
                        row["executed_price"] = executed_info.get("executed_price", "")
                        row["executed_quantity"] = executed_info.get("executed_quantity", "")
                        match_found = True
                        break
            if not match_found:
                # Retain the row as is if no match is found
                row.setdefault("order_status", "")
                row.setdefault("executed_price", "")
                row.setdefault("executed_quantity", "")
            updated_rows.append(row)
        
        # Write header and updated rows back to the CSV file
        with open(file_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        
        print(f"CSV order statuses updated in {file_path}")
        
    def record_option_target(self,option_dict,order_id,order_status):
        sell_position_target = option_dict['sell_position_target']
        buy_position_target = option_dict['buy_position_target']
        
        # Add order_id and status to both dictionaries
        if sell_position_target is not None:
            sell_position_target['order_id'] = order_id
            sell_position_target['order_status'] = order_status
        if buy_position_target is not None:
            buy_position_target['order_id'] = order_id
            buy_position_target['order_status'] = order_status

        # Create the "executed_order_tracker" folder if it doesn't exist
        folder_name = "executed_order_tracker"
        os.makedirs(folder_name, exist_ok=True)

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create the CSV file path
        file_name = f"{current_date}.csv"
        file_path = os.path.join(folder_name, file_name)

        # Define the fieldnames for the CSV file - add order_id and order_status
        fieldnames = ["ticker", "call_put", "strike", "price", "volume", "spread", "expire", "action", "order_id", "order_status"]

        # Write the sell_position_target to the CSV file
        with open(file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Write the sell_position_target as a new row
            if sell_position_target is not None:
                writer.writerow(sell_position_target)
            if buy_position_target is not None:
                writer.writerow(buy_position_target)

        print(f"Sell position target dumped to CSV file: {file_path}")

    def analyze_price_difference(self, date, executed_orders: list[dict]) -> list[dict]:
        """
        Cross-reference executed orders from E*TRADE with tracker records and calculate the difference.
        Also compare CSV data at ~1pm to the Polygon.io data.
        
        Matching is done by ticker symbol, strike, option type, and expiry date.
        """

        records = []
        folder_name = "executed_order_tracker"
        file_name = f"{date}.csv"
        file_path = os.path.join(folder_name, file_name)
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                records.append(row)

        differences = []
        polygonio_differences = []

        # --------------------------------------------------------------------------
        # 1) Compare each executed order to the CSV records (your existing logic).
        # --------------------------------------------------------------------------
        for order in executed_orders:
            # Extract order details
            ticker = order.get("symbol")
            executed_price = order.get("executed_price")
            expired_date = order.get("expiry_date")
            executed_date = order.get("executed_date")
            call_put = order.get("option_type")
            strike = order.get("strike_price")
            order_action = order.get("order_action")

            if ticker is None or executed_price is None or executed_date is None or strike is None:
                print(f"Skipping order with missing details: {order}")
                continue

            try:
                executed_price = float(executed_price)
                order_strike = float(strike)
                dt_executed = datetime.strptime(executed_date, "%Y-%m-%d %H:%M:%S")
            except Exception:
                print(f"Skipping order with invalid price or date format: {order}")
                continue

            action_map = {"BUY_OPEN": "buy", "SELL_OPEN": "sell"}
            mapped_order_action = action_map.get(order_action, order_action.lower())

            # Compare to CSV records
            for record in records:
                if record.get("ticker") != ticker:
                    continue

                rec_strike = record.get("strike")
                rec_date_str = record.get("order_time")
                rec_action = record.get("action")
                if rec_strike is None or rec_date_str is None:
                    continue

                try:
                    rec_strike = float(rec_strike)
                    dt_record = datetime.strptime(rec_date_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

                if rec_strike != order_strike:
                    continue

                # Allow up to 5 minutes delta
                if abs(dt_executed - dt_record) > timedelta(minutes=5):
                    continue

                if rec_action and rec_action.lower() != mapped_order_action:
                    continue

                try:
                    target_price = float(record.get("price", 0))
                    diff = executed_price - target_price
                    differences.append({
                        "ticker": ticker,
                        "target_price": target_price,
                        "executed_price": executed_price,
                        "difference": round(diff, 3),
                        "order_action": mapped_order_action,
                        "order_executed_date": executed_date,
                        "record_executed_date": rec_date_str,
                        "strike": order_strike,
                        "volume": record.get("volume"),
                    })
                except Exception as e:
                    print(f"Error processing record for ticker {ticker}: {e}")

            # ----------------------------------------------------------------------
            # 2) Compare each executed order to the Polygon.io data (your existing logic).
            # ----------------------------------------------------------------------
            load_stored_option_data(ticker)  # Fills stored_option_price for ticker
            from data_and_research.polygonio_improvequery import stored_option_price
            ticker_data = stored_option_price.get(ticker, {})
            if ticker_data:
                last_date = sorted(ticker_data.keys())[-1]
                last_date_prices = ticker_data.get(last_date, {})
                poly_records = []
                for strike_key, date_dict in last_date_prices.items():
                    try:
                        rec_strike = float(strike_key)
                    except Exception:
                        continue
                    for date_str, options in date_dict.items():
                        for opt_type, data in options.items():
                            if opt_type not in ["call", "put"]:
                                continue
                            poly_records.append({
                                "ticker": ticker,
                                "strike_price": rec_strike,
                                "expiry_date": date_str,
                                "option_type": opt_type,
                                "price": data.get("close_price"),
                                "volume": data.get("close_volume")
                            })

                # Compare executed order to the polygonio records
                for record in poly_records:
                    rec_ticker = record.get("ticker")
                    try:
                        rec_strike = float(record.get("strike_price", 0))
                    except Exception:
                        continue
                    if rec_ticker != ticker or rec_strike != order_strike:
                        continue
                    
                    if expired_date != record.get("expiry_date"):
                        continue

                    option_type = record.get("option_type")
                    option_type_lower = option_type.lower() if option_type else None
                    call_put_lower = call_put.lower() if call_put else None
                    if option_type_lower and option_type_lower != call_put_lower:
                        continue

                    try:
                        polygon_close_price = float(record.get("price", 0))
                    except Exception:
                        continue

                    if mapped_order_action == "buy":
                        diff = polygon_close_price - executed_price
                    elif mapped_order_action == "sell":
                        diff = executed_price - polygon_close_price
                    else:
                        diff = 0  # fallback

                    polygonio_differences.append({
                        "ticker": ticker,
                        "strike": order_strike,
                        "source": "polygonio",
                        "order_action": mapped_order_action,
                        "executed_date": executed_date,
                        "expiry_date": expired_date,
                        "polygon_close_price": polygon_close_price,
                        "executed_price": executed_price,
                        "difference": round(diff / executed_price * 100, 1)
                    })
                    print(f"Polygonio difference: {polygonio_differences[-1]}")

        target_time = pd.to_datetime(f"{date} 13:00:00")
        history_folder = "history_option"
        history_polygon_differences = []
        
        for fname in os.listdir(history_folder):
            if date not in fname:
                continue
            pkl_path = os.path.join(history_folder, fname)
            try:
                history_df = pd.read_pickle(pkl_path)
            except Exception as e:
                print(f"Error reading {pkl_path}: {e}")
                continue
        
            if history_df.empty or "Timestamp" not in history_df.columns:
                continue
        
            # Extract ticker from the filename (assumes format: TICKER_date.pkl)
            ticker_h = fname.split("_")[0]
        
            # Load Polygon.io data for this ticker once per file.
            load_stored_option_data(ticker_h)  # Populates stored_option_price for ticker_h
            from data_and_research.polygonio_improvequery import stored_option_price
            ticker_data = stored_option_price.get(ticker_h, {})
            if not ticker_data:
                print(f"No polygon data found for ticker {ticker_h}. Skipping file {fname}.")
                continue

            last_date_prices = ticker_data.get(date, {})
        
            # Build lookup dict outside of the group loop
            poly_lookup = {}
            for strike_key, date_dict in last_date_prices.items():
                try:
                    poly_strike = float(strike_key)
                except Exception:
                    continue
                for date_str, opt_types in date_dict.items():
                    for opt_type, data in opt_types.items():
                        if opt_type not in ["call", "put"]:
                            continue
                        # Use lowercase for option_type and convert expiry to string for consistency
                        poly_key = (ticker_h, poly_strike, opt_type.lower(), str(date_str))
                        poly_lookup[poly_key] = data.get("close_price", 0)

            # Ensure Timestamp is datetime
            history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
        
            # Group by unique combination of Ticker, Strike_Price, Option_Type, Expiry_Date
            groups = history_df.groupby(["Ticker", "Strike_Price", "Option_Type", "Expiry_Date"])
            for key, group_df in groups:
                group_df = group_df.copy()
                group_df["time_delta"] = group_df["Timestamp"].apply(lambda t: abs(t - target_time))
                best_row = group_df.loc[group_df["time_delta"].idxmin()]
                # Determine a history price from available Ask_Price and Bid_Price
                try:
                    ask = float(best_row["Ask_Price"]) if best_row["Ask_Price"] is not None else 0
                except Exception:
                    ask = 0
                try:
                    bid = float(best_row["Bid_Price"]) if best_row["Bid_Price"] is not None else 0
                except Exception:
                    bid = 0
                best_row_volume = best_row["Volume"] if best_row["Volume"] is not None else 0
                if ask and bid:
                    history_price = (ask + bid) / 2.0
                else:
                    history_price = ask or bid
                
                # key contains (Ticker, Strike_Price, Option_Type, Expiry_Date)
                _, strike_h, opt_type_h, expire_h = key
        
                # Form the lookup key from history data.
                polygon_key = (ticker_h, float(strike_h), opt_type_h.lower(), str(expire_h))
                if polygon_key not in poly_lookup:
                    continue
        
                try:
                    poly_price = float(poly_lookup[polygon_key])
                except Exception:
                    poly_price = 0
        
                price_diff = poly_price - history_price
                history_polygon_differences.append({
                    "ticker": ticker_h,
                    "strike": float(strike_h),  # convert np.float64 to float
                    "option_type": opt_type_h,
                    "expiry_date": expire_h.isoformat() if hasattr(expire_h, "isoformat") else str(expire_h),
                    "history_price": round(float(history_price), 3),
                    "polygon_close_price": round(float(poly_price), 3),
                    "difference": round(price_diff / poly_price * 100, 1) if poly_price != 0 else 100,
                    "volume": int(best_row["Volume"]) if best_row["Volume"] is not None else 0,
                    "Timestamp": best_row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                })
                if best_row_volume > 500:
                    entry = history_polygon_differences[-1]
                    if abs(entry["difference"]) > 20:
                        print(f"\033[31m{entry}\033[0m")
                    else:
                        print(f"{entry}")

    def neutralize_option_delta(self, input_position: StockPosition, all_positions: List[StockPosition]):
        """
        Generate a pair of short call and short put options based on the input short call/put position.
        The two output positions will have deltas similar in magnitude but opposite in polarity.
        The premium of the initial output option is close to half of the input position's price.
        Assumes the input position will be closed.

        Additionally, generate buy_close positions for the input position and an existing opposite short position.

        :param input_position: StockPosition object representing the short call or put position.
        :param all_positions: List of StockPosition objects representing the current portfolio.
        :return: Dictionary with 'sell_call', 'sell_put', 'buy_close_input', and 'buy_close_opposite' StockPosition objects,
                 or None if no suitable pair or positions are found.
        """
        # Validate that the input is a short position
        if input_position.quantity >= 0:
            raise ValueError("Input position must be a short position (quantity < 0).")

        # Extract input position details
        symbol = input_position.symbol
        input_price = input_position.last_price
        target_expiration = input_position.expiration_date
        input_type = input_position.call_put.upper()
        input_strike = input_position.strike_price
        input_qty = abs(input_position.quantity)  # Absolute quantity for closing

        # Target premium for the initial output option
        target_premium = input_price / 2

        # Get current stock price
        stock_price = self.get_stock_price(symbol)
        if stock_price is None:
            print("Failed to retrieve stock price.")
            return None

        # Convert target_expiration to string format "YYYY-MM-DD" for comparison
        if isinstance(target_expiration, date):
            target_expiration_str = target_expiration.strftime("%Y-%m-%d")
        else:
            target_expiration_str = str(target_expiration)

        # Verify the expiration date is available
        available_expirations = self.get_available_expirations(symbol)
        if not available_expirations or target_expiration_str not in available_expirations:
            print(f"No matching expiration date available. Target: {target_expiration_str}")
            return None

        # Fetch option chain data (simplified for this example)
        # In practice, replace with actual API call
        url = f"{self.base_url}/v1/market/optionchains.json"
        params = {
            "symbol": symbol,
            "expiryYear": target_expiration.year,
            "expiryMonth": target_expiration.month,
            "expiryDay": target_expiration.day,
        }
        response = self.session.get(url, params=params, auth=self.session.auth)
        if response.status_code != 200:
            print(f"Error fetching option chain: {response.status_code}")
            return None

        # Parse option chain data (simplified example)
        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
        if not option_pairs:
            print("No option pairs available in the option chain.")
            return None

        # Determine initial and opposite option types
        if input_type == "CALL":
            initial_type = "Call"
            opposite_type = "Put"
            strike_direction = "higher"
        elif input_type == "PUT":
            initial_type = "Put"
            opposite_type = "Call"
            strike_direction = "lower"
        else:
            raise ValueError("Input position must be a CALL or PUT option.")

        # Step 1: Select the initial short option
        initial_options = []
        for option_pair in option_pairs:
            option = option_pair.get(initial_type)
            if not option:
                continue
            strike = option.get("strikePrice")
            if (strike_direction == "higher" and strike > input_strike) or \
               (strike_direction == "lower" and strike < input_strike):
                price = (option.get("bid", 0) + option.get("ask", 0)) / 2
                delta = -option.get("OptionGreeks", {}).get("delta", 0)  # Delta for short position
                initial_options.append({
                    "option": option,
                    "price": price,
                    "delta": delta,
                    "strike": strike
                })

        if not initial_options:
            print(f"No suitable {initial_type} options found with {strike_direction} strike.")
            return None

        # Choose the initial option with price closest to target_premium
        initial_option = min(initial_options, key=lambda x: abs(x["price"] - target_premium))
        initial_delta = initial_option["delta"]

        # Step 2: Select the opposite option with similar delta
        opposite_options = []
        for option_pair in option_pairs:
            option = option_pair.get(opposite_type)
            if not option:
                continue
            price = (option.get("bid", 0) + option.get("ask", 0)) / 2
            delta = -option.get("OptionGreeks", {}).get("delta", 0)  # Delta for short position
            opposite_options.append({
                "option": option,
                "price": price,
                "delta": delta,
                "strike": option.get("strikePrice")
            })

        if not opposite_options:
            print(f"No suitable {opposite_type} options found.")
            return None

        # Choose the opposite option with delta closest to initial_delta
        opposite_option = min(opposite_options, key=lambda x: abs(abs(x["delta"]) - abs(initial_delta)))

        # Step 3: Create StockPosition objects for sell_call and sell_put
        if initial_type == "Call":
            sell_call = StockPosition(
                symbol=symbol,
                quantity=-input_qty,
                last_price=initial_option["price"],
                strike_price=initial_option["strike"],
                expiration_date=target_expiration,
                call_put="CALL",
                delta=initial_option["delta"]
            )
            sell_put = StockPosition(
                symbol=symbol,
                quantity=-input_qty,
                last_price=opposite_option["price"],
                strike_price=opposite_option["strike"],
                expiration_date=target_expiration,
                call_put="PUT",
                delta=opposite_option["delta"]
            )
        else:
            sell_put = StockPosition(
                symbol=symbol,
                quantity=-input_qty,
                last_price=initial_option["price"],
                strike_price=initial_option["strike"],
                expiration_date=target_expiration,
                call_put="PUT",
                delta=initial_option["delta"]
            )
            sell_call = StockPosition(
                symbol=symbol,
                quantity=-input_qty,
                last_price=opposite_option["price"],
                strike_price=opposite_option["strike"],
                expiration_date=target_expiration,
                call_put="CALL",
                delta=opposite_option["delta"]
            )

        # Step 4: Generate buy_close for the input_position
        buy_close_input = StockPosition(
            symbol=symbol,
            quantity=input_qty,  # Positive to close the short position
            last_price=input_price,
            strike_price=input_strike,
            expiration_date=target_expiration,
            call_put=input_type,
            delta=input_position.delta
        )

        # Step 5: Find an existing short position of the opposite type to close
        opposite_positions = [
            pos for pos in all_positions
            if pos.symbol == symbol and pos.call_put == opposite_type.upper() and pos.quantity < 0
        ]

        if not opposite_positions:
            print(f"No existing short {opposite_type} positions found for {symbol}.")
            return None

        # Select an opposite position with sufficient quantity
        suitable_opposite = None
        for pos in opposite_positions:
            if abs(pos.quantity) >= input_qty:
                suitable_opposite = pos
                break

        if not suitable_opposite:
            print(f"No short {opposite_type} position with sufficient quantity (>= {input_qty}).")
            return None

        # Create buy_close for the opposite position
        buy_close_opposite = StockPosition(
            symbol=symbol,
            quantity=input_qty,  # Close the same quantity as input_position
            last_price=suitable_opposite.last_price,
            strike_price=suitable_opposite.strike_price,
            expiration_date=suitable_opposite.expiration_date,
            call_put=opposite_type,
            delta=suitable_opposite.delta
        )

        # Return the result
        return {
            "sell_call": sell_call,
            "sell_put": sell_put,
            "buy_close_input": buy_close_input,
            "buy_close_opposite": buy_close_opposite
        }

    def custom_option_order(self, sell_call, sell_put, buy_close_input, buy_close_opposite):
        """
        Generate a single multi-leg option order for E*TRADE with four legs to neutralize option delta.

        :param sell_call: StockPosition for the new short call to sell to open.
        :param sell_put: StockPosition for the new short put to sell to open.
        :param buy_close_input: StockPosition to buy to close the input short position.
        :param buy_close_opposite: StockPosition to buy to close the opposite short position.
        :return: List containing one dictionary representing the multi-leg order.
        :raises ValueError: If inputs are not StockPosition objects or symbols differ.
        """
        # Validate inputs
        positions = [sell_call, sell_put, buy_close_input, buy_close_opposite]
        if not all(isinstance(pos, StockPosition) for pos in positions):
            raise ValueError("All inputs must be StockPosition objects.")

        symbol = sell_call.symbol
        if not all(pos.symbol == symbol for pos in positions):
            raise ValueError("All positions must have the same underlying symbol.")

        # Generate unique client order ID
        client_order_id = random.randint(1000000000, 9999999999)

        # Construct the four legs
        legs = [
            # Leg 1: Sell to Open the new short call
            {
                'symbol': symbol,
                'orderAction': "SELL_OPEN",
                'quantity': abs(sell_call.quantity),  # Negative quantity becomes positive
                'callPut': sell_call.call_put.upper(),
                'expiryYear': sell_call.expiration_date.year,
                'expiryMonth': sell_call.expiration_date.month,
                'expiryDay': sell_call.expiration_date.day,
                'strikePrice': sell_call.strike_price,
            },
            # Leg 2: Sell to Open the new short put
            {
                'symbol': symbol,
                'orderAction': "SELL_OPEN",
                'quantity': abs(sell_put.quantity),
                'callPut': sell_put.call_put.upper(),
                'expiryYear': sell_put.expiration_date.year,
                'expiryMonth': sell_put.expiration_date.month,
                'expiryDay': sell_put.expiration_date.day,
                'strikePrice': sell_put.strike_price,
            },
            # Leg 3: Buy to Close the input position
            {
                'symbol': symbol,
                'orderAction': "BUY_CLOSE",
                'quantity': buy_close_input.quantity,  # Already positive
                'callPut': buy_close_input.call_put.upper(),
                'expiryYear': buy_close_input.expiration_date.year,
                'expiryMonth': buy_close_input.expiration_date.month,
                'expiryDay': buy_close_input.expiration_date.day,
                'strikePrice': buy_close_input.strike_price,
            },
            # Leg 4: Buy to Close the opposite position
            {
                'symbol': symbol,
                'orderAction': "BUY_CLOSE",
                'quantity': buy_close_opposite.quantity,
                'callPut': buy_close_opposite.call_put.upper(),
                'expiryYear': buy_close_opposite.expiration_date.year,
                'expiryMonth': buy_close_opposite.expiration_date.month,
                'expiryDay': buy_close_opposite.expiration_date.day,
                'strikePrice': buy_close_opposite.strike_price,
            }
        ]

        # Build the order dictionary
        order = {
            'client_order_id': client_order_id,
            'securityType': "OPTN",
            'orderTerm': 'GOOD_FOR_DAY',
            'orderAction': "SPREAD",
            'spreadType': "CUSTOM",  # Four-leg custom strategy
            'orderType': 'SPREADS',
            'priceType': "LIMIT",
            'limitPrice': 0.1,  # Market order
            'legs': legs,
            'required_margin': 0  # Placeholder; adjust if needed
        }

        # Return as a list to match generate_option_order
        return order

    def option_value_final(self, all_positions: List[StockPosition]):
        """
        Generate separate plots of aggregated option value vs. underlying stock price
        for each (ticker, expiration_date) group and save them in the 'option_value_plot' directory.

        Args:
            all_positions (List[StockPosition]): List of StockPosition objects from portfolio().
        """
        from collections import defaultdict
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timedelta

        os.makedirs("option_value_plot", exist_ok=True)

        # Group options by (ticker, expiration_date)
        ticker_exp_to_options = defaultdict(list)
        for position in all_positions:
            if position.security_type == "Option":
                key = (_aggregate_option_symbol(position.symbol), position.expiration_date)
                ticker_exp_to_options[key].append(position)

        for (ticker, exp_date), options in ticker_exp_to_options.items():
            if not options:
                continue

            if exp_date in (None, "", "None"):
                print(f"[WARN] option_value_final: skipping {ticker} positions with missing expiration date.")
                continue
            try:
                exp_dt = exp_date if hasattr(exp_date, "strftime") else datetime.strptime(str(exp_date), "%Y-%m-%d")
            except Exception:
                print(f"[WARN] option_value_final: could not parse expiration '{exp_date}' for {ticker}; skipping plot.")
                continue

            current_price = options[0].underlying_last_price
            stock_quantity = sum(
                pos.quantity for pos in all_positions
                if pos.security_type == "Stock" and _aggregate_option_symbol(pos.symbol) == ticker
            )

            unique_strikes = sorted(set(option.strike_price for option in options))
            S_min = current_price * 0.8
            S_max = current_price * 1.2
            S_values = np.linspace(S_min, S_max, 1000)
            total_value = np.zeros_like(S_values)

            for option in options:
                if option.call_put.upper() == "CALL":
                    payoff = np.maximum(S_values - option.strike_price, 0)
                elif option.call_put.upper() == "PUT":
                    payoff = np.maximum(option.strike_price - S_values, 0)
                else:
                    continue
                total_value += option.quantity * 100 * payoff

            include_stock_list = ['QQQ', 'BRK.B', 'GOOGL']
            if stock_quantity != 0 and ticker in include_stock_list:
                total_value += stock_quantity * S_values

            total_value_at_strikes = []
            for strike in unique_strikes:
                total = 0
                for option in options:
                    if option.call_put.upper() == "CALL":
                        payoff = max(strike - option.strike_price, 0)
                    elif option.call_put.upper() == "PUT":
                        payoff = max(option.strike_price - strike, 0)
                    else:
                        continue
                    total += option.quantity * 100 * payoff
                if stock_quantity != 0 and ticker in include_stock_list:
                    total += stock_quantity * strike
                total_value_at_strikes.append(total)

            # Plotting
            plt.figure()
            plt.plot(S_values, total_value, label='Total Value')
            plt.scatter(unique_strikes, total_value_at_strikes, color='blue', marker='o', s=20, label='Strike Prices')
            plt.axvline(x=current_price, color='r', linestyle='--', label=f'Current Price: ${current_price:.2f}')
            plt.xlabel("Underlying Stock Price ($)")
            plt.ylabel("Total Value ($)")
            plt.title(f"{ticker} Option Value (Exp: {exp_dt.strftime('%Y-%m-%d')})")
            plt.legend()
            plt.grid(True)

            for strike, value in zip(unique_strikes, total_value_at_strikes):
                plt.vlines(x=strike, ymin=0, ymax=value, color='gray', linestyle='--', linewidth=0.5)

            plt.xticks(rotation=45)
            plt.subplots_adjust(bottom=0.2)

            # Format expiration date in filename
            exp_str = exp_dt.strftime("%Y-%m-%d")
            plot_filename = os.path.join("option_value_plot", f"{ticker}_exp_{exp_str}.png")
            plt.savefig(plot_filename)
            plt.close()
