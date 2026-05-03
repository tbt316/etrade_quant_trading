import yfinance as yf
import pandas as pd
import datetime
import argparse
from dateutil.relativedelta import relativedelta
import pytz

import datetime
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from backtesting.polygonio_dailytrade import get_historical_prices

def calculate_probability(ticker, price_offset_percent, day_of_week, look_back_years, expiration_week):
    """
    Calculate the probability that the stock price on a future Friday will exceed a target price.
    Excludes weeks containing earnings announcements.
    
    Parameters:
    ticker (str): Stock ticker symbol
    price_offset_percent (float): Percentage offset to calculate target price
    day_of_week (str): Starting day of week
    look_back_years (int): Number of years to look back for historical data
    expiration_week (int): Number of weeks to expiration
    
    Returns:
    dict: Results including probability and other relevant data
    """
    # Convert day_of_week to lowercase for case-insensitive comparison
    day_of_week = day_of_week.lower()
    
    # Map day of week from string to integer (0=Monday, 1=Tuesday, ..., 6=Sunday)
    days_mapping = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
    
    if ticker == "BRK.B":
        ticker = "BRK-B"
    # Check if the provided day_of_week is valid
    if day_of_week not in days_mapping:
        return f"Invalid day of week: {day_of_week}. Please use monday, tuesday, etc."
    
    dow_int = days_mapping[day_of_week]
    
    # Get end date (today) and start date (look_back_years ago)
    end_date = datetime.datetime.now()
    start_date = end_date - relativedelta(years=look_back_years)
    
    # Fetch historical data using get_historical_prices
    try:
        # print(f"Fetching historical data for {ticker} from {start_date.date()} to {end_date.date()}...")
        stock_data = get_historical_prices(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            data_source='yfinance'
        )
        # Convert date column to datetime and set as index, rename close to Close
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data = stock_data.set_index('date')[['close']].rename(columns={'close': 'Close'})
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"
    
    if stock_data.empty:
        return f"No data found for ticker symbol: {ticker}"
    
    # Get earnings dates
    try:
        # print(f"Fetching earnings dates for {ticker}...")
        ticker_info = yf.Ticker(ticker)
        earnings_dates = ticker_info.earnings_dates
        
        if earnings_dates is not None and not earnings_dates.empty:
            # print(f"Found {len(earnings_dates)} earnings dates for {ticker}")
            # Get the timezone of earnings dates (usually 'America/New_York')
            earnings_tz = earnings_dates.index.tz
            # print(f"Earnings dates timezone: {earnings_tz}")
        else:
            print(f"No earnings dates found for {ticker}")
            earnings_dates = pd.DataFrame(index=pd.DatetimeIndex([]))
            earnings_tz = None
    except Exception as e:
        print(f"Error fetching earnings dates: {str(e)}")
        earnings_dates = pd.DataFrame(index=pd.DatetimeIndex([]))
        earnings_tz = None
    
    # Add day of week column
    stock_data['day_of_week'] = stock_data.index.dayofweek
    
    # Filter for the specified day of week
    day_data = stock_data[stock_data['day_of_week'] == dow_int].copy()
    
    if day_data.empty:
        return f"No data found for {day_of_week} in the historical data."
    
    # For each specified day of week, calculate the corresponding Friday date and check prices
    results = []
    earnings_week_count = 0
    
    # Use NY timezone as default for market dates if we couldn't get it from earnings
    ny_tz = pytz.timezone('America/New_York') if earnings_tz is None else earnings_tz
    
    for date, row in day_data.iterrows():
        # Convert date to tz-aware (matching earnings_dates timezone)
        # We need to handle date as a datetime, not a Pandas Timestamp
        date_dt = date.to_pydatetime().replace(tzinfo=None)
        date_aware = ny_tz.localize(date_dt)
        
        # Check if this week contains an earnings announcement
        week_start = date_aware - datetime.timedelta(days=date_aware.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        
        earnings_in_week = False
        if not earnings_dates.empty:
            try:
                week_earnings = earnings_dates[(earnings_dates.index >= week_start) & 
                                              (earnings_dates.index <= week_end)]
                earnings_in_week = not week_earnings.empty
            except Exception as e:
                print(f"Warning: Error checking earnings for week of {week_start.date()}: {str(e)}")
                earnings_in_week = False
        
        # Skip this week if it contains earnings
        if earnings_in_week:
            earnings_week_count += 1
            continue
        
        # Get the closing price (make sure it's a scalar)
        start_price = row['Close'].item()
        
        # Calculate the target price based on the day's closing price and price_offset_percent
        target_price = start_price * (1 + price_offset_percent / 100)
        
        # Calculate how many days to the next Friday (or same day if it's a Friday)
        days_to_friday = (4 - date.weekday()) % 7
        
        # Calculate the target Friday date
        friday_date = date + datetime.timedelta(days=days_to_friday)
        
        # Add the specified number of weeks to get the expiration Friday
        target_friday = friday_date + datetime.timedelta(weeks=expiration_week)
        
        # Convert target_friday to tz-aware
        target_friday_dt = target_friday.to_pydatetime().replace(tzinfo=None)
        target_friday_aware = ny_tz.localize(target_friday_dt)
        
        # Check if the target week contains earnings (if so, skip it)
        target_week_start = target_friday_aware - datetime.timedelta(days=target_friday_aware.weekday())
        target_week_end = target_week_start + datetime.timedelta(days=6)
        
        target_earnings_in_week = False
        if not earnings_dates.empty:
            try:
                target_week_earnings = earnings_dates[(earnings_dates.index >= target_week_start) & 
                                                     (earnings_dates.index <= target_week_end)]
                target_earnings_in_week = not target_week_earnings.empty
            except Exception as e:
                print(f"Warning: Error checking earnings for target week of {target_week_start.date()}: {str(e)}")
                target_earnings_in_week = False
        
        if target_earnings_in_week:
            earnings_week_count += 1
            continue
        
        # Find the closest trading day to the target Friday (within 3 business days)
        closest_trading_day = None
        min_days_diff = float('inf')
        
        # Look for trading days within a window of +/- 3 days from the target Friday
        for delta in range(-3, 4):
            check_date = target_friday + datetime.timedelta(days=delta)
            if check_date in stock_data.index:
                days_diff = abs(delta)
                if days_diff < min_days_diff:
                    min_days_diff = days_diff
                    closest_trading_day = check_date
        
        # If we found a trading day close to the target Friday
        if closest_trading_day is not None:
            # Get future price and ensure it's a scalar
            future_price = stock_data.loc[closest_trading_day, 'Close'].item()
            
            # Check if the future price exceeds or falls below the target price depending on offset direction
            if price_offset_percent >= 0:
                # For positive offset, check if price exceeds target
                exceeds_target = future_price >= target_price
            else:
                # For negative offset, check if price falls below target
                exceeds_target = future_price <= target_price
            
            results.append({
                'start_date': date,
                'start_price': start_price,
                'target_price': target_price,
                'future_date': closest_trading_day,
                'future_price': future_price,
                'exceeds_target': exceeds_target
            })
    
    if not results:
        return f"No matching data points found for {day_of_week} with {expiration_week} weeks to expiration (after excluding earnings weeks)."
    
    # Calculate probability
    exceed_count = sum(1 for result in results if result['exceeds_target'])
    total_count = len(results)
    probability = (exceed_count / total_count) * 100
    
    # Get the most recent price
    latest_price = day_data.iloc[-1]['Close'].item()
    latest_target_price = latest_price * (1 + price_offset_percent / 100)
    
    # Calculate the Friday date for the current prediction
    latest_date = day_data.index[-1]
    days_to_friday = (4 - latest_date.weekday()) % 7
    friday_date = latest_date + datetime.timedelta(days=days_to_friday)
    target_friday = friday_date + datetime.timedelta(weeks=expiration_week)
    
    return {
        'ticker': ticker,
        'latest_price': latest_price,
        'latest_target_price': latest_target_price,
        'price_offset_percent': price_offset_percent,
        'target_friday': target_friday.date(),
        'probability': probability,
        'historical_data_points': total_count,
        'exceed_count': exceed_count,
        'earnings_weeks_excluded': earnings_week_count
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate probability of stock price exceeding a target price.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('price_offset_percent', type=float, help='Price offset percentage (positive or negative)')
    parser.add_argument('day_of_week', type=str, help='Day of the week to start from')
    parser.add_argument('look_back_years', type=int, help='Number of years to look back')
    parser.add_argument('expiration_week', type=int, help='Number of weeks to expiration')
    
    args = parser.parse_args()
    
    result = calculate_probability(
        args.ticker,
        args.price_offset_percent,
        args.day_of_week,
        args.look_back_years,
        args.expiration_week
    )
    
    if isinstance(result, str):
        print(result)
    else:
        print(f"\nResults for {result['ticker']}:")
        print("-" * 40)
        print(f"Latest {args.day_of_week.capitalize()} Price: ${result['latest_price']:.2f}")
        print(f"Target Price: ${result['latest_target_price']:.2f} ({result['price_offset_percent']:+.2f}%)")
        print(f"Target Friday: {result['target_friday']}")
        
        direction = "exceeding" if result['price_offset_percent'] >= 0 else "falling below"
        print(f"\nHistorical Probability of price {direction} target: {result['probability']:.2f}%")
        print(f"(Based on {result['exceed_count']} out of {result['historical_data_points']} historical instances)")
        print(f"Earnings weeks excluded: {result['earnings_weeks_excluded']}")

if __name__ == "__main__":
    main()