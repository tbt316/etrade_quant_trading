from typing import final
import yfinance as yf
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime
import json
from scipy.stats import norm, expon, lognorm, kstest, anderson, probplot
import sys
import time

def convert_timestamps(obj):
    """
    Recursively converts Timestamp objects to ISO format strings in a nested dictionary.
    """
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    return obj

def save_results(results, ticker):
    """
    Save the results to a JSON file.

    Args:
        results (dict): Results dictionary to save.
        ticker (str): Stock ticker symbol for naming the file.
    """
    # Convert Timestamp objects to strings
    converted_results = convert_timestamps(results)

    # Define the output file path
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"option_limit/{ticker}_results_{date_str}.json"

    with open(filename, "w") as file:
        json.dump(converted_results, file, indent=4)

    print(f"Results saved to {filename}")

def load_results(ticker):
    """
    Load the latest results file containing the specified ticker symbol.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: The loaded results.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    folder = "option_limit"
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # List all files containing the ticker symbol
    matching_files = [
        f for f in os.listdir(folder)
        if ticker in f and f.endswith(".json")
    ]

    if not matching_files:
        raise FileNotFoundError(f"No files found for ticker: {ticker}")

    # Sort files by modification time (newest first)
    matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    latest_file = matching_files[0]
    filepath = os.path.join(folder, latest_file)

    with open(filepath, "r") as file:
        results = json.load(file)
    print(f"Results loaded from {filepath}")
    return results

def fetch_data(ticker, lookback_window, today_date=None):
    """
    Fetch historical data for the given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        lookback_window (int): The lookback period for calculating the moving average and standard deviation.
        today_date (str or datetime, optional): The end date for fetching data. Defaults to None (current date).

    Returns:
        pd.DataFrame: Historical data with calculated moving average and standard deviation.
    """
    if today_date is None:
        today_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    else:
        today_date = pd.Timestamp(today_date).strftime('%Y-%m-%d')  # Ensure string format 'YYYY-MM-DD'

    # Calculate the start date based on the lookback period
    start_date = (pd.Timestamp(today_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

    try:
        df = yf.download(ticker, start=start_date, end=today_date)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker {ticker} from {start_date} to {today_date}. Error: {e}")

    if df.empty:
        raise ValueError(f"Failed to fetch data for ticker {ticker}. Please check the ticker symbol and date range.")

    df['Return'] = df['Close'].pct_change()
    df['MA'] = df['Close'].rolling(window=lookback_window).mean()
    df['StdDev'] = df['Return'].rolling(window=lookback_window).std()

    df['UpperBand'] = df['MA'] * (1 + (3 * df['StdDev']))
    df['LowerBand'] = df['MA'] * (1 - (3 * df['StdDev']))

    # Fetch earnings dates
    t = yf.Ticker(ticker)
    try:
        df_earnings_dates = t.get_earnings_dates(limit=50)
        print(df_earnings_dates)
        breakpoint()
        if df_earnings_dates is not None:
            # Normalize earnings dates and DataFrame index
            earnings_dates_normalized = pd.to_datetime(df_earnings_dates.index).date
            df['Earnings'] = df.index.to_series().apply(lambda x: x.date()).isin(earnings_dates_normalized)
        else:
            df['Earnings'] = False
    except Exception as e:
        print(f"Warning: Unable to fetch earnings dates for {ticker}. Error: {e}")
        df['Earnings'] = False

    return df.dropna()

def get_next_friday(current_date, num_wks, df_index):
    """
    Find the next Friday after the given date within the DataFrame index.

    Args:
        current_date (datetime.date): The current date.
        df_index (pd.DatetimeIndex): The index of the DataFrame containing dates.

    Returns:
        datetime.date: The next valid Friday date in the DataFrame index.
    """
    current_date = pd.Timestamp(current_date).date()
    # Calculate the next Friday
    days_to_friday = (4 - current_date.weekday()) % 7 + 7*num_wks
    next_friday = pd.Timestamp(current_date + pd.Timedelta(days=days_to_friday))

    while next_friday not in df_index:
        next_friday -= pd.Timedelta(1, unit='D')  # Move one day earlier
    return next_friday


def bollinger_breach_analysis(df, multiplier, hedge=1000, sample_dates=500, num_wks=0, exclude_earning=False):
    """
    Analyze Bollinger Band breaches and expected losses.

    Args:
        df (pd.DataFrame): Historical data with calculated moving average and standard deviation.
        lookback_window (int): The lookback period for calculating the moving average and standard deviation.
        days_to_expire (int): The window to check for Bollinger Band breaches.
        multiplier (float): Multiplier for the standard deviation in Bollinger Bands.
        sample_dates (int): Number of random starting points for analysis.
        exclude_earning (bool): Whether to exclude windows overlapping earnings dates.

    Returns:
        tuple: Probabilities, median losses, highest losses, and all losses for upper and lower band breaches.
    """
    base_line = df['Close']
    base_line = base_line.squeeze()  # Ensure base_line is a Series

    eligible_dates = df.index[:-1]

    sampled_dates = []
    # Filter eligible dates to include only Fridays
    friday_dates = [date for date in eligible_dates if date.weekday() == 4]

    if len(friday_dates) < sample_dates:
        raise ValueError("Not enough consecutive Fridays available in the data to meet the sample size.")

    # Ensure sampled_dates contains consecutive Fridays
    for i in range(sample_dates):
        potential_date = friday_dates[i]
        next_friday_idx = i + 1

        # Ensure the next Friday index is within bounds
        if next_friday_idx >= len(friday_dates):
            break

        next_friday = friday_dates[next_friday_idx]
        start_idx = df.index.get_loc(potential_date)
        end_idx = df.index.get_loc(next_friday)

        # Skip if the range includes earnings and exclude_earning is True
        if exclude_earning and df.iloc[start_idx:end_idx]['Earnings'].any():
            continue

        sampled_dates.append((potential_date, next_friday))

    upper_breach_count = 0
    lower_breach_count = 0
    upper_losses = []
    lower_losses = []
    upper_losses_raw = []
    lower_losses_raw = []
    upper_highest_loss = 0
    lower_highest_loss = 0
    upper_highest_loss_date = None
    lower_highest_loss_date = None
    upper_highest_loss_limit = None
    lower_highest_loss_limit = None


    for potential_date, next_friday in sampled_dates:
        start_idx = df.index.get_loc(potential_date)
        next_friday_idx = df.index.get_loc(next_friday)

        upper_band_value = round(base_line.iloc[start_idx] * (1 + (multiplier * df['StdDev'].iloc[start_idx] * np.sqrt((next_friday - potential_date).days))),2)
        lower_band_value = round(base_line.iloc[start_idx] * (1 - (multiplier * df['StdDev'].iloc[start_idx] * np.sqrt((next_friday - potential_date).days))),2)
        
        if isinstance(df['Close'].iloc[next_friday_idx], pd.Series):
            final_close = df['Close'].iloc[next_friday_idx].iloc[0]  # Extract value if Series
        else:
            final_close = df['Close'].iloc[next_friday_idx]  # Use directly if scalar

        final_close = round(final_close,2)

        upper_losses_raw.append(final_close-upper_band_value)
        lower_losses_raw.append(lower_band_value-final_close)

        if final_close > upper_band_value:
            upper_breach_count += 1
            loss = final_close - upper_band_value
            upper_losses.append(min(loss,hedge))
            if loss > upper_highest_loss:
                upper_highest_loss = loss
                upper_highest_loss_date = potential_date
                upper_highest_loss_limit = upper_band_value
                # print(f"date: {potential_date} final_close: {final_close}, upper limit: {upper_highest_loss_limit},multiplier: {multiplier}")
        else:
            upper_losses.append(0)

        if final_close < lower_band_value:
            lower_breach_count += 1
            loss = lower_band_value - final_close
            lower_losses.append(min(loss,hedge))
            if loss > lower_highest_loss:
                lower_highest_loss = loss
                lower_highest_loss_date = potential_date
                lower_highest_loss_limit = lower_band_value
                # print(f"date: {potential_date} final_close: {final_close}, lower limit: {lower_highest_loss_limit},multiplier: {multiplier}")
        else:
            lower_losses.append(0)

    upper_breach_probability = (upper_breach_count / sample_dates) * 100
    lower_breach_probability = (lower_breach_count / sample_dates) * 100

    upper_avg_loss = np.mean(upper_losses) if upper_losses else 0
    lower_avg_loss = np.mean(lower_losses) if lower_losses else 0

    PLOT_DISTRIBUTION = False
    if PLOT_DISTRIBUTION:
        # Plot histograms
        plt.figure(figsize=(12, 6))
        plt.hist(upper_losses, bins=30, alpha=0.7, label="Upper Losses", color="red")
        plt.hist(lower_losses, bins=30, alpha=0.7, label="Lower Losses", color="blue")
        plt.hist(upper_losses_raw, bins=30, alpha=0.7, label="Upper Losses raw", color="orange")
        plt.hist(lower_losses_raw, bins=30, alpha=0.7, label="Lower Losses raw", color="green")
        plt.title("Distribution of Upper and Lower Losses")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Perform normal distribution fit and test
        upper_mu, upper_std = norm.fit(upper_losses)
        lower_mu, lower_std = norm.fit(lower_losses)
        upper_raw_mu, upper_raw_std = norm.fit(upper_losses_raw)
        lower_raw_mu, lower_raw_std = norm.fit(lower_losses_raw)

        upper_raw_ks_stat, upper_raw_ks_p_value = kstest(upper_losses_raw, "norm", args=(upper_raw_mu, upper_raw_std))
        lower_raw_ks_stat, lower_raw_ks_p_value = kstest(lower_losses_raw, "norm", args=(lower_raw_mu, lower_raw_std))

        print(f"Upper_raw Loss Normal Fit: mu={upper_raw_mu:.2f}, std={upper_raw_std:.2f}")
        print(f"Upper_raw KS Statistic: {upper_raw_ks_stat:.4f}, P-value: {upper_raw_ks_p_value:.4f}")

        print(f"Lower_raw Loss Normal Fit: mu={lower_raw_mu:.2f}, std={lower_raw_std:.2f}")
        print(f"Lower_raw KS Statistic: {lower_raw_ks_stat:.4f}, P-value: {lower_raw_ks_p_value:.4f}")

        upper_ks_stat, upper_ks_p_value = kstest(upper_losses, "norm", args=(upper_mu, upper_std))
        lower_ks_stat, lower_ks_p_value = kstest(lower_losses, "norm", args=(lower_mu, lower_std))

        print(f"Upper Loss Normal Fit: mu={upper_mu:.2f}, std={upper_std:.2f}")
        print(f"Upper KS Statistic: {upper_ks_stat:.4f}, P-value: {upper_ks_p_value:.4f}")

        print(f"Lower Loss Normal Fit: mu={lower_mu:.2f}, std={lower_std:.2f}")
        print(f"Lower KS Statistic: {lower_ks_stat:.4f}, P-value: {lower_ks_p_value:.4f}")
        
    return {
        "probability": {
            "upper_breach_probability": upper_breach_probability,
            "lower_breach_probability": lower_breach_probability
        },
        "expected_loss": {
            "upper_avg_loss": upper_avg_loss,
            "lower_avg_loss": lower_avg_loss
        },
        "misc": {
            "upper_highest_loss_limit": upper_highest_loss_limit,
            "lower_highest_loss_limit": lower_highest_loss_limit,
            "upper_loss_distributions": upper_losses,
            "lower_loss_distributions": lower_losses,
            "upper_highest_losses": upper_highest_loss,
            "lower_highest_losses": lower_highest_loss,
            "highest_loss_dates": {
                "upper": upper_highest_loss_date,
                "lower": lower_highest_loss_date
            }
        }
    }

def generate_3d_plot(results, hedges, multipliers, ticker):
    """
    Generate a 3D mesh plot of average losses against hedges and multipliers.

    Args:
        results (dict): Results from the Bollinger breach analysis.
        hedges (list): List of hedge values.
        multipliers (list): List of multipliers.
        ticker (str): Stock ticker symbol.
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Prepare data for 3D plotting
    hedge_mesh, multiplier_mesh = np.meshgrid(hedges, multipliers)
    upper_avg_loss_mesh = np.zeros_like(hedge_mesh, dtype=float)
    lower_avg_loss_mesh = np.zeros_like(hedge_mesh, dtype=float)

    for i, hedge in enumerate(hedges):
        for j, multiplier in enumerate(multipliers):
            result = results[hedge][multiplier]
            upper_avg_loss_mesh[j, i] = result["expected_loss"]["upper_avg_loss"]
            lower_avg_loss_mesh[j, i] = result["expected_loss"]["lower_avg_loss"]

    # Plot for Upper Band Average Losses
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(multiplier_mesh, hedge_mesh, upper_avg_loss_mesh, cmap='Reds', alpha=0.8)
    ax.set_title(f"Upper Band Average Losses for {ticker}")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Hedge")
    ax.set_zlabel("Upper Avg Loss")
    plt.show()

    # Plot for Lower Band Average Losses
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(multiplier_mesh, hedge_mesh, lower_avg_loss_mesh, cmap='Blues', alpha=0.8)
    ax.set_title(f"Lower Band Average Losses for {ticker}")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Hedge")
    ax.set_zlabel("Lower Avg Loss")
    plt.show()


def plot_bollinger_analysis(ticker, raw_data, today_date, num_wks, sample_dates, hedges, multipliers, plot_breach=False, plot_against="hedge"):
    df = raw_data
    # Post-process to truncate DataFrame up to today_date
    if today_date:
        today_date = pd.Timestamp(today_date)
        df = df[df.index <= today_date]
        # Truncate to the latest 2 years (504 rows)
        df_len_wanted = 3*252
        if len(df) > df_len_wanted:
            df = df.tail(df_len_wanted)

    if df.empty:
        raise ValueError(f"No data available for {ticker} up to {today_date}. Please check the date or ticker.")

    results = {}

    for hedge in hedges:
        results[hedge] = {}
        for multiplier in multipliers:
            analysis_result = bollinger_breach_analysis(
                df, multiplier, hedge, sample_dates=sample_dates, num_wks=num_wks, exclude_earning=True
            )
            results[hedge][multiplier] = analysis_result

    # Extract data for plotting
    for hedge, hedge_results in results.items():

        # Plot breaches for highest losses
        if plot_breach:
            for multiplier in multipliers:
                upper_date = hedge_results[multiplier]["misc"]["highest_loss_dates"]["upper"]
                lower_date = hedge_results[multiplier]["misc"]["highest_loss_dates"]["lower"]
                upper_highest_loss_limit = hedge_results[multiplier]["misc"]["upper_highest_loss_limit"]
                lower_highest_loss_limit = hedge_results[multiplier]["misc"]["lower_highest_loss_limit"]
                upper_highest_losses = hedge_results[multiplier]["misc"]["upper_highest_losses"]
                lower_highest_losses = hedge_results[multiplier]["misc"]["lower_highest_losses"]

                if upper_date:
                    start_idx = df.index.get_loc(upper_date)
                    end_idx = df.index.get_loc(get_next_friday(upper_date, num_wks, df.index)) + 1

                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index[start_idx:end_idx], df['Close'][start_idx:end_idx], label='Close', color='blue')
                    plt.plot(df.index[start_idx:end_idx], [upper_highest_loss_limit] * len(df.index[start_idx:end_idx]), label='Upper Band', color='red', linestyle='--')
                    plt.title(f"Highest Upper Band Loss Breach Plot for {ticker} (Multiplier: {round(multiplier, 2)})")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.xticks(df.index[start_idx:end_idx], [date.strftime('%A') for date in df.index[start_idx:end_idx]], rotation=45)

                    # Annotate the loss value on the plot
                    loss_value = upper_highest_losses
                    plt.annotate(f"Loss: {loss_value:.2f}, Limit: {upper_highest_loss_limit}, Date: {upper_date}",
                                xy=(df.index[start_idx], df['Close'].iloc[start_idx]),
                                xytext=(df.index[start_idx], df['Close'].iloc[start_idx] + 5),
                                arrowprops=dict(facecolor='black', arrowstyle='->'),
                                fontsize=10, color='red')

                    plt.legend()
                    plt.grid(True)
                    plt.show()

                if lower_date:
                    start_idx = df.index.get_loc(lower_date)
                    end_idx = df.index.get_loc(get_next_friday(lower_date, num_wks, df.index)) + 1

                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index[start_idx:end_idx], df['Close'][start_idx:end_idx], label='Close', color='blue')
                    plt.plot(df.index[start_idx:end_idx], [lower_highest_loss_limit] * len(df.index[start_idx:end_idx]), label='Lower Band', color='green', linestyle='--')
                    plt.title(f"Highest Lower Band Loss Breach Plot for {ticker} (Multiplier: {round(multiplier, 2)})")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.xticks(df.index[start_idx:end_idx], [date.strftime('%A') for date in df.index[start_idx:end_idx]], rotation=45)

                    # Annotate the loss value on the plot
                    loss_value = lower_highest_losses
                    plt.annotate(f"Loss: {loss_value:.2f}, Limit: {lower_highest_loss_limit}, Date: {lower_date}",
                                xy=(df.index[start_idx], df['Close'].iloc[start_idx]),
                                xytext=(df.index[start_idx], df['Close'].iloc[start_idx] + 5),
                                arrowprops=dict(facecolor='black', arrowstyle='->'),
                                fontsize=10, color='blue')

                    plt.legend()
                    plt.grid(True)
                    plt.show()
    
    return results

def plot_results(results, ticker, plot_against="multiplier", plot_breach=False):
    """
    Plot the results from the Bollinger breach analysis.

    Args:
        results (dict): Results from the Bollinger breach analysis.
        ticker (str): Stock ticker symbol.
        plot_against (str): Either "multiplier" or "hedge" to specify the x-axis for the plots.
    """
    # Determine available hedges and multipliers
    hedges = sorted(results.keys())
    multipliers = sorted(next(iter(results.values())).keys())

    # Dynamic variable selection for plotting
    if plot_against == "multiplier":
        x_axis = multipliers
        x_label = "Multiplier"
        outer_loop = hedges
        inner_key = "hedge"
    elif plot_against == "hedge":
        x_axis = hedges
        x_label = "Hedge"
        outer_loop = multipliers
        inner_key = "multiplier"
    else:
        raise ValueError("Invalid value for plot_against. Use 'multiplier' or 'hedge'.")

    # Extract data and plot for each outer loop variable
    for outer_value in outer_loop:
        upper_breach_probabilities = []
        lower_breach_probabilities = []
        upper_avg_losses = []
        lower_avg_losses = []

        for x_value in x_axis:
            if plot_against == "multiplier":
                data = results[outer_value][x_value]
            else:  # plot_against == "hedge"
                data = results[x_value][outer_value]

            upper_breach_probabilities.append(data["probability"]["upper_breach_probability"])
            lower_breach_probabilities.append(data["probability"]["lower_breach_probability"])
            upper_avg_losses.append(data["expected_loss"]["upper_avg_loss"])
            lower_avg_losses.append(data["expected_loss"]["lower_avg_loss"])

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Plot breach probabilities
        label_suffix = f"{inner_key.capitalize()}={outer_value}" if isinstance(outer_value, str) else f"{inner_key.capitalize()}={outer_value:.2f}"
        ax1.plot(x_axis, upper_breach_probabilities, label=f'Upper Band Breach Probability ({label_suffix})', marker='o', color='red')
        ax1.plot(x_axis, lower_breach_probabilities, label=f'Lower Band Breach Probability ({label_suffix})', marker='x', color='blue')
        ax1.set_title(f"Breach Probabilities for {ticker}")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Breach Probability (%)")
        ax1.legend()
        ax1.grid(True)

        # Add numerical labels
        for x, y in zip(x_axis, upper_breach_probabilities):
            ax1.text(x, y, f"{y:.1f}", fontsize=8, color='red')
        for x, y in zip(x_axis, lower_breach_probabilities):
            ax1.text(x, y, f"{y:.1f}", fontsize=8, color='blue')

        # Plot expected losses
        ax2.plot(x_axis, upper_avg_losses, label=f'Upper Band Avg Loss ({label_suffix})', marker='o', color='red')
        ax2.plot(x_axis, lower_avg_losses, label=f'Lower Band Avg Loss ({label_suffix})', marker='x', color='blue')
        ax2.set_title(f"Average Losses for {ticker}")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Average Loss")
        ax2.legend()
        ax2.grid(True)

        # Add numerical labels for losses
        for x, y in zip(x_axis, upper_avg_losses):
            ax2.text(x, y, f"{y:.2f}", fontsize=8, color='red')
        for x, y in zip(x_axis, lower_avg_losses):
            ax2.text(x, y, f"{y:.2f}", fontsize=8, color='blue')

        plt.tight_layout()
        plt.show()

def generate_random_dates_within_past_years(years, num_dates):
    """
    Generate an array of unique random dates within the past `years` years, ensuring no dates fall within the same calendar week.

    Args:
        years (int): Number of past years to consider.
        num_dates (int): Number of random dates to generate.

    Returns:
        list: List of unique random dates.
    """
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=years)

    # Determine all possible weeks
    all_possible_dates = pd.date_range(start=start_date, end=end_date).tolist()
    unique_weeks = {(date.isocalendar()[0], date.isocalendar()[1]) for date in all_possible_dates}
    total_possible_weeks = len(unique_weeks)

    # Check if the number of unique weeks is less than requested
    if total_possible_weeks < num_dates:
        print(
            f"Warning: Only {total_possible_weeks} unique weeks are available "
            f"within the past {years} years. Returning dates for all available weeks."
        )
        # Return one random date per unique week
        return [min([date for date in all_possible_dates if (date.isocalendar()[0], date.isocalendar()[1]) == week])
                for week in unique_weeks]

    # Generate unique random dates
    random_dates = []
    used_weeks = set()

    while len(random_dates) < num_dates:
        random_date = random.choice(all_possible_dates)
        year, week, _ = random_date.isocalendar()  # Get ISO year and week

        # Ensure the week of the new date has not already been used
        if (year, week) not in used_weeks:
            random_dates.append(random_date)
            used_weeks.add((year, week))  # Mark the week as used

    return random_dates
# ticker = "NVDA"
# lookback_window = 20
# sample_dates = 520
# hedges = [10]
# multipliers = [1]

def get_option_probability(ticker,lookback_window,num_wks,dates_per_test,num_test,hedges,multipliers):
    # Example usage:

    raw_data = fetch_data(ticker, lookback_window)

    # Generate random dates for the past 4 years
    random_dates = generate_random_dates_within_past_years(years=1, num_dates=num_test)
    total_results = []

    for idx, today_date in enumerate(random_dates, start=1):
        results = plot_bollinger_analysis(ticker, raw_data, today_date, num_wks, dates_per_test, hedges, multipliers, plot_breach=False)
        total_results.append(results)
        sys.stdout.write(f"\rProcessed {idx}/{len(random_dates)} dates")
        sys.stdout.flush()

    # To ensure the final message stays in the terminal
    print("\nProcessing complete.")

    upper_avg_losses = []
    lower_avg_losses = []
    upper_breach_probabilities = []
    lower_breach_probabilities = []

    # Extract data from total_results
    for result in total_results:
        for hedge, hedge_results in result.items():
            for multiplier, multiplier_results in hedge_results.items():
                upper_avg_losses.append(multiplier_results["expected_loss"]["upper_avg_loss"])
                lower_avg_losses.append(multiplier_results["expected_loss"]["lower_avg_loss"])
                upper_breach_probabilities.append(multiplier_results["probability"]["upper_breach_probability"])
                lower_breach_probabilities.append(multiplier_results["probability"]["lower_breach_probability"])

    # Create subplots
    GENERATE_HIST=False
    if GENERATE_HIST:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot distributions of upper and lower average losses
        ax1.hist(upper_avg_losses, bins=30, alpha=0.7, label="Upper Avg Loss", color="red")
        ax1.hist(lower_avg_losses, bins=30, alpha=0.7, label="Lower Avg Loss", color="blue")
        ax1.set_title(f"Distribution of Upper and Lower Avg Losses ({ticker})")
        ax1.set_xlabel("Average Loss")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True)

        # Plot distributions of upper and lower breach probabilities
        ax2.hist(upper_breach_probabilities, bins=30, alpha=0.7, label="Upper Breach Probability", color="green")
        ax2.hist(lower_breach_probabilities, bins=30, alpha=0.7, label="Lower Breach Probability", color="purple")
        ax2.set_title(f"Distribution of Upper and Lower Breach Probabilities ({ticker})")
        ax2.set_xlabel("Breach Probability (%)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    # Initialize a 3D array to store results
    percentile_results = np.empty((len(hedges), len(multipliers)), dtype=object)
    # Loop through combinations of hedges and multipliers
    for h_idx, hedge in enumerate(hedges):
        for m_idx, multiplier in enumerate(multipliers):
            upper_avg_losses = []
            lower_avg_losses = []
            upper_breach_probabilities = []
            lower_breach_probabilities = []

            # Extract relevant data for the current hedge and multiplier
            for result in total_results:
                if hedge in result and multiplier in result[hedge]:
                    data = result[hedge][multiplier]
                    upper_avg_losses.append(data["expected_loss"]["upper_avg_loss"])
                    lower_avg_losses.append(data["expected_loss"]["lower_avg_loss"])
                    upper_breach_probabilities.append(data["probability"]["upper_breach_probability"])
                    lower_breach_probabilities.append(data["probability"]["lower_breach_probability"])

            # Compute the 75th percentile for losses and probabilities
            upper_loss_75th = np.percentile(upper_avg_losses, 75) if upper_avg_losses else None
            lower_loss_75th = np.percentile(lower_avg_losses, 75) if lower_avg_losses else None
            upper_prob_75th = np.percentile(upper_breach_probabilities, 75) if upper_breach_probabilities else None
            lower_prob_75th = np.percentile(lower_breach_probabilities, 75) if lower_breach_probabilities else None

            # Store results in the array
            percentile_results[h_idx, m_idx] = {
                "upper_loss_75th": round(upper_loss_75th, 3) if upper_loss_75th is not None else None,
                "lower_loss_75th": round(lower_loss_75th, 3) if lower_loss_75th is not None else None,
                "upper_prob_75th": round(upper_prob_75th, 3) if upper_prob_75th is not None else None,
                "lower_prob_75th": round(lower_prob_75th, 3) if lower_prob_75th is not None else None,
            }

    print(percentile_results)
    return percentile_results

# save_results(results, ticker)
# loaded_result = load_results(ticker)
# plot_results(loaded_result,ticker)