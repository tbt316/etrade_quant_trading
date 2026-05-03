import os
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import json
import logging
from datetime import datetime, date

# Replace these with your actual imports and definitions.
from data_and_research import polygonio_config  # Assumes polygonio_config.API_KEY exists
from backtesting.polygonio_dailytrade import monthly_recursive_backtest, load_stored_option_data, save_stored_option_data, PolygonAPIClient,plot_recursive_results,get_historical_prices
from datetime import datetime
import numpy as np
import glob
import re
import json
from itertools import product

def load_latest_parameters():
    """
    Find and load the most recent parameters file.
    
    Returns:
        tuple: (param_list, expiration_week) where
               param_list is a list of parameter dictionaries
               expiration_week is the expiration week extracted from the file
    """
    import glob
    import os
    import re
    import json
    
    # Find all parameter files
    param_files = glob.glob("./trade_parameters/final_parameters_*.txt")
    if not param_files:
        print("No existing parameter files found. Starting fresh.")
        return [], None
        
    # Get the most recent file based on timestamp in filename
    latest_file = max(param_files, key=lambda f: os.path.getmtime(f))
    print(f"Loading parameters from: {latest_file}")
    
    # Read and parse the file
    param_list = []
    expiration_week = None
    ticker_set = set()  # To track tickers we've already seen
    
    try:
        with open(latest_file, "r") as f:
            content = f.read()
            
            # Extract the expiration week as an integer
            exp_pattern = r"expiration_week\s*=\s*(\d+)"
            exp_match = re.search(exp_pattern, content)
            if exp_match:
                try:
                    expiration_week = int(exp_match.group(1))
                    print(f"Found expiration week: {expiration_week}")
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse expiration_week as integer")
                    expiration_week = None
            
            # Extract the parameters list using regex
            pattern = r"parameters\s*=\s*\[(.*?)\]"
            matches = re.search(pattern, content, re.DOTALL)
            if not matches:
                print("Could not parse parameters format in file.")
                return [], expiration_week
                
            param_entries = matches.group(1).strip()
            
            # Process each entry that's not commented out
            for line in param_entries.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and "{" in line:
                    # Clean up the line to make it valid JSON
                    line = line.rstrip(',')
                    try:
                        # Convert single quotes to double quotes for JSON parsing
                        line = line.replace("'", '"')
                        param_dict = json.loads(line)
                        ticker = param_dict["ticker"]
                        # Only add if we haven't seen this ticker before
                        if ticker not in ticker_set:
                            param_list.append(param_dict)
                            ticker_set.add(ticker)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line}\nError: {e}")
                        continue
                        
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        return [], None
        
    return param_list, expiration_week

# Configuration and placeholder variables
tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK-B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
# tickers = ['GOOGL','PLTR','MRNA','T','AMZN','AAPL','INTC','WMT','BAC','AMD','AVGO','DIS','PFE','UNH','GOOG','NKE','CSCO','FFIV','OMC','COR','ORCL','LVS','APH','SBUX','MSFT','TSLA','KO','MU','DAL','UBER','NFLX','GM','BA','VZ','C','SMCI','F','META','NEE','MDT','PEP','JPM','NWSA','PANW','CPRT','BMY','ADM','FCX']
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'BRK.B', 'WMT', 'JPM', 'LLY', 'V', 'ORCL', 'MA', 'XOM', 'UNH', 'NFLX','TSM','BA','COST','PG', 'HD','AXP','T','GDX','UPS','TXN','DIS','ASML','ADBE','PEP','CSCO','INTC','QCOM','OXY','GOLD']
extra_ticker = ['QQQ','SPY','DIA','IWM','VONE','GLD','VOO']
non_tech_ticker = ['XLU','XLP','XLV','XLF','XLE','TLT']
mega_cap_ticker = ['MSFT','COST','AMZN','GOOGL','BRK.B','TSM','NFLX','COST','META']
tickers = mega_cap_ticker + extra_ticker
tickers = ['QQQ','IWM','DIA','VOO','BRK.B']
tickers = ['ORCL','BA']
# tickers = ['QQQ','DIA','IWM','VONE','GLD','VOO']
do_not_use = ['VB','VTI','VT']
# tickers = ['GLD','GDX','IWM','TLT','HYG','SLV','XLF']
# tickers = ['SPX', 'NDX', 'RUT', 'DJX', 'XND', 'NQX', 'XSP']
# tickers = ['BRK.B']
global_start_date = "2020-01-01"
# global_start_date = "2024-06-01"
global_end_date   = datetime.now().strftime("%Y-%m-%d")
# global_end_date   = "2022-01-01"
lookback_months   = 0
lookforward_months  = 240
target_premium_otm    = np.arange(0.1,0.3,0.05)
target_premium_otm    = [0.25]
# target_premium_otm    = [0.2]
# target_premium_otm    = [None]
target_delta = [0.025]
target_delta = [0.015]
target_delta = [None]
# target_steer = [0]
target_steer = [0]
iron_condor_width = [15,20,25]
iron_condor_width = [20]

# target_premium = [0.1]
expiring_wks      = [2,3,4]  # Your expiring weeks data
expiring_wks      = [6]  # Your expiring weeks data
vix_correlation    = [0.05,0.1,0.15,0]
vix_correlation    = [0.0]
vix_threshold      = [20,25,15]
vix_threshold      = [20]
# roll_methods      = ['close price','loss','roll']
roll_methods      = [None]
stop_loss_action = ['roll_in']  # 'roll' or 'close' or 'skip'
stop_profit_percent = np.arange(0.2,0.8,0.2)
stop_profit_percent = [0.2]  # Your stop loss percentage(s)
day_of_week       = ['Monday','Tuesday','Wednesday','Thursday','Friday']
# day_of_week       = ['Friday']
trade_type       = ['iron_condor','put_spread']
trade_type       = ['call_credit_spread']

trade_parameters = [
        {
            'expiring_wks': expiring_wks,
            'target_premium_otm': target_premium_otm,
            'target_steer': target_steer,
            'target_delta': target_delta,
            'iron_condor_width': iron_condor_width,
            'stop_loss_action': stop_loss_action,
            'stop_profit_percent': stop_profit_percent,
            'day_of_week': day_of_week,
            'vix_correlation': vix_correlation,
            'vix_threshold': vix_threshold,
            'trade_type': trade_type,
        }
        for expiring_wks, target_premium_otm, target_steer, target_delta, iron_condor_width, stop_loss_action, stop_profit_percent, vix_correlation, vix_threshold, trade_type in product(
            expiring_wks, target_premium_otm, target_steer, target_delta, iron_condor_width, stop_loss_action, stop_profit_percent, vix_correlation, vix_threshold, trade_type
        )
    ]

import os
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed

df = {}
log_dir = "./option_test_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def process_ticker_with_data(ticker):
    try:
        # Load data for this ticker only and release immediately after processing
        ticker_df = get_historical_prices(ticker, global_start_date, global_end_date)
        vix_df = get_historical_prices('VIX', global_start_date, global_end_date)
        
        if ticker_df is None or ticker_df.empty:
            print(f"Failed to load data for {ticker}.")
            return {"ticker": ticker, "final_pnl": 0, "error": "No data"}
        
        print(f"Loaded data for {ticker}.")
        load_stored_option_data(ticker)
        
        # Process with the data
        result = asyncio.run(run_ticker_internal(ticker, {'df': ticker_df, 'vix_df': vix_df},))
        
        # Release memory
        del ticker_df
        return result
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {"ticker": ticker, "final_pnl": 0, "error": str(e)}
    finally:
        # Force garbage collection
        import gc
        gc.collect()

async def run_ticker_internal(ticker, ticker_df):
    print(f"\n--- Starting Backtest for {ticker} ---")
    async with PolygonAPIClient(api_key=polygonio_config.API_KEY, max_concurrent_requests=10) as client:
        final_pnl, dt_series, pnl_cumulative_series, parameter_history, pnl_series, details_m,pnl_cumulative_realized_series = await monthly_recursive_backtest(
            ticker=ticker,
            global_start_date=global_start_date,
            global_end_date=global_end_date,
            lookback_months=lookback_months,
            lookforward_months=lookforward_months,
            trade_parameters=trade_parameters,  # Pass the list of trade parameters
            client=client,
            save_file=True,
            trade_type=trade_type,
            input_df=ticker_df,
        )
        
        save_stored_option_data(ticker)
        print(f"Recursive monthly approach => Final PnL for {ticker}: {final_pnl:.2f}")
        
        if (final_pnl > 200 and len(tickers) > 1) or len(tickers) < 8:
            plot_recursive_results(
                ticker,
                final_pnl,
                details_m,
                pnl_cumulative_series,
                pnl_cumulative_realized_series,
                parameter_history,
                global_start_date,
                global_end_date,
                ticker_df
            )
        return {
            "ticker": ticker,
            "final_pnl": final_pnl,
            "weekly_pnl": pnl_series,
            "weekly_pnl_cumulative": pnl_cumulative_series,
            "dates": dt_series,
            "parameter_history": parameter_history
        }
    
def main():
    # Process tickers in smaller batches to limit memory usage
    BATCH_SIZE = 24  # Adjust based on your machine's capabilities
    recursive_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_file = f"./trade_parameters/final_parameters_{timestamp}.txt"
    
    # Initialize the parameters file with the opening bracket
    with open(params_file, "w") as f:
        f.write(f"expiration_week = {expiring_wks[0]}\n")
        f.write("parameters = [\n")
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch_tickers = tickers[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}: {batch_tickers}")
        
        with ProcessPoolExecutor() as executor:
            future_to_ticker = {executor.submit(process_ticker_with_data, ticker): ticker for ticker in batch_tickers}
            
            for future in as_completed(future_to_ticker):
                try:
                    result = future.result()
                    if "error" not in result:  # Only add successful results
                        ticker_result = result['ticker']
                        recursive_results[ticker_result] = result
                        
                        # Immediately append this ticker's parameters to the file if it meets criteria
                        if result['final_pnl'] > 400 and result['parameter_history']:
                            last_params = result['parameter_history'][-1]
                            param_line = (f"  {{\"ticker\": '{ticker_result}', "
                                         f"\"pnl\": {result['final_pnl']:.1f}, "
                                         f"\"target_premium_call\": {last_params.get('target_premium_call', 0.0):.4f}, "
                                         f"\"target_premium_put\": {last_params.get('target_premium_put', 0.0):.4f}, "
                                         f"\"hedge_ratio_call\": {last_params.get('call_hedge', 0)}, "
                                         f"\"hedge_ratio_put\": {last_params.get('put_hedge', 0)}, "
                                         f"\"qty\": {5}}},\n")
                            
                            with open(params_file, "a") as f:
                                f.write(param_line)
                            
                            # Also print to console
                            print(f"Added parameters for {ticker_result} with PnL: {result['final_pnl']:.1f}")
                            
                except Exception as exc:
                    ticker = future_to_ticker[future]
                    print(f'Processing ticker {ticker} generated an exception: {exc}')
                    
                    # Log the error to the parameters file as a comment
                    with open(params_file, "a") as f:
                        f.write(f"  # Error processing {ticker}: {exc}\n")
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    # Close the parameters file with the closing bracket
    with open(params_file, "a") as f:
        f.write("]\n")
    
    # Print a summary to console
    successful_tickers = len(recursive_results)
    qualifying_tickers = sum(1 for t in recursive_results if 
                            recursive_results[t]['final_pnl'] > 400 and 
                            recursive_results[t]['parameter_history'])
    
    print(f"\nProcessing complete: {successful_tickers} tickers processed, "
          f"{qualifying_tickers} met the PnL threshold.")
    print(f"Final parameters saved to: {params_file}")

if __name__ == '__main__':
    main()