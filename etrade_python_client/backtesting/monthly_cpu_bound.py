# monthly_cpu_bound.py

import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Some global toggles or directories:
LOAD_MONTHLY_DATA = True
MONTHLY_BACKTEST_DIR = "./monthly_backtest_data"

###############################################################################
# Placeholder: parameters_match, get_monthly_backtest_file
# If you have these in another file, import them or define them here.
###############################################################################
def parameters_match(loaded_dict: Dict[str, Any], current_dict: Dict[str, Any]) -> bool:
    """
    Check if the loaded parameters match the current parameters exactly.
    """
    keys_to_check = [
        "ticker", "global_start_date", "global_end_date",
        "lookback_months", "hedge_values", "multiplier_values",
        "target_price_baselines", "expiring_wks"
    ]
    for k in keys_to_check:
        if loaded_dict.get(k) != current_dict.get(k):
            return False
    return True

def get_monthly_backtest_file(ticker: str, global_start_date: str, global_end_date: str) -> str:
    """
    Return a suitable file path for saving monthly backtest results.
    """
    # Example approach: embed timestamp so we don't overwrite older data
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monthly_{ticker}_{global_start_date}_to_{global_end_date}_{timestamp_str}.pkl"
    return os.path.join(MONTHLY_BACKTEST_DIR, filename)

###############################################################################
# The CPU-bound version of your monthly_recursive_backtest
###############################################################################
def run_monthly_backtest_cpu_bound(
    ticker: str,
    global_start_date: str,
    global_end_date: str,
    lookback_months: int,
    hedge_values: List[float],
    multiplier_values: List[float],
    target_price_baselines: List[str],
    expiring_wks: List[int],
    save_file: bool = True
) -> bytes:
    """
    A CPU-bound function that replicates the logic of monthly_recursive_backtest,
    but is purely synchronous. This can be run in a separate process via
    ProcessPoolExecutor or multiprocessing.
    
    Return a pickled bytes object containing the tuple:
      (final_pnl, dt_series, pnl_series, parameter_history)
    """

    # 1) Create the parameter dictionary
    current_parameters = {
        "ticker": ticker,
        "global_start_date": global_start_date,
        "global_end_date": global_end_date,
        "lookback_months": lookback_months,
        "hedge_values": hedge_values,
        "multiplier_values": multiplier_values,
        "target_price_baselines": target_price_baselines,
        "expiring_wks": expiring_wks
    }

    # Ensure the directory for saving loaded data exists
    if not os.path.exists(MONTHLY_BACKTEST_DIR):
        os.makedirs(MONTHLY_BACKTEST_DIR)

    # 2) Attempt to load from file if LOAD_MONTHLY_DATA is True
    if LOAD_MONTHLY_DATA:
        for fname in os.listdir(MONTHLY_BACKTEST_DIR):
            if (fname.startswith(f"monthly_{ticker}_{global_start_date}_to_{global_end_date}")
                and fname.endswith(".pkl")):
                fullpath = os.path.join(MONTHLY_BACKTEST_DIR, fname)
                with open(fullpath, "rb") as f:
                    loaded_data = pickle.load(f)
                loaded_parameters = loaded_data.get("parameters", {})

                if parameters_match(loaded_parameters, current_parameters):
                    print(f"[CPU-Bound] Loaded monthly backtest from file: {fname}")

                    loaded_dt_series  = loaded_data["dt_series"]
                    loaded_pnl_series = loaded_data["pnl_series"]

                    # Slice dt/pnl to [global_start_date, global_end_date]
                    start_dt = datetime.strptime(global_start_date, "%Y-%m-%d")
                    end_dt   = datetime.strptime(global_end_date,   "%Y-%m-%d")

                    dt_sliced, pnl_sliced = [], []
                    for d, p in zip(loaded_dt_series, loaded_pnl_series):
                        # If 'd' was originally stored as datetime; if stored as string, parse it
                        if isinstance(d, str):
                            d = datetime.fromisoformat(d)
                        if start_dt <= d <= end_dt:
                            dt_sliced.append(d)
                            pnl_sliced.append(p)

                    final_pnl_sliced = pnl_sliced[-1] if pnl_sliced else 0.0
                    param_hist = loaded_data["parameter_history"]

                    # Return it => must be pickled so the main process can unpickle
                    result_tuple = (final_pnl_sliced, dt_sliced, pnl_sliced, param_hist)
                    return pickle.dumps(result_tuple)

    print(f"[CPU-Bound] Running monthly_recursive_backtest from scratch for {ticker}...")

    ############################################################################
    # 3) The "scratch" logic for monthly_recursive_backtest
    #    This is a synchronous approach to replicate your existing code.
    #    If your original code calls async functions, you must replace them
    #    with synchronous versions or pre-fetched data.
    ############################################################################

    # Convert strings to datetime
    start_dt = datetime.strptime(global_start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(global_end_date, "%Y-%m-%d")

    global_weekly_dates = []
    global_weekly_pnls = []
    parameter_history = []
    cumulative_pnl = 0.0

    # Helper: get first day of each month
    def month_starts_between(sd, ed):
        dates = []
        current = datetime(sd.year, sd.month, 1)
        while current <= ed:
            dates.append(current)
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        return dates

    all_month_starts = month_starts_between(start_dt, end_dt)
    if not all_month_starts:
        print("[CPU-Bound] No months found in the specified range!")
        # Return empty
        result_tuple = (0.0, [], [], [])
        return pickle.dumps(result_tuple)

    # Example iteration
    for i in range(len(all_month_starts)):
        trade_month_start = all_month_starts[i]
        if i == len(all_month_starts) - 1:
            break
        next_month_start = all_month_starts[i + 1]

        if trade_month_start < start_dt:
            if next_month_start <= start_dt:
                continue
            trade_month_start = start_dt
        if next_month_start > end_dt:
            if trade_month_start >= end_dt:
                break
            trade_month_end = end_dt
        else:
            trade_month_end = next_month_start - timedelta(days=1)
            if trade_month_end > end_dt:
                trade_month_end = end_dt

        training_end = trade_month_start - timedelta(days=1)
        training_start = training_end - timedelta(days=30 * lookback_months)
        if training_start < start_dt:
            training_start = start_dt
        if training_end <= training_start:
            continue

        combo_results = []
        for h in hedge_values:
            for call_m in multiplier_values:
                for put_m in multiplier_values:
                    for t in target_price_baselines:
                        for w in expiring_wks:
                            ############################################################################
                            # TODO: Instead of "await backtest_options_sync_or_async",
                            # you must do a synchronous version or a pre-fetched approach.
                            # We'll just mock some final_pnl data to show how you'd accumulate:
                            ############################################################################
                            # final_pnl, details, sharpe, _, _ = await backtest_options_sync_or_async(...)
                            # Instead, we do an example:
                            final_pnl = 100.0  # Fake
                            sharpe = 1.23      # Fake
                            # End of placeholder

                            adj_sharpe = sharpe / h if abs(h) > 1e-9 else 0.0
                            combo_results.append({
                                "hedge": h,
                                "call_mult": call_m,
                                "put_mult": put_m,
                                "baseline": t,
                                "expiring_wk": w,
                                "final_pnl": final_pnl,
                                "sharpe": sharpe,
                                "adj_sharpe": adj_sharpe,
                            })

        if not combo_results:
            continue

        best_combo = max(combo_results, key=lambda x: x["adj_sharpe"])
        h_star = best_combo["hedge"]
        cm_star = best_combo["call_mult"]
        pm_star = best_combo["put_mult"]
        tb_star = best_combo["baseline"]
        ew_star = best_combo["expiring_wk"]

        print(f"[CPU-Bound] Selected best combo for {trade_month_start.strftime('%Y-%m')} => "
              f"hedge={h_star}, callM={cm_star}, putM={pm_star}, exp_wk={ew_star}, baseline={tb_star}, "
              f"train_sharpe={best_combo['sharpe']:.3f}, train_sharpe_over_h={best_combo['adj_sharpe']:.3f}")

        parameter_history.append({
            "month": trade_month_start.strftime("%Y-%m"),
            "start_date": trade_month_start.strftime("%Y-%m-%d"),
            "end_date": trade_month_end.strftime("%Y-%m-%d"),
            "hedge": h_star,
            "call_multiplier": cm_star,
            "put_multiplier": pm_star,
            "target_price_baseline": tb_star,
            "expiring_weeks": ew_star,
        })

        if trade_month_end < trade_month_start:
            continue

        # Suppose we run the final best strategy:
        # final_pnl_m, details_m, sharpe_m, month_dates, month_pnls = ...
        # We'll just simulate some dummy monthly results:
        final_pnl_m = 200.0
        month_dates = [trade_month_start, trade_month_end]
        month_pnls = [cumulative_pnl, cumulative_pnl + final_pnl_m]

        if month_dates and month_pnls:
            offset = cumulative_pnl
            shifted_pnls = [p + offset for p in month_pnls]
            # Actually, we already added offset in the example
            cumulative_pnl += final_pnl_m

            # Extend the global arrays
            if global_weekly_dates and (month_dates[0] <= global_weekly_dates[-1]):
                pass  # Overlapping check or ignore

            global_weekly_dates.extend(month_dates)
            global_weekly_pnls.extend(month_pnls)

    final_val = 0.0
    if global_weekly_pnls:
        final_val = global_weekly_pnls[-1]

    # 4) Optionally save to .pkl
    if save_file:
        out_filename = get_monthly_backtest_file(ticker, global_start_date, global_end_date)
        data_to_save = {
            "parameters": current_parameters,
            "final_pnl": final_val,
            "dt_series": global_weekly_dates,
            "pnl_series": global_weekly_pnls,
            "parameter_history": parameter_history,
        }
        with open(out_filename, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"[CPU-Bound] Saved monthly_recursive_backtest results to {out_filename}")

    # 5) Return the final results as pickled bytes
    # The main process will unpickle them
    result_tuple = (final_val, global_weekly_dates, global_weekly_pnls, parameter_history)
    return pickle.dumps(result_tuple)