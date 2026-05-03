import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import contextlib
import io
import concurrent.futures
from functools import partial

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels
from backtesting.backtest_runner import run_put_credit_spread_backtest

def calculate_metrics(res):
    """Calculate Sharpe, Sortino, and Profit Factor from BacktestResult."""
    if not res.nvl_history:
        return {"sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0}
    
    df = pd.DataFrame(res.nvl_history, columns=['date', 'nlv'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Daily returns
    df['returns'] = df['nlv'].pct_change().fillna(0)
    
    # Sharpe Ratio (Annualized)
    mean_ret = df['returns'].mean()
    std_ret = df['returns'].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Sortino Ratio (Annualized)
    downside_df = df[df['returns'] < 0]
    downside_std = downside_df['returns'].std() if not downside_df.empty else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    
    # Profit Factor
    closed = [t for t in res.trades if t.status == "closed"]
    wins = sum(t.pnl_per_contract * t.num_contracts for t in closed if t.pnl_per_contract > 0)
    losses = abs(sum(t.pnl_per_contract * t.num_contracts for t in closed if t.pnl_per_contract <= 0))
    profit_factor = wins / losses if losses > 0 else (float('inf') if wins > 0 else 1.0)
    
    # Max DD %
    peak = df['nlv'].cummax()
    dd = (df['nlv'] - peak) / peak
    max_dd_pct = abs(dd.min())
    
    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 2),
        "max_dd_pct": round(max_dd_pct, 4)
    }

def backtest_worker_sync(params, reg_dict, labels, spy_prices, vix_prices, start_date, end_date):
    """
    Synchronous worker for multiprocessing.
    Runs one backtest and returns the result dictionary.
    """
    import asyncio
    # Need to import inside worker because of multiprocessing
    from backtesting.backtest_runner import run_put_credit_spread_backtest
    
    # We use asyncio.run to execute the async backtest in the worker process
    # Redirect stdout to suppress Phase 4 logs in workers
    with contextlib.redirect_stdout(io.StringIO()):
        res = asyncio.run(run_put_credit_spread_backtest(
            underlying="SPY",
            start_date=start_date,
            end_date=end_date,
            target_short_delta=params['short_delta'],
            panic_delta_multiplier=params['panic_delta_mult'],
            panic_dte_target=63,
            panic_swap_enabled=params['panic_swap'],
            regimes=reg_dict,
            regime_labels=labels,
            underlying_prices=spy_prices,
            vix_prices=vix_prices,
            backtest_qty=2,
            plot=False,
            enable_logging=False
        ))
    
    metrics = calculate_metrics(res)
    return {
        **params,
        'total_pnl': res.total_pnl,
        'max_drawdown': abs(res.max_drawdown),
        'sharpe': metrics['sharpe'],
        'sortino': metrics['sortino'],
        'profit_factor': metrics['profit_factor'],
        'max_dd_pct': metrics['max_dd_pct'],
        'score': metrics['sharpe'],
        'win_rate': res.win_count / max(1, res.total_trades)
    }

async def optimize():
    # 1. Fetch data once
    print("Fetching historical data...")
    df_hist = fetch_historical_data()
    
    # 2. Define parameter grid (Reduced for speed)
    param_grid = {
        'short_delta': [-0.12, -0.14, -0.16, -0.18, -0.20],
        'panic_delta_mult': [2.0, 4.0],
        'panic_swap': [True, False],
        'k': [3, 4]
    }
    
    # Prepare all combinations
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results_list = []
    
    # Cache HMM results for each K to avoid retraining
    hmm_cache = {}

    # Common prices
    start_date = "2024-01-01"
    end_date = "2026-04-25"
    
    spy_close = df_hist['SPY_Close'].copy()
    spy_close.index = spy_close.index.strftime("%Y-%m-%d")
    vix_close = df_hist['VIX_Close'].copy()
    vix_close.index = vix_close.index.strftime("%Y-%m-%d")
    
    mask = (spy_close.index >= start_date) & (spy_close.index <= end_date)
    spy_prices = spy_close[mask]
    vix_prices = vix_close[mask]

    # --- PHASE 0: PRE-WARM CACHE ---
    print("\n" + "="*80)
    print(f"  PHASE 0: PRE-WARMING CACHE ({start_date} -> {end_date})")
    print("="*80)
    print("Running a single backtest to populate SQLite cache for all expirations...")
    # Using default params for pre-warm pass (Phase 1-3)
    await run_put_credit_spread_backtest(
        underlying="SPY",
        start_date=start_date,
        end_date=end_date,
        underlying_prices=spy_prices,
        vix_prices=vix_prices,
        plot=False,
        enable_logging=False
    )
    print("✓ Cache pre-warmed. All subsequent runs will be 100% cache-driven.")

    # --- PHASE 1: EXECUTION ---
    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    print(f"\nTotal combinations to test: {len(combinations)}")
    
    results_list = []
    
    # Group by K to avoid redundant HMM training
    from collections import defaultdict
    by_k = defaultdict(list)
    for p in combinations:
        by_k[p['k']].append(p)
    
    num_workers = os.cpu_count() or 4
    print(f"Starting parallel execution with {num_workers} workers...")

    for k, params_list in by_k.items():
        print(f"\n--- Training HMM for K={k} ---")
        best_hmm, best_k, feature_df = train_regime_hmm(df_hist, n_components=k)
        labels = get_regime_labels(best_hmm)
        reg_dict = {d.strftime('%Y-%m-%d'): s for d, s in feature_df['HMM_State'].to_dict().items()}
        
        print(f"Executing {len(params_list)} combinations for K={k} in parallel...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Prepare worker function with fixed arguments
            worker_fn = partial(
                backtest_worker_sync,
                reg_dict=reg_dict,
                labels=labels,
                spy_prices=spy_prices,
                vix_prices=vix_prices,
                start_date=start_date,
                end_date=end_date
            )
            
            # Run tasks
            futures = [executor.submit(worker_fn, p) for p in params_list]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res_dict = future.result()
                results_list.append(res_dict)
                print(f"  [{i+1}/{len(params_list)}] K={k} | Delta={res_dict['short_delta']} | Sharpe={res_dict['sharpe']:.2f} | PnL=${res_dict['total_pnl']:,.0f}")

    # 3. Report Best
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='score', ascending=False)
    
    print("\n" + "="*80)
    print("  OPTIMIZATION RESULTS (Top 10)")
    print("="*80)
    print(results_df.head(10).to_string(index=False))
    
    best = results_df.iloc[0]
    print("\nBEST COMBINATION:")
    print(best)
    
    results_df.to_csv("scratch/optimization_results.csv", index=False)
    print(f"\nFull results saved to scratch/optimization_results.csv")

if __name__ == "__main__":
    asyncio.run(optimize())
