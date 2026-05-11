import os
import sys
import asyncio
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta

# Adjust path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading.ev_engine import fetch_historical_data, train_regime_hmm, get_regime_labels, fit_gmm, query_gmm, calendar_days_to_trading_days
from backtesting.option_data_cache import OptionDataCache
from backtesting.massive_api_client import MassiveAPIClient
from backtesting.greeks_calculator import compute_chain_deltas

CACHE_VERSION = 3
TRAINING_CACHE_PATH = "backtest_cache/regime_audit_training.pkl"
RESULTS_CACHE_PATH = "backtest_cache/regime_audit_results.json"


def _build_cache_metadata(df, audit_start, audit_end, horizon, trading_horizon):
    return {
        "cache_version": CACHE_VERSION,
        "data_start": df.index.min().strftime("%Y-%m-%d"),
        "data_end": df.index.max().strftime("%Y-%m-%d"),
        "row_count": int(len(df)),
        "audit_start": audit_start,
        "audit_end": audit_end,
        "horizon_calendar_days": int(horizon),
        "trading_horizon": int(trading_horizon),
        "expanding_window": True,
    }


def _load_json_cache(path, expected_metadata):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"Failed to load cache {path}: {exc}")
        return None

    if isinstance(payload, list):
        return None

    if payload.get("metadata") != expected_metadata:
        return None

    return payload


def _save_json_cache(path, metadata, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_training_cache(path, expected_metadata):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"Failed to load training cache {path}: {exc}")
        return None

    if payload.get("metadata") != expected_metadata:
        return None

    return payload


def _save_training_cache(path, metadata, best_hmm, best_k, feature_df, regime_labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "metadata": metadata,
        "best_hmm": best_hmm,
        "best_k": best_k,
        "feature_df": feature_df,
        "regime_labels": regime_labels,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


async def generate_regime_probability_audit():
    audit_start = "2025-01-01"
    audit_end = "2026-05-01"
    horizon = 42
    trading_horizon = calendar_days_to_trading_days(horizon)

    # 1. Setup Data
    print("Fetching historical price data (SPY/VIX)...")
    df = fetch_historical_data()
    if df.empty:
        print("Error: No price data found.")
        return

    # Ensure Log_Return is present for probability calculations
    df['Log_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    cache_metadata = _build_cache_metadata(df, audit_start, audit_end, horizon, trading_horizon)

    cached_results_payload = _load_json_cache(RESULTS_CACHE_PATH, cache_metadata)
    if cached_results_payload is not None:
        print(f"Loading cached audit results from {RESULTS_CACHE_PATH}...")
        results = cached_results_payload["results"]
        for r in results:
            r['date'] = pd.to_datetime(r['date'])
    else:
        training_payload = _load_training_cache(TRAINING_CACHE_PATH, cache_metadata)
        if training_payload is not None:
            print(f"Loading cached regime training from {TRAINING_CACHE_PATH}...")
            best_hmm = training_payload["best_hmm"]
            best_k = training_payload["best_k"]
            feature_df = training_payload["feature_df"]
            regime_labels = training_payload["regime_labels"]
        else:
            print("Running expanding window HMM training (causal)...")
            best_hmm, best_k, feature_df = train_regime_hmm(df, expanding_window=True)
            regime_labels = get_regime_labels(best_hmm)
            _save_training_cache(
                TRAINING_CACHE_PATH,
                cache_metadata,
                best_hmm,
                best_k,
                feature_df,
                regime_labels,
            )

        # 3. Initialize Options Cache
        db_path = os.path.join(os.getcwd(), "backtest_cache", "option_data.db")
        cache = OptionDataCache(db_path)

        # 4. Audit Window
        audit_dates = feature_df.loc[audit_start:audit_end].index
        results = []
        print(f"Auditing {len(audit_dates)} trading days...")

        future_return_cache = {}

        async with MassiveAPIClient(cache=cache) as client:
            for i, d in enumerate(audit_dates):
                d_str = d.strftime("%Y-%m-%d")
                spot = df.loc[d, 'SPY_Close']
                current_state = int(feature_df.loc[d, 'HMM_State'])
                current_pos = df.index.get_loc(d)
                
                # A. Find the ~15 delta strike for that day (approx 42 DTE)
                target_exp_dt = d + timedelta(days=horizon)
                effective_calendar_horizon = horizon
                
                # Fetch strikes for this day from cache
                chain = cache.get_chain_for_date("SPY", None, "put", d_str) 
                strike_price = None
                delta_prob = 0.15 # Target
                
                if not chain:
                    # Fallback approximation if cache empty
                    vol = df.loc[:d, 'Log_Return'].tail(30).std() * np.sqrt(252)
                    strike_price = spot * (1 - 1.04 * vol * np.sqrt(horizon/365))
                else:
                    # Find the expiration closest to 42 DTE
                    exps = sorted(list(set([c['expiration'] for c in chain])))
                    best_exp = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - target_exp_dt).days))
                    effective_calendar_horizon = max(1, (datetime.strptime(best_exp, "%Y-%m-%d") - d).days)
                    exp_chain = [c for c in chain if c['expiration'] == best_exp]
                    
                    # Compute deltas for this chain
                    dte_years = (datetime.strptime(best_exp, "%Y-%m-%d") - d).days / 365.0
                    strikes = [c['strike'] for c in exp_chain]
                    prices = [c['close'] for c in exp_chain]
                    # Filter out zero prices
                    valid_indices = [idx for idx, p in enumerate(prices) if p > 0]
                    if valid_indices:
                        v_strikes = [strikes[idx] for idx in valid_indices]
                        v_prices = [prices[idx] for idx in valid_indices]
                        deltas = compute_chain_deltas(
                            spot,
                            v_strikes,
                            v_prices,
                            dte_years,
                            risk_free_rate=0.04,
                            dividend_yield=0.0,
                            option_type="put",
                        )
                        
                        # Find strike closest to -0.15 delta
                        best_idx = min(range(len(deltas)), key=lambda idx: abs(deltas[idx][1] - (-0.15)))
                        strike_price = deltas[best_idx][0]
                        delta_prob = abs(deltas[best_idx][1])
                    else:
                        vol = df.loc[:d, 'Log_Return'].tail(30).std() * np.sqrt(252)
                        strike_price = spot * (1 - 1.04 * vol * np.sqrt(horizon/365))

                effective_trading_horizon = calendar_days_to_trading_days(effective_calendar_horizon)
                if effective_trading_horizon not in future_return_cache:
                    col = f"future_ret_{effective_trading_horizon}"
                    df[col] = df['SPY_Close'].shift(-effective_trading_horizon) / df['SPY_Close'] - 1
                    future_return_cache[effective_trading_horizon] = col
                future_col = future_return_cache[effective_trading_horizon]
                resolved_cutoff_pos = current_pos - effective_trading_horizon
                if resolved_cutoff_pos < 0:
                    continue

                resolved_index = df.index[:resolved_cutoff_pos + 1]
                dist_to_strike = (strike_price / spot) - 1

                # B. Historical Baseline Prob (resolved sample only)
                hist_subset = df.loc[resolved_index].dropna(subset=[future_col])
                all_rets = hist_subset[future_col].values
                dist_to_strike = (strike_price / spot) - 1
                prob_hist = np.sum(all_rets <= dist_to_strike) / len(all_rets) if len(all_rets) > 0 else 0
                
                # C. Regime Conditioned Prob
                eligible_feature_df = feature_df.loc[feature_df.index.intersection(resolved_index)]
                mask = eligible_feature_df['HMM_State'] == current_state
                matching_dates = mask[mask].index
                regime_rets = df.loc[matching_dates, future_col].dropna().values
                
                prob_regime = np.sum(regime_rets <= dist_to_strike) / len(regime_rets) if len(regime_rets) > 0 else 0
                
                # D. Terminal Outcome (Future data)
                terminal_pos = current_pos + effective_trading_horizon
                if terminal_pos < len(df):
                    s_terminal = df['SPY_Close'].iloc[terminal_pos]
                    outcome_price_dist = s_terminal - strike_price
                else:
                    outcome_price_dist = np.nan
    
                results.append({
                    'date': d.strftime("%Y-%m-%d"),
                    'spot': spot,
                    'strike': strike_price,
                    'otm_pct': (spot - strike_price) / spot if strike_price else 0,
                    'delta_prob': float(delta_prob),
                    'hist_prob': float(prob_hist),
                    'regime_prob': float(prob_regime),
                    'outcome_dist': float(outcome_price_dist) if not np.isnan(outcome_price_dist) else None,
                    'regime_state': int(current_state),
                    'regime_label': feature_df.loc[d, 'Regime_Label'] if 'Regime_Label' in feature_df.columns else regime_labels.get(current_state, f"State {current_state}"),
                    'effective_calendar_horizon': int(effective_calendar_horizon),
                    'effective_trading_horizon': int(effective_trading_horizon),
                })
                
                if i % 20 == 0:
                    print(f"  [{i}/{len(audit_dates)}] {d_str} | Delta: {delta_prob:.2f} | Hist: {prob_hist:.2f} | Regime: {prob_regime:.2f}")

        _save_json_cache(RESULTS_CACHE_PATH, cache_metadata, results)
        for r in results:
            r['date'] = pd.to_datetime(r['date'])

    # 5. Visualization
    audit_df = pd.DataFrame(results).set_index('date')
    
    # Use integer index for x-axis to remove gaps from weekends/holidays
    x = np.arange(len(audit_df))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # --- SUBPLOT 1: Probabilities ---
    ax1.plot(x, audit_df['delta_prob'], label='Option Delta (15%)', color='black', linestyle='--', alpha=0.6)
    ax1.plot(x, audit_df['hist_prob'], label='Historical Baseline (2019-Present)', color='blue', alpha=0.8)
    ax1.plot(x, audit_df['regime_prob'], label='Regime-Conditioned Prob', color='red', linewidth=2)
    
    # Highlight regions where Regime Prob < Delta Prob (Potential edge)
    ax1.fill_between(x, audit_df['regime_prob'], audit_df['delta_prob'], 
                     where=(audit_df['regime_prob'] < audit_df['delta_prob']), 
                     color='green', alpha=0.1, label='Regime Advantage')
    
    ax1.set_ylabel('Probability of Assignment')
    ax1.set_title('Probability Audit: Delta vs History vs Regime (SPY 42 DTE)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.4)
    
    # 2nd Y-Axis for OTM %
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, audit_df['otm_pct'] * 100, color='purple', label='Strike OTM %', alpha=0.4, linewidth=1.5)
    ax1_twin.set_ylabel('Strike OTM % (Distance from Spot)')
    ax1_twin.set_ylim(0, 10) # 0% to 10% OTM
    ax1_twin.legend(loc='upper right')
    
    # --- SUBPLOT 2: Terminal Outcome ---
    ax2.fill_between(x, 0, audit_df['outcome_dist'], 
                     where=(audit_df['outcome_dist'] >= 0), color='green', alpha=0.3, label='Safe (OTM)')
    ax2.fill_between(x, 0, audit_df['outcome_dist'], 
                     where=(audit_df['outcome_dist'] < 0), color='red', alpha=0.3, label='Assigned (ITM)')
    
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('Terminal Price - Strike ($)')
    ax2.set_title('Realized Outcome (42 Days Later)')
    ax2.legend(loc='upper left')
    
    # SPY Price on twin axis for context
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, audit_df['spot'], color='gray', alpha=0.3, label='SPY Spot')
    ax2_twin.set_ylabel('SPY Price')
    
    # --- SUBPLOT 3: Market Regime ---
    # Map regime states to numeric for plotting
    states = audit_df['regime_state'].values
    ax3.step(x, states, where='post', color='darkorange', linewidth=2, label='Detected Regime')
    
    # Fill background by regime
    unique_regimes = sorted(audit_df['regime_state'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
    for idx, r in enumerate(unique_regimes):
        label = audit_df[audit_df['regime_state'] == r]['regime_label'].iloc[0]
        ax3.fill_between(x, -0.5, len(unique_regimes)-0.5, where=(audit_df['regime_state'] == r), 
                         color=colors[idx], alpha=0.1, label=label)
    
    ax3.set_ylabel('Regime ID')
    ax3.set_yticks(unique_regimes)
    ax3.set_yticklabels([audit_df[audit_df['regime_state'] == r]['regime_label'].iloc[0] for r in unique_regimes])
    ax3.set_title('Market Regime Detection (Walk-Forward HMM)')
    ax3.grid(True, alpha=0.2)
    # ax3.legend(loc='upper left', ncol=2, fontsize='small') # Legend can get crowded
    
    # Formatting X-axis with dates (sampling to avoid clutter)
    n_ticks = 12
    tick_indices = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(audit_df.index[tick_indices].strftime('%Y-%m-%d'), rotation=30)
    
    plt.tight_layout()
    output_png = 'research_reports/regime_probability_audit_2025_2026.png'
    os.makedirs('research_reports', exist_ok=True)
    plt.savefig(output_png, dpi=150)
    print(f"\n✓ Audit complete. Plot saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    asyncio.run(generate_regime_probability_audit())
