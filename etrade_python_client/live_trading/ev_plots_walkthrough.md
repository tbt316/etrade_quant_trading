# `ev_plots.py` Walkthrough

This document explains how `live_trading/ev_plots.py` and its core math engine `live_trading/ev_engine.py` work, from data collection to GMM probability modeling to final EV charts.

## 1. What the script does

`ev_plots.py` compares SPY and SPX put-credit-spread opportunities by:

1. **Fetching Market Data**: Gets live SPY/SPX/VIX prices and a historical
   multi-source feature panel, then models only exact NYSE sessions.
2. **HMM Regime Detection**: Uses a Gaussian **Hidden Markov Model (HMM)** on
   prefix-causal scaled/PCA features. VIX close is excluded from HMM emissions
   and retained for the separately named stress overlay.
3. **GMM Probability Projection**: Uses a BIC-selected Gaussian Mixture Model
   within each exact raw-HMM taxonomy to estimate multi-week breach
   probabilities ($P(spot \le strike)$).
4. **Optional Forward Projection**: When explicitly enabled, projects the
   causal tail posterior to expiration using the validated HMM transition
   matrix. It is disabled by default.
5. **EV & Risk Integration**: Numerically integrates the payout function over the projected probability mixture to find Expected Value (EV), Expected Shortfall (ES), and Loss Probability.
6. **Portfolio Normalization**: Scales metrics to a consistent $10,000 margin budget.
7. **Liquidity Analysis**: Tracks slippage relative to net credit and volume/OI across deltas.

It supports these research diagnostic modes:

- `--distributions`: Plots historical return distributions by VIX regime with normality tests.
- `--gmm-dist`: Overlays GMM density fits on empirical return histograms (with AIC/BIC metrics).
- `--timeline`: Plots an explicitly calibrated causal raw-HMM trace and a
  separately labeled final stress overlay.
- `--regime-log-return-gmm`: Fits research return models to a causal raw-HMM
  trace.

`--gmm-plots`, `--calibrate`, and `--samples` are intentionally disabled with
stable error codes. Their legacy forward-outcome paths do not satisfy the
typed exact-as-of return-bucket protocol.

## 2. Core Math Engine (`ev_engine.py`)

To ensure reliability and speed, the heavy lifting is moved to `ev_engine.py`:

- **Data Persistence**: Caches raw yfinance data (`s_and_p_data/spy_vix_historical_raw.csv`) including SPY, VIX, and VVIX.
- **Feature Engineering**: Applies prefix-fitted stationarity decisions,
  trailing volatility filters, causal scaling, and prefix-fitted PCA.
- **HMM Training**: Dynamically selects the optimal number of hidden states ($K$) using **AIC/BIC scores** to prevent overfitting.
- **Regime Bucketing**: Buckets only strictly resolved $T$-horizon returns by
  an exact raw-HMM taxonomy, never by a final stress-overlay integer.
- **GMM Implementation**: Logic for fitting BIC-selected mixtures to multi-week horizon data (capturing fat tails only when supported by the data) and querying CDFs is centralized here.

## 3. Global configuration

Key globals near the top of the file:

- `PROBABILITY_MODEL = 'gmm'`
  - Options: `'bootstrap'`, `'parametric'`, `'gmm'`
- `USE_MARKOV_TRANSITIONS = False`
  - The current default uses the causal tail posterior. If explicitly enabled,
    a validated raw-HMM transition matrix projects that posterior.
- `TARGET_MARGIN_DOLLARS = 10000.0`
  - Used to normalize portfolio EV/ES across different spread widths.
- `COST_PER_SPREAD = 1.0`
  - Flat transaction cost used in payout math.

## 3. Historical data and regime setup

### `fetch_historical_data()`

- Downloads ~15 years of SPY, VIX, and **VVIX** from `yfinance`.
- Caches raw data to `s_and_p_data/spy_vix_historical_raw.csv`.
- Reuses cache if it is fresher than 24 hours.

### `train_regime_hmm(df)`

- Prepares stationary features through a causal feature manifest and a
  prefix-fitted scaler/PCA pipeline.
- Walk-forward plot traces require an explicit `fit_end` strictly before the
  first displayed or evaluated session.
- The returned fitted model carries an exact raw-HMM taxonomy ID. Numeric HMM
  states are meaningful only with that ID. Walk-forward output records the
  exact taxonomy per row because each refit creates a new fitted identity.

### `build_regime_return_arrays(cache_key, horizon=45, as_of_date=...)`

- Converts the calendar-day option horizon to a trading-session horizon.
- Requires an explicit date-only `as_of_date`; there is no wall-clock default.
- Records the exact terminal session for every forward return and includes it
  only when that session is strictly before `as_of_date`.
- Returns `RegimeReturnBuckets`, not a plain dictionary. The bucket values,
  horizon, resolution cutoff, model cutoff, and exact fitted-HMM taxonomy are
  bound into the cache contract.
- Rejects legacy/unbound cache content instead of silently reusing it.

## 4. Markov transition modeling

### HMM Transitions

Instead of a custom build, the engine uses the **Transition Matrix** (`transmat_`) directly from the trained HMM model.

- Each `transmat_[i, j]` represents the daily probability of shifting from HMM State $i$ to State $j$.

### `get_probability_engine(...)`

1. **Contract validation**: Requires typed return buckets whose taxonomy,
   training cutoff, state count, and horizon exactly match the HMM.
2. **Posterior Probability**: Uses the causal tail probabilities stored by the
   model pipeline.
3. **Matrix Exponentiation**: The Transition Matrix is raised to the power of the `horizon` (days to expiration) using `np.linalg.matrix_power`.
4. **Future Projection**: Today's state vector is multiplied by the projected matrix to yield the expected regime distribution AT expiration.
5. **Mixture Summation**: The final `prob_func` is a weighted sum of the GMMs for each state, using the projected weights.

## 5. Probability engines

The script needs a function `prob_func(strike)` that returns:

- `P(spot <= strike)` at expiration horizon.

### A) Bootstrap model

Inside `_build_single_regime_prob_func(...)`:

- Resamples daily returns from selected regime.
- Simulates paths for `days`.
- Uses minimum path price for breach probability proxy.

### B) Parametric model (`Student-t`)

- Fits Student-t to log returns: `np.log1p(bucket)`.
- Scales location by `days`, scale by `sqrt(days)`.
- Uses `stats.t.cdf(...)`.

### C) GMM model

Two-step implementation for speed:

1. `fit_gmm(bucket_returns, regime_label="")`
   - Fits `GaussianMixture` candidates, including `K=1`, to the multi-week horizon data and selects by BIC.
   - **No scaling required**: The data itself already represents the total duration (e.g. 43 days). This avoids the 'persistence error' where crash-day means were previously scaled linearly.
   - Falls back to 1-component Gaussian if sample size is too small or fit fails.
2. `query_gmm(cached_params, spot_price, strike_price)`
   - Computes weighted Gaussian CDF mixture quickly.

This design captures the real-world Non-Normality (fat tails, skew) observed in 15 years of trade durations.

## 6. Building the final probability function

### `get_probability_engine(spot_price, current_vix, regime_buckets, horizon=45, hmm_model=None)`

Returns one `ProbabilityEngineResult` containing:

- `probability(strike)`: The final weighted-mixture probability function.
- the dominant raw HMM state and current/projected probabilities;
- the exact taxonomy ID and resolved-through date; and
- permanent `UNVERIFIED` / `execution_eligible=false` status.

Raw `dict[str, array]` buckets are not accepted. Final stress-overlay integers
also cannot index raw HMM return buckets.

In **Markov-Switching** mode, the engine dynamically adjusts the return distribution's "tails" as expiration time increases, naturally reflecting the increasing risk of transitioning into a high-volatility crash state over longer trade durations.

## 8. Expected Value & Risk Calculation

### `calculate_yield_metrics(...)`

For a candidate spread (`short_strike`, `long_strike`, `net_credit_per_share`):

- **Payout Function**: Defines contract P&L for any given spot price at expiration.
- **Numerical Integration**: Iterates through strike buckets (and tails) to calculate the weighted average P&L.
- **Metrics**:
  - `net_yield`: Max possible profit if the trade is successful (after flat costs).
  - `expected_shortfall` (ES): The average loss in scenarios where the short strike is breached.
  - `ev_per_contract`: Final statistical expectation.
  - `prob_of_loss`: Probability of the trade ending with a negative P&L.

## 9. Main Runtime Flow (`main(args)`)

1. **Authentication**: OAuth login to E*TRADE.
2. **Price Discovery**: Fetches live SPY/SPX/VIX. Uses `fetch_cached_yf_close` as a fallback for VIX to prevent script failure during API hiccups.
3. **Engine Initialization**:
   - Builds rolling-horizon return buckets (e.g., if expiration is 43 days away, it fits models to 43-day historical windows).
   - Projects regime weights using the Markov transition matrix.
4. **Option Chain Sweep**:
   - Fetches Put chains for SPY and SPX.
   - Sets baseline short leg at $\Delta \approx -0.15$.
   - Sweeps long-leg deltas from $-0.14 \to -0.01$.
5. **Visualization**: Generates a 4-panel visual dashboard.

## 10. Visualization Dashboard

The script produces a high-fidelity plot with 4 panels:

1. **Portfolio EV**: Total Expected Value normalized to a fixed $10,000 margin budget.
2. **EV Efficiency**: The ratio of `EV / |ES|`, representing risk-adjusted return expectation.
3. **Premiums**: Dual-axis plot of short and long leg premiums for both SPY and SPX.
4. **Liquidity Fingerprint**: Relative slippage (bid-ask spread as % of net credit) overlaid with volume/OI bars.

Saved to: `/Users/btian/.gemini/antigravity/artifacts/ev_delta_normalized_margin_plot.png`

## 11. Diagnostic Modes

### A) Regime Distributions (`--distributions`)
- Plots return histograms grouped by HMM State.
- Runs **Shapiro-Wilk** and **Jarque-Bera** tests on each regime's multi-week returns.
- Typically shows that specific HMM states are significantly more non-Normal than others.

### B) GMM Clustering (`--gmm-plots`)
- Disabled with `UNSAFE_REGIME_GMM_CLUSTER_DIAGNOSTIC_DISABLED` until rebuilt
  on exact-as-of typed return buckets.

### C) GMM Distribution Fit (`--gmm-dist`)
- Overlays the final GMM density mixture on histograms for each HMM state.
- Displays **Log-Likelihood**, **AIC**, and **BIC** to justify the use of Gaussian Mixtures over simple Parametric models.

### D) Model Calibration (`--calibrate`)
- Disabled with `UNSAFE_REGIME_CALIBRATION_BACKTEST_DISABLED`. Use the typed
  backtest protocol and prospective Regime V2 calibration workflow instead.

### E) Prediction Samples (`--samples`)
- Disabled with `UNSAFE_REGIME_SAMPLE_OUTCOMES_DISABLED` until migrated to
  completed-NYSE-session snapshots and strictly resolved outcomes.

## 12. CLI Usage Examples

### Diagnostic Suite (No Login Needed)
```bash
# View GMM density fits with AIC/BIC
python live_trading/ev_plots.py --gmm-dist

# View a causal raw-HMM timeline with separate stress overlays
python live_trading/ev_plots.py --timeline
```

### Full Analysis (Requires Login)
```bash
python live_trading/ev_plots.py --username YOUR_USER --password YOUR_PASS
```

The live analysis path treats the raw HMM label as display-only. It cannot
veto or resize an order. Missing VIX fails closed, and risk-gate overrides stay
disabled until a durable time-scoped operator-override ledger exists.

## 13. System Requirements

- `hmmlearn`: Required for `GaussianHMM` regime detection.
- `scikit-learn`: Required for `GaussianMixture` (internal state distributions).
- `yfinance`: For daily historical data and the bounded VIX fallback; if both
  configured VIX sources fail, the live analysis stops.
- `scipy`: For statistical tests (Kruskal-Wallis, Shapiro-Wilk) and GMM mixture CDFs.
