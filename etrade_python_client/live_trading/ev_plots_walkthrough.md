# `ev_plots.py` Walkthrough

This document explains how `live_trading/ev_plots.py` and its core math engine `live_trading/ev_engine.py` work, from data collection to GMM probability modeling to final EV charts.

## 1. What the script does

`ev_plots.py` compares SPY and SPX put-credit-spread opportunities by:

1. **Fetching Market Data**: Gets live prices (SPY, SPX, VIX) and 15 years of daily history via `yfinance`. Now includes **VVIX** for state detection.
2. **HMM Regime Detection**: Uses a Continuous **Hidden Markov Model (HMM)** to dynamically detect hidden market states based on Log Returns, **EWMA Realized Volatility**, VIX, and VVIX.
3. **GMM Probability Projection**: Uses a 2-component Gaussian Mixture Model (GMM) *within* each HMM state to estimate multi-week breach probabilities ($P(spot \le strike)$).
4. **Forward Projection**: Projects today's state probabilities forward to option expiration using **Matrix Exponentiation** of the HMM transition matrix.
5. **EV & Risk Integration**: Numerically integrates the payout function over the projected probability mixture to find Expected Value (EV), Expected Shortfall (ES), and Loss Probability.
6. **Portfolio Normalization**: Scales metrics to a consistent $10,000 margin budget.
7. **Liquidity Analysis**: Tracks slippage relative to net credit and volume/OI across deltas.

It supports multiple diagnostic modes:

- `--distributions`: Plots historical return distributions by VIX regime with normality tests.
- `--gmm-plots`: Visualizes internal GMM clustering of historical returns.
- `--gmm-dist`: Overlays GMM density fits on empirical return histograms (with AIC/BIC metrics).
- `--calibrate`: Runs a temporal walk-forward backtest to validate probability accuracy.
- `--samples`: Performs spot-checks on specific historical dates to see model predictions vs outcomes.

## 2. Core Math Engine (`ev_engine.py`)

To ensure reliability and speed, the heavy lifting is moved to `ev_engine.py`:

- **Data Persistence**: Caches raw yfinance data (`s_and_p_data/spy_vix_historical_raw.csv`) including SPY, VIX, and VVIX.
- **Feature Engineering**: Implements 10-day **EWMA Volatility** to eliminate the "ghosting" lag of standard rolling volatility windows.
- **HMM Training**: Dynamically selects the optimal number of hidden states ($K$) using **AIC/BIC scores** to prevent overfitting.
- **Regime Bucketing**: Buckets $T$-horizon returns by the starting HMM state rather than static VIX levels.
- **GMM Implementation**: Logic for fitting 2-component mixtures to multi-week horizon data (capturing fat tails) and querying CDFs is centralized here.

## 3. Global configuration

Key globals near the top of the file:

- `PROBABILITY_MODEL = 'gmm'`
  - Options: `'bootstrap'`, `'parametric'`, `'gmm'`
- `USE_MARKOV_TRANSITIONS = True`
  - If `True`, probabilities are blended across projected future VIX regimes instead of using only the current regime.
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

- Standardizes features: `[Log_Return, EWMA_Vol_10d, VIX, VVIX]`.
- Iterates through $K=1 \to 6$ states.
- Selects the model with the minimum **BIC** to strike a balance between granularity and noise filtering.

### `build_regime_return_arrays(cache_key, horizon=45)`

- Computes forward-looking returns for the given horizon: `df['SPY_Close'].shift(-horizon) / df['SPY_Close'] - 1`.
- Buckets returns by the **discovered HMM State** (e.g., State 0, State 1, ... State $K-1$).
- Captures the empirical return behavior associated with each machine-learned regime.

## 4. Markov transition modeling

### HMM Transitions

Instead of a custom build, the engine uses the **Transition Matrix** (`transmat_`) directly from the trained HMM model.

- Each `transmat_[i, j]` represents the daily probability of shifting from HMM State $i$ to State $j$.

### `get_probability_engine(...)`

1. **Posterior Probability**: Today's features are passed to `hmm_model.predict_proba()` to find the probability distribution of current states (e.g., 80% State 0, 20% State 1).
2. **Matrix Exponentiation**: The Transition Matrix is raised to the power of the `horizon` (days to expiration) using `np.linalg.matrix_power`.
3. **Future Projection**: Today's state vector is multiplied by the projected matrix to yield the expected regime distribution AT expiration.
4. **Mixture Summation**: The final `prob_func` is a weighted sum of the GMMs for each state, using the projected weights.

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
   - Fits 2-component `GaussianMixture` to the multi-week horizon data.
   - **No scaling required**: The data itself already represents the total duration (e.g. 43 days). This avoids the 'persistence error' where crash-day means were previously scaled linearly.
   - Falls back to 1-component Gaussian if sample size is too small or fit fails.
2. `query_gmm(cached_params, spot_price, strike_price)`
   - Computes weighted Gaussian CDF mixture quickly.

This design captures the real-world Non-Normality (fat tails, skew) observed in 15 years of trade durations.

## 6. Building the final probability function

### `get_probability_engine(spot_price, current_vix, regime_dict, horizon=45, hmm_model=None)`

Returns:

- `prob_func`: The final weighted-mixture probability function.
- `regime_name`: The label of the currently dominant state.
- `projected_weights`: The estimated regime probabilities at expiration.

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
- Visualizes how the 2-component GMM finds "hidden" sub-regimes within the HMM-learned states, identifying internal multi-modality.

### C) GMM Distribution Fit (`--gmm-dist`)
- Overlays the final GMM density mixture on histograms for each HMM state.
- Displays **Log-Likelihood**, **AIC**, and **BIC** to justify the use of Gaussian Mixtures over simple Parametric models.

### D) Model Calibration (`--calibrate`)
- A rigorous **Out-Of-Sample (OOS) Walk-Forward Backtest**.
- **Refit Mechanism**: Every 21 trading days (approx. 1 month), the HMM is retrained from scratch using only data *strictly prior* to the validation date.
- **Statistical Separation**: Applies the **Kruskal-Wallis** H-test to verify that the predicted HMM regimes legitimately correlate with statistically distinct future return outcomes out-of-sample.
- Prints overall **Brier Score**, **Log-Loss**, and **ECE** (Expected Calibration Error).

### E) Prediction Samples (`--samples`)
- Performs spot-checks on specific historical dates (e.g., 2020 peak, 2022 bear) to see model predictions vs realized outcomes without look-forward bias.

## 12. CLI Usage Examples

### Diagnostic Suite (No Login Needed)
```bash
# View GMM clustering internals
python live_trading/ev_plots.py --gmm-plots

# View GMM density fits with AIC/BIC
python live_trading/ev_plots.py --gmm-dist

# Check historical calibration
python live_trading/ev_plots.py --calibrate
```

### Full Analysis (Requires Login)
```bash
python live_trading/ev_plots.py --username YOUR_USER --password YOUR_PASS
```

## 13. System Requirements

- `hmmlearn`: Required for `GaussianHMM` regime detection.
- `scikit-learn`: Required for `GaussianMixture` (internal state distributions).
- `yfinance`: For daily historical data (SPY, VIX, VVIX) and price fallback.
- `scipy`: For statistical tests (Kruskal-Wallis, Shapiro-Wilk) and GMM mixture CDFs.
