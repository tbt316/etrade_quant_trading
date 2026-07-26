# Market Regime Detection Audit (2019 - 2026)

## 1. Algorithmic Walkthrough & Methodology

The Market Regime Detector is a sophisticated unsupervised machine learning engine designed to dynamically segment market environments (e.g., bull markets, crashes, sideways chop) without relying on rigid, hardcoded thresholds (like "VIX > 20 means panic"). It organically clusters market geometries based on their underlying risk/return profiles.

### Data Ingestion & Stationarity
1. **Multi-Asset Sourcing**: The engine pulls daily pricing and macroeconomic data, including SPY, VIX, Treasury Yields, High Yield Spreads, and optionally Bitcoin and Crude Oil.
2. **Stationarity via Fractional Differencing**: Financial time series are often non-stationary, which mathematically breaks Markov models. The engine uses the Augmented Dickey-Fuller (ADF) test to dynamically apply **fractional differencing**. This removes non-stationarity while preserving the "memory" of the price series far better than standard first-order differencing (which completely erases the baseline levels).

### Causal Feature Extraction (PCA)
To handle highly correlated inputs (like SPY returns and High Yield Spreads), the model uses **Sparse Principal Component Analysis (SPCA)** to fuse the features into orthogonal dimensions (PC1 and PC2).
- **Rolling Robust Scaling**: The data is scaled using an expanding 5-year window median/IQR to ensure that extreme outliers (like the COVID crash) do not distort the entire dataset globally. This is calculated with a strict look-behind window.

### Gaussian Mixture Hidden Markov Model (GMM-HMM)
1. **The Core Engine**: The engine assumes the market is driven by an unobservable hidden Markov process that transitions between $K$ discrete states. In the current diagnostic artifact, the walk-forward engine stores filtered point-in-time state probabilities and keeps raw HMM labels separate from the final actionable overlay. Historical backtests must not consume full-sample Viterbi-smoothed paths.
2. **Causal Walk-Forward Refit**: The model is strictly causal. It is refitted every 21 days on past data only (expanding window). This eliminates "look-ahead bias" where the parameters of the 2020 crash would otherwise be accidentally known in 2019.
3. **Symmetric KL-Divergence Alignment**: During refitting, the engine mathematically maps the newly learned states to the previous states using Kullback-Leibler (KL) divergence. This anchors the states and prevents them from randomly swapping IDs across refit boundaries (e.g., State 0 suddenly becoming State 2).

### Adaptive Archetype Ranking
Instead of static rules, the model ranks the physical centroids of each state relative to one another to create semantic meaning. The state with the highest intrinsic VIX centroid is labeled **Market Turmoil**, while the state with the highest Sharpe Ratio is labeled **Robust Expansion**.

### Actionable Overlay
Raw HMM archetype labels are diagnostic. The final trading-facing taxonomy is produced by an overlay that requires observable market stress evidence from close-T SPY/VIX data before mapping a raw high-volatility archetype into `Panic / Crisis`. The signal timestamp in the latest diagnostics is `close_T_for_next_session`, so intraday backtests must consume the prior trading day's regime.

---

## 2. Model Comparison: Full Data vs. Ex-Crypto/Commodities

We ran the engine from 2019 to 2025 under two configurations:
1. **Full Model**: Includes SPY, VIX, Macro Yields, Bitcoin (BTC), and WTI Crude Oil.
2. **Partial Model**: Strips out Bitcoin and Crude Oil.

### Full Model (Including BTC & Crude)
![Full Model Analysis](./regime_full.png)

### Partial Model (Excluding BTC & Crude)
![Partial Model Analysis](./regime_partial.png)

---

## 3. Findings & Reality Check

1. **Noise Reduction and Precision**: The Partial Model (excluding BTC and Oil) generally produces much more stable and logical regime transitions for an equity-focused strategy. Bitcoin and Oil have distinct, idiosyncratic volatility cycles that frequently desynchronize from the broader US equity market. Including them forces the PCA to allocate variance to these assets, leading to "false alarms" or choppy regime flipping in the SPY timeline. Removing them creates a much cleaner macro signal.
2. **Does it match reality?**: The regime detector makes profound sense when compared to historical reality:
   - **Q1 2020 (COVID-19)**: The model rapidly spikes into "Market Turmoil" exactly during the liquidity crash in February/March 2020. Because it is causal, it detects the shift in real-time as the geometry changes, accurately capturing the regime change without knowing the future.
   - **2021 (Hyper-Expansion)**: The model correctly locks into "Robust Expansion" and "Emerging Expansion" during the post-COVID quantitative easing run.
   - **2022 (Bear Market)**: Accurately identifies a sustained "Cautious Decline" and "High Vol Chop" as the Federal Reserve aggressively raised interest rates.
   - **2023-2024 (AI Bull Run)**: Transitions back to stable Expansion regimes, demonstrating that the rolling scaler successfully adapted to the new higher-rate environment without falsely flagging it as turmoil.

---

## 4. Latest Diagnostic Result (Generated 2026-05-24)

Latest artifacts:
- `research_reports/regime_diagnostics/market_regime_diagnostics.html`
- `research_reports/regime_diagnostics/causal_regime_trace.csv`
- `research_reports/regime_diagnostics/spy_log_return_gmm_summary.csv`
- `audit_plots/regime_probability_audit.html`
- `audit_plots/option_assignment_summary.csv`

### Regime Trace
The latest regime diagnostics were generated on 2026-05-24 17:54 with fetch window 2015-01-01 to 2026-05-24 and analysis window 2020-01-01 to 2026-05-24. The saved causal trace spans 2020-01-03 to 2026-05-22.

Final detected-state counts in the trace:
- `Expansion (0)`: 981 days
- `Cautious Decline (1)`: 219 days
- `Panic / Crisis (2)`: 60 days

Latest available trace row:
- Date: 2026-05-22
- SPY close: 745.64
- VIX close: 16.70
- SPY log return: +0.3924%
- Latest HMM refit date: 2026-05-14
- Raw HMM label: `Market Turmoil (2)` with raw posterior concentrated in state 2
- Final actionable overlay: `Expansion (0)` because no stress overlay was triggered
- Signal timestamp: `close_T_for_next_session`

### Return Distribution Fit
The state-conditioned SPY daily log-return GMM audit selected a single Gaussian component for every final regime by BIC:
- `Expansion (0)`: n=981, mean daily log return +0.1341%, std 0.8310%
- `Cautious Decline (1)`: n=219, mean daily log return -0.1172%, std 1.6930%
- `Panic / Crisis (2)`: n=60, mean daily log return -0.7534%, std 3.4677%

Interpretation: the final overlay produced economically sensible ordering across regimes, but the daily-return GMM did not justify multiple components after BIC penalty in this artifact.

### SPY vs SPX Probability Audit
The probability audit was generated on 2026-05-24 17:56. The daily SPY/SPX trace spans 2016-03-02 to 2026-05-21, and the resolved 42-DTE option-assignment outcomes span 2018-02-27 to 2026-04-09.

Key results:
- SPY and SPX daily log returns are effectively equivalent for this audit: overall correlation 0.99727 and KS p-value 0.9994.
- For 42-DTE skew-adjusted short-put assignment, the overall realized assignment frequency was below the target delta buckets.
- At target delta -0.15, realized skew-adjusted assignment was 6.18% for SPY and 6.23% for SPX, versus a 15% target, implying positive probability edge of about +8.82% for SPY and +8.77% for SPX in this sample.
- At target delta -0.20, realized skew-adjusted assignment was 9.75% for both SPY and SPX, versus a 20% target.

## 5. Current Limitations / Open Issues

1. **Raw HMM state is not an absolute trading signal**: On the latest row, the raw HMM label is `Market Turmoil (2)` while the final overlay is `Expansion (0)`. This confirms the raw relative HMM archetype can drift or overstate stress in a calm absolute-volatility environment. Downstream trading logic should consume the final overlay, not raw state IDs.
2. **Semantic drift is present but bounded in the latest trace**: `HMM_Semantic_Drift_Flag` is true on 12 of 1260 diagnostic rows. This does not invalidate the run, but it prevents treating raw HMM state identity as stable without the overlay and diagnostics.
3. **Crisis sample size is small**: The final `Panic / Crisis` return bucket has only 60 daily observations, and the option-assignment summary has only 29 resolved Market Turmoil outcomes. Crisis-regime assignment edge estimates should be treated as underpowered.
4. **The audit HTML headline and CSV disagree for crisis assignment**: `audit_plots/regime_probability_audit.html` includes a headline claiming -6.85% crisis edge at -0.15 delta, while `audit_plots/option_assignment_summary.csv` shows zero realized skew-adjusted assignments in 29 Market Turmoil rows. The CSV should be treated as the current tabular source of truth until the HTML card generation is audited.
5. **Backtest causality still requires explicit checklist verification**: For any strategy report using these regimes, document the calibration/refit schedule, test range, inference method, one-session regime lag for intraday entries, and return-bucket censoring. The latest diagnostic artifacts satisfy point-in-time trace generation, but they are not by themselves proof that every downstream backtest consumed the regime map causally.
