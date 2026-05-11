# Market Regime Detection Audit (2019 - 2025)

## 1. Algorithmic Walkthrough & Methodology

The Market Regime Detector is a sophisticated unsupervised machine learning engine designed to dynamically segment market environments (e.g., bull markets, crashes, sideways chop) without relying on rigid, hardcoded thresholds (like "VIX > 20 means panic"). It organically clusters market geometries based on their underlying risk/return profiles.

### Data Ingestion & Stationarity
1. **Multi-Asset Sourcing**: The engine pulls daily pricing and macroeconomic data, including SPY, VIX, Treasury Yields, High Yield Spreads, and optionally Bitcoin and Crude Oil.
2. **Stationarity via Fractional Differencing**: Financial time series are often non-stationary, which mathematically breaks Markov models. The engine uses the Augmented Dickey-Fuller (ADF) test to dynamically apply **fractional differencing**. This removes non-stationarity while preserving the "memory" of the price series far better than standard first-order differencing (which completely erases the baseline levels).

### Causal Feature Extraction (PCA)
To handle highly correlated inputs (like SPY returns and High Yield Spreads), the model uses **Sparse Principal Component Analysis (SPCA)** to fuse the features into orthogonal dimensions (PC1 and PC2).
- **Rolling Robust Scaling**: The data is scaled using an expanding 5-year window median/IQR to ensure that extreme outliers (like the COVID crash) do not distort the entire dataset globally. This is calculated with a strict look-behind window.

### Gaussian Mixture Hidden Markov Model (GMM-HMM)
1. **The Core Engine**: The engine assumes the market is driven by an unobservable (hidden) Markov process that transitions between $K$ discrete states. In each state, the PCA features (emissions) are modeled as a Gaussian Mixture distribution. It uses the Viterbi/Forward-Backward algorithm to decode the most likely probability sequence of hidden states.
2. **Causal Walk-Forward Refit**: The model is strictly causal. It is refitted every 21 days on past data only (expanding window). This eliminates "look-ahead bias" where the parameters of the 2020 crash would otherwise be accidentally known in 2019.
3. **Symmetric KL-Divergence Alignment**: During refitting, the engine mathematically maps the newly learned states to the previous states using Kullback-Leibler (KL) divergence. This anchors the states and prevents them from randomly swapping IDs across refit boundaries (e.g., State 0 suddenly becoming State 2).

### Adaptive Archetype Ranking
Instead of static rules, the model ranks the physical centroids of each state relative to one another to create semantic meaning. The state with the highest intrinsic VIX centroid is labeled **Market Turmoil**, while the state with the highest Sharpe Ratio is labeled **Robust Expansion**.

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
