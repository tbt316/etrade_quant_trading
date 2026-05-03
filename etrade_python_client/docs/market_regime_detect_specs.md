# Market Regime Detection Specification

## 1. Core Philosophy: Absolute Non-Anticipativity
The primary goal of the Market Regime Detection (MRD) engine is to provide causal, real-time identification of market archetypes without **look-ahead bias**. In quantitative finance, data leakage from the future into the past (e.g., global scaling, Viterbi decoding over a full sample) results in "beautiful" backtests that fail in live production. This specification mandates a strictly causal pipeline.

---

## 2. Data Pipeline & Stationarity

### 2.1 Fractional Differencing
All macroeconomic and price-based features must be stationary to prevent variance pollution in the HMM. 
- **Regulation**: Features like SPY Price or VIX level must not be used raw. Use log-returns or fractionally differenced series where the order of integration $d$ is the minimum required to pass an ADF test while preserving memory.
- **Reasoning**: HMMs assume the emission distribution is stable within a state. Non-stationary data causes the model to "chase" secular trends rather than detecting structural shifts.

### 2.2 Causal Scaling
- **Regulation**: Never use `StandardScaler` or `RobustScaler` from `sklearn` directly on the entire dataset before training or inference.
- **Implementation**: The agent **MUST** implement a custom `ExpandingRobustScaler` or use a strict rolling window. Calling `.fit()` or `.fit_transform()` on a DataFrame containing future data relative to the current evaluation step is a catastrophic failure. At time $t$, the features are scaled using ONLY the median/IQR calculated from $X[0 \dots t]$.
- **Reasoning**: Global scaling leaks future volatility into the past. If the 2020 VIX spike is in the dataset, global scaling suppresses the relative magnitude of the 2018 Volmageddon event, altering historical regime classification.

---

## 3. Dimensionality Reduction (PCA): Causal Orthogonalization

### 3.1 Expanding Window PCA
- **Regulation**: PCA must NEVER be fit on the global dataset. 
- **Implementation**: The agent must implement an expanding window PCA. The rotation matrix (loadings) used to transform features at time $t$ must only be derived from a PCA `.fit()` on data from $0$ to $t-1$.
- **Reasoning**: Global PCA leaks future correlation structures into past data. 

### 3.2 Strict Orthogonality Enforcement
- **Regulation**: HMMs with diagonal covariance matrices assume feature independence. `SparsePCA` does not guarantee mathematically orthogonal components. 
- **Implementation**: In `verify_pca`, if the correlation between any two principal components exceeds the 0.1 threshold, the script **MUST NOT** just log a warning. It must strictly throw an Exception or automatically fall back to standard `PCA` to enforce orthogonality. 
- **Reasoning**: Non-orthogonal inputs into an HMM will cause the covariance matrices to distort, leading to overlapping state emissions and erratic "flickering" between regimes.

---

## 4. Regime Inference: Filtering vs. Smoothing

### 4.1 Forbidding the Viterbi Algorithm in Backtesting
- **Regulation**: The agent is strictly forbidden from using `model.predict(X)` to generate historical regime traces for backtesting. 
- **Implementation**: `hmmlearn.predict()` utilizes Viterbi decoding, which is a *smoothing* algorithm that uses data from $t+1 \dots T$ to classify the state at $t$. The agent MUST use `model.predict_proba(X)` and implement a custom Forward Algorithm (filtering) pass to ensure the probability of being in State $K$ at time $t$ relies **only** on data up to time $t$. 
- **Reasoning**: This is the single most common cause of quant illusion. Viterbi will make the model look like it flawlessly predicted market crashes the day before they happened, because it "cheats" by looking at the crash data to label the preceding day.

### 4.2 Minimum State Sojourn Time (Debouncing)
- **Regulation**: HMMs are prone to rapid state-switching noise during transition phases. 
- **Implementation**: Implement a "debounce" mechanism. A regime shift signal is only valid if the filtering probability of the new state exceeds the baseline threshold (e.g., $> 0.70$) for a minimum of $N$ consecutive periods (e.g., 3 days). 
- **Reasoning**: In live trading, executing a portfolio rotation costs basis points. We cannot afford to rotate the portfolio on a 1-day regime flicker caused by a single macroeconomic data print.

---

## 5. HMM Engine: Stability & Mapping

### 5.1 Architecture
- **Model Type**: Gaussian Mixture Hidden Markov Model (`GMMHMM`) with diagonal covariance.
- **State Selection**: Use **Bayesian Information Criterion (BIC)** to dynamically select the number of states $K$ (typically 3 to 5).
- **Persistence Prior**: Initialize the transition matrix with high diagonal dominance (e.g., 0.95) and use a **Dirichlet Prior** to penalize high-frequency state switching.

### 5.2 State Identity Preservation (KL-Divergence)
During walk-forward refits (e.g., every 21 days), the HMM might re-order its states. "State 0" in January must mean the same thing as "State 0" in February.
- **Regulation**: Use **Symmetric KL-Divergence** to map new model states to the previous model's states.
- **Reasoning**: Prevents "identity drift" which would break downstream strategy logic tied to specific regime IDs.

### 5.3 Adaptive Archetype Ranking
- **Regulation**: Labels (e.g., "Market Turmoil") must be assigned by ranking states relative to each other in the current model.
- **Ranking Criteria**:
    1. **Market Turmoil**: State with highest median VIX.
    2. **Robust Expansion**: Remaining state with highest Sharpe Ratio (Return/Vol).
    3. **Cautious Decline**: Remaining state with lowest (most negative) return.
- **Reasoning**: Absolute thresholds fail across secular shifts. Relative ranking ensures the engine always identifies the current extremes.

---

## 6. Defensive Engineering
- **Numerical Stability**: Add a "jitter" (regularization) to the covariance diagonal ($1e-6$) to prevent Singular Matrix errors during fitting.
- **Health Checks**: Validate `is_hmm_healthy` after every fit. If the model contains NaNs or Inf in the transition matrix, fall back to the previous stable model.
- **Caching**: Results must be cached with a timestamp and a "Feature Hash". If the feature set changes, the cache must be invalidated automatically.

---

## 7. Goal: Long-Term Memory
This specification serves as the "source of truth". Any refactor to `ev_engine.py`, `data_ingestion.py`, or `pca_fusion.py` **must not** violate the Non-Anticipativity or Structural Consistency rules. If a change is made that increases "beauty" at the expense of "causality", it is a regression.
