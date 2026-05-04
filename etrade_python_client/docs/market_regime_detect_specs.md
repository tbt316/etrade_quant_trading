# Market Regime Detection Specification

## 1. Core Philosophy: Absolute Non-Anticipativity
The primary goal of the Market Regime Detection (MRD) engine is to provide causal, real-time identification of market archetypes without **look-ahead bias**. In quantitative finance, data leakage from the future into the past (e.g., global scaling, Viterbi decoding over a full sample) results in "beautiful" backtests that fail in live production. This specification mandates a strictly causal pipeline.

**Rule 1.1: Rolling vs. Expanding Windows**
While an expanding window prevents data leakage, it dilutes recent structural breaks as the sample size grows to infinity. All "expanding" logic must feature a parameterized memory cap (e.g., rolling window or Exponentially Weighted Moving Average) so the models can adapt to new market behaviors without being weighed down by decades of stale data.

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

---

## 8. Data Ingestion & Preprocessing Constraints (data_ingestion.py)

### Mandate 8.1: Strict Stationarity Enforcement
Unsupervised models (HMMs/GMMs) and PCA will fail or produce spurious regimes if applied to non-stationary price data.
- **Agent Instruction**: The script must enforce transformations (e.g., log returns, fractional differencing) on raw price data. Before passing data downstream, the code must dynamically verify stationarity using the adfuller test (rejecting the null hypothesis at p<0.05). If it fails, the script must throw an explicit error or dynamically difference the data.

- **Agent Instruction**: Modify or replace the expanding scaler with a RollingRobustScaler. Outliers from 10 years ago should not skew the interquartile range (IQR) today.

### Mandate 8.3: Earnings-Neutral Volatility Inputs
Unsupervised models are highly sensitive to sudden, deterministic volatility spikes.
- **Agent Instruction**: Ensure that the volatility features fed into the HMM/GMM do not interpret standard quarterly earnings seasons as sudden structural market regime breaks. The engine must explicitly ingest the forward factor with earnings-induced jumps removed to ensure the model reacts only to genuine macroeconomic shifts rather than scheduled calendar events.

### Mandate 8.4: Rolling Window Execution Limits
- **Agent Instruction**: When implementing the RollingRobustScaler (from Mandate 8.2), the agent must absolutely avoid using slow, iterative pandas.Series.rolling().apply() loops in the core execution path. It must utilize numpy striding (numpy.lib.stride_tricks) or built-in vectorized functions to ensure the scaling logic does not introduce unacceptable execution latency for live trading.

---

## 9. Dimensionality Reduction Constraints (pca_fusion.py)

### Mandate 9.1: Sequential Subspace Fitting
- **Agent Instruction**: PCA and SparsePCA must never be instantiated and fit using fit_transform() over the full dataset during backtesting. The agent must implement a sequential loop that fits the PCA strictly on T−window and only transforms the vector at T.

- **Agent Instruction**: The agent must enforce eigenvector sign alignment across consecutive rolling windows. Calculate the cosine similarity (using scipy.spatial.distance.cosine) between the principal components of T and T−1. If the correlation is negative, the agent must multiply the current component by −1 to ensure continuous, stable feature generation.

### Mandate 9.3: First-Window Edge Cases in PCA Alignment
- **Agent Instruction**: When implementing the eigenvector sign-flipping logic (Mandate 9.2), the agent must elegantly handle the T=0 edge case. The script must initialize an empty array or safely bypass the cosine similarity check on the very first rolling window, otherwise the script will throw a NoneType or IndexError during the initial backtest step.

---

## 10. Regime Detection Engine Constraints (ev_engine.py)

### Mandate 10.1: Deterministic State Alignment (Label Switching)
Unsupervised models like GaussianMixture and hmm assign arbitrary integer labels (e.g., State 0, State 1) to regimes. When the model is retrained, "State 0" could spontaneously become the high-volatility bear market instead of the low-volatility bull market, wrecking the downstream strategy.
- **Agent Instruction**: The agent must implement a deterministic state-mapping heuristic immediately after .fit(). For example, the script must automatically calculate the variance of the emissions for each state, and strictly map the states such that State 0 = Lowest Variance, State 1 = Medium Variance, State N = Highest Variance.

### Mandate 10.2: Causal Viterbi Decoding
hmmlearn relies on the Viterbi algorithm (.predict()), which naturally looks forward in time to smooth out the hidden state path over a sequence.
- **Agent Instruction**: During live trading or backtesting, if the agent runs .predict() on a sequence of data, it must only extract the final integer of the output array as the regime for time T. Feeding the smoothed historical sequence back into the backtester will inherently introduce look-ahead bias.

- **Agent Instruction**: When retraining the model on a rolling basis, the agent must capture the previous window's transition matrix, means, and covariances, and pass them into the hmm.GaussianHMM(init_params='') constructor as the starting weights for the new window.

### Mandate 10.4: Holistic State Alignment
A common implementation failure when mapping HMM states deterministically (e.g., mapping State 0 to Lowest Variance) is only re-labeling the final predictions and ignoring the internal model attributes.
- **Agent Instruction**: When sorting and reassigning the hidden states, the script must simultaneously reorder the hmm.transmat_ (Transition Matrix), hmm.means_, and hmm.covars_. Failing to realign the underlying probability distributions with the new integer labels will completely decouple the mathematical state from the predicted output, causing a cascading failure when the model attempts a warm start on the next step.

### Mandate 10.5: Warm-Start Fallback Protocol
- **Agent Instruction**: When injecting the previous window's parameters into the init_params='' constructor for a warm start, the agent must wrap the initialization in a try-except block. If the matrix dimensions misalign or if the model encounters a singular covariance matrix during an anomalous market jump, the code must cleanly fall back to a kmeans initialization rather than crashing the entire pipeline.
