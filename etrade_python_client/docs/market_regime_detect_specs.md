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
- **Implementation**: The agent **MUST** implement a custom `ExpandingRobustScaler` or use a strict rolling window. Calling `.fit()` or `.fit_transform()` on a DataFrame containing future data relative to the current evaluation step is a catastrophic failure. At time $t$, the features are scaled using only data that would be observable before the decision timestamp. If the regime is used for a next-session trade after the close, scaling may use data through close $t$. If the regime is used for an intraday or same-session trade, scaling must use only $X[0 \dots t-1]$ and the resulting signal must be timestamped accordingly.
- **Reasoning**: Global scaling leaks future volatility into the past. If the 2020 VIX spike is in the dataset, global scaling suppresses the relative magnitude of the 2018 Volmageddon event, altering historical regime classification.

### 2.3 Signal Availability Boundary
- **Regulation**: Causal math and tradable availability are separate requirements. A feature value can be causal in an end-of-day research sense while still being unavailable for a same-day trade.
- **Implementation**: Every regime feature frame must carry or document whether each row is available at `open_T`, `intraday_T`, `close_T`, or `next_open_T_plus_1`. Backtests must consume the regime value only after that timestamp. For daily close inputs such as `SPY_Close`, `VIX_Close`, realized volatility, and option end-of-day marks, the default tradable regime for date `T` is the previous trading day's regime unless the strategy explicitly models close-after-finalization execution.
- **Reasoning**: Same-row contamination can enter through rolling scalers, trailing filters, and close-based features even when no future calendar date is used.

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
- **Regulation**: The engine must explicitly defend against HMM label switching after every refit. Use deterministic emission sorting as the primary identity rule and, when warm-starting from a prior model, use **Symmetric KL-Divergence** or an equivalent emission-distance assignment to map newly fitted states to the previous model's state order.
- **Implementation**: State alignment must be applied to the model object itself, not only to the predicted state vector. Reorder start probabilities, transition matrix rows/columns, means, covariances, mixture weights, and any probability columns together.
- **Reasoning**: Prevents "identity drift" which would break downstream strategy logic tied to specific regime IDs.

### 5.3 Adaptive Archetype Ranking
- **Regulation**: Labels (e.g., "Market Turmoil") must be assigned by ranking states relative to each other in the current model.
- **Ranking Criteria**:
    1. **Market Turmoil**: State with highest median VIX.
    2. **Robust Expansion**: Remaining state with highest Sharpe Ratio (Return/Vol).
    3. **Cautious Decline**: Remaining state with lowest (most negative) return.
- **Reasoning**: Absolute thresholds fail across secular shifts. Relative ranking ensures the engine always identifies the current extremes.
- **Boundary**: These labels are relative archetype descriptions only. They must not be treated as absolute macro regimes or hard trading permissions without the actionable overlay in Section 5.4.

### 5.4 Actionable Three-Regime Overlay
- **Regulation**: Relative HMM archetype labels are diagnostic labels, not sufficient proof of an actionable crisis regime. A state labeled "Market Turmoil" only because it has the highest VIX among current HMM states must not be promoted directly to final `Panic / Crisis`.
- **Implementation**: The final three-regime taxonomy (`Expansion`, `Cautious Decline`, `Panic / Crisis`) must require causal close-T market stress evidence such as SPY drawdown, recent SPY log return shock, or absolute VIX stress. Preserve raw HMM state/label columns for audit, but downstream plotting/backtesting should consume the final detected regime only after this overlay is applied and timestamped.
- **Reasoning**: In calm bull markets, the highest relative VIX state can still represent low absolute volatility. Directly mapping that state to `Panic / Crisis` creates false crisis classifications during healthy uptrends.

### 5.5 Semantic Drift and Absolute Anchors
- **Regulation**: The spec distinguishes label switching from semantic drift. Label switching is a state-permutation bug and must be corrected by sorting/alignment. Semantic drift is the gradual change in the fitted emission distributions and must be disclosed, bounded, or anchored before regime IDs are used for trading.
- **Implementation**: Any regime-aware backtest or live rule must state whether HMM emissions are adaptive or anchored. If adaptive, downstream logic must use relative HMM labels only as diagnostics and must rely on the final three-regime overlay or another documented absolute risk gate for trade decisions. If absolute regime identity is required, train an anchor model on a representative multi-cycle calibration set and either freeze emission parameters or constrain refits against those anchor emissions while allowing transition probabilities to adapt.
- **Implementation**: Absolute context features such as VIX level may be used as raw-market overlay inputs. If they are fed into the HMM emission vector, they must be causally timestamped, transformed or scaled without look-ahead, and documented as intentional absolute anchors; they must not violate stationarity and scaling rules by being globally normalized.
- **Reasoning**: A quiet state in 2008 and a quiet state in 2017 are both relatively quiet inside their local sample, but they are not necessarily the same absolute market condition. Trading logic must not silently depend on that equivalence.

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
- **Agent Instruction**: The agent must implement a deterministic state-mapping heuristic immediately after every `.fit()`. For equity-index regimes, the default order is emission variance from lowest to highest: State 0 = quietest, State 1 = medium variance, State N = highest variance. The variance metric must be computed from the fitted emission distribution used by the HMM, not from downstream relabeled predictions.
- **Agent Instruction**: If a prior walk-forward model exists, the refit must also perform continuity alignment against the prior model using symmetric KL divergence, Wasserstein distance, or another documented emission-distance assignment. The implementation must record which mapping was applied for each refit in diagnostics or logs.

### Mandate 10.2: Causal Viterbi Decoding
hmmlearn relies on the Viterbi algorithm (.predict()), which naturally looks forward in time to smooth out the hidden state path over a sequence.
- **Agent Instruction**: During live trading or backtesting, if the agent runs .predict() on a sequence of data, it must only extract the final integer of the output array as the regime for time T. Feeding the smoothed historical sequence back into the backtester will inherently introduce look-ahead bias.

- **Agent Instruction**: When retraining the model on a rolling or expanding basis, the agent must capture the previous window's transition matrix, means, covariances, mixture weights if present, and start probabilities, and pass them into the next HMM with `init_params=''` or the hmmlearn-equivalent warm-start path.
- **Agent Instruction**: Random or k-means initialization is allowed only for the first causal training window or after the explicit warm-start fallback protocol in Mandate 10.5 fires. Routine refits must not silently reinitialize from scratch.

### Mandate 10.4: Holistic State Alignment
A common implementation failure when mapping HMM states deterministically (e.g., mapping State 0 to Lowest Variance) is only re-labeling the final predictions and ignoring the internal model attributes.
- **Agent Instruction**: When sorting and reassigning the hidden states, the script must simultaneously reorder the hmm.transmat_ (Transition Matrix), hmm.means_, and hmm.covars_. Failing to realign the underlying probability distributions with the new integer labels will completely decouple the mathematical state from the predicted output, causing a cascading failure when the model attempts a warm start on the next step.

### Mandate 10.5: Warm-Start Fallback Protocol
- **Agent Instruction**: When injecting the previous window's parameters into the init_params='' constructor for a warm start, the agent must wrap the initialization in a try-except block. If the matrix dimensions misalign or if the model encounters a singular covariance matrix during an anomalous market jump, the code must cleanly fall back to a kmeans initialization rather than crashing the entire pipeline.
- **Agent Instruction**: Every fallback must be visible in diagnostics. The trace or report must include enough information to identify fallback dates, reason, fitted K, and whether state alignment was restored afterward.

### Mandate 10.6: Causal Probability Trace Extraction
When visualizing or backtesting regimes, downstream scripts (e.g., plotting utilities) often require the full historical probability trace.
- **Agent Instruction**: The agent is strictly forbidden from manually reconstructing historical probabilities using a `for` loop that calls `predict_proba()` sequentially on the *final* returned global model (which contains parameters fitted on future data). This introduces severe look-ahead bias.
- **Agent Instruction**: Instead, the regime training engine (`train_regime_hmm`) must calculate, store, and return the causal probabilities directly within its walk-forward refit loop. Downstream scripts must exclusively read these pre-calculated causal probability traces from the returned DataFrame.

### Mandate 10.7: Scaler Inverse Transformation Safety
- **Agent Instruction**: The `RollingRobustScaler` (or any custom causal scaler) used in `data_ingestion.py` must explicitly implement an `inverse_transform` method. Downstream logic in `get_regime_labels` relies on this method to convert physical PCA centroids back to raw VIX/Return values for semantic labeling.
- **Agent Instruction**: Bare `try-except Exception:` blocks in `get_regime_labels` that swallow missing method errors and silently assign fallback values (like `avg_vix=20.0` for all states) are strictly forbidden. This silently destroys archetype ranking, causing the engine to misidentify regimes (e.g., mistaking an expansion for turmoil).

### Mandate 10.8: Semantic Drift Diagnostics
- **Agent Instruction**: Walk-forward diagnostics must report the raw HMM emission summaries for each refit or at least for each analysis period: state mean, variance, mapped prior-state ID, relative archetype label, final detected regime, and overlay reason.
- **Agent Instruction**: If the same numeric state's emission variance, return mean, or reconstructed VIX proxy changes beyond a documented tolerance between adjacent refits, the report must flag semantic drift. The flag does not automatically invalidate the model, but it prevents downstream code from treating the raw HMM state ID as an absolute regime.
- **Agent Instruction**: Plots and HTML reports must visibly distinguish raw HMM state/archetype labels from final actionable regime labels.

---

## 11. Empirical Return Construction Constraints

### Mandate 11.1: Calendar vs. Trading Day Alignment
- **Agent Instruction**: When constructing forward return buckets (e.g., `build_regime_return_arrays`), the agent must mathematically distinguish between calendar days and trading days. Option horizons (`days_to_exp`) are calculated in calendar days, whereas the historical `DataFrame` index consists only of trading days.
- **Agent Instruction**: The script must never blindly execute `df.shift(-horizon)` when `horizon` is derived from calendar days. A row shift of 45 on a trading-day dataset steps forward ~63 calendar days, artificially inflating historical standard deviation and causing massive mispricing in the empirical expected value (EV) engine. The agent must reliably convert the calendar day horizon into equivalent trading days (e.g., `int(horizon * 252 / 365)`) before applying the positional `.shift()`.

### Mandate 11.2: Walk-Forward Return Bucket Availability
- **Agent Instruction**: A backtest must never price a trade at date T using a regime return bucket that includes future terminal returns whose entry date or exit date is after T. Forward returns are valid labels for research, but they become data leakage when used to estimate the probability model for earlier trades.
- **Agent Instruction**: For historical backtests, `regime_dict`, daily GMM models, and any EV probability engine must be rebuilt or incrementally updated using only observations whose full forward horizon has already resolved by T. For a 42-calendar-day option, the latest eligible bucket entry at T is approximately `T - trading_horizon`.
- **Reasoning**: Grouping all 2015-2026 future returns by causal regime is still non-causal if a 2016 trade is priced with 2020 crash outcomes. Live trading may use all history available as of today; backtests must simulate that availability date by date.

### Mandate 11.3: Cache Key Completeness
- **Agent Instruction**: Regime caches must be keyed by `as_of_date`, `horizon_calendar_days`, converted `trading_horizon`, `n_components`, feature list/hash, scaler/PCA version, model class, and data vintage. Timestamp-only caches are forbidden.
- **Agent Instruction**: If `build_regime_return_arrays` is called with a different horizon or K than the cached object was built with, the cache must be invalidated. The function must not return a 7-day bucket to a 42-day strategy or a 42-day bucket to a 21-day strategy.
- **Reasoning**: Reusing stale buckets silently changes the payoff distribution and can make the EV surface appear stable when it is using the wrong horizon.

### Mandate 11.4: Entry-Date Censoring in Research Audits
- **Agent Instruction**: Any script that audits probabilities at trade date `T` must censor the empirical return sample to entries whose terminal outcome is already known by `T`. If the return label is `r_{t->t+h}`, then the latest admissible entry row in the sample is `t <= T - h_trading`.
- **Agent Instruction**: `df.loc[:T].dropna(subset=[future_return])` is forbidden when `future_return` was precomputed with a forward shift, because the final `h_trading` rows before `T` still encode outcomes from after `T`.
- **Reasoning**: This is the audit-script version of look-ahead bias. The sample appears historical because the row index is in the past, but the label itself contains future prices beyond the decision date.

### Mandate 11.5: Return-Distribution GMM Component Selection
- **Agent Instruction**: Scripts must not hard-code a 2-component GMM for regime-conditioned SPY return distributions. The fitting routine must include `K=1` as a candidate and choose the component count with a penalized model-selection criterion such as BIC, subject to a minimum observation-per-component guardrail.
- **Agent Instruction**: Reports must disclose the selected component count, BIC/AIC, and candidate scores for each regime. A single Gaussian is the preferred fit whenever additional mixture components do not improve the penalized criterion.
- **Reasoning**: The priority is to describe each regime's empirical return distribution parsimoniously. Some regimes are close to Gaussian, while others may need more mixture components to capture skew, tails, or multimodality.

---

## 12. Current `ev_plots.py` / `ev_engine.py` Audit Regulations

### Mandate 12.1: No Centered Rolling Filters
- **Issue Observed**: `data_ingestion.py` applies a 5-day rolling median with `center=True` to volatility columns. A centered window uses future rows (`t+1`, `t+2`) to rewrite the feature at T.
- **Agent Instruction**: All feature smoothing used for training, backtesting, live inference, charts, or return bucketing must be strictly trailing. `rolling(..., center=True)` is forbidden in the MRD pipeline. Use `center=False`, EWMA, or a trailing median and document whether the signal is available at close T or next open T+1.
- **Reasoning**: Centered filtering is direct look-ahead bias and will make volatility shocks appear earlier and cleaner than they were in real time.

### Mandate 12.2: Walk-Forward Model Selection
- **Issue Observed**: `train_regime_hmm(expanding_window=True)` currently performs BIC state-count selection on the full feature sample before the walk-forward inference loop.
- **Agent Instruction**: If K is selected dynamically, K must be selected inside the walk-forward refit at each refit date using only training data available at that date. Alternatively, K must be fixed by an out-of-sample research decision before the backtest period starts and recorded in configuration.
- **Reasoning**: Full-sample K selection leaks future regime complexity into the past. A model that knows 2020-like turmoil requires more states in 2015 is not causal.

### Mandate 12.3: No Future-Trained Warm Starts
- **Issue Observed**: The walk-forward loop initializes `current_hmm` from a model fitted on the full sample, then warm-starts historical refits from those parameters.
- **Agent Instruction**: A walk-forward model at date T must not use transition matrices, mixture weights, means, covariances, scaler state, PCA loadings, or labels from any model fitted with data after T. The first historical model must be initialized from scratch using only the first training window. Subsequent warm starts may use only the immediately prior walk-forward model.
- **Reasoning**: EM warm starts can retain future information even after refitting on a shorter window, especially when iteration counts are capped.

### Mandate 12.4: Scaler/PCA/Model Window Consistency
- **Issue Observed**: Refit code mixes causal rolling-scaled features with a freshly fit `RobustScaler` on raw historical data for semantic labels. This can make `get_regime_labels` invert centroids through a scaler that was not used to produce the HMM emissions.
- **Agent Instruction**: Every HMM snapshot must carry the exact scaler snapshot and PCA/fusion snapshot used to generate the emissions on which it was fit. Label inversion, current inference, and plotted centroids must use those exact objects. Re-fitting a different scaler merely for labeling is forbidden.
- **Reasoning**: If the inverse transform is not the inverse of the transform that trained the model, archetype rankings can flip and downstream strategy logic will trade the wrong regime.

### Mandate 12.5: Full Forward Filtering, Not Posterior History Reuse
- **Agent Instruction**: For live decisions, using `predict_proba(X[:T])[-1]` is acceptable as a practical filtering approximation because the final posterior has no future observations inside that prefix. For historical traces, scripts must only store that final probability for each prefix. They must never reuse earlier rows from `predict_proba(X[:T])` as labels for earlier dates.
- **Reasoning**: Earlier rows in a prefix are smoothed by later observations within that prefix. Only the final row is causal for decision time T.

### Mandate 12.6: Decision Timestamp Discipline
- **Agent Instruction**: Every regime value must declare its tradable timestamp: `close_T_for_next_session`, `intraday_T`, or `open_T`. If a feature uses daily close, VIX close, or same-day option chain settlement data, trades may only be entered after that data is actually observable, usually next session unless the strategy explicitly runs at the close after all source data is finalized.
- **Reasoning**: A model can be mathematically causal and still operationally leaked if it enters before the data used to compute the signal was known.

### Mandate 12.6a: Backtest Regime Map Lag
- **Agent Instruction**: Daily backtests that enter during session `T` must pass a one-trading-day-lagged regime map into `run_put_credit_spread_backtest` when the regime detector uses daily close or end-of-day features. A dictionary keyed by `T` must contain only the latest regime observable before that trade decision, normally the regime calculated from `T-1`.
- **Agent Instruction**: Unshifted maps such as `{date_T: HMM_State_T}` are allowed only for pure end-of-day reporting or for strategies that explicitly execute after all `T` inputs are finalized and whose option pricing also reflects that same decision timestamp.
- **Reasoning**: `HMM_State_T` may be causal as a label for the close of `T`, but it is not available for selecting a trade earlier on `T`.

### Mandate 12.7: Hard Risk Gates Must Be Enforced
- **Issue Observed**: `ev_plots.py` prints "SKIP PUT SELLING" when Market Turmoil, negative GEX, or VIX backwardation is detected, but the script continues into put-spread selection.
- **Agent Instruction**: A hard gate must return before trade construction unless the caller passes an explicit override flag that is logged with timestamp, reason, and operator. Warnings that continue into trade generation are not risk controls.
- **Reasoning**: Tail-risk gates only protect capital if they affect execution.

### Mandate 12.8: Expiration Selection Must Be Dynamic
- **Issue Observed**: `ev_plots.py` hard-codes `target_exp = "2026-05-29"`.
- **Agent Instruction**: Live and backtest strategy code must select expiration from the available option expirations based on configured target DTE and liquidity filters. Hard-coded future dates are forbidden outside one-off research scripts.
- **Reasoning**: A hard-coded expiration silently becomes stale and can select unavailable, illiquid, or unintended contracts.

### Mandate 12.9: Audit Horizon Consistency
- **Agent Instruction**: If an audit script prices a contract using the actual listed expiration selected from the chain, then the realized benchmark and empirical return bucket must use that exact expiration date or exact day count for that trade date. Mixing an approximate trading-day shift for the probability estimate with a separate calendar-date lookup for realized outcome is forbidden.
- **Reasoning**: If the forecast horizon and realized horizon differ, the audit is no longer comparing like with like; the measured edge can come from horizon mismatch rather than regime skill.

### Mandate 12.10: Regime Label Provenance
- **Agent Instruction**: Historical audit rows must display the semantic label generated by the HMM snapshot that existed at that row's decision time, or an explicitly versioned mapping derived only from information available then. Re-labeling all historical rows with `get_regime_labels(final_model)` is forbidden.
- **Reasoning**: Even if the integer state trace is causal, projecting final-model semantics backward can rewrite the narrative of past regimes and hide state-identity drift.

### Mandate 12.11: Audit Cache Validity
- **Agent Instruction**: Research audit caches must include `audit_start`, `audit_end`, target horizon, selected expiration policy, data vintage, feature hash, and code/spec version. A bare JSON cache keyed only by filename is forbidden.
- **Agent Instruction**: If any of those inputs change, the audit must recompute rather than load stale results.
- **Reasoning**: Cached research output is part of the evidence base. Reusing a stale audit after model or data changes is functionally equivalent to looking at the wrong experiment.

### Mandate 12.12: Raw-vs-Filtered Feature Disclosure
- **Agent Instruction**: Any regime audit or chart that relies on filtered volatility features (for example a trailing median on VIX or realized vol) must explicitly disclose that the HMM operated on filtered inputs rather than raw closes. Where the objective is raw-data forensic validation, the script must offer a no-filter mode.
- **Reasoning**: A trailing filter can be causal and still materially reshape shocks, persistence, and state boundaries. If that transformation is not disclosed, reviewers may attribute behavior to the raw market tape when it was created by preprocessing.

### Mandate 12.13: Existing Backtest Script Safety
- **Issue Observed**: `scratch/run_best_backtest.py` historically called `train_regime_hmm(df_hist, n_components=k)` without `expanding_window=True`, then passed the resulting same-date regime dictionary directly into the backtester.
- **Agent Instruction**: Any script used for regime-aware historical backtests must either call the walk-forward causal path (`expanding_window=True`) or load a precomputed causal trace. It must not use the default non-expanding/global model for 2020-2026 performance claims.
- **Agent Instruction**: If the declared experiment is "2020-2026 using historical data up to 2018", calibration, K selection, scaler/PCA snapshots, label mapping, and return/probability buckets must be fit or selected using data no later than `2018-12-31`. Subsequent 2020-2026 regime inference must be out-of-sample filtering unless the experiment explicitly declares walk-forward refits and their allowed training window.
- **Reasoning**: A script can reference causal APIs elsewhere in the codebase and still run a non-causal experiment if it uses the wrong defaults or an unlagged regime map.

---

## 13. Regime-to-Strategy Adaptation Rules

The active backtest strategy definitions live in `backtesting/strategies/*.yaml`. Regime detection should be used as a risk overlay first, and as an entry optimizer second.

### Mandate 13.1: Baseline Strategy Must Remain a Control
- **Agent Instruction**: `baseline_put_spread` should remain `regime_aware: false` as the experimental control. Do not tune its entries by regime. Use it to measure whether the HMM overlay adds value after costs.

### Mandate 13.2: Regime Overlay for Put Credit Spreads
- **Agent Instruction**: For regime-aware put selling, use these default controls until walk-forward tests justify different values:
    1. **Robust Expansion**: put credit spreads allowed; target short delta may be increased modestly from -0.15 toward -0.20; normal DTE and normal margin cap may be used.
    2. **Emerging Expansion**: put credit spreads allowed at baseline risk; target short delta around -0.15; avoid increasing margin until the regime persists after debounce.
    3. **High Vol Chop**: reduce risk; target short delta around -0.10, lower `margin_limit_pct`, require higher minimum credit/EV, and prefer wider long-leg protection only if EV per unit expected shortfall improves.
    4. **Cautious Decline**: defensive mode; either skip new put spreads or use very low delta, reduced size, shorter holding periods, and stricter profit-taking. Do not roll losing puts mechanically into larger exposure.
    5. **Market Turmoil**: no new short puts by default. Only hedged/risk-defined trades with explicit override are allowed. Close or reduce existing short-vol exposure according to a pre-declared crisis protocol.
- **Reasoning**: Put credit spreads are short downside convexity. The correct regime response is primarily to reduce left-tail exposure, not to chase higher premium.

### Mandate 13.3: Panic Swap Controls
- **Agent Instruction**: `dynamic_delta_variant` must not force-close and replace positions 1:1 during Market Turmoil unless a walk-forward test proves that the replacement improves drawdown and expected shortfall after transaction costs. If enabled, `panic_qty_multiplier` should default below 1.0, and the strategy must cap aggregate short put delta, margin usage, and portfolio expected shortfall.
- **Reasoning**: A 1:1 panic swap can realize losses, pay wide spreads, and immediately reload short crash convexity at the worst liquidity point.

### Mandate 13.4: EV-Optimized Strategy Validation
- **Agent Instruction**: `ev_optimized_put_spread` may use the GMM/HMM probability engine only with walk-forward return buckets available as of the trade date. Entry requires positive EV, acceptable EV/expected-shortfall ratio, liquidity filters, and a hard block in Market Turmoil unless explicitly overridden.
- **Reasoning**: EV optimization is highly sensitive to tail distribution estimation. If the regime return buckets leak or are stale, the optimizer will select the most overfit spread.

---

## 14. Enforcement & Agent Operating Rules

### Mandate 14.1: Spec Must Be Loaded Before MRD Work
- **Agent Instruction**: Before modifying, reviewing, or running market-regime detection, EV probability, or regime-aware backtest code, the coding agent must read this file and cite which mandates govern the task.
- **Scope**: This applies at minimum to `live_trading/ev_engine.py`, `live_trading/ev_plots.py`, `live_trading/data_ingestion.py`, `live_trading/pca_fusion.py`, `backtesting/backtest_runner.py`, active strategy YAML files in `backtesting/strategies/`, and `scratch/*regime*` / `scratch/*backtest*` scripts.

### Mandate 14.2: Specs Are Not Self-Enforcing
- **Agent Instruction**: A Markdown specification is advisory unless it is connected to agent instructions, tests, linters, CI checks, or runtime assertions. Any critical non-anticipativity rule must have at least one executable guard where practical.
- **Recommended Guards**:
  1. A repo-level `AGENTS.md` that requires agents to read this spec before MRD/backtest work.
  2. Regression tests that fail on global scaler/PCA fitting, non-expanding backtest traces, unlagged daily regime maps, stale cache keys, and unresolved forward-return buckets.
  3. Runtime assertions in backtest scripts that record `train_end`, `regime_signal_timestamp`, and whether regime maps are lagged.
- **Reasoning**: Agents and humans do not automatically ingest every Markdown file in a repository. The spec must be promoted into the working instructions and backed by failing checks.
