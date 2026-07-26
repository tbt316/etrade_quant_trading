# Repository Instructions for Coding Agents (AGENTS.md)

Welcome! This is the root instructions and guidelines file (equivalent to `CLAUDE.md`) that all AI coding agents must check and adhere to when working in this repository.

> [!IMPORTANT]
> **Mandatory Session Context**
> Before executing *any* task in this repository, you **MUST** read and treat the following files as mandatory context:
> 1. [`docs/project_overview.md`](file:///Users/btian/EtradePythonClient/etrade_python_client/docs/project_overview.md) — The visual project map, architectural overview, execution flows, and backend-to-artifact associations.
> 2. [`docs/karpathy_guidelines.md`](file:///Users/btian/EtradePythonClient/etrade_python_client/docs/karpathy_guidelines.md) — The operational rules derived from Andrej Karpathy's LLM coding observations.
> 3. [`RELIABILITY.md`](file:///Users/btian/EtradePythonClient/etrade_python_client/RELIABILITY.md) — The evolving incident ledger and safety invariants for live E*TRADE data and dashboard changes.

---

## 🚀 The 4 Karpathy-Inspired Coding Principles

To prevent common agentic failure modes (wrong assumptions, overengineering, side effects, and vague completion criteria), you must enforce these principles at all times:

### 1. Think Before Coding
* **Don't Assume:** If a request is ambiguous or has multiple interpretations, **stop** and ask the user for clarification.
* **Surface Tradeoffs:** Clearly explain the architectural options, dependencies, and potential edge cases before coding.
* **Push Back on Complexity:** Suggest simpler alternatives if the requested implementation is unnecessarily complex.

### 2. Simplicity First
* **Minimum Viable Code:** Write the absolute minimum code required to solve the problem. Do not implement speculative features, abstractions, configurations, or unused fallback paths.
* **Refactor Bloat:** If 200 lines can be written cleanly in 50 lines, prefer the shorter, cleaner approach.

### 3. Surgical Changes
* **Strict Scope Limits:** Only modify files and lines directly related to the task.
* **No Unrelated Changes:** Do not "improve" adjacent formatting, rewrite comments, or refactor unrelated code unless explicitly requested.
* **Style Conformance:** Match the existing coding style, indentation, and structure of the surrounding files.

### 4. Goal-Driven Execution
* **Define Success Criteria:** Establish clear, verifiable outcomes before writing implementation code.
* **Concrete Verification:** Validate your work using actual unit tests, logs, or UI output. Do not declare success based on "it should work."
* **HTML Visual Verification:** Whenever a task changes, debugs, or verifies an HTML/dashboard artifact, you **MUST** open or render the actual HTML and visually inspect the result. Do not rely only on source inspection, grep output, generated markup, or syntax checks for HTML-facing changes.

### Dashboard Verification Contract
* **Name the Exact Symptom First:** If a screenshot could mean more than one thing, do not infer which visual or data issue the user means. Ask for the exact missing or incorrect element before changing code.
* **Verify the User's Artifact:** An offline renderer, fixture, source file, or locally patched copy is not proof that the live dashboard the user sees is fixed. Verify the same live endpoint and generated artifact used by the open dashboard.
* **"Fixed" Means the Served HTML Changed:** A source-code patch, successful test, process restart, or regenerated test file is not completion. Before saying an issue is fixed, regenerate the production HTML if required, reload the exact page the user opens, and confirm that the served HTML visibly contains the intended result.
* **Check for Stale Delivery:** Confirm that the running process loaded the new code and that browser, iframe, service-worker, and generated-file caching are not still serving an older artifact.
* **Check Rendered Data, Not Only Code:** For time-series charts, record the final rendered label and every final plotted value, then compare them with the current source of truth. A successful syntax check or the presence of an array entry is insufficient.
* **Inspect the Actual Canvas:** Confirm that the expected point, label, or line is visibly present in the rendered chart at the user's selected time span and viewport.
* **Do Not Claim Unobserved Success:** If browser or canvas tooling prevents visual confirmation, state that verification is incomplete and do not call the task complete. Never substitute verification of a different symptom.

---

## 📊 Quantitative Market Regime & Backtesting Constraints

If your task involves **Market Regime Detection**, **EV Probability**, or **Regime-Aware Backtesting**, you **MUST** strictly adhere to the causal rules documented in:
👉 [`docs/market_regime_detect_specs.md`](file:///Users/btian/EtradePythonClient/etrade_python_client/docs/market_regime_detect_specs.md)

This applies at a minimum when modifying or running:
* `live_trading/ev_engine.py` (Regime/GMM/EV calculations)
* `live_trading/ev_plots.py` (Visualization of regimes/EV margins)
* `live_trading/data_ingestion.py` (Fractional differencing & stationarity)
* `live_trading/pca_fusion.py` (Causal dimensionality reduction)
* `backtesting/backtest_runner.py` (Causal option strategy simulation)
* Scripts under `scratch/` matching `*regime*` or `*backtest*` (e.g., `regime_probability_audit.py`)

### Critical Causal Checklist for Backtests
For any regime-aware historical backtest execution or reporting, you must verify and document:
- [ ] **Training/Calibration End Date:** When was the HMM model calibrated?
- [ ] **Test Date Range:** What is the out-of-sample evaluation period?
- [ ] **Inference Method:** Is HMM inference running in walk-forward mode or fixed out-of-sample?
- [ ] **Regime Lagging:** Are daily regime values lagged by at least 1 trading day before trade entry?
- [ ] **Return Bucket Causality:** Do empirical return/probability buckets include only outcomes resolved *prior* to each trade decision date?

> [!CAUTION]
> Do not present a regime-aware backtest as valid if any of these causal fields are unverified or unknown. Look-ahead leakage violates the absolute non-anticipativity contract.
