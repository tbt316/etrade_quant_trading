# Project Overview

This repository houses two major quantitative trading systems:
1. **A historical option strategy backtesting stack** (highly simulated, multi-variant testing harness).
2. **A causal market-regime detection & expected value (EV) engine** (featuring Hidden Markov Models, Gaussian Mixture Models, and option pricing calculations).

This document serves as the high-level map of the repository, illustrating the end-to-end architecture, visual execution flows, command shortcuts, and the direct associations between backend scripts and frontend UI/HTML artifacts.

---

## 🗺️ Architectural Flows

### 1. Backtesting Stack & Artifact Flow
The backtesting stack evaluates option strategies (configured via YAML) across a range of parameters (variants) and compiles the results into interactive Plotly HTML dashboards.

```mermaid
flowchart TD
    subgraph Input Configurations
        YAML["backtesting/strategies/*.yaml<br/>(Strategy Rule Specs)"]
    end

    subgraph Batch Execution
        RC["backtesting/run_comparison.py<br/>(Batch Harness)"]
        TC["backtesting/temp_config_*.json<br/>(Temporary Variant Config)"]
    end

    subgraph Execution Engine
        BR["backtesting/backtest_runner.py<br/>(Historical Simulator)"]
        Cache[("option_data_cache.db / Shard DBs<br/>(Historical Option Prices)")]
    end

    subgraph Outputs & Artifacts
        REP["backtesting/reports/report_*.html<br/>(Per-Run Interactive Report)"]
        LOG["backtesting/experiments_log.jsonl<br/>(Run History Registry)"]
        GER["backtesting/generate_experiments_report.py<br/>(Dashboard Compiler)"]
        DASH["backtesting/experiments_dashboard.html<br/>(Experiments Leaderboard)"]
    end

    YAML -->|Strategy Config| RC
    RC -->|Generates Temp Configs| TC
    TC --> BR
    Cache -->|Option & Spot Quotes| BR
    BR -->|1. Generates interactive Plotly report| REP
    BR -->|2. Appends summary metadata| LOG
    LOG --> GER
    GER -->|3. Compiles final leaderboard| DASH
```

### 2. Regime Detection, EV Engine, & Audit Flow
The Market Regime Detection (MRD) stack retains the existing causal HMM/GMM
research path and adds a separate V2 shadow path. V2 validates immutable
SPY/VIX evidence and separates persistent background stress from fast event
shocks. It is not connected to order execution. The provider-backed path is a
library-only, entitlement-gated shadow collector; it does not schedule itself
or read credentials at import time.

```mermaid
flowchart TD
    subgraph Data & Feature Layer
        DI["live_trading/data_ingestion.py<br/>(Fractional-Diff & Stationarity Checks)"]
        YF[("yfinance Cache / Flat Files<br/>(SPY, SPX, VIX, IRX Close Prices)")]
        MPG["regime_market_data_gateway.py<br/>(Bounded Massive/Cboe acquisition)"]
        RPE["regime_provider_evidence.py<br/>(Strict retained-byte parsers)"]
        RMD["live_trading/regime_market_data.py<br/>(Immutable SPY/VIX Evidence Contract)"]
        RES[("regime_evidence_store.py<br/>(Bytes, Receipts, Revisions, Verified Snapshots)")]
    end

    subgraph Dimensionality Reduction
        PF["live_trading/pca_fusion.py<br/>(Causal PCA & eigenvector sign align)"]
    end

    subgraph Modeling & Calibration
        EE["live_trading/ev_engine.py<br/>(Causal HMM/GMM Fitting & EV Logic)"]
        Snap[("backtest_cache/regime_snapshots/*.pkl<br/>(Causal HMM/GMM Cache)")]
        RD2["live_trading/regime_detector_v2.py<br/>(Background + Shock Shadow Detector)"]
        RCAL["live_trading/regime_calibration.py<br/>(Purged Candidate Selection + Immutable Artifact)"]
        RSIG["live_trading/regime_signal.py<br/>(Typed, Shadow-Only Signal Contract)"]
        RSP["live_trading/regime_shadow_publish.py<br/>(Entitlement-Gated Publisher)"]
        RSS[("live_trading/runtime/regime_v2_shadow.json<br/>(Atomic, Owner-Only Read Model)")]
        RART[("regime_v2_calibration_artifact.json<br/>(Research-only, Content Addressed)")]
    end

    subgraph Visual Analytics
        EP["live_trading/ev_plots.py<br/>(Diagnostic Plotter)"]
        RPA["scratch/regime_probability_audit.py<br/>(SPY vs SPX Statistical Auditor)"]
        RVA["scratch/regime_detector_v2_audit.py<br/>(Unverified Legacy Replay Audit)"]
        BANN["backtesting/backtest_runner.py<br/>(Exact-Date V2 Audit Annotations)"]
        VDASH["/api/regime_v2_shadow + dashboard_template.html<br/>(Redacted Advisory Display)"]
    end

    subgraph Generated Frontends & Images
        PNGS["s_and_p_data/*.png<br/>(Regime Timelines, GMM Fits, EV Plots)"]
        ARTS["/Users/btian/.gemini/antigravity/artifacts/*.png<br/>(GMM Sub-Regimes & Density fits)"]
        AUD["audit_plots/regime_probability_audit.html<br/>(Interactive Audit Dashboard)"]
        CSVS["audit_plots/*.csv<br/>(Causal Audit Trace Tables)"]
    end

    Massive["Massive SPY daily summary"] --> MPG
    Cboe["Cboe VIX history CSV"] --> MPG
    MPG -->|"exact decoded parser-input bytes"| RES
    MPG --> RPE
    RPE -->|"parser receipts + observations"| RES
    RES -->|"verified immutable snapshot"| RMD
    YF --> DI
    YF --> RMD
    RMD --> RD2
    RD2 --> RCAL
    RCAL --> RART
    RD2 --> RSIG
    RART --> RSIG
    RSIG --> BANN
    RES --> RSP
    RD2 --> RSP
    RSP --> RSIG
    RSIG --> RSS
    RSS --> VDASH
    RD2 --> RVA
    DI --> PF
    PF --> EE
    EE -->|Caches causal states| Snap
    Snap --> EE
    EE --> EP
    EE --> RPA
    EP -->|Saves charts| PNGS
    EP -->|Saves core clusters/densities| ARTS
    RPA -->|Saves interactive audit UI| AUD
    RPA -->|Saves tabular CSV audits| CSVS
```

---

## 🛠️ Stack Component Descriptions

### 1. Backtesting Stack

*   **Strategy Specifications** (`backtesting/strategies/*.yaml`): YAML files (e.g., `baseline_put_spread.yaml`, `put_call_credit_spread.yaml`) defining trade structure, short/long delta targets, margin caps, early profit exits, DTEs, and rolling behavior. This is the stable configuration layer.
*   **Batch Harness** ([`run_comparison.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/run_comparison.py)): Orchestrates parameters sweeps. It maps out variants, writes temporary configuration JSONs, and triggers the `backtest_runner.py` for each variant before compiling the leaderboard.
*   **Historical Simulator** ([`backtest_runner.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/backtest_runner.py)): The core execution engine. It simulates daily trade lifecycles, parses the option chain databases, applies margin math, tracks PnL, logs trade events, and generates an interactive, detailed HTML report. V2 regime signals enter through a separate typed, exact-date audit lane and cannot alter the legacy strategy path in R4.
*   **Dashboard Compiler** ([`generate_experiments_report.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/generate_experiments_report.py)): Reads the flat-file JSONL experiments log and compiles a central HTML dashboard leaderboard for comparison.
*   **Experiment Manager** ([`experiment_manager.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/experiment_manager.py)): A utility script providing CLI commands to delete, rebuild, or manage individual runs in the experiments log.

### 2. Regime Detection & EV Engine

*   **Data Ingestion** ([`data_ingestion.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/data_ingestion.py)): Ingests raw market series (SPY, SPX, VIX, IRX) and transforms them to stationary inputs (log returns, fractional differencing) while running ADF (Augmented Dickey-Fuller) stationarity assertions.
*   **PCA Fusion** ([`pca_fusion.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/pca_fusion.py)): Projects scaled stationary features into mathematically orthogonal components using rolling/expanding window PCA, enforcing eigenvector sign alignment over consecutive steps.
*   **Core Engine** ([`ev_engine.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/ev_engine.py)): Implements walk-forward Hidden Markov Model (HMM) fits, deterministic state mapping (by variance/VIX to prevent label switching), and GMM (Gaussian Mixture Model) conditional forward return density estimates to calculate quantitative Expected Values (EV) for OTM puts.
*   **V2 Evidence Contract** ([`regime_market_data.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_market_data.py)): Defines exact NYSE/Cboe clocks, immutable source observations, deterministic input hashes, and policy-bound provenance metadata.
*   **V2 Provider Gateway and Parsers** ([`regime_market_data_gateway.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_market_data_gateway.py), [`regime_provider_evidence.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_provider_evidence.py)): A caller-invoked, bounded Massive SPY/Cboe VIX acquisition library. It captures exact decoded parser-input bytes before retry decisions, retains credential-free fetch receipts, and derives strict parser receipts. It has no scheduler, import-time network call, or execution link.
*   **V2 Evidence Store** ([`regime_evidence_store.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_evidence_store.py)): Persists raw BLOBs, source attempts/health, fetch and parser receipts, append-only corrections, and channel-scoped verified snapshots in SQLite. It replays retained bytes before a `shadow` publication; one-leg failures preserve the previously verified head.
*   **V2 Shadow Detector** ([`regime_detector_v2.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_detector_v2.py)): Produces independent background and shock states from causal daily SPY/VIX inputs. Every current output is execution-ineligible.
*   **V2 Causal Calibration** ([`regime_calibration.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_calibration.py)): Evaluates a predeclared detector grid on purged, resolved-only future-risk outcomes; binds detector/build, data-prefix, calendar, source-policy, plan, and artifact hashes; and keeps the selected profile research-only. The frozen protocol is `docs/regime_v2_calibration_plan.json`.
*   **V2 Typed Signal Contract** ([`regime_signal.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_signal.py)): Converts a structurally valid close-T detector trace into immutable background-plus-shock annotations keyed only to the exact next NYSE session. It preserves artifact, source, evidence, runtime, and causal lineage; never maps into HMM integers; never fills missing sessions; and has no action projection.
*   **V2 Shadow Publisher** ([`regime_shadow_publish.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_shadow_publish.py)): Requires an externally validated entitlement capability before provider I/O, advances only an exact newly verified decision-time snapshot, seals only a non-unavailable tail signal, and preserves the prior read model on every failure.
*   **V2 Dashboard Read Model** ([`regime_shadow_store.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/regime_shadow_store.py)): Atomically publishes and descriptor-validates an owner-only sealed signal. The authenticated dashboard consumes a fixed redacted projection from a separate same-origin endpoint; missing, unsafe, future, or stale state is explicitly unavailable and cannot authorize execution.
*   **Plotting & Diagnostics** ([`ev_plots.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/ev_plots.py)): Orchestrates visualizations of regime timelines, HMM state returns, GMM distribution fits, and Expected Value curves. It is also equipped to trigger out-of-sample calibration backtests.
*   **Regime Audit** ([`regime_probability_audit.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/scratch/regime_probability_audit.py)): A rigorous statistical audit script that merges SPY/SPX data, fits a causal walk-forward HMM, checks for statistical equivalence via Kolmogorov-Smirnov (KS) tests, audits options assignment frequencies against BS/Skew probabilities, and compiles a comprehensive audit report.
*   **V2 Legacy Replay Audit** ([`regime_detector_v2_audit.py`](file:///Users/btian/EtradePythonClient/etrade_python_client/scratch/regime_detector_v2_audit.py)): Wraps legacy cache values in an explicitly unverified snapshot, surfaces conflicts/quarantined rows, and compares the two-timescale shadow result with the old overlay.

The provider-backed route begins with an explicit `snapshot_start` bootstrap or
incremental history assembly: each candidate must cover the complete contiguous
NYSE range through its new endpoint, not merely the newest session. Observing a
response after the NYSE/Cboe close is the availability policy, not a provider
finality promise; later provider corrections append a revision and require a
new verified snapshot. The [provider entitlement gate](regime_data_provider_entitlements.md)
is unresolved, so retained provider data and all derived V2 outputs remain
shadow/research-only and cannot affect E*TRADE order eligibility.

---

## ⚡ Core Command Cheat-Sheet (Quick Reference)

Use these standard commands in the shell to run processes and refresh the UI artifacts:

### Backtesting Commands
```bash
# Run the complete batch strategy comparison sweep (clears log by default)
python backtesting/run_comparison.py

# Run the sweep but append results instead of clearing the leaderboard
python backtesting/run_comparison.py --retain-existing-results

# Run a single backtest variant directly (e.g. baseline_put_spread from 2020-2026)
python backtesting/backtest_runner.py --strategy baseline_put_spread --start 2020-01-01 --end 2026-05-23 --log

# Rebuild the experiments dashboard leaderboard from the log file
python backtesting/generate_experiments_report.py
```

### Regime Detection & EV Diagnostic Commands
```bash
# Plot the 2015-Present Causal HMM Regime Timeline (SPY & VIX overlay)
python live_trading/ev_plots.py --timeline

# Plot Daily Return Histograms and Student-t fits bucketed by HMM state
python live_trading/ev_plots.py --distributions

# Plot the Causal Timeline along with BIC-Selected GMM Fits on Log Returns
python live_trading/ev_plots.py --regime-log-return-gmm

# Plot GMM Sub-Regime Cluster Scatterplots (Horizon MAE Returns)
python live_trading/ev_plots.py --gmm-plots

# Plot GMM Mixture Component Density Curves vs Horizon Returns
python live_trading/ev_plots.py --gmm-dist
```

### Quantitative Audit Commands
```bash
# Run the SPY vs SPX quantitative regime probability & option assignment audit
python scratch/regime_probability_audit.py

# Run the read-only V2 background/shock replay (always unverified)
python scratch/regime_detector_v2_audit.py

# Reproduce the frozen R3 calibration and its 2026 case studies
python scratch/regime_detector_v2_calibrate.py
```

---

## 🔗 Backend-to-Frontend Artifact Mapping

When the user mentions a specific **dashboard**, **HTML**, or **chart**, check this mapping table to locate the file and understand which backend script generates it:

| Frontend UI / Artifact | Generated By | Primary Purpose & Contents |
| :--- | :--- | :--- |
| **[`experiments_dashboard.html`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/experiments_dashboard.html)** | `generate_experiments_report.py` | The main leaderboard. Compiles terminal statistics, margin metrics, drawdowns, Sharpe ratios, and links for all simulated variants in the log. |
| **[`backtesting/reports/report_*.html`](file:///Users/btian/EtradePythonClient/etrade_python_client/backtesting/reports/)** | `backtest_runner.py` | Single backtest interactive report. Contains step-by-step trade logs, equity curves, drawdown curves, trade-by-trade details, and embedded strategy config details. |
| **[`audit_plots/regime_probability_audit.html`](file:///Users/btian/EtradePythonClient/etrade_python_client/audit_plots/regime_probability_audit.html)** | `regime_probability_audit.py` | Interactive statistical dashboard comparing SPY & SPX. Displays distribution overlays, moments tables, and option assignment edges. |
| **[`live_trading/dashboard_template.html`](file:///Users/btian/EtradePythonClient/etrade_python_client/live_trading/dashboard_template.html)** | `RefreshHandler` serves the source template; `/api/regime_v2_shadow` reads `live_trading/runtime/regime_v2_shadow.json` | Authenticated operator dashboard. The V2 card displays a redacted background-plus-shock advisory beside the legacy lane and permanently states that it cannot authorize execution. |
| **[`research_reports/regime_v2_calibration_artifact.json`](file:///Users/btian/EtradePythonClient/etrade_python_client/research_reports/regime_v2_calibration_artifact.json)** | `regime_detector_v2_calibrate.py` | Canonical R3 candidate metrics, causal folds, selected baseline, provenance limitations, and execution-ineligible promotion status. |
| **[`s_and_p_data/regime_timeline_2015.png`](file:///Users/btian/EtradePythonClient/etrade_python_client/s_and_p_data/regime_timeline_2015.png)** | `ev_plots.py --timeline` | Visually maps out-of-sample HMM regimes (Expansion, Decline, Panic) as background colors overlaid on SPY Close and VIX. |
| **[`s_and_p_data/spy_return_distributions_by_hmm.png`](file:///Users/btian/EtradePythonClient/etrade_python_client/s_and_p_data/spy_return_distributions_by_hmm.png)** | `ev_plots.py --distributions` | Multi-panel histogram displaying daily returns bucketed by HMM state and overlaid with fitted fat-tailed Student-t densities. |
| **[`s_and_p_data/regime_log_return_timeline.png`](file:///Users/btian/EtradePythonClient/etrade_python_client/s_and_p_data/regime_log_return_timeline.png)** | `ev_plots.py --regime-log-return-gmm` | Causal regime timeline from 2015-Present including posterior HMM state probability stacked areas. |
| **[`s_and_p_data/regime_log_return_gmm_fits.png`](file:///Users/btian/EtradePythonClient/etrade_python_client/s_and_p_data/regime_log_return_gmm_fits.png)** | `ev_plots.py --regime-log-return-gmm` | Subplot layout displaying BIC-selected Gaussian Mixture density components fitted over each HMM state's daily log returns. |
| **[`/Users/btian/.gemini/antigravity/artifacts/gmm_regime_clusters.png`](file:///Users/btian/.gemini/antigravity/artifacts/gmm_regime_clusters.png)** | `ev_plots.py --gmm-plots` | Scatterplots of horizon returns over time, colored by their sub-component classification to reveal internal sub-regimes. |
| **[`/Users/btian/.gemini/antigravity/artifacts/gmm_distribution_fits.png`](file:///Users/btian/.gemini/antigravity/artifacts/gmm_distribution_fits.png)** | `ev_plots.py --gmm-dist` | Empirical density histograms of horizon returns overlaid with multi-component Gaussian mixture probability curves. |

---

## 📜 Causal Operating Contract

When writing code or verifying backtests/plots, ensure the following core quantitative boundaries are **never** breached:

> [!WARNING]
> **No Look-Ahead Volatility Centering**
> All moving averages, standard deviations, or volatility feature smoothing (e.g. VIX or realized vol filters) must be strictly trailing (`center=False`). Centering a rolling filter leaks future information.

> [!CAUTION]
> **Walk-Forward Validation Only**
> Do not use global scaler transformations or Viterbi-smoothed histories (`model.predict()`) to generate historical backtest regime states. You must run causal walk-forward scaling and out-of-sample forward filtering (`model.predict_proba()[-1]`), lagging daily states by 1 trading day before trade entry to mimic real-world execution.
