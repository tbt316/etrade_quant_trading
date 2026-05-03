# Backtest Strategy Registry

This file is the single source of truth for all backtesting strategies.
The backtest runner (`backtest_runner.py`) reads this file to configure strategy execution.

## How to Add a New Strategy

1. Copy the template below and fill in all required fields.
2. Add the strategy block to the **Strategy Catalog** section.
3. Implement the corresponding entry/exit logic in `backtest_runner.py` (register in `STRATEGY_DISPATCH`).

---

## Strategy Schema

Each strategy is defined by a YAML-like block with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique snake_case identifier |
| `name` | string | ✅ | Human-readable name |
| `description` | string | ✅ | What the strategy does |
| `instrument` | string | ✅ | `put_credit_spread`, `call_credit_spread`, `iron_condor`, etc. |
| `underlying` | string | ✅ | Default underlying ticker |
| `entry.method` | string | ✅ | How the short strike is selected: `fixed_delta`, `ev_optimized`, `fixed_strike_pct` |
| `entry.target_dte` | int | ✅ | Target days to expiration at entry |
| `entry.short_delta` | float | ❌ | Required if method is `fixed_delta` |
| `entry.spread_width` | float | ✅ | Width of the spread in dollars |
| `exit.method` | string | ✅ | `fixed_dte`, `profit_target`, `trailing_stop`, `expiration` |
| `exit.close_dte` | int | ❌ | Required if exit method includes `fixed_dte` |
| `exit.early_profit_pct` | float | ❌ | Close early if this % of max profit is captured |
| `sizing.method` | string | ✅ | `fixed_qty`, `pct_of_nlv`, `kelly` |
| `sizing.value` | float | ✅ | Qty if `fixed_qty`, fraction if `pct_of_nlv` |
| `filters` | list | ❌ | Optional filters: `min_volume`, `min_credit`, `max_spread_width_pct` |
| `status` | string | ❌ | `implemented` or `planned`. Defaults to `implemented`. |
| `regime_aware` | bool | ❌ | If `false`, skips the HMM regime detection process (speeds up backtest). Defaults to `true`. |

### Global Operational Rules

1. **Immediate Release**: Margin budget from positions that are closed (early profit/scheduled) or expired (at 0 DTE) is released **on the same day**. This budget is immediately available for opening new positions in the same daily loop.
2. **Cash-Based Sizing**: The maximum allowed margin (`allowed_margin`) is calculated based on the current cash balance (`current_cash * margin_limit_pct`), not Net Liquidation Value (NLV). This prevents unrealized losses from prematurely forcing a reduction in exposure.
3. **Margin Limit**: The `margin_limit_pct` is a hard cap on the sum of `margin_required` for all *currently open* trades.
4. **ITM Rolling Rule**: When a position is less than or equal to 7 days from expiration and is in-the-money (ITM), it must be rolled into a future expiration.
   - **Fixed Delta Strategy**: Find the target option by using the *current* delta of the opened position (at the time of the roll, not entry).
   - **Adaptive Strategy** (e.g., dynamic delta/spread multiplier): Follow strategy-specific settings for finding the new position, applying the delta multiplier to the *current* delta.
   - **Special Delta Handling**:
     - If the current delta is between -1.0 and -0.5 (exclusive/inclusive as per market conditions), set the target delta to -1.0.
     - If the current delta is already close to -1.0, roll into the same strike price in the target expiration.

---

## Data Integrity & Institutional Memory

### 1. Dividend Adjustment (SPY Spot Price & Pricing Models)
- **Rule**: NEVER use "Adjusted Close" for the underlying `spot_price` during historical backtests.
- **Reason**: Option strikes in historical databases (Massive API, Polygon) are almost always **unadjusted**. If the backtester uses an adjusted SPY price (e.g., $170 in 2015 when the real price was $205), an OTM strike like $185 will appear deep ITM.
- **Implementation**:
    - Ensure `yf.download(..., auto_adjust=False)` is used.
    - Black-Scholes models (pricing and Greeks) must explicitly incorporate `dividend_yield` ($q$) to account for the deterministic drag on spot prices over long DTEs.
- **Verification**: Check 2015-01-02 SPY price (~$205.43). Run `scratch/test_greeks_failure.py` to verify $q$ impact.

### 2. Causal Feature Preparation (Expanding Windows)
- **Rule**: Feature scaling (RobustScaler) and dimensionality reduction (SparsePCA) must be computed in an **expanding-window (causal)** manner during all backtesting and model training.
- **Reason**: Global fitting (e.g. `scaler.fit(full_dataset)`) allows information from the future (e.g. the 2020 COVID crash) to contaminate historical data points (e.g. 2015). This look-ahead bias results in artificially stable regime detection and inflated backtest performance.
- **Implementation**:
    - Use `DataIngestor.scale_features(df, expanding=True)`.
    - Ensure `train_regime_hmm(expanding_window=True)` is used for all historical analysis.
- **Verification**: Run `scratch/verify_scaling_fix.py`.

---

## Strategy Catalog

---

### Strategy: `fixed_delta_put_spread`

```yaml
id: fixed_delta_put_spread
name: Fixed-Delta Put Credit Spread
status: implemented
regime_aware: false
description: >
  Sells a put credit spread at a fixed short-leg delta target.
  Uses Black-Scholes delta from OHLCV close prices to select strikes.
  If the position is in-the-money (ITM) at the scheduled close date (half-DTE),
  it is held until expiration instead of being closed, even if it results in assignment.
  This is the baseline mechanical strategy — no GMM/EV optimization.
instrument: put_credit_spread
underlying: SPY

entry:
  method: fixed_delta
  target_dte: 42
  short_delta: -0.15
  spread_width: 20.0
  risk_free_rate: 0.05

exit:
  method: fixed_dte + profit_target
  close_dte: 21
  early_profit_pct: 0.70
  hold_itm_to_expiration: true

sizing:
  method: pct_of_nlv
  value: 0.50
  margin_limit_pct: 0.50

filters:
  min_chain_strikes: 10
  min_credit: 0.01
  strike_tolerance_pct: 0.30
```

**Implementation:** `backtest_runner.py` → `run_put_credit_spread_backtest()`

**Status:** ✅ Implemented and tested

---

### Strategy: `ev_optimized_put_spread` (PLANNED)

```yaml
id: ev_optimized_put_spread
name: EV-Optimized Put Credit Spread
status: planned
description: >
  Uses the GMM probability engine to select the long-leg delta that
  maximizes EV/ES (execution edge) for each entry. The short leg is
  still selected by fixed delta, but the spread width is dynamic.
  This strategy tests whether the GMM engine adds alpha over the
  fixed-delta baseline.
instrument: put_credit_spread
underlying: SPY

entry:
  method: ev_optimized
  target_dte: 42
  short_delta: -0.15
  spread_width: dynamic  # GMM engine selects optimal width
  risk_free_rate: 0.05
  ev_engine:
    model: gmm
    horizon: auto  # matches DTE
    regime_source: hmm
    expanding_window: true  # prevents look-ahead bias

exit:
  method: fixed_dte + profit_target
  close_dte: 21
  early_profit_pct: 0.70

sizing:
  method: pct_of_nlv
  value: 0.25
  margin_limit_pct: 0.25

filters:
  min_ev_per_contract: 0.0  # only enter positive-EV trades
  min_chain_strikes: 10
  min_credit: 0.01
```

---

### Strategy: `dynamic_delta_variant`

```yaml
id: dynamic_delta_variant
name: Dynamic Delta Variant (Panic Swapper)
status: implemented
regime_aware: true
description: >
  A regime-aware strategy that aggressively adjusts risk in high-volatility environments.
  When entering the 'Panic / Crisis' regime, a "Panic Swap" is triggered (if `panic_swap_enabled` is true):
  - All open positions are immediately closed.
  - Every closed position is replaced 1:1 on the same day with a new trade using panic parameters (higher delta, wider spreads, longer DTE).
  - The regular daily scheduled entry continues uninterrupted, ensuring exposure is maintained.
  - The `panic_qty_multiplier` (default 0.5x) is applied to all entries while in the panic regime.
  - The `panic_swap_enabled` flag (default false in registry) determines if existing positions are force-closed.
instrument: put_credit_spread
underlying: SPY

entry:
  method: dynamic_delta_variant
  target_dte: 42
  short_delta: -0.12
  spread_width: 20.0
  panic_delta_multiplier: 1
  panic_width_multiplier: 1
  panic_qty_multiplier: 1
  panic_swap_enabled: false
  risk_free_rate: 0.05

exit:
  method: fixed_dte + profit_target
  close_dte: 21
  early_profit_pct: 0.7
  hold_itm_to_expiration: true

sizing:
  method: pct_of_nlv
  value: 0.50
  margin_limit_pct: 0.50
```

---

## Loading Strategies in Python

The backtest runner loads strategies from this file using the helper below:

```python
# In backtest_runner.py:
from backtesting.strategy_loader import load_strategy

config = load_strategy("fixed_delta_put_spread")
# config = {"id": "fixed_delta_put_spread", "entry": {"method": "fixed_delta", ...}, ...}
```

The `strategy_loader.py` module parses the YAML blocks from this markdown file
and returns them as Python dicts. See `strategy_loader.py` for implementation.
