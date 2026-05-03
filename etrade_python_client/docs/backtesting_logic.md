# Core Trading Logic: Backtesting Engine

This document outlines the specific trading logics implemented in the `backtesting` directory, primarily derived from `polygonio_dailytrade.py` and its core simulation loop.

## 1. Simulation Engine State & Loop
- **Day-by-Day Loop:** The simulator iterates chronologically over marked historic trading days. On each day, it evaluates stop-profits and dynamically opens new spreads simulating the time progression.
- **API Polling & Pricing Sources:** To compute mark-to-markets and test fills, the script queries Polygon.io options APIs. The exact polling method depends on the configured `PREMIUM_PRICE_MODE`:
  - `trade` mode: Queries `https://api.polygon.io/v3/trades/{option_symbol}`. It fetches the actual **Historical Trade** data prints executed on that day (filtering by `limit: 10`, `order: desc`, `sort: timestamp` for the specified `pricing_date`).
  - `mid` mode: Queries `https://api.polygon.io/v3/quotes/{option_symbol}`. It pulls the **NBBO (National Best Bid and Offer) Quotes** published by the broker/exchanges at that time and derives the theoretical fill price as `(bid + ask) / 2`.
  - `close` mode: Queries `https://api.polygon.io/v1/open-close/...`. It relies on the daily aggregated EOD official closing price.
- **Time of Day Polled:** Because the Polygon API requests specify `timestamp=YYYY-MM-DD` coupled with a descending `timestamp` sort order, the backtester effectively pulls the **End-of-Day (EOD)** pricing snapshot (i.e., the last available quote or trade occurring exactly at market close, typically 16:00 or 16:15 ET). *Note: This is a critical distinction from the live trading agent, which polls repeatedly at various increments throughout the live intraday trading window (e.g., 07:15 to 13:30).*
- **End-of-Day Settlement:** Positions that reach their expiration date evaluate their terminal payoff profiles (e.g. `max(close_price - strike, 0)`). PnLs are settled automatically against the realized closing price of the underlying on the expiration date, deducting those figures from the initial collected credits.

## 2. Dynamic Premium Targeting & VIX Sizing
- Similar to the live trader, the backtest system uses the VIX closing print on the entry date to set dynamic targets.
- **VIX Scaling Baseline:** 
  - Call Target Premium = `baseline / 1.5` or `baseline * (1 - (vix_value - vix_threshold) * adjust)`.
  - Put Target Premium = `baseline / 1.5` or `baseline * (1 + (vix_value - vix_threshold) * adjust)`.
- **Target Type Modes:** Allows toggling between solving for a specific `delta_target` vs solving for a `premium_target`.

## 3. Strike Selection Constraints & Search
- **Bidirectional Best-Fit Search:** The algorithm sweeps outwards from the At-The-Money (ATM) strike, interpolating up and down the chain to locate the tightest match for the targeted premium or delta criteria.
- **OTM Filtering:** Enforces an absolute rule strictly checking that options selected are Out-of-the-Money (`call_strike >= stock_price` and `put_strike <= stock_price`). 
- **Monotonicity Check (`enforce_mid_monotonic`):** Ingests raw chain data and filters out pricing anomalies by strictly enforcing that Call bid/ask midpoints must strictly non-increase as strikes ascend, and Put midpoints must non-increase as strikes descend.
- **Volume/Liquidity Thresholds:** The engine requires selected strikes to maintain a minimum trading volume threshold. If the theoretically closest target strike fails this floor, the system will explicitly select a slightly worse matching strike that clears the liquidity requirement.

## 4. Mark-to-Market & Missing Interpolation
- During the middle of a trade's lifecycle, the backtester continuously monitors the "Price to Close" (MTM cost) to validate intra-trade conditions.
- **Missing Data Interpolation:** Because deep OTM or thinly traded options lack clean daily pricing quotes, the engine employs a complex interpolation methodology `interpolate_option_price()`. It anchors on the underlying price and triangulates the theoretical premium value based on nearby liquid strikes to calculate accurate daily position mark-to-markets.

## 5. Intra-Trade PnL Exits (Stop Profit / Stop Loss)
- The engine actively tests `close_call_cost` and `close_put_cost` values each day.
- **Stop Profit Execution:** 
  - If the absolute cost to close drops to `current_cost <= stop_profit_percent * entry_credit`. (Meaning you can buy it back for pennies relative to what you collected).
  - *Hard-coded constraint check:* Alternatively, some strategies define very strict `0.20 * entry_credit` stop profit mandates if a `hold_to_expiration` flag is raised but the market aggressively decays the option early.
- **Time-Decay Taking Profit:** If the option reaches half of its initial life-span left (`expiring_soon`), the exit tolerance expands, dynamically allowing trades to exit smoothly even if they only hit `current_cost <= 2.0 * entry_credit`.
- **Stop Loss Execution:** 
  - If market action causes `current_cost >= 50 * entry_credit`, the leg triggers a catastrophic stop-loss tag (`call_closed_by_stop = True`), exiting immediately.

## 6. Logic Parity Notables
- The core dynamic premium formulas and VIX scaling mechanisms exactly mirror the `live_trading` parameters.
- Both systems aggressively target 6-week (typically Friday) OTM credit spread collections.
- Unlike `live_trading`, the `backtester` enforces much stricter raw option-chain mathematical sanitation protocols (Monotonicity and Interpolation checks) to avoid taking advantage of bad historical datasets that wouldn't actually fill in reality.
