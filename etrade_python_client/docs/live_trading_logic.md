# Core Trading Logic: Live Trading Agent

This document outlines the specific trading logics currently implemented in the `live_trading` directory, primarily derived from the master `etrade_cover_call_new.py` execution script.

## 1. Operating Window & Market Gating
- **Trading Window:** The core trading loop is active between `07:15:00` and `13:30:00` Eastern Time.
- **Market State Gate:** Operations are halted on holidays, weekends, or outside extended hours. Trades are skipped if the system detects the market as `CLOSED_HOLIDAY`. 

## 2. Dynamic Target Premium & VIX Adjustment
- The algorithm calculates a `days_to_expire` target (typically ~6 weeks out, maturing on a Friday).
- **Baseline Premium Targeting:** Baseline premiums scale with the square root of the days to expiration (`days_to_expire ** 0.5`). Additionally, put spreads receive a `(1 + target_steer)` multiplier, and call spreads receive a `(1 - target_steer)` multiplier. 
- **VIX Adjustment:** 
  - The system checks the current VIX level against a baseline threshold (typically `20`).
  - `adjust = vix_correlation` (e.g., typically `0.05`).
  - Target call premiums are scaled down dynamically as VIX rises above the threshold (anticipating market rebounds).
  - Target put premiums are scaled up dynamically as VIX rises, increasing collected premium for downside protection during volatility spikes. 

## 3. Position Filtering & Strike Selection
- **Earnings Date Skips:** The agent fetches known earnings dates for the underlying ticker. If an earnings event falls between the current date and the target expiration date, it automatically skips opening new positions for that ticker.
- **Iterative Search:** Uses current price and days to expiration to scan the option chain for spreads matching the dynamically calculated target premium. If no matches are found, it decrements the target days to expiration by 7 days and tries again until a valid spread setup is found or `days_to_expire < 2`.
- **Profitability Constraint:** Proposed spread orders are rejected locally if the net credit generated per spread drops below `$0.01` during the final price verification.

## 4. Risk Management & Extrinsic Value Monitoring
- **Extrinsic Value Alerts:** Scans all active short Option positions that are In-The-Money (ITM). If the Extrinsic Value drops to `$1.00` or below (heightened assignment risk), it automatically triggers an email alert.
- **Same-Day ITM Put Spread Market Auto-Close:** Specifically triggers after `12:50:00` ET. The script scans expiring put credit spreads. If the short leg is ITM (underlying price below short put strike), the system automatically attempts to close the entire spread at `MARKET` to prevent assignment.

## 5. Early Profit Taking (High-Gain Spread Detection)
- Continuously scans the portfolio for active spreads demonstrating strong profitability.
- If it detects a spread with a `gain_loss_percentage > 70%`, it extracts the current Bid/Ask midpoints for the corresponding legs and automatically proposes an order to close the spread at a net debit matching the midpoint.

## 6. Execution & Auto-Adjustment Workflows
- **Approval Flow:** Proposed orders (both opening and high-gain closing) are batched, sorted by Annualized ROI, and sent as an email payload.
- **Execution Validation:** Before actually transacting after an approval, it checks if the refreshed market prices for the spread still result in a valid credit setup. 
- **Auto-Adjustment:** Once an order is placed, if it is not immediately filled, the system initiates a monitoring loop that adjusts the order's limit price by `$0.01` every `30` seconds (up to ~90 minutes) until a fill is achieved.
- **Margin Check Release:** If an order triggers an `INSUFFICIENT_FUNDS` error, the agent invokes a margin-release function, conditionally cancelling existing resting orders to free up buying power, before retrying.
