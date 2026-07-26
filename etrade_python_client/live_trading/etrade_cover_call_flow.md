# E*TRADE Live Trading Agent Execution Flow

This document outlines the execution logic of `etrade_cover_call_new.py`. The trading agent acts as the backend for the local dashboard (`http://localhost:8765`), which serves as the primary UI and source of truth for all trading parameters.

## 1. Startup & Initialization
- **Argument Parsing**: Reads `--trade`, `--sandbox`, `--username`, `--password`, etc.
- **Authentication**: Performs OAuth login with E*TRADE. Supports auto-login via iMessage/Browser if configured.
- **Account Selection**: Explicitly selects the designated brokerage account (e.g., Roth IRA ending in 8703) during initialization.
- **Background Services**: 
    - Starts an HTTP server (default port `8765`) to host the interactive Dashboard.
    - Handles `/api/settings` to load/save user configurations.
    - Handles `/api/preview_spread` and `/api/execute_manual_order` for manual trading.
- **Time Setup**: Defines trading windows (07:15 - 13:30 PT) and log windows.

## 2. The Dashboard Interaction Model
The dashboard (`dashboard_template.html`) is the **absolute source of truth** for both manual and automated position openings. 

- **Settings Management**: User configurations (Target Delta, Spread Width, Target Weeks, Auto-Open toggle, Trading Side) are stored in `live_trading_settings.json`.
- **Live Preview**: The dashboard constantly queries `/api/preview_spread` to show the user the exact spread (Call, Put, or Both) that matches their current slider settings based on live E*TRADE quotes.
- **Parameter Locking (Manual Trades)**: When the user clicks **"Execute Order Now"**, the dashboard sends the exact `sell_strike`, `buy_strike`, `expiration`, and `side` to the backend. The backend **bypasses any background search logic** and exclusively executes those specific strikes.
- **Parameter Loading (Auto Trades)**: If Auto-Open is enabled, the backend reads `live_trading_settings.json` at the start of the trading day and searches for candidates using those precise UI-defined parameters.

## 3. Main Execution Loop
The script runs in an infinite loop, performing the following steps:

### A. Session & Status Maintenance
- **Session Renewal**: Renews the E*TRADE access token every 60 minutes.
- **Market Status Gate**: Checks if the NYSE is `OPEN`, `PRE_MARKET`, `AFTER_HOURS`, or `CLOSED`.
    - **Holiday/Weekend**: Sleeps until the next trading day.
    - **Pre/After Hours**: Updates the HTML dashboard but skips active trading.
    - **Manual Trade Override**: If a manual trade is requested via the dashboard, the engine immediately bypasses sleep/time gates to execute the trade.
- **Data Refresh**: Fetches account balance, full portfolio, margin usage, and calculates portfolio-wide Net Delta.

### B. Risk Monitoring & Margin Tracking
- **Extrinsic Value Alerts**: Scans for ITM short options with **<$1.00 extrinsic value**. Sends an email warning of assignment risk.
- **Stale Limit Order Nudging**: Monitors open limit orders. If an order remains unfilled, it nudges the limit price by $0.01 every 30 seconds to chase the market, and it will keep following favorable price movement instead of blindly reverting a better price.

### C. Automated Management Actions
- **Expiring ITM Put Close**: After 12:50 PM PT, the script automatically sends market orders to close any ITM put spreads expiring today to avoid assignment.
- **High-Gain Close Proposals**:
    - Detects spreads reaching the UI-defined **Auto-Close Target** (e.g., closing cost < $0.30 or >70% profit).
    - Calculates mid-price from live quotes.
    - Sends an **Approval Email** to the user.
    - If approved (via dashboard or email), executes the closing trade with the price-chasing algorithm.

### D. New Trade Entry (Auto or Manual)
- **Trigger**: Runs if `auto_open_enabled` is true (and no trade has occurred today) OR if `is_manual_trade` is true.
- **Parameter Source**:
    - **Manual**: Extracts explicit strikes and expirations sent from the dashboard.
    - **Auto**: Loads `target_delta`, `target_weeks`, `hedge_spread`, and `trade_side` (Call/Put/Both) from `live_trading_settings.json`.
- **Earnings Filter**: For auto-trades, skips tickers with earnings announcements prior to the target expiration.
- **Strategy Selection**:
    - Generates order JSON based purely on the defined/requested side (Call or Put).
- **Approval Workflow**:
    - For Auto-Trades: Sends an Order Approval Email with ROI and EV stats, pausing for up to 30 minutes.
    - For Manual Trades: Skips email approval (since the user clicked "Execute Now" on the dashboard).
- **Order Execution**:
    - Refreshes prices immediately before submission.
    - Places Limit orders.
    - Engages the **Price Chasing (Nudge) logic** to ensure execution if the market moves away from the initial limit price.

### E. Error Handling & Logging
- **Retries**: Implements exponential backoff for E*TRADE API rate limits (e.g., "Too Many Requests").
- **Logging**: Detailed rotation logs in `python_client.log`.
- **Notifications**: Sends Gmail notifications for login failures or critical errors.
