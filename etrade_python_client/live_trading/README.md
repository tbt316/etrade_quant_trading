# E*TRADE Live Trading Dashboard & Bot

This module contains the live trading agent and its associated web-based management dashboard.

## 🚀 Quick Start

1. **Provision safety inputs**:
   Configure client credentials outside Git using local `config.ini` or environment variables, and create an owner-only `live_trading_settings.json` with a non-default dashboard username, a 16+ character password, and an 8+ digit action PIN. Production also requires `ETRADE_PRODUCTION_ARMING_SECRET` and an owner-only, signed, unexpired production-arm document.

2. **Start the trading bot with an explicit environment and account identity**:

   ```zsh
   python3 -m live_trading.etrade_cover_call_new \
     --environment sandbox \
     --expected-account-id '<display-id>' \
     --expected-account-id-key '<account-key>' \
     --expected-institution-type '<institution-type>' \
     --trade
   ```

   For production, issue a fresh arm document immediately before startup. The secret is read only from the environment; do not put it in the command, settings file, or Git.

   ```zsh
   export ETRADE_PRODUCTION_ARMING_SECRET='<32+-character secret>'
   python3 -m live_trading.runtime_safety issue-production-arm \
     --output .production-arm.json \
     --expected-account-id '<display-id>' \
     --expected-account-id-key '<account-key>' \
     --expected-institution-type '<institution-type>' \
     --ttl-seconds 300

   python3 -m live_trading.etrade_cover_call_new \
     --environment production \
     --production-arm-file .production-arm.json \
     --expected-account-id '<display-id>' \
     --expected-account-id-key '<account-key>' \
     --expected-institution-type '<institution-type>' \
     --trade
   ```

   The generated document is owner-only (`0600`), binds the exact account identity, and is valid for at most 15 minutes. A missing or ambiguous environment never defaults to production.

3. **Access the dashboard locally**:
   Open `http://localhost:8765/dashboard` in your browser. The server binds to loopback only; it does not create a public tunnel.

---

## 🔒 Remote access

The order-capable dashboard deliberately has no automatic public tunnel. If remote access is later approved, place it behind a separately managed TLS reverse proxy with strong authentication, request-rate controls, and network allowlisting. Do not expose the local HTTP listener directly.

## 📊 Dashboard Features

*   **Live Settings**: Adjust Target Delta, Spread Width, and Trade Quantities in real-time without restarting the bot.
*   **Auto-Open Toggle**: Enable or disable the automatic opening of new positions.
*   **Position Monitoring**: View your active portfolio and profit/loss metrics with Chart.js visualizations.
*   **Manual Execution**: Override the bot to execute a specific credit spread immediately based on your current settings.
*   **Auto-Close Thresholds**: Set specific Gain % and Cost thresholds for automatic closing of profitable positions.

---

## ⚙️ Configuration

*   **Settings**: `live_trading_settings.json` stores dynamic parameters and must be owner-only (`0600`). It is not a place for E*TRADE OAuth credentials.
*   **E*TRADE credentials**: configure sandbox values through `ETRADE_SANDBOX_CONSUMER_KEY` / `ETRADE_SANDBOX_CONSUMER_SECRET` and production values through `ETRADE_LIVE_CONSUMER_KEY` / `ETRADE_LIVE_CONSUMER_SECRET`, or matching local `config.ini` entries. Never commit values.
*   **Refresh Frequency**: 
    *   Main Loop Evaluation: Every 60 seconds.
    *   Stale Order Monitoring: Every 120 seconds.
    *   UI Status Polling: Every 60 seconds.

---

## 🛠️ Components

*   `etrade_cover_call_new.py`: The core trading engine and HTTP server logic.
*   `dashboard_template.html`: Premium Glassmorphic UI for bot management.
*   `spy_position_tracker.py`: Handles historical PnL and position snapshotting.
*   `ev_engine.py`: Markov-Switching GMM engine for expected value calculations.
