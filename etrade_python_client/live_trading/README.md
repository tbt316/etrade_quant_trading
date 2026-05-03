# E*TRADE Live Trading Dashboard & Bot

This module contains the live trading agent and its associated web-based management dashboard.

## 🚀 Quick Start

1. **Start the Trading Bot**:
   Run the main script with the `--trade` flag. This will automatically start the background HTTP server on port `8765`.
   ```zsh
   python3 -m live_trading.etrade_cover_call_new --no-sandbox --trade
   ```

2. **Access the Dashboard**:
   * **Local**: Open `http://localhost:8765/dashboard` in your browser.
   * **PIN**: Default PIN is `1234` (configurable in `live_trading_settings.json`).

---

## 🌐 Public Access Setup (ngrok)

To access your dashboard from your phone while away from home, use **ngrok** to create a secure tunnel.

### 1. Installation
```zsh
brew install ngrok/ngrok/ngrok
```

1. Open `config.ini` in the root directory.
2. Add your token to the `[NGROK]` section:
   ```ini
   [NGROK]
   AUTH_TOKEN = your_token_here
   STATIC_DOMAIN = your-name.ngrok-free.app
   ```
3. Run the authentication command once:
   ```zsh
   ngrok config add-authtoken <YOUR_AUTH_TOKEN>
   ```

### 3. Claim a Static Domain
In your ngrok dashboard, go to **Cloud Edge -> Domains** and claim your free static domain (e.g., `your-name.ngrok-free.app`).

### 4. Run the Tunnel
```zsh
ngrok http --url=your-name.ngrok-free.app 8765
```

---

## 📊 Dashboard Features

*   **Live Settings**: Adjust Target Delta, Spread Width, and Trade Quantities in real-time without restarting the bot.
*   **Auto-Open Toggle**: Enable or disable the automatic opening of new positions.
*   **Position Monitoring**: View your active portfolio and profit/loss metrics with Chart.js visualizations.
*   **Manual Execution**: Override the bot to execute a specific credit spread immediately based on your current settings.
*   **Auto-Close Thresholds**: Set specific Gain % and Cost thresholds for automatic closing of profitable positions.

---

## ⚙️ Configuration

*   **Settings**: `live_trading_settings.json` stores all dynamic parameters.
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
