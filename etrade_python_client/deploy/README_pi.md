# Raspberry Pi Live Runtime

This repo can stay edited on the MacBook while the live dashboard and E*TRADE bot run on a Raspberry Pi. The intended shape is:

1. MacBook Pro: editing, testing, and pushing/syncing code.
2. Raspberry Pi: always-on runtime for `python -m live_trading.etrade_cover_call_new --no-sandbox --trade`.
3. `systemd`: keeps the process alive and restarts it after Pi reboot or process failure.
4. `ngrok`: still comes from `config.ini` and is started by the Python process when `[NGROK]` is configured.

No trading logic changes are required for this deployment model.

## Success Criteria

- Closing the MacBook does not stop the dashboard.
- The Pi serves `http://<pi-ip>:8765/dashboard` on the local network.
- If `config.ini` has `[NGROK] AUTH_TOKEN` and `STATIC_DOMAIN`, the public dashboard URL stays hosted by the Pi.
- `sudo systemctl status etrade-live` shows the bot running.
- `journalctl -u etrade-live -f` shows the same startup/login/runtime output that used to appear in the Mac terminal.

## One-Time Pi Setup

On the Pi, place or clone the repo somewhere stable, for example:

```bash
mkdir -p /home/pi/etrade_python_client
```

From the Mac, sync the current working tree to that directory:

```bash
./deploy/sync_to_pi.sh
```

Then on the Pi:

```bash
cd /home/pi/etrade_python_client
./deploy/pi_bootstrap.sh
```

The bootstrap creates `.venv`, installs the Python requirements used by the live bot, and installs Chromium/chromedriver packages when available.

The current Raspberry Pi 4 checked at `pi@raspberrypi.local` is running Raspbian 10 buster with Python 3.7.3. That OS is too old for the pinned live runtime dependencies. Use Raspberry Pi OS 64-bit Bookworm, or install Python 3.10+ manually and run bootstrap with:

```bash
PYTHON_BIN=/path/to/python3.10 ./deploy/pi_bootstrap.sh
```

## Private Runtime State

The live bot needs files that are intentionally not committed:

- `config.ini`
- `.env`, if used
- `.etrade_oauth`, if present
- `live_trading_settings.json`
- `trade_status.json`
- `manual_order_status.json`
- `spy_tracking_data.json`
- `spy_vix_price_cache.json`
- `nudge_history.json`, if present
- `nudge_state.json`, if present

Sync those from the Mac only when you intentionally want the Pi to inherit the current runtime state:

```bash
./deploy/sync_to_pi.sh --state
```

Do this before the first service start. After the Pi is live, treat the Pi as the source of truth for runtime state files that change during trading hours.

## Install The Always-On Service

On the Pi:

```bash
cd /home/pi/etrade_python_client
./deploy/install_pi_service.sh
sudo systemctl start etrade-live
```

Check status and logs:

```bash
sudo systemctl status etrade-live
journalctl -u etrade-live -f
```

Stop or restart:

```bash
sudo systemctl stop etrade-live
sudo systemctl restart etrade-live
```

The installer enables the service at boot. It does not start the bot unless you start it manually or run:

```bash
START_NOW=1 ./deploy/install_pi_service.sh
```

## Normal Edit/Deploy Loop From The Mac

Edit on the MacBook, then sync code and restart the Pi service:

```bash
./deploy/sync_to_pi.sh --restart
```

This sync intentionally excludes secrets, OAuth tokens, logs, virtualenvs, and generated runtime files. Use `--state` only for deliberate private-state migration.

## First Login Notes

If `.etrade_oauth` copied from the Mac renews successfully on the Pi, startup should continue without a fresh browser login. If renewal fails, the bot will use the existing automated login path from `core_api/stock_trade_class.py`; the Pi therefore needs browser support from the bootstrap step and valid credentials in `config.ini` or `.env`.

For the first migration, start the service while you can watch logs:

```bash
journalctl -u etrade-live -f
```

Confirm `/api/status` and the dashboard before relying on the Pi during market hours.
