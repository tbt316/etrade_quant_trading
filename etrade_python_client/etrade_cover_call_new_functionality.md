# etrade_cover_call_new.py — Functionality Specification

## Overview

`etrade_cover_call_new.py` is the **main entry-point script** for an automated E\*TRADE options trading system. It continuously monitors the market, opens credit spreads (call and put) across a configurable list of tickers, detects high-gain spreads for early closure, and manages order execution — all gated behind an **email-based approval workflow**.

The script runs as a **long-lived loop** that:
1. Authenticates with E\*TRADE via OAuth 1.
2. Checks NYSE market status (holiday / pre-market / open / after-hours).
3. On trading days within the configured trading window, opens new spread positions once per day.
4. Continuously monitors existing positions for high-gain close opportunities.
5. Generates an HTML dashboard (`screened_option_pairs.html`) with live position data.
6. Accepts manual refresh requests via a built-in HTTP server.

---

## Configuration & Dependencies

### External Config
- **`config.ini`** — holds OAuth keys (`SANDBOX_CONSUMER_KEY`, `PROD_CONSUMER_KEY`, etc.), email credentials (`[EMAIL]` section: `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `NOTIFY_EMAIL`, `APPROVAL_TIMEOUT_MINUTES`).

### Key Internal Modules
| Module | Purpose |
|---|---|
| `accounts.accounts_bo.Accounts` | Portfolio, balance, option chain queries, order generation, screening |
| `market.market_bo.Market` | Real-time quote data |
| `stock_trade_class.LiveTradeAgent` | Order placement, gain tracking, price adjustment |
| `polygon_multi.load_latest_parameters` | Load optimized trading parameters per ticker |
| `option_assign_probability.calculate_probability` | Estimate assignment probability for a spread |
| `polygonio_improvequery.get_earnings_dates` | Earnings-date lookup to avoid trading around earnings |
| `spy_position_tracker` | Daily SPY position & margin snapshot tracking |

### Persistent Files
| File | Purpose |
|---|---|
| `trade_status.json` | Tracks the last date trades were executed (prevents duplicate trading) |
| `.etrade_oauth` | Cached OAuth access tokens for session reuse/renewal |
| `screened_option_pairs.html` | Auto-generated dashboard of current positions |
| `python_client.log` | Rotating log file (5 MB max, 3 backups) |

---

## Command-Line Arguments

```
--sandbox / --no-sandbox    Use sandbox or live E*TRADE environment
--trade / --no-trade        Enable live trading mode (required for main loop)
--use_existing_file         Re-use cached backtest results
--username TEXT              E*TRADE login username (required)
--password TEXT              E*TRADE login password (required)
--no-headless               Disable headless browser for OAuth login
```

---

## Module-Level Components

### 1. HTTP Refresh & Approval Server (lines 45–121)

**`class RefreshHandler`** — A lightweight `BaseHTTPRequestHandler` serving on port 8765.

| Endpoint | Method | Purpose |
|---|---|---|
| `/refresh` | POST | Signals the main loop to refresh pricing data immediately |
| `/approve_orders` | GET | Approves pending orders (clicked from email) |
| `/reject_orders` | GET | Rejects pending orders (clicked from email) |

**Global threading events:**
- `REFRESH_REQUESTED` — set by `/refresh`, cleared by main loop after processing.
- `ORDER_APPROVED` / `ORDER_REJECTED` — set by `/approve_orders` or `/reject_orders`.

**`start_refresh_server(port=8765)`** — Starts the HTTP server on a daemon thread.

**`_get_local_ip()`** — Resolves the machine's LAN IP for clickable links in emails.

---

### 2. Email Approval System (lines 137–599)

#### `send_order_approval_email(preview_orders, total_net_credit, total_required_margin, roi_display, port=8765)`
- Sends an HTML email summarizing all proposed **opening** orders.
- Includes per-order details: ticker, type (call/put/cover call), strikes, expiry, credit, margin, ROI, distance-to-strike.
- Contains clickable **Approve / Reject** links pointing to the local HTTP server.
- Fallback: user can reply with "APPROVE" or "REJECT" in email body.
- Returns `(success: bool, message_id: str)`.

#### `send_close_spread_approval_email(close_proposals, port=8765)`
- Same pattern but for **closing** high-gain (>70%) spread positions.
- Summarizes each spread's ticker, strike pair, expiry, quantity, gain %, midpoint, and total debit.

#### `_check_imap_for_approval(sent_after, gmail_address, gmail_app_password)`
- Polls Gmail IMAP (INBOX, Sent Mail, All Mail) for replies containing "APPROVE" or "REJECT".
- Filters by date and subject line, handles MIME-encoded subjects, timezone-aware date comparison.
- Returns `'APPROVE'`, `'REJECT'`, or `None`.

#### `wait_for_order_approval(timeout_minutes=30)`
- Dual-channel wait loop: checks HTTP events every 1s, polls IMAP every 15s.
- Returns `True` (approved) or `False` (rejected / timed out).
- [DONE] After approval, `refresh_spread_limit_price()` re-fetches live quotes and updates the limit price before execution. Orders whose credit has dropped to ≤$0.01 are skipped.

#### `refresh_spread_limit_price(market_instance, ticker, sell_strike, buy_strike, call_put, expiration_date, order_dict, is_credit=True)`
- Builds OSI keys for both spread legs, fetches live quotes via `market.get_quote()`.
- Calculates midpoint credit (opening) or debit (closing) from current bid/ask.
- Updates `order_dict['limitPrice']` in-place; returns `(is_valid, new_price)`.
- Graceful fallback: if quotes fail, returns the original price and proceeds.
---

### 3. Market Status Utilities (lines 636–722)

#### `is_trading_day(check_date)`
- Uses `pandas_market_calendars` (NYSE calendar) to determine if a date is a valid trading day.
- Falls back to weekday-only check if the library fails.

#### `is_market_open(check_datetime=None)`
- Returns `(is_open, status, market_open, market_close)`.
- `status` ∈ `{"OPEN", "PRE_MARKET", "AFTER_HOURS", "CLOSED_HOLIDAY"}`.
- Includes timezone-aware handling (US/Eastern) and a heuristic fallback.

#### `load_trade_status()` / `save_trade_status(last_trade_date)`
- Simple JSON read/write for `trade_status.json`.

---

### 4. Margin Release (lines 724–833)

#### `release_margin(all_positions, cover_call_list=None, etrade_instance=None, max_positions=5)`
- Identifies the top N highest-margin-ratio spreads via `find_highest_margin_ratios()`.
- Generates BUY_CLOSE (short leg) + SELL_CLOSE (long leg) orders for each.
- Places each order with a LIMIT price ±$0.01 from last price.
- Prints before/after margin buying power.
- Returns the number of positions processed.

---

### 5. OAuth Authentication (lines 853–1013)

#### `get_etrade_oauth(use_sandbox)` / `save_etrade_oauth(token, use_sandbox)`
- Cache-based OAuth token persistence, keyed by environment (sandbox/live).

#### `oauth(use_sandbox, auto_login=True, username=None, password=None, headless=None)`
- **Token renewal path**: If `.etrade_oauth` exists, attempts session renewal via the renew endpoint.
- **New auth path**: Initiates a full OAuth 1 flow using `rauth.OAuth1Service`.
  - Uses `get_token_automated()` (from external module) for headless browser-based E\*TRADE login.
  - Custom UTF-8 decoder handles E\*TRADE's non-ASCII error responses.
- Returns `(session, base_url)`.

---

### 6. Helper Utilities

#### `extract_ticker_ask_bid(data)` (line 835)
- Parses E\*TRADE `QuoteResponse` JSON into `[{symbol, ask, bid}]`.

#### `environment_key(use_sandbox)` (line 850)
- Returns `"sandbox"` or `"live"`.

---

## Main Loop (lines 1015–1749)

### Initialization (lines 1015–1116)
1. Parse CLI arguments.
2. Authenticate via `oauth()`.
3. Create `LiveTradeAgent`, `Accounts`, and `Market` instances.
4. Start the HTTP refresh server on port 8765.
5. Guard: exit if `--trade` flag is missing.

### Loop Logic (inside `while True`)

The main loop runs indefinitely with the following flow on each iteration:

```
┌─────────────────────────────────────────┐
│ 1. Get current time / load trade status │
├─────────────────────────────────────────┤
│ 2. Renew OAuth session (every 60 min)   │
├─────────────────────────────────────────┤
│ 3. Check market status                  │
│   ├── CLOSED_HOLIDAY → sleep until next │
│   │   trading day (interruptible)       │
│   ├── PRE_MARKET / AFTER_HOURS →        │
│   │   update HTML, sleep until open     │
│   └── OPEN → continue                  │
├─────────────────────────────────────────┤
│ 4. Trading window check                 │
│   (08:15–13:30 Pacific)                 │
│   Outside window → update HTML, sleep   │
├─────────────────────────────────────────┤
│ 5. Read-only operations:                │
│   - account_list, balance, portfolio    │
│   - option_gain_new (weekly P&L)        │
│   - SPY position tracker snapshot       │
│   - get_option_trade (rollable options) │
│   - update_csv_order_statuses           │
├─────────────────────────────────────────┤
│ 6. OPENING NEW POSITIONS (once/day):    │
│   a. Load parameters per ticker         │
│   b. Check for earnings in period       │
│   c. Calculate target premiums          │
│      (VIX-adjusted, sqrt-scaled by DTE) │
│   d. Find call/put spread candidates    │
│   e. Generate preview orders            │
│   f. Calculate ROI, assignment prob     │
│   g. Email approval gate                │
│   h. Execute approved orders            │
│   i. Auto-adjust price until filled     │
│   j. Record trade + save status         │
├─────────────────────────────────────────┤
│ 7. POSITION MANAGEMENT (every tick):    │
│   a. Re-fetch portfolio & screen        │
│   b. Render HTML dashboard              │
│   c. Detect spreads with >70% gain      │
│   d. Fetch live quotes for legs         │
│   e. Build close orders at midpoint     │
│   f. Email approval gate                │
│   g. Execute approved closes            │
│   h. Record closed gains (SPY tracker)  │
├─────────────────────────────────────────┤
│ 8. Position neutralization (BA-specific)│
│   - Delta-neutral hedge for BA PUT 1700 │
├─────────────────────────────────────────┤
│ 9. Sleep 5 min (interruptible by        │
│    manual refresh), then repeat         │
└─────────────────────────────────────────┘
```

### Error Handling
- Exponential backoff on main-loop exceptions: `30 * 2^(retry-1)` seconds, capped at 5 minutes.
- All errors logged to `python_client.log` and printed to stdout.

---

## Spread Selection Logic (lines 1296–1465)

### Parameter Loading
- `load_latest_parameters()` returns per-ticker params (target premiums, hedge ratios, quantities) and expiration week count.
- **Cover call tickers** (in `cover_call_list` dict) are appended with `is_cover_call: True`.

### Premium Targeting
- **Baseline**: `target_premium_otm * sqrt(DTE) * steer_factor`
  - `target_steer = 0.9` biases toward put spreads (90% of premium budget to puts).
- **VIX adjustment**: If VIX > 20, premiums are widened; if VIX > 30, adjustment quadruples.
- **Floor**: premium ≥ baseline / 1.5.

### Candidate Search
- `accounts.get_option_spread_by_price(ticker, call_or_put, days_to_expire, target_premium, hedge_ratio, hedge_spread=20, qty)`
- If no candidates found for the target DTE, retries with 7 fewer days (minimum 2 days).
- Auto-corrects if actual expiration date doesn't match requested DTE.

### Order Generation
- Orders are generated via `accounts.generate_option_order()` with `priceType=NET_CREDIT`.
- Minimum profit threshold: `$0.01`.

---

## High-Gain Spread Detection (lines 1576–1708)

1. **Screen** all positions via `accounts.screen_option()`.
2. **Filter** spread entries where `short_lot.gain_loss_percentage > 70`.
3. **Fetch live quotes** using OSI-format option keys.
4. **Calculate midpoint** debit = `(mid_short - mid_long)`.
5. **Generate close order** as `NET_DEBIT` spread at midpoint price.
6. **Batch** all proposals into a single approval email.
7. On approval, execute each close with auto-price-adjustment until filled.

---

## Key Constants & Defaults

| Constant | Value | Description |
|---|---|---|
| Trade window | 08:15–13:30 | Pacific time window for opening trades |
| Datalog window | 06:45–13:15 | *(currently unused)* |
| Session renewal | 60 min | Re-authenticate OAuth every hour |
| High-gain threshold | 70% | Trigger spread close proposals |
| HTTP server port | 8765 | Refresh & approval endpoints |
| IMAP poll interval | 15 sec | Email reply check frequency |
| Approval timeout | 30 min | Default; configurable via `config.ini` |
| Price adjustment step | $0.01 / 30s | Auto-adjust unfilled orders |
| Post-trade sleep | 5 min | Between main loop iterations |

---

## Future Extension Points

<!-- Add your planned features here -->

- [DONE] **Feature**: In `screened_option_pairs.html`, the total Portfolio value chart now shows SPY closing price (2nd y-axis, green) and VIX closing price (3rd y-axis, orange dashed) via `yfinance`.
- [DONE] **Feature**: For the email summary sent out at the begining and end of the trading each day, only do that during market opened days
- [DONE] **Feature**: The plots in the html, only include days where the market is open
- [DONE] **Feature**: When SPY has dropped more than 0.6% (make this configurable somewhere) from the previous day closing price and the position-open trade of the day hasn`t already been executed or rejected, then automatically proceed to open the position bypassing the email approval.
- [NEW] **Feature**: For the above feature, when the drop is detected, set the strike price of the short leg to the nearest strike price below the current SPY price, and then long leg to be 20 dollars below the short leg, and then set the qty to be 2. 
