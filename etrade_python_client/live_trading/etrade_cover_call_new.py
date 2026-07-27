#!/usr/bin/env python
import subprocess
from tabnanny import check
import time as t
from cgi import test
import argparse
from json.decoder import JSONDecodeError
from signal import signal
from tracemalloc import start
import pyetrade
import json
import ast
import os
import sys
import stat
import traceback
import random
import secrets
import tempfile
import hmac
import hashlib
import smtplib
from http.cookies import SimpleCookie
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import timedelta
from datetime import datetime, date, timezone
from pathlib import Path
import logging
import pandas as pd
from live_trading.ev_engine import (
    build_regime_return_arrays, get_probability_engine, calculate_yield_metrics,
    fetch_cached_yf_close, fetch_historical_data, _build_causal_regime_feature_frame,
    ProbabilityEngineUnavailable,
)
import yfinance as yf
from backtesting import backtest_bo
import matplotlib.pyplot as plt
import numpy as np
from itertools import product                                                               
# import option_price
import csv
from accounts.accounts_bo import StockPosition
from core_api.stock_trade_class import *
import webbrowser
from rauth import OAuth1Service
from accounts.accounts_bo import Accounts, calculate_std_dev, calculate_margin, print_margin_report, find_highest_margin_ratios, _select_nearest_expiration, is_etrade_token_expired_response
from market.market_bo import Market
import configparser
import multiprocessing
from typing import List
from queue import Queue
from uuid import uuid4
from backtesting import option_limit_backtest
from data_and_research.polygonio_improvequery import get_earnings_dates
from data_and_research.option_assign_probability import calculate_probability
from backtesting.polygonio_dailytrade import fetch_yfinance_data
from pandas_market_calendars import get_calendar
from live_trading.regime_shadow_store import (
    RegimeShadowStore,
    unavailable_dashboard_payload,
)
from live_trading.positions_artifact import (
    PositionsArtifactSigningKey,
    build_positions_snapshot,
    positions_identity_fingerprint,
)
from live_trading.positions_artifact_publisher import (
    PositionsArtifactPublisher,
)
from live_trading.runtime_config import (
    RuntimeConfigError,
    load_runtime_config,
    resolve_positions_artifact_hmac_key,
    validate_runtime_directories,
)
from live_trading.runtime_safety import (
    LegacyExecutionDisabled,
    RuntimeSafetyError,
    build_runtime_safety_boundary,
    configure_owner_only_logger,
    read_owner_only_json,
    reject_legacy_execution,
    secure_append_text,
    secure_lock_file,
    validate_dashboard_credentials,
    write_owner_only_json,
)
from live_trading.spy_position_tracker import update_spy_daily_snapshot, record_closed_spy_gain
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import base64
import io
TRADE_STATUS_FILE = "trade_status.json"
DASHBOARD_LOG_FILE = "dashboard_requests.log"
AUDIT_LOG_FILE = "order_audit_log.csv"
MANUAL_TRADE_STATUS_FILE = "manual_order_status.json"
ETRADE_SESSION_REFRESH_LOCK = threading.RLock()
REGIME_V2_SHADOW_PATH = (
    Path(__file__).resolve().parent
    / "runtime"
    / "regime_v2_shadow.json"
)
REGIME_V2_SHADOW_STORE = RegimeShadowStore(REGIME_V2_SHADOW_PATH)
REGIME_V2_PUBLIC_DASHBOARD_FIELDS = frozenset({
    "available",
    "status",
    "source_family",
    "as_of_session",
    "effective_session",
    "background_state",
    "shock_state",
    "composite_label",
    "availability",
    "reason_codes",
    "abstain_reasons",
    "stale",
    "may_authorize_execution",
})
DISABLED_DASHBOARD_EXECUTION_PATHS = frozenset({
    "/api/execute_manual_order",
    "/api/execute_neutralize_order",
    "/api/review_close_position",
    "/api/close_position",
    "/api/execute_close_order",
})
POSITIONS_READ_ONLY_MARKER = (
    "Read only — all E*TRADE order actions are disabled"
)
MAX_POSITIONS_ARTIFACT_BYTES = 8 * 1024 * 1024
FORBIDDEN_POSITIONS_ARTIFACT_MARKERS = frozenset({
    *DISABLED_DASHBOARD_EXECUTION_PATHS,
    "data-close-position",
    "action-cell",
})
POSITIONS_FRAME_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
    "script-src 'none'; connect-src 'none'; frame-ancestors 'self'; "
    "base-uri 'none'; form-action 'none'"
)


def _read_only_positions_fallback() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Positions unavailable</title>
  <style>
    body {{ margin: 0; padding: 16px; color: #e2e8f0; background: #0f172a;
            font: 15px/1.5 system-ui, sans-serif; }}
    .notice {{ border: 1px solid #ef4444; border-radius: 10px; padding: 14px;
               background: #450a0a; }}
    strong {{ display: block; margin-bottom: 6px; color: #fecaca; }}
  </style>
</head>
<body data-positions-artifact-state="unavailable">
  <div class="notice" role="alert">
    <strong>{POSITIONS_READ_ONLY_MARKER}</strong>
    Position data is temporarily unavailable because the generated artifact
    predates the read-only renderer. Refresh after the monitoring loop creates
    a current artifact.
  </div>
</body>
</html>"""


def _validated_positions_artifact(content):
    """Return only a current read-only artifact; fail closed on stale HTML."""

    if not isinstance(content, str):
        return _read_only_positions_fallback(), False
    repaired = _repair_benchmark_option_value_gaps(content)
    lowered = repaired.lower()
    if (
        len(repaired.encode("utf-8")) > MAX_POSITIONS_ARTIFACT_BYTES
        or POSITIONS_READ_ONLY_MARKER not in repaired
        or any(
            marker.lower() in lowered
            for marker in FORBIDDEN_POSITIONS_ARTIFACT_MARKERS
        )
    ):
        return _read_only_positions_fallback(), False
    return repaired, True

def log_order_execution(order_info, reason, status="PLACED"):
    """Log order execution details to a permanent CSV file."""
    try:
        file_exists = os.path.exists(AUDIT_LOG_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ticker = order_info.get('ticker', 'N/A')
        order_type = "CLOSE" if order_info.get('is_close') else "OPEN"
        strikes = f"{order_info.get('sell_strike', 'N/A')}/{order_info.get('long_strike', 'N/A')}"
        qty = order_info.get('pair_quantity', order_info.get('qty', 1))
        order_id = order_info.get('order_id', 'N/A')
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        if not file_exists:
            writer.writerow(("timestamp", "ticker", "type", "strikes", "qty", "reason", "order_id", "status"))
        writer.writerow((timestamp, ticker, order_type, strikes, qty, reason, order_id, status))
        secure_append_text(AUDIT_LOG_FILE, buffer.getvalue())
    except Exception as e:
        print(f"⚠️ Error writing to audit log: {e}")

def send_trade_notification_email(order_info, reason):
    """Send an immediate email notification for a trade execution."""
    try:
        gmail_address = config.get('EMAIL', 'GMAIL_ADDRESS', fallback=None)
        gmail_app_password = config.get('EMAIL', 'GMAIL_APP_PASSWORD', fallback=None)
        notify_email = config.get('EMAIL', 'NOTIFY_EMAIL', fallback=gmail_address)

        if not gmail_address or not gmail_app_password or gmail_app_password == 'REPLACE_WITH_APP_PASSWORD':
            return

        ticker = order_info.get('ticker', 'N/A')
        order_type = "CLOSE" if order_info.get('is_close') else "OPEN"
        strikes = f"{order_info.get('sell_strike', 'N/A')}/{order_info.get('long_strike', 'N/A')}"
        qty = order_info.get('pair_quantity', order_info.get('qty', 1))
        
        subject = f"🚀 Trade Executed: {order_type} {ticker} {strikes}"
        body = f"""
        <h2>Trade Execution Notification</h2>
        <p><b>Type:</b> {order_type}</p>
        <p><b>Ticker:</b> {ticker}</p>
        <p><b>Strikes:</b> {strikes}</p>
        <p><b>Quantity:</b> {qty}</p>
        <p><b>Reason:</b> {reason}</p>
        <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr>
        <p><i>Sent by Antigravity Trading Bot</i></p>
        """
        
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = gmail_address
        msg['To'] = notify_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.send_message(msg)
            print(f"📧 Notification email sent to {notify_email}")
    except Exception as e:
        print(f"⚠️ Failed to send notification email: {e}")

_DASHBOARD_LOG_SECRET_MARKERS = (
    "pin", "pass", "password", "secret", "token", "credential", "oauth",
    "consumer", "authorization", "auth",
)


def _redact_dashboard_log_data(value):
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(marker in str(key).lower() for marker in _DASHBOARD_LOG_SECRET_MARKERS)
                else _redact_dashboard_log_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_dashboard_log_data(item) for item in value]
    return value


def log_dashboard_request(path, data):
    """Log request metadata without persisting credentials or action secrets."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = _redact_dashboard_log_data(data)
            
        entry = {
            "timestamp": timestamp,
            "path": path,
            "data": log_data
        }
        secure_append_text(DASHBOARD_LOG_FILE, json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Error logging dashboard request: {e}")

def _repair_benchmark_option_value_gaps(html):
    try:
        match = re.search(r'(optionValues:\s*)(\[[^\]]*\])', html)
        if not match:
            return html
        values = json.loads(match.group(2))
        repaired = []
        last_value = None
        changed = False
        for value in values:
            if value is not None:
                last_value = value
                repaired.append(value)
            elif last_value is not None:
                repaired.append(last_value)
                changed = True
            else:
                repaired.append(value)
        if not changed:
            return html
        return html[:match.start(2)] + json.dumps(repaired) + html[match.end(2):]
    except Exception as e:
        print(f"⚠️ Could not repair benchmark option value gaps: {e}")
        return html

# Global flag to signal a manual refresh request from the web UI
REFRESH_REQUESTED = threading.Event()
PORTFOLIO_REFRESH_LOCK = threading.Lock()
PORTFOLIO_REFRESH_STATUS = {
    "requested_generation": 0,
    "running_generation": 0,
    "completed_generation": 0,
    "last_error_generation": 0,
    "last_error": None,
}


def _queue_portfolio_refresh():
    with PORTFOLIO_REFRESH_LOCK:
        PORTFOLIO_REFRESH_STATUS["requested_generation"] += 1
        PORTFOLIO_REFRESH_STATUS["last_error"] = None
        return PORTFOLIO_REFRESH_STATUS["requested_generation"]


def _start_portfolio_refresh():
    with PORTFOLIO_REFRESH_LOCK:
        generation = PORTFOLIO_REFRESH_STATUS["requested_generation"]
        if generation <= PORTFOLIO_REFRESH_STATUS["completed_generation"]:
            generation = PORTFOLIO_REFRESH_STATUS["completed_generation"] + 1
            PORTFOLIO_REFRESH_STATUS["requested_generation"] = generation
        PORTFOLIO_REFRESH_STATUS["running_generation"] = generation
        return generation


def _finish_portfolio_refresh(generation, error=None):
    with PORTFOLIO_REFRESH_LOCK:
        PORTFOLIO_REFRESH_STATUS["completed_generation"] = max(
            PORTFOLIO_REFRESH_STATUS["completed_generation"],
            generation,
        )
        if PORTFOLIO_REFRESH_STATUS["running_generation"] == generation:
            PORTFOLIO_REFRESH_STATUS["running_generation"] = 0
        if error:
            PORTFOLIO_REFRESH_STATUS["last_error_generation"] = generation
            PORTFOLIO_REFRESH_STATUS["last_error"] = str(error)
        elif PORTFOLIO_REFRESH_STATUS["last_error_generation"] == generation:
            PORTFOLIO_REFRESH_STATUS["last_error_generation"] = 0
            PORTFOLIO_REFRESH_STATUS["last_error"] = None


def _portfolio_refresh_snapshot():
    with PORTFOLIO_REFRESH_LOCK:
        snapshot = dict(PORTFOLIO_REFRESH_STATUS)
    if snapshot["running_generation"]:
        snapshot["state"] = "refreshing"
    elif snapshot["completed_generation"] < snapshot["requested_generation"]:
        snapshot["state"] = "queued"
    elif (
        snapshot["last_error"]
        and snapshot["last_error_generation"] == snapshot["completed_generation"]
    ):
        snapshot["state"] = "failed"
    else:
        snapshot["state"] = "idle"
    return snapshot

# Global flags for order flow
MANUAL_TRADE_REQUESTED = threading.Event()
MANUAL_TRADE_PARAMS = {}
MANUAL_TRADE_QUEUE = Queue()
MANUAL_TRADE_STATUS = {}
MANUAL_TRADE_STATUS_LOCK = threading.Lock()
CURRENT_CLOSE_PROPOSALS = [] # Global for dashboard access
CURRENT_NEUTRALIZE_PROPOSALS = [] # Global for dashboard access
ACTIVE_DASHBOARD_ORDERS = set() # Global for tracking orders submitted to E*TRADE
MARKET_CLOSE_PRICE_REFRESHED_DATE = None

NEUTRALIZE_DELTA_THRESHOLD = 0.20
NEUTRALIZE_TRIGGER_DTE = 21
NEUTRALIZE_TARGET_DTE = 42
AUTO_REFRESH_INTERVAL_SECONDS = 300
DASHBOARD_ON_DEMAND_REFRESH_ONLY = True
READ_ONLY_POSITIONS_PUBLISHER = None
READ_ONLY_POSITIONS_BROKER_ENVIRONMENT = None


def _publish_confirmed_positions(positions, *, observed_at=None):
    """Publish one complete raw portfolio snapshot for the isolated dashboard."""

    publisher = READ_ONLY_POSITIONS_PUBLISHER
    environment = READ_ONLY_POSITIONS_BROKER_ENVIRONMENT
    if (
        type(publisher) is not PositionsArtifactPublisher
        or environment not in {"sandbox", "production"}
    ):
        raise RuntimeError(
            "read-only positions publisher is not configured"
        )
    source_as_of = (
        datetime.now(timezone.utc)
        if observed_at is None
        else observed_at
    )
    snapshot = build_positions_snapshot(
        positions,
        broker_environment=environment,
        source_as_of=source_as_of,
    )
    receipt = publisher.publish(snapshot)
    print(
        "✅ [Read-only Publisher] "
        f"{receipt.state}: {receipt.source_generation[:12]} "
        f"({receipt.size_bytes} bytes)"
    )
    return receipt


def _load_stable_positions(accounts):
    """Require two page-complete scans with identical identity and quantity."""

    first = accounts.portfolio(
        print_enable=False,
        minimal=True,
        require_success=True,
    )
    source_as_of = datetime.now(timezone.utc)
    second = accounts.portfolio(
        print_enable=False,
        require_success=True,
    )
    if (
        positions_identity_fingerprint(first)
        != positions_identity_fingerprint(second)
    ):
        raise RuntimeError(
            "E*TRADE portfolio changed between confirmation reads; "
            "the prior positions artifact was preserved"
        )
    return second, source_as_of


def _status_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _next_refresh_sleep_seconds(cycle_started_at, deadline_seconds=None):
    elapsed = max(0, int((datetime.now() - cycle_started_at).total_seconds()))
    remaining = max(0, AUTO_REFRESH_INTERVAL_SECONDS - elapsed)
    if deadline_seconds is not None:
        remaining = min(remaining, max(0, int(deadline_seconds)))
    return remaining

def _wait_for_dashboard_work():
    print("💤 On-demand dashboard mode: waiting for browser refresh or manual trade request.")
    last_close_check_at = None
    while not REFRESH_REQUESTED.is_set() and not MANUAL_TRADE_REQUESTED.is_set():
        now = datetime.now()
        if last_close_check_at is None or (now - last_close_check_at).total_seconds() >= 30:
            last_close_check_at = now
            try:
                if _market_close_refresh_due():
                    print("🔄 [Market Close] Waking dashboard loop to capture closing SPY/SPX/VIX data.")
                    REFRESH_REQUESTED.set()
                    break
            except Exception as close_check_err:
                print(f"⚠️ [Market Close] Could not check close-refresh timing: {close_check_err}")
        t.sleep(1)

def _sync_manual_trade_event():
    if MANUAL_TRADE_QUEUE.empty():
        MANUAL_TRADE_REQUESTED.clear()
    else:
        MANUAL_TRADE_REQUESTED.set()

def _save_manual_trade_status_locked():
    try:
        tmp_file = MANUAL_TRADE_STATUS_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(MANUAL_TRADE_STATUS, f, indent=2)
        os.replace(tmp_file, MANUAL_TRADE_STATUS_FILE)
    except Exception as e:
        print(f"⚠️ Error saving manual trade status: {e}")

def _load_manual_trade_status_file():
    if not os.path.exists(MANUAL_TRADE_STATUS_FILE):
        return {}
    try:
        with open(MANUAL_TRADE_STATUS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"⚠️ Error loading manual trade status: {e}")
        return {}

def _hydrate_manual_trade_status_from_file():
    persisted = _load_manual_trade_status_file()
    if not persisted:
        return
    with MANUAL_TRADE_STATUS_LOCK:
        for request_id, record in persisted.items():
            if not isinstance(record, dict):
                continue
            current = MANUAL_TRADE_STATUS.get(request_id)
            if not current or str(record.get("updated_at", "")) > str(current.get("updated_at", "")):
                MANUAL_TRADE_STATUS[request_id] = record

def update_manual_trade_status(request_id, **updates):
    if not request_id:
        return
    with MANUAL_TRADE_STATUS_LOCK:
        status = MANUAL_TRADE_STATUS.setdefault(request_id, {"request_id": request_id})
        status.update(updates)
        status["updated_at"] = _status_now()
        _save_manual_trade_status_locked()

def get_manual_trade_status_record(request_id):
    if not request_id:
        return {}
    with MANUAL_TRADE_STATUS_LOCK:
        return dict(MANUAL_TRADE_STATUS.get(request_id, {}))

def append_manual_trade_status(request_id, key, value):
    if not request_id:
        return
    with MANUAL_TRADE_STATUS_LOCK:
        status = MANUAL_TRADE_STATUS.setdefault(request_id, {"request_id": request_id})
        status.setdefault(key, []).append(value)
        status["updated_at"] = _status_now()
        _save_manual_trade_status_locked()

def _audit_status_to_manual_status(status):
    normalized = str(status or "").strip().upper()
    if normalized in ("FILLED", "EXECUTED"):
        return "filled"
    if normalized in ("OPEN", "PLACED", "QUEUED", "PROCESSING"):
        return "placed"
    if normalized in ("FAILED", "REJECTED"):
        return "failed"
    if normalized == "CANCELLED":
        return "cancelled"
    return normalized.lower() or "unknown"

def _today_order_audit_records(today_prefix):
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    records = []
    try:
        with open(AUDIT_LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                timestamp = row.get("timestamp", "")
                if not timestamp.startswith(today_prefix):
                    continue
                order_id = str(row.get("order_id") or "").strip()
                ticker = str(row.get("ticker") or "").strip()
                order_type = str(row.get("type") or "order").strip()
                strikes = str(row.get("strikes") or "").strip()
                strike_parts = strikes.split("/", 1)
                status = _audit_status_to_manual_status(row.get("status"))
                reason = str(row.get("reason") or "").strip()
                message_parts = [p for p in [ticker, order_type, strikes, row.get("status"), reason] if p]
                records.append({
                    "request_id": f"audit-{order_id}" if order_id and order_id != "N/A" else f"audit-{timestamp}-{ticker}-{strikes}",
                    "request_type": order_type.lower() if order_type else "order",
                    "status": status,
                    "ticker": ticker,
                    "side": "",
                    "short_strike": strike_parts[0] if strike_parts else "",
                    "long_strike": strike_parts[1] if len(strike_parts) > 1 else "",
                    "quantity": row.get("qty"),
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "order_ids": [order_id] if order_id and order_id != "N/A" else [],
                    "message": " | ".join(message_parts),
                    "messages": [reason] if reason else []
                })
    except Exception as e:
        print(f"⚠️ Error reading order audit log: {e}")
    return records

def _dashboard_request_record_from_log(entry):
    path = entry.get("path")
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    timestamp = entry.get("timestamp", "")
    if path not in ("/api/execute_manual_order", "/api/close_position", "/api/execute_close_order", "/api/execute_neutralize_order"):
        return None
    if "error" in data:
        return None

    if path == "/api/execute_neutralize_order":
        ticker = data.get("ticker")
        request_type = "neutralize"
        side = ""
        short_strike = ""
        long_strike = ""
        quantity = data.get("qty")
        message = f"Dashboard neutralize request {data.get('proposal_id', '')}".strip()
    elif path == "/api/execute_manual_order":
        ticker = data.get("ticker") or data.get("symbol")
        request_type = "open"
        side = data.get("side") or data.get("call_put") or data.get("cp")
        short_strike = data.get("sell_strike") or data.get("short_strike") or data.get("strike")
        long_strike = data.get("buy_strike") or data.get("long_strike")
        quantity = data.get("qty") or data.get("quantity") or data.get("pair_quantity")
        message = f"Dashboard open request {short_strike}/{long_strike}".strip()
    else:
        ticker = data.get("ticker") or data.get("symbol")
        request_type = "close"
        side = data.get("side") or data.get("call_put") or data.get("cp")
        short_strike = data.get("sell_strike") or data.get("short_strike") or data.get("strike")
        long_strike = data.get("long_strike")
        quantity = data.get("qty") or data.get("quantity") or data.get("pair_quantity")
        message = f"Dashboard close request {short_strike}/{long_strike}".strip()

    request_id = f"request-{timestamp}-{path}-{ticker}-{short_strike}-{long_strike}-{quantity}"
    return {
        "request_id": request_id,
        "request_type": request_type,
        "status": "placed",
        "ticker": ticker,
        "side": side,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "quantity": quantity,
        "created_at": timestamp,
        "updated_at": timestamp,
        "order_ids": [],
        "message": message,
        "messages": []
    }

def _today_dashboard_request_records(today_prefix):
    if not os.path.exists(DASHBOARD_LOG_FILE):
        return []
    records = []
    try:
        with open(DASHBOARD_LOG_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                timestamp = entry.get("timestamp", "")
                if not timestamp.startswith(today_prefix):
                    continue
                record = _dashboard_request_record_from_log(entry)
                if record:
                    records.append(record)
    except Exception as e:
        print(f"⚠️ Error reading dashboard request log: {e}")
    return records

def _manual_records_from_open_orders(open_orders):
    if not open_orders:
        return []
    grouped = {}
    for leg in open_orders:
        order_id = str(leg.get("orderId") or "").strip()
        if not order_id:
            continue
        grouped.setdefault(order_id, []).append(leg)

    records = []
    for order_id, legs in grouped.items():
        first_leg = legs[0]
        placed_time = first_leg.get("placedTime")
        if placed_time:
            try:
                created_at = datetime.fromtimestamp(int(placed_time) / 1000).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at = _status_now()
        else:
            created_at = _status_now()

        actions = {str(leg.get("orderAction") or "").upper() for leg in legs}
        if any(action.endswith("_CLOSE") for action in actions):
            request_type = "close"
        elif any(action.endswith("_OPEN") for action in actions):
            request_type = "open"
        else:
            request_type = "order"

        strikes = [str(leg.get("strikePrice")) for leg in legs if leg.get("strikePrice") is not None]
        message = f"Broker open order {order_id}"
        if first_leg.get("limitPrice") is not None:
            message += f" @ {first_leg.get('limitPrice')}"
        if strikes:
            message += f" | strikes {'/'.join(strikes)}"

        records.append({
            "request_id": f"broker-open-{order_id}",
            "request_type": request_type,
            "status": "placed",
            "ticker": first_leg.get("symbol"),
            "side": first_leg.get("callPut"),
            "short_strike": strikes[0] if strikes else "",
            "long_strike": strikes[1] if len(strikes) > 1 else "",
            "quantity": first_leg.get("quantity"),
            "created_at": created_at,
            "updated_at": created_at,
            "order_ids": [order_id],
            "message": message,
            "messages": []
        })
    return records

def _manual_records_from_executed_orders(executed_orders):
    if not executed_orders:
        return []
    grouped = {}
    for leg in executed_orders:
        order_id = str(leg.get("order_id") or "").strip()
        if not order_id:
            continue
        grouped.setdefault(order_id, []).append(leg)

    records = []
    for order_id, legs in grouped.items():
        first_leg = legs[0]
        actions = {str(leg.get("order_action") or "").upper() for leg in legs}
        if any(action.endswith("_CLOSE") for action in actions):
            request_type = "close"
            short_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_CLOSE") and str(leg.get("order_action") or "").upper().startswith("BUY")), legs[0])
            long_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_CLOSE") and str(leg.get("order_action") or "").upper().startswith("SELL")), legs[-1])
        elif any(action.endswith("_OPEN") for action in actions):
            request_type = "open"
            short_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_OPEN") and str(leg.get("order_action") or "").upper().startswith("SELL")), legs[0])
            long_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_OPEN") and str(leg.get("order_action") or "").upper().startswith("BUY")), legs[-1])
        else:
            request_type = "order"
            short_leg = legs[0]
            long_leg = legs[-1]

        short_strike = short_leg.get("strike_price")
        long_strike = long_leg.get("strike_price")
        quantity = first_leg.get("executed_quantity")
        message = f"Broker executed order {order_id}"
        if short_strike is not None or long_strike is not None:
            message += f" | strikes {short_strike}/{long_strike}"

        records.append({
            "request_id": f"broker-executed-{order_id}",
            "request_type": request_type,
            "status": "filled",
            "ticker": first_leg.get("symbol"),
            "side": first_leg.get("option_type"),
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
            "created_at": first_leg.get("executed_date") or _status_now(),
            "updated_at": first_leg.get("executed_date") or _status_now(),
            "order_ids": [order_id],
            "message": message,
            "messages": []
        })
    return records

def _manual_records_from_cancelled_orders(cancelled_orders):
    if not cancelled_orders:
        return []
    grouped = {}
    for leg in cancelled_orders:
        order_id = str(leg.get("order_id") or "").strip()
        if not order_id:
            continue
        grouped.setdefault(order_id, []).append(leg)

    records = []
    for order_id, legs in grouped.items():
        first_leg = legs[0]
        actions = {str(leg.get("order_action") or "").upper() for leg in legs}
        if any(action.endswith("_CLOSE") for action in actions):
            request_type = "close"
            short_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_CLOSE") and str(leg.get("order_action") or "").upper().startswith("BUY")), legs[0])
            long_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_CLOSE") and str(leg.get("order_action") or "").upper().startswith("SELL")), legs[-1])
        elif any(action.endswith("_OPEN") for action in actions):
            request_type = "open"
            short_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_OPEN") and str(leg.get("order_action") or "").upper().startswith("SELL")), legs[0])
            long_leg = next((leg for leg in legs if str(leg.get("order_action") or "").upper().endswith("_OPEN") and str(leg.get("order_action") or "").upper().startswith("BUY")), legs[-1])
        else:
            request_type = "order"
            short_leg = legs[0]
            long_leg = legs[-1]

        short_strike = short_leg.get("strike_price")
        long_strike = long_leg.get("strike_price")
        quantity = first_leg.get("cancelled_quantity")
        message = f"Broker cancelled order {order_id}"
        if short_strike is not None or long_strike is not None:
            message += f" | strikes {short_strike}/{long_strike}"

        records.append({
            "request_id": f"broker-cancelled-{order_id}",
            "request_type": request_type,
            "status": "cancelled",
            "ticker": first_leg.get("symbol"),
            "side": first_leg.get("option_type"),
            "short_strike": short_strike,
            "long_strike": long_strike,
            "quantity": quantity,
            "created_at": first_leg.get("cancelled_date") or _status_now(),
            "updated_at": first_leg.get("cancelled_date") or _status_now(),
            "order_ids": [order_id],
            "message": message,
            "messages": []
        })
    return records

def _normalize_manual_order_value(value):
    if value is None:
        return ""
    try:
        return str(float(value))
    except Exception:
        return str(value).strip().upper()

def _manual_order_match_key(record):
    ticker = str(record.get("ticker") or "").upper()
    if ticker.startswith("SPX"):
        ticker = "SPX"
    return (
        ticker,
        str(record.get("request_type") or "").lower(),
        _normalize_manual_order_value(record.get("short_strike")),
        _normalize_manual_order_value(record.get("long_strike")),
        _normalize_manual_order_value(record.get("quantity")),
    )

def _manual_order_created_key(record):
    return (
        str(record.get("created_at") or ""),
        str(record.get("ticker") or "").upper(),
        str(record.get("request_type") or "").lower(),
    )

def _merge_manual_order_records(records, additions):
    merged = list(records)
    seen_request_ids = {str(r.get("request_id")) for r in merged if r.get("request_id")}
    seen_order_ids = {
        str(order_id)
        for record in merged
        for order_id in (record.get("order_ids") or [])
        if order_id
    }
    seen_match_keys = {
        _manual_order_match_key(record)
        for record in merged
        if _manual_order_match_key(record)[0] and _manual_order_match_key(record)[2]
    }
    seen_created_keys = {
        _manual_order_created_key(record)
        for record in merged
        if _manual_order_created_key(record)[0] and _manual_order_created_key(record)[1]
    }
    for record in additions:
        request_id = str(record.get("request_id")) if record.get("request_id") else ""
        order_ids = {str(order_id) for order_id in (record.get("order_ids") or []) if order_id}
        match_key = _manual_order_match_key(record)
        created_key = _manual_order_created_key(record)
        if request_id and request_id in seen_request_ids:
            continue
        if order_ids and order_ids.intersection(seen_order_ids):
            continue
        if not order_ids and match_key[0] and match_key[2] and match_key in seen_match_keys:
            continue
        if not order_ids and created_key[0] and created_key[1] and created_key in seen_created_keys:
            continue
        merged.append(record)
        if request_id:
            seen_request_ids.add(request_id)
        seen_order_ids.update(order_ids)
        if match_key[0] and match_key[2]:
            seen_match_keys.add(match_key)
        if created_key[0] and created_key[1]:
            seen_created_keys.add(created_key)
    return merged

def _normalize_local_manual_status_record(record):
    normalized = dict(record)
    order_ids = normalized.get("order_ids") or []
    message = str(normalized.get("message") or "")
    if normalized.get("status") == "failed" and order_ids and "sell_position_target" in message:
        normalized["status"] = "placed"
        normalized["message"] = f"Order submitted; target recording failed for order {', '.join(map(str, order_ids))}"
    return normalized

def get_manual_trade_status_snapshot(broker_open_orders=None, broker_executed_orders=None, broker_cancelled_orders=None):
    _hydrate_manual_trade_status_from_file()
    with MANUAL_TRADE_STATUS_LOCK:
        local_records = [_normalize_local_manual_status_record(record) for record in MANUAL_TRADE_STATUS.values()]
    today_prefix = datetime.now().strftime("%Y-%m-%d")
    records = []
    records = _merge_manual_order_records(records, _manual_records_from_executed_orders(broker_executed_orders))
    records = _merge_manual_order_records(records, _manual_records_from_open_orders(broker_open_orders))
    records = _merge_manual_order_records(records, _manual_records_from_cancelled_orders(broker_cancelled_orders))
    records = _merge_manual_order_records(records, local_records)
    records = _merge_manual_order_records(records, _today_order_audit_records(today_prefix))
    records = _merge_manual_order_records(records, _today_dashboard_request_records(today_prefix))
    records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    today_records = [r for r in records if str(r.get("created_at", "")).startswith(today_prefix)]
    return {
        "latest": records[0] if records else {},
        "requests": records[:25],
        "today": today_records,
        "queued_count": MANUAL_TRADE_QUEUE.qsize()
    }

def enqueue_manual_trade_request(data, request_type, enqueue=True):
    reject_legacy_execution("legacy dashboard manual-trade queue")

def dequeue_manual_trade_request():
    """The read-only dashboard never yields executable work."""

    MANUAL_TRADE_REQUESTED.clear()
    return None

def send_login_failure_notification(error_message, screenshot_path=None):
    """Send an email notification when automated login fails."""
    try:
        # Check if the error message indicates scheduled maintenance
        err_msg_lower = error_message.lower() if error_message else ""
        maintenance_keywords = [
            "maintenance", "temporarily unavailable", "system unavailable", 
            "service unavailable", "down for maintenance", "scheduled maintenance", 
            "schedule maintenance", "scheduled maintainance", "schedule maintainance"
        ]
        if any(kw in err_msg_lower for kw in maintenance_keywords):
            print(f"🛑 Scheduled maintenance detected. Skipping login failure email notification. Error: {error_message}")
            return

        config = configparser.ConfigParser()
        config.read("config.ini")
        
        gmail_address = config.get("EMAIL", "GMAIL_ADDRESS", fallback=None)
        gmail_app_password = config.get("EMAIL", "GMAIL_APP_PASSWORD", fallback=None)
        notify_email = config.get("EMAIL", "NOTIFY_EMAIL", fallback=gmail_address)

        if not gmail_address or not gmail_app_password:
            print("⚠️ Email notification failed: Gmail credentials not found in config.ini")
            return

        # Strip quotes
        gmail_address = gmail_address.strip("'\"")
        gmail_app_password = gmail_app_password.strip("'\"")
        notify_email = notify_email.strip("'\"")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"⚠️ E*TRADE Login Failure - {now_str}"
        
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = gmail_address
        msg["To"] = notify_email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #d93025;">❌ E*TRADE Automated Login Failed</h2>
            <p>The automated login process encountered an error at <strong>{now_str}</strong>.</p>
            <p><strong>Error Message:</strong> {error_message}</p>
            <p>Manual intervention may be required (e.g., verifying iMessage access or updating credentials).</p>
        """
        
        if screenshot_path and os.path.exists(screenshot_path):
            html_body += f'<p><strong>Screenshot of the failure:</strong></p><img src="cid:failure_screenshot" style="max-width: 100%; border: 1px solid #ccc;"/>'
        
        html_body += """
            <p style="color: #888; font-size: 12px; margin-top: 20px;">Sent automatically by E*Trade Cover Call Script.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))

        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-ID", "<failure_screenshot>")
                img.add_header("Content-Disposition", "inline", filename=os.path.basename(screenshot_path))
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.sendmail(gmail_address, notify_email, msg.as_string())
        
        print(f"📧 Login failure notification sent to {notify_email}")
    except Exception as e:
        print(f"⚠️ Failed to send login failure notification: {e}")

def send_extrinsic_value_alert(alerts):
    """Send an email notification for short options that are ITM and have low extrinsic value."""
    try:
        config = configparser.ConfigParser()
        config.read("config.ini")
        
        gmail_address = config.get("EMAIL", "GMAIL_ADDRESS", fallback=None)
        gmail_app_password = config.get("EMAIL", "GMAIL_APP_PASSWORD", fallback=None)
        notify_email = config.get("EMAIL", "NOTIFY_EMAIL", fallback=gmail_address)

        if not gmail_address or not gmail_app_password:
            print("⚠️ Email notification failed: Gmail credentials not found in config.ini")
            return

        gmail_address = gmail_address.strip("'\"")
        gmail_app_password = gmail_app_password.strip("'\"")
        notify_email = notify_email.strip("'\"")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"⚠️ E*TRADE ITM Option Assignment Risk - {now_str}"
        
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = gmail_address
        msg["To"] = notify_email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #d93025;">⚠️ High Assignment Risk: ITM Options</h2>
            <p>The following short options are In-The-Money with <strong>$1.00 or less</strong> in extrinsic value.</p>
            <p>Please close these positions to avoid assignment.</p>
            <ul>
        """
        for alert in alerts:
            html_body += f"<li style='margin-bottom: 10px;'>{alert}</li>"
            
        html_body += """
            </ul>
            <p style="color: #888; font-size: 12px; margin-top: 20px;">Sent automatically by E*Trade Cover Call Script.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.sendmail(gmail_address, notify_email, msg.as_string())
        
        print(f"📧 Extrinsic value alert sent to {notify_email}")
    except Exception as e:
        print(f"⚠️ Failed to send extrinsic value alert: {e}")

LIVE_SETTINGS_FILE = "live_trading_settings.json"
DASHBOARD_SESSION_COOKIE = "etrade_dashboard_session"
DASHBOARD_SESSION_DAYS = 7


def _validate_live_settings_parent():
    parent = Path(LIVE_SETTINGS_FILE).parent
    try:
        parent_metadata = os.lstat(parent)
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise RuntimeSafetyError("live settings parent is unsafe")
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeSafetyError("live settings parent is unsafe") from exc
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            opened_parent.st_dev != parent_metadata.st_dev
            or opened_parent.st_ino != parent_metadata.st_ino
        ):
            raise RuntimeSafetyError("live settings parent is unsafe")
    finally:
        os.close(parent_descriptor)
    return parent


def load_live_settings():
    """Load trading settings from JSON file."""
    try:
        if os.path.lexists(LIVE_SETTINGS_FILE):
            _validate_live_settings_parent()
            metadata = os.stat(LIVE_SETTINGS_FILE, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise RuntimeSafetyError("live settings must be an owner-only regular file")
            descriptor = os.open(
                LIVE_SETTINGS_FILE,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            with os.fdopen(descriptor, "r", encoding="utf-8") as f:
                settings = json.load(f)
                # Dynamic self-healing migration for SPY & SPX spread width and pair quantity settings
                modified = False
                if 'spy_hedge_spread' not in settings:
                    settings['spy_hedge_spread'] = float(settings.get('hedge_spread', 20.0) or 20.0)
                    modified = True
                if 'spx_hedge_spread' not in settings:
                    settings['spx_hedge_spread'] = 200.0
                    modified = True
                if 'spy_pair_quantity' not in settings:
                    settings['spy_pair_quantity'] = int(settings.get('pair_quantity', 15) or 15)
                    modified = True
                if 'spx_pair_quantity' not in settings:
                    settings['spx_pair_quantity'] = 2
                    modified = True
                if 'spy_target_expiration' not in settings:
                    settings['spy_target_expiration'] = settings.get('target_expiration')
                    modified = True
                if 'spx_target_expiration' not in settings:
                    settings['spx_target_expiration'] = settings.get('target_expiration')
                    modified = True
                if 'dashboard_auth_secret' not in settings:
                    settings['dashboard_auth_secret'] = secrets.token_hex(32)
                    modified = True
                if settings.get('auto_open_enabled') is not False:
                    settings['auto_open_enabled'] = False
                    modified = True
                if modified:
                    save_live_settings(settings)
                return settings
    except RuntimeSafetyError:
        raise
    except Exception as e:
        print(f"⚠️ Error loading settings: {e}")
    
    # Default fallback
    return {
        "target_delta": 0.13,
        "hedge_spread": 20.0,
        "spy_hedge_spread": 20.0,
        "spx_hedge_spread": 200.0,
        "trade_start_time": "07:15:00",
        "trade_end_time": "13:30:00",
        "auto_close_midpoint_threshold": 0.30,
        "auto_close_gain_threshold": 70.0,
        "pair_quantity": 15,
        "spy_pair_quantity": 15,
        "spx_pair_quantity": 2,
        "target_weeks": 6,
        "target_expiration": None,
        "spy_target_expiration": None,
        "spx_target_expiration": None,
        "auto_open_enabled": False,
        "pin": "",
        "dashboard_user": "",
        "dashboard_pass": "",
        "dashboard_auth_secret": secrets.token_hex(32)
    }

def save_live_settings(settings):
    """Atomically save validated dashboard settings as an owner-only file."""
    try:
        settings = dict(settings)
        settings["auto_open_enabled"] = False
        validate_dashboard_credentials(settings)
        target = Path(LIVE_SETTINGS_FILE)
        parent = _validate_live_settings_parent()
        if os.path.lexists(target) and stat.S_ISLNK(os.lstat(target).st_mode):
            raise RuntimeSafetyError("live settings path must not be a symlink")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(settings, handle, indent=4)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
            parent_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
        return True
    except Exception as e:
        print(f"⚠️ Error saving settings: {e}")
        return False


def _dashboard_auth_configured(settings=None):
    settings = settings or load_live_settings()
    return bool(settings.get('dashboard_user') and settings.get('dashboard_pass'))


def _dashboard_session_signature(settings, username, expires_at):
    secret = settings.get('dashboard_auth_secret') or ''
    payload = f"{username}|{expires_at}"
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _create_dashboard_session_cookie(settings, username, secure=False):
    expires_at = int(t.time()) + DASHBOARD_SESSION_DAYS * 24 * 60 * 60
    signature = _dashboard_session_signature(settings, username, expires_at)
    value = f"{expires_at}|{signature}"
    max_age = DASHBOARD_SESSION_DAYS * 24 * 60 * 60
    secure_attribute = "; Secure" if secure else ""
    return f"{DASHBOARD_SESSION_COOKIE}={value}; Max-Age={max_age}; Path=/; HttpOnly; SameSite=Lax{secure_attribute}"


def _clear_dashboard_session_cookie(secure=False):
    secure_attribute = "; Secure" if secure else ""
    return f"{DASHBOARD_SESSION_COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax{secure_attribute}"


def _is_valid_dashboard_session(headers, settings):
    cookie_header = headers.get('Cookie')
    if not cookie_header:
        return False
    try:
        cookie = SimpleCookie(cookie_header)
        morsel = cookie.get(DASHBOARD_SESSION_COOKIE)
        if not morsel:
            return False
        expires_raw, signature = morsel.value.split('|', 1)
        expires_at = int(expires_raw)
        if expires_at < int(t.time()):
            return False
        username = settings.get('dashboard_user')
        expected = _dashboard_session_signature(settings, username, expires_at)
        return hmac.compare_digest(signature, expected)
    except Exception:
        return False


def _dashboard_login_html():
    return """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>E*TRADE Dashboard Login</title>
    <style>
        body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101418; color: #eef2f6; }
        form { width: min(360px, calc(100vw - 32px)); display: grid; gap: 12px; }
        h1 { margin: 0 0 8px; font-size: 22px; font-weight: 650; }
        input, button { box-sizing: border-box; width: 100%; border-radius: 6px; border: 1px solid #34404c; padding: 12px; font-size: 16px; }
        input { background: #171d23; color: #eef2f6; }
        button { border: 0; background: #2383e2; color: white; font-weight: 650; cursor: pointer; }
        .error { min-height: 20px; color: #ff7676; font-size: 14px; }
    </style>
</head>
<body>
    <form id="login-form">
        <h1>Dashboard Login</h1>
        <input id="username" autocomplete="username" placeholder="Username" required>
        <input id="password" type="password" autocomplete="current-password" placeholder="Password" required>
        <button type="submit">Sign In</button>
        <div id="error" class="error"></div>
    </form>
    <script>
        document.getElementById('login-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const error = document.getElementById('error');
            error.textContent = '';
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    username: document.getElementById('username').value,
                    password: document.getElementById('password').value
                })
            });
            if (res.ok) {
                window.location.href = '/dashboard';
            } else {
                error.textContent = 'Invalid username or password.';
            }
        });
    </script>
</body>
</html>"""


def _dashboard_manifest():
    return {
        "id": "/dashboard",
        "name": "Antigravity Trader Dashboard",
        "short_name": "Trader",
        "description": "Private mobile dashboard for the E*TRADE trading system.",
        "start_url": "/dashboard",
        "scope": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#ffffff",
    }


def _select_target_expiration(expirations, target_date, today_date):
    parsed = [
        datetime.strptime(exp, "%Y-%m-%d").date() if isinstance(exp, str) else exp
        for exp in (expirations or [])
    ]
    candidates = [exp for exp in parsed if exp >= today_date]
    friday_candidates = [exp for exp in candidates if exp.weekday() == 4]
    if friday_candidates:
        candidates = friday_candidates
    later_candidates = [exp for exp in candidates if exp >= target_date]
    if later_candidates:
        return min(later_candidates, key=lambda exp: (exp - target_date).days)
    if candidates:
        return min(candidates, key=lambda exp: abs((exp - target_date).days))
    return None


def _target_expiration_from_settings(accounts_obj, ticker, live_settings, today_date, available_expirations=None):
    ticker_key = "spx" if str(ticker).upper() in ("SPX", "SPXW") else str(ticker).lower()
    saved_expiry = live_settings.get(f"{ticker_key}_target_expiration") or live_settings.get("target_expiration")
    try:
        if saved_expiry:
            saved_expiry = datetime.strptime(saved_expiry, "%Y-%m-%d").date()
            if saved_expiry >= today_date:
                return saved_expiry
    except Exception:
        pass

    target_weeks = int(live_settings.get("target_weeks", 6) or 6)
    target_date = today_date + timedelta(weeks=target_weeks)
    try:
        expirations = available_expirations if available_expirations is not None else accounts_obj.get_available_expirations(ticker)
        selected = _select_target_expiration(expirations, target_date, today_date)
        if selected:
            return selected
    except Exception as e:
        print(f"⚠️ Failed to resolve target expiration for {ticker}: {e}")

    return target_date


def _add_months(base_date, months):
    month_index = base_date.month - 1 + months
    year = base_date.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(year, month, min(base_date.day, days_in_month[month - 1]))


def _normalize_order_payload(order):
    """Return a single order dict from generate_option_order output."""
    if isinstance(order, list):
        return order[0] if order else None
    return order


def _date_parts(expiration):
    if hasattr(expiration, "year"):
        return expiration.year, expiration.month, expiration.day
    parsed = datetime.strptime(str(expiration), "%Y-%m-%d")
    return parsed.year, parsed.month, parsed.day


def _template_leg_symbol(template_order, fallback):
    if isinstance(template_order, dict):
        for leg in template_order.get("legs", []):
            symbol = leg.get("symbol")
            if symbol:
                return symbol
    return fallback


def _build_neutralize_spread_order(ticker, detail, qty, price_type, limit_price, short_action, long_action, template_order=None):
    year, month, day = _date_parts(detail.get("expiration"))
    call_put = str(detail.get("call_put", "")).upper()
    short_strike = float(detail.get("short_strike"))
    long_strike = float(detail.get("long_strike"))
    symbol = _template_leg_symbol(template_order, ticker)
    width = abs(short_strike - long_strike)

    return {
        "client_order_id": random.randint(1000000000, 9999999999),
        "securityType": "OPTN",
        "orderTerm": "GOOD_FOR_DAY",
        "orderAction": "SPREAD",
        "spreadType": "VERTICAL",
        "orderType": "SPREADS",
        "priceType": price_type,
        "limitPrice": round(float(limit_price or 0), 2),
        "legs": [
            {
                "symbol": symbol,
                "orderAction": short_action,
                "quantity": qty,
                "callPut": call_put,
                "expiryYear": year,
                "expiryMonth": month,
                "expiryDay": day,
                "strikePrice": short_strike,
            },
            {
                "symbol": symbol,
                "orderAction": long_action,
                "quantity": qty,
                "callPut": call_put,
                "expiryYear": year,
                "expiryMonth": month,
                "expiryDay": day,
                "strikePrice": long_strike,
            },
        ],
        "required_margin": 0 if price_type == "NET_DEBIT" else width * qty * 100,
    }


def _format_osi(symbol, exp_date, call_put, strike):
    """Build an E*TRADE OSI string for an option leg."""
    if hasattr(exp_date, "year"):
        return f"{symbol}:{exp_date.year}:{exp_date.month:02d}:{exp_date.day:02d}:{call_put}:{strike}"
    parts = str(exp_date).split("-")
    return f"{symbol}:{parts[0]}:{parts[1]}:{parts[2]}:{call_put}:{strike}"


def _quote_spread_midpoint(market, short_lot, long_lot):
    """Quote a vertical spread and return midpoint plus leg prices."""
    short_osi = _format_osi(short_lot.symbol, short_lot.expiration_date, short_lot.call_put, short_lot.strike_price)
    long_osi = _format_osi(long_lot.symbol, long_lot.expiration_date, long_lot.call_put, long_lot.strike_price)

    resp = market.get_quote([short_osi, long_osi], resp_format="json")
    quote_data = resp.get("QuoteResponse", {}).get("QuoteData", [])

    def _match_quote(osi_str):
        parts = osi_str.split(':')
        for q in quote_data:
            p = q.get("Product", {})
            try:
                if (p.get("symbol") == parts[0] and
                    str(p.get("expiryYear")) == parts[1] and
                    int(p.get("expiryMonth")) == int(parts[2]) and
                    int(p.get("expiryDay")) == int(parts[3]) and
                    p.get("callPut") == parts[4] and
                    abs(float(p.get("strikePrice", 0)) - float(parts[5])) < 0.01):
                    return q.get("All", {})
            except Exception:
                continue
        return None

    short_q = _match_quote(short_osi)
    long_q = _match_quote(long_osi)
    if not short_q or not long_q:
        return None

    short_bid = float(short_q.get("bid", 0))
    short_ask = float(short_q.get("ask", 0))
    long_bid = float(long_q.get("bid", 0))
    long_ask = float(long_q.get("ask", 0))
    short_mid = (short_bid + short_ask) / 2.0
    long_mid = (long_bid + long_ask) / 2.0
    return {
        "midpoint": short_mid - long_mid,
        "short_bid": short_bid,
        "short_ask": short_ask,
        "long_bid": long_bid,
        "long_ask": long_ask,
    }


def build_close_proposals(screened, accounts, market, live_settings, rejected_proposals_today):
    """Quote screened spreads and return positions that meet close thresholds."""
    from copy import copy

    proposals = []
    close_mid_threshold = float(live_settings.get("auto_close_midpoint_threshold", 0.30))
    target_gain_threshold = float(live_settings.get("auto_close_gain_threshold", 70.0))

    for entry in screened:
        if not (entry.get("is_spread") and entry.get("short_lot") and entry.get("long_lot")):
            continue

        short_lot = entry["short_lot"]
        long_lot = entry["long_lot"]
        proposal_id = f"{short_lot.symbol}_{short_lot.strike_price}_{short_lot.call_put}_{short_lot.expiration_date}"
        if proposal_id in rejected_proposals_today:
            continue

        try:
            print(f"   Checking quote for {short_lot.symbol} close candidate...")
            quote_info = _quote_spread_midpoint(market, short_lot, long_lot)
            if not quote_info:
                continue

            midpoint = quote_info["midpoint"]
            gain_val = float(entry.get("pair_gain_loss", 0) or 0)
            if gain_val < target_gain_threshold or abs(midpoint) > close_mid_threshold:
                continue

            print(f"\n🚀 DETECTION: Profit Target Met in {short_lot.symbol}")
            print(f"   Gain: {gain_val:.1f}% >= {target_gain_threshold}%")
            print(f"   Cost: ${abs(midpoint):.2f} <= ${close_mid_threshold:.2f}")
            print(f"   Pair Qty: {entry['pair_quantity']}")

            s_lot = copy(short_lot)
            l_lot = copy(long_lot)
            s_lot.quantity = entry["pair_quantity"]
            l_lot.quantity = entry["pair_quantity"]
            close_order = accounts.generate_option_order(
                single_leg_stock_position=None,
                action="SPREAD",
                spread_sell_option=l_lot,
                spread_buy_option=s_lot,
                priceType={"priceType": "NET_DEBIT", "limitPrice": abs(midpoint)}
            )

            expiry_str = short_lot.expiration_date.strftime("%Y-%m-%d") if hasattr(short_lot.expiration_date, "strftime") else str(short_lot.expiration_date)
            proposals.append({
                "ticker": short_lot.symbol,
                "short_strike": short_lot.strike_price,
                "long_strike": long_lot.strike_price,
                "call_put": short_lot.call_put,
                "expiration": expiry_str,
                "pair_quantity": entry["pair_quantity"],
                "gain_pct": short_lot.gain_loss_percentage if hasattr(short_lot, "gain_loss_percentage") else gain_val,
                "midpoint": midpoint,
                "short_bid": quote_info["short_bid"],
                "short_ask": quote_info["short_ask"],
                "long_bid": quote_info["long_bid"],
                "long_ask": quote_info["long_ask"],
                "close_order": close_order,
                "proposal_id": proposal_id,
            })
        except Exception as e:
            print(f"   Error in auto-close logic for {short_lot.symbol}: {e}")
            traceback.print_exc()

    return proposals


def _neutralize_target_expiration(accounts, ticker, live_settings, today):
    return _target_expiration_from_settings(accounts, ticker, live_settings, today)


def _spread_uses_expiration(spread, expiration_date):
    if not spread:
        return False
    sell_option = spread.get("sell_option")
    buy_option = spread.get("buy_option")
    return (
        sell_option is not None and buy_option is not None and
        getattr(sell_option, "expiration_date", None) == expiration_date and
        getattr(buy_option, "expiration_date", None) == expiration_date
    )


def _execute_neutralize_leg(etrade_instance, ticker, leg_name, order_to_send, short_strike, long_strike, call_put, qty, close_reason):
    reject_legacy_execution("legacy dashboard neutralize worker")


def build_neutralize_proposals(screened, accounts, market, live_settings, rejected_proposals_today):
    """Build neutralize proposals for risky short spreads."""
    proposals = []
    target_delta = float(live_settings.get("target_delta", 0.15) or 0.15)
    hedge_spread = float(live_settings.get("hedge_spread", 20.0) or 20.0)
    today = datetime.now().date()
    side_margin_by_expiration = {}

    for entry in screened:
        if not entry.get("is_spread"):
            continue
        short_lot = entry.get("short_lot")
        long_lot = entry.get("long_lot")
        if not short_lot or not long_lot:
            continue
        if getattr(short_lot, "call_put", None) != getattr(long_lot, "call_put", None):
            continue
        if getattr(short_lot, "quantity", 0) >= 0 or getattr(long_lot, "quantity", 0) <= 0:
            continue
        exp = getattr(short_lot, "expiration_date", None)
        if exp is None:
            continue
        try:
            qty = int(entry.get("pair_quantity", 1) or 1)
            width = abs(float(long_lot.strike_price) - float(short_lot.strike_price))
        except Exception:
            continue
        key = (short_lot.symbol, exp, short_lot.call_put)
        side_margin_by_expiration[key] = side_margin_by_expiration.get(key, 0.0) + width * qty * 100.0

    for entry in screened:
        if not entry.get("is_spread"):
            continue

        short_lot = entry.get("short_lot")
        long_lot = entry.get("long_lot")
        if not short_lot or not long_lot:
            continue
        if getattr(short_lot, "call_put", None) != getattr(long_lot, "call_put", None):
            continue
        if getattr(short_lot, "quantity", 0) >= 0 or getattr(long_lot, "quantity", 0) <= 0:
            continue
        if getattr(short_lot, "expiration_date", None) is None:
            continue

        try:
            dte = (short_lot.expiration_date - today).days
        except Exception:
            continue

        try:
            short_delta = float(getattr(short_lot, "delta", 0) or 0)
        except Exception:
            continue

        is_delta_risk = abs(short_delta) > NEUTRALIZE_DELTA_THRESHOLD and dte <= NEUTRALIZE_TRIGGER_DTE
        is_put_delta_risk = short_lot.call_put == "PUT" and is_delta_risk
        has_put_margin = side_margin_by_expiration.get((short_lot.symbol, short_lot.expiration_date, "PUT"), 0.0) > 0
        is_orphan_call_margin = (
            short_lot.symbol in ["SPY", "SPX"] and
            short_lot.call_put == "CALL" and
            not has_put_margin
        )

        if dte < 0 or not (is_delta_risk or is_orphan_call_margin):
            continue

        proposal_id = f"{short_lot.symbol}_{short_lot.strike_price}_{short_lot.call_put}_{short_lot.expiration_date}_neutralize"
        if proposal_id in rejected_proposals_today:
            continue

        try:
            quote_info = _quote_spread_midpoint(market, short_lot, long_lot)
        except Exception as e:
            print(f"   [Neutralize] Quote fetch failed for {short_lot.symbol}: {e}")
            continue
        if not quote_info:
            continue

        qty = int(entry.get("pair_quantity", 1) or 1)
        close_debit = abs(quote_info["midpoint"])
        side_title = "Call" if short_lot.call_put == "CALL" else "Put"
        opposite_title = "Put" if side_title == "Call" else "Call"
        original_spread_width = abs(float(long_lot.strike_price) - float(short_lot.strike_price))
        if original_spread_width <= 0:
            continue

        target_expiration = _neutralize_target_expiration(accounts, short_lot.symbol, live_settings, today)
        days_to_expire = (target_expiration - today).days
        if days_to_expire < 1:
            continue

        try:
            replacement_target_delta = abs(short_delta) / 2.0 if is_put_delta_risk else target_delta
            same_side_spread = accounts.get_option_spread_by_price(
                short_lot.symbol,
                side_title,
                days_to_expire=days_to_expire,
                target_premium=0,
                hedge_ratio=1,
                hedge_spread=original_spread_width,
                qty=qty,
                target_delta=replacement_target_delta
            )

            include_offset = not is_put_delta_risk
            opposite_side_spread = None
            opposite_target_delta = None
            if include_offset:
                if short_lot.symbol == "SPX":
                    pos_hedge_spread = float(live_settings.get('spx_hedge_spread', live_settings.get('hedge_spread', 200.0)) or 200.0)
                else:
                    pos_hedge_spread = float(live_settings.get('spy_hedge_spread', live_settings.get('hedge_spread', 20.0)) or 20.0)

                opposite_target_delta = target_delta
                opposite_side_spread = accounts.get_option_spread_by_price(
                    short_lot.symbol,
                    opposite_title,
                    days_to_expire=days_to_expire,
                    target_premium=0,
                    hedge_ratio=1,
                    hedge_spread=pos_hedge_spread,
                    qty=qty,
                    target_delta=opposite_target_delta
                )
        except Exception as e:
            print(f"   [Neutralize] Failed to build replacement spreads for {short_lot.symbol}: {e}")
            continue

        if not same_side_spread or (include_offset and not opposite_side_spread):
            continue
        if not _spread_uses_expiration(same_side_spread, target_expiration):
            print(f"   [Neutralize] Skipping replacement that fell back before target expiration {target_expiration}.")
            continue
        if include_offset and not _spread_uses_expiration(opposite_side_spread, target_expiration):
            print(f"   [Neutralize] Skipping offset that fell back before target expiration {target_expiration}.")
            continue

        same_profit = float(same_side_spread.get("profit", 0) or 0)
        opposite_profit = float(opposite_side_spread.get("profit", 0) or 0) if include_offset else 0.0
        if same_profit <= 0.01 or (include_offset and opposite_profit <= 0.01):
            continue

        same_sell = same_side_spread.get("sell_option")
        same_buy = same_side_spread.get("buy_option")
        opp_sell = opposite_side_spread.get("sell_option") if include_offset else None
        opp_buy = opposite_side_spread.get("buy_option") if include_offset else None
        if not same_sell or not same_buy or (include_offset and (not opp_sell or not opp_buy)):
            continue

        close_order = accounts.generate_option_order(
            single_leg_stock_position=None,
            action="SPREAD",
            spread_sell_option=long_lot,
            spread_buy_option=short_lot,
            priceType={"priceType": "NET_DEBIT", "limitPrice": round(close_debit, 2)}
        )
        same_order = accounts.generate_option_order(
            single_leg_stock_position=None,
            action="SPREAD",
            spread_sell_option=same_sell,
            spread_buy_option=same_buy,
            priceType={"priceType": "NET_CREDIT", "limitPrice": round(same_profit, 2)}
        )
        opp_order = None
        if include_offset:
            opp_order = accounts.generate_option_order(
                single_leg_stock_position=None,
                action="SPREAD",
                spread_sell_option=opp_sell,
                spread_buy_option=opp_buy,
                priceType={"priceType": "NET_CREDIT", "limitPrice": round(opposite_profit, 2)}
            )

        offset_details = None
        if include_offset:
            offset_details = {
                "call_put": opposite_title.upper(),
                "expiration": opp_sell.expiration_date.strftime("%Y-%m-%d"),
                "short_strike": float(opp_sell.strike_price),
                "long_strike": float(opp_buy.strike_price),
                "spread_width": round(abs(float(opp_buy.strike_price) - float(opp_sell.strike_price)), 2),
                "delta": round(float(getattr(opp_sell, "delta", 0) or 0), 4),
                "target_delta": round(float(opposite_target_delta), 4),
                "credit": round(float(opposite_side_spread.get("profit", 0) or 0), 2),
            }

        proposals.append({
            "proposal_id": proposal_id,
            "ticker": short_lot.symbol,
            "qty": qty,
            "trigger": " & ".join([r for r, cond in [("Delta Risk", is_delta_risk), ("Orphan CALL Margin", is_orphan_call_margin)] if cond]) or "Unknown",
            "original": {
                "call_put": short_lot.call_put,
                "expiration": short_lot.expiration_date.strftime("%Y-%m-%d"),
                "short_strike": float(short_lot.strike_price),
                "long_strike": float(long_lot.strike_price),
                "delta": round(short_delta, 4),
                "dte": int(dte),
                "midpoint_debit": round(close_debit, 2),
                "spread_width": round(original_spread_width, 2),
            },
            "replacement": {
                "call_put": side_title.upper(),
                "expiration": same_sell.expiration_date.strftime("%Y-%m-%d"),
                "short_strike": float(same_sell.strike_price),
                "long_strike": float(same_buy.strike_price),
                "spread_width": round(abs(float(same_buy.strike_price) - float(same_sell.strike_price)), 2),
                "delta": round(float(getattr(same_sell, "delta", 0) or 0), 4),
                "target_delta": round(float(replacement_target_delta), 4),
                "credit": round(float(same_side_spread.get("profit", 0) or 0), 2),
            },
            "offset": offset_details,
            "estimated_close_debit": round(close_debit, 2),
            "estimated_open_credit": round(same_profit + opposite_profit, 2),
            "estimated_net_credit": round(same_profit + opposite_profit - close_debit, 2),
            "has_open_order": False,
            "orders": {
                "close_order": _normalize_order_payload(close_order),
                "replacement_order": _normalize_order_payload(same_order),
                "offset_order": _normalize_order_payload(opp_order) if include_offset else None,
            }
        })

    return proposals


# --- SPY GEX Calculation & Mock Fallback ---

# In-memory cache for SPY GEX results to prevent spamming E*TRADE API
spy_gex_cache = {
    "timestamp": 0.0,
    "data": None
}

spy_regime_cache = {
    "timestamp": 0.0,
    "data": None
}


def _pct_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value) * 100.0
    except Exception:
        return None


def _latest_regime_from_diagnostic_csv(error=None):
    path = os.path.join("research_reports", "regime_diagnostics", "causal_regime_trace.csv")
    if not os.path.exists(path):
        return {
            "available": False,
            "error": error or "No HMM model loaded and no diagnostic trace found.",
            "source": "unavailable",
        }

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError("diagnostic trace is empty")
        row = df.iloc[-1]
        return _format_regime_row(row, df.index[-1], source="diagnostic_csv_fallback", error=error)
    except Exception as exc:
        return {
            "available": False,
            "error": error or str(exc),
            "source": "unavailable",
        }


def _format_regime_row(row, row_date, source, error=None):
    prob_cols = [c for c in row.index if str(c).startswith("prob_state_")]
    raw_confidence = None
    if prob_cols:
        try:
            raw_confidence = float(pd.to_numeric(row[prob_cols], errors="coerce").max())
        except Exception:
            raw_confidence = None

    detected_prob_cols = [c for c in row.index if str(c).startswith("detected_prob_state_")]
    detected_confidence = None
    if detected_prob_cols:
        try:
            detected_confidence = float(pd.to_numeric(row[detected_prob_cols], errors="coerce").max())
        except Exception:
            detected_confidence = None

    def get_int(field):
        try:
            value = row.get(field)
            if pd.isna(value):
                return None
            return int(value)
        except Exception:
            return None

    def get_str(field, default=None):
        value = row.get(field, default)
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return str(value)

    date_str = pd.Timestamp(row_date).strftime("%Y-%m-%d")
    return {
        "available": True,
        "source": source,
        "stale": source != "live_hmm",
        "error": error,
        "as_of_date": date_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_timestamp": get_str("Regime_Signal_Timestamp", "close_T_for_next_session"),
        "hmm_refit_date": get_str("HMM_Refit_Date"),
        "raw_hmm_state": get_int("Raw_HMM_State") if "Raw_HMM_State" in row.index else get_int("HMM_State"),
        "raw_hmm_label": get_str("Raw_Regime_Label") if "Raw_Regime_Label" in row.index else get_str("Regime_Label"),
        "raw_hmm_confidence": raw_confidence,
        "detected_regime_state": get_int("Detected_Regime_State"),
        "detected_regime_label": get_str("Detected_Regime_Label"),
        "detected_regime_confidence": detected_confidence,
        "stress_overlay": get_str("Stress_Overlay", "none"),
        "spy_close": None if pd.isna(row.get("SPY_Close", np.nan)) else float(row.get("SPY_Close")),
        "vix_close": None if pd.isna(row.get("VIX_Close", row.get("Stress_VIX_Close", np.nan))) else float(row.get("VIX_Close", row.get("Stress_VIX_Close"))),
        "stress_21d_drawdown_pct": _pct_or_none(row.get("Stress_21d_Drawdown")),
        "stress_5d_log_return_pct": _pct_or_none(row.get("Stress_5d_Log_Return")),
        "stress_1d_log_return_pct": _pct_or_none(row.get("Stress_1d_Log_Return")),
    }


def calculate_spy_regime_status(hmm_model):
    """
    Return the latest HMM regime status for the dashboard.

    The HMM uses close-T market data, so the result is labeled for next-session
    use and should be displayed as risk context rather than intraday clairvoyance.
    """
    global spy_regime_cache

    now = t.time()
    if spy_regime_cache["data"] is not None and (now - spy_regime_cache["timestamp"]) < 900:
        return spy_regime_cache["data"]

    if hmm_model is None:
        data = _latest_regime_from_diagnostic_csv("No live HMM model is loaded.")
        spy_regime_cache = {"timestamp": now, "data": data}
        return data

    try:
        historical_df = fetch_historical_data()
        if historical_df.empty:
            raise ValueError("historical data is empty")

        as_of_ts = pd.Timestamp(datetime.now()).normalize()
        causal_df = historical_df.loc[historical_df.index <= as_of_ts].copy()
        if causal_df.empty:
            raise ValueError(f"no historical rows available as of {as_of_ts.date()}")

        feature_df = _build_causal_regime_feature_frame(causal_df, hmm_model, as_of_ts)
        if feature_df.empty:
            raise ValueError("HMM scoring returned no rows")

        last_date = feature_df.index[-1]
        row = feature_df.iloc[-1].copy()
        raw_row = causal_df.reindex(feature_df.index).iloc[-1]
        for col in ("SPY_Close", "VIX_Close"):
            if col in raw_row.index and col not in row.index:
                row[col] = raw_row[col]

        data = _format_regime_row(row, last_date, source="live_hmm")
        spy_regime_cache = {"timestamp": now, "data": data}
        return data
    except Exception as exc:
        print(f"⚠️ Error calculating SPY HMM regime status: {exc}")
        data = _latest_regime_from_diagnostic_csv(str(exc))
        spy_regime_cache = {"timestamp": now, "data": data}
        return data


def calculate_regime_v2_shadow_status():
    """Read the sealed V2 advisory snapshot without touching trading state."""

    try:
        payload = REGIME_V2_SHADOW_STORE.dashboard_payload()
    except Exception:
        print("⚠️ Regime V2 shadow status is unavailable.")
        return unavailable_dashboard_payload("invalid")
    if (
        not isinstance(payload, dict)
        or set(payload) != REGIME_V2_PUBLIC_DASHBOARD_FIELDS
        or payload.get("may_authorize_execution") is not False
    ):
        return unavailable_dashboard_payload("invalid")
    return payload


def fetch_option_chain_for_gex(accounts_instance, symbol, expiration, allow_auth_refresh=True):
    """
    Fetch the option chain for a given symbol and expiration, near the spot price.
    """
    try:
        spot_price = accounts_instance.get_stock_price(symbol)
        if not spot_price:
            return []
            
        url = f"{accounts_instance.base_url}/v1/market/optionchains.json"
        symbol_converted = "BRK.B" if symbol == "BRKB" else symbol
        
        if isinstance(expiration, str):
            dt_obj = datetime.strptime(expiration, "%Y-%m-%d")
        elif hasattr(expiration, "strftime"):
            dt_obj = datetime.combine(expiration, datetime.min.time())
        else:
            dt_obj = expiration

        params = {
            "symbol": symbol_converted,
            "expiryYear": dt_obj.year,
            "expiryMonth": dt_obj.month,
            "expiryDay": dt_obj.day,
            "includeWeekly": True,
            "skipAdjusted": True,
            "optionCategory": "STANDARD",
            "strikePriceNear": round(spot_price),
            "noOfStrikes": 100
        }
        
        response = accounts_instance.session.get(url, params=params, auth=accounts_instance.session.auth)
        if is_etrade_token_expired_response(response) and allow_auth_refresh:
            refreshed = _refresh_etrade_session(f"{symbol} GEX option chain")
            if refreshed:
                refreshed_accounts = globals().get("accounts", accounts_instance)
                return fetch_option_chain_for_gex(refreshed_accounts, symbol, expiration, allow_auth_refresh=False)
        if response.status_code != 200:
            print(f"⚠️ Error fetching option chain for {symbol} on {expiration}: {response.status_code}")
            return []
            
        data = response.json()
        option_pairs = data.get("OptionChainResponse", {}).get("OptionPair", [])
        return option_pairs
    except Exception as e:
        print(f"⚠️ Exception in fetch_option_chain_for_gex: {e}")
        return []

def generate_mock_spy_gex(spot_price=510.0):
    """
    Generate mathematically consistent mock GEX data for SPY when live options data is unavailable.
    """
    import random
    
    strikes = list(range(int(spot_price) - 30, int(spot_price) + 30))
    net_gex = []
    call_gex = []
    put_gex = []
    
    for s in strikes:
        dist = s - spot_price
        # Gamma is highest ATM and decays exponentially
        approx_gamma = float(np.exp(-(dist**2) / (2 * 8.0**2)) * (0.05 + random.uniform(-0.005, 0.005)))
        
        # Puts (negative GEX)
        approx_put_oi = float(np.exp(-((dist + 10)**2) / (2 * 12.0**2)) * 8000 * random.uniform(0.8, 1.2) if dist < 5 else 100)
        # Calls (positive GEX)
        approx_call_oi = float(np.exp(-((dist - 10)**2) / (2 * 12.0**2)) * 9000 * random.uniform(0.8, 1.2) if dist > -5 else 100)
        
        p_gex = -approx_gamma * approx_put_oi * 100.0 * spot_price / 1_000_000.0
        c_gex = approx_gamma * approx_call_oi * 100.0 * spot_price / 1_000_000.0
        
        call_gex.append(float(round(c_gex, 2)))
        put_gex.append(float(round(p_gex, 2)))
        net_gex.append(float(round(c_gex + p_gex, 2)))
        
    call_wall = int(strikes[int(np.argmax(call_gex))])
    put_wall = int(strikes[int(np.argmin(put_gex))])
    
    # Zero gamma flip search
    zero_gamma = float(spot_price)
    for i in range(len(strikes) - 1):
        if net_gex[i] < 0 and net_gex[i+1] > 0:
            zero_gamma = float(strikes[i] + (strikes[i+1] - strikes[i]) * (-net_gex[i]) / (net_gex[i+1] - net_gex[i]))
            break
            
    total_net = float(sum(net_gex))
    total_call = float(sum(call_gex))
    total_put = float(sum(put_gex))
    
    return {
        "spot_price": float(round(spot_price, 2)),
        "total_net_gex": float(round(total_net, 2)),
        "total_call_gex": float(round(total_call, 2)),
        "total_put_gex": float(round(total_put, 2)),
        "call_wall": int(call_wall),
        "put_wall": int(put_wall),
        "zero_gamma": float(round(zero_gamma, 2)),
        "expirations": [(datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d") for d in [1, 3, 5, 8, 15]],
        "chart_data": {
            "strikes": [int(s) for s in strikes],
            "net_gex": [float(v) for v in net_gex],
            "call_gex": [float(v) for v in call_gex],
            "put_gex": [float(v) for v in put_gex]
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (MOCK)",
        "is_mock": True
    }

def calculate_spy_gex(accounts_instance):
    """
    Fetch SPY option chains and calculate Gamma Exposure (GEX) profile.
    """
    global spy_gex_cache
    
    now = t.time()
    if spy_gex_cache["data"] is not None and (now - spy_gex_cache["timestamp"]) < 60:
        print("⚡ Returning cached SPY GEX data")
        return spy_gex_cache["data"]
        
    print("🔄 Fetching SPY option chains for GEX calculation...")
    
    spot_price = None
    try:
        spot_price = accounts_instance.get_stock_price("SPY")
    except Exception as e:
        print(f"⚠️ Failed to get SPY price from E*TRADE: {e}")
        
    if not spot_price:
        # Fallback to yfinance
        try:
            ticker_obj = yf.Ticker("SPY")
            data = ticker_obj.history(period="1d")
            if not data.empty:
                spot_price = float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ yfinance fallback for spot price failed: {e}")
            
    if not spot_price:
        print("⚠️ Unable to get a live SPY spot price. Returning mock GEX profile.")
        return generate_mock_spy_gex(510.0)
        
    expirations = None
    try:
        expirations = accounts_instance.get_available_expirations("SPY")
    except Exception as e:
        print(f"⚠️ Failed to get SPY expirations from E*TRADE: {e}")
        
    if not expirations:
        # Fallback to yfinance
        try:
            ticker_obj = yf.Ticker("SPY")
            expirations = ticker_obj.options
        except Exception as e:
            print(f"⚠️ yfinance fallback for expirations failed: {e}")
            
    if not expirations:
        print("⚠️ No expirations found. Returning mock GEX profile.")
        return generate_mock_spy_gex(spot_price)
        
    # Standardize and sort expirations
    parsed_expirations = []
    for exp in expirations:
        try:
            if isinstance(exp, str):
                parsed_expirations.append(datetime.strptime(exp, "%Y-%m-%d").date())
            else:
                parsed_expirations.append(exp)
        except Exception:
            continue
    parsed_expirations = sorted(parsed_expirations)
    
    today = datetime.now().date()
    active_expirations = [exp for exp in parsed_expirations if exp >= today]
    
    target_expirations = active_expirations[:5]
    if not target_expirations:
        print("⚠️ No active expirations found. Returning mock GEX profile.")
        return generate_mock_spy_gex(spot_price)
        
    print(f"Targeting SPY expirations: {target_expirations}")
    
    strike_gex = {}
    strike_call_gex = {}
    strike_put_gex = {}
    
    total_net_gex = 0.0
    total_call_gex = 0.0
    total_put_gex = 0.0
    
    success_count = 0
    for expiration in target_expirations:
        option_pairs = fetch_option_chain_for_gex(accounts_instance, "SPY", expiration)
        if option_pairs:
            success_count += 1
            
        for pair in option_pairs:
            for opt_type in ["Call", "Put"]:
                opt = pair.get(opt_type)
                if not opt:
                    continue
                    
                strike = float(opt.get("strikePrice", 0.0))
                oi = int(opt.get("openInterest", 0) or 0)
                
                greeks = opt.get("OptionGreeks") or {}
                gamma = 0.0
                if greeks:
                    try:
                        gamma = float(greeks.get("gamma", 0.0) or 0.0)
                    except (ValueError, TypeError):
                        gamma = 0.0
                        
                if gamma == 100.0 or gamma < 0.0 or gamma > 5.0:
                    gamma = 0.0
                    
                is_call = (opt_type == "Call")
                sign = 1.0 if is_call else -1.0
                
                gex_value = gamma * oi * 100.0 * spot_price * sign
                gex_value_millions = gex_value / 1_000_000.0
                
                if strike not in strike_gex:
                    strike_gex[strike] = 0.0
                    strike_call_gex[strike] = 0.0
                    strike_put_gex[strike] = 0.0
                    
                strike_gex[strike] += gex_value_millions
                if is_call:
                    strike_call_gex[strike] += gex_value_millions
                    total_call_gex += gex_value_millions
                else:
                    strike_put_gex[strike] += gex_value_millions
                    total_put_gex += gex_value_millions
                    
                total_net_gex += gex_value_millions
                
    if success_count == 0 or not strike_gex:
        print("⚠️ E*TRADE returned empty option chains or failed. Falling back to mock GEX profile.")
        return generate_mock_spy_gex(spot_price)
        
    sorted_strikes = sorted(strike_gex.keys())
    call_wall = max(strike_call_gex.items(), key=lambda x: x[1], default=(0.0, 0.0))[0]
    put_wall = min(strike_put_gex.items(), key=lambda x: x[1], default=(0.0, 0.0))[0]
    
    zero_gamma = None
    for i in range(len(sorted_strikes) - 1):
        s1, s2 = sorted_strikes[i], sorted_strikes[i+1]
        g1, g2 = strike_gex[s1], strike_gex[s2]
        if g1 < 0 and g2 > 0:
            zero_gamma = float(s1 + (s2 - s1) * (-g1) / (g2 - g1))
            break
            
    if zero_gamma is None:
        zero_gamma = float(min(strike_gex.items(), key=lambda x: abs(x[1]), default=(0.0, 0.0))[0])
        
    lower_bound = spot_price * 0.94
    upper_bound = spot_price * 1.06
    filtered_strikes = [s for s in sorted_strikes if lower_bound <= s <= upper_bound]
    
    if not filtered_strikes:
        filtered_strikes = sorted_strikes
        
    chart_data = {
        "strikes": [int(s) for s in filtered_strikes],
        "net_gex": [float(round(strike_gex[s], 2)) for s in filtered_strikes],
        "call_gex": [float(round(strike_call_gex[s], 2)) for s in filtered_strikes],
        "put_gex": [float(round(strike_put_gex[s], 2)) for s in filtered_strikes],
    }
    
    result = {
        "spot_price": float(round(spot_price, 2)),
        "total_net_gex": float(round(total_net_gex, 2)),
        "total_call_gex": float(round(total_call_gex, 2)),
        "total_put_gex": float(round(total_put_gex, 2)),
        "call_wall": float(round(call_wall, 2)),
        "put_wall": float(round(put_wall, 2)),
        "zero_gamma": float(round(zero_gamma, 2)),
        "expirations": [exp.strftime("%Y-%m-%d") for exp in target_expirations],
        "chart_data": chart_data,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_mock": False
    }
    
    spy_gex_cache["data"] = result
    spy_gex_cache["timestamp"] = now
    
    return result

def log_gex_minute(accounts_instance):
    """
    Calculate SPY GEX and log it to spy_gex_intraday_log.csv.
    Logs: Timestamp, SPY_Spot, Total_Net_GEX, Zero_Gamma, Call_Wall, Put_Wall, Setup (LONG/SHORT_GAMMA), Is_Mock
    """
    CSV_FILE = "spy_gex_intraday_log.csv"
    try:
        gex_data = calculate_spy_gex(accounts_instance)
        if not gex_data:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        spy_spot = gex_data.get("spot_price")
        net_gex = gex_data.get("total_net_gex")
        zero_gamma = gex_data.get("zero_gamma")
        call_wall = gex_data.get("call_wall")
        put_wall = gex_data.get("put_wall")
        is_mock = gex_data.get("is_mock", False)
        
        setup = "LONG_GAMMA" if net_gex >= 0 else "SHORT_GAMMA"
        
        # Check if file exists to write header
        write_header = not os.path.exists(CSV_FILE)
        
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "Timestamp", "SPY_Spot", "Total_Net_GEX_M", 
                    "Zero_Gamma_Flip", "Call_Wall", "Put_Wall", 
                    "Setup", "Is_Mock"
                ])
            writer.writerow([
                timestamp, spy_spot, net_gex, 
                zero_gamma, call_wall, put_wall, 
                setup, is_mock
            ])
            
        print(f"📝 [GEX Logger] Logged intraday data to {CSV_FILE}: SPY=${spy_spot}, NetGEX={net_gex}M, Flip=${zero_gamma}, Wall=C${call_wall}/P${put_wall} ({setup})")
    except Exception as e:
        print(f"⚠️ [GEX Logger] Error logging GEX data: {e}")


class RefreshHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for the dashboard and API."""
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def _request_uses_https(self):
        forwarded_proto = self.headers.get('X-Forwarded-Proto', '').split(',', 1)[0].strip().lower()
        if forwarded_proto == 'https':
            return True
        forwarded = self.headers.get('Forwarded', '').lower().replace(',', ';')
        return any(part.strip() == 'proto=https' for part in forwarded.split(';'))
    
    def _send_safe_response(self, code, content, content_type='application/json', headers=None):
        """Send a response while safely handling BrokenPipeError."""
        try:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            response_headers = {
                'Cache-Control': 'no-store, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer',
                'Cross-Origin-Resource-Policy': 'same-origin',
            }
            response_headers.update(headers or {})
            for key, value in response_headers.items():
                self.send_header(key, value)
            self.end_headers()
            if content is None:
                return
            if isinstance(content, str):
                self.wfile.write(content.encode())
            elif isinstance(content, bytes):
                self.wfile.write(content)
            else:
                self.wfile.write(json.dumps(content).encode())
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we could finish sending the response
            pass
        except Exception as e:
            print(f"⚠️ Error sending response: {e}")

    def _send_private_json_response(self, code, content):
        """Send authenticated same-origin JSON without permissive CORS."""

        try:
            encoded = json.dumps(
                content,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(code)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8",
            )
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Vary", "Authorization, Cookie")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            print("⚠️ Error sending private dashboard response.")

    def do_OPTIONS(self):
        """Reject cross-origin preflight for this same-origin dashboard."""
        try:
            self._send_private_json_response(405, {"error": "Method not allowed"})
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def check_auth(self, auth_header):
        """Verify Basic Auth credentials against settings."""
        settings = load_live_settings()
        user = settings.get('dashboard_user')
        password = settings.get('dashboard_pass')
        
        if not user or not password:
            return False

        if _is_valid_dashboard_session(self.headers, settings):
            return True
            
        if not auth_header or not auth_header.startswith('Basic '):
            return False
            
        try:
            auth_decoded = base64.b64decode(auth_header[6:]).decode('utf-8')
            u, p = auth_decoded.split(':', 1)
            return u == user and p == password
        except Exception:
            return False

    def _send_unauthorized(self):
        self._send_safe_response(401, {"error": "Authentication required"})

    def do_POST(self):
        """Handle POST requests safely."""
        try:
            self._do_POST_logic()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"⚠️ POST Handler Error: {e}")
            traceback.print_exc()
            self._send_safe_response(500, {"error": str(e)})

    def _do_POST_logic(self):
        if self.path.startswith('/api/login'):
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data.decode('utf-8'))
            settings = load_live_settings()
            if not _dashboard_auth_configured(settings):
                self._send_safe_response(400, {"error": "Dashboard username/password are not configured."})
                return
            username = str(data.get('username') or '')
            password = str(data.get('password') or '')
            if username == settings.get('dashboard_user') and password == settings.get('dashboard_pass'):
                self._send_safe_response(
                    200,
                    {"status": "ok", "session_days": DASHBOARD_SESSION_DAYS},
                    headers={"Set-Cookie": _create_dashboard_session_cookie(
                        settings, username, secure=self._request_uses_https()
                    )}
                )
                return
            self._send_safe_response(403, {"error": "Invalid username or password"})
            return

        if self.path.startswith('/api/logout'):
            self._send_safe_response(
                200,
                {"status": "ok"},
                headers={"Set-Cookie": _clear_dashboard_session_cookie(
                    secure=self._request_uses_https()
                )}
            )
            return

        # Security: Check Basic Auth
        if not self.check_auth(self.headers.get('Authorization')):
            self._send_unauthorized()
            return

        request_path = self.path.split("?", 1)[0]
        if request_path in DISABLED_DASHBOARD_EXECUTION_PATHS:
            try:
                reject_legacy_execution(f"legacy dashboard endpoint {request_path}")
            except LegacyExecutionDisabled:
                self._send_safe_response(
                    503,
                    {
                        "code": "LEGACY_EXECUTION_DISABLED",
                        "error": "Trading actions are disabled.",
                        "read_only": True,
                        "execution_enabled": False,
                    },
                )
            return

        if request_path == '/refresh':
            print("\n🔄 [Refresh Server] Manual refresh requested via web UI")
            positions_version = None
            if os.path.exists("screened_option_pairs.html"):
                positions_version = str(os.stat("screened_option_pairs.html").st_mtime_ns)
            refresh_generation = _queue_portfolio_refresh()
            REFRESH_REQUESTED.set()
            self._send_safe_response(200, {
                "status": "ok",
                "message": "Refresh triggered.",
                "positions_version": positions_version,
                "refresh_generation": refresh_generation,
            })

        elif self.path == '/api/verify_pin':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length else b'{}'
            data = json.loads(post_data.decode('utf-8'))
            settings = load_live_settings()
            submitted_pin = str(data.get('pin') or '')
            configured_pin = str(settings.get('pin') or '')
            if not submitted_pin or not hmac.compare_digest(submitted_pin, configured_pin):
                self._send_safe_response(403, {"error": "Invalid PIN"})
                return
            self._send_safe_response(200, {"status": "ok"})
        
        elif self.path.startswith('/api/settings'):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_settings = json.loads(post_data.decode('utf-8'))
            current_settings = load_live_settings()
            
            if new_settings.get('pin') != current_settings.get('pin'):
                log_dashboard_request(self.path, {"error": "Invalid PIN attempt"})
                self._send_safe_response(403, {"error": "Invalid PIN"})
                return
            
            log_dashboard_request(self.path, new_settings)

            for key in ['target_delta', 'hedge_spread', 'spy_hedge_spread', 'spx_hedge_spread', 'trade_start_time', 'trade_end_time', 'auto_close_midpoint_threshold', 'auto_close_gain_threshold', 'pair_quantity', 'spy_pair_quantity', 'spx_pair_quantity', 'target_weeks', 'target_expiration', 'spy_target_expiration', 'spx_target_expiration', 'trade_side', 'dashboard_user', 'dashboard_pass']:
                if key in new_settings:
                    current_settings[key] = new_settings[key]
            current_settings['auto_open_enabled'] = False
            
            if save_live_settings(current_settings):
                self._send_safe_response(200, {"status": "ok"})
            else:
                self._send_safe_response(500, {"error": "Failed to save settings"})

        else:
            self._send_safe_response(404, {"error": "Not found"})

    def do_GET(self):
        """Handle GET requests safely."""
        try:
            self._do_GET_logic()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            if self.path.split("?", 1)[0] == "/api/regime_v2_shadow":
                print("⚠️ Regime V2 shadow request failed.")
                self._send_private_json_response(
                    500,
                    unavailable_dashboard_payload("invalid"),
                )
                return
            print(f"⚠️ GET Handler Error: {e}")
            traceback.print_exc()
            self._send_safe_response(500, {"error": str(e)})

    def _do_GET_logic(self):
        request_path = self.path.split("?", 1)[0]

        if request_path == '/manifest.webmanifest':
            self._send_safe_response(200, _dashboard_manifest(), 'application/manifest+json')
            return

        if request_path == '/login':
            self._send_safe_response(200, _dashboard_login_html(), 'text/html')
            return

        # Security: Check Basic Auth
        if not self.check_auth(self.headers.get('Authorization')):
            if request_path == "/api/regime_v2_shadow":
                self._send_private_json_response(
                    401,
                    {"error": "Authentication required"},
                )
            elif request_path == '/dashboard' or request_path == '/':
                self.send_response(302)
                self.send_header('Location', '/login')
                self.end_headers()
            else:
                self._send_unauthorized()
            return

        if request_path == '/':
            self.send_response(302)
            self.send_header('Location', '/dashboard')
            self.end_headers()
            return
        
        elif request_path == '/dashboard':
            template_path = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
            if not os.path.exists(template_path):
                template_path = "live_trading/dashboard_template.html"
            
            with open(template_path, "r") as f:
                html = f.read()
            self._send_safe_response(200, html, 'text/html')

        elif request_path == "/api/regime_v2_shadow":
            payload = calculate_regime_v2_shadow_status()
            status = 200 if payload.get("available") else 503
            self._send_private_json_response(status, payload)

        elif self.path.startswith('/api/settings'):
            settings = load_live_settings()
            hidden_settings = {'pin', 'dashboard_pass', 'dashboard_auth_secret'}
            display_settings = {k: v for k, v in settings.items() if k not in hidden_settings}
            self._send_safe_response(200, display_settings)

        elif self.path.startswith('/api/status'):
            is_open, market_status, _, _ = is_market_open()
            trade_status = load_trade_status()
            settings = load_live_settings()
            today = datetime.now()
            fallback_target_expiration = today.date() + timedelta(weeks=int(settings.get('target_weeks', 6) or 6))
            spy_target_expiration = fallback_target_expiration
            spx_target_expiration = fallback_target_expiration
            
            spy_available_expirations = []
            spx_available_expirations = []
            if 'accounts' in globals() and accounts is not None:
                today_str = today.strftime("%Y-%m-%d")
                expiration_cutoff = _add_months(today.date(), 4).strftime("%Y-%m-%d")
                try:
                    spy_available_expirations = [
                        exp for exp in accounts.get_available_expirations("SPY")
                        if today_str <= exp <= expiration_cutoff
                    ]
                    spy_target_expiration = _target_expiration_from_settings(
                        accounts, "SPY", settings, today.date(), spy_available_expirations
                    )
                except Exception as e:
                    print(f"⚠️ Failed to get SPY expirations: {e}")
                try:
                    spx_available_expirations = [
                        exp for exp in accounts.get_available_expirations("SPX")
                        if today_str <= exp <= expiration_cutoff
                    ]
                    spx_target_expiration = _target_expiration_from_settings(
                        accounts, "SPX", settings, today.date(), spx_available_expirations
                    )
                except Exception as e:
                    print(f"⚠️ Failed to get SPX expirations: {e}")

            broker_open_orders = []
            broker_executed_orders = []
            broker_cancelled_orders = []
            if 'etrade_instance' in globals() and etrade_instance is not None:
                try:
                    broker_open_orders = etrade_instance.order.get_open_orders()
                except Exception as e:
                    print(f"[Manual Order Status] Warning: Could not fetch open broker orders: {e}")
                try:
                    broker_executed_orders = etrade_instance.order.get_executed_orders(today.strftime("%Y-%m-%d"))
                except Exception as e:
                    print(f"[Manual Order Status] Warning: Could not fetch executed broker orders: {e}")
                try:
                    broker_cancelled_orders = etrade_instance.order.get_cancelled_orders(today.strftime("%Y-%m-%d"))
                except Exception as e:
                    print(f"[Manual Order Status] Warning: Could not fetch cancelled broker orders: {e}")
            manual_orders_snapshot = get_manual_trade_status_snapshot(broker_open_orders, broker_executed_orders, broker_cancelled_orders)
            status = {
                "market_status": market_status,
                "is_open": is_open,
                "last_trade_date": trade_status.get("last_trade_date"),
                "target_expiration": spy_target_expiration.strftime("%Y-%m-%d"),
                "spy_target_expiration": spy_target_expiration.strftime("%Y-%m-%d"),
                "spx_target_expiration": spx_target_expiration.strftime("%Y-%m-%d"),
                "available_expirations": spy_available_expirations,
                "spy_available_expirations": spy_available_expirations,
                "spx_available_expirations": spx_available_expirations,
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "account_id": accounts.account.get('accountId', 'N/A') if accounts and accounts.account else 'N/A',
                "manual_order": manual_orders_snapshot.get("latest", {}),
                "manual_orders": manual_orders_snapshot
            }
            self._send_safe_response(200, status)

        elif self.path.startswith('/api/preview_spread'):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            settings = load_live_settings()
            ticker = query.get('ticker', ["SPY"])[0].upper()
            if ticker not in ["SPY", "SPX"]:
                ticker = "SPY"

            target_delta = float(query.get('delta', [settings.get('target_delta', 0.15)])[0])
            
            if ticker == "SPX":
                default_width = settings.get('spx_hedge_spread', settings.get('hedge_spread', 200.0))
            else:
                default_width = settings.get('spy_hedge_spread', settings.get('hedge_spread', 20.0))

            hedge_spread = float(query.get('width', [default_width])[0])
            target_weeks = int(query.get('weeks', [settings.get('target_weeks', 6)])[0])
            explicit_expiry = query.get('expiration', [None])[0]
            automatic_expiry = explicit_expiry == ""
            side = query.get('side', ["Put"])[0].capitalize()
            
            today = datetime.now()
            if explicit_expiry:
                expiry_dt = datetime.strptime(explicit_expiry, "%Y-%m-%d")
                selected_expiration = expiry_dt.date()
                days_to_expire = (expiry_dt.date() - today.date()).days
                preview_expiration = explicit_expiry
            else:
                preview_settings = dict(settings)
                preview_settings["target_weeks"] = target_weeks
                if automatic_expiry:
                    ticker_key = "spx" if ticker == "SPX" else "spy"
                    preview_settings[f"{ticker_key}_target_expiration"] = None
                    preview_settings["target_expiration"] = None
                target_expiration = _target_expiration_from_settings(accounts, ticker, preview_settings, today.date())
                selected_expiration = target_expiration
                days_to_expire = (target_expiration - today.date()).days
                if days_to_expire < 1:
                    days_to_expire = 7
                    selected_expiration = (today + timedelta(days=7)).date()
                    preview_expiration = selected_expiration.strftime("%Y-%m-%d")
                else:
                    preview_expiration = target_expiration.strftime("%Y-%m-%d")

            if 'accounts' not in globals() or accounts is None:
                self._send_safe_response(503, {"error": "Engine loading..."})
                return

            spread = accounts.get_option_spread_by_price(
                ticker, side, days_to_expire=days_to_expire, 
                target_premium=0, hedge_ratio=1, 
                hedge_spread=hedge_spread, qty=1, 
                target_delta=target_delta,
                target_expiration=selected_expiration
            )
            
            if spread:
                sell_leg = spread.get('sell_option')
                buy_leg = spread.get('buy_option')
                sell_strike = sell_leg.strike_price if sell_leg else "N/A"
                
                spot_price = None
                if 'accounts' in globals() and accounts is not None:
                    try:
                        spot_price = accounts.get_stock_price(ticker)
                    except: pass
                
                otm_pct = None
                if spot_price and sell_strike != "N/A":
                    otm_pct = abs(float(sell_strike) - float(spot_price)) / float(spot_price) * 100.0

                res = {
                    "ticker": ticker,
                    "sell_strike": sell_strike,
                    "buy_strike": buy_leg.strike_price if buy_leg else "None",
                    "premium": spread.get('profit', 0),
                    "expiration": sell_leg.expiration_date.strftime("%Y-%m-%d") if sell_leg and hasattr(sell_leg, 'expiration_date') else "N/A",
                    "delta": sell_leg.delta if sell_leg and hasattr(sell_leg, 'delta') else target_delta,
                    "otm_pct": round(otm_pct, 2) if otm_pct is not None else None,
                    "side": side.upper(),
                    "is_mock": False
                }
            else:
                self._send_safe_response(
                    503,
                    {
                        "error": f"Live {ticker} {side.upper()} spread preview unavailable. Check E*TRADE authentication and option-chain access.",
                        "ticker": ticker,
                        "side": side.upper(),
                        "expiration": preview_expiration,
                        "is_mock": False
                    }
                )
                return
            self._send_safe_response(200, res)

        elif self.path.startswith('/api/positions_version'):
            positions_version = None
            updated_at = None
            if os.path.exists("screened_option_pairs.html"):
                stat_result = os.stat("screened_option_pairs.html")
                positions_version = str(stat_result.st_mtime_ns)
                updated_at = datetime.fromtimestamp(stat_result.st_mtime).isoformat()
            self._send_safe_response(200, {
                "positions_version": positions_version,
                "updated_at": updated_at,
                "refresh": _portfolio_refresh_snapshot(),
            })

        elif self.path.startswith('/api/positions'):
            if os.path.exists("screened_option_pairs.html"):
                try:
                    with open(
                        "screened_option_pairs.html",
                        "r",
                        encoding="utf-8",
                    ) as f:
                        content = f.read(MAX_POSITIONS_ARTIFACT_BYTES + 1)
                except (OSError, UnicodeError):
                    content = None
                content, valid = _validated_positions_artifact(content)
                self._send_safe_response(
                    200 if valid else 503,
                    content,
                    'text/html',
                    headers={
                        "X-Frame-Options": "SAMEORIGIN",
                        "Content-Security-Policy": POSITIONS_FRAME_CSP,
                    },
                )
            else:
                self._send_safe_response(
                    503,
                    _read_only_positions_fallback(),
                    'text/html',
                    headers={
                        "X-Frame-Options": "SAMEORIGIN",
                        "Content-Security-Policy": POSITIONS_FRAME_CSP,
                    },
                )

        elif self.path.startswith('/api/high_gain_spreads'):
            clean_proposals = []
            for p in CURRENT_CLOSE_PROPOSALS:
                cp = p.copy()
                if 'close_order' in cp: del cp['close_order']
                cp['has_open_order'] = cp.get('proposal_id') in ACTIVE_DASHBOARD_ORDERS
                clean_proposals.append(cp)
            self._send_safe_response(200, clean_proposals)
        elif self.path.startswith('/api/neutralize_risk'):
            clean_proposals = []
            for p in CURRENT_NEUTRALIZE_PROPOSALS:
                cp = p.copy()
                cp.pop('orders', None)
                cp['has_open_order'] = cp.get('proposal_id') in ACTIVE_DASHBOARD_ORDERS
                clean_proposals.append(cp)
            self._send_safe_response(200, clean_proposals)
        elif self.path.startswith('/api/gex'):
            if 'accounts' not in globals() or accounts is None:
                self._send_safe_response(503, {"error": "Engine loading..."})
                return
            try:
                gex_results = calculate_spy_gex(accounts)
                gex_results["regime"] = calculate_spy_regime_status(globals().get("best_hmm"))
                self._send_safe_response(200, gex_results)
            except Exception as e:
                print(f"⚠️ Error calculating SPY GEX: {e}")
                traceback.print_exc()
                self._send_safe_response(500, {"error": str(e)})
        else:
            self._send_safe_response(404, {"error": "Not found"})

def start_ngrok_tunnel(config, port):
    """Start an ngrok tunnel using credentials from config.ini if available."""
    try:
        if not config.has_section('NGROK'):
            return

        token = config.get('NGROK', 'AUTH_TOKEN', fallback=None)
        domain = config.get('NGROK', 'STATIC_DOMAIN', fallback=None)

        if not token or not domain:
            print("ℹ️ Ngrok auto-start skipped: AUTH_TOKEN or STATIC_DOMAIN missing in config.ini")
            return

        print(f"🌐 Starting ngrok tunnel for domain: {domain}...")
        
        # Configure token (run once, shouldn't hurt to run again)
        subprocess.run(["ngrok", "config", "add-authtoken", token.strip()], capture_output=True)
        
        # Start the tunnel in the background
        # Use Popen so it doesn't block
        cmd = [
            "ngrok", "http", 
            f"--url=https://{domain.strip()}", 
            str(port),
            "--log=stdout"
        ]
        
        # Redirect output to a log file for debugging
        with open("ngrok.log", "a") as log_file:
            subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        
        # Give it a moment to stabilize
        t.sleep(2)
        print(f"✅ Ngrok tunnel active! Public URL: https://{domain}")
        
    except FileNotFoundError:
        print("⚠️ Warning: ngrok command not found. Please install it using 'brew install ngrok/ngrok/ngrok'")
    except Exception as e:
        print(f"⚠️ Error starting ngrok tunnel: {e}")

def start_refresh_server(port=8765, host="127.0.0.1"):
    """Start the order-capable dashboard on loopback only."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise RuntimeSafetyError("dashboard host must be loopback")
    server = ThreadingHTTPServer((host, port), RefreshHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"🌐 [Refresh Server] Started on http://localhost:{port}")
    
    return server


def _get_local_ip():
    """Get the local network IP address for clickable links from other devices."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return 'localhost'




# import backtrader as bt

# from backtesting.test import SMA
# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# logger settings
logger = configure_owner_only_logger('my_logger')
'''
    Grab the option expire dates and option chains for the specified symbol.
    Save as a JSON file

'''
OAUTH_KEYS = {
    "sandbox": {
        "consumer_key": config['DEFAULT'].get('SANDBOX_CONSUMER_KEY', os.getenv("ETRADE_SANDBOX_CONSUMER_KEY")),
        "consumer_secret": config['DEFAULT'].get('SANDBOX_CONSUMER_SECRET', os.getenv("ETRADE_SANDBOX_CONSUMER_SECRET")),
    },
    "live": {
        "consumer_key": config['DEFAULT'].get('PROD_CONSUMER_KEY', os.getenv("ETRADE_LIVE_CONSUMER_KEY")),
        "consumer_secret": config['DEFAULT'].get('PROD_CONSUMER_SECRET', os.getenv("ETRADE_LIVE_CONSUMER_SECRET")),
    }
}

# File to cache OAuth tokens so you don't have to re-authenticate each time
ETRADE_OAUTH_FILE = ".etrade_oauth"


def configured_oauth_keys(use_sandbox):
    """Return non-placeholder client credentials or fail before broker I/O."""

    prefix = "SANDBOX" if use_sandbox else "LIVE"
    config_key = "SANDBOX" if use_sandbox else "PROD"
    consumer_key = os.getenv(f"ETRADE_{prefix}_CONSUMER_KEY") or config["DEFAULT"].get(f"{config_key}_CONSUMER_KEY")
    consumer_secret = os.getenv(f"ETRADE_{prefix}_CONSUMER_SECRET") or config["DEFAULT"].get(f"{config_key}_CONSUMER_SECRET")
    placeholders = {"", "default_live_key", "default_live_secret", "default_sandbox_key", "default_sandbox_secret"}
    if (
        not isinstance(consumer_key, str)
        or not isinstance(consumer_secret, str)
        or consumer_key.strip() in placeholders
        or consumer_secret.strip() in placeholders
    ):
        raise RuntimeSafetyError("E*TRADE client credentials are not configured")
    return {"consumer_key": consumer_key.strip(), "consumer_secret": consumer_secret.strip()}

def is_trading_day(check_date):
    """
    Check if the given date is a trading day for the NYSE.

    Args:
        check_date (date): The date to check.

    Returns:
        bool: True if it is a trading day, False otherwise.
    """
    try:
        nyse = get_calendar('NYSE')
        # Check if the date is a valid trading day
        schedule = nyse.valid_days(start_date=check_date, end_date=check_date)
        return len(schedule) > 0
    except Exception as e:
        logging.error(f"Error checking trading day for {check_date}: {e}")
        # Fallback to basic weekday check if library fails
        return check_date.weekday() < 5

def is_market_open(check_datetime=None):
    """
    Check the current NYSE market status using pandas_market_calendars.

    Args:
        check_datetime (datetime, optional): The datetime to check. Defaults to now (US/Eastern).

    Returns:
        tuple: (is_open: bool, status: str, market_open: datetime|None, market_close: datetime|None)
            status is one of: "OPEN", "PRE_MARKET", "AFTER_HOURS", "CLOSED_HOLIDAY"
            market_open/market_close are timezone-aware datetimes (or None on holidays).
    """
    import pytz
    eastern = pytz.timezone('US/Eastern')

    if check_datetime is None:
        check_datetime = datetime.now(eastern)
    elif check_datetime.tzinfo is None:
        # Assume local time, convert to Eastern
        check_datetime = eastern.localize(check_datetime)

    check_date = check_datetime.date()

    try:
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=check_date, end_date=check_date)

        if schedule.empty:
            return False, "CLOSED_HOLIDAY", None, None

        market_open = schedule.iloc[0]['market_open'].to_pydatetime()
        market_close = schedule.iloc[0]['market_close'].to_pydatetime()

        # Ensure check_datetime is tz-aware for comparison
        if check_datetime < market_open:
            return False, "PRE_MARKET", market_open, market_close
        elif check_datetime > market_close:
            return False, "AFTER_HOURS", market_open, market_close
        else:
            return True, "OPEN", market_open, market_close

    except Exception as e:
        logging.error(f"Error checking market status for {check_datetime}: {e}")
        # Fallback: use basic weekday + time heuristic
        if check_date.weekday() >= 5:
            return False, "CLOSED_HOLIDAY", None, None
        market_open_fallback = eastern.localize(datetime.combine(check_date, datetime.strptime('09:30:00', '%H:%M:%S').time()))
        market_close_fallback = eastern.localize(datetime.combine(check_date, datetime.strptime('16:00:00', '%H:%M:%S').time()))
        if check_datetime < market_open_fallback:
            return False, "PRE_MARKET", market_open_fallback, market_close_fallback
        elif check_datetime > market_close_fallback:
            return False, "AFTER_HOURS", market_open_fallback, market_close_fallback
        else:
            return True, "OPEN", market_open_fallback, market_close_fallback

def _market_close_refresh_due(check_datetime=None, market_status=None, market_close_time=None):
    """
    Return True once per trading day shortly after the regular NYSE close.
    This lets on-demand dashboard mode wake itself to freeze SPY/SPX/VIX closes.
    """
    global MARKET_CLOSE_PRICE_REFRESHED_DATE
    import pytz

    eastern = pytz.timezone('US/Eastern')
    if check_datetime is None:
        check_datetime = datetime.now(eastern)
    elif check_datetime.tzinfo is None:
        check_datetime = eastern.localize(check_datetime)
    else:
        check_datetime = check_datetime.astimezone(eastern)

    today_key = check_datetime.date().isoformat()
    if MARKET_CLOSE_PRICE_REFRESHED_DATE == today_key:
        return False

    if market_status is None or market_close_time is None:
        _, market_status, _, market_close_time = is_market_open(check_datetime)

    if market_status != "AFTER_HOURS" or market_close_time is None:
        return False
    return check_datetime >= market_close_time + timedelta(minutes=5)

def load_trade_status():
    """Load the trade status from the JSON file."""
    try:
        with open(TRADE_STATUS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_trade_date": None}

def save_trade_status(last_trade_date):
    """Save the trade status to the JSON file."""
    with open(TRADE_STATUS_FILE, 'w') as f:
        json.dump({"last_trade_date": last_trade_date}, f)

def record_market_close_prices(market_instance):
    """
    Fetch SPY, SPX and VIX prices and record them in the close-price cache.
    Yahoo Finance daily closes are preferred after market close; E*TRADE
    lastTrade is used as the fallback when Yahoo has not published today yet.
    """
    try:
        import pytz
        import json
        
        CACHE_FILE = "spy_vix_price_cache.json"
        eastern = pytz.timezone('US/Eastern')
        today_str = datetime.now(eastern).strftime("%Y-%m-%d")
        
        print(f"[Market Close] Recording SPY, SPX and VIX prices for {today_str}...")
        
        def _latest_yahoo_close(symbol):
            try:
                hist = yf.download(symbol, period="5d", progress=False, auto_adjust=False)
                if hist is None or hist.empty:
                    return None
                if hasattr(hist.columns, "levels") and len(hist.columns.levels) > 1:
                    try:
                        hist = hist.droplevel(1, axis=1)
                    except Exception:
                        pass
                close_col = "Close"
                if close_col not in hist.columns:
                    for col in hist.columns:
                        if (isinstance(col, tuple) and "Close" in col) or (isinstance(col, str) and "Close" in col):
                            close_col = col
                            break
                if close_col not in hist.columns:
                    return None
                for idx in reversed(hist.index):
                    dt_str = str(idx.date()) if hasattr(idx, "date") else str(idx).split(" ")[0]
                    if dt_str != today_str:
                        continue
                    value = hist.loc[idx, close_col]
                    if hasattr(value, "iloc"):
                        value = value.iloc[0]
                    if pd.isna(value):
                        return None
                    return round(float(value), 2)
            except Exception as yf_err:
                print(f"[Market Close] Yahoo close fetch failed for {symbol}: {yf_err}")
            return None

        prices = {}
        sources = {}

        yahoo_spy = _latest_yahoo_close("SPY")
        yahoo_spx = _latest_yahoo_close("^SPX")
        yahoo_vix = _latest_yahoo_close("^VIX")
        if yahoo_spy:
            prices["SPY"] = yahoo_spy
            sources["SPY"] = "Yahoo"
        if yahoo_spx:
            prices["SPX"] = yahoo_spx
            sources["SPX"] = "Yahoo"
        if yahoo_vix:
            prices["VIX"] = yahoo_vix
            sources["VIX"] = "Yahoo"

        # E*TRADE fallback: useful immediately after the bell before Yahoo daily
        # bars are available.
        if "SPY" not in prices or "SPX" not in prices or "VIX" not in prices:
            resp = market_instance.get_quote(['SPY', 'SPX', '^VIX'], resp_format="json")
            if not resp or 'QuoteResponse' not in resp or 'QuoteData' not in resp['QuoteResponse']:
                print("[Market Close] Warning: Could not fetch fallback quotes from E*TRADE.")
            else:
                quotes = resp['QuoteResponse']['QuoteData']
                for q in quotes:
                    sym = str(q.get('Product', {}).get('symbol') or '').upper()
                    last_price = q.get('All', {}).get('lastTrade')
                    if not last_price:
                        continue
                    if sym == "SPY" and "SPY" not in prices:
                        prices["SPY"] = round(float(last_price), 2)
                        sources["SPY"] = "E*TRADE"
                    elif sym in ("SPX", "SPXW") and "SPX" not in prices:
                        prices["SPX"] = round(float(last_price), 2)
                        sources["SPX"] = "E*TRADE"
                    elif sym in ("^VIX", "VIX") and "VIX" not in prices:
                        prices["VIX"] = round(float(last_price), 2)
                        sources["VIX"] = "E*TRADE"

        if "SPY" not in prices or "SPX" not in prices or "VIX" not in prices:
            print(f"[Market Close] Warning: Missing close data after fetch: {prices}")

        # Load and update cache
        cache = {"SPY": {}, "SPX": {}, "VIX": {}}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache = json.load(f)
            except Exception:
                pass
        
        updated = False
        if 'SPY' in prices:
            cache.setdefault("SPY", {})[today_str] = prices['SPY']
            updated = True
        if 'SPX' in prices:
            cache.setdefault("SPX", {})[today_str] = prices['SPX']
            updated = True
        if 'VIX' in prices:
            cache.setdefault("VIX", {})[today_str] = prices['VIX']
            updated = True
            
        if updated:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)
            print(f"[Market Close] Successfully recorded prices in {CACHE_FILE}: {prices} (sources={sources})")
        return updated
            
    except Exception as e:
        print(f"[Market Close] Error recording prices: {e}")
        return False

def get_previous_trading_day_close(ticker):
    """
    Retrieve the most recent close price from spy_vix_price_cache.json.
    Excludes today's date if it's already in the cache.
    """
    try:
        CACHE_FILE = "spy_vix_price_cache.json"
        if not os.path.exists(CACHE_FILE):
            return None
            
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            
        key = "SPY" if ticker.upper() == "SPY" else "VIX"
        data = cache.get(key, {})
        if not data:
            return None
            
        # Get all dates, sorted
        sorted_dates = sorted(data.keys())
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Filter out today (we want the *previous* close)
        past_dates = [d for d in sorted_dates if d < today_str]
        
        if not past_dates:
            return None
            
        prev_date = past_dates[-1]
        return data[prev_date]
    except Exception as e:
        print(f"[Prev Close] Error retrieving for {ticker}: {e}")
        return None

def release_margin(all_positions, cover_call_list=None, etrade_instance=None, max_positions=5):
    reject_legacy_execution("legacy automatic margin release")

def refresh_spread_limit_price(market_instance, ticker, sell_strike, buy_strike, call_put, expiration_date, order_dict, is_credit=True):
    """
    Re-fetch live quotes for a spread's legs and update order_dict['limitPrice']
    to the current midpoint.

    Args:
        market_instance: Market API instance.
        ticker: Underlying symbol (e.g. 'SPY').
        sell_strike: Strike price of the sold leg.
        buy_strike: Strike price of the bought leg.
        call_put: 'CALL' or 'PUT'.
        expiration_date: date object or 'YYYY-MM-DD' string.
        order_dict: The order dictionary (or list of dicts) whose 'limitPrice' will be updated.
        is_credit: True for credit spreads (opening), False for debit spreads (closing).

    Returns:
        (is_valid, new_price): is_valid=False if spread credit <= $0.01.
    """
    try:
        # Build expiration string component
        if hasattr(expiration_date, 'year'):
            exp_str = f"{expiration_date.year}:{expiration_date.month:02}:{expiration_date.day:02}"
        else:
            parts = str(expiration_date).split('-')
            exp_str = f"{parts[0]}:{parts[1]}:{parts[2]}"

        sell_osi = f"{ticker}:{exp_str}:{call_put}:{sell_strike}"
        buy_osi = f"{ticker}:{exp_str}:{call_put}:{buy_strike}"

        print(f"   🔄 [Price Refresh] Fetching latest quotes: {sell_osi}, {buy_osi}")
        resp = market_instance.get_quote([sell_osi, buy_osi], resp_format="json")
        quote_data = resp.get("QuoteResponse", {}).get("QuoteData", [])

        sell_q, buy_q = None, None
        for q in quote_data:
            prod = q.get("Product", {})
            strike = float(prod.get("strikePrice", 0))
            if abs(strike - float(sell_strike)) < 0.01:
                sell_q = q.get("All", {})
            elif abs(strike - float(buy_strike)) < 0.01:
                buy_q = q.get("All", {})

        if not sell_q or not buy_q:
            print("   ⚠️ [Price Refresh] Could not match quotes to legs. Using original price.")
            original = order_dict.get('limitPrice', 0) if isinstance(order_dict, dict) else 0
            return True, original

        sell_mid = (float(sell_q.get("bid", 0)) + float(sell_q.get("ask", 0))) / 2
        buy_mid = (float(buy_q.get("bid", 0)) + float(buy_q.get("ask", 0))) / 2

        if is_credit:
            new_price = _snap_option_limit_price(sell_mid - buy_mid)
        else:
            new_price = _snap_option_limit_price(abs(sell_mid - buy_mid))

        # Determine old price for logging
        if isinstance(order_dict, dict):
            old_price = order_dict.get('limitPrice', 0)
        elif isinstance(order_dict, list):
            old_price = order_dict[0].get('limitPrice', 0) if order_dict else 0
        else:
            old_price = 0

        label = "Credit" if is_credit else "Debit"
        print(f"   💰 [Price Refresh] {label}: ${old_price:.2f} → ${new_price:.2f}")

        if is_credit and new_price <= 0.01:
            print(f"   ❌ [Price Refresh] Spread no longer profitable (credit=${new_price:.2f}). Skipping.")
            return False, new_price

        # Update limitPrice in the order dict(s)
        if isinstance(order_dict, dict):
            order_dict['limitPrice'] = new_price
        elif isinstance(order_dict, list):
            for od in order_dict:
                if isinstance(od, dict) and 'limitPrice' in od:
                    od['limitPrice'] = new_price

        return True, new_price

    except Exception as e:
        print(f"   ⚠️ [Price Refresh] Error refreshing price: {e}")
        original = order_dict.get('limitPrice', 0) if isinstance(order_dict, dict) else 0
        return True, original

def build_manual_open_conflict_order(ticker, spread_candidate, quantity, request_id):
    """Build one spread order that uses BUY_CLOSE for an already-short bought leg."""
    buy_conflict_qty = int(spread_candidate.get("buy_conflict_qty") or 0)
    if buy_conflict_qty >= 0:
        return None

    sell_opt = spread_candidate.get("sell_option")
    buy_opt = spread_candidate.get("buy_option")
    if not sell_opt or not buy_opt:
        return None

    adjusted_qty = min(abs(buy_conflict_qty), int(quantity))
    if adjusted_qty <= 0:
        return None

    detail = {
        "expiration": sell_opt.expiration_date,
        "call_put": sell_opt.call_put,
        "short_strike": sell_opt.strike_price,
        "long_strike": buy_opt.strike_price,
    }
    order = _build_neutralize_spread_order(
        sell_opt.symbol or ticker,
        detail,
        adjusted_qty,
        "NET_CREDIT",
        spread_candidate.get("profit", 0),
        "SELL_OPEN",
        "BUY_CLOSE",
    )

    message = (
        f"Using BUY_CLOSE for existing short {ticker} {buy_opt.call_put} {buy_opt.strike_price} x{adjusted_qty} "
        f"inside {sell_opt.strike_price}/{buy_opt.strike_price} order"
    )
    update_manual_trade_status(request_id, status="adjusted", message=message)
    print(f"   🔁 [Manual Trade] {message}")

    order_type = f"{sell_opt.call_put.capitalize()} spread conflict adjustment"
    return {
        "ticker": ticker,
        "quote_ticker": sell_opt.symbol or ticker,
        "type": order_type,
        "order": order,
        "spread_data": spread_candidate,
        "quantity": adjusted_qty,
        "audit": {
            "ticker": ticker,
            "sell_strike": sell_opt.strike_price,
            "long_strike": buy_opt.strike_price,
            "qty": adjusted_qty,
            "is_close": False,
        },
    }

def check_manual_open_buy_conflict(accounts, ticker, buy_opt):
    conflict_qty = accounts.check_conflict_position(buy_opt, "buy", buy_opt.quantity)
    if conflict_qty != 0 or ticker != "SPX":
        return conflict_qty

    original_symbol = buy_opt.symbol
    try:
        for symbol in ("SPXW", "SPX"):
            if symbol == original_symbol:
                continue
            buy_opt.symbol = symbol
            conflict_qty = accounts.check_conflict_position(buy_opt, "buy", buy_opt.quantity)
            if conflict_qty != 0:
                return conflict_qty
    finally:
        buy_opt.symbol = original_symbol
    return 0

def _has_exact_manual_open_fields(data):
    return all(data.get(key) not in (None, "") for key in ("sell_strike", "buy_strike", "expiration"))

def _manual_open_target_row(option, action):
    return {
        "ticker": option.symbol,
        "call_put": option.call_put,
        "strike": option.strike_price,
        "price": option.last_price,
        "volume": getattr(option, "volume", ""),
        "spread": getattr(option, "ask_bid_spread", ""),
        "expire": option.expiration_date,
        "action": action,
    }

def submit_manual_open_fast_async(etrade_instance, accounts, market, data, request_id):
    reject_legacy_execution("legacy dashboard manual-open scheduler")

def _manual_open_fast_worker(etrade_instance, accounts, market, data, request_id):
    reject_legacy_execution("legacy dashboard manual-open worker")

def build_close_order_payload(ticker, expiration, call_put, short_strike, long_strike, quantity, limit_price, position_qty=None):
    ticker = (ticker or "").upper()
    call_put = (call_put or "").upper()
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    qty = abs(int(quantity or 1))
    limit_price = round(abs(float(limit_price)), 2)
    client_id = random.randint(1000000000, 9999999999)

    if long_strike:
        return {
            "client_order_id": client_id,
            "securityType": "OPTN",
            "orderTerm": "GOOD_FOR_DAY",
            "orderAction": "SPREAD",
            "spreadType": "VERTICAL",
            "orderType": "SPREADS",
            "priceType": "NET_DEBIT",
            "limitPrice": limit_price,
            "legs": [
                {
                    "symbol": ticker,
                    "orderAction": "SELL_CLOSE",
                    "quantity": qty,
                    "callPut": call_put,
                    "expiryYear": exp_date.year,
                    "expiryMonth": exp_date.month,
                    "expiryDay": exp_date.day,
                    "strikePrice": float(long_strike),
                },
                {
                    "symbol": ticker,
                    "orderAction": "BUY_CLOSE",
                    "quantity": qty,
                    "callPut": call_put,
                    "expiryYear": exp_date.year,
                    "expiryMonth": exp_date.month,
                    "expiryDay": exp_date.day,
                    "strikePrice": float(short_strike),
                },
            ],
            "required_margin": 0,
        }

    close_action = "SELL_CLOSE" if position_qty is not None and float(position_qty) > 0 else "BUY_CLOSE"
    return {
        "client_order_id": client_id,
        "symbol": ticker,
        "quantity": qty,
        "securityType": "OPTN",
        "orderType": "LIMIT",
        "priceType": "LIMIT",
        "orderTerm": "GOOD_FOR_DAY",
        "limitPrice": limit_price,
        "orderAction": close_action,
        "callPut": call_put,
        "expiryYear": exp_date.year,
        "expiryMonth": exp_date.month,
        "expiryDay": exp_date.day,
        "strikePrice": float(short_strike),
        "required_margin": 0,
    }

def submit_order_async(
    etrade_instance,
    accounts,
    market,
    order_info,
    reason,
    request_id=None,
    preview_only=False,
    max_checks=180,
    refresh_spec=None,
    record_target=False,
    record_spy_close_payload=None,
    active_proposal_id=None,
    trade_date_to_mark=None,
):
    reject_legacy_execution("legacy dashboard order scheduler")

def _order_worker(
    etrade_instance,
    accounts,
    market,
    order_info,
    reason,
    request_id,
    preview_only,
    max_checks,
    refresh_spec,
    record_target,
    record_spy_close_payload,
    active_proposal_id,
    trade_date_to_mark,
):
    reject_legacy_execution("legacy dashboard order worker")

def log_nudge_action(action_type, order_id, ticker, details):
    """Log nudge actions to a persistent JSON file for EOD reporting."""
    history_file = "nudge_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        date_str = datetime.now().date().isoformat()
        if date_str not in history:
            history[date_str] = []
        
        history[date_str].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "order_id": order_id,
            "ticker": ticker,
            "type": action_type,
            "details": details
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"   ⚠️ [Nudge Log] Error logging action: {e}")

def _option_tick_size(price):
    try:
        return 0.05 if abs(float(price)) < 3 else 0.10
    except Exception:
        return 0.10

def _snap_option_limit_price(price):
    try:
        value = abs(float(price))
        tick = _option_tick_size(value)
        snapped = int((value / tick) + 0.5) * tick
        return round(max(tick, snapped), 2)
    except Exception:
        return 0.0

def send_nudge_summary_email(target_date):
    """Read nudge_history.json and send an EOD summary email."""
    history_file = "nudge_history.json"
    if not os.path.exists(history_file):
        return

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except Exception:
        return

    actions = history.get(target_date, [])
    if not actions:
        print(f"   [Nudge Report] No nudge actions recorded for {target_date}.")
        return

    gmail_address = config.get('EMAIL', 'GMAIL_ADDRESS', fallback=None)
    gmail_app_password = config.get('EMAIL', 'GMAIL_APP_PASSWORD', fallback=None)
    notify_email = config.get('EMAIL', 'NOTIFY_EMAIL', fallback=gmail_address)

    if not gmail_address or not gmail_app_password or gmail_app_password == 'REPLACE_WITH_APP_PASSWORD':
        print("   ⚠️ [Nudge Report] Email not configured. Skipping report.")
        return

    rows_html = ""
    for a in actions:
        rows_html += f"""
        <tr>
            <td style="padding:8px;border:1px solid #ddd;">{a['time']}</td>
            <td style="padding:8px;border:1px solid #ddd;">{a['ticker']}</td>
            <td style="padding:8px;border:1px solid #ddd;">{a['order_id']}</td>
            <td style="padding:8px;border:1px solid #ddd;">{a['type']}</td>
            <td style="padding:8px;border:1px solid #ddd;">{a['details']}</td>
        </tr>"""

    html_body = f"""
    <html><body>
    <h2>🚀 Stale Order Nudge Summary - {target_date}</h2>
    <table style="border-collapse:collapse;width:100%;">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="padding:8px;border:1px solid #ddd;">Time</th>
          <th style="padding:8px;border:1px solid #ddd;">Ticker</th>
          <th style="padding:8px;border:1px solid #ddd;">Order ID</th>
          <th style="padding:8px;border:1px solid #ddd;">Action</th>
          <th style="padding:8px;border:1px solid #ddd;">Details</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </body></html>
    """

    msg = MIMEMultipart()
    msg['From'] = gmail_address
    msg['To'] = notify_email
    msg['Subject'] = f"🚀 Nudge Report: {target_date}"
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.send_message(msg)
        server.quit()
        print(f"   ✅ [Nudge Report] Email sent for {target_date}.")
    except Exception as e:
        print(f"   ⚠️ [Nudge Report] Email failed: {e}")

def monitor_and_nudge_stale_orders(etrade_instance, market_instance, dry_run=False):
    reject_legacy_execution("legacy stale-order nudge worker")

def extract_ticker_ask_bid(data):
    result = []
    try:
        quotes = data['QuoteResponse']['QuoteData']
        for quote in quotes:
            symbol = quote['Product']['symbol']
            ask = float(quote['All']['ask'])
            bid = float(quote['All']['bid'])
            result.append({'symbol': symbol, 'ask': ask, 'bid': bid})
    except KeyError as e:
        print(f"Key error: {e}")
    except (ValueError, TypeError) as e:
        print(f"Value or Type error: {e}")
    return result

def environment_key(use_sandbox) -> str:
    return "sandbox" if use_sandbox else "live"

def get_etrade_oauth(use_sandbox) -> dict:
    try:
        tokens = read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache")
        return tokens[environment_key(use_sandbox)]
    except (KeyError, TypeError, RuntimeSafetyError):
        print("Couldn't load cached OAuth safely.")
        return None

# Save the token, merging in with existing tokens
def save_etrade_oauth(token, use_sandbox) -> bool:
    try:
        if os.path.lexists(ETRADE_OAUTH_FILE):
            tokens = read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache")
        else:
            tokens = {}
        tokens[environment_key(use_sandbox)] = token
        write_owner_only_json(ETRADE_OAUTH_FILE, tokens)
        return True
    except (KeyError, TypeError, RuntimeSafetyError):
        print("Couldn't save cached OAuth safely.")
        return False

from urllib.parse import parse_qsl

def oauth(use_sandbox, auto_login=True, username=None, password=None, headless=None):
    """Allows user authorization for the sample application with OAuth 1"""
    if use_sandbox not in {True, False}:
        raise RuntimeSafetyError("E*TRADE environment must be explicit")
    keys = configured_oauth_keys(use_sandbox)
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]
    
    print(f"Environment: {'Sandbox' if use_sandbox else 'Live'}")
    print(f"Base URL: {'https://apisb.etrade.com' if use_sandbox else 'https://api.etrade.com'}")
    sys.stdout.flush()

    if use_sandbox:
        base_url = "https://apisb.etrade.com"
    else:
        base_url = "https://api.etrade.com"
    
    # Create session with standard User-Agent and legacy consumerkey header
    session_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "consumerkey": consumer_key
    }

    etrade = OAuth1Service(
        name="etrade",
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        request_token_url=f"{base_url}/oauth/request_token",
        access_token_url=f"{base_url}/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url=base_url
    )

    token_file = ETRADE_OAUTH_FILE

    tokens = (
        read_owner_only_json(token_file, label="OAuth cache")
        if os.path.lexists(token_file)
        else None
    )
    if isinstance(tokens, dict) and {"access_token", "access_token_secret"} <= set(tokens):
        
        session = etrade.get_session((tokens['access_token'], tokens['access_token_secret']))
        
        renew_url = f"{base_url}/oauth/renew_access_token"
        try:
            response = session.get(renew_url)
            if response.status_code == 200:
                # Accept any 200 response as successful renewal
                # Log the actual content for debugging purposes
                try:
                    content_preview = response.content.decode('utf-8', errors='replace')[:100]
                    print(f"Session renewal response: {content_preview}")
                except:
                    pass
                print("Session renewed successfully. Continuing with existing tokens.")
                return session, base_url
            else:
                print(f"Session renewal failed. Status code: {response.status_code}")
                # Log the short error message if possible to help debug rejection
                try:
                    error_msg = response.content.decode('utf-8', errors='replace')
                    if 'oauth_problem' in error_msg:
                        print(f"Server response: {error_msg[:100]}...")
                except:
                    pass
        except Exception as e:
            print(f"Error during renewal: {e}")
    else:
        print("No existing tokens found. Starting new OAuth flow.")

    # If we get here, either there were no existing tokens or renewal failed
    # Start a new OAuth flow
    # Fix: Provide a custom UTF-8 decoder to handle non-ASCII responses and HTML errors gracefully
    def utf8_decoder(content):
        # Decode as UTF-8, replacing invalid sequences with a placeholder
        # This prevents the 'ascii' codec or 'utf-8' failures on E*Trade's HTML error pages
        try:
            decoded_str = content.decode('utf-8', errors='replace')
            return dict(parse_qsl(decoded_str))
        except Exception as e:
            print(f"Decoder error: {e}")
            return {}

    request_token, request_token_secret = etrade.get_request_token(
        params={"oauth_callback": "oob", "format": "json"},
        headers=session_headers,
        decoder=utf8_decoder  # Override default decoder
    )

    authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)

    if auto_login:
        headless_mode = headless
        if headless_mode is None:
            # Fallback to args global if not provided
            try:
                headless_mode = not getattr(args, 'no_headless', False)
            except NameError:
                headless_mode = True
        
        # Check global username/password if not provided arguments
        if username is None:
            try:
                username = globals().get('username')
            except: pass
        if password is None:
            try:
                password = globals().get('password')
            except: pass

        text_code = get_token_automated(authorize_url, username=username, password=password, headless=headless_mode)
    else:
        # For autonomous operation, we strongly prefer auto_login=True.
        # If False, we log an error and try to continue if possible, 
        # but this will likely fail in a headless/non-interactive environment.
        print("CRITICAL: Manual login requested in autonomous mode. Attempting webbrowser.open...")
        webbrowser.open(authorize_url)
        # text_code = input("Please accept agreement and enter verification code from browser: ")
        # Fallback/Placeholder: In a real autonomous setup, this would be a failure point.
        raise RuntimeError("Manual OAuth verification code input is not supported in autonomous mode.")

    session = etrade.get_auth_session(
        request_token,
        request_token_secret,
        params={"oauth_verifier": text_code},
        decoder=utf8_decoder  # ensure access-token response is decoded as UTF-8
    )

    # Save the new tokens
    tokens = {
        'access_token': session.access_token,
        'access_token_secret': session.access_token_secret
    }
    write_owner_only_json(token_file, tokens)

    print("New session created and tokens saved.")
    return session, base_url


def _attach_etrade_auth_refresh_callbacks():
    callback = _refresh_etrade_session
    clients = [globals().get("accounts"), globals().get("market")]
    etrade = globals().get("etrade_instance")
    if etrade is not None:
        clients.extend([
            getattr(etrade, "account", None),
            getattr(etrade, "market", None),
            getattr(etrade, "order", None),
        ])
    for client in clients:
        if client is not None:
            setattr(client, "auth_refresh_callback", callback)


def _select_runtime_account(account_client):
    """Resolve the selected account through the immutable runtime boundary."""

    safety = globals().get("runtime_safety")
    if safety is None:
        # Importable helpers retain legacy sandbox behavior outside the main process.
        return account_client.account_list(1)
    account_client.account_list(**safety.account_selection_kwargs)
    safety.verify_account(account_client.account)
    return account_client.account


def _refresh_etrade_session(reason="E*TRADE request"):
    with ETRADE_SESSION_REFRESH_LOCK:
        use_sandbox_value = globals().get("use_sandbox", False)
        print(f"🔐 E*TRADE OAuth token expired during {reason}. Refreshing session...")
        new_session, new_base_url = oauth(use_sandbox_value, auto_login=True)

        globals()["session"] = new_session
        globals()["base_url"] = new_base_url
        globals()["last_renewal_time"] = datetime.now()

        if globals().get("accounts") is not None:
            refreshed_accounts = Accounts(
                new_session, new_base_url, use_sandbox=use_sandbox_value,
                consumer_key=globals().get("runtime_consumer_key"),
            )
            _select_runtime_account(refreshed_accounts)
            globals()["accounts"] = refreshed_accounts
        if globals().get("market") is not None:
            globals()["market"] = Market(
                new_session, new_base_url, use_sandbox=use_sandbox_value,
                consumer_key=globals().get("runtime_consumer_key"),
            )

        etrade = globals().get("etrade_instance")
        if etrade is not None:
            etrade.refresh_session(new_session, new_base_url)
            safety = globals().get("runtime_safety")
            if safety is None:
                raise RuntimeSafetyError("runtime account safety is unavailable during refresh")
            safety.verify_account(etrade.account.account)

        _attach_etrade_auth_refresh_callbacks()
        return new_session, new_base_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grab all the option chains for the specified symbol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--environment', choices=('sandbox', 'production'))
    parser.add_argument('--sandbox', help='legacy explicit sandbox mode', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--trade', help='start live trade?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_existing_file', help='re-use the backtest results?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-regime', help='skip heavy market regime detection?', action='store_true')
    parser.add_argument('--expected-account-id')
    parser.add_argument('--expected-account-id-key')
    parser.add_argument('--expected-institution-type')
    parser.add_argument('--production-arm-file', type=Path)
    parser.add_argument(
        '--runtime-config',
        required=True,
        type=Path,
        help='strict runtime configuration for state and artifact paths',
    )

    parser.add_argument('--username', help='username for login', type=str, required=False)
    parser.add_argument('--password', help='password for login', type=str, required=False)
    parser.add_argument('--no-headless', help='disable headless mode for login', action='store_true')
    args = parser.parse_args()

    if args.trade is not True:
        parser.error("--trade is required for this order-capable process")

    try:
        runtime_safety = build_runtime_safety_boundary(
            environment=args.environment,
            legacy_sandbox=args.sandbox,
            expected_account_id=args.expected_account_id,
            expected_account_id_key=args.expected_account_id_key,
            expected_institution_type=args.expected_institution_type,
            production_arm_file=args.production_arm_file,
            production_arm_secret=os.getenv("ETRADE_PRODUCTION_ARMING_SECRET"),
        )
    except RuntimeSafetyError as exc:
        parser.error(str(exc))

    try:
        runtime_config = load_runtime_config(args.runtime_config)
        validate_runtime_directories(runtime_config.paths)
        configured_account = runtime_config.selected_account
        configured_identity = (
            None
            if configured_account is None
            else (
                configured_account.account_id,
                configured_account.account_id_key,
                configured_account.institution_type,
            )
        )
        safety_identity = (
            runtime_safety.expected_account_id,
            runtime_safety.expected_account_id_key,
            runtime_safety.expected_institution_type,
        )
        if (
            runtime_config.broker_environment
            != runtime_safety.environment
            or configured_identity is None
            or configured_identity != safety_identity
        ):
            raise RuntimeConfigError(
                "runtime configuration does not match the exact "
                "broker environment and account safety boundary"
            )
        READ_ONLY_POSITIONS_PUBLISHER = (
            PositionsArtifactPublisher.from_runtime_config(
                runtime_config,
                signing_key=PositionsArtifactSigningKey.from_text(
                    resolve_positions_artifact_hmac_key()
                ),
            )
        )
        READ_ONLY_POSITIONS_BROKER_ENVIRONMENT = (
            runtime_config.broker_environment
        )
        AUTO_REFRESH_INTERVAL_SECONDS = min(
            AUTO_REFRESH_INTERVAL_SECONDS,
            max(
                1,
                runtime_config.data.max_snapshot_age_seconds // 2,
            ),
        )
        DASHBOARD_ON_DEMAND_REFRESH_ONLY = False
    except (OSError, RuntimeConfigError, RuntimeError, ValueError) as exc:
        parser.error(f"runtime configuration rejected: {exc}")

    # --- SINGLETON LOCK ---
    try:
        lock_file = secure_lock_file(Path.cwd() / "etrade_trader.lock")
    except RuntimeSafetyError as exc:
        parser.error(str(exc))
    use_sandbox = runtime_safety.use_sandbox
    runtime_consumer_key = configured_oauth_keys(use_sandbox)["consumer_key"]
    start_trade = args.trade
    use_existing_file = args.use_existing_file
    no_regime = args.no_regime

    # use_sandbox = True
    username = args.username
    password = args.password

    # Load from config.ini if not provided via CLI
    if not username or not password:
        try:
            if config.has_section('ETRADE'):
                if not username: username = config.get('ETRADE', 'USER', fallback=None)
                if not password: password = config.get('ETRADE', 'PASS', fallback=None)
        except Exception as e:
            print(f"⚠️ Warning: Could not load credentials from config.ini: {e}")
    
    # Fallback to .env if still missing
    if not username or not password:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if not username: username = os.getenv('ETRADE_USER')
            if not password: password = os.getenv('ETRADE_PASS')
        except ImportError:
            pass

    if not username or not password:
        print("❌ Error: E*TRADE username and password must be provided via --username/--password OR in config.ini")
        sys.exit(1)
    bypass_etrade = False
    start_log = False
    end_log = False

    try:
        live_settings = load_live_settings()
        validate_dashboard_credentials(live_settings)
        configured_oauth_keys(use_sandbox)
    except RuntimeSafetyError as exc:
        parser.error(str(exc))

    trade_executed_flag = False # flag to be reset every ticks, indicating the trade strategy had been exeucted for the current tick
    # Add these lines
    SKIP_CALL_FLAG = False  # Flag to skip call spread orders if needed
    SKIP_PUT_FLAG = False   # Flag to skip put spread orders if needed

    if not bypass_etrade: 
        try:
            session, base_url = oauth(use_sandbox, auto_login=True)
        except LoginFailureException as e:
            send_login_failure_notification(str(e), e.screenshot_path)
            sys.exit(1)
        except Exception as e:
            send_login_failure_notification(f"Unexpected error during initial login: {e}")
            sys.exit(1)
    else:
        session = base_url = None

    authenticated = 0
    
    # Load dynamic settings only after the safety boundary has accepted them.
    trade_start_time = datetime.strptime(live_settings.get('trade_start_time', '07:15:00'), '%H:%M:%S').time()
    trade_end_time = datetime.strptime(live_settings.get('trade_end_time', '13:30:00'), '%H:%M:%S').time()

    datalog_start_time = datetime.strptime('06:45:00', '%H:%M:%S').time()
    datalog_end_time = datetime.strptime('13:15:00', '%H:%M:%S').time()

    last_renewal_time = datetime.now()

    stock_positions: List[StockPosition] = []

    if start_trade:
        if not bypass_etrade: 
            selection = runtime_safety.account_selection_kwargs
            etrade_instance = LiveTradeAgent(
                None,
                session,
                base_url,
                selected_account=selection["selected_account_id"],
                use_sandbox=use_sandbox,
                expected_account_id_key=runtime_safety.expected_account_id_key,
                expected_account_id=runtime_safety.expected_account_id,
                expected_institution_type=runtime_safety.expected_institution_type,
                runtime_safety=runtime_safety,
                consumer_key=runtime_consumer_key,
            )
            runtime_safety.verify_account(etrade_instance.account.account)
        else:
            etrade_instance = LiveTradeAgent()
        print('Live trade agent id: ', etrade_instance.agent_id)

        accounts = Accounts(session, base_url, use_sandbox=use_sandbox, consumer_key=runtime_consumer_key)
        _select_runtime_account(accounts)
        market = Market(session, base_url, use_sandbox=use_sandbox, consumer_key=runtime_consumer_key)
        _attach_etrade_auth_refresh_callbacks()



        # breakpoint()
                    # ticker = "SPY"
                    # daily_gain_threshold = -0.02  # Focus on days with a 5%+ drop
                    # print("Downside Exposure:", downside_exposure)
        # vix_spread = accounts.get_protect_spread('VIX',quantity=5,days_to_expiration=30,target_delta=0.8, long_call_gap=1)
        # vix_spread_order_1 = accounts.generate_option_order(roll_stock_position=None,action="SPREAD",custom_order_id=None,spread_sell_option=vix_spread['sell_call_option'],spread_buy_option=vix_spread['buy_call_option_low'])
        # vix_spread_order_2 = accounts.generate_option_order(roll_stock_position=None,action="SPREAD",custom_order_id=None,spread_sell_option=None,spread_buy_option=vix_spread['buy_call_option_high'])
        # print(vix_spread_order_1)
        # print(vix_spread_order_2)
        # breakpoint()

        # breakpoint()
                    # positions_to_roll = accounts.portfolio(stock_positions)

        ##########manual option query##########
                # TSLA_CALL=accounts.manual_option_input("TSLA","CALL","2024-11-22",360)
                # TSLA_CALL=accounts.manual_option_input("TSLA","CALL","2025-02-21",350)
                # TSLA_CALL=accounts.manual_option_input("TSLA","CALL","2025-08-15",340)
                # print(TSLA_CALL)
                # TSLA_CALL.volatility=calculate_std_dev('TSLA',5)
                # print("14 day volatility: ", TSLA_CALL.volatility)
                # print(accounts.get_option_chain(TSLA_CALL,"CALL","IN",show_options=True))
                # breakpoint()

        # quick_order = {
        #     'callPut':"CALL"
        # }
        # breakpoint()        
    positions_to_roll=None
    now = None
    auto_closed_spread_ids = set()
    extrinsic_alert_sent = set()
    rejected_proposals_today = set()
    prev_date = None
    nudge_email_sent_for_date = None

    last_stale_check_time = datetime.now() - timedelta(minutes=6)

    # Load EV/Probability engine data lazily after the market-status gate.
    # Pre-market refreshes do not need a yfinance-backed regime sync.
    regime_dict, best_hmm, daily_models = {}, None, []
    regime_load_deferred_logged = False
    if no_regime:
        print("⏭️  Skipping market regime detection (--no-regime).")
    else:
        print("📈 Regime-based return data will load when the market is OPEN.")


    # Start the HTTP server for manual refresh requests from web UI
    try:
        refresh_server = start_refresh_server(port=8765)
        startup_refresh_generation = _queue_portfolio_refresh()
        REFRESH_REQUESTED.set()
        print(f"🔄 [Startup Refresh] Queued portfolio generation {startup_refresh_generation}.")

    except Exception as e:
        print(f"⚠️ Could not start refresh server: {e}")
        refresh_server = None

    retry_count = 0
    prev_market_status = None
    
    # Guard: Ensure --trade flag was passed, otherwise accounts/etrade_instance are undefined
    if not start_trade:
        print("❌ Error: You must run with --trade flag for the refresh functionality to work.")
        print("   See live_trading/README.md for explicit environment and production-arm usage.")
        sys.exit(1)
    
    while True:
        cycle_started_at = datetime.now()
        try:
            if DASHBOARD_ON_DEMAND_REFRESH_ONLY and not REFRESH_REQUESTED.is_set() and not MANUAL_TRADE_REQUESTED.is_set():
                _wait_for_dashboard_work()

            # Check for manual refresh request
            is_manual_refresh = REFRESH_REQUESTED.is_set()
            refresh_generation = None
            if is_manual_refresh:
                REFRESH_REQUESTED.clear()
                refresh_generation = _start_portfolio_refresh()

            # Refresh live settings on each tick
            live_settings = load_live_settings()
            trade_start_time = datetime.strptime(live_settings.get('trade_start_time', '07:15:00'), '%H:%M:%S').time()
            trade_end_time = datetime.strptime(live_settings.get('trade_end_time', '13:30:00'), '%H:%M:%S').time()

            # Get the current time and date
            now = datetime.now()
            current_time = now.time()
            current_date = now.date().isoformat()
            
            # Reset daily trackers
            if prev_date != current_date:
                rejected_proposals_today.clear()
                extrinsic_alert_sent.clear() # Optional: also reset alerts daily
                prev_date = current_date
    
            # Load trade status to check if trade was executed today
            trade_status = load_trade_status()
            last_trade_date = trade_status.get("last_trade_date")
    
            # Renew session if needed (every 60 minutes)
            if (now - last_renewal_time) >= timedelta(minutes=60):
                print("Renewing session...", now)
                try:
                    session, base_url = oauth(use_sandbox)
                    last_renewal_time = datetime.now()
                    accounts = Accounts(session, base_url, use_sandbox=use_sandbox, consumer_key=runtime_consumer_key)
                    market = Market(session, base_url, use_sandbox=use_sandbox, consumer_key=runtime_consumer_key)
                    _select_runtime_account(accounts)
                    if start_trade and not bypass_etrade:
                        etrade_instance.refresh_session(session, base_url)
                        runtime_safety.verify_account(etrade_instance.account.account)
                    _attach_etrade_auth_refresh_callbacks()
                except LoginFailureException as e:
                    logging.error(f"Failed to renew session (LoginFailure): {e}")
                    send_login_failure_notification(f"Session renewal failed: {e}", e.screenshot_path)
                except Exception as e:
                    logging.error(f"Failed to renew session: {e}")
                    send_login_failure_notification(f"Session renewal encountered an unexpected error: {e}")
                    # We'll continue and hope the next tick works or the main loop catch handles it
    
            # ── Market status gate (checked BEFORE any portfolio/order API calls) ──
            is_open, market_status, market_open_time, market_close_time = is_market_open()
            print(f"\n📊 Market status: {market_status}  (checked at {now.strftime('%Y-%m-%d %H:%M:%S')})")

            # Publish the authoritative raw-position view before slower model,
            # analytics, or candidate scans. A failed/incomplete broker read
            # leaves the prior atomic artifact untouched.
            _select_runtime_account(accounts)
            cycle_positions, positions_source_as_of = (
                _load_stable_positions(accounts)
            )
            _publish_confirmed_positions(
                cycle_positions,
                observed_at=positions_source_as_of,
            )

            # Capture close prices even in on-demand mode or when the process starts after close.
            if _market_close_refresh_due(market_status=market_status, market_close_time=market_close_time):
                if record_market_close_prices(etrade_instance.market):
                    MARKET_CLOSE_PRICE_REFRESHED_DATE = current_date
            prev_market_status = market_status

            if not no_regime and best_hmm is None:
                if market_status == "OPEN":
                    print("📈 Loading regime-based return data for EV engine...")
                    # build_regime_return_arrays performs yfinance sync; avoid it before regular hours.
                    regime_dict, best_hmm, daily_models = build_regime_return_arrays(int(t.time()/86400), horizon=7)
                elif market_status == "PRE_MARKET" and not regime_load_deferred_logged:
                    print("⏭️  Deferring yfinance-backed regime refresh until market is OPEN (currently PRE_MARKET).")
                    regime_load_deferred_logged = True

            # Log GEX by minute for intraday analysis (only during market hours)
            if is_open:
                log_gex_minute(accounts)

            next_manual_request = dequeue_manual_trade_request()
            if next_manual_request:
                MANUAL_TRADE_PARAMS = next_manual_request
                is_manual_trade = True
            else:
                MANUAL_TRADE_PARAMS = {}
                is_manual_trade = False

            # The dashboard Refresh Data button is a read-only portfolio refresh.
            # Render it before slower candidate/GEX/order-management scans so the
            # browser does not keep reloading yesterday's generated HTML.
            if is_manual_refresh and not is_manual_trade:
                print(f"🔄 [Manual Refresh] Regenerating portfolio HTML ({market_status})...")
                try:
                    _select_runtime_account(accounts)
                    refresh_positions = cycle_positions
                    from copy import deepcopy
                    refresh_screened = accounts.screen_option(deepcopy(refresh_positions))

                    try:
                        from accounts.accounts_bo import _spy_margin_totals_external
                        spy_total_margin, spy_call_margin, spy_put_margin = _spy_margin_totals_external(refresh_screened)
                        update_spy_daily_snapshot(
                            refresh_positions,
                            spy_total_margin,
                            spy_call_margin,
                            spy_put_margin,
                        )
                    except Exception as tracker_err:
                        print(f"[SPY Tracker] Warning: Could not update tracker during manual refresh: {tracker_err}")

                    html_path = accounts.render_screened_option_pairs_html(
                        refresh_screened,
                        out_path="screened_option_pairs.html",
                        order_instance=etrade_instance.order,
                        show_refresh=False,
                    )
                    _finish_portfolio_refresh(refresh_generation)
                    print(f"✅ [Manual Refresh] HTML updated: {html_path}")
                except Exception as refresh_err:
                    _finish_portfolio_refresh(refresh_generation, error=refresh_err)
                    print(f"⚠️ [Manual Refresh] Error updating HTML: {refresh_err}")
                    traceback.print_exc()
                continue
            
            if market_status == "CLOSED_HOLIDAY" and not is_manual_trade:
                print(f"📅 Today ({current_date}) is not a trading day (market holiday).")

                # If manual refresh was requested, fetch portfolio and update HTML
                if is_manual_refresh:
                    print("🔄 [Manual Refresh] Fetching portfolio and updating HTML (market closed)...")
                    try:
                        _select_runtime_account(accounts)
                        all_positions = accounts.portfolio(print_enable=False, require_success=True)
                        screened = accounts.screen_option(all_positions)
                        html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
                        CURRENT_NEUTRALIZE_PROPOSALS = build_neutralize_proposals(
                            screened,
                            accounts,
                            market,
                            live_settings,
                            rejected_proposals_today
                        )
                        print(f"✅ [Manual Refresh] HTML updated: {html_path}")
                    except Exception as refresh_err:
                        print(f"⚠️ [Manual Refresh] Error updating HTML: {refresh_err}")

                # Find next trading day
                next_day = now.date() + timedelta(days=1)
                while not is_trading_day(next_day):
                    next_day += timedelta(days=1)
                next_trade_time = datetime.combine(next_day, trade_start_time)
                seconds_until_next = (next_trade_time - now).total_seconds()
                if seconds_until_next > 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sleeping until next trading day: {next_trade_time}...")
                    # Use interruptible sleep to allow manual refresh (check every second, max 5 min)
                    for _ in range(_next_refresh_sleep_seconds(cycle_started_at, seconds_until_next)):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # ── Market is a trading day (PRE_MARKET, OPEN, or AFTER_HOURS) ──
            # Run read-only account/portfolio operations
            _select_runtime_account(accounts)
            today = datetime.now()
            passed_monday = today - timedelta(days=((today.weekday()) % 7))
            etrade_instance.order.option_gain_new(passed_monday.strftime("%Y-%m-%d"))
            # etrade_instance.order.option_gain_new('2026-01-01')
            accounts.balance()
    
            all_positions = cycle_positions

            # --- EXTRINSIC VALUE ALERTS FOR ITM SHORT OPTIONS ---
            # Skip this check outside regular market hours because option quotes can be stale.
            if is_open:
                try:
                    alerts_to_send = []
                    for pos in all_positions:
                        if getattr(pos, "security_type", "") == "Option" and getattr(pos, "quantity", 0) < 0:
                            ul = getattr(pos, "underlying_last_price", 0.0)
                            strike = getattr(pos, "strike_price", 0.0)
                            price = getattr(pos, "last_price", 0.0)
                            cp = getattr(pos, "call_put", None)
                            sym = getattr(pos, "symbol", "")
                            exp = getattr(pos, "expiration_date", "")
                            
                            if ul and strike and price and cp:
                                intrinsic = max(0.0, ul - strike) if cp == "CALL" else max(0.0, strike - ul)
                                if intrinsic > 0:
                                    extrinsic = max(0.0, price - intrinsic)
                                    if extrinsic <= 1.0:
                                        alert_id = f"{sym}_{strike}_{cp}_{exp}_{current_date}"
                                        if alert_id not in extrinsic_alert_sent:
                                            alerts_to_send.append(
                                                f"<strong>{sym} {cp} ${strike}</strong> (Exp: {exp})<br>"
                                                f"Underlying: ${ul:.2f} | Option Price: ${price:.2f}<br>"
                                                f"Intrinsic: <span style='color:#d93025;'>${intrinsic:.2f}</span> | "
                                                f"Extrinsic: <span style='color:#d93025;font-weight:bold;'>${extrinsic:.2f}</span>"
                                            )
                                            extrinsic_alert_sent.add(alert_id)
                    if alerts_to_send:
                        send_extrinsic_value_alert(alerts_to_send)
                except Exception as alert_e:
                    print(f"[Extrinsic Alert Check] Warning: {alert_e}")
    
            # Update SPY position tracker BEFORE screen_option (which modifies quantities in-place)
            try:
                # Refresh portfolio right before snapshot to ensure fresh pricing
                all_positions = accounts.portfolio(print_enable=False, require_success=True)
                
                # Need to call screen_option first just to get margin values, then restore quantities
                from copy import deepcopy
                positions_copy = deepcopy(all_positions)
                screened_temp = accounts.screen_option(positions_copy)
                from accounts.accounts_bo import _spy_margin_totals_external
                spy_total_margin, spy_call_margin, spy_put_margin = _spy_margin_totals_external(screened_temp)
                update_spy_daily_snapshot(all_positions, spy_total_margin, spy_call_margin, spy_put_margin)

            except Exception as e:
                print(f"[SPY Tracker] Warning: Could not update tracker: {e}")
    
            # Prepare for opening/managing positions
            positions_to_roll = accounts.get_option_trade(all_positions)
            executed_orders_list = etrade_instance.order.get_executed_orders(passed_monday.strftime("%Y-%m-%d"))
            accounts.update_csv_order_statuses(executed_orders_list)
            
            try:
                open_orders = etrade_instance.order.get_open_orders()
                if open_orders is not None:
                    ACTIVE_DASHBOARD_ORDERS.clear()
                    for o in open_orders:
                        if o.get("orderAction") in ("BUY_CLOSE", "SELL_CLOSE"):
                            sym = o.get("symbol")
                            strike = o.get("strikePrice")
                            cp_opt = o.get("callPut")
                            exp = o.get("expiryDate")
                            if sym and strike and cp_opt and exp:
                                # Add variants to match both datetime and date string representations
                                ACTIVE_DASHBOARD_ORDERS.add(f"{sym}_{float(strike)}_{cp_opt}_{exp}")
                                ACTIVE_DASHBOARD_ORDERS.add(f"{sym}_{float(strike)}_{cp_opt}_{exp} 00:00:00")
            except Exception as e:
                print(f"[Open Orders Sync] Warning: Could not fetch open orders: {e}")

            # If market is not open (PRE_MARKET or AFTER_HOURS), update HTML but skip position actions
            if market_status in ("PRE_MARKET", "AFTER_HOURS") and not is_manual_trade:
                if is_manual_refresh:
                    print(f"🔄 [Manual Refresh] Fetching portfolio and updating HTML ({market_status})...")
                else:
                    print(f"\n--- Updating screened_option_pairs.html ({market_status}) ---")
                all_positions = accounts.portfolio(print_enable=False, require_success=True)
                screened = accounts.screen_option(all_positions)
                html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
                CURRENT_NEUTRALIZE_PROPOSALS = build_neutralize_proposals(
                    screened,
                    accounts,
                    market,
                    live_settings,
                    rejected_proposals_today
                )
                print(f"HTML updated: {html_path}")

                if market_status == "PRE_MARKET" and market_open_time is not None:
                    # Wake at market open so read-only scans begin immediately.
                    import pytz
                    eastern = pytz.timezone('US/Eastern')
                    seconds_until_open = (market_open_time - datetime.now(eastern)).total_seconds()
                elif market_status == "AFTER_HOURS":
                    # Sleep until trade_start_time on the next trading day
                    next_day = now.date() + timedelta(days=1)
                    while not is_trading_day(next_day):
                        next_day += timedelta(days=1)
                    next_trade_time = datetime.combine(next_day, trade_start_time)
                    seconds_until_open = (next_trade_time - now).total_seconds()
                else:
                    seconds_until_open = 600

                if seconds_until_open > 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Market {market_status}. Sleeping {int(seconds_until_open)}s...")
                    sleep_seconds = _next_refresh_sleep_seconds(cycle_started_at, seconds_until_open)
                    for _ in range(sleep_seconds):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # ── Trading window check (market is OPEN, but are we within user's trading hours?) ──
            if not (trade_start_time <= current_time <= trade_end_time) and not is_manual_trade:
                print(f"\n⏰ Market is OPEN but outside trading window ({trade_start_time}–{trade_end_time}). Running read-only scans; skipping position actions.")
                print("--- Updating positions and close candidates (outside trading window) ---")
                all_positions = accounts.portfolio(print_enable=False, require_success=True)
                screened = accounts.screen_option(all_positions)
                html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
                accounts.option_value_final(all_positions)
                CURRENT_CLOSE_PROPOSALS = build_close_proposals(
                    screened,
                    accounts,
                    market,
                    live_settings,
                    rejected_proposals_today
                )
                CURRENT_NEUTRALIZE_PROPOSALS = build_neutralize_proposals(
                    screened,
                    accounts,
                    market,
                    live_settings,
                    rejected_proposals_today
                )
                print(f"HTML updated: {html_path}")

                # Calculate seconds until trading window starts
                next_trade_time = datetime.combine(now.date(), trade_start_time)
                if current_time > trade_end_time:
                    # Past trading window for today, sleep until next trading day
                    
                    # 🚀 Send EOD Nudge Summary Log
                    if nudge_email_sent_for_date != current_date:
                        print(f"--- Sending EOD Nudge Summary for {current_date} ---")
                        send_nudge_summary_email(current_date)
                        nudge_email_sent_for_date = current_date

                    next_day = now.date() + timedelta(days=1)
                    while not is_trading_day(next_day):
                        next_day += timedelta(days=1)
                    next_trade_time = datetime.combine(next_day, trade_start_time)
                seconds_until_trade = (next_trade_time - now).total_seconds()
                if seconds_until_trade > 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Read-only scan complete. Next refresh in at most 5 minutes.")
                    sleep_seconds = _next_refresh_sleep_seconds(cycle_started_at, seconds_until_trade)
                    for _ in range(sleep_seconds):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # Expiring/ITM spreads remain visible in the analytical screens.
            # The legacy process no longer submits automatic closing orders.

            # Source of truth for execution: dashboard manual trade or auto-open enabled
            is_manual_close = is_manual_trade and MANUAL_TRADE_PARAMS.get('is_close', False)
            is_manual_neutralize = is_manual_trade and MANUAL_TRADE_PARAMS.get('is_neutralize', False)
            is_manual_open = is_manual_trade and not is_manual_close and not is_manual_neutralize

            # --- POSITION OPENING ACTION SECTION ---
            # Attempt to open new positions if auto-open is enabled and we haven't traded yet today
            # OR if manual refresh/manual open is requested. Manual close/neutralize requests are
            # isolated from normal open automation so a risk action cannot create unrelated spreads.
            auto_open_enabled = False
            if is_manual_trade:
                print("\n⚡ [Main Loop] Manual trade request detected. Proceeding to execution logic...")
                _sync_manual_trade_event()
            
            # --- OPENING EXECUTION GATE ---
            # Only search for new positions if it's an AUTO-OPEN or a MANUAL-OPEN request
            execute_open_now = (
                (auto_open_enabled and last_trade_date != current_date and not is_manual_close and not is_manual_neutralize)
                or is_manual_open
            )
            
            if execute_open_now or is_manual_refresh:
                if is_manual_refresh:
                     print(f"🔄 [Manual Refresh] Re-evaluating candidates for dashboard...")
                if is_manual_open:
                     print("\n⚡ [Main Loop] Manual OPEN request detected. Proceeding to execution logic...")
                     _sync_manual_trade_event()
                cover_call_list = {}
                # 'AAPL': 5,
                # 'AMD': 1,
                # 'BRK.B': 2,
                # 'JPM': 1,
                # 'NVDA': 1,
                # 'QCOM': 1,

        
                # Execute trades automatically at trading start based on dashboard settings
                expiration_weeks = live_settings.get('target_weeks', 6)
                trade_side = live_settings.get('trade_side', 'BOTH')
                
                # Source of truth is now the dashboard settings
                parameters = [
                    {
                        "ticker": "SPY",
                        "pnl": 0,
                        "target_premium_call": 0,
                        "target_premium_put": 0,
                        "hedge_ratio_call": 1 if trade_side in ['CALL', 'BOTH'] else 0,
                        "hedge_ratio_put": 1 if trade_side in ['PUT', 'BOTH'] else 0,
                        "qty": live_settings.get('spy_pair_quantity', live_settings.get('pair_quantity', 15))
                    },
                    {
                        "ticker": "SPX",
                        "pnl": 0,
                        "target_premium_call": 0,
                        "target_premium_put": 0,
                        "hedge_ratio_call": 1 if trade_side in ['CALL', 'BOTH'] else 0,
                        "hedge_ratio_put": 1 if trade_side in ['PUT', 'BOTH'] else 0,
                        "qty": live_settings.get('spx_pair_quantity', 1)
                    }
                ]

                if is_manual_open:
                    requested_ticker = (MANUAL_TRADE_PARAMS.get('ticker') or "").upper()
                    if requested_ticker not in ("SPY", "SPX"):
                        update_manual_trade_status(
                            MANUAL_TRADE_PARAMS.get("request_id"),
                            status="failed",
                            message=f"Unsupported manual open ticker: {requested_ticker or 'missing'}"
                        )
                        parameters = []
                    else:
                        parameters = [p for p in parameters if p["ticker"] == requested_ticker]
                        print(f"🎯 [Manual Trade] Restricting open search to {requested_ticker}")
                
                # Calculate days to next Friday
                today = datetime.now()
                friday = today + timedelta(days=((4 - today.weekday()) % 7) + 7 * (expiration_weeks-1))
                days_to_expire = (friday - today).days            
        
                PREVIEW_ONLY = True
                preview_orders = []
        
                for parameter in parameters:
                    spread_options_call_candidate, spread_options_put_candidate = None, None
                    call_spread_orders, put_spread_orders = None, None
        
                    ticker = parameter['ticker']
                    hedge_ratio_call = parameter['hedge_ratio_call']
                    hedge_ratio_put = parameter['hedge_ratio_put']
        
                    if ticker not in ['SPY', 'GLD', 'DIA']:
                        earnings_date = get_earnings_dates(ticker, today.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d"))
                        print(f"Earnings date for {ticker}: {earnings_date}")
                        earning_flag = False
                        if earnings_date is not None:
                            for ed in earnings_date:
                                if today.date() <= ed < friday.date():
                                    print(f"Earnings date for {ticker} is within the trading period. Skipping...")
                                    earning_flag = True
                            if earning_flag:
                                print(f"Earnings date for {ticker} is within the trading period. Skipping...")
                                continue
        
                    current_price = accounts.get_stock_price(ticker)
                    print(f"Current price for {ticker}: {current_price}")
        
                    target_premium_otm_call = 0.3
                    target_premium_otm_put = 0.3
                    target_steer = 0.9
                    today_date = today.date()
                    friday_date = _target_expiration_from_settings(accounts, ticker, live_settings, today_date)
                    days_to_expire = (friday_date - today_date).days
        
                    print(f"Target Friday: {friday_date}, Days from today: {days_to_expire}")
        
                    vix_value = accounts.get_stock_price('VIX')
                    print(f"VIX value: {vix_value}")
        
                    vix_threshold = 20
                    vix_correlation = 0.05
                    trade_type = 'put_credit_spread'
                    adjust = vix_correlation
                    if vix_value > vix_threshold * 1.5:
                        adjust = vix_correlation * 4
        
                    target_premium_call_baseline = target_premium_otm_call * (days_to_expire)**0.5 * (1-target_steer)
                    target_premium_put_baseline = target_premium_otm_put * (days_to_expire)**0.5 * (1+target_steer)
                    # Skip VIX-based premium adjustment when using delta target (-0.15)
                    target_premium_call = target_premium_call_baseline if trade_type != 'put_credit_spread' else 0
                    target_premium_put = target_premium_put_baseline if trade_type != 'call_credit_spread' else 0

                    if is_manual_trade and MANUAL_TRADE_PARAMS:
                        requested_side = str(MANUAL_TRADE_PARAMS.get('side', 'BOTH')).upper()
                        if requested_side == 'CALL':
                            hedge_ratio_put = 0
                        elif requested_side == 'PUT':
                            hedge_ratio_call = 0
                        
                        # Use provided quantity if available
                        if 'qty' in MANUAL_TRADE_PARAMS:
                            parameter['qty'] = int(MANUAL_TRADE_PARAMS['qty'])
        
                    target_delta_val = live_settings.get('target_delta', 0.15)
                    if ticker == 'SPX':
                        hedge_spread_val = float(live_settings.get('spx_hedge_spread', live_settings.get('hedge_spread', 200.0)) or 200.0)
                    else:
                        hedge_spread_val = float(live_settings.get('spy_hedge_spread', live_settings.get('hedge_spread', 20.0)) or 20.0)
                    pair_qty = parameter['qty']
                    print(f"🎯 Searching for {ticker} spread candidates by Target Delta: {target_delta_val} (Width: ${hedge_spread_val}, Qty: {pair_qty})")
                    # Log target premiums only as secondary reference
                    # print(f"Target premium call: {target_premium_call}, Target premium put: {target_premium_put}")
        
                    if is_manual_trade and MANUAL_TRADE_PARAMS and 'sell_strike' in MANUAL_TRADE_PARAMS:
                        sell_k = float(MANUAL_TRADE_PARAMS.get('sell_strike', 0))
                        buy_k = float(MANUAL_TRADE_PARAMS.get('buy_strike', 0))
                        target_exp = MANUAL_TRADE_PARAMS.get('expiration', '')
                        
                        print(f"🎯 [Manual Trade] Using specific strikes from dashboard: {sell_k}/{buy_k} Exp: {target_exp}")
                        
                        side = MANUAL_TRADE_PARAMS.get('side', 'PUT').capitalize()
                        
                        try:
                            # manual_option_input returns a StockPosition with last_price populated from E*TRADE
                            sell_opt = accounts.manual_option_input(ticker, side.upper(), target_exp, sell_k)
                            buy_opt = accounts.manual_option_input(ticker, side.upper(), target_exp, buy_k) if buy_k else None
                            
                            if sell_opt is not None and buy_opt is not None:
                                # Populate missing fields for generate_option_order
                                target_exp_dt = datetime.strptime(target_exp, "%Y-%m-%d").date()
                                
                                sell_opt.strike_price = sell_k
                                sell_opt.expiration_date = target_exp_dt
                                sell_opt.call_put = side.upper()
                                sell_opt.quantity = parameter['qty']
                                sell_opt.distance_to_strike = (sell_k - current_price) / current_price * 100 if current_price else 0
                                
                                buy_opt.strike_price = buy_k
                                buy_opt.expiration_date = target_exp_dt
                                buy_opt.call_put = side.upper()
                                buy_opt.quantity = parameter['qty']
                                buy_opt.distance_to_strike = (buy_k - current_price) / current_price * 100 if current_price else 0
                                buy_conflict_qty = check_manual_open_buy_conflict(accounts, ticker, buy_opt)
                                
                                profit = MANUAL_TRADE_PARAMS.get('premium')
                                if profit is not None:
                                    profit = float(profit)
                                    print(f"   [Manual Trade] Using dashboard premium: ${profit:.2f}")
                                else:
                                    profit = round(sell_opt.last_price - buy_opt.last_price, 2)
                                
                                candidate = {
                                    'sell_option': sell_opt,
                                    'buy_option': buy_opt,
                                    'profit': profit,
                                    'ticker': ticker,
                                    'buy_conflict_qty': buy_conflict_qty,
                                }
                                print(f"   ✅ Manual spread candidate created: {sell_k}/{buy_k} at net credit ${profit}")
                                if buy_conflict_qty < 0:
                                    update_manual_trade_status(
                                        MANUAL_TRADE_PARAMS.get("request_id"),
                                        message=f"Detected existing short {ticker} {side.upper()} {buy_k} qty {buy_conflict_qty}; using BUY_CLOSE on long leg"
                                    )
                                
                                # Proceed with execution logic...
                                print(f"   🚀 Executing manual trade for {ticker}...")
                                try:
                                    # Logic to place order...
                                    # (This usually happens later in the loop, but for manual trade we can force it)
                                    # Actually, let's just add it to the candidates list and let the normal logic handle it
                                    # but we need to make sure it's selected.
                                    pass
                                except Exception as e:
                                    print(f"   ❌ Error in manual execution: {e}")
                                if side == 'Call':
                                    spread_options_call_candidate = candidate
                                else:
                                    spread_options_put_candidate = candidate
                                    
                                print(f"✅ [Manual Trade] Successfully loaded specific spread. Net Credit: ${profit:.2f}")
                                
                            else:
                                print(f"⚠️ [Manual Trade] Could not retrieve quotes for both legs. Aborting manual execution.")
                                update_manual_trade_status(
                                    MANUAL_TRADE_PARAMS.get("request_id"),
                                    status="failed",
                                    message=f"Could not retrieve quotes for {ticker} {sell_k}/{buy_k}"
                                )
                                continue
                        except Exception as e:
                            print(f"⚠️ [Manual Trade] Error loading specific strikes: {e}. Aborting submitted dashboard order.")
                            update_manual_trade_status(
                                MANUAL_TRADE_PARAMS.get("request_id"),
                                status="failed",
                                message=f"Error loading requested {ticker} {side.upper()} {sell_k}/{buy_k} exp {target_exp}: {e}"
                            )
                            traceback.print_exc()
                            continue

                    while spread_options_call_candidate is None and spread_options_put_candidate is None:
                        # Use target_delta (sign-aware in accounts_bo.py) to determine the short leg
                        spread_options_call_candidate = accounts.get_option_spread_by_price(ticker, "Call", days_to_expire=days_to_expire, target_premium=target_premium_call, hedge_ratio=hedge_ratio_call, hedge_spread=hedge_spread_val, qty=pair_qty, target_delta=target_delta_val)
                        spread_options_put_candidate = accounts.get_option_spread_by_price(ticker, "Put", days_to_expire=days_to_expire, target_premium=target_premium_put, hedge_ratio=hedge_ratio_put, hedge_spread=hedge_spread_val, qty=pair_qty, target_delta=target_delta_val)
                        
                        if spread_options_call_candidate is None and spread_options_put_candidate is None:
                            days_to_expire -= 7
                            if days_to_expire < 2:
                                break
                            continue
                        actual_expiration_date = spread_options_call_candidate['sell_option'].expiration_date if spread_options_call_candidate is not None else spread_options_put_candidate['sell_option'].expiration_date
                        if actual_expiration_date is None:
                            print(f"Expiration date is None for {ticker}, {spread_options_put_candidate}, {spread_options_call_candidate}. Retrying...")
                            # breakpoint()
                            break
                        actual_days_to_expire = (actual_expiration_date - today_date).days
                        if actual_days_to_expire != days_to_expire:
                            spread_options_call_candidate, spread_options_put_candidate = None, None
                            print(f"Actual days to expire {actual_days_to_expire} does not match target {days_to_expire}. Retrying...")
                            days_to_expire = actual_days_to_expire
                            target_premium_call_baseline = target_premium_otm_call * (days_to_expire)**0.5 * (1-target_steer)
                            target_premium_put_baseline = target_premium_otm_put * (days_to_expire)**0.5 * (1+target_steer)
                            # Skip VIX-based premium adjustment when using delta target (-0.15)
                            target_premium_call = target_premium_call_baseline if trade_type != 'put_credit_spread' else 0
                            target_premium_put = target_premium_put_baseline if trade_type != 'call_credit_spread' else 0
                            print(f"Target premium call: {target_premium_call}, Target premium put: {target_premium_put}")
        
                    single_leg_stock_position_call = None
                    single_leg_stock_position_put = None

                    if hedge_ratio_call == 1:
                        call_action = "SPREAD"
                    else:
                        call_action = "SPREAD"

                    if spread_options_put_candidate is not None and spread_options_put_candidate['profit'] > 0.01:
                        put_action = "SPREAD"
                    else:
                        put_action = "SPREAD"
        
                    call_has_manual_conflict = (
                        is_manual_open
                        and spread_options_call_candidate
                        and int(spread_options_call_candidate.get("buy_conflict_qty") or 0) < 0
                    )
                    put_has_manual_conflict = (
                        is_manual_open
                        and spread_options_put_candidate
                        and int(spread_options_put_candidate.get("buy_conflict_qty") or 0) < 0
                    )

                    call_conflict_order = None
                    if call_has_manual_conflict:
                        call_conflict_order = build_manual_open_conflict_order(
                            ticker,
                            spread_options_call_candidate,
                            parameter['qty'],
                            MANUAL_TRADE_PARAMS.get("request_id"),
                        )

                    put_conflict_order = None
                    if put_has_manual_conflict:
                        put_conflict_order = build_manual_open_conflict_order(
                            ticker,
                            spread_options_put_candidate,
                            parameter['qty'],
                            MANUAL_TRADE_PARAMS.get("request_id"),
                        )

                    if spread_options_call_candidate is not None and spread_options_call_candidate['profit'] > 0.01 and not call_has_manual_conflict:
                        call_spread_orders = accounts.generate_option_order(
                            single_leg_stock_position=single_leg_stock_position_call,
                            action=call_action,
                            custom_order_id=None,
                            spread_sell_option=spread_options_call_candidate['sell_option'],
                            spread_buy_option=spread_options_call_candidate['buy_option'],
                            priceType={"priceType": "NET_CREDIT", "limitPrice": round(spread_options_call_candidate['profit'], 2)}
                        )
        
                    if spread_options_put_candidate is not None and spread_options_put_candidate['profit'] > 0.01 and not put_has_manual_conflict:
                        put_spread_orders = accounts.generate_option_order(
                            single_leg_stock_position=single_leg_stock_position_put,
                            action=put_action,
                            custom_order_id=None,
                            spread_sell_option=spread_options_put_candidate['sell_option'],
                            spread_buy_option=spread_options_put_candidate['buy_option'],
                            priceType={"priceType": "NET_CREDIT", "limitPrice": round(spread_options_put_candidate['profit'], 2)}
                        )
        
                    if call_conflict_order is not None and SKIP_CALL_FLAG is False:
                        preview_orders.append(call_conflict_order)
                    elif call_has_manual_conflict and SKIP_CALL_FLAG is False:
                        update_manual_trade_status(
                            MANUAL_TRADE_PARAMS.get("request_id"),
                            status="failed",
                            message="Could not build BUY_CLOSE-adjusted call spread order"
                        )
                    elif call_spread_orders is not None and SKIP_CALL_FLAG is False and not call_has_manual_conflict:
                        for call_spread_order in call_spread_orders:
                            if call_spread_order is None:
                                continue
                            preview_orders.append({
                                "ticker": ticker,
                                "quote_ticker": spread_options_call_candidate['sell_option'].symbol,
                                "type": "Call spread" if parameter.get('is_cover_call') is None else "Cover call",
                                "order": call_spread_order,
                                "spread_data": spread_options_call_candidate,
                                "quantity": parameter['qty']
                            })
                    else:
                        print(f"No call_spread_orders for {ticker}")
        
                    if put_conflict_order is not None and SKIP_PUT_FLAG is False:
                        preview_orders.append(put_conflict_order)
                    elif put_has_manual_conflict and SKIP_PUT_FLAG is False:
                        update_manual_trade_status(
                            MANUAL_TRADE_PARAMS.get("request_id"),
                            status="failed",
                            message="Could not build BUY_CLOSE-adjusted put spread order"
                        )
                    elif put_spread_orders is not None and SKIP_PUT_FLAG is False and not put_has_manual_conflict:
                        for put_spread_order in put_spread_orders:
                            if put_spread_order is None:
                                continue
                            preview_orders.append({
                                "ticker": ticker,
                                "quote_ticker": spread_options_put_candidate['sell_option'].symbol,
                                "type": "Put spread",
                                "order": put_spread_order,
                                "spread_data": spread_options_put_candidate,
                                "quantity": parameter['qty']
                            })
                    else:
                        print(f"No put spread order for {ticker}")
        
                # Calculate ROI and sort orders
                print("\n===== ORDERS SUMMARY =====")
                total_net_credit = 0
                total_required_margin = 0
        
                for order in preview_orders:
                    order_margin = order['order'].get('required_margin', 0)
                    net_credit = order['spread_data']['profit'] * 100 * order['quantity']
                    if order_margin > 0 and net_credit > 0:
                        days_to_expire = 7
                        roi = (net_credit / order_margin) * 100
                        annualized_roi = roi * (365 / days_to_expire)
                    else:
                        roi = 0
                        annualized_roi = 0
                    order['roi'] = roi
                    order['annualized_roi'] = annualized_roi
                    order['net_credit'] = net_credit
                    order['margin'] = order_margin

                    # Calculate EV and Probability of Assignment
                    try:
                        spot_price = fetch_cached_yf_close(order['ticker']) or order['spread_data']['sell_option'].last_price # rough fallback
                        # Probability engine for current VIX and ticker
                        probability_engine = get_probability_engine(
                            spot_price,
                            vix_value,
                            regime_dict,
                            horizon=7,
                            hmm_model=best_hmm,
                        )
                        
                        metrics = calculate_yield_metrics(
                            order['spread_data']['sell_option'].strike_price,
                            order['spread_data']['buy_option'].strike_price,
                            order['spread_data']['profit'],
                            probability_engine.probability,
                        )
                        order['ev_data'] = {
                            **metrics,
                            'status': probability_engine.validity_status,
                            'execution_eligible': (
                                probability_engine.execution_eligible
                            ),
                            'regime_name': probability_engine.regime_name,
                        }
                    except ProbabilityEngineUnavailable as ev_e:
                        print(
                            "⚠️ EV unavailable for "
                            f"{order['ticker']}: {ev_e.code}"
                        )
                        order['ev_data'] = {
                            'status': 'UNAVAILABLE',
                            'execution_eligible': False,
                            'reason_code': ev_e.code,
                        }
                    except Exception:
                        print(
                            "⚠️ EV unavailable for "
                            f"{order['ticker']}: ENGINE_FAILURE"
                        )
                        order['ev_data'] = {
                            'status': 'UNAVAILABLE',
                            'execution_eligible': False,
                            'reason_code': 'ENGINE_FAILURE',
                        }
        
                preview_orders.sort(key=lambda x: x['roi'], reverse=True)
        
                day_of_week = datetime.now().weekday()
                for i, order in enumerate(preview_orders):
                    assign_result = calculate_probability(order['ticker'], order['spread_data']['sell_option'].distance_to_strike, 'monday', 5, expiration_weeks-1)
                    print(f"Assign result: {assign_result}")
                    assign_probability = assign_result['probability']
                    print(f"{i+1}. {order['ticker']} - {order['type']} - " +
                          f"Credit: ${order['net_credit']:.2f}, " +
                          f"Margin: ${order['margin']:.2f}, " +
                          f"ROI: {order['annualized_roi']:.2f}%, " +
                          f"Distance: {order['spread_data']['sell_option'].distance_to_strike}%, " +
                          f"Assign: {assign_probability:.2f}%")
                    total_net_credit += order['net_credit']
                    total_required_margin += order['margin']
        
                if total_required_margin > 0:
                    portfolio_roi = (total_net_credit / total_required_margin) * 100
                    days_to_expire = 7
                    annualized_portfolio_roi = portfolio_roi * (365 / days_to_expire)
                    roi_display = f", ROI: {portfolio_roi:.2f}%, Annualized ROI: {annualized_portfolio_roi:.2f}%"
                else:
                    roi_display = ""
        
                print("\n=== PORTFOLIO SUMMARY ===")
                print(f"Total Net Credit: ${total_net_credit:.2f}")
                print(f"Total Required Margin: ${total_required_margin:.2f}{roi_display}")
        
                if preview_orders:
                    print(f"Read-only candidate scan complete: {len(preview_orders)} spread candidate(s); no orders submitted.")
                else:
                    print("Read-only candidate scan complete: no valid spread candidates.")
            # --- POSITION DETECTION & MANAGEMENT SECTION ---
            # Now refresh portfolio state to detect high-gain spreads or needed rolls
            print("\n--- Running Position Detection & Management ---")
            all_positions = accounts.portfolio(print_enable=False, require_success=True)
            screened = accounts.screen_option(all_positions)
            html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
            accounts.option_value_final(all_positions)

            # Propose closing high-gain spreads automatically (email approval)
            close_proposals = build_close_proposals(
                screened,
                accounts,
                market,
                live_settings,
                rejected_proposals_today
            )

            # Update global list for dashboard
            CURRENT_CLOSE_PROPOSALS = close_proposals
            CURRENT_NEUTRALIZE_PROPOSALS = build_neutralize_proposals(
                screened,
                accounts,
                market,
                live_settings,
                rejected_proposals_today
            )

            # State Mismatch Bug Fix: Check if requested manual close still exists
            if is_manual_close:
                found = False
                req_ticker = MANUAL_TRADE_PARAMS.get('ticker')
                req_strike = MANUAL_TRADE_PARAMS.get('sell_strike')
                for cp in close_proposals:
                    if cp['ticker'] == req_ticker and float(cp['short_strike']) == float(req_strike):
                        found = True
                        break
                if not found:
                    print(f"⚠️ [Manual Trade] Position {req_ticker} (Strike {req_strike}) no longer qualifies for auto-close thresholds. Skipping execution.")
                    update_manual_trade_status(
                        MANUAL_TRADE_PARAMS.get("request_id"),
                        status="failed",
                        message=f"Position {req_ticker} {req_strike} no longer qualifies for close"
                    )
                    _sync_manual_trade_event()
                    is_manual_close = False

            if close_proposals:
                print(f"Read-only risk scan: {len(close_proposals)} close candidate(s); no orders submitted.")
            else:
                print("   No high-gain spreads detected for closing.")
            # Existing position neutralization logic (unchanged)
            for position in all_positions:
                if position.security_type == "Option" and position.symbol == "BA" and position.strike_price == 1700 and position.call_put == "PUT":
                    result = accounts.neutralize_option_delta(position, all_positions)
                    if result:
                        print("Sell Call:", result["sell_call"].strike_price, result["sell_call"].expiration_date, result["sell_call"].last_price, result["sell_call"].delta)
                        print("Sell Put:", result["sell_put"].strike_price, result["sell_put"].expiration_date, result["sell_put"].last_price, result["sell_put"].delta)
                        print("buy_close_input", result["buy_close_input"].strike_price, result["buy_close_input"].expiration_date, result["buy_close_input"].last_price, result["buy_close_input"].delta)
                        print("buy_close_opposite", result["buy_close_opposite"].strike_price, result["buy_close_opposite"].expiration_date, result["buy_close_opposite"].last_price, result["buy_close_opposite"].delta)
                        print(f"original delta: {result['buy_close_input'].delta} resulting delta change: {(result['sell_call'].delta + result['sell_put'].delta - result['buy_close_input'].delta - result['buy_close_opposite'].delta)}")
                        print(f"Total cost: {(result['sell_call'].last_price + result['sell_put'].last_price - result['buy_close_input'].last_price - result['buy_close_opposite'].last_price)}")
                        order = accounts.custom_option_order(result["sell_call"], result["sell_put"], result["buy_close_input"], result["buy_close_opposite"])
                        print(order)
                        # PREVIEW_ONLY = True
                        # PREVIEW_ONLY = True
                    # breakpoint()

            # --- STALE ORDER MONITORING SECTION ---
            if (datetime.now() - last_stale_check_time).total_seconds() >= 300:
                print("\n--- Stale-order mutation monitoring disabled (read-only mode) ---")
                last_stale_check_time = datetime.now()

            # End of detection loop, wait before next refresh if trade already executed
            sleep_seconds = _next_refresh_sleep_seconds(cycle_started_at)
            if last_trade_date == current_date:
                # Use shorter sleep intervals to allow manual refresh interruption
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tick complete. Next refresh in {sleep_seconds}s.")
                for _ in range(sleep_seconds):
                    if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                        print("🔄 [Refresh Server] Processing manual request...")
                        break

                    t.sleep(1)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tick complete. Next refresh in {sleep_seconds}s.")
                for _ in range(sleep_seconds):
                    if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                        break
                    t.sleep(1)
    
            
            retry_count = 0  # Success, reset retry counter
        except Exception as e:
            retry_count += 1
            wait_time = min(60 * 5, 30 * (2 ** (retry_count - 1)))
            logging.error(f"Error in main loop: {e}")
            logging.error(traceback.format_exc())
            print(f"Error occurred: {e}")
            print(traceback.format_exc())
            print(f"Retrying in {wait_time} seconds... (Attempt {retry_count})")
            t.sleep(wait_time)
