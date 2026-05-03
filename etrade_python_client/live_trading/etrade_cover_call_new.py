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
import traceback
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import timedelta
from datetime import datetime
from logging.handlers import RotatingFileHandler
import logging
import pandas as pd
from live_trading.ev_engine import (
    build_regime_return_arrays, get_probability_engine, calculate_yield_metrics,
    fetch_cached_yf_close
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
from logging.handlers import RotatingFileHandler
from accounts.accounts_bo import Accounts, calculate_std_dev, calculate_margin, print_margin_report, find_highest_margin_ratios
from market.market_bo import Market
import configparser
import multiprocessing
from typing import List
from backtesting import option_limit_backtest
from data_and_research.polygonio_improvequery import get_earnings_dates
from data_and_research.option_assign_probability import calculate_probability
from backtesting.polygonio_dailytrade import fetch_yfinance_data
from pandas_market_calendars import get_calendar
from live_trading.spy_position_tracker import update_spy_daily_snapshot, record_closed_spy_gain
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import base64
TRADE_STATUS_FILE = "trade_status.json"
DASHBOARD_LOG_FILE = "dashboard_requests.log"
AUDIT_LOG_FILE = "order_audit_log.csv"

def log_order_execution(order_info, reason, status="PLACED"):
    """Log order execution details to a permanent CSV file."""
    try:
        file_exists = os.path.exists(AUDIT_LOG_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(AUDIT_LOG_FILE, "a") as f:
            if not file_exists:
                f.write("timestamp,ticker,type,strikes,qty,reason,order_id,status\n")
            
            ticker = order_info.get('ticker', 'N/A')
            order_type = "CLOSE" if order_info.get('is_close') else "OPEN"
            strikes = f"{order_info.get('sell_strike', 'N/A')}/{order_info.get('long_strike', 'N/A')}"
            qty = order_info.get('pair_quantity', order_info.get('qty', 1))
            order_id = order_info.get('order_id', 'N/A')
            
            f.write(f"{timestamp},{ticker},{order_type},{strikes},{qty},{reason},{order_id},{status}\n")
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

def log_dashboard_request(path, data):
    """Log all dashboard API requests to a persistent file."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Strip PIN for security in logs
        log_data = data.copy() if isinstance(data, dict) else data
        if isinstance(log_data, dict) and 'pin' in log_data:
            log_data['pin'] = "****"
            
        entry = {
            "timestamp": timestamp,
            "path": path,
            "data": log_data
        }
        with open(DASHBOARD_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Error logging dashboard request: {e}")

# Global flag to signal a manual refresh request from the web UI
REFRESH_REQUESTED = threading.Event()

# Global flags for order flow
MANUAL_TRADE_REQUESTED = threading.Event()
MANUAL_TRADE_PARAMS = {}
CURRENT_CLOSE_PROPOSALS = [] # Global for dashboard access

def send_login_failure_notification(error_message, screenshot_path=None):
    """Send an email notification when automated login fails."""
    try:
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

def load_live_settings():
    """Load trading settings from JSON file."""
    try:
        if os.path.exists(LIVE_SETTINGS_FILE):
            with open(LIVE_SETTINGS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading settings: {e}")
    
    # Default fallback
    return {
        "target_delta": 0.15,
        "hedge_spread": 20.0,
        "trade_start_time": "07:15:00",
        "trade_end_time": "13:30:00",
        "auto_close_midpoint_threshold": 0.30,
        "auto_close_gain_threshold": 70.0,
        "pair_quantity": 1,
        "target_weeks": 6,
        "target_expiration": null,
        "auto_open_enabled": false,
        "pin": "1234"
    }

def save_live_settings(settings):
    """Save trading settings to JSON file."""
    try:
        with open(LIVE_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"⚠️ Error saving settings: {e}")
        return False


class RefreshHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for the dashboard and API."""
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass
    
    def _send_safe_response(self, code, content, content_type='application/json'):
        """Send a response while safely handling BrokenPipeError."""
        try:
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
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

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        try:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def check_auth(self, auth_header):
        """Verify Basic Auth credentials against settings."""
        settings = load_live_settings()
        user = settings.get('dashboard_user')
        password = settings.get('dashboard_pass')
        
        # If no auth is configured, allow access (fallback to PIN only)
        if not user or not password:
            return True
            
        if not auth_header or not auth_header.startswith('Basic '):
            return False
            
        try:
            auth_decoded = base64.b64decode(auth_header[6:]).decode('utf-8')
            u, p = auth_decoded.split(':', 1)
            return u == user and p == password
        except Exception:
            return False

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
        global MANUAL_TRADE_PARAMS
        
        # Security: Check Basic Auth
        if not self.check_auth(self.headers.get('Authorization')):
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="ETrade Dashboard"')
            self.end_headers()
            return

        if self.path == '/refresh':
            print("\n🔄 [Refresh Server] Manual refresh requested via web UI")
            REFRESH_REQUESTED.set()
            self._send_safe_response(200, {"status": "ok", "message": "Refresh triggered."})
        
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

            for key in ['target_delta', 'hedge_spread', 'trade_start_time', 'trade_end_time', 'auto_close_midpoint_threshold', 'auto_close_gain_threshold', 'pair_quantity', 'target_weeks', 'target_expiration', 'auto_open_enabled', 'trade_side', 'dashboard_user', 'dashboard_pass']:
                if key in new_settings:
                    current_settings[key] = new_settings[key]
            
            if save_live_settings(current_settings):
                self._send_safe_response(200, {"status": "ok"})
            else:
                self._send_safe_response(500, {"error": "Failed to save settings"})

        elif self.path.startswith('/api/execute_manual_order'):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            settings = load_live_settings()
            
            if data.get('pin') != settings.get('pin'):
                log_dashboard_request(self.path, {"error": "Invalid PIN attempt"})
                self._send_safe_response(403, {"error": "Invalid PIN"})
                return

            log_dashboard_request(self.path, data)
            qty = int(data.get('pair_quantity') or data.get('quantity') or data.get('qty') or 1)
            print(f"\n⚡ [Dashboard] Manual order execution requested for: {data.get('side', 'PUT')} (Qty: {qty})")
            MANUAL_TRADE_PARAMS = data
            MANUAL_TRADE_REQUESTED.set()
            self._send_safe_response(200, {"status": "ok", "message": "Order request sent."})

        elif self.path.startswith('/api/close_position') or self.path.startswith('/api/execute_close_order'):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            settings = load_live_settings()
            
            if data.get('pin') != settings.get('pin'):
                log_dashboard_request(self.path, {"error": "Invalid PIN attempt"})
                self._send_safe_response(403, {"error": "Invalid PIN"})
                return

            log_dashboard_request(self.path, data)
            sell_k = data.get('sell_strike') or data.get('strike') or data.get('short_strike')
            long_k = data.get('long_strike')
            qty = int(data.get('quantity') or data.get('qty') or 1)
            print(f"\n⚡ [Dashboard] Manual CLOSE requested for: {data.get('symbol') or data.get('ticker')} {data.get('expiry')} {data.get('cp') or data.get('call_put')} {sell_k}/{long_k} (Qty: {qty})")
            
            # Format for the trading engine
            data['is_close'] = True
            data['ticker'] = data.get('symbol') or data.get('ticker')
            data['sell_strike'] = sell_k
            data['long_strike'] = long_k
            data['side'] = data.get('cp') or data.get('call_put')
            data['pair_quantity'] = qty
            
            MANUAL_TRADE_PARAMS = data
            MANUAL_TRADE_REQUESTED.set()
            self._send_safe_response(200, {"status": "ok", "message": "Close order sent."})
        else:
            self._send_safe_response(404, {"error": "Not found"})

    def do_GET(self):
        """Handle GET requests safely."""
        try:
            self._do_GET_logic()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"⚠️ GET Handler Error: {e}")
            traceback.print_exc()
            self._send_safe_response(500, {"error": str(e)})

    def _do_GET_logic(self):
        # Security: Check Basic Auth
        if not self.check_auth(self.headers.get('Authorization')):
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="ETrade Dashboard"')
            self.end_headers()
            return

        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/dashboard')
            self.end_headers()
            return
        
        elif self.path == '/dashboard':
            template_path = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
            if not os.path.exists(template_path):
                template_path = "live_trading/dashboard_template.html"
            
            with open(template_path, "r") as f:
                html = f.read()
            self._send_safe_response(200, html, 'text/html')

        elif self.path.startswith('/api/settings'):
            settings = load_live_settings()
            display_settings = {k: v for k, v in settings.items() if k != 'pin'}
            self._send_safe_response(200, display_settings)

        elif self.path.startswith('/api/status'):
            is_open, market_status, _, _ = is_market_open()
            trade_status = load_trade_status()
            settings = load_live_settings()
            today = datetime.now()
            target_weeks = settings.get('target_weeks', 6)
            friday = today + timedelta(days=((4 - today.weekday()) % 7) + (target_weeks - 1) * 7)
            
            available_expirations = []
            if 'accounts' in globals() and accounts is not None:
                try:
                    available_expirations = accounts.get_available_expirations("SPY")
                    today_str = today.strftime("%Y-%m-%d")
                    available_expirations = [exp for exp in available_expirations if exp >= today_str][:20]
                except: pass

            status = {
                "market_status": market_status,
                "is_open": is_open,
                "last_trade_date": trade_status.get("last_trade_date"),
                "target_expiration": friday.strftime("%Y-%m-%d"),
                "available_expirations": available_expirations,
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "account_id": accounts.account.get('accountId', 'N/A') if accounts and accounts.account else 'N/A'
            }
            self._send_safe_response(200, status)

        elif self.path.startswith('/api/preview_spread'):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            settings = load_live_settings()
            ticker = "SPY"
            target_delta = float(query.get('delta', [settings.get('target_delta', 0.15)])[0])
            hedge_spread = float(query.get('width', [settings.get('hedge_spread', 20)])[0])
            target_weeks = int(query.get('weeks', [settings.get('target_weeks', 6)])[0])
            explicit_expiry = query.get('expiration', [None])[0]
            side = query.get('side', ["Put"])[0].capitalize()
            
            today = datetime.now()
            if explicit_expiry:
                expiry_dt = datetime.strptime(explicit_expiry, "%Y-%m-%d")
                days_to_expire = (expiry_dt.date() - today.date()).days
            else:
                friday = today + timedelta(days=((4 - today.weekday()) % 7) + (target_weeks - 1) * 7)
                days_to_expire = (friday.date() - today.date()).days
                if days_to_expire < 1: days_to_expire = 7

            if 'accounts' not in globals() or accounts is None:
                self._send_safe_response(503, {"error": "Engine loading..."})
                return

            spread = accounts.get_option_spread_by_price(
                ticker, side, days_to_expire=days_to_expire, 
                target_premium=0, hedge_ratio=1, 
                hedge_spread=hedge_spread, qty=1, 
                target_delta=target_delta
            )
            
            if spread:
                sell_leg = spread.get('sell_option')
                buy_leg = spread.get('buy_option')
                res = {
                    "ticker": ticker,
                    "sell_strike": sell_leg.strike_price if sell_leg else "N/A",
                    "buy_strike": buy_leg.strike_price if buy_leg else "None",
                    "premium": spread.get('profit', 0),
                    "expiration": sell_leg.expiration_date.strftime("%Y-%m-%d") if sell_leg and hasattr(sell_leg, 'expiration_date') else "N/A",
                    "delta": sell_leg.delta if sell_leg and hasattr(sell_leg, 'delta') else target_delta,
                    "side": side.upper(),
                    "is_mock": False
                }
            else:
                res = {
                    "ticker": ticker,
                    "sell_strike": 700.0 - (target_delta * 100),
                    "buy_strike": 700.0 - (target_delta * 100) - hedge_spread,
                    "premium": 1.25 + (target_delta * 5),
                    "expiration": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "delta": target_delta,
                    "is_mock": True
                }
            self._send_safe_response(200, res)

        elif self.path.startswith('/api/positions'):
            if os.path.exists("screened_option_pairs.html"):
                with open("screened_option_pairs.html", "r") as f:
                    content = f.read()
                self._send_safe_response(200, content, 'text/html')
            else:
                self._send_safe_response(404, "No position data available.", 'text/plain')

        elif self.path.startswith('/api/high_gain_spreads'):
            clean_proposals = []
            for p in CURRENT_CLOSE_PROPOSALS:
                cp = p.copy()
                if 'close_order' in cp: del cp['close_order']
                clean_proposals.append(cp)
            self._send_safe_response(200, clean_proposals)
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

def start_refresh_server(port=8765):
    """Start the HTTP refresh server in a background thread."""
    server = HTTPServer(('0.0.0.0', port), RefreshHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"🌐 [Refresh Server] Started on http://localhost:{port}")
    
    # Start ngrok tunnel if configured
    start_ngrok_tunnel(config, port)
    
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
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("python_client.log", maxBytes=5*1024*1024, backupCount=3)
FORMAT = "%(asctime)-15s %(message)s"
fmt = logging.Formatter(FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
handler.setFormatter(fmt)
logger.addHandler(handler)
'''
    Grab the option expire dates and option chains for the specified symbol.
    Save as a JSON file

'''
OAUTH_KEYS = {
    "sandbox": {
        "consumer_key": config['DEFAULT'].get('SANDBOX_CONSUMER_KEY', os.getenv("ETRADE_SANDBOX_CONSUMER_KEY", "default_sandbox_key")),
        "consumer_secret": config['DEFAULT'].get('SANDBOX_CONSUMER_SECRET', os.getenv("ETRADE_SANDBOX_CONSUMER_SECRET", "default_sandbox_secret")),
    },
    "live": {
        "consumer_key": config['DEFAULT'].get('PROD_CONSUMER_KEY', os.getenv("ETRADE_LIVE_CONSUMER_KEY", "default_live_key")),
        "consumer_secret": config['DEFAULT'].get('PROD_CONSUMER_SECRET', os.getenv("ETRADE_LIVE_CONSUMER_SECRET", "default_live_secret")),
    }
}

# File to cache OAuth tokens so you don't have to re-authenticate each time
ETRADE_OAUTH_FILE = ".etrade_oauth"

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
    Fetch SPY and VIX prices via E*TRADE and record them in the cache at market close.
    """
    try:
        from datetime import datetime as dt
        import json
        
        CACHE_FILE = "spy_vix_price_cache.json"
        today_str = dt.now().strftime("%Y-%m-%d")
        
        print(f"[Market Close] Recording SPY and VIX prices for {today_str}...")
        
        # Fetch quotes
        resp = market_instance.get_quote(['SPY', '^VIX'], resp_format="json")
        if not resp or 'QuoteResponse' not in resp or 'QuoteData' not in resp['QuoteResponse']:
            print("[Market Close] Error: Could not fetch quotes from E*TRADE.")
            return

        quotes = resp['QuoteResponse']['QuoteData']
        prices = {}
        for q in quotes:
            sym = q.get('Product', {}).get('symbol')
            # E*TRADE might return SPY or ^VIX
            last_price = q.get('All', {}).get('lastTrade')
            if sym and last_price:
                prices[sym] = round(float(last_price), 2)


        if 'SPY' not in prices or '^VIX' not in prices:
             # Try to search for VIX if ^VIX didn't work (E*TRADE symbol convention varies)
             if 'VIX' in prices and '^VIX' not in prices:
                 prices['^VIX'] = prices['VIX']
             else:
                 print(f"[Market Close] Warning: Missing data in quotes: {prices}")

        # Load and update cache
        cache = {"SPY": {}, "VIX": {}}
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
        if '^VIX' in prices or 'VIX' in prices:
            v_price = prices.get('^VIX') or prices.get('VIX')
            cache.setdefault("VIX", {})[today_str] = v_price
            updated = True
            
        if updated:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)
            print(f"[Market Close] Successfully recorded prices in {CACHE_FILE}: {prices}")
            
    except Exception as e:
        print(f"[Market Close] Error recording prices: {e}")

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
    """
    Release margin by closing high-margin spreads.

    Args:
        all_positions: List of all positions in the portfolio.
        cover_call_list: Dictionary mapping ticker symbols to number of exempt covered call contracts.
        etrade_instance: Instance of E*TRADE client.
        max_positions: Maximum number of positions to close. Default is 5.

    Returns:
        int: Number of positions processed.
    """
    import time

    # Set to track positions already being closed
    already_closing = set()
    positions_processed = 0

    # Calculate initial margin
    total_margin, margin_details = calculate_margin(all_positions, cover_call_list)
    initial_margin = total_margin

    balance_data_start = accounts.balance()
    print("Margin Buying Power: " + str('${:,.2f}'.format(balance_data_start["Computed"]["marginBuyingPower"])))

    # Get the top candidates using the updated function (top_n equals max_positions)
    candidates = find_highest_margin_ratios(margin_details, already_closing, top_n=max_positions)

    orders = []
    for candidate in candidates:
        ticker, pair, ratio = candidate
        # Mark the candidate as already being processed
        position_id = (
            ticker,
            pair.get("expiry"),
            pair.get("type"),
            pair.get("short_strike"),
            pair.get("long_strike")
        )
        already_closing.add(position_id)

        expiry = pair.get("expiry")
        # Create the StockPosition object for the short position
        short_position = StockPosition(
            symbol=ticker,
            quantity=pair.get("quantity", 1)
        )
        short_position.security_type = "Option"
        short_position.call_put = pair.get("type")
        short_position.strike_price = pair.get("short_strike")
        short_position.expiration_date = expiry
        short_position.last_price = round(pair.get("short_price", 0),2)

        # Create the StockPosition object for the long position
        long_position = StockPosition(
            symbol=ticker,
            quantity=pair.get("quantity", 1)
        )
        long_position.security_type = "Option"
        long_position.call_put = pair.get("type")
        long_position.strike_price = pair.get("long_strike")
        long_position.expiration_date = expiry
        long_position.last_price = round(pair.get("long_price", 0),2)

        # Generate orders for closing the positions
        close_short_orders = accounts.generate_option_order(
            single_leg_stock_position=short_position,
            action="BUY_CLOSE",
            custom_order_id=None,
            spread_sell_option=None,
            spread_buy_option=None,
            priceType={"priceType": "LIMIT", "limitPrice": round(pair.get("short_price", 0)+0.01,2)}
        )

        close_long_orders = accounts.generate_option_order(
            single_leg_stock_position=long_position,
            action="SELL_CLOSE",
            custom_order_id=None,
            spread_sell_option=None,
            spread_buy_option=None,
            priceType={"priceType": "LIMIT", "limitPrice": round(pair.get("long_price", 0)-0.01,2)}
        )

        # orders.extend(close_short_orders + close_long_orders)
        orders.extend(close_short_orders+close_long_orders)

    # Process each order generated for the selected candidates
    for order in orders:
        print(f"Processing order: {order}")
        if etrade_instance:
            preview_response = etrade_instance.order.place_order(order, preview_only=True)
            print(f"Preview response: {preview_response}")
            if order['limitPrice'] > 0.05:
                print("Order executed automatically (Auto-approval enabled).")
                # user_input = input("Would you like to execute this order for real? (yes/no): ").strip().lower()
                # if user_input not in ['yes', 'y']:
                #     print("Order skipped.")
                #     continue
            order_id = etrade_instance.order.place_order(order, preview_only=False)
            print(f"Order placed with ID: {order_id}")
        positions_processed += 1
        if positions_processed >= max_positions:
            break

    balance_data_end = accounts.balance()
    print("Margin Buying Power start: " + str('${:,.2f}'.format(balance_data_start["Computed"]["marginBuyingPower"])))
    print("Margin Buying Power end: " + str('${:,.2f}'.format(balance_data_end["Computed"]["marginBuyingPower"])))

    return positions_processed

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
            new_price = round(sell_mid - buy_mid, 2)
        else:
            new_price = round(abs(sell_mid - buy_mid), 2)

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
    """
    Monitor open limit orders, nudge those > 10m old if near midpoint ($0.05).
    Strategy: Nudge +$0.01, wait 1m, if not executed, revert and wait 5m.
    """
    state_file = "nudge_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                nudge_state = json.load(f)
        except Exception:
            nudge_state = {}
    else:
        nudge_state = {}

    try:
        open_orders = etrade_instance.order.get_open_orders()
    except Exception as e:
        print(f"   ⚠️ [Stale Monitor] Error fetching open orders: {e}")
        return

    if not open_orders:
        return

    # Group by orderId
    orders_grouped = {}
    for o in open_orders:
        oid = str(o['orderId'])
        if oid not in orders_grouped:
            orders_grouped[oid] = []
        orders_grouped[oid].append(o)

    now = datetime.now()
    now_ms = int(now.timestamp() * 1000)
    state_changed = False

    for oid, legs in orders_grouped.items():
        first_leg = legs[0]
        placed_time = first_leg.get('placedTime')
        order_term = first_leg.get('orderTerm')
        
        # Only monitor Good-for-Day limit orders
        if order_term != 'GOOD_FOR_DAY':
            continue
        
        price_type = first_leg.get('priceType')
        if price_type not in ('NET_CREDIT', 'NET_DEBIT', 'LIMIT'):
            continue

        # Check age > 10m
        if placed_time and (now_ms - int(placed_time)) < (10 * 60 * 1000):
            continue

        state = nudge_state.get(oid, {"phase": "NORMAL"})
        
        # 1. Handling "NUDGE_TEST" phase (Wait 1 minute)
        if state['phase'] == "NUDGE_TEST":
            nudge_start_ms = state.get('nudge_start_time', 0)
            if (now_ms - nudge_start_ms) >= (1 * 60 * 1000):
                print(f"   🕒 [Nudge] Order {oid} nudge time expired (1m). Reverting...")
                orig_price = state.get('original_price')
                if orig_price is not None:
                    if not dry_run:
                        print(f"   🔄 [Nudge] Reverting order {oid} to original price ${orig_price:.2f}")
                        ok, new_id = etrade_instance.order.change_order_limit(int(oid), orig_price, first_leg['allDetail'], first_leg['orderType'])
                        if ok:
                            log_nudge_action("REVERT", oid, first_leg['symbol'], f"Price ${orig_price:.2f}")
                            nudge_state[oid] = {
                                "phase": "COOL_DOWN",
                                "last_revert_time": now_ms,
                                "original_price": orig_price
                            }
                            if new_id:
                                # New order ID after change; move state
                                nudge_state[str(new_id)] = nudge_state[oid]
                                del nudge_state[oid]
                            state_changed = True
                    else:
                        print(f"   [DRY RUN] Would revert order {oid} to ${orig_price:.2f}")
                        state['phase'] = "COOL_DOWN"
                        state['last_revert_time'] = now_ms
                        state_changed = True
            continue

        # 2. Handling "COOL_DOWN" phase (Wait 5 minutes)
        if state['phase'] == "COOL_DOWN":
            revert_time = state.get('last_revert_time', 0)
            if (now_ms - revert_time) < (5 * 60 * 1000):
                continue
            else:
                state['phase'] = "NORMAL"
                state_changed = True

        # 3. Handling "NORMAL" phase: Check for eligibility
        # Calculate current midpoint
        osi_list = []
        for leg in legs:
            symbol = leg['symbol']
            if leg['securityType'] == 'OPTN' and leg.get('expiryDate'):
                exp = leg['expiryDate'].replace('-', ':')
                osi = f"{symbol}:{exp}:{leg['callPut']}:{leg['strikePrice']}"
                osi_list.append(osi)
            else:
                osi_list.append(symbol)

        try:
            resp = market_instance.get_quote(osi_list, resp_format="json")
            quotes = resp.get("QuoteResponse", {}).get("QuoteData", [])
            
            # Map quotes back to legs to compute net mid
            net_mid = 0
            found_all = True
            for leg in legs:
                leg_mid = None
                for q in quotes:
                    q_symbol = q.get("Product", {}).get("symbol")
                    # Match symbol or OSI
                    if q_symbol == leg['symbol'] or (leg['securityType'] == 'OPTN' and leg['symbol'] in q_symbol):
                        all_q = q.get("All", {})
                        leg_mid = (float(all_q.get("bid", 0)) + float(all_q.get("ask", 0))) / 2
                        break
                
                if leg_mid is not None:
                    # Direction depends on orderAction
                    action = leg.get('orderAction', '')
                    if action in ('SELL', 'SELL_SHORT', 'SELL_OPEN'):
                        net_mid += leg_mid
                    else:
                        net_mid -= leg_mid
                else:
                    found_all = False
                    break
            
            if not found_all:
                continue
            
            # If net_mid is negative (buy), we flip it for comparison with limitPrice which is usually positive abs
            current_limit = float(first_leg.get('limitPrice', 0) or 0)
            
            # Threshold Check ($0.05)
            if abs(current_limit - abs(net_mid)) <= 0.05:
                # Nudge it!
                nudge_step = 0.01
                if price_type == 'NET_CREDIT':
                    new_limit = current_limit + nudge_step
                elif price_type == 'NET_DEBIT':
                    new_limit = max(0.01, current_limit - nudge_step)
                else:
                    # Infer from action if needed, but usually LIMIT is for single leg
                    # For a simple buy limit, more favorable is lower price
                    if any(l.get('orderAction','').startswith('BUY') for l in legs):
                        new_limit = max(0.01, current_limit - nudge_step)
                    else:
                        new_limit = current_limit + nudge_step
                
                print(f"   🎯 [Stale Nudge] Order {oid} is near mid (${abs(net_mid):.2f}). Nudging ${current_limit:.2f} -> ${new_limit:.2f}")
                if not dry_run:
                    ok, new_id = etrade_instance.order.change_order_limit(int(oid), new_limit, first_leg['allDetail'], first_leg['orderType'])
                    if ok:
                        log_nudge_action("NUDGE", oid, first_leg['symbol'], f"Price {current_limit:.2f} -> {new_limit:.2f}")
                        nudge_state[oid] = {
                            "phase": "NUDGE_TEST",
                            "nudge_start_time": now_ms,
                            "original_price": current_limit
                        }
                        if new_id:
                            nudge_state[str(new_id)] = nudge_state[oid]
                            del nudge_state[oid]
                        state_changed = True
                else:
                    print(f"   [DRY RUN] Would nudge order {oid} to ${new_limit:.2f}")
                    nudge_state[oid] = {
                        "phase": "NUDGE_TEST",
                        "nudge_start_time": now_ms,
                        "original_price": current_limit
                    }
                    state_changed = True

        except Exception as e:
            print(f"   ⚠️ [Stale Monitor] Error during nudge logic for {oid}: {e}")
            continue

    if state_changed:
        with open(state_file, 'w') as f:
            json.dump(nudge_state, f, indent=4)

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
        with open(ETRADE_OAUTH_FILE) as f:
            tokens = json.load(f)
            return tokens[environment_key(use_sandbox)]
    except (KeyError, TypeError, FileNotFoundError, JSONDecodeError) as err:
        print("Couldn't find/parse cached OAuth in {} ({}: {})".format(ETRADE_OAUTH_FILE, err))
        return None

# Save the token, merging in with existing tokens
def save_etrade_oauth(token, use_sandbox) -> bool:
    try:
        try:
            with open(ETRADE_OAUTH_FILE) as f:
                tokens = json.load(f)
        except FileNotFoundError:
            tokens = {}
        tokens[environment_key(use_sandbox)] = token
        with open(os.open(ETRADE_OAUTH_FILE, os.O_CREAT | os.O_WRONLY, 0o600), "w") as f:
            f.write(json.dumps(tokens))
    except (KeyError, JSONDecodeError) as err:
        print("Couldn't write cached OAuth in {} ({})".format(ETRADE_OAUTH_FILE, err))
        sys.exit(1)

from urllib.parse import parse_qsl

def oauth(use_sandbox, auto_login=True, username=None, password=None, headless=None):
    """Allows user authorization for the sample application with OAuth 1"""
    keys = OAUTH_KEYS[environment_key(use_sandbox)]
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]
    
    print(f"Environment: {'Sandbox' if use_sandbox else 'Live'}")
    print(f"Base URL: {'https://apisb.etrade.com' if use_sandbox else 'https://api.etrade.com'}")
    print(f"Consumer Key used: {consumer_key[:4]}...{consumer_key[-4:]}")
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

    token_file = ".etrade_oauth"

    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            tokens = json.load(f)
        
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
    with open(token_file, 'w') as f:
        json.dump(tokens, f)

    print("New session created and tokens saved.")
    return session, base_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grab all the option chains for the specified symbol',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', help='use sandbox?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--trade', help='start live trade?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_existing_file', help='re-use the backtest results?', action=argparse.BooleanOptionalAction)
    parser.add_argument('--no-regime', help='skip heavy market regime detection?', action='store_true')

    parser.add_argument('--username', help='username for login', type=str, required=False)
    parser.add_argument('--password', help='password for login', type=str, required=False)
    parser.add_argument('--no-headless', help='disable headless mode for login', action='store_true')
    args = parser.parse_args()

    # --- SINGLETON LOCK ---
    import fcntl
    lock_file_path = '/tmp/etrade_trader.lock'
    lock_file = open(lock_file_path, 'w')
    try:
        fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print(f"❌ Error: Another instance of etrade_cover_call_new is already running.")
        sys.exit(1)
    use_sandbox = args.sandbox
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

    keys = OAUTH_KEYS[environment_key(use_sandbox)]
    consumer_key = keys["consumer_key"]
    consumer_secret = keys["consumer_secret"]

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
    
    # Load dynamic settings
    live_settings = load_live_settings()
    trade_start_time = datetime.strptime(live_settings.get('trade_start_time', '07:15:00'), '%H:%M:%S').time()
    trade_end_time = datetime.strptime(live_settings.get('trade_end_time', '13:30:00'), '%H:%M:%S').time()

    datalog_start_time = datetime.strptime('06:45:00', '%H:%M:%S').time()
    datalog_end_time = datetime.strptime('13:15:00', '%H:%M:%S').time()

    last_renewal_time = datetime.now()

    stock_positions: List[StockPosition] = []

    if start_trade:
        if not bypass_etrade: 
            etrade_instance = LiveTradeAgent(None,session,base_url, use_sandbox=use_sandbox)
        else:
            etrade_instance = LiveTradeAgent()
        print('Live trade agent id: ', etrade_instance.agent_id)

        accounts = Accounts(session, base_url)
        accounts.account_list(1) # Select the Individual Brokerage account ending in 8703
        market = Market(session, base_url)



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
        # close_order_id = etrade_instance.order.place_order(vix_spread_order_1, preview_only=True)
        # close_order_id = etrade_instance.order.place_order(vix_spread_order_2, preview_only=True)

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
        # etrade_instance.order.place_order(quick_order)
        # breakpoint()        
    positions_to_roll=None
    now = None
    auto_closed_spread_ids = set()
    extrinsic_alert_sent = set()
    rejected_proposals_today = set()
    prev_date = None
    nudge_email_sent_for_date = None

    last_stale_check_time = datetime.now() - timedelta(minutes=6)

    # Load EV/Probability engine data
    if no_regime:
        print("⏭️  Skipping market regime detection (--no-regime).")
        regime_dict, best_hmm, daily_models = {}, None, []
    else:
        print("📈 Loading regime-based return data for EV engine...")
        # build_regime_return_arrays will now check for cached data first
        regime_dict, best_hmm, daily_models = build_regime_return_arrays(int(t.time()/86400), horizon=7) 


    # Start the HTTP server for manual refresh requests from web UI
    try:
        refresh_server = start_refresh_server(port=8765)

    except Exception as e:
        print(f"⚠️ Could not start refresh server: {e}")
        refresh_server = None

    retry_count = 0
    prev_market_status = None
    
    # Guard: Ensure --trade flag was passed, otherwise accounts/etrade_instance are undefined
    if not start_trade:
        print("❌ Error: You must run with --trade flag for the refresh functionality to work.")
        print("   Example: python3 etrade_cover_call_new.py --no-sandbox --trade --username <user> --password <pass>")
        sys.exit(1)
    
    while True:
        try:
            # Check for manual refresh request
            is_manual_refresh = REFRESH_REQUESTED.is_set()
            if is_manual_refresh:
                REFRESH_REQUESTED.clear()

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
                    accounts = Accounts(session, base_url)
                    market = Market(session, base_url)
                    accounts.account_list(1)
                    if start_trade and not bypass_etrade:
                        etrade_instance.refresh_session(session, base_url)
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

            # Detect Market Close Transition to record prices
            if prev_market_status == "OPEN" and market_status == "AFTER_HOURS":
                record_market_close_prices(etrade_instance.market)
            prev_market_status = market_status

            is_manual_trade = MANUAL_TRADE_REQUESTED.is_set()
            
            # --- MANUAL CLOSE ORDER HANDLING ---
            if is_manual_trade and MANUAL_TRADE_PARAMS.get('is_close'):
                print("\n🔐 [Manual Trade] Processing CLOSE order from dashboard...")
                data = MANUAL_TRADE_PARAMS
                MANUAL_TRADE_REQUESTED.clear()
                
                try:
                    ticker = data.get('ticker') or data.get('symbol')
                    exp = data.get('expiry') or data.get('expiration')
                    cp = data.get('side') or data.get('call_put') or data.get('cp')
                    short_strike = data.get('sell_strike') or data.get('short_strike') or data.get('strike')
                    long_strike = data.get('long_strike') # May be None for single leg
                    qty = int(data.get('pair_quantity') or data.get('quantity') or data.get('qty') or 1)
                    
                    exp_parts = exp.split('-')
                    # format: SYMBOL:YYYY:MM:DD:TYPE:STRIKE
                    short_osi = f"{ticker}:{exp_parts[0]}:{exp_parts[1]}:{exp_parts[2]}:{cp}:{short_strike}"
                    
                    if long_strike:
                        long_osi = f"{ticker}:{exp_parts[0]}:{exp_parts[1]}:{exp_parts[2]}:{cp}:{long_strike}"
                        symbols_to_quote = [short_osi, long_osi]
                    else:
                        symbols_to_quote = [short_osi]
                    
                    # Check if midpoint was passed from dashboard for consistency
                    limit_debit = data.get('midpoint')
                    if limit_debit is not None:
                        limit_debit = abs(float(limit_debit))
                        print(f"   [Manual Trade] Using dashboard midpoint: ${limit_debit:.2f}")
                    else:
                        # Fetch latest midpoint for limit price
                        resp = market.get_quote(symbols_to_quote, resp_format="json")
                        q_data = resp.get("QuoteResponse", {}).get("QuoteData", [])
                        
                        def find_quote(osi, data_list):
                            parts = osi.split(':')
                            for q in data_list:
                                p = q.get("Product", {})
                                try:
                                    if (p.get("symbol") == parts[0] and 
                                        str(p.get("expiryYear")) == parts[1] and 
                                        int(p.get("expiryMonth")) == int(parts[2]) and 
                                        int(p.get("expiryDay")) == int(parts[3]) and 
                                        p.get("callPut") == parts[4] and 
                                        abs(float(p.get("strikePrice", 0)) - float(parts[5])) < 0.01):
                                        return q.get("All", {})
                                except: continue
                            return None

                        s_q = find_quote(short_osi, q_data)
                        l_q = find_quote(long_osi, q_data) if long_strike else None
                        
                        if s_q:
                            mid_s = (float(s_q.get('bid',0)) + float(s_q.get('ask',0))) / 2
                            if long_strike and l_q:
                                mid_l = (float(l_q.get('bid',0)) + float(l_q.get('ask',0))) / 2
                                limit_debit = abs(mid_s - mid_l)
                            else:
                                limit_debit = mid_s
                        else:
                            limit_debit = 0.05 # Fallback

                    if limit_debit:
                        from accounts.accounts_bo import StockPosition
                        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                        s_leg = StockPosition(symbol=ticker, quantity=-qty, security_type="Option", 
                                              strike_price=float(short_strike), call_put=cp, expiration_date=exp_date)
                        
                        if long_strike:
                            l_leg = StockPosition(symbol=ticker, quantity=qty, security_type="Option", 
                                                  strike_price=float(long_strike), call_put=cp, expiration_date=exp_date)
                            
                            close_order = accounts.generate_option_order(
                                single_leg_stock_position=None,
                                action="SPREAD",
                                spread_sell_option=l_leg,
                                spread_buy_option=s_leg,
                                priceType={'priceType': 'NET_DEBIT', 'limitPrice': round(limit_debit, 2)}
                            )
                            print(f"   Submitting SPREAD close order for {ticker} {cp} {short_strike}/{long_strike} @ ${limit_debit:.2f} (Qty: {qty})")
                        else:
                            close_order = accounts.generate_option_order(
                                single_leg_stock_position=s_leg,
                                action="BUY_TO_CLOSE",
                                priceType={'priceType': 'LIMIT', 'limitPrice': round(limit_debit, 2)}
                            )
                            print(f"   Submitting SINGLE leg close order for {ticker} {cp} {short_strike} @ ${limit_debit:.2f} (Qty: {qty})")
                        
                        if isinstance(close_order, list): close_order = close_order[0]
                        order_id = etrade_instance.order.place_order(close_order, preview_only=False)
                        if order_id:
                            print(f"   ✅ Close order submitted! ID: {order_id}")
                            if ticker == 'SPY':
                                try: record_closed_spy_gain(data)
                                except: pass
                        else:
                            print(f"   ❌ Close order submission failed.")
                    else:
                        print(f"   ❌ Could not fetch latest quotes for {symbols_to_quote}.")
                except Exception as e:
                    print(f"   ❌ Error executing manual close: {e}")
                    traceback.print_exc()
                
                REFRESH_REQUESTED.set()
                continue
            
            if market_status == "CLOSED_HOLIDAY" and not is_manual_trade:
                print(f"📅 Today ({current_date}) is not a trading day (market holiday).")

                # If manual refresh was requested, fetch portfolio and update HTML
                if is_manual_refresh:
                    print("🔄 [Manual Refresh] Fetching portfolio and updating HTML (market closed)...")
                    try:
                        accounts.account_list(1)
                        all_positions = accounts.portfolio(print_enable=False)
                        screened = accounts.screen_option(all_positions)
                        html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
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
                    # Use interruptible sleep to allow manual refresh (check every second, max 10 min)
                    for _ in range(min(600, int(seconds_until_next))):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # ── Market is a trading day (PRE_MARKET, OPEN, or AFTER_HOURS) ──
            # Run read-only account/portfolio operations
            accounts.account_list(1)
            today = datetime.now()
            passed_monday = today - timedelta(days=((today.weekday()) % 7))
            etrade_instance.order.option_gain_new(passed_monday.strftime("%Y-%m-%d"))
            # etrade_instance.order.option_gain_new('2026-01-01')
            accounts.balance()
    
            all_positions = accounts.portfolio(print_enable=True)  

            # --- EXTRINSIC VALUE ALERTS FOR ITM SHORT OPTIONS ---
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
                all_positions = accounts.portfolio(print_enable=False)
                
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
            # etrade_instance.order.refresh_order_limit()

            # If market is not open (PRE_MARKET or AFTER_HOURS), update HTML but skip position actions
            if market_status in ("PRE_MARKET", "AFTER_HOURS") and not is_manual_trade:
                if is_manual_refresh:
                    print(f"🔄 [Manual Refresh] Fetching portfolio and updating HTML ({market_status})...")
                else:
                    print(f"\n--- Updating screened_option_pairs.html ({market_status}) ---")
                all_positions = accounts.portfolio(print_enable=False)
                screened = accounts.screen_option(all_positions)
                html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
                print(f"HTML updated: {html_path}")

                if market_status == "PRE_MARKET" and market_open_time is not None:
                    # Sleep until market open (or our trade_start_time, whichever is later)
                    import pytz
                    eastern = pytz.timezone('US/Eastern')
                    trade_start_dt = eastern.localize(datetime.combine(now.date(), trade_start_time))
                    wake_target = max(market_open_time, trade_start_dt)
                    seconds_until_open = (wake_target - datetime.now(eastern)).total_seconds()
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
                    sleep_seconds = min(600, int(seconds_until_open))
                    for _ in range(sleep_seconds):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # ── Trading window check (market is OPEN, but are we within user's trading hours?) ──
            is_manual_trade = MANUAL_TRADE_REQUESTED.is_set()
            if not (trade_start_time <= current_time <= trade_end_time) and not is_manual_trade:
                print(f"\n⏰ Market is OPEN but outside trading window ({trade_start_time}–{trade_end_time}). Skipping position actions.")
                # Still update the HTML so users can see current positions
                print("--- Updating screened_option_pairs.html (outside trading window) ---")
                all_positions = accounts.portfolio(print_enable=False)
                screened = accounts.screen_option(all_positions)
                html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
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
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sleeping until trading window: {next_trade_time}...")
                    sleep_seconds = min(600, int(seconds_until_trade))
                    for _ in range(sleep_seconds):
                        if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                            print("🔄 [Refresh Server] Processing manual request...")
                            break
                        t.sleep(1)
                continue

            # --- AUTO-CLOSE EXPIRING ITM PUT SPREADS ---
            if current_time >= datetime.strptime('12:50:00', '%H:%M:%S').time():
                for pair in screened_temp:
                    if pair.get("is_spread") and pair["short_lot"].call_put == "PUT" and pair["long_lot"].call_put == "PUT":
                        short_lot = pair["short_lot"]
                        long_lot = pair["long_lot"]
                        exp_date_str = str(short_lot.expiration_date)[:10]
                        
                        if exp_date_str == current_date:
                            # Short put is ITM if underlying price is below strike
                            if 0 < short_lot.underlying_last_price < short_lot.strike_price:
                                spread_id = f"{short_lot.symbol}_{short_lot.strike_price}_{long_lot.strike_price}_{current_date}"
                                if spread_id not in auto_closed_spread_ids:
                                    print(f"\n🔥 [Auto-Close] {short_lot.symbol} Put Credit Spread (Short {short_lot.strike_price}) ITM and expiring today! Closing at MARKET...")
                                    try:
                                        price_type_market = {'priceType': 'MARKET', 'limitPrice': 0.1}
                                        close_orders = accounts.generate_option_order(
                                            single_leg_stock_position=None, 
                                            action="SPREAD", 
                                            custom_order_id=0,
                                            spread_sell_option=long_lot,  # Sell our long leg
                                            spread_buy_option=short_lot,  # Buy our short leg
                                            priceType=price_type_market
                                        )
                                        if close_orders:
                                            order_id = etrade_instance.order.place_order(close_orders[0], preview_only=False)
                                            if order_id:
                                                print(f"✅ Market order {order_id} placed to close spread {spread_id}.")
                                                auto_closed_spread_ids.add(spread_id)
                                    except Exception as e:
                                        print(f"❌ Failed to auto-close spread {spread_id}: {e}")

            # --- POSITION OPENING ACTION SECTION ---
            # Attempt to open new positions if auto-open is enabled and we haven't traded yet today 
            # OR if manual refresh/manual trade is requested
            auto_open_enabled = live_settings.get('auto_open_enabled', False)
            # is_manual_trade already checked above
            if is_manual_trade:
                print("\n⚡ [Main Loop] Manual trade request detected. Proceeding to execution logic...")
                MANUAL_TRADE_REQUESTED.clear()
            
            # Source of truth for execution: dashboard manual trade or auto-open enabled
            is_manual_close = is_manual_trade and MANUAL_TRADE_PARAMS.get('is_close', False)
            is_manual_open = is_manual_trade and not is_manual_close
            
            # --- OPENING EXECUTION GATE ---
            # Only search for new positions if it's an AUTO-OPEN or a MANUAL-OPEN request
            execute_open_now = (auto_open_enabled and last_trade_date != current_date) or is_manual_open
            
            if execute_open_now or is_manual_refresh:
                if is_manual_refresh:
                     print(f"🔄 [Manual Refresh] Re-evaluating candidates for dashboard...")
                if is_manual_open:
                     print("\n⚡ [Main Loop] Manual OPEN request detected. Proceeding to execution logic...")
                     MANUAL_TRADE_REQUESTED.clear()
                cover_call_list = {}
                # 'AAPL': 5,
                # 'AMD': 1,
                # 'BRK.B': 2,
                # 'JPM': 1,
                # 'NVDA': 1,
                # 'QCOM': 1,

        
                RELEASE_MARGIN_ENABLE = False
                if RELEASE_MARGIN_ENABLE:
                    etrade_instance.order.cancel_all_order()
                    positions_processed = 0
                    while positions_processed < 10:
                        print(f"Start releasing margin...")
                        all_positions = accounts.portfolio()   
                        accounts.balance() 
                        positions_processed = release_margin(
                            all_positions=all_positions, 
                            cover_call_list=None, 
                            etrade_instance=etrade_instance,
                            max_positions=5
                        )
                        # breakpoint()
                # Execute trades automatically at trading start based on dashboard settings
                expiration_weeks = live_settings.get('target_weeks', 6)
                trade_side = live_settings.get('trade_side', 'BOTH')
                
                # Source of truth is now the dashboard settings
                parameters = [{
                    "ticker": "SPY",
                    "pnl": 0,
                    "target_premium_call": 0,
                    "target_premium_put": 0,
                    "hedge_ratio_call": 1 if trade_side in ['CALL', 'BOTH'] else 0,
                    "hedge_ratio_put": 1 if trade_side in ['PUT', 'BOTH'] else 0,
                    "qty": live_settings.get('pair_quantity', 1)
                }]
                
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
                    saved_expiry = live_settings.get('target_expiration')
                    if saved_expiry:
                        try:
                            # Use saved expiration if valid and in future
                            friday_date = datetime.strptime(saved_expiry, "%Y-%m-%d").date()
                            if friday_date < today_date:
                                raise ValueError("Past expiration")
                            days_to_expire = (friday_date - today_date).days
                        except Exception:
                            # Fallback to weeks if saved expiry is invalid or past
                            tw = live_settings.get('target_weeks', 6)
                            target_date = today_date + timedelta(weeks=tw)
                            days_until_friday = (4 - target_date.weekday()) % 7
                            friday_date = target_date + timedelta(days=days_until_friday)
                            days_to_expire = (friday_date - today_date).days
                    else:
                        tw = live_settings.get('target_weeks', 6)
                        target_date = today_date + timedelta(weeks=tw)
                        days_until_friday = (4 - target_date.weekday()) % 7
                        friday_date = target_date + timedelta(days=days_until_friday)
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
                        requested_side = MANUAL_TRADE_PARAMS.get('side', 'BOTH')
                        if requested_side == 'CALL':
                            hedge_ratio_put = 0
                        elif requested_side == 'PUT':
                            hedge_ratio_call = 0
                        
                        # Use provided quantity if available
                        if 'qty' in MANUAL_TRADE_PARAMS:
                            parameter['qty'] = int(MANUAL_TRADE_PARAMS['qty'])
        
                    target_delta_val = live_settings.get('target_delta', 0.15)
                    hedge_spread_val = live_settings.get('hedge_spread', 20)
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
                                    'ticker': ticker
                                }
                                print(f"   ✅ Manual spread candidate created: {sell_k}/{buy_k} at net credit ${profit}")
                                
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
                                
                                # Clear the manual trade flag so it doesn't run again next iteration
                                MANUAL_TRADE_REQUESTED.clear()
                                is_manual_trade = False
                                MANUAL_TRADE_PARAMS = {}
                            else:
                                print(f"⚠️ [Manual Trade] Could not retrieve quotes for both legs. Aborting manual execution.")
                                # Clear the manual trade flag so it doesn't run again next iteration
                                MANUAL_TRADE_REQUESTED.clear()
                                is_manual_trade = False
                                MANUAL_TRADE_PARAMS = {}
                        except Exception as e:
                            print(f"⚠️ [Manual Trade] Error loading specific strikes: {e}. Falling back to search.")
                            traceback.print_exc()

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
        
                    if spread_options_call_candidate is not None and spread_options_call_candidate['profit'] > 0.01:
                        call_spread_orders = accounts.generate_option_order(
                            single_leg_stock_position=single_leg_stock_position_call,
                            action=call_action,
                            custom_order_id=None,
                            spread_sell_option=spread_options_call_candidate['sell_option'],
                            spread_buy_option=spread_options_call_candidate['buy_option'],
                            priceType={"priceType": "NET_CREDIT", "limitPrice": round(spread_options_call_candidate['profit'], 2)}
                        )
        
                    if spread_options_put_candidate is not None and spread_options_put_candidate['profit'] > 0.01:
                        put_spread_orders = accounts.generate_option_order(
                            single_leg_stock_position=single_leg_stock_position_put,
                            action=put_action,
                            custom_order_id=None,
                            spread_sell_option=spread_options_put_candidate['sell_option'],
                            spread_buy_option=spread_options_put_candidate['buy_option'],
                            priceType={"priceType": "NET_CREDIT", "limitPrice": round(spread_options_put_candidate['profit'], 2)}
                        )
        
                    if call_spread_orders is not None and SKIP_CALL_FLAG is False:
                        for call_spread_order in call_spread_orders:
                            if call_spread_order is None:
                                continue
                            preview_orders.append({
                                "ticker": ticker,
                                "type": "Call spread" if parameter.get('is_cover_call') is None else "Cover call",
                                "order": call_spread_order,
                                "spread_data": spread_options_call_candidate,
                                "quantity": parameter['qty']
                            })
                            call_order_id = etrade_instance.order.place_order(call_spread_order, preview_only=PREVIEW_ONLY)
                            if call_order_id is not None and PREVIEW_ONLY == False:
                                print("Place order OK, order ID: ", call_order_id)
                            elif call_order_id is None and PREVIEW_ONLY == False:
                                print("Place close order failed: ", call_spread_order)
                    else:
                        print(f"No call_spread_orders for {ticker}")
        
                    if put_spread_orders is not None and SKIP_PUT_FLAG is False:
                        for put_spread_order in put_spread_orders:
                            if put_spread_order is None:
                                continue
                            preview_orders.append({
                                "ticker": ticker,
                                "type": "Put spread",
                                "order": put_spread_order,
                                "spread_data": spread_options_put_candidate,
                                "quantity": parameter['qty']
                            })
                            put_order_id = etrade_instance.order.place_order(put_spread_order, preview_only=PREVIEW_ONLY)
                            if put_order_id is not None and PREVIEW_ONLY == False:
                                print("Place order OK, order ID: ", put_order_id)
                            elif put_order_id is None and PREVIEW_ONLY == False:
                                print("Place open order failed: ", put_spread_order)
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
                        prob_func, regime_name, _ = get_probability_engine(spot_price, vix_value, regime_dict, horizon=7, hmm_model=best_hmm)
                        # print(f"Using {regime_name} regime for EV calculation")
                        
                        metrics = calculate_yield_metrics(
                            order['spread_data']['sell_option'].strike_price,
                            order['spread_data']['buy_option'].strike_price,
                            order['spread_data']['profit'],
                            prob_func
                        )
                        order['ev_data'] = metrics
                    except Exception as ev_e:
                        print(f"⚠️ EV Calculation Error for {order['ticker']}: {ev_e}")
                        order['ev_data'] = {}
        
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
        
                if len(preview_orders) > 0 and execute_open_now:
                    # ── Auto-approval logic ──
                    # If auto_open_enabled=True or is_manual_trade=True, we proceed to execute.
                    # Manual refresh alone will NOT reach this block.
                    print("⏩ Proceeding with Approval...")
                    # Determine reason for audit
                    open_reason = "Manual Dashboard Open" if is_manual_open else "Automatic System Open"
                    log_dashboard_request("EXECUTION_START", {
                        "is_manual": is_manual_open,
                        "auto_open": auto_open_enabled,
                        "reason": open_reason,
                        "valid_orders": [o['ticker'] for o in preview_orders]
                    })

                    print("✅ Executing orders...")
                    any_executed = False
                    for order_info in preview_orders:
                        PREVIEW_ONLY = False

                        # Re-fetch latest price after approval delay
                        spread_data = order_info.get('spread_data', {})
                        sell_opt = spread_data.get('sell_option')
                        buy_opt = spread_data.get('buy_option')
                        if sell_opt and buy_opt:
                            is_valid, refreshed_price = refresh_spread_limit_price(
                                market, sell_opt.symbol,
                                sell_opt.strike_price, buy_opt.strike_price,
                                sell_opt.call_put, sell_opt.expiration_date,
                                order_info['order'], is_credit=True
                            )
                            if not is_valid:
                                print(f"   ⏭️ Skipping {order_info['ticker']} {order_info['type']} — no longer profitable after price refresh.")
                                continue

                        order_id = etrade_instance.order.place_order(order_info['order'], preview_only=PREVIEW_ONLY)
                        while order_id == "INSUFFICIENT_FUNDS":
                            all_positions = accounts.portfolio()
                            positions_processed = release_margin(
                                all_positions=all_positions,
                                cover_call_list=None,
                                etrade_instance=etrade_instance,
                                max_positions=5
                            )
                            t.sleep(5)
                            accounts.balance()
                            order_id = etrade_instance.order.place_order(order_info['order'], preview_only=PREVIEW_ONLY)
        
                        if order_id is not None:
                            print(f"Placed {order_info['ticker']} {order_info['type']} order - ID: {order_id}")
                            try:
                                print(f"Monitoring order {order_id} for execution and adjusting price by $0.01 every 30s if still open...")
                                executed, final_order_id = etrade_instance.order.wait_and_adjust_until_filled(order_id, step=0.01, interval_sec=30, max_checks=180)
                                if executed:
                                    accounts.record_option_target(order_info['spread_data'], order_id=final_order_id, order_status="EXECUTED")
                                    any_executed = True
                                    
                                    # Permanent Logging & Notification
                                    order_audit = {
                                        'ticker': order_info['ticker'],
                                        'sell_strike': order_info['spread_data']['sell_option'].strike_price,
                                        'long_strike': order_info['spread_data']['buy_option'].strike_price,
                                        'qty': order_info['quantity'],
                                        'order_id': final_order_id,
                                        'is_close': False
                                    }
                                    log_order_execution(order_audit, open_reason, "FILLED")
                                    send_trade_notification_email(order_audit, open_reason)
                            except Exception as e:
                                print(f"Auto-adjust loop for order {order_id} encountered an error: {e}")
                        else:
                            print(f"Failed to place {order_info['ticker']} {order_info['type']} order")
                        t.sleep(5)
                    print("All orders have been processed")
                    # Mark trade as executed for today only if at least one order is executed
                    if any_executed:
                        save_trade_status(current_date)
                        last_trade_date = current_date
                        trade_executed_flag = True
                else:
                    if not execute_now:
                        print("Search complete. No orders executed (Refresh mode).")
                    else:
                        print("No valid orders found to execute.")

            # --- POSITION DETECTION & MANAGEMENT SECTION ---
            # Now refresh portfolio state to detect high-gain spreads or needed rolls
            print("\n--- Running Position Detection & Management ---")
            all_positions = accounts.portfolio(print_enable=False)
            screened = accounts.screen_option(all_positions)
            html_path = accounts.render_screened_option_pairs_html(screened, out_path="screened_option_pairs.html", order_instance=etrade_instance.order, show_refresh=False)
            accounts.option_value_final(all_positions)

            # Propose closing high-gain spreads automatically (email approval)
            close_proposals = []   # collect all close-worthy spreads first
            for entry in screened:
                if entry.get("is_spread") and entry.get("short_lot") and entry.get("long_lot"):
                    short_lot = entry["short_lot"]
                    long_lot = entry["long_lot"]
                    
                    try:
                        # Construct OSI keys for quotes
                        short_osi = f"{short_lot.symbol}:{short_lot.expiration_date.year}:{short_lot.expiration_date.month:02}:{short_lot.expiration_date.day:02}:{short_lot.call_put}:{short_lot.strike_price}"
                        long_osi = f"{long_lot.symbol}:{long_lot.expiration_date.year}:{long_lot.expiration_date.month:02}:{long_lot.expiration_date.day:02}:{long_lot.call_put}:{long_lot.strike_price}"
                        
                        # Check if already rejected today to avoid redundant quotes/emails
                        proposal_id = f"{short_lot.symbol}_{short_lot.strike_price}_{short_lot.call_put}_{short_lot.expiration_date}"
                        if proposal_id in rejected_proposals_today:
                            continue

                        print(f"   Checking quote for {short_lot.symbol} close candidate...")
                        resp = market.get_quote([short_osi, long_osi], resp_format="json")
                        quote_data = resp.get("QuoteResponse", {}).get("QuoteData", [])
                        
                        quotes_dict = {}
                        for q in quote_data:
                            p = q.get("Product", {})
                            ticker_resp = p.get("symbol")
                            quotes_dict[ticker_resp] = q.get("All", {})
                        
                        short_q = quotes_dict.get(short_osi)
                        long_q = quotes_dict.get(long_osi)
                        
                        # Fallback matching
                        if not short_q or not long_q:
                            for q in quote_data:
                                all_q = q.get("All", {})
                                prod = q.get("Product", {})
                                if abs(float(prod.get("strikePrice", 0)) - float(short_lot.strike_price)) < 0.01 and prod.get("callPut") == short_lot.call_put:
                                    short_q = all_q
                                if abs(float(prod.get("strikePrice", 0)) - float(long_lot.strike_price)) < 0.01 and prod.get("callPut") == long_lot.call_put:
                                    long_q = all_q

                        if short_q and long_q:
                            bid_s = float(short_q.get("bid", 0))
                            ask_s = float(short_q.get("ask", 0))
                            bid_l = float(long_q.get("bid", 0))
                            ask_l = float(long_q.get("ask", 0))
                            
                            mid_s = (bid_s + ask_s) / 2
                            mid_l = (bid_l + ask_l) / 2
                            midpoint = mid_s - mid_l
                            
                            # NEW Threshold: Check closing cost midpoint AND profit percentage
                            close_mid_threshold = live_settings.get('auto_close_midpoint_threshold', 0.30)
                            target_gain_threshold = live_settings.get('auto_close_gain_threshold', 70.0)
                            
                            # Both conditions must be met:
                            # 1. Gain must be >= target_gain (e.g. 70%)
                            # 2. Closing cost must be <= threshold (e.g. $0.30)
                            gain_val = entry.get('pair_gain_loss', 0)
                            gain_met = gain_val >= target_gain_threshold
                            cost_met = abs(midpoint) <= close_mid_threshold
                            
                            if gain_met and cost_met:
                                print(f"\n🚀 DETECTION: Profit Target Met in {short_lot.symbol}")
                                print(f"   Gain: {gain_val:.1f}% >= {target_gain_threshold}%")
                                print(f"   Cost: ${abs(midpoint):.2f} <= ${close_mid_threshold:.2f}")
                                print(f"   Pair Qty: {entry['pair_quantity']}")
                                
                                # Build the close order ahead of time
                                from copy import copy
                                s_lot = copy(short_lot)
                                l_lot = copy(long_lot)
                                s_lot.quantity = entry['pair_quantity']
                                l_lot.quantity = entry['pair_quantity']
                                
                                close_order = accounts.generate_option_order(
                                    single_leg_stock_position=None,
                                    action="SPREAD",
                                    spread_sell_option=l_lot,
                                    spread_buy_option=s_lot,
                                    priceType={'priceType': 'NET_DEBIT', 'limitPrice': abs(midpoint)}
                                )
                                
                                expiry_str = short_lot.expiration_date.strftime('%Y-%m-%d') if hasattr(short_lot.expiration_date, 'strftime') else str(short_lot.expiration_date)
                                close_proposals.append({
                                    'ticker': short_lot.symbol,
                                    'short_strike': short_lot.strike_price,
                                    'long_strike': long_lot.strike_price,
                                    'call_put': short_lot.call_put,
                                    'expiration': expiry_str,
                                    'pair_quantity': entry['pair_quantity'],
                                    'gain_pct': short_lot.gain_loss_percentage if hasattr(short_lot, 'gain_loss_percentage') else entry.get('pair_gain_loss', 0),
                                    'midpoint': midpoint,
                                    'short_bid': bid_s, 'short_ask': ask_s,
                                    'long_bid': bid_l, 'long_ask': ask_l,
                                    'close_order': close_order,
                                    'proposal_id': proposal_id
                                })
                        else:
                            # print(f"   Warning: Could not fetch quotes for legs of {short_lot.symbol} spread.")
                            pass
                    except Exception as e:
                        print(f"   Error in auto-close logic for {short_lot.symbol}: {e}")
                        import traceback
                        traceback.print_exc()

            # Update global list for dashboard
            CURRENT_CLOSE_PROPOSALS = close_proposals

            # --- Auto-approval (Guarded by master toggle OR manual override) ---
            if close_proposals:
                if auto_open_enabled or is_manual_close:
                    close_reason = "Manual Dashboard Close" if is_manual_close else "Automatic System Close"
                    print(f"✅ Close-spread proposals approved ({close_reason}). Executing...")
                    for cp in close_proposals:
                        close_order = cp['close_order']
                        if isinstance(close_order, list): 
                            close_order = close_order[0]

                        # Re-fetch latest price after approval delay
                        _, refreshed_debit = refresh_spread_limit_price(
                            market, cp['ticker'],
                            cp['short_strike'], cp['long_strike'],
                            cp['call_put'], cp['expiration'],
                            close_order, is_credit=False
                        )

                        limit_price_val = close_order.get('limitPrice', 'N/A')
                        print(f"   Submitting close order for {cp['ticker']} {cp['call_put']} {cp['short_strike']}/{cp['long_strike']} @ ${limit_price_val}...")
                        order_id = etrade_instance.order.place_order(close_order, preview_only=False)
                        if order_id:
                            print(f"   Order submitted! ID: {order_id}")
                            try:
                                executed, final_id = etrade_instance.order.wait_and_adjust_until_filled(
                                    order_id, step=0.01, interval_sec=30, max_checks=60
                                )
                                if executed:
                                    print(f"   ✅ Close order filled (ID: {final_id})")
                                    try:
                                        record_closed_spy_gain(cp)
                                        
                                        # Permanent Logging & Notification
                                        order_audit = {
                                            'ticker': cp['ticker'],
                                            'sell_strike': cp['short_strike'],
                                            'long_strike': cp['long_strike'],
                                            'qty': cp.get('pair_quantity', 1),
                                            'order_id': final_id,
                                            'is_close': True
                                        }
                                        log_order_execution(order_audit, close_reason, "FILLED")
                                        send_trade_notification_email(order_audit, close_reason)
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"   Auto-adjust for close order {order_id} error: {e}")
                        else:
                            print(f"   ❌ Close order submission failed for {cp['ticker']}")
                        t.sleep(3)
                    # Clear flags
                    MANUAL_TRADE_REQUESTED.clear()
                    is_manual_close = False
                else:
                    print(f"📝 {len(close_proposals)} Close-spread proposals detected. Waiting for manual approval on dashboard (Automatic Open is OFF).")
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
                        # call_order_id = etrade_instance.order.place_order(order, preview_only=PREVIEW_ONLY)
                    # breakpoint()

            # --- STALE ORDER MONITORING SECTION ---
            if (datetime.now() - last_stale_check_time).total_seconds() >= 120:
                print("\n--- Monitoring Stale Orders ---")
                monitor_and_nudge_stale_orders(etrade_instance, market, dry_run=False)
                last_stale_check_time = datetime.now()

            # End of detection loop, wait before next refresh if trade already executed
            if last_trade_date == current_date:
                # Use shorter sleep intervals to allow manual refresh interruption
                for _ in range(300):  # 300 x 1 second = 5 minutes max
                    if REFRESH_REQUESTED.is_set() or MANUAL_TRADE_REQUESTED.is_set():
                        print("🔄 [Refresh Server] Processing manual request...")
                        break
                    
                    if (datetime.now() - last_stale_check_time).total_seconds() >= 120:
                        print("\n--- Monitoring Stale Orders (Sleep) ---")
                        monitor_and_nudge_stale_orders(etrade_instance, market, dry_run=False)
                        last_stale_check_time = datetime.now()

                    t.sleep(1)
            else:
                # Main Loop Tick: Every 60 seconds (when trade not yet executed)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tick complete. Sleeping 60s...")
                for _ in range(60):
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
