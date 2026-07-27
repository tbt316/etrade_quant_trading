#!/usr/bin/env python3
"""
Quarantined legacy E*TRADE options prototype.

This module is retained only for compatibility and offline code archaeology.
It is not production-grade, and its authentication and broker-execution entry
points fail closed. Broker mutations must use the durable E*TRADE order gateway.
"""

import argparse
import configparser
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl

import numpy as np
import pandas as pd
import yfinance as yf
from rauth import OAuth1Service

# Custom modules (assume these are refactored into separate files in a package)
from accounts.accounts_bo import Accounts, StockPosition, calculate_margin, find_highest_margin_ratios
from market.market_bo import Market
from data_and_research.option_assign_probability import calculate_probability
from backtesting.polygonio_dailytrade import fetch_yfinance_data
from data_and_research.polygonio_improvequery import get_earnings_dates
from backtesting.polygon_multi import load_latest_parameters
from live_trading.runtime_safety import (
    RuntimeSafetyError,
    configure_owner_only_logger,
    read_owner_only_json,
    reject_legacy_execution,
    write_owner_only_json,
)

# Constants
ETRADE_OAUTH_FILE = ".etrade_oauth"

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

OAUTH_KEYS = {
    "sandbox": {
        "consumer_key": os.getenv("ETRADE_SANDBOX_CONSUMER_KEY") or config.get('DEFAULT', 'SANDBOX_CONSUMER_KEY', fallback=None),
        "consumer_secret": os.getenv("ETRADE_SANDBOX_CONSUMER_SECRET") or config.get('DEFAULT', 'SANDBOX_CONSUMER_SECRET', fallback=None),
    },
    "live": {
        "consumer_key": os.getenv("ETRADE_LIVE_CONSUMER_KEY") or config.get('DEFAULT', 'PROD_CONSUMER_KEY', fallback=None),
        "consumer_secret": os.getenv("ETRADE_LIVE_CONSUMER_SECRET") or config.get('DEFAULT', 'PROD_CONSUMER_SECRET', fallback=None),
    }
}

# Setup logging
logger = configure_owner_only_logger('etrade_trader', "etrade_trader.log")

def environment_key(use_sandbox: bool) -> str:
    """Determine the environment key based on sandbox flag."""
    return "sandbox" if use_sandbox else "live"

def load_oauth_tokens(use_sandbox: bool) -> Optional[Dict[str, str]]:
    """Load cached OAuth tokens from file."""
    try:
        tokens = read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache")
        return tokens.get(environment_key(use_sandbox))
    except (RuntimeSafetyError, KeyError, TypeError) as e:
        logger.warning(f"Failed to load OAuth tokens: {e}")
        return None

def save_oauth_tokens(tokens: Dict[str, str], use_sandbox: bool) -> None:
    """Save OAuth tokens to file securely."""
    try:
        existing_tokens = (
            read_owner_only_json(ETRADE_OAUTH_FILE, label="OAuth cache")
            if os.path.lexists(ETRADE_OAUTH_FILE)
            else {}
        )
        existing_tokens[environment_key(use_sandbox)] = tokens
        write_owner_only_json(ETRADE_OAUTH_FILE, existing_tokens)
        logger.info("OAuth tokens saved successfully.")
    except (RuntimeSafetyError, KeyError, TypeError) as e:
        logger.error(f"Failed to save OAuth tokens: {e}")
        raise RuntimeSafetyError("could not persist OAuth cache safely") from e

def get_etrade_session(use_sandbox: bool, auto_login: bool = True, username: Optional[str] = None, password: Optional[str] = None) -> Tuple[Optional[OAuth1Service], str]:
    """Authenticate and get E*TRADE OAuth session with renewal support."""
    raise RuntimeSafetyError(
        "etrade_put_credit_spread is quarantined; use etrade_cover_call_new "
        "with an explicit RuntimeSafetyBoundary"
    )
    keys = OAUTH_KEYS[environment_key(use_sandbox)]
    base_url = config.get('DEFAULT', 'SANDBOX_BASE_URL') if use_sandbox else config.get('DEFAULT', 'PROD_BASE_URL')

    etrade = OAuth1Service(
        name="etrade",
        consumer_key=keys["consumer_key"],
        consumer_secret=keys["consumer_secret"],
        request_token_url=f"{base_url}/oauth/request_token",
        access_token_url=f"{base_url}/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url=base_url
    )

    tokens = load_oauth_tokens(use_sandbox)
    if tokens:
        session = etrade.get_session((tokens['oauth_token'], tokens['oauth_token_secret']))
        renew_url = f"{base_url}/oauth/renew_access_token"
        response = session.get(renew_url)
        if response.status_code == 200:
            logger.info("Session renewed successfully.")
            return session, base_url
        logger.warning("Session renewal failed. Starting new authentication.")

    # New authentication flow
    def utf8_decoder(content: bytes) -> Dict[str, str]:
        return dict(parse_qsl(content.decode('utf-8', errors='replace')))

    try:
        request_token, request_token_secret = etrade.get_request_token(
            params={"oauth_callback": "oob", "format": "json"},
            decoder=utf8_decoder
        )
        authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)

        if auto_login and username and password:
            # Implement automated login logic (e.g., using Selenium if needed)
            # For security, avoid plain password handling; use secure input.
            logger.warning("Automated login not implemented securely. Use manual verification.")
            text_code = input("Enter verification code from browser: ")
        else:
            import webbrowser
            webbrowser.open(authorize_url)
            text_code = input("Enter verification code from browser: ")

        session = etrade.get_auth_session(
            request_token,
            request_token_secret,
            params={"oauth_verifier": text_code}
        )

        new_tokens = {
            'oauth_token': session.access_token,
            'oauth_token_secret': session.access_token_secret
        }
        save_oauth_tokens(new_tokens, use_sandbox)
        logger.info("New session created and tokens saved.")
        return session, base_url
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None, base_url

def release_margin(
    accounts: Accounts,
    etrade_instance,
    all_positions: List[StockPosition],
    cover_call_list: Optional[Dict[str, int]] = None,
    max_positions: int = 5
) -> int:
    """Release margin by closing high-margin positions."""
    reject_legacy_execution("live_trading.etrade_put_credit_spread.release_margin")

def main():
    parser = argparse.ArgumentParser(description="E*TRADE Options Trading Application")
    parser.error("etrade_put_credit_spread is quarantined; use etrade_cover_call_new with explicit runtime safety")
    parser.add_argument('--sandbox', action='store_true', help='Use sandbox environment')
    parser.add_argument('--trade', action='store_true', help='Enable live trading')
    parser.add_argument('--use_existing_file', action='store_true', help='Reuse backtest results')
    parser.add_argument('--username', required=True, help='Username for login')
    parser.add_argument('--password', required=True, help='Password for login')
    args = parser.parse_args()

    session, base_url = get_etrade_session(args.sandbox, auto_login=True, username=args.username, password=args.password)
    if not session:
        logger.error("Failed to obtain E*TRADE session. Exiting.")
        sys.exit(1)

    accounts = Accounts(session, base_url)
    # market = Market(session, base_url)  # Uncomment if needed

    # Define trading windows
    trade_start = datetime.strptime('06:30:00', '%H:%M:%S').time()
    trade_end = datetime.strptime('13:30:00', '%H:%M:%S').time()

    last_renewal = datetime.now()
    while True:
        now = datetime.now()
        if (now - last_renewal) >= timedelta(minutes=1):
            session, base_url = get_etrade_session(args.sandbox)
            last_renewal = now
            accounts = Accounts(session, base_url)

        if args.trade and trade_start <= now.time() <= trade_end:
            try:
                # Core trading logic here (refactored from original while loop)
                all_positions = accounts.portfolio(print_enable=True)
                accounts.screen_option(all_positions)
                accounts.option_value_final(all_positions)

                # Example: Process positions for neutralization or spreads
                # Add your trading strategy calls here

                # Margin release example
                # release_margin(accounts, etrade_instance, all_positions)

                time.sleep(60)  # Throttle loop
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(300)  # Backoff on error
        else:
            time.sleep(300)  # Sleep outside trading window

if __name__ == "__main__":
    main()
