from __future__ import annotations
"""Synchronous Polygon option quote helpers (v3 endpoints).

This module provides a small fallback/synchronous path for fetching a
single option's quote from Polygon and returning ask/bid/mid fields,
mirroring the behavior that existed in the original monolith.
"""
from typing import Any, Dict, List, Union
import logging

import requests

from .config import get_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------
# Symbol formatting
# ---------------------------------------------------------

def format_polygon_option_symbol(underlying: str, strike_price: float, call_put: str, expiration_date: str) -> str:
    """Format a Polygon option symbol.

    Pattern: ``O:{UNDERLYING}{YYMMDD}{C/P}{STRIKE*1000:08d}``
    Example: ``O:AAPL241220C00175000`` for AAPL 2024-12-20 175C.
    """
    exp_fmt = expiration_date[2:].replace("-", "")  # YYMMDD
    strike_fmt = f"{int(round(strike_price * 1000)):08d}"
    cp = "C" if call_put.lower() == "call" else "P"
    return f"O:{underlying}{exp_fmt}{cp}{strike_fmt}"


# ---------------------------------------------------------
# Quote lookup (sync)
# ---------------------------------------------------------

def get_option_quote(
    underlying_ticker: str,
    strike_price: float,
    call_put: str,
    expiration_date: str,
    quote_timestamp: str,
) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """Fetch latest quote for a single option from Polygon v3/quotes.

    Returns
    -------
    list of dict
        Each dict contains ``ask_price``, ``bid_price``, ``ask_size``, ``bid_size``, ``mid_price``.
        (We typically use the first element.)
    dict
        An error dict of the form ``{"error": message}`` on failure.
    """
    option_symbol = format_polygon_option_symbol(underlying_ticker, strike_price, call_put, expiration_date)
    url = f"https://api.polygon.io/v3/quotes/{option_symbol}"
    params = {
        "timestamp": quote_timestamp,
        "order": "desc",
        "sort": "timestamp",
        "limit": 1,
        "apiKey": get_settings().polygon_api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return {"error": "No valid data"}
        out: List[Dict[str, Any]] = []
        for r in results:
            ask = r.get("ask_price", 0.0)
            bid = r.get("bid_price", 0.0)
            if ask <= 0 or bid <= 0:
                continue
            out.append(
                {
                    "ask_price": ask,
                    "bid_price": bid,
                    "ask_size": r.get("ask_size", 0),
                    "bid_size": r.get("bid_size", 0),
                    "mid_price": round((ask + bid) / 2.0, 3),
                }
            )
        return out or {"error": "No valid bid/ask"}
    except requests.RequestException as e:
        log.error("Polygon quote request failed: %s", e)
        return {"error": str(e)}


def query_polygon_for_option_price(
    ticker: str,
    strike_price: float,
    call_put: str,
    expiration_date: str,
    pricing_date: str,
) -> Dict[str, Any]:
    """Convenience wrapper → single dict with ask/bid/mid, or empty dict if invalid."""
    data = get_option_quote(ticker, strike_price, call_put, expiration_date, pricing_date)
    if isinstance(data, dict) and "error" in data:
        return {}
    if not data:
        return {}
    return data[0]

