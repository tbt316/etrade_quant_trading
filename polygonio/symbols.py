from __future__ import annotations
from typing import Literal

# ---------------------------------------------------------
# Ticker normalization helpers
# ---------------------------------------------------------

# Known quirks between vendors
# - Polygon may use dot-class tickers (BRK.B), E*TRADE prefers dash (BRK-B)
# - Yahoo uses ^-prefixed indices (e.g., ^VIX). Polygon uses VIX/underlying references.
# - Add as we discover more.

_POLY_TO_ETRADE: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "SPX": "^GSPC",  # often used as index proxy when fetching spot from Yahoo
    "VIX": "^VIX",    # map to Yahoo-style for analytics; E*TRADE index symbols may differ
    "VIXW": "^VIX",
}

_YF_CANONICAL: dict[str, str] = {
    # Yahoo expects dashes for .B classes and carets for indices
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "VIXW": "^VIX",
}


def convert_polygon_to_etrade_ticker(polygon_ticker: str) -> str:
    """Backwards-compatible shim for existing code paths.
    Returns the corrected symbol for E*TRADE when Polygon provides a variant.
    """
    return _POLY_TO_ETRADE.get(polygon_ticker, polygon_ticker)


def to_vendor_ticker(ticker: str, vendor: Literal["yfinance", "etrade", "polygon"] = "yfinance") -> str:
    """Normalize *ticker* for a specific vendor API.

    Parameters
    ----------
    ticker : str
        Input ticker (may be Polygon-style or raw).
    vendor : {"yfinance", "etrade", "polygon"}
        Target data source / brokerage symbol style.

    Returns
    -------
    str
        Vendor-appropriate symbol string.
    """
    t = ticker.strip()

    if vendor == "yfinance":
        return _YF_CANONICAL.get(t, t)

    if vendor == "etrade":
        # Prefer the same mapping we used historically for E*TRADE
        return _POLY_TO_ETRADE.get(t, t)

    if vendor == "polygon":
        # Polygon generally accepts raw tickers; return as-is
        return t

    # Fallback: no change
    return t

