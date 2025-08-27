from __future__ import annotations
"""Option math utilities: Black–Scholes pricing, Vega, and implied volatility.

This module mirrors the behavior of the original script while organizing
functions cleanly for reuse.
"""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.optimize import newton
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSInputs:
    S: float  # underlying spot
    K: float  # strike
    T: float  # time to expiry in years
    r: float  # risk-free rate (annualized, cont comp)
    q: float  # dividend yield (annualized, cont comp)
    sigma: float  # volatility (annualized)
    option_type: OptionType = "call"


# ------------------------------
# Black–Scholes core functions
# ------------------------------

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * np.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType = "call") -> float:
    """Black–Scholes price for a European call/put with continuous dividend yield."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Derivative of price w.r.t volatility (per 1.00 of sigma, not 1%)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# ------------------------------
# Implied volatility via Newton's method
# ------------------------------

def calculate_implied_volatility(
    close_price: float,
    strike_price: float,
    option_price: float,
    days_to_expire: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType = "call",
) -> Optional[float]:
    """Solve for annualized implied volatility using Newton–Raphson.

    Parameters
    ----------
    close_price : float
        Current underlying price S.
    strike_price : float
        Option strike K.
    option_price : float
        Observed market option price.
    days_to_expire : float
        Days until expiration.
    risk_free_rate : float
        Annualized risk-free rate, e.g. 0.05 for 5%.
    dividend_yield : float
        Annualized continuous dividend yield, e.g. 0.02 for 2%.
    option_type : {"call", "put"}
        Option side.

    Returns
    -------
    float | None
        Implied volatility in decimal (e.g., 0.2) or None if it cannot be found
        or inputs are invalid. (Matches legacy behavior.)
    """
    # Convert days to years
    T = days_to_expire / 365.0

    # Basic input validation — intentionally conservative to mirror legacy checks
    if (
        close_price <= 0
        or strike_price <= 0
        or option_price <= 0
        or T <= 0
        or option_price >= close_price  # original script's guardrail
    ):
        return None

    # Objective function f(sigma) = model_price - market_price
    def f(sigma: float) -> float:
        return bs_price(close_price, strike_price, T, risk_free_rate, dividend_yield, sigma, option_type) - option_price

    # Vega for Newton step
    def fprime(sigma: float) -> float:
        return vega(close_price, strike_price, T, risk_free_rate, dividend_yield, sigma)

    try:
        iv = float(newton(f, x0=0.2, fprime=fprime, maxiter=100, tol=1e-6))
        # Sanity bounds (legacy-style): positive and capped at 500%
        if iv <= 0 or iv > 5.0:
            return None
        return iv
    except Exception:
        return None


__all__ = [
    "BSInputs",
    "bs_price",
    "vega",
    "calculate_implied_volatility",
]

