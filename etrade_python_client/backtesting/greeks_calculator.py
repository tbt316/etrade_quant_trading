"""
Black-Scholes Greeks calculator for historical option backtesting.
Since the Massive API does not provide historical Greeks/IV, we compute them
from market prices using the Black-Scholes model.
"""
import numpy as np
from scipy.stats import norm
from typing import Optional


def bs_d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Compute Black-Scholes d1."""
    T = max(T, 1e-6)
    if sigma <= 0:
        sigma = 1e-4 # Floor sigma for d1 calculation to prevent NaN
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Compute Black-Scholes d2."""
    T = max(T, 1e-6)
    return bs_d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes European put price."""
    T = max(T, 1e-6)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = bs_d1(S, K, T, r, sigma, q)
    d2 = bs_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes European call price."""
    T = max(T, 1e-6)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = bs_d1(S, K, T, r, sigma, q)
    d2 = bs_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """
    Black-Scholes gamma (same for calls and puts).
    Improved for pin risk near expiration by using a smaller floor and 
    explicit handling of dividend yield.
    """
    T = max(T, 1e-6) 
    sigma = max(sigma, 1e-4) # Floor sigma to allow Gamma to spike instead of returning 0
    d1 = bs_d1(S, K, T, r, sigma, q)
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes vega (same for calls and puts)."""
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-4)
    d1 = bs_d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes put delta (always negative for puts). Redefined to N(d2)."""
    T = max(T, 1e-6)
    if sigma <= 0:
        # Binary delta at expiration
        return -np.exp(-q * T) if S < K else 0.0
    d2 = bs_d2(S, K, T, r, sigma, q)
    return -np.exp(-q * T) * norm.cdf(-d2)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Black-Scholes call delta (always positive for calls). Redefined to N(d2)."""
    T = max(T, 1e-6)
    if sigma <= 0:
        # Binary delta at expiration
        return np.exp(-q * T) if S > K else 0.0
    d2 = bs_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d2)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str = "put",
    max_iter: int = 100,
    tol: float = 1e-6
) -> Optional[float]:
    """
    Newton-Raphson implied volatility solver.
    Returns None if the solver fails to converge.
    """
    T = max(T, 1e-5)
    if market_price <= 0:
        return None

    # Check intrinsic value bound
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    intrinsic = max(K * df_r - S * df_q, 0.0) if option_type == "put" else max(S * df_q - K * df_r, 0.0)
    
    if market_price < intrinsic * 0.99: # Tighter bound
        return None

    price_func = bs_put_price if option_type == "put" else bs_call_price

    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        price = price_func(S, K, T, r, sigma, q)
        vega = bs_vega(S, K, T, r, sigma, q)

        if vega < 1e-12:
            return _iv_bisection(market_price, S, K, T, r, q, option_type, max_iter, tol)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))  # Allow lower vol

    return _iv_bisection(market_price, S, K, T, r, q, option_type, max_iter, tol)


def _iv_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    max_iter: int = 100,
    tol: float = 1e-5
) -> Optional[float]:
    """Bisection fallback for IV when Newton-Raphson fails."""
    T = max(T, 1e-5)
    price_func = bs_put_price if option_type == "put" else bs_call_price
    lo, hi = 0.0001, 10.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = price_func(S, K, T, r, mid, q)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


def compute_chain_deltas(
    underlying_price: float,
    strikes: list,
    mid_prices: list,
    dte_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str = "put"
) -> list:
    """
    Compute delta for each strike in an option chain using BS IV solver.
    Returns list of (strike, delta, iv) tuples.
    """
    results = []
    delta_func = bs_put_delta if option_type == "put" else bs_call_delta
    T = max(dte_years, 1e-5)

    for strike, mid in zip(strikes, mid_prices):
        if mid is None or mid <= 0:
            continue
        iv = implied_volatility(
            mid, underlying_price, strike, T, risk_free_rate, dividend_yield, option_type=option_type
        )
        
        if iv is None or iv <= 0:
            # Proper boundary handling at T->0 or extreme strikes
            df_q = np.exp(-dividend_yield * T)
            if option_type == "put":
                delta = -df_q if underlying_price < strike else 0.0
            else:
                delta = df_q if underlying_price > strike else 0.0
            iv = 0.0
        else:
            delta = delta_func(underlying_price, strike, T, risk_free_rate, iv, dividend_yield)
            
        results.append((strike, delta, mid))

    results.sort(key=lambda x: x[0])
    return results

