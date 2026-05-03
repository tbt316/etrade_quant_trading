#!/usr/bin/env python3
"""
Portfolio Manager Module

Provides functions to fetch and extract current holdings from an authenticated
E*Trade session. This module serves as the first stage of the Analyst Report Agent.
"""

from typing import List, Optional
from accounts.accounts_bo import Accounts, StockPosition


def get_current_holdings(accounts: Accounts, include_options: bool = True) -> List[str]:
    """
    Fetch current stock positions and return unique ticker symbols.
    
    Uses the existing authenticated Accounts session to call portfolio()
    and extracts unique underlying ticker symbols.
    
    Args:
        accounts: An authenticated Accounts instance with valid session.
        include_options: If True, includes underlying symbols from option positions.
                        Defaults to True.
    
    Returns:
        List of unique ticker symbols (e.g., ['AAPL', 'MSFT', 'NVDA'])
    
    Raises:
        RuntimeError: If the accounts session is invalid or API call fails.
    
    Example:
        >>> accounts = Accounts(session, base_url)
        >>> tickers = get_current_holdings(accounts)
        >>> print(tickers)  # ['AAPL', 'GOOGL', 'MSFT', ...]
    """
    if accounts is None or accounts.session is None:
        raise RuntimeError("Invalid accounts session. Please authenticate first.")
    
    # Fetch all positions using minimal mode for faster response
    try:
        positions = accounts.portfolio(print_enable=False, minimal=True)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch portfolio: {e}")
    
    if positions is None:
        return []
    
    tickers = set()
    
    for position in positions:
        if not isinstance(position, StockPosition):
            continue
            
        symbol = position.symbol
        
        # Skip invalid symbols
        if not symbol or symbol == "N/A":
            continue
        
        # Normalize symbol names
        symbol = _normalize_symbol(symbol)
        
        # Filter by security type
        security_type = getattr(position, 'security_type', None)
        
        if security_type == "EQ" or security_type == "Stock":
            # Equity positions: add directly
            tickers.add(symbol)
        elif security_type == "Option" and include_options:
            # Option positions: add the underlying ticker
            tickers.add(symbol)
    
    # Sort for consistent ordering
    return sorted(list(tickers))


def get_holdings_with_details(accounts: Accounts) -> List[dict]:
    """
    Fetch holdings with additional details useful for analysis.
    
    Returns a list of dictionaries with ticker and position details.
    
    Args:
        accounts: An authenticated Accounts instance.
    
    Returns:
        List of dicts with keys: ticker, quantity, market_value, security_type
    """
    if accounts is None or accounts.session is None:
        raise RuntimeError("Invalid accounts session. Please authenticate first.")
    
    try:
        positions = accounts.portfolio(print_enable=False, minimal=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch portfolio: {e}")
    
    if positions is None:
        return []
    
    holdings = []
    seen_tickers = {}  # Track aggregated equity positions
    
    for position in positions:
        if not isinstance(position, StockPosition):
            continue
        
        symbol = position.symbol
        if not symbol or symbol == "N/A":
            continue
        
        symbol = _normalize_symbol(symbol)
        security_type = getattr(position, 'security_type', None)
        
        # Only include equity positions for detailed analysis
        if security_type == "EQ":
            quantity = getattr(position, 'quantity', 0)
            market_value = getattr(position, 'market_value', 0)
            last_price = getattr(position, 'last_price', 0)
            
            if symbol in seen_tickers:
                # Aggregate multiple lots
                seen_tickers[symbol]['quantity'] += quantity
                seen_tickers[symbol]['market_value'] += market_value
            else:
                seen_tickers[symbol] = {
                    'ticker': symbol,
                    'quantity': quantity,
                    'market_value': market_value,
                    'last_price': last_price,
                    'security_type': 'EQ'
                }
    
    return list(seen_tickers.values())


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize ticker symbols to a consistent format.
    
    Handles E*Trade specific ticker variations.
    
    Args:
        symbol: Raw ticker symbol from E*Trade API.
    
    Returns:
        Normalized ticker symbol.
    """
    if not symbol:
        return symbol
    
    symbol = symbol.upper().strip()
    
    # E*Trade to standard mappings
    symbol_mappings = {
        'BRKB': 'BRK.B',
        'BRK B': 'BRK.B',
        'VIXW': 'VIX',
    }
    
    return symbol_mappings.get(symbol, symbol)


def filter_tickers_for_research(
    tickers: List[str],
    exclude_etfs: bool = False,
    exclude_indices: bool = True
) -> List[str]:
    """
    Filter tickers to those suitable for analyst research.
    
    Some tickers (like VIX, indices) may not have analyst reports available.
    
    Args:
        tickers: List of ticker symbols.
        exclude_etfs: If True, exclude common ETF symbols. Defaults to False.
        exclude_indices: If True, exclude index symbols. Defaults to True.
    
    Returns:
        Filtered list of tickers.
    """
    # Indices and volatility products typically don't have analyst reports
    index_symbols = {'VIX', 'VIXW', 'SPX', 'NDX', 'DJX', 'RUT'}
    
    # Common ETFs (may or may not have reports)
    etf_symbols = {'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'VXX', 'UVXY'}
    
    filtered = []
    for ticker in tickers:
        if exclude_indices and ticker in index_symbols:
            continue
        if exclude_etfs and ticker in etf_symbols:
            continue
        filtered.append(ticker)
    
    return filtered


if __name__ == "__main__":
    # Standalone test (requires authenticated session)
    print("Portfolio Manager Module")
    print("This module requires an authenticated E*Trade session to run.")
    print("Use it via main_analyst_agent.py or integrate with etrade_cover_call_new.py")
