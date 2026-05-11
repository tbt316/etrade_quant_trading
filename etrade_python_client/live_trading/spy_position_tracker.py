"""
SPY Position Tracker Module

Tracks daily SPY option position snapshots including:
- Total option price
- Total margin required
- Individual position details with full greeks
- YTD cumulative realized gains from closed positions

Data is stored in spy_tracking_data.json with one entry per day,
updated continuously during the day and frozen at end of day.
"""

import json
import os
import pytz
from datetime import datetime, date, time
from typing import List, Dict, Any, Optional

TRACKER_FILE = "spy_tracking_data.json"


def _load_tracker_data() -> Dict[str, Any]:
    """Load existing tracker data or return empty structure."""
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[SPY Tracker] Warning: Could not load tracker file: {e}")
    return {
        "daily_snapshots": {},
        "ytd_realized_gain": 0.0
    }


def _save_tracker_data(data: Dict[str, Any]) -> bool:
    """Save tracker data to file."""
    try:
        with open(TRACKER_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except IOError as e:
        print(f"[SPY Tracker] Error saving tracker file: {e}")
        return False



def _extract_spy_positions(all_positions: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract SPY option positions from the full position list.
    
    Args:
        all_positions: List of StockPosition objects from accounts.portfolio()
        
    Returns:
        List of dictionaries with position details and greeks
    """
    spy_positions = []
    
    for pos in all_positions:
        symbol = getattr(pos, 'symbol', '') or ''
        if symbol.upper() != 'SPY':
            continue
        if getattr(pos, 'security_type', '') != 'Option':
            continue
            
        # Extract all relevant fields including greeks
        position_data = {
            "symbol": symbol,
            "call_put": getattr(pos, 'call_put', None),
            "strike_price": _to_float(getattr(pos, 'strike_price', None)),
            "expiration_date": str(getattr(pos, 'expiration_date', None)),
            "quantity": _to_int(getattr(pos, 'quantity', 0)),
            "last_price": _to_float(getattr(pos, 'last_price', None)),
            "price_paid": _to_float(getattr(pos, 'price_paid', None)),
            "underlying_last_price": _to_float(getattr(pos, 'underlying_last_price', None)),
            # Greeks
            "delta": _to_float(getattr(pos, 'delta', None)),
            "gamma": _to_float(getattr(pos, 'gamma', None)),
            "theta": _to_float(getattr(pos, 'theta', None)),
            "vega": _to_float(getattr(pos, 'vega', None)),
            "rho": _to_float(getattr(pos, 'rho', None)),
            "implied_volatility": _to_float(getattr(pos, 'implied_volatility', None)),
            # Additional useful fields
            "days_to_expiration": _to_int(getattr(pos, 'days_to_expiration', None)),
            "distance_to_strike": _to_float(getattr(pos, 'distance_to_strike', None)),
        }
        spy_positions.append(position_data)
    
    return spy_positions


def _to_float(val) -> Optional[float]:
    """Safely convert to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val) -> Optional[int]:
    """Safely convert to int."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse option symbol to extract components.
    
    Handles format: SPY_2025-02-07_PUT_577
    
    Returns dict with: underlying, expiry_date, option_type (CALL/PUT), strike
    """
    result = {"underlying": None, "expiry_date": None, "option_type": None, "strike": None}
    try:
        parts = symbol.split("_")
        if len(parts) >= 4:
            result["underlying"] = parts[0]
            result["expiry_date"] = parts[1]
            result["option_type"] = parts[2].upper()
            result["strike"] = float(parts[3])
    except Exception:
        pass
    return result


def _reconstruct_historical_positions(all_trades: List[Dict], current_positions: List[Dict], 
                                       start_date: str, end_date: str) -> Dict[str, List[Dict]]:
    """
    Reconstruct SPY option positions for historical dates by working backward from current positions.
    
    Algorithm:
    1. Start with today's positions
    2. Sort trades by date descending (newest first)
    3. For each trade going backward:
       - SELL_OPEN: Position opened, so remove going backward (didn't exist before)
       - BUY_CLOSE: Position closed, so add going backward (existed before close)
       - BUY_OPEN: Long position opened, remove going backward
       - SELL_CLOSE: Long position closed, add going backward
    
    Returns: Dict mapping date -> list of positions on that date
    """
    from datetime import datetime, timedelta
    from copy import deepcopy
    
    # Start with current positions (keyed by symbol)
    # Handle two formats:
    # 1. Trade format: SPY_2026-02-27_PUT_611
    # 2. Snapshot format: {symbol: SPY, call_put: PUT, strike_price: 611, expiration_date: 2026-02-27}
    position_map = {}  # {symbol: {quantity, strike, option_type, expiry_date}}
    for pos in current_positions:
        raw_symbol = pos.get("symbol") or pos.get("ticker")
        
        # Check if it's snapshot format (has call_put field)
        if pos.get("call_put") and pos.get("strike_price") and pos.get("expiration_date"):
            # Convert snapshot format to trade symbol format
            opt_type = pos.get("call_put", "").upper()
            strike = float(pos.get("strike_price", 0))
            expiry = pos.get("expiration_date", "")
            symbol = f"SPY_{expiry}_{opt_type}_{int(strike)}"
            position_map[symbol] = {
                "symbol": symbol,
                "quantity": pos.get("quantity", 0),
                "strike": strike,
                "option_type": opt_type,
                "expiry_date": expiry
            }
        elif raw_symbol and "SPY" in str(raw_symbol).upper() and "_" in str(raw_symbol):
            # Trade symbol format
            parsed = _parse_option_symbol(raw_symbol)
            position_map[raw_symbol] = {
                "symbol": raw_symbol,
                "quantity": pos.get("quantity", 0),
                "strike": parsed.get("strike", 0),
                "option_type": parsed.get("option_type", ""),
                "expiry_date": parsed.get("expiry_date", "")
            }
    
    # Sort trades by date descending
    dated_trades = []
    for trade in all_trades:
        tdate = trade.get("date", "")
        if tdate and tdate >= start_date and tdate <= end_date:
            dated_trades.append(trade)
    dated_trades.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    # Group trades by date
    from collections import defaultdict
    trades_by_date = defaultdict(list)
    for trade in dated_trades:
        trades_by_date[trade.get("date", "")].append(trade)
    
    # Build historical positions
    historical_positions = {}
    
    # Record today's positions FIRST (this is our starting point)
    today = datetime.now().strftime("%Y-%m-%d")
    historical_positions[today] = deepcopy(list(position_map.values()))
    
    # Process dates in descending order (newest to oldest)
    for trade_date in sorted(trades_by_date.keys(), reverse=True):
        trades_for_date = trades_by_date[trade_date]
        
        # Process ALL trades for this date (reverse their effects)
        for trade in trades_for_date:
            action = trade.get("action", "")
            symbol = trade.get("symbol", "")
            quantity = abs(float(trade.get("quantity", 0)))
            
            if not symbol or "SPY" not in symbol.upper():
                continue
            
            parsed = _parse_option_symbol(symbol)
            
            if action in ["SELL_OPEN", "BUY_OPEN"]:
                # Position was opened, so it didn't exist before - remove/reduce
                if symbol in position_map:
                    position_map[symbol]["quantity"] -= (-quantity if action == "SELL_OPEN" else quantity)
                    if abs(position_map[symbol]["quantity"]) < 0.01:
                        del position_map[symbol]
            elif action in ["BUY_CLOSE", "SELL_CLOSE"]:
                # Position was closed, so it existed before - add back
                if symbol not in position_map:
                    position_map[symbol] = {
                        "symbol": symbol,
                        "quantity": 0,
                        "strike": parsed.get("strike", 0),
                        "option_type": parsed.get("option_type", ""),
                        "expiry_date": parsed.get("expiry_date", "")
                    }
                position_map[symbol]["quantity"] += (-quantity if action == "BUY_CLOSE" else quantity)
        
        # Record positions AFTER reversing all trades for this date
        # This represents what positions existed at START of this trade date
        historical_positions[trade_date] = deepcopy(list(position_map.values()))
    
    return historical_positions


def _calculate_margin_from_spreads(positions: List[Dict]) -> float:
    """
    Calculate margin required from spread widths.
    
    For credit spreads, margin = max spread width × contracts × 100
    
    Logic:
    1. Group positions by expiry_date and option_type
    2. Within each group, find short + long pairs (opposite signs)
    3. Spread width = abs(short_strike - long_strike)
    4. Margin = width × min(short_qty, long_qty) × 100
    """
    from collections import defaultdict
    
    # Group by expiry_date
    expiry_groups = defaultdict(lambda: {"CALL": 0.0, "PUT": 0.0})
    
    # Pre-group by expiry and type to match correctly
    type_groups = defaultdict(list)
    for pos in positions:
        qty = pos.get("quantity", 0)
        if qty == 0:
            continue
        key = (pos.get("expiry_date", ""), pos.get("option_type", ""))
        type_groups[key].append(pos)

    # Process each expiry
    expiries = set(k[0] for k in type_groups.keys())
    for expiry in expiries:
        for opt_type in ["CALL", "PUT"]:
            group_positions = type_groups.get((expiry, opt_type), [])
            # Separate shorts (negative qty) and longs (positive qty)
            shorts = [p for p in group_positions if p.get("quantity", 0) < 0]
            longs = [p for p in group_positions if p.get("quantity", 0) > 0]
            
            type_margin = 0.0
            # Match shorts with longs to form spreads
            for short in shorts:
                short_strike = short.get("strike", 0)
                short_qty = abs(short.get("quantity", 0))
                
                for long in longs:
                    long_strike = long.get("strike", 0)
                    long_qty = long.get("quantity", 0)
                    
                    if long_qty <= 0:
                        continue
                    
                    matched_qty = min(short_qty, long_qty)
                    if matched_qty > 0:
                        spread_width = abs(short_strike - long_strike)
                        type_margin += spread_width * matched_qty * 100
                        
                        short_qty -= matched_qty
                        long["quantity"] -= matched_qty
                        
                        if short_qty <= 0:
                            break
            expiry_groups[expiry][opt_type] = type_margin
    
    # Total margin is the sum of max(CALL, PUT) for each expiry
    total_margin = sum(max(v["CALL"], v["PUT"]) for v in expiry_groups.values())
    return total_margin



def update_spy_daily_snapshot(
    all_positions: List[Any],
    spy_total_margin: float = 0.0,
    spy_call_margin: float = 0.0,
    spy_put_margin: float = 0.0
) -> bool:
    """
    Update today's SPY position snapshot.
    
    This should be called after each portfolio refresh. It will overwrite
    the current day's entry, so only the last snapshot of the day is kept.
    
    Args:
        all_positions: List of StockPosition objects from accounts.portfolio()
        spy_total_margin: Total SPY margin required (already accounts for max(call, put) per expiry)
        spy_call_margin: SPY call margin required (for recording breakdown)
        spy_put_margin: SPY put margin required (for recording breakdown)
        
    Returns:
        True if saved successfully
    """
    data = _load_tracker_data()
    
    # Consistently use US/Eastern for both date key and market hours check
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    today_key = now_et.strftime("%Y-%m-%d")
    
    # Market hours check (9:30 AM - 4:10 PM ET)
    market_open = time(9, 30)
    market_close_freeze = time(16, 10)  # Freeze after 4:10 PM ET
    
    # ── Trading Day Check ────────────────────────────────────────────────────
    try:
        from pandas_market_calendars import get_calendar
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=now_et.date(), end_date=now_et.date())
        is_trading_day_nyse = not schedule.empty
        
        if not is_trading_day_nyse:
            print(f"[SPY Tracker] Market is closed today ({now_et.date()}). Skipping update.")
            return True
    except Exception as e:
        # Fallback to simple weekend check
        if now_et.weekday() >= 5:
            print(f"[SPY Tracker] Market is closed (Weekend: {now_et.strftime('%A')}). Skipping update.")
            return True

    is_outside_hours = now_et.time() < market_open or now_et.time() > market_close_freeze

    # If it's a weekday but outside hours:
    if is_outside_hours:
        # If we already have a snapshot, always skip (freeze)
        if today_key in data.get("daily_snapshots", {}):
            print(f"[SPY Tracker] Market is closed (ET: {now_et.strftime('%H:%M:%S')}) and snapshot for {today_key} exists. Skipping update to freeze prices.")
            return True
            
        # If no snapshot exists yet and it's Pre-market (midnight to 9:30 AM), skip 
        # to avoid creating an empty snapshot for a day that hasn't started trading yet.
        if now_et.time() < market_open:
            print(f"[SPY Tracker] Pre-market (ET: {now_et.strftime('%H:%M:%S')}) and no snapshot for {today_key} exists. Skipping to avoid empty snapshot.")
            return True

    # Extract SPY positions
    spy_positions = _extract_spy_positions(all_positions)
    
    # Calculate total option price (signed quantity * price * 100)
    # Filter out long-only positions (only include shorts and matched longs)
    from collections import defaultdict
    type_groups = defaultdict(list)
    for pos in spy_positions:
        qty = pos.get('quantity', 0) or 0
        if qty == 0: continue
        # Handle both formats: snapshot format and trade object format
        expiry = pos.get('expiration_date', '') or pos.get('expiry_date', '')
        call_put = pos.get('call_put', '') or pos.get('option_type', '')
        key = (str(expiry), str(call_put).upper())
        type_groups[key].append({
            'qty': qty,
            'price': pos.get('last_price', 0) or pos.get('price', 0) or 0,
            'strike': pos.get('strike_price', 0) or pos.get('strike', 0) or 0
        })

    total_option_price = 0.0
    for key, group in type_groups.items():
        shorts = [p for p in group if p['qty'] < 0]
        longs = [p for p in group if p['qty'] > 0]
        
        opt_type = key[1]
        # Sort to match spread legs appropriately
        # For PUTs, higher strike matches to short first
        if opt_type == 'PUT':
            longs.sort(key=lambda x: x['strike'], reverse=True)
            shorts.sort(key=lambda x: x['strike'], reverse=True)
        else:
            longs.sort(key=lambda x: x['strike'])
            shorts.sort(key=lambda x: x['strike'])

        for short in shorts:
            # Add short position value (qty is negative, so value is negative)
            total_option_price += short['qty'] * short['price'] * 100.0
            
            # Match longs to this short to form a spread
            short_qty_abs = abs(short['qty'])
            for long in longs:
                if long['qty'] <= 0: continue
                matched_qty = min(short_qty_abs, long['qty'])
                if matched_qty > 0:
                    # Add matched long position value (positive)
                    total_option_price += matched_qty * long['price'] * 100.0
                    short_qty_abs -= matched_qty
                    long['qty'] -= matched_qty
                if short_qty_abs <= 0:
                    break
    
    # Total margin
    total_margin = spy_total_margin

    
    # Get current YTD realized gain
    ytd_gain = data.get("ytd_realized_gain", 0.0)
    
    # Create/update today's snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "total_option_price": round(total_option_price, 2),
        "total_margin": round(total_margin, 2),
        "spy_call_margin": round(spy_call_margin, 2),
        "spy_put_margin": round(spy_put_margin, 2),
        "position_count": len(spy_positions),
        "positions": spy_positions,
        "ytd_realized_gain": round(ytd_gain, 2)
    }
    
    data["daily_snapshots"][today_key] = snapshot
    
    print(f"[SPY Tracker] Updated snapshot for {today_key}: "
          f"Option Price=${total_option_price:,.2f}, Margin=${total_margin:,.2f}, "
          f"YTD Gain=${ytd_gain:,.2f}")
    
    return _save_tracker_data(data)


def record_closed_spy_gain(gain_amount: float, description: str = "") -> bool:
    """
    Record a realized gain from a closed SPY position.
    
    This adds to the cumulative YTD realized gain.
    
    Args:
        gain_amount: The realized gain (positive) or loss (negative)
        description: Optional description of the closed trade
        
    Returns:
        True if saved successfully
    """
    data = _load_tracker_data()
    
    current_ytd = data.get("ytd_realized_gain", 0.0)
    new_ytd = current_ytd + gain_amount
    data["ytd_realized_gain"] = round(new_ytd, 2)
    
    print(f"[SPY Tracker] Recorded closed gain: ${gain_amount:,.2f} "
          f"(YTD total: ${new_ytd:,.2f}) {description}")
    
    return _save_tracker_data(data)


def get_spy_tracking_history() -> Dict[str, Any]:
    """
    Get all SPY tracking history for plotting.
    
    Returns:
        Dictionary with:
        - dates: list of date strings
        - total_option_prices: list of floats
        - total_margins: list of floats  
        - ytd_realized_gains: list of floats
        - ytd_current: current YTD realized gain
    """
    data = _load_tracker_data()
    
    snapshots = data.get("daily_snapshots", {})
    
    # Sort by date
    sorted_dates = sorted(snapshots.keys())
    
    dates = []
    total_option_prices = []
    total_margins = []
    ytd_realized_gains = []
    
    for date_key in sorted_dates:
        snap = snapshots[date_key]
        dates.append(date_key)
        total_option_prices.append(snap.get("total_option_price", 0.0))
        total_margins.append(snap.get("total_margin", 0.0))
        ytd_realized_gains.append(snap.get("ytd_realized_gain", 0.0))
    
    return {
        "dates": dates,
        "total_option_prices": total_option_prices,
        "total_margins": total_margins,
        "ytd_realized_gains": ytd_realized_gains,
        "ytd_current": data.get("ytd_realized_gain", 0.0)
    }


def reset_ytd_gain() -> bool:
    """
    Reset the YTD realized gain to zero (e.g., at start of new year).
    
    Returns:
        True if saved successfully
    """
    data = _load_tracker_data()
    data["ytd_realized_gain"] = 0.0
    print("[SPY Tracker] Reset YTD realized gain to $0.00")
    return _save_tracker_data(data)


SPY_GAINS_CACHE_FILE = "spy_gains_cache.json"


def _load_gains_cache() -> Dict[str, Any]:
    """Load cached SPY gains data or return empty structure."""
    if os.path.exists(SPY_GAINS_CACHE_FILE):
        try:
            with open(SPY_GAINS_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[SPY Tracker] Warning: Could not load gains cache: {e}")
    return {
        "daily_cash_flows": {},  # date -> daily cash flow (not cumulative)
        "last_update_date": None
    }


def _save_gains_cache(data: Dict[str, Any]) -> bool:
    """Save gains cache to file."""
    try:
        with open(SPY_GAINS_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"[SPY Tracker] Error saving gains cache: {e}")
        return False


def calculate_spy_daily_gains(order_instance, start_date: str = "2025-01-01") -> Dict[str, float]:
    """
    Calculate daily cumulative SPY option gains from executed orders.
    
    Uses local caching to avoid repeated E*TRADE API calls. Only fetches
    new data for dates after the last cached date.
    
    Args:
        order_instance: An Order instance with active E*TRADE session
        start_date: Start date in YYYY-MM-DD format
        
    Returns:
        Dictionary mapping date strings to cumulative gain amounts
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    import re
    
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    today = datetime.now().date()
    
    # Load existing cache
    cache = _load_gains_cache()
    cached_flows = cache.get("daily_cash_flows", {})
    cached_trades = cache.get("trade_details", {})
    cached_close_days = set(cache.get("days_with_close_events", []))
    last_update = cache.get("last_update_date")

    # Lookback window for syncing (14 days)
    # This ensures we always re-verify recent history to capture intra-day trades.
    lookback_start = today - timedelta(days=14)
    
    # Proposed fetch start: 14 days ago or start_date, whichever is later
    fetch_start = max(start_date_dt, lookback_start)
    
    # If the cache hasn't been updated in a long time (more than 14 days),
    # ensure we start from the day after the last update to avoid gaps.
    if last_update:
        last_update_dt = datetime.strptime(last_update, '%Y-%m-%d').date()
        fetch_start = min(fetch_start, last_update_dt + timedelta(days=1))
    
    # Skip fetch ONLY if market is closed and we already fetched today's final data
    if last_update and last_update == today.strftime('%Y-%m-%d'):
        # Check if it's weekend or after-hours
        import pytz as _pytz
        _et_tz = _pytz.timezone('US/Eastern')
        _now_et = datetime.now(_et_tz)
        _is_weekend = _now_et.weekday() >= 5
        _is_after_close = _now_et.hour >= 17 or (_now_et.hour == 16 and _now_et.minute >= 45)
        
        if _is_weekend or _is_after_close:
            print(f"[SPY Tracker] Using cached data (sync complete for {today})")
            return {
                "cash_flows": _calculate_cumulative_from_flows(cached_flows, start_date),
                "realized_gains": {},
                "days_with_close_events": list(cached_close_days)
            }

    
    # ===== CHECK FOR RAW API CACHE =====
    # If spy_raw_api_debug.json exists, use it instead of re-polling E*TRADE
    raw_cache_file = os.path.join(os.path.dirname(__file__), "spy_raw_api_debug.json")
    all_orders = []
    all_transactions = []
    use_raw_cache = False
    
    if os.path.exists(raw_cache_file):
        try:
            with open(raw_cache_file, 'r') as f:
                raw_cache = json.load(f)
            
            # Check if raw cache covers our date range
            raw_start = raw_cache.get("fetch_start_date", "")
            raw_end = raw_cache.get("fetch_end_date", "")
            
            if raw_start and raw_end:
                raw_start_dt = datetime.strptime(raw_start, '%Y-%m-%d').date()
                raw_end_dt = datetime.strptime(raw_end, '%Y-%m-%d').date()
                
                # Use raw cache if it covers the date range we need
                if raw_start_dt <= fetch_start and raw_end_dt >= today:
                    all_orders = raw_cache.get("raw_orders", [])
                    use_raw_cache = True
                    print(f"[SPY Tracker] Using raw API cache ({len(all_orders)} orders from {raw_start} to {raw_end})")
        except Exception as e:
            print(f"[SPY Tracker] Could not load raw cache: {e}")
    
    if not use_raw_cache:
        print(f"[SPY Tracker] Fetching new data from {fetch_start} to {today}")
    if not use_raw_cache:
        # Get the session and base_url from order instance
        base_url = order_instance.base_url
        session = order_instance.session
        account_key = order_instance.account['accountIdKey']
        consumer_key = order_instance.consumer_key
        
        # Build date ranges for new data only (1-day chunks)
        date_ranges = []
        current_start = fetch_start
        while current_start <= today:
            current_end = min(current_start + timedelta(days=1), today)
            date_ranges.append((current_start.strftime('%m%d%Y'), current_end.strftime('%m%d%Y')))
            current_start = current_end + timedelta(days=1)
        
        # Fetch executed orders for new date range
        url = f"{base_url}/v1/accounts/{account_key}/orders.json"
        headers = {"consumerKey": consumer_key}
        
        for start, end in date_ranges:
            marker = None
            while True:
                params = {
                    "status": "EXECUTED",
                    "count": 100,
                    "fromDate": start,
                    "toDate": end,
                }
                if marker:
                    params['marker'] = marker
                
                try:
                    response = session.get(url, params=params, headers=headers)
                    if response.status_code != 200:
                        break
                    
                    chunk_data = response.json()
                    if "OrdersResponse" in chunk_data and "Order" in chunk_data["OrdersResponse"]:
                        orders = chunk_data["OrdersResponse"]["Order"]
                        if orders:
                            all_orders.extend(orders)
                            if "marker" in chunk_data["OrdersResponse"] and chunk_data["OrdersResponse"]["marker"]:
                                marker = chunk_data["OrdersResponse"]["marker"]
                            else:
                                break
                        else:
                            break
                    else:
                        break
                except Exception as e:
                    print(f"[SPY Tracker] Error fetching orders: {e}")
                    break
        
        # ===== DEBUG: Save raw API response for offline analysis =====
        debug_cache_file = os.path.join(os.path.dirname(__file__), "spy_raw_api_debug.json")
        try:
            debug_data = {
                "fetch_timestamp": datetime.now().isoformat(),
                "fetch_start_date": fetch_start.strftime('%Y-%m-%d'),
                "fetch_end_date": today.strftime('%Y-%m-%d'),
                "total_orders_fetched": len(all_orders),
                "raw_orders": all_orders
            }
            with open(debug_cache_file, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)
            print(f"[SPY Tracker DEBUG] Saved {len(all_orders)} raw orders to {debug_cache_file}")
        except Exception as e:
            print(f"[SPY Tracker DEBUG] Failed to save debug cache: {e}")
        
        # ===== FETCH TRANSACTIONS API (for UNSOLICITED TRADE orders) =====
        # The Orders API doesn't return all trades - some like broker-initiated sells
        # only appear in the Transactions API
        transactions_url = f"{base_url}/v1/accounts/{account_key}/transactions.json"
        
        for start, end in date_ranges:
            try:
                # Convert date format from MMDDYYYY to MM/DD/YYYY for transactions API
                start_formatted = f"{start[:2]}/{start[2:4]}/{start[4:]}"
                end_formatted = f"{end[:2]}/{end[2:4]}/{end[4:]}"
                
                trans_params = {
                    "startDate": start_formatted,
                    "endDate": end_formatted,
                    "count": 50,  # Max allowed
                }
                
                response = session.get(transactions_url, params=trans_params, headers=headers)
                if response.status_code == 200:
                    trans_data = response.json()
                    if "TransactionListResponse" in trans_data:
                        txns = trans_data["TransactionListResponse"].get("Transaction", [])
                        if txns:
                            all_transactions.extend(txns)
            except Exception as e:
                print(f"[SPY Tracker] Error fetching transactions: {e}")
        
        print(f"[SPY Tracker] Fetched {len(all_transactions)} transactions from Transactions API")
        
        # Save transactions to debug file as well
        try:
            trans_debug_file = os.path.join(os.path.dirname(__file__), "spy_raw_transactions_debug.json")
            trans_debug_data = {
                "fetch_timestamp": datetime.now().isoformat(),
                "total_transactions": len(all_transactions),
                "transactions": all_transactions
            }
            with open(trans_debug_file, 'w') as f:
                json.dump(trans_debug_data, f, indent=2, default=str)
            print(f"[SPY Tracker DEBUG] Saved {len(all_transactions)} transactions to {trans_debug_file}")
        except Exception as e:
            print(f"[SPY Tracker DEBUG] Failed to save transactions debug: {e}")
    
    # Calculate daily cash flows and gains for SPY only from new orders
    # cash_flows = all inflows/outflows when positions are opened/closed
    # realized_gains = only SELL_CLOSE and BUY_CLOSE (profit from closing positions)
    new_daily_cash_flows = defaultdict(float)
    new_trade_details = {}  # Store individual trade records for debugging
    days_with_close_events = set()  # Track days that have close events
    used_equity_orders = set()  # Track matched equity orders for assignments
    seen_order_instruments = set()  # Track already-processed order+instrument combos to prevent duplicates
    
    # First pass: identify all orders for processing
    spy_option_orders = []
    spy_equity_orders = []
    
    for order in all_orders:
        order_id = order.get("orderId")
        order_type = order.get("orderType")
        for detail in order.get("OrderDetail", []):
            if detail.get("status") != "EXECUTED":
                continue
            for instrument in detail.get("Instrument", []):
                sec_type = instrument['Product']['securityType']
                raw_symbol = instrument['Product']['symbol']
                
                # Construct unique symbol for Options to ensure FIFO works
                if sec_type == 'OPTN':
                    prod = instrument['Product']
                    # Try to get details
                    year = prod.get('expiryYear', '')
                    month = prod.get('expiryMonth', '')
                    day = prod.get('expiryDay', '')
                    strike = prod.get('strikePrice', 0)
                    cp = prod.get('callPut', '') # CALL or PUT
                    
                    if year and month and day and strike:
                         # Format: SPY_2025-01-17_P_500.0
                         option_ticker = f"SPY_{year}-{month:02d}-{day:02d}_{cp}_{strike}"
                    else:
                         option_ticker = raw_symbol
                else:
                    option_ticker = raw_symbol

                # Extract underlying asset from option ticker
                underlying_asset = raw_symbol
                try:
                    if "--" in raw_symbol:
                        underlying_asset = raw_symbol.split("--")[0]
                    else:
                        match = re.match(r"([A-Za-z]+)", raw_symbol)
                        underlying_asset = match.group(1) if match else raw_symbol
                except:
                    pass
                
                # Only process SPY
                if underlying_asset.upper() != "SPY":
                    continue
                
                executed_time = datetime.fromtimestamp(detail.get('executedTime') / 1000)
                executed_date = executed_time.strftime('%Y-%m-%d')
                order_action = instrument.get("orderAction", "")
                filled_quantity = float(instrument.get("filledQuantity", 0))
                avg_price = instrument.get("averageExecutionPrice")
                
                if avg_price is None:
                    continue
                
                # Create unique key for this order+instrument to detect duplicates
                # Use reconstructed option_ticker to be specific
                instrument_key = f"{order_id}_{option_ticker}_{order_action}_{filled_quantity}"
                if instrument_key in seen_order_instruments:
                    continue  # Skip duplicate
                seen_order_instruments.add(instrument_key)
                
                trade_info = {
                    "order": order,
                    "order_id": order_id,
                    "order_type": order_type,
                    "sec_type": sec_type,
                    "symbol": option_ticker,
                    "underlying": underlying_asset,
                    "action": order_action,
                    "quantity": filled_quantity,
                    "avg_price": avg_price,
                    "date": executed_date,
                }
                
                # Route by SECURITY TYPE, not order type
                # This ensures equity from OPTION_ASSIGNMENT goes to equity processing
                if sec_type == "EQ":
                    # Filter: Only include equity trades with qty that's a multiple of 100
                    # Option assignments always result in 100-share multiples (1 contract = 100 shares)
                    # Manual equity trades (like long-term LONG positions) typically have qty < 100
                    if filled_quantity >= 100 and filled_quantity % 100 == 0:
                        spy_equity_orders.append(trade_info)
                    else:
                        # Skip manual equity trades (not from option assignments)
                        continue
                elif sec_type == "OPTN":
                    spy_option_orders.append(trade_info)
    
    # Process SPY option orders (including OPTION_ASSIGNMENT)
    for trade in spy_option_orders:
        order_type = trade["order_type"]
        order_action = trade["action"]
        filled_quantity = trade["quantity"]
        avg_price = trade["avg_price"]
        executed_date = trade["date"]
        sec_type = trade["sec_type"]
        underlying = trade["underlying"]
        order_id = trade["order_id"]
        
        # Handle OPTION_ASSIGNMENT - SKIP these entirely here
        # They will be processed as equity trades in the equity loop
        # and FIFO will match BUY with SELL naturally
        if order_type == "OPTION_ASSIGNMENT":
            # Store assignment details for logging, but don't create trade records here
            # The assignment creates an equity position which will be in spy_equity_orders
            continue
        
        # Calculate trade amount
        if sec_type == "OPTN":
            trade_amount = filled_quantity * avg_price * 100
        else:
            # For EQ from assignment
            trade_amount = filled_quantity * avg_price
        
        is_close_event = order_action in ["SELL_CLOSE", "BUY_CLOSE", "SELL", "BUY"]
        
        # Calculate cash flow impact
        if order_action in ["SELL_OPEN", "SELL_CLOSE", "SELL", "SELL_TO_COVER"]:
            cash_impact = trade_amount
            new_daily_cash_flows[executed_date] += trade_amount
        elif order_action in ["BUY_OPEN", "BUY_CLOSE", "BUY", "BUY_TO_COVER"]:
            cash_impact = -trade_amount
            new_daily_cash_flows[executed_date] -= trade_amount
        else:
            cash_impact = 0
        
        # Store trade details for debugging
        trade_record = {
            "date": executed_date,
            "action": order_action,
            "symbol": trade["symbol"],
            "quantity": filled_quantity,
            "price": avg_price,
            "cash_impact": round(cash_impact, 2),
            "order_id": order_id
        }
        if executed_date not in new_trade_details:
            new_trade_details[executed_date] = []
        new_trade_details[executed_date].append(trade_record)
        
        # Mark days with close events for realized gain snapshot
        if is_close_event:
            days_with_close_events.add(executed_date)
    
    # ===== PROCESS ALL SPY EQUITY ORDERS =====
    # Store ALL equity trades for proper FIFO matching
    # This includes: assignment BUYs at strike price, and manual SELLs at market price
    for trade in spy_equity_orders:
        order_action = trade["action"]
        filled_quantity = trade["quantity"]
        avg_price = trade["avg_price"]  # Raw execution price
        executed_date = trade["date"]
        order_id = trade["order_id"]
        
        # NO SKIP - cache ALL equity trades for FIFO
        
        # Calculate cash flow (equity is 1x multiplier)
        trade_amount = filled_quantity * avg_price
        
        if order_action in ["SELL", "SELL_TO_COVER"]:
            cash_impact = trade_amount
            new_daily_cash_flows[executed_date] += trade_amount
        elif order_action in ["BUY", "BUY_TO_COVER"]:
            cash_impact = -trade_amount
            new_daily_cash_flows[executed_date] -= trade_amount
        else:
            cash_impact = 0
        
        # Store equity trade for FIFO
        trade_record = {
            "date": executed_date,
            "action": order_action,
            "symbol": "SPY",  # Equity symbol
            "quantity": filled_quantity,
            "price": avg_price,
            "cash_impact": round(cash_impact, 2),
            "order_id": order_id,
            "sec_type": "EQ"  # Mark as equity for FIFO multiplier
        }
        if executed_date not in new_trade_details:
            new_trade_details[executed_date] = []
        new_trade_details[executed_date].append(trade_record)
        
        if order_action in ["SELL", "SELL_TO_COVER"]:
            days_with_close_events.add(executed_date)
    
    # RECONCILIATION: Clear existing cache entries for the sync window
    # to avoid double-counting or keeping stale records.
    sync_date_str = fetch_start.strftime('%Y-%m-%d')
    print(f"[SPY Tracker] Syncing cache for dates >= {sync_date_str}")
    
    # Filter out dates in the sync window from current cache
    cached_flows = {d: v for d, v in cached_flows.items() if d < sync_date_str}
    cached_trades = {d: t for d, t in cached_trades.items() if d < sync_date_str}
    cached_close_days = {d for d in cached_close_days if d < sync_date_str}

    # Merge new data with cached data
    for date_key, flow_value in new_daily_cash_flows.items():
        cached_flows[date_key] = flow_value
    
    # Merge trade details
    for date_key, trades in new_trade_details.items():
        cached_trades[date_key] = trades
    
    # Track days with close events
    cached_close_days.update(days_with_close_events)
    
    # Update cache
    cache["daily_cash_flows"] = cached_flows
    cache["trade_details"] = cached_trades
    cache["days_with_close_events"] = list(cached_close_days)
    # Only mark today as fully fetched AFTER market close (4:30 PM ET)
    # Before that, set to yesterday so today's trades get re-fetched next run
    import pytz as _pytz
    _et_tz = _pytz.timezone('US/Eastern')
    _now_et = datetime.now(_et_tz)
    _market_close = _now_et.replace(hour=16, minute=30, second=0, microsecond=0)
    if _now_et >= _market_close or _now_et.weekday() >= 5:
        # After market close or weekend: safe to mark today as done
        cache["last_update_date"] = today.strftime('%Y-%m-%d')
    else:
        # During/before trading hours: only mark through yesterday
        # so today's trades get re-fetched on the next run
        yesterday = today - timedelta(days=1)
        cache["last_update_date"] = yesterday.strftime('%Y-%m-%d')
        print(f"[SPY Tracker] Market still open — will re-fetch today's trades next run")
    _save_gains_cache(cache)
    
    print(f"[SPY Tracker] Cached {len(cached_flows)} days of SPY data ({len(cached_close_days)} close days)")
    
    # Return cumulative cash flows and realized gains on close days
    return {
        "cash_flows": _calculate_cumulative_from_flows(cached_flows, start_date),
        "realized_gains": {},
        "days_with_close_events": list(cached_close_days)
    }


def _calculate_cumulative_from_flows(daily_flows: Dict[str, float], start_date: str) -> Dict[str, float]:
    """Convert daily cash flows to cumulative gains starting from start_date."""
    from datetime import datetime
    
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    # Filter to dates >= start_date and sort
    filtered_dates = sorted([d for d in daily_flows.keys() 
                             if datetime.strptime(d, '%Y-%m-%d').date() >= start_date_dt])
    
    cumulative_gains = {}
    running_total = 0.0
    
    for date_key in filtered_dates:
        running_total += daily_flows[date_key]
        cumulative_gains[date_key] = round(running_total, 2)
    
    return cumulative_gains


def _recalculate_gains_fifo(trade_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Re-calculate realized gains using FIFO matching logic.
    Decouples Realized Gains from Cash Flow by matching Open/Close trades.
    """
    from collections import defaultdict
    
    daily_realized_gains = defaultdict(float)
    open_positions = defaultdict(list)  # {symbol: [queue of open legs]}
    
    # Sort trades by date, then order_id
    flat_trades = []
    if isinstance(trade_list, dict):
        for date, trades in trade_list.items():
            flat_trades.extend(trades)
    else:
        flat_trades = trade_list
        
    flat_trades.sort(key=lambda x: (x['date'], x['order_id']))
    
    for trade in flat_trades:
        action = trade['action']
        symbol = trade['symbol']
        qty = float(trade['quantity'])
        price = float(trade['price'])
        date = trade['date']
        
        # Determine if Open or Close
        is_open = 'OPEN' in action
        is_close = 'CLOSE' in action
        
        # Map SPY Equity actions (Assignment/Liquidation)
        if 'SPY' == symbol and 'OPTN' not in str(trade.get('sec_type', '')): 
             # If symbol is exactly "SPY", assume Equity.
             if action == 'BUY':
                 is_open = True
                 is_close = False
             elif action == 'SELL':
                 is_open = False
                 is_close = True

        if is_open:
            open_positions[symbol].append({
                'qty': qty,
                'price': price,
                'action': action,
                'date': date
            })
            
        elif is_close:
            needed = qty
            pnl = 0.0
            
            while needed > 0 and open_positions[symbol]:
                match = open_positions[symbol].pop(0) # FIFO
                matched_qty = min(needed, match['qty'])
                
                entry_price = match['price']
                exit_price = price
                
                # Check multiplier: 100 for Options, 1 for Equity (SPY)
                # If symbol is exactly "SPY", assume multiplier 1
                multiplier = 100 if symbol != 'SPY' else 1
                
                if 'SELL' in match['action']: # We were Short
                    # Profit = Entry - Exit
                    trade_pnl = (entry_price - exit_price) * matched_qty * multiplier
                else: # We were Long
                    # Profit = Exit - Entry
                    trade_pnl = (exit_price - entry_price) * matched_qty * multiplier
                
                pnl += trade_pnl
                needed -= matched_qty
                
                if match['qty'] > matched_qty:
                    match['qty'] -= matched_qty
                    open_positions[symbol].insert(0, match) # Put back at front
            
            daily_realized_gains[date] += pnl
    
    # ===== EXPIRATION SETTLEMENT =====
    # Any open option positions that have expired should be closed at $0
    # This realizes the full premium for expired short options
    from datetime import datetime
    today = datetime.now().date()
    
    for symbol, legs in list(open_positions.items()):
        if symbol == 'SPY':
            continue  # Skip equity - don't auto-settle
        
        # Parse expiration from symbol (SPY_2025-02-21_CALL_619)
        parts = symbol.split('_')
        if len(parts) >= 2:
            try:
                exp_date = datetime.strptime(parts[1], '%Y-%m-%d').date()
            except:
                continue  # Skip if can't parse
        else:
            continue
        
        if exp_date >= today:
            continue  # Not expired yet
        
        # Close all expired positions at $0
        exp_date_str = exp_date.strftime('%Y-%m-%d')
        for leg in legs:
            multiplier = 100
            if 'SELL' in leg['action']:
                # Short expired worthless = full profit (keep premium)
                pnl = leg['price'] * leg['qty'] * multiplier
            else:
                # Long expired worthless = full loss (lost premium)
                pnl = -leg['price'] * leg['qty'] * multiplier
            daily_realized_gains[exp_date_str] += pnl
        
        # Clear expired positions
        open_positions[symbol] = []
            
    # Convert daily P&L to cumulative
    sorted_dates = sorted(daily_realized_gains.keys())
    cumulative_gains = {}
    running_total = 0.0
    
    for date in sorted_dates:
        running_total += daily_realized_gains[date]
        cumulative_gains[date] = round(running_total, 2)
        
    return cumulative_gains


def _get_historical_gains(order_instance, start_date: str) -> Dict[str, Dict[str, float]]:
    """
    Helper to fetch historical gains if order_instance is available.
    Wraps calculate_spy_daily_gains.
    """
    if order_instance is None:
        return {}
        
    try:
        return calculate_spy_daily_gains(order_instance, start_date)
    except Exception as e:
        print(f"[SPY Tracker] Could not fetch historical gains: {e}")
        return {}


def get_spy_tracking_history_with_gains(order_instance=None, start_date: str = "2025-01-01") -> Dict[str, Any]:
    """
    Get SPY tracking history merged with historical gains from E*TRADE.
    Uses FIFO logic for Realized Gains.
    Uses backward position reconstruction for historical margin calculation.
    """
    data = _load_tracker_data()
    snapshots = data.get("daily_snapshots", {})
    
    # Get historical data (cached or fresh)
    historical_data = _get_historical_gains(order_instance, start_date)
    historical_cash_flows = historical_data.get("cash_flows", {})
    
    # Calculate Realized Gains using FIFO on trade details
    # We need access to cache's trade_details
    cache = _load_gains_cache()
    cached_trades = cache.get("trade_details", {})
    
    if cached_trades:
        historical_realized_gains = _recalculate_gains_fifo(cached_trades)
    else:
        historical_realized_gains = {}
    
    # ===== RECONSTRUCT HISTORICAL POSITIONS FOR MARGIN =====
    # Build flat list of all trades for position reconstruction
    all_trades_list = []
    for date_key, trades in cached_trades.items():
        for trade in trades:
            trade_copy = dict(trade)
            trade_copy["date"] = date_key
            all_trades_list.append(trade_copy)
    
    # Get current positions from latest snapshot or empty
    today = datetime.now().strftime("%Y-%m-%d")
    latest_snapshot = snapshots.get(today, {})
    current_positions = latest_snapshot.get("positions", [])
    
    # Reconstruct historical positions
    historical_positions = {}
    if all_trades_list and current_positions:
        historical_positions = _reconstruct_historical_positions(
            all_trades_list, current_positions, start_date, today
        )
    
    # Combine all dates
    all_dates = set(snapshots.keys()) | set(historical_cash_flows.keys()) | set(historical_realized_gains.keys())
    sorted_dates = sorted(all_dates)
    
    dates = []
    total_option_prices = []
    total_margins = []
    cash_flows = []
    realized_gains = []
    
    # Determine the latest values to carry forward
    latest_realized_gain = 0.0
    latest_cash_flow = 0.0
    
    for date_key in sorted_dates:
        dates.append(date_key)
        
        # Snapshot data
        snap = snapshots.get(date_key, {})
        total_option_prices.append(snap.get("total_option_price", None))
        
        # Margin: use stored value if available, otherwise calculate from reconstructed positions
        stored_margin = snap.get("total_margin")
        if stored_margin is not None:
            total_margins.append(stored_margin)
        elif date_key in historical_positions:
            # Calculate margin from reconstructed positions
            reconstructed_margin = _calculate_margin_from_spreads(historical_positions[date_key])
            total_margins.append(reconstructed_margin)
        else:
            total_margins.append(None)
        
        # Cash flows (Cumulative)
        if date_key in historical_cash_flows:
            val = historical_cash_flows[date_key]
            cash_flows.append(val)
            latest_cash_flow = val
        else:
            # Carry forward previous value
            cash_flows.append(latest_cash_flow)
        
        # Realized gains (Stepped Line)
        if date_key in historical_realized_gains:
            val = historical_realized_gains[date_key]
            realized_gains.append(val)
            latest_realized_gain = val
        else:
            # Carry forward previous value
            if realized_gains:
                 realized_gains.append(latest_realized_gain)
            else:
                 realized_gains.append(0.0)
    
    return {
        "dates": dates,
        "total_option_prices": total_option_prices,
        "total_margins": total_margins,
        "cash_flows": cash_flows,
        "realized_gains": realized_gains,
        "ytd_current": latest_realized_gain
    }

