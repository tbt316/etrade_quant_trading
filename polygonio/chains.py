from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import get_settings, PREMIUM_FIELD_MAP
from .cache_io import stored_option_price
from .poly_client import PolygonAPIClient
from .option_lookup import get_option_quote


@dataclass(frozen=True)
class StrikeRange:
    min_strike: Optional[float]
    max_strike: Optional[float]


def _scaling_for_ticker(ticker: str) -> float:
    """Match legacy scaling logic for index/ETF products.

    - SPY/QQQ/TQQQ/SQQQ => 1
    - SPX/NDX           => 0.5 (wider dollar strikes; narrow percentage window)
    - else              => 1
    """
    t = ticker.upper()
    if t in {"SPY", "QQQ", "TQQQ", "SQQQ"}:
        return 1.0
    if t in {"SPX", "NDX"}:
        return 0.5
    return 1.0


def _price_window(close_price: float, *, scale: float, force_otm: bool) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute (call_low, call_high), (put_low, put_high) strike windows around spot.

    Mirrors the original behavior:
    - If force_otm is False → allow slight ITM (price_limit_percent = -0.5)
    - If force_otm is True  → strictly OTM (price_limit_percent = 0)
    - Calls:   (spot * (1+limit*scale), spot * (1 + 0.5*scale))
    - Puts:    (spot * (1 - 0.5*scale), spot * (1 - limit*scale))
    """
    limit = 0.0 if force_otm else -0.5
    call_low = close_price * (1 + limit * scale)
    call_high = close_price * (1 + 0.5 * scale)
    put_low = close_price * (1 - 0.5 * scale)
    put_high = close_price * (1 - limit * scale)
    return (call_low, call_high), (put_low, put_high)


async def pull_option_chain_data(
    ticker: str,
    call_put: str,
    expiration_str: str,
    as_of_str: str,
    close_price: float,
    *,
    client: PolygonAPIClient,
    force_otm: bool = False,
    force_update: bool = False,
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Dict[str, Optional[float]]]],
]:
    """Assemble chains and premiums around spot for both calls and puts.

    Returns
    -------
    (all_call_data, all_put_data, call_options, put_options, strike_range_dict)

    Notes
    -----
    - Matches legacy semantics (window construction, cache reuse, batch fetching)
    - Only fetches missing legs unless `force_update=True` or global chain update forced
    - Respects `PREMIUM_PRICE_MODE` for which premium field is considered valid
    """
    s = get_settings()
    premium_field = PREMIUM_FIELD_MAP.get(s.premium_price_mode, "trade_price")

    # Ensure we have both call & put chains for the (expiry, as_of)
    unique_chain_requests = [
        (expiration_str, as_of_str, "call"),
        (expiration_str, as_of_str, "put"),
    ]
    chain_data = await client.get_option_chains_batch_async(
        ticker, unique_chain_requests, force_update=force_update
    )

    call_strike_dict: Dict[float, str] = chain_data[ticker][expiration_str][as_of_str].get("call", {}) or {}
    put_strike_dict: Dict[float, str] = chain_data[ticker][expiration_str][as_of_str].get("put", {}) or {}

    if not call_strike_dict and not put_strike_dict:
        return None, None, None, None, None

    # Build strike windows
    scale = _scaling_for_ticker(ticker)
    (call_low, call_high), (put_low, put_high) = _price_window(close_price, scale=scale, force_otm=force_otm)

    # Assemble candidate option dicts within the window
    call_options: List[Dict[str, Any]] = []
    for strike, sym in call_strike_dict.items():
        if call_low < strike < call_high:
            call_options.append(
                {
                    "strike_price": strike,
                    "call_put": "call",
                    "expiration_date": expiration_str,
                    "quote_timestamp": as_of_str,
                    "option_ticker": sym,
                }
            )

    put_options: List[Dict[str, Any]] = []
    for strike, sym in put_strike_dict.items():
        if put_low < strike < put_high:
            put_options.append(
                {
                    "strike_price": strike,
                    "call_put": "put",
                    "expiration_date": expiration_str,
                    "quote_timestamp": as_of_str,
                    "option_ticker": sym,
                }
            )

    # Pre-compute min/max ranges
    def _range(arr: List[Dict[str, Any]]) -> StrikeRange:
        if not arr:
            return StrikeRange(None, None)
        strikes = [o["strike_price"] for o in arr]
        return StrikeRange(min(strikes), max(strikes))

    call_rng = _range(call_options)
    put_rng = _range(put_options)

    strike_range_dict: Optional[Dict[str, Dict[str, Optional[float]]]]
    if call_rng.min_strike is not None and put_rng.min_strike is not None:
        strike_range_dict = {
            "call": {"min_strike": call_rng.min_strike, "max_strike": call_rng.max_strike},
            "put": {"min_strike": put_rng.min_strike, "max_strike": put_rng.max_strike},
        }
    else:
        strike_range_dict = None

    # Decide whether range is sufficient (OPTION_RANGE check) — legacy behavior
    range_ok = False
    if strike_range_dict is not None:
        range_ok = (
            (call_rng.max_strike or 0) > close_price * (1 + s.option_range)
            and (put_rng.min_strike or 1e12) < close_price * (1 - s.option_range)
        )

    # Gather existing cached premiums
    t = ticker.upper()
    all_call_data: List[Optional[Dict[str, Any]]] = []
    all_put_data: List[Optional[Dict[str, Any]]] = []
    call_to_fetch: List[Dict[str, Any]] = []
    put_to_fetch: List[Dict[str, Any]] = []

    # Threshold: if forcing update we demand strictly fresh data, otherwise accept any positive premium
    threshold = 1 if (force_update or s.option_chain_force_update) else 0

    for opt in call_options:
        strike = round(float(opt["strike_price"]), 2)
        data = (
            stored_option_price.get(t, {})
            .get(as_of_str, {})
            .get(strike, {})
            .get(expiration_str, {})
            .get("call", {})
        )
        if data and ((data.get(premium_field, -1) >= threshold)) and not s.option_chain_force_update:
            all_call_data.append(data)
        else:
            if "call" in call_put:
                call_to_fetch.append(opt)
            all_call_data.append(None)

    for opt in put_options:
        strike = round(float(opt["strike_price"]), 2)
        data = (
            stored_option_price.get(t, {})
            .get(as_of_str, {})
            .get(strike, {})
            .get(expiration_str, {})
            .get("put", {})
        )
        if data and ((data.get(premium_field, -1) >= threshold)) and not s.option_chain_force_update:
            all_put_data.append(data)
        else:
            if "put" in call_put:
                put_to_fetch.append(opt)
            all_put_data.append(None)

    # Heuristic from legacy: only batch-fetch if >10% missing or force_update
    need_fetch = (
        (len(call_to_fetch) > 0.1 * max(1, len(call_options)))
        or (len(put_to_fetch) > 0.1 * max(1, len(put_options)))
        or force_update
    )

    if need_fetch and (call_to_fetch or put_to_fetch):
        fetched = await client.get_option_prices_batch_async(ticker, call_to_fetch + put_to_fetch)
        fetched_call = fetched[: len(call_to_fetch)]
        fetched_put = fetched[len(call_to_fetch) :]

        # Fill back the placeholders in order
        if "call" in call_put:
            idx = 0
            for i in range(len(all_call_data)):
                if all_call_data[i] is None:
                    all_call_data[i] = fetched_call[idx] if idx < len(fetched_call) else {}
                    idx += 1
        if "put" in call_put:
            idx = 0
            for i in range(len(all_put_data)):
                if all_put_data[i] is None:
                    all_put_data[i] = fetched_put[idx] if idx < len(fetched_put) else {}
                    idx += 1

    # If we decided not to batch fetch (few gaps), fill the gaps synchronously via v3/quotes
    elif (call_to_fetch or put_to_fetch):
        if "call" in call_put and call_to_fetch:
            for i, opt in enumerate(call_options):
                if all_call_data[i] is None:
                    q = get_option_quote(
                        underlying_ticker=ticker,
                        strike_price=float(opt["strike_price"]),
                        call_put="call",
                        expiration_date=expiration_str,
                        quote_timestamp=as_of_str,
                    )
                    all_call_data[i] = q[0] if isinstance(q, list) and q else {}
        if "put" in call_put and put_to_fetch:
            for i, opt in enumerate(put_options):
                if all_put_data[i] is None:
                    q = get_option_quote(
                        underlying_ticker=ticker,
                        strike_price=float(opt["strike_price"]),
                        call_put="put",
                        expiration_date=expiration_str,
                        quote_timestamp=as_of_str,
                    )
                    all_put_data[i] = q[0] if isinstance(q, list) and q else {}

    return (
        all_call_data or None,
        all_put_data or None,
        call_options or None,
        put_options or None,
        strike_range_dict,
    )
