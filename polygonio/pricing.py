from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import get_settings, PREMIUM_FIELD_MAP
from .cache_io import stored_option_price
from .chains import pull_option_chain_data

log = logging.getLogger(__name__)

# ---------------------------------------------------------
# Delta calculation (finite differences across neighboring strikes)
# ---------------------------------------------------------

def calculate_delta(
    ticker: str,
    pricing_date: str,
    expiration_date: str,
    call_put: str,
    *,
    force_delta_update: bool = False,
) -> None:
    """Populate `close_price_delta` and `mid_price_delta` for an option chain slice.

    Logic mirrors the legacy implementation:
    - For each strike, compute forward/backward diffs to nearest neighbors
    - Average available diffs and round to 4 decimals
    - Skip if already present unless `force_delta_update` is True
    """
    t = ticker.upper()

    if t not in stored_option_price or pricing_date not in stored_option_price[t]:
        return

    # Collect all strikes with this expiry & call/put present
    strikes: List[float] = []
    for K, expiries in stored_option_price[t][pricing_date].items():
        if (
            isinstance(expiries, dict)
            and expiration_date in expiries
            and call_put in expiries[expiration_date]
        ):
            strikes.append(K)

    if not strikes:
        return

    # Sorting consistent with legacy (calls asc, puts desc)
    if call_put == "call":
        strikes.sort()
    elif call_put == "put":
        strikes.sort(reverse=True)

    for K in strikes:
        opt_dict = stored_option_price[t][pricing_date][K][expiration_date][call_put]

        # Skip if both deltas already present and not forced
        if not force_delta_update and (
            "close_price_delta" in opt_dict and "mid_price_delta" in opt_dict
        ):
            continue

        # Find neighbors
        K_prev = max([s for s in strikes if s < K], default=None)
        K_next = min([s for s in strikes if s > K], default=None)

        close_delta = 0.0
        mid_delta = 0.0
        close_cnt = 0
        mid_cnt = 0

        # Current prices
        pK = opt_dict

        # Forward difference (K -> K_next)
        if K_next is not None:
            p_next = stored_option_price[t][pricing_date][K_next][expiration_date][call_put]
            if (
                "close_price" in pK
                and "close_price" in p_next
                and pK["close_price"] is not None
                and p_next["close_price"] is not None
            ):
                close_delta += (p_next["close_price"] - pK["close_price"]) / (K_next - K)
                close_cnt += 1
            if (
                "mid_price" in pK
                and "mid_price" in p_next
                and pK["mid_price"] is not None
                and p_next["mid_price"] is not None
            ):
                mid_delta += (p_next["mid_price"] - pK["mid_price"]) / (K_next - K)
                mid_cnt += 1

        # Backward difference (K_prev -> K)
        if K_prev is not None:
            p_prev = stored_option_price[t][pricing_date][K_prev][expiration_date][call_put]
            if (
                "close_price" in pK
                and "close_price" in p_prev
                and pK["close_price"] is not None
                and p_prev["close_price"] is not None
            ):
                close_delta += (pK["close_price"] - p_prev["close_price"]) / (K - K_prev)
                close_cnt += 1
            if (
                "mid_price" in pK
                and "mid_price" in p_prev
                and pK["mid_price"] is not None
                and p_prev["mid_price"] is not None
            ):
                mid_delta += (pK["mid_price"] - p_prev["mid_price"]) / (K - K_prev)
                mid_cnt += 1

        opt_dict["close_price_delta"] = round(close_delta / close_cnt, 4) if close_cnt else 0.0
        opt_dict["mid_price_delta"] = round(mid_delta / mid_cnt, 4) if mid_cnt else 0.0


# ---------------------------------------------------------
# Premium interpolation / extrapolation
# ---------------------------------------------------------

async def interpolate_option_price(
    ticker: str,
    close_price_today: float,
    strike_price_to_interpolate: float,
    option_type: str,  # 'call' or 'put'
    expiration_date: str,  # YYYY-MM-DD
    pricing_date: str,  # YYYY-MM-DD
    *,
    premium_field: Optional[str] = None,  # e.g., 'mid_price' | 'close_price' | 'trade_price'
    price_interpolate_flag: bool = True,
    client: Optional[Any] = None,  # PolygonAPIClient
) -> float:
    """Estimate an option premium at a given strike by interpolation/extrapolation.

    Behavior matches legacy function with guardrails:
    - If direct price exists in cache and is valid → return it
    - Otherwise request a chain slice (calls+puts) for the (expiry, as_of) and
      collect nearby points to PCHIP-interpolate
    - Temporal cohesion: prefer points within ≤60s of median timestamp, expand up to 300s
    - Enforce intrinsic value floor; preserve monotonicity for puts
    - On failure, return 0.0
    """
    s = get_settings()
    premium_field = premium_field or PREMIUM_FIELD_MAP.get(s.premium_price_mode, "trade_price")

    def _is_valid_price(d: Dict[str, Any], field: str) -> bool:
        if not isinstance(d, dict) or d.get(field) is None:
            return False
        val = d.get(field, 0.0)
        if not isinstance(val, (int, float)) or val <= 0:
            return False
        if field == "mid_price":
            return d.get("ask_size", 0) > 0 and d.get("bid_size", 0) > 0
        if field == "close_price":
            return d.get("close_volume", 0) > 0
        if field == "trade_price":
            return d.get("trade_size", 0) > 0
        return False

    def _linear_extrap(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)

    t = ticker.upper()
    Kt = round(float(strike_price_to_interpolate), 2)
    cp = option_type.lower()

    # 1) Direct lookup
    direct = (
        stored_option_price.get(t, {})
        .get(pricing_date, {})
        .get(Kt, {})
        .get(expiration_date, {})
        .get(cp, {})
    )
    if _is_valid_price(direct, premium_field):
        return float(direct[premium_field])

    if not price_interpolate_flag:
        log.debug(
            "No direct data and interpolation disabled: %s %s K=%s exp=%s asof=%s",
            t,
            cp,
            Kt,
            expiration_date,
            pricing_date,
        )
        return 0.0

    # 2) Ensure we have a populated chain window to source points from
    if client is None:
        log.warning("interpolate_option_price called without client; cannot fetch chain data")
        return 0.0

    all_call_data, all_put_data, call_opts, put_opts, _ = await pull_option_chain_data(
        t,
        cp,
        str(expiration_date),
        str(pricing_date),
        close_price_today,
        client=client,
    )

    option_opts = call_opts if cp == "call" else put_opts
    option_data = all_call_data if cp == "call" else all_put_data

    if option_opts is None or option_data is None:
        log.debug("No chain data for %s %s exp=%s asof=%s", t, cp, expiration_date, pricing_date)
        return 0.0

    # 3) Collect usable points near target strike
    points: List[Dict[str, float]] = []
    for opt, prem in zip(option_opts, option_data):
        if not prem:
            continue
        val = float(prem.get(premium_field, 0.0))
        if val <= 0.0:
            continue
        strike = float(opt["strike_price"])
        # restrict to +/-50% window around target (legacy ratio check)
        if abs(strike - Kt) / max(Kt, 1e-9) < 0.5:
            ts = float(prem.get("sip_timestamp", 0)) / 1e9  # ns → s
            points.append({"strike": strike, "price": val, "ts": ts})

    if len(points) < 2:
        log.debug(
            "Insufficient points for interpolation: %s %s K=%s close=%.2f exp=%s asof=%s",
            t,
            cp,
            Kt,
            close_price_today,
            expiration_date,
            pricing_date,
        )
        return 0.0

    points.sort(key=lambda x: x["strike"])  # ensure sorted by strike

    strikes = [p["strike"] for p in points]
    prices = [p["price"] for p in points]
    stamps = [p["ts"] for p in points]

    # 4) Temporal cohesion: prefer a tight window around median timestamp
    selected = []
    time_window = 60.0
    while len(selected) < 4 and time_window < 300.0 and stamps:
        if max(stamps) - min(stamps) > time_window:
            median_ts = float(np.median(stamps))
            selected = [p for p in points if abs(p["ts"] - median_ts) <= time_window]
        time_window += 30.0

    if len(selected) < 4:
        selected = points  # fallback to all available

    s_strikes = [p["strike"] for p in selected]
    s_prices = [p["price"] for p in selected]

    # 5) Interpolate or extrapolate
    inside = min(s_strikes) <= Kt <= max(s_strikes)

    if inside:
        interp = PchipInterpolator(s_strikes, s_prices)
        est = float(interp(Kt))

        # Preserve monotonicity for puts around neighbors
        if cp == "put":
            lower = max([s for s in s_strikes if s < Kt], default=None)
            higher = min([s for s in s_strikes if s > Kt], default=None)
            if lower is not None:
                est = max(est, s_prices[s_strikes.index(lower)])
            if higher is not None:
                est = min(est, s_prices[s_strikes.index(higher)])
    else:
        if len(s_strikes) < 2:
            return 0.0
        if Kt < min(s_strikes):
            x1, x2 = s_strikes[0], s_strikes[1]
            y1, y2 = s_prices[0], s_prices[1]
            est = _linear_extrap(Kt, x1, y1, x2, y2)
            if cp == "put":
                est = min(est, y1)  # puts non-decreasing with strike
        else:
            x1, x2 = s_strikes[-2], s_strikes[-1]
            y1, y2 = s_prices[-2], s_prices[-1]
            est = _linear_extrap(Kt, x1, y1, x2, y2)
            if cp == "put":
                est = max(est, y2)

    # 6) Intrinsic value floor & non-negativity
    intrinsic = max(close_price_today - Kt, 0.0) if cp == "call" else max(Kt - close_price_today, 0.0)
    est = float(max(est, intrinsic, 0.0))

    log.debug(
        "Estimated price for %s %s K=%s: %.3f (spot=%.2f, exp=%s, asof=%s)",
        t,
        cp,
        Kt,
        est,
        close_price_today,
        expiration_date,
        pricing_date,
    )
    return est

