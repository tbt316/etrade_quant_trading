from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Iterable

import asyncio

from .config import get_settings, PREMIUM_FIELD_MAP
from .cache_io import load_stored_option_data, stored_option_chain, stored_option_price
from .poly_client import PolygonAPIClient


def _to_date(d: str | date | None) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def _ds(d: str | date | None) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, str):
        return d
    return d.strftime("%Y-%m-%d")


def _price_field() -> str:
    mode = (get_settings().premium_price_mode or "mid").lower()
    return PREMIUM_FIELD_MAP.get(mode, "mid_price")


FALLBACK_MAX_WEEKS = 5


def _has_chain_for(ticker: str, as_of_s: str, exp_s: str) -> Tuple[bool, Dict[float, Any], Dict[float, Any]]:
    bucket = stored_option_chain.get(ticker, {}).get(exp_s, {}).get(as_of_s, {})
    calls = bucket.get("call") or {}
    puts = bucket.get("put") or {}
    return (len(calls) > 0 and len(puts) > 0, calls, puts)


def find_available_expiration(
    ticker: str,
    as_of: str | date,
    target_expiration: str | date,
    max_weeks: int = FALLBACK_MAX_WEEKS,
) -> Tuple[Optional[str], Optional[Dict[float, Any]], Optional[Dict[float, Any]]]:
    as_of_s = _ds(as_of)
    target_d = _to_date(target_expiration)
    if as_of_s is None or target_d is None:
        return None, None, None

    # exact
    ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(target_d))
    if ok:
        return _ds(target_d), calls, puts

    # forward
    for k in range(1, max_weeks + 1):
        cand = target_d + timedelta(days=7 * k)
        ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(cand))
        if ok:
            return _ds(cand), calls, puts

    # backward
    for k in range(1, max_weeks + 1):
        cand = target_d - timedelta(days=7 * k)
        ok, calls, puts = _has_chain_for(ticker, as_of_s, _ds(cand))
        if ok:
            return _ds(cand), calls, puts

    return None, None, None


@lru_cache(maxsize=2048)
def _get_price_bucket(ticker: str, as_of_s: str) -> Dict[float, Dict[str, Dict[str, Dict[str, float]]]]:
    return stored_option_price.get(ticker, {}).get(as_of_s, {})


def _extract_chain_for_expiration(
    *,
    ticker: str,
    as_of_s: str,
    expiration_s: str,
) -> Tuple[Dict[float, str], Dict[float, str]]:
    bucket = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {})
    call_symbols = bucket.get("call") or {}
    put_symbols = bucket.get("put") or {}
    return call_symbols, put_symbols


def _extract_premiums_for_strikes(
    *,
    ticker: str,
    as_of_s: str,
    expiration_s: str,
    strikes: Iterable[float],
    opt_type: str,
) -> Dict[float, float]:
    pf = _price_field()
    price_bucket = _get_price_bucket(ticker, as_of_s)
    out: Dict[float, float] = {}
    for k in strikes:
        exp_map = price_bucket.get(k, {})
        type_map = exp_map.get(expiration_s, {}).get(opt_type, {})
        val = type_map.get(pf)
        if val is not None:
            try:
                out[float(k)] = float(val)
            except Exception:
                pass
    return out


def _window_strikes(
    *,
    strikes: Iterable[float],
    spot: Optional[float],
    option_range: float,
    force_otm: bool,
    is_call: bool,
) -> Iterable[float]:
    if spot is None or spot <= 0:
        return sorted(set(float(k) for k in strikes))

    lo = spot * (1.0 - option_range)
    hi = spot * (1.0 + option_range)

    filt = []
    for k in strikes:
        try:
            kf = float(k)
        except Exception:
            continue
        if lo <= kf <= hi:
            if not force_otm:
                filt.append(kf)
            else:
                if is_call and kf >= spot:
                    filt.append(kf)
                elif (not is_call) and kf <= spot:
                    filt.append(kf)
    return sorted(set(filt))


@dataclass(frozen=True)
class ChainResult:
    ticker: str
    as_of: str
    expiration: str
    call_options: Dict[float, Dict[str, Any]]
    put_options: Dict[float, Dict[str, Any]]


def get_option_chain_for_date(
    *,
    ticker: str,
    as_of_str: Optional[str] = None,
    expiration_str: Optional[str] = None,
    as_of: Optional[str | date] = None,
    expiration: Optional[str | date] = None,
    spot: Optional[float] = None,
    option_range: Optional[float] = None,
    force_otm: Optional[bool] = None,
) -> Optional[ChainResult]:
    settings = get_settings()
    option_range = settings.option_range if option_range is None else option_range
    force_otm = settings.force_otm if force_otm is None else force_otm

    as_of_s = _ds(as_of_str) or _ds(as_of)
    expiration_s = _ds(expiration_str) or _ds(expiration)
    if as_of_s is None or expiration_s is None:
        return None

    call_syms = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("call") or {}
    put_syms  = stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("put") or {}
    if not call_syms or not put_syms:
        # Attempt to fetch missing chain data from Polygon
        async def _fetch() -> None:
            async with PolygonAPIClient() as client:
                reqs = [(expiration_s, as_of_s, "call"), (expiration_s, as_of_s, "put")]
                await client.get_option_chains_batch_async(ticker, reqs)

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                new_loop = asyncio.new_event_loop()
                new_loop.run_until_complete(_fetch())
                new_loop.close()
            else:
                loop.run_until_complete(_fetch())
        except Exception:
            pass

        call_syms = (
            stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("call")
            or call_syms
        )
        put_syms = (
            stored_option_chain.get(ticker, {}).get(expiration_s, {}).get(as_of_s, {}).get("put")
            or put_syms
        )

    if not call_syms or not put_syms:
        chosen_exp, calls_fallback, puts_fallback = find_available_expiration(
            ticker=ticker, as_of=as_of_s, target_expiration=expiration_s
        )
        if not chosen_exp:
            return None
        expiration_s = chosen_exp
        call_syms = calls_fallback or {}
        put_syms  = puts_fallback or {}

    all_call_strikes = list(call_syms.keys())
    all_put_strikes = list(put_syms.keys())

    win_call_strikes = _window_strikes(
        strikes=all_call_strikes, spot=spot, option_range=option_range, force_otm=force_otm, is_call=True
    )
    win_put_strikes = _window_strikes(
        strikes=all_put_strikes, spot=spot, option_range=option_range, force_otm=force_otm, is_call=False
    )

    call_prem = _extract_premiums_for_strikes(
        ticker=ticker, as_of_s=as_of_s, expiration_s=expiration_s, strikes=win_call_strikes, opt_type="call"
    )
    put_prem = _extract_premiums_for_strikes(
        ticker=ticker, as_of_s=as_of_s, expiration_s=expiration_s, strikes=win_put_strikes, opt_type="put"
    )

    call_options: Dict[float, Dict[str, Any]] = {}
    for k in win_call_strikes:
        sym = call_syms.get(k)
        prem = call_prem.get(k)
        if sym is not None and prem is not None:
            call_options[float(k)] = {"symbol": sym, "premium": float(prem)}

    put_options: Dict[float, Dict[str, Any]] = {}
    for k in win_put_strikes:
        sym = put_syms.get(k)
        prem = put_prem.get(k)
        if sym is not None and prem is not None:
            put_options[float(k)] = {"symbol": sym, "premium": float(prem)}

    return ChainResult(
        ticker=ticker,
        as_of=as_of_s,
        expiration=expiration_s,
        call_options=call_options,
        put_options=put_options,
    )

# -----------------------------------------------------------------------------
# Legacy-compatible async API
# -----------------------------------------------------------------------------
async def pull_option_chain_data(
    ticker: str,
    call_put: str,
    expiration_str: str,
    as_of_str: str,
    close_price: float | None = None,
    *,
    client=None,
    force_otm: bool = False,
    force_update: bool = False,
):
    """
    Legacy-compatible wrapper used by pricing/recursive_backtest.

    Returns a 5-tuple:
        (all_call_data, all_put_data, call_opts, put_opts, strike_range)

    - *_opts* are lists of meta dicts aligned by index with *_data* lists.
    - *_data* rows are the stored price dicts containing fields like
      'ask_price','bid_price','mid_price','close_price','trade_price', etc.

    We will try the requested expiration; if either side is missing, we walk
    expirations (forward weeks, then backward) to find one with BOTH sides,
    mirroring dailytrade.py.
    """
    # Step 1: check/request chain at desired expiration
    # (We do not use `call_put` to filter returns; both sides are provided.)

    # Attempt direct
    call_syms = stored_option_chain.get(ticker, {}).get(expiration_str, {}).get(as_of_str, {}).get("call") or {}
    put_syms  = stored_option_chain.get(ticker, {}).get(expiration_str, {}).get(as_of_str, {}).get("put") or {}

    # If either side is missing, attempt to fetch from Polygon via the provided client
    if (not call_syms or not put_syms) and client is not None:
        try:
            reqs = [(expiration_str, as_of_str, "call"), (expiration_str, as_of_str, "put")]
            print(f"[DEBUG] Fetching missing chain data for {ticker} exp {expiration_str} asof {as_of_str} @ {datetime.utcnow().isoformat()}")
            chain_data = await client.get_option_chains_batch_async(
                ticker, reqs, force_update=force_update
            )
            print(f"[DEBUG] Fetched missing chain data for {ticker} exp {expiration_str} asof {as_of_str} @ {datetime.utcnow().isoformat()}")
            call_syms = (
                chain_data.get(ticker, {})
                .get(expiration_str, {})
                .get(as_of_str, {})
                .get("call")
                or call_syms
            )
            put_syms = (
                chain_data.get(ticker, {})
                .get(expiration_str, {})
                .get(as_of_str, {})
                .get("put")
                or put_syms
            )
        except Exception:
            # Swallow network/client errors and fall back to stored data
            pass

    # If still missing, walk backward day-by-day (preferring Fridays) and
    # attempt to pull chains for nearby expirations, mirroring the reference
    # dailytrade logic.
    if not call_syms or not put_syms:
        try:
            exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
            as_of_date = datetime.strptime(as_of_str, "%Y-%m-%d").date()
        except Exception:
            exp_date = None
            as_of_date = None

        counter = 0
        current = exp_date
        while (
            current is not None
            and as_of_date is not None
            and (not call_syms or not put_syms)
            and current > as_of_date
            and counter < 30
        ):
            counter += 1
            current -= timedelta(days=1)
            # After the first shift, only consider Friday expirations
            if counter > 1 and current.weekday() != 4:
                continue
            exp_cand = current.strftime("%Y-%m-%d")
            call_syms = (
                stored_option_chain.get(ticker, {})
                .get(exp_cand, {})
                .get(as_of_str, {})
                .get("call")
                or {}
            )
            put_syms = (
                stored_option_chain.get(ticker, {})
                .get(exp_cand, {})
                .get(as_of_str, {})
                .get("put")
                or {}
            )
            if (not call_syms or not put_syms) and client is not None:
                try:
                    reqs = [(exp_cand, as_of_str, "call"), (exp_cand, as_of_str, "put")]
                    await client.get_option_chains_batch_async(
                        ticker, reqs, force_update=force_update
                    )
                    call_syms = (
                        stored_option_chain.get(ticker, {})
                        .get(exp_cand, {})
                        .get(as_of_str, {})
                        .get("call")
                        or {}
                    )
                    put_syms = (
                        stored_option_chain.get(ticker, {})
                        .get(exp_cand, {})
                        .get(as_of_str, {})
                        .get("put")
                        or {}
                    )
                except Exception:
                    pass
            if call_syms and put_syms:
                expiration_str = exp_cand
                break

    exp_chosen = expiration_str
    if not call_syms or not put_syms:
        chosen_exp, calls_fb, puts_fb = find_available_expiration(
            ticker=ticker, as_of=as_of_str, target_expiration=expiration_str
        )
        if not chosen_exp:
            # return empty shapes
            return [], [], [], [], None
        exp_chosen = chosen_exp
        call_syms = calls_fb or {}
        put_syms  = puts_fb or {}

    # Pull aligned price rows from stored_option_price
    pf_bucket = stored_option_price.get(ticker, {}).get(as_of_str, {})  # strike -> exp -> type -> fields

    # Sort strikes as in legacy: calls asc, puts desc
    call_strikes = sorted(call_syms.keys())
    put_strikes  = sorted(put_syms.keys(), reverse=True)

    def _row_for(strike: float, opt_type: str) -> dict:
        return (pf_bucket.get(strike, {}).get(exp_chosen, {}).get(opt_type, {}) or {}).copy()

    # Build meta lists (aligned) and data lists
    call_opts = [{"strike_price": float(k), "expiration_date": exp_chosen, "option_ticker": call_syms[k]} for k in call_strikes]
    put_opts  = [{"strike_price": float(k), "expiration_date": exp_chosen, "option_ticker": put_syms[k]}  for k in put_strikes]

    all_call_data = [_row_for(k, "call") for k in call_strikes]
    all_put_data  = [_row_for(k, "put")  for k in put_strikes]

    # ------------------------------------------------------------------
    # Ensure each option has pricing; fetch missing premiums from Polygon
    # ------------------------------------------------------------------
    pf = _price_field()
    reqs: list[dict[str, Any]] = []
    call_missing_idx: list[int] = []
    put_missing_idx: list[int] = []

    for i, (opt, data) in enumerate(zip(call_opts, all_call_data)):
        val = data.get(pf) if isinstance(data, dict) else None
        if force_update or not val or float(val) <= 0.0:
            reqs.append({
                "strike_price": opt["strike_price"],
                "call_put": "call",
                "expiration_date": opt["expiration_date"],
                "quote_timestamp": as_of_str,
                "option_ticker": opt["option_ticker"],
            })
            call_missing_idx.append(i)

    for i, (opt, data) in enumerate(zip(put_opts, all_put_data)):
        val = data.get(pf) if isinstance(data, dict) else None
        if force_update or not val or float(val) <= 0.0:
            reqs.append({
                "strike_price": opt["strike_price"],
                "call_put": "put",
                "expiration_date": opt["expiration_date"],
                "quote_timestamp": as_of_str,
                "option_ticker": opt["option_ticker"],
            })
            put_missing_idx.append(i)

    if reqs and client is not None:
        try:
            fetched = await client.get_option_prices_batch_async(ticker, reqs)
        except Exception:
            fetched = []
        # Map results back to placeholders
        j = 0
        for idx in call_missing_idx:
            all_call_data[idx] = fetched[j] if j < len(fetched) else {}
            j += 1
        for idx in put_missing_idx:
            all_put_data[idx] = fetched[j] if j < len(fetched) else {}
            j += 1

    # Drop options still lacking the required premium
    def _valid(prem: dict) -> bool:
        try:
            return float(prem.get(pf, 0.0)) > 0.0
        except Exception:
            return False

    call_filtered = [(opt, prem) for opt, prem in zip(call_opts, all_call_data) if _valid(prem)]
    put_filtered  = [(opt, prem) for opt, prem in zip(put_opts,  all_put_data)  if _valid(prem)]

    call_opts, all_call_data = zip(*call_filtered) if call_filtered else ([], [])
    put_opts,  all_put_data  = zip(*put_filtered)  if put_filtered  else ([], [])
    call_opts, all_call_data = list(call_opts), list(all_call_data)
    put_opts,  all_put_data  = list(put_opts),  list(all_put_data)

    # Compute strike range for debugging
    smin = smax = None
    all_strikes = [opt["strike_price"] for opt in call_opts] + [opt["strike_price"] for opt in put_opts]
    if all_strikes:
        smin = float(min(all_strikes))
        smax = float(max(all_strikes))
    strike_range = (smin, smax) if smin is not None else None

    return all_call_data, all_put_data, call_opts, put_opts, strike_range
