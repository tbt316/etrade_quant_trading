# polygonio/recursive_backtest.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, Iterable, List, Optional, Tuple

class _DebugCounters:
    def __init__(self):
        self.days_total = 0
        self.days_no_price = 0
        self.days_skipped_earnings = 0
        self.expiries_considered = 0
        self.expiries_skipped_no_chain = 0
        self.expiries_skipped_no_strikes = 0
        self.positions_built = 0
        self.exceptions = 0
    def summary(self):
        return (
            f"[DBG] days={self.days_total} no_price={self.days_no_price} "
            f"earnings_skips={self.days_skipped_earnings} expiries={self.expiries_considered} "
            f"no_chain={self.expiries_skipped_no_chain} no_strikes={self.expiries_skipped_no_strikes} "
            f"positions_built={self.positions_built} exceptions={self.exceptions}"
        )


from .cache_io import load_stored_option_data
from .config import get_settings, PREMIUM_FIELD_MAP
from .prices import get_historical_prices
from .earnings import get_earnings_dates
from .market_calendar import list_expiries  # maps your old get_all_weekdays
from .poly_client import PolygonAPIClient
from .chains import pull_option_chain_data
from .pricing import interpolate_option_price, calculate_delta
from .cache_io import stored_option_price, save_stored_option_data
from .symbols import convert_polygon_to_etrade_ticker
from .paths import ROOT_DIR

# --- helpers for PCS selection ---
def _mid_from_quotes(d: dict) -> float | None:
    bid = d.get("bid_price") or d.get("bid")
    ask = d.get("ask_price") or d.get("ask")
    try:
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        pass
    return None

def _price_from_data(d: dict, premium_field: str) -> float | None:
    if not isinstance(d, dict):
        return None
    v = d.get(premium_field)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    # fallbacks
    for k in ("mid_price", "mark_price", "last_price", "close_price", "trade_price", "price"):
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return _mid_from_quotes(d)
# --- end helpers ---

def _safe_premium(d: dict, preferred_key: str) -> float | None:
    if not isinstance(d, dict):
        return None
    for k in (preferred_key, "trade_price", "mid_price", "close_price", "ask_price", "bid_price"):
        v = d.get(k)
        try:
            if v is None: 
                continue
            fv = float(v)
            if fv > 0:
                return fv
        except Exception:
            continue
    return None

def _probe_put_credit_spread(*, ticker: str, as_of: str, expiry: str, spot: float, put_data: list | None, put_options: list | None, premium_field: str, width: float | int, min_credit: float | None = None, steer: float | None = None) -> None:
    print(f"[DBG-PROBE] PCS probe: {ticker} as_of={as_of} expiry={expiry} spot={spot:g} width={width} min_credit={min_credit} steer={steer}")
    if not put_data or not put_options:
        print("[DBG-PROBE] No put_data/put_options returned.")
        return
    # build list of candidate puts below spot
    puts = []
    for i, opt in enumerate(put_options):
        try:
            k = float(opt.get("strike_price"))
        except Exception:
            continue
        if k >= spot:
            continue
        pd = put_data[i] if i < len(put_data) else {}
        prem = _safe_premium(pd, premium_field)
        if prem is None:
            continue
        puts.append((k, prem, pd))
    puts.sort(key=lambda x: x[0], reverse=True)  # nearest OTM first
    cnt = len(puts)
    ge_025 = sum(1 for _, p, _ in puts if p >= 0.25)
    print(f"[DBG-PROBE] puts<spot={cnt}, with prem>=0.25: {ge_025}")
    print("[DBG-PROBE] top puts (strike@prem):", ", ".join([f"{int(k)}@{prem:.2f}" for k,prem,_ in puts[:8]]))

    if not puts:
        print("[DBG-PROBE] No OTM puts with usable premium.")
        return

    # pick a plausible short put near-the-money OTM
    short_k, short_prem, short_pd = puts[0]
    long_k = short_k - float(width)
    # find long leg in list or closest below desired
    long_candidates = [t for t in puts if abs(t[0] - long_k) < 1e-6] or [t for t in puts if t[0] < short_k and t[0] <= long_k]
    if not long_candidates:
        print(f"[DBG-PROBE] Could not find long leg at or below {long_k}. Short={short_k}.")
        return
    long_k, long_prem, long_pd = long_candidates[-1]  # furthest down if exact not found
    credit = max(0.0, (short_prem - long_prem))
    eff_width = float(short_k - long_k)
    print(f"[DBG-PROBE] probe pair: short={int(short_k)}@{short_prem:.2f}, long={int(long_k)}@{long_prem:.2f}, width={eff_width:.0f}, credit≈{credit:.2f}")
    if min_credit is not None:
        ok = credit >= float(min_credit)
        print(f"[DBG-PROBE] credit {'meets' if ok else 'below'} min_credit={min_credit}")


from strategies.strategies import sides_for_trade_type, get_strategy


# -------------------------------
# Public orchestration API
# -------------------------------

@dataclass
class RecursionConfig:
    ticker: str
    global_start_date: str  # "YYYY-MM-DD"
    global_end_date: str    # "YYYY-MM-DD"
    trade_type: str
    # core
    expiring_weekday: str = "Friday"
    expiring_wks: int = 1
    contract_qty: int = 1
    # dailytrade.py parity params
    iron_condor_width: Optional[float] = None
    target_premium_otm: Optional[float] = None
    target_delta: Optional[float] = None      # accepted; not used for strike selection
    target_steer: float = 0.0
    stop_profit_percent: Optional[float] = None
    stop_loss_action: Optional[str] = None
    vix_threshold: Optional[float] = None
    vix_correlation: Optional[str] = None


def monthly_recursive_backtest(
    ticker: str,
    global_start_date: str,
    global_end_date: str,
    *,
    trade_type: str,
    expiring_weekday: str = "Friday",
    expiring_wks: int = 1,
    contract_qty: int = 1,
    iron_condor_width: float | None = None,
    target_premium_otm: float | None = None,
    target_delta: float | None = None,
    target_steer: float = 0.0,
    stop_profit_percent: float | None = None,
    stop_loss_action: str | None = None,
    vix_threshold: float | None = None,
    vix_correlation: str | None = None,
) -> Dict[str, Any]:
    """
    Replacement for your old monthly_recursive_backtest(...).
    Splits the global window and for each slice runs the per-day engine.

    NOTE: Adjust the slicing policy below to match your original:
          (lookback windows, validation hops, etc). For now we run one pass
          over the whole range to keep behavior simple and identical to your
          daily engine.
    """
    s = get_settings()

    # ---- If you previously used lookback + validation hops, rebuild that here.
    # For now: single call covering the entire [start, end] range.
    cfg = RecursionConfig(
        iron_condor_width=iron_condor_width,
        target_premium_otm=target_premium_otm,
        target_delta=target_delta,
        target_steer=target_steer,
        stop_profit_percent=stop_profit_percent,
        stop_loss_action=stop_loss_action,
        vix_threshold=vix_threshold,
        vix_correlation=vix_correlation,
        ticker=ticker,
        global_start_date=global_start_date,
        global_end_date=global_end_date,
        trade_type=trade_type,
        expiring_weekday=expiring_weekday,
        expiring_wks=expiring_wks,
        contract_qty=contract_qty,
    )

    # Run the async per-day engine
    print("[DEBUG] Launching async backtest...")
    print("[DEBUG] launching async backtest_options_sync_or_async...")
    results = asyncio.run(
        backtest_options_sync_or_async(cfg)  # identical role to your old function
    )

    # If you used to persist monthly results to a PKL, do it here.
    # from .paths import get_monthly_backtest_file
    # out_file = get_monthly_backtest_file(ticker, global_start_date, global_end_date)
    # with open(out_file, "wb") as f:
    #     pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[DEBUG] Finished monthly_recursive_backtest, returning results")
    print("[DEBUG] monthly_recursive_backtest done. Results keys:", (list(results.keys()) if isinstance(results, dict) else type(results)))
    return results

# Compatibility wrapper for list_expiries signature drift
def _list_expiries_compat(*, weekday, start_date, end_date):

    """Build a trading_dates_df compatible with market_calendar.list_expiries().
    We avoid depending on any helper methods that may not exist on TradingCalendar
    by constructing the schedule directly with pandas_market_calendars.
    """
    import pandas as pd
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('NYSE')  # default
        sched = cal.schedule(start_date=start_date, end_date=end_date)
        df = sched.reset_index().rename(columns={'index': 'date'})
        # Normalize to naive datetimes/dates as expected
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        if 'market_open' in df.columns:
            df['market_open'] = pd.to_datetime(df['market_open']).dt.tz_localize(None)
        if 'market_close' in df.columns:
            df['market_close'] = pd.to_datetime(df['market_close']).dt.tz_localize(None)
        trading_dates_df = df
    except Exception:
        # Fallback: business days only; not as precise as the official schedule
        trading_dates_df = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='B')})

    return list_expiries(
        weekday=weekday,
        start_date=start_date,
        end_date=end_date,
        trading_dates_df=trading_dates_df,
    )
# -------------------------------

def _target_expiry_compat(*, weekday: str, as_of: date, weeks: int) -> Optional[datetime]:
    import pandas as pd
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar('NYSE')
        horizon = as_of + timedelta(days=weeks * 7 + 20)
        sched = cal.schedule(start_date=as_of, end_date=horizon)
        trading_days = (
            pd.to_datetime(sched.index).tz_localize(None).date.tolist()
        )
    except Exception:
        trading_days = pd.bdate_range(start=as_of, end=as_of + timedelta(days=weeks * 7 + 20)).date.tolist()

    weekday_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}
    w = weekday_map.get(weekday, 4)
    probe = as_of + timedelta(weeks=weeks)
    for d in trading_days:
        if d >= probe and d.weekday() == w:
            return datetime.combine(d, datetime.min.time())
    return None
# Daily loop (async) — main worker
# -------------------------------

async def backtest_options_sync_or_async(cfg: RecursionConfig) -> Dict[str, Any]:

    """
    Mirrors your original inner engine:
      - iterate pricing dates
      - enumerate target expirations
      - fetch chains/quotes
      - select strikes (PASTE YOUR LOGIC)
      - compute premiums (interpolate if needed)
      - assemble positions via strategies layer (no logic change)
      - simulate PnL / exit (PASTE YOUR LOGIC)
    Returns whatever aggregate structure you expect downstream (dict here).
    """

    # Preload on-disk buffered caches into memory so we fetch only true misses
    try:
        cache_dir = None
        if "_lite" in (cfg.trade_type or ""):
            cache_dir = (ROOT_DIR / "polygon_api_option_data").resolve()
        load_stored_option_data(cfg.ticker, cache_dir=cache_dir)
        print(f"[DEBUG] preloaded cached PKLs for {cfg.ticker}")
    except Exception as e:
        print(f"[DEBUG] preload skipped for {cfg.ticker}: {e}")
    s = get_settings()
    premium_field = PREMIUM_FIELD_MAP.get(s.premium_price_mode, "trade_price")

    # 0) Prep: underlying history (for spot/MA/vol, same as your original)
    hist = get_historical_prices(
        cfg.ticker, cfg.global_start_date, cfg.global_end_date,
        vol_lookback=5, data_source="yfinance"  # matches your default path
    )
    if hist.empty:
        return {"error": "no_price_data", "ticker": cfg.ticker}

    # Convert to a dict keyed by date for quick lookup
    close_by_date = {
        (d.date() if hasattr(d, "date") else d): float(px)
        for d, px in zip(hist["date"], hist["close"])
    }

    # 1) Earnings filter if desired
    earnings_dates: Optional[set[date]] = None
    print(f"[DEBUG] skip_earnings={s.skip_earnings}")
    if s.skip_earnings:
        earnings_dates = get_earnings_dates(cfg.ticker, cfg.global_start_date, cfg.global_end_date)

    # 2) Async client
    print("[DEBUG] creating PolygonAPIClient...")
    async with PolygonAPIClient(api_key=s.polygon_api_key) as client:
        print("[DEBUG] PolygonAPIClient ready")
        dbg = _DebugCounters()
        # Collect results
        daily_positions: List[Dict[str, Any]] = []
        daily_pnls: List[Dict[str, Any]] = []
        open_positions: List[Dict[str, Any]] = []

        def _pos_open_date(pos: Dict[str, Any]) -> Optional[date]:
            """Return the date a position was opened, or None if unknown."""
            od = pos.get("position_open_date") or pos.get("opened_at")
            if isinstance(od, datetime):
                return od.date()
            if isinstance(od, date):
                return od
            if isinstance(od, str):
                try:
                    return datetime.strptime(od, "%Y-%m-%d").date()
                except Exception:
                    return None
            return None

        # 3) Iterate every pricing day in the window
        start_dt = datetime.strptime(cfg.global_start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(cfg.global_end_date, "%Y-%m-%d").date()
        cur = start_dt
        print(f"[DEBUG] pricing day loop: start={start_dt}, end={end_dt}")

        # Strategy -> which sides (call/put/both)
        needed_sides = sides_for_trade_type(cfg.trade_type)
        call_put_flag = "call_put_both" if set(needed_sides) == {"call", "put"} else needed_sides[0]

        # Cache of trading calendar dates (if you have one already, reuse)
        # Otherwise, we'll derive expiries per pricing day via list_expiries(...)
        while cur <= end_dt:
            dbg.days_total += 1
            pcs_ts = None
            # skip non-price days
            spot = close_by_date.get(cur)
            if (cur.toordinal() - start_dt.toordinal()) % 20 == 0:
                print(f"[DEBUG] day={cur} spot={spot}")
            if spot is None:
                dbg.days_no_price += 1
                cur += timedelta(days=1)
                continue

            # skip earnings if configured
            if earnings_dates and cur in earnings_dates:
                dbg.days_skipped_earnings += 1
                cur += timedelta(days=1)
                continue

            as_of_str = cur.strftime("%Y-%m-%d")

            # 3a) Choose a single target expiration (Friday/Wed cadence)
            print(f"[DEBUG] computing expiries for as_of={as_of_str}")
            target_dt = _target_expiry_compat(
                weekday=cfg.expiring_weekday, as_of=cur, weeks=cfg.expiring_wks
            )
            if not target_dt:
                cur += timedelta(days=1)
                continue
            expiration_str = target_dt.strftime("%Y-%m-%d")

            # Determine whether a position with a different expiration was opened today
            already_open = any(
                (_pos_open_date(p) == cur) and p.get("expiration") != expiration_str
                for p in open_positions
            )

            # 3b) Pull chains + maybe batch fetch missing quotes
            # _target_expiry_compat returns a datetime; convert to date for comparisons
            this_exp = target_dt.date()
            counter = 0
            while True:
                expiration_str = this_exp.strftime("%Y-%m-%d")
                print(
                    f"[DEBUG] pulling option chain: expiry={expiration_str}, as_of={as_of_str}, side={call_put_flag}"
                )
                call_data, put_data, call_opts, put_opts, strike_range = await pull_option_chain_data(
                    ticker=cfg.ticker,
                    call_put=call_put_flag,
                    expiration_str=expiration_str,
                    as_of_str=as_of_str,
                    close_price=spot,
                    client=client,
                    force_otm=False,
                    force_update=False,
                )
                print(
                    f"[DEBUG] chain pulled: calls={len(call_data) if call_data else 0}, puts={len(put_data) if put_data else 0}, strike_range={strike_range}"
                )
                dbg.expiries_considered += 1
                need_call = "call" in call_put_flag
                need_put = "put" in call_put_flag
                have_call = bool(call_data) or not need_call
                have_put = bool(put_data) or not need_put
                if have_call and have_put:
                    break
                counter += 1
                # Compare dates to avoid type mismatch when loop index is a `date`
                if this_exp <= cur or counter > 30:
                    break
                this_exp -= timedelta(days=1)
                if counter > 1 and this_exp.weekday() != 4:
                    continue

            if ("call" in call_put_flag and not call_data) or ("put" in call_put_flag and not put_data):
                dbg.expiries_skipped_no_chain += 1
                cur += timedelta(days=1)
                print(f"[DEBUG] skipping expiry {expiration_str}: no chain data")
                continue

            position = None
            if not already_open:
                sc_k = lc_k = sp_k = lp_k = None
                dbg_sel = {'puts_total': 0, 'puts_below_spot': 0, 'meets_premium': 0, 'chosen_short_put': None, 'chosen_long_put': None}   # float
                sc_p = lc_p = sp_p = lp_p = None   # float
                have_short_call = have_long_call = False
                have_short_put = have_long_put = False

                # ========== PASTE BLOCK 1: STRIKE SELECTION (unchanged) ==========
                # Use your existing strike selection logic here to compute:
                #   sc_k, sc_p  (short call strike/premium)
                #   lc_k, lc_p  (long  call strike/premium)
                #   sp_k, sp_p  (short put  strike/premium)
                #   lp_k, lp_p  (long  put  strike/premium)
                #
                # Notes:
                # - If your chosen premium is missing (0 or None), call interpolate_option_price()
                #   to estimate (same guards/flags as your old code).
                # - Examples for interpolation:
                #
                # sc_p = sc_p or (await interpolate_option_price(
                #     ticker=cfg.ticker,
                #     close_price_today=spot,
                #     strike_price_to_interpolate=sc_k,
                #     option_type="call",
                #     expiration_date=expiration_str,
                #     pricing_date=as_of_str,
                #     stored_option_price=stored_option_price,
                #     premium_field=premium_field,
                #     price_interpolate_flag=s.price_interpolate,
                #     client=client,
                # ))
                #
                # Compute deltas if you need them for filters:
                # calculate_delta(cfg.ticker, as_of_str, expiration_str, "call", force_delta_update=False)
                # calculate_delta(cfg.ticker, as_of_str, expiration_str, "put",  force_delta_update=False)
                #
                # === BEGIN PCS selection using (put_opts, put_data); target_prem_otm = target PRICE ===
                try:
                    # settings decide which premium field to read from the data array
                    s = get_settings()
                    premium_field = PREMIUM_FIELD_MAP.get(s.premium_price_mode, "trade_price")

                    # Build candidates by zipping meta (put_opts) with data (put_data)
                    candidates = []
                    _metas = put_opts or []      # meta rows: {'strike_price', 'expiration_date', 'option_ticker', ...}
                    _datas = put_data or []      # price rows aligned by index: {'trade_price'/'mid_price'/...}
                    if not _metas:
                        print(f"[DBG] no put_opts for {cfg.ticker} {as_of_str}->{expiration_str} (strike_range={strike_range})")

                    for i, meta in enumerate(_metas):
                        try:
                            k = float(meta.get("strike_price"))
                        except Exception:
                            print("[DBG] skipping put meta with invalid strike_price:", meta)
                            continue
                        d = _datas[i] if i < len(_datas) else {}
                        price = _price_from_data(d, premium_field)
                        candidates.append({"strike": k, "price": price, "meta": meta, "data": d})

                    if candidates:
                        kmin = min(x["strike"] for x in candidates)
                        kmax = max(x["strike"] for x in candidates)
                        priced = sum(1 for x in candidates if x["price"] is not None)
                        print(f"[DBG] put candidates: n={len(candidates)} priced={priced} strikes=[{kmin},{kmax}] spot={spot} mode={s.premium_price_mode}")

                    # OTM only with a usable price
                    otm_puts = [r for r in candidates if r["strike"] < spot and (r["price"] is not None)]
                    if not otm_puts:
                        print(f"[DBG] no OTM put candidates w/ price for {cfg.ticker} {as_of_str}->{expiration_str} (spot={spot})")
                    else:
                        # knobs
                        width = float(getattr(cfg, "iron_condor_width", 10.0) or 10.0)

                        # 1) PRICE target (target_prem_otm == desired option price)
                        target_price = None
                        if getattr(cfg, "target_premium_otm", None) is not None:
                            try:
                                target_price = float(cfg.target_premium_otm)
                            except Exception:
                                target_price = None

                        # 2) DELTA target (optionally steered)
                        target_delta = None
                        if getattr(cfg, "target_delta", None) is not None:
                            try:
                                target_delta = float(cfg.target_delta)
                            except Exception:
                                target_delta = None
                        if target_delta is not None and getattr(cfg, "target_steer", None):
                            try:
                                target_delta *= float(cfg.target_steer)
                            except Exception:
                                pass
                        if target_delta is not None:
                            target_delta = max(0.01, min(0.49, abs(target_delta)))
                            # get per-strike delta (map by strike_price)
                            try:
                                delta_map = calculate_delta(cfg.ticker, as_of_str, expiration_str, "put", force_delta_update=False)
                            except Exception:
                                delta_map = {}
                        else:
                            delta_map = {}

                        # Build scored list
                        scored = []
                        for r in otm_puts:
                            k = r["strike"]; price = r["price"]
                            otm_pct = (spot - k) / spot if spot else 0.0
                            d = None
                            if delta_map:
                                d = delta_map.get(round(k, 2)) or delta_map.get(k)
                                try:
                                    d = abs(float(d)) if d is not None else None
                                except Exception:
                                    d = None
                            scored.append({"strike": k, "price": price, "otm_pct": otm_pct, "delta": d})

                        # Choose short put
                        sp = None
                        reason = ""
                        if target_price is not None:
                            cands = [x for x in scored if x["price"] is not None]
                            if cands:
                                sp = min(cands, key=lambda x: abs(x["price"] - target_price))
                                reason = f"price≈{sp['price']:.3f} vs target {target_price:.3f}"

                        if sp is None and target_delta is not None:
                            cands = [x for x in scored if x["delta"] is not None]
                            if cands:
                                sp = min(cands, key=lambda x: abs(x["delta"] - target_delta))
                                reason = f"delta≈{sp['delta']:.3f} vs target {target_delta:.3f}"

                        if sp is None:
                            # fallback ~10% OTM
                            sp = min(scored, key=lambda x: abs(x["otm_pct"] - 0.10))
                            reason = f"fallback OTM≈{sp['otm_pct']:.2%}"

                        sp_k, sp_p = sp["strike"], sp["price"]

                        # Long put: aim width lower; nearest available ≤ target with price
                        lp_target = sp_k - width
                        under = [x for x in scored if x["strike"] <= lp_target and x["price"] is not None]

                        # If nothing at/below target, pick the CLOSEST strike strictly BELOW the short
                        if not under:
                            under = [x for x in scored if x["strike"] < sp_k and x["price"] is not None]

                        lp_k = lp_p = None
                        if under:
                            lp = min(under, key=lambda x: abs(x["strike"] - lp_target))
                            lp_k, lp_p = lp["strike"], lp["price"]

                        # FINAL sanity: long must be strictly below short; otherwise try the best available below short
                        if lp_k is None or lp_k >= sp_k:
                            lower = [x for x in scored if x["strike"] < sp_k and x["price"] is not None]
                            if lower:
                                # choose the highest strike below short (closest, ensures positive width)
                                best = max(lower, key=lambda x: x["strike"])
                                lp_k, lp_p = best["strike"], best["price"]
                            else:
                                # no valid long; skip building the spread for this day
                                have_long_put = False
                                have_short_put = sp_k is not None and sp_p is not None
                                print(f"[DBG] PCS skip: no long put below SP {sp_k} available; strikes range min={min(x['strike'] for x in scored):g}")
                            # only set have_long_put if we ended up with a valid one
                        if lp_k is not None and lp_k < sp_k:
                            have_long_put = True
                        else:
                            have_long_put = False

                        have_short_put = (sp_k is not None and sp_p is not None)

                        ts_now = datetime.utcnow().isoformat()
                        print(
                            f"[DBG] PCS {cfg.ticker} {as_of_str}->{expiration_str}: "
                            f"SP {sp_k} @ {sp_p} ({reason}); "
                            f"LP target {lp_target} → {lp_k} @ {lp_p}; "
                            f"width={(sp_k - lp_k) if (lp_k is not None and sp_k is not None) else 'NA'} @ {ts_now}"
                        )
                        pcs_ts = time.perf_counter()
                except Exception as e:
                    print(f"[DBG] PCS selection exception: {e}")
                # === END PCS selection using (put_opts, put_data); target_prem_otm = target PRICE ===

                # 3c) Build the position using strategies (no logic change to shape/margin)
                strat = get_strategy(cfg.trade_type)
                build_kwargs: Dict[str, Any] = dict(
                    underlying=cfg.ticker,
                    expiration=expiration_str,
                    opened_at=as_of_str,
                    qty=int(cfg.contract_qty),
                )
                if "call" in needed_sides:
                    if have_short_call and sc_k is not None and sc_p:
                        build_kwargs["short_call"] = (float(sc_k), float(sc_p))
                    if have_long_call and lc_k is not None and lc_p:
                        build_kwargs["long_call"] = (float(lc_k), float(lc_p))
                if "put" in needed_sides:
                    if have_short_put and sp_k is not None and sp_p:
                        build_kwargs["short_put"] = (float(sp_k), float(sp_p))
                    if have_long_put and lp_k is not None and lp_p:
                        build_kwargs["long_put"] = (float(lp_k), float(lp_p))

                    # Strategy may raise if a required leg is missing; guard as you did before
                    try:
                        position = strat.build_position(**build_kwargs).to_dict()
                    except Exception as e:
                        # skip this date/expiry if legs incomplete
                        position = None

                if position is not None:
                    daily_positions.append(position)
                    dbg.positions_built += 1

            # ========== PASTE BLOCK 2: P&L / EXIT / ACCOUNTING (unchanged) ==========
            # Here paste your existing code that:
            #   - computes cashflows (credit/debit)
            #   - tracks margin requirements
            #   - exits early if profit target / stop is hit
            #   - realizes P&L at expiration or at exit date
            #
            # Append a summary dict to daily_pnls (or however you used to record it).
            #
            else:
                print(f"[DEBUG] day={cur} skipped: already have open position for expiry {expiration_str}")
# ---> BEGIN YOUR P&L / EXIT LOGIC
            # --- Early exit & MTM logic (inspired by polygonio_dailytrade.py) ---
            # Normalize convenience
            t = cfg.ticker.upper()
            premium_field = PREMIUM_FIELD_MAP.get(get_settings().premium_price_mode, "trade_price")

            def _get_price_from_store(side: str, strike: float) -> float | None:
                try:
                    d1 = stored_option_price.get(t, {}).get(as_of_str, {})
                    d2 = d1.get(round(float(strike), 2), {})
                    d3 = d2.get(expiration_str, {})
                    leaf = d3.get(side, {}) if isinstance(d3, dict) else {}
                    v = leaf.get(premium_field)
                    if v is not None:
                        return float(v)
                    bid = leaf.get("bid_price"); ask = leaf.get("ask_price")
                    if bid is not None and ask is not None:
                        return (float(bid) + float(ask)) / 2.0
                except Exception:
                    return None
                return None

            # Helper to append/update open_positions on open day
            def _register_open_position(position_dict: Dict[str, Any]):
                """Register a newly-opened position in the open list.

                ``position_dict`` is already appended to ``daily_positions`` above.
                To ensure any later mutations (e.g. call/put closure fields) are
                reflected in the final ``positions`` output, we must operate on
                the same dictionary object rather than a copy.  Otherwise the
                open/close logic below would update a separate object and the
                caller would never see the enriched fields.
                """

                # Use the original dict so that open_positions and daily_positions
                # share the same reference.
                pos = position_dict

                pos.setdefault("position_open_date", datetime.strptime(as_of_str, "%Y-%m-%d"))
                pos.setdefault("call_closed_by_stop", False)
                pos.setdefault("put_closed_by_stop", False)
                pos.setdefault("call_closed_date", None)
                pos.setdefault("put_closed_date", None)

                # Extract per-leg info
                for leg in position_dict.get("legs", []):
                    side = leg.get("side"); action = leg.get("action")
                    strike = float(leg.get("strike", 0.0)); prem = float(leg.get("premium", 0.0))
                    if side == "call" and action == "sell":
                        pos["call_strike_sold"] = strike
                        pos["short_call_prem_open"] = prem
                    if side == "call" and action == "buy":
                        pos["call_strike_bought"] = strike
                        pos["long_call_prem_open"] = prem
                    if side == "put" and action == "sell":
                        pos["put_strike_sold"] = strike
                        pos["short_put_prem_open"] = prem
                    if side == "put" and action == "buy":
                        pos["put_strike_bought"] = strike
                        pos["long_put_prem_open"] = prem

                open_positions.append(pos)

            # Register this newly-opened position, if any
            if position is not None:
                _register_open_position(position)

            # Evaluate ALL open positions (including the one we just opened) for early exits or expiration
            # DTE window: 'expiring soon' = half of configured expiring_wks
            dte_soon_days = int((cfg.expiring_wks or 1) * 7 / 2)

            still_open: List[Dict[str, Any]] = []
            for pos in open_positions:
                try:
                    exp_dt = datetime.strptime(pos.get("expiration"), "%Y-%m-%d").date()
                except Exception:
                    # if expiration is already a date object
                    exp_dt = pos.get("expiration")
                    if isinstance(exp_dt, str):
                        try:
                            exp_dt = datetime.strptime(exp_dt, "%Y-%m-%d").date()
                        except Exception:
                            continue
                # Skip invalid
                if exp_dt is None:
                    continue

                # Entry credits (dollars)
                entry_credit_call = 0.0
                if pos.get("short_call_prem_open") and pos.get("long_call_prem_open") is not None:
                    entry_credit_call = (float(pos.get("short_call_prem_open", 0.0)) - float(pos.get("long_call_prem_open", 0.0))) * 100.0
                entry_credit_put = 0.0
                if pos.get("short_put_prem_open") and pos.get("long_put_prem_open") is not None:
                    entry_credit_put = (float(pos.get("short_put_prem_open", 0.0)) - float(pos.get("long_put_prem_open", 0.0))) * 100.0

                # If expired today or earlier: settle at intrinsic
                if exp_dt <= cur:
                    close_price = spot if spot is not None else 0.0
                    # CALL vertical payoff (loss positive in points)
                    if pos.get("short_call_prem_open", 0) and pos.get("call_closed_date") is None:
                        sc_loss = max(close_price - pos.get("call_strike_sold", 0.0), 0.0)
                        lc_gain = max(close_price - pos.get("call_strike_bought", 0.0), 0.0)
                        call_loss_final = sc_loss - lc_gain  # points
                        pos["call_closed_date"] = cur
                        pos["call_closed_by_stop"] = True
                        pos["call_closed_profit"] = entry_credit_call - (call_loss_final * 100.0)
                    if pos.get("short_put_prem_open", 0) and pos.get("put_closed_date") is None:
                        sp_loss = max(pos.get("put_strike_sold", 0.0) - close_price, 0.0)
                        lp_gain = max(pos.get("put_strike_bought", 0.0) - close_price, 0.0)
                        put_loss_final = sp_loss - lp_gain  # points
                        pos["put_closed_date"] = cur
                        pos["put_closed_by_stop"] = True
                        pos["put_closed_profit"] = entry_credit_put - (put_loss_final * 100.0)
                    # drop from open list
                    continue

                # Otherwise, try an early exit
                dte = (exp_dt - cur).days
                expiring_soon = dte <= dte_soon_days

                # CALL leg close cost (points)
                close_call_cost = None
                if pos.get("short_call_prem_open", 0) and not pos.get("call_closed_by_stop", False):
                    sc = pos.get("call_strike_sold"); lc = pos.get("call_strike_bought")
                    sc_p = _get_price_from_store("call", sc)
                    lc_p = _get_price_from_store("call", lc) if lc is not None else 0.0

                    if sc_p is None or lc_p is None:
                        try:
                            from .pricing import interpolate_option_price as _interp
                            exp_s = exp_dt.strftime("%Y-%m-%d")  # exp_dt is a date
                            # IMPORTANT: await the coroutine in this async function
                            if sc_p is None:
                                sc_p = await _interp(
                                    t, float(spot or 0.0), float(sc), "call",
                                    exp_s, as_of_str,
                                    premium_field=premium_field,
                                    price_interpolate_flag=get_settings().price_interpolate,
                                    client=client
                                )
                            if lc is not None and lc_p is None:
                                lc_p = await _interp(
                                    t, float(spot or 0.0), float(lc), "call",
                                    exp_s, as_of_str,
                                    premium_field=premium_field,
                                    price_interpolate_flag=get_settings().price_interpolate,
                                    client=client
                                )
                        except Exception:
                            pass

                    # Only compute if both legs have numbers
                    try:
                        if sc_p is not None and lc_p is not None:
                            close_call_cost = float(sc_p) - float(lc_p)
                    except Exception:
                        close_call_cost = None

                # PUT leg close cost (points)
                close_put_cost = None
                if pos.get("short_put_prem_open", 0) and not pos.get("put_closed_by_stop", False):
                    sp = pos.get("put_strike_sold"); lp = pos.get("put_strike_bought")
                    sp_p = _get_price_from_store("put", sp)
                    lp_p = _get_price_from_store("put", lp) if lp is not None else 0.0

                    if sp_p is None or lp_p is None:
                        try:
                            from .pricing import interpolate_option_price as _interp
                            exp_s = exp_dt.strftime("%Y-%m-%d")
                            if sp_p is None:
                                sp_p = await _interp(
                                    t, float(spot or 0.0), float(sp), "put",
                                    exp_s, as_of_str,
                                    premium_field=premium_field,
                                    price_interpolate_flag=get_settings().price_interpolate,
                                    client=client
                                )
                            if lp is not None and lp_p is None:
                                lp_p = await _interp(
                                    t, float(spot or 0.0), float(lp), "put",
                                    exp_s, as_of_str,
                                    premium_field=premium_field,
                                    price_interpolate_flag=get_settings().price_interpolate,
                                    client=client
                                )
                        except Exception:
                            pass

                    try:
                        if sp_p is not None and lp_p is not None:
                            close_put_cost = float(sp_p) - float(lp_p)
                    except Exception:
                        close_put_cost = None

                # Triggers per leg (mirror dailytrade.py semantics)
                tp = float(cfg.stop_profit_percent) if (cfg.stop_profit_percent not in (None, 0, "0")) else None
                # For now we don't implement hold_to_expiration toggles; always use tp if provided.
                commission = 2 * 0.5  # $1 round trip placeholder

                # CALL leg decision
                if close_call_cost is not None and entry_credit_call > 0 and not pos.get("call_closed_by_stop", False):
                    cc_dollars = close_call_cost * 100.0
                    call_trigger_profit = (tp is not None) and (0 < cc_dollars <= tp * entry_credit_call)
                    call_trigger_loss   = cc_dollars >= 50 * entry_credit_call
                    call_trigger_exp    = (0 < cc_dollars <= 2 * entry_credit_call) and expiring_soon
                    if call_trigger_profit or call_trigger_exp or call_trigger_loss:
                        realised_loss = round(-cc_dollars, 2) - commission
                        pos["call_closed_by_stop"] = True
                        pos["call_closed_date"] = cur
                        pos["call_closed_profit"] = entry_credit_call + realised_loss

                # PUT leg decision
                if close_put_cost is not None and entry_credit_put > 0 and not pos.get("put_closed_by_stop", False):
                    pc_dollars = close_put_cost * 100.0
                    put_trigger_profit = (tp is not None) and (0 < pc_dollars <= tp * entry_credit_put)
                    put_trigger_loss   = pc_dollars >= 50 * entry_credit_put
                    put_trigger_exp    = (0 < pc_dollars <= 2 * entry_credit_put) and expiring_soon
                    if put_trigger_profit or put_trigger_exp or put_trigger_loss:
                        realised_loss = round(-pc_dollars, 2) - commission
                        pos["put_closed_by_stop"] = True
                        pos["put_closed_date"] = cur
                        pos["put_closed_profit"] = entry_credit_put + realised_loss

                # Keep for filtering after evaluating both legs
                still_open.append(pos)

            # Replace open_positions with positions that still have an open leg
            # Treat a leg as closed if it was never opened (no corresponding short premium)
            open_positions = [
                p
                for p in still_open
                if not (
                    (p.get("call_closed_by_stop", False) or p.get("short_call_prem_open") is None)
                    and (p.get("put_closed_by_stop", False) or p.get("short_put_prem_open") is None)
                )
            ]
            
            # bookkeeping row
            pnl_row = {
                "as_of": as_of_str,
                "expiration": expiration_str,
                "trade_type": cfg.trade_type,
                "underlying": cfg.ticker,
                "qty": cfg.contract_qty,
                "spot": spot,
                "open_positions": len(open_positions),
            }
            ts_now = datetime.utcnow().isoformat()
            if pcs_ts is not None:
                elapsed = time.perf_counter() - pcs_ts
                print(
                    f"pnl_row init: {pnl_row} open_positions={len(open_positions)} "
                    f"dt={elapsed:.2f}s @ {ts_now}"
                )
            else:
                print(
                    f"pnl_row init: {pnl_row} open_positions={len(open_positions)} @ {ts_now}"
                )
            daily_pnls.append(pnl_row)
# <--- END YOUR P&L / EXIT LOGIC
# <--- END YOUR P&L / EXIT LOGIC
            # =================================================================

            cur += timedelta(days=1)

    # Optionally persist caches as you go (same as before)
    # try:
    #     print("[DEBUG] saving cached option data...")
    #     cache_dir = None
    #     if "_lite" in (cfg.trade_type or ""):
    #         cache_dir = (ROOT_DIR / "polygon_api_option_data").resolve()
    #     save_stored_option_data(cfg.ticker, cache_dir=cache_dir)
    #     print("[DEBUG] saved.")
    # except Exception:
    #     pass
    
    print(dbg.summary())
    # Return whatever structure you expect downstream
    return {
        "ticker": cfg.ticker,
        "start": cfg.global_start_date,
        "end": cfg.global_end_date,
        "trade_type": cfg.trade_type,
        "positions": daily_positions,
        "pnl": daily_pnls,
        "debug": dbg.__dict__,
    }
