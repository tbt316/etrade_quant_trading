"""Utility CLI to compute Forward Factor (FF) time series for a ticker.

This script fetches the nearest front/back expirations (defaults 60/90 DTE)
using the calendar helpers from ``polygonio.recursive_backtest`` and computes
the forward implied volatility plus the normalized forward factor metric.

Example:

    python -m etrade_quant_trading.scripts.compute_forward_factor \
        --ticker SPY --start 2024-03-01 --end 2025-03-01 --csv spy_ff.csv
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from datetime import datetime
import pandas as pd

from etrade_quant_trading.polygonio.cache_io import (
    load_stored_option_data,
    save_stored_option_data,
    stored_option_chain,
)
from etrade_quant_trading.polygonio.config import get_settings, resolve_premium_field
from etrade_quant_trading.polygonio.option_math import calculate_implied_volatility
from etrade_quant_trading.polygonio.poly_client import PolygonAPIClient
from etrade_quant_trading.polygonio.prices import get_historical_prices
from etrade_quant_trading.polygonio.earnings import get_earnings_dates
from etrade_quant_trading.polygonio.calendar_utils import (
    calc_implied_vol,
    forward_factor,
    gather_calendar_pair,
)
from etrade_quant_trading.polygonio.recursive_backtest import (
    RecursionConfig,
    _price_from_data,
    _calendar_dte_targets,
    _calendar_expiry_candidates,
)


def _fmt_ns_hms(ns_val: Any) -> str:
    try:
        ns = int(ns_val)
        if ns <= 0:
            return ""
        return datetime.fromtimestamp(ns / 1_000_000_000).strftime("%H:%M:%S")
    except Exception:
        return ""


def _fmt_ns_full(ns_val: Any) -> str:
    try:
        ns = int(ns_val)
        if ns <= 0:
            return ""
        dt = datetime.fromtimestamp(ns / 1_000_000_000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _format_timestamp_field(val: Any) -> Any:
    formatted = _fmt_ns_hms(val)
    if not formatted:
        return val
    return f"{formatted} ({val})"


@dataclass
class ForwardFactorRow:
    date: str
    spot: float
    front_expiry: str
    back_expiry: str
    strike_front: float
    strike_back: float
    iv_front: float
    iv_back: float
    forward_vol: float
    forward_factor: float
    front_time: Optional[str] = None  # HH:MM:SS used for pricing
    back_time: Optional[str] = None   # HH:MM:SS used for pricing
    front_dte: Optional[int] = None
    back_dte: Optional[int] = None
    front_offset_minutes: Optional[float] = None
    back_offset_minutes: Optional[float] = None

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "date": self.date,
            "spot": self.spot,
            "front_expiry": self.front_expiry,
            "back_expiry": self.back_expiry,
            "strike_front": self.strike_front,
            "strike_back": self.strike_back,
            "iv_front": self.iv_front,
            "iv_back": self.iv_back,
            "forward_vol": self.forward_vol,
            "forward_factor": self.forward_factor,
            "front_time": self.front_time or "",
            "back_time": self.back_time or "",
            "front_dte": "" if self.front_dte is None else self.front_dte,
            "back_dte": "" if self.back_dte is None else self.back_dte,
            "front_offset_minutes": "" if self.front_offset_minutes is None else self.front_offset_minutes,
            "back_offset_minutes": "" if self.back_offset_minutes is None else self.back_offset_minutes,
        }




async def _compute_series(
    *,
    ticker: str,
    start: date,
    end: date,
    front_dte: int,
    back_dte: int,
    debug: bool = False,
    break_on_bundle_miss: bool = False,
) -> Tuple[List[ForwardFactorRow], Dict[str, float], Set[Any], Counter[str]]:
    settings = get_settings()
    premium_field = resolve_premium_field(settings)

    # --- Simple risk-free estimator using Yahoo yields
    def _infer_risk_free(as_of: date, dte_days: int) -> float:
        try:
            tenor = max(1, int(dte_days))
            if tenor <= 90:
                sym = "^IRX"   # 13-week T-bill
            elif tenor <= 365:
                sym = "^FVX"   # 5-year as rough proxy for 6-12m
            else:
                sym = "^TNX"   # 10-year
            ds = as_of.strftime("%Y-%m-%d")
            df = get_historical_prices(sym, ds, ds)
            if df is not None and not df.empty:
                y = float(df.iloc[-1]["close"]) / 100.0
                return max(0.0, y)
        except Exception:
            pass
        return 0.0

    def _dividend_yield_default(tkr: str) -> float:
        try:
            return float(get_settings().dividend_yield_default or 0.0)
        except Exception:
            return 0.0

    df_prices = get_historical_prices(ticker, start.isoformat(), end.isoformat())
    close_prices_available: Dict[str, float] = {}
    try:
        df_prices = df_prices.copy()
        df_prices["date"] = pd.to_datetime(df_prices["date"], errors="coerce")
        if debug:
            print("[FF-DEBUG] DataFrame prices:")
            print(df_prices.tail(25))
    except Exception as exc:
        if debug:
            print(f"[FF-DEBUG] Failed to normalize price dates: {exc}")
    close_by_date: Dict[date, float] = {}
    for dt_val, px in zip(df_prices["date"], df_prices["close"]):
        try:
            if pd.isna(dt_val):
                continue
            px_f = float(px)
            close_by_date[dt_val.date()] = px_f
            close_prices_available[dt_val.date().strftime("%Y-%m-%d")] = px_f
        except Exception:
            continue
    if debug:
        print(
            f"[FF-DEBUG] Loaded {len(close_by_date)} spot close prices for {ticker} "
            f"from {start.isoformat()} to {end.isoformat()}"
        )
    
    print(f"[FF-DEBUG] DataFrame prices:\n{df_prices}")
    print(f"[FF-DEBUG] Loaded {len(close_by_date)} spot close prices for {ticker} from {start} to {end}")

    cfg = RecursionConfig(
        ticker=ticker,
        global_start_date=start.isoformat(),
        global_end_date=end.isoformat(),
        trade_type="forward_iv_calender",
        expiring_weekday="Friday",
        calendar_front_dte=front_dte,
        calendar_back_dte=back_dte,
    )

    rows: List[ForwardFactorRow] = []
    skip_counts: Counter[str] = Counter()

    try:
        load_stored_option_data(ticker, strike_type="atm")
    except Exception as exc:
        if debug:
            print(f"[FF-DEBUG] Failed to preload option cache for {ticker}: {exc}")

    async with PolygonAPIClient(api_key=settings.polygon_api_key) as client:
        cur = start
        while cur <= end:
            spot = close_by_date.get(cur)
            if spot is None:
                skip_counts["no_spot_close"] += 1
                if debug:
                    print(f"[FF-DEBUG] {cur}: no spot close available; skipping")
                cur += timedelta(days=1)
                continue

            f_tgt, b_tgt = _calendar_dte_targets(cfg)
            pair_result, failure_reason = await gather_calendar_pair(
                ticker=ticker,
                as_of_date=cur,
                spot=float(spot),
                front_target=f_tgt,
                back_target=b_tgt,
                weekday=cfg.expiring_weekday,
                client=client,
                premium_field=premium_field,
                option_type="call",
                debug=debug,
            )
            if pair_result is None:
                reason_key = failure_reason or "no_premium_pair"
                if reason_key == "no_spot":
                    reason_key = "no_spot_close"
                skip_counts[reason_key] += 1
                if debug:
                    print(f"[FF-DEBUG] {cur}: calendar pair unavailable (reason={reason_key})")
                cur += timedelta(days=1)
                continue

            front_date = pair_result.front_date
            back_date = pair_result.back_date
            strike_front = pair_result.strike_front
            strike_back = pair_result.strike_back
            entry_front = pair_result.entry_front
            entry_back = pair_result.entry_back
            premium_front = pair_result.premium_front
            premium_back = pair_result.premium_back
            used_ts_front = pair_result.used_ts_front
            used_ts_back = pair_result.used_ts_back
            chosen_front = pair_result.chosen_front
            chosen_back = pair_result.chosen_back
            front_samples_all = pair_result.front_samples
            back_samples_all = pair_result.back_samples
            target_dt = pair_result.target_dt
            target_ns = pair_result.target_ns
            window_lo_ns = pair_result.window_lo_ns
            window_hi_ns = pair_result.window_hi_ns

            # Persist chosen samples & full window on the entry quotes for downstream use
            def _prepare_store_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    copy = dict(s)
                    copy.pop("_price", None)
                    copy.setdefault("target_timestamp", target_ns)
                    out.append(copy)
                return out

            for entry, sample, samples_all in (
                (entry_front, chosen_front, front_samples_all),
                (entry_back, chosen_back, back_samples_all),
            ):
                quote_dest = entry.setdefault("quote", {}) if isinstance(entry, dict) else {}
                cleaned_sample = dict(sample)
                cleaned_sample.pop("_price", None)
                quote_dest.update(cleaned_sample)
                quote_dest["_samples"] = _prepare_store_samples(samples_all)

            q_front = entry_front.get("quote", {}) if isinstance(entry_front, dict) else {}
            q_back = entry_back.get("quote", {}) if isinstance(entry_back, dict) else {}
            ts_front_ns = used_ts_front or q_front.get("sip_timestamp")
            ts_back_ns = used_ts_back or q_back.get("sip_timestamp")
            if debug:
                front_ts_hms = _fmt_ns_hms(ts_front_ns)
                front_ts_full = _fmt_ns_full(ts_front_ns)
                back_ts_hms = _fmt_ns_hms(ts_back_ns)
                back_ts_full = _fmt_ns_full(ts_back_ns)
                def _compose_ts_display(hms: str, full: str, raw: Any) -> str:
                    if hms and full:
                        return f"{hms} ({full})"
                    if hms:
                        return hms
                    if full:
                        return full
                    return str(raw)
                print(
                    f"[FF-DEBUG] {cur}: using premiums source={premium_field} "
                    f"front_ts={_compose_ts_display(front_ts_hms, front_ts_full, ts_front_ns)} "
                    f"back_ts={_compose_ts_display(back_ts_hms, back_ts_full, ts_back_ns)}"
                )

            # Infer rates
            dte_front = max(0, (front_date - cur).days)
            dte_back = max(0, (back_date - cur).days)
            r_front = _infer_risk_free(cur, int(dte_front or 0))
            r_back = _infer_risk_free(cur, int(dte_back or 0))
            q_yield = _dividend_yield_default(ticker)

            iv_front = calc_implied_vol(
                spot=spot,
                strike=strike_front,
                premium=premium_front,
                dte=int(dte_front),
                option_type="call",
                risk_free_rate=r_front,
                dividend_yield=q_yield,
            )
            iv_back = calc_implied_vol(
                spot=spot,
                strike=strike_back,
                premium=premium_back,
                dte=int(dte_back),
                option_type="call",
                risk_free_rate=r_back,
                dividend_yield=q_yield,
            )
            if iv_front is None or iv_back is None:
                skip_counts["iv_failure"] += 1
                if debug:
                    print(
                        f"[FF-DEBUG] {cur}: iv computation failed front={iv_front} back={iv_back}"
                    )
                cur += timedelta(days=1)
                continue

            ff_pair = forward_factor(
                iv_front=iv_front,
                iv_back=iv_back,
                dte_front=int(dte_front),
                dte_back=int(dte_back),
            )
            if ff_pair is None:
                skip_counts["forward_factor_failure"] += 1
                if debug:
                    print(
                        f"[FF-DEBUG] {cur}: forward factor could not be computed"
                    )
                cur += timedelta(days=1)
                continue

            ff, fwd_vol = ff_pair
            def _offset_minutes(ts_ns: Optional[int]) -> Optional[float]:
                if ts_ns is None or ts_ns <= 0 or not target_dt:
                    return None
                try:
                    ts_dt = datetime.fromtimestamp(ts_ns / 1_000_000_000)
                    return round((ts_dt - target_dt).total_seconds() / 60.0, 3)
                except Exception:
                    return None

            rows.append(
                ForwardFactorRow(
                    date=cur.isoformat(),
                    spot=float(spot),
                    front_expiry=front_date.strftime("%Y-%m-%d"),
                    back_expiry=back_date.strftime("%Y-%m-%d"),
                    strike_front=float(strike_front),
                    strike_back=float(strike_back),
                    iv_front=float(iv_front),
                    iv_back=float(iv_back),
                    forward_vol=float(fwd_vol),
                    forward_factor=float(ff),
                    front_time=_fmt_ns_hms(ts_front_ns),
                    back_time=_fmt_ns_hms(ts_back_ns),
                    front_dte=int(dte_front),
                    back_dte=int(dte_back),
                    front_offset_minutes=_offset_minutes(used_ts_front),
                    back_offset_minutes=_offset_minutes(used_ts_back),
                )
            )
            if debug:
                print(f"[FF-DEBUG] {cur}: FF={ff:.4f} (front_exp={front_date} back_exp={back_date})")
            cur += timedelta(days=1)

    earnings_dates = get_earnings_dates(
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    return rows, close_prices_available, earnings_dates, skip_counts


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_front = int(getattr(settings, "calendar_front_dte_default", 60))
    default_back = int(getattr(settings, "calendar_back_dte_default", 90))
    parser = argparse.ArgumentParser(description="Compute Forward Factor time series")
    parser.add_argument("--ticker", required=True, help="Underlying symbol, e.g. SPY")
    parser.add_argument(
        "--start",
        help="Start date YYYY-MM-DD (default: today minus 365 days)",
    )
    parser.add_argument(
        "--end",
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--front-dte",
        type=int,
        default=default_front,
        help=f"Target front DTE (default: {default_front})",
    )
    parser.add_argument(
        "--back-dte",
        type=int,
        default=default_back,
        help=f"Target back DTE (default: {default_back})",
    )
    parser.add_argument(
        "--csv",
        help="Optional path to write CSV output (default: <ticker>_ff_<timestamp>.csv)",
    )
    parser.add_argument(
        "--plot",
        help="Optional path to save forward factor plot PNG (defaults to CSV path with .png)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Number of rows to print at the end (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-day debug information during computation",
    )
    parser.add_argument(
        "--break-on-bundle-miss",
        action="store_true",
        help="Enter debugger when front/back expiry bundle cannot be built",
    )
    return parser.parse_args()


def _coerce_date(value: Optional[str], fallback: date) -> date:
    if not value:
        return fallback
    return datetime.strptime(value, "%Y-%m-%d").date()


def _write_csv(path: Path, rows: List[ForwardFactorRow]) -> None:
    headers = list(rows[0].as_dict().keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if headers:
            fh.write(",".join(headers) + "\n")
        for row in rows:
            values = row.as_dict()
            fh.write(",".join(str(values[h]) for h in headers) + "\n")


def _write_plot(
    path: Path,
    rows: List[ForwardFactorRow],
    spot_price_map: Dict[str, float],
    earnings_dates: Set[date] | Set[datetime] | Set[Any],
) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[FF-DEBUG] Unable to import matplotlib for plotting: {exc}")
        return

    try:
        date_values = [datetime.strptime(r.date[:10], "%Y-%m-%d") for r in rows]
    except Exception:
        date_values = list(range(len(rows)))

    def _safe_float(val: Any, default: float = float("nan")) -> float:
        try:
            return float(val)
        except Exception:
            return default

    ff_values = [_safe_float(r.forward_factor) for r in rows]
    spot_values = [_safe_float(r.spot) for r in rows]
    iv_front_values = [_safe_float(r.iv_front) for r in rows]
    iv_back_values = [_safe_float(r.iv_back) for r in rows]
    front_dte_values = [_safe_float(r.front_dte, 0.0) for r in rows]
    back_dte_values = [_safe_float(r.back_dte, 0.0) for r in rows]
    front_offset_values = [_safe_float(r.front_offset_minutes) for r in rows]
    back_offset_values = [_safe_float(r.back_offset_minutes) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_spot, ax_iv, ax_offset, ax_ff, ax_dte) = plt.subplots(5, 1, figsize=(10, 11), sharex=True)

    ax_spot.plot(date_values, spot_values, color="tab:blue", linewidth=1.5, label="Spot")
    ax_spot.set_title("Spot Price Over Time")
    ax_spot.set_ylabel("Spot")
    ax_spot.grid(alpha=0.3)
    spot_only_dates: List[datetime] = []
    spot_only_values: List[float] = []
    if spot_price_map:
        row_dates_set = {str(getattr(r, "date", ""))[:10] for r in rows}
        for ds, px in sorted(spot_price_map.items()):
            if ds not in row_dates_set:
                try:
                    dt_obj = datetime.strptime(ds, "%Y-%m-%d")
                except Exception:
                    continue
                spot_only_dates.append(dt_obj)
                spot_only_values.append(px)
    if spot_only_dates and date_values and isinstance(date_values[0], datetime):
        ax_spot.scatter(spot_only_dates, spot_only_values, color="tab:red", marker="o", s=15, label="Spot only")
    earnings_plot_dates: List[datetime] = []
    if earnings_dates:
        if isinstance(date_values[0], datetime):
            for ed in sorted(earnings_dates):
                try:
                    ed_date = ed if isinstance(ed, datetime) else datetime.combine(ed, datetime.min.time())
                except Exception:
                    continue
                earnings_plot_dates.append(ed_date)
            for ed in earnings_plot_dates:
                ax_spot.axvline(ed, color="tab:purple", linestyle="--", linewidth=0.8, alpha=0.7)
    handles, labels = ax_spot.get_legend_handles_labels()
    if handles:
        ax_spot.legend()

    ax_iv.plot(date_values, iv_front_values, label="Front IV", color="tab:orange", linewidth=1.2)
    ax_iv.plot(date_values, iv_back_values, label="Back IV", color="tab:green", linewidth=1.2)
    ax_iv.set_ylabel("Implied Vol")
    ax_iv.grid(alpha=0.3)
    ax_iv.legend()

    ax_offset.plot(date_values, front_offset_values, label="Front offset", color="tab:purple", linewidth=1.2)
    ax_offset.plot(date_values, back_offset_values, label="Back offset", color="tab:brown", linewidth=1.2)
    ax_offset.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_offset.set_ylabel("Offset (min)")
    ax_offset.grid(alpha=0.3)
    ax_offset.legend()

    ax_ff.plot(date_values, ff_values, marker="o", linewidth=1.5, markersize=4)
    ax_ff.set_title("Forward Factor Over Time")
    ax_ff.set_ylabel("Forward Factor")
    ax_ff.grid(alpha=0.3)

    ax_dte.plot(date_values, front_dte_values, label="Front DTE", linewidth=1.2)
    ax_dte.plot(date_values, back_dte_values, label="Back DTE", linewidth=1.2)
    ax_dte.set_ylabel("Days to Expiration")
    ax_dte.set_xlabel("Date")
    ax_dte.grid(alpha=0.3)
    ax_dte.legend()

    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150)
        print(f"Saved forward factor plot to {path}")
    except Exception as exc:
        print(f"[FF-DEBUG] Failed to save plot {path}: {exc}")
    finally:
        plt.close(fig)


def main() -> int:
    args = _parse_args()
    today = date.today()
    start = _coerce_date(args.start, today - timedelta(days=365))
    end = _coerce_date(args.end, today)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    rows, spot_price_map, earnings_dates, skip_summary = asyncio.run(
        _compute_series(
            ticker=args.ticker,
            start=start,
            end=end,
            front_dte=args.front_dte,
            back_dte=args.back_dte,
            debug=args.debug,
            break_on_bundle_miss=args.break_on_bundle_miss,
        )
    )

    # Persist updated option cache to ATM-specific files
    try:
        save_stored_option_data(args.ticker, strike_type="atm")
        if args.debug:
            print(f"[FF-DEBUG] Saved ATM option cache for {args.ticker}")
    except Exception as exc:
        if args.debug:
            print(f"[FF-DEBUG] Failed to save ATM option cache: {exc}")

    base_dir = Path("cover_call_plot")
    csv_path: Optional[Path] = None
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
    elif rows:
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        csv_path = base_dir / f"{args.ticker.upper()}_ff_{timestamp}.csv"

    if csv_path and rows:
        _write_csv(csv_path, rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")

    plot_path: Optional[Path] = None
    if args.plot:
        plot_path = Path(args.plot)
        if not plot_path.is_absolute():
            plot_path = base_dir / plot_path
    elif csv_path:
        plot_path = csv_path.with_suffix(".png")

    if plot_path:
        if rows:
            _write_plot(plot_path, rows, spot_price_map, earnings_dates)
        else:
            print(f"[FF-DEBUG] Skipping plot generation for {plot_path} (no rows)")

    tail_n = max(0, int(args.tail))
    if tail_n:
        print(f"Last {min(tail_n, len(rows))} rows:")
        for row in rows[-tail_n:]:
            print(row.as_dict())
    else:
        print(f"Computed {len(rows)} rows (tail output disabled)")

    reason_labels = {
        "no_spot_close": "Missing spot close",
        "no_expiries": "No calendar expiries",
        "no_front_contracts": "No front-leg contracts",
        "no_back_contracts": "No back-leg contracts",
        "no_atm_strike": "No ATM strike candidates",
        "no_strike_pair": "No acceptable strike pair",
        "no_premium_pair": "No aligned premium pair",
        "iv_failure": "IV computation failure",
        "forward_factor_failure": "Forward factor failure",
    }
    total_missing = sum(skip_summary.values())
    print("Missing-day summary:")
    print(f"  Total missing days: {total_missing}")
    for key, label in reason_labels.items():
        print(f"  {label}: {skip_summary.get(key, 0)}")
    extra_keys = [k for k in skip_summary.keys() if k not in reason_labels]
    for key in extra_keys:
        print(f"  {key}: {skip_summary[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
