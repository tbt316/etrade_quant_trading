import os, asyncio, pytest

pytestmark = pytest.mark.integration

# Import your real entry points
from polygonio.recursive_backtest import (
    backtest_options_sync_or_async, RecursionConfig
)

def _pick_daily(res: dict):
    # try top-level common names
    for k in ["daily", "days", "timeline", "daily_results", "day_results"]:
        v = res.get(k)
        if isinstance(v, (list, tuple)):
            return v
    # try a nested container like 'report' or 'results'
    for container in ["report", "results", "summary"]:
        c = res.get(container)
        if isinstance(c, dict):
            for k in ["daily", "days", "timeline", "daily_results", "day_results"]:
                v = c.get(k) if isinstance(c, dict) else None
                if isinstance(v, (list, tuple)):
                    return v
    return None

def _pick_positions(res: dict):
    for k in ["positions", "trades", "opened_positions", "closed_positions", "legs"]:
        v = res.get(k)
        if isinstance(v, (list, tuple)):
            return v
    # sometimes positions are nested under 'results' or 'report'
    for container in ["results", "report", "summary"]:
        c = res.get(container)
        if isinstance(c, dict):
            for k in ["positions", "trades", "opened_positions", "closed_positions", "legs"]:
                v = c.get(k)
                if isinstance(v, (list, tuple)):
                    return v
    return None

def test_backtest_smoke_online():
    assert os.getenv("POLYGON_API_KEY"), "POLYGON_API_KEY must be set"
    print(f"KEY length: {len(os.getenv('POLYGON_API_KEY'))}")
    
    cfg = RecursionConfig(
        ticker="SPY",
        global_start_date="2025-07-10",
        global_end_date="2025-07-21",
        trade_type="put_credit_spread",
        expiring_weekday="Friday",
        expiring_wks=6,
        contract_qty=1,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    res = loop.run_until_complete(backtest_options_sync_or_async(cfg))
    assert isinstance(res, dict), f"Unexpected result type: {type(res)}"

    # Minimal invariants that should hold for a successful run
    daily = _pick_daily(res)
    positions = _pick_positions(res)

    # at least one of daily/positions should exist and be a non-empty list
    assert (daily and len(daily) > 0) or (positions and len(positions) > 0), (
        f"Result did not contain recognizable daily/positions lists. Keys: {list(res.keys())}"
    )

    # Optional: basic counters if present
    dbg = res.get("debug") or res.get("stats") or {}
    if isinstance(dbg, dict):
        for k in ["days_total", "days_no_price", "exceptions"]:
            if k in dbg:
                assert isinstance(dbg[k], int)