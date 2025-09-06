import os
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from pathlib import Path

# Ensure the repository root is on sys.path so `polygonio` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from polygonio import plot_results
from polygonio.plot_results import plot_from_backtest_results


def test_plot_from_backtest_results_argument_order(monkeypatch):
    res = {
        "ticker": "TST",
        "start": "2020-01-01",
        "end": "2020-01-05",
        "positions": [
            {
                "position_open_date": "2020-01-01",
                "expiration": "2020-01-07",
                "call_closed_profit": 10.0,
                "call_closed_date": "2020-01-03",
            }
        ],
    }

    captured = {}

    def fake_plot_recursive_results(*args):
        captured["args"] = args
        return "ok"

    monkeypatch.setattr(
        "polygonio.plot_results.plot_recursive_results", fake_plot_recursive_results
    )

    out = plot_from_backtest_results(res)
    assert out == "ok"
    assert captured["args"][0] == "TST"
    # parameter_history should be in position 5
    assert captured["args"][5][0]["start_date"] == "2020-01-01"
    # start and end should follow parameter_history
    assert captured["args"][6] == "2020-01-01"
    assert captured["args"][7] == "2020-01-05"


def test_plot_recursive_results_creates_png(tmp_path, monkeypatch):
    res = {
        "ticker": "TST",
        "start": "2023-01-01",
        "end": "2023-01-03",
        "positions": [
            {
                "position_open_date": "2023-01-01",
                "expiration": "2023-01-05",
                "call_closed_profit": 5.0,
                "call_closed_date": "2023-01-02",
                "short_call_prem_open": 1.0,
                "long_call_prem_open": 0.5,
                "call_strike_sold": 105,
                "call_strike_bought": 110,
                "open_distance_call": 0.05,
                "required_margin": 1000,
            }
        ],
    }

    price_df = pd.DataFrame(
        {"close": [100, 101, 102]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )

    monkeypatch.setattr(plot_results, "PLOT_DIR", tmp_path)

    fig = plot_from_backtest_results(res, price_df=price_df)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 8
    files = list(Path(tmp_path).glob("recursive_monthly_TST_*"))
    assert files, "PNG file was not created"

