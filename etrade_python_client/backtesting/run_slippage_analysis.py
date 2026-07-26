import subprocess
import sys
import os
import json

START_DATE = "2025-01-01"
END_DATE = "2025-03-31"

# Define the strategy configuration for Variant 6
config_base = {
    "id": "baseline_put_spread",
    "name": "ATM Long Put Spread (10% profit)",
    "status": "implemented",
    "regime_aware": False,
    "description": "Variant 6 under slippage testing (2022-2026)",
    "instrument": "put_credit_spread",
    "underlying": "SPX",
    "entry": {
        "method": "fixed_delta",
        "target_dte": 42,
        "short_delta": -0.1,
        "spread_width": 200.0,
        "risk_free_rate": 0.05,
        "dynamic_delta_method": "atm_long_spread"
    },
    "exit": {
        "method": "dynamic_dte + profit_target",
        "close_dte": 0,
        "early_profit_pct": 0.1,
        "conditional_half_dte_exit": False,
        "hold_itm_to_expiration": False
    },
    "sizing": {
        "method": "daily_pacing",
        "initial_capital": 1000000.0,
        "margin_limit_pct": 0.25,
        "daily_pacing_slots": 10
    },
    "rolling": {
        "trigger": "itm_at_close_dte",
        "target_dte": 42,
        "long_leg_target": "atm",
        "short_leg_offset": "spread_width",
        "spread_width_multiplier": 2.0,
        "qty_reduction": "ceil_half",
        "repeat_itm_at_expiration": True,
        "prefer_monthly_on_deviation": True,
        "stop_when_chain_breakeven": True,
        "enabled": False
    },
    "filters": {
        "min_chain_strikes": 10,
        "min_credit": 0.01,
        "strike_tolerance_pct": 0.3,
        "delta_tolerance_pct": 0.15
    }
}

models = [
    {"slippage_model": "none", "name": "V6 (Mid-Price Baseline)"},
    {"slippage_model": "steer_50", "name": "V6 (50% Steer)"},
    {"slippage_model": "worst_case", "name": "V6 (Worst Case)"}
]

def run_backtest(model_cfg):
    slippage = model_cfg["slippage_model"]
    name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"RUNNING SLIPPAGE ANALYSIS: {name} (Slippage: {slippage})")
    print(f"{'='*60}")
    
    cfg = config_base.copy()
    cfg["name"] = name
    cfg["description"] = f"Variant 6 (ATM Long Put Spread 10% Profit) from {START_DATE} to {END_DATE} using slippage model: {slippage}."
    
    config_path = f"backtesting/temp_slippage_config_{slippage}.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f)
        
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
        
    try:
        cmd = [
            sys.executable, "backtesting/backtest_runner.py",
            "--strategy", "baseline_put_spread",
            "--strategy_config", config_path,
            "--underlying", "SPX",
            "--start", START_DATE,
            "--end", END_DATE,
            "--slippage_model", slippage,
            "--log"
        ]
        subprocess.run(cmd, check=True, env=env)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    for m in models:
        run_backtest(m)
        
    # Run dashboard report generation
    print("\nGenerating final dashboard with slippage metrics...")
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
    subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"], env=env)
    print("\nDONE. View dashboard at backtesting/experiments_dashboard.html")
