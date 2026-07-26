import subprocess
import sys
import os
import json
import argparse
try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

START_DATE = "2016-01-01"
END_DATE = "2026-05-23"

LOG_FILE = "backtesting/experiments_log.jsonl"

# Define variants
VARIANTS = [
    {
        "strategy_id": "baseline_put_spread",
        "description": "Baseline SPY Put Spread (42 DTE, -0.12 delta, $20 width, 50% profit target)",
        "config": {
            "name": "Baseline (SPY, $20 width, 50% exit)",
            "underlying": "SPY",
            "entry": {
                "spread_width": 20.0,
            },
            "exit": {
                "early_profit_pct": 0.50,
            },
            "sizing": {
                "daily_pacing_slots": 15,
                "dynamic_margin_method": None,
            },
        },
    },
    {
        "strategy_id": "baseline_put_spread",
        "description": "Variant 1: Change to SPX ($200 width)",
        "config": {
            "name": "Variant 1: SPX ($200 width)",
            "underlying": "SPX",
            "entry": {
                "spread_width": 200.0,
            },
            "exit": {
                "early_profit_pct": 0.50,
            },
            "sizing": {
                "daily_pacing_slots": 15,
                "dynamic_margin_method": None,
            },
        },
    },
    {
        "strategy_id": "baseline_put_spread",
        "description": "Variant 2: Early Exit Profit Target 80%",
        "config": {
            "name": "Variant 2: 80% Profit Exit",
            "underlying": "SPY",
            "entry": {
                "spread_width": 20.0,
            },
            "exit": {
                "early_profit_pct": 0.80,
            },
            "sizing": {
                "daily_pacing_slots": 15,
                "dynamic_margin_method": None,
            },
        },
    },
    {
        "strategy_id": "baseline_put_spread",
        "description": "Variant 3: Pacing Slots 10 (Capital * 25% / 10)",
        "config": {
            "name": "Variant 3: 25%/10 Pacing",
            "underlying": "SPY",
            "entry": {
                "spread_width": 20.0,
            },
            "exit": {
                "early_profit_pct": 0.50,
            },
            "sizing": {
                "daily_pacing_slots": 10,
                "dynamic_margin_method": None,
            },
        },
    },
    {
        "strategy_id": "baseline_put_spread",
        "description": "Variant 4: Dynamic Margin Limit (15% if VIX < 25 else 25%)",
        "config": {
            "name": "Variant 4: Dynamic 15%/25% Margin",
            "underlying": "SPY",
            "entry": {
                "spread_width": 20.0,
            },
            "exit": {
                "early_profit_pct": 0.50,
            },
            "sizing": {
                "daily_pacing_slots": 15,
                "dynamic_margin_method": "vix_scaled_15_25",
            },
        },
    },
]

def run_experiment(strategy_id, description, config):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    # Save config to temp file
    config_path = f"backtesting/temp_config_{strategy_id}.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # Inherit current environment to preserve PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
        
    try:
        cmd = [
            sys.executable, "backtesting/backtest_runner.py",
            "--strategy", strategy_id,
            "--strategy_config", config_path,
            "--start", START_DATE,
            "--end", END_DATE,
            "--log"
        ]
        subprocess.run(cmd, check=True, env=env)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the comparison batch of backtests.")
    parser.add_argument(
        "--retain-existing-results",
        action="store_true",
        help="Keep existing experiments_log.jsonl entries and append new runs to the dashboard.",
    )
    args = parser.parse_args()

    if args.retain_existing_results:
        print(f"Retaining existing experiments log ({LOG_FILE}). New runs will be appended.")
    else:
        print(f"Clearing experiments log ({LOG_FILE}) to start fresh...")
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("")
            print("✓ Successfully cleared leaderboard log.")
        except Exception as e:
            print(f"WARNING: Failed to clear leaderboard log: {e}")

    print(f"Starting execution of {len(VARIANTS)} variants from {START_DATE} to {END_DATE}...")
    for i, var in enumerate(tqdm(VARIANTS, desc="Variant Progress")):
        run_experiment(var["strategy_id"], var["description"], var["config"])
    
    # Final dashboard update
    print("\nGenerating final dashboard...")
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
    subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"], env=env)
    print("\nDONE. View dashboard at backtesting/experiments_dashboard.html")
