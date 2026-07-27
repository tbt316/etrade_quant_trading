import subprocess
import sys
import os
import json
from tqdm import tqdm

START_DATE = "2019-01-01"
END_DATE = "2026-05-17"

LOG_FILE = "backtesting/experiments_log.jsonl"

# Generate delta variants from -0.11 to -0.15 in steps of 0.01
deltas = [0.11, 0.12, 0.13, 0.14, 0.15]
VARIANTS = []

for d in deltas:
    VARIANTS.append({
        "strategy_id": "baseline_put_spread",
        "description": f"Delta Variant: Put Spread ({d:.2f} Delta)",
        "config": {
            "name": f"Put Spread ({d:.2f} Delta)",
            "exit": {"conditional_half_dte_exit": False},
            "rolling": {"roll_dte_multiplier": 1.0, "roll_strike_behavior": "same_short"},
            "sizing": {"dynamic_margin_method": None},
            "entry": {
                "dynamic_delta_method": None,
                "short_delta": -d
            }
        }
    })

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
    print(f"Starting execution of {len(VARIANTS)} delta variants from {START_DATE} to {END_DATE}...")
    print(f"Existing entries in {LOG_FILE} will be preserved.")
    
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
