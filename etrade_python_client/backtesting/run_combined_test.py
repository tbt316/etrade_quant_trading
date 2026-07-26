import subprocess
import sys
import os
import json

START_DATE = "2019-01-01"
END_DATE = "2026-05-17"
STRATEGY_ID = "baseline_put_spread"

combined_config = {
    "name": "Put Spread (0.13 Delta + Cond Exit)",
    "exit": {
        "conditional_half_dte_exit": "profitable_only"
    },
    "rolling": {
        "roll_dte_multiplier": 1.0, 
        "roll_strike_behavior": "same_short"
    },
    "sizing": {
        "dynamic_margin_method": None
    },
    "entry": {
        "dynamic_delta_method": None,
        "short_delta": -0.13
    }
}

def main():
    print(f"\n{'='*60}")
    print(f"RUNNING: Put Spread (0.13 Delta + Cond Exit)")
    print(f"{'='*60}")
    
    # Save config to temp file
    config_path = f"backtesting/temp_config_combined.json"
    with open(config_path, "w") as f:
        json.dump(combined_config, f)
    
    # Inherit current environment to preserve PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
        
    try:
        cmd = [
            sys.executable, "backtesting/backtest_runner.py",
            "--strategy", STRATEGY_ID,
            "--strategy_config", config_path,
            "--start", START_DATE,
            "--end", END_DATE,
            "--log"
        ]
        subprocess.run(cmd, check=True, env=env)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
            
    # Regenerate the dashboard
    print("\nRegenerating experiments dashboard...")
    subprocess.run([sys.executable, "backtesting/generate_experiments_report.py"], env=env)
    print("\nDONE. View dashboard at backtesting/experiments_dashboard.html")

if __name__ == "__main__":
    main()
