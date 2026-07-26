import subprocess
import sys
import os

START_DATE = "2025-01-01"
END_DATE = "2025-03-31"
STRATEGY_ID = "baseline_put_spread"

def main():
    print(f"\n{'='*60}")
    print(f"RUNNING SPY TEST BACKTEST: {STRATEGY_ID} from {START_DATE} to {END_DATE}")
    print(f"{'='*60}")
    
    # Inherit current environment to preserve PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "."
    else:
        env["PYTHONPATH"] = "." + os.pathsep + env["PYTHONPATH"]
        
    cmd = [
        sys.executable, "backtesting/backtest_runner.py",
        "--strategy", STRATEGY_ID,
        "--underlying", "SPY",
        "--start", START_DATE,
        "--end", END_DATE,
        "--log"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print("\n✓ Backtest executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during backtest execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
