"""
Experiment Tracker: Records backtest outcomes to experiments_log.jsonl
and snapshots engine files for future diffing.
"""
import os
import json
import hashlib
import shutil
import uuid
from datetime import datetime
import numpy as np

_TRACKER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments_log.jsonl"
)

_SNAPSHOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".engine_snapshots"
)

# Core files to hash for the automatic engine version
_CORE_FILES = [
    "backtest_runner.py",
    "massive_api_client.py",
    "greeks_calculator.py",
    "option_data_cache.py",
    "../live_trading/data_ingestion.py"
]

class ExperimentTracker:
    def __init__(self, log_file: str = _TRACKER_FILE):
        self.log_file = log_file

    def _get_engine_hash(self) -> str:
        """Calculate a hash of the core backtesting engine files."""
        hasher = hashlib.sha256()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        for file_name in _CORE_FILES:
            file_path = os.path.join(base_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
            else:
                hasher.update(file_name.encode('utf-8'))
                
        return hasher.hexdigest()[:8]

    def _snapshot_engine(self, engine_hash: str):
        """
        Save a copy of the core engine files under .engine_snapshots/{hash}/
        so we can diff between any two engine versions later.
        """
        snapshot_dir = os.path.join(_SNAPSHOTS_DIR, engine_hash)
        if os.path.exists(snapshot_dir):
            return  # Already snapshotted
        
        os.makedirs(snapshot_dir, exist_ok=True)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        for file_name in _CORE_FILES:
            src = os.path.join(base_dir, file_name)
            if os.path.exists(src):
                # Flatten the path for storage (replace / with __)
                safe_name = file_name.replace("/", "__").replace("..", "_parent_")
                dst = os.path.join(snapshot_dir, safe_name)
                shutil.copy2(src, dst)

    def _get_strategy_hash(self, strategy_config: dict) -> str:
        """Calculate a hash of the strategy configuration."""
        config_str = json.dumps(strategy_config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

    def _compute_sharpe_ratio(self, result, risk_free_rate: float = 0.05) -> float:
        """
        Compute annualized Sharpe ratio from the daily NLV history.
        Sharpe = (mean(daily_returns) - daily_rf) / std(daily_returns) * sqrt(252)
        """
        if not hasattr(result, "nvl_history") or not result.nvl_history:
            return 0.0
        
        nlvs = [val for _, val in result.nvl_history]
        if len(nlvs) < 2:
            return 0.0
        
        nlvs = np.array(nlvs, dtype=float)
        daily_returns = np.diff(nlvs) / nlvs[:-1]
        
        if len(daily_returns) == 0 or np.std(daily_returns) == 0:
            return 0.0
        
        # Check if result has dynamic risk_free_rates
        if hasattr(result, "risk_free_rates") and not result.risk_free_rates.empty:
            # Shift dates to match the returns (which are index 1 onwards in nvl_history)
            dates = [d for d, _ in result.nvl_history[1:]]
            daily_rf = []
            for dt in dates:
                if dt in result.risk_free_rates.index:
                    rf_ann = result.risk_free_rates.loc[dt]
                else:
                    rf_ann = risk_free_rate
                daily_rf.append(rf_ann / 252.0)
            daily_rf = np.array(daily_rf, dtype=float)
        else:
            daily_rf = risk_free_rate / 252.0
            
        excess_returns = daily_returns - daily_rf
        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
        return float(sharpe)

    def record_experiment(
        self, 
        strategy_config: dict, 
        start_date: str, 
        end_date: str, 
        result, # BacktestResult
        log_path: str = "",
        report_path: str = ""
    ):
        """Record the outcome of a backtest experiment."""
        
        # Calculate financial metrics
        initial_capital = getattr(result, "initial_capital", 100000.0)
        total_pnl = getattr(result, "total_pnl", 0.0)
        
        total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
        
        # Calculate Annualized Return (CAGR)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        years = max(days / 365.25, 0.01) # Avoid division by zero
        
        cagr_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        
        # Max Drawdown from NLV history
        max_drawdown_pct = 0.0
        if hasattr(result, "nvl_history") and result.nvl_history:
            nlvs = [val for _, val in result.nvl_history]
            if nlvs:
                peak = nlvs[0]
                mdd = 0.0
                for val in nlvs:
                    if val > peak:
                        peak = val
                    dd = (peak - val) / peak if peak > 0 else 0
                    if dd > mdd:
                        mdd = dd
                max_drawdown_pct = mdd * 100

        # Calmar Ratio
        calmar_ratio = cagr_pct / max_drawdown_pct if max_drawdown_pct > 0 else cagr_pct

        # Sharpe Ratio
        risk_free_rate = strategy_config.get("entry", {}).get("risk_free_rate", 0.05)
        sharpe_ratio = self._compute_sharpe_ratio(result, risk_free_rate)

        # Basic Stats
        total_trades = getattr(result, "total_trades", 0)
        win_count = getattr(result, "win_count", 0)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = getattr(result, "avg_pnl", 0.0)
        
        engine_hash = self._get_engine_hash()
        
        # Snapshot the engine files for this hash (no-op if already snapshotted)
        self._snapshot_engine(engine_hash)
        
        record = {
            "experiment_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "engine_hash": engine_hash,
            "strategy_id": strategy_config.get("name", strategy_config.get("id", "unknown")),
            "strategy_hash": self._get_strategy_hash(strategy_config),
            "start_date": start_date,
            "end_date": end_date,
            "metrics": {
                "total_return_pct": total_return_pct,
                "cagr_pct": cagr_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "calmar_ratio": calmar_ratio,
                "sharpe_ratio": sharpe_ratio,
                "win_rate_pct": win_rate,
                "total_trades": total_trades,
                "avg_pnl": avg_pnl,
                "api_calls": getattr(result, "api_calls", 0),
                "data_gap_count": getattr(result, "data_gap_count", 0),
                "critical_gap_count": getattr(result, "critical_gap_count", 0)
            },
            "log_path": log_path,
            "report_path": report_path,
            "strategy_config": strategy_config
        }

        # Append to jsonl
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        print(f"\n  ✓ Experiment {record['experiment_id']} recorded to {self.log_file}")
        print(f"    Engine Hash: {engine_hash} | Strategy Hash: {record['strategy_hash']}")
        print(f"    CAGR: {cagr_pct:.2f}% | Max DD: {max_drawdown_pct:.2f}% | Calmar: {calmar_ratio:.2f} | Sharpe: {sharpe_ratio:.2f}")
