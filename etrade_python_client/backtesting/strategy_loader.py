"""Load strategy YAML files from backtesting/strategies."""
import os
import glob
from typing import Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_STRATEGIES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "strategies",
)

def _simple_yaml_parse(text: str) -> dict:
    """A very basic fallback parser for simple YAML-like structures."""
    result = {}
    current_key = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            
            # Handle booleans
            if val.lower() == "true": val = True
            elif val.lower() == "false": val = False
            # Handle numbers
            else:
                try:
                    if "." in val: val = float(val)
                    else: val = int(val)
                except ValueError:
                    pass
            
            # Simple nesting (one level for entry/exit)
            if not raw_line.startswith(" "): # Root key
                if not val:
                    result[key] = {}
                else:
                    result[key] = val
                current_key = key
            elif current_key and raw_line.startswith(" "): # Nested key
                if not isinstance(result.get(current_key), dict):
                    result[current_key] = {}
                result[current_key][key] = val
    return result

def load_all_strategies(strategies_dir: str = _STRATEGIES_DIR) -> Dict[str, dict]:
    """Load all strategies from the strategies directory."""
    if not os.path.exists(strategies_dir):
        return {}
        
    strategies = {}
    for file_path in glob.glob(os.path.join(strategies_dir, "*.yaml")) + glob.glob(os.path.join(strategies_dir, "*.yml")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            if HAS_YAML:
                parsed = yaml.safe_load(content)
            else:
                parsed = _simple_yaml_parse(content)
                
            if isinstance(parsed, dict) and "id" in parsed:
                strategies[parsed["id"]] = parsed
        except Exception as e:
            print(f"Error parsing strategy file {file_path}: {e}")
            continue
    return strategies


def load_strategy(
    strategy_id: str,
    strategies_dir: str = _STRATEGIES_DIR,
) -> dict:
    """Load a single strategy by ID from the strategies directory."""
    strategies = load_all_strategies(strategies_dir)
    if strategy_id not in strategies:
        available = list(strategies.keys())
        raise KeyError(
            f"Strategy '{strategy_id}' not found in {strategies_dir}. Available: {available}"
        )
    return strategies[strategy_id]


def list_strategies(strategies_dir: str = _STRATEGIES_DIR) -> list:
    """List all available strategy IDs and names."""
    strategies = load_all_strategies(strategies_dir)
    return [
        {"id": s["id"], "name": s.get("name", s["id"]), "status": s.get("status", "implemented")}
        for s in strategies.values()
    ]


if __name__ == "__main__":
    print("Available strategies:")
    for s in list_strategies():
        print(f"  [{s['status']}] {s['id']}: {s['name']}")
    print()
    try:
        config = load_strategy("baseline_put_spread")
        print("Loaded config for 'baseline_put_spread':")
        for key, val in config.items():
            print(f"  {key}: {val}")
    except KeyError as e:
        print(e)
