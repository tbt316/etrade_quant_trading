"""
Strategy loader: parses YAML blocks from strategy_registry.md
and returns strategy configs as Python dicts.
"""
import os
import re
from typing import Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "strategy_registry.md",
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

def _parse_yaml_blocks(markdown_text: str) -> Dict[str, dict]:
    """Extract all ```yaml ... ``` blocks from a markdown file."""
    pattern = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)
    strategies = {}
    for match in pattern.finditer(markdown_text):
        yaml_content = match.group(1)
        try:
            if HAS_YAML:
                parsed = yaml.safe_load(yaml_content)
            else:
                parsed = _simple_yaml_parse(yaml_content)
                
            if isinstance(parsed, dict) and "id" in parsed:
                strategies[parsed["id"]] = parsed
        except Exception:
            continue
    return strategies


def load_all_strategies(registry_path: str = _REGISTRY_PATH) -> Dict[str, dict]:
    """Load all strategies from the registry markdown file."""
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Strategy registry not found: {registry_path}")
    with open(registry_path, "r", encoding="utf-8") as f:
        content = f.read()
    return _parse_yaml_blocks(content)


def load_strategy(
    strategy_id: str,
    registry_path: str = _REGISTRY_PATH,
) -> dict:
    """Load a single strategy by ID from the registry."""
    strategies = load_all_strategies(registry_path)
    if strategy_id not in strategies:
        available = list(strategies.keys())
        raise KeyError(
            f"Strategy '{strategy_id}' not found. Available: {available}"
        )
    return strategies[strategy_id]


def list_strategies(registry_path: str = _REGISTRY_PATH) -> list:
    """List all available strategy IDs and names."""
    strategies = load_all_strategies(registry_path)
    return [
        {"id": s["id"], "name": s.get("name", s["id"]), "status": s.get("status", "implemented")}
        for s in strategies.values()
    ]


if __name__ == "__main__":
    print("Available strategies:")
    for s in list_strategies():
        print(f"  [{s['status']}] {s['id']}: {s['name']}")
    print()
    config = load_strategy("fixed_delta_put_spread")
    print("Loaded config for 'fixed_delta_put_spread':")
    for key, val in config.items():
        print(f"  {key}: {val}")
