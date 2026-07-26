#!/usr/bin/env python3
"""
Diff two engine snapshots by hash.

Usage:
    python -m backtesting.diff_engines <hash1> <hash2>
    
Example:
    python -m backtesting.diff_engines 8c790d81 a3f1b2c4
    
This will show a unified diff of all core engine files that changed
between the two snapshot hashes.
"""
import os
import sys
import difflib
import json

_SNAPSHOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".engine_snapshots"
)

_LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments_log.jsonl"
)


def list_available_hashes():
    """List all engine hashes that have been snapshotted."""
    if not os.path.exists(_SNAPSHOTS_DIR):
        return []
    return sorted(os.listdir(_SNAPSHOTS_DIR))


def diff_engines(hash1: str, hash2: str, context_lines: int = 3):
    """Show unified diff between two engine snapshots."""
    dir1 = os.path.join(_SNAPSHOTS_DIR, hash1)
    dir2 = os.path.join(_SNAPSHOTS_DIR, hash2)
    
    if not os.path.exists(dir1):
        print(f"ERROR: Snapshot for hash '{hash1}' not found.")
        print(f"  Available hashes: {list_available_hashes()}")
        return False
    if not os.path.exists(dir2):
        print(f"ERROR: Snapshot for hash '{hash2}' not found.")
        print(f"  Available hashes: {list_available_hashes()}")
        return False
    
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    all_files = sorted(files1 | files2)
    
    has_diff = False
    
    for fname in all_files:
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)
        
        if fname not in files1:
            print(f"\n{'='*70}")
            print(f"NEW FILE in {hash2}: {fname}")
            print(f"{'='*70}")
            has_diff = True
            continue
        if fname not in files2:
            print(f"\n{'='*70}")
            print(f"DELETED FILE in {hash2}: {fname}")
            print(f"{'='*70}")
            has_diff = True
            continue
        
        with open(path1, "r", encoding="utf-8", errors="replace") as f:
            lines1 = f.readlines()
        with open(path2, "r", encoding="utf-8", errors="replace") as f:
            lines2 = f.readlines()
        
        diff = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=f"{hash1}/{fname}",
            tofile=f"{hash2}/{fname}",
            n=context_lines
        ))
        
        if diff:
            has_diff = True
            print(f"\n{'='*70}")
            print(f"CHANGED: {fname}")
            print(f"{'='*70}")
            for line in diff:
                # Color output
                if line.startswith('+') and not line.startswith('+++'):
                    print(f"\033[92m{line}\033[0m", end="")
                elif line.startswith('-') and not line.startswith('---'):
                    print(f"\033[91m{line}\033[0m", end="")
                elif line.startswith('@@'):
                    print(f"\033[96m{line}\033[0m", end="")
                else:
                    print(line, end="")
    
    if not has_diff:
        print(f"No differences found between {hash1} and {hash2}.")
    
    return has_diff


def show_experiments_for_hash(engine_hash: str):
    """Show which experiments used a given engine hash."""
    if not os.path.exists(_LOG_FILE):
        return
    
    experiments = []
    with open(_LOG_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                exp = json.loads(line)
                if exp.get("engine_hash") == engine_hash:
                    experiments.append(exp)
            except json.JSONDecodeError:
                continue
    
    if experiments:
        print(f"\nExperiments using engine hash {engine_hash}:")
        for exp in experiments:
            print(f"  {exp['experiment_id']} | {exp['strategy_id']} | {exp['start_date']}→{exp['end_date']} | {exp['timestamp'][:16]}")
    else:
        print(f"\nNo experiments found for engine hash {engine_hash}.")


def main():
    if len(sys.argv) < 2:
        print("Engine Snapshot Diff Tool")
        print("=" * 40)
        print(f"\nUsage:")
        print(f"  python -m backtesting.diff_engines <hash1> <hash2>   # Diff two snapshots")
        print(f"  python -m backtesting.diff_engines --list             # List available hashes")
        print(f"  python -m backtesting.diff_engines --info <hash>      # Show experiments for a hash")
        
        hashes = list_available_hashes()
        if hashes:
            print(f"\nAvailable snapshots: {', '.join(hashes)}")
        else:
            print(f"\nNo snapshots found. Run a backtest to create the first snapshot.")
        return
    
    if sys.argv[1] == "--list":
        hashes = list_available_hashes()
        if hashes:
            print(f"Available engine snapshots ({len(hashes)}):")
            for h in hashes:
                print(f"  {h}")
        else:
            print("No snapshots found.")
        return
    
    if sys.argv[1] == "--info" and len(sys.argv) >= 3:
        show_experiments_for_hash(sys.argv[2])
        return
    
    if len(sys.argv) < 3:
        print("ERROR: Two hashes required for diffing.")
        print("  Usage: python -m backtesting.diff_engines <hash1> <hash2>")
        return
    
    diff_engines(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
