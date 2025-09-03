import sys, re
from pathlib import Path
from textwrap import dedent

TARGET = Path("etrade_quant_trading/polygonio/cache_io.py")

def insert_after_merge_nested(text: str) -> str:
    """Insert merge_nested_dicts_with_change after merge_nested_dicts if missing."""
    if "def merge_nested_dicts_with_change(" in text:
        return text  # already present

    # Find end of merge_nested_dicts function
    m = re.search(r"\ndef\s+merge_nested_dicts\s*\([^)]*\):[\s\S]*?\n(?=def|\Z)", text)
    if not m:
        return text  # not found

    insert_pos = m.end()
    snippet = dedent('''
    def merge_nested_dicts_with_change(dst: dict, src: dict) -> bool:
        """
        Like merge_nested_dicts, but returns True if *dst* changed as a result of the merge.
        Used to avoid unnecessary pickle writes when nothing is new.
        """
        changed = False
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                if merge_nested_dicts_with_change(dst[k], v):
                    changed = True
            else:
                if k not in dst or dst.get(k) != v:
                    dst[k] = v
                    changed = True
        return changed

    ''')
    return text[:insert_pos] + snippet + text[insert_pos:]

def patch_save_func(text: str) -> str:
    """Replace unconditional write in save_stored_option_data with change-aware writes."""
    fstart = re.search(r"\ndef\s+save_stored_option_data\s*\(\s*ticker\s*:\s*str\s*\)\s*->\s*None\s*:\s*\n", text)
    if not fstart:
        return text
    start_idx = fstart.end()
    fend = re.search(r"\ndef\s+\w", text[start_idx:])
    end_idx = start_idx + (fend.start() if fend else len(text) - start_idx)
    body = text[start_idx:end_idx]

    # Ensure parent dirs for files
    body = re.sub(
        r"ensure_dir\([^)]*POLY_OPTION_DIR[^)]*\)",
        "ensure_dir(price_cache_file.parent)\n    ensure_dir(chain_cache_file.parent)",
        body
    )

    # Replace merge+write block with change-aware logic
    merge_pat = re.compile(
        r"merge_nested_dicts\(\s*existing_price\s*,\s*stored_option_price\[t\]\s*\)\s*\n\s*"
        r"merge_nested_dicts\(\s*existing_chain\s*,\s*stored_option_chain\[t\]\s*\)\s*\n"
    )
    if merge_pat.search(body):
        body = merge_pat.sub(dedent('''
        price_changed = merge_nested_dicts_with_change(existing_price, stored_option_price[t])
        chain_changed = merge_nested_dicts_with_change(existing_chain, stored_option_chain[t])

        if price_changed:
            with price_cache_file.open("wb") as f:
                pickle.dump(existing_price, f, protocol=pickle.HIGHEST_PROTOCOL)
            if price_cache_file.stat().st_size == 0:
                print(f"Error: {price_cache_file} is empty after write!")

        if chain_changed:
            with chain_cache_file.open("wb") as f:
                pickle.dump(existing_chain, f, protocol=pickle.HIGHEST_PROTOCOL)
            if chain_cache_file.stat().st_size == 0:
                print(f"Error: {chain_cache_file} is empty after write!")

        '''), body, count=1)

    return text[:start_idx] + body + text[end_idx:]

def main():
    p = TARGET
    if not p.exists():
        print(f"[ERROR] Cannot find {p}. Run this from repo root.")
        sys.exit(1)
    original = p.read_text(encoding="utf-8")
    mod = insert_after_merge_nested(original)
    mod2 = patch_save_func(mod)
    if mod2 == original:
        print("[INFO] No changes applied (file may already be patched or patterns not found).")
    else:
        p.write_text(mod2, encoding="utf-8")
        print("[OK] Patched", p)

if __name__ == "__main__":
    main()