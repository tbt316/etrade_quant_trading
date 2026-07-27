#!/usr/bin/env python3
"""
Legacy integration script to:
1. Fetch current portfolio tickers from E*Trade.
2. Download analyst research reports for each ticker using browser_use_hybrid.
"""

import asyncio
import argparse
import os
import sys
import shutil
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import E*Trade infrastructure
try:
    from live_trading.etrade_cover_call_new import oauth
    from accounts.accounts_bo import Accounts
    from live_trading.portfolio_manager import get_current_holdings
    from browser_use_hybrid import run_hybrid_download
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_portfolio_reports(username, password, use_sandbox=False, headless=True, verify_only=False):
    """
    Main workflow:
    1. OAuth Login (API)
    2. Get Portfolio Tickers (from ALL accounts)
    3. Web Login & Download Reports (Browser)
    """
    
    # 1. API Login & Fetch Tickers
    print("\n" + "="*60)
    print("   📊 FETCHING E*TRADE PORTFOLIO")
    print("="*60)
    
    all_tickers = set()
    
    try:
        # Use existing oauth logic from etrade_cover_call_new
        session, base_url = oauth(
            use_sandbox=use_sandbox,
            auto_login=True,
            username=username,
            password=password,
            headless=headless
        )
        
        accounts = Accounts(session, base_url)
        
        # Get list of all accounts
        # calling account_list(0) initially to get the full list
        # We will ignore the 'self.account' side effect for a moment and iterate properly
        account_list_info = accounts.account_list(0)
        
        if not account_list_info:
            print("❌ No accounts found.")
            return

        print(f"\nFound {len(account_list_info)} accounts. Scanning all for holdings...")

        # Iterate through each account
        for i, acc_info in enumerate(account_list_info):
            acc_id, acc_desc, _, _ = acc_info
            print(f"  Scanning Account {i+1}: {acc_desc} ({acc_id})...")
            
            # Switch to this account
            # This sets accounts.account = accounts[i]
            accounts.account_list(i)
            
            try:
                # Fetch holdings for the current account
                holdings = get_current_holdings(accounts, include_options=True)
                if holdings:
                    print(f"    Found {len(holdings)} tickers.")
                    all_tickers.update(holdings)
                else:
                    print("    No holdings found.")
            except Exception as e:
                print(f"    ❌ Error scanning account {acc_id}: {e}")

        sorted_tickers = sorted(list(all_tickers))
        
        if not sorted_tickers:
            print("⚠️ No tickers found in any portfolio.")
            return
            
        print(f"\n✅ Total unique tickers found: {len(sorted_tickers)}")
        print(f"Tickers: {', '.join(sorted_tickers)}")
        
        if verify_only:
            print("\n⏹️ Verification mode enabled. Exiting without downloading reports.")
            return
        
    except Exception as e:
        print(f"❌ Failed to fetch portfolio: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Download Reports (Persistent Browser)
    print("\n" + "="*60)
    print("   📥 DOWNLOADING RESEARCH REPORTS (PERSISTENT BROWSER)")
    print("="*60)
    
    # Set environment variables for browser app auto_login
    if username: os.environ["ETRADE_USER"] = username
    if password: os.environ["ETRADE_PASS"] = password
    
    playwright_instance = None
    browser_context = None

    try:
        # Launch browser once
        from browser_use_hybrid import launch_persistent_browser, download_report_for_ticker
        playwright_instance, browser_context = await launch_persistent_browser(headless=headless)

        # Iterate tickers and download using the same browser
        for i, ticker in enumerate(sorted_tickers, 1):
            print(f"\n[{i}/{len(sorted_tickers)}] Processing {ticker}...")
            
            try:
                download_dir = Path("analyst_reports") / ticker
                
                # Use the persistent context
                await download_report_for_ticker(browser_context, ticker, download_dir)
                
                # Small delay between tickers to be polite
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to process {ticker}: {e}")
                continue

        print("\n✅ All portfolio tickers processed.")

    except Exception as e:
        print(f"❌ Browser session failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if browser_context:
            print("Closing browser context...")
            await browser_context.close()
        if playwright_instance:
            print("Stopping Playwright...")
            await playwright_instance.stop()

    # 3. Aggregate Reports
    aggregate_reports()

def aggregate_reports():
    """
    Copy all PDF files from ticker-specific folders into a central 'aggregated' folder.
    """
    base_dir = Path("analyst_reports")
    agg_dir = base_dir / "aggregated"
    
    if not base_dir.exists():
        return

    print("\n" + "="*60)
    print("   📂 AGGREGATING REPORTS INTO 'aggregated/' FOLDER")
    print("="*60)
    
    agg_dir.mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    for ticker_dir in base_dir.iterdir():
        if ticker_dir.is_dir() and ticker_dir.name != "aggregated":
            for pdf_file in ticker_dir.glob("*.pdf"):
                dest_file = agg_dir / pdf_file.name
                try:
                    shutil.copy2(pdf_file, dest_file)
                    file_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to copy {pdf_file.name}: {e}")
    
    print(f"✅ Aggregation complete. Total files in aggregated/: {file_count}")

def main():
    parser = argparse.ArgumentParser(description='Fetch portfolio and download research reports.')
    parser.add_argument('--username', '-u', help='E*Trade web username (optional if ETRADE_USER set in .env)')
    parser.add_argument('--password', '-p', help='E*Trade web password (optional if ETRADE_PASS set in .env)')
    parser.add_argument('--sandbox', action='store_true', help='Use Sandbox environment')
    parser.add_argument('--no-headless', action='store_true', help='Show browser UI')
    parser.add_argument('--verify-tickers-only', action='store_true', help='Only fetch and print tickers, do not download')

    args = parser.parse_args()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Resolve credentials
    username = args.username or os.environ.get("ETRADE_USER")
    password = args.password or os.environ.get("ETRADE_PASS")

    if not username or not password:
        print("❌ Error: Username/Password required. Set ETRADE_USER/ETRADE_PASS in .env or use --username/--password")
        sys.exit(1)
    
    try:
        asyncio.run(process_portfolio_reports(
            username=username,
            password=password,
            use_sandbox=args.sandbox,
            headless=not args.no_headless,
            verify_only=args.verify_tickers_only
        ))
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
