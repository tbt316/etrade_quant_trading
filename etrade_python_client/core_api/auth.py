#!/usr/bin/env python3
"""
E*Trade Browser Session Authentication

This script opens a browser for manual or automated login and saves the authenticated 
session (cookies and storage state) to a file. This allows the main scraper to reuse
the session without needing to log in every time.

Usage:
    python auth.py
    python auth.py --username YOUR_USER --password YOUR_PASS
    
The script will:
1. Open a visible browser window
2. Navigate to E*Trade login page
3. Attempt automated login if credentials provided
4. Wait for you to complete MFA if required
5. Save the session to etrade_session.json
6. Future runs of the scraper can use --use-session to skip login
"""

import asyncio
import json
import os
import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    raise ImportError(
        "Playwright is required. Install with: pip install playwright && playwright install chromium"
    )


SESSION_FILE = Path("etrade_session.json")
ETRADE_LOGIN_URL = "https://us.etrade.com/home/welcome-back"


def _print(message: str, prefix: str = "[Auth]"):
    """Print with immediate flush for real-time output."""
    print(f"{prefix} {message}", flush=True)


async def save_session(username=None, password=None):
    """
    Open browser for manual or automated login and save the authenticated session.
    """
    _print("Starting E*Trade Authentication Session Saver")
    _print("=" * 60)
    
    async with async_playwright() as p:
        # Launch browser in visible mode (NOT headless)
        browser = await p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
            ]
        )
        
        # Create context with realistic settings
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            accept_downloads=True,
        )
        
        # Remove webdriver flag
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        page = await context.new_page()
        
        _print(f"Navigating to E*Trade login page...")
        await page.goto(ETRADE_LOGIN_URL, wait_until='domcontentloaded')
        
        # Automated login steps
        if username and password:
            _print("Attempting automated login...")
            try:
                # Username field
                username_selectors = ['#USER', 'input[name="USER"]', '#user_orig', 'input[name="username"]']
                username_field = None
                for selector in username_selectors:
                    try:
                        username_field = await page.wait_for_selector(selector, timeout=3000)
                        if username_field: break
                    except: continue
                
                if username_field:
                    await username_field.fill(username)
                    _print("  Username entered ✓")
                    
                    # Password field
                    password_field = await page.wait_for_selector('#password, #password_orig, input[type="password"]', timeout=3000)
                    if password_field:
                        await password_field.fill(password)
                        _print("  Password entered ✓")
                        
                        # Login button
                        login_btn = await page.wait_for_selector('#mfaLogonButton, #logOnbtn, button[type="submit"], input[type="submit"]', timeout=3000)
                        if login_btn:
                            await login_btn.click()
                            _print("  Login form submitted ✓")
                else:
                    _print("  ⚠ Could not find login fields, please proceed manually.")
            except Exception as e:
                _print(f"  ⚠ Automated login step failed: {e}")

        _print("")
        _print("╔═════════════════════════════════════════════════════════════╗")
        _print("║  MANUAL ACTION MAY BE REQUIRED                              ║")
        _print("║                                                             ║")
        _print("║  1. Complete MFA if challenged                              ║")
        _print("║  2. Complete login if it didn't finish automatically       ║")
        _print("║  3. Once you see your account dashboard, press ENTER here   ║")
        _print("╚═════════════════════════════════════════════════════════════╝")
        _print("")
        
        # Wait for user to complete login in the actual terminal
        # Using loop to check for the ENTER input so we can handle it properly in async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "Press ENTER after you have successfully logged in...")
        
        # Verify login was successful
        current_url = page.url
        _print(f"Current URL: {current_url}")
        
        if 'login' in current_url.lower():
            _print("⚠️  Warning: You may still be on a login page!")
        
        # Save the storage state (cookies + localStorage)
        _print("Saving session state...")
        storage_state = await context.storage_state()
        
        # Add metadata
        session_data = {
            "saved_at": datetime.now().isoformat(),
            "url_at_save": current_url,
            "storage_state": storage_state
        }
        
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        _print(f"✓ Session saved to: {SESSION_FILE.absolute()}")
        _print(f"  Cookies saved: {len(storage_state.get('cookies', []))}")
        _print("")
        _print("You can now run the scraper with --use-session flag:")
        _print(f"  python main_analyst_agent.py -t AAPL --use-session")
        
        await browser.close()


def load_session() -> dict:
    """
    Load saved session from file.
    Returns dict with storage_state for Playwright context, or None if not found.
    """
    if not SESSION_FILE.exists():
        return None
    
    try:
        with open(SESSION_FILE, 'r') as f:
            data = json.load(f)
        
        saved_at = data.get("saved_at", "unknown")
        _print(f"Loaded session from {saved_at}")
        
        return data.get("storage_state")
    except Exception as e:
        _print(f"Error loading session: {e}")
        return None


def session_exists() -> bool:
    """Check if a saved session file exists."""
    return SESSION_FILE.exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E*Trade Session Authentication")
    parser.add_argument("--username", help="E*Trade username")
    parser.add_argument("--password", help="E*Trade password")
    args = parser.parse_args()
    
    # Check env vars if not provided via CLI
    username = args.username or os.environ.get("ETRADE_USERNAME")
    password = args.password or os.environ.get("ETRADE_PASSWORD")
    
    asyncio.run(save_session(username, password))
