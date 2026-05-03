#!/usr/bin/env python3
"""
Report Scraper Module

Async Playwright-based scraper for downloading analyst PDF reports from E*Trade's
web interface. Handles web login, navigation, and PDF downloads with anti-detection
measures.
"""

import asyncio
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

try:
    from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout
except ImportError:
    raise ImportError(
        "Playwright is required. Install with: pip install playwright && playwright install chromium"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _print(message: str, prefix: str = "    [Scraper]"):
    """Print with immediate flush for real-time output."""
    print(f"{prefix} {message}", flush=True)


class ETradeReportScraper:
    """
    Async web scraper for E*Trade analyst reports using Playwright.
    
    Handles the complete workflow:
    1. Login to E*Trade web interface (with MFA support)
    2. Navigate to Research pages for each ticker
    3. Download available analyst PDF reports
    
    Features:
    - Anti-detection: Random delays, realistic user-agent
    - MFA handling: Waits for manual MFA input
    - Dynamic waits: Uses wait_for_selector instead of fixed sleeps
    - Error resilience: Gracefully handles missing reports
    """
    
    # Use direct SSO login page (more stable than deprecated E*Trade paths)
    ETRADE_LOGIN_URL = "https://idp.etrade.com/idp/SSO.saml2?SAMLRequest"
    # Fallback login URL if SSO page doesn't have form directly
    ETRADE_LEGACY_LOGIN_URL = "https://us.etrade.com/e/t/user/login"
    # Research URL - wallst.com with SSO handshake (confirmed by user)
    # Research URL - Direct E*Trade portal snapshot (more stable for SPA navigation)
    ETRADE_RESEARCH_URL_TEMPLATE = "https://us.etrade.com/etx/hw/v2/quotes/snapshot?symbol={ticker}"
    
    # Analyst Research deep link
    ETRADE_ANALYST_RESEARCH_URL_TEMPLATE = (
        "https://us.etrade.com/e/t/invest/quotesandresearch?cmenu=DetQ&sym={ticker}#/analystResearch"
    )
    
    # Wallst.com variant for detailed research (fallback)
    ETRADE_WALLST_RESEARCH_URL = (
        "https://www.etrade.wallst.com/etrade-web/research"
        "?ChallengeUrl=https://idp.etrade.com/idp/SSO.saml2"
        "&reinitiate-handshake=0"
        "&AuthnContext=authenticated"
        "&env=PRD"
        "&symbol={ticker}"
    )
    
    def __init__(
        self,
        headless: bool = True,
        download_dir: Optional[Path] = None,
        timeout_ms: int = 30000,
        session_file: Optional[Path] = None,
        debug_dir: Optional[Path] = None
    ):
        """
        Initialize the scraper.
        
        Args:
            headless: Run browser in headless mode. Set False for debugging.
            download_dir: Directory for downloaded PDFs. Defaults to ./analyst_reports/
            timeout_ms: Default timeout for page operations in milliseconds.
            session_file: Optional path to a saved session file (from auth.py).
            debug_dir: Directory for debug artifacts (screenshots, HTML). Defaults to ./debug/
        """
        self.headless = headless
        self.download_dir = download_dir or Path("./analyst_reports")
        self.timeout_ms = timeout_ms
        self.session_file = session_file
        self.debug_dir = debug_dir or Path("./debug")
        self.capture_steps = True  # Always capture page state at key steps for debugging
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._playwright = None
        self._logged_in = False
        self._step_counter = 0  # For sequencing debug files
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _init_browser(self):
        """Initialize Playwright browser with stealth settings."""
        _print("Starting Playwright...")
        self._playwright = await async_playwright().start()
        
        _print(f"Launching Chromium browser (headless={self.headless})...")
        # Launch Chromium with stealth options
        self.browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        _print("Browser launched successfully")
        
        _print("Creating browser context...")
        
        # Load session state if available
        storage_state = None
        if self.session_file and self.session_file.exists():
            try:
                import json
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                storage_state = session_data.get("storage_state")
                _print(f"  ✓ Loaded saved session from {self.session_file}")
            except Exception as e:
                _print(f"  ⚠ Failed to load session: {e}")
        
        # Create context with realistic settings (and optional storage state)
        context_options = {
            'viewport': {'width': 1920, 'height': 1080},
            # Spoof real Chrome on Windows as requested to pass WAF
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'locale': 'en-US',
            'timezone_id': 'America/New_York',
            'accept_downloads': True,
            # CRITICAL SSO FIX: Allow cross-domain SAML handoff to wallst.com
            # Bypass Content Security Policy and HTTPS errors to ensure cookies pass 
            'bypass_csp': True,
            'ignore_https_errors': True,
            # Explicitly allow permissions that might be needed
            'permissions': ['geolocation', 'notifications'], 
        }
        
        if storage_state:
            context_options['storage_state'] = storage_state
            self._logged_in = True  # Assume logged in if we have saved session
        
        context = await self.browser.new_context(**context_options)
        
        # MANUAL SSO BYPASS: Inject wallst.com cookies if provided
        # This bypasses the fragile SAML handshake by using a manually captured session
        wallst_cookies_file = Path("wallst_cookies.json")
        if wallst_cookies_file.exists():
            try:
                import json
                with open(wallst_cookies_file, 'r') as f:
                    cookies = json.load(f)
                
                # Filter/Sanitize cookies to ensure they are for the right domain if needed
                # or just add them all. Playwright handles domain matching.
                if isinstance(cookies, list):
                    await context.add_cookies(cookies)
                    _print(f"  ✓ Injected {len(cookies)} manual cookies from {wallst_cookies_file}")
                else:
                    _print(f"  ⚠ Invalid cookie format in {wallst_cookies_file} (expected list)")
            except Exception as e:
                _print(f"  ⚠ Failed to inject manual cookies: {e}")
        
        # Remove webdriver flag
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        self.page = await context.new_page()
        
        # Add routing to force PDF downloads (triggers expect_download even for inline PDFs)
        async def force_pdf_download(route):
            try:
                if route.request.resource_type in ["image", "font", "stylesheet"]:
                    await route.continue_()
                    return

                response = await route.fetch()
                headers = response.headers.copy()
                content_type = headers.get('content-type', '').lower()
                url = route.request.url.lower()
                
                # Force download for PDF content types or URLs containing pdf/docKey
                if 'pdf' in content_type or '.pdf' in url or 'pdf.asp' in url or 'dockey=' in url:
                    headers['content-disposition'] = 'attachment'
                    await route.fulfill(response=response, headers=headers)
                else:
                    await route.fulfill(response=response)
            except Exception:
                try:
                    await route.continue_()
                except:
                    pass
                
        await self.page.context.route("**", force_pdf_download)
        
        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        _print("Browser initialized successfully ✓")
    
    async def close(self):
        """Close browser and cleanup."""
        _print("Closing browser...")
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        _print("Browser closed ✓")
    
    async def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0):
        """Add random delay to mimic human behavior."""
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)
    
    async def _human_type(self, element, text: str):
        """Type text with random delays between keystrokes."""
        for char in text:
            await element.type(char, delay=random.randint(50, 150))
    
    async def _save_debug_artifacts(self, context: str = "step"):
        """
        Save screenshot and HTML page source for debugging.
        
        This creates files that can be shared with the AI agent or
        reviewed manually to understand page state on failures.
        
        Args:
            context: A short label for the artifact files (e.g., 'login_page', 'research_page')
        """
        if not self.page:
            return None, None
        
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self._step_counter += 1
            step_prefix = f"{self._step_counter:02d}"
            
            # Save screenshot
            screenshot_path = self.debug_dir / f"{step_prefix}_{context}.png"
            await self.page.screenshot(path=str(screenshot_path), full_page=True)
            _print(f"  📸 [{step_prefix}] Screenshot: {screenshot_path.name}")
            
            # Save HTML page source
            html_path = self.debug_dir / f"{step_prefix}_{context}.html"
            content = await self.page.content()
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(content)
            _print(f"  📄 [{step_prefix}] HTML: {html_path.name}")
            
            # Also save current URL
            url_path = self.debug_dir / f"{step_prefix}_{context}_url.txt"
            with open(url_path, 'w') as f:
                f.write(f"URL: {self.page.url}\n")
                f.write(f"Title: {await self.page.title()}\n")
            
            return screenshot_path, html_path
        except Exception as e:
            _print(f"  ⚠️ Failed to save debug artifacts: {e}")
            return None, None
    
    async def login(
        self,
        username: str,
        password: str,
        mfa_timeout_sec: int = 120
    ) -> bool:
        """
        Handle E*Trade web login with MFA support.
        Optimized to check page state rather than using fixed delays.
        
        Args:
            username: E*Trade username.
            password: E*Trade password.
            mfa_timeout_sec: Seconds to wait for MFA completion.
        
        Returns:
            True if login successful, False otherwise.
        """
        if not self.page:
            await self._init_browser()
        
        # PROACTIVE: If we have a session file, check if we're already logged in
        if self.session_file and self.session_file.exists():
            _print("Session file found, checking if already logged in...")
            try:
                # Navigate to a safe page to check session
                await self.page.goto("https://us.etrade.com/home", wait_until='domcontentloaded', timeout=10000)
                await asyncio.sleep(1)
                
                # Check if we are NOT on a login page
                current_url = self.page.url.lower()
                if 'login' not in current_url and 'welcome-back' not in current_url:
                    _print("Active session detected! Skipping login steps. ✓")
                    self._logged_in = True
                    return True
                else:
                    _print("Session expired or invalid, proceeding with login...")
            except Exception as e:
                _print(f"Error checking session: {e}, proceeding with login...")

        _print(f"Navigating to E*Trade login page...")
        
        # Try multiple login URLs as E*Trade has been deprecating paths
        login_urls = [
            "https://us.etrade.com/home/welcome-back",  # Welcome back login page
            "https://us.etrade.com/e/t/user/login",  # Classic login page
            self.ETRADE_LOGIN_URL,  # SSO login
        ]
        
        username_field = None
        
        for login_url in login_urls:
            _print(f"  Trying: {login_url[:50]}...")
            try:
                await self.page.goto(login_url, wait_until='domcontentloaded', timeout=15000)
            except Exception as nav_err:
                _print(f"    Navigation failed: {str(nav_err)[:50]}")
                continue
            
            # Check if we hit a 404 page
            page_content = await self.page.content()
            if "404" in page_content and "Not Found" in page_content:
                _print(f"    ⚠ 404 error on this URL, trying next...")
                continue
            
            # CAPTURE: Login page loaded - capture state for debugging
            if self.capture_steps:
                await self._save_debug_artifacts("login_page_loaded")
            
            # Look for username field FIRST (many pages have form directly visible)
            _print("  Looking for username field...")
            username_selectors = ['#USER', 'input[name="USER"]', '#user_orig', 'input[name="username"]']
            
            for selector in username_selectors:
                try:
                    username_field = await self.page.wait_for_selector(selector, timeout=3000)
                    if username_field:
                        _print(f"    Found username field ✓")
                        break
                except PlaywrightTimeout:
                    continue
            
            if username_field:
                break
            
            # If username field not found, check if we need to click a "Log on" link
            _print("  Username field not immediately visible, looking for 'Log on' link...")
            logon_selectors = [
                'a:has-text("Log on")', 
                'a:has-text("Log On")',
                'text=Log on',
                'text=Log On'
            ]
            
            for selector in logon_selectors:
                try:
                    logon_link = await self.page.query_selector(selector)
                    if logon_link and await logon_link.is_visible():
                        _print(f"    Found 'Log on' link, clicking...")
                        await logon_link.click()
                        await self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                        break
                except Exception:
                    continue

            # Try again after clicking Log on link
            for selector in username_selectors:
                try:
                    username_field = await self.page.wait_for_selector(selector, timeout=3000)
                    if username_field:
                        _print(f"    Found username field ✓")
                        break
                except PlaywrightTimeout:
                    continue
            
            if username_field:
                break
            else:
                _print("    Username field not found on this page, trying next URL...")
        
        # Wait for and handle login form
        try:
            if not username_field:
                _print("✗ Could not find username field on any login page!")
                return False
            
            # Enter username immediately when field is found
            _print("Entering username...")
            await username_field.click()
            await self._human_type(username_field, username)
            _print("  Username entered ✓")
            
            # Password field - should already be on page
            _print("Looking for password field...")
            password_field = await self.page.wait_for_selector(
                '#password, #password_orig, input[type="password"]', timeout=3000
            )
            
            if not password_field:
                _print("✗ Could not find password field!")
                return False
            
            _print("  Found password field ✓")
            _print("Entering password...")
            await password_field.click()
            await self._human_type(password_field, password)
            _print("  Password entered ✓")
            
            # Click login button immediately
            _print("Looking for login button...")
            login_btn = await self.page.wait_for_selector(
                '#mfaLogonButton, #logOnbtn, button[type="submit"], input[type="submit"]',
                timeout=3000
            )
            
            if login_btn:
                _print("  Found login button ✓")
                _print("Clicking login button...")
                await login_btn.click()
                _print("  Login form submitted ✓")
            
            # CAPTURE: After login form submission - capture state for debugging
            if self.capture_steps:
                await asyncio.sleep(2)  # Wait a moment for page to react
                await self._save_debug_artifacts("after_login_submit")
            
            # PROACTIVE CHECK: Look for success indicators immediately
            # This handles cases where navigation is fast or already complete
            _print("Checking for login success or MFA...")
            
            # Short loop to check status without waiting for full network idle
            for _ in range(10):  # Check for 5 seconds total
                current_url = self.page.url
                
                # Success Check
                if any(x in current_url.lower() for x in ['accountshome', 'portfolio', 'home', 'welcome', 'dashboard']):
                    if 'login' not in current_url.lower():
                        _print("Login detected! Waiting for session to settle...")
                        try:
                            # Wait for network to settle - crucial for authentication handshakes
                            await self.page.wait_for_load_state('networkidle', timeout=5000)
                        except:
                            pass
                        _print("Login successful! ✓")
                        self._logged_in = True
                        return True
                
                # MFA Check
                mfa_url_patterns = ['sendotpcode', 'otp', 'mfa', 'securid', 'verify', '2fa']
                if any(pattern in current_url.lower() for pattern in mfa_url_patterns):
                    _print(f"⚠️  MFA/OTP PAGE DETECTED!")
                    mfa_detected = True
                    break
                
                # Check for account indicators on page
                try:
                    # Look for research link or log out link as signs of being logged in
                    success_indicator = await self.page.query_selector('a[href*="research"], a:has-text("Log out"), a:has-text("Log Out")')
                    if success_indicator:
                        _print("Login detected (via elements)! Waiting for session to settle...")
                        try:
                            # Wait for network to settle - crucial for authentication handshakes
                            await self.page.wait_for_load_state('networkidle', timeout=5000)
                        except:
                            pass
                        _print("Login successful! ✓")
                        self._logged_in = True
                        return True
                except:
                    pass
                
                await asyncio.sleep(0.5)
            
            # If not detected proactively, fall back to standard wait
            try:
                await self.page.wait_for_load_state('networkidle', timeout=5000)
            except PlaywrightTimeout:
                pass
            
            # Final check of URL
            current_url = self.page.url
            _print(f"  Current URL: {current_url[:70]}...")
            
            # Check if already logged in successfully
            if any(x in current_url.lower() for x in ['accountshome', 'portfolio', 'home', 'welcome', 'dashboard']):
                if 'login' not in current_url.lower():
                    _print("Login successful! ✓")
                    self._logged_in = True
                    return True
            
            # Check for MFA/OTP page via URL (fastest check)
            mfa_url_patterns = ['sendotpcode', 'otp', 'mfa', 'securid', 'verify', '2fa']
            mfa_detected = any(pattern in current_url.lower() for pattern in mfa_url_patterns)
            
            if mfa_detected:
                _print(f"⚠️  MFA/OTP PAGE DETECTED!")
            else:
                # Quick check for MFA input fields (short timeout)
                _print("Checking for MFA input fields...")
                mfa_selectors = ['#passcode', 'input[name="otp"]', 'input[name="passcode"]', 'input[type="tel"]']
                
                for selector in mfa_selectors:
                    try:
                        mfa_field = await self.page.wait_for_selector(selector, timeout=1500)
                        if mfa_field:
                            mfa_detected = True
                            _print(f"⚠️  MFA INPUT FIELD DETECTED!")
                            break
                    except PlaywrightTimeout:
                        continue
            
            if mfa_detected:
                # Wait for user to complete MFA manually
                _print(f"")
                _print(f"╔═══════════════════════════════════════════════════════════════╗")
                _print(f"║  MFA/OTP REQUIRED - Please complete authentication manually   ║")
                _print(f"║  Waiting up to {mfa_timeout_sec} seconds for completion...                    ║")
                _print(f"╚═══════════════════════════════════════════════════════════════╝")
                _print(f"")
                
                start_time = asyncio.get_event_loop().time()
                last_print = 0
                
                while asyncio.get_event_loop().time() - start_time < mfa_timeout_sec:
                    elapsed = int(asyncio.get_event_loop().time() - start_time)
                    
                    # Print every 10 seconds
                    if elapsed >= last_print + 10:
                        _print(f"  ⏳ Waiting for MFA completion... ({elapsed}s / {mfa_timeout_sec}s)")
                        last_print = elapsed
                    
                    # Check URL for successful navigation away from MFA
                    new_url = self.page.url
                    if any(x in new_url.lower() for x in ['accountshome', 'portfolio', 'home', 'research']):
                        if not any(p in new_url.lower() for p in mfa_url_patterns):
                            _print("MFA completed successfully! ✓")
                            self._logged_in = True
                            return True
                    
                    await asyncio.sleep(1)  # Check every second
                
                _print("✗ MFA timeout - login failed")
                return False
            
            # Not MFA page - verify login by checking for account elements
            _print("Verifying login status...")
            
            # Quick check for account indicators
            account_indicators = ['a[href*="research"]', '.account-summary', '[data-testid="account"]']
            
            for selector in account_indicators:
                try:
                    await self.page.wait_for_selector(selector, timeout=3000)
                    _print("Login successful! ✓")
                    self._logged_in = True
                    return True
                except PlaywrightTimeout:
                    continue
            
            # Final URL check
            final_url = self.page.url
            if 'etrade.com' in final_url and 'login' not in final_url.lower():
                _print("Login appears successful (on E*Trade domain) ✓")
                self._logged_in = True
                return True
            
            _print("✗ Login verification failed - unexpected page state")
            await self._save_debug_artifacts("login_failed")
            return False
            
        except Exception as e:
            _print(f"✗ Login error: {e}")
            await self._save_debug_artifacts("login_exception")
            return False
    
    async def download_analyst_reports(
        self,
        ticker: str,
        max_reports: int = 5
    ) -> List[Path]:
        """
        Navigate to Research page and download PDF analyst reports.
        
        The research page has analyst report providers on the right sidebar
        (Morgan Stanley, Argus, Refinitiv, etc.). We need to click on those
        provider links to access the actual PDF reports.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            max_reports: Maximum number of reports to download.
        
        Returns:
            List of paths to downloaded PDF files.
        """
        if not self._logged_in:
            raise RuntimeError("Not logged in. Call login() first.")
        
        downloaded_files = []
        ticker_dir = self.download_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Try up to 3 times to navigate to the correct ticker research page
        is_correct = False
        for attempt in range(3):
            _print(f"  Attempt {attempt + 1}: Navigating to {ticker} research page...")
            
            try:
                # STEP 1: Click the Search button/icon first to reveal the search input
                _print(f"    Step 1: Clicking Search trigger to reveal search box...")
            
                # Based on successful autonomous test, use these selectors in order
                search_trigger_selectors = [
                    'text=Search',  # Text-based selector - WORKED in test
                    'button:has-text("Search")',
                    'a:has-text("Search")',
                    '#UDPRightSearchDiv',
                    '.enav-search-toggle',
                ]
                
                search_trigger = None
                for selector in search_trigger_selectors:
                    try:
                        search_trigger = await self.page.wait_for_selector(selector, timeout=3000)
                        if search_trigger and await search_trigger.is_visible():
                            _print(f"    Found search trigger ✓")
                            break
                        search_trigger = None
                    except:
                        continue
            
                if not search_trigger:
                    _print(f"  ⚠ Could not find search trigger, trying direct navigation...")
                    research_url = self.ETRADE_RESEARCH_URL_TEMPLATE.format(ticker=ticker)
                    await self.page.goto(research_url, wait_until='networkidle')
                else:
                    # Click the search trigger to reveal input
                    await search_trigger.click()
                    await self._random_delay(0.5, 1)
                    
                    # CAPTURE: After clicking search trigger
                    if self.capture_steps:
                        await self._save_debug_artifacts(f"after_click_search_{ticker}")
                    
                    # Now look for the search input that appeared
                    search_input_selectors = [
                        'input[type="text"]',  # WORKED in test
                        'input[type="search"]',
                        'input[placeholder*="symbol"]',
                        'input[name*="symbol"]',
                    ]
                    
                    search_input = None
                    for selector in search_input_selectors:
                        try:
                            search_input = await self.page.wait_for_selector(selector, timeout=3000)
                            if search_input and await search_input.is_visible():
                                _print(f"    Found search input ✓")
                                break
                            search_input = None
                        except:
                            continue
                    
                    if not search_input:
                        _print(f"  ⚠ Search input not found after click, falling back to URL")
                        research_url = self.ETRADE_RESEARCH_URL_TEMPLATE.format(ticker=ticker)
                        await self.page.goto(research_url, wait_until='networkidle')
                    else:
                        # Type ticker and submit
                        # CLEAR FIRST: Using fill('') then type with delay
                        await search_input.click()
                        await search_input.fill("")
                        await self._random_delay(0.2, 0.4)
                        
                        _print(f"    Typing ticker: {ticker}...")
                        await search_input.type(ticker, delay=100)
                        
                        # Wait for autocomplete suggestions to appear (optimized)
                        _print("    Waiting for autocomplete suggestions...")
                        combined_autocomplete_sel = (
                            '.enav-suggestions, #customer-suggestions, .typeahead-results, '
                            '.search-results, .layout-search-results, .autocomplete-results, '
                            '[class*="results"], [class*="suggestions"]'
                        )
                        
                        suggestion_ready = False
                        try:
                            # Use one combined selector for speed
                            await self.page.wait_for_selector(combined_autocomplete_sel, timeout=4000, state='visible')
                            suggestion_ready = True
                            await asyncio.sleep(0.2) # Minimal breath for JS to settle
                        except:
                            # Last ditch fallback before searching for elements
                            await asyncio.sleep(0.5)

                        # NEW: Robust submission - look for suggestions first
                        # As requested: click the specific ticker in the list
                        try:
                            if not suggestion_ready:
                                # Quick check again
                                suggestion_ready = await self.page.locator(combined_autocomplete_sel).count() > 0

                            # Targets discovered from E*Trade HTML and user screenshot
                            # We want the 'TSLA' symbol specifically (usually in .enav-suggestion-left)
                            symbol_locators = [
                                self.page.locator('.enav-suggestion-left').get_by_text(ticker.upper(), exact=True),
                                self.page.locator('.autocomplete-results, .typeahead-results, .enav-customer-suggestions').get_by_text(ticker.upper(), exact=True),
                                self.page.locator('tr[data-name="symbol"]').filter(has_text=ticker.upper()),
                                self.page.locator(f'text="{ticker.upper()}"').first
                            ]
                            
                            found_suggestion = None
                            for loc in symbol_locators:
                                try:
                                    count = await loc.count()
                                    if count > 0:
                                        # Use the first visible one
                                        for i in range(count):
                                            candidate = loc.nth(i)
                                            if await candidate.is_visible():
                                                found_suggestion = candidate
                                                break
                                    if found_suggestion: break
                                except: continue
                            
                            if found_suggestion:
                                await found_suggestion.scroll_into_view_if_needed()
                                await found_suggestion.click()
                                _print(f"    Successfully clicked suggestion for {ticker} ✓")
                            else:
                                # Fallback: fuzzy match including table rows and cells
                                _print(f"    ⚠ Exact suggestion for {ticker} not found, trying fuzzy match...")
                                fuzzy_locators = [
                                    self.page.locator('.enav-suggestions tr, .autocomplete-results li, .typeahead-results div').filter(has_text=ticker.upper()),
                                    self.page.locator(f'td:has-text("{ticker.upper()}")'),
                                    self.page.locator(f'li:has-text("{ticker.upper()}")'),
                                ]
                                
                                for loc in fuzzy_locators:
                                    try:
                                        count = await loc.count()
                                        if count > 0:
                                            for i in range(count):
                                                candidate = loc.nth(i)
                                                if await candidate.is_visible():
                                                    found_suggestion = candidate
                                                    break
                                        if found_suggestion: break
                                    except: continue
                                
                                if found_suggestion:
                                    await found_suggestion.click()
                                    _print(f"    Selected fuzzy suggestion for {ticker} ✓")
                                else:
                                    _print(f"    ⚠ Could not find suggestion for {ticker}, falling back to Enter")
                                    await search_input.press("Enter")
                        except Exception as e:
                            _print(f"    ⚠ Error during suggestion selection: {e}. Falling back to Enter...")
                            try:
                                await search_input.press("Enter")
                            except:
                                pass
                        
                        # Wait for navigation
                        try:
                            await self.page.wait_for_load_state('networkidle', timeout=10000)
                        except:
                            pass
                        
                        # VERIFY: Did we land on the right ticker? (Check URL AND page content/title)
                        _print("    Verifying navigation result...")
                        is_correct = False
                        for _ in range(5): # Increase check frequency
                            try:
                                current_url = self.page.url.upper()
                                page_title = (await self.page.title()).upper()
                                
                                # Strict check: Ticker must be in the title
                                # This prevents the case where URL is correct but SPA content hasn't updated
                                if ticker.upper() in page_title and ticker.upper() in current_url:
                                    is_correct = True
                                    break
                                elif ticker.upper() in current_url:
                                    # URL is right but title isn't yet. Wait for title.
                                    _print(f"      Waiting for title to match {ticker} (current: '{page_title[:30]}')...")
                            except Exception as e:
                                # Handle "Execution context was destroyed" or other transient errors
                                if "destroyed" in str(e).lower():
                                    await asyncio.sleep(0.5)
                                    continue
                                break
                            await asyncio.sleep(1.0)
                        
                        if is_correct:
                            _print(f"    ✓ Verified ticker research page for {ticker}")
                            break # SUCCESS: Exit attempt loop
                        
                        # If we get here, verification failed for this attempt
                        current_url = self.page.url.upper()
                        try: 
                            page_title = (await self.page.title()).upper()
                        except: 
                            page_title = "UNKNOWN"
                        
                        _print(f"    Page state: Title='{page_title[:40]}...', URL={current_url[:60]}...")
                        
                        # Fallback Plan: Try to search again ON the current (wrong) page
                        if attempt < 2:
                            _print(f"    ⚠ Ticker mismatch! Trying on-page 'Go' search box for {ticker}...")
                            
                            # User requested specifically to use the box next to 'Go' button
                            # Logic: Find 'Go' button, then find the input next to it
                            try:
                                go_button = self.page.locator('button:has-text("Go"), input[value="Go"], a:has-text("Go")').first
                                if await go_button.count() > 0 and await go_button.is_visible():
                                    _print("    Found 'Go' button, looking for input...")
                                    # Try to find the input sibling or nearby input
                                    # Common pattern: Input is previous sibling or in same container
                                    # We'll try a few broad selectors for the input
                                    page_inputs = [
                                        'input[name="symbol"]', 
                                        'input[placeholder*="Symbol"]',
                                        'input.form-control',
                                        'input[type="text"]' # Broad but might work if we filter by visibility
                                    ]
                                    
                                    found_input = None
                                    for inp_sel in page_inputs:
                                        # Get visible inputs
                                        inputs = self.page.locator(inp_sel)
                                        count = await inputs.count()
                                        for i in range(count):
                                            inp = inputs.nth(i)
                                            if await inp.is_visible():
                                                # Check if it's near the Go button? 
                                                # Or just assume the first visible empty or short input is it.
                                                # E*TRADE QQQ page often has the ticker in the input.
                                                val = await inp.get_attribute('value')
                                                if val and len(val) < 6: # Likely a ticker input
                                                    found_input = inp
                                                    break
                                            if found_input: break
                                        if found_input: break
                                    
                                    if found_input:
                                        await found_input.click()
                                        await found_input.fill(ticker)
                                        # Explicit delay before clicking Go as requested
                                        await asyncio.sleep(1.0)
                                        await go_button.click()
                                        _print(f"    Clicked 'Go' for {ticker} ✓")
                                        await self.page.wait_for_load_state('networkidle', timeout=5000)
                                        # Re-verify in next loop iteration (which is actually this same loop, but we need to verify NOW)
                                        # Let's verify here to break early if successful
                                        await asyncio.sleep(2)
                                        try:
                                            if ticker.upper() in (await self.page.title()).upper():
                                                is_correct = True
                                                _print(f"    ✓ Verified ticker after 'Go' retry")
                                                break
                                        except: pass
                                    else:
                                        _print("    ⚠ Could not find input next to Go button")
                            except Exception as e:
                                _print(f"    ⚠ Error during on-page Go retry: {e}")
                            
                            # If the above specific logic didn't fix it, we continue to next attempt 
                            # (which will retry global search or deep link)
                            continue 
                        
            except Exception as e:
                _print(f"    ⚠ Attempt {attempt + 1} encountered error: {e}")
                if attempt < 2: 
                    await asyncio.sleep(1)
                    continue
                else: 
                    break
            
            # Final last-ditch fallback outside the loop if all search attempts failed
            if not is_correct:
                _print(f"    ⚠ All search attempts failed. Forcing deep-link navigation to {ticker}...")
                research_url = self.ETRADE_ANALYST_RESEARCH_URL_TEMPLATE.format(ticker=ticker)
                try:
                    await self.page.goto(research_url, wait_until='domcontentloaded')
                    await asyncio.sleep(3)
                    page_title = (await self.page.title()).upper()
                    if ticker.upper() in self.page.url.upper() or ticker.upper() in page_title:
                        _print(f"    ✓ Verified ticker via final deep-link fallback")
                        is_correct = True
                except: 
                    pass

        if not is_correct:
            _print(f"  ❌ Failed to navigate to {ticker} research page after 3 attempts.")
            return []
            
        # Explicit 1s delay after page load as requested
        await asyncio.sleep(1.0)
        await self._random_delay(1, 2)
        
        # CAPTURE: After search - see what page we landed on
        if self.capture_steps:
            await self._save_debug_artifacts(f"after_search_{ticker}")
        
        # STEP 2: Click the "Analyst Research" tab
        _print(f"  Step 2: Looking for Analyst Research tab...")
        
        analyst_tab_selectors = [
            'a:has-text("Analyst Research")',
            '[data-tab="analyst"]',
            'button:has-text("Analyst Research")',
            '.tab-item:has-text("Analyst Research")'
        ]
        
        analyst_tab = None
        for selector in analyst_tab_selectors:
            try:
                analyst_tab = await self.page.query_selector(selector)
                if analyst_tab and await analyst_tab.is_visible():
                    _print(f"    Found 'Analyst Research' tab ✓")
                    break
            except:
                continue
        
        if analyst_tab:
            await analyst_tab.click()
            await self._random_delay(2, 3)
            try:
                await self.page.wait_for_load_state('networkidle', timeout=8000)
            except:
                pass
                
            # NEW: Handle potential SSO hang (etrade.wallst.com)
            # User reported getting stuck on /sso/saml2/login.ashx
            current_url = self.page.url
            if "wallst.com" in current_url or "sso" in current_url:
                _print(f"    ⚠ Detected SSO redirection: {current_url[:50]}...")
                _print("    Waiting for SSO handshake to complete...")
                # Wait up to 10s for URL to change back or content to load
                try: 
                    # Wait for us to NOT be on the SSO page anymore (url change)
                    # or for known content to appear
                    await self.page.wait_for_function(
                        "window.location.href.indexOf('sso/saml2') === -1", 
                        timeout=10000
                    )
                    _print("    SSO redirection completed ✓")
                except:
                    _print("    ⚠ SSO appears to be stuck. Forcing reload...")
                    try:
                        await self.page.reload(wait_until='domcontentloaded')
                        await asyncio.sleep(4)
                        _print("    Page reloaded. Re-clicking Analyst Research tab...")
                        
                        # Re-find and re-click the tab because reload resets SPA state
                        analyst_tab = None
                        for selector in analyst_tab_selectors:
                            try:
                                analyst_tab = await self.page.query_selector(selector)
                                if analyst_tab and await analyst_tab.is_visible():
                                    await analyst_tab.click()
                                    _print(f"    Re-clicked 'Analyst Research' tab ✓")
                                    await asyncio.sleep(2)
                                    break
                            except: continue
                    except Exception as e:
                        _print(f"    Error during SSO reload/re-click: {e}")

        else:
            _print(f"    ⚠ Analyst Research tab not found")
            
        # CAPTURE: After clicking Analyst Research tab
        if self.capture_steps:
            await self._save_debug_artifacts(f"analyst_research_tab_{ticker}")
        
        # STEP 3: Wait for analyst content to render
        _print(f"  Step 3: Waiting for analyst content to load...")
            
        # Look for known provider names that appear on the page
        # These were discovered via autonomous browser testing
        provider_sync_selectors = [
            'a:has-text("Morgan Stanley")',
            'a:has-text("Argus")',
            'a:has-text("Refinitiv")',  # Discovered in test - LSEG shows as Refinitiv
            'a:has-text("Additional reports")',
            'a:has-text("TipRanks")',
            'a:has-text("Market Edge")',
        ]
            
        links_ready = False
        for _ in range(12):  # Poll up to 6 seconds
            for selector in provider_sync_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        for el in elements:
                            if await el.is_visible():
                                links_ready = True
                                _print(f"    Found analyst provider links ✓")
                                break
                    if links_ready: break
                except: continue
            if links_ready:
                # Success; wait a tiny bit more for final rendering
                await asyncio.sleep(1.0)
                break
            await asyncio.sleep(0.5)

            if not links_ready:
                _print(f"  ⚠ Still no provider links ready after waiting. Attempting scan...")

            # IMPORTANT: Based on autonomous debugging, the actual PDF report links use
            # the pattern: pdf.asp?docKey=... for providers like Refinitiv, Argus, MarketEdge
            # We should target these directly instead of clicking on provider headers.
        _print(f"  Looking for direct PDF report links on analyst page...")
        
        # Find all PDF links with docKey pattern (these are the actual reports)
        report_links = []
            
        # Target patterns discovered via debug: pdf.asp?docKey=...
        pdf_selectors = [
            'a[href*="pdf.asp?docKey="]',
            'a[href*="pdf.asp?wsodIssue="]',
        ]
        
        # Known equity research providers (link text patterns)
        equity_providers = [
            'Refinitiv', 'Argus', 'MarketEdge', 'LSEG',
            'Options Report', 'Analyst', 'Quantitative', 'Research'
        ]
        
        # Terms to SKIP (fund reports, disclosures, not equity research)
        skip_terms = [
            'fund report', 'morningstar on', 'mstar', 'etf', 
            'disclosure', 'risk statement', 'relationship summary',
            'form crs', 'agreement', '606 report', 'quarterly report'
        ]
            
        for selector in pdf_selectors:
            try:
                links = await self.page.query_selector_all(selector)
                for link in links:
                    if not await link.is_visible():
                        continue
                    
                    href = await link.get_attribute('href') or ""
                    text = (await link.inner_text()).strip()
                    
                    # Skip if text matches skip terms (fund reports, disclosures)
                    text_lower = text.lower()
                    href_lower = href.lower()
                    
                    if any(skip in text_lower for skip in skip_terms):
                        _print(f"    Skipping (excluded term): {text[:40]}...")
                        continue
                    if any(skip in href_lower for skip in skip_terms):
                        continue
                    
                    # Accept if text contains equity provider names
                    is_equity_report = any(p.lower() in text_lower for p in equity_providers)
                    
                    # Also accept if docKey contains the ticker symbol (strong signal)
                    ticker_in_url = ticker.upper() in href.upper()
                    
                    if is_equity_report or ticker_in_url:
                        # Clean the text for display
                        clean_text = text.replace('picture_as_pdf', '').replace('description', '').strip()
                        report_links.append({
                            'element': link,
                            'text': clean_text,
                            'href': href,
                            'has_ticker_in_url': ticker_in_url
                        })
                        _print(f"    Found PDF report: {clean_text[:40]}... {'(TICKER IN URL)' if ticker_in_url else ''}")
                    
            except Exception as e:
                continue
        
        # Deduplicate by href
        unique_links = []
        seen_hrefs = set()
        for link in report_links:
            if link['href'] not in seen_hrefs:
                seen_hrefs.add(link['href'])
                unique_links.append(link)
            
        # Sort: prefer links with ticker in URL first
        unique_links.sort(key=lambda x: (not x['has_ticker_in_url']))
        report_links = unique_links[:max_reports]
        
        _print(f"  Found {len(report_links)} equity research PDF link(s)")
        
        # Directly download each PDF link found
        if not report_links:
            _print(f"  ⚠️ No equity research PDF links found for {ticker}")
            return downloaded_files
        
        for report_info in report_links:
            if len(downloaded_files) >= max_reports:
                break
                
            report_text = report_info['text']
            _print(f"  Downloading: {report_text[:40]}...")
                
            try:
                await self._random_delay(1, 2)
                
                element = report_info['element']
                href = report_info['href']
                
                # Generate safe filename
                safe_name = re.sub(r'[^\w\s-]', '', report_text)[:30].strip()
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{ticker}_{safe_name}_{timestamp}.pdf"
                filepath = ticker_dir / filename
                
                # Construct full URL for the report viewer
                import urllib.parse
                if href.startswith('http'):
                    viewer_url = href
                else:
                    # Use a proper base URL for wallst.com
                    base_url = "https://www.etrade.wallst.com/v1/common/"
                    viewer_url = urllib.parse.urljoin(base_url, href)

                _print(f"  Accessing viewer: {viewer_url[:60]}...")
                
                # Click or navigate to viewer and handle extraction
                try:
                    # Create new page for extraction to preserve current page state
                    new_page = await self.page.context.new_page()
                    await new_page.goto(viewer_url, wait_until='networkidle', timeout=30000)
                    await asyncio.sleep(2)
                    
                    # Get HTML content and look for frame-based PDF
                    html_content = await new_page.content()
                    
                    # Pattern for frame-based PDF (found in debug)
                    # <frame src="/cgi-bin/upload.dll/file.pdf?..."
                    frame_match = re.search(r'<frame[^>]+src=["\']([^"\']*(?:\.pdf|upload\.dll)[^"\']*)["\']', html_content, re.IGNORECASE)
                    
                    pdf_url = None
                    if frame_match:
                        pdf_path = frame_match.group(1)
                        if pdf_path.startswith('/'):
                            pdf_url = f"https://www.etrade.wallst.com{pdf_path}"
                        else:
                            pdf_url = urllib.parse.urljoin(new_page.url, pdf_path)
                        _print(f"    Found frame PDF URL ✓")
                    else:
                        # Look for iframe
                        iframe_match = re.search(r'<iframe[^>]+src=["\']([^"\']*\.pdf[^"\']*)["\']', html_content, re.IGNORECASE)
                        if iframe_match:
                            pdf_url = urllib.parse.urljoin(new_page.url, iframe_match.group(1))
                            _print(f"    Found iframe PDF URL ✓")
                    
                    if pdf_url:
                        try:
                            _print(f"    Fetching direct PDF (request context)...")
                            # Use use request context to fetch raw bytes directly
                            # This avoids browser download events which can be flaky
                            response = await self.page.context.request.get(pdf_url)
                            content = await response.body()
                            
                            if content.startswith(b'%PDF-'):
                                with open(filepath, 'wb') as f:
                                    f.write(content)
                                downloaded_files.append(filepath)
                                _print(f"    ✓ Saved actual PDF (request): {filename} ({len(content):,} bytes)")
                            else:
                                _print(f"    ⚠ Response is not a valid PDF content")
                        except Exception as d_err:
                            _print(f"    ⚠ Direct fetch failed: {d_err}, trying fallback...")
                            # Fallback to rendered capture
                            content = await new_page.pdf()
                            if content:
                                with open(filepath, 'wb') as f:
                                    f.write(content)
                                downloaded_files.append(filepath)
                                _print(f"    ✓ Saved rendered PDF: {filename} ({len(content):,} bytes)")
                    else:
                        # Fallback to rendered PDF if no direct link found
                        _print(f"    Falling back to rendered PDF capture...")
                        content = await new_page.pdf()
                        if content:
                            with open(filepath, 'wb') as f:
                                f.write(content)
                            downloaded_files.append(filepath)
                            _print(f"    ✓ Saved rendered PDF: {filename} ({len(content):,} bytes)")

                    await new_page.close()
                            
                except Exception as extraction_err:
                    _print(f"    ⚠ Extraction failed: {extraction_err}")
                        
            except Exception as e:
                _print(f"    ⚠️ Error downloading: {e}")
                continue
            # Method completion
            return downloaded_files
    
    def _is_valid_pdf(self, content: bytes) -> bool:
        """Check if content looks like a PDF."""
        # Check magic bytes %PDF-
        return content.startswith(b'%PDF-')

    async def _save_pdf_content(self, content: bytes, filepath: Path) -> bool:
        """Save content to file if it is a valid PDF."""
        if self._is_valid_pdf(content):
            with open(filepath, 'wb') as f:
                f.write(content)
            return True
        else:
            # Try to read first few chars to see what it is
            try:
                preview = content[:100].decode('utf-8', errors='ignore').replace('\n', ' ')
                _print(f"      ⚠ Downloaded content is not a PDF. Start: {preview}...")
            except:
                _print(f"      ⚠ Downloaded content is not a valid PDF (bytes).")
            return False

    async def _download_pdfs_from_current_page(
        self,
        ticker_dir: Path,
        max_reports: int,
        source_name: str = "page"
    ) -> List[Path]:
        """
        Download PDF files from the current page.
        
        Args:
            ticker_dir: Directory to save PDFs.
            max_reports: Maximum number of PDFs to download.
            source_name: Name of the source for logging.
        
        Returns:
            List of paths to downloaded PDFs.
        """
        downloaded_files = []
        ticker = ticker_dir.name
        
        _print(f"    Searching for PDF links on {source_name}...")
        
        # Look for PDF links
        pdf_link_selectors = [
            'a[href$=".pdf"]',
            'a[href*=".pdf"]',
            'a:has-text("PDF")',
            'a:has-text("Download Report")',
            'a:has-text("View Report")',
            'a:has-text("Full Report")',
            '[data-testid="report-download"]',
        ]
        
        # Terms to skip (not actual analyst reports)
        skip_terms = [
            'disclosure', 'terms', 'privacy', 'risk statement', 
            'relationship summary', 'form crs', 'customer agreement',
            'disclaimer', 'important information', 'notices'
        ]
        
        pdf_links = []
        for selector in pdf_link_selectors:
            try:
                links = await self.page.query_selector_all(selector)
                for link in links:
                    href = await link.get_attribute('href')
                    text = await link.inner_text()
                    
                    if not href:
                        continue
                        
                    # Filter out non-report PDFs
                    text_lower = text.lower() if text else ""
                    href_lower = href.lower()
                    
                    if any(skip in href_lower for skip in skip_terms) or \
                       any(skip in text_lower for skip in skip_terms):
                        continue
                        
                    pdf_links.append((link, href, text.strip() if text else 'report'))
            except Exception:
                continue
        
        # Deduplicate by href
        seen_hrefs = set()
        unique_links = []
        for link, href, text in pdf_links:
            if href not in seen_hrefs:
                seen_hrefs.add(href)
                unique_links.append((link, href, text))
        
        pdf_links = unique_links[:max_reports]
        
        if not pdf_links:
            _print(f"    No PDF report links found on {source_name}")
            return []
        
        _print(f"    Found {len(pdf_links)} PDF link(s)")
        
        # Download each PDF
        for i, (link, href, text) in enumerate(pdf_links):
            try:
                await self._random_delay(1, 2)
                
                # Clean filename
                safe_name = re.sub(r'[^\w\s-]', '', text)[:30]
                safe_source = re.sub(r'[^\w\s-]', '', source_name)[:20]
                timestamp = datetime.now().strftime('%Y%m%d')
                filename = f"{ticker}_{safe_source}_{safe_name}_{timestamp}.pdf"
                filepath = ticker_dir / filename
                
                _print(f"    Downloading [{i+1}/{len(pdf_links)}]: {safe_name}...")
                
                # Try to download. We handle two main scenarios:
                # 1. Click triggers a download event
                # 2. Click opens a new tab/page (blob or direct PDF)
                try:
                    async with self.page.context.expect_page() as new_page_info:
                        try:
                            async with self.page.expect_download(timeout=10000) as download_info:
                                await link.click()
                        
                            # If we get a download event, save it
                            download = await download_info.value
                            await download.save_as(filepath)
                            downloaded_files.append(filepath)
                            _print(f"      Saved via download event ✓")
                            continue
                        except PlaywrightTimeout:
                            # No download event. Check if a new tab was opened
                            new_page = await new_page_info.value
                            
                            # Give it a moment to start loading
                            try:
                                await new_page.wait_for_load_state('domcontentloaded', timeout=5000)
                            except:
                                pass
                                
                            pdf_url = new_page.url
                            _print(f"      Opened in new tab: {pdf_url[:60]}...")
                            
                            if pdf_url.startswith('blob:'):
                                # Special handling for blob URLs via evaluation
                                _print(f"      Attempting blob evaluation via XHR...")
                                try:
                                    # Use XMLHttpRequest as it sometimes bypasses fetch CSP
                                    content_b64 = await new_page.evaluate("""async (url) => {
                                        return new Promise((resolve, reject) => {
                                            const xhr = new XMLHttpRequest();
                                            xhr.open('GET', url, true);
                                            xhr.responseType = 'blob';
                                            xhr.onload = function() {
                                                const reader = new FileReader();
                                                reader.onloadend = () => resolve(reader.result.split(',')[1]);
                                                reader.onerror = reject;
                                                reader.readAsDataURL(xhr.response);
                                            };
                                            xhr.onerror = () => reject(new Error('XHR failed'));
                                            xhr.send();
                                        });
                                    }""", pdf_url)
                                    
                                    import base64
                                    pdf_bytes = base64.b64decode(content_b64)
                                    
                                    if await self._save_pdf_content(pdf_bytes, filepath):
                                        downloaded_files.append(filepath)
                                        _print(f"      Saved via blob evaluation ✓")
                                        await new_page.close()
                                        continue
                                except Exception as blob_err:
                                    _print(f"      ⚠ Blob evaluation failed: {blob_err}")
                            
                            # For regular URLs in new tab, we try to fetch via page context
                            if pdf_url.startswith('http'):
                                await new_page.close()
                                response = await self.page.request.get(pdf_url, headers={
                                    "Referer": self.page.url
                                })
                                if response.ok:
                                    content = await response.body()
                                    if await self._save_pdf_content(content, filepath):
                                        downloaded_files.append(filepath)
                                        _print(f"      Saved via session-aware fetch ✓")
                                        continue
                            else:
                                await new_page.close()

                except Exception as e:
                    _print(f"      ⚠ Event-based download failed: {str(e)[:100]}")
                
                # Fallback: Direct link fetch (if not already downloaded)
                if href.startswith('http') or href.startswith('/'):
                    full_href = href if href.startswith('http') else f"https://www.etrade.wallst.com{href}"
                    _print(f"      Trying direct fetch: {full_href[:60]}...")
                    response = await self.page.request.get(full_href, headers={
                        "Referer": self.page.url
                    })
                    if response.ok:
                        content = await response.body()
                        if await self._save_pdf_content(content, filepath):
                            downloaded_files.append(filepath)
                            _print(f"      Saved via fallback fetch ✓")
                    else:
                        _print(f"      ✗ Failed to download (HTTP {response.status})")
                
            except Exception as e:
                _print(f"      ✗ Failed: {e}")
                continue
        
        return downloaded_files
    
    async def scrape_all_tickers(
        self,
        tickers: List[str],
        max_reports_per_ticker: int = 5,
        rate_limit_sec: float = 5.0
    ) -> Dict[str, List[Path]]:
        """
        Batch process all tickers with rate limiting.
        
        Args:
            tickers: List of ticker symbols.
            max_reports_per_ticker: Max reports to download per ticker.
            rate_limit_sec: Minimum seconds between ticker requests.
        
        Returns:
            Dict mapping ticker to list of downloaded PDF paths.
        """
        results = {}
        
        _print(f"Starting to scrape {len(tickers)} ticker(s)...")
        
        for i, ticker in enumerate(tickers):
            _print(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")
            
            try:
                pdfs = await self.download_analyst_reports(
                    ticker, max_reports=max_reports_per_ticker
                )
                results[ticker] = pdfs
                if pdfs:
                    _print(f"  Downloaded {len(pdfs)} report(s) for {ticker} ✓")
                else:
                    _print(f"  No reports found for {ticker}")
            except Exception as e:
                _print(f"  ✗ Failed to process {ticker}: {e}")
                results[ticker] = []
            
            # Rate limiting
            if i < len(tickers) - 1:
                delay = random.uniform(rate_limit_sec, rate_limit_sec + 3)
                _print(f"  Rate limiting: waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # Summary
        total_downloaded = sum(len(pdfs) for pdfs in results.values())
        tickers_with_reports = sum(1 for pdfs in results.values() if pdfs)
        
        _print(f"\n✓ Scraping complete: {total_downloaded} PDF(s) from {tickers_with_reports}/{len(tickers)} tickers")
        
        return results


async def scrape_reports_for_tickers(
    tickers: List[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    download_dir: Optional[Path] = None,
    headless: bool = True,
    max_reports: int = 5,
    session_file: Optional[Path] = None
) -> Dict[str, List[Path]]:
    """
    Convenience function to scrape reports for a list of tickers.
    
    Args:
        tickers: List of ticker symbols.
        username: E*Trade web username (optional if session_file provided).
        password: E*Trade web password (optional if session_file provided).
        download_dir: Directory for downloaded PDFs.
        headless: Run browser in headless mode.
        max_reports: Max reports per ticker.
        session_file: Path to saved session file.
    
    Returns:
        Dict mapping ticker to list of downloaded PDF paths.
    """
    _print("Creating scraper instance...")
    
    async with ETradeReportScraper(
        headless=headless,
        download_dir=download_dir,
        session_file=session_file
    ) as scraper:
        _print("Attempting to log in to E*Trade...")
        login_success = await scraper.login(username, password)
        
        if not login_success:
            _print("✗ Login failed! Cannot proceed.")
            raise RuntimeError("Failed to log in to E*Trade")
        
        _print("Login successful! Starting report scraping...")
        return await scraper.scrape_all_tickers(
            tickers, max_reports_per_ticker=max_reports
        )


if __name__ == "__main__":
    # Standalone test
    import sys
    
    async def test():
        print("E*Trade Report Scraper")
        print("=" * 50)
        
        # Get credentials from environment or prompt
        username = os.environ.get('ETRADE_USERNAME')
        password = os.environ.get('ETRADE_PASSWORD')
        
        if not username or not password:
            print("Set ETRADE_USERNAME and ETRADE_PASSWORD environment variables")
            print("Or pass credentials programmatically")
            sys.exit(1)
        
        test_tickers = ['AAPL']  # Start with one ticker for testing
        
        results = await scrape_reports_for_tickers(
            tickers=test_tickers,
            username=username,
            password=password,
            headless=False,  # Visible for debugging
            max_reports=1
        )
        
        print("\nResults:")
        for ticker, pdfs in results.items():
            print(f"  {ticker}: {len(pdfs)} report(s)")
            for pdf in pdfs:
                print(f"    - {pdf}")
    
    asyncio.run(test())
