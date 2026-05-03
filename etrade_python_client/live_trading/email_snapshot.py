#!/usr/bin/env python3
"""
Email a full-page screenshot of screened_option_pairs.html (with rendered
Chart.js charts) to the configured NOTIFY_EMAIL address.

Usage:
    python3 email_snapshot.py            # screenshot + email
    python3 email_snapshot.py --dry-run  # screenshot only, no email
"""

import argparse
import configparser
import os
import smtplib
import sys
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
HTML_FILE = SCRIPT_DIR / "screened_option_pairs.html"
SNAPSHOT_FILE = SCRIPT_DIR / "snapshot.png"
CONFIG_FILE = SCRIPT_DIR / "config.ini"


def take_screenshot(html_path: Path, out_path: Path) -> Path:
    """Render the HTML in headless Chromium and save a full-page screenshot."""
    from playwright.sync_api import sync_playwright

    file_url = html_path.as_uri()
    print(f"📸 Rendering {file_url} ...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(file_url, wait_until="networkidle")

        # Wait for Chart.js to finish rendering on both canvases
        page.wait_for_selector("canvas#spyPriceChart", timeout=10_000)
        page.wait_for_selector("canvas#spyChart", timeout=10_000)
        # Extra small delay to let Chart.js animations complete
        page.wait_for_timeout(1500)

        page.screenshot(path=str(out_path), full_page=True)
        browser.close()

    size_kb = out_path.stat().st_size / 1024
    print(f"✅ Screenshot saved: {out_path} ({size_kb:.0f} KB)")
    return out_path


def send_email(image_path: Path, config: configparser.ConfigParser) -> bool:
    """Send the screenshot as an inline-embedded email."""
    gmail_address = config.get("EMAIL", "GMAIL_ADDRESS", fallback=None)
    gmail_app_password = config.get("EMAIL", "GMAIL_APP_PASSWORD", fallback=None)
    notify_email = config.get("EMAIL", "NOTIFY_EMAIL", fallback=gmail_address)

    # Strip quotes that ConfigParser may preserve
    for attr in (gmail_address, gmail_app_password, notify_email):
        if attr:
            attr = attr.strip("\"'")
    gmail_address = gmail_address.strip("\"'") if gmail_address else None
    gmail_app_password = gmail_app_password.strip("\"'") if gmail_app_password else None
    notify_email = notify_email.strip("\"'") if notify_email else None

    if not gmail_address or not gmail_app_password:
        print("⚠️  Gmail credentials not configured in config.ini [EMAIL] section.")
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"📊 Options Snapshot – {now_str}"

    # Build multipart message with inline image
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = notify_email

    html_body = f"""\
<html>
<body style="font-family: Arial, sans-serif; color: #333; padding: 16px;">
  <h2 style="color:#1a73e8;">📊 Screened Option Pairs Snapshot</h2>
  <p style="color:#666;">Generated at <strong>{now_str}</strong></p>
  <img src="cid:snapshot" style="max-width:100%; border:1px solid #ddd; border-radius:8px;" />
  <p style="color:#999; font-size:12px; margin-top:16px;">
    Sent automatically from E*Trade Cover Call Script
  </p>
</body>
</html>"""

    msg_alt = MIMEMultipart("alternative")
    msg_alt.attach(MIMEText(f"Options Snapshot – {now_str}\nSee attached image.", "plain"))
    msg_alt.attach(MIMEText(html_body, "html"))
    msg.attach(msg_alt)

    # Attach image with Content-ID for inline display
    with open(image_path, "rb") as f:
        img = MIMEImage(f.read(), _subtype="png")
    img.add_header("Content-ID", "<snapshot>")
    img.add_header("Content-Disposition", "inline", filename="snapshot.png")
    msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.sendmail(gmail_address, notify_email, msg.as_string())
        print(f"📧 Snapshot email sent to {notify_email}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Screenshot & email screened_option_pairs.html")
    parser.add_argument("--dry-run", action="store_true", help="Take screenshot but do not send email")
    args = parser.parse_args()

    # ── Trading Day Check ────────────────────────────────────────────────────
    try:
        from pandas_market_calendars import get_calendar
        import pytz
        
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        today_et = now_et.date()
        
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=today_et, end_date=today_et)
        
        if schedule.empty:
            print(f"📅 Today ({today_et}) is not a trading day (NYSE). Skipping snapshot.")
            sys.exit(0)
    except Exception as e:
        print(f"⚠️  Warning: Could not verify trading day status: {e}")
        # Continue as fallback

    if not HTML_FILE.exists():
        print(f"❌ HTML file not found: {HTML_FILE}")
        sys.exit(1)

    # Take screenshot
    take_screenshot(HTML_FILE, SNAPSHOT_FILE)

    if args.dry_run:
        print("🏁 Dry run complete — email not sent.")
        return

    # Load config and send
    config = configparser.ConfigParser()
    config.read(str(CONFIG_FILE))
    success = send_email(SNAPSHOT_FILE, config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
