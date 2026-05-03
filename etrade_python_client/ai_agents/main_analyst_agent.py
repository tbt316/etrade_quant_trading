#!/usr/bin/env python3
"""
Main Analyst Agent - Integration Module

Orchestrates the complete workflow:
1. Fetch portfolio holdings via E*Trade API
2. Scrape analyst PDF reports via Playwright
3. Analyze reports with Gemini AI
4. Generate consolidated recommendations
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_trading.portfolio_manager import get_current_holdings, filter_tickers_for_research
from ai_agents.report_scraper import ETradeReportScraper, scrape_reports_for_tickers
from ai_agents.gemini_analyst import GeminiAnalyst, AnalysisResult

# Constants
SESSION_FILE = Path("etrade_session.json")

# Configure logging to show on console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def print_step(step_num: int, message: str, char: str = "►"):
    """Print a visible step marker to the console."""
    print(f"\n{char * 3} STEP {step_num}: {message} {char * 3}")
    sys.stdout.flush()


def print_progress(message: str):
    """Print a progress message."""
    print(f"    → {message}")
    sys.stdout.flush()


def print_success(message: str):
    """Print a success message."""
    print(f"    ✓ {message}")
    sys.stdout.flush()


def print_warning(message: str):
    """Print a warning message."""
    print(f"    ⚠ {message}")
    sys.stdout.flush()


def print_error(message: str):
    """Print an error message."""
    print(f"    ✗ {message}")
    sys.stdout.flush()


class AnalystReportAgent:
    """
    Main orchestrator for the Analyst Report Agent.
    
    Coordinates portfolio fetching, web scraping, and AI analysis
    to provide actionable investment recommendations.
    """
    
    def __init__(
        self,
        accounts=None,
        session=None,
        base_url: str = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        download_dir: Optional[Path] = None,
        headless: bool = True,
        session_file: Optional[Path] = None
    ):
        """
        Initialize the Analyst Report Agent.
        
        Args:
            accounts: Authenticated Accounts instance (optional if session provided).
            session: Authenticated session for E*Trade API.
            base_url: E*Trade API base URL.
            gemini_api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
            gemini_model: Gemini model to use for analysis.
            download_dir: Directory for downloaded PDFs.
            headless: Run browser in headless mode.
            session_file: Path to saved web session file.
        """
        self.accounts = accounts
        self.session = session
        self.base_url = base_url
        self.gemini_api_key = gemini_api_key or os.environ.get('GEMINI_API_KEY')
        self.gemini_model = gemini_model
        self.download_dir = download_dir or Path("./analyst_reports")
        self.headless = headless
        self.session_file = session_file
        
        # Lazy initialization
        self._gemini_analyst = None
        self._scraper = None
    
    @property
    def gemini_analyst(self) -> GeminiAnalyst:
        """Lazy-initialize Gemini analyst."""
        if self._gemini_analyst is None:
            self._gemini_analyst = GeminiAnalyst(
                api_key=self.gemini_api_key,
                model=self.gemini_model
            )
        return self._gemini_analyst
    
    async def run(
        self,
        web_username: Optional[str] = None,
        web_password: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        max_reports_per_ticker: int = 5,
        exclude_etfs: bool = False,
        skip_scraping: bool = False
    ) -> List[AnalysisResult]:
        """
        Run the complete analyst agent workflow.
        
        Args:
            web_username: E*Trade web username for scraping.
            web_password: E*Trade web password for scraping.
            tickers: Optional list of tickers to analyze. If None, fetches from portfolio.
            max_reports_per_ticker: Maximum PDF reports to download per ticker.
            exclude_etfs: If True, exclude common ETFs from analysis.
            skip_scraping: If True, skip scraping and use existing PDFs.
        
        Returns:
            List of AnalysisResult for each analyzed ticker.
        """
        print("\n" + "=" * 60)
        print("   🤖 ANALYST REPORT AGENT - Starting")
        print("=" * 60)
        print(f"   Model: {self.gemini_model}")
        print(f"   Download Dir: {self.download_dir}")
        print(f"   Headless Mode: {self.headless}")
        print("=" * 60)
        sys.stdout.flush()
        
        # Step 1: Get tickers to analyze
        print_step(1, "GETTING TICKERS TO ANALYZE")
        
        if tickers is None:
            print_progress("Fetching portfolio holdings from E*Trade API...")
            tickers = await self._fetch_holdings(exclude_etfs)
        else:
            print_progress(f"Using provided tickers: {tickers}")
        
        if not tickers:
            print_error("No tickers found to analyze. When running standalone, use --tickers to specify tickers.")
            return []
        
        print_success(f"Found {len(tickers)} ticker(s): {', '.join(tickers)}")
        
        # Step 2: Scrape analyst reports
        print_step(2, "SCRAPING ANALYST REPORTS")
        
        ticker_pdf_map = {}
        if not skip_scraping:
            print_progress(f"Launching browser (headless={self.headless})...")
            print_progress(f"Will download up to {max_reports_per_ticker} PDF(s) per ticker")
            ticker_pdf_map = await self._scrape_reports(
                tickers, web_username, web_password, max_reports_per_ticker
            )
        else:
            print_progress("Skipping scraping, using existing PDFs...")
            ticker_pdf_map = self._find_existing_pdfs(tickers)
        
        # Filter tickers with reports
        tickers_with_reports = {k: v for k, v in ticker_pdf_map.items() if v}
        tickers_without_reports = [k for k, v in ticker_pdf_map.items() if not v]
        
        if tickers_without_reports:
            print_warning(f"No reports found for: {', '.join(tickers_without_reports)}")
        
        if not tickers_with_reports:
            print_error("No analyst reports found for any ticker. Cannot proceed to analysis.")
            return []
        
        total_pdfs = sum(len(v) for v in tickers_with_reports.values())
        print_success(f"Found {total_pdfs} PDF(s) for {len(tickers_with_reports)} ticker(s)")
        
        for ticker, pdfs in tickers_with_reports.items():
            print_progress(f"  {ticker}: {len(pdfs)} report(s)")
        
        # Step 3: Analyze with Gemini
        print_step(3, "ANALYZING REPORTS WITH GEMINI AI")
        print_progress(f"Using model: {self.gemini_model}")
        
        results = await self._analyze_reports(tickers_with_reports)
        
        # Step 4: Generate summary
        print_step(4, "GENERATING ANALYSIS SUMMARY")
        
        self._print_summary(results)
        
        return results
    
    async def _fetch_holdings(self, exclude_etfs: bool) -> List[str]:
        """Fetch portfolio holdings."""
        if self.accounts is None:
            # If no accounts provided, try to create from session
            if self.session is None or self.base_url is None:
                print_error("No accounts or session provided. Use --tickers to specify tickers manually.")
                return []
            
            print_progress("Creating Accounts instance from session...")
            from accounts.accounts_bo import Accounts
            self.accounts = Accounts(self.session, self.base_url)
        
        try:
            print_progress("Calling portfolio API...")
            tickers = get_current_holdings(self.accounts)
            tickers = filter_tickers_for_research(
                tickers, exclude_etfs=exclude_etfs
            )
            return tickers
        except Exception as e:
            print_error(f"Failed to fetch holdings: {e}")
            return []
    
    async def _scrape_reports(
        self,
        tickers: List[str],
        username: Optional[str],
        password: Optional[str],
        max_reports: int
    ) -> Dict[str, List[Path]]:
        """Scrape analyst reports for tickers."""
        try:
            print_progress("Initializing Playwright browser...")
            results = await scrape_reports_for_tickers(
                tickers=tickers,
                username=username,
                password=password,
                download_dir=self.download_dir,
                headless=self.headless,
                max_reports=max_reports,
                session_file=self.session_file
            )
            
            total_pdfs = sum(len(pdfs) for pdfs in results.values())
            print_success(f"Scraping complete: {total_pdfs} PDF(s) downloaded")
            
            return results
        except Exception as e:
            print_error(f"Scraping failed: {e}")
            return {}
    
    def _find_existing_pdfs(self, tickers: List[str]) -> Dict[str, List[Path]]:
        """Find existing PDFs for tickers."""
        result = {}
        for ticker in tickers:
            ticker_dir = self.download_dir / ticker
            if ticker_dir.exists():
                pdfs = list(ticker_dir.glob("*.pdf"))
                result[ticker] = pdfs[:2]  # Limit to 2
                if pdfs:
                    print_progress(f"Found {len(pdfs)} existing PDF(s) for {ticker}")
            else:
                result[ticker] = []
        
        return result
    
    async def _analyze_reports(
        self,
        ticker_pdf_map: Dict[str, List[Path]]
    ) -> List[AnalysisResult]:
        """Analyze reports with Gemini."""
        results = []
        total = len(ticker_pdf_map)
        
        for i, (ticker, pdfs) in enumerate(ticker_pdf_map.items(), 1):
            print_progress(f"[{i}/{total}] Analyzing {ticker} ({len(pdfs)} PDF(s))...")
            
            try:
                result = await self.gemini_analyst.analyze_holdings(ticker, pdfs)
                results.append(result)
                
                if result.error:
                    print_warning(f"  Analysis had error: {result.error}")
                else:
                    rating_str = f" - Rating: {result.rating}" if result.rating else ""
                    print_success(f"  Analysis complete{rating_str}")
                    
            except Exception as e:
                print_error(f"  Analysis failed: {e}")
                results.append(AnalysisResult(
                    ticker=ticker,
                    bull_thesis="",
                    bear_thesis="",
                    actionable_advice="",
                    confidence="Low",
                    error=str(e)
                ))
            
            # Small delay between API calls
            if i < total:
                await asyncio.sleep(1)
        
        successful = sum(1 for r in results if not r.error)
        print_success(f"Successfully analyzed {successful}/{len(results)} ticker(s)")
        
        return results
    
    def _print_summary(self, results: List[AnalysisResult]):
        """Print summary of all analysis results."""
        print("\n" + "=" * 60)
        print("   📊 ANALYSIS RESULTS")
        print("=" * 60)
        
        for result in results:
            print(result)
        
        # Aggregate recommendations
        buy_signals = [r for r in results if r.rating and 'buy' in r.rating.lower()]
        hold_signals = [r for r in results if r.rating and 'hold' in r.rating.lower()]
        sell_signals = [r for r in results if r.rating and 'sell' in r.rating.lower()]
        
        print("\n" + "=" * 60)
        print("   📈 AGGREGATED SIGNALS")
        print("=" * 60)
        print(f"   🟢 BUY signals:  {len(buy_signals)} - {[r.ticker for r in buy_signals]}")
        print(f"   🟡 HOLD signals: {len(hold_signals)} - {[r.ticker for r in hold_signals]}")
        print(f"   🔴 SELL signals: {len(sell_signals)} - {[r.ticker for r in sell_signals]}")
        print("=" * 60)
    
    def export_results(
        self,
        results: List[AnalysisResult],
        output_path: Optional[Path] = None,
        format: str = "json"
    ) -> Path:
        """
        Export analysis results to file.
        
        Args:
            results: List of AnalysisResult.
            output_path: Output file path.
            format: Output format ('json' or 'csv').
        
        Returns:
            Path to the exported file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = self.download_dir / f"analysis_results_{timestamp}.{format}"
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(
                    [r.to_dict() for r in results],
                    f,
                    indent=2
                )
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'ticker', 'rating', 'price_target', 'confidence',
                    'bull_thesis', 'bear_thesis', 'actionable_advice', 'sources'
                ])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        'ticker': r.ticker,
                        'rating': r.rating,
                        'price_target': r.price_target,
                        'confidence': r.confidence,
                        'bull_thesis': r.bull_thesis,
                        'bear_thesis': r.bear_thesis,
                        'actionable_advice': r.actionable_advice,
                        'sources': ', '.join(r.sources)
                    })
        
        print_success(f"Results exported to: {output_path}")
        return output_path


async def run_analyst_agent(
    session,
    base_url: str,
    web_username: str,
    web_password: str,
    gemini_api_key: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    headless: bool = True
) -> List[AnalysisResult]:
    """
    Convenience function to run the complete analyst agent.
    
    This is the main entry point for integration with existing code.
    
    Args:
        session: Authenticated E*Trade API session.
        base_url: E*Trade API base URL.
        web_username: E*Trade web username.
        web_password: E*Trade web password.
        gemini_api_key: Gemini API key (optional, uses env var).
        tickers: Optional list of specific tickers to analyze.
        headless: Run browser in headless mode.
    
    Returns:
        List of AnalysisResult for each analyzed ticker.
    
    Example usage in etrade_cover_call_new.py:
        
        from ai_agents.main_analyst_agent import run_analyst_agent
        
        # After your existing OAuth flow:
        # session, base_url = oauth(use_sandbox, auto_login=True)
        
        results = asyncio.run(run_analyst_agent(
            session=session,
            base_url=base_url,
            web_username=username,
            web_password=password,
            gemini_api_key=os.environ.get('GEMINI_API_KEY'),
            headless=True
        ))
        
        for result in results:
            print(result)
    """
    agent = AnalystReportAgent(
        session=session,
        base_url=base_url,
        gemini_api_key=gemini_api_key,
        headless=headless
    )
    
    return await agent.run(
        web_username=web_username,
        web_password=web_password,
        tickers=tickers
    )


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Analyst Report Agent - Automated investment analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--username', '-u',
        help='E*Trade web username'
    )
    parser.add_argument(
        '--password', '-p',
        help='E*Trade web password'
    )
    parser.add_argument(
        '--tickers', '-t',
        nargs='+',
        help='Specific tickers to analyze (REQUIRED when running standalone)'
    )
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Run browser in visible mode (for debugging)'
    )
    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='Skip scraping, use existing PDFs'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file for results'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--model',
        default='gemini-2.0-flash',
        help='Gemini model to use'
    )
    parser.add_argument(
        '--max-reports', '-m',
        type=int,
        default=5,
        help='Maximum PDF reports to download per ticker'
    )
    parser.add_argument(
        '--use-session',
        action='store_true',
        help='Use saved session from auth.py (skip login, no username/password needed)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   🚀 ANALYST REPORT AGENT - Initializing")
    print("=" * 60)
    
    # Validate environment
    print_progress("Checking GEMINI_API_KEY environment variable...")
    gemini_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_key:
        print_error("GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    print_success("GEMINI_API_KEY found")
    
    # When running standalone, tickers are required
    if not args.tickers:
        print_error("--tickers is required when running standalone (no E*Trade session)")
        print("   Example: python main_analyst_agent.py -u USER -p PASS -t AAPL MSFT")
        sys.exit(1)
    
    print_success(f"Tickers to analyze: {args.tickers}")
    print_progress(f"Model: {args.model}")
    print_progress(f"Headless mode: {not args.no_headless}")
    
    async def run():
        session_file = SESSION_FILE if args.use_session else None
        
        agent = AnalystReportAgent(
            gemini_api_key=gemini_key,
            gemini_model=args.model,
            headless=not args.no_headless,
            session_file=session_file
        )
        
        results = await agent.run(
            web_username=args.username,
            web_password=args.password,
            tickers=args.tickers,
            max_reports_per_ticker=args.max_reports,
            skip_scraping=args.skip_scraping
        )
        
        if results and args.output:
            agent.export_results(results, args.output, args.format)
        
        return results
    
    try:
        results = asyncio.run(run())
        print(f"\n✅ Analysis complete. Processed {len(results)} ticker(s).")
    except KeyboardInterrupt:
        print("\n⚠️  Aborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
