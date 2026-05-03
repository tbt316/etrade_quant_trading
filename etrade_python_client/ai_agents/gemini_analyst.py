#!/usr/bin/env python3
"""
Gemini Analyst Module

Uses Google's Gemini AI to analyze analyst PDF reports and provide
investment recommendations with Bull/Bear thesis.

Updated to use the new google.genai SDK (replacing deprecated google.generativeai)
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required. Install with: pip install google-genai"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structured result from Gemini analysis of analyst reports."""
    
    ticker: str
    bull_thesis: str
    bear_thesis: str
    actionable_advice: str
    confidence: str  # High, Medium, Low
    price_target: Optional[str] = None
    rating: Optional[str] = None  # Buy, Hold, Sell if mentioned
    sources: List[str] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'bull_thesis': self.bull_thesis,
            'bear_thesis': self.bear_thesis,
            'actionable_advice': self.actionable_advice,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'rating': self.rating,
            'sources': self.sources,
            'error': self.error,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        if self.error:
            return f"[{self.ticker}] Error: {self.error}"
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║ {self.ticker} Analysis Summary
╠══════════════════════════════════════════════════════════════════╣
║ Rating: {self.rating or 'N/A'} | Price Target: {self.price_target or 'N/A'} | Confidence: {self.confidence}
╠══════════════════════════════════════════════════════════════════╣
║ 📈 BULL THESIS:
║ {self._wrap_text(self.bull_thesis)}
╠══════════════════════════════════════════════════════════════════╣
║ 📉 BEAR THESIS:
║ {self._wrap_text(self.bear_thesis)}
╠══════════════════════════════════════════════════════════════════╣
║ 💡 ACTIONABLE ADVICE:
║ {self._wrap_text(self.actionable_advice)}
╠══════════════════════════════════════════════════════════════════╣
║ Sources: {', '.join(self.sources) if self.sources else 'N/A'}
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def _wrap_text(self, text: str, width: int = 60) -> str:
        """Wrap text for display."""
        if not text:
            return "N/A"
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n║ '.join(lines)


class GeminiAnalyst:
    """
    Gemini-powered analyst for investment reports.
    
    Uses Google's new genai SDK to upload and analyze PDF analyst reports,
    extracting Bull/Bear thesis and providing actionable recommendations.
    """
    
    ANALYSIS_PROMPT = """You are a senior investment analyst. I'm providing you with analyst reports for {ticker}.

Please analyze these reports thoroughly and provide:

1. **BULL THESIS** (2-3 sentences): The strongest arguments for why this stock could outperform. Focus on catalysts, competitive advantages, and growth drivers.

2. **BEAR THESIS** (2-3 sentences): The key risks and concerns that could cause underperformance. Focus on valuation concerns, competitive threats, and execution risks.

3. **ACTIONABLE ADVICE** (2-3 sentences): Based on the reports, what specific action should an investor take? Be specific about position sizing if relevant (e.g., "Hold current position", "Trim by 25%", "Add on pullbacks below $X").

4. **CONFIDENCE LEVEL**: Rate as High, Medium, or Low based on the quality and consensus of the analyst reports.

5. **PRICE TARGET**: If mentioned in the reports, provide the consensus or range.

6. **RATING**: If an overall rating is given (Buy/Hold/Sell or equivalent), state it.

Format your response as:
BULL_THESIS: [your bull thesis]
BEAR_THESIS: [your bear thesis]
ACTIONABLE_ADVICE: [your advice]
CONFIDENCE: [High/Medium/Low]
PRICE_TARGET: [target or "Not specified"]
RATING: [rating or "Not specified"]
SOURCES: [list the analyst firms that authored the reports, comma-separated]
"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash"
    ):
        """
        Initialize Gemini analyst.
        
        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Gemini model to use. Options: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-pro-latest
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )
        
        # Initialize the new google.genai client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        
        logger.info(f"Gemini analyst initialized with model: {model}")
    
    async def analyze_holdings(
        self,
        ticker: str,
        pdf_paths: List[Path]
    ) -> AnalysisResult:
        """
        Upload PDFs and analyze with Gemini.
        
        Args:
            ticker: Stock ticker symbol.
            pdf_paths: List of paths to analyst report PDFs.
        
        Returns:
            AnalysisResult with Bull/Bear thesis and recommendations.
        """
        if not pdf_paths:
            return AnalysisResult(
                ticker=ticker,
                bull_thesis="",
                bear_thesis="",
                actionable_advice="",
                confidence="Low",
                error="No PDF reports provided"
            )
        
        logger.info(f"Analyzing {len(pdf_paths)} report(s) for {ticker}...")
        
        try:
            # Upload PDFs to Gemini using new SDK
            uploaded_files = []
            for pdf_path in pdf_paths:
                if not pdf_path.exists():
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                
                logger.info(f"Uploading {pdf_path.name}...")
                
                # Upload file using new SDK
                uploaded_file = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda p=pdf_path: self.client.files.upload(file=p)
                )
                uploaded_files.append(uploaded_file)
                
                # Wait for file to be processed
                while uploaded_file.state == "PROCESSING":
                    await asyncio.sleep(1)
                    uploaded_file = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda f=uploaded_file: self.client.files.get(name=f.name)
                    )
                
                if uploaded_file.state == "FAILED":
                    logger.error(f"Failed to process {pdf_path.name}")
                    continue
            
            if not uploaded_files:
                return AnalysisResult(
                    ticker=ticker,
                    bull_thesis="",
                    bear_thesis="",
                    actionable_advice="",
                    confidence="Low",
                    error="Failed to upload any PDF files"
                )
            
            # Generate analysis using new SDK
            prompt = self.ANALYSIS_PROMPT.format(ticker=ticker)
            
            # Build content with uploaded files and prompt
            content_parts = []
            for f in uploaded_files:
                content_parts.append(types.Part.from_uri(file_uri=f.uri, mime_type="application/pdf"))
            content_parts.append(prompt)
            
            # Generate analysis with retry logic for 429 errors
            max_retries = 5
            retry_delay = 10
            
            for attempt in range(max_retries):
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.models.generate_content(
                            model=self.model_name,
                            contents=content_parts
                        )
                    )
                    break # Success!
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if attempt < max_retries - 1:
                            # Try to parse wait time from error message
                            # Pattern: "Please retry in 36.180037638s."
                            import re
                            wait_match = re.search(r'retry in (\d+(\.\d+)?)s', err_str)
                            
                            if wait_match:
                                wait_time = float(wait_match.group(1)) + 2.0  # Add 2s buffer
                                logger.warning(f"Quota exceeded. limits say wait {wait_match.group(1)}s. Sleeping {wait_time:.1f}s...")
                            else:
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"Quota exceeded (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                            
                            await asyncio.sleep(wait_time)
                            continue
                    raise e # Re-raise if not 429 or last attempt
            
            # Parse response
            result = self._parse_response(ticker, response.text, pdf_paths)
            
            # Cleanup uploaded files
            for f in uploaded_files:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda fn=f.name: self.client.files.delete(name=fn)
                    )
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            # Check for quota errors
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error(f"Quota exceeded for {ticker}. Please check your Gemini API plan.")
            elif "404" in str(e) and "not found" in str(e):
                logger.error(f"Model '{self.model_name}' not found. Please verify the model name.")
            
            logger.error(f"Analysis failed for {ticker}: {e}")
            return AnalysisResult(
                ticker=ticker,
                bull_thesis="",
                bear_thesis="",
                actionable_advice="",
                confidence="Low",
                error=str(e)
            )
    
    def _parse_response(
        self,
        ticker: str,
        response_text: str,
        pdf_paths: List[Path]
    ) -> AnalysisResult:
        """Parse structured response from Gemini."""
        
        def extract_field(text: str, field: str) -> str:
            """Extract a field value from the response."""
            import re
            pattern = rf'{field}:\s*(.+?)(?=\n[A-Z_]+:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        
        bull_thesis = extract_field(response_text, "BULL_THESIS")
        bear_thesis = extract_field(response_text, "BEAR_THESIS")
        actionable_advice = extract_field(response_text, "ACTIONABLE_ADVICE")
        confidence = extract_field(response_text, "CONFIDENCE") or "Medium"
        price_target = extract_field(response_text, "PRICE_TARGET")
        rating = extract_field(response_text, "RATING")
        sources_str = extract_field(response_text, "SOURCES")
        
        sources = []
        if sources_str:
            sources = [s.strip() for s in sources_str.split(',')]
        
        # Add PDF filenames as sources if none extracted
        if not sources:
            sources = [p.stem for p in pdf_paths]
        
        return AnalysisResult(
            ticker=ticker,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            actionable_advice=actionable_advice,
            confidence=confidence,
            price_target=price_target if price_target != "Not specified" else None,
            rating=rating if rating != "Not specified" else None,
            sources=sources,
            raw_response=response_text
        )
    
    async def analyze_multiple(
        self,
        ticker_pdf_map: dict
    ) -> List[AnalysisResult]:
        """
        Analyze multiple tickers in parallel.
        
        Args:
            ticker_pdf_map: Dict mapping ticker to list of PDF paths.
        
        Returns:
            List of AnalysisResult for each ticker.
        """
        tasks = [
            self.analyze_holdings(ticker, pdfs)
            for ticker, pdfs in ticker_pdf_map.items()
        ]
        
        # Run sequentially to avoid rate limiting
        results = []
        batch_size = 1  # Process 1 at a time (sequential)
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Analysis failed: {result}")
                else:
                    results.append(result)
            
            # Rate limit between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(2)
        
        return results


async def analyze_holdings(
    ticker: str,
    pdf_paths: List[Path],
    model: str = "gemini-1.5-pro",
    api_key: Optional[str] = None
) -> AnalysisResult:
    """
    Convenience function to analyze holdings for a single ticker.
    
    Args:
        ticker: Stock ticker symbol.
        pdf_paths: List of paths to analyst report PDFs.
        model: Gemini model to use.
        api_key: Gemini API key (falls back to env var).
    
    Returns:
        AnalysisResult with Bull/Bear thesis and recommendations.
    """
    analyst = GeminiAnalyst(api_key=api_key, model=model)
    return await analyst.analyze_holdings(ticker, pdf_paths)


if __name__ == "__main__":
    # Standalone test
    import sys
    
    async def test():
        print("Gemini Analyst Module")
        print("=" * 50)
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("Set GEMINI_API_KEY environment variable")
            sys.exit(1)
        
        # Test with a sample PDF if available
        test_pdfs = list(Path("./analyst_reports").glob("**/*.pdf"))
        
        if not test_pdfs:
            print("No test PDFs found in ./analyst_reports/")
            print("Run the scraper first to download some reports.")
            sys.exit(1)
        
        ticker = test_pdfs[0].parent.name  # Assume ticker is parent dir name
        
        print(f"\nAnalyzing {ticker} with {len(test_pdfs)} PDF(s)...")
        
        result = await analyze_holdings(ticker, test_pdfs[:2])
        print(result)
    
    asyncio.run(test())
