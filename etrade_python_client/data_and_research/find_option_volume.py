import asyncio
import aiohttp
import datetime
import logging
import certifi
import ssl
from typing import List, Dict, Any, Optional
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import time
from data_and_research import polygonio_config

@dataclass
class OptionVolumeResult:
    """Data class for storing option volume analysis results"""
    ticker: str
    current_price: float
    atm_strike: float
    total_option_volume: int
    option_trades_count: int
    avg_daily_volume: float

class PolygonAPIError(Exception):
    """Custom exception for Polygon API errors"""
    def __init__(self, status: int, message: str, endpoint: str):
        self.status = status
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"Polygon API Error: {status} - {message} (Endpoint: {endpoint})")

class PolygonOptionVolumeAnalyzer:
    def __init__(self):
        """Initialize the Polygon Option Volume Analyzer"""
        self.api_key = polygonio_config.API_KEY
        self.max_retries = getattr(polygonio_config, 'MAX_RETRIES', 3)
        self.rate_limit_per_min = getattr(polygonio_config, 'RATE_LIMIT_PER_MIN', 300)
        
        # Rate limiting setup
        self.request_interval = 60.0 / self.rate_limit_per_min
        self.last_request_time = 0
        
        # SSL and session setup
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('polygon_analysis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize tickers
        self.sp500_tickers = self._get_sp500_tickers()
        if not self.sp500_tickers:
            raise ValueError("Failed to retrieve S&P 500 tickers")

    async def _make_api_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Dict[str, Any],
        endpoint_name: str
    ) -> Dict[str, Any]:
        """Make an API request with rate limiting and retries"""
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
        
        # Add API key to params
        params = {**params, 'apiKey': self.api_key}
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, params=params) as response:
                    response_text = await response.text()
                    
                    if response.status == 429:  # Rate limit exceeded
                        wait_time = float(response.headers.get('retry-after', 60))
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    if response.status == 403:
                        error_msg = f"Authorization failed for {endpoint_name}"
                        self.logger.error(error_msg)
                        raise PolygonAPIError(403, error_msg, endpoint_name)
                        
                    if response.status != 200:
                        error_msg = f"Request failed: {response_text}"
                        self.logger.error(error_msg)
                        raise PolygonAPIError(response.status, error_msg, endpoint_name)
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise PolygonAPIError(0, str(e), endpoint_name)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise PolygonAPIError(0, "Max retries exceeded", endpoint_name)

    def _get_sp500_tickers(self) -> List[str]:
        """Retrieve S&P 500 tickers from Wikipedia"""
        try:
            with requests.Session() as session:
                session.verify = certifi.where()
                response = session.get(
                    # 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                    'https://en.wikipedia.org/wiki/Nasdaq-100',
                    timeout=10
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            
            if not table:
                raise ValueError("S&P 500 constituents table not found")

            tickers = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if cells:
                    # ticker = cells[0].text.strip().replace('.', '-').upper()  #cell 0 for sp500
                    ticker = cells[1].text.strip().replace('.', '-').upper()    #cell 1 for nasdaq100
                    tickers.append(ticker)

            self.logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")

            print("Tickers:", tickers)
            return tickers

        except Exception as e:
            self.logger.error(f"Failed to fetch S&P 500 tickers: {str(e)}")
            raise

    async def _create_session(self) -> aiohttp.ClientSession:
        """Create an aiohttp session with proper configuration"""
        return aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context),
            timeout=self.session_timeout,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    async def get_top_option_volume_sp500(
        self,
        days_back: int = 90,
        top_n: int = 50,
        min_price: float = 10.0
    ) -> List[OptionVolumeResult]:
        """Get top S&P 500 stocks by option volume"""
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        async with await self._create_session() as session:
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
            
            async def process_ticker(ticker):
                async with semaphore:
                    return await self._analyze_ticker_option_volume(
                        session, ticker, start_date, end_date, min_price
                    )
            
            tasks = [process_ticker(ticker) for ticker in self.sp500_tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for ticker, result in zip(self.sp500_tickers, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing {ticker}: {result}")
                continue
            if result is not None:
                processed_results.append(result)
        
        # Sort and return top N results
        top_results = sorted(
            processed_results,
            key=lambda x: x.total_option_volume,
            reverse=True
        )[:top_n]

        return top_results

    async def _analyze_ticker_option_volume(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        min_price: float
    ) -> Optional[OptionVolumeResult]:
        """Analyze option volume for a specific ticker"""
        try:
            # Get current stock price
            stock_info = await self._get_current_stock_price(session, ticker)
            if not stock_info or stock_info < min_price:
                return None

            # Get option chain
            option_chain = await self._get_option_chain(
                session, ticker, end_date.strftime('%Y-%m-%d')
            )
            if not option_chain:
                return None

            # Filter for call options only
            call_options = [opt for opt in option_chain if opt.get('contract_type') == 'call']
            
            # Get today's date and find next Friday
            today = datetime.date.today()
            days_until_friday = (4 - today.weekday()) % 7  # 4 represents Friday
            next_friday = today + datetime.timedelta(days=days_until_friday)
            
            # Find the closest expiration date to next Friday
            sorted_expirations = sorted(
                call_options,
                key=lambda x: abs(datetime.datetime.strptime(x['expiration_date'], '%Y-%m-%d').date() - next_friday)
            )
            
            if not sorted_expirations:
                self.logger.warning(f"No valid options found for {ticker}")
                return None
                
            closest_expiration = sorted_expirations[0]['expiration_date']
            
            # Filter for options with the closest expiration
            closest_calls = [
                opt for opt in call_options 
                if opt['expiration_date'] == closest_expiration
            ]
            
            # Find ATM strike
            atm_strike = min(
                closest_calls,
                key=lambda x: abs(float(x['strike_price']) - stock_info)
            )['strike_price']
            
            # Find the last trading day (most recent business day)
            last_trading_day = today
            while last_trading_day.weekday() > 4:  # Skip weekends
                last_trading_day -= datetime.timedelta(days=1)
                
            self.logger.info(
                f"{ticker}: Found ATM strike {atm_strike} "
                f"for closest expiration date {closest_expiration}, "
                f"using last trading day {last_trading_day}"
            )

            # Get option trades
            trades = await self._get_atm_option_trades(
                session, ticker, atm_strike,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                closest_expiration,
                last_trading_day
            )

            # Calculate metrics
            total_volume = sum(trade.get('volume', 0) for trade in trades)
            avg_daily_volume = total_volume / max(1, (end_date - start_date).days)

            return OptionVolumeResult(
                ticker=ticker,
                current_price=stock_info,
                atm_strike=atm_strike,
                total_option_volume=total_volume,
                option_trades_count=len(trades),
                avg_daily_volume=avg_daily_volume
            )

        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
            return None

    async def _get_current_stock_price(
        self,
        session: aiohttp.ClientSession,
        ticker: str
    ) -> Optional[float]:
        """Get current stock price using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                return None
            return hist['Close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None

    async def _get_option_chain(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        date: str
    ) -> List[Dict]:
        """Get option chain data"""
        try:
            data = await self._make_api_request(
                session,
                "https://api.polygon.io/v3/reference/options/contracts",
                {
                    "underlying_ticker": ticker,
                    "as_of": date,
                    "limit": 250
                },
                "option_chain"
            )
            return data.get('results', [])
        except PolygonAPIError as e:
            self.logger.error(f"Error fetching option chain for {ticker}: {str(e)}")
            return []

    async def _get_atm_option_trades(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        strike_price: float,
        start_date: str,
        end_date: str,
        closest_expiration: str,
        last_trading_day: datetime.date
    ) -> List[Dict]:
        """
        Get ATM option trades using the open-close endpoint for options
        
        Args:
            session: Async HTTP session
            ticker: Stock ticker symbol
            strike_price: Strike price of the option
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            closest_expiration: Closest expiration date (YYYY-MM-DD format)
            last_trading_day: Last trading day to use for as_of parameter
            
        Returns:
            List of trade dictionaries
        """
        all_trades = []
        try:
            # Format the option symbol using the closest expiration date
            option_type = 'C'  # Using calls by default
            expiration_date_formatted = closest_expiration[2:].replace("-", "")  # Format: YYMMDD
            strike_price_formatted = f"{int(strike_price * 1000):08d}"  # Format: 00150000 for 150.00
            option_symbol = f"O:{ticker.upper()}{expiration_date_formatted}{option_type}{strike_price_formatted}"
            
            # Construct the URL for the open-close endpoint
            url = f"https://api.polygon.io/v1/open-close/{option_symbol}/{last_trading_day}"
            
            # Make the API request
            try:
                data = await self._make_api_request(
                    session,
                    url,
                    {"apiKey": self.api_key},
                    "option_open_close"
                )
                
                # Format the response into a trade-like structure
                if data and 'volume' in data:
                    trade = {
                        'symbol': option_symbol,
                        'volume': data.get('volume', 0),
                        'open': data.get('open', 0),
                        'close': data.get('close', 0),
                        'high': data.get('high', 0),
                        'low': data.get('low', 0)
                    }
                    all_trades.append(trade)
                    
                    self.logger.info(
                        f"Retrieved option data for {option_symbol}: "
                        f"Volume={trade['volume']}, "
                        f"Open={trade['open']}, "
                        f"Close={trade['close']}"
                    )
            
            except PolygonAPIError as e:
                if e.status == 404:
                    self.logger.warning(f"No data found for option {option_symbol}")
                else:
                    raise

            return all_trades

        except Exception as e:
            self.logger.error(
                f"Error fetching option trades for {ticker} at strike {strike_price}: {str(e)}"
            )
            return []

async def main():
    """Example usage"""
    try:
        analyzer = PolygonOptionVolumeAnalyzer()
        results = await analyzer.get_top_option_volume_sp500(
            days_back=300,
            top_n=30,
            min_price=20.0
        )
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {
                'Ticker': r.ticker,
                'Price': f"${r.current_price:.2f}",
                'ATM Strike': f"${r.atm_strike:.2f}",
                'Total Volume': f"{r.total_option_volume:,}",
                'Avg Daily Volume': f"{r.avg_daily_volume:.0f}",
                'Trade Count': r.option_trades_count
            }
            for r in results
        ])

        ticker_list = [r.ticker for r in results]
        formatted_tickers = "['" + "','".join(ticker_list) + "']"
        print("\nTickers in list format:")
        print(formatted_tickers)

        print("\nTop S&P 500 Stocks by Option Volume:")
        print(df.to_string(index=False))
        
    except Exception as e:
        logging.error(f"Execution error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())