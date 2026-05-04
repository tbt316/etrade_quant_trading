import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import RobustScaler
import asyncio
import logging
import threading
import time
from copy import deepcopy

class RollingRobustScaler:
    """
    A strictly causal, stateful scaler that maintains a rolling window of historical medians and IQRs.
    Mandate 8.4: Optimized with numpy-based vectorized operations to avoid slow pandas loops.
    """
    def __init__(self, window=252 * 5):
        self.window = window
        self.history = None # Will be initialized as np.array
        self.center_ = None
        self.scale_ = None

    def update_and_transform(self, x_row):
        """Update history with new row and return scaled value."""
        val = x_row.values if hasattr(x_row, 'values') else x_row
        val = np.array(val).reshape(1, -1)
        
        if self.history is None:
            self.history = val
        else:
            self.history = np.vstack([self.history, val])
            if len(self.history) > self.window:
                self.history = self.history[-self.window:]
            
        if len(self.history) < 20:
            return x_row * np.nan
        
        # Mandate 8.4: Vectorized median and percentile calculation
        self.center_ = np.median(self.history, axis=0)
        q1 = np.percentile(self.history, 25, axis=0)
        q3 = np.percentile(self.history, 75, axis=0)
        self.scale_ = q3 - q1
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        
        return (x_row - self.center_) / self.scale_

    def transform(self, df):
        """Standard transform using CURRENTLY learned parameters (no update)."""
        if self.center_ is None:
            return df * np.nan
        return (df - self.center_) / self.scale_

    def batch_rolling_transform(self, df):
        """
        Mandate 8.4: Batch version using numpy stride_tricks for maximum performance.
        Avoids iterative loops by calculating all rolling windows at once.
        """
        data = df.values
        n_samples, n_features = data.shape
        if n_samples < 20:
            return df * np.nan
            
        # Use a smaller window for batch if the full window is too large for memory
        # but here we use self.window
        w = min(self.window, n_samples)
        
        # Create rolling windows using stride_tricks
        # Shape: (n_samples - w + 1, w, n_features)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(data, (w, n_features)).squeeze()
        # windows shape might be (n_samples-w+1, w) if n_features=1, or (n_samples-w+1, w, n_features)
        
        if n_features == 1:
            windows = windows[:, :, np.newaxis]
            
        # Calculate rolling medians and IQRs
        centers = np.median(windows, axis=1)
        q1 = np.percentile(windows, 25, axis=1)
        q3 = np.percentile(windows, 75, axis=1)
        scales = q3 - q1
        scales = np.where(scales == 0, 1.0, scales)
        
        # The result at index t corresponds to the window ending at t
        # So centers[0] is for t = w-1
        result = np.full_like(data, np.nan)
        result[w-1:] = (data[w-1:] - centers) / scales
        
        return pd.DataFrame(result, index=df.index, columns=df.columns)

# Adjust path for project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache
from backtesting.greeks_calculator import bs_gamma, implied_volatility

logger = logging.getLogger(__name__)

def red_alert(msg):
    """Print a prominent red alert to the console."""
    print(f"\033[91m\033[1m🚨 [RED ALERT] {msg}\033[0m")
    logger.error(f"RED ALERT: {msg}")

class DataCacheManager:
    def __init__(self, cache_dir="s_and_p_data/api_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def get_cache_path(self, symbol):
        return os.path.join(self.cache_dir, f"{symbol.replace('^', 'INDEX_')}.parquet")

    def load(self, symbol):
        path = self.get_cache_path(symbol)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:
                logger.error(f"Error reading cache for {symbol}: {e}")
        return pd.DataFrame()

    def save(self, symbol, df):
        if df.empty: return
        path = self.get_cache_path(symbol)
        with self._lock:
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                # Combine and drop duplicates (keep newest)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep='last')].sort_index()
            df.to_parquet(path)

    def check_coverage(self, symbol, start_date, end_date):
        """Check if cache covers the requested range."""
        df = self.load(symbol)
        if df.empty: return False
        
        cache_start = df.index.min()
        cache_end = df.index.max()
        
        req_start = pd.to_datetime(start_date).tz_localize(None).normalize()
        req_end = pd.to_datetime(end_date).tz_localize(None).normalize()
        
        # 1. Start Date Lenience:
        # If the symbol (like BTC) only started trading after req_start, 
        # we consider it covered if cache_start is reasonably close to its inception.
        start_covered = (cache_start <= req_start)
        if not start_covered:
            # Lenience if we have at least 5 years of data
            if (cache_end - cache_start).days > 365 * 5:
                start_covered = True

        # 2. End Date Lenience:
        # Macro data (FRED) is often monthly and delayed. 
        # We use a 90-day buffer for macro symbols to prevent infinite polling.
        is_macro = symbol.isupper() and (len(symbol) > 5 or any(c in symbol for c in ['-', '=', '_']))
        # Specifically for FEDFUNDS or other FRED-like tickers
        if "FED" in symbol or "TREASURY" in symbol:
            is_macro = True
            
        buffer_days = 90 if is_macro else 2
        end_covered = (cache_end >= req_end - timedelta(days=buffer_days))
        
        return start_covered and end_covered

class DataIngestor:
    def __init__(self, cache_path=None):
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_cache", "option_data.db")
        self.cache = OptionDataCache(cache_path)
        self.rolling_scaler = RollingRobustScaler()
        self._fred_cache = None # Cache for the session
        self.cache_manager = DataCacheManager()
        self._sync_active = set()

    def start_background_sync(self, symbols, source='yf', start_date=None, end_date=None):
        """Start a background thread to sync symbols if not already syncing."""
        if not start_date: start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        if not end_date: end_date = datetime.now().strftime("%Y-%m-%d")
        
        sync_key = tuple(sorted(symbols))
        if sync_key in self._sync_active: return
        
        def sync_task():
            self._sync_active.add(sync_key)
            print(f"🔄 [DataIngestor] Background sync started for {len(symbols)} symbols from {source}...")
            while True:
                try:
                    all_covered = True
                    for sym in symbols:
                        if not self.cache_manager.check_coverage(sym, start_date, end_date):
                            all_covered = False
                            print(f"📡 [DataIngestor] Polling API for {sym}...")
                            if source == 'yf':
                                # Small fetch for just the gap might be better but for now we fetch the range
                                df = self._fetch_yf_raw(sym, start_date, end_date)
                                if not df.empty: self.cache_manager.save(sym, df)
                            else:
                                df = self._fetch_fred_raw([sym], start_date, end_date)
                                if not df.empty: self.cache_manager.save(sym, df)
                    
                    if all_covered:
                        print(f"✅ [DataIngestor] Background sync complete for {source} symbols.")
                        break
                    
                    # Wait 5 mins before retrying failures
                    time.sleep(300)
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
                    time.sleep(60)
            self._sync_active.remove(sync_key)

        t = threading.Thread(target=sync_task, daemon=True)
        t.start()

    def _fetch_fred_raw(self, symbols, start_date, end_date):
        try:
            df = web.DataReader(symbols, 'fred', start_date, end_date)
            return self._normalize_index(df)
        except Exception as e:
            logger.error(f"FRED fetch error: {e}")
            return pd.DataFrame()

    def _fetch_yf_raw(self, symbol, start_date, end_date):
        try:
            # 🚨 CRITICAL: Use auto_adjust=False to get raw Close price (NOT dividend-adjusted).
            # Historical option strikes (Massive API/Polygon) are UNADJUSTED. 
            # Using Adjusted Close (auto_adjust=True) will cause spot/strike misalignment (e.g. SPY 2015).
            # See backtesting/strategy_registry.md -> Data Integrity for details.
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                # Robust extraction of 'Close' and flattening to a simple 1-column DataFrame
                if isinstance(df.columns, pd.MultiIndex):
                    if 'Close' in df.columns.get_level_values(0):
                        df = df['Close']
                        if isinstance(df, pd.DataFrame):
                            df = df.iloc[:, 0] # Extract first ticker if multiple (should be 1)
                    else:
                        df = df.iloc[:, 0]
                elif 'Close' in df.columns:
                    df = df['Close']
                else:
                    df = df.iloc[:, 0]
                
                # Force to a simple 1-column DataFrame named 'Close'
                if isinstance(df, pd.Series):
                    df = df.to_frame('Close')
                elif isinstance(df, pd.DataFrame):
                    df = df.iloc[:, [0]]
                    df.columns = ['Close']
                    
                return self._normalize_index(df)
        except Exception as e:
            logger.error(f"YF fetch error for {symbol}: {e}")
        return pd.DataFrame()

    def _normalize_index(self, df):
        """Standardize index to tz-naive, normalized DatetimeIndex."""
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df

    def fetch_fred_data(self, start_date, end_date):
        """Fetch macroeconomic data from FRED with local caching."""
        symbols = {
            'DGS10': '10Y_Treasury',
            'DTB3': '3M_TBill',
            'BAA10Y': 'High_Yield_Spread',
            'FEDFUNDS': 'Fed_Funds_Rate',
            'USEPUINDXD': 'EPU_Index'
        }
        
        combined_df = pd.DataFrame()
        missing = []
        for sym in symbols.keys():
            cached = self.cache_manager.load(sym)
            if cached.empty or not self.cache_manager.check_coverage(sym, start_date, end_date):
                missing.append(sym)
            if not cached.empty:
                if combined_df.empty: combined_df = cached
                else: combined_df = pd.merge(combined_df, cached, left_index=True, right_index=True, how='outer')
        
        if missing:
            red_alert(f"FRED data missing for {missing}. Starting background sync.")
            self.start_background_sync(missing, source='fred', start_date=start_date, end_date=end_date)
        
        if combined_df.empty: return pd.DataFrame()
        
        # Select and rename
        cols_to_use = [c for c in symbols.keys() if c in combined_df.columns]
        df = combined_df[cols_to_use].copy()
        df.rename(columns=symbols, inplace=True)
        if '10Y_Treasury' in df.columns and '3M_TBill' in df.columns:
            df['Yield_Curve_10Y_3M'] = df['10Y_Treasury'] - df['3M_TBill']
        
        # Filter to requested range
        req_start = pd.to_datetime(start_date).tz_localize(None).normalize()
        req_end = pd.to_datetime(end_date).tz_localize(None).normalize()
        df = df[(df.index >= req_start) & (df.index <= req_end)]
        
        return self._normalize_index(df)

    def fetch_yf_data(self, start_date, end_date):
        """Fetch price data from Yahoo Finance with local caching."""
        symbols = {
            'SPY': 'SPY_Close',
            '^VIX': 'VIX_Close',
            '^VVIX': 'VVIX_Close',
            '^VIX3M': 'VXV_Close',
            'BTC-USD': 'BTC_Close',
            'CL=F': 'WTI_Oil'
        }
        
        combined_df = pd.DataFrame()
        missing = []
        for sym, name in symbols.items():
            cached = self.cache_manager.load(sym)
            if cached.empty or not self.cache_manager.check_coverage(sym, start_date, end_date):
                missing.append(sym)
            if not cached.empty:
                cached.columns = [name]
                if combined_df.empty: combined_df = cached
                else: combined_df = pd.merge(combined_df, cached, left_index=True, right_index=True, how='outer')

        if missing:
            red_alert(f"YFinance data missing for {missing}. Starting background sync.")
            self.start_background_sync(missing, source='yf', start_date=start_date, end_date=end_date)

        if combined_df.empty: return pd.DataFrame()

        # Feature engineering (on what we have)
        df = combined_df
        if 'SPY_Close' in df.columns:
            # Drop NaNs for calculation to avoid window pollution on weekends
            spy_series = df['SPY_Close'].dropna()
            spy_returns = np.log(spy_series / spy_series.shift(1))
            # Calculate 21-day Realized Volatility (Annualized) on trading days
            spy_rv = spy_returns.rolling(window=21).std() * np.sqrt(252)
            
            df['SPY_Log_Return'] = spy_returns
            df['SPY_Realized_Vol_21d'] = spy_rv
        if 'BTC_Close' in df.columns:
            df['BTC_Log_Return'] = np.log(df['BTC_Close'] / df['BTC_Close'].shift(1))
        if 'VIX_Close' in df.columns and 'VXV_Close' in df.columns:
            df['VXV_VIX_Ratio'] = df['VXV_Close'] / df['VIX_Close']

        # Filter to requested range
        req_start = pd.to_datetime(start_date).tz_localize(None).normalize()
        req_end = pd.to_datetime(end_date).tz_localize(None).normalize()
        df = df[(df.index >= req_start) & (df.index <= req_end)]
        
        return self._normalize_index(df)

    async def calculate_gex_metrics(self, underlying, spot_price):
        """Calculate Gamma Exposure (GEX) and Zero Gamma from Massive API snapshot."""
        async with MassiveAPIClient(cache=self.cache) as client:
            snapshot = await client.fetch_option_snapshot(underlying)
            if not snapshot:
                return {"net_gex": 0.0, "zero_gamma": 0.0}
            
            total_gex = 0.0
            strike_gex = {}
            
            for opt in snapshot:
                greeks = opt.get('greeks', {})
                gamma = greeks.get('gamma')
                oi = opt.get('open_interest', 0)
                
                if gamma is not None and oi > 0:
                    details = opt.get('details', {})
                    ticker = details.get('ticker', '')
                    strike = details.get('strike_price', 0)
                    is_call = details.get('contract_type') == 'call'
                    
                    # Standard OMM GEX assumption: OMM Long Calls, Short Puts
                    # Long call = +Gamma, Short put = -Gamma
                    gex_value = gamma * oi * 100 * spot_price
                    if is_call:
                        total_gex += gex_value
                    else:
                        total_gex -= gex_value
                    
                    strike_gex[strike] = strike_gex.get(strike, 0) + (gex_value if is_call else -gex_value)
            
            # Find Zero Gamma (strike where GEX flips from negative to positive)
            sorted_strikes = sorted(strike_gex.keys())
            zero_gamma = 0.0
            for i in range(len(sorted_strikes) - 1):
                s1, s2 = sorted_strikes[i], sorted_strikes[i+1]
                if strike_gex[s1] < 0 and strike_gex[s2] > 0:
                    # Simple interpolation
                    zero_gamma = s1 + (s2 - s1) * abs(strike_gex[s1]) / (abs(strike_gex[s1]) + abs(strike_gex[s2]))
                    break
            
            return {
                "net_gex": total_gex / 1e9, # In billions
                "zero_gamma": zero_gamma
            }

    def fractional_diff(self, series, d, threshold=1e-4):
        """Compute fractional differentiation to preserve memory."""
        weights = [1.0]
        k = 1
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold or k > 126:
                break
            weights.append(w)
            k += 1
            
        weights = np.array(weights)[::-1]
        
        diff_series = pd.Series(index=series.index, dtype=float)
        for i in range(len(weights) - 1, len(series)):
            window = series.iloc[i - len(weights) + 1 : i + 1]
            diff_series.iloc[i] = np.dot(weights, window)
            
        return diff_series

    def ensure_stationarity(self, df):
        """Apply ADF test and fractional differencing to ensure all features are stationary."""
        stationary_df = df.copy()
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            
            # FIX: VIX is naturally mean-reverting and structurally stationary. 
            # Differencing it (even d=0.2) pulls the mean toward zero and breaks labeling logic.
            if any(kw in col for kw in ['VIX', 'VVIX', 'Ratio', 'Log_Return']):
                logger.info(f"Feature {col} is inherently stationary/bounded. Skipping differencing.")
                continue

            # REMOVED Hardcoded d=0.2 for levels: The required d is dynamic (e.g. higher in rate-hike cycles).
            # We now run a dynamic ADF search for all features to find the minimum d (floor 0.15) 
            # that achieves stationarity while maximizing memory preservation.

            if series.var() < 1e-12:
                logger.info(f"Feature {col} has near-zero variance. Differencing to maintain stationarity.")
                stationary_df[col] = series.diff()
                continue

            try:
                # Check for stationarity on the raw series
                res = adfuller(series, autolag='AIC')
                p_value = res[1]
                threshold = 0.05

                if p_value > threshold:
                    logger.info(f"Feature {col} is non-stationary (p={p_value:.4f}). Searching for optimal d...")
                    
                    best_d = None
                    # Search from d=0.15 (memory preservation) to d=1.0 (first-order diff)
                    for d in np.arange(0.15, 1.05, 0.05):
                        diffed = self.fractional_diff(series, d)
                        diffed_clean = diffed.dropna()
                        if not diffed_clean.empty and diffed_clean.var() > 1e-12:
                            try:
                                if adfuller(diffed_clean)[1] < threshold:
                                    best_d = d
                                    break
                            except Exception:
                                pass
                    
                    if best_d is None:
                        logger.info(f"Fractional diff failed to achieve stationarity for {col}. Falling back to first-order diff (d=1.0)")
                        stationary_df[col] = series.diff()
                    else:
                        logger.info(f"Optimal fractional d for {col} = {best_d:.2f}")
                        stationary_df[col] = self.fractional_diff(series, best_d)
                else:
                    # Feature is already stationary
                    stationary_df[col] = series
            except Exception as e:
                if "is constant" in str(e):
                    logger.info(f"Feature {col} is constant. Differencing.")
                    stationary_df[col] = series.diff()
                else:
                    logger.warning(f"ADF test failed for {col}: {e}")
        
        return stationary_df.dropna()

    def fit_scaler(self, df, window=252*5):
        """
        Regulation 2.2: Fitting on a global block is only allowed if the block 
        is explicitly historical training data and NOT being used for causal inference.
        """
        temp_scaler = RobustScaler()
        temp_scaler.fit(df)
        # Seed the rolling scaler history
        self.rolling_scaler.window = window
        self.rolling_scaler.history = df.values.tolist()[-window:]
        self.rolling_scaler.center_ = temp_scaler.center_
        self.rolling_scaler.scale_ = temp_scaler.scale_

    def transform_features(self, df):
        """Apply the already-fitted rolling scaler to the data."""
        if self.rolling_scaler.center_ is None:
             raise ValueError("RollingRobustScaler must be fitted or warmed up before calling transform_features.")
        return self.rolling_scaler.transform(df)

    def scale_features(self, df, rolling=False, window=252*5):
        """
        Scale features using strictly causal logic.
        """
        if rolling:
            return self.rolling_scale_features(df, window=window)
        
        if self.rolling_scaler.center_ is None:
            red_alert("RollingRobustScaler not fitted. This will cause NaNs in inference.")
            return pd.DataFrame(index=df.index, columns=df.columns)
            
        return self.transform_features(df)

    def rolling_scale_features(self, df, window=252*5):
        """
        Regulation 8.2 & 8.4: Optimized strictly causal rolling-window scaling.
        """
        scaler = RollingRobustScaler(window=window)
        # Use optimized batch transform if possible
        scaled_df = scaler.batch_rolling_transform(df)
        
        # Update self.rolling_scaler with the final state
        self.rolling_scaler = scaler
        # We need to manually populate history for future updates
        self.rolling_scaler.history = df.values[-window:]
        
        return scaled_df

    def verify_data(self, df):
        """🛑 Verification Checkpoint 1: Data Integrity."""
        if df.empty:
            raise ValueError("Final dataset is empty!")
        
        # NaN check
        nan_pct = df.isna().mean().max()
        if nan_pct > 0.05: # Allow slightly more for macro data gaps
            raise ValueError(f"Data integrity failed: Max NaN ratio is {nan_pct:.2%}, exceeds 5% limit.")
        
        # Date alignment check
        if len(df) < 20:
            raise ValueError(f"Insufficient data points: {len(df)}")
            
        logger.info("Verification Checkpoint 1 passed.")
        return True

    async def build_fused_dataset(self, start_date, end_date, underlying="SPY", scale=True, rolling=False):
        """Main pipeline to build the high-dimensional feature set."""
        fred_df = self.fetch_fred_data(start_date, end_date)
        yf_df = self.fetch_yf_data(start_date, end_date)
        
        # Merge basic features
        if fred_df.empty:
            logger.warning("FRED data is missing. Proceeding with YFinance data only.")
            combined = yf_df
        else:
            combined = pd.concat([yf_df, fred_df], axis=1).ffill()
        
        # Drop columns that completely failed to download (e.g. due to yfinance rate limits)
        # before we run dropna(), otherwise an all-NaN column wipes out all rows!
        missing_ratios = combined.isna().mean()
        bad_cols = missing_ratios[missing_ratios > 0.8].index # Be more lenient with macro data
        if len(bad_cols) > 0:
            logger.warning(f"Dropping features with >80% missing data: {list(bad_cols)}")
            combined = combined.drop(columns=bad_cols)
            
        # Instead of aggressive dropna, we should drop rows only if core features are missing
        core_cols = ['SPY_Close', 'VIX_Close']
        existing_core = [c for c in core_cols if c in combined.columns]
        if existing_core:
            combined = combined.dropna(subset=existing_core)
        else:
            combined = combined.dropna()
        
        if combined.empty:
            raise ValueError("Dataset is empty after dropping NaNs! Check your internet connection or API rate limits.")
        
        # Mandate 8.3: Earnings-Neutral Volatility Inputs
        # Standardize spikes in VIX/Realized Vol that are likely earnings-driven.
        vol_cols = [c for c in combined.columns if any(kw in c for kw in ['VIX', 'Vol', 'VVIX'])]
        for col in vol_cols:
            # 5-day rolling median filter suppresses short-lived spikes (earnings jumps)
            combined[col] = combined[col].rolling(window=5, center=True, min_periods=1).median().ffill().bfill()
            logger.info(f"Applied earnings-neutral median filter to {col}")

        # Stationarity
        stationary = self.ensure_stationarity(combined)
        
        # Verification
        self.verify_data(stationary)
        
        # Scaling
        if scale:
            scaled = self.scale_features(stationary, rolling=rolling)
            return scaled
            
        return stationary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = DataIngestor()
    # Fetch 2 years to ensure ADF has enough data
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    async def run():
        df = await ingestor.build_fused_dataset(start, end)
        print("\n--- Fused Feature Set (First 5 rows) ---")
        print(df.head())
        print(f"\nFinal Features: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        
        # Test GEX
        # spot = yf.Ticker("SPY").fast_info['lastPrice']
        # gex = await ingestor.calculate_gex_metrics("SPY", spot)
        # print(f"\nLive GEX Metrics for SPY: {gex}")

    asyncio.run(run())
