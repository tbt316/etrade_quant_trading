import os
import numpy as np
import pandas as pd
import yfinance as yf
# import pandas_datareader.data as web  # Disabled due to pandas deprecate_kwarg compatibility issue
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

    def inverse_transform(self, df):
        """Mandate 10.7: Inverse transform using currently learned parameters."""
        if self.center_ is None:
            return df * np.nan
        return (df * self.scale_) + self.center_

    def batch_rolling_transform(self, df, include_current=True):
        """
        Mandate 8.4: Batch version using numpy stride_tricks for maximum performance.
        Avoids iterative loops by calculating all rolling windows at once.

        include_current=True is appropriate for a close_T signal consumed after
        the close. Set include_current=False when the transformed value will be
        used for an intraday or same-session decision before row T is observable.
        """
        data = df.astype(float).values
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
        
        result = np.full(data.shape, np.nan, dtype=float)
        if include_current:
            # The result at index t corresponds to the window ending at t.
            # This is causal for close_T reporting, but not same-session entry.
            result[w-1:] = (data[w-1:] - centers) / scales
        else:
            # The result at index t uses the window ending at t-1.
            # centers[0] covers rows 0..w-1 and transforms row w.
            result[w:] = (data[w:] - centers[:-1]) / scales[:-1]
        
        return pd.DataFrame(result, index=df.index, columns=df.columns)

# Adjust path for project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.massive_api_client import MassiveAPIClient
from backtesting.option_data_cache import OptionDataCache
from backtesting.greeks_calculator import bs_call_delta, bs_gamma, bs_put_delta, implied_volatility
from backtesting.sharded_option_data_cache import _root_aliases

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def red_alert(msg):
    """Print a prominent red alert to the console."""
    print(f"\033[91m\033[1m🚨 [RED ALERT] {msg}\033[0m")
    logger.error(f"RED ALERT: {msg}")

class DataCacheManager:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(PROJECT_ROOT, "s_and_p_data", "api_cache")
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
                    all_covered = self.sync_data(symbols, source=source, start_date=start_date, end_date=end_date)
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

    def sync_data(self, symbols, source='yf', start_date=None, end_date=None):
        """
        Synchronously sync symbols. Returns True if all symbols are now covered.
        """
        if not start_date: start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        if not end_date: end_date = datetime.now().strftime("%Y-%m-%d")
        
        all_covered = True
        for sym in symbols:
            cached = self.cache_manager.load(sym)
            has_ohlc = not cached.empty and 'Open' in cached.columns
            covered = self.cache_manager.check_coverage(sym, start_date, end_date)
            if source == 'yf':
                covered = covered and has_ohlc

            if not covered:
                all_covered = False
                print(f"📡 [DataIngestor] Syncing {sym} from {source}...")
                if source == 'yf':
                    df = self._fetch_yf_raw(sym, start_date, end_date)
                    if not df.empty: self.cache_manager.save(sym, df)
                else:
                    df = self._fetch_fred_raw([sym], start_date, end_date)
                    if not df.empty: self.cache_manager.save(sym, df)
        return all_covered

    def _fetch_fred_raw(self, symbols, start_date, end_date):
        try:
            combined = pd.DataFrame()
            for sym in symbols:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sym}"
                df = pd.read_csv(url, index_col='DATE', parse_dates=True)
                df.index = pd.to_datetime(df.index)
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                df[sym] = pd.to_numeric(df[sym], errors='coerce')
                if combined.empty:
                    combined = df
                else:
                    combined = combined.merge(df, left_index=True, right_index=True, how='outer')
            return self._normalize_index(combined)
        except Exception as e:
            logger.error(f"FRED fetch error: {e}")
            return pd.DataFrame()

    def _fetch_yf_raw(self, symbol, start_date, end_date):
        try:
            # 🚨 CRITICAL: Use auto_adjust=False to get raw prices.
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                # Standardize columns to [Open, High, Low, Close, Volume]
                if isinstance(df.columns, pd.MultiIndex):
                    # Multi-ticker or multi-level index: extract first ticker
                    tickers = df.columns.get_level_values(1).unique() if df.columns.nlevels > 1 else [None]
                    df = df.xs(tickers[0], axis=1, level=1) if tickers[0] else df
                
                # Filter to only the core OHLCV columns we need
                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                df = df[cols]
                
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

    def fetch_fred_data(self, start_date, end_date, wait=False):
        """Fetch macroeconomic data from FRED with local caching.
        
        NOTE: Fed_Funds_Rate (FEDFUNDS) and 3M_TBill (DTB3) deliberately
        removed — slow macro rate trends were hijacking PC1 and diluting
        leading-indicator signal. Yield_Curve_10Y_3M also removed as it
        depended on 3M_TBill.
        """
        symbols = {
            'DGS10': '10Y_Treasury',
            'BAA10Y': 'High_Yield_Spread',
            'USEPUINDXD': 'EPU_Index'
        }
        
        if wait:
            self.sync_data(list(symbols.keys()), source='fred', start_date=start_date, end_date=end_date)

        combined_df = pd.DataFrame()
        missing = []
        for sym in symbols.keys():
            cached = self.cache_manager.load(sym)
            if cached.empty or not self.cache_manager.check_coverage(sym, start_date, end_date):
                missing.append(sym)
            if not cached.empty:
                if combined_df.empty: combined_df = cached
                else: combined_df = pd.merge(combined_df, cached, left_index=True, right_index=True, how='outer')
        
        if missing and not wait:
            red_alert(f"FRED data missing for {missing}. Starting background sync.")
            self.start_background_sync(missing, source='fred', start_date=start_date, end_date=end_date)
        
        if combined_df.empty: return pd.DataFrame()
        
        # Select and rename
        cols_to_use = [c for c in symbols.keys() if c in combined_df.columns]
        df = combined_df[cols_to_use].copy()
        df.rename(columns=symbols, inplace=True)
        
        # Filter to requested range
        req_start = pd.to_datetime(start_date).tz_localize(None).normalize()
        req_end = pd.to_datetime(end_date).tz_localize(None).normalize()
        df = df[(df.index >= req_start) & (df.index <= req_end)]
        
        return self._normalize_index(df)

    def calculate_yang_zhang_volatility(self, ohlc_df, window=21):
        """
        Implementation of Yang-Zhang Volatility Estimator.
        It is roughly 14x more efficient than Close-to-Close estimators.
        """
        if len(ohlc_df) < window + 1:
            return pd.Series(index=ohlc_df.index, dtype=float)
            
        log_ho = np.log(ohlc_df['High'] / ohlc_df['Open'])
        log_lo = np.log(ohlc_df['Low'] / ohlc_df['Open'])
        log_co = np.log(ohlc_df['Close'] / ohlc_df['Open'])
        
        log_oc = np.log(ohlc_df['Open'] / ohlc_df['Close'].shift(1))
        log_oc_sq = log_oc**2
        
        log_cc = np.log(ohlc_df['Close'] / ohlc_df['Close'].shift(1))
        log_cc_sq = log_cc**2
        
        # Rogers-Satchell component
        rs_comp = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs_var = rs_comp.rolling(window=window).mean()
        
        # Overnight and Open-to-Close components
        overnight_var = log_oc_sq.rolling(window=window).var()
        open_to_close_var = log_co.rolling(window=window).var()
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        yz_var = overnight_var + k * open_to_close_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * 252)

    def fetch_yf_data(self, start_date, end_date, wait=False):
        """Fetch price data from Yahoo Finance with local caching."""
        symbols = {
            'SPY': 'SPY',
            '^VIX': 'VIX',
            '^VVIX': 'VVIX',
            '^VIX3M': 'VXV',
            'BTC-USD': 'BTC',
            'CL=F': 'WTI'
        }
        
        if wait:
            self.sync_data(list(symbols.keys()), source='yf', start_date=start_date, end_date=end_date)

        def load_yf_symbol(sym):
            cached = self.cache_manager.load(sym)
            if cached.empty or 'Open' not in cached.columns:
                return pd.DataFrame()
            return cached

        ohlc_map = {}
        missing = []
        for sym, name in symbols.items():
            cached = load_yf_symbol(sym)
            if cached.empty or not self.cache_manager.check_coverage(sym, start_date, end_date):
                missing.append(sym)
            if not cached.empty:
                ohlc_map[name] = cached

        if missing and not wait:
            red_alert(f"YFinance OHLC data missing or outdated for {missing}. Syncing...")
            self.start_background_sync(missing, source='yf', start_date=start_date, end_date=end_date)
        elif wait and missing:
            # sync_data() already attempted the refresh; reload the local cache now
            ohlc_map = {}
            for sym, name in symbols.items():
                cached = load_yf_symbol(sym)
                if not cached.empty:
                    ohlc_map[name] = cached

        if not ohlc_map: return pd.DataFrame()

        # Build combined Close-only DF for most features
        close_df = pd.DataFrame()
        for name, df in ohlc_map.items():
            if 'Close' in df.columns:
                series = df['Close'].rename(f"{name}_Close")
                if close_df.empty: close_df = series.to_frame()
                else: close_df = pd.merge(close_df, series, left_index=True, right_index=True, how='outer')

        # Feature engineering
        df = close_df
        if 'SPY_Close' in df.columns:
            spy_series = df['SPY_Close'].dropna()
            spy_returns = np.log(spy_series / spy_series.shift(1))
            df['SPY_Log_Return'] = spy_returns
            
            # NOTE: SPY_Realized_Vol_21d deliberately removed — backward-looking
            # historical variance lags the model. Yang-Zhang is the sole vol estimator.
            
            # MANDATE: Add Yang-Zhang Volatility for the primary asset
            if 'SPY' in ohlc_map:
                df['SPY_Yang_Zhang_21d'] = self.calculate_yang_zhang_volatility(ohlc_map['SPY'])
                logger.info("Successfully integrated Yang-Zhang Volatility for SPY.")

        if 'BTC_Close' in df.columns:
            df['BTC_Log_Return'] = np.log(df['BTC_Close'] / df['BTC_Close'].shift(1))
        if 'VIX_Close' in df.columns and 'VXV_Close' in df.columns:
            df['VXV_VIX_Ratio'] = df['VXV_Close'] / df['VIX_Close']
        
        # Keep VIX_Close available in the raw feature frame for overlay and audit
        # paths. The HMM training pipeline drops it before scaler/PCA fitting.

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
            return {
                "net_gex": total_gex / 1e9, # In billions
                "zero_gamma": zero_gamma
            }

    def fetch_historical_gex(self, dates, underlying="SPX"):
        """
        Calculates Net GEX for a list of string dates using the sharded options database.
        Highly optimized using vectorized pandas grouping instead of hot-loop queries.
        """
        gex_series = pd.Series(index=pd.to_datetime(dates), dtype=float)
        cache = OptionDataCache()
        missing_years = []
        
        underlying_cache = os.path.join(PROJECT_ROOT, "s_and_p_data", f"underlying_{underlying}.csv")
        if not os.path.exists(underlying_cache):
            underlying_cache = os.path.join(PROJECT_ROOT, "s_and_p_data", f"underlying_SPY.csv")
            
        if os.path.exists(underlying_cache):
            try:
                spot_df = pd.read_csv(underlying_cache, index_col=0)
                spot_df.index = pd.to_datetime(spot_df.index).tz_localize(None).normalize()
            except Exception:
                spot_df = pd.DataFrame()
        else:
            spot_df = pd.DataFrame()
            
        # Group dates by year to query only once per shard
        dates_df = pd.DataFrame(index=pd.to_datetime(dates))
        dates_df['year'] = dates_df.index.strftime("%Y")
        
        for year, group in dates_df.groupby('year'):
            try:
                # 1. Get connection to options DB
                if cache.use_shards:
                    conn = cache._ensure_price_conn(underlying, year)
                else:
                    conn = cache.conn

                year_start = pd.Timestamp(f"{year}-01-01")
                year_end = pd.Timestamp(f"{int(year) + 1}-01-01")

                # Fetch only the option rows needed for this year
                query = """
                    SELECT pricing_date, contract_type, gamma, open_interest
                    FROM option_prices
                    WHERE underlying = ? AND pricing_date >= ? AND pricing_date < ?
                """
                df_year = pd.read_sql(
                    query,
                    conn,
                    params=(underlying, year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")),
                )
                if df_year.empty:
                    continue
                
                df_year['pricing_date'] = pd.to_datetime(df_year['pricing_date'])

                usable = df_year['gamma'].fillna(0).gt(0) & df_year['open_interest'].fillna(0).gt(0)
                if not usable.any():
                    missing_years.append(year)
                    continue
                df_year = df_year.loc[usable].copy()
                
                # Map spot prices
                if not spot_df.empty:
                    df_year = df_year.merge(spot_df.iloc[:, 0].rename('spot'), left_on='pricing_date', right_index=True, how='left')
                else:
                    df_year['spot'] = 4000.0
                df_year['spot'] = df_year['spot'].fillna(4000.0)
                
                # Vectorized GEX calculation
                df_year['sign'] = np.where(df_year['contract_type'].str.lower() == 'call', 1.0, -1.0)
                df_year['gex'] = df_year['sign'] * df_year['open_interest'] * 100 * df_year['gamma'] * (df_year['spot'] ** 2) * 0.01
                
                # Aggregate by date
                daily_gex = df_year.groupby('pricing_date')['gex'].sum() / 1e9 # Billions
                
                # Assign back to series
                gex_series.loc[daily_gex.index] = daily_gex.values
            except Exception as e:
                logger.warning(f"Error in vectorized GEX for {year}: {e}")

        if missing_years:
            logger.warning(
                f"No usable GEX rows found for {underlying} in years {', '.join(missing_years)}; "
                "leaving feature missing for those years."
            )

        if gex_series.notna().sum() < max(10, int(len(gex_series) * 0.1)):
            logger.warning(f"Historical GEX coverage too sparse for {underlying}; dropping feature.")
            return np.full(len(gex_series), np.nan, dtype=float)

        return gex_series.values

    def fetch_historical_skew(self, dates, underlying="SPX"):
        """
        Calculates 25-Delta Option Implied Skew (IV_Put - IV_Call) closest to 30 DTE.
        Uses cached option quote history and derives IV from the stored mid price
        when explicit implied_vol / delta fields are missing in the shard rows.
        """
        skew_series = pd.Series(index=pd.to_datetime(dates), dtype=float)
        cache = OptionDataCache()
        aliases = _root_aliases(underlying) or [underlying]

        spot_path = os.path.join(PROJECT_ROOT, "s_and_p_data", f"underlying_{underlying}.csv")
        if not os.path.exists(spot_path) and underlying != "SPY":
            spot_path = os.path.join(PROJECT_ROOT, "s_and_p_data", "underlying_SPY.csv")
        try:
            spot_df = pd.read_csv(spot_path, index_col=0)
            spot_df.index = pd.to_datetime(spot_df.index).tz_localize(None).normalize()
            spot_series = spot_df.iloc[:, 0].astype(float)
        except Exception:
            spot_series = pd.Series(dtype=float)

        rate_path = os.path.join(PROJECT_ROOT, "s_and_p_data", "underlying_^IRX.csv")
        try:
            rate_df = pd.read_csv(rate_path, index_col=0)
            rate_df.index = pd.to_datetime(rate_df.index).tz_localize(None).normalize()
            rate_series = (rate_df.iloc[:, 0].astype(float) / 100.0).reindex(pd.to_datetime(dates)).ffill().bfill()
        except Exception:
            rate_series = pd.Series(index=pd.to_datetime(dates), data=0.0, dtype=float)

        spot_series = spot_series.reindex(pd.to_datetime(dates)).ffill().bfill()
        dates_df = pd.DataFrame(index=pd.to_datetime(dates))
        dates_df['year'] = dates_df.index.strftime("%Y")
        
        for year, group in dates_df.groupby('year'):
            try:
                year_start = pd.Timestamp(f"{year}-01-01")
                year_end = pd.Timestamp(f"{int(year) + 1}-01-01")

                frames = []
                for alias in aliases:
                    if cache.use_shards:
                        conn = cache._ensure_price_conn(alias, year)
                    else:
                        conn = cache.conn
                    if alias == "SPX":
                        pattern = "O:SPX[0-9]*"
                    elif alias == "SPXW":
                        pattern = "O:SPXW*"
                    else:
                        pattern = f"O:{alias}*"
                    query = """
                        SELECT option_ticker, bid, ask, mid, close, pricing_date
                        FROM option_prices
                        WHERE pricing_date >= ? AND pricing_date < ?
                          AND option_ticker GLOB ?
                    """
                    df_alias = pd.read_sql(
                        query,
                        conn,
                        params=(year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d"), pattern),
                    )
                    if not df_alias.empty:
                        frames.append(df_alias)

                if not frames:
                    continue

                df_year = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['option_ticker', 'pricing_date'])
                if df_year.empty:
                    continue

                parsed = df_year['option_ticker'].str.extract(
                    r'^(?:O:)?(?P<underlying>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike_raw>\d{8})$'
                )
                df_year = pd.concat([df_year, parsed], axis=1)
                df_year = df_year.dropna(subset=['yy', 'mm', 'dd', 'cp', 'strike_raw'])
                if df_year.empty:
                    logger.warning(f"No parseable skew rows found for {underlying} in {year}; leaving feature missing.")
                    continue

                df_year['pricing_date'] = pd.to_datetime(df_year['pricing_date']).dt.normalize()
                df_year['expiration'] = pd.to_datetime(
                    "20" + df_year['yy'] + "-" + df_year['mm'] + "-" + df_year['dd'],
                    errors='coerce'
                )
                df_year['contract_type'] = np.where(df_year['cp'] == 'C', 'call', 'put')
                df_year['strike'] = pd.to_numeric(df_year['strike_raw'], errors='coerce') / 1000.0
                df_year = df_year.dropna(subset=['expiration', 'strike'])
                if df_year.empty:
                    continue

                df_year['mid_px'] = pd.to_numeric(df_year['mid'], errors='coerce')
                df_year['mid_px'] = df_year['mid_px'].where(df_year['mid_px'] > 0)
                df_year['mid_px'] = df_year['mid_px'].fillna(pd.to_numeric(df_year['close'], errors='coerce'))
                if 'bid' in df_year.columns and 'ask' in df_year.columns:
                    fallback_mid = (pd.to_numeric(df_year['bid'], errors='coerce') + pd.to_numeric(df_year['ask'], errors='coerce')) / 2.0
                    df_year['mid_px'] = df_year['mid_px'].fillna(fallback_mid)
                df_year = df_year[df_year['mid_px'].notna() & (df_year['mid_px'] > 0)].copy()
                if df_year.empty:
                    logger.warning(f"No usable quote mids found for {underlying} in {year}; leaving feature missing.")
                    continue

                df_year['dte'] = (df_year['expiration'] - df_year['pricing_date']).dt.days
                df_year['dte_diff'] = (df_year['dte'] - 30).abs()
                df_valid = df_year[(df_year['dte'] >= 20) & (df_year['dte'] <= 60)].copy()
                if df_valid.empty:
                    df_valid = df_year.copy()

                best_exp = (
                    df_valid.groupby(['pricing_date', 'expiration'], as_index=False)['dte_diff']
                    .min()
                    .sort_values(['pricing_date', 'dte_diff', 'expiration'])
                    .groupby('pricing_date', as_index=False)
                    .first()[['pricing_date', 'expiration']]
                )

                df_target = df_year.merge(best_exp, on=['pricing_date', 'expiration'], how='inner')
                if df_target.empty:
                    continue

                daily_rows = []
                for pricing_date, grp in df_target.groupby('pricing_date'):
                    spot = float(spot_series.get(pricing_date, np.nan))
                    rate = float(rate_series.get(pricing_date, 0.0))
                    if not np.isfinite(spot) or spot <= 0:
                        continue
                    exp = grp['expiration'].iloc[0]
                    dte = max(int((exp - pricing_date).days), 1)
                    t_years = max(dte / 365.0, 1e-5)

                    def _score_side(side_df, option_type):
                        if side_df.empty:
                            return None
                        rows = []
                        for _, row in side_df.iterrows():
                            iv = row.get('implied_vol')
                            if iv is None or not np.isfinite(iv) or iv <= 0:
                                iv = implied_volatility(
                                    float(row['mid_px']),
                                    spot,
                                    float(row['strike']),
                                    t_years,
                                    rate,
                                    0.0,
                                    option_type=option_type,
                                )
                            if iv is None or not np.isfinite(iv) or iv <= 0:
                                continue
                            if option_type == 'call':
                                delta = bs_call_delta(spot, float(row['strike']), t_years, rate, iv, 0.0)
                                delta_diff = abs(delta - 0.25)
                            else:
                                delta = bs_put_delta(spot, float(row['strike']), t_years, rate, iv, 0.0)
                                delta_diff = abs(abs(delta) - 0.25)
                            rows.append((float(row['strike']), float(iv), float(delta), float(delta_diff)))
                        if not rows:
                            return None
                        return min(rows, key=lambda item: item[3])

                    call_best = _score_side(grp[grp['contract_type'] == 'call'], 'call')
                    put_best = _score_side(grp[grp['contract_type'] == 'put'], 'put')
                    if call_best is None or put_best is None:
                        continue

                    call_iv = call_best[1]
                    put_iv = put_best[1]
                    daily_rows.append((pricing_date, put_iv - call_iv))

                if daily_rows:
                    daily_skew = pd.Series(
                        data=[v for _, v in daily_rows],
                        index=pd.to_datetime([d for d, _ in daily_rows]),
                        dtype=float
                    )
                    skew_series.loc[daily_skew.index] = daily_skew.values
            except Exception as e:
                logger.warning(f"Error in vectorized Skew for {year}: {e}")

        if skew_series.notna().sum() < max(10, int(len(skew_series) * 0.1)):
            logger.warning(f"Historical skew coverage too sparse for {underlying}; dropping feature.")
            return np.full(len(skew_series), np.nan, dtype=float)

        return skew_series.values

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
            if any(kw in col for kw in ['VIX', 'VVIX', 'Ratio', 'Log_Return', 'Skew', 'Momentum', 'GEX']):
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

    def scale_features(self, df, rolling=False, window=252*5, include_current=True):
        """
        Scale features using strictly causal logic.
        """
        if rolling:
            return self.rolling_scale_features(df, window=window, include_current=include_current)
        
        if self.rolling_scaler.center_ is None:
            red_alert("RollingRobustScaler not fitted. This will cause NaNs in inference.")
            return pd.DataFrame(index=df.index, columns=df.columns)
            
        return self.transform_features(df)

    def rolling_scale_features(self, df, window=252*5, include_current=True):
        """
        Regulation 8.2 & 8.4: Optimized strictly causal rolling-window scaling.
        """
        scaler = RollingRobustScaler(window=window)
        # Use optimized batch transform if possible
        scaled_df = scaler.batch_rolling_transform(df, include_current=include_current)
        
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

    async def build_fused_dataset(self, start_date, end_date, underlying="SPY", scale=True, rolling=False, wait=True):
        """Main pipeline to build the high-dimensional feature set."""
        fred_df = self.fetch_fred_data(start_date, end_date, wait=wait)
        yf_df = self.fetch_yf_data(start_date, end_date, wait=wait)
        
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
            
        # Add High Yield Spread Momentum
        if 'High_Yield_Spread' in combined.columns:
            fast = combined['High_Yield_Spread'].ewm(span=5, adjust=False).mean()
            slow = combined['High_Yield_Spread'].ewm(span=21, adjust=False).mean()
            combined['High_Yield_Spread_Momentum'] = fast - slow
            logger.info("Successfully integrated High Yield Spread Momentum.")

        # Add Option-Implied Skew only.
        # Historical SPX open interest is not present in the available OptionsDX
        # text cache, so Net_GEX cannot be reconstructed causally here.
        dates = combined.index.strftime("%Y-%m-%d").tolist()
        combined['Implied_Skew'] = self.fetch_historical_skew(dates, underlying="SPX")
        logger.info("Successfully integrated Implied_Skew feature.")

        # Re-run the missingness filter after adding the expensive option features.
        missing_ratios = combined.isna().mean()
        bad_cols = missing_ratios[missing_ratios > 0.8].index
        if len(bad_cols) > 0:
            logger.warning(f"Dropping features with >80% missing data after enrichment: {list(bad_cols)}")
            combined = combined.drop(columns=bad_cols)
            
        # Instead of aggressive dropna, we should drop rows only if core features are missing
        core_cols = ['SPY_Close']
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
            # 5-day trailing median filter suppresses short-lived spikes without
            # rewriting T with observations from T+1/T+2.
            combined[col] = combined[col].rolling(window=5, center=False, min_periods=1).median().ffill()
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
