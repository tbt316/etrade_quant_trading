
import yfinance as yf
from datetime import datetime, timedelta

def test_yf():
    start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    print(f"Testing yfinance download for SPY from {start}...")
    df = yf.download('SPY', start=start, progress=False, auto_adjust=False)
    print("Columns:", df.columns)
    print("Head:\n", df.head())
    
    # Test the extraction logic
    temp_df = df.copy()
    if hasattr(temp_df.columns, 'levels') and len(temp_df.columns.levels) > 1:
        print("Detected MultiIndex columns")
        temp_df = temp_df.droplevel(1, axis=1)
    
    close_col = 'Close'
    if close_col not in temp_df.columns:
        print(f"'{close_col}' not in columns. Searching...")
        for col in temp_df.columns:
            if isinstance(col, tuple) and 'Close' in col:
                close_col = col
                break
            elif isinstance(col, str) and 'Close' in col:
                close_col = col
                break
    
    print(f"Using close_col: {close_col}")
    if close_col in temp_df.columns:
        for idx in temp_df.index:
            dt_str = str(idx.date())
            val = temp_df.loc[idx, close_col]
            if hasattr(val, 'iloc'):
                val = val.iloc[0]
            print(f"{dt_str}: {val}")
    else:
        print("FAILED to find Close column")

if __name__ == "__main__":
    test_yf()
