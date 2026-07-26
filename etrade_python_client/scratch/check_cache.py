import sqlite3

db_path = "backtest_cache/option_data.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT option_ticker) FROM option_prices WHERE underlying = 'SPX'")
    count, distinct = cursor.fetchone()
    print(f"SPX Option Prices in Database: {count:,} records across {distinct:,} distinct tickers")
    
    cursor.execute("SELECT MIN(pricing_date), MAX(pricing_date) FROM option_prices WHERE underlying = 'SPX'")
    min_date, max_date = cursor.fetchone()
    print(f"Date Range for SPX in DB: {min_date} to {max_date}")
except Exception as e:
    print("Error:", e)

conn.close()
