import ccxt
import os
import pandas as pd
import time


def load_dataset(timeframe):
    RAW_FILE = f'btcusdt_{timeframe}_raw.parquet'
    interval = ccxt.binance().parse_timeframe(timeframe) * 1000
    if os.path.exists(RAW_FILE):
        df = pd.read_parquet(RAW_FILE)
    else:
        exchange = ccxt.binance()
        symbol = 'BTC/USDT'
        since = exchange.parse8601('2022-01-01T00:00:00Z')

        all_rows = []
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            all_rows += batch
            since = batch[-1][0] + interval
            time.sleep(exchange.rateLimit / 1000)

        now_ms = exchange.milliseconds()
        all_rows = [row for row in all_rows if row[0] + interval <= now_ms]

        df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_parquet(RAW_FILE)

    
    if df.timestamp.is_monotonic_increasing == False:
        raise ValueError("Timestamps not monotonic increasing")
    
    if df.timestamp.duplicated().any() == True:
        raise ValueError("Timestamps are not unique!")

    
    if df.timestamp.iloc[-1] + interval > time.time() * 1000:
        raise ValueError("Last candle is still in progress!")
    
    return df