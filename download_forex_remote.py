#!/usr/bin/env python3
import requests
import pandas as pd
from datetime import datetime, timedelta
import struct
import lzma

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed"
CURRENCY_PAIRS = ['EURUSD', 'USDJPY', 'GBPUSD', 'EURGBP', 'USDCHF', 'EURJPY', 'EURCHF', 'USDCAD', 'AUDUSD', 'GBPJPY']

def download_day(pair, date):
    year, month, day = date.year, date.month - 1, date.day - 1
    ticks = []
    for hour in range(24):
        try:
            url = f"{DUKASCOPY_URL}/{pair}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
            r = requests.get(url, timeout=30)
            if r.status_code != 200: continue
            data = lzma.decompress(r.content)
            for i in range(len(data) // 20):
                chunk = data[i*20:(i+1)*20]
                ts_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
                ts = datetime(year, month+1, day+1, hour) + timedelta(milliseconds=ts_ms)
                ticks.append({'timestamp': ts, 'pair': pair, 'bid': bid/100000, 'ask': ask/100000, 'bid_volume': bid_vol, 'ask_volume': ask_vol})
        except: pass
    if ticks:
        df = pd.DataFrame(ticks)
        filename = f"/root/forex_data_node/{pair}_{date.strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {pair}_{date.strftime('%Y%m%d')}.csv: {len(df):,} ticks")
        return len(df)
    return 0

for date_str in ['2026-01-05', '2026-01-06', '2026-01-07']:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    if date.weekday() >= 5: continue
    print(f"\n{date_str}:")
    for pair in CURRENCY_PAIRS:
        download_day(pair, date)
