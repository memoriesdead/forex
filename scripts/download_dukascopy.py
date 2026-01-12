#!/usr/bin/env python3
"""Download historical forex data from Dukascopy."""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import struct
import lzma

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed"

CURRENCY_PAIRS = [
    'EURUSD', 'USDJPY', 'GBPUSD', 'EURGBP', 'USDCHF',
    'EURJPY', 'EURCHF', 'USDCAD', 'AUDUSD', 'GBPJPY'
]

def download_dukascopy_day(pair: str, date: datetime, output_dir: Path):
    """Download one day of tick data from Dukascopy."""
    year = date.year
    month = date.month - 1
    day = date.day - 1

    all_ticks = []

    for hour in range(24):
        url = f"{DUKASCOPY_URL}/{pair}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue

            data = lzma.decompress(response.content)
            num_ticks = len(data) // 20

            for i in range(num_ticks):
                chunk = data[i*20:(i+1)*20]
                timestamp_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)

                timestamp = datetime(year, month + 1, day + 1, hour) + timedelta(milliseconds=timestamp_ms)

                all_ticks.append({
                    'timestamp': timestamp,
                    'pair': pair,
                    'bid': bid / 100000,
                    'ask': ask / 100000,
                    'bid_volume': bid_vol,
                    'ask_volume': ask_vol
                })

        except Exception as e:
            print(f"Error downloading {pair} {year}-{month+1:02d}-{day+1:02d} {hour:02d}:00: {e}")
            continue

    if all_ticks:
        df = pd.DataFrame(all_ticks)
        filename = f"{pair}_{date.strftime('%Y%m%d')}.csv"
        output_file = output_dir / filename
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} ticks to {filename}")
        return len(df)

    return 0

def download_recent_days(days: int = 4):
    """Download last N days of data for all pairs."""
    output_dir = Path(__file__).parent.parent / "data" / "dukascopy_local"
    output_dir.mkdir(parents=True, exist_ok=True)

    for day_offset in range(days):
        target_date = datetime.now() - timedelta(days=day_offset + 1)

        if target_date.weekday() >= 5:
            print(f"Skipping weekend: {target_date.strftime('%Y-%m-%d')}")
            continue

        print(f"\nDownloading data for {target_date.strftime('%Y-%m-%d')}...")

        for pair in CURRENCY_PAIRS:
            print(f"  {pair}...", end=" ")
            ticks = download_dukascopy_day(pair, target_date, output_dir)
            if ticks > 0:
                print(f"{ticks:,} ticks")

if __name__ == "__main__":
    download_recent_days(days=4)
