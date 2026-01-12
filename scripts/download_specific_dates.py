#!/usr/bin/env python3
"""Download forex data for specific dates."""

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
        print(f"Saved {filename}: {len(df):,} ticks")
        return len(df)

    return 0

def download_dates(date_strings: list):
    """Download data for specific dates."""
    output_dir = Path(__file__).parent.parent / "data" / "dukascopy_local"
    output_dir.mkdir(parents=True, exist_ok=True)

    for date_str in date_strings:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')

        if target_date.weekday() >= 5:
            print(f"Skipping weekend: {date_str}")
            continue

        print(f"\nDownloading {date_str}...")

        total_ticks = 0
        for pair in CURRENCY_PAIRS:
            ticks = download_dukascopy_day(pair, target_date, output_dir)
            total_ticks += ticks

        print(f"  Total: {total_ticks:,} ticks across {len(CURRENCY_PAIRS)} pairs")

if __name__ == "__main__":
    # Download 1/4 through 1/7/2026
    download_dates(['2026-01-04', '2026-01-05', '2026-01-06', '2026-01-07'])
