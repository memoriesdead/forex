#!/usr/bin/env python3
"""
Oracle Cloud 24/7 Auto-Downloader
Runs daily via cron, downloads forex data, cleans it, tracks availability
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import struct
import lzma
import json

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed"

# Import all forex pairs
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from all_forex_pairs import ALL_PAIRS

CURRENCY_PAIRS = ALL_PAIRS  # 90+ pairs

# Oracle Cloud paths
BASE_DIR = Path("/home/ubuntu/projects/forex")
DATA_RAW = BASE_DIR / "data" / "dukascopy_local"
DATA_CLEANED = BASE_DIR / "data_cleaned" / "dukascopy_local"
TRACKER_FILE = BASE_DIR / "download_tracker.json"


def download_day(pair: str, date: datetime) -> int:
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
            print(f"Error downloading {pair} {date.strftime('%Y-%m-%d')} {hour:02d}:00: {e}")
            continue

    if all_ticks:
        df = pd.DataFrame(all_ticks)
        filename = f"{pair}_{date.strftime('%Y%m%d')}.csv"
        output_file = DATA_RAW / filename
        df.to_csv(output_file, index=False)
        return len(df)

    return 0


def clean_file(input_file: Path, output_file: Path) -> int:
    """Clean and validate a single forex CSV file."""
    try:
        df = pd.read_csv(input_file)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop_duplicates(subset=['timestamp', 'pair'])

        valid_spread = df['bid'] < df['ask']
        if not valid_spread.all():
            invalid_count = (~valid_spread).sum()
            print(f"  Removed {invalid_count} invalid spreads from {input_file.name}")
            df = df[valid_spread]

        df = df.sort_values('timestamp')

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        return len(df)

    except Exception as e:
        print(f"  Error cleaning {input_file.name}: {e}")
        return 0


def update_tracker(date: datetime, pairs_downloaded: list):
    """Update download tracker JSON."""
    tracker = {}
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, 'r') as f:
            tracker = json.load(f)

    date_str = date.strftime('%Y%m%d')
    tracker[date_str] = {
        'date': date.strftime('%Y-%m-%d'),
        'pairs': pairs_downloaded,
        'downloaded_at': datetime.now().isoformat(),
        'status': 'available'
    }

    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)


def main():
    """Main auto-download routine - downloads yesterday's data."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CLEANED.mkdir(parents=True, exist_ok=True)

    # Download yesterday's data (market closes, data becomes available)
    yesterday = datetime.now() - timedelta(days=1)

    # Skip weekends
    if yesterday.weekday() >= 5:
        print(f"Skipping weekend: {yesterday.strftime('%Y-%m-%d')}")
        return

    print(f"Auto-downloading: {yesterday.strftime('%Y-%m-%d')}")

    # Download raw data
    pairs_downloaded = []
    total_ticks = 0

    for pair in CURRENCY_PAIRS:
        ticks = download_day(pair, yesterday)
        if ticks > 0:
            total_ticks += ticks
            pairs_downloaded.append(pair)
            print(f"  {pair}: {ticks:,} ticks")

    if not pairs_downloaded:
        print("No data downloaded")
        return

    print(f"\nDownloaded {len(pairs_downloaded)} pairs, {total_ticks:,} ticks")

    # Clean data
    print("\nCleaning data...")
    date_str = yesterday.strftime('%Y%m%d')

    for pair in pairs_downloaded:
        input_file = DATA_RAW / f"{pair}_{date_str}.csv"
        output_file = DATA_CLEANED / f"{pair}_{date_str}.csv"

        cleaned_ticks = clean_file(input_file, output_file)
        if cleaned_ticks > 0:
            print(f"  {pair}: {cleaned_ticks:,} ticks")

    # Update tracker
    update_tracker(yesterday, pairs_downloaded)
    print(f"\nTracker updated: {TRACKER_FILE}")


if __name__ == "__main__":
    main()
