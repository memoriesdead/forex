#!/usr/bin/env python3
"""Clean and validate recently downloaded forex data."""

import pandas as pd
from pathlib import Path
from datetime import datetime

def clean_forex_file(input_file: Path, output_file: Path):
    """Clean and validate a single forex CSV file."""
    try:
        df = pd.read_csv(input_file)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'pair'])

        # Validate bid < ask (basic sanity check)
        valid_spread = df['bid'] < df['ask']
        if not valid_spread.all():
            invalid_count = (~valid_spread).sum()
            print(f"  Warning: {invalid_count} invalid spreads in {input_file.name}")
            df = df[valid_spread]

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Save cleaned data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        return len(df)

    except Exception as e:
        print(f"  Error cleaning {input_file.name}: {e}")
        return 0

def clean_recent_files():
    """Clean recently downloaded files (1/5-1/7/2026)."""
    data_dir = Path(__file__).parent.parent / "data" / "dukascopy_local"
    cleaned_dir = Path(__file__).parent.parent / "data_cleaned" / "dukascopy_local"

    # Find files for 1/5-1/7/2026
    target_dates = ['20260105', '20260106', '20260107']
    files_to_clean = []

    for date_str in target_dates:
        files_to_clean.extend(data_dir.glob(f"*_{date_str}.csv"))

    print(f"Cleaning {len(files_to_clean)} files...")

    total_ticks = 0
    for input_file in sorted(files_to_clean):
        output_file = cleaned_dir / input_file.name
        ticks = clean_forex_file(input_file, output_file)
        if ticks > 0:
            total_ticks += ticks
            print(f"  {input_file.name}: {ticks:,} ticks")

    print(f"\nCleaned {len(files_to_clean)} files, {total_ticks:,} total ticks")
    print(f"Output: {cleaned_dir}")

if __name__ == "__main__":
    clean_recent_files()
