#!/usr/bin/env python3
"""Download latest forex tick data from TrueFX."""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

TRUEFX_URL = "https://webrates.truefx.com/rates/connect.html"

CURRENCY_PAIRS = [
    'EUR/USD', 'USD/JPY', 'GBP/USD', 'EUR/GBP', 'USD/CHF',
    'EUR/JPY', 'EUR/CHF', 'USD/CAD', 'AUD/USD', 'GBP/JPY'
]

def download_truefx_data(output_dir: Path, hours: int = 24):
    """Download TrueFX tick data for the last N hours."""
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = None
    try:
        response = requests.get(f"{TRUEFX_URL}?f=html&c=" + ",".join(CURRENCY_PAIRS))
        session_id = response.text
        print(f"TrueFX session: {session_id}")
    except Exception as e:
        print(f"Failed to get TrueFX session: {e}")
        return

    all_data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    print(f"Collecting data from {start_time} to {end_time}")

    try:
        for _ in range(hours * 60):
            response = requests.get(f"{TRUEFX_URL}?id={session_id}")
            if response.status_code == 200 and response.text:
                lines = response.text.strip().split('\n')
                timestamp = datetime.now()

                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        pair = parts[0]
                        bid = parts[1]
                        ask = parts[3]
                        all_data.append({
                            'timestamp': timestamp,
                            'pair': pair,
                            'bid': bid,
                            'ask': ask
                        })

            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping data collection...")

    if all_data:
        df = pd.DataFrame(all_data)
        filename = f"truefx_tick_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_dir / filename, index=False)
        print(f"Saved {len(df)} ticks to {filename}")
    else:
        print("No data collected")

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "truefx_tick"
    download_truefx_data(data_dir, hours=1)
