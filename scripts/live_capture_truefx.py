"""
TrueFX Live Data Capture Script (24/7 on Oracle Cloud)

Fetches real-time forex tick data every 1 second from TrueFX (free)
Stores to daily CSV files with automatic rotation and cleanup

Deploy to: /home/ubuntu/projects/forex/scripts/live_capture_truefx.py
Run as: systemd service (auto-restart on failure)
"""

import requests
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import signal
import sys

# Configuration
BASE_DIR = Path("/home/ubuntu/projects/forex")
DATA_DIR = BASE_DIR / "data" / "live"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "live_capture.log"

TRUEFX_URL = "https://webrates.truefx.com/rates/connect.html"
FETCH_INTERVAL = 1.0  # seconds
RETENTION_DAYS = 7  # Keep last 7 days of data

# TrueFX pairs (10 major pairs available for free)
PAIRS = [
    "EUR/USD", "USD/JPY", "GBP/USD", "EUR/GBP", "USD/CHF",
    "EUR/JPY", "EUR/CHF", "USD/CAD", "AUD/USD", "GBP/JPY"
]

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    logger.info("Shutdown signal received, stopping gracefully...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def fetch_truefx_data() -> List[Dict]:
    """
    Fetch current tick data from TrueFX
    Returns list of dicts with pair, timestamp, bid, ask, spread
    """
    try:
        response = requests.get(
            TRUEFX_URL,
            params={'f': 'csv', 'c': 'EUR/USD,USD/JPY,GBP/USD,EUR/GBP,USD/CHF,EUR/JPY,EUR/CHF,USD/CAD,AUD/USD,GBP/JPY'},
            timeout=5
        )
        response.raise_for_status()

        # Parse TrueFX CSV response
        # Format: Pair,Timestamp,Bid_big,Bid_point,Ask_big,Ask_point,High,Low,Open
        # Example: EUR/USD,1704729600000,1.09,845,1.09,847,1.09850,1.09840,1.09842

        lines = response.text.strip().split('\n')
        results = []
        timestamp = datetime.utcnow()

        for line in lines:
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 6:
                continue

            pair = parts[0].replace('/', '')  # EUR/USD -> EURUSD

            # Parse bid/ask from TrueFX format
            bid_big = parts[2]
            bid_point = parts[3]
            ask_big = parts[4]
            ask_point = parts[5]

            bid = float(f"{bid_big}{bid_point}")
            ask = float(f"{ask_big}{ask_point}")
            spread = round(ask - bid, 5)

            results.append({
                'pair': pair,
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'spread': spread
            })

        return results

    except requests.RequestException as e:
        logger.error(f"TrueFX API request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing TrueFX data: {e}")
        return []


def get_daily_file(pair: str, date: datetime) -> Path:
    """Get file path for pair on specific date"""
    date_str = date.strftime('%Y-%m-%d')
    day_dir = DATA_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{pair}_{date_str}.csv"


def append_to_csv(pair: str, timestamp: datetime, bid: float, ask: float, spread: float):
    """Append tick data to daily CSV file"""
    file_path = get_daily_file(pair, timestamp)

    # Create file with header if doesn't exist
    if not file_path.exists():
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'bid', 'ask', 'spread'])
        logger.info(f"Created new file: {file_path}")

    # Append data
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],  # milliseconds
            bid,
            ask,
            spread
        ])


def cleanup_old_files():
    """Delete data older than RETENTION_DAYS"""
    cutoff_date = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    deleted_count = 0

    if not DATA_DIR.exists():
        return

    for day_dir in DATA_DIR.iterdir():
        if not day_dir.is_dir():
            continue

        try:
            dir_date = datetime.strptime(day_dir.name, '%Y-%m-%d')
            if dir_date < cutoff_date:
                # Delete entire day directory
                for file in day_dir.iterdir():
                    file.unlink()
                day_dir.rmdir()
                deleted_count += 1
                logger.info(f"Deleted old data: {day_dir.name}")
        except ValueError:
            # Not a date directory, skip
            pass

    if deleted_count > 0:
        logger.info(f"Cleanup complete: removed {deleted_count} old directories")


def get_stats() -> Dict:
    """Get capture statistics"""
    stats = {
        'total_files': 0,
        'total_size_mb': 0,
        'oldest_date': None,
        'newest_date': None
    }

    if not DATA_DIR.exists():
        return stats

    dates = []
    for day_dir in DATA_DIR.iterdir():
        if not day_dir.is_dir():
            continue

        try:
            dates.append(datetime.strptime(day_dir.name, '%Y-%m-%d'))
            for file in day_dir.iterdir():
                stats['total_files'] += 1
                stats['total_size_mb'] += file.stat().st_size / (1024 * 1024)
        except ValueError:
            pass

    if dates:
        stats['oldest_date'] = min(dates).strftime('%Y-%m-%d')
        stats['newest_date'] = max(dates).strftime('%Y-%m-%d')

    return stats


def main():
    """Main capture loop"""
    logger.info("="*60)
    logger.info("TrueFX Live Data Capture Started")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Fetch interval: {FETCH_INTERVAL}s")
    logger.info(f"Retention: {RETENTION_DAYS} days")
    logger.info(f"Pairs: {len(PAIRS)}")
    logger.info("="*60)

    # Initial cleanup
    cleanup_old_files()

    tick_count = 0
    error_count = 0
    last_cleanup = datetime.utcnow()
    last_stats = datetime.utcnow()

    while not shutdown_flag:
        loop_start = time.time()

        # Fetch data
        tick_data = fetch_truefx_data()

        if tick_data:
            # Save each pair
            for data in tick_data:
                try:
                    append_to_csv(
                        data['pair'],
                        data['timestamp'],
                        data['bid'],
                        data['ask'],
                        data['spread']
                    )
                except Exception as e:
                    logger.error(f"Failed to save {data['pair']}: {e}")
                    error_count += 1

            tick_count += 1

            # Log progress every 60 seconds
            if tick_count % 60 == 0:
                logger.info(f"Captured {tick_count} ticks, {len(tick_data)} pairs, {error_count} errors")
        else:
            error_count += 1
            logger.warning("No data received from TrueFX")

        # Cleanup old files once per hour
        if (datetime.utcnow() - last_cleanup).total_seconds() > 3600:
            cleanup_old_files()
            last_cleanup = datetime.utcnow()

        # Log stats every 15 minutes
        if (datetime.utcnow() - last_stats).total_seconds() > 900:
            stats = get_stats()
            logger.info(f"Storage: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB, "
                       f"range: {stats['oldest_date']} to {stats['newest_date']}")
            last_stats = datetime.utcnow()

        # Sleep until next interval
        elapsed = time.time() - loop_start
        sleep_time = max(0, FETCH_INTERVAL - elapsed)

        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final stats
    logger.info("="*60)
    logger.info("Capture stopped gracefully")
    logger.info(f"Total ticks captured: {tick_count}")
    logger.info(f"Total errors: {error_count}")
    stats = get_stats()
    logger.info(f"Final storage: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
