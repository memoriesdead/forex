"""
TrueFX Live Tick Data Capture (HTTP Polling)
Polls TrueFX CSV API every 1 second and saves to CSV files.
Runs 24/7 on Oracle Cloud.
"""

import requests
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TrueFX pairs (free API)
TRUEFX_PAIRS = [
    'EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF', 'CHFJPY'
]

# TrueFX CSV API endpoint (no auth required for these pairs)
TRUEFX_API_URL = "https://webrates.truefx.com/rates/connect.html?f=csv"


class LiveTickCapture:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffer ticks in memory, flush every 50 ticks per pair
        self.tick_buffers = defaultdict(list)
        self.buffer_size = 50

        # Track stats
        self.tick_counts = defaultdict(int)
        self.start_time = datetime.now(timezone.utc)
        self.last_data = {}  # Deduplicate

        # File handles (open once, append continuously)
        self.file_handles = {}
        self.csv_writers = {}

        self.running = False
        self.session = requests.Session()

    def get_output_file(self, pair: str) -> Path:
        """Get output CSV file path for today's date."""
        date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        return self.output_dir / f"{pair}_{date_str}_live.csv"

    def init_csv_writer(self, pair: str):
        """Initialize CSV writer for a pair."""
        if pair in self.csv_writers:
            return

        file_path = self.get_output_file(pair)

        # Check if file exists to determine if we need header
        file_exists = file_path.exists()

        # Open in append mode
        fh = open(file_path, 'a', newline='', buffering=1)
        writer = csv.writer(fh)

        # Write header if new file
        if not file_exists:
            writer.writerow(['timestamp', 'pair', 'bid', 'ask', 'bid_points', 'ask_points'])

        self.file_handles[pair] = fh
        self.csv_writers[pair] = writer

        logger.info(f"Initialized CSV writer for {pair}: {file_path}")

    def flush_buffer(self, pair: str):
        """Flush buffered ticks to CSV file."""
        if pair not in self.tick_buffers or len(self.tick_buffers[pair]) == 0:
            return

        # Ensure CSV writer exists
        self.init_csv_writer(pair)

        # Write all buffered ticks
        writer = self.csv_writers[pair]
        for tick in self.tick_buffers[pair]:
            writer.writerow(tick)

        # Flush to disk
        self.file_handles[pair].flush()

        # Clear buffer
        count = len(self.tick_buffers[pair])
        self.tick_buffers[pair].clear()

        logger.debug(f"Flushed {count} ticks for {pair}")

    def parse_truefx_csv(self, csv_data: str):
        """
        Parse TrueFX CSV format.
        Format: EUR/USD,1704758400123,1.08234,1.08235,1.08236,1.08237
        Fields: pair,timestamp,bid_big,bid_points,ask_big,ask_points
        """
        lines = csv_data.strip().split('\n')

        for line in lines:
            if not line:
                continue

            try:
                parts = line.split(',')
                if len(parts) < 6:
                    continue

                # Parse fields
                pair_raw = parts[0]
                timestamp_ms = int(parts[1])
                bid_big = parts[2]
                bid_points = parts[3]
                ask_big = parts[4]
                ask_points = parts[5]

                # Convert pair format: EUR/USD -> EURUSD
                pair = pair_raw.replace('/', '')

                # Only capture pairs we're tracking
                if pair not in TRUEFX_PAIRS:
                    continue

                # Build bid/ask prices
                bid = f"{bid_big}{bid_points}"
                ask = f"{ask_big}{ask_points}"

                # Convert timestamp to ISO format
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # Deduplicate - skip if same as last tick
                tick_key = f"{pair}_{bid}_{ask}"
                if self.last_data.get(pair) == tick_key:
                    continue
                self.last_data[pair] = tick_key

                # Add to buffer
                tick = [timestamp_str, pair, bid, ask, bid_points, ask_points]
                self.tick_buffers[pair].append(tick)

                # Update stats
                self.tick_counts[pair] += 1

                # Flush if buffer is full
                if len(self.tick_buffers[pair]) >= self.buffer_size:
                    self.flush_buffer(pair)

            except Exception as e:
                logger.error(f"Error parsing line: {e} - Line: {line}")

    def poll_truefx(self):
        """Poll TrueFX API for latest ticks."""
        try:
            response = self.session.get(TRUEFX_API_URL, timeout=5)
            response.raise_for_status()

            csv_data = response.text
            self.parse_truefx_csv(csv_data)

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"Error in poll_truefx: {e}")

    def capture_loop(self):
        """Main capture loop - polls every 1 second."""
        logger.info("Starting capture loop...")

        while self.running:
            try:
                self.poll_truefx()
                time.sleep(1)  # Poll every 1 second

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(5)  # Wait before retry

        # Flush all buffers on exit
        logger.info("Flushing all buffers...")
        for pair in list(self.tick_buffers.keys()):
            self.flush_buffer(pair)

        # Close all file handles
        for fh in self.file_handles.values():
            fh.close()

        logger.info("Capture loop stopped")

    def start(self):
        """Start capturing ticks."""
        self.running = True
        logger.info(f"Starting live tick capture to {self.output_dir}")
        logger.info(f"Polling TrueFX API every 1 second for {len(TRUEFX_PAIRS)} pairs")

        # Start stats reporting thread
        stats_thread = threading.Thread(target=self.report_stats, daemon=True)
        stats_thread.start()

        # Start capture loop (blocking)
        try:
            self.capture_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.stop()

    def stop(self):
        """Stop capturing ticks."""
        self.running = False
        logger.info("Live capture stopped")

    def report_stats(self):
        """Report statistics every 60 seconds."""
        while self.running:
            time.sleep(60)

            uptime = datetime.now(timezone.utc) - self.start_time
            total_ticks = sum(self.tick_counts.values())

            logger.info(f"=== STATS ===")
            logger.info(f"Uptime: {uptime}")
            logger.info(f"Total ticks: {total_ticks:,}")
            if uptime.total_seconds() > 0:
                logger.info(f"Ticks/min: {total_ticks / (uptime.total_seconds() / 60):.1f}")
            logger.info(f"Pairs active: {len(self.tick_counts)}")

            # Top 5 pairs by tick count
            if self.tick_counts:
                top_pairs = sorted(self.tick_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for pair, count in top_pairs:
                    logger.info(f"  {pair}: {count:,} ticks")


def main():
    """Main entry point."""
    # Oracle Cloud path
    output_dir = Path("/home/ubuntu/projects/forex/data/truefx_live")

    # Create capture instance
    capture = LiveTickCapture(output_dir)

    try:
        capture.start()
    except KeyboardInterrupt:
        capture.stop()


if __name__ == "__main__":
    main()
