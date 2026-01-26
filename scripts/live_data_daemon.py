#!/usr/bin/env python3
"""
24/7 LIVE FOREX DATA DAEMON
===========================
Streams forex pairs from Interactive Brokers + TrueFX backup
Saves to data/live/ as parquet files

Run: python scripts/live_data_daemon.py
"""

import sys
sys.path.insert(0, '.')

import time
import signal
import logging
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd
import requests

# Setup
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data" / "live"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'live_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TrueFX - free real-time data
TRUEFX_PAIRS = "EUR/USD,USD/JPY,GBP/USD,EUR/GBP,USD/CHF,EUR/JPY,EUR/CHF,USD/CAD,AUD/USD,GBP/JPY,AUD/JPY,NZD/USD,EUR/AUD,EUR/CAD,GBP/CHF"
TRUEFX_URL = "https://webrates.truefx.com/rates/connect.html"

# Global
running = True
tick_buffers = {}
tick_counts = {}
last_save = {}
lock = threading.Lock()


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def add_tick(tick: dict):
    """Add tick to buffer."""
    pair = tick['pair']

    with lock:
        if pair not in tick_buffers:
            tick_buffers[pair] = []
            tick_counts[pair] = 0
            last_save[pair] = datetime.now()

        tick_buffers[pair].append(tick)
        tick_counts[pair] += 1

        # Save every 300 ticks or 5 minutes
        now = datetime.now()
        if len(tick_buffers[pair]) >= 300 or (now - last_save[pair]).seconds >= 300:
            save_buffer(pair)


def save_buffer(pair: str):
    """Save buffer to parquet."""
    if not tick_buffers.get(pair):
        return

    try:
        df = pd.DataFrame(tick_buffers[pair])
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']

        today = datetime.now().strftime('%Y%m%d')
        pair_dir = DATA_DIR / pair
        pair_dir.mkdir(parents=True, exist_ok=True)
        filepath = pair_dir / f"{pair}_{today}_live.parquet"

        if filepath.exists():
            existing = pd.read_parquet(filepath)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(filepath, index=False)
        count = len(tick_buffers[pair])
        tick_buffers[pair] = []
        last_save[pair] = datetime.now()
        logger.info(f"Saved {pair}: {count} ticks")

    except Exception as e:
        logger.error(f"Save error {pair}: {e}")


def run_truefx():
    """TrueFX stream - 15 major pairs."""
    global running
    session = requests.Session()

    logger.info("TrueFX stream started (15 pairs)")

    while running:
        try:
            r = session.get(f"{TRUEFX_URL}?f=csv&c={TRUEFX_PAIRS}", timeout=10)
            if r.status_code == 200:
                for line in r.text.strip().split('\n'):
                    p = line.split(',')
                    if len(p) >= 6:
                        try:
                            pair = p[0].replace('/', '')
                            bid = float(p[2] + p[3])
                            ask = float(p[4] + p[5])
                            add_tick({
                                'timestamp': datetime.now(),
                                'pair': pair,
                                'bid': bid,
                                'ask': ask,
                                'source': 'TrueFX'
                            })
                        except:
                            pass
            time.sleep(1)
        except Exception as e:
            logger.error(f"TrueFX: {e}")
            time.sleep(5)


def run_ib():
    """IB Gateway stream - all forex pairs."""
    global running

    try:
        from ib_insync import IB, Forex, util
        util.startLoop()
    except ImportError:
        logger.error("ib_insync not installed")
        return

    ib = IB()
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
             'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD']

    def on_pending_tickers(tickers):
        for t in tickers:
            if t.bid and t.ask and t.bid > 0:
                pair = t.contract.symbol + t.contract.currency
                add_tick({
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'bid': float(t.bid),
                    'ask': float(t.ask),
                    'source': 'IB'
                })

    while running:
        try:
            if not ib.isConnected():
                logger.info("Connecting to IB Gateway...")
                ib.connect('127.0.0.1', 4004, clientId=10)
                logger.info(f"IB connected: {ib.managedAccounts()}")

                ib.pendingTickersEvent += on_pending_tickers

                for pair in pairs:
                    contract = Forex(pair)
                    ib.qualifyContracts(contract)
                    ib.reqMktData(contract)
                    time.sleep(0.2)

                logger.info(f"IB subscribed to {len(pairs)} pairs")

            ib.sleep(1)

        except Exception as e:
            logger.error(f"IB: {e}")
            time.sleep(10)

    if ib.isConnected():
        ib.disconnect()


def status_printer():
    """Print status every minute."""
    global running
    start = datetime.now()

    while running:
        time.sleep(60)
        total = sum(tick_counts.values())
        elapsed = (datetime.now() - start).total_seconds()
        tps = total / elapsed if elapsed > 0 else 0
        pairs = len([p for p, c in tick_counts.items() if c > 0])
        logger.info(f"STATUS: {pairs} pairs | {total:,} ticks | {tps:.1f}/sec")


def main():
    global running

    print("=" * 60)
    print("24/7 LIVE FOREX DATA DAEMON")
    print("=" * 60)
    print(f"TrueFX: 15 major pairs (free)")
    print(f"IB Gateway: 14 pairs (if connected)")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()

    # Start threads
    threads = [
        threading.Thread(target=run_truefx, daemon=True),
        threading.Thread(target=run_ib, daemon=True),
        threading.Thread(target=status_printer, daemon=True),
    ]

    for t in threads:
        t.start()

    # Wait
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False

    # Save remaining
    logger.info("Saving remaining buffers...")
    with lock:
        for pair in list(tick_buffers.keys()):
            save_buffer(pair)

    logger.info("Daemon stopped.")


if __name__ == '__main__':
    main()
