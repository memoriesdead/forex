#!/usr/bin/env python3
"""
Trading CLI
===========
Main entry point for multi-symbol forex trading.

Usage:
    python scripts/trade.py --tier majors --mode paper
    python scripts/trade.py --symbols EURUSD,GBPUSD --mode live
    python scripts/trade.py --tier all --mode paper
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
import logging
import signal
from typing import Optional

from core.symbol.registry import SymbolRegistry
from core.trading import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-symbol forex trading bot"
    )
    parser.add_argument(
        '--tier',
        choices=['majors', 'crosses', 'exotics', 'all'],
        help='Symbol tier to trade'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Starting capital (default: 100000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='IB Gateway host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=4001,
        help='IB Gateway port (default: 4001)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    return parser.parse_args()


def show_status():
    """Show system status."""
    from core.models.loader import ModelLoader

    print("\n" + "=" * 60)
    print("MULTI-SYMBOL FOREX TRADING SYSTEM STATUS")
    print("=" * 60)

    # Symbol registry
    registry = SymbolRegistry.get()
    print(f"\nSymbols:")
    for tier in ['majors', 'crosses', 'exotics']:
        pairs = registry.get_enabled(tier=tier)
        print(f"  {tier.capitalize()}: {len(pairs)} pairs")
    print(f"  Total: {len(registry.get_enabled())} pairs")

    # Models
    loader = ModelLoader()
    available = loader.get_available()
    print(f"\nModels:")
    print(f"  Available: {len(available)}")
    if available:
        print(f"  Symbols: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")

    # Check IB connection
    print(f"\nIB Gateway:")
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect('localhost', 4001, clientId=99, readonly=True)
        print(f"  Status: Connected")
        ib.disconnect()
    except Exception as e:
        print(f"  Status: Not connected ({e})")

    print("\n" + "=" * 60 + "\n")


def run_bot(args):
    """Run the trading bot."""
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Live mode confirmation
    if args.mode == 'live':
        print("\n" + "!" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("This will execute REAL trades with REAL money!")
        print("!" * 60)
        confirm = input("\nType 'LIVE' to confirm: ")
        if confirm != 'LIVE':
            print("Aborted.")
            return

    # Create bot
    bot = TradingBot(
        capital=args.capital,
        symbols=symbols,
        tier=args.tier,
        mode=args.mode,
        ib_host=args.host,
        ib_port=args.port,
    )

    # Setup signal handlers
    def handle_shutdown(signum, frame):
        logger.info("Shutdown signal received")
        bot.stop()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Print startup info
    print("\n" + "=" * 60)
    print(f"TRADING BOT STARTED")
    print("=" * 60)
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Symbols: {len(bot.symbols)}")
    print(f"  IB: {args.host}:{args.port}")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")

    # Run
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()

        # Print summary
        status = bot.status()
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Ticks processed: {status['ticks_processed']:,}")
        print(f"  Signals generated: {status['signals_generated']:,}")
        print(f"  Trades executed: {status['trades_executed']:,}")
        if status['positions']['pnl']:
            pnl = status['positions']['pnl']
            print(f"  Realized P&L: ${pnl['realized']:,.2f}")
            print(f"  Unrealized P&L: ${pnl['unrealized']:,.2f}")
        print("=" * 60 + "\n")


def main():
    args = parse_args()

    if args.status:
        show_status()
        return

    if not args.tier and not args.symbols:
        print("Error: Must specify --tier or --symbols")
        print("Example: python scripts/trade.py --tier majors --mode paper")
        sys.exit(1)

    run_bot(args)


if __name__ == '__main__':
    main()
