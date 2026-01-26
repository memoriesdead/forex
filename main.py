#!/usr/bin/env python3
"""
Forex HFT Trading System - Main Entry Point
============================================

Single entry point for all operations:
  python main.py trade --mode paper --symbols EURUSD,GBPUSD,USDJPY
  python main.py train --symbols EURUSD --target direction_10
  python main.py backtest --strategy ml_ensemble --days 30
  python main.py status
  python main.py data download --source truefx --days 7

Architecture:
  main.py (this file) - CLI entry point
  core/               - Core modules (feature engines, models, data loaders)
  scripts/            - Utility scripts (called via main.py)
  models/             - Trained model files
  data/               - Market data
  training_package/   - Training data and scripts
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_trade(args):
    """Start the trading bot."""
    from scripts.hft_trading_bot import HFTTradingBot

    symbols = args.symbols.split(',')

    logger.info(f"Starting trading bot in {args.mode} mode")
    logger.info(f"Symbols: {symbols}")

    bot = HFTTradingBot(
        mode=args.mode,
        symbols=symbols,
        ib_host=args.host,
        ib_port=args.port
    )

    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot.stop()


def cmd_train(args):
    """Train models locally on RTX 5080."""
    from training_package.train_models import main as train_main

    logger.info(f"Training models for: {args.symbols}")
    logger.info(f"Target: {args.target}")

    # Run training
    train_main()


def cmd_backtest(args):
    """Run backtesting."""
    from backtest import run_backtest

    logger.info(f"Running backtest: {args.strategy}")
    logger.info(f"Period: {args.days} days")

    run_backtest(
        strategy=args.strategy,
        days=args.days,
        symbols=args.symbols.split(',') if args.symbols else None
    )


def cmd_status(args):
    """Check system status."""
    import torch

    print("\n" + "="*60)
    print("FOREX HFT SYSTEM STATUS")
    print("="*60)

    # GPU Status
    print("\n[GPU]")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}")
    else:
        print("  No GPU available")

    # Models Status
    print("\n[MODELS]")
    models_dir = PROJECT_ROOT / "models" / "production"
    if models_dir.exists():
        models = list(models_dir.glob("*.pkl"))
        print(f"  Location: {models_dir}")
        print(f"  Models found: {len(models)}")
        for m in models[:5]:
            print(f"    - {m.name}")
    else:
        print("  No models found")

    # Data Status
    print("\n[DATA]")
    data_dir = PROJECT_ROOT / "training_package"
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        symbol_dir = data_dir / symbol
        if symbol_dir.exists():
            train_file = symbol_dir / "train.parquet"
            if train_file.exists():
                import pandas as pd
                df = pd.read_parquet(train_file)
                print(f"  {symbol}: {len(df):,} samples")

    # IB Gateway Status
    print("\n[IB GATEWAY]")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 4004))
        if result == 0:
            print("  Status: Connected (localhost:4004)")
        else:
            print("  Status: Not connected")
        sock.close()
    except Exception as e:
        print(f"  Status: Error - {e}")

    print("\n" + "="*60 + "\n")


def cmd_data(args):
    """Data management commands."""
    if args.data_cmd == 'download':
        logger.info(f"Downloading data from {args.source}")
        logger.info(f"Days: {args.days}")

        if args.source == 'truefx':
            from scripts.download_truefx_data import download
            download(days=args.days)
        elif args.source == 'dukascopy':
            from scripts.download_dukascopy import download
            download(days=args.days)

    elif args.data_cmd == 'prepare':
        logger.info("Preparing training data...")
        from scripts.prepare_hft_training import main as prepare_main
        prepare_main()

    elif args.data_cmd == 'status':
        # Show data status
        data_dir = PROJECT_ROOT / "data"
        print(f"\nData directory: {data_dir}")
        if data_dir.exists():
            for item in sorted(data_dir.iterdir())[:10]:
                print(f"  {item.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Forex HFT Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py trade --mode paper --symbols EURUSD,GBPUSD
  python main.py train
  python main.py status
  python main.py data download --source truefx --days 7
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Start trading bot')
    trade_parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                              help='Trading mode (default: paper)')
    trade_parser.add_argument('--symbols', default='EURUSD,GBPUSD,USDJPY',
                              help='Comma-separated symbols')
    trade_parser.add_argument('--host', default='localhost',
                              help='IB Gateway host')
    trade_parser.add_argument('--port', type=int, default=4004,
                              help='IB Gateway port')
    trade_parser.set_defaults(func=cmd_trade)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models locally')
    train_parser.add_argument('--symbols', default='EURUSD,GBPUSD,USDJPY',
                              help='Comma-separated symbols')
    train_parser.add_argument('--target', default='direction_10',
                              help='Target variable')
    train_parser.set_defaults(func=cmd_train)

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--strategy', default='ml_ensemble',
                                 help='Strategy name')
    backtest_parser.add_argument('--days', type=int, default=30,
                                 help='Days to backtest')
    backtest_parser.add_argument('--symbols', help='Comma-separated symbols')
    backtest_parser.set_defaults(func=cmd_backtest)

    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    status_parser.set_defaults(func=cmd_status)

    # Data command
    data_parser = subparsers.add_parser('data', help='Data management')
    data_subparsers = data_parser.add_subparsers(dest='data_cmd')

    download_parser = data_subparsers.add_parser('download', help='Download data')
    download_parser.add_argument('--source', choices=['truefx', 'dukascopy'],
                                 default='truefx', help='Data source')
    download_parser.add_argument('--days', type=int, default=7,
                                 help='Days to download')

    prepare_parser = data_subparsers.add_parser('prepare', help='Prepare training data')
    status_data_parser = data_subparsers.add_parser('status', help='Show data status')

    data_parser.set_defaults(func=cmd_data)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
