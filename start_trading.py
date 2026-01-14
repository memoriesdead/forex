"""
FOREX TRADING SYSTEM - UNIFIED ENTRY POINT
Production-ready trading for 6:30am PST and 9pm PST sessions

Usage:
    python start_trading.py --mode paper --session auto
    python start_trading.py --mode live --session morning --broker ib
    python start_trading.py --mode live --session evening --broker oanda
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pytz

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_config():
    """Load all configuration files"""
    config_dir = project_root / "config"

    with open(config_dir / "trading_sessions.json") as f:
        sessions_config = json.load(f)

    return {
        'sessions': sessions_config,
        'timezone': pytz.timezone(sessions_config.get('timezone', 'America/Los_Angeles'))
    }


def detect_current_session(config):
    """Auto-detect current trading session"""
    now = datetime.now(config['timezone'])
    current_time = now.time()

    for session_name, session_data in config['sessions']['sessions'].items():
        if session_name == 'off_hours':
            continue

        start = datetime.strptime(session_data['time_start'], '%H:%M').time()
        end = datetime.strptime(session_data['time_end'], '%H:%M').time()

        if start <= current_time <= end:
            return session_name

    return 'off_hours'


def print_banner(mode, session, broker, config):
    """Print startup banner"""
    print("="*70)
    print("FOREX TRADING SYSTEM")
    print("="*70)
    print(f"Mode:     {mode.upper()}")
    print(f"Broker:   {broker.upper()}")
    print(f"Session:  {session.upper()}")

    now = datetime.now(config['timezone'])
    print(f"Time:     {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if session != 'off_hours':
        session_data = config['sessions']['sessions'][session]
        pairs = session_data.get('pairs', [])
        strategy = session_data.get('strategy', 'N/A')

        print(f"Strategy: {strategy}")
        print(f"Pairs:    {', '.join(pairs)}")
    else:
        print("Status:   OFF HOURS - Not recommended for trading")
        print("Next:     9pm PST (Evening) or 6:30am PST (Morning)")

    print("="*70)
    print()


def start_paper_trading(session, broker, config):
    """Start paper trading"""
    print("[INFO] Starting paper trading...")
    print("[INFO] This will use demo/practice account (no real money)")

    if broker == 'ib':
        # Use IB paper trading bot
        from scripts.ib_paper_trading_bot import IBTradingBot

        bot = IBTradingBot(port=7497, client_id=2)  # TWS paper trading port
        bot.connect()

        if session == 'auto':
            session = detect_current_session(config)

        print(f"[INFO] Session: {session}")

        # Run test mode for now
        bot.run_simple_test()
        bot.disconnect()

    elif broker == 'oanda':
        # Use OANDA paper trading bot
        from scripts.session_aware_trading_bot import SessionAwareTradingBot

        print("[INFO] Using session-aware momentum strategy")
        print("[INFO] Starting bot...")

        # This will run continuously
        import subprocess
        subprocess.run([
            sys.executable,
            str(project_root / "scripts" / "session_aware_trading_bot.py")
        ])

    else:
        print(f"[ERROR] Unknown broker: {broker}")
        return False

    return True


def start_live_trading(session, broker, config):
    """Start live trading (REAL MONEY)"""
    print("="*70)
    print("WARNING: LIVE TRADING MODE")
    print("="*70)
    print("This will trade with REAL MONEY on your account.")
    print("Make sure you understand the risks.")
    print()

    confirm = input("Type 'START LIVE TRADING' to confirm: ").strip()

    if confirm != "START LIVE TRADING":
        print("[CANCELLED] Live trading not started")
        return False

    print()
    print("[INFO] Starting live trading...")

    if broker == 'ib':
        # Use IB live trading
        from scripts.ib_paper_trading_bot import IBTradingBot

        bot = IBTradingBot(port=7497)  # Live trading port
        bot.connect()

        if session == 'auto':
            session = detect_current_session(config)

        print(f"[INFO] Session: {session}")

        # Get account summary
        balance, positions = bot.get_account_summary()

        print(f"[INFO] Account Balance: ${balance:,.2f}")
        print(f"[INFO] Open Positions: {len(positions)}")

        # Run in live mode (not yet implemented fully)
        print("[WARNING] Full live trading not yet implemented")
        print("[INFO] Use --mode paper to test first")

        bot.disconnect()

    elif broker == 'oanda':
        print("[ERROR] OANDA live trading not yet implemented")
        print("[INFO] Use --mode paper to test with OANDA practice account")
        return False

    else:
        print(f"[ERROR] Unknown broker: {broker}")
        return False

    return True


def check_prerequisites():
    """Check if system is ready"""
    print("[CHECK] Verifying system prerequisites...")

    issues = []

    # Check config files
    config_dir = project_root / "config"
    if not (config_dir / "trading_sessions.json").exists():
        issues.append("Missing config/trading_sessions.json")

    # Check .env file
    if not (project_root / ".env").exists():
        issues.append("Missing .env file (copy from .env.example)")

    # Check data directory
    data_dir = project_root / "data" / "live"
    if not data_dir.exists():
        issues.append("Missing data/live directory (run data sync first)")

    # Check dependencies
    try:
        import ib_insync
    except ImportError:
        issues.append("Missing ib_insync (pip install ib_insync)")

    try:
        import pytz
    except ImportError:
        issues.append("Missing pytz (pip install pytz)")

    if issues:
        print("[ERROR] System not ready:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("Fix these issues and try again.")
        return False

    print("[OK] All prerequisites met")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Forex Trading System - Unified Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (auto-detect session)
  python start_trading.py --mode paper --session auto --broker ib

  # Paper trading (specific session)
  python start_trading.py --mode paper --session morning --broker oanda

  # Live trading (real money)
  python start_trading.py --mode live --session auto --broker ib
        """
    )

    parser.add_argument('--mode',
                       choices=['paper', 'live'],
                       required=True,
                       help='Trading mode (paper=demo, live=real money)')

    parser.add_argument('--session',
                       choices=['auto', 'morning', 'evening', 'off_hours'],
                       default='auto',
                       help='Trading session (auto=detect current)')

    parser.add_argument('--broker',
                       choices=['ib', 'oanda'],
                       default='ib',
                       help='Broker to use (ib=Interactive Brokers, oanda=OANDA)')

    parser.add_argument('--skip-checks',
                       action='store_true',
                       help='Skip prerequisite checks')

    args = parser.parse_args()

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            sys.exit(1)

    # Load config
    config = load_config()

    # Auto-detect session if needed
    session = args.session
    if session == 'auto':
        session = detect_current_session(config)

    # Print banner
    print_banner(args.mode, session, args.broker, config)

    # Start trading
    if args.mode == 'paper':
        success = start_paper_trading(session, args.broker, config)
    else:
        success = start_live_trading(session, args.broker, config)

    if success:
        print()
        print("[INFO] Trading session ended")
    else:
        print()
        print("[ERROR] Trading failed to start")
        sys.exit(1)


if __name__ == "__main__":
    main()
