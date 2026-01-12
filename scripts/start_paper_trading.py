"""
Start Paper Trading System
Launches live tick capture and paper trading bot together.
"""

import threading
import time
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_live_capture():
    """Start live tick capture in background thread."""
    from truefx_live_capture_local import LiveTickCapture, _capture_instance

    # Create local output directory
    output_dir = Path(__file__).parent.parent / "data" / "truefx_live"

    # Create and start capture
    capture = LiveTickCapture(output_dir)

    # Make it globally accessible
    import truefx_live_capture_local
    truefx_live_capture_local._capture_instance = capture

    logger.info("Starting live tick capture...")
    capture.start()


def start_paper_trading():
    """Start paper trading bot."""
    from paper_trading import PaperTradingBot
    from truefx_live_capture_local import get_capture_instance

    # Wait for capture to initialize
    logger.info("Waiting for live capture to start...")
    time.sleep(5)

    capture = get_capture_instance()

    if capture is None:
        logger.error("Live capture not available!")
        return

    # Create paper trading bot
    logger.info("Starting paper trading bot...")
    bot = PaperTradingBot(
        capture,
        pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    )

    # Run bot
    bot.run()


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("FOREX PAPER TRADING SYSTEM")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This will:")
    logger.info("1. Start live tick capture from TrueFX (every 1 second)")
    logger.info("2. Start paper trading bot (processes ticks in real-time)")
    logger.info("")
    logger.info("Initial balance: $10,000")
    logger.info("Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    logger.info("")

    # Start live capture in background thread
    capture_thread = threading.Thread(target=start_live_capture, daemon=True)
    capture_thread.start()

    # Give it a moment to start
    time.sleep(3)

    # Start paper trading in main thread
    try:
        start_paper_trading()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
