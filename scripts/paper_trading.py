"""
Paper Trading System
Reads live ticks and simulates trading in real-time.
"""

import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'paper_trading.log')
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingAccount:
    """Simulated trading account."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}  # pair -> {'side': 'long'/'short', 'entry_price': float, 'size': float}
        self.trade_history = []
        self.equity_curve = []

    def open_position(self, pair: str, side: str, entry_price: float, size: float):
        """Open a new position."""
        if pair in self.positions:
            logger.warning(f"Position already open for {pair}, closing first")
            self.close_position(pair, entry_price)

        self.positions[pair] = {
            'side': side,
            'entry_price': entry_price,
            'size': size,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"OPEN {side.upper()} {pair} @ {entry_price:.5f} size={size:.2f}")

    def close_position(self, pair: str, exit_price: float):
        """Close an existing position."""
        if pair not in self.positions:
            logger.warning(f"No position to close for {pair}")
            return

        pos = self.positions[pair]

        # Calculate P&L
        if pos['side'] == 'long':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:  # short
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        self.balance += pnl

        # Record trade
        trade = {
            'pair': pair,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'size': pos['size'],
            'pnl': pnl,
            'entry_time': pos['timestamp'],
            'exit_time': datetime.now(timezone.utc).isoformat()
        }
        self.trade_history.append(trade)

        logger.info(f"CLOSE {pos['side'].upper()} {pair} @ {exit_price:.5f} PnL=${pnl:.2f} Balance=${self.balance:.2f}")

        # Remove position
        del self.positions[pair]

    def get_status(self) -> dict:
        """Get account status."""
        total_pnl = self.balance - self.initial_balance
        return {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_pnl': total_pnl,
            'pnl_pct': (total_pnl / self.initial_balance) * 100,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'win_trades': sum(1 for t in self.trade_history if t['pnl'] > 0),
            'lose_trades': sum(1 for t in self.trade_history if t['pnl'] < 0)
        }


class SimpleStrategy:
    """Simple example strategy - replace with your ML models."""

    def __init__(self, pair: str, lookback: int = 10):
        self.pair = pair
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)

    def update(self, tick: dict):
        """Update with new tick data."""
        mid_price = (tick['bid'] + tick['ask']) / 2
        self.price_history.append(mid_price)

    def get_signal(self) -> str:
        """Get trading signal: 'long', 'short', or 'neutral'."""
        if len(self.price_history) < self.lookback:
            return 'neutral'

        # Simple moving average crossover example
        recent_avg = sum(list(self.price_history)[-5:]) / 5
        older_avg = sum(list(self.price_history)[-10:]) / 10

        if recent_avg > older_avg * 1.0001:  # 0.01% above
            return 'long'
        elif recent_avg < older_avg * 0.9999:  # 0.01% below
            return 'short'
        else:
            return 'neutral'


class PaperTradingBot:
    """Main paper trading bot."""

    def __init__(self, capture_instance, pairs: list = None):
        self.capture = capture_instance
        self.pairs = pairs or ['EURUSD', 'GBPUSD', 'USDJPY']

        self.account = PaperTradingAccount(initial_balance=10000.0)
        self.strategies = {pair: SimpleStrategy(pair) for pair in self.pairs}

        self.running = False

    def process_tick(self, pair: str, tick: dict):
        """Process a single tick."""
        if pair not in self.strategies:
            return

        # Update strategy
        strategy = self.strategies[pair]
        strategy.update(tick)

        # Get signal
        signal = strategy.get_signal()

        # Execute trades based on signal
        if signal == 'long' and pair not in self.account.positions:
            # Open long position
            self.account.open_position(pair, 'long', tick['ask'], size=1000.0)

        elif signal == 'short' and pair not in self.account.positions:
            # Open short position
            self.account.open_position(pair, 'short', tick['bid'], size=1000.0)

        elif signal == 'neutral' and pair in self.account.positions:
            # Close position
            pos = self.account.positions[pair]
            if pos['side'] == 'long':
                self.account.close_position(pair, tick['bid'])
            else:
                self.account.close_position(pair, tick['ask'])

    def run(self):
        """Main trading loop."""
        self.running = True
        logger.info(f"Starting paper trading bot for {self.pairs}")
        logger.info(f"Initial balance: ${self.account.initial_balance:.2f}")
        logger.info("Press Ctrl+C to stop")

        last_status_time = time.time()

        try:
            while self.running:
                # Get latest ticks
                all_ticks = self.capture.get_all_latest_ticks()

                # Process each pair
                for pair in self.pairs:
                    if pair in all_ticks:
                        tick = all_ticks[pair]
                        self.process_tick(pair, tick)

                # Print status every 60 seconds
                if time.time() - last_status_time > 60:
                    status = self.account.get_status()
                    logger.info("=== ACCOUNT STATUS ===")
                    logger.info(f"Balance: ${status['balance']:.2f}")
                    logger.info(f"Total PnL: ${status['total_pnl']:.2f} ({status['pnl_pct']:.2f}%)")
                    logger.info(f"Open positions: {status['open_positions']}")
                    logger.info(f"Trades: {status['total_trades']} (W:{status['win_trades']} L:{status['lose_trades']})")

                    if status['total_trades'] > 0:
                        win_rate = (status['win_trades'] / status['total_trades']) * 100
                        logger.info(f"Win rate: {win_rate:.1f}%")

                    last_status_time = time.time()

                # Sleep briefly to avoid busy loop
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Stopping paper trading bot...")
            self.stop()

    def stop(self):
        """Stop the bot."""
        self.running = False

        # Close all open positions
        all_ticks = self.capture.get_all_latest_ticks()
        for pair in list(self.account.positions.keys()):
            if pair in all_ticks:
                tick = all_ticks[pair]
                pos = self.account.positions[pair]
                if pos['side'] == 'long':
                    self.account.close_position(pair, tick['bid'])
                else:
                    self.account.close_position(pair, tick['ask'])

        # Final status
        status = self.account.get_status()
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Initial balance: ${self.account.initial_balance:.2f}")
        logger.info(f"Final balance: ${status['balance']:.2f}")
        logger.info(f"Total PnL: ${status['total_pnl']:.2f} ({status['pnl_pct']:.2f}%)")
        logger.info(f"Total trades: {status['total_trades']}")

        if status['total_trades'] > 0:
            win_rate = (status['win_trades'] / status['total_trades']) * 100
            avg_win = sum(t['pnl'] for t in self.account.trade_history if t['pnl'] > 0) / max(status['win_trades'], 1)
            avg_loss = sum(t['pnl'] for t in self.account.trade_history if t['pnl'] < 0) / max(status['lose_trades'], 1)

            logger.info(f"Win rate: {win_rate:.1f}%")
            logger.info(f"Avg win: ${avg_win:.2f}")
            logger.info(f"Avg loss: ${avg_loss:.2f}")


def main():
    """Main entry point."""
    # Import the capture instance
    from truefx_live_capture_local import get_capture_instance

    capture = get_capture_instance()

    if capture is None:
        logger.error("Live capture not running! Start truefx_live_capture_local.py first.")
        return

    # Create paper trading bot
    bot = PaperTradingBot(
        capture,
        pairs=['EURUSD', 'GBPUSD', 'USDJPY']  # Start with major pairs
    )

    # Run bot
    bot.run()


if __name__ == "__main__":
    main()
