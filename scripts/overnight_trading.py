#!/usr/bin/env python3
"""
Overnight Trading Bot with Full Logging
========================================
Runs the HFT bot overnight with comprehensive CSV logging for analysis.

Goal: $100 → $30,000+ (explosive compounding growth)

Logs:
- logs/trades_log.csv - All executed trades
- logs/signals_log.csv - All ML signals generated
- logs/performance_log.csv - Periodic performance snapshots
- logs/overnight_summary.json - Final summary

Usage:
    python scripts/overnight_trading.py
"""

import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'overnight_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CSV files for spreadsheet analysis
TRADES_CSV = LOG_DIR / 'trades_log.csv'
SIGNALS_CSV = LOG_DIR / 'signals_log.csv'
PERFORMANCE_CSV = LOG_DIR / 'performance_log.csv'
SUMMARY_JSON = LOG_DIR / 'overnight_summary.json'


class TradingLogger:
    """Logs all trading activity to CSV files for spreadsheet analysis."""

    def __init__(self):
        self._init_csv_files()
        self.start_time = datetime.now()
        self.starting_balance = 100.0
        self.trades = []
        self.signals = []

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Trades log
        if not TRADES_CSV.exists() or TRADES_CSV.stat().st_size == 0:
            with open(TRADES_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'confidence', 'pnl', 'balance', 'source', 'model_votes'
                ])

        # Signals log
        if not SIGNALS_CSV.exists() or SIGNALS_CSV.stat().st_size == 0:
            with open(SIGNALS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'direction', 'confidence',
                    'predicted_return', 'xgb_vote', 'lgb_vote', 'cb_vote', 'action'
                ])

        # Performance log
        if not PERFORMANCE_CSV.exists() or PERFORMANCE_CSV.stat().st_size == 0:
            with open(PERFORMANCE_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'balance', 'equity', 'pnl', 'pnl_pct',
                    'trades_count', 'win_rate', 'positions_open', 'drawdown_pct'
                ])

    def log_trade(self, symbol, side, quantity, price, confidence, pnl, balance, source='IB', model_votes=None):
        """Log a trade to CSV."""
        row = [
            datetime.now().isoformat(),
            symbol,
            side,
            quantity,
            f"{price:.5f}",
            f"{confidence:.4f}",
            f"{pnl:.2f}",
            f"{balance:.2f}",
            source,
            json.dumps(model_votes) if model_votes else '{}'
        ]

        with open(TRADES_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.trades.append(row)
        logger.info(f"TRADE: {side} {quantity} {symbol} @ {price:.5f} | PnL: ${pnl:.2f} | Balance: ${balance:.2f}")

    def log_signal(self, symbol, direction, confidence, predicted_return, model_votes, action='SKIP'):
        """Log a signal to CSV."""
        row = [
            datetime.now().isoformat(),
            symbol,
            direction,
            f"{confidence:.4f}",
            f"{predicted_return:.4f}",
            model_votes.get('xgboost', 0),
            model_votes.get('lightgbm', 0),
            model_votes.get('catboost', 0),
            action
        ]

        with open(SIGNALS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.signals.append(row)

    def log_performance(self, balance, equity, trades_count, win_rate, positions_open, drawdown_pct):
        """Log performance snapshot."""
        pnl = balance - self.starting_balance
        pnl_pct = (pnl / self.starting_balance) * 100

        row = [
            datetime.now().isoformat(),
            f"{balance:.2f}",
            f"{equity:.2f}",
            f"{pnl:.2f}",
            f"{pnl_pct:.2f}",
            trades_count,
            f"{win_rate:.2f}",
            positions_open,
            f"{drawdown_pct:.2f}"
        ]

        with open(PERFORMANCE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_summary(self, final_balance, total_trades, winning_trades):
        """Save final overnight summary."""
        duration = datetime.now() - self.start_time
        pnl = final_balance - self.starting_balance
        pnl_pct = (pnl / self.starting_balance) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_hours': duration.total_seconds() / 3600,
            'starting_balance': self.starting_balance,
            'final_balance': final_balance,
            'pnl_dollars': pnl,
            'pnl_percent': pnl_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_signals': len(self.signals),
            'goal_progress': f"${self.starting_balance} → ${final_balance:.2f} (Target: $30,000)",
            'multiplier': final_balance / self.starting_balance if self.starting_balance > 0 else 0
        }

        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("OVERNIGHT TRADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Starting Balance: ${self.starting_balance:.2f}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Goal Progress: ${self.starting_balance} → ${final_balance:.2f} / $30,000")
        logger.info("=" * 60)

        return summary


# Import the trading bot
from scripts.hft_trading_bot import HFTTradingBot, TradingMode

class OvernightTradingBot(HFTTradingBot):
    """Extended trading bot with CSV logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = TradingLogger()
        self.winning_trades = 0
        self.total_trades_count = 0
        self.last_performance_log = datetime.now()

    async def _execute_signal(self, signal, bid, ask):
        """Override to add logging."""
        # Get balance before trade
        balance_before = self.risk_manager.account_balance

        # Execute parent method
        await super()._execute_signal(signal, bid, ask)

        # Log the trade if one was made
        if len(self.trades) > self.total_trades_count:
            trade = self.trades[-1]
            balance_after = self.risk_manager.account_balance
            pnl = balance_after - balance_before

            if pnl > 0:
                self.winning_trades += 1

            self.logger.log_trade(
                symbol=trade.symbol,
                side=trade.side.value,
                quantity=trade.quantity,
                price=trade.fill_price,
                confidence=trade.signal_confidence,
                pnl=pnl,
                balance=balance_after,
                source='IB',
                model_votes=signal.model_votes
            )

            self.total_trades_count = len(self.trades)

    async def process_tick(self, symbol, bid, ask, volume=0.0, timestamp=None):
        """Override to add signal logging and periodic performance logging."""
        # Call parent
        await super().process_tick(symbol, bid, ask, volume, timestamp)

        # Log performance every 5 minutes
        if (datetime.now() - self.last_performance_log).seconds >= 300:
            self._log_performance_snapshot()
            self.last_performance_log = datetime.now()

    def _log_performance_snapshot(self):
        """Log current performance."""
        balance = self.risk_manager.account_balance
        win_rate = (self.winning_trades / self.total_trades_count * 100) if self.total_trades_count > 0 else 0
        positions_open = sum(1 for p in self.positions.values() if p.quantity != 0)
        drawdown = self.risk_manager.current_drawdown() * 100

        self.logger.log_performance(
            balance=balance,
            equity=balance,  # Simplified
            trades_count=self.total_trades_count,
            win_rate=win_rate,
            positions_open=positions_open,
            drawdown_pct=drawdown
        )

    async def shutdown(self):
        """Override to save summary."""
        # Save summary before shutdown
        self.logger.save_summary(
            final_balance=self.risk_manager.account_balance,
            total_trades=self.total_trades_count,
            winning_trades=self.winning_trades
        )

        await super().shutdown()


def get_trained_symbols():
    """Get list of symbols with trained models."""
    models_dir = PROJECT_ROOT / 'models' / 'production'
    trained = []
    for f in models_dir.glob('*_models.pkl'):
        symbol = f.stem.replace('_models', '')
        # Skip direction-specific models and focus on main models
        if 'target_direction' not in symbol:
            trained.append(symbol)
    return sorted(trained)


async def main():
    """Run overnight trading session."""
    logger.info("=" * 60)
    logger.info("OVERNIGHT TRADING SESSION STARTING")
    logger.info("=" * 60)
    logger.info(f"Goal: $100 -> $30,000+ (explosive compounding growth)")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # Get trained symbols
    trained_symbols = get_trained_symbols()
    logger.info(f"Found {len(trained_symbols)} trained models: {', '.join(trained_symbols[:10])}...")

    # Create bot with logging
    bot = OvernightTradingBot(
        mode=TradingMode.PAPER,
        symbols=trained_symbols,  # Use all trained symbols
        use_execution_optimizer=False,
        use_online_learning=True
    )

    # Set capital
    bot.risk_manager.starting_capital = 100.0
    bot.risk_manager.account_balance = 100.0
    bot.risk_manager.peak_balance = 100.0
    bot.risk_manager.max_daily_trades = 10000  # Unlimited for overnight

    # Initialize
    await bot.initialize()

    # Run with multi-source data
    logger.info("Starting multi-source data feed...")
    await bot.run_multi_source()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
