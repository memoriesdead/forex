"""
Session-Aware Paper Trading Bot

Automatically selects optimal pairs and parameters based on current trading session
- 6:30am PST: London+NY overlap (EUR/USD, GBP/USD focus)
- 9pm PST: Tokyo session (USD/JPY, AUD/USD focus)

Usage:
    python scripts/session_aware_trading_bot.py [--vastai-endpoint http://IP:5000]
"""

import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from dotenv import load_dotenv
import pytz

load_dotenv()

# Import base classes from existing bot
import sys
sys.path.append(str(Path(__file__).parent))

# Configuration
OANDA_PRACTICE_KEY = os.getenv('OANDA_PRACTICE_API_KEY')
OANDA_PRACTICE_ACCOUNT = os.getenv('OANDA_PRACTICE_ACCOUNT_ID')
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"

# Paths
DATA_DIR = Path("data/live/latest")
LOG_DIR = Path("logs")
CONFIG_DIR = Path("config")
LOG_DIR.mkdir(exist_ok=True)

# Load session configuration
SESSION_CONFIG_FILE = CONFIG_DIR / "trading_sessions.json"
with open(SESSION_CONFIG_FILE, 'r') as f:
    SESSION_CONFIG = json.load(f)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "session_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SessionManager:
    """Manages trading sessions and determines current active session"""

    def __init__(self, timezone_str: str = "America/Los_Angeles"):
        self.timezone = pytz.timezone(timezone_str)
        self.sessions = SESSION_CONFIG['sessions']

    def get_current_session(self) -> Tuple[str, Dict]:
        """
        Get current trading session based on time
        Returns: (session_name, session_config)
        """
        now = datetime.now(self.timezone)
        current_time = now.strftime('%H:%M')

        logger.debug(f"Current time: {current_time} {self.timezone}")

        # Check each session
        for session_name, session_config in self.sessions.items():
            if session_name == 'off_hours':
                continue  # Check this last

            start_time = session_config['time_start']
            end_time = session_config['time_end']

            # Handle sessions that cross midnight
            if end_time < start_time:
                if current_time >= start_time or current_time <= end_time:
                    return session_name, session_config
            else:
                if start_time <= current_time <= end_time:
                    return session_name, session_config

        # Default to off_hours
        return 'off_hours', self.sessions['off_hours']

    def get_active_pairs(self) -> List[str]:
        """Get pairs that should be traded in current session"""
        session_name, session_config = self.get_current_session()
        return session_config['pairs']

    def get_session_params(self) -> Dict:
        """Get trading parameters for current session"""
        session_name, session_config = self.get_current_session()
        return {
            'session_name': session_name,
            'description': session_config['description'],
            'pairs': session_config['pairs'],
            'min_confidence': session_config['min_confidence'],
            'position_size_multiplier': session_config['position_size_multiplier'],
            'max_risk_per_trade_pct': session_config['max_risk_per_trade_pct'],
            'max_daily_risk_pct': session_config['max_daily_risk_pct'],
            'stop_loss_pips': session_config['stop_loss_pips'],
            'take_profit_pips': session_config['take_profit_pips'],
            'trading_style': session_config['trading_style'],
            'strategy': session_config['strategy']
        }

    def should_trade_now(self) -> bool:
        """Check if current time is within active trading sessions"""
        session_name, _ = self.get_current_session()
        return session_name != 'off_hours'

    def get_pair_metadata(self, pair: str) -> Dict:
        """Get metadata for specific pair"""
        return SESSION_CONFIG['pair_metadata'].get(pair, {})


# Import OANDA client and other classes from base bot
from paper_trading_bot import (
    OANDAClient,
    FeatureEngine,
    VastAIInference,
    RiskManager
)


class SessionAwareBot:
    """Trading bot with session awareness"""

    def __init__(self, vastai_endpoint: Optional[str] = None):
        if not OANDA_PRACTICE_KEY or OANDA_PRACTICE_KEY == 'your_practice_key_here':
            raise ValueError("OANDA Practice API key not configured in .env")

        self.oanda = OANDAClient(OANDA_PRACTICE_KEY, OANDA_PRACTICE_ACCOUNT)
        self.feature_engine = FeatureEngine()
        self.vastai = VastAIInference(vastai_endpoint) if vastai_endpoint else None
        self.session_manager = SessionManager()
        self.risk_manager = None

        # State
        self.positions = {}
        self.is_running = False
        self.current_session_params = None

    def initialize(self):
        """Initialize bot"""
        logger.info("="*60)
        logger.info("SESSION-AWARE PAPER TRADING BOT")
        logger.info("="*60)

        try:
            account = self.oanda.get_account_summary()
            balance = float(account['balance'])
            logger.info(f"Account balance: ${balance:,.2f}")

            self.risk_manager = RiskManager(balance)

            # Log current session
            session_params = self.session_manager.get_session_params()
            logger.info(f"\nCurrent Session: {session_params['session_name']}")
            logger.info(f"Description: {session_params['description']}")
            logger.info(f"Active Pairs: {', '.join(session_params['pairs'])}")
            logger.info(f"Strategy: {session_params['strategy']}")
            logger.info(f"Min Confidence: {session_params['min_confidence']}")

            if not self.session_manager.should_trade_now():
                logger.warning("\n⚠️  OFF HOURS - Not recommended to trade now")
                logger.warning("Best trading times:")
                logger.warning("  - 6:30am-9am PST (London+NY overlap)")
                logger.warning("  - 9pm-11pm PST (Tokyo session)")

            logger.info("="*60)
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def sync_latest_data(self):
        """Sync latest data from Oracle Cloud"""
        import subprocess
        try:
            subprocess.run(
                ["python", "scripts/sync_live_data_v2.py", "--latest", "1m"],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Data sync failed: {e}")

    def update_session_if_changed(self):
        """Check if session has changed and update parameters"""
        new_params = self.session_manager.get_session_params()

        if self.current_session_params is None or \
           self.current_session_params['session_name'] != new_params['session_name']:

            logger.info("\n" + "="*60)
            logger.info(f"SESSION CHANGE: {new_params['session_name']}")
            logger.info(f"Description: {new_params['description']}")
            logger.info(f"Active Pairs: {', '.join(new_params['pairs'])}")
            logger.info(f"Strategy: {new_params['strategy']}")
            logger.info("="*60 + "\n")

            self.current_session_params = new_params

            # Close positions not in new session
            self.close_out_of_session_positions()

    def close_out_of_session_positions(self):
        """Close positions for pairs not active in current session"""
        active_pairs = set(self.session_manager.get_active_pairs())

        for oanda_pair, units in list(self.positions.items()):
            if units != 0 and oanda_pair not in active_pairs:
                try:
                    self.oanda.close_position(oanda_pair)
                    logger.info(f"Closed {oanda_pair} (not active in current session)")
                    self.positions[oanda_pair] = 0
                except Exception as e:
                    logger.error(f"Failed to close {oanda_pair}: {e}")

    def process_pair(self, pair: str):
        """Process one pair with session-aware parameters"""
        try:
            # Get session parameters
            session_params = self.session_manager.get_session_params()
            min_confidence = session_params['min_confidence']
            position_multiplier = session_params['position_size_multiplier']

            # Load data and calculate features
            df = self.feature_engine.load_latest_data(pair)
            features = self.feature_engine.calculate_features(df)

            # Get prediction
            if self.vastai:
                signal, confidence = self.vastai.predict(pair, features)
            else:
                # Simple strategy based on session
                if session_params['trading_style'] == 'breakout_trend':
                    # Trend following for morning session
                    if features['returns_5'] > 0.0001 and features['ma_5'] > features['ma_20']:
                        signal, confidence = 'long', 0.6
                    elif features['returns_5'] < -0.0001 and features['ma_5'] < features['ma_20']:
                        signal, confidence = 'short', 0.6
                    else:
                        signal, confidence = 'neutral', 0.0
                else:
                    # Range trading for evening session
                    if features['returns_1'] < -0.0002:  # Oversold
                        signal, confidence = 'long', 0.6
                    elif features['returns_1'] > 0.0002:  # Overbought
                        signal, confidence = 'short', 0.6
                    else:
                        signal, confidence = 'neutral', 0.0

            logger.info(f"{pair}: mid={features['mid_price']:.5f}, signal={signal}, conf={confidence:.2f}")

            # Risk checks
            if self.risk_manager.check_daily_loss_limit():
                return

            # Position management
            current_position = self.positions.get(pair, 0)

            # Entry logic
            if signal == 'long' and confidence >= min_confidence:
                if current_position <= 0:
                    units = self.risk_manager.calculate_position_size(confidence)
                    units = int(units * position_multiplier)  # Apply session multiplier

                    if units > 0:
                        if current_position < 0:
                            self.oanda.close_position(pair)

                        result = self.oanda.place_market_order(pair, units)
                        logger.info(f"{pair}: OPENED LONG {units} units")
                        self.positions[pair] = units

            elif signal == 'short' and confidence >= min_confidence:
                if current_position >= 0:
                    units = self.risk_manager.calculate_position_size(confidence)
                    units = int(units * position_multiplier)

                    if units > 0:
                        if current_position > 0:
                            self.oanda.close_position(pair)

                        result = self.oanda.place_market_order(pair, -units)
                        logger.info(f"{pair}: OPENED SHORT {units} units")
                        self.positions[pair] = -units

            elif signal == 'neutral' and current_position != 0:
                self.oanda.close_position(pair)
                logger.info(f"{pair}: CLOSED POSITION")
                self.positions[pair] = 0

        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")

    def run(self, interval_seconds: int = 5):
        """Main trading loop"""
        self.is_running = True
        logger.info(f"Starting session-aware trading loop (interval={interval_seconds}s)")

        try:
            while self.is_running:
                loop_start = time.time()

                # Check for session changes
                self.update_session_if_changed()

                # Skip trading if off hours
                if not self.session_manager.should_trade_now():
                    logger.info("Off hours - waiting for active session...")
                    time.sleep(60)  # Check every minute
                    continue

                # Sync data
                self.sync_latest_data()

                # Get active pairs for current session
                active_pairs = self.session_manager.get_active_pairs()

                if not active_pairs:
                    logger.warning("No active pairs for current session")
                    time.sleep(interval_seconds)
                    continue

                # Process each active pair
                for pair in active_pairs:
                    self.process_pair(pair)

                # Update account status
                account = self.oanda.get_account_summary()
                balance = float(account['balance'])
                unrealized = float(account.get('unrealizedPL', 0))

                session_name = self.session_manager.get_session_params()['session_name']
                logger.info(f"[{session_name}] Balance: ${balance:,.2f}, P/L: ${unrealized:,.2f}")

                # Sleep
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Stopping bot (user interrupt)")
        finally:
            self.is_running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Session-Aware Paper Trading Bot')
    parser.add_argument('--vastai-endpoint', help='Vast.ai model server endpoint')
    parser.add_argument('--interval', type=int, default=5, help='Trading interval in seconds')

    args = parser.parse_args()

    bot = SessionAwareBot(vastai_endpoint=args.vastai_endpoint)

    if not bot.initialize():
        return 1

    try:
        bot.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        logger.info("Session-aware trading bot stopped")

    return 0


if __name__ == "__main__":
    exit(main())
