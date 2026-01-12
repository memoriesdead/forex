"""
24/7 Forex Trading Daemon
Session-aware automated trading with ML models and risk management

Features:
- Connects to Interactive Brokers API
- Session-aware trading (morning, afternoon, evening, overnight)
- ML model ensemble predictions
- Risk management (1% per trade, 3% daily max)
- Runs as background daemon

Usage:
    python scripts/trading_daemon.py --mode paper     # Paper trading (default)
    python scripts/trading_daemon.py --mode live      # Live trading
    python scripts/trading_daemon.py --status         # Check status
"""

import os
import sys
import json
import time
import pickle
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ib_insync import IB, Forex, MarketOrder, util
except ImportError:
    try:
        from ib_async import IB, Forex, MarketOrder, util  # New library (replaces ib_insync)
    except ImportError:
        print("ERROR: ib_async not installed. Run: pip install ib_async")
        sys.exit(1)

# Configuration
CONFIG_DIR = PROJECT_ROOT / "config"
# Institutional-grade models (HMM + Kalman + XGBoost/LightGBM/CatBoost)
MODELS_DIR = PROJECT_ROOT / "models" / "institutional"
# Fallback to old models if institutional not available
LEGACY_MODELS_DIR = PROJECT_ROOT / "trading" / "supreme_system" / "models_h100"
LOG_DIR = PROJECT_ROOT / "logs"
PID_FILE = PROJECT_ROOT / "trading_daemon.pid"

LOG_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "trading_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RiskManager:
    """Manages position sizing and daily risk limits"""

    def __init__(self, initial_balance: float,
                 max_risk_per_trade_pct: float = 1.0,
                 max_daily_risk_pct: float = 3.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_risk_pct = max_daily_risk_pct
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()

    def update_balance(self, new_balance: float):
        """Update current balance and calculate daily P&L"""
        today = datetime.now().date()
        if today != self.last_reset:
            # Reset daily P&L at start of new day
            self.initial_balance = new_balance
            self.daily_pnl = 0.0
            self.last_reset = today
            logger.info(f"New trading day - reset daily P&L. Balance: ${new_balance:,.2f}")

        self.daily_pnl = new_balance - self.initial_balance
        self.current_balance = new_balance

    def check_daily_loss_limit(self) -> bool:
        """Returns True if daily loss limit exceeded"""
        max_loss = self.initial_balance * (self.max_daily_risk_pct / 100)
        if self.daily_pnl < -max_loss:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:,.2f} (max: ${-max_loss:,.2f})")
            return True
        return False

    def calculate_position_size(self, pip_risk: float, pair: str) -> int:
        """Calculate position size based on risk per trade"""
        max_risk = self.current_balance * (self.max_risk_per_trade_pct / 100)

        # Pip value calculation (simplified)
        # For standard lot: 1 pip = $10 for most pairs, $8.33 for JPY pairs
        if 'JPY' in pair:
            pip_value_per_lot = 8.33
        else:
            pip_value_per_lot = 10.0

        # Calculate lot size
        if pip_risk > 0:
            lot_size = max_risk / (pip_risk * pip_value_per_lot)
            # Convert to units (1 lot = 100,000 units)
            units = int(lot_size * 100000)
            # Minimum 1000 units, max 100000
            units = max(1000, min(units, 100000))
            return units
        return 1000

    def get_status(self) -> Dict:
        """Get current risk status"""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            'daily_limit_reached': self.check_daily_loss_limit()
        }


class SessionManager:
    """Manages trading sessions and determines current active session"""

    def __init__(self, config_file: Path):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.timezone = pytz.timezone(self.config.get('timezone', 'America/Los_Angeles'))
        self.sessions = self.config['sessions']

    def get_current_session(self) -> Tuple[str, Dict]:
        """Get current trading session based on time"""
        now = datetime.now(self.timezone)
        current_time = now.strftime('%H:%M')

        for session_name, session_config in self.sessions.items():
            if session_name == 'off_hours':
                continue

            start_time = session_config['time_start']
            end_time = session_config['time_end']

            # Handle sessions that cross midnight
            if end_time < start_time:
                if current_time >= start_time or current_time <= end_time:
                    return session_name, session_config
            else:
                if start_time <= current_time <= end_time:
                    return session_name, session_config

        return 'off_hours', self.sessions['off_hours']

    def should_trade(self) -> bool:
        """Check if current time is within active trading sessions"""
        session_name, session_config = self.get_current_session()
        return session_config.get('strategy') != 'disabled'

    def get_active_pairs(self) -> List[str]:
        """Get pairs for current session"""
        _, session_config = self.get_current_session()
        return session_config.get('pairs', [])

    def get_session_params(self) -> Dict:
        """Get all parameters for current session"""
        session_name, session_config = self.get_current_session()
        return {
            'name': session_name,
            'pairs': session_config.get('pairs', []),
            'min_confidence': session_config.get('min_confidence', 0.55),
            'position_multiplier': session_config.get('position_size_multiplier', 1.0),
            'stop_loss_pips': session_config.get('stop_loss_pips', 20),
            'take_profit_pips': session_config.get('take_profit_pips', 40),
            'strategy': session_config.get('strategy', 'moderate'),
            'max_risk_per_trade': session_config.get('max_risk_per_trade_pct', 1.0),
            'max_daily_risk': session_config.get('max_daily_risk_pct', 3.0)
        }


class MLPredictor:
    """
    Institutional-grade ML predictor wrapper.

    Uses ACTUAL methods from billion-dollar quants:
    - Hidden Markov Models (Renaissance Technologies) - Regime detection
    - Kalman Filters (Goldman Sachs) - Dynamic parameter estimation
    - XGBoost/LightGBM/CatBoost ensemble (What actually wins)

    NOT using (overfits/doesn't work):
    - Deep Learning / Neural Networks
    - Transformers
    - Deep RL
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.predictor = None
        self.loaded = False

    def load_models(self) -> bool:
        """Load institutional-grade models."""
        try:
            # Try to import institutional predictor
            from core.institutional_predictor import create_predictor

            # Try institutional models first
            if self.models_dir.exists():
                self.predictor = create_predictor(self.models_dir)
                if hasattr(self.predictor, 'loaded') and self.predictor.loaded:
                    logger.info("Loaded institutional-grade models (HMM + Kalman + Ensemble)")
                    self.loaded = True
                    return True

            # Fallback to legacy models
            if LEGACY_MODELS_DIR.exists():
                self.predictor = create_predictor(LEGACY_MODELS_DIR)
                if hasattr(self.predictor, 'loaded') and self.predictor.loaded:
                    logger.info("Loaded legacy models")
                    self.loaded = True
                    return True

            logger.warning("No models loaded - using fallback predictor")
            from core.institutional_predictor import FallbackPredictor
            self.predictor = FallbackPredictor()
            return False

        except ImportError as e:
            logger.warning(f"Could not import institutional predictor: {e}")
            return self._load_legacy_models()

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def _load_legacy_models(self) -> bool:
        """Fallback: Load legacy XGBoost/LightGBM/CatBoost models."""
        try:
            self.models = {}
            self.scaler = None
            self.weights = {}

            model_files = {
                'xgboost': 'hft_xgboost.pkl',
                'lightgbm': 'hft_lightgbm.pkl',
                'catboost': 'hft_catboost.pkl'
            }

            models_dir = LEGACY_MODELS_DIR if LEGACY_MODELS_DIR.exists() else self.models_dir

            for name, filename in model_files.items():
                model_path = models_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    logger.info(f"Loaded legacy model: {name}")

            # Load scaler and weights
            for fname, attr in [('hft_scaler.pkl', 'scaler'), ('hft_weights.pkl', 'weights')]:
                path = models_dir / fname
                if path.exists():
                    with open(path, 'rb') as f:
                        setattr(self, attr, pickle.load(f))

            self.loaded = bool(self.models)
            return self.loaded

        except Exception as e:
            logger.error(f"Legacy model loading failed: {e}")
            return False

    def predict(self, pair: str, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Get prediction using institutional-grade methods.

        Returns:
            Tuple of (signal: 'long'/'short'/'neutral', confidence: 0-1)
        """
        if self.predictor is not None:
            try:
                result = self.predictor.predict(pair, data)
                return result['signal'], result['confidence']
            except Exception as e:
                logger.warning(f"Institutional prediction failed: {e}")

        # Fallback to legacy prediction
        return self._legacy_predict(data)

    def get_full_prediction(self, pair: str, data: pd.DataFrame) -> Dict:
        """
        Get full prediction with regime detection and all details.

        Returns comprehensive prediction including:
        - Signal and confidence
        - Regime information (HMM)
        - Kalman filter estimates
        - Position size multiplier
        """
        if self.predictor is not None:
            try:
                return self.predictor.predict(pair, data)
            except Exception as e:
                logger.warning(f"Full prediction failed: {e}")

        # Fallback
        signal, confidence = self._legacy_predict(data)
        return {
            'signal': signal,
            'confidence': confidence,
            'regime': {'state': 1, 'action': 'normal', 'confidence': 0.5},
            'position_multiplier': 1.0,
            'notes': 'Legacy predictor'
        }

    def _legacy_predict(self, data: pd.DataFrame) -> Tuple[str, float]:
        """Legacy prediction using simple signals."""
        try:
            mid = data['mid'].values if 'mid' in data.columns else data.iloc[:, 0].values

            # Simple MA crossover
            sma_fast = pd.Series(mid).rolling(10).mean().iloc[-1]
            sma_slow = pd.Series(mid).rolling(50).mean().iloc[-1]

            if sma_fast > sma_slow:
                return 'long', 0.52
            elif sma_fast < sma_slow:
                return 'short', 0.52
            else:
                return 'neutral', 0.5
        except:
            return 'neutral', 0.0


class TradingDaemon:
    """Main 24/7 trading daemon"""

    def __init__(self, mode: str = 'paper', port: int = None):
        self.mode = mode
        self.port = port or (7496 if mode == 'paper' else 7497)
        self.ib = IB()
        self.session_manager = SessionManager(CONFIG_DIR / "trading_sessions.json")
        self.risk_manager = None
        self.ml_predictor = MLPredictor(MODELS_DIR)

        self.running = False
        self.positions = {}
        self.last_session = None

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def connect(self) -> bool:
        """Connect to IB Gateway/TWS"""
        try:
            logger.info(f"Connecting to IB on port {self.port}...")
            self.ib.connect('127.0.0.1', self.port, clientId=1)

            if self.ib.isConnected():
                logger.info(f"Connected to IB ({self.mode} mode)")

                # Get account info
                account_values = self.ib.accountValues()
                balance = 0
                for item in account_values:
                    if item.tag == 'NetLiquidation' and item.currency == 'USD':
                        balance = float(item.value)
                        break

                logger.info(f"Account balance: ${balance:,.2f}")

                # Initialize risk manager
                session_params = self.session_manager.get_session_params()
                self.risk_manager = RiskManager(
                    balance,
                    max_risk_per_trade_pct=session_params['max_risk_per_trade'],
                    max_daily_risk_pct=session_params['max_daily_risk']
                )

                return True
            else:
                logger.error("Failed to connect to IB")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.ib.isConnected():
            # Close all positions before disconnect
            self._close_all_positions()
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def _get_market_data(self, pair: str) -> Optional[Dict]:
        """Get current market data for a pair"""
        try:
            ib_symbol = pair.replace('_', '')
            contract = Forex(ib_symbol)

            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return None

            ticker = self.ib.reqMktData(qualified[0], '', False, False)
            self.ib.sleep(2)

            if ticker.bid and ticker.ask:
                return {
                    'pair': pair,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'mid': (ticker.bid + ticker.ask) / 2,
                    'spread': ticker.ask - ticker.bid
                }
            return None

        except Exception as e:
            logger.error(f"Market data error for {pair}: {e}")
            return None

    def _calculate_simple_signal(self, market_data: Dict, session_params: Dict) -> Tuple[str, float]:
        """Calculate simple trading signal when ML models not available"""
        # This is a placeholder - in production, implement proper TA
        return 'neutral', 0.0

    def _place_order(self, pair: str, action: str, units: int) -> bool:
        """Place a market order"""
        try:
            ib_symbol = pair.replace('_', '')
            contract = Forex(ib_symbol)

            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return False

            order = MarketOrder(action, units)
            trade = self.ib.placeOrder(qualified[0], order)

            logger.info(f"ORDER: {action} {units} {ib_symbol}")

            # Wait for fill
            timeout = 10
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == 'Filled':
                logger.info(f"FILLED @ {trade.orderStatus.avgFillPrice}")
                return True
            else:
                logger.warning(f"Order status: {trade.orderStatus.status}")
                return False

        except Exception as e:
            logger.error(f"Order error: {e}")
            return False

    def _close_position(self, pair: str):
        """Close position for a pair"""
        current_pos = self.positions.get(pair, 0)
        if current_pos == 0:
            return

        action = 'SELL' if current_pos > 0 else 'BUY'
        units = abs(current_pos)

        if self._place_order(pair, action, units):
            self.positions[pair] = 0
            logger.info(f"Closed position for {pair}")

    def _close_all_positions(self):
        """Close all open positions"""
        for pair in list(self.positions.keys()):
            if self.positions[pair] != 0:
                self._close_position(pair)

    def _update_account(self):
        """Update account balance and risk status"""
        try:
            account_values = self.ib.accountValues()
            balance = 0
            for item in account_values:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    balance = float(item.value)
                    break

            self.risk_manager.update_balance(balance)

        except Exception as e:
            logger.error(f"Account update error: {e}")

    def _process_pair(self, pair: str, session_params: Dict):
        """Process trading logic for one pair"""
        try:
            # Get market data
            market_data = self._get_market_data(pair)
            if not market_data:
                return

            # Get prediction (ML or simple)
            if self.ml_predictor.models:
                # TODO: Extract features from market data and recent history
                # For now, use simple signal
                signal, confidence = self._calculate_simple_signal(market_data, session_params)
            else:
                signal, confidence = self._calculate_simple_signal(market_data, session_params)

            min_confidence = session_params['min_confidence']
            current_pos = self.positions.get(pair, 0)

            logger.info(f"{pair}: mid={market_data['mid']:.5f}, signal={signal}, conf={confidence:.2f}")

            # Risk checks
            if self.risk_manager.check_daily_loss_limit():
                logger.warning("Daily loss limit - skipping trade")
                return

            # Position sizing
            pip_risk = session_params['stop_loss_pips']
            units = self.risk_manager.calculate_position_size(pip_risk, pair)
            units = int(units * session_params['position_multiplier'])

            # Entry/exit logic
            if signal == 'long' and confidence >= min_confidence:
                if current_pos <= 0:
                    if current_pos < 0:
                        self._close_position(pair)
                    if self._place_order(pair, 'BUY', units):
                        self.positions[pair] = units

            elif signal == 'short' and confidence >= min_confidence:
                if current_pos >= 0:
                    if current_pos > 0:
                        self._close_position(pair)
                    if self._place_order(pair, 'SELL', units):
                        self.positions[pair] = -units

            elif signal == 'neutral' and current_pos != 0:
                self._close_position(pair)

        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")

    def _check_session_change(self):
        """Check if trading session has changed"""
        session_params = self.session_manager.get_session_params()
        current_session = session_params['name']

        if current_session != self.last_session:
            logger.info("=" * 60)
            logger.info(f"SESSION CHANGE: {current_session.upper()}")
            logger.info(f"Pairs: {', '.join(session_params['pairs'])}")
            logger.info(f"Strategy: {session_params['strategy']}")
            logger.info(f"Min confidence: {session_params['min_confidence']}")
            logger.info("=" * 60)

            # Close positions not in new session
            active_pairs = set(session_params['pairs'])
            for pair in list(self.positions.keys()):
                if pair not in active_pairs and self.positions[pair] != 0:
                    logger.info(f"Closing {pair} (not in current session)")
                    self._close_position(pair)

            self.last_session = current_session

    def run(self, interval: int = 5):
        """Main trading loop"""
        logger.info("=" * 60)
        logger.info("24/7 FOREX TRADING DAEMON")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        # Load ML models
        if self.ml_predictor.load_models():
            logger.info("ML models loaded successfully")
        else:
            logger.warning("Running without ML models - using simple signals")

        # Connect to IB
        if not self.connect():
            logger.error("Failed to connect to IB - exiting")
            return

        self.running = True
        logger.info(f"Starting trading loop (interval={interval}s)")

        try:
            while self.running:
                loop_start = time.time()

                # Check session changes
                self._check_session_change()

                # Check if we should trade
                if not self.session_manager.should_trade():
                    logger.debug("Off hours - waiting...")
                    time.sleep(60)
                    continue

                # Get session parameters
                session_params = self.session_manager.get_session_params()
                pairs = session_params['pairs']

                if not pairs:
                    time.sleep(interval)
                    continue

                # Process each pair
                for pair in pairs:
                    self._process_pair(pair, session_params)

                # Update account
                self._update_account()

                # Log status
                risk_status = self.risk_manager.get_status()
                logger.info(f"[{session_params['name']}] Balance: ${risk_status['current_balance']:,.2f}, "
                          f"Daily P/L: ${risk_status['daily_pnl']:,.2f} ({risk_status['daily_pnl_pct']:.2f}%)")

                # Sleep
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            self.disconnect()
            if PID_FILE.exists():
                PID_FILE.unlink()
            logger.info("Trading daemon stopped")


def check_status():
    """Check if daemon is running"""
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            os.kill(pid, 0)
            print(f"Trading daemon is running (PID: {pid})")
            return True
        except (OSError, ValueError):
            print("Trading daemon is not running (stale PID file)")
            PID_FILE.unlink()
            return False
    else:
        print("Trading daemon is not running")
        return False


def main():
    parser = argparse.ArgumentParser(description='24/7 Forex Trading Daemon')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--port', type=int, help='IB port (default: 7496 paper, 7497 live)')
    parser.add_argument('--interval', type=int, default=5, help='Trading interval in seconds')
    parser.add_argument('--status', action='store_true', help='Check daemon status')

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    daemon = TradingDaemon(mode=args.mode, port=args.port)

    try:
        daemon.run(interval=args.interval)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
