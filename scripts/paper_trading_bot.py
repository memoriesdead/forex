"""
Paper Trading Bot for Forex (OANDA Practice API)

Integrates:
- Live data from Oracle Cloud (via sync)
- Feature engineering for ML models
- Vast.ai inference via API
- OANDA practice account execution
- Risk management and position sizing
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from dotenv import load_dotenv
import json

load_dotenv()

# Configuration
OANDA_PRACTICE_KEY = os.getenv('OANDA_PRACTICE_API_KEY')
OANDA_PRACTICE_ACCOUNT = os.getenv('OANDA_PRACTICE_ACCOUNT_ID')
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"

MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PERCENT', 5))
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PERCENT', 27))
HFT_MIN_CONFIDENCE = float(os.getenv('HFT_MIN_CONFIDENCE', 0.52))
SNIPER_MIN_CONFIDENCE = float(os.getenv('SNIPER_MIN_CONFIDENCE', 0.85))

# Paths
DATA_DIR = Path("data/live/latest")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading pairs (using OANDA format)
PAIRS = {
    'EURUSD': 'EUR_USD',
    'GBPUSD': 'GBP_USD',
    'USDJPY': 'USD_JPY',
    'AUDUSD': 'AUD_USD',
    'USDCAD': 'USD_CAD'
}


class OANDAClient:
    """OANDA Practice API client"""

    def __init__(self, api_key: str, account_id: str):
        self.api_key = api_key
        self.account_id = account_id
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def get_account_summary(self) -> Dict:
        """Get account balance and stats"""
        url = f"{OANDA_BASE_URL}/v3/accounts/{self.account_id}/summary"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()['account']

    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        url = f"{OANDA_BASE_URL}/v3/accounts/{self.account_id}/openPositions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get('positions', [])

    def place_market_order(self, instrument: str, units: int) -> Dict:
        """
        Place market order
        units > 0 = buy (long)
        units < 0 = sell (short)
        """
        url = f"{OANDA_BASE_URL}/v3/accounts/{self.account_id}/orders"

        order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': instrument,
                'units': str(units),
                'timeInForce': 'FOK',  # Fill or kill
                'positionFill': 'DEFAULT'
            }
        }

        response = requests.post(url, headers=self.headers, json=order_data)
        response.raise_for_status()
        return response.json()

    def close_position(self, instrument: str) -> Dict:
        """Close position for instrument"""
        url = f"{OANDA_BASE_URL}/v3/accounts/{self.account_id}/positions/{instrument}/close"
        response = requests.put(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class FeatureEngine:
    """Calculate features from live data for ML inference"""

    @staticmethod
    def load_latest_data(pair: str, lookback_seconds: int = 300) -> pd.DataFrame:
        """Load latest N seconds of data for pair"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        file_path = DATA_DIR / f"{pair}_{today}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"No data file for {pair}")

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get last N seconds
        cutoff = datetime.utcnow() - timedelta(seconds=lookback_seconds)
        df = df[df['timestamp'] >= cutoff].copy()

        return df

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate ML features from price data"""
        if len(df) < 20:
            raise ValueError("Not enough data for features")

        # Mid price
        df['mid'] = (df['bid'] + df['ask']) / 2

        # Returns
        df['returns_1'] = df['mid'].pct_change(1)
        df['returns_5'] = df['mid'].pct_change(5)
        df['returns_20'] = df['mid'].pct_change(20)

        # Moving averages
        df['ma_5'] = df['mid'].rolling(5).mean()
        df['ma_20'] = df['mid'].rolling(20).mean()

        # Volatility
        df['volatility'] = df['returns_1'].rolling(20).std()

        # Spread metrics
        df['avg_spread'] = df['spread'].rolling(20).mean()

        # Volume imbalance (bid/ask spread changes)
        df['spread_change'] = df['spread'].diff()
        df['vol_imbalance'] = df['spread_change'].rolling(10).sum()

        # Get latest values
        latest = df.iloc[-1]

        features = {
            'mid_price': latest['mid'],
            'returns_1': latest['returns_1'],
            'returns_5': latest['returns_5'],
            'returns_20': latest['returns_20'],
            'ma_5': latest['ma_5'],
            'ma_20': latest['ma_20'],
            'volatility': latest['volatility'],
            'spread': latest['spread'],
            'avg_spread': latest['avg_spread'],
            'vol_imbalance': latest['vol_imbalance']
        }

        # Handle NaN values
        features = {k: (v if pd.notna(v) else 0.0) for k, v in features.items()}

        return features


class VastAIInference:
    """Interface to vast.ai model server"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def predict(self, pair: str, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Get prediction from vast.ai model server
        Returns: (signal, confidence)
        signal = 'long', 'short', or 'neutral'
        """
        try:
            response = requests.post(
                f"{self.endpoint}/predict",
                json={'pair': pair, 'features': features},
                timeout=2
            )
            response.raise_for_status()

            result = response.json()
            return result['signal'], result['confidence']

        except requests.RequestException as e:
            logger.error(f"Vast.ai inference failed: {e}")
            return 'neutral', 0.0


class RiskManager:
    """Position sizing and risk management"""

    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.daily_pnl = 0.0
        self.daily_start = datetime.utcnow().date()

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        if datetime.utcnow().date() > self.daily_start:
            # New day, reset
            self.daily_pnl = 0.0
            self.daily_start = datetime.utcnow().date()
            return False

        loss_limit = self.account_balance * (MAX_DAILY_LOSS_PCT / 100)
        if self.daily_pnl < -loss_limit:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return True

        return False

    def calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on confidence and risk limits"""
        # Max position value
        max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

        # Scale by confidence
        position_value = max_position_value * confidence

        # Convert to units (forex is typically 100,000 units = 1 standard lot)
        # For micro lots: 1,000 units
        units = int(position_value / 100)  # Conservative sizing

        return units

    def update_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl


class PaperTradingBot:
    """Main paper trading bot"""

    def __init__(self, vastai_endpoint: Optional[str] = None):
        if not OANDA_PRACTICE_KEY or OANDA_PRACTICE_KEY == 'your_practice_key_here':
            raise ValueError("OANDA Practice API key not configured in .env")

        self.oanda = OANDAClient(OANDA_PRACTICE_KEY, OANDA_PRACTICE_ACCOUNT)
        self.feature_engine = FeatureEngine()
        self.vastai = VastAIInference(vastai_endpoint) if vastai_endpoint else None
        self.risk_manager = None

        # State
        self.positions = {}
        self.is_running = False

    def initialize(self):
        """Initialize bot and load account info"""
        logger.info("="*60)
        logger.info("INITIALIZING PAPER TRADING BOT")
        logger.info("="*60)

        try:
            account = self.oanda.get_account_summary()
            balance = float(account['balance'])
            logger.info(f"Account balance: ${balance:,.2f}")
            logger.info(f"Currency: {account['currency']}")

            self.risk_manager = RiskManager(balance)

            # Load existing positions
            positions = self.oanda.get_positions()
            logger.info(f"Open positions: {len(positions)}")

            for pos in positions:
                instrument = pos['instrument']
                units = int(pos['long']['units']) - int(pos['short']['units'])
                self.positions[instrument] = units
                logger.info(f"  {instrument}: {units} units")

            logger.info("="*60)
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def sync_latest_data(self):
        """Sync latest minute of data from Oracle Cloud"""
        import subprocess
        try:
            subprocess.run(
                ["python", "scripts/sync_live_data_v2.py", "--latest", "1m"],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Data sync failed: {e}")

    def process_pair(self, pair: str):
        """Process one trading pair"""
        oanda_pair = PAIRS[pair]

        try:
            # Load data and calculate features
            df = self.feature_engine.load_latest_data(pair)
            features = self.feature_engine.calculate_features(df)

            logger.info(f"{pair}: mid={features['mid_price']:.5f}, "
                       f"returns_1={features['returns_1']:.6f}, "
                       f"volatility={features['volatility']:.6f}")

            # Get prediction from vast.ai (if configured)
            if self.vastai:
                signal, confidence = self.vastai.predict(pair, features)
            else:
                # Simple momentum strategy if no ML model
                if features['returns_5'] > 0.0001 and features['ma_5'] > features['ma_20']:
                    signal, confidence = 'long', 0.6
                elif features['returns_5'] < -0.0001 and features['ma_5'] < features['ma_20']:
                    signal, confidence = 'short', 0.6
                else:
                    signal, confidence = 'neutral', 0.0

            logger.info(f"{pair}: signal={signal}, confidence={confidence:.2f}")

            # Risk checks
            if self.risk_manager.check_daily_loss_limit():
                logger.warning("Daily loss limit reached, skipping trades")
                return

            # Position management
            current_position = self.positions.get(oanda_pair, 0)

            # Entry logic
            if signal == 'long' and confidence >= HFT_MIN_CONFIDENCE:
                if current_position <= 0:  # Not already long
                    units = self.risk_manager.calculate_position_size(confidence)
                    if units > 0:
                        # Close short if exists
                        if current_position < 0:
                            self.oanda.close_position(oanda_pair)

                        # Open long
                        result = self.oanda.place_market_order(oanda_pair, units)
                        logger.info(f"{pair}: OPENED LONG {units} units")
                        self.positions[oanda_pair] = units

            elif signal == 'short' and confidence >= HFT_MIN_CONFIDENCE:
                if current_position >= 0:  # Not already short
                    units = self.risk_manager.calculate_position_size(confidence)
                    if units > 0:
                        # Close long if exists
                        if current_position > 0:
                            self.oanda.close_position(oanda_pair)

                        # Open short
                        result = self.oanda.place_market_order(oanda_pair, -units)
                        logger.info(f"{pair}: OPENED SHORT {units} units")
                        self.positions[oanda_pair] = -units

            # Exit logic
            elif signal == 'neutral' and current_position != 0:
                self.oanda.close_position(oanda_pair)
                logger.info(f"{pair}: CLOSED POSITION")
                self.positions[oanda_pair] = 0

        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")

    def run(self, interval_seconds: int = 5):
        """Main trading loop"""
        self.is_running = True
        logger.info(f"Starting trading loop (interval={interval_seconds}s)")

        try:
            while self.is_running:
                loop_start = time.time()

                # Sync latest data
                self.sync_latest_data()

                # Process each pair
                for pair in PAIRS.keys():
                    self.process_pair(pair)

                # Update account status
                account = self.oanda.get_account_summary()
                balance = float(account['balance'])
                unrealized = float(account.get('unrealizedPL', 0))

                logger.info(f"Balance: ${balance:,.2f}, Unrealized P/L: ${unrealized:,.2f}")

                # Sleep until next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Stopping bot (user interrupt)")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.is_running = False

    def stop(self):
        """Stop trading loop"""
        self.is_running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--vastai-endpoint', help='Vast.ai model server endpoint (e.g., http://123.45.67.89:5000)')
    parser.add_argument('--interval', type=int, default=5, help='Trading interval in seconds (default: 5)')

    args = parser.parse_args()

    bot = PaperTradingBot(vastai_endpoint=args.vastai_endpoint)

    if not bot.initialize():
        logger.error("Failed to initialize bot")
        return 1

    try:
        bot.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        logger.info("Paper trading bot stopped")

    return 0


if __name__ == "__main__":
    exit(main())
