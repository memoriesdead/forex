"""
Vast.ai Live Inference System
Sends live tick data to Vast.ai GPU for real-time predictions.
"""

import requests
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Optional
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

VAST_API_KEY = os.getenv('VAST_AI_API_KEY')
LOCAL_LIVE_DIR = Path(__file__).parent.parent / "data" / "truefx_live"

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / "vastai_inference.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VastAILiveInference:
    """Real-time inference using Vast.ai GPU."""

    def __init__(self, model_endpoint: str, api_key: str = None):
        """
        Initialize Vast.ai live inference.

        Args:
            model_endpoint: Vast.ai instance URL (e.g., http://123.456.789.10:5000)
            api_key: Vast.ai API key (from .env if not provided)
        """
        self.endpoint = model_endpoint
        self.api_key = api_key or VAST_API_KEY
        self.last_predictions = {}

        logger.info(f"Vast.ai Inference initialized: {model_endpoint}")

    def read_latest_ticks(self, pair: str, num_ticks: int = 100) -> pd.DataFrame:
        """Read latest N ticks for a pair from live data files."""
        try:
            live_file = LOCAL_LIVE_DIR / f"{pair}_{datetime.now().strftime('%Y-%m-%d')}_live.csv"

            if not live_file.exists():
                logger.warning(f"Live data file not found: {live_file}")
                return None

            # Read last N lines
            df = pd.read_tail(live_file, num_ticks, names=['timestamp', 'pair', 'bid', 'ask', 'bid_vol', 'ask_vol'])

            return df

        except Exception as e:
            logger.error(f"Error reading ticks for {pair}: {e}")
            return None

    def calculate_features(self, df: pd.DataFrame) -> Dict:
        """Calculate features from tick data for ML model."""
        if df is None or len(df) < 20:
            return None

        try:
            # Calculate mid price
            df['mid'] = (df['bid'] + df['ask']) / 2

            # Returns at different timeframes
            returns_1 = (df['mid'].iloc[-1] - df['mid'].iloc[-2]) / df['mid'].iloc[-2]
            returns_5 = (df['mid'].iloc[-1] - df['mid'].iloc[-6]) / df['mid'].iloc[-6]
            returns_20 = (df['mid'].iloc[-1] - df['mid'].iloc[-21]) / df['mid'].iloc[-21]

            # Moving averages
            ma_5 = df['mid'].tail(5).mean()
            ma_20 = df['mid'].tail(20).mean()

            # Volatility
            volatility = df['mid'].tail(20).std()

            # Spread
            spread = df['ask'].iloc[-1] - df['bid'].iloc[-1]
            avg_spread = (df['ask'] - df['bid']).tail(20).mean()

            # Volume features
            bid_vol_avg = df['bid_vol'].tail(20).mean()
            ask_vol_avg = df['ask_vol'].tail(20).mean()
            vol_imbalance = (ask_vol_avg - bid_vol_avg) / (ask_vol_avg + bid_vol_avg)

            features = {
                'returns_1': float(returns_1),
                'returns_5': float(returns_5),
                'returns_20': float(returns_20),
                'ma_5': float(ma_5),
                'ma_20': float(ma_20),
                'volatility': float(volatility),
                'spread': float(spread),
                'avg_spread': float(avg_spread),
                'vol_imbalance': float(vol_imbalance),
                'current_price': float(df['mid'].iloc[-1])
            }

            return features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None

    def get_prediction(self, pair: str, features: Dict, timeout: float = 5.0) -> Optional[Dict]:
        """Get prediction from Vast.ai GPU model."""
        try:
            # Send request to Vast.ai model endpoint
            response = requests.post(
                f"{self.endpoint}/predict",
                json={
                    'pair': pair,
                    'features': features
                },
                headers={'Authorization': f'Bearer {self.api_key}'},
                timeout=timeout
            )

            if response.status_code == 200:
                prediction = response.json()
                self.last_predictions[pair] = prediction
                return prediction
            else:
                logger.error(f"Prediction failed: {response.status_code} - {response.text}")
                return None

        except requests.Timeout:
            logger.warning(f"Prediction timeout for {pair} (>{timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Error getting prediction for {pair}: {e}")
            return None

    def get_trading_signal(self, pair: str) -> Optional[str]:
        """
        Get trading signal for a pair.

        Returns:
            'long': Buy signal
            'short': Sell signal
            'neutral': Hold/no action
            None: No data or error
        """
        # Read latest ticks
        df = self.read_latest_ticks(pair, num_ticks=100)
        if df is None:
            return None

        # Calculate features
        features = self.calculate_features(df)
        if features is None:
            return None

        # Get prediction from Vast.ai
        prediction = self.get_prediction(pair, features)
        if prediction is None:
            return None

        # Interpret prediction
        # Assuming model returns {'signal': 'long/short/neutral', 'confidence': 0.0-1.0}
        signal = prediction.get('signal', 'neutral')
        confidence = prediction.get('confidence', 0.0)

        logger.info(f"{pair}: {signal} (confidence: {confidence:.2f})")

        return signal if confidence > 0.55 else 'neutral'

    def stream_inference(self, pairs: list, interval: float = 1.0):
        """
        Continuously stream inference for multiple pairs.

        Args:
            pairs: List of currency pairs
            interval: Seconds between inference cycles
        """
        logger.info("=" * 60)
        logger.info("STARTING LIVE INFERENCE STREAM")
        logger.info(f"Pairs: {pairs}")
        logger.info(f"Interval: {interval}s")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info("=" * 60)

        cycle = 0

        try:
            while True:
                start_time = time.time()

                signals = {}
                for pair in pairs:
                    signal = self.get_trading_signal(pair)
                    if signal:
                        signals[pair] = signal

                cycle += 1

                # Log every 10 cycles
                if cycle % 10 == 0:
                    logger.info(f"[{cycle}] Signals: {signals}")

                # Maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 60)
            logger.info("INFERENCE STREAM STOPPED")
            logger.info(f"Total cycles: {cycle}")
            logger.info("=" * 60)


def pd_read_tail(file_path: Path, num_lines: int, names: list) -> pd.DataFrame:
    """Read last N lines of CSV file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_lines = lines[-num_lines:]

    # Parse lines
    data = []
    for line in last_lines:
        data.append(line.strip().split(','))

    df = pd.DataFrame(data, columns=names)

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['bid'] = df['bid'].astype(float)
    df['ask'] = df['ask'].astype(float)
    df['bid_vol'] = df['bid_vol'].astype(int)
    df['ask_vol'] = df['ask_vol'].astype(int)

    return df


# Monkey patch pandas
pd.read_tail = pd_read_tail


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Vast.ai Live Inference')
    parser.add_argument('--endpoint', required=True, help='Vast.ai model endpoint URL')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Currency pairs to trade')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Seconds between inference cycles')

    args = parser.parse_args()

    # Initialize and run
    inference = VastAILiveInference(args.endpoint)
    inference.stream_inference(args.pairs, args.interval)
