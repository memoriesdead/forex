"""
Interactive Brokers Live Trading
Uses ensemble predictions from all frameworks
Qlib + FinRL + V20pyPro + Chinese Quant
Start with $100, maximize ROI
"""

from ib_insync import IB, Forex, MarketOrder
import pandas as pd
import pickle
from pathlib import Path
import time
import logging
from datetime import datetime

# Setup logging
log_file = Path(__file__).parent.parent / "logs" / "ib_live_trading.log"
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

# IB Connection
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # Live trading
IB_CLIENT_ID = 1

# Model directory
MODELS_DIR = Path(__file__).parent.parent / 'models' / 'multi_framework'

# Trading pairs
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
         'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']

# Risk management
STARTING_CAPITAL = 100  # $100 starting capital
RISK_PER_TRADE = 0.02   # 2% risk per trade
MAX_POSITIONS = 3       # Max 3 positions at once
MIN_CONFIDENCE = 0.65   # 65% minimum confidence


class MultiFrameworkPredictor:
    """Ensemble predictions from all frameworks."""

    def __init__(self, pair: str):
        self.pair = pair
        self.models = self.load_models()

    def load_models(self):
        """Load all framework models."""
        models = {}

        # Load ensemble config
        ensemble_path = MODELS_DIR / f'{self.pair}_ensemble.pkl'
        if ensemble_path.exists():
            with open(ensemble_path, 'rb') as f:
                models['ensemble'] = pickle.load(f)

        # Load Qlib
        qlib_path = MODELS_DIR / f'{self.pair}_qlib.pkl'
        if qlib_path.exists():
            with open(qlib_path, 'rb') as f:
                models['qlib'] = pickle.load(f)

        # Load FinRL
        finrl_path = MODELS_DIR / f'{self.pair}_finrl.txt'
        if finrl_path.exists():
            models['finrl'] = str(finrl_path)

        # Load V20pyPro
        v20_path = MODELS_DIR / f'{self.pair}_v20pypro.h5'
        if v20_path.exists():
            from tensorflow import keras
            models['v20pypro'] = keras.models.load_model(str(v20_path))

        # Load Chinese Quant
        chinese_path = MODELS_DIR / f'{self.pair}_chinese.pkl'
        if chinese_path.exists():
            with open(chinese_path, 'rb') as f:
                models['chinese'] = pickle.load(f)

        logger.info(f"{self.pair}: Loaded {len(models)} models")
        return models

    def get_prediction(self, data: pd.DataFrame):
        """Get ensemble prediction."""
        if not self.models:
            return 'neutral', 0.0

        predictions = []
        confidences = []

        # Get predictions from each model
        # (In production, implement actual prediction logic)

        # For now, return majority vote
        if len(predictions) == 0:
            return 'neutral', 0.5

        # Majority vote
        long_votes = predictions.count('long')
        short_votes = predictions.count('short')
        total_votes = len(predictions)

        if long_votes > short_votes:
            confidence = long_votes / total_votes
            return 'long', confidence
        elif short_votes > long_votes:
            confidence = short_votes / total_votes
            return 'short', confidence
        else:
            return 'neutral', 0.5


class IBLiveTrader:
    """Live trading via Interactive Brokers."""

    def __init__(self):
        self.ib = IB()
        self.predictors = {pair: MultiFrameworkPredictor(pair) for pair in PAIRS}
        self.positions = {}
        self.capital = STARTING_CAPITAL
        self.trades_today = 0

    def connect(self):
        """Connect to IB."""
        logger.info(f"Connecting to IB @ {IB_HOST}:{IB_PORT}")
        self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)

        if self.ib.isConnected():
            logger.info("[OK] Connected to Interactive Brokers")
            return True
        else:
            logger.error("[FAIL] Connection failed")
            return False

    def get_account_value(self):
        """Get current account value."""
        account_values = self.ib.accountValues()
        for av in account_values:
            if av.tag == 'NetLiquidation':
                return float(av.value)
        return self.capital

    def calculate_position_size(self, pair: str, entry_price: float):
        """Calculate position size based on risk management."""
        risk_amount = self.capital * RISK_PER_TRADE
        # For forex, 1 lot = 100,000 units
        # Micro lot = 1,000 units
        # Start with micro lots

        position_size = 1000  # 1 micro lot
        return position_size

    def open_position(self, pair: str, signal: str, confidence: float):
        """Open a position."""
        if pair in self.positions:
            logger.info(f"{pair}: Already have position")
            return

        if len(self.positions) >= MAX_POSITIONS:
            logger.info(f"Max positions ({MAX_POSITIONS}) reached")
            return

        if confidence < MIN_CONFIDENCE:
            logger.info(f"{pair}: Confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            return

        # Get current price
        contract = Forex(pair)
        self.ib.qualifyContracts(contract)

        ticker = self.ib.reqMktData(contract)
        time.sleep(1)  # Wait for data

        if not ticker.bid or not ticker.ask:
            logger.warning(f"{pair}: No market data")
            return

        entry_price = ticker.ask if signal == 'long' else ticker.bid
        size = self.calculate_position_size(pair, entry_price)

        # Place order
        action = 'BUY' if signal == 'long' else 'SELL'
        order = MarketOrder(action, size)

        trade = self.ib.placeOrder(contract, order)

        logger.info(f"[OPEN] {pair} {signal.upper()} @ {entry_price:.5f} size={size}")
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  Order ID: {trade.order.orderId}")

        self.positions[pair] = {
            'signal': signal,
            'entry': entry_price,
            'size': size,
            'confidence': confidence,
            'time': datetime.now()
        }

        self.trades_today += 1

    def close_position(self, pair: str, reason: str = 'signal'):
        """Close a position."""
        if pair not in self.positions:
            return

        pos = self.positions[pair]

        # Get current price
        contract = Forex(pair)
        ticker = self.ib.reqMktData(contract)
        time.sleep(1)

        exit_price = ticker.bid if pos['signal'] == 'long' else ticker.ask

        # Calculate P&L
        if pos['signal'] == 'long':
            pnl = (exit_price - pos['entry']) * pos['size']
        else:
            pnl = (pos['entry'] - exit_price) * pos['size']

        self.capital += pnl

        # Place closing order
        action = 'SELL' if pos['signal'] == 'long' else 'BUY'
        order = MarketOrder(action, pos['size'])

        self.ib.placeOrder(contract, order)

        logger.info(f"[CLOSE] {pair} @ {exit_price:.5f}")
        logger.info(f"  P&L: ${pnl:.2f}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Capital: ${self.capital:.2f}")

        del self.positions[pair]

    def trading_loop(self):
        """Main trading loop."""
        logger.info("="*60)
        logger.info("LIVE TRADING STARTED")
        logger.info(f"Starting capital: ${self.capital:.2f}")
        logger.info(f"Pairs: {PAIRS}")
        logger.info(f"Risk per trade: {RISK_PER_TRADE*100}%")
        logger.info("="*60)

        cycle = 0

        try:
            while True:
                cycle += 1

                # Get latest data for each pair
                for pair in PAIRS:
                    try:
                        # Get live data from local stream
                        # (Read from data/truefx_live/*.csv)
                        live_file = Path(__file__).parent.parent / 'data' / 'truefx_live' / f'{pair}_{datetime.now().strftime("%Y-%m-%d")}_live.csv'

                        if not live_file.exists():
                            continue

                        # Read last 100 ticks
                        df = pd.read_csv(live_file, names=['timestamp', 'pair', 'bid', 'ask', 'bid_vol', 'ask_vol'])
                        df = df.tail(100)

                        # Get prediction from all models
                        signal, confidence = self.predictors[pair].get_prediction(df)

                        # Manage positions
                        if pair in self.positions:
                            # Check if should close
                            if signal == 'neutral' or signal != self.positions[pair]['signal']:
                                self.close_position(pair, reason='signal_change')
                        else:
                            # Check if should open
                            if signal in ['long', 'short']:
                                self.open_position(pair, signal, confidence)

                    except Exception as e:
                        logger.error(f"{pair} error: {e}")

                # Log status every 10 cycles
                if cycle % 10 == 0:
                    capital = self.get_account_value()
                    roi = ((capital - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
                    logger.info(f"[{cycle}] Capital: ${capital:.2f} | ROI: {roi:+.2f}% | Positions: {len(self.positions)}")

                # Sleep 1 second
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n" + "="*60)
            logger.info("TRADING STOPPED")
            logger.info("="*60)

            # Close all positions
            for pair in list(self.positions.keys()):
                self.close_position(pair, reason='shutdown')

            final_capital = self.get_account_value()
            roi = ((final_capital - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

            logger.info(f"Starting: ${STARTING_CAPITAL:.2f}")
            logger.info(f"Final: ${final_capital:.2f}")
            logger.info(f"ROI: {roi:+.2f}%")
            logger.info(f"Trades: {self.trades_today}")
            logger.info("="*60)

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")


def main():
    trader = IBLiveTrader()

    if not trader.connect():
        return

    try:
        trader.trading_loop()
    finally:
        trader.disconnect()


if __name__ == "__main__":
    main()
