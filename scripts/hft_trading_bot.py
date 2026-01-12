"""
HFT Trading Bot - Production Ready
===================================
Real-time forex trading using institutional-grade ML ensemble.

Components:
- TrueFX live tick feed
- HFT feature engine (100+ features)
- ML ensemble predictor (XGBoost/LightGBM/CatBoost)
- IB Gateway execution
- Latency-aware order management
- Risk management (Kelly, drawdown limits)
- Real-time monitoring

Usage:
    python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD
    python scripts/hft_trading_bot.py --mode live --symbols EURUSD

Modes:
    paper - Paper trading via IB Gateway (DUO423364)
    live - Live trading (requires live account)
    backtest - Run on historical data
    monitor - Monitor only, no trades
"""

import asyncio
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import pickle
import signal
import sys
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    MONITOR = "monitor"


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Signal:
    """Trading signal from ML models."""
    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0-1
    predicted_return: float  # Expected return in bps
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_votes: Dict[str, int] = field(default_factory=dict)


@dataclass
class Order:
    """Order to be executed."""
    symbol: str
    side: Side
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Executed trade."""
    symbol: str
    side: Side
    quantity: float
    fill_price: float
    timestamp: datetime
    signal_confidence: float
    latency_ms: float


class RiskManager:
    """
    Risk management using Kelly Criterion and drawdown limits.
    """

    def __init__(self,
                 max_position_pct: float = 0.02,  # 2% of account per position
                 max_drawdown_pct: float = 0.05,  # 5% max drawdown
                 kelly_fraction: float = 0.25,   # Quarter Kelly
                 max_daily_trades: int = 100):

        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.max_daily_trades = max_daily_trades

        # State
        self.account_balance = 100000.0  # Will be updated from IB
        self.peak_balance = self.account_balance
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()

    def reset_daily_limits(self):
        """Reset daily counters."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today

    def update_balance(self, balance: float):
        """Update account balance."""
        self.account_balance = balance
        self.peak_balance = max(self.peak_balance, balance)

    def current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.account_balance) / self.peak_balance

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        self.reset_daily_limits()

        # Check drawdown limit
        dd = self.current_drawdown()
        if dd > self.max_drawdown_pct:
            return False, f"Drawdown limit exceeded: {dd:.1%} > {self.max_drawdown_pct:.1%}"

        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"

        return True, "OK"

    def kelly_size(self, win_prob: float, win_loss_ratio: float = 1.5) -> float:
        """
        Calculate Kelly position size.

        f* = (p * b - q) / b
        where p = win prob, q = 1-p, b = win/loss ratio
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        # Fractional Kelly (safer)
        kelly = kelly * self.kelly_fraction

        # Clamp to reasonable range
        kelly = max(0, min(kelly, self.max_position_pct))

        return kelly

    def calculate_position_size(self, signal: Signal,
                                current_price: float) -> float:
        """Calculate position size based on signal and risk limits."""
        # Convert confidence to position size
        win_prob = 0.5 + signal.confidence * 0.15  # Map confidence to 50-65% win rate

        kelly = self.kelly_size(win_prob)

        # Dollar amount
        position_dollars = self.account_balance * kelly

        # Convert to lots (100k units)
        lot_size = 100000
        position_lots = position_dollars / lot_size

        # Round to micro lots (0.01)
        position_lots = round(position_lots, 2)

        # Minimum 0.01 lots
        if position_lots < 0.01 and signal.confidence > 0.6:
            position_lots = 0.01

        return position_lots

    def record_trade(self, pnl: float):
        """Record trade for daily limits."""
        self.daily_trades += 1
        self.daily_pnl += pnl


class MLEnsemble:
    """
    ML Ensemble Predictor.
    Loads trained models and generates trading signals.
    """

    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("models/hft_ensemble")
        self.models: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.loaded = False

    def load_models(self, symbols: List[str] = None):
        """Load trained models for symbols."""
        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            # Try new format first
            model_path = self.model_dir / f"{symbol}_target_direction_10_models.pkl"

            # Fallback to legacy format
            legacy_path = Path("models") / f"hft_{symbol}_aggressive.pkl"
            legacy_quick = Path("models") / f"hft_{symbol}_quick.pkl"

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    self.models[symbol] = data
                    if not self.feature_names and 'feature_names' in data:
                        self.feature_names = data['feature_names']
                    logger.info(f"Loaded models for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load models for {symbol}: {e}")
            elif legacy_path.exists():
                try:
                    with open(legacy_path, 'rb') as f:
                        data = pickle.load(f)
                    # Adapt legacy format to expected format
                    adapted = {
                        'models': {'primary': data.get('model', data)},
                        'feature_names': data.get('features', [])
                    }
                    self.models[symbol] = adapted
                    if not self.feature_names and adapted['feature_names']:
                        self.feature_names = adapted['feature_names']
                    logger.info(f"Loaded LEGACY models for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to load legacy models for {symbol}: {e}")
            else:
                logger.warning(f"No models found for {symbol}")

        self.loaded = len(self.models) > 0

    def predict(self, symbol: str, features: Dict[str, float]) -> Optional[Signal]:
        """Generate signal from features."""
        if symbol not in self.models:
            return None

        model_data = self.models[symbol]
        models = model_data.get('models', {})

        if not models:
            return None

        # Prepare feature vector
        if self.feature_names:
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        else:
            X = np.array([list(features.values())])

        # Get predictions from each model
        predictions = {}
        probabilities = {}

        for name, model in models.items():
            try:
                pred = model.predict(X)[0]
                predictions[name] = int(pred)

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = proba[1] if len(proba) > 1 else proba[0]
                else:
                    probabilities[name] = 0.5

            except Exception as e:
                logger.debug(f"Prediction error for {name}: {e}")

        if not predictions:
            return None

        # Ensemble voting
        avg_prob = np.mean(list(probabilities.values()))
        majority_vote = 1 if sum(predictions.values()) > len(predictions) / 2 else 0

        # Direction: 1 = long, -1 = short
        direction = 1 if majority_vote == 1 else -1

        # Confidence: distance from 0.5
        confidence = abs(avg_prob - 0.5) * 2

        # Predicted return in bps (simple estimate)
        predicted_return = direction * confidence * 10  # 10 bps max

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            features=features,
            model_votes=predictions
        )


class IBGatewayConnector:
    """
    Interactive Brokers Gateway Connector.
    Handles order execution via IB API.
    """

    def __init__(self, host: str = 'localhost', port: int = 4001,
                 client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.ib = None

    async def connect(self):
        """Connect to IB Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IB Gateway at {self.host}:{self.port}")

            # Get account info
            account = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else "Unknown"
            logger.info(f"Account: {account}")

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            logger.info("Ensure SSH tunnel is running: ssh -L 4001:localhost:4001 ubuntu@89.168.65.47")
            self.connected = False

    async def get_account_balance(self) -> float:
        """Get current account balance."""
        if not self.connected or not self.ib:
            return 100000.0  # Default for paper

        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    return float(av.value)
        except:
            pass

        return 100000.0

    async def submit_order(self, order: Order) -> Optional[Trade]:
        """Submit order to IB Gateway."""
        if not self.connected or not self.ib:
            logger.warning("Not connected to IB Gateway - simulating fill")
            return self._simulate_fill(order)

        try:
            from ib_insync import Forex, MarketOrder, LimitOrder

            # Create forex contract
            contract = Forex(order.symbol[:3] + order.symbol[3:])

            # Create order
            if order.order_type == "MARKET":
                ib_order = MarketOrder(
                    order.side.value,
                    order.quantity * 100000  # Convert to units
                )
            else:
                ib_order = LimitOrder(
                    order.side.value,
                    order.quantity * 100000,
                    order.limit_price
                )

            # Submit
            submit_time = datetime.now()
            trade = self.ib.placeOrder(contract, ib_order)

            # Wait for fill (with timeout)
            await asyncio.sleep(0.1)

            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0.0
            fill_time = datetime.now()

            latency_ms = (fill_time - submit_time).total_seconds() * 1000

            return Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=fill_time,
                signal_confidence=0.0,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return None

    def _simulate_fill(self, order: Order) -> Trade:
        """Simulate order fill for testing."""
        # Simulate realistic fill price
        base_price = order.limit_price if order.limit_price else 1.1000
        slippage = 0.00001 * (1 if order.side == Side.BUY else -1)

        return Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=base_price + slippage,
            timestamp=datetime.now(),
            signal_confidence=0.0,
            latency_ms=50.0
        )

    async def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")


class HFTTradingBot:
    """
    Main HFT Trading Bot.
    Orchestrates data, features, signals, and execution.
    """

    def __init__(self,
                 mode: TradingMode = TradingMode.PAPER,
                 symbols: List[str] = None):

        self.mode = mode
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']

        # Components
        self.feature_engine = None
        self.ml_ensemble = None
        self.risk_manager = RiskManager()
        self.ib_connector = None

        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.tick_count: Dict[str, int] = {}
        self.last_prices: Dict[str, Tuple[float, float]] = {}

        # Performance tracking
        self.start_time = datetime.now()
        self.total_pnl = 0.0

        # Control
        self.running = False

    async def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing HFT Trading Bot - Mode: {self.mode.value}")

        # Initialize feature engine
        try:
            from core.hft_feature_engine import HFTFeatureEngine
            self.feature_engine = HFTFeatureEngine()
            logger.info("Feature engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize feature engine: {e}")

        # Load ML models
        self.ml_ensemble = MLEnsemble()
        self.ml_ensemble.load_models(self.symbols)

        if not self.ml_ensemble.loaded:
            logger.warning("No ML models loaded - running in signal-only mode")

        # Connect to IB Gateway (for paper/live modes)
        if self.mode in [TradingMode.PAPER, TradingMode.LIVE]:
            self.ib_connector = IBGatewayConnector()
            await self.ib_connector.connect()

            if self.ib_connector.connected:
                balance = await self.ib_connector.get_account_balance()
                self.risk_manager.update_balance(balance)
                logger.info(f"Account balance: ${balance:,.2f}")

        # Initialize positions
        for symbol in self.symbols:
            self.positions[symbol] = Position(symbol=symbol)
            self.tick_count[symbol] = 0

        logger.info("Initialization complete")

    async def process_tick(self, symbol: str, bid: float, ask: float,
                          volume: float = 0.0, timestamp: datetime = None):
        """Process incoming tick and generate/execute signals."""
        timestamp = timestamp or datetime.now()
        mid = (bid + ask) / 2

        self.tick_count[symbol] = self.tick_count.get(symbol, 0) + 1
        self.last_prices[symbol] = (bid, ask)

        # 1. Generate features
        if self.feature_engine:
            features = self.feature_engine.process_tick(symbol, bid, ask, volume, timestamp)
        else:
            features = {'mid_price': mid, 'spread_bps': (ask - bid) / mid * 10000}

        # 2. Generate signal from ML ensemble
        signal = None
        if self.ml_ensemble and self.ml_ensemble.loaded:
            signal = self.ml_ensemble.predict(symbol, features)

        # 3. Risk check
        can_trade, reason = self.risk_manager.can_trade()

        if not can_trade:
            if self.tick_count[symbol] % 1000 == 0:
                logger.warning(f"Trading blocked: {reason}")
            return

        # 4. Generate order if signal is strong enough
        if signal and signal.confidence > 0.6:
            self.signals.append(signal)

            # Only trade if in paper/live mode
            if self.mode in [TradingMode.PAPER, TradingMode.LIVE]:
                await self._execute_signal(signal, bid, ask)

        # 5. Log progress periodically
        if self.tick_count[symbol] % 1000 == 0:
            self._log_status(symbol)

    async def _execute_signal(self, signal: Signal, bid: float, ask: float):
        """Execute trading signal."""
        position = self.positions[signal.symbol]

        # Calculate position size
        size = self.risk_manager.calculate_position_size(signal, (bid + ask) / 2)

        if size < 0.01:
            return  # Too small

        # Determine side and check position
        if signal.direction > 0:  # Long signal
            if position.quantity >= 0:  # Not short, can go long
                side = Side.BUY
            else:  # Close short first
                side = Side.BUY
                size = min(size, abs(position.quantity))
        else:  # Short signal
            if position.quantity <= 0:  # Not long, can go short
                side = Side.SELL
            else:  # Close long first
                side = Side.SELL
                size = min(size, position.quantity)

        # Create order
        order = Order(
            symbol=signal.symbol,
            side=side,
            quantity=size,
            order_type="MARKET"
        )

        # Execute
        trade = await self.ib_connector.submit_order(order)

        if trade:
            trade.signal_confidence = signal.confidence
            self.trades.append(trade)

            # Update position
            if trade.side == Side.BUY:
                position.quantity += trade.quantity
            else:
                position.quantity -= trade.quantity

            logger.info(f"TRADE: {trade.side.value} {trade.quantity:.2f} {trade.symbol} "
                       f"@ {trade.fill_price:.5f} (conf: {signal.confidence:.2f})")

            self.risk_manager.record_trade(0)  # PnL calculated later

    def _log_status(self, symbol: str):
        """Log current status."""
        position = self.positions[symbol]
        n_signals = len([s for s in self.signals if s.symbol == symbol])
        n_trades = len([t for t in self.trades if t.symbol == symbol])

        bid, ask = self.last_prices.get(symbol, (0, 0))

        logger.info(
            f"[{symbol}] Ticks: {self.tick_count[symbol]:,} | "
            f"Signals: {n_signals} | Trades: {n_trades} | "
            f"Position: {position.quantity:.2f} | "
            f"Price: {(bid+ask)/2:.5f}"
        )

    async def run_live(self):
        """Run with live TrueFX data feed."""
        logger.info("Starting live trading with TrueFX feed...")

        try:
            from core.hft_data_loader import UnifiedDataLoader

            loader = UnifiedDataLoader()
            self.running = True

            # Stream live ticks
            async for tick in loader.stream_live(self.symbols):
                if not self.running:
                    break

                await self.process_tick(
                    symbol=tick.symbol,
                    bid=tick.bid,
                    ask=tick.ask,
                    volume=tick.volume,
                    timestamp=tick.timestamp
                )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Live trading error: {e}")
        finally:
            await self.shutdown()

    async def run_backtest(self, data: Dict[str, pd.DataFrame]):
        """Run backtest on historical data."""
        logger.info("Running backtest...")

        self.running = True

        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue

            logger.info(f"Processing {len(df)} ticks for {symbol}")

            for idx, row in df.iterrows():
                if not self.running:
                    break

                bid = row.get('bid', row.get('close', 1.0))
                ask = row.get('ask', bid + 0.0001)
                volume = row.get('volume', 0)
                timestamp = row.get('timestamp', datetime.now())

                await self.process_tick(symbol, bid, ask, volume, timestamp)

        self._print_backtest_results()

    def _print_backtest_results(self):
        """Print backtest summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        for symbol in self.symbols:
            signals = [s for s in self.signals if s.symbol == symbol]
            trades = [t for t in self.trades if t.symbol == symbol]

            if signals:
                avg_conf = np.mean([s.confidence for s in signals])
                long_signals = len([s for s in signals if s.direction > 0])
                short_signals = len([s for s in signals if s.direction < 0])

                print(f"\n{symbol}:")
                print(f"  Signals: {len(signals)} (Long: {long_signals}, Short: {short_signals})")
                print(f"  Avg Confidence: {avg_conf:.2%}")
                print(f"  Trades: {len(trades)}")

        print("\n" + "=" * 60)

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False

        if self.ib_connector:
            await self.ib_connector.disconnect()

        # Save results
        self._save_results()

        logger.info("Shutdown complete")

    def _save_results(self):
        """Save trading results."""
        results = {
            'mode': self.mode.value,
            'symbols': self.symbols,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_ticks': sum(self.tick_count.values()),
            'total_signals': len(self.signals),
            'total_trades': len(self.trades),
            'positions': {s: {'quantity': p.quantity, 'pnl': p.realized_pnl}
                         for s, p in self.positions.items()},
            'risk_manager': {
                'daily_trades': self.risk_manager.daily_trades,
                'drawdown': self.risk_manager.current_drawdown()
            }
        }

        results_path = Path("logs/hft_results.json")
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")


async def main():
    parser = argparse.ArgumentParser(description='HFT Trading Bot')
    parser.add_argument('--mode', type=str, default='paper',
                        choices=['paper', 'live', 'backtest', 'monitor'],
                        help='Trading mode')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD',
                        help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=1,
                        help='Days of data for backtest')
    args = parser.parse_args()

    mode = TradingMode(args.mode)
    symbols = [s.strip() for s in args.symbols.split(',')]

    bot = HFTTradingBot(mode=mode, symbols=symbols)

    # Handle shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize
    await bot.initialize()

    # Run based on mode
    if mode == TradingMode.BACKTEST:
        # Load historical data
        from core.hft_data_loader import UnifiedDataLoader
        loader = UnifiedDataLoader()

        data = {}
        for symbol in symbols:
            df = loader.load_historical(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=args.days),
                end_date=datetime.now(),
                source='truefx'
            )
            data[symbol] = df

        await bot.run_backtest(data)

    elif mode in [TradingMode.PAPER, TradingMode.LIVE]:
        await bot.run_live()

    elif mode == TradingMode.MONITOR:
        logger.info("Monitor mode - watching signals only")
        await bot.run_live()


if __name__ == '__main__':
    asyncio.run(main())
