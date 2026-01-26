"""
Trading Bot
===========
Slim orchestrator for multi-symbol forex trading.
Delegates to specialized modules for signal, risk, execution.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
import logging
import threading

from core.symbol.registry import SymbolRegistry
from core.risk.portfolio import PortfolioRiskManager
from core.models.loader import ModelLoader
from .signal import SignalGenerator, Signal
from .position import PositionManager
from .executor import OrderExecutor, Order

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Multi-symbol trading bot orchestrator.

    Slim design (~120 lines) - delegates to specialized modules:
    - SymbolRegistry: Symbol configuration
    - SignalGenerator: ML predictions
    - PortfolioRiskManager: Risk checks
    - OrderExecutor: Order submission
    - PositionManager: Position tracking
    """

    def __init__(
        self,
        capital: float = 100000.0,
        symbols: List[str] = None,
        tier: str = None,
        mode: str = "paper",
        ib_host: str = "localhost",
        ib_port: int = 4001,
    ):
        """
        Initialize trading bot.

        Args:
            capital: Starting capital
            symbols: Specific symbols to trade (None = use tier)
            tier: Symbol tier (majors, crosses, exotics, all)
            mode: Trading mode (paper, live)
            ib_host: IB Gateway host
            ib_port: IB Gateway port
        """
        self.capital = capital
        self.mode = mode

        # Get symbols
        self.registry = SymbolRegistry.get()
        if symbols:
            self.symbols = symbols
        elif tier:
            self.symbols = [
                p.symbol for p in self.registry.get_enabled(tier=tier)
            ]
        else:
            self.symbols = [p.symbol for p in self.registry.get_enabled()]

        # Initialize modules
        self.model_loader = ModelLoader()
        self.signal_gen = SignalGenerator(self.model_loader)
        self.risk_manager = PortfolioRiskManager(capital)
        self.position_manager = PositionManager()
        self.executor = OrderExecutor(
            host=ib_host,
            port=ib_port,
            connect=(mode != "backtest")
        )

        # State
        self._running = False
        self._lock = threading.RLock()
        self._tick_count = 0
        self._signal_count = 0
        self._trade_count = 0

        # Feature engine (lazy loaded)
        self._feature_engine = None

        logger.info(
            f"TradingBot initialized: {len(self.symbols)} symbols, "
            f"mode={mode}, capital=${capital:,.0f}"
        )

    def _get_feature_engine(self):
        """Lazy load feature engine."""
        if self._feature_engine is None:
            from core.features.engine import HFTFeatureEngine
            self._feature_engine = HFTFeatureEngine()
        return self._feature_engine

    async def run(self):
        """Main trading loop."""
        self._running = True
        logger.info(f"Starting trading loop for {self.symbols}")

        # Preload models for active symbols
        available = self.model_loader.get_available()
        to_preload = [s for s in self.symbols if s in available]
        if to_preload:
            self.model_loader.preload(to_preload[:10])  # Preload up to 10

        try:
            # In real implementation, this would stream ticks
            # For now, a simple polling loop
            while self._running:
                await asyncio.sleep(0.1)  # 100ms tick interval

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        finally:
            self._running = False

    def process_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float = 0.0,
        timestamp: datetime = None
    ):
        """
        Process a market tick.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            volume: Tick volume
            timestamp: Tick timestamp
        """
        if not self._running:
            return

        self._tick_count += 1
        timestamp = timestamp or datetime.now()
        mid_price = (bid + ask) / 2
        spread_pips = (ask - bid) * 10000

        # 1. Update position P&L
        self.position_manager.update_prices({symbol: mid_price})

        # 2. Generate features
        engine = self._get_feature_engine()
        features = engine.process_tick(
            symbol=symbol,
            bid=bid,
            ask=ask,
            volume=volume,
            timestamp=timestamp
        )

        if not features:
            return

        # 3. Get ML signal
        signal = self.signal_gen.predict(symbol, features)
        if signal is None or not signal.is_valid:
            return

        self._signal_count += 1

        # 4. Risk check
        can_trade, reason = self.risk_manager.can_trade(symbol, spread_pips)
        if not can_trade:
            logger.debug(f"Risk blocked {symbol}: {reason}")
            return

        # 5. Check confidence threshold
        config = self.registry.get_config(symbol)
        min_conf = config.get('min_confidence', 0.15)
        if signal.confidence < min_conf:
            return

        # 6. Calculate position size
        size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=signal.confidence,
            account_balance=self.capital
        )

        if size < 1000:  # Minimum size check
            return

        # 7. Check existing position
        position = self.position_manager.get_position(symbol)
        if position.is_open:
            # Already have a position - check if signal agrees
            if position.direction == signal.direction:
                return  # Same direction, don't add
            # Opposite direction - close existing
            self._close_position(symbol, mid_price)

        # 8. Execute trade
        self._execute_trade(symbol, signal, size, mid_price)

    def _execute_trade(
        self,
        symbol: str,
        signal: Signal,
        size: float,
        price: float
    ):
        """Execute a trade."""
        self._trade_count += 1

        def on_fill(order: Order):
            if order.status.value == "filled":
                self.position_manager.update_position(
                    symbol,
                    order.direction * order.filled_qty,
                    order.avg_fill_price
                )
                self.risk_manager.record_trade(symbol, order.avg_fill_price)

        self.executor.submit(
            symbol=symbol,
            direction=signal.direction,
            quantity=size,
            order_type="MKT",
            callback=on_fill
        )

        logger.info(
            f"Trade {symbol}: {'BUY' if signal.direction > 0 else 'SELL'} "
            f"{size:,.0f} @ {price:.5f} (conf={signal.confidence:.2%})"
        )

    def _close_position(self, symbol: str, price: float):
        """Close an existing position."""
        position = self.position_manager.get_position(symbol)
        if not position.is_open:
            return

        self.executor.submit(
            symbol=symbol,
            direction=-position.direction,
            quantity=abs(position.quantity),
            order_type="MKT"
        )

        logger.info(f"Closing {symbol} position: {position.quantity:,.0f}")

    def stop(self):
        """Stop the trading bot."""
        self._running = False
        self.executor.disconnect()
        logger.info("Trading bot stopped")

    def status(self) -> Dict[str, Any]:
        """Get bot status."""
        return {
            'running': self._running,
            'mode': self.mode,
            'symbols': self.symbols,
            'ticks_processed': self._tick_count,
            'signals_generated': self._signal_count,
            'trades_executed': self._trade_count,
            'positions': self.position_manager.summary(),
            'risk': self.risk_manager.summary(),
            'executor': self.executor.stats(),
            'models': self.model_loader.stats(),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
