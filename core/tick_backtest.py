"""
Tick-Level Backtesting Engine for HFT
======================================
Provides realistic tick-by-tick simulation with:
- Order book state tracking
- Queue position modeling
- Latency simulation
- Realistic slippage

Inspired by HftBacktest (https://github.com/nkaz001/hftbacktest)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from .order_book_l3 import OrderBookL3, BookUpdate, OrderType, Side as BookSide
from .queue_position import QueuePositionTracker, QueueOrder, Side
from .fill_probability import FillProbabilityEngine, SlippageEstimator

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status in backtest."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class BacktestOrder:
    """Order in backtesting engine."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float]
    timestamp: datetime

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fill_time: Optional[datetime] = None

    # Queue tracking
    queue_position: float = 0.0
    initial_queue_position: float = 0.0

    # Fees and slippage
    slippage_bps: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestFill:
    """Fill event in backtest."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    slippage_bps: float
    commission: float
    is_maker: bool


@dataclass
class BacktestTrade:
    """Trade (position change) in backtest."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_bps: float = 0.0


@dataclass
class TickData:
    """Single tick of market data."""
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float = 1.0
    ask_size: float = 1.0
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    is_trade: bool = False


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_per_lot: float = 7.0  # USD per 100k lot
    slippage_model: str = "realistic"  # "none", "fixed", "realistic"
    fixed_slippage_bps: float = 0.5
    latency_ms: float = 50.0  # Simulated latency
    fill_model: str = "queue"  # "instant", "queue", "probabilistic"
    max_position: float = 10.0  # Max lots
    lot_size: float = 100000.0  # Forex lot size


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def on_tick(self, tick: TickData, engine: 'TickBacktestEngine') -> None:
        """Called on each tick."""
        pass

    @abstractmethod
    def on_fill(self, fill: BacktestFill, engine: 'TickBacktestEngine') -> None:
        """Called when order is filled."""
        pass

    def on_start(self, engine: 'TickBacktestEngine') -> None:
        """Called at backtest start."""
        pass

    def on_end(self, engine: 'TickBacktestEngine') -> None:
        """Called at backtest end."""
        pass


class TickBacktestEngine:
    """
    Tick-Level Backtesting Engine.

    Features:
    - Tick-by-tick simulation
    - Order book reconstruction
    - Queue position tracking
    - Realistic fill simulation
    - Latency modeling
    - PnL tracking

    Usage:
        engine = TickBacktestEngine(config)
        engine.load_data("EURUSD", tick_data)
        engine.run(strategy)
        results = engine.get_results()
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()

        # Market data
        self.symbols: List[str] = []
        self.tick_data: Dict[str, pd.DataFrame] = {}
        self.current_tick: Dict[str, TickData] = {}

        # Order book simulation
        self.order_books: Dict[str, OrderBookL3] = {}
        self.queue_trackers: Dict[str, QueuePositionTracker] = {}
        self.fill_engine = FillProbabilityEngine()

        # Orders and fills
        self.orders: Dict[str, BacktestOrder] = {}
        self.pending_orders: List[str] = []
        self.fills: List[BacktestFill] = []

        # Position tracking
        self.positions: Dict[str, float] = defaultdict(float)  # symbol -> quantity
        self.avg_entry_prices: Dict[str, float] = defaultdict(float)
        self.trades: List[BacktestTrade] = []

        # Account
        self.cash = self.config.initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []

        # State
        self.current_time: Optional[datetime] = None
        self._order_counter = 0
        self._trade_counter = 0

    def load_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Load tick data for symbol.

        Expects DataFrame with columns:
        - timestamp: datetime
        - bid: float
        - ask: float
        - bid_size: float (optional)
        - ask_size: float (optional)
        - last_price: float (optional, for trades)
        - last_size: float (optional, for trades)
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        if 'bid' not in data.columns or 'ask' not in data.columns:
            raise ValueError("Data must have 'bid' and 'ask' columns")

        self.symbols.append(symbol)
        self.tick_data[symbol] = data.sort_values('timestamp').reset_index(drop=True)

        # Initialize order book
        tick_size = 0.0001 if 'JPY' not in symbol else 0.01
        self.order_books[symbol] = OrderBookL3(tick_size=tick_size)
        self.queue_trackers[symbol] = QueuePositionTracker(model='probabilistic')

        logger.info(f"Loaded {len(data)} ticks for {symbol}")

    def run(self, strategy: Strategy) -> None:
        """
        Run backtest with strategy.

        Args:
            strategy: Trading strategy to test
        """
        if not self.symbols:
            raise ValueError("No data loaded")

        logger.info(f"Starting backtest with {len(self.symbols)} symbols")

        # Merge all tick data into single timeline
        all_ticks = self._merge_tick_data()

        strategy.on_start(self)

        # Main backtest loop
        for _, row in all_ticks.iterrows():
            symbol = row['symbol']
            self.current_time = row['timestamp']

            # Create tick
            tick = TickData(
                timestamp=row['timestamp'],
                bid=row['bid'],
                ask=row['ask'],
                bid_size=row.get('bid_size', 1.0),
                ask_size=row.get('ask_size', 1.0),
                last_price=row.get('last_price'),
                last_size=row.get('last_size'),
                is_trade='last_price' in row and pd.notna(row.get('last_price'))
            )

            self.current_tick[symbol] = tick

            # Update order book
            self._update_order_book(symbol, tick)

            # Process pending orders
            self._process_orders(symbol, tick)

            # Call strategy
            strategy.on_tick(tick, self)

            # Update equity curve
            self._update_equity()

        strategy.on_end(self)

        logger.info(f"Backtest complete: {len(self.fills)} fills, {len(self.trades)} trades")

    def _merge_tick_data(self) -> pd.DataFrame:
        """Merge tick data from all symbols into timeline."""
        frames = []
        for symbol, df in self.tick_data.items():
            df = df.copy()
            df['symbol'] = symbol
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values('timestamp').reset_index(drop=True)

    def _update_order_book(self, symbol: str, tick: TickData) -> None:
        """Update order book from tick."""
        book = self.order_books[symbol]

        # Update L2 quotes
        book._update_level2_bid(tick.timestamp, tick.bid, tick.bid_size)
        book._update_level2_ask(tick.timestamp, tick.ask, tick.ask_size)

        # Process trade if present
        if tick.is_trade and tick.last_price and tick.last_size:
            # Determine trade side (at ask = buy, at bid = sell)
            if abs(tick.last_price - tick.ask) < abs(tick.last_price - tick.bid):
                trade_side = BookSide.BUY
            else:
                trade_side = BookSide.SELL

            update = BookUpdate(
                timestamp=tick.timestamp,
                update_type=OrderType.TRADE,
                side=trade_side,
                price=tick.last_price,
                quantity=tick.last_size
            )
            book.process_update(update)

            # Update fill probability engine
            self.fill_engine.record_trade(tick.timestamp, tick.last_size)

            # Advance queues
            tracker = self.queue_trackers[symbol]
            side = Side.BUY if trade_side == BookSide.BUY else Side.SELL
            tracker.on_trade(tick.last_price, tick.last_size, side)

    def _process_orders(self, symbol: str, tick: TickData) -> None:
        """Process pending orders against current tick."""
        orders_to_remove = []

        for order_id in self.pending_orders:
            order = self.orders.get(order_id)
            if not order or order.symbol != symbol:
                continue

            filled = False

            if order.order_type == OrderType.MARKET:
                # Market orders fill immediately
                filled = self._fill_market_order(order, tick)

            elif order.order_type == OrderType.LIMIT:
                # Limit orders check price and queue
                filled = self._check_limit_fill(order, tick)

            if filled:
                orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            if order_id in self.pending_orders:
                self.pending_orders.remove(order_id)

    def _fill_market_order(self, order: BacktestOrder, tick: TickData) -> bool:
        """Fill market order."""
        if order.side == OrderSide.BUY:
            fill_price = tick.ask
        else:
            fill_price = tick.bid

        # Apply slippage
        slippage = self._calculate_slippage(order, tick)
        if order.side == OrderSide.BUY:
            fill_price *= (1 + slippage / 10000)
        else:
            fill_price *= (1 - slippage / 10000)

        # Create fill
        self._create_fill(order, fill_price, order.quantity, tick.timestamp, slippage, is_maker=False)

        return True

    def _check_limit_fill(self, order: BacktestOrder, tick: TickData) -> bool:
        """Check if limit order fills."""
        if order.limit_price is None:
            return False

        # Check price crossing
        if order.side == OrderSide.BUY:
            # Buy limit: fills if ask <= limit price
            if tick.ask > order.limit_price:
                return False
            fill_price = order.limit_price
        else:
            # Sell limit: fills if bid >= limit price
            if tick.bid < order.limit_price:
                return False
            fill_price = order.limit_price

        # Queue-based fill model
        if self.config.fill_model == "queue":
            tracker = self.queue_trackers[order.symbol]
            fill_prob = tracker.get_fill_probability(order.order_id)

            # Only fill if probability high enough
            if fill_prob < 0.5 and np.random.random() > fill_prob:
                return False

        # Create fill
        self._create_fill(order, fill_price, order.quantity, tick.timestamp, 0.0, is_maker=True)

        return True

    def _calculate_slippage(self, order: BacktestOrder, tick: TickData) -> float:
        """Calculate slippage in basis points."""
        if self.config.slippage_model == "none":
            return 0.0

        if self.config.slippage_model == "fixed":
            return self.config.fixed_slippage_bps

        # Realistic slippage based on order size and spread
        spread = (tick.ask - tick.bid) / ((tick.ask + tick.bid) / 2) * 10000
        size_impact = 0.1 * order.quantity / 100  # Scale by lots

        return 0.5 + spread * 0.2 + size_impact

    def _create_fill(self, order: BacktestOrder, fill_price: float, quantity: float,
                     timestamp: datetime, slippage: float, is_maker: bool) -> None:
        """Create fill and update position."""
        # Calculate commission
        commission = self.config.commission_per_lot * quantity

        # Apply maker rebate / taker fee
        if is_maker:
            commission *= 0.5  # Maker rebate

        fill = BacktestFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            timestamp=timestamp,
            slippage_bps=slippage,
            commission=commission,
            is_maker=is_maker
        )

        self.fills.append(fill)

        # Update order status
        order.filled_quantity = quantity
        order.filled_price = fill_price
        order.fill_time = timestamp
        order.slippage_bps = slippage
        order.commission = commission
        order.status = OrderStatus.FILLED

        # Update position
        self._update_position(fill)

        logger.debug(f"Fill: {order.side.value} {quantity} {order.symbol} @ {fill_price:.5f}")

    def _update_position(self, fill: BacktestFill) -> None:
        """Update position from fill."""
        symbol = fill.symbol
        prev_pos = self.positions[symbol]
        prev_avg = self.avg_entry_prices[symbol]

        if fill.side == OrderSide.BUY:
            new_pos = prev_pos + fill.quantity
            if new_pos > 0:
                # Calculate new average entry
                total_cost = prev_pos * prev_avg + fill.quantity * fill.price
                self.avg_entry_prices[symbol] = total_cost / new_pos if new_pos != 0 else 0
        else:
            new_pos = prev_pos - fill.quantity
            if new_pos < 0:
                # Short position, update average
                total_cost = abs(prev_pos) * prev_avg + fill.quantity * fill.price
                self.avg_entry_prices[symbol] = total_cost / abs(new_pos) if new_pos != 0 else 0

        # Check for position close (PnL realization)
        if prev_pos != 0 and np.sign(new_pos) != np.sign(prev_pos):
            # Position flipped or closed
            closed_qty = min(abs(prev_pos), fill.quantity)
            self._record_trade_close(symbol, fill.side, closed_qty, fill.price, fill.timestamp)

        elif prev_pos != 0 and abs(new_pos) < abs(prev_pos):
            # Partial close
            closed_qty = abs(prev_pos) - abs(new_pos)
            self._record_trade_close(symbol, fill.side, closed_qty, fill.price, fill.timestamp)

        self.positions[symbol] = new_pos

        # Deduct commission
        self.cash -= fill.commission

    def _record_trade_close(self, symbol: str, side: OrderSide, quantity: float,
                            exit_price: float, exit_time: datetime) -> None:
        """Record closed trade with PnL."""
        entry_price = self.avg_entry_prices[symbol]

        if side == OrderSide.SELL:
            # Was long, now selling
            pnl = (exit_price - entry_price) * quantity * self.config.lot_size
        else:
            # Was short, now buying
            pnl = (entry_price - exit_price) * quantity * self.config.lot_size

        pnl_bps = ((exit_price / entry_price) - 1) * 10000 if entry_price != 0 else 0

        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=f"trade_{self._trade_counter}",
            symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.SELL else OrderSide.BUY,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=exit_time,  # Approximate
            exit_price=exit_price,
            exit_time=exit_time,
            pnl=pnl,
            pnl_bps=pnl_bps
        )

        self.trades.append(trade)
        self.cash += pnl

    def _update_equity(self) -> None:
        """Update equity curve."""
        # Mark-to-market PnL
        mtm_pnl = 0.0

        for symbol, position in self.positions.items():
            if position == 0:
                continue

            tick = self.current_tick.get(symbol)
            if tick:
                mid = (tick.bid + tick.ask) / 2
                entry = self.avg_entry_prices[symbol]
                if position > 0:
                    mtm_pnl += (mid - entry) * abs(position) * self.config.lot_size
                else:
                    mtm_pnl += (entry - mid) * abs(position) * self.config.lot_size

        equity = self.cash + mtm_pnl
        self.equity_curve.append((self.current_time, equity))

    # ==================== ORDER SUBMISSION ====================

    def submit_order(self, symbol: str, side: OrderSide, quantity: float,
                     order_type: OrderType = OrderType.MARKET,
                     limit_price: Optional[float] = None) -> str:
        """
        Submit order to backtest engine.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity (in lots)
            order_type: MARKET or LIMIT
            limit_price: Limit price (required for LIMIT orders)

        Returns:
            Order ID
        """
        # Validate
        if symbol not in self.symbols:
            raise ValueError(f"Unknown symbol: {symbol}")

        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")

        # Check position limits
        current_pos = self.positions[symbol]
        if side == OrderSide.BUY:
            new_pos = current_pos + quantity
        else:
            new_pos = current_pos - quantity

        if abs(new_pos) > self.config.max_position:
            logger.warning(f"Order would exceed max position, rejecting")
            return None

        # Create order
        self._order_counter += 1
        order_id = f"order_{self._order_counter}"

        order = BacktestOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            timestamp=self.current_time,
            status=OrderStatus.SUBMITTED
        )

        # For limit orders, track queue position
        if order_type == OrderType.LIMIT:
            book = self.order_books[symbol]
            queue_side = Side.BUY if side == OrderSide.BUY else Side.SELL
            order.queue_position = book.estimate_queue_ahead(limit_price, queue_side)
            order.initial_queue_position = order.queue_position

            # Add to queue tracker
            tracker = self.queue_trackers[symbol]
            tracker.add_order(order_id, limit_price, quantity, queue_side, order.queue_position)

        self.orders[order_id] = order
        self.pending_orders.append(order_id)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.SUBMITTED:
            return False

        order.status = OrderStatus.CANCELLED

        if order_id in self.pending_orders:
            self.pending_orders.remove(order_id)

        # Remove from queue tracker
        tracker = self.queue_trackers.get(order.symbol)
        if tracker:
            tracker.remove_order(order_id)

        return True

    # ==================== ACCESSORS ====================

    def get_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0.0)

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get current mid price."""
        tick = self.current_tick.get(symbol)
        if tick:
            return (tick.bid + tick.ask) / 2
        return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread."""
        tick = self.current_tick.get(symbol)
        if tick:
            return tick.ask - tick.bid
        return None

    def get_book_imbalance(self, symbol: str) -> float:
        """Get order book imbalance."""
        book = self.order_books.get(symbol)
        if book:
            return book.get_imbalance()
        return 0.0

    # ==================== RESULTS ====================

    def get_results(self) -> Dict[str, Any]:
        """Get backtest results summary."""
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])

        # Calculate metrics
        initial = self.config.initial_capital
        final = equity_df['equity'].iloc[-1]
        returns = equity_df['equity'].pct_change().dropna()

        total_return = (final / initial - 1) * 100
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if len(returns) > 1 else 0

        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # Max drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_dd = drawdown.min() * 100

        # Average trade metrics
        avg_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        avg_pnl_bps = np.mean([t.pnl_bps for t in self.trades]) if self.trades else 0

        return {
            'initial_capital': initial,
            'final_equity': final,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_trades': len(self.trades),
            'total_fills': len(self.fills),
            'win_rate_pct': win_rate,
            'avg_pnl_per_trade': avg_pnl,
            'avg_pnl_bps': avg_pnl_bps,
            'total_commission': sum(f.commission for f in self.fills),
            'equity_curve': equity_df,
            'trades': self.trades,
            'fills': self.fills
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        data = []
        for trade in self.trades:
            data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_bps': trade.pnl_bps,
                'exit_time': trade.exit_time
            })
        return pd.DataFrame(data)


# Example strategy for testing
class SimpleMovingAverageStrategy(Strategy):
    """Simple MA crossover strategy for testing."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: Dict[str, List[float]] = defaultdict(list)

    def on_tick(self, tick: TickData, engine: TickBacktestEngine) -> None:
        # Get symbol from current tick
        for symbol, t in engine.current_tick.items():
            if t == tick:
                self.prices[symbol].append((t.bid + t.ask) / 2)

                # Need enough history
                if len(self.prices[symbol]) < self.slow_period:
                    return

                # Calculate MAs
                fast_ma = np.mean(self.prices[symbol][-self.fast_period:])
                slow_ma = np.mean(self.prices[symbol][-self.slow_period:])

                position = engine.get_position(symbol)

                # Simple crossover logic
                if fast_ma > slow_ma and position <= 0:
                    # Buy signal
                    if position < 0:
                        engine.submit_order(symbol, OrderSide.BUY, abs(position))
                    engine.submit_order(symbol, OrderSide.BUY, 0.1)

                elif fast_ma < slow_ma and position >= 0:
                    # Sell signal
                    if position > 0:
                        engine.submit_order(symbol, OrderSide.SELL, position)
                    engine.submit_order(symbol, OrderSide.SELL, 0.1)

    def on_fill(self, fill: BacktestFill, engine: TickBacktestEngine) -> None:
        pass


if __name__ == '__main__':
    print("Tick Backtest Engine Test")
    print("=" * 50)

    # Generate synthetic tick data
    np.random.seed(42)
    n_ticks = 1000

    timestamps = pd.date_range('2026-01-01', periods=n_ticks, freq='1s')
    prices = 1.1000 + np.cumsum(np.random.randn(n_ticks) * 0.0001)

    tick_data = pd.DataFrame({
        'timestamp': timestamps,
        'bid': prices - 0.00005,
        'ask': prices + 0.00005,
        'bid_size': np.random.uniform(1, 10, n_ticks),
        'ask_size': np.random.uniform(1, 10, n_ticks)
    })

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        slippage_model="realistic",
        fill_model="queue"
    )

    engine = TickBacktestEngine(config)
    engine.load_data("EURUSD", tick_data)

    strategy = SimpleMovingAverageStrategy(fast_period=5, slow_period=20)
    engine.run(strategy)

    # Print results
    results = engine.get_results()
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"  Avg PnL/Trade: ${results['avg_pnl_per_trade']:.2f}")
    print(f"  Total Commission: ${results['total_commission']:.2f}")
