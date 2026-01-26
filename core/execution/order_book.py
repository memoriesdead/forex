"""
Order Book Level 2/Level 3 Processing for HFT
==============================================
Implements order book reconstruction, imbalance signals, and queue position estimation.

Based on:
- NautilusTrader order book architecture
- HftBacktest Level-3 FIFO model
- Chinese quant OFI/OEI research

Source: https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Side(Enum):
    """Order side."""
    BUY = 1
    SELL = -1


class OrderType(Enum):
    """Order book update types."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    TRADE = "trade"


@dataclass
class Order:
    """Single order in the book."""
    order_id: str
    price: float
    quantity: float
    side: Side
    timestamp: datetime
    is_own: bool = False  # Track our own orders


@dataclass
class PriceLevel:
    """Orders at a single price level."""
    price: float
    orders: List[Order] = field(default_factory=list)

    @property
    def total_quantity(self) -> float:
        return sum(o.quantity for o in self.orders)

    @property
    def order_count(self) -> int:
        return len(self.orders)

    def add_order(self, order: Order) -> int:
        """Add order, return queue position."""
        self.orders.append(order)
        return len(self.orders) - 1

    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order by ID."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                return self.orders.pop(i)
        return None

    def get_queue_position(self, order_id: str) -> int:
        """Get queue position (0 = front)."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                return i
        return -1


@dataclass
class BookUpdate:
    """Order book update event."""
    timestamp: datetime
    update_type: OrderType
    side: Side
    price: float
    quantity: float
    order_id: Optional[str] = None


class OrderBookL3:
    """
    Level 3 Order Book with queue position tracking.

    Features:
    - Full order-by-order book reconstruction
    - Queue position estimation
    - Order book imbalance signals
    - Depth ratio signals
    - Fill probability estimation

    Usage:
        book = OrderBookL3(tick_size=0.0001)
        book.process_update(update)
        imbalance = book.get_imbalance()
        queue_pos = book.estimate_queue_position(1.1000, Side.BUY)
    """

    def __init__(self, tick_size: float = 0.0001, max_levels: int = 20):
        """
        Initialize order book.

        Args:
            tick_size: Minimum price increment (0.0001 for forex = 1 pip)
            max_levels: Maximum depth levels to track
        """
        self.tick_size = tick_size
        self.max_levels = max_levels

        # Order book: price -> PriceLevel
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}

        # Order index for fast lookup
        self.order_index: Dict[str, Tuple[Side, float]] = {}

        # Trade history for queue advancement
        self.recent_trades: List[BookUpdate] = []
        self.max_trade_history = 1000

        # Statistics
        self.update_count = 0
        self.last_update_time: Optional[datetime] = None

    def process_update(self, update: BookUpdate) -> None:
        """Process a single order book update."""
        self.update_count += 1
        self.last_update_time = update.timestamp

        if update.update_type == OrderType.ADD:
            self._add_order(update)
        elif update.update_type == OrderType.MODIFY:
            self._modify_order(update)
        elif update.update_type == OrderType.DELETE:
            self._delete_order(update)
        elif update.update_type == OrderType.TRADE:
            self._process_trade(update)

    def _add_order(self, update: BookUpdate) -> None:
        """Add new order to book."""
        book = self.bids if update.side == Side.BUY else self.asks
        price = self._round_price(update.price)

        if price not in book:
            book[price] = PriceLevel(price=price)

        order = Order(
            order_id=update.order_id or f"order_{self.update_count}",
            price=price,
            quantity=update.quantity,
            side=update.side,
            timestamp=update.timestamp
        )

        book[price].add_order(order)
        self.order_index[order.order_id] = (update.side, price)

    def _modify_order(self, update: BookUpdate) -> None:
        """Modify existing order quantity."""
        if update.order_id not in self.order_index:
            return

        side, price = self.order_index[update.order_id]
        book = self.bids if side == Side.BUY else self.asks

        if price in book:
            for order in book[price].orders:
                if order.order_id == update.order_id:
                    order.quantity = update.quantity
                    break

    def _delete_order(self, update: BookUpdate) -> None:
        """Remove order from book."""
        if update.order_id not in self.order_index:
            return

        side, price = self.order_index[update.order_id]
        book = self.bids if side == Side.BUY else self.asks

        if price in book:
            book[price].remove_order(update.order_id)
            if book[price].order_count == 0:
                del book[price]

        del self.order_index[update.order_id]

    def _process_trade(self, update: BookUpdate) -> None:
        """Process trade (removes liquidity from book)."""
        # Trade at ask = buy aggressor, remove from asks
        # Trade at bid = sell aggressor, remove from bids
        book = self.asks if update.side == Side.BUY else self.bids
        price = self._round_price(update.price)

        self.recent_trades.append(update)
        if len(self.recent_trades) > self.max_trade_history:
            self.recent_trades.pop(0)

        if price in book:
            remaining = update.quantity
            level = book[price]

            # Remove orders FIFO
            while remaining > 0 and level.orders:
                front_order = level.orders[0]
                if front_order.quantity <= remaining:
                    remaining -= front_order.quantity
                    level.orders.pop(0)
                    if front_order.order_id in self.order_index:
                        del self.order_index[front_order.order_id]
                else:
                    front_order.quantity -= remaining
                    remaining = 0

            if level.order_count == 0:
                del book[price]

    def _round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size

    # ==================== GETTERS ====================

    def get_best_bid(self) -> Optional[float]:
        """Get best (highest) bid price."""
        if not self.bids:
            return None
        return max(self.bids.keys())

    def get_best_ask(self) -> Optional[float]:
        """Get best (lowest) ask price."""
        if not self.asks:
            return None
        return min(self.asks.keys())

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        spread = self.get_spread()
        mid = self.get_mid_price()
        if spread is None or mid is None or mid == 0:
            return None
        return (spread / mid) * 10000

    # ==================== SIGNALS ====================

    def get_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Imbalance ratio: (bid_qty - ask_qty) / (bid_qty + ask_qty)
            Range: -1 (all asks) to +1 (all bids)
            Positive = buying pressure, Negative = selling pressure
        """
        bid_qty = self._get_total_quantity(self.bids, levels)
        ask_qty = self._get_total_quantity(self.asks, levels)

        total = bid_qty + ask_qty
        if total == 0:
            return 0.0

        return (bid_qty - ask_qty) / total

    def get_depth_ratio(self, levels: int = 10) -> float:
        """
        Calculate depth ratio.

        Returns:
            Ratio: bid_depth / ask_depth
            >1 = more buy interest, <1 = more sell interest
        """
        bid_qty = self._get_total_quantity(self.bids, levels)
        ask_qty = self._get_total_quantity(self.asks, levels)

        if ask_qty == 0:
            return float('inf') if bid_qty > 0 else 1.0

        return bid_qty / ask_qty

    def get_weighted_mid_price(self, levels: int = 5) -> Optional[float]:
        """
        Calculate volume-weighted mid price.

        More accurate than simple mid when book is imbalanced.
        """
        if not self.bids or not self.asks:
            return None

        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]

        if not bid_prices or not ask_prices:
            return None

        bid_qty = sum(self.bids[p].total_quantity for p in bid_prices)
        ask_qty = sum(self.asks[p].total_quantity for p in ask_prices)

        if bid_qty + ask_qty == 0:
            return (bid_prices[0] + ask_prices[0]) / 2

        # Weight by opposing side (more ask volume = price closer to bid)
        wmid = (bid_prices[0] * ask_qty + ask_prices[0] * bid_qty) / (bid_qty + ask_qty)
        return wmid

    def get_microprice(self) -> Optional[float]:
        """
        Calculate microprice (imbalance-adjusted mid).

        Better predictor of next trade direction than mid price.
        Source: Gatheral & Oomen (2010)
        """
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid is None or ask is None:
            return None

        bid_qty = self.bids[bid].total_quantity if bid in self.bids else 0
        ask_qty = self.asks[ask].total_quantity if ask in self.asks else 0

        total = bid_qty + ask_qty
        if total == 0:
            return (bid + ask) / 2

        # Microprice: weighted by opposite side quantity
        return (bid * ask_qty + ask * bid_qty) / total

    def _get_total_quantity(self, book: Dict[float, PriceLevel], levels: int) -> float:
        """Get total quantity across N levels."""
        if not book:
            return 0.0

        if book is self.bids:
            prices = sorted(book.keys(), reverse=True)[:levels]
        else:
            prices = sorted(book.keys())[:levels]

        return sum(book[p].total_quantity for p in prices)

    # ==================== QUEUE POSITION ====================

    def estimate_queue_position(self, price: float, side: Side) -> int:
        """
        Estimate queue position if order placed at price.

        Args:
            price: Limit order price
            side: BUY or SELL

        Returns:
            Estimated queue position (0 = front of queue)
        """
        price = self._round_price(price)
        book = self.bids if side == Side.BUY else self.asks

        if price not in book:
            return 0  # Would be first at this level

        return book[price].order_count

    def estimate_queue_ahead(self, price: float, side: Side) -> float:
        """
        Estimate total quantity ahead in queue.

        Returns:
            Total quantity that must trade before our order fills
        """
        price = self._round_price(price)
        book = self.bids if side == Side.BUY else self.asks

        if price not in book:
            return 0.0

        return book[price].total_quantity

    def get_own_queue_position(self, order_id: str) -> int:
        """Get queue position of our own order."""
        if order_id not in self.order_index:
            return -1

        side, price = self.order_index[order_id]
        book = self.bids if side == Side.BUY else self.asks

        if price not in book:
            return -1

        return book[price].get_queue_position(order_id)

    # ==================== SNAPSHOT ====================

    def get_snapshot(self, levels: int = 10) -> Dict:
        """
        Get order book snapshot.

        Returns:
            Dict with bids, asks, and summary statistics
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]

        return {
            'timestamp': self.last_update_time,
            'bids': [
                {'price': p, 'quantity': self.bids[p].total_quantity, 'orders': self.bids[p].order_count}
                for p in bid_prices
            ],
            'asks': [
                {'price': p, 'quantity': self.asks[p].total_quantity, 'orders': self.asks[p].order_count}
                for p in ask_prices
            ],
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'mid': self.get_mid_price(),
            'spread': self.get_spread(),
            'imbalance': self.get_imbalance(levels),
            'depth_ratio': self.get_depth_ratio(levels),
            'microprice': self.get_microprice(),
        }

    def to_dataframe(self, levels: int = 10) -> pd.DataFrame:
        """Convert order book to DataFrame."""
        snapshot = self.get_snapshot(levels)

        # Create bid/ask columns
        data = {'level': list(range(levels))}

        for i in range(levels):
            if i < len(snapshot['bids']):
                data[f'bid_price_{i}'] = snapshot['bids'][i]['price']
                data[f'bid_qty_{i}'] = snapshot['bids'][i]['quantity']
            if i < len(snapshot['asks']):
                data[f'ask_price_{i}'] = snapshot['asks'][i]['price']
                data[f'ask_qty_{i}'] = snapshot['asks'][i]['quantity']

        return pd.DataFrame([data])

    # ==================== RECONSTRUCTION ====================

    @classmethod
    def from_ticks(cls, ticks: pd.DataFrame, tick_size: float = 0.0001) -> 'OrderBookL3':
        """
        Reconstruct order book from tick data.

        Expects DataFrame with columns:
        - timestamp: datetime
        - type: 'trade' or 'quote'
        - side: 'buy' or 'sell' (for trades)
        - price: float
        - bid: float (for quotes)
        - ask: float (for quotes)
        - bid_size: float (for quotes)
        - ask_size: float (for quotes)
        - size: float (for trades)
        """
        book = cls(tick_size=tick_size)

        for _, row in ticks.iterrows():
            if row.get('type') == 'quote' or 'bid' in row:
                # Process quote update
                if pd.notna(row.get('bid')):
                    book._update_level2_bid(row['timestamp'], row['bid'], row.get('bid_size', 1.0))
                if pd.notna(row.get('ask')):
                    book._update_level2_ask(row['timestamp'], row['ask'], row.get('ask_size', 1.0))

            elif row.get('type') == 'trade' or 'size' in row:
                # Process trade
                side = Side.BUY if row.get('side', 'buy').lower() == 'buy' else Side.SELL
                update = BookUpdate(
                    timestamp=row['timestamp'],
                    update_type=OrderType.TRADE,
                    side=side,
                    price=row['price'],
                    quantity=row.get('size', row.get('volume', 1.0))
                )
                book.process_update(update)

        return book

    def _update_level2_bid(self, timestamp: datetime, price: float, quantity: float) -> None:
        """Update Level 2 bid (aggregate at price level)."""
        price = self._round_price(price)

        # Clear old best bid if price changed
        old_best = self.get_best_bid()
        if old_best and old_best != price:
            if old_best in self.bids:
                for order in self.bids[old_best].orders:
                    if order.order_id in self.order_index:
                        del self.order_index[order.order_id]
                del self.bids[old_best]

        # Update new level
        if quantity > 0:
            if price not in self.bids:
                self.bids[price] = PriceLevel(price=price)

            # Replace with single aggregate order
            self.bids[price].orders = [Order(
                order_id=f"l2_bid_{price}",
                price=price,
                quantity=quantity,
                side=Side.BUY,
                timestamp=timestamp
            )]
        elif price in self.bids:
            del self.bids[price]

    def _update_level2_ask(self, timestamp: datetime, price: float, quantity: float) -> None:
        """Update Level 2 ask (aggregate at price level)."""
        price = self._round_price(price)

        # Clear old best ask if price changed
        old_best = self.get_best_ask()
        if old_best and old_best != price:
            if old_best in self.asks:
                for order in self.asks[old_best].orders:
                    if order.order_id in self.order_index:
                        del self.order_index[order.order_id]
                del self.asks[old_best]

        # Update new level
        if quantity > 0:
            if price not in self.asks:
                self.asks[price] = PriceLevel(price=price)

            # Replace with single aggregate order
            self.asks[price].orders = [Order(
                order_id=f"l2_ask_{price}",
                price=price,
                quantity=quantity,
                side=Side.SELL,
                timestamp=timestamp
            )]
        elif price in self.asks:
            del self.asks[price]


class OrderBookSignals:
    """
    Generate trading signals from order book data.

    Signals include:
    - Order book imbalance (OBI)
    - Depth ratio
    - Microprice momentum
    - Spread regime
    - Trade flow imbalance
    """

    def __init__(self, lookback: int = 100):
        """
        Initialize signal generator.

        Args:
            lookback: Number of snapshots for rolling calculations
        """
        self.lookback = lookback
        self.snapshots: List[Dict] = []
        self.trade_history: List[BookUpdate] = []

    def update(self, book: OrderBookL3) -> Dict[str, float]:
        """
        Update with new book state and generate signals.

        Returns:
            Dict of signal values
        """
        snapshot = book.get_snapshot()
        self.snapshots.append(snapshot)

        if len(self.snapshots) > self.lookback:
            self.snapshots.pop(0)

        # Store recent trades
        for trade in book.recent_trades[-10:]:
            if trade not in self.trade_history:
                self.trade_history.append(trade)

        if len(self.trade_history) > self.lookback * 10:
            self.trade_history = self.trade_history[-self.lookback * 5:]

        return self.generate_signals()

    def generate_signals(self) -> Dict[str, float]:
        """Generate all order book signals."""
        if len(self.snapshots) < 2:
            return {}

        current = self.snapshots[-1]

        signals = {
            # Current state signals
            'obi': current.get('imbalance', 0),
            'depth_ratio': current.get('depth_ratio', 1),
            'spread_bps': (current.get('spread', 0) / current.get('mid', 1)) * 10000 if current.get('mid') else 0,

            # Microprice vs mid
            'microprice_bias': self._microprice_bias(current),

            # Momentum signals
            'obi_momentum': self._obi_momentum(),
            'microprice_momentum': self._microprice_momentum(),

            # Trade flow
            'trade_flow_imbalance': self._trade_flow_imbalance(),

            # Spread regime
            'spread_percentile': self._spread_percentile(),

            # Combined signal
            'book_signal': self._combined_signal(),
        }

        return signals

    def _microprice_bias(self, snapshot: Dict) -> float:
        """Calculate microprice bias from mid."""
        mid = snapshot.get('mid')
        microprice = snapshot.get('microprice')

        if mid is None or microprice is None or mid == 0:
            return 0.0

        return (microprice - mid) / mid * 10000  # In basis points

    def _obi_momentum(self) -> float:
        """Calculate OBI momentum (change in imbalance)."""
        if len(self.snapshots) < 5:
            return 0.0

        recent = [s.get('imbalance', 0) for s in self.snapshots[-5:]]
        older = [s.get('imbalance', 0) for s in self.snapshots[-10:-5]] if len(self.snapshots) >= 10 else recent

        return np.mean(recent) - np.mean(older)

    def _microprice_momentum(self) -> float:
        """Calculate microprice momentum."""
        if len(self.snapshots) < 5:
            return 0.0

        microprices = [s.get('microprice', s.get('mid', 0)) for s in self.snapshots[-10:]]
        microprices = [m for m in microprices if m is not None and m > 0]

        if len(microprices) < 2:
            return 0.0

        # Return change in basis points
        return (microprices[-1] / microprices[0] - 1) * 10000

    def _trade_flow_imbalance(self) -> float:
        """Calculate trade flow imbalance."""
        if len(self.trade_history) < 10:
            return 0.0

        recent = self.trade_history[-20:]

        buy_volume = sum(t.quantity for t in recent if t.side == Side.BUY)
        sell_volume = sum(t.quantity for t in recent if t.side == Side.SELL)

        total = buy_volume + sell_volume
        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def _spread_percentile(self) -> float:
        """Calculate current spread percentile vs history."""
        if len(self.snapshots) < 10:
            return 50.0

        spreads = [s.get('spread', 0) for s in self.snapshots]
        current = spreads[-1]

        if current is None or current == 0:
            return 50.0

        percentile = sum(1 for s in spreads[:-1] if s < current) / len(spreads[:-1]) * 100
        return percentile

    def _combined_signal(self) -> float:
        """
        Generate combined book signal.

        Range: -1 (strong sell) to +1 (strong buy)
        """
        if len(self.snapshots) < 5:
            return 0.0

        current = self.snapshots[-1]

        # Weights for different components
        obi = current.get('imbalance', 0) * 0.3
        microprice_bias = self._microprice_bias(current) / 10 * 0.2  # Normalize
        obi_mom = self._obi_momentum() * 0.2
        trade_flow = self._trade_flow_imbalance() * 0.3

        signal = obi + microprice_bias + obi_mom + trade_flow

        # Clip to [-1, 1]
        return max(-1, min(1, signal))


if __name__ == '__main__':
    # Test order book
    print("Order Book L3 Test")
    print("=" * 50)

    book = OrderBookL3(tick_size=0.0001)

    # Simulate some orders
    from datetime import datetime
    now = datetime.now()

    # Add bids
    for i, (price, qty) in enumerate([(1.0995, 100), (1.0994, 200), (1.0993, 150)]):
        book.process_update(BookUpdate(
            timestamp=now,
            update_type=OrderType.ADD,
            side=Side.BUY,
            price=price,
            quantity=qty,
            order_id=f"bid_{i}"
        ))

    # Add asks
    for i, (price, qty) in enumerate([(1.0996, 80), (1.0997, 120), (1.0998, 100)]):
        book.process_update(BookUpdate(
            timestamp=now,
            update_type=OrderType.ADD,
            side=Side.SELL,
            price=price,
            quantity=qty,
            order_id=f"ask_{i}"
        ))

    # Print snapshot
    snapshot = book.get_snapshot(levels=5)
    print(f"\nBest Bid: {snapshot['best_bid']}")
    print(f"Best Ask: {snapshot['best_ask']}")
    print(f"Mid Price: {snapshot['mid']:.5f}")
    print(f"Spread: {snapshot['spread']:.5f}")
    print(f"Imbalance: {snapshot['imbalance']:.3f}")
    print(f"Depth Ratio: {snapshot['depth_ratio']:.3f}")
    print(f"Microprice: {snapshot['microprice']:.5f}")

    print("\nBids:")
    for bid in snapshot['bids']:
        print(f"  {bid['price']:.5f}: {bid['quantity']} ({bid['orders']} orders)")

    print("\nAsks:")
    for ask in snapshot['asks']:
        print(f"  {ask['price']:.5f}: {ask['quantity']} ({ask['orders']} orders)")

    # Queue position
    print(f"\nQueue position at 1.0995 BUY: {book.estimate_queue_position(1.0995, Side.BUY)}")
    print(f"Queue ahead at 1.0995: {book.estimate_queue_ahead(1.0995, Side.BUY)}")

    # Simulate trade
    print("\n--- Simulating trade at 1.0996 (buy aggressor) ---")
    book.process_update(BookUpdate(
        timestamp=now,
        update_type=OrderType.TRADE,
        side=Side.BUY,
        price=1.0996,
        quantity=50
    ))

    snapshot = book.get_snapshot(levels=5)
    print(f"New Best Ask: {snapshot['best_ask']}")
    print(f"Ask qty at 1.0996: {book.asks.get(1.0996, PriceLevel(0)).total_quantity}")
