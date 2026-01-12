"""
Queue Position Tracking and Fill Probability Models
=====================================================
Implements HftBacktest-style queue position estimation.

Three models implemented:
1. Risk-Averse Model - Conservative, order advances only on trades
2. Probabilistic Model - Statistical queue position estimation
3. Level-3 FIFO Model - Exact simulation with price-time priority

Source: https://hftbacktest.readthedocs.io/en/latest/tutorials/Probability%20Queue%20Models.html
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = 1
    SELL = -1


@dataclass
class QueueOrder:
    """Order in queue with position tracking."""
    order_id: str
    price: float
    quantity: float
    side: Side
    timestamp: datetime
    initial_queue_position: float = 0.0  # Quantity ahead when order placed
    current_queue_position: float = 0.0  # Current quantity ahead


@dataclass
class QueueAdvancement:
    """Result of queue advancement calculation."""
    order_id: str
    previous_position: float
    new_position: float
    advancement: float
    fill_probability: float


class QueueModel(ABC):
    """Abstract base class for queue models."""

    @abstractmethod
    def estimate_position(self, order: QueueOrder, book_state: Dict) -> float:
        """Estimate current queue position."""
        pass

    @abstractmethod
    def advance_queue(self, order: QueueOrder, trade_volume: float, trade_price: float) -> QueueAdvancement:
        """Advance queue position based on trade."""
        pass

    @abstractmethod
    def fill_probability(self, order: QueueOrder, book_state: Dict) -> float:
        """Estimate probability of fill."""
        pass


class RiskAverseQueueModel(QueueModel):
    """
    Risk-Averse Queue Model.

    Conservative model where queue only advances when trades occur
    at the order's price level. No credit for cancellations.

    Best for: Conservative fill probability estimation
    """

    def __init__(self):
        self.name = "RiskAverse"

    def estimate_position(self, order: QueueOrder, book_state: Dict) -> float:
        """
        Estimate queue position.

        Position = initial_position - trades_at_level
        """
        return max(0, order.current_queue_position)

    def advance_queue(self, order: QueueOrder, trade_volume: float, trade_price: float) -> QueueAdvancement:
        """
        Advance queue only if trade at same price level.

        Args:
            order: Order to advance
            trade_volume: Volume of trade
            trade_price: Price of trade

        Returns:
            QueueAdvancement with new position
        """
        previous = order.current_queue_position

        # Only advance if trade at our price level
        if abs(trade_price - order.price) < 1e-8:
            advancement = min(trade_volume, order.current_queue_position)
            order.current_queue_position = max(0, order.current_queue_position - advancement)
        else:
            advancement = 0

        # Estimate fill probability
        if order.current_queue_position <= 0:
            fill_prob = 1.0
        else:
            fill_prob = order.quantity / (order.current_queue_position + order.quantity)

        return QueueAdvancement(
            order_id=order.order_id,
            previous_position=previous,
            new_position=order.current_queue_position,
            advancement=advancement,
            fill_probability=fill_prob
        )

    def fill_probability(self, order: QueueOrder, book_state: Dict) -> float:
        """Estimate fill probability based on queue position."""
        if order.current_queue_position <= 0:
            return 1.0

        # Simple model: P(fill) = order_size / (position + order_size)
        return order.quantity / (order.current_queue_position + order.quantity)


class ProbabilisticQueueModel(QueueModel):
    """
    Probabilistic Queue Model.

    Uses statistical model to estimate queue position based on:
    - Historical fill rates at price level
    - Order arrival/cancellation rates
    - Trade frequency

    Best for: Realistic average-case estimation
    """

    def __init__(self, fill_rate: float = 0.3, cancel_rate: float = 0.5):
        """
        Initialize probabilistic model.

        Args:
            fill_rate: Base fill rate per unit time
            cancel_rate: Rate at which orders ahead cancel
        """
        self.name = "Probabilistic"
        self.fill_rate = fill_rate
        self.cancel_rate = cancel_rate

        # Historical data for calibration
        self.trade_arrivals: List[Tuple[datetime, float]] = []
        self.max_history = 1000

    def estimate_position(self, order: QueueOrder, book_state: Dict) -> float:
        """
        Probabilistic position estimate.

        Accounts for expected cancellations ahead.
        """
        raw_position = order.current_queue_position

        # Estimate cancellations (exponential decay)
        time_in_queue = (datetime.now() - order.timestamp).total_seconds()
        expected_cancels = raw_position * (1 - np.exp(-self.cancel_rate * time_in_queue / 60))

        return max(0, raw_position - expected_cancels)

    def advance_queue(self, order: QueueOrder, trade_volume: float, trade_price: float) -> QueueAdvancement:
        """
        Advance with probabilistic cancellation credit.
        """
        previous = order.current_queue_position

        # Record trade for statistics
        self.trade_arrivals.append((datetime.now(), trade_volume))
        if len(self.trade_arrivals) > self.max_history:
            self.trade_arrivals.pop(0)

        advancement = 0.0

        # Trade at our level
        if abs(trade_price - order.price) < 1e-8:
            advancement = min(trade_volume, order.current_queue_position)
            order.current_queue_position = max(0, order.current_queue_position - advancement)

        # Add probabilistic cancellation credit
        time_in_queue = (datetime.now() - order.timestamp).total_seconds()
        cancel_credit = order.current_queue_position * self.cancel_rate * (time_in_queue / 3600)
        order.current_queue_position = max(0, order.current_queue_position - cancel_credit)

        # Calculate fill probability
        fill_prob = self._calculate_fill_probability(order)

        return QueueAdvancement(
            order_id=order.order_id,
            previous_position=previous,
            new_position=order.current_queue_position,
            advancement=advancement,
            fill_probability=fill_prob
        )

    def fill_probability(self, order: QueueOrder, book_state: Dict) -> float:
        """Calculate fill probability using Poisson model."""
        return self._calculate_fill_probability(order)

    def _calculate_fill_probability(self, order: QueueOrder) -> float:
        """
        Calculate fill probability using Poisson arrival model.

        P(fill) = P(trades >= queue_position within time horizon)
        """
        if order.current_queue_position <= 0:
            return 1.0

        # Estimate trade arrival rate from history
        if len(self.trade_arrivals) < 10:
            # Default to simple model
            return order.quantity / (order.current_queue_position + order.quantity)

        # Calculate average trade rate
        time_span = (self.trade_arrivals[-1][0] - self.trade_arrivals[0][0]).total_seconds()
        if time_span <= 0:
            return 0.5

        total_volume = sum(t[1] for t in self.trade_arrivals)
        rate = total_volume / time_span  # Volume per second

        # Poisson probability of filling within 60 seconds
        expected_trades = rate * 60
        position = order.current_queue_position

        # P(X >= position) where X ~ Poisson(rate * time)
        if expected_trades <= 0:
            return 0.0

        # Approximate using normal for large lambda
        if expected_trades > 20:
            z = (position - expected_trades) / np.sqrt(expected_trades)
            return 1 - 0.5 * (1 + np.tanh(z * 0.7))

        # Exact Poisson for small lambda
        prob = 0.0
        factorial = 1
        for k in range(int(position)):
            if k > 0:
                factorial *= k
            prob += (expected_trades ** k * np.exp(-expected_trades)) / factorial

        return 1 - prob

    def calibrate(self, trade_history: pd.DataFrame) -> None:
        """
        Calibrate model from historical trades.

        Expects DataFrame with: timestamp, price, volume
        """
        if len(trade_history) < 10:
            return

        # Calculate fill rate
        time_span = (trade_history['timestamp'].max() - trade_history['timestamp'].min()).total_seconds()
        if time_span > 0:
            self.fill_rate = len(trade_history) / time_span * 60  # Fills per minute


class L3FIFOQueueModel(QueueModel):
    """
    Level-3 FIFO Queue Model.

    Exact simulation using full order book with price-time priority.
    Tracks every order ahead in queue.

    Best for: High-precision HFT simulation
    """

    def __init__(self):
        self.name = "L3FIFO"

        # Track orders at each price level
        self.queue_state: Dict[float, List[str]] = {}  # price -> [order_ids]
        self.order_quantities: Dict[str, float] = {}  # order_id -> quantity

    def update_book(self, price: float, order_id: str, quantity: float, action: str) -> None:
        """
        Update internal queue state from L3 feed.

        Args:
            price: Price level
            order_id: Order identifier
            quantity: Order quantity
            action: 'add', 'modify', 'delete', 'fill'
        """
        if price not in self.queue_state:
            self.queue_state[price] = []

        if action == 'add':
            self.queue_state[price].append(order_id)
            self.order_quantities[order_id] = quantity

        elif action == 'modify':
            if order_id in self.order_quantities:
                self.order_quantities[order_id] = quantity

        elif action in ('delete', 'fill'):
            if order_id in self.queue_state[price]:
                self.queue_state[price].remove(order_id)
            if order_id in self.order_quantities:
                del self.order_quantities[order_id]

    def estimate_position(self, order: QueueOrder, book_state: Dict) -> float:
        """
        Get exact queue position from L3 state.
        """
        price = order.price

        if price not in self.queue_state:
            return 0.0

        queue = self.queue_state[price]

        # Find our position
        try:
            idx = queue.index(order.order_id)
        except ValueError:
            return order.current_queue_position

        # Sum quantity ahead of us
        quantity_ahead = 0.0
        for i in range(idx):
            oid = queue[i]
            quantity_ahead += self.order_quantities.get(oid, 0)

        return quantity_ahead

    def advance_queue(self, order: QueueOrder, trade_volume: float, trade_price: float) -> QueueAdvancement:
        """
        Advance queue using FIFO matching.
        """
        previous = order.current_queue_position

        if abs(trade_price - order.price) < 1e-8:
            # Recalculate position from L3 state
            order.current_queue_position = self.estimate_position(order, {})

        advancement = previous - order.current_queue_position

        # Calculate fill probability
        fill_prob = 1.0 if order.current_queue_position <= 0 else 0.0

        return QueueAdvancement(
            order_id=order.order_id,
            previous_position=previous,
            new_position=order.current_queue_position,
            advancement=max(0, advancement),
            fill_probability=fill_prob
        )

    def fill_probability(self, order: QueueOrder, book_state: Dict) -> float:
        """Binary fill probability based on position."""
        position = self.estimate_position(order, book_state)
        return 1.0 if position <= 0 else 0.0


class QueuePositionTracker:
    """
    High-level queue position tracker.

    Manages multiple orders and models, provides unified interface.

    Usage:
        tracker = QueuePositionTracker(model='probabilistic')
        tracker.add_order(order)
        tracker.on_trade(trade)
        fill_prob = tracker.get_fill_probability(order_id)
    """

    def __init__(self, model: str = 'probabilistic'):
        """
        Initialize tracker.

        Args:
            model: 'risk_averse', 'probabilistic', or 'l3fifo'
        """
        if model == 'risk_averse':
            self.model = RiskAverseQueueModel()
        elif model == 'probabilistic':
            self.model = ProbabilisticQueueModel()
        elif model == 'l3fifo':
            self.model = L3FIFOQueueModel()
        else:
            raise ValueError(f"Unknown model: {model}")

        self.orders: Dict[str, QueueOrder] = {}
        self.book_state: Dict = {}

    def add_order(self, order_id: str, price: float, quantity: float,
                  side: Side, queue_ahead: float) -> QueueOrder:
        """
        Add new order to track.

        Args:
            order_id: Unique order identifier
            price: Limit order price
            quantity: Order quantity
            side: BUY or SELL
            queue_ahead: Quantity ahead in queue when placed

        Returns:
            QueueOrder object
        """
        order = QueueOrder(
            order_id=order_id,
            price=price,
            quantity=quantity,
            side=side,
            timestamp=datetime.now(),
            initial_queue_position=queue_ahead,
            current_queue_position=queue_ahead
        )

        self.orders[order_id] = order

        # Update L3 model if used
        if isinstance(self.model, L3FIFOQueueModel):
            self.model.update_book(price, order_id, quantity, 'add')

        return order

    def remove_order(self, order_id: str) -> Optional[QueueOrder]:
        """Remove order from tracking."""
        if order_id not in self.orders:
            return None

        order = self.orders.pop(order_id)

        if isinstance(self.model, L3FIFOQueueModel):
            self.model.update_book(order.price, order_id, 0, 'delete')

        return order

    def on_trade(self, price: float, volume: float, side: Side) -> List[QueueAdvancement]:
        """
        Process trade event, advance queues.

        Args:
            price: Trade price
            volume: Trade volume
            side: Aggressor side (BUY = took from asks, SELL = took from bids)

        Returns:
            List of queue advancements for affected orders
        """
        advancements = []

        # Determine which orders are affected
        # Buy aggressor hits asks, sell aggressor hits bids
        affected_side = Side.SELL if side == Side.BUY else Side.BUY

        for order in self.orders.values():
            if order.side == affected_side and abs(order.price - price) < 1e-8:
                advancement = self.model.advance_queue(order, volume, price)
                advancements.append(advancement)

        return advancements

    def get_position(self, order_id: str) -> float:
        """Get current queue position for order."""
        if order_id not in self.orders:
            return -1

        return self.model.estimate_position(self.orders[order_id], self.book_state)

    def get_fill_probability(self, order_id: str) -> float:
        """Get fill probability for order."""
        if order_id not in self.orders:
            return 0.0

        return self.model.fill_probability(self.orders[order_id], self.book_state)

    def get_all_probabilities(self) -> Dict[str, float]:
        """Get fill probabilities for all tracked orders."""
        return {
            oid: self.model.fill_probability(order, self.book_state)
            for oid, order in self.orders.items()
        }

    def update_book_state(self, book_state: Dict) -> None:
        """Update book state for probability calculations."""
        self.book_state = book_state

    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all tracked orders."""
        data = []
        for oid, order in self.orders.items():
            data.append({
                'order_id': oid,
                'side': order.side.name,
                'price': order.price,
                'quantity': order.quantity,
                'initial_position': order.initial_queue_position,
                'current_position': order.current_queue_position,
                'position_improvement': order.initial_queue_position - order.current_queue_position,
                'fill_probability': self.get_fill_probability(oid),
                'age_seconds': (datetime.now() - order.timestamp).total_seconds()
            })

        return pd.DataFrame(data)


class AdverseSelectionModel:
    """
    Adverse Selection Risk Model.

    Estimates the probability that a fill will be followed by
    an adverse price move (informed trading).

    Based on: Kyle (1985), Glosten-Milgrom (1985)
    """

    def __init__(self, informed_fraction: float = 0.1):
        """
        Initialize model.

        Args:
            informed_fraction: Estimated fraction of informed traders
        """
        self.informed_fraction = informed_fraction
        self.fills: List[Dict] = []
        self.max_history = 500

    def record_fill(self, fill_price: float, fill_side: Side,
                    price_after_1s: float, price_after_5s: float) -> None:
        """
        Record a fill with subsequent price moves.

        Used to calibrate adverse selection model.
        """
        if fill_side == Side.BUY:
            # We bought, adverse move is price going down
            adverse_1s = fill_price > price_after_1s
            adverse_5s = fill_price > price_after_5s
            move_1s = (price_after_1s - fill_price) / fill_price * 10000
            move_5s = (price_after_5s - fill_price) / fill_price * 10000
        else:
            # We sold, adverse move is price going up
            adverse_1s = fill_price < price_after_1s
            adverse_5s = fill_price < price_after_5s
            move_1s = (fill_price - price_after_1s) / fill_price * 10000
            move_5s = (fill_price - price_after_5s) / fill_price * 10000

        self.fills.append({
            'side': fill_side,
            'price': fill_price,
            'adverse_1s': adverse_1s,
            'adverse_5s': adverse_5s,
            'move_1s_bps': move_1s,
            'move_5s_bps': move_5s
        })

        if len(self.fills) > self.max_history:
            self.fills.pop(0)

    def adverse_selection_probability(self) -> float:
        """
        Calculate probability of adverse selection.

        Returns probability that a fill will be followed by adverse move.
        """
        if len(self.fills) < 10:
            return self.informed_fraction

        adverse_count = sum(1 for f in self.fills if f['adverse_5s'])
        return adverse_count / len(self.fills)

    def expected_adverse_move_bps(self) -> float:
        """Calculate expected adverse move in basis points."""
        if len(self.fills) < 10:
            return 1.0  # Default 1 bps

        moves = [f['move_5s_bps'] for f in self.fills if f['adverse_5s']]
        if not moves:
            return 0.0

        return np.mean(moves)

    def toxicity_score(self) -> float:
        """
        Calculate VPIN-style toxicity score.

        Higher = more informed trading, worse fills expected.
        Range: 0 (no toxicity) to 1 (all informed)
        """
        if len(self.fills) < 20:
            return 0.5

        # Rolling window toxicity
        recent = self.fills[-20:]

        # Measure: How often do we get adversely selected?
        adverse_rate = sum(1 for f in recent if f['adverse_1s']) / len(recent)

        # Measure: How large are the adverse moves?
        adverse_moves = [f['move_1s_bps'] for f in recent if f['adverse_1s']]
        avg_move = np.mean(adverse_moves) if adverse_moves else 0

        # Combine into toxicity score
        toxicity = 0.5 * adverse_rate + 0.5 * min(1, avg_move / 5)

        return toxicity


if __name__ == '__main__':
    print("Queue Position Tracking Test")
    print("=" * 50)

    # Test with probabilistic model
    tracker = QueuePositionTracker(model='probabilistic')

    # Add some orders
    tracker.add_order("order_1", 1.1000, 100, Side.BUY, queue_ahead=500)
    tracker.add_order("order_2", 1.1000, 50, Side.BUY, queue_ahead=600)
    tracker.add_order("order_3", 1.0999, 100, Side.BUY, queue_ahead=200)

    print("\nInitial State:")
    print(tracker.get_summary().to_string())

    # Simulate trades at 1.1000
    print("\n--- Trade: 100 volume at 1.1000 (sell aggressor) ---")
    advancements = tracker.on_trade(1.1000, 100, Side.SELL)
    for adv in advancements:
        print(f"Order {adv.order_id}: {adv.previous_position:.1f} -> {adv.new_position:.1f}, P(fill)={adv.fill_probability:.3f}")

    print("\n--- Trade: 200 volume at 1.1000 (sell aggressor) ---")
    advancements = tracker.on_trade(1.1000, 200, Side.SELL)
    for adv in advancements:
        print(f"Order {adv.order_id}: {adv.previous_position:.1f} -> {adv.new_position:.1f}, P(fill)={adv.fill_probability:.3f}")

    print("\nFinal State:")
    print(tracker.get_summary().to_string())

    # Test adverse selection
    print("\n" + "=" * 50)
    print("Adverse Selection Model Test")

    as_model = AdverseSelectionModel()

    # Simulate some fills
    for i in range(50):
        # Random fill with some adverse moves
        fill_price = 1.1000
        if np.random.random() < 0.3:  # 30% informed
            price_1s = fill_price * (1 - 0.0002)  # Price moves against us
            price_5s = fill_price * (1 - 0.0005)
        else:
            price_1s = fill_price * (1 + 0.0001 * np.random.randn())
            price_5s = fill_price * (1 + 0.0002 * np.random.randn())

        as_model.record_fill(fill_price, Side.BUY, price_1s, price_5s)

    print(f"Adverse Selection Probability: {as_model.adverse_selection_probability():.2%}")
    print(f"Expected Adverse Move: {as_model.expected_adverse_move_bps():.2f} bps")
    print(f"Toxicity Score: {as_model.toxicity_score():.3f}")
