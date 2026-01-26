"""
Position Manager
================
Track and manage trading positions across symbols.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 0.001

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.001

    @property
    def is_short(self) -> bool:
        return self.quantity < -0.001

    @property
    def direction(self) -> int:
        if self.is_long:
            return 1
        elif self.is_short:
            return -1
        return 0

    def update_pnl(self, current_price: float, pip_value: float = 10.0):
        """Update unrealized P&L based on current price."""
        if not self.is_open:
            self.unrealized_pnl = 0.0
            return

        price_diff = current_price - self.avg_price
        # Convert to pips (for non-JPY pairs, 1 pip = 0.0001)
        pips = price_diff * 10000

        self.unrealized_pnl = pips * pip_value * self.quantity
        self.last_update = datetime.now()


class PositionManager:
    """
    Manage positions across all trading symbols.

    Thread-safe position tracking with P&L calculation.
    """

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._lock = threading.RLock()
        self._trade_history: List[Dict] = []

    def get_position(self, symbol: str) -> Position:
        """Get or create position for a symbol."""
        with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = Position(symbol=symbol)
            return self._positions[symbol]

    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        is_fill: bool = True
    ):
        """
        Update position after a trade.

        Args:
            symbol: Trading symbol
            quantity: Signed quantity (positive = buy, negative = sell)
            price: Execution price
            is_fill: Whether this is a fill (vs theoretical update)
        """
        with self._lock:
            pos = self.get_position(symbol)

            if not pos.is_open:
                # Opening new position
                pos.quantity = quantity
                pos.avg_price = price
                pos.entry_time = datetime.now()
            else:
                # Modifying existing position
                old_qty = pos.quantity
                new_qty = old_qty + quantity

                if abs(new_qty) < 0.001:
                    # Closing position
                    realized = self._calculate_realized_pnl(
                        pos, abs(quantity), price
                    )
                    pos.realized_pnl += realized
                    pos.quantity = 0.0
                    pos.avg_price = 0.0
                    pos.unrealized_pnl = 0.0

                elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                    # Adding to position
                    total_cost = (pos.avg_price * abs(old_qty) +
                                 price * abs(quantity))
                    pos.avg_price = total_cost / abs(new_qty)
                    pos.quantity = new_qty

                else:
                    # Reversing position
                    close_qty = abs(old_qty)
                    realized = self._calculate_realized_pnl(pos, close_qty, price)
                    pos.realized_pnl += realized

                    # Open new position with remaining
                    pos.quantity = new_qty
                    pos.avg_price = price
                    pos.entry_time = datetime.now()

            pos.last_update = datetime.now()

            if is_fill:
                self._record_trade(symbol, quantity, price)

    def _calculate_realized_pnl(
        self,
        pos: Position,
        close_qty: float,
        close_price: float
    ) -> float:
        """Calculate realized P&L for closing a position."""
        price_diff = close_price - pos.avg_price
        if pos.is_short:
            price_diff = -price_diff  # Profit when price goes down for short

        pips = price_diff * 10000  # Convert to pips
        return pips * 10.0 * close_qty  # Assume $10/pip standard lot

    def _record_trade(self, symbol: str, quantity: float, price: float):
        """Record trade in history."""
        self._trade_history.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat(),
        })

    def update_prices(self, prices: Dict[str, float]):
        """Update unrealized P&L for all positions."""
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol].update_pnl(price)

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        with self._lock:
            return [p for p in self._positions.values() if p.is_open]

    def get_total_pnl(self) -> Dict[str, float]:
        """Get total P&L across all positions."""
        with self._lock:
            unrealized = sum(p.unrealized_pnl for p in self._positions.values())
            realized = sum(p.realized_pnl for p in self._positions.values())
            return {
                'unrealized': unrealized,
                'realized': realized,
                'total': unrealized + realized,
            }

    def close_all(self, prices: Dict[str, float]):
        """Close all positions at given prices."""
        with self._lock:
            for symbol, pos in self._positions.items():
                if pos.is_open and symbol in prices:
                    self.update_position(
                        symbol,
                        -pos.quantity,
                        prices[symbol]
                    )

    def summary(self) -> Dict:
        """Get position summary."""
        with self._lock:
            open_positions = self.get_open_positions()
            pnl = self.get_total_pnl()

            return {
                'open_count': len(open_positions),
                'positions': {
                    p.symbol: {
                        'quantity': p.quantity,
                        'avg_price': p.avg_price,
                        'unrealized_pnl': p.unrealized_pnl,
                        'direction': 'LONG' if p.is_long else 'SHORT',
                    }
                    for p in open_positions
                },
                'pnl': pnl,
                'trade_count': len(self._trade_history),
            }

    def __len__(self) -> int:
        return len(self.get_open_positions())
