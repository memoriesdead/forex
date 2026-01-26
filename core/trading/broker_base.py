"""
Multi-Broker Abstraction Layer
==============================
Professional-grade broker interface for institutional forex trading.

Supports:
- Interactive Brokers (IB Gateway)
- OANDA v20 API
- Forex.com (GAIN Capital REST API)
- tastyfx (IG Group REST API)
- IG Markets (REST API)

Architecture:
    BrokerBase (ABC)
    ├── IBBroker
    ├── OANDABroker
    ├── ForexComBroker
    ├── TastyFXBroker
    └── IGBroker

    BrokerRouter → Routes orders to optimal broker
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    IB = "interactive_brokers"
    OANDA = "oanda"
    FOREX_COM = "forex_com"
    TASTYFX = "tastyfx"
    IG = "ig_markets"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class BrokerConfig:
    """Broker configuration."""
    broker_type: BrokerType
    name: str
    enabled: bool = True

    # Connection
    host: str = ""
    port: int = 0
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""

    # Mode
    paper: bool = True

    # Rate limits
    max_orders_per_second: float = 10.0
    max_requests_per_minute: int = 120

    # Timeouts
    connect_timeout: float = 30.0
    request_timeout: float = 10.0

    # Retry
    max_retries: int = 3
    retry_delay: float = 1.0

    # Symbols this broker handles
    symbols: List[str] = field(default_factory=list)

    # Priority (lower = higher priority for routing)
    priority: int = 0

    # Additional broker-specific params
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerOrder:
    """Unified order representation across all brokers."""
    # Identity
    internal_id: str
    broker_id: Optional[str] = None
    broker_type: Optional[BrokerType] = None

    # Order details
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Error handling
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Broker-specific data
    broker_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR
        )

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL
        )


@dataclass
class BrokerPosition:
    """Unified position representation."""
    symbol: str
    broker_type: BrokerType
    account_id: str

    # Position
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Margin
    margin_used: float = 0.0

    # Timestamps
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class BrokerAccount:
    """Unified account information."""
    account_id: str
    broker_type: BrokerType

    # Balance
    balance: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Currency
    currency: str = "USD"

    # Status
    is_active: bool = True
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BrokerQuote:
    """Unified quote/tick representation."""
    symbol: str
    broker_type: BrokerType

    # Prices
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0

    # Size
    bid_size: float = 0.0
    ask_size: float = 0.0

    # Spread
    spread: float = 0.0
    spread_pips: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if quote is valid."""
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid


class BrokerBase(ABC):
    """
    Abstract base class for all broker implementations.

    All broker adapters must implement this interface.
    """

    def __init__(self, config: BrokerConfig):
        """
        Initialize broker.

        Args:
            config: Broker configuration
        """
        self.config = config
        self._connected = False
        self._lock = threading.RLock()
        self._orders: Dict[str, BrokerOrder] = {}
        self._positions: Dict[str, BrokerPosition] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._order_counter = 0

    # ==================== Connection ====================

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def ping(self) -> bool:
        """
        Ping broker to check connection health.

        Returns:
            True if connection is healthy
        """
        pass

    # ==================== Account ====================

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """
        Get account information.

        Returns:
            Account information
        """
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """
        Get account balance.

        Returns:
            Account balance
        """
        pass

    # ==================== Orders ====================

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> BrokerOrder:
        """
        Submit an order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            **kwargs: Additional broker-specific parameters

        Returns:
            Order object
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation was successful
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None
        """
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        pass

    # ==================== Positions ====================

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """
        Get all positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str, quantity: Optional[float] = None) -> BrokerOrder:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            quantity: Quantity to close (None = close all)

        Returns:
            Order object for the closing trade
        """
        pass

    # ==================== Market Data ====================

    @abstractmethod
    def get_quote(self, symbol: str) -> BrokerQuote:
        """
        Get current quote for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote object
        """
        pass

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """
        Get current spread in pips.

        Args:
            symbol: Trading symbol

        Returns:
            Spread in pips
        """
        pass

    # ==================== Symbol Info ====================

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get list of tradeable symbols.

        Returns:
            List of symbol names
        """
        pass

    @abstractmethod
    def get_pip_value(self, symbol: str) -> float:
        """
        Get pip value for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Pip value
        """
        pass

    @abstractmethod
    def get_min_quantity(self, symbol: str) -> float:
        """
        Get minimum order quantity for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Minimum quantity
        """
        pass

    # ==================== Helper Methods ====================

    def _generate_order_id(self) -> str:
        """Generate unique internal order ID."""
        with self._lock:
            self._order_counter += 1
            return f"{self.config.broker_type.value}_{self._order_counter:08d}"

    def _update_order(self, order: BrokerOrder) -> None:
        """Update order in internal storage."""
        with self._lock:
            self._orders[order.internal_id] = order
            order.updated_at = datetime.now()

    def _notify_callbacks(self, event: str, data: Any) -> None:
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for events.

        Events:
            - order_update
            - position_update
            - quote_update
            - connection_status
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        orders = list(self._orders.values())
        return {
            'broker': self.config.broker_type.value,
            'connected': self.is_connected(),
            'account_id': self.config.account_id,
            'paper': self.config.paper,
            'total_orders': len(orders),
            'open_orders': sum(1 for o in orders if o.is_active),
            'filled_orders': sum(1 for o in orders if o.status == OrderStatus.FILLED),
            'positions': len(self._positions),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config.broker_type.value}, account={self.config.account_id})"

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
