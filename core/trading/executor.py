"""
Order Executor
==============
Execute orders via IB Gateway with queue management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from enum import Enum
import threading
import queue
import logging
import time

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    direction: int          # 1 = buy, -1 = sell
    quantity: float
    order_type: str = "MKT"  # MKT, LMT, STP
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    ib_order_id: Optional[int] = None

    @property
    def side(self) -> str:
        return "BUY" if self.direction > 0 else "SELL"

    @property
    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.ERROR
        )


class OrderExecutor:
    """
    Execute orders via IB Gateway.

    Features:
    - Order queue with rate limiting
    - Async submission with callbacks
    - Position tracking integration
    - Error handling and retry logic
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4001,
        client_id: int = 1,
        max_orders_per_second: float = 10.0,
        connect: bool = True
    ):
        """
        Initialize order executor.

        Args:
            host: IB Gateway host
            port: IB Gateway port
            client_id: Client ID for IB connection
            max_orders_per_second: Rate limit
            connect: Whether to connect immediately
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.rate_limit = 1.0 / max_orders_per_second

        # Connection state
        self._ib = None
        self._connected = False
        self._lock = threading.RLock()

        # Order management
        self._orders: Dict[str, Order] = {}
        self._order_queue: queue.Queue = queue.Queue()
        self._next_order_id = 1
        self._callbacks: Dict[str, List[Callable]] = {}

        # Worker thread
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        if connect:
            self.connect()

    def connect(self) -> bool:
        """Connect to IB Gateway."""
        if self._connected:
            return True

        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
                readonly=False
            )
            self._connected = True
            self._start_worker()
            logger.info(f"Connected to IB Gateway at {self.host}:{self.port}")
            return True

        except ImportError:
            logger.warning("ib_insync not installed, running in simulation mode")
            self._connected = False
            self._start_worker()
            return False

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            self._start_worker()
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("Disconnected from IB Gateway")

    def _start_worker(self):
        """Start order processing worker."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_orders,
            daemon=True,
            name="OrderExecutor-Worker"
        )
        self._worker_thread.start()

    def _process_orders(self):
        """Worker thread to process order queue."""
        last_order_time = 0.0

        while self._running:
            try:
                # Get order with timeout
                try:
                    order = self._order_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Rate limiting
                now = time.time()
                elapsed = now - last_order_time
                if elapsed < self.rate_limit:
                    time.sleep(self.rate_limit - elapsed)

                # Submit order
                self._submit_order(order)
                last_order_time = time.time()

            except Exception as e:
                logger.error(f"Order processing error: {e}")

    def submit(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        callback: Optional[Callable[[Order], None]] = None
    ) -> Order:
        """
        Submit an order.

        Args:
            symbol: Trading symbol
            direction: 1 for buy, -1 for sell
            quantity: Order quantity
            order_type: MKT, LMT, or STP
            limit_price: Limit price (for LMT orders)
            callback: Function called on order update

        Returns:
            Order object
        """
        with self._lock:
            order_id = f"ORD-{self._next_order_id:06d}"
            self._next_order_id += 1

        order = Order(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            quantity=abs(quantity),
            order_type=order_type,
            limit_price=limit_price,
        )

        with self._lock:
            self._orders[order_id] = order
            if callback:
                self._callbacks[order_id] = [callback]

        # Queue for processing
        self._order_queue.put(order)
        logger.info(f"Queued {order.side} {order.quantity} {symbol} ({order_id})")

        return order

    def _submit_order(self, order: Order):
        """Submit order to IB Gateway or simulate."""
        try:
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self._notify_callbacks(order)

            if self._connected and self._ib:
                self._submit_to_ib(order)
            else:
                self._simulate_fill(order)

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.updated_at = datetime.now()
            logger.error(f"Order {order.order_id} error: {e}")
            self._notify_callbacks(order)

    def _submit_to_ib(self, order: Order):
        """Submit order to IB Gateway."""
        from ib_insync import Forex, MarketOrder, LimitOrder

        # Create contract
        contract = Forex(order.symbol)

        # Create order
        if order.order_type == "MKT":
            ib_order = MarketOrder(
                order.side,
                order.quantity
            )
        elif order.order_type == "LMT":
            ib_order = LimitOrder(
                order.side,
                order.quantity,
                order.limit_price
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Submit
        trade = self._ib.placeOrder(contract, ib_order)
        order.ib_order_id = trade.order.orderId

        # Wait for fill (with timeout)
        start = time.time()
        while not trade.isDone() and time.time() - start < 30:
            self._ib.sleep(0.1)

        # Update order from trade
        if trade.orderStatus.status == 'Filled':
            order.status = OrderStatus.FILLED
            order.filled_qty = float(trade.orderStatus.filled)
            order.avg_fill_price = float(trade.orderStatus.avgFillPrice)
        elif trade.orderStatus.status == 'Cancelled':
            order.status = OrderStatus.CANCELLED
        else:
            order.status = OrderStatus.PARTIAL
            order.filled_qty = float(trade.orderStatus.filled)

        order.updated_at = datetime.now()
        logger.info(
            f"Order {order.order_id} {order.status.value}: "
            f"{order.filled_qty}@{order.avg_fill_price}"
        )
        self._notify_callbacks(order)

    def _simulate_fill(self, order: Order):
        """Simulate order fill for testing."""
        # Simulate small delay
        time.sleep(0.05)

        # Simulate fill
        order.status = OrderStatus.FILLED
        order.filled_qty = order.quantity
        # Simulate price (would come from market data in reality)
        order.avg_fill_price = order.limit_price or 1.0
        order.updated_at = datetime.now()

        logger.info(
            f"Simulated fill {order.order_id}: "
            f"{order.side} {order.filled_qty} {order.symbol}"
        )
        self._notify_callbacks(order)

    def _notify_callbacks(self, order: Order):
        """Notify callbacks of order update."""
        callbacks = self._callbacks.get(order.order_id, [])
        for callback in callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Callback error for {order.order_id}: {e}")

    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order."""
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return False

            if order.is_complete:
                return False

            if self._connected and self._ib and order.ib_order_id:
                try:
                    self._ib.cancelOrder(order.ib_order_id)
                except Exception as e:
                    logger.error(f"Cancel error: {e}")

            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self._notify_callbacks(order)
            return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [
            o for o in self._orders.values()
            if not o.is_complete
        ]

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [
            o for o in self._orders.values()
            if o.symbol == symbol
        ]

    @property
    def is_connected(self) -> bool:
        """Check if connected to IB Gateway."""
        return self._connected

    def stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        orders = list(self._orders.values())
        return {
            'connected': self._connected,
            'total_orders': len(orders),
            'pending': sum(1 for o in orders if o.status == OrderStatus.PENDING),
            'submitted': sum(1 for o in orders if o.status == OrderStatus.SUBMITTED),
            'filled': sum(1 for o in orders if o.status == OrderStatus.FILLED),
            'cancelled': sum(1 for o in orders if o.status == OrderStatus.CANCELLED),
            'errors': sum(1 for o in orders if o.status == OrderStatus.ERROR),
            'queue_size': self._order_queue.qsize(),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
