"""
Interactive Brokers Adapter
===========================
Professional IB Gateway integration via ib_insync.

Connection:
- IB Gateway Docker: localhost:4004 (paper)
- Account: DUO423364

Features:
- Market and limit orders
- Position tracking
- Real-time quotes
- Account monitoring
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import logging
import time

from .broker_base import (
    BrokerBase, BrokerConfig, BrokerType,
    BrokerOrder, BrokerPosition, BrokerAccount, BrokerQuote,
    OrderSide, OrderType, OrderStatus, PositionSide
)

logger = logging.getLogger(__name__)


class IBBroker(BrokerBase):
    """
    Interactive Brokers adapter using ib_insync.

    Connects to IB Gateway (Docker) for forex trading.
    """

    # IB-specific symbol mapping
    SYMBOL_MAP = {
        'EURUSD': 'EUR.USD',
        'GBPUSD': 'GBP.USD',
        'USDJPY': 'USD.JPY',
        'USDCHF': 'USD.CHF',
        'AUDUSD': 'AUD.USD',
        'USDCAD': 'USD.CAD',
        'NZDUSD': 'NZD.USD',
        'EURJPY': 'EUR.JPY',
        'GBPJPY': 'GBP.JPY',
        'EURGBP': 'EUR.GBP',
        'EURCHF': 'EUR.CHF',
        'AUDJPY': 'AUD.JPY',
        'EURAUD': 'EUR.AUD',
        'GBPAUD': 'GBP.AUD',
    }

    # Pip values for common pairs
    PIP_VALUES = {
        'EURUSD': 0.0001,
        'GBPUSD': 0.0001,
        'USDJPY': 0.01,
        'USDCHF': 0.0001,
        'AUDUSD': 0.0001,
        'USDCAD': 0.0001,
        'NZDUSD': 0.0001,
        'EURJPY': 0.01,
        'GBPJPY': 0.01,
        'EURGBP': 0.0001,
    }

    def __init__(self, config: BrokerConfig):
        """
        Initialize IB broker.

        Args:
            config: Broker configuration with IB-specific settings
        """
        super().__init__(config)
        self._ib = None
        self._contracts: Dict[str, Any] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> bool:
        """Connect to IB Gateway."""
        if self._connected:
            return True

        try:
            from ib_insync import IB
            self._ib = IB()

            # Connect with timeout
            self._ib.connect(
                host=self.config.host or 'localhost',
                port=self.config.port or 4004,
                clientId=self.config.extra.get('client_id', 1),
                readonly=False,
                timeout=self.config.connect_timeout
            )

            self._connected = True
            self._start_event_loop()

            logger.info(f"Connected to IB Gateway at {self.config.host}:{self.config.port}")
            logger.info(f"Account: {self._ib.managedAccounts()}")

            return True

        except ImportError:
            logger.error("ib_insync not installed: pip install ib_insync")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("Disconnected from IB Gateway")

    def is_connected(self) -> bool:
        """Check if connected."""
        if not self._connected or not self._ib:
            return False
        return self._ib.isConnected()

    def ping(self) -> bool:
        """Ping IB Gateway."""
        if not self.is_connected():
            return False
        try:
            # Request server time as ping
            self._ib.reqCurrentTime()
            return True
        except Exception:
            return False

    def _start_event_loop(self):
        """Start IB event loop in background thread."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="IB-EventLoop"
        )
        self._worker_thread.start()

    def _run_event_loop(self):
        """Run IB event loop."""
        while self._running and self._connected:
            try:
                self._ib.sleep(0.1)
            except Exception as e:
                logger.error(f"IB event loop error: {e}")
                break

    # ==================== Account ====================

    def get_account(self) -> BrokerAccount:
        """Get IB account information."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")

        account_values = self._ib.accountValues()

        # Parse account values
        balance = 0.0
        equity = 0.0
        margin_used = 0.0

        for av in account_values:
            if av.tag == 'TotalCashBalance' and av.currency == 'USD':
                balance = float(av.value)
            elif av.tag == 'NetLiquidation' and av.currency == 'USD':
                equity = float(av.value)
            elif av.tag == 'MaintMarginReq' and av.currency == 'USD':
                margin_used = float(av.value)

        return BrokerAccount(
            account_id=self.config.account_id or self._ib.managedAccounts()[0],
            broker_type=BrokerType.IB,
            balance=balance,
            equity=equity,
            margin_used=margin_used,
            margin_available=equity - margin_used,
            currency='USD',
            is_active=True,
            updated_at=datetime.now()
        )

    def get_balance(self) -> float:
        """Get account balance."""
        account = self.get_account()
        return account.balance

    # ==================== Orders ====================

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
        """Submit order to IB Gateway."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")

        from ib_insync import Forex, MarketOrder, LimitOrder, StopOrder, StopLimitOrder

        # Create order record
        order = BrokerOrder(
            internal_id=self._generate_order_id(),
            broker_type=BrokerType.IB,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )

        try:
            # Get or create contract
            contract = self._get_contract(symbol)

            # Create IB order
            ib_side = "BUY" if side == OrderSide.BUY else "SELL"

            if order_type == OrderType.MARKET:
                ib_order = MarketOrder(ib_side, quantity)
            elif order_type == OrderType.LIMIT:
                ib_order = LimitOrder(ib_side, quantity, limit_price)
            elif order_type == OrderType.STOP:
                ib_order = StopOrder(ib_side, quantity, stop_price)
            elif order_type == OrderType.STOP_LIMIT:
                ib_order = StopLimitOrder(ib_side, quantity, limit_price, stop_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Submit
            trade = self._ib.placeOrder(contract, ib_order)
            order.broker_id = str(trade.order.orderId)
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()

            # Wait for fill with timeout
            start = time.time()
            timeout = kwargs.get('timeout', 30.0)
            while not trade.isDone() and time.time() - start < timeout:
                self._ib.sleep(0.1)

            # Update order from trade
            self._update_order_from_trade(order, trade)

            # Store order
            self._update_order(order)

            logger.info(
                f"IB Order {order.internal_id}: {order.side.value} {order.quantity} {symbol} "
                f"-> {order.status.value}"
            )

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.updated_at = datetime.now()
            self._update_order(order)
            logger.error(f"IB order error: {e}")
            return order

    def _get_contract(self, symbol: str):
        """Get or create IB Forex contract."""
        if symbol in self._contracts:
            return self._contracts[symbol]

        from ib_insync import Forex

        # Map symbol to IB format
        ib_symbol = self.SYMBOL_MAP.get(symbol, symbol)

        # Create contract
        if '.' in ib_symbol:
            pair = ib_symbol.split('.')
            contract = Forex(pair=ib_symbol)
        else:
            contract = Forex(symbol)

        # Qualify contract
        self._ib.qualifyContracts(contract)
        self._contracts[symbol] = contract

        return contract

    def _update_order_from_trade(self, order: BrokerOrder, trade):
        """Update order from IB trade object."""
        status_map = {
            'PendingSubmit': OrderStatus.PENDING,
            'PreSubmitted': OrderStatus.SUBMITTED,
            'Submitted': OrderStatus.SUBMITTED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELLED,
            'ApiCancelled': OrderStatus.CANCELLED,
            'Inactive': OrderStatus.REJECTED,
        }

        ib_status = trade.orderStatus.status
        order.status = status_map.get(ib_status, OrderStatus.PENDING)

        if trade.orderStatus.filled > 0:
            order.filled_qty = float(trade.orderStatus.filled)
            order.avg_fill_price = float(trade.orderStatus.avgFillPrice)

            if order.filled_qty >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
            elif order.filled_qty > 0:
                order.status = OrderStatus.PARTIAL

        order.updated_at = datetime.now()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an IB order."""
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            if order.broker_id:
                # Find the trade
                for trade in self._ib.openTrades():
                    if str(trade.order.orderId) == order.broker_id:
                        self._ib.cancelOrder(trade.order)
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now()
                        return True

            return False

        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open orders."""
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ==================== Positions ====================

    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """Get all positions."""
        if not self.is_connected():
            return []

        positions = []
        for pos in self._ib.positions():
            # Extract symbol from contract
            pos_symbol = pos.contract.symbol
            if hasattr(pos.contract, 'pair'):
                pos_symbol = pos.contract.pair.replace('.', '')

            if symbol and pos_symbol != symbol:
                continue

            side = PositionSide.LONG if pos.position > 0 else (
                PositionSide.SHORT if pos.position < 0 else PositionSide.FLAT
            )

            bp = BrokerPosition(
                symbol=pos_symbol,
                broker_type=BrokerType.IB,
                account_id=pos.account,
                side=side,
                quantity=abs(pos.position),
                avg_entry_price=pos.avgCost,
                updated_at=datetime.now()
            )
            positions.append(bp)

        return positions

    def close_position(self, symbol: str, quantity: Optional[float] = None) -> BrokerOrder:
        """Close a position."""
        positions = self.get_positions(symbol)
        if not positions:
            raise ValueError(f"No position found for {symbol}")

        pos = positions[0]
        close_qty = quantity or pos.quantity
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY

        return self.submit_order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            order_type=OrderType.MARKET
        )

    # ==================== Market Data ====================

    def get_quote(self, symbol: str) -> BrokerQuote:
        """Get current quote."""
        if not self.is_connected():
            raise ConnectionError("Not connected to IB Gateway")

        contract = self._get_contract(symbol)

        # Request ticker
        ticker = self._ib.reqMktData(contract, '', False, False)
        self._ib.sleep(1.0)  # Wait for data

        bid = ticker.bid if ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask > 0 else 0

        pip_value = self.PIP_VALUES.get(symbol, 0.0001)
        spread = ask - bid if bid > 0 and ask > 0 else 0
        spread_pips = spread / pip_value if pip_value > 0 else 0

        self._ib.cancelMktData(contract)

        return BrokerQuote(
            symbol=symbol,
            broker_type=BrokerType.IB,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2 if bid > 0 and ask > 0 else 0,
            spread=spread,
            spread_pips=spread_pips,
            timestamp=datetime.now()
        )

    def get_spread(self, symbol: str) -> float:
        """Get current spread in pips."""
        quote = self.get_quote(symbol)
        return quote.spread_pips

    # ==================== Symbol Info ====================

    def get_symbols(self) -> List[str]:
        """Get tradeable forex symbols."""
        return list(self.SYMBOL_MAP.keys())

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.PIP_VALUES.get(symbol, 0.0001)

    def get_min_quantity(self, symbol: str) -> float:
        """Get minimum order quantity (lot size)."""
        # IB forex minimum is typically 25,000 units for majors
        return 25000.0


def create_ib_broker(
    host: str = 'localhost',
    port: int = 4004,
    account_id: str = 'DUO423364',
    client_id: int = 1,
    paper: bool = True
) -> IBBroker:
    """
    Factory function to create IB broker.

    Args:
        host: IB Gateway host
        port: IB Gateway port (4004 for paper)
        account_id: IB account ID
        client_id: Client ID for connection
        paper: Whether paper trading

    Returns:
        Configured IBBroker instance
    """
    config = BrokerConfig(
        broker_type=BrokerType.IB,
        name="Interactive Brokers",
        host=host,
        port=port,
        account_id=account_id,
        paper=paper,
        extra={'client_id': client_id},
        symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
        priority=1
    )

    return IBBroker(config)
