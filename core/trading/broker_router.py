"""
Multi-Broker Router
===================
Intelligent order routing across multiple forex brokers.

Features:
- Best execution routing (lowest spread)
- Load balancing across brokers
- Failover handling
- Position aggregation
- Real-time spread comparison

Supported Brokers:
1. Interactive Brokers (IB Gateway)
2. OANDA v20 API
3. Forex.com (GAIN Capital)
4. tastyfx (IG Group)
5. IG Markets

Routing Strategies:
- BEST_SPREAD: Route to broker with lowest spread
- PRIORITY: Route based on broker priority
- ROUND_ROBIN: Distribute orders evenly
- FAILOVER: Primary broker with fallback
- SPLIT: Split large orders across brokers
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import logging
import time
from collections import defaultdict

from .broker_base import (
    BrokerBase, BrokerConfig, BrokerType,
    BrokerOrder, BrokerPosition, BrokerAccount, BrokerQuote,
    OrderSide, OrderType, OrderStatus
)
from .broker_ib import IBBroker, create_ib_broker
from .broker_oanda import OANDABroker, create_oanda_broker
from .broker_forex_com import ForexComBroker, create_forex_com_broker
from .broker_tastyfx import TastyFXBroker, IGBroker, create_tastyfx_broker, create_ig_broker

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Order routing strategy."""
    BEST_SPREAD = "best_spread"      # Route to lowest spread
    PRIORITY = "priority"            # Route based on priority
    ROUND_ROBIN = "round_robin"      # Distribute evenly
    FAILOVER = "failover"            # Primary with fallback
    SPLIT = "split"                  # Split across brokers


@dataclass
class BrokerStatus:
    """Broker connection status."""
    broker_type: BrokerType
    connected: bool = False
    last_ping: Optional[datetime] = None
    latency_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    orders_today: int = 0
    fill_rate: float = 1.0


@dataclass
class RoutingDecision:
    """Routing decision for an order."""
    broker: BrokerBase
    reason: str
    spread: float = 0.0
    latency_ms: float = 0.0
    alternatives: List[BrokerBase] = field(default_factory=list)


class BrokerRouter:
    """
    Multi-broker order router.

    Routes orders to optimal broker based on strategy.
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BEST_SPREAD,
        max_retries: int = 3,
        spread_tolerance_pips: float = 0.5
    ):
        """
        Initialize broker router.

        Args:
            strategy: Default routing strategy
            max_retries: Max retry attempts per broker
            spread_tolerance_pips: Accept spread within tolerance of best
        """
        self.strategy = strategy
        self.max_retries = max_retries
        self.spread_tolerance = spread_tolerance_pips

        self._brokers: Dict[BrokerType, BrokerBase] = {}
        self._status: Dict[BrokerType, BrokerStatus] = {}
        self._lock = threading.RLock()

        # Round robin counter
        self._rr_counter = 0

        # Order history per broker
        self._order_history: Dict[BrokerType, List[BrokerOrder]] = defaultdict(list)

        # Spread cache (symbol -> broker -> spread)
        self._spread_cache: Dict[str, Dict[BrokerType, float]] = defaultdict(dict)
        self._spread_cache_time: Dict[str, datetime] = {}

    # ==================== Broker Management ====================

    def add_broker(self, broker: BrokerBase) -> None:
        """Add a broker to the router."""
        with self._lock:
            broker_type = broker.config.broker_type
            self._brokers[broker_type] = broker
            self._status[broker_type] = BrokerStatus(
                broker_type=broker_type,
                connected=False
            )
            logger.info(f"Added broker: {broker_type.value}")

    def remove_broker(self, broker_type: BrokerType) -> None:
        """Remove a broker from the router."""
        with self._lock:
            if broker_type in self._brokers:
                broker = self._brokers[broker_type]
                broker.disconnect()
                del self._brokers[broker_type]
                del self._status[broker_type]
                logger.info(f"Removed broker: {broker_type.value}")

    def get_broker(self, broker_type: BrokerType) -> Optional[BrokerBase]:
        """Get broker by type."""
        return self._brokers.get(broker_type)

    def get_brokers(self) -> List[BrokerBase]:
        """Get all brokers."""
        return list(self._brokers.values())

    def get_active_brokers(self) -> List[BrokerBase]:
        """Get connected brokers sorted by priority."""
        active = [
            b for b in self._brokers.values()
            if b.is_connected() and b.config.enabled
        ]
        return sorted(active, key=lambda b: b.config.priority)

    # ==================== Connection ====================

    def connect_all(self) -> Dict[BrokerType, bool]:
        """Connect to all brokers."""
        results = {}
        for broker_type, broker in self._brokers.items():
            try:
                success = broker.connect()
                results[broker_type] = success
                self._status[broker_type].connected = success
                if success:
                    logger.info(f"Connected to {broker_type.value}")
                else:
                    logger.warning(f"Failed to connect to {broker_type.value}")
            except Exception as e:
                results[broker_type] = False
                self._status[broker_type].connected = False
                self._status[broker_type].last_error = str(e)
                logger.error(f"Error connecting to {broker_type.value}: {e}")

        return results

    def disconnect_all(self) -> None:
        """Disconnect from all brokers."""
        for broker_type, broker in self._brokers.items():
            try:
                broker.disconnect()
                self._status[broker_type].connected = False
            except Exception as e:
                logger.error(f"Error disconnecting from {broker_type.value}: {e}")

    def health_check(self) -> Dict[BrokerType, BrokerStatus]:
        """Check health of all brokers."""
        for broker_type, broker in self._brokers.items():
            status = self._status[broker_type]
            try:
                start = time.time()
                success = broker.ping()
                latency = (time.time() - start) * 1000

                status.connected = success
                status.last_ping = datetime.now()
                status.latency_ms = latency

                if not success:
                    status.error_count += 1

            except Exception as e:
                status.connected = False
                status.error_count += 1
                status.last_error = str(e)

        return dict(self._status)

    # ==================== Order Routing ====================

    def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy: Optional[RoutingStrategy] = None,
        preferred_broker: Optional[BrokerType] = None,
        **kwargs
    ) -> BrokerOrder:
        """
        Route order to optimal broker.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            strategy: Override routing strategy
            preferred_broker: Force specific broker
            **kwargs: Additional order parameters

        Returns:
            Executed order
        """
        routing_strategy = strategy or self.strategy

        # Get routing decision
        decision = self._get_routing_decision(
            symbol=symbol,
            side=side,
            quantity=quantity,
            strategy=routing_strategy,
            preferred_broker=preferred_broker
        )

        if not decision.broker:
            raise RuntimeError(f"No available broker for {symbol}")

        logger.info(
            f"Routing {side.value} {quantity} {symbol} to {decision.broker.config.broker_type.value} "
            f"({decision.reason})"
        )

        # Execute with retries
        last_error = None
        brokers_to_try = [decision.broker] + decision.alternatives

        for broker in brokers_to_try[:self.max_retries]:
            try:
                order = broker.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    **kwargs
                )

                # Track order
                self._order_history[broker.config.broker_type].append(order)

                if order.status not in (OrderStatus.ERROR, OrderStatus.REJECTED):
                    return order

                last_error = order.error_message

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order failed on {broker.config.broker_type.value}: {e}")
                continue

        # All brokers failed
        error_order = BrokerOrder(
            internal_id=f"FAILED_{datetime.now().timestamp()}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=OrderStatus.ERROR,
            error_message=f"All brokers failed: {last_error}"
        )
        return error_order

    def _get_routing_decision(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        strategy: RoutingStrategy,
        preferred_broker: Optional[BrokerType] = None
    ) -> RoutingDecision:
        """Get routing decision based on strategy."""
        active_brokers = self.get_active_brokers()

        if not active_brokers:
            return RoutingDecision(broker=None, reason="No active brokers")

        # Filter brokers that support the symbol
        eligible = [
            b for b in active_brokers
            if symbol in b.config.symbols or not b.config.symbols
        ]

        if not eligible:
            return RoutingDecision(broker=None, reason=f"No broker supports {symbol}")

        # Handle preferred broker
        if preferred_broker:
            broker = self._brokers.get(preferred_broker)
            if broker and broker.is_connected():
                alternatives = [b for b in eligible if b != broker]
                return RoutingDecision(
                    broker=broker,
                    reason="Preferred broker",
                    alternatives=alternatives
                )

        # Apply strategy
        if strategy == RoutingStrategy.BEST_SPREAD:
            return self._route_best_spread(symbol, eligible)

        elif strategy == RoutingStrategy.PRIORITY:
            return self._route_priority(eligible)

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(eligible)

        elif strategy == RoutingStrategy.FAILOVER:
            return self._route_failover(eligible)

        else:
            # Default to priority
            return self._route_priority(eligible)

    def _route_best_spread(
        self,
        symbol: str,
        brokers: List[BrokerBase]
    ) -> RoutingDecision:
        """Route to broker with best spread."""
        spreads = []

        for broker in brokers:
            try:
                spread = self._get_spread_cached(symbol, broker)
                spreads.append((broker, spread))
            except Exception:
                continue

        if not spreads:
            return self._route_priority(brokers)

        # Sort by spread (lowest first)
        spreads.sort(key=lambda x: x[1])

        best_broker, best_spread = spreads[0]
        alternatives = [b for b, s in spreads[1:]]

        return RoutingDecision(
            broker=best_broker,
            reason=f"Best spread ({best_spread:.1f} pips)",
            spread=best_spread,
            alternatives=alternatives
        )

    def _route_priority(self, brokers: List[BrokerBase]) -> RoutingDecision:
        """Route based on broker priority."""
        # Already sorted by priority
        return RoutingDecision(
            broker=brokers[0],
            reason=f"Priority {brokers[0].config.priority}",
            alternatives=brokers[1:]
        )

    def _route_round_robin(self, brokers: List[BrokerBase]) -> RoutingDecision:
        """Distribute orders evenly across brokers."""
        with self._lock:
            self._rr_counter += 1
            idx = self._rr_counter % len(brokers)

        broker = brokers[idx]
        alternatives = brokers[:idx] + brokers[idx+1:]

        return RoutingDecision(
            broker=broker,
            reason="Round robin",
            alternatives=alternatives
        )

    def _route_failover(self, brokers: List[BrokerBase]) -> RoutingDecision:
        """Primary broker with failover."""
        return RoutingDecision(
            broker=brokers[0],
            reason="Primary (failover enabled)",
            alternatives=brokers[1:]
        )

    def _get_spread_cached(
        self,
        symbol: str,
        broker: BrokerBase,
        max_age_seconds: float = 5.0
    ) -> float:
        """Get spread with caching."""
        broker_type = broker.config.broker_type
        cache_time = self._spread_cache_time.get(symbol)

        if cache_time and (datetime.now() - cache_time).total_seconds() < max_age_seconds:
            if broker_type in self._spread_cache[symbol]:
                return self._spread_cache[symbol][broker_type]

        # Fetch fresh spread
        spread = broker.get_spread(symbol)
        self._spread_cache[symbol][broker_type] = spread
        self._spread_cache_time[symbol] = datetime.now()

        return spread

    # ==================== Aggregated Operations ====================

    def get_all_positions(self, symbol: Optional[str] = None) -> Dict[BrokerType, List[BrokerPosition]]:
        """Get positions from all brokers."""
        positions = {}
        for broker_type, broker in self._brokers.items():
            if broker.is_connected():
                try:
                    positions[broker_type] = broker.get_positions(symbol)
                except Exception as e:
                    logger.error(f"Failed to get positions from {broker_type.value}: {e}")
                    positions[broker_type] = []
        return positions

    def get_all_quotes(self, symbol: str) -> Dict[BrokerType, BrokerQuote]:
        """Get quotes from all brokers for comparison."""
        quotes = {}
        for broker_type, broker in self._brokers.items():
            if broker.is_connected():
                try:
                    quotes[broker_type] = broker.get_quote(symbol)
                except Exception:
                    pass
        return quotes

    def get_best_quote(self, symbol: str) -> Optional[BrokerQuote]:
        """Get best quote (lowest spread) across all brokers."""
        quotes = self.get_all_quotes(symbol)
        if not quotes:
            return None

        valid_quotes = [q for q in quotes.values() if q.is_valid]
        if not valid_quotes:
            return None

        return min(valid_quotes, key=lambda q: q.spread_pips)

    def get_aggregate_balance(self) -> Dict[str, float]:
        """Get aggregate balance across all brokers."""
        total_balance = 0.0
        total_equity = 0.0
        total_margin_used = 0.0

        by_broker = {}

        for broker_type, broker in self._brokers.items():
            if broker.is_connected():
                try:
                    account = broker.get_account()
                    total_balance += account.balance
                    total_equity += account.equity
                    total_margin_used += account.margin_used
                    by_broker[broker_type.value] = account.balance
                except Exception:
                    pass

        return {
            'total_balance': total_balance,
            'total_equity': total_equity,
            'total_margin_used': total_margin_used,
            'by_broker': by_broker
        }

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            'brokers': len(self._brokers),
            'connected': sum(1 for b in self._brokers.values() if b.is_connected()),
            'strategy': self.strategy.value,
            'status': {
                bt.value: {
                    'connected': s.connected,
                    'latency_ms': s.latency_ms,
                    'error_count': s.error_count,
                    'orders_today': s.orders_today,
                }
                for bt, s in self._status.items()
            },
            'order_counts': {
                bt.value: len(orders)
                for bt, orders in self._order_history.items()
            }
        }

    def __enter__(self):
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_all()


def create_multi_broker_router(
    brokers_config: Dict[str, Dict[str, Any]],
    strategy: RoutingStrategy = RoutingStrategy.BEST_SPREAD
) -> BrokerRouter:
    """
    Factory function to create configured multi-broker router.

    Args:
        brokers_config: Dict of broker configs keyed by broker name
        strategy: Default routing strategy

    Returns:
        Configured BrokerRouter

    Example:
        config = {
            'ib': {
                'host': 'localhost',
                'port': 4004,
                'account_id': 'DUO423364',
                'client_id': 1,
            },
            'oanda': {
                'api_key': 'xxx',
                'account_id': '101-001-xxx',
            },
            'forex_com': {
                'username': 'xxx',
                'password': 'xxx',
                'app_key': 'xxx',
            },
            'tastyfx': {
                'api_key': 'xxx',
                'username': 'xxx',
                'password': 'xxx',
            }
        }
        router = create_multi_broker_router(config)
    """
    router = BrokerRouter(strategy=strategy)

    # Create IB broker
    if 'ib' in brokers_config:
        ib_config = brokers_config['ib']
        broker = create_ib_broker(
            host=ib_config.get('host', 'localhost'),
            port=ib_config.get('port', 4004),
            account_id=ib_config.get('account_id', 'DUO423364'),
            client_id=ib_config.get('client_id', 1),
            paper=ib_config.get('paper', True)
        )
        router.add_broker(broker)

    # Create OANDA broker
    if 'oanda' in brokers_config:
        oanda_config = brokers_config['oanda']
        broker = create_oanda_broker(
            api_key=oanda_config['api_key'],
            account_id=oanda_config['account_id'],
            paper=oanda_config.get('paper', True)
        )
        router.add_broker(broker)

    # Create Forex.com broker
    if 'forex_com' in brokers_config:
        fc_config = brokers_config['forex_com']
        broker = create_forex_com_broker(
            username=fc_config['username'],
            password=fc_config['password'],
            app_key=fc_config['app_key'],
            paper=fc_config.get('paper', True)
        )
        router.add_broker(broker)

    # Create tastyfx broker
    if 'tastyfx' in brokers_config:
        tfx_config = brokers_config['tastyfx']
        broker = create_tastyfx_broker(
            api_key=tfx_config['api_key'],
            username=tfx_config['username'],
            password=tfx_config['password'],
            paper=tfx_config.get('paper', True)
        )
        router.add_broker(broker)

    # Create IG broker
    if 'ig' in brokers_config:
        ig_config = brokers_config['ig']
        broker = create_ig_broker(
            api_key=ig_config['api_key'],
            username=ig_config['username'],
            password=ig_config['password'],
            paper=ig_config.get('paper', True)
        )
        router.add_broker(broker)

    return router
