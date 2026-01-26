"""
Trading Module
==============
Modular trading components for multi-symbol forex trading.

Multi-Broker Support (2026-01-18):
- Interactive Brokers (IB Gateway)
- OANDA v20 API
- Forex.com (GAIN Capital)
- tastyfx (IG Group)
- IG Markets
"""

from .signal import SignalGenerator, Signal
from .position import PositionManager, Position
from .executor import OrderExecutor, Order, OrderStatus
from .bot import TradingBot

# Multi-broker abstraction layer
from .broker_base import (
    BrokerBase,
    BrokerConfig,
    BrokerType,
    BrokerOrder,
    BrokerPosition,
    BrokerAccount,
    BrokerQuote,
    OrderSide,
    OrderType,
    OrderStatus as BrokerOrderStatus,
    PositionSide
)

# Broker implementations
from .broker_ib import IBBroker, create_ib_broker
from .broker_oanda import OANDABroker, create_oanda_broker
from .broker_forex_com import ForexComBroker, create_forex_com_broker
from .broker_tastyfx import TastyFXBroker, IGBroker, create_tastyfx_broker, create_ig_broker

# Multi-broker router
from .broker_router import (
    BrokerRouter,
    RoutingStrategy,
    RoutingDecision,
    BrokerStatus,
    create_multi_broker_router
)

__all__ = [
    # Original
    'SignalGenerator',
    'Signal',
    'PositionManager',
    'Position',
    'OrderExecutor',
    'Order',
    'OrderStatus',
    'TradingBot',
    # Broker base
    'BrokerBase',
    'BrokerConfig',
    'BrokerType',
    'BrokerOrder',
    'BrokerPosition',
    'BrokerAccount',
    'BrokerQuote',
    'OrderSide',
    'OrderType',
    'BrokerOrderStatus',
    'PositionSide',
    # Broker implementations
    'IBBroker',
    'create_ib_broker',
    'OANDABroker',
    'create_oanda_broker',
    'ForexComBroker',
    'create_forex_com_broker',
    'TastyFXBroker',
    'IGBroker',
    'create_tastyfx_broker',
    'create_ig_broker',
    # Router
    'BrokerRouter',
    'RoutingStrategy',
    'RoutingDecision',
    'BrokerStatus',
    'create_multi_broker_router',
]
