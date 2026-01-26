"""
Forex.com (GAIN Capital) Adapter
================================
Professional Forex.com integration via REST API.

Forex.com uses GAIN Capital's FOREX.com REST Trading API.
API Docs: https://developer.forex.com/

Features:
- Market and limit orders
- Position management
- Real-time quotes
- Account monitoring
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import logging
import hashlib
import hmac
import time
import json

from .broker_base import (
    BrokerBase, BrokerConfig, BrokerType,
    BrokerOrder, BrokerPosition, BrokerAccount, BrokerQuote,
    OrderSide, OrderType, OrderStatus, PositionSide
)

logger = logging.getLogger(__name__)


class ForexComBroker(BrokerBase):
    """
    Forex.com (GAIN Capital) API adapter.

    Uses REST API with OAuth2 authentication.
    """

    # API endpoints
    DEMO_URL = "https://ciapi.cityindex.com/TradingAPI"
    LIVE_URL = "https://ciapi.forex.com/TradingAPI"

    # Symbol mapping (Forex.com uses market IDs)
    SYMBOL_TO_MARKET_ID = {
        'EURUSD': 401484347,
        'GBPUSD': 401484352,
        'USDJPY': 401484360,
        'USDCHF': 401484363,
        'AUDUSD': 401484329,
        'USDCAD': 401484358,
        'NZDUSD': 401484355,
        'EURJPY': 401484344,
        'GBPJPY': 401484349,
        'EURGBP': 401484341,
        'EURCHF': 401484338,
        'AUDJPY': 401484326,
    }

    MARKET_ID_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_MARKET_ID.items()}

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
        """Initialize Forex.com broker."""
        super().__init__(config)
        self._session = None
        self._base_url = self.DEMO_URL if config.paper else self.LIVE_URL
        self._session_token = None
        self._trading_account_id = None

    def connect(self) -> bool:
        """Connect to Forex.com API."""
        if self._connected:
            return True

        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json'
            })

            # Authenticate
            auth_response = self._session.post(
                f"{self._base_url}/session",
                json={
                    'UserName': self.config.extra.get('username', ''),
                    'Password': self.config.api_key,
                    'AppKey': self.config.api_secret,
                    'AppVersion': '1.0',
                    'AppComments': 'ForexML Trading System'
                },
                timeout=self.config.connect_timeout
            )

            if auth_response.status_code == 200:
                result = auth_response.json()
                self._session_token = result.get('Session')
                self._trading_account_id = result.get('TradingAccounts', [{}])[0].get('TradingAccountId')

                # Add session header for subsequent requests
                self._session.headers['Session'] = self._session_token
                self._session.headers['UserName'] = self.config.extra.get('username', '')

                self._connected = True
                logger.info(f"Connected to Forex.com ({self._base_url})")
                logger.info(f"Trading Account: {self._trading_account_id}")
                return True
            else:
                logger.error(f"Forex.com auth failed: {auth_response.text}")
                return False

        except ImportError:
            logger.error("requests not installed: pip install requests")
            return False

        except Exception as e:
            logger.error(f"Forex.com connection error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Forex.com."""
        if self._session and self._session_token:
            try:
                self._session.delete(
                    f"{self._base_url}/session",
                    timeout=5.0
                )
            except Exception:
                pass

        if self._session:
            self._session.close()
            self._session = None

        self._session_token = None
        self._connected = False
        logger.info("Disconnected from Forex.com")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._session_token is not None

    def ping(self) -> bool:
        """Ping Forex.com API."""
        if not self.is_connected():
            return False
        try:
            response = self._session.get(
                f"{self._base_url}/UserAccount/ClientAndTradingAccount",
                timeout=self.config.request_timeout
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_market_id(self, symbol: str) -> int:
        """Get Forex.com market ID for symbol."""
        market_id = self.SYMBOL_TO_MARKET_ID.get(symbol)
        if not market_id:
            raise ValueError(f"Unknown symbol: {symbol}")
        return market_id

    # ==================== Account ====================

    def get_account(self) -> BrokerAccount:
        """Get account info."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Forex.com")

        response = self._session.get(
            f"{self._base_url}/UserAccount/ClientAndTradingAccount",
            timeout=self.config.request_timeout
        )

        if response.status_code == 200:
            data = response.json()
            trading_account = data.get('TradingAccounts', [{}])[0]

            return BrokerAccount(
                account_id=str(self._trading_account_id),
                broker_type=BrokerType.FOREX_COM,
                balance=float(trading_account.get('TradingAccountBalance', 0)),
                equity=float(trading_account.get('TradingAccountEquity', 0)),
                margin_used=float(trading_account.get('TradingAccountMarginUsed', 0)),
                margin_available=float(trading_account.get('TradingAccountMarginAvailable', 0)),
                currency=trading_account.get('TradingAccountCurrency', 'USD'),
                is_active=True,
                updated_at=datetime.now()
            )

        raise RuntimeError(f"Failed to get account: {response.text}")

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
        """Submit order to Forex.com."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Forex.com")

        order = BrokerOrder(
            internal_id=self._generate_order_id(),
            broker_type=BrokerType.FOREX_COM,
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
            market_id = self._get_market_id(symbol)

            # Direction: true = buy, false = sell
            direction = side == OrderSide.BUY

            order_request = {
                'MarketId': market_id,
                'Direction': 'buy' if direction else 'sell',
                'Quantity': quantity,
                'TradingAccountId': self._trading_account_id,
                'PositionMethodId': 1,  # LongOrShortOnly
            }

            if order_type == OrderType.MARKET:
                # Market order endpoint
                endpoint = f"{self._base_url}/order/newtradeorder"
            elif order_type == OrderType.LIMIT:
                endpoint = f"{self._base_url}/order/newstoplimitorder"
                order_request['OrderType'] = 'Limit'
                order_request['TriggerPrice'] = limit_price
            elif order_type == OrderType.STOP:
                endpoint = f"{self._base_url}/order/newstoplimitorder"
                order_request['OrderType'] = 'Stop'
                order_request['TriggerPrice'] = stop_price
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            response = self._session.post(
                endpoint,
                json=order_request,
                timeout=self.config.request_timeout
            )

            result = response.json()

            if response.status_code == 200:
                order.broker_id = str(result.get('OrderId', ''))
                order.status = OrderStatus.FILLED if result.get('Status') == 1 else OrderStatus.SUBMITTED
                order.submitted_at = datetime.now()

                if order.status == OrderStatus.FILLED:
                    order.filled_qty = quantity
                    order.avg_fill_price = float(result.get('Price', 0))
                    order.filled_at = datetime.now()
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get('ErrorMessage', str(result))

            order.updated_at = datetime.now()
            self._update_order(order)

            logger.info(
                f"Forex.com Order {order.internal_id}: {order.side.value} {order.quantity} {symbol} "
                f"-> {order.status.value}"
            )

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.updated_at = datetime.now()
            self._update_order(order)
            logger.error(f"Forex.com order error: {e}")
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a Forex.com order."""
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            if order.broker_id:
                response = self._session.post(
                    f"{self._base_url}/order/cancel",
                    json={
                        'OrderId': int(order.broker_id),
                        'TradingAccountId': self._trading_account_id
                    },
                    timeout=self.config.request_timeout
                )

                if response.status_code == 200:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    return True

            return False

        except Exception as e:
            logger.error(f"Forex.com cancel error: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open orders."""
        if not self.is_connected():
            return []

        response = self._session.get(
            f"{self._base_url}/order/openpositions?TradingAccountId={self._trading_account_id}",
            timeout=self.config.request_timeout
        )

        orders = []
        if response.status_code == 200:
            for o in response.json().get('OpenPositions', []):
                market_id = o.get('MarketId')
                sym = self.MARKET_ID_TO_SYMBOL.get(market_id, str(market_id))

                if symbol and sym != symbol:
                    continue

                broker_order = BrokerOrder(
                    internal_id=f"forexcom_{o.get('OrderId')}",
                    broker_id=str(o.get('OrderId')),
                    broker_type=BrokerType.FOREX_COM,
                    symbol=sym,
                    side=OrderSide.BUY if o.get('Direction') == 'buy' else OrderSide.SELL,
                    quantity=float(o.get('Quantity', 0)),
                    status=OrderStatus.SUBMITTED
                )
                orders.append(broker_order)

        return orders

    # ==================== Positions ====================

    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """Get all positions."""
        if not self.is_connected():
            return []

        response = self._session.get(
            f"{self._base_url}/order/openpositions?TradingAccountId={self._trading_account_id}",
            timeout=self.config.request_timeout
        )

        positions = []
        if response.status_code == 200:
            for p in response.json().get('OpenPositions', []):
                market_id = p.get('MarketId')
                sym = self.MARKET_ID_TO_SYMBOL.get(market_id, str(market_id))

                if symbol and sym != symbol:
                    continue

                direction = p.get('Direction', '').lower()
                side = PositionSide.LONG if direction == 'buy' else PositionSide.SHORT

                bp = BrokerPosition(
                    symbol=sym,
                    broker_type=BrokerType.FOREX_COM,
                    account_id=str(self._trading_account_id),
                    side=side,
                    quantity=float(p.get('Quantity', 0)),
                    avg_entry_price=float(p.get('OpenPrice', 0)),
                    unrealized_pnl=float(p.get('UnrealisedPnL', 0)),
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
            raise ConnectionError("Not connected to Forex.com")

        market_id = self._get_market_id(symbol)

        response = self._session.get(
            f"{self._base_url}/market/{market_id}/tickhistory?PriceTicks=1",
            timeout=self.config.request_timeout
        )

        if response.status_code == 200:
            ticks = response.json().get('PriceTicks', [])
            if ticks:
                tick = ticks[0]
                bid = float(tick.get('Bid', 0))
                ask = float(tick.get('Ask', 0))

                pip_value = self.PIP_VALUES.get(symbol, 0.0001)
                spread = ask - bid
                spread_pips = spread / pip_value if pip_value > 0 else 0

                return BrokerQuote(
                    symbol=symbol,
                    broker_type=BrokerType.FOREX_COM,
                    bid=bid,
                    ask=ask,
                    mid=(bid + ask) / 2,
                    spread=spread,
                    spread_pips=spread_pips,
                    timestamp=datetime.now()
                )

        return BrokerQuote(symbol=symbol, broker_type=BrokerType.FOREX_COM)

    def get_spread(self, symbol: str) -> float:
        """Get current spread in pips."""
        quote = self.get_quote(symbol)
        return quote.spread_pips

    # ==================== Symbol Info ====================

    def get_symbols(self) -> List[str]:
        """Get tradeable symbols."""
        return list(self.SYMBOL_TO_MARKET_ID.keys())

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.PIP_VALUES.get(symbol, 0.0001)

    def get_min_quantity(self, symbol: str) -> float:
        """Get minimum order quantity."""
        return 1000.0  # Forex.com minimum is typically 1k


def create_forex_com_broker(
    username: str,
    password: str,
    app_key: str,
    paper: bool = True
) -> ForexComBroker:
    """
    Factory function to create Forex.com broker.

    Args:
        username: Forex.com username
        password: Forex.com password
        app_key: API app key
        paper: Whether demo account

    Returns:
        Configured ForexComBroker instance
    """
    config = BrokerConfig(
        broker_type=BrokerType.FOREX_COM,
        name="Forex.com",
        api_key=password,
        api_secret=app_key,
        paper=paper,
        extra={'username': username},
        symbols=list(ForexComBroker.SYMBOL_TO_MARKET_ID.keys()),
        priority=3
    )

    return ForexComBroker(config)
