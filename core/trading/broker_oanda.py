"""
OANDA v20 API Adapter
=====================
Professional OANDA integration via v20 REST API.

Features:
- Market and limit orders
- Position management
- Real-time streaming quotes
- Account monitoring
- Supports practice and live accounts

API Docs: https://developer.oanda.com/rest-live-v20/introduction/
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
import logging
import time
import json

from .broker_base import (
    BrokerBase, BrokerConfig, BrokerType,
    BrokerOrder, BrokerPosition, BrokerAccount, BrokerQuote,
    OrderSide, OrderType, OrderStatus, PositionSide
)

logger = logging.getLogger(__name__)


class OANDABroker(BrokerBase):
    """
    OANDA v20 API adapter.

    Uses REST API for trading operations.
    """

    # API endpoints
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
    STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"

    # OANDA symbol format (underscore)
    SYMBOL_MAP = {
        'EURUSD': 'EUR_USD',
        'GBPUSD': 'GBP_USD',
        'USDJPY': 'USD_JPY',
        'USDCHF': 'USD_CHF',
        'AUDUSD': 'AUD_USD',
        'USDCAD': 'USD_CAD',
        'NZDUSD': 'NZD_USD',
        'EURJPY': 'EUR_JPY',
        'GBPJPY': 'GBP_JPY',
        'EURGBP': 'EUR_GBP',
        'EURCHF': 'EUR_CHF',
        'AUDJPY': 'AUD_JPY',
        'EURAUD': 'EUR_AUD',
        'GBPAUD': 'GBP_AUD',
        'AUDCAD': 'AUD_CAD',
        'AUDCHF': 'AUD_CHF',
        'AUDNZD': 'AUD_NZD',
        'CADCHF': 'CAD_CHF',
        'CADJPY': 'CAD_JPY',
        'CHFJPY': 'CHF_JPY',
        'EURCAD': 'EUR_CAD',
        'EURNZD': 'EUR_NZD',
        'GBPCAD': 'GBP_CAD',
        'GBPCHF': 'GBP_CHF',
        'GBPNZD': 'GBP_NZD',
        'NZDCAD': 'NZD_CAD',
        'NZDCHF': 'NZD_CHF',
        'NZDJPY': 'NZD_JPY',
    }

    REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}

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
        """Initialize OANDA broker."""
        super().__init__(config)
        self._session = None
        self._base_url = self.PRACTICE_URL if config.paper else self.LIVE_URL
        self._stream_url = self.STREAM_PRACTICE_URL if config.paper else self.STREAM_LIVE_URL

    def connect(self) -> bool:
        """Connect to OANDA API."""
        if self._connected:
            return True

        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
                'Accept-Datetime-Format': 'RFC3339'
            })

            # Test connection by fetching account
            response = self._request('GET', f'/v3/accounts/{self.config.account_id}')

            if response.status_code == 200:
                self._connected = True
                account_data = response.json().get('account', {})
                logger.info(f"Connected to OANDA ({self._base_url})")
                logger.info(f"Account: {self.config.account_id}, Balance: {account_data.get('balance')}")
                return True
            else:
                logger.error(f"OANDA connection failed: {response.text}")
                return False

        except ImportError:
            logger.error("requests not installed: pip install requests")
            return False

        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from OANDA."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from OANDA")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._session is not None

    def ping(self) -> bool:
        """Ping OANDA API."""
        if not self.is_connected():
            return False
        try:
            response = self._request('GET', f'/v3/accounts/{self.config.account_id}')
            return response.status_code == 200
        except Exception:
            return False

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None):
        """Make API request."""
        url = f"{self._base_url}{endpoint}"

        if method == 'GET':
            return self._session.get(url, timeout=self.config.request_timeout)
        elif method == 'POST':
            return self._session.post(url, json=data, timeout=self.config.request_timeout)
        elif method == 'PUT':
            return self._session.put(url, json=data, timeout=self.config.request_timeout)
        elif method == 'DELETE':
            return self._session.delete(url, timeout=self.config.request_timeout)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _to_oanda_symbol(self, symbol: str) -> str:
        """Convert to OANDA symbol format."""
        return self.SYMBOL_MAP.get(symbol, symbol.replace('', '_'))

    def _from_oanda_symbol(self, oanda_symbol: str) -> str:
        """Convert from OANDA symbol format."""
        return self.REVERSE_SYMBOL_MAP.get(oanda_symbol, oanda_symbol.replace('_', ''))

    # ==================== Account ====================

    def get_account(self) -> BrokerAccount:
        """Get OANDA account info."""
        if not self.is_connected():
            raise ConnectionError("Not connected to OANDA")

        response = self._request('GET', f'/v3/accounts/{self.config.account_id}')
        data = response.json().get('account', {})

        return BrokerAccount(
            account_id=self.config.account_id,
            broker_type=BrokerType.OANDA,
            balance=float(data.get('balance', 0)),
            equity=float(data.get('NAV', 0)),
            margin_used=float(data.get('marginUsed', 0)),
            margin_available=float(data.get('marginAvailable', 0)),
            unrealized_pnl=float(data.get('unrealizedPL', 0)),
            realized_pnl=float(data.get('pl', 0)),
            currency=data.get('currency', 'USD'),
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
        """Submit order to OANDA."""
        if not self.is_connected():
            raise ConnectionError("Not connected to OANDA")

        # Create order record
        order = BrokerOrder(
            internal_id=self._generate_order_id(),
            broker_type=BrokerType.OANDA,
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
            # Convert symbol
            oanda_symbol = self._to_oanda_symbol(symbol)

            # Calculate units (positive for buy, negative for sell)
            units = int(quantity) if side == OrderSide.BUY else int(-quantity)

            # Build order request
            order_request = {
                'order': {
                    'instrument': oanda_symbol,
                    'units': str(units),
                    'timeInForce': 'FOK',  # Fill or kill for market
                    'positionFill': 'DEFAULT'
                }
            }

            if order_type == OrderType.MARKET:
                order_request['order']['type'] = 'MARKET'
            elif order_type == OrderType.LIMIT:
                order_request['order']['type'] = 'LIMIT'
                order_request['order']['price'] = str(limit_price)
                order_request['order']['timeInForce'] = 'GTC'
            elif order_type == OrderType.STOP:
                order_request['order']['type'] = 'STOP'
                order_request['order']['price'] = str(stop_price)
                order_request['order']['timeInForce'] = 'GTC'
            elif order_type == OrderType.STOP_LIMIT:
                order_request['order']['type'] = 'STOP'
                order_request['order']['price'] = str(stop_price)
                order_request['order']['priceBound'] = str(limit_price)
                order_request['order']['timeInForce'] = 'GTC'

            # Submit order
            response = self._request(
                'POST',
                f'/v3/accounts/{self.config.account_id}/orders',
                order_request
            )

            result = response.json()

            if response.status_code in (200, 201):
                # Check for fill
                if 'orderFillTransaction' in result:
                    fill = result['orderFillTransaction']
                    order.broker_id = fill.get('id')
                    order.status = OrderStatus.FILLED
                    order.filled_qty = abs(float(fill.get('units', 0)))
                    order.avg_fill_price = float(fill.get('price', 0))
                    order.filled_at = datetime.now()
                    order.commission = float(fill.get('commission', 0))
                elif 'orderCreateTransaction' in result:
                    create = result['orderCreateTransaction']
                    order.broker_id = create.get('id')
                    order.status = OrderStatus.SUBMITTED
                else:
                    order.status = OrderStatus.SUBMITTED

                order.submitted_at = datetime.now()

            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get('errorMessage', str(result))

            order.updated_at = datetime.now()
            self._update_order(order)

            logger.info(
                f"OANDA Order {order.internal_id}: {order.side.value} {order.quantity} {symbol} "
                f"-> {order.status.value}"
            )

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.updated_at = datetime.now()
            self._update_order(order)
            logger.error(f"OANDA order error: {e}")
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an OANDA order."""
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            if order.broker_id:
                response = self._request(
                    'PUT',
                    f'/v3/accounts/{self.config.account_id}/orders/{order.broker_id}/cancel'
                )

                if response.status_code == 200:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    return True

            return False

        except Exception as e:
            logger.error(f"OANDA cancel error: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open orders."""
        if not self.is_connected():
            return []

        response = self._request(
            'GET',
            f'/v3/accounts/{self.config.account_id}/pendingOrders'
        )

        orders = []
        if response.status_code == 200:
            for o in response.json().get('orders', []):
                oanda_symbol = o.get('instrument', '')
                sym = self._from_oanda_symbol(oanda_symbol)

                if symbol and sym != symbol:
                    continue

                broker_order = BrokerOrder(
                    internal_id=f"oanda_{o['id']}",
                    broker_id=o['id'],
                    broker_type=BrokerType.OANDA,
                    symbol=sym,
                    side=OrderSide.BUY if float(o.get('units', 0)) > 0 else OrderSide.SELL,
                    quantity=abs(float(o.get('units', 0))),
                    status=OrderStatus.SUBMITTED
                )
                orders.append(broker_order)

        return orders

    # ==================== Positions ====================

    def get_positions(self, symbol: Optional[str] = None) -> List[BrokerPosition]:
        """Get all positions."""
        if not self.is_connected():
            return []

        response = self._request(
            'GET',
            f'/v3/accounts/{self.config.account_id}/openPositions'
        )

        positions = []
        if response.status_code == 200:
            for p in response.json().get('positions', []):
                oanda_symbol = p.get('instrument', '')
                sym = self._from_oanda_symbol(oanda_symbol)

                if symbol and sym != symbol:
                    continue

                long_units = float(p.get('long', {}).get('units', 0))
                short_units = abs(float(p.get('short', {}).get('units', 0)))

                if long_units > 0:
                    side = PositionSide.LONG
                    qty = long_units
                    avg_price = float(p.get('long', {}).get('averagePrice', 0))
                    unrealized = float(p.get('long', {}).get('unrealizedPL', 0))
                elif short_units > 0:
                    side = PositionSide.SHORT
                    qty = short_units
                    avg_price = float(p.get('short', {}).get('averagePrice', 0))
                    unrealized = float(p.get('short', {}).get('unrealizedPL', 0))
                else:
                    continue

                bp = BrokerPosition(
                    symbol=sym,
                    broker_type=BrokerType.OANDA,
                    account_id=self.config.account_id,
                    side=side,
                    quantity=qty,
                    avg_entry_price=avg_price,
                    unrealized_pnl=unrealized,
                    updated_at=datetime.now()
                )
                positions.append(bp)

        return positions

    def close_position(self, symbol: str, quantity: Optional[float] = None) -> BrokerOrder:
        """Close a position."""
        oanda_symbol = self._to_oanda_symbol(symbol)

        # Determine close amount
        close_data = {}
        if quantity:
            positions = self.get_positions(symbol)
            if positions:
                side = positions[0].side
                units = int(quantity) if side == PositionSide.SHORT else int(-quantity)
                if side == PositionSide.LONG:
                    close_data = {'longUnits': str(abs(units))}
                else:
                    close_data = {'shortUnits': str(abs(units))}
        else:
            close_data = {'longUnits': 'ALL', 'shortUnits': 'ALL'}

        response = self._request(
            'PUT',
            f'/v3/accounts/{self.config.account_id}/positions/{oanda_symbol}/close',
            close_data
        )

        # Create order record for the close
        order = BrokerOrder(
            internal_id=self._generate_order_id(),
            broker_type=BrokerType.OANDA,
            symbol=symbol,
            side=OrderSide.SELL,  # Simplified
            order_type=OrderType.MARKET,
            quantity=quantity or 0,
            status=OrderStatus.FILLED if response.status_code == 200 else OrderStatus.ERROR,
            created_at=datetime.now()
        )

        if response.status_code == 200:
            result = response.json()
            if 'longOrderFillTransaction' in result:
                fill = result['longOrderFillTransaction']
                order.filled_qty = abs(float(fill.get('units', 0)))
                order.avg_fill_price = float(fill.get('price', 0))
            elif 'shortOrderFillTransaction' in result:
                fill = result['shortOrderFillTransaction']
                order.filled_qty = abs(float(fill.get('units', 0)))
                order.avg_fill_price = float(fill.get('price', 0))

        self._update_order(order)
        return order

    # ==================== Market Data ====================

    def get_quote(self, symbol: str) -> BrokerQuote:
        """Get current quote."""
        if not self.is_connected():
            raise ConnectionError("Not connected to OANDA")

        oanda_symbol = self._to_oanda_symbol(symbol)

        response = self._request(
            'GET',
            f'/v3/accounts/{self.config.account_id}/pricing?instruments={oanda_symbol}'
        )

        if response.status_code == 200:
            prices = response.json().get('prices', [])
            if prices:
                p = prices[0]
                bid = float(p.get('bids', [{}])[0].get('price', 0))
                ask = float(p.get('asks', [{}])[0].get('price', 0))

                pip_value = self.PIP_VALUES.get(symbol, 0.0001)
                spread = ask - bid
                spread_pips = spread / pip_value if pip_value > 0 else 0

                return BrokerQuote(
                    symbol=symbol,
                    broker_type=BrokerType.OANDA,
                    bid=bid,
                    ask=ask,
                    mid=(bid + ask) / 2,
                    spread=spread,
                    spread_pips=spread_pips,
                    timestamp=datetime.now()
                )

        return BrokerQuote(symbol=symbol, broker_type=BrokerType.OANDA)

    def get_spread(self, symbol: str) -> float:
        """Get current spread in pips."""
        quote = self.get_quote(symbol)
        return quote.spread_pips

    # ==================== Symbol Info ====================

    def get_symbols(self) -> List[str]:
        """Get tradeable symbols."""
        return list(self.SYMBOL_MAP.keys())

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.PIP_VALUES.get(symbol, 0.0001)

    def get_min_quantity(self, symbol: str) -> float:
        """Get minimum order quantity (OANDA allows 1 unit)."""
        return 1.0


def create_oanda_broker(
    api_key: str,
    account_id: str,
    paper: bool = True
) -> OANDABroker:
    """
    Factory function to create OANDA broker.

    Args:
        api_key: OANDA API key
        account_id: OANDA account ID
        paper: Whether practice account

    Returns:
        Configured OANDABroker instance
    """
    config = BrokerConfig(
        broker_type=BrokerType.OANDA,
        name="OANDA",
        api_key=api_key,
        account_id=account_id,
        paper=paper,
        symbols=list(OANDABroker.SYMBOL_MAP.keys()),
        priority=2
    )

    return OANDABroker(config)
