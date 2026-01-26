"""
tastyfx / IG Markets Adapter
============================
Professional integration via IG Group REST API.

tastyfx is the US forex brand of IG Group.
Uses same API as IG Markets with US-specific endpoints.

API Docs: https://labs.ig.com/rest-trading-api-reference

Features:
- Market and limit orders
- Position management
- Real-time streaming
- Account monitoring
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


class TastyFXBroker(BrokerBase):
    """
    tastyfx (IG Group) API adapter.

    Uses IG REST API for trading operations.
    """

    # API endpoints
    DEMO_URL = "https://demo-api.ig.com/gateway/deal"
    LIVE_URL = "https://api.ig.com/gateway/deal"

    # IG uses EPIC codes for instruments
    SYMBOL_TO_EPIC = {
        'EURUSD': 'CS.D.EURUSD.MINI.IP',
        'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
        'USDJPY': 'CS.D.USDJPY.MINI.IP',
        'USDCHF': 'CS.D.USDCHF.MINI.IP',
        'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
        'USDCAD': 'CS.D.USDCAD.MINI.IP',
        'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
        'EURJPY': 'CS.D.EURJPY.MINI.IP',
        'GBPJPY': 'CS.D.GBPJPY.MINI.IP',
        'EURGBP': 'CS.D.EURGBP.MINI.IP',
        'EURCHF': 'CS.D.EURCHF.MINI.IP',
        'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
        'EURAUD': 'CS.D.EURAUD.MINI.IP',
        'GBPAUD': 'CS.D.GBPAUD.MINI.IP',
    }

    EPIC_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_EPIC.items()}

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
        """Initialize tastyfx broker."""
        super().__init__(config)
        self._session = None
        self._base_url = self.DEMO_URL if config.paper else self.LIVE_URL
        self._cst = None  # Client Security Token
        self._x_security_token = None
        self._account_id = None

    def connect(self) -> bool:
        """Connect to IG API."""
        if self._connected:
            return True

        try:
            import requests
            self._session = requests.Session()

            # IG requires specific headers
            self._session.headers.update({
                'Content-Type': 'application/json; charset=UTF-8',
                'Accept': 'application/json; charset=UTF-8',
                'X-IG-API-KEY': self.config.api_key,
                'Version': '3'
            })

            # Authenticate
            auth_response = self._session.post(
                f"{self._base_url}/session",
                json={
                    'identifier': self.config.extra.get('username', ''),
                    'password': self.config.api_secret,
                },
                timeout=self.config.connect_timeout
            )

            if auth_response.status_code == 200:
                # Extract security tokens from headers
                self._cst = auth_response.headers.get('CST')
                self._x_security_token = auth_response.headers.get('X-SECURITY-TOKEN')

                # Add tokens to session headers
                self._session.headers['CST'] = self._cst
                self._session.headers['X-SECURITY-TOKEN'] = self._x_security_token

                result = auth_response.json()
                self._account_id = result.get('currentAccountId')

                self._connected = True
                logger.info(f"Connected to tastyfx/IG ({self._base_url})")
                logger.info(f"Account: {self._account_id}")
                return True
            else:
                logger.error(f"tastyfx auth failed: {auth_response.text}")
                return False

        except ImportError:
            logger.error("requests not installed: pip install requests")
            return False

        except Exception as e:
            logger.error(f"tastyfx connection error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IG API."""
        if self._session and self._cst:
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

        self._cst = None
        self._x_security_token = None
        self._connected = False
        logger.info("Disconnected from tastyfx/IG")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._cst is not None

    def ping(self) -> bool:
        """Ping IG API."""
        if not self.is_connected():
            return False
        try:
            response = self._session.get(
                f"{self._base_url}/accounts",
                timeout=self.config.request_timeout
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_epic(self, symbol: str) -> str:
        """Get IG EPIC code for symbol."""
        epic = self.SYMBOL_TO_EPIC.get(symbol)
        if not epic:
            raise ValueError(f"Unknown symbol: {symbol}")
        return epic

    # ==================== Account ====================

    def get_account(self) -> BrokerAccount:
        """Get account info."""
        if not self.is_connected():
            raise ConnectionError("Not connected to tastyfx")

        response = self._session.get(
            f"{self._base_url}/accounts",
            timeout=self.config.request_timeout
        )

        if response.status_code == 200:
            accounts = response.json().get('accounts', [])
            account = next(
                (a for a in accounts if a.get('accountId') == self._account_id),
                accounts[0] if accounts else {}
            )

            balance_info = account.get('balance', {})

            return BrokerAccount(
                account_id=self._account_id,
                broker_type=BrokerType.TASTYFX,
                balance=float(balance_info.get('balance', 0)),
                equity=float(balance_info.get('available', 0)),
                margin_used=float(balance_info.get('deposit', 0)),
                margin_available=float(balance_info.get('available', 0)),
                unrealized_pnl=float(balance_info.get('profitLoss', 0)),
                currency=account.get('currency', 'USD'),
                is_active=account.get('status') == 'ENABLED',
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
        """Submit order to IG."""
        if not self.is_connected():
            raise ConnectionError("Not connected to tastyfx")

        order = BrokerOrder(
            internal_id=self._generate_order_id(),
            broker_type=BrokerType.TASTYFX,
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
            epic = self._get_epic(symbol)

            # IG order direction
            direction = "BUY" if side == OrderSide.BUY else "SELL"

            order_request = {
                'epic': epic,
                'expiry': '-',  # No expiry for spot forex
                'direction': direction,
                'size': quantity,
                'orderType': 'MARKET',
                'timeInForce': 'FILL_OR_KILL',
                'guaranteedStop': False,
                'forceOpen': False,
                'currencyCode': 'USD'
            }

            if order_type == OrderType.MARKET:
                # Use positions endpoint for market orders
                endpoint = f"{self._base_url}/positions/otc"
            elif order_type == OrderType.LIMIT:
                endpoint = f"{self._base_url}/workingorders/otc"
                order_request['orderType'] = 'LIMIT'
                order_request['level'] = limit_price
                order_request['timeInForce'] = 'GOOD_TILL_CANCELLED'
            elif order_type == OrderType.STOP:
                endpoint = f"{self._base_url}/workingorders/otc"
                order_request['orderType'] = 'STOP'
                order_request['level'] = stop_price
                order_request['timeInForce'] = 'GOOD_TILL_CANCELLED'
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            response = self._session.post(
                endpoint,
                json=order_request,
                timeout=self.config.request_timeout
            )

            result = response.json()

            if response.status_code == 200:
                deal_reference = result.get('dealReference')
                order.broker_id = deal_reference
                order.submitted_at = datetime.now()

                # Check deal status
                if order_type == OrderType.MARKET:
                    # Confirm the deal
                    confirm_response = self._session.get(
                        f"{self._base_url}/confirms/{deal_reference}",
                        timeout=self.config.request_timeout
                    )

                    if confirm_response.status_code == 200:
                        confirm = confirm_response.json()
                        if confirm.get('dealStatus') == 'ACCEPTED':
                            order.status = OrderStatus.FILLED
                            order.filled_qty = float(confirm.get('size', quantity))
                            order.avg_fill_price = float(confirm.get('level', 0))
                            order.filled_at = datetime.now()
                        else:
                            order.status = OrderStatus.REJECTED
                            order.error_message = confirm.get('reason', '')
                else:
                    order.status = OrderStatus.SUBMITTED

            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get('errorCode', str(result))

            order.updated_at = datetime.now()
            self._update_order(order)

            logger.info(
                f"tastyfx Order {order.internal_id}: {order.side.value} {order.quantity} {symbol} "
                f"-> {order.status.value}"
            )

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)
            order.updated_at = datetime.now()
            self._update_order(order)
            logger.error(f"tastyfx order error: {e}")
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an IG order."""
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            if order.broker_id:
                response = self._session.delete(
                    f"{self._base_url}/workingorders/otc/{order.broker_id}",
                    timeout=self.config.request_timeout
                )

                if response.status_code == 200:
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    return True

            return False

        except Exception as e:
            logger.error(f"tastyfx cancel error: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get open orders."""
        if not self.is_connected():
            return []

        response = self._session.get(
            f"{self._base_url}/workingorders",
            timeout=self.config.request_timeout
        )

        orders = []
        if response.status_code == 200:
            for wo in response.json().get('workingOrders', []):
                epic = wo.get('marketData', {}).get('epic', '')
                sym = self.EPIC_TO_SYMBOL.get(epic, epic)

                if symbol and sym != symbol:
                    continue

                broker_order = BrokerOrder(
                    internal_id=f"ig_{wo.get('workingOrderData', {}).get('dealId')}",
                    broker_id=wo.get('workingOrderData', {}).get('dealId'),
                    broker_type=BrokerType.TASTYFX,
                    symbol=sym,
                    side=OrderSide.BUY if wo.get('workingOrderData', {}).get('direction') == 'BUY' else OrderSide.SELL,
                    quantity=float(wo.get('workingOrderData', {}).get('orderSize', 0)),
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
            f"{self._base_url}/positions",
            timeout=self.config.request_timeout
        )

        positions = []
        if response.status_code == 200:
            for p in response.json().get('positions', []):
                pos_data = p.get('position', {})
                market_data = p.get('market', {})

                epic = market_data.get('epic', '')
                sym = self.EPIC_TO_SYMBOL.get(epic, epic)

                if symbol and sym != symbol:
                    continue

                direction = pos_data.get('direction', '').upper()
                side = PositionSide.LONG if direction == 'BUY' else PositionSide.SHORT

                bp = BrokerPosition(
                    symbol=sym,
                    broker_type=BrokerType.TASTYFX,
                    account_id=self._account_id,
                    side=side,
                    quantity=float(pos_data.get('dealSize', 0)),
                    avg_entry_price=float(pos_data.get('openLevel', 0)),
                    unrealized_pnl=float(market_data.get('netChange', 0)),
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
            raise ConnectionError("Not connected to tastyfx")

        epic = self._get_epic(symbol)

        response = self._session.get(
            f"{self._base_url}/markets/{epic}",
            timeout=self.config.request_timeout
        )

        if response.status_code == 200:
            market = response.json()
            snapshot = market.get('snapshot', {})

            bid = float(snapshot.get('bid', 0))
            offer = float(snapshot.get('offer', 0))

            pip_value = self.PIP_VALUES.get(symbol, 0.0001)
            spread = offer - bid
            spread_pips = spread / pip_value if pip_value > 0 else 0

            return BrokerQuote(
                symbol=symbol,
                broker_type=BrokerType.TASTYFX,
                bid=bid,
                ask=offer,
                mid=(bid + offer) / 2,
                spread=spread,
                spread_pips=spread_pips,
                timestamp=datetime.now()
            )

        return BrokerQuote(symbol=symbol, broker_type=BrokerType.TASTYFX)

    def get_spread(self, symbol: str) -> float:
        """Get current spread in pips."""
        quote = self.get_quote(symbol)
        return quote.spread_pips

    # ==================== Symbol Info ====================

    def get_symbols(self) -> List[str]:
        """Get tradeable symbols."""
        return list(self.SYMBOL_TO_EPIC.keys())

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol."""
        return self.PIP_VALUES.get(symbol, 0.0001)

    def get_min_quantity(self, symbol: str) -> float:
        """Get minimum order quantity."""
        return 0.5  # IG allows mini lots (0.5)


# Also create IG Markets broker (same as tastyfx but different branding)
class IGBroker(TastyFXBroker):
    """
    IG Markets broker (same API as tastyfx).

    Just changes the broker type for identification.
    """

    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        # Override broker type
        self.config.broker_type = BrokerType.IG


def create_tastyfx_broker(
    api_key: str,
    username: str,
    password: str,
    paper: bool = True
) -> TastyFXBroker:
    """
    Factory function to create tastyfx broker.

    Args:
        api_key: IG API key
        username: tastyfx username
        password: tastyfx password
        paper: Whether demo account

    Returns:
        Configured TastyFXBroker instance
    """
    config = BrokerConfig(
        broker_type=BrokerType.TASTYFX,
        name="tastyfx",
        api_key=api_key,
        api_secret=password,
        paper=paper,
        extra={'username': username},
        symbols=list(TastyFXBroker.SYMBOL_TO_EPIC.keys()),
        priority=4
    )

    return TastyFXBroker(config)


def create_ig_broker(
    api_key: str,
    username: str,
    password: str,
    paper: bool = True
) -> IGBroker:
    """
    Factory function to create IG Markets broker.

    Args:
        api_key: IG API key
        username: IG username
        password: IG password
        paper: Whether demo account

    Returns:
        Configured IGBroker instance
    """
    config = BrokerConfig(
        broker_type=BrokerType.IG,
        name="IG Markets",
        api_key=api_key,
        api_secret=password,
        paper=paper,
        extra={'username': username},
        symbols=list(TastyFXBroker.SYMBOL_TO_EPIC.keys()),
        priority=5
    )

    return IGBroker(config)
