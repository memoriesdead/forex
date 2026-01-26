#!/usr/bin/env python3
"""
Multi-Broker Trading Bot
========================
Professional multi-exchange forex trading with intelligent order routing.

Supported Brokers:
1. Interactive Brokers (IB Gateway) - PRIMARY
2. OANDA v20 API
3. Forex.com (GAIN Capital)
4. tastyfx (IG Group)
5. IG Markets

Features:
- Best execution routing (lowest spread)
- Automatic failover between brokers
- Position aggregation across exchanges
- Real-time spread comparison
- Chinese Quant online learning
- 51 forex pairs, 575 features

Usage:
    python scripts/multi_broker_trading.py --mode paper --symbols EURUSD,GBPUSD,USDJPY
    python scripts/multi_broker_trading.py --mode paper --all-brokers
    python scripts/multi_broker_trading.py --status
"""

import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(PROJECT_ROOT / '.env.oracle')

from config.brokers import (
    load_broker_config_from_env,
    get_enabled_brokers,
    print_broker_status
)
from core.trading.broker_router import (
    BrokerRouter,
    RoutingStrategy,
    create_multi_broker_router
)
from core.trading.broker_base import OrderSide, OrderType, OrderStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultiBrokerTradingBot:
    """
    Multi-broker trading bot with intelligent order routing.

    Routes orders to optimal broker based on:
    - Spread comparison
    - Broker availability
    - Position limits
    - Rate limits
    """

    def __init__(
        self,
        symbols: List[str],
        capital: float = 100.0,
        mode: str = 'paper',
        routing_strategy: RoutingStrategy = RoutingStrategy.BEST_SPREAD,
        online_learning: bool = True
    ):
        """
        Initialize multi-broker trading bot.

        Args:
            symbols: Trading symbols
            capital: Starting capital (USD)
            mode: Trading mode ('paper' or 'live')
            routing_strategy: Order routing strategy
            online_learning: Enable online learning
        """
        self.symbols = symbols
        self.capital = capital
        self.mode = mode
        self.routing_strategy = routing_strategy
        self.online_learning = online_learning

        self.router: Optional[BrokerRouter] = None
        self.models = {}
        self.running = False

        # Performance tracking
        self.trades = []
        self.positions = {}
        self.daily_pnl = 0.0

    def initialize(self) -> bool:
        """Initialize the trading bot."""
        logger.info("=" * 60)
        logger.info("MULTI-BROKER FOREX TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Capital: ${self.capital:,.2f}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Routing: {self.routing_strategy.value}")
        logger.info(f"Online Learning: {'ENABLED' if self.online_learning else 'DISABLED'}")
        logger.info("=" * 60)

        # Load broker configurations
        broker_configs = load_broker_config_from_env()
        enabled = get_enabled_brokers()

        if not enabled:
            logger.error("No brokers enabled! Check your .env configuration.")
            logger.info("Run: python config/brokers.py for configuration template")
            return False

        logger.info(f"Enabled brokers: {', '.join(enabled)}")

        # Create router
        self.router = create_multi_broker_router(
            brokers_config=broker_configs,
            strategy=self.routing_strategy
        )

        # Connect to brokers
        logger.info("Connecting to brokers...")
        results = self.router.connect_all()

        connected = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        if connected:
            logger.info(f"Connected: {', '.join(str(b) for b in connected)}")
        if failed:
            logger.warning(f"Failed: {', '.join(str(b) for b in failed)}")

        if not connected:
            logger.error("No brokers connected!")
            return False

        # Load ML models
        self._load_models()

        return True

    def _load_models(self):
        """Load ML models for each symbol."""
        from core.models import ModelLoader

        logger.info("Loading ML models...")
        loader = ModelLoader()

        for symbol in self.symbols:
            try:
                model = loader.load(symbol)
                if model:
                    self.models[symbol] = model
                    logger.info(f"  Loaded model for {symbol}")
                else:
                    logger.warning(f"  No model found for {symbol}")
            except Exception as e:
                logger.warning(f"  Failed to load {symbol}: {e}")

        logger.info(f"Loaded {len(self.models)}/{len(self.symbols)} models")

    async def run(self):
        """Run the trading bot."""
        self.running = True
        logger.info("Starting trading loop...")

        try:
            while self.running:
                await self._trading_iteration()
                await asyncio.sleep(1.0)  # 1 second tick

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            if self.router:
                self.router.disconnect_all()

    async def _trading_iteration(self):
        """Single trading iteration."""
        for symbol in self.symbols:
            try:
                await self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    async def _process_symbol(self, symbol: str):
        """Process a single symbol."""
        # Get best quote across all brokers
        quote = self.router.get_best_quote(symbol)
        if not quote or not quote.is_valid:
            return

        # Get model prediction
        if symbol not in self.models:
            return

        model = self.models[symbol]

        # Generate features (simplified - use full feature engine in production)
        features = self._generate_features(symbol, quote)

        # Predict
        try:
            prediction = model.predict([features])[0]
            confidence = model.predict_proba([features])[0].max() if hasattr(model, 'predict_proba') else 0.6
        except Exception as e:
            logger.debug(f"Prediction error for {symbol}: {e}")
            return

        # Trading decision
        if confidence < 0.55:
            return  # Not confident enough

        # Check position
        positions = self.router.get_all_positions(symbol)
        total_position = sum(
            sum(p.quantity * (1 if p.side.value == 'long' else -1) for p in plist)
            for plist in positions.values()
        )

        # Determine action
        if prediction == 1 and total_position <= 0:
            # Buy signal
            await self._execute_trade(symbol, OrderSide.BUY, confidence)
        elif prediction == 0 and total_position >= 0:
            # Sell signal
            await self._execute_trade(symbol, OrderSide.SELL, confidence)

    def _generate_features(self, symbol: str, quote) -> List[float]:
        """Generate features for prediction (simplified)."""
        # In production, use full HFTFeatureEngine
        return [
            quote.bid,
            quote.ask,
            quote.mid,
            quote.spread,
            quote.spread_pips,
        ]

    async def _execute_trade(
        self,
        symbol: str,
        side: OrderSide,
        confidence: float
    ):
        """Execute a trade with intelligent routing."""
        # Calculate position size (Kelly criterion)
        win_prob = confidence
        risk_fraction = (win_prob - (1 - win_prob)) / 1.5  # Simplified Kelly
        risk_fraction = max(0, min(0.25, risk_fraction))  # Cap at 25%

        # Calculate quantity based on capital
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        position_value = self.capital * risk_fraction
        quantity = int(position_value * 100)  # Simplified lot calculation

        if quantity < 1000:
            return  # Too small

        # Route order
        logger.info(f"Executing {side.value} {quantity} {symbol} (conf: {confidence:.2%})")

        order = self.router.route_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )

        if order.status == OrderStatus.FILLED:
            logger.info(
                f"FILLED: {order.side.value} {order.filled_qty} {symbol} "
                f"@ {order.avg_fill_price:.5f} via {order.broker_type.value}"
            )
            self.trades.append(order)
        else:
            logger.warning(f"Order {order.status.value}: {order.error_message}")

    def get_status(self) -> Dict[str, Any]:
        """Get bot status."""
        status = {
            'running': self.running,
            'mode': self.mode,
            'capital': self.capital,
            'symbols': self.symbols,
            'models_loaded': len(self.models),
            'trades_today': len(self.trades),
            'daily_pnl': self.daily_pnl,
        }

        if self.router:
            status['router'] = self.router.get_stats()
            status['balances'] = self.router.get_aggregate_balance()

        return status

    def print_status(self):
        """Print current status."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("MULTI-BROKER TRADING BOT STATUS")
        print("=" * 60)
        print(f"Running:     {status['running']}")
        print(f"Mode:        {status['mode'].upper()}")
        print(f"Capital:     ${status['capital']:,.2f}")
        print(f"Symbols:     {', '.join(status['symbols'])}")
        print(f"Models:      {status['models_loaded']} loaded")
        print(f"Trades:      {status['trades_today']} today")
        print(f"Daily P&L:   ${status['daily_pnl']:+,.2f}")

        if 'balances' in status:
            print("\nAGGREGATE BALANCE:")
            print(f"  Total:     ${status['balances']['total_balance']:,.2f}")
            print(f"  Equity:    ${status['balances']['total_equity']:,.2f}")
            for broker, balance in status['balances'].get('by_broker', {}).items():
                print(f"  {broker}: ${balance:,.2f}")

        if 'router' in status:
            print("\nBROKER STATUS:")
            for broker, data in status['router'].get('status', {}).items():
                connected = "ONLINE" if data['connected'] else "OFFLINE"
                latency = f"{data['latency_ms']:.0f}ms" if data['latency_ms'] > 0 else "N/A"
                print(f"  {broker}: {connected} ({latency})")

        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Broker Forex Trading Bot"
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode'
    )
    parser.add_argument(
        '--symbols', '-s',
        default='EURUSD,GBPUSD,USDJPY',
        help='Comma-separated symbols'
    )
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=100.0,
        help='Starting capital (USD)'
    )
    parser.add_argument(
        '--routing',
        choices=['best_spread', 'priority', 'round_robin', 'failover'],
        default='best_spread',
        help='Order routing strategy'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show broker status and exit'
    )
    parser.add_argument(
        '--all-brokers',
        action='store_true',
        help='Show all available brokers'
    )
    parser.add_argument(
        '--no-online-learning',
        action='store_true',
        help='Disable online learning'
    )

    args = parser.parse_args()

    # Status check
    if args.status or args.all_brokers:
        print_broker_status()
        return

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Parse routing strategy
    routing_map = {
        'best_spread': RoutingStrategy.BEST_SPREAD,
        'priority': RoutingStrategy.PRIORITY,
        'round_robin': RoutingStrategy.ROUND_ROBIN,
        'failover': RoutingStrategy.FAILOVER,
    }
    routing = routing_map[args.routing]

    # Create and run bot
    bot = MultiBrokerTradingBot(
        symbols=symbols,
        capital=args.capital,
        mode=args.mode,
        routing_strategy=routing,
        online_learning=not args.no_online_learning
    )

    if not bot.initialize():
        logger.error("Failed to initialize bot")
        sys.exit(1)

    # Run
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        bot.print_status()


if __name__ == "__main__":
    main()
