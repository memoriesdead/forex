#!/usr/bin/env python3
"""
Execution Optimization HTTP API
===============================
REST API for FX execution optimization - deploy on Oracle Cloud.

Endpoints:
- GET  /health                      Health check
- POST /api/execution/optimize      Get optimal execution decision
- POST /api/execution/impact        Estimate market impact
- POST /api/execution/schedule/twap Create TWAP schedule
- POST /api/execution/schedule/vwap Create VWAP schedule
- POST /api/execution/schedule/ac   Create Almgren-Chriss trajectory
- GET  /api/execution/sessions      Get current session info
- GET  /api/execution/strategies    List available strategies
- POST /api/execution/compare       Compare strategies for a trade

Usage:
    python scripts/mcp_execution_api.py --port 8080

    # Test
    curl http://localhost:8080/health
    curl -X POST http://localhost:8080/api/execution/optimize \
        -H "Content-Type: application/json" \
        -d '{"symbol": "EURUSD", "direction": 1, "quantity": 100000}'
"""

import argparse
import json
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutionAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for execution optimization API."""

    # Lazy-loaded components
    _engine = None
    _impact_model = None
    _twap_scheduler = None
    _vwap_scheduler = None
    _ac_optimizer = None

    @classmethod
    def get_engine(cls):
        """Lazy load execution engine."""
        if cls._engine is None:
            try:
                from core.execution.optimization import get_execution_engine
                cls._engine = get_execution_engine()
                logger.info("Execution engine loaded")
            except ImportError as e:
                logger.error(f"Failed to load execution engine: {e}")
                raise
        return cls._engine

    @classmethod
    def get_impact_model(cls):
        """Lazy load market impact model."""
        if cls._impact_model is None:
            try:
                from core.execution.optimization import get_market_impact_model
                cls._impact_model = get_market_impact_model()
                logger.info("Market impact model loaded")
            except ImportError as e:
                logger.error(f"Failed to load impact model: {e}")
                raise
        return cls._impact_model

    @classmethod
    def get_twap_scheduler(cls):
        """Lazy load TWAP scheduler."""
        if cls._twap_scheduler is None:
            try:
                from core.execution.optimization import get_twap_scheduler
                cls._twap_scheduler = get_twap_scheduler()
                logger.info("TWAP scheduler loaded")
            except ImportError as e:
                logger.error(f"Failed to load TWAP scheduler: {e}")
                raise
        return cls._twap_scheduler

    @classmethod
    def get_vwap_scheduler(cls):
        """Lazy load VWAP scheduler."""
        if cls._vwap_scheduler is None:
            try:
                from core.execution.optimization import get_vwap_scheduler
                cls._vwap_scheduler = get_vwap_scheduler()
                logger.info("VWAP scheduler loaded")
            except ImportError as e:
                logger.error(f"Failed to load VWAP scheduler: {e}")
                raise
        return cls._vwap_scheduler

    @classmethod
    def get_ac_optimizer(cls):
        """Lazy load Almgren-Chriss optimizer."""
        if cls._ac_optimizer is None:
            try:
                from core.execution.optimization import get_ac_optimizer
                cls._ac_optimizer = get_ac_optimizer()
                logger.info("Almgren-Chriss optimizer loaded")
            except ImportError as e:
                logger.error(f"Failed to load AC optimizer: {e}")
                raise
        return cls._ac_optimizer

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def send_error_response(self, message: str, status: int = 400):
        """Send error response."""
        self.send_json_response({'error': message, 'success': False}, status)

    def read_json_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON body: {e}")
            return None

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {
            '/health': self.handle_health,
            '/api/execution/sessions': self.handle_get_sessions,
            '/api/execution/strategies': self.handle_get_strategies,
        }

        handler = routes.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.exception(f"Error handling GET {path}")
                self.send_error_response(str(e), 500)
        else:
            self.send_error_response(f"Unknown endpoint: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {
            '/api/execution/optimize': self.handle_optimize,
            '/api/execution/impact': self.handle_impact,
            '/api/execution/schedule/twap': self.handle_twap_schedule,
            '/api/execution/schedule/vwap': self.handle_vwap_schedule,
            '/api/execution/schedule/ac': self.handle_ac_trajectory,
            '/api/execution/compare': self.handle_compare,
        }

        handler = routes.get(path)
        if handler:
            try:
                body = self.read_json_body()
                if body is None:
                    self.send_error_response("Invalid JSON body")
                    return
                handler(body)
            except Exception as e:
                logger.exception(f"Error handling POST {path}")
                self.send_error_response(str(e), 500)
        else:
            self.send_error_response(f"Unknown endpoint: {path}", 404)

    def handle_health(self):
        """Health check endpoint."""
        self.send_json_response({
            'status': 'healthy',
            'service': 'execution-optimization-api',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
        })

    def handle_get_sessions(self):
        """Get current FX session info."""
        try:
            from core.execution.optimization import (
                get_current_session,
                get_session_config,
                FXSession,
            )

            current = get_current_session()
            config = get_session_config(current)

            sessions = []
            for session in FXSession:
                sess_config = get_session_config(session)
                sessions.append({
                    'name': session.value,
                    'start_utc': sess_config.start_hour,
                    'end_utc': sess_config.end_hour,
                    'liquidity_multiplier': sess_config.liquidity_multiplier,
                    'spread_multiplier': sess_config.spread_multiplier,
                    'volatility_multiplier': sess_config.volatility_multiplier,
                })

            self.send_json_response({
                'success': True,
                'current_session': current.value,
                'current_config': {
                    'liquidity_multiplier': config.liquidity_multiplier,
                    'spread_multiplier': config.spread_multiplier,
                    'volatility_multiplier': config.volatility_multiplier,
                },
                'all_sessions': sessions,
                'utc_hour': datetime.utcnow().hour,
            })
        except Exception as e:
            self.send_error_response(str(e), 500)

    def handle_get_strategies(self):
        """List available execution strategies."""
        from core.execution.optimization import ExecutionStrategy

        strategies = []
        for strategy in ExecutionStrategy:
            strategies.append({
                'name': strategy.value,
                'description': self._get_strategy_description(strategy),
            })

        self.send_json_response({
            'success': True,
            'strategies': strategies,
        })

    def _get_strategy_description(self, strategy) -> str:
        """Get description for a strategy."""
        from core.execution.optimization import ExecutionStrategy

        descriptions = {
            ExecutionStrategy.MARKET: "Immediate execution at current market price",
            ExecutionStrategy.LIMIT: "Passive limit order at specified price",
            ExecutionStrategy.TWAP: "Time-Weighted Average Price - spread execution over time",
            ExecutionStrategy.VWAP: "Volume-Weighted Average Price - follow volume profile",
            ExecutionStrategy.ALMGREN_CHRISS: "Optimal trajectory minimizing impact + variance",
            ExecutionStrategy.ADAPTIVE: "RL-based adaptive execution",
        }
        return descriptions.get(strategy, "Unknown strategy")

    def handle_optimize(self, body: Dict[str, Any]):
        """Get optimal execution decision."""
        # Validate required fields
        required = ['symbol', 'direction', 'quantity']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        engine = self.get_engine()

        decision = engine.optimize(
            symbol=body['symbol'],
            direction=body['direction'],
            quantity=body['quantity'],
            mid_price=body.get('mid_price', 1.0),
            spread_bps=body.get('spread_bps', 1.0),
            volatility=body.get('volatility'),
            urgency=body.get('urgency', 0.5),
            signal_confidence=body.get('signal_confidence', 0.5),
        )

        self.send_json_response({
            'success': True,
            'decision': {
                'strategy': decision.strategy.value,
                'expected_cost_bps': decision.expected_cost_bps,
                'expected_slippage_bps': decision.expected_slippage_bps,
                'horizon_seconds': decision.horizon_seconds,
                'num_slices': len(decision.slices) if decision.slices else 0,
                'limit_price': decision.limit_price,
                'reason': decision.reason,
            },
            'slices': [
                {
                    'time_offset_seconds': s.time_offset_seconds,
                    'quantity': s.quantity,
                    'order_type': s.order_type,
                    'limit_price': s.limit_price,
                    'urgency': s.urgency,
                }
                for s in (decision.slices or [])
            ],
        })

    def handle_impact(self, body: Dict[str, Any]):
        """Estimate market impact."""
        required = ['symbol', 'quantity']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        model = self.get_impact_model()

        estimate = model.estimate_impact(
            symbol=body['symbol'],
            quantity=body['quantity'],
            mid_price=body.get('mid_price', 1.0),
            spread_bps=body.get('spread_bps', 1.0),
            volatility=body.get('volatility'),
            horizon_seconds=body.get('horizon_seconds', 300),
        )

        self.send_json_response({
            'success': True,
            'impact': {
                'temporary_bps': estimate.temporary_bps,
                'permanent_bps': estimate.permanent_bps,
                'total_bps': estimate.total_bps,
                'spread_cost_bps': estimate.spread_cost_bps,
                'timing_risk_bps': estimate.timing_risk_bps,
                'session': estimate.session.value,
                'session_multiplier': estimate.session_multiplier,
            },
        })

    def handle_twap_schedule(self, body: Dict[str, Any]):
        """Create TWAP execution schedule."""
        required = ['symbol', 'quantity', 'direction']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        scheduler = self.get_twap_scheduler()

        schedule = scheduler.create_schedule(
            symbol=body['symbol'],
            total_quantity=body['quantity'],
            direction=body['direction'],
            horizon_seconds=body.get('horizon_seconds', 300),
            mid_price=body.get('mid_price', 1.0),
        )

        self.send_json_response({
            'success': True,
            'schedule': {
                'symbol': schedule.symbol,
                'total_quantity': schedule.total_quantity,
                'direction': schedule.direction,
                'horizon_seconds': schedule.horizon_seconds,
                'num_slices': len(schedule.slices),
                'expected_cost_bps': schedule.expected_cost_bps,
            },
            'slices': [
                {
                    'time_offset_seconds': s.time_offset_seconds,
                    'quantity': s.quantity,
                    'weight': s.weight,
                    'session': s.session.value if hasattr(s, 'session') else None,
                }
                for s in schedule.slices
            ],
        })

    def handle_vwap_schedule(self, body: Dict[str, Any]):
        """Create VWAP execution schedule."""
        required = ['symbol', 'quantity', 'direction']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        scheduler = self.get_vwap_scheduler()

        schedule = scheduler.create_schedule(
            symbol=body['symbol'],
            total_quantity=body['quantity'],
            direction=body['direction'],
            horizon_seconds=body.get('horizon_seconds', 300),
            mid_price=body.get('mid_price', 1.0),
            max_participation=body.get('max_participation', 0.1),
        )

        self.send_json_response({
            'success': True,
            'schedule': {
                'symbol': schedule.symbol,
                'total_quantity': schedule.total_quantity,
                'direction': schedule.direction,
                'horizon_seconds': schedule.horizon_seconds,
                'num_slices': len(schedule.slices),
                'expected_cost_bps': schedule.expected_cost_bps,
            },
            'slices': [
                {
                    'time_offset_seconds': s.time_offset_seconds,
                    'quantity': s.quantity,
                    'weight': s.weight,
                    'volume_participation': getattr(s, 'volume_participation', None),
                }
                for s in schedule.slices
            ],
        })

    def handle_ac_trajectory(self, body: Dict[str, Any]):
        """Create Almgren-Chriss optimal trajectory."""
        required = ['symbol', 'quantity', 'direction']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        optimizer = self.get_ac_optimizer()

        trajectory = optimizer.compute_trajectory(
            symbol=body['symbol'],
            quantity=body['quantity'],
            direction=body['direction'],
            mid_price=body.get('mid_price', 1.0),
            spread_bps=body.get('spread_bps', 1.0),
            volatility=body.get('volatility'),
            horizon_seconds=body.get('horizon_seconds', 300),
            risk_aversion=body.get('risk_aversion'),
        )

        self.send_json_response({
            'success': True,
            'trajectory': {
                'symbol': trajectory.symbol,
                'total_quantity': trajectory.total_quantity,
                'horizon_seconds': trajectory.horizon_seconds,
                'num_periods': trajectory.num_periods,
                'expected_cost_bps': trajectory.expected_cost_bps,
                'cost_variance_bps': trajectory.cost_variance_bps,
                'risk_aversion': trajectory.risk_aversion,
            },
            'trade_schedule': trajectory.trade_schedule.tolist() if hasattr(trajectory.trade_schedule, 'tolist') else list(trajectory.trade_schedule),
            'inventory_path': trajectory.inventory_path.tolist() if hasattr(trajectory.inventory_path, 'tolist') else list(trajectory.inventory_path),
        })

    def handle_compare(self, body: Dict[str, Any]):
        """Compare multiple strategies for a trade."""
        required = ['symbol', 'direction', 'quantity']
        missing = [f for f in required if f not in body]
        if missing:
            self.send_error_response(f"Missing required fields: {missing}")
            return

        from core.execution.optimization import ExecutionStrategy

        engine = self.get_engine()

        # Get costs for each strategy at different urgency levels
        urgency_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}

        for urgency in urgency_levels:
            decision = engine.optimize(
                symbol=body['symbol'],
                direction=body['direction'],
                quantity=body['quantity'],
                mid_price=body.get('mid_price', 1.0),
                spread_bps=body.get('spread_bps', 1.0),
                urgency=urgency,
                signal_confidence=body.get('signal_confidence', 0.5),
            )

            results[f"urgency_{urgency}"] = {
                'recommended_strategy': decision.strategy.value,
                'expected_cost_bps': decision.expected_cost_bps,
                'expected_slippage_bps': decision.expected_slippage_bps,
                'reason': decision.reason,
            }

        self.send_json_response({
            'success': True,
            'symbol': body['symbol'],
            'quantity': body['quantity'],
            'direction': body['direction'],
            'comparison': results,
            'recommendation': "Use low urgency for passive fills, high urgency for immediate execution",
        })

    def log_message(self, format: str, *args):
        """Custom logging."""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(host: str = '0.0.0.0', port: int = 8080):
    """Run the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, ExecutionAPIHandler)

    logger.info(f"Starting Execution Optimization API server on {host}:{port}")
    logger.info(f"Health check: http://{host}:{port}/health")
    logger.info(f"API docs: http://{host}:{port}/api/execution/strategies")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execution Optimization HTTP API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    run_server(args.host, args.port)
