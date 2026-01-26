#!/usr/bin/env python3
"""
MCP Server for Execution Optimization
======================================
Exposes the FX execution optimization engine via Model Context Protocol (MCP).

Tools provided:
- execution_optimize: Get optimal execution strategy for an order
- execution_estimate_impact: Estimate market impact for a trade
- execution_create_schedule: Create a TWAP/VWAP/AC execution schedule
- execution_get_session: Get current FX session and liquidity info
- execution_list_strategies: List available execution strategies

Usage:
    # Run as MCP server (stdio transport)
    python scripts/mcp_execution_server.py

    # Or via HTTP
    python scripts/mcp_execution_server.py --http --port 8090

Configuration via environment:
    MCP_EXEC_LOG_LEVEL=INFO
    MCP_EXEC_DEFAULT_HORIZON=300
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import execution optimization modules
from core.execution.optimization.config import (
    ExecutionConfig, ExecutionStrategy, FXSession,
    get_current_session, get_session_config, get_symbol_config
)
from core.execution.optimization.engine import (
    ExecutionEngine, get_execution_engine
)
from core.execution.optimization.market_impact_fx import (
    FXMarketImpactModel, get_market_impact_model
)
from core.execution.optimization.almgren_chriss import get_ac_optimizer
from core.execution.optimization.twap import get_twap_scheduler
from core.execution.optimization.vwap import get_vwap_scheduler

# Setup logging
log_level = os.environ.get('MCP_EXEC_LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp_execution')


@dataclass
class MCPToolResult:
    """Standard MCP tool result format."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class ExecutionMCPServer:
    """
    MCP Server for Execution Optimization.

    Provides tools for:
    - Strategy selection
    - Market impact estimation
    - Schedule creation
    - Session awareness
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.engine = get_execution_engine(self.config)
        self.impact_model = get_market_impact_model(self.config)
        self.ac_optimizer = get_ac_optimizer(self.config)
        self.twap_scheduler = get_twap_scheduler(self.config)
        self.vwap_scheduler = get_vwap_scheduler(self.config)

        # Tool registry
        self.tools = {
            'execution_optimize': self.tool_optimize,
            'execution_estimate_impact': self.tool_estimate_impact,
            'execution_create_schedule': self.tool_create_schedule,
            'execution_get_session': self.tool_get_session,
            'execution_list_strategies': self.tool_list_strategies,
            'execution_compare_strategies': self.tool_compare_strategies,
        }

        logger.info("Execution MCP Server initialized")

    def get_tool_definitions(self) -> List[Dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "execution_optimize",
                "description": "Get optimal execution strategy for an FX order. Returns recommended strategy (MARKET, LIMIT, TWAP, VWAP, or Almgren-Chriss) based on order size, urgency, and market conditions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Currency pair (e.g., EURUSD, GBPUSD)"
                        },
                        "direction": {
                            "type": "integer",
                            "description": "1 for buy, -1 for sell"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Order size in base currency units"
                        },
                        "mid_price": {
                            "type": "number",
                            "description": "Current mid price"
                        },
                        "spread_bps": {
                            "type": "number",
                            "description": "Current spread in basis points"
                        },
                        "urgency": {
                            "type": "number",
                            "description": "Urgency level 0-1 (0=patient, 1=immediate)"
                        },
                        "signal_confidence": {
                            "type": "number",
                            "description": "ML signal confidence 0-1"
                        },
                        "volatility": {
                            "type": "number",
                            "description": "Current volatility (optional, default 0.0001)"
                        }
                    },
                    "required": ["symbol", "direction", "quantity", "mid_price", "spread_bps", "urgency", "signal_confidence"]
                }
            },
            {
                "name": "execution_estimate_impact",
                "description": "Estimate market impact for an FX trade. Returns breakdown of dealer spread, size impact, and information leakage costs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Currency pair"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Order size"
                        },
                        "direction": {
                            "type": "integer",
                            "description": "1 for buy, -1 for sell"
                        },
                        "mid_price": {
                            "type": "number",
                            "description": "Current mid price"
                        },
                        "spread_bps": {
                            "type": "number",
                            "description": "Current spread in bps (optional)"
                        },
                        "session": {
                            "type": "string",
                            "description": "FX session: tokyo, london, new_york, overlap_ln, off_hours (optional, auto-detected)"
                        }
                    },
                    "required": ["symbol", "quantity", "direction", "mid_price"]
                }
            },
            {
                "name": "execution_create_schedule",
                "description": "Create an execution schedule for TWAP, VWAP, or Almgren-Chriss strategy. Returns detailed slice-by-slice execution plan.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "enum": ["twap", "vwap", "almgren_chriss"],
                            "description": "Execution strategy"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Currency pair"
                        },
                        "direction": {
                            "type": "integer",
                            "description": "1 for buy, -1 for sell"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Total order size"
                        },
                        "horizon_seconds": {
                            "type": "integer",
                            "description": "Execution window in seconds (default 300)"
                        },
                        "mid_price": {
                            "type": "number",
                            "description": "Current mid price (required for AC)"
                        }
                    },
                    "required": ["strategy", "symbol", "direction", "quantity"]
                }
            },
            {
                "name": "execution_get_session",
                "description": "Get current FX trading session and liquidity characteristics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "execution_list_strategies",
                "description": "List all available execution strategies with descriptions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "execution_compare_strategies",
                "description": "Compare execution costs across all strategies for a given order.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Currency pair"
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Order size"
                        },
                        "mid_price": {
                            "type": "number",
                            "description": "Current mid price"
                        },
                        "spread_bps": {
                            "type": "number",
                            "description": "Current spread in bps"
                        }
                    },
                    "required": ["symbol", "quantity", "mid_price", "spread_bps"]
                }
            }
        ]

    async def tool_optimize(self, params: Dict) -> MCPToolResult:
        """Get optimal execution strategy."""
        try:
            symbol = params['symbol']
            direction = int(params['direction'])
            quantity = float(params['quantity'])
            mid_price = float(params['mid_price'])
            spread_bps = float(params['spread_bps'])
            urgency = float(params['urgency'])
            signal_confidence = float(params['signal_confidence'])
            volatility = float(params.get('volatility', 0.0001))

            decision = self.engine.optimize(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                mid_price=mid_price,
                spread_bps=spread_bps,
                volatility=volatility,
                urgency=urgency,
                signal_confidence=signal_confidence
            )

            # Format schedule if present
            schedule_info = None
            if decision.schedule:
                schedule_info = {
                    'order_id': decision.schedule.order_id,
                    'num_slices': decision.schedule.num_slices,
                    'horizon_seconds': decision.schedule.horizon_seconds,
                    'slices': [
                        {
                            'slice_id': s.slice_id,
                            'target_time': s.target_time.isoformat(),
                            'target_quantity': s.target_quantity,
                            'strategy': s.strategy.value
                        }
                        for s in decision.schedule.slices[:10]  # First 10 slices
                    ]
                }

            return MCPToolResult(
                success=True,
                data={
                    'strategy': decision.strategy.value,
                    'expected_cost_bps': round(decision.expected_cost_bps, 2),
                    'expected_time_seconds': decision.expected_time_seconds,
                    'use_limit': decision.use_limit,
                    'limit_offset_bps': round(decision.limit_offset_bps, 2),
                    'aggressiveness': round(decision.aggressiveness, 2),
                    'confidence': round(decision.confidence, 2),
                    'reasoning': decision.reasoning,
                    'alternatives': {k: round(v, 2) for k, v in decision.alternatives.items()},
                    'schedule': schedule_info
                }
            )

        except Exception as e:
            logger.error(f"optimize error: {e}")
            return MCPToolResult(success=False, data={}, error=str(e))

    async def tool_estimate_impact(self, params: Dict) -> MCPToolResult:
        """Estimate market impact."""
        try:
            symbol = params['symbol']
            quantity = float(params['quantity'])
            direction = int(params['direction'])
            mid_price = float(params['mid_price'])
            spread_bps = params.get('spread_bps')
            session_str = params.get('session')

            session = None
            if session_str:
                session = FXSession(session_str)

            impact = self.impact_model.estimate_impact(
                symbol=symbol,
                quantity=quantity,
                direction=direction,
                mid_price=mid_price,
                spread_bps=spread_bps,
                session=session
            )

            return MCPToolResult(
                success=True,
                data={
                    'total_impact_bps': round(impact.total_impact_bps, 2),
                    'dealer_spread_bps': round(impact.dealer_spread_bps, 2),
                    'size_impact_bps': round(impact.size_impact_bps, 2),
                    'info_leakage_bps': round(impact.info_leakage_bps, 2),
                    'session_multiplier': round(impact.session_multiplier, 2),
                    'confidence': round(impact.confidence, 2)
                }
            )

        except Exception as e:
            logger.error(f"estimate_impact error: {e}")
            return MCPToolResult(success=False, data={}, error=str(e))

    async def tool_create_schedule(self, params: Dict) -> MCPToolResult:
        """Create execution schedule."""
        try:
            strategy = params['strategy'].lower()
            symbol = params['symbol']
            direction = int(params['direction'])
            quantity = float(params['quantity'])
            horizon = int(params.get('horizon_seconds', 300))
            mid_price = float(params.get('mid_price', 1.0))

            order_id = f"MCP_{datetime.now().strftime('%H%M%S')}"

            if strategy == 'twap':
                schedule = self.twap_scheduler.create_schedule(
                    order_id=order_id,
                    symbol=symbol,
                    direction=direction,
                    total_quantity=quantity,
                    horizon_seconds=horizon
                )
            elif strategy == 'vwap':
                schedule = self.vwap_scheduler.create_schedule(
                    order_id=order_id,
                    symbol=symbol,
                    direction=direction,
                    total_quantity=quantity,
                    horizon_seconds=horizon
                )
            elif strategy == 'almgren_chriss':
                schedule = self.ac_optimizer.create_schedule(
                    order_id=order_id,
                    symbol=symbol,
                    direction=direction,
                    total_quantity=quantity,
                    mid_price=mid_price,
                    horizon_seconds=horizon
                )
            else:
                return MCPToolResult(
                    success=False,
                    data={},
                    error=f"Unknown strategy: {strategy}"
                )

            return MCPToolResult(
                success=True,
                data={
                    'order_id': schedule.order_id,
                    'symbol': schedule.symbol,
                    'direction': 'BUY' if schedule.direction > 0 else 'SELL',
                    'total_quantity': schedule.total_quantity,
                    'strategy': schedule.strategy.value,
                    'horizon_seconds': schedule.horizon_seconds,
                    'num_slices': schedule.num_slices,
                    'expected_cost_bps': round(schedule.expected_cost_bps, 2),
                    'slices': [
                        {
                            'slice_id': s.slice_id,
                            'target_time': s.target_time.isoformat(),
                            'target_quantity': round(s.target_quantity, 0),
                            'strategy': s.strategy.value
                        }
                        for s in schedule.slices
                    ]
                }
            )

        except Exception as e:
            logger.error(f"create_schedule error: {e}")
            return MCPToolResult(success=False, data={}, error=str(e))

    async def tool_get_session(self, params: Dict) -> MCPToolResult:
        """Get current FX session."""
        try:
            session = get_current_session()
            config = get_session_config(session)

            return MCPToolResult(
                success=True,
                data={
                    'session': session.value,
                    'start_hour_utc': config.start_hour,
                    'end_hour_utc': config.end_hour,
                    'liquidity_multiplier': config.liquidity_multiplier,
                    'spread_multiplier': config.spread_multiplier,
                    'volatility_multiplier': config.volatility_multiplier,
                    'volume_weight': config.volume_weight,
                    'current_time_utc': datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            logger.error(f"get_session error: {e}")
            return MCPToolResult(success=False, data={}, error=str(e))

    async def tool_list_strategies(self, params: Dict) -> MCPToolResult:
        """List available execution strategies."""
        return MCPToolResult(
            success=True,
            data={
                'strategies': [
                    {
                        'name': 'market',
                        'description': 'Immediate execution at current market price',
                        'best_for': 'Urgent trades, small orders',
                        'typical_cost_bps': '0.5-2.0'
                    },
                    {
                        'name': 'limit',
                        'description': 'Passive execution at specified price',
                        'best_for': 'Patient trades, tight spreads',
                        'typical_cost_bps': '0.1-0.5'
                    },
                    {
                        'name': 'twap',
                        'description': 'Time-weighted execution spread evenly over time',
                        'best_for': 'Medium orders, reduce timing risk',
                        'typical_cost_bps': '0.3-1.0'
                    },
                    {
                        'name': 'vwap',
                        'description': 'Volume-weighted execution aligned with market volume',
                        'best_for': 'Large orders during liquid periods',
                        'typical_cost_bps': '0.2-0.8'
                    },
                    {
                        'name': 'almgren_chriss',
                        'description': 'Optimal trajectory minimizing cost + risk',
                        'best_for': 'Institutional orders, risk-averse execution',
                        'typical_cost_bps': '0.15-0.6'
                    }
                ]
            }
        )

    async def tool_compare_strategies(self, params: Dict) -> MCPToolResult:
        """Compare execution costs across strategies."""
        try:
            symbol = params['symbol']
            quantity = float(params['quantity'])
            mid_price = float(params['mid_price'])
            spread_bps = float(params['spread_bps'])

            session = get_current_session()

            # Get costs for different strategies
            costs = self.engine._estimate_strategy_costs(
                symbol=symbol,
                quantity=quantity,
                mid_price=mid_price,
                spread_bps=spread_bps,
                volatility=0.0001,
                session=session
            )

            # Sort by cost
            sorted_costs = sorted(costs.items(), key=lambda x: x[1])

            return MCPToolResult(
                success=True,
                data={
                    'symbol': symbol,
                    'quantity': quantity,
                    'session': session.value,
                    'comparison': [
                        {
                            'strategy': strategy.value,
                            'expected_cost_bps': round(cost, 2),
                            'rank': i + 1
                        }
                        for i, (strategy, cost) in enumerate(sorted_costs)
                    ],
                    'recommendation': sorted_costs[0][0].value,
                    'savings_vs_market': round(
                        costs.get(ExecutionStrategy.MARKET, 0) - sorted_costs[0][1], 2
                    )
                }
            )

        except Exception as e:
            logger.error(f"compare_strategies error: {e}")
            return MCPToolResult(success=False, data={}, error=str(e))

    async def handle_request(self, request: Dict) -> Dict:
        """Handle an MCP request."""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')

        if method == 'initialize':
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {
                    'protocolVersion': '0.1.0',
                    'serverInfo': {
                        'name': 'forex-execution-optimizer',
                        'version': '1.0.0'
                    },
                    'capabilities': {
                        'tools': True
                    }
                }
            }

        elif method == 'tools/list':
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {
                    'tools': self.get_tool_definitions()
                }
            }

        elif method == 'tools/call':
            tool_name = params.get('name')
            tool_args = params.get('arguments', {})

            if tool_name not in self.tools:
                return {
                    'jsonrpc': '2.0',
                    'id': request_id,
                    'error': {
                        'code': -32601,
                        'message': f'Unknown tool: {tool_name}'
                    }
                }

            result = await self.tools[tool_name](tool_args)

            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {
                    'content': [
                        {
                            'type': 'text',
                            'text': json.dumps(asdict(result), indent=2)
                        }
                    ]
                }
            }

        else:
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'error': {
                    'code': -32601,
                    'message': f'Unknown method: {method}'
                }
            }

    async def run_stdio(self):
        """Run MCP server over stdio."""
        logger.info("Starting MCP server on stdio...")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().create_connection(
            lambda: protocol, sys.stdin
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode())
                response = await self.handle_request(request)

                print(json.dumps(response), flush=True)

            except Exception as e:
                logger.error(f"Error processing request: {e}")


async def run_http_server(server: ExecutionMCPServer, host: str, port: int):
    """Run as HTTP server."""
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp not installed. Install with: pip install aiohttp")
        return

    async def handle_mcp(request):
        data = await request.json()
        response = await server.handle_request(data)
        return web.json_response(response)

    async def handle_health(request):
        return web.json_response({
            'status': 'healthy',
            'service': 'forex-execution-mcp',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    app = web.Application()
    app.router.add_post('/mcp', handle_mcp)
    app.router.add_get('/health', handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"HTTP server running on http://{host}:{port}")
    logger.info(f"  MCP endpoint: POST /mcp")
    logger.info(f"  Health check: GET /health")

    # Keep running
    while True:
        await asyncio.sleep(3600)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Execution Optimization MCP Server')
    parser.add_argument('--http', action='store_true', help='Run as HTTP server')
    parser.add_argument('--host', default='127.0.0.1', help='HTTP host')
    parser.add_argument('--port', type=int, default=8090, help='HTTP port')
    args = parser.parse_args()

    server = ExecutionMCPServer()

    if args.http:
        asyncio.run(run_http_server(server, args.host, args.port))
    else:
        asyncio.run(server.run_stdio())


if __name__ == '__main__':
    main()
