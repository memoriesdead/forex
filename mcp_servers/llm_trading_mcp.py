#!/usr/bin/env python3
"""
LLM Trading MCP Server - Forex-Quant Multi-Agent Integration
============================================================
Exposes LLM trading tools via MCP stdio protocol.

Uses custom forex-quant model fine-tuned on 1,172+ quant formulas with
Chinese quant techniques (幻方量化, 九坤投资).

Tools available:
- llm_analyze_trade: Full multi-agent analysis (Bull/Bear debate)
- llm_validate_signal: Quick signal validation for HFT
- llm_explain_regime: Market regime explanation
- llm_fast_analyze: Fast mode analysis (<5s)
- llm_get_stats: Get LLM decision statistics
- llm_set_mode: Set integration mode (advisory/validation/autonomous)
- llm_warmup: Warmup the model

Usage:
    python mcp_servers/llm_trading_mcp.py

Integrates with:
- forex-quant:latest via Ollama (falls back to deepseek-r1:8b)
- core/ml/llm_reasoner.py (multi-agent system)
"""

import sys
import json
import asyncio
from typing import Any
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml.llm_reasoner import (
    MultiAgentTradingReasoner,
    TradingBotLLMIntegration,
    MarketContext,
    create_llm_integration
)

# Global reasoner instance (reused across calls)
_reasoner: MultiAgentTradingReasoner = None
_integration: TradingBotLLMIntegration = None


def send_response(response: dict):
    """Send JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def send_error(id: Any, code: int, message: str):
    """Send JSON-RPC error response."""
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })


def send_result(id: Any, result: Any):
    """Send JSON-RPC success response."""
    send_response({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })


def get_reasoner() -> MultiAgentTradingReasoner:
    """Get or create the global reasoner instance."""
    global _reasoner
    if _reasoner is None:
        _reasoner = MultiAgentTradingReasoner()
    return _reasoner


def get_integration() -> TradingBotLLMIntegration:
    """Get or create the global integration instance."""
    global _integration
    if _integration is None:
        _integration = create_llm_integration(mode="validation")
    return _integration


# Tool implementations
def tool_analyze_trade(params: dict) -> dict:
    """Full multi-agent trade analysis with Bull/Bear debate."""
    try:
        reasoner = get_reasoner()

        # Build market context
        context = MarketContext(
            symbol=params.get("symbol", "EURUSD"),
            current_price=float(params.get("price", 1.0)),
            spread_pips=float(params.get("spread_pips", 1.0)),
            direction_prediction=int(params.get("direction", 0)),
            confidence=float(params.get("confidence", 0.5)),
            regime=params.get("regime", "unknown"),
            features=params.get("features", {}),
            account_balance=float(params.get("balance", 10000)),
            current_position=float(params.get("position", 0)),
            daily_pnl=float(params.get("daily_pnl", 0))
        )

        # Run async analysis
        fast_mode = params.get("fast_mode", False)

        async def run_analysis():
            await reasoner.initialize()
            return await reasoner.analyze_trade(context, fast_mode=fast_mode)

        decision = asyncio.run(run_analysis())

        return {
            "success": True,
            "action": decision.action,
            "confidence": decision.confidence,
            "position_size_pct": decision.position_size_pct,
            "reasoning": decision.reasoning,
            "bull_argument": decision.bull_argument,
            "bear_argument": decision.bear_argument,
            "risk_assessment": decision.risk_assessment,
            "latency_ms": decision.latency_ms,
            "should_override_ml": decision.should_override_ml
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_validate_signal(params: dict) -> dict:
    """Quick signal validation for HFT - fast yes/no decision."""
    try:
        reasoner = get_reasoner()

        context = MarketContext(
            symbol=params.get("symbol", "EURUSD"),
            current_price=float(params.get("price", 1.0)),
            spread_pips=float(params.get("spread_pips", 1.0)),
            direction_prediction=int(params.get("direction", 0)),
            confidence=float(params.get("confidence", 0.5)),
            regime=params.get("regime", "unknown"),
            features=params.get("features", {})
        )

        async def run_validation():
            await reasoner.initialize()
            return await reasoner.validate_trade_entry(context)

        should_trade, reason, latency = asyncio.run(run_validation())

        return {
            "success": True,
            "should_trade": should_trade,
            "reason": reason,
            "latency_ms": latency
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_explain_regime(params: dict) -> dict:
    """Get LLM explanation of current market regime."""
    try:
        reasoner = get_reasoner()

        regime = params.get("regime", "unknown")
        features = params.get("features", {})

        async def run_explanation():
            await reasoner.initialize()
            return await reasoner.explain_regime(regime, features)

        explanation = asyncio.run(run_explanation())

        return {
            "success": True,
            "regime": regime,
            "explanation": explanation
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_fast_analyze(params: dict) -> dict:
    """Fast mode analysis for HFT (<5s response time)."""
    params["fast_mode"] = True
    return tool_analyze_trade(params)


def tool_get_stats(params: dict) -> dict:
    """Get LLM decision statistics."""
    try:
        reasoner = get_reasoner()
        stats = reasoner.get_statistics()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_set_mode(params: dict) -> dict:
    """Set LLM integration mode."""
    try:
        integration = get_integration()
        mode = params.get("mode", "validation")

        mode_map = {
            "advisory": TradingBotLLMIntegration.Mode.ADVISORY,
            "validation": TradingBotLLMIntegration.Mode.VALIDATION,
            "autonomous": TradingBotLLMIntegration.Mode.AUTONOMOUS
        }

        if mode not in mode_map:
            return {"success": False, "error": f"Invalid mode: {mode}. Use advisory/validation/autonomous"}

        integration.set_mode(mode_map[mode])

        return {
            "success": True,
            "mode": mode,
            "description": {
                "advisory": "LLM provides reasoning but doesn't change ML decisions",
                "validation": "LLM can veto trades that ML suggests",
                "autonomous": "LLM has full control over trading decisions"
            }.get(mode)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_warmup(params: dict) -> dict:
    """Warmup the DeepSeek-R1 model for faster subsequent calls."""
    try:
        reasoner = get_reasoner()

        async def run_warmup():
            latency = await reasoner.initialize()
            return latency

        latency = asyncio.run(run_warmup())

        return {
            "success": True,
            "message": "DeepSeek-R1 model warmed up and ready",
            "warmup_latency_ms": latency
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_process_signal(params: dict) -> dict:
    """Process ML signal through LLM integration layer."""
    try:
        integration = get_integration()

        async def run_process():
            await integration.initialize()
            return await integration.process_signal(
                symbol=params.get("symbol", "EURUSD"),
                ml_signal=int(params.get("ml_signal", 0)),
                ml_confidence=float(params.get("ml_confidence", 0.5)),
                current_price=float(params.get("price", 1.0)),
                spread_pips=float(params.get("spread_pips", 1.0)),
                regime=params.get("regime", "unknown"),
                features=params.get("features", {}),
                account_balance=float(params.get("balance", 10000)),
                current_position=float(params.get("position", 0)),
                daily_pnl=float(params.get("daily_pnl", 0)),
                fast_mode=params.get("fast_mode", True)
            )

        final_signal, position_size, reasoning = asyncio.run(run_process())

        return {
            "success": True,
            "final_signal": final_signal,
            "position_size_pct": position_size,
            "reasoning": reasoning,
            "mode": integration.mode.value
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# MCP Tool definitions
TOOLS = {
    "llm_analyze_trade": {
        "description": "Full multi-agent trade analysis with Bull/Bear debate using DeepSeek-R1. Returns action, confidence, reasoning from multiple agent perspectives.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Currency pair (e.g., EURUSD)"},
                "price": {"type": "number", "description": "Current price"},
                "spread_pips": {"type": "number", "description": "Current spread in pips"},
                "direction": {"type": "integer", "description": "ML prediction: 1=buy, -1=sell, 0=hold"},
                "confidence": {"type": "number", "description": "ML confidence (0-1)"},
                "regime": {"type": "string", "description": "Market regime (trending/mean_reverting/volatile)"},
                "features": {"type": "object", "description": "Feature dict with key indicators"},
                "balance": {"type": "number", "description": "Account balance"},
                "position": {"type": "number", "description": "Current position size"},
                "daily_pnl": {"type": "number", "description": "Daily P&L"},
                "fast_mode": {"type": "boolean", "description": "Use fast mode for quicker response"}
            },
            "required": ["symbol"]
        },
        "handler": tool_analyze_trade
    },
    "llm_validate_signal": {
        "description": "Quick signal validation for HFT - returns yes/no decision with brief reason. Optimized for low latency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Currency pair"},
                "price": {"type": "number", "description": "Current price"},
                "spread_pips": {"type": "number", "description": "Spread in pips"},
                "direction": {"type": "integer", "description": "Trade direction (1/-1/0)"},
                "confidence": {"type": "number", "description": "ML confidence"},
                "regime": {"type": "string", "description": "Market regime"},
                "features": {"type": "object", "description": "Key features"}
            },
            "required": ["symbol", "direction", "confidence"]
        },
        "handler": tool_validate_signal
    },
    "llm_explain_regime": {
        "description": "Get LLM explanation of current market regime and its trading implications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "description": "Market regime (trending/mean_reverting/volatile/unknown)"},
                "features": {"type": "object", "description": "Key market features for context"}
            },
            "required": ["regime"]
        },
        "handler": tool_explain_regime
    },
    "llm_fast_analyze": {
        "description": "Fast mode trade analysis (<5s) - simplified multi-agent decision for HFT.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Currency pair"},
                "price": {"type": "number", "description": "Current price"},
                "spread_pips": {"type": "number", "description": "Spread in pips"},
                "direction": {"type": "integer", "description": "ML direction prediction"},
                "confidence": {"type": "number", "description": "ML confidence"},
                "regime": {"type": "string", "description": "Market regime"}
            },
            "required": ["symbol"]
        },
        "handler": tool_fast_analyze
    },
    "llm_get_stats": {
        "description": "Get LLM decision statistics including action breakdown, latency metrics, and override rate.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "handler": tool_get_stats
    },
    "llm_set_mode": {
        "description": "Set LLM integration mode: advisory (reasoning only), validation (can veto), autonomous (full control).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["advisory", "validation", "autonomous"],
                    "description": "Integration mode"
                }
            },
            "required": ["mode"]
        },
        "handler": tool_set_mode
    },
    "llm_warmup": {
        "description": "Warmup the DeepSeek-R1 model for faster subsequent calls. Call this at startup.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "handler": tool_warmup
    },
    "llm_process_signal": {
        "description": "Process ML signal through the full LLM integration layer. Combines ML prediction with LLM validation based on current mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Currency pair"},
                "ml_signal": {"type": "integer", "description": "ML signal (1=buy, -1=sell, 0=hold)"},
                "ml_confidence": {"type": "number", "description": "ML confidence (0-1)"},
                "price": {"type": "number", "description": "Current price"},
                "spread_pips": {"type": "number", "description": "Spread in pips"},
                "regime": {"type": "string", "description": "Market regime"},
                "features": {"type": "object", "description": "Feature dict"},
                "balance": {"type": "number", "description": "Account balance"},
                "position": {"type": "number", "description": "Current position"},
                "daily_pnl": {"type": "number", "description": "Daily P&L"},
                "fast_mode": {"type": "boolean", "description": "Use fast mode (default: true)"}
            },
            "required": ["symbol", "ml_signal", "ml_confidence"]
        },
        "handler": tool_process_signal
    }
}


def handle_request(request: dict):
    """Handle incoming JSON-RPC request."""
    method = request.get("method", "")
    id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        send_result(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "llm-trading",
                "version": "2.0.0",
                "description": "Forex-Quant Model (Chinese quant fine-tuned on 1,172+ formulas)"
            }
        })

    elif method == "tools/list":
        tools_list = []
        for name, tool in TOOLS.items():
            tools_list.append({
                "name": name,
                "description": tool["description"],
                "inputSchema": tool["inputSchema"]
            })
        send_result(id, {"tools": tools_list})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name]["handler"](tool_args)
                send_result(id, {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                })
            except Exception as e:
                send_result(id, {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True
                })
        else:
            send_error(id, -32601, f"Unknown tool: {tool_name}")

    elif method == "notifications/initialized":
        pass  # No response needed

    else:
        send_error(id, -32601, f"Method not found: {method}")


def main():
    """Main MCP server loop."""
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            handle_request(request)
        except json.JSONDecodeError:
            send_error(None, -32700, "Parse error")
        except Exception as e:
            send_error(None, -32603, f"Internal error: {str(e)}")


if __name__ == "__main__":
    main()
