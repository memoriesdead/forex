#!/usr/bin/env python3
"""
Memory Keeper MCP Server
Wraps the REST API on Oracle Cloud as an MCP stdio server.

Usage:
    python mcp_servers/memory_keeper_mcp.py

This creates MCP tools for:
- context_session_start
- context_save
- context_search
- context_checkpoint
- context_batch_save
"""

import sys
import json
import requests
from typing import Any

# REST API base URL (via SSH tunnel)
API_BASE = "http://localhost:3847"
PROJECT = "forex"

# Current session ID
current_session_id = None


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


# Tool implementations
def tool_session_start(params: dict) -> dict:
    """Start a new memory session."""
    global current_session_id

    try:
        resp = requests.post(
            f"{API_BASE}/{PROJECT}/session/start",
            json={
                "name": params.get("name", "Claude Session"),
                "description": params.get("description", ""),
                "projectDir": params.get("projectDir", "C:/Users/kevin/forex/forex"),
                "defaultChannel": params.get("defaultChannel", "default")
            },
            timeout=10
        )
        data = resp.json()
        current_session_id = data.get("id")
        return {"success": True, "sessionId": current_session_id, "project": PROJECT}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_context_save(params: dict) -> dict:
    """Save context item to memory."""
    global current_session_id

    if not current_session_id:
        # Auto-start session
        tool_session_start({"name": "Auto Session"})

    try:
        resp = requests.post(
            f"{API_BASE}/{PROJECT}/context/save",
            json={
                "sessionId": current_session_id,
                "key": params.get("key", f"item_{int(__import__('time').time())}"),
                "value": params.get("value", ""),
                "channel": params.get("channel", "default"),
                "category": params.get("category", "note"),
                "priority": params.get("priority", "normal")
            },
            timeout=10
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_context_search(params: dict) -> dict:
    """Search context items."""
    try:
        resp = requests.get(
            f"{API_BASE}/{PROJECT}/context/search",
            params={
                "query": params.get("query", ""),
                "category": params.get("category", ""),
                "channel": params.get("channel", ""),
                "priority": params.get("priority", ""),
                "limit": params.get("limit", 20)
            },
            timeout=10
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_context_checkpoint(params: dict) -> dict:
    """Create a checkpoint."""
    global current_session_id

    if not current_session_id:
        tool_session_start({"name": "Auto Session"})

    try:
        resp = requests.post(
            f"{API_BASE}/{PROJECT}/checkpoint",
            json={
                "sessionId": current_session_id,
                "name": params.get("name", f"checkpoint_{int(__import__('time').time())}"),
                "description": params.get("description", "")
            },
            timeout=10
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_batch_save(params: dict) -> dict:
    """Save multiple context items."""
    global current_session_id

    if not current_session_id:
        tool_session_start({"name": "Auto Session"})

    try:
        items = params.get("items", [])
        for item in items:
            item["sessionId"] = current_session_id

        resp = requests.post(
            f"{API_BASE}/{PROJECT}/context/batch-save",
            json={"sessionId": current_session_id, "items": items},
            timeout=10
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_get_status(params: dict) -> dict:
    """Get memory keeper status."""
    try:
        resp = requests.get(f"{API_BASE}/{PROJECT}/status", timeout=10)
        data = resp.json()
        data["currentSessionId"] = current_session_id
        return data
    except Exception as e:
        return {"success": False, "error": str(e)}


# MCP Tool definitions
TOOLS = {
    "context_session_start": {
        "description": "Start a new memory session for context tracking",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Session name"},
                "description": {"type": "string", "description": "Session description"},
                "projectDir": {"type": "string", "description": "Project directory"},
                "defaultChannel": {"type": "string", "description": "Default channel"}
            }
        },
        "handler": tool_session_start
    },
    "context_save": {
        "description": "Save a context item to memory (decision, progress, error, note)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Unique key for this item"},
                "value": {"type": "string", "description": "Content to save"},
                "channel": {"type": "string", "description": "Channel (default: default)"},
                "category": {"type": "string", "enum": ["decision", "progress", "error", "note", "task", "warning"], "description": "Category"},
                "priority": {"type": "string", "enum": ["high", "normal", "low"], "description": "Priority"}
            },
            "required": ["key", "value"]
        },
        "handler": tool_context_save
    },
    "context_search": {
        "description": "Search for context items by query, category, or channel",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {"type": "string", "description": "Filter by category"},
                "channel": {"type": "string", "description": "Filter by channel"},
                "priority": {"type": "string", "description": "Filter by priority"},
                "limit": {"type": "integer", "description": "Max results (default 20)"}
            }
        },
        "handler": tool_context_search
    },
    "context_checkpoint": {
        "description": "Create a checkpoint to preserve current context state",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Checkpoint name"},
                "description": {"type": "string", "description": "Checkpoint description"}
            },
            "required": ["name"]
        },
        "handler": tool_context_checkpoint
    },
    "context_batch_save": {
        "description": "Save multiple context items at once",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                            "channel": {"type": "string"},
                            "category": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["key", "value"]
                    },
                    "description": "Array of items to save"
                }
            },
            "required": ["items"]
        },
        "handler": tool_batch_save
    },
    "context_status": {
        "description": "Get memory keeper status and current session info",
        "inputSchema": {
            "type": "object",
            "properties": {}
        },
        "handler": tool_get_status
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
                "name": "memory-keeper",
                "version": "1.0.0"
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
        pass  # No response needed for notifications

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
