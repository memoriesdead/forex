#!/usr/bin/env python3
"""
Remote MCP Server Setup for Oracle Cloud
Migrates MCP servers from local to Oracle Cloud to reduce local resource usage
"""

import subprocess
from pathlib import Path
from oracle_helper import get_oracle_cloud


MCP_SERVERS_CONFIG = """
# MCP Servers for Forex Trading Project
# Running on Oracle Cloud to offload local compute

# Memory Keeper MCP - GitHub backed storage
[memory-keeper]
command = npx
args = ["-y", "@modelcontextprotocol/server-memory"]
env = {
    "MEMORY_STORAGE_TYPE": "github",
    "GITHUB_TOKEN": "${GITHUB_TOKEN}",
    "GITHUB_REPO": "${GITHUB_MEMORY_REPO}",
    "GITHUB_BRANCH": "main",
    "PROJECT_ID": "forex-trading-ml"
}

# IDE MCP - VS Code integration (optional if using IDE on cloud)
[ide]
command = npx
args = ["-y", "@anthropic-ai/mcp-server-ide"]

# File System MCP - Access to forex project files
[filesystem]
command = npx
args = ["-y", "@modelcontextprotocol/server-filesystem"]
env = {
    "ALLOWED_DIRECTORIES": "/home/ubuntu/projects/forex"
}
"""


SYSTEMD_SERVICE_TEMPLATE = """[Unit]
Description=MCP Server - {name}
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/forex
Environment="PATH=/usr/bin:/usr/local/bin"
{env_vars}
ExecStart={command} {args}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""


def setup_github_memory():
    """Set up GitHub-based memory storage"""
    print("[INFO] GitHub-based memory storage setup")
    print()
    print("You'll need to:")
    print("1. Create a private GitHub repo for memory storage")
    print("   Example: https://github.com/yourusername/forex-memory")
    print()
    print("2. Generate a GitHub Personal Access Token")
    print("   - Go to: https://github.com/settings/tokens")
    print("   - Click: Generate new token (classic)")
    print("   - Scopes needed: repo (all)")
    print()
    print("3. Add to your .env file:")
    print("   GITHUB_TOKEN=your_token_here")
    print("   GITHUB_MEMORY_REPO=yourusername/forex-memory")
    print()

    return input("Have you completed these steps? (yes/no): ").lower() == "yes"


def install_mcp_dependencies():
    """Install MCP server dependencies on Oracle Cloud"""
    oracle = get_oracle_cloud()

    print("[1/4] Installing Node.js dependencies on Oracle Cloud...")

    # Install npx and MCP servers
    commands = [
        "npm install -g npm@latest",
        "npx -y @modelcontextprotocol/server-memory --version || echo 'Will install on first run'",
    ]

    for cmd in commands:
        print(f"  Running: {cmd}")
        success, stdout, stderr = oracle.ssh_command(f"cd /home/ubuntu/projects/forex && {cmd}")
        if stdout:
            print(f"  {stdout}")
        if not success and stderr:
            print(f"  Warning: {stderr}")

    return True


def create_mcp_server_script():
    """Create MCP server manager script on Oracle Cloud"""
    oracle = get_oracle_cloud()

    print("[2/4] Creating MCP server manager script...")

    manager_script = """#!/bin/bash
# MCP Server Manager for Forex Trading Project
# Runs MCP servers on Oracle Cloud

PROJECT_DIR="/home/ubuntu/projects/forex"
MCP_DIR="$PROJECT_DIR/mcp_servers"
LOG_DIR="$PROJECT_DIR/logs/mcp"

mkdir -p "$LOG_DIR"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(cat "$PROJECT_DIR/.env" | grep -v '^#' | xargs)
fi

case "$1" in
    start-memory)
        echo "Starting Memory Keeper MCP server..."
        cd "$PROJECT_DIR"
        MEMORY_STORAGE_TYPE=github \\
        GITHUB_TOKEN="${GITHUB_TOKEN}" \\
        GITHUB_REPO="${GITHUB_MEMORY_REPO}" \\
        PROJECT_ID="forex-trading-ml" \\
        npx -y @modelcontextprotocol/server-memory > "$LOG_DIR/memory.log" 2>&1 &
        echo $! > "$MCP_DIR/memory.pid"
        echo "Memory Keeper started (PID: $(cat $MCP_DIR/memory.pid))"
        ;;

    stop-memory)
        if [ -f "$MCP_DIR/memory.pid" ]; then
            kill $(cat "$MCP_DIR/memory.pid") 2>/dev/null
            rm "$MCP_DIR/memory.pid"
            echo "Memory Keeper stopped"
        fi
        ;;

    status)
        echo "=== MCP Server Status ==="
        if [ -f "$MCP_DIR/memory.pid" ]; then
            if ps -p $(cat "$MCP_DIR/memory.pid") > /dev/null; then
                echo "Memory Keeper: RUNNING (PID: $(cat $MCP_DIR/memory.pid))"
            else
                echo "Memory Keeper: STOPPED (stale PID file)"
                rm "$MCP_DIR/memory.pid"
            fi
        else
            echo "Memory Keeper: STOPPED"
        fi
        ;;

    logs-memory)
        tail -f "$LOG_DIR/memory.log"
        ;;

    restart-memory)
        $0 stop-memory
        sleep 2
        $0 start-memory
        ;;

    *)
        echo "Usage: $0 {start-memory|stop-memory|restart-memory|status|logs-memory}"
        exit 1
        ;;
esac
"""

    # Upload script
    local_script = Path("/tmp/mcp_manager.sh")
    local_script.write_text(manager_script)

    oracle.upload_file(local_script, "mcp_servers/mcp_manager.sh")
    oracle.ssh_command("chmod +x /home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh")

    print("  MCP manager script created")
    return True


def create_env_file():
    """Upload .env file to Oracle Cloud"""
    oracle = get_oracle_cloud()

    print("[3/4] Uploading environment configuration...")

    # Read local .env
    local_env = Path("C:/Users/kevin/forex/.env")
    if not local_env.exists():
        print("  Error: .env file not found locally")
        return False

    # Upload to Oracle Cloud
    oracle.upload_file(local_env, ".env")

    print("  Environment file uploaded")
    return True


def setup_local_config():
    """Configure local Claude Code to use remote MCP servers"""
    print("[4/4] Configuring local Claude Code...")

    local_mcp_config = {
        "mcpServers": {
            "memory-keeper-remote": {
                "command": "ssh",
                "args": [
                    "-i", str(Path("C:/Users/kevin/forex/ssh-key-2026-01-07 (1).key")),
                    "ubuntu@89.168.65.47",
                    "/home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh",
                    "start-memory"
                ],
                "env": {},
                "transport": "stdio"
            }
        }
    }

    print()
    print("Add this to your Claude Code MCP configuration:")
    print()
    import json
    print(json.dumps(local_mcp_config, indent=2))
    print()

    return True


def main():
    print("=" * 60)
    print("Oracle Cloud MCP Server Setup")
    print("Forex Trading Project")
    print("=" * 60)
    print()

    # Step 0: GitHub setup
    if not setup_github_memory():
        print("[ERROR] GitHub setup incomplete. Exiting.")
        return

    # Step 1: Install dependencies
    if not install_mcp_dependencies():
        print("[ERROR] Failed to install MCP dependencies")
        return

    # Step 2: Create manager script
    if not create_mcp_server_script():
        print("[ERROR] Failed to create manager script")
        return

    # Step 3: Upload .env
    if not create_env_file():
        print("[ERROR] Failed to upload environment file")
        return

    # Step 4: Configure local
    if not setup_local_config():
        print("[ERROR] Failed to configure local setup")
        return

    print()
    print("=" * 60)
    print("[SUCCESS] Remote MCP server setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. SSH into Oracle Cloud and start MCP servers:")
    print("   bash scripts/oracle.sh ssh")
    print("   /home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh start-memory")
    print()
    print("2. Check status:")
    print("   /home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh status")
    print()
    print("3. View logs:")
    print("   /home/ubuntu/projects/forex/mcp_servers/mcp_manager.sh logs-memory")
    print()
    print("MCP servers now running on Oracle Cloud instead of local machine!")


if __name__ == "__main__":
    main()
