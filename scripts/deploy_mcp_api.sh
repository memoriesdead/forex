#!/bin/bash
# Deploy MCP HTTP API to Oracle Cloud
# This allows vast.ai instances to read model configs and download models

set -e

ORACLE_HOST="89.168.65.47"
ORACLE_USER="ubuntu"
ORACLE_KEY="./ssh-key-2026-01-07 (1).key"
ORACLE_PROJECT_DIR="/home/ubuntu/projects/forex"

echo "=========================================="
echo "DEPLOYING MCP HTTP API TO ORACLE CLOUD"
echo "=========================================="

# Upload API script
echo ""
echo "[1/4] Uploading MCP API script..."
scp -i "$ORACLE_KEY" \
    scripts/mcp_http_api.py \
    "$ORACLE_USER@$ORACLE_HOST:$ORACLE_PROJECT_DIR/scripts/"

echo "[OK] Script uploaded"

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" << 'EOF'
pip3 install flask flask-cors requests --quiet
EOF

echo "[OK] Dependencies installed"

# Create systemd service
echo ""
echo "[3/4] Creating systemd service..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" << 'EOF'
cat > /tmp/forex-mcp-api.service << 'SERVICE'
[Unit]
Description=MCP HTTP API for Vast.ai Access
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/forex
ExecStart=/usr/bin/python3 /home/ubuntu/projects/forex/scripts/mcp_http_api.py --port 8080 --host 0.0.0.0
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/projects/forex/logs/mcp_api.log
StandardError=append:/home/ubuntu/projects/forex/logs/mcp_api.log

[Install]
WantedBy=multi-user.target
SERVICE

sudo mv /tmp/forex-mcp-api.service /etc/systemd/system/
sudo systemctl daemon-reload
EOF

echo "[OK] Systemd service created"

# Enable and start service
echo ""
echo "[4/4] Starting service..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "sudo systemctl enable forex-mcp-api && sudo systemctl start forex-mcp-api"

echo "[OK] Service started"

# Check status
echo ""
echo "Checking status..."
sleep 2
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "sudo systemctl status forex-mcp-api --no-pager" || true

# Test API
echo ""
echo "Testing API..."
sleep 2
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "curl -s http://localhost:8080/health" | python -m json.tool || true

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "MCP HTTP API is running on Oracle Cloud"
echo "  Internal: http://localhost:8080"
echo "  External: http://89.168.65.47:8080 (if firewall allows)"
echo ""
echo "Available endpoints:"
echo "  GET  /health"
echo "  POST /api/memory/search"
echo "  POST /api/memory/get"
echo "  GET  /api/models/list"
echo "  GET  /api/models/download/<model_name>"
echo "  GET  /api/data/latest"
echo ""
echo "Useful commands:"
echo "  Check status:  bash scripts/oracle.sh ssh 'sudo systemctl status forex-mcp-api'"
echo "  View logs:     bash scripts/oracle.sh ssh 'tail -f /home/ubuntu/projects/forex/logs/mcp_api.log'"
echo "  Restart:       bash scripts/oracle.sh ssh 'sudo systemctl restart forex-mcp-api'"
echo ""
echo "Test from local machine (if port 8080 is open):"
echo "  curl http://89.168.65.47:8080/health"
echo ""
