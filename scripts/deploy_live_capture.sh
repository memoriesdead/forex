#!/bin/bash
# Deploy TrueFX Live Capture to Oracle Cloud
# Run this from local machine to set up 24/7 data capture

set -e

ORACLE_HOST="89.168.65.47"
ORACLE_USER="ubuntu"
ORACLE_KEY="./ssh-key-2026-01-07 (1).key"
ORACLE_PROJECT_DIR="/home/ubuntu/projects/forex"

echo "=========================================="
echo "DEPLOYING TRUEFX LIVE CAPTURE TO ORACLE CLOUD"
echo "=========================================="

# Upload capture script
echo ""
echo "[1/5] Uploading capture script..."
scp -i "$ORACLE_KEY" \
    scripts/live_capture_truefx.py \
    "$ORACLE_USER@$ORACLE_HOST:$ORACLE_PROJECT_DIR/scripts/"

echo "[OK] Script uploaded"

# Create systemd service file
echo ""
echo "[2/5] Creating systemd service..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" << 'EOF'
cat > /tmp/forex-live-capture.service << 'SERVICE'
[Unit]
Description=TrueFX Live Forex Data Capture (24/7)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/projects/forex
ExecStart=/usr/bin/python3 /home/ubuntu/projects/forex/scripts/live_capture_truefx.py
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/projects/forex/logs/live_capture_systemd.log
StandardError=append:/home/ubuntu/projects/forex/logs/live_capture_systemd.log

[Install]
WantedBy=multi-user.target
SERVICE

sudo mv /tmp/forex-live-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
EOF

echo "[OK] Systemd service created"

# Enable and start service
echo ""
echo "[3/5] Enabling service..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "sudo systemctl enable forex-live-capture"

echo "[OK] Service enabled (will auto-start on boot)"

echo ""
echo "[4/5] Starting service..."
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "sudo systemctl start forex-live-capture"

echo "[OK] Service started"

# Check status
echo ""
echo "[5/5] Checking status..."
sleep 3
ssh -i "$ORACLE_KEY" "$ORACLE_USER@$ORACLE_HOST" \
    "sudo systemctl status forex-live-capture --no-pager" || true

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Service is now running 24/7 on Oracle Cloud"
echo ""
echo "Useful commands:"
echo "  Check status:  bash scripts/oracle.sh ssh 'sudo systemctl status forex-live-capture'"
echo "  View logs:     bash scripts/oracle.sh ssh 'tail -f /home/ubuntu/projects/forex/logs/live_capture.log'"
echo "  Stop service:  bash scripts/oracle.sh ssh 'sudo systemctl stop forex-live-capture'"
echo "  Start service: bash scripts/oracle.sh ssh 'sudo systemctl start forex-live-capture'"
echo ""
echo "Data location: $ORACLE_PROJECT_DIR/data/live/"
echo ""
echo "To sync data locally:"
echo "  python scripts/sync_live_data_v2.py --latest 1m"
echo "  python scripts/sync_live_data_v2.py --stream"
echo ""
