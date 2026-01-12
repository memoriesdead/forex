#!/bin/bash
# Start Paper Trading System End-to-End
#
# Usage: bash scripts/start_paper_trading.sh [--vastai-endpoint http://IP:5000]

set -e

VASTAI_ENDPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vastai-endpoint)
            VASTAI_ENDPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "STARTING PAPER TRADING SYSTEM"
echo "========================================"

# Check prerequisites
echo ""
echo "[1/5] Checking prerequisites..."

# Check OANDA API key
if ! grep -q "OANDA_PRACTICE_API_KEY=" .env || grep -q "your_practice_key_here" .env; then
    echo "[ERROR] OANDA Practice API key not configured"
    echo ""
    echo "Please add to .env file:"
    echo "  OANDA_PRACTICE_API_KEY=your_key"
    echo "  OANDA_PRACTICE_ACCOUNT_ID=your_account_id"
    echo ""
    echo "Get free practice account at: https://www.oanda.com/demo-account/"
    exit 1
fi

echo "[OK] OANDA API key configured"

# Check live data capture
echo ""
echo "[2/5] Checking live data capture on Oracle Cloud..."
python scripts/sync_live_data_v2.py --status || {
    echo "[WARNING] Live capture may not be running"
    echo "[ACTION] Deploy it with: bash scripts/deploy_live_capture.sh"
}

# Sync initial data
echo ""
echo "[3/5] Syncing latest data from Oracle Cloud..."
python scripts/sync_live_data_v2.py --latest 5m || {
    echo "[ERROR] Failed to sync data"
    exit 1
}

echo "[OK] Data synced"

# Check vast.ai endpoint
echo ""
echo "[4/5] Checking vast.ai model server..."

if [ -z "$VASTAI_ENDPOINT" ]; then
    echo "[INFO] No vast.ai endpoint provided"
    echo "[INFO] Will use simple momentum strategy"
    echo ""
    echo "To use ML models, start vast.ai server and pass:"
    echo "  --vastai-endpoint http://VASTAI_IP:5000"
else
    # Test connection
    if curl -f -s "${VASTAI_ENDPOINT}/health" > /dev/null; then
        echo "[OK] Vast.ai server reachable at $VASTAI_ENDPOINT"
    else
        echo "[ERROR] Cannot reach vast.ai server at $VASTAI_ENDPOINT"
        echo "[ACTION] Check if server is running and accessible"
        exit 1
    fi
fi

# Start paper trading bot
echo ""
echo "[5/5] Starting paper trading bot..."
echo ""
echo "========================================"
echo "PAPER TRADING BOT STARTING"
echo "========================================"
echo ""

if [ -z "$VASTAI_ENDPOINT" ]; then
    python scripts/paper_trading_bot.py --interval 5
else
    python scripts/paper_trading_bot.py --vastai-endpoint "$VASTAI_ENDPOINT" --interval 5
fi
