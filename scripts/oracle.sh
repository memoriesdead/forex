#!/bin/bash
# Quick wrapper for Oracle Cloud sync operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_SYNC="$SCRIPT_DIR/oracle_sync.py"

# Shortcuts for common operations
case "$1" in
    # Push operations
    up|upload|push)
        python3 "$ORACLE_SYNC" push "${2:-all}" "${@:3}"
        ;;

    # Pull operations
    down|download|pull)
        python3 "$ORACLE_SYNC" pull "${2:-all}" "${@:3}"
        ;;

    # Status check
    status|space|check)
        python3 "$ORACLE_SYNC" status
        ;;

    # List files
    ls|list)
        python3 "$ORACLE_SYNC" list "${2:-all}"
        ;;

    # Clean local
    clean|rm)
        python3 "$ORACLE_SYNC" clean "${2:-all}"
        ;;

    # SSH into Oracle Cloud
    ssh)
        SSH_KEY="$SCRIPT_DIR/../ssh-key-2026-01-07 (1).key"
        ssh -i "$SSH_KEY" ubuntu@89.168.65.47 "${@:2}"
        ;;

    # Help
    help|--help|-h|"")
        echo "Oracle Cloud Storage Manager - Quick Commands"
        echo ""
        echo "Usage: ./oracle.sh <command> [target] [options]"
        echo ""
        echo "Commands:"
        echo "  up/upload/push [target]     - Upload data to cloud (default: all)"
        echo "  down/download/pull [target] - Download data from cloud (default: all)"
        echo "  status/space/check          - Check cloud storage usage"
        echo "  ls/list [target]            - List files on cloud (default: all)"
        echo "  clean/rm [target]           - Remove local data after upload"
        echo "  ssh [command]               - SSH into Oracle Cloud instance"
        echo ""
        echo "Targets: data, data_cleaned, models, logs, all"
        echo ""
        echo "Examples:"
        echo "  ./oracle.sh up data              # Upload data/ to cloud"
        echo "  ./oracle.sh down models          # Download models from cloud"
        echo "  ./oracle.sh status               # Check cloud storage"
        echo "  ./oracle.sh ssh 'df -h'          # Run command on cloud"
        echo "  ./oracle.sh up all --dry-run     # Preview what would upload"
        echo ""
        echo "For full options, run: python3 $ORACLE_SYNC --help"
        ;;

    *)
        echo "[ERROR] Unknown command: $1"
        echo "Run './oracle.sh help' for usage"
        exit 1
        ;;
esac
