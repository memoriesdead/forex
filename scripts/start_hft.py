"""
HFT System Master Controller
============================
One-stop control for the entire HFT system.

Commands:
    python scripts/start_hft.py prepare   # Prepare training data
    python scripts/start_hft.py train     # Launch Vast.ai training
    python scripts/start_hft.py paper     # Start paper trading
    python scripts/start_hft.py live      # Start live trading
    python scripts/start_hft.py status    # Check system status
    python scripts/start_hft.py all       # Full pipeline: prepare -> train -> paper

Requirements:
    - SSH tunnel to Oracle Cloud (for IB Gateway)
    - Vast.ai account (for GPU training)
    - TrueFX credentials (for live data)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import os

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print HFT system banner."""
    print("""
================================================================
              HFT FOREX TRADING SYSTEM
              Renaissance-Level ML (306 Features)
================================================================
  Components:
  - 306 HFT Features (Alpha101, Alpha191, Renaissance)
  - ML Ensemble (XGBoost + LightGBM + CatBoost)
  - Tick-Level Backtesting
  - Latency-Aware Execution
  - IB Gateway Integration
================================================================
""")


def check_ssh_tunnel():
    """Check if SSH tunnel to Oracle Cloud is running."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 4001))
    sock.close()

    if result == 0:
        print("[OK] SSH Tunnel: Connected (localhost:4001)")
        return True
    else:
        print("[X] SSH Tunnel: Not connected")
        print("  Run: powershell -File scripts\\start_oracle_tunnel.ps1")
        return False


def check_models():
    """Check if trained models exist."""
    model_dir = PROJECT_ROOT / "models" / "hft_ensemble"

    if model_dir.exists():
        models = list(model_dir.glob("*.pkl"))
        if models:
            print(f"[OK] Models: {len(models)} model files found")
            for m in models[:3]:
                print(f"    - {m.name}")
            return True

    print("[X] Models: No trained models found")
    print("  Run: python scripts/start_hft.py prepare && train on Vast.ai")
    return False


def check_data():
    """Check if training data exists."""
    data_dir = PROJECT_ROOT / "training_package"

    if data_dir.exists():
        parquets = list(data_dir.glob("**/*.parquet"))
        if parquets:
            print(f"[OK] Training Data: {len(parquets)} data files ready")
            return True

    print("[X] Training Data: Not prepared")
    print("  Run: python scripts/start_hft.py prepare")
    return False


def cmd_status():
    """Check system status."""
    print("\n=== System Status ===\n")

    tunnel_ok = check_ssh_tunnel()
    models_ok = check_models()
    data_ok = check_data()

    # Check IB Gateway
    print()
    if tunnel_ok:
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect('localhost', 4001, clientId=999, timeout=5)
            account = ib.managedAccounts()[0] if ib.managedAccounts() else "Unknown"
            print(f"[OK] IB Gateway: Connected (Account: {account})")
            ib.disconnect()
        except Exception as e:
            print(f"[X] IB Gateway: Not responding ({e})")

    print("\n=== Readiness ===")

    if models_ok and tunnel_ok:
        print("\n[OK] READY FOR PAPER TRADING")
        print("  Run: python scripts/start_hft.py paper")
    elif data_ok:
        print("\n-> Next Step: Train models on Vast.ai")
        print("  1. Upload training_package/ to Vast.ai")
        print("  2. Run: ./run_training.sh")
        print("  3. Download models to models/hft_ensemble/")
    else:
        print("\n-> Next Step: Prepare training data")
        print("  Run: python scripts/start_hft.py prepare")


def cmd_prepare(symbols: str = "EURUSD,GBPUSD,USDJPY", days: int = 30):
    """Prepare training data."""
    print("\n=== Preparing Training Data ===\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_hft_training.py"),
        "--symbols", symbols,
        "--days", str(days),
        "--output", "training_package"
    ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    print("\n=== Next Steps ===")
    print("1. Rent H100 on Vast.ai (~$2.50/hr)")
    print("2. Upload: scp -r training_package/* root@<VAST_IP>:/workspace/")
    print("3. SSH: ssh root@<VAST_IP>")
    print("4. Run: ./run_training.sh")
    print("5. Download: scp root@<VAST_IP>:/workspace/hft_models.tar.gz .")
    print("6. Extract to: models/hft_ensemble/")


def cmd_paper(symbols: str = "EURUSD,GBPUSD"):
    """Start paper trading."""
    print("\n=== Starting Paper Trading ===\n")

    # Check prerequisites
    if not check_ssh_tunnel():
        print("\nPlease start SSH tunnel first.")
        return

    if not check_models():
        print("\nModels not found. Running in signal-only mode.")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "hft_trading_bot.py"),
        "--mode", "paper",
        "--symbols", symbols
    ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_live(symbols: str = "EURUSD"):
    """Start live trading."""
    print("\n⚠️  LIVE TRADING - REAL MONEY ⚠️\n")

    confirm = input("Type 'LIVE' to confirm: ")
    if confirm != "LIVE":
        print("Cancelled.")
        return

    if not check_ssh_tunnel():
        print("\nPlease start SSH tunnel first.")
        return

    if not check_models():
        print("\nNo models - cannot trade live!")
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "hft_trading_bot.py"),
        "--mode", "live",
        "--symbols", symbols
    ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_backtest(symbols: str = "EURUSD", days: int = 7):
    """Run backtest."""
    print("\n=== Running Backtest ===\n")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "hft_trading_bot.py"),
        "--mode", "backtest",
        "--symbols", symbols,
        "--days", str(days)
    ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def cmd_tunnel():
    """Start SSH tunnel to Oracle Cloud."""
    print("\n=== Starting SSH Tunnel ===\n")

    # Windows
    if sys.platform == "win32":
        ps_script = PROJECT_ROOT / "scripts" / "start_oracle_tunnel.ps1"
        if ps_script.exists():
            subprocess.Popen(
                ["powershell", "-File", str(ps_script)],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print("SSH tunnel started in new window")
        else:
            print("SSH tunnel script not found")
            print("Manual command:")
            print('  ssh -i "ssh-key-2026-01-07 (1).key" -L 4001:localhost:4001 ubuntu@89.168.65.47 -N')
    else:
        print("Run manually:")
        print('  ssh -L 4001:localhost:4001 ubuntu@89.168.65.47 -N &')


def main():
    print_banner()

    parser = argparse.ArgumentParser(description='HFT System Controller')
    parser.add_argument('command', nargs='?', default='status',
                        choices=['status', 'prepare', 'paper', 'live', 'backtest', 'tunnel'],
                        help='Command to run')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD',
                        help='Trading symbols')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of data')
    args = parser.parse_args()

    commands = {
        'status': cmd_status,
        'prepare': lambda: cmd_prepare(args.symbols, args.days),
        'paper': lambda: cmd_paper(args.symbols),
        'live': lambda: cmd_live(args.symbols),
        'backtest': lambda: cmd_backtest(args.symbols, args.days),
        'tunnel': cmd_tunnel,
    }

    commands[args.command]()


if __name__ == '__main__':
    main()
