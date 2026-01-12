"""
Interactive Brokers Manual Trading - Quick trades for testing
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ib_insync import IB, Forex, MarketOrder


def quick_trade(pair: str, action: str, quantity: int, port=7497):
    """Execute a quick trade on IB"""
    ib = IB()

    try:
        print(f"\nConnecting to IB...")
        ib.connect('127.0.0.1', port, clientId=999)

        if not ib.isConnected():
            print("[ERROR] Cannot connect to IB")
            return

        print(f"[OK] Connected")

        # Convert EUR_USD to EURUSD
        ib_symbol = pair.replace('_', '')
        contract = Forex(ib_symbol)

        # Qualify contract
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"[ERROR] Contract {ib_symbol} not found")
            return

        # Get current price
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)

        if ticker.bid and ticker.ask:
            print(f"\nCurrent {ib_symbol}:")
            print(f"  Bid: {ticker.bid:.5f}")
            print(f"  Ask: {ticker.ask:.5f}")
            print(f"  Mid: {(ticker.bid + ticker.ask) / 2:.5f}")

            # Confirm
            print(f"\n[TRADE] {action} {quantity:,} units of {ib_symbol}")
            confirm = input("Confirm? (yes/no): ").strip().lower()

            if confirm != 'yes':
                print("[CANCELLED]")
                return

            # Place order
            order = MarketOrder(action, quantity)
            trade = ib.placeOrder(qualified[0], order)

            print(f"\n[ORDER PLACED] ID: {trade.order.orderId}")
            print("Waiting for fill...")

            # Wait for fill
            timeout = 30
            while not trade.isDone() and timeout > 0:
                ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == 'Filled':
                print(f"\n[FILLED] {action} {quantity:,} {ib_symbol}")
                print(f"  Fill Price: {trade.orderStatus.avgFillPrice:.5f}")
                print(f"  Commission: ${abs(trade.orderStatus.commission):.2f}")
                print(f"\n[SUCCESS] Trade complete")
            else:
                print(f"\n[WARNING] Order status: {trade.orderStatus.status}")

        else:
            print("[ERROR] No market data available")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n[OK] Disconnected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quick IB Trade')
    parser.add_argument('pair', help='Pair (e.g., EUR_USD)')
    parser.add_argument('action', choices=['BUY', 'SELL'], help='BUY or SELL')
    parser.add_argument('quantity', type=int, help='Quantity (e.g., 20000 = $20k notional)')
    parser.add_argument('--port', type=int, default=7497, help='IB port')

    args = parser.parse_args()

    print("="*60)
    print("INTERACTIVE BROKERS - MANUAL TRADE")
    print("="*60)

    quick_trade(args.pair, args.action, args.quantity, args.port)
