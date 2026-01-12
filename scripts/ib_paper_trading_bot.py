"""
Interactive Brokers Paper Trading Bot
Session-aware trading for 6:30am and 9pm PST sessions
"""

import sys
import os
from datetime import datetime, time
from pathlib import Path
import json
import time as time_module
from typing import Dict, Optional
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ib_insync import IB, Forex, MarketOrder, util
except ImportError:
    print("ERROR: ib_insync not installed")
    print("Install with: pip install ib_insync")
    sys.exit(1)


class IBTradingBot:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timezone = pytz.timezone('America/Los_Angeles')

        # Load trading sessions config
        config_file = project_root / "config" / "trading_sessions.json"
        with open(config_file) as f:
            self.config = json.load(f)

        self.positions = {}
        self.daily_pnl = 0.0

    def connect(self):
        """Connect to IB Gateway/TWS"""
        print("Connecting to IB...")
        self.ib.connect(self.host, self.port, clientId=self.client_id)

        if self.ib.isConnected():
            print(f"[OK] Connected to IB on port {self.port}")
            return True
        else:
            print("[ERROR] Failed to connect to IB")
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("[OK] Disconnected from IB")

    def get_current_session(self) -> str:
        """Determine current trading session based on PST time"""
        now = datetime.now(self.timezone)
        current_time = now.time()

        for session_name, session_data in self.config['sessions'].items():
            if session_name == 'off_hours':
                continue

            start = datetime.strptime(session_data['time_start'], '%H:%M').time()
            end = datetime.strptime(session_data['time_end'], '%H:%M').time()

            if start <= current_time <= end:
                return session_name

        return 'off_hours'

    def get_session_pairs(self, session_name: str) -> list:
        """Get trading pairs for current session"""
        if session_name not in self.config['sessions']:
            return []
        return self.config['sessions'][session_name]['pairs']

    def get_session_config(self, session_name: str) -> dict:
        """Get full config for session"""
        return self.config['sessions'].get(session_name, {})

    def get_market_data(self, pair: str) -> Optional[Dict]:
        """Get current bid/ask for a pair"""
        try:
            # Convert EUR_USD to EURUSD for IB
            ib_symbol = pair.replace('_', '')
            contract = Forex(ib_symbol)

            # Qualify contract
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                print(f"[ERROR] Contract {ib_symbol} not found")
                return None

            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Wait for data

            if ticker.bid and ticker.ask:
                mid = (ticker.bid + ticker.ask) / 2
                spread = ticker.ask - ticker.bid

                # Calculate spread in pips
                if 'JPY' in ib_symbol:
                    spread_pips = spread * 100
                else:
                    spread_pips = spread * 10000

                return {
                    'pair': pair,
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'mid': mid,
                    'spread': spread,
                    'spread_pips': spread_pips
                }
            else:
                print(f"[WARNING] No market data for {ib_symbol}")
                return None

        except Exception as e:
            print(f"[ERROR] Getting market data for {pair}: {e}")
            return None

    def place_order(self, pair: str, action: str, quantity: int) -> bool:
        """Place a market order"""
        try:
            ib_symbol = pair.replace('_', '')
            contract = Forex(ib_symbol)

            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                print(f"[ERROR] Cannot place order - contract {ib_symbol} not found")
                return False

            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(qualified[0], order)

            print(f"[ORDER] {action} {quantity} {ib_symbol}")
            print(f"  Order ID: {trade.order.orderId}")

            # Wait for fill
            timeout = 10
            while not trade.isDone() and timeout > 0:
                self.ib.sleep(1)
                timeout -= 1

            if trade.orderStatus.status == 'Filled':
                print(f"[FILLED] {action} {quantity} {ib_symbol} @ {trade.orderStatus.avgFillPrice}")
                return True
            else:
                print(f"[WARNING] Order status: {trade.orderStatus.status}")
                return False

        except Exception as e:
            print(f"[ERROR] Placing order for {pair}: {e}")
            return False

    def get_account_summary(self):
        """Get account balance and positions"""
        try:
            account_values = self.ib.accountValues()

            balance = 0
            for item in account_values:
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    balance = float(item.value)
                    break

            positions = self.ib.positions()

            print(f"\n{'='*60}")
            print(f"ACCOUNT SUMMARY")
            print(f"{'='*60}")
            print(f"Balance: ${balance:,.2f}")
            print(f"Open Positions: {len(positions)}")

            if positions:
                print(f"\nPositions:")
                for pos in positions:
                    print(f"  {pos.contract.symbol}: {pos.position} @ {pos.avgCost}")

            print(f"{'='*60}\n")

            return balance, positions

        except Exception as e:
            print(f"[ERROR] Getting account summary: {e}")
            return 0, []

    def run_simple_test(self):
        """Run a simple test to verify everything works"""
        print("\n" + "="*60)
        print("IB PAPER TRADING BOT - SIMPLE TEST")
        print("="*60)

        # Get current session
        session = self.get_current_session()
        session_config = self.get_session_config(session)
        pairs = self.get_session_pairs(session)

        print(f"\nCurrent Time: {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Current Session: {session.upper()}")
        print(f"Session Name: {session_config.get('name', 'N/A')}")
        print(f"Strategy: {session_config.get('strategy', 'N/A')}")
        print(f"Trading Pairs: {', '.join(pairs) if pairs else 'None (off hours)'}")

        if not pairs:
            print(f"\n[INFO] Currently in OFF HOURS - not recommended for trading")
            print(f"[INFO] Next session: 9pm PST (Evening) or 6:30am PST (Morning)")
            print(f"\n[TEST] Checking EUR/USD market data anyway...")
            pairs = ['EUR_USD']  # Test with EUR/USD

        # Get account info
        balance, positions = self.get_account_summary()

        # Check market data for session pairs
        print(f"\nMarket Data Check:")
        print("-" * 60)

        for pair in pairs:
            data = self.get_market_data(pair)
            if data:
                print(f"{pair:10s} | Bid: {data['bid']:.5f} | Ask: {data['ask']:.5f} | Spread: {data['spread_pips']:.1f} pips")
            else:
                print(f"{pair:10s} | [NO DATA]")

        print(f"\n{'='*60}")
        print("TEST COMPLETE")
        print("="*60)
        print("\nBot is ready. Run with --live flag to start actual trading.")
        print("Recommended: Wait until 9pm PST for your evening session.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IB Paper Trading Bot')
    parser.add_argument('--port', type=int, default=7497, help='IB port (7497=live, 7496=paper)')
    parser.add_argument('--test', action='store_true', help='Run simple test')
    parser.add_argument('--live', action='store_true', help='Start live trading')

    args = parser.parse_args()

    bot = IBTradingBot(port=args.port)

    try:
        if not bot.connect():
            print("[ERROR] Cannot connect to IB. Make sure TWS/Gateway is running.")
            return

        if args.test or not args.live:
            bot.run_simple_test()
        else:
            print("[INFO] Live trading mode not yet implemented")
            print("[INFO] Use --test to run test mode")

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
