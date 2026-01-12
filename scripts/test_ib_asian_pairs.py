"""
Test Asian Currency Pairs on Interactive Brokers
Check availability of Chinese Yuan (CNH) and other Asian currencies
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')  # UTF-8

try:
    from ib_insync import IB, Forex
except ImportError:
    print("ERROR: ib_insync not installed")
    print("Install with: pip install ib_insync")
    sys.exit(1)


def test_asian_pairs():
    """Test all Asian currency pairs on IB."""
    print("="*70)
    print("INTERACTIVE BROKERS - ASIAN CURRENCY PAIRS TEST")
    print("="*70)

    ib = IB()

    try:
        # Connect to IB (live trading port since paper is not open)
        print("\nConnecting to IB (live trading port 7497)...")
        ib.connect('127.0.0.1', 7497, clientId=999, timeout=10)

        if not ib.isConnected():
            print("[ERROR] Failed to connect to IB")
            print("\nMake sure:")
            print("1. TWS or IB Gateway is running")
            print("2. API is enabled (File -> Global Configuration -> API)")
            print("3. Socket port 7496 is configured")
            return

        print("[OK] Connected to IB\n")

        # Define pairs to test
        pairs = {
            'Major Asian': [
                ('USDJPY', 'USD/JPY - Japan'),
                ('AUDUSD', 'AUD/USD - Australia'),
                ('NZDUSD', 'NZD/USD - New Zealand'),
                ('AUDJPY', 'AUD/JPY - Australia/Japan'),
                ('EURJPY', 'EUR/JPY - Europe/Japan'),
                ('GBPJPY', 'GBP/JPY - UK/Japan'),
            ],
            'Chinese Yuan (CNH)': [
                ('USDCNH', 'USD/CNH - Chinese Yuan Offshore'),
                ('AUDCNH', 'AUD/CNH - Australia/China'),
                ('EURCNH', 'EUR/CNH - Europe/China'),
                ('GBPCNH', 'GBP/CNH - UK/China'),
                ('JPYCNH', 'JPY/CNH - Japan/China'),
            ],
            'Other Asian': [
                ('USDSGD', 'USD/SGD - Singapore'),
                ('USDHKD', 'USD/HKD - Hong Kong'),
                ('USDKRW', 'USD/KRW - South Korea'),
                ('USDTHB', 'USD/THB - Thailand'),
                ('USDINR', 'USD/INR - India'),
            ],
            'Russian Ruble': [
                ('USDRUB', 'USD/RUB - Russian Ruble'),
                ('EURRUB', 'EUR/RUB - Euro/Ruble'),
            ]
        }

        results = {
            'available': [],
            'no_data': [],
            'not_found': []
        }

        # Test each category
        for category, pair_list in pairs.items():
            print(f"\n{'='*70}")
            print(f"{category}")
            print('='*70)

            for symbol, name in pair_list:
                print(f"\nTesting {symbol} ({name})...")

                try:
                    # Create contract
                    contract = Forex(symbol)
                    qualified = ib.qualifyContracts(contract)

                    if not qualified:
                        print(f"  [X] Contract not found")
                        results['not_found'].append((symbol, name))
                        continue

                    print(f"  [OK] Contract qualified: {qualified[0]}")

                    # Request market data
                    ticker = ib.reqMktData(contract)
                    ib.sleep(2)  # Wait for data

                    if ticker.bid and ticker.ask:
                        spread = ticker.ask - ticker.bid

                        # Calculate pip value (depends on pair)
                        if 'JPY' in symbol:
                            # JPY pairs: 0.01 = 1 pip
                            spread_pips = spread * 100
                        else:
                            # Other pairs: 0.0001 = 1 pip
                            spread_pips = spread * 10000

                        mid = (ticker.bid + ticker.ask) / 2

                        print(f"  [OK] AVAILABLE - Live data received")
                        print(f"    Bid: {ticker.bid:.5f}")
                        print(f"    Ask: {ticker.ask:.5f}")
                        print(f"    Mid: {mid:.5f}")
                        print(f"    Spread: {spread_pips:.1f} pips")

                        results['available'].append((symbol, name, spread_pips))

                        # Spread warning
                        if spread_pips > 10:
                            print(f"    [!]  WIDE SPREAD (>{spread_pips:.1f} pips)")
                    else:
                        print(f"  [!]  No market data (may need subscription)")
                        results['no_data'].append((symbol, name))

                    # Cancel market data
                    ib.cancelMktData(contract)

                except Exception as e:
                    print(f"  [X] Error: {e}")
                    results['not_found'].append((symbol, name))

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print(f"\n[OK] AVAILABLE WITH DATA ({len(results['available'])}):")
        print("-" * 70)
        for symbol, name, spread in results['available']:
            status = "[!] WIDE SPREAD" if spread > 10 else "[OK] Good"
            print(f"  {symbol:10s} {name:40s} {spread:6.1f} pips  {status}")

        if results['no_data']:
            print(f"\n[!]  AVAILABLE BUT NO DATA ({len(results['no_data'])}):")
            print("-" * 70)
            print("These pairs exist but may need market data subscription:")
            for symbol, name in results['no_data']:
                print(f"  {symbol:10s} {name}")

        if results['not_found']:
            print(f"\n[X] NOT FOUND ({len(results['not_found'])}):")
            print("-" * 70)
            for symbol, name in results['not_found']:
                print(f"  {symbol:10s} {name}")

        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR YOUR 9PM PST SESSION")
        print("="*70)

        # Filter good pairs
        good_pairs = [p for p in results['available'] if p[2] <= 10]  # spread <= 10 pips

        print("\nPrimary Pairs (Tight Spreads < 10 pips):")
        for symbol, name, spread in good_pairs:
            if any(x in symbol for x in ['AUD', 'JPY', 'CNH']):
                print(f"  [*] {symbol:10s} {name:40s} {spread:.1f} pips")

        print("\nSecondary Pairs (Wider Spreads but Tradeable):")
        wide_pairs = [p for p in results['available'] if 10 < p[2] <= 30]
        for symbol, name, spread in wide_pairs:
            if any(x in symbol for x in ['CNH', 'SGD']):
                print(f"  [+] {symbol:10s} {name:40s} {spread:.1f} pips")

        print("\nAvoid (Spreads > 30 pips):")
        very_wide = [p for p in results['available'] if p[2] > 30]
        for symbol, name, spread in very_wide:
            print(f"  [X] {symbol:10s} {name:40s} {spread:.1f} pips (TOO WIDE)")

        # Market data subscriptions
        if results['no_data']:
            print("\n" + "="*70)
            print("MARKET DATA SUBSCRIPTIONS NEEDED")
            print("="*70)
            print("\nSome pairs are available but need market data subscriptions:")
            print("\nTo add subscriptions:")
            print("1. Log in to IB Account Management")
            print("2. Go to: Settings -> Market Data Subscriptions")
            print("3. Consider adding:")
            print("   - Asia-Pacific Quotes ($1.50/month) - for CNH pairs")
            print("   - European Quotes - for RUB pairs (if needed)")
            print("\nOr test in paper trading first - may have data there")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n[OK] Disconnected from IB")


if __name__ == "__main__":
    test_asian_pairs()
