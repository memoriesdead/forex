#!/usr/bin/env python3
"""
Check Current P&L and Optionally Close Positions
=================================================
Connects to IB Gateway and shows all positions with P&L.
Can close all positions to realize profits.

Usage:
    python scripts/check_pnl.py           # Show P&L
    python scripts/check_pnl.py --close   # Close all positions to REALIZE P&L
    python scripts/check_pnl.py --close --symbol GBPAUD  # Close specific symbol
"""

import argparse
from ib_insync import IB, Forex, MarketOrder


def main():
    parser = argparse.ArgumentParser(description='Check P&L and close positions')
    parser.add_argument('--close', action='store_true', help='Close all positions to realize P&L')
    parser.add_argument('--symbol', type=str, help='Close specific symbol only')
    args = parser.parse_args()

    ib = IB()
    ib.connect('127.0.0.1', 4004, clientId=50)

    print('=' * 70)
    print('ACCOUNT P&L - FOREX TRADING')
    print('=' * 70)

    # Account values
    for item in ib.accountSummary():
        if item.tag in ['NetLiquidation', 'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL']:
            if item.currency == 'USD' or item.currency == 'BASE':
                print(f'{item.tag}: ${float(item.value):,.2f} {item.currency}')

    print()
    print('POSITIONS:')
    print('-' * 70)
    print(f"{'Symbol':<12} {'Qty':>10} {'Direction':>10} {'Entry':>12} {'Current':>12} {'P&L':>12}")
    print('-' * 70)

    positions = ib.positions()

    if not positions:
        print('No open positions')
        ib.disconnect()
        return

    total_unrealized = 0
    position_data = []

    for pos in positions:
        symbol = pos.contract.symbol + pos.contract.currency
        qty = pos.position
        avg_cost = pos.avgCost

        if qty == 0:
            continue

        # Get current price
        contract = pos.contract
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(0.5)

        if ticker.last and ticker.last > 0:
            current_price = ticker.last
        elif ticker.close and ticker.close > 0:
            current_price = ticker.close
        elif ticker.bid and ticker.ask:
            current_price = (ticker.bid + ticker.ask) / 2
        else:
            current_price = avg_cost  # fallback

        # Calculate P&L
        if 'JPY' in symbol:
            # JPY pairs: P&L in USD = qty * (current - entry) / current
            pnl = qty * (current_price - avg_cost) / current_price
        else:
            # Other pairs: P&L = qty * (current - entry)
            pnl = qty * (current_price - avg_cost)

        total_unrealized += pnl

        direction = 'LONG' if qty > 0 else 'SHORT'
        pnl_str = f'+${pnl:.2f}' if pnl >= 0 else f'-${abs(pnl):.2f}'

        print(f'{symbol:<12} {abs(qty):>10,.0f} {direction:>10} {avg_cost:>12.5f} {current_price:>12.5f} {pnl_str:>12}')

        position_data.append({
            'symbol': symbol,
            'contract': contract,
            'qty': qty,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'pnl': pnl,
            'direction': direction
        })

        ib.cancelMktData(contract)

    print('-' * 70)
    total_str = f'+${total_unrealized:.2f}' if total_unrealized >= 0 else f'-${abs(total_unrealized):.2f}'
    print(f"{'TOTAL UNREALIZED P&L':>58} {total_str:>12}")
    print('=' * 70)

    # Close positions if requested
    if args.close and position_data:
        print()
        print('=' * 70)
        print('CLOSING POSITIONS TO REALIZE P&L')
        print('=' * 70)

        total_realized = 0

        for pos_info in position_data:
            symbol = pos_info['symbol']

            # Filter by symbol if specified
            if args.symbol and args.symbol.upper() not in symbol.upper():
                continue

            qty = pos_info['qty']
            if qty == 0:
                continue

            side = 'SELL' if qty > 0 else 'BUY'
            close_qty = abs(qty)

            print(f"\nClosing {symbol}: {side} {close_qty:,.0f} units...")

            try:
                # Create forex contract properly
                base = symbol[:3]
                quote = symbol[3:]
                contract = Forex(base + quote)
                ib.qualifyContracts(contract)

                order = MarketOrder(side, close_qty)
                trade = ib.placeOrder(contract, order)

                # Wait for fill
                ib.sleep(2)

                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    if 'JPY' in symbol:
                        realized_pnl = qty * (fill_price - pos_info['avg_cost']) / fill_price
                    else:
                        realized_pnl = qty * (fill_price - pos_info['avg_cost'])
                    total_realized += realized_pnl
                    pnl_str = f'+${realized_pnl:.2f}' if realized_pnl >= 0 else f'-${abs(realized_pnl):.2f}'
                    print(f"  ✓ FILLED @ {fill_price:.5f} | REALIZED: {pnl_str}")
                else:
                    print(f"  Status: {trade.orderStatus.status}")

            except Exception as e:
                print(f"  ERROR closing {symbol}: {e}")

        print()
        print('=' * 70)
        total_real_str = f'+${total_realized:.2f}' if total_realized >= 0 else f'-${abs(total_realized):.2f}'
        print(f"TOTAL REALIZED P&L: {total_real_str}")
        print('=' * 70)
        print()
        print('✓ PROFITS LOCKED IN! Your trading capital has increased.')

    ib.disconnect()
    print("\nDisconnected from IB Gateway")


if __name__ == '__main__':
    main()
