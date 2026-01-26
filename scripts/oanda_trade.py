#!/usr/bin/env python3
"""OANDA ALL-PAIRS trading - 68 forex pairs, no commission."""
import os
import sys
import time
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# OANDA config
API_KEY = os.getenv('OANDA_API_KEY')
ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
IS_PRACTICE = os.getenv('OANDA_PAPER', 'true').lower() == 'true'
BASE_URL = 'https://api-fxpractice.oanda.com/v3' if IS_PRACTICE else 'https://api-fxtrade.oanda.com/v3'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

# Trading config
CAPITAL = 200.0
MAX_POSITION_PCT = 0.05  # 5% per trade (smaller for 68 pairs)
MAX_POSITIONS = 15  # Max concurrent positions


def get_all_instruments():
    """Get ALL tradable instruments from OANDA."""
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/instruments"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.ok:
        return [i['name'] for i in resp.json().get('instruments', [])]
    return []


class SimpleFeatureGenerator:
    """Generate 14 features for ML models."""

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history: Dict[str, List[float]] = {}

    def process_tick(self, symbol: str, bid: float, ask: float, volume: float = 0.0) -> Dict[str, float]:
        mid = (bid + ask) / 2
        spread = ask - bid

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        history = self.price_history[symbol]
        history.append(mid)

        if len(history) > self.lookback:
            history.pop(0)

        def calc_return(n):
            if len(history) > n:
                return (history[-1] / history[-n-1]) - 1.0
            return 0.0

        def calc_vol(n):
            if len(history) > n:
                prices = history[-n:]
                returns = [(prices[i] / prices[i-1]) - 1.0 for i in range(1, len(prices))]
                if len(returns) > 1:
                    return float(np.std(returns))
            return 0.0

        return {
            'bid': bid, 'ask': ask, 'bid_volume': volume, 'ask_volume': volume,
            'mid': mid, 'spread': spread, 'close': mid, 'volume': volume,
            'ret_1': calc_return(1), 'ret_5': calc_return(5),
            'ret_10': calc_return(10), 'ret_20': calc_return(20),
            'vol_20': calc_vol(20), 'vol_50': calc_vol(50),
        }


print(f"OANDA Mode: {'PRACTICE' if IS_PRACTICE else 'LIVE'}")

# Get all OANDA instruments
print("Fetching OANDA instruments...")
ALL_SYMBOLS = get_all_instruments()
print(f"OANDA has {len(ALL_SYMBOLS)} forex pairs available")

# Load ML models
print("Loading ML models...")
from core.ml.adaptive_ensemble import create_adaptive_ensemble
ensemble = create_adaptive_ensemble()

# Convert OANDA symbols to our format and load available models
oanda_to_our = {s: s.replace('_', '') for s in ALL_SYMBOLS}
our_to_oanda = {v: k for k, v in oanda_to_our.items()}

# Check which models we have
from pathlib import Path
model_dir = Path("models/production")
available_models = []
for sym in oanda_to_our.values():
    if (model_dir / f"{sym}_models.pkl").exists():
        available_models.append(sym)

print(f"Models available: {len(available_models)}/{len(ALL_SYMBOLS)}")
ensemble.load_models(available_models)
print(f"Loaded: {available_models[:10]}..." if len(available_models) > 10 else f"Loaded: {available_models}")

# Get tradeable OANDA symbols (ones we have models for)
TRADEABLE_SYMBOLS = [our_to_oanda[s] for s in available_models if s in our_to_oanda]
print(f"Trading {len(TRADEABLE_SYMBOLS)} pairs")

# Feature generator
feature_gen = SimpleFeatureGenerator()

# MCP Server for live tuning
MCP_URL = "http://localhost:8082"
try:
    resp = requests.get(f"{MCP_URL}/health", timeout=2)
    HAS_MCP = resp.ok
    print(f"MCP Server: Connected")
except:
    HAS_MCP = False
    print(f"MCP Server: Not available")

print("\n" + "="*70)
print("OANDA ALL-PAIRS TRADING - NO COMMISSION")
print(f"Capital: ${CAPITAL:.2f} | Pairs: {len(TRADEABLE_SYMBOLS)} | Max Positions: {MAX_POSITIONS}")
print(f"MCP Live Tuning: {'ON' if HAS_MCP else 'OFF'}")
print("="*70 + "\n")


def get_prices(symbols):
    """Get current prices from OANDA."""
    # OANDA limits to ~40 instruments per request
    all_prices = {}
    for i in range(0, len(symbols), 40):
        batch = symbols[i:i+40]
        url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/pricing?instruments={','.join(batch)}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.ok:
            for p in resp.json().get('prices', []):
                sym = p['instrument']
                if p.get('bids') and p.get('asks'):
                    all_prices[sym] = {
                        'bid': float(p['bids'][0]['price']),
                        'ask': float(p['asks'][0]['price']),
                    }
    return all_prices


def get_account():
    """Get OANDA account info."""
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    return resp.json()['account'] if resp.ok else None


# State
positions = {}
balance = CAPITAL
trades = []
tick_count = 0
trade_count = 0

# Check account
acct = get_account()
if acct:
    print(f"OANDA Account Balance: ${float(acct['balance']):.2f}\n")

try:
    while True:
        prices = get_prices(TRADEABLE_SYMBOLS)
        if not prices:
            time.sleep(1)
            continue

        tick_count += 1

        for oanda_sym, price in prices.items():
            our_sym = oanda_to_our.get(oanda_sym)
            if not our_sym or our_sym not in available_models:
                continue

            bid, ask = price['bid'], price['ask']

            # Generate features
            features = feature_gen.process_tick(our_sym, bid, ask)

            # Need history
            if len(feature_gen.price_history.get(our_sym, [])) < 25:
                continue

            # Get prediction
            result = ensemble.predict(our_sym, features)
            if result is None:
                continue

            # Handle both Signal objects and dicts
            if hasattr(result, 'direction'):
                direction = result.direction
                confidence = result.confidence
            else:
                direction = result.get('direction', 0)
                confidence = result.get('confidence', 0)

            current_pos = positions.get(oanda_sym)

            # Close on reversal or take profit
            if current_pos:
                curr_price = bid if current_pos['direction'] == 1 else ask
                pnl = (curr_price - current_pos['entry']) * current_pos['units'] * current_pos['direction']
                if 'JPY' in oanda_sym:
                    pnl /= 100

                # Close conditions: reversal with confidence OR 15+ pips profit OR 10+ pips loss
                pips = abs(curr_price - current_pos['entry']) * (100 if 'JPY' in oanda_sym else 10000)
                should_close = (
                    (current_pos['direction'] != direction and confidence > 0.15) or
                    (pnl > 0 and pips >= 15) or  # Take profit
                    (pnl < 0 and pips >= 10)     # Stop loss
                )

                if should_close:
                    balance += pnl
                    trade_count += 1
                    trades.append({'pnl': pnl, 'symbol': our_sym})

                    print(f"[{datetime.now().strftime('%H:%M:%S')}] CLOSE {our_sym}: "
                          f"{'LONG' if current_pos['direction']==1 else 'SHORT'} "
                          f"{current_pos['units']} @ {curr_price:.5f} | P&L: ${pnl:.2f}")

                    # Record to MCP for live tuning
                    if HAS_MCP:
                        try:
                            requests.post(f"{MCP_URL}/api/tuning/record_outcome", json={
                                "symbol": our_sym,
                                "ml_direction": current_pos['direction'],
                                "ml_confidence": current_pos.get('conf', 0.5),
                                "actual_direction": 1 if pnl > 0 else -1,
                                "pnl_dollars": pnl
                            }, timeout=2)
                        except:
                            pass

                    del positions[oanda_sym]

            # Open new position (if under max and confidence high enough)
            if oanda_sym not in positions and len(positions) < MAX_POSITIONS and confidence > 0.20:
                mid = (bid + ask) / 2
                units = int(balance * MAX_POSITION_PCT * 50 / mid)
                if units > 0:
                    entry = ask if direction == 1 else bid
                    positions[oanda_sym] = {
                        'units': units, 'entry': entry, 'direction': direction, 'conf': confidence
                    }
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] OPEN {our_sym}: "
                          f"{'BUY' if direction==1 else 'SELL'} {units} @ {entry:.5f} "
                          f"(conf: {confidence:.2f})")

        # Status every 30 ticks
        if tick_count % 30 == 0:
            unrealized = 0
            for sym, pos in positions.items():
                if sym in prices:
                    curr = prices[sym]['bid'] if pos['direction'] == 1 else prices[sym]['ask']
                    pnl = (curr - pos['entry']) * pos['units'] * pos['direction']
                    if 'JPY' in sym:
                        pnl /= 100
                    unrealized += pnl

            wins = len([t for t in trades if t['pnl'] > 0])
            wr = wins / len(trades) * 100 if trades else 0
            total_pnl = balance - CAPITAL + unrealized

            print(f"\n{'='*70}")
            print(f"Tick {tick_count} | Positions: {len(positions)}/{MAX_POSITIONS} | "
                  f"Trades: {trade_count} | WR: {wr:.0f}%")
            print(f"Realized: ${balance-CAPITAL:.2f} | Unrealized: ${unrealized:.2f} | "
                  f"TOTAL: ${total_pnl:.2f} ({total_pnl/CAPITAL*100:+.1f}%)")
            print(f"{'='*70}\n")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total Trades: {trade_count}")
    wins = len([t for t in trades if t['pnl'] > 0])
    print(f"Win Rate: {wins}/{trade_count} ({wins/trade_count*100:.1f}%)" if trade_count else "No trades")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total P&L: ${balance-CAPITAL:.2f} ({(balance/CAPITAL-1)*100:+.1f}%)")
