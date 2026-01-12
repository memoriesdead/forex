"""
Order Flow Features for HFT
============================
Implements advanced order flow analysis signals.

Features:
- Order Flow Imbalance (OFI)
- Order Execution Imbalance (OEI)
- Depth Ratio signals
- Hawkes Process intensity
- Trade direction inference

Based on: Chinese A-Share research, Gatheral & Oomen (2010)
Source: https://arxiv.org/html/2505.22678v1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowState:
    """Current order flow state."""
    ofi: float = 0.0  # Order Flow Imbalance
    oei: float = 0.0  # Order Execution Imbalance
    depth_ratio: float = 1.0
    trade_imbalance: float = 0.0
    hawkes_intensity: float = 0.0
    vpin: float = 0.0  # Volume-synchronized PIN


class OrderFlowFeatures:
    """
    Advanced Order Flow Feature Generator.

    Generates signals from:
    - Order book updates (L2/L3)
    - Trade flow
    - Quote changes

    Usage:
        ofi = OrderFlowFeatures()
        ofi.update_book(bid, ask, bid_size, ask_size)
        ofi.update_trade(price, size, side)
        signals = ofi.get_signals()
    """

    def __init__(self, lookback: int = 100, vpin_buckets: int = 50):
        """
        Initialize order flow feature generator.

        Args:
            lookback: Number of updates for rolling calculations
            vpin_buckets: Number of volume buckets for VPIN
        """
        self.lookback = lookback
        self.vpin_buckets = vpin_buckets

        # Order book history
        self.book_history: deque = deque(maxlen=lookback)

        # Trade history
        self.trade_history: deque = deque(maxlen=lookback * 10)

        # VPIN volume buckets
        self.volume_buckets: List[Dict] = []
        self.bucket_volume: float = 1000.0  # Volume per bucket

        # Hawkes process parameters
        self.hawkes_mu = 0.1  # Base intensity
        self.hawkes_alpha = 0.5  # Self-excitation
        self.hawkes_beta = 1.0  # Decay rate

        # Previous state
        self.prev_bid = None
        self.prev_ask = None
        self.prev_bid_size = None
        self.prev_ask_size = None

    def update_book(self, bid: float, ask: float,
                    bid_size: float, ask_size: float,
                    timestamp: datetime = None) -> None:
        """
        Update with new book state.

        Args:
            bid: Best bid price
            ask: Best ask price
            bid_size: Quantity at best bid
            ask_size: Quantity at best ask
            timestamp: Update timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate OFI changes
        ofi_delta = 0.0

        if self.prev_bid is not None:
            # Bid side OFI
            if bid > self.prev_bid:
                ofi_delta += bid_size
            elif bid == self.prev_bid:
                ofi_delta += bid_size - self.prev_bid_size
            else:
                ofi_delta -= self.prev_bid_size

            # Ask side OFI
            if ask < self.prev_ask:
                ofi_delta -= ask_size
            elif ask == self.prev_ask:
                ofi_delta -= ask_size - self.prev_ask_size
            else:
                ofi_delta += self.prev_ask_size

        self.book_history.append({
            'timestamp': timestamp,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'ofi_delta': ofi_delta,
            'mid': (bid + ask) / 2,
            'spread': ask - bid
        })

        # Update previous state
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_bid_size = bid_size
        self.prev_ask_size = ask_size

    def update_trade(self, price: float, size: float, side: str,
                     timestamp: datetime = None) -> None:
        """
        Update with new trade.

        Args:
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell' (aggressor side)
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        direction = 1 if side.lower() == 'buy' else -1

        self.trade_history.append({
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'side': side,
            'direction': direction,
            'signed_volume': size * direction
        })

        # Update VPIN
        self._update_vpin(size, direction)

        # Update Hawkes
        self._update_hawkes(timestamp)

    def _update_vpin(self, volume: float, direction: int) -> None:
        """Update VPIN calculation."""
        if not self.volume_buckets:
            self.volume_buckets.append({
                'buy_volume': 0,
                'sell_volume': 0,
                'total_volume': 0
            })

        current_bucket = self.volume_buckets[-1]

        if direction > 0:
            current_bucket['buy_volume'] += volume
        else:
            current_bucket['sell_volume'] += volume

        current_bucket['total_volume'] += volume

        # Start new bucket if full
        if current_bucket['total_volume'] >= self.bucket_volume:
            self.volume_buckets.append({
                'buy_volume': 0,
                'sell_volume': 0,
                'total_volume': 0
            })

            # Keep only recent buckets
            if len(self.volume_buckets) > self.vpin_buckets:
                self.volume_buckets.pop(0)

    def _update_hawkes(self, timestamp: datetime) -> None:
        """Update Hawkes process intensity estimate."""
        # Calculate intensity from recent trades
        now = timestamp
        recent_trades = [t for t in self.trade_history
                         if (now - t['timestamp']).total_seconds() < 60]

        if not recent_trades:
            return

        # Hawkes intensity: λ(t) = μ + Σ α * exp(-β * (t - ti))
        intensity = self.hawkes_mu

        for trade in recent_trades:
            time_diff = (now - trade['timestamp']).total_seconds()
            intensity += self.hawkes_alpha * np.exp(-self.hawkes_beta * time_diff)

        # Store for retrieval
        if self.trade_history:
            self.trade_history[-1]['hawkes_intensity'] = intensity

    # ==================== SIGNAL CALCULATIONS ====================

    def get_ofi(self, window: int = None) -> float:
        """
        Calculate Order Flow Imbalance.

        OFI = Σ(bid_size_change - ask_size_change) weighted by price changes

        Returns:
            OFI value (positive = buying pressure)
        """
        if not self.book_history:
            return 0.0

        window = window or len(self.book_history)
        recent = list(self.book_history)[-window:]

        ofi = sum(update['ofi_delta'] for update in recent)

        # Normalize by total volume
        total_vol = sum(update['bid_size'] + update['ask_size'] for update in recent)
        if total_vol > 0:
            ofi /= total_vol

        return ofi

    def get_oei(self, window: int = None) -> float:
        """
        Calculate Order Execution Imbalance.

        OEI measures efficiency of bid/ask execution.

        Returns:
            OEI value (positive = more efficient buying)
        """
        if not self.trade_history:
            return 0.0

        window = window or len(self.trade_history)
        recent = list(self.trade_history)[-window:]

        buy_volume = sum(t['size'] for t in recent if t['direction'] > 0)
        sell_volume = sum(t['size'] for t in recent if t['direction'] < 0)

        total = buy_volume + sell_volume
        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def get_depth_ratio(self) -> float:
        """
        Calculate depth ratio.

        Returns:
            bid_depth / ask_depth (>1 = more buy interest)
        """
        if not self.book_history:
            return 1.0

        current = self.book_history[-1]
        bid_size = current['bid_size']
        ask_size = current['ask_size']

        if ask_size == 0:
            return float('inf') if bid_size > 0 else 1.0

        return bid_size / ask_size

    def get_trade_imbalance(self, window: int = 20) -> float:
        """
        Calculate trade imbalance.

        Returns:
            (buy_trades - sell_trades) / total_trades
        """
        if not self.trade_history:
            return 0.0

        recent = list(self.trade_history)[-window:]

        buy_trades = sum(1 for t in recent if t['direction'] > 0)
        sell_trades = sum(1 for t in recent if t['direction'] < 0)

        total = buy_trades + sell_trades
        if total == 0:
            return 0.0

        return (buy_trades - sell_trades) / total

    def get_vpin(self) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading.

        VPIN = Σ|buy_vol - sell_vol| / Σ(buy_vol + sell_vol)

        Higher = more informed trading (toxic flow)
        """
        if len(self.volume_buckets) < 10:
            return 0.5  # Neutral default

        # Use last N buckets
        buckets = self.volume_buckets[-self.vpin_buckets:]

        abs_imbalance = sum(
            abs(b['buy_volume'] - b['sell_volume'])
            for b in buckets
        )
        total_volume = sum(b['total_volume'] for b in buckets)

        if total_volume == 0:
            return 0.5

        return abs_imbalance / total_volume

    def get_hawkes_intensity(self) -> float:
        """
        Get current Hawkes process intensity.

        Higher = more self-exciting (momentum)
        """
        if not self.trade_history:
            return self.hawkes_mu

        return self.trade_history[-1].get('hawkes_intensity', self.hawkes_mu)

    def get_spread_percentile(self, lookback: int = 100) -> float:
        """
        Calculate current spread percentile vs history.

        Returns:
            Percentile (0-100)
        """
        if len(self.book_history) < 10:
            return 50.0

        spreads = [update['spread'] for update in self.book_history][-lookback:]
        current = spreads[-1]

        percentile = sum(1 for s in spreads[:-1] if s < current) / len(spreads[:-1]) * 100
        return percentile

    def get_microprice_momentum(self, window: int = 20) -> float:
        """
        Calculate microprice momentum in basis points.
        """
        if len(self.book_history) < window:
            return 0.0

        recent = list(self.book_history)[-window:]

        # Calculate microprices
        microprices = []
        for update in recent:
            bid = update['bid']
            ask = update['ask']
            bid_size = update['bid_size']
            ask_size = update['ask_size']

            total = bid_size + ask_size
            if total > 0:
                microprice = (bid * ask_size + ask * bid_size) / total
                microprices.append(microprice)

        if len(microprices) < 2:
            return 0.0

        return (microprices[-1] / microprices[0] - 1) * 10000

    # ==================== COMBINED SIGNALS ====================

    def get_signals(self) -> Dict[str, float]:
        """
        Get all order flow signals.

        Returns:
            Dict with signal names and values
        """
        return {
            'ofi': self.get_ofi(),
            'ofi_short': self.get_ofi(window=10),
            'ofi_long': self.get_ofi(window=50),
            'oei': self.get_oei(),
            'depth_ratio': self.get_depth_ratio(),
            'trade_imbalance': self.get_trade_imbalance(),
            'vpin': self.get_vpin(),
            'hawkes_intensity': self.get_hawkes_intensity(),
            'spread_percentile': self.get_spread_percentile(),
            'microprice_momentum': self.get_microprice_momentum(),
            'combined_signal': self._combined_signal()
        }

    def _combined_signal(self) -> float:
        """
        Generate combined order flow signal.

        Returns:
            Signal in range [-1, 1] (positive = bullish)
        """
        # Get component signals
        ofi = self.get_ofi()
        oei = self.get_oei()
        depth = self.get_depth_ratio()
        trade_imb = self.get_trade_imbalance()

        # Normalize depth ratio
        depth_signal = np.tanh(depth - 1)  # Center around 1

        # Weight components
        signal = (
            0.30 * ofi +
            0.25 * oei +
            0.20 * depth_signal +
            0.25 * trade_imb
        )

        # Clip to [-1, 1]
        return max(-1, min(1, signal))

    def get_state(self) -> OrderFlowState:
        """Get current order flow state."""
        return OrderFlowState(
            ofi=self.get_ofi(),
            oei=self.get_oei(),
            depth_ratio=self.get_depth_ratio(),
            trade_imbalance=self.get_trade_imbalance(),
            hawkes_intensity=self.get_hawkes_intensity(),
            vpin=self.get_vpin()
        )

    # ==================== DATAFRAME GENERATION ====================

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate order flow features from tick DataFrame.

        Expects columns:
        - timestamp
        - bid, ask
        - bid_size, ask_size (optional)
        - last_price, last_size (for trades)
        - side (for trades, 'buy' or 'sell')

        Returns:
            DataFrame with additional OFI/OEI columns
        """
        result = df.copy()

        # Initialize feature columns
        result['ofi'] = 0.0
        result['oei'] = 0.0
        result['depth_ratio'] = 1.0
        result['trade_imbalance'] = 0.0
        result['vpin'] = 0.5
        result['hawkes_intensity'] = 0.0
        result['combined_flow_signal'] = 0.0

        # Reset state
        self.book_history.clear()
        self.trade_history.clear()
        self.volume_buckets.clear()
        self.prev_bid = None
        self.prev_ask = None

        # Process each row
        for idx, row in df.iterrows():
            # Update book if quotes present
            if 'bid' in row and 'ask' in row:
                bid_size = row.get('bid_size', 1.0)
                ask_size = row.get('ask_size', 1.0)
                self.update_book(
                    row['bid'], row['ask'],
                    bid_size, ask_size,
                    row.get('timestamp', datetime.now())
                )

            # Update trade if present
            if 'last_price' in row and pd.notna(row.get('last_price')):
                side = row.get('side', 'buy')  # Default to buy if not specified
                self.update_trade(
                    row['last_price'],
                    row.get('last_size', 1.0),
                    side,
                    row.get('timestamp', datetime.now())
                )

            # Get signals
            signals = self.get_signals()
            result.loc[idx, 'ofi'] = signals['ofi']
            result.loc[idx, 'oei'] = signals['oei']
            result.loc[idx, 'depth_ratio'] = signals['depth_ratio']
            result.loc[idx, 'trade_imbalance'] = signals['trade_imbalance']
            result.loc[idx, 'vpin'] = signals['vpin']
            result.loc[idx, 'hawkes_intensity'] = signals['hawkes_intensity']
            result.loc[idx, 'combined_flow_signal'] = signals['combined_signal']

        return result


class TradeDirectionInference:
    """
    Trade Direction Inference using Lee-Ready and variants.

    Infers trade direction when side is not provided.
    """

    def __init__(self, method: str = 'lee_ready'):
        """
        Initialize inference model.

        Args:
            method: 'lee_ready', 'tick', or 'bulk'
        """
        self.method = method
        self.prev_price = None
        self.prev_mid = None

    def infer_direction(self, price: float, bid: float, ask: float,
                        prev_price: float = None) -> str:
        """
        Infer trade direction.

        Args:
            price: Trade price
            bid: Best bid
            ask: Best ask
            prev_price: Previous trade price (for tick rule)

        Returns:
            'buy' or 'sell'
        """
        mid = (bid + ask) / 2

        if self.method == 'lee_ready':
            # Quote rule: above mid = buy, below mid = sell
            if price > mid:
                return 'buy'
            elif price < mid:
                return 'sell'
            else:
                # At mid, use tick rule
                if prev_price is not None:
                    if price > prev_price:
                        return 'buy'
                    elif price < prev_price:
                        return 'sell'
                return 'buy'  # Default

        elif self.method == 'tick':
            # Pure tick rule
            if prev_price is not None:
                if price > prev_price:
                    return 'buy'
                elif price < prev_price:
                    return 'sell'
            return 'buy'

        elif self.method == 'bulk':
            # Bulk classification: closer to ask = buy
            if abs(price - ask) < abs(price - bid):
                return 'buy'
            else:
                return 'sell'

        return 'buy'  # Default


if __name__ == '__main__':
    print("Order Flow Features Test")
    print("=" * 50)

    ofi = OrderFlowFeatures()

    # Simulate book updates
    np.random.seed(42)
    base_price = 1.1000

    for i in range(100):
        price = base_price + np.cumsum([np.random.randn() * 0.0001])[0]
        spread = 0.0001

        bid = price - spread / 2
        ask = price + spread / 2
        bid_size = np.random.uniform(1, 10)
        ask_size = np.random.uniform(1, 10)

        ofi.update_book(bid, ask, bid_size, ask_size)

        # Simulate some trades
        if np.random.random() < 0.3:
            trade_price = ask if np.random.random() < 0.6 else bid
            side = 'buy' if trade_price == ask else 'sell'
            ofi.update_trade(trade_price, np.random.uniform(1, 5), side)

    # Get signals
    signals = ofi.get_signals()

    print("\nOrder Flow Signals:")
    for name, value in signals.items():
        print(f"  {name}: {value:.4f}")

    # Get state
    state = ofi.get_state()
    print(f"\nOrder Flow State:")
    print(f"  OFI: {state.ofi:.4f}")
    print(f"  OEI: {state.oei:.4f}")
    print(f"  Depth Ratio: {state.depth_ratio:.4f}")
    print(f"  VPIN: {state.vpin:.4f}")
    print(f"  Hawkes Intensity: {state.hawkes_intensity:.4f}")
