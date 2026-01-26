"""
Limit Order Book (LOB) Features for HFT
========================================
Sources:
- HftBacktest (nkaz001) - 3.5k stars
- QuantsPlaybook - 券商金工研报
- Cont et al. (2014) - Order Flow Imbalance
- Cartea et al. (2015) - Algorithmic Trading

Gold Standard LOB Features Used by:
- 幻方量化 (High-Flyer)
- 九坤投资 (Ubiquant)
- Citadel / Two Sigma / DE Shaw

Features:
1. Book Imbalance (多级不平衡)
2. Queue Imbalance (队列不平衡)
3. Trade Imbalance (Lee-Ready)
4. Depth Imbalance (深度不平衡)
5. Slope Features (订单簿斜率)
6. Spread Decomposition (价差分解)
7. Resilience (弹性)
8. Price Impact Curve (价格冲击曲线)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction classification."""
    BUY = 1
    SELL = -1
    UNKNOWN = 0


@dataclass
class LOBSnapshot:
    """Single LOB snapshot."""
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_direction: Optional[TradeDirection] = None


class LeeReadyClassifier:
    """
    Lee-Ready Trade Classification.

    Source: Lee & Ready (1991) - Classic algorithm

    Rules:
    1. Quote Rule: Compare trade price to midpoint
       - Above mid -> Buy
       - Below mid -> Sell
    2. Tick Rule: If at mid, use price change
       - Up-tick -> Buy
       - Down-tick -> Sell
       - Zero-tick -> Use previous classification

    Accuracy: ~85% on equity markets
    """

    def __init__(self, delay_quotes: int = 0):
        """
        Args:
            delay_quotes: Quote delay in ticks (5-second rule = 0 for HFT)
        """
        self.delay_quotes = delay_quotes
        self.prev_direction = TradeDirection.UNKNOWN

    def classify(self, trade_price: float, bid: float, ask: float,
                prev_trade_price: Optional[float] = None) -> TradeDirection:
        """
        Classify single trade.

        Args:
            trade_price: Trade execution price
            bid: Best bid at trade time
            ask: Best ask at trade time
            prev_trade_price: Previous trade price for tick rule

        Returns:
            TradeDirection
        """
        mid = (bid + ask) / 2

        # Quote rule
        if trade_price > mid:
            direction = TradeDirection.BUY
        elif trade_price < mid:
            direction = TradeDirection.SELL
        else:
            # Tick rule for trades at midpoint
            if prev_trade_price is not None:
                if trade_price > prev_trade_price:
                    direction = TradeDirection.BUY
                elif trade_price < prev_trade_price:
                    direction = TradeDirection.SELL
                else:
                    # Zero tick - use previous
                    direction = self.prev_direction
            else:
                direction = TradeDirection.UNKNOWN

        self.prev_direction = direction
        return direction

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify all trades in DataFrame.

        Args:
            df: DataFrame with 'price', 'bid', 'ask' columns

        Returns:
            Series of trade directions (-1, 0, 1)
        """
        directions = []
        prev_price = None

        for idx, row in df.iterrows():
            price = row.get('price', row.get('close'))
            bid = row.get('bid', price * 0.9999)
            ask = row.get('ask', price * 1.0001)

            direction = self.classify(price, bid, ask, prev_price)
            directions.append(direction.value)
            prev_price = price

        return pd.Series(directions, index=df.index, name='trade_direction')


class BookImbalance:
    """
    Multi-Level Book Imbalance.

    Source: Multiple papers, HftBacktest

    Measures:
    - Level 1 imbalance (BBO only)
    - Weighted imbalance (exponential decay)
    - Volume imbalance (total depth)

    Formula:
    I_n = sum(w_i * (bid_size_i - ask_size_i)) / sum(w_i * (bid_size_i + ask_size_i))

    For Forex HFT:
    - Primary alpha signal
    - Short-term direction prediction
    - Market making skew
    """

    def __init__(self, n_levels: int = 5, decay: float = 0.5):
        self.n_levels = n_levels
        self.decay = decay

    def calculate_level1(self, bid_size: float, ask_size: float) -> float:
        """Level 1 (BBO) imbalance."""
        total = bid_size + ask_size
        if total == 0:
            return 0.0
        return (bid_size - ask_size) / total

    def calculate_weighted(self, bids: List[Tuple[float, float]],
                          asks: List[Tuple[float, float]]) -> float:
        """
        Weighted multi-level imbalance.

        Deeper levels have exponentially decaying weight.
        """
        numerator = 0.0
        denominator = 0.0

        for i in range(min(self.n_levels, len(bids), len(asks))):
            weight = self.decay ** i
            bid_size = bids[i][1]
            ask_size = asks[i][1]

            numerator += weight * (bid_size - ask_size)
            denominator += weight * (bid_size + ask_size)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def calculate_volume_imbalance(self, bids: List[Tuple[float, float]],
                                   asks: List[Tuple[float, float]]) -> float:
        """Total volume imbalance across all levels."""
        total_bid = sum(size for _, size in bids[:self.n_levels])
        total_ask = sum(size for _, size in asks[:self.n_levels])

        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total

    def calculate_depth_ratio(self, bids: List[Tuple[float, float]],
                             asks: List[Tuple[float, float]],
                             price_range_pct: float = 0.001) -> float:
        """
        Depth ratio within price range.

        Measures liquidity asymmetry.
        """
        mid = (bids[0][0] + asks[0][0]) / 2
        range_size = mid * price_range_pct

        bid_depth = sum(size for price, size in bids if price >= mid - range_size)
        ask_depth = sum(size for price, size in asks if price <= mid + range_size)

        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total


class QueueImbalance:
    """
    Queue Position Imbalance.

    Source: HftBacktest, Cont (2014)

    Tracks changes in queue positions to predict fills.

    For Forex HFT:
    - Estimate fill probability
    - Detect order flow changes
    - Improve execution timing
    """

    def __init__(self):
        self.prev_bid_size = None
        self.prev_ask_size = None
        self.prev_bid_price = None
        self.prev_ask_price = None

    def calculate(self, bid: float, ask: float,
                 bid_size: float, ask_size: float) -> float:
        """
        Calculate queue imbalance from quote changes.

        Returns imbalance in [-1, 1] range.
        """
        if self.prev_bid_price is None:
            self.prev_bid_price = bid
            self.prev_ask_price = ask
            self.prev_bid_size = bid_size
            self.prev_ask_size = ask_size
            return 0.0

        # Bid side contribution
        if bid > self.prev_bid_price:
            bid_contrib = bid_size  # New aggressive bid
        elif bid == self.prev_bid_price:
            bid_contrib = bid_size - self.prev_bid_size  # Size change
        else:
            bid_contrib = -self.prev_bid_size  # Bid dropped

        # Ask side contribution
        if ask < self.prev_ask_price:
            ask_contrib = -ask_size  # New aggressive ask
        elif ask == self.prev_ask_price:
            ask_contrib = -(ask_size - self.prev_ask_size)  # Size change
        else:
            ask_contrib = self.prev_ask_size  # Ask lifted

        # Update state
        self.prev_bid_price = bid
        self.prev_ask_price = ask
        self.prev_bid_size = bid_size
        self.prev_ask_size = ask_size

        # Normalize
        total = abs(bid_contrib) + abs(ask_contrib)
        if total == 0:
            return 0.0
        return (bid_contrib + ask_contrib) / total


class SpreadDecomposition:
    """
    Spread Decomposition Analysis.

    Source: Academic market microstructure literature

    Components:
    1. Adverse Selection Cost (信息成本)
    2. Inventory Cost (库存成本)
    3. Order Processing Cost (处理成本)

    For Forex HFT:
    - Estimate true trading cost
    - Identify informed flow
    - Optimize quote placement
    """

    def __init__(self, window: int = 100):
        self.window = window

    def calculate_realized_spread(self, df: pd.DataFrame,
                                  horizon: int = 10) -> pd.Series:
        """
        Realized spread - actual profit to liquidity provider.

        Realized_Spread = 2 * direction * (trade_price - mid_future)
        """
        mid = (df['bid'] + df['ask']) / 2
        mid_future = mid.shift(-horizon)
        trade_price = df['close']

        # Estimate direction
        direction = np.sign(trade_price - mid)

        realized_spread = 2 * direction * (trade_price - mid_future)
        return realized_spread.fillna(0)

    def calculate_effective_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Effective spread - actual cost to liquidity taker.

        Effective_Spread = 2 * |trade_price - mid|
        """
        mid = (df['bid'] + df['ask']) / 2
        trade_price = df['close']

        effective_spread = 2 * np.abs(trade_price - mid)
        return effective_spread

    def calculate_adverse_selection(self, df: pd.DataFrame,
                                   horizon: int = 10) -> pd.Series:
        """
        Adverse selection component.

        AS = Effective_Spread - Realized_Spread
        """
        effective = self.calculate_effective_spread(df)
        realized = self.calculate_realized_spread(df, horizon)

        return effective - realized

    def decompose(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Full spread decomposition."""
        return {
            'quoted_spread': df['ask'] - df['bid'],
            'effective_spread': self.calculate_effective_spread(df),
            'realized_spread': self.calculate_realized_spread(df),
            'adverse_selection': self.calculate_adverse_selection(df)
        }


class LOBSlope:
    """
    Order Book Slope Features.

    Source: Naik & Yadav, verified in Chinese research

    Measures how quickly depth decays from BBO.
    Steep slope = thin book = high impact
    Flat slope = thick book = low impact

    For Forex HFT:
    - Estimate price impact
    - Detect thin markets
    - Size order appropriately
    """

    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels

    def calculate_slope(self, prices: List[float], sizes: List[float]) -> float:
        """
        Calculate slope using linear regression.

        Slope = d(cumulative_size) / d(price_distance)
        """
        if len(prices) < 2:
            return 0.0

        cumulative_sizes = np.cumsum(sizes)
        price_distances = np.abs(np.array(prices) - prices[0])

        if price_distances[-1] == 0:
            return float('inf')  # All at same price

        # Simple slope
        slope = cumulative_sizes[-1] / price_distances[-1]
        return slope

    def calculate_bid_slope(self, bids: List[Tuple[float, float]]) -> float:
        """Bid side slope (positive = thick book)."""
        prices = [p for p, _ in bids[:self.n_levels]]
        sizes = [s for _, s in bids[:self.n_levels]]
        return self.calculate_slope(prices, sizes)

    def calculate_ask_slope(self, asks: List[Tuple[float, float]]) -> float:
        """Ask side slope (positive = thick book)."""
        prices = [p for p, _ in asks[:self.n_levels]]
        sizes = [s for _, s in asks[:self.n_levels]]
        return self.calculate_slope(prices, sizes)

    def calculate_slope_imbalance(self, bids: List[Tuple[float, float]],
                                  asks: List[Tuple[float, float]]) -> float:
        """
        Slope imbalance.

        Positive = bid side thicker = buy pressure
        """
        bid_slope = self.calculate_bid_slope(bids)
        ask_slope = self.calculate_ask_slope(asks)

        if bid_slope + ask_slope == 0:
            return 0.0

        return (bid_slope - ask_slope) / (bid_slope + ask_slope)


class Resilience:
    """
    Order Book Resilience.

    Source: Large & Payne (2007), Chinese HFT research

    Measures how quickly the book recovers after large trades.
    High resilience = healthy market
    Low resilience = fragile market

    For Forex HFT:
    - Detect market stress
    - Avoid trading in fragile conditions
    - Time execution optimally
    """

    def __init__(self, recovery_window: int = 10):
        self.recovery_window = recovery_window

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate resilience metric.

        Resilience = Speed of spread recovery after widening
        """
        spread = df['ask'] - df['bid']
        spread_ma = spread.rolling(50).mean()

        # Detect spread widening events
        spread_deviation = (spread - spread_ma) / (spread_ma + 1e-10)
        widening = spread_deviation > 0.5  # 50% above average

        # Measure recovery time
        resilience = []
        for i in range(len(df)):
            if widening.iloc[i]:
                # Look forward for recovery
                recovery_time = self.recovery_window
                for j in range(1, min(self.recovery_window + 1, len(df) - i)):
                    if spread_deviation.iloc[i + j] < 0.2:
                        recovery_time = j
                        break
                # Faster recovery = higher resilience
                resilience.append(1.0 / recovery_time)
            else:
                resilience.append(1.0)  # Normal conditions

        return pd.Series(resilience, index=df.index).rolling(50).mean().fillna(1.0)


class PriceImpactCurve:
    """
    Price Impact Curve Estimation.

    Source: Almgren-Chriss, Gatheral

    Models: I(q) = sigma * sign(q) * |q|^delta

    Where:
    - sigma = volatility scaling
    - q = order size
    - delta = impact exponent (typically 0.5)

    For Forex HFT:
    - Estimate execution cost
    - Optimal order slicing
    - VWAP/TWAP calibration
    """

    def __init__(self, delta: float = 0.5):
        self.delta = delta
        self.sigma = None

    def calibrate(self, df: pd.DataFrame, window: int = 100) -> float:
        """
        Calibrate impact model from historical data.

        Returns estimated sigma.
        """
        returns = df['close'].pct_change().abs()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Regression: |return| = sigma * volume^delta
        log_returns = np.log(returns + 1e-10)
        log_volume = np.log(volume + 1)

        # Rolling estimation
        cov = log_returns.rolling(window).cov(log_volume)
        var = log_volume.rolling(window).var()

        estimated_delta = cov / (var + 1e-10)

        # Use median delta for sigma estimation
        self.delta = estimated_delta.median()
        self.sigma = (returns / (volume ** self.delta + 1e-10)).rolling(window).mean().iloc[-1]

        return self.sigma

    def estimate_impact(self, order_size: float, volatility: float) -> float:
        """
        Estimate price impact for given order size.

        Args:
            order_size: Order size (can be negative for sells)
            volatility: Current volatility estimate

        Returns:
            Expected price impact in price units
        """
        if self.sigma is None:
            self.sigma = 0.1  # Default

        return volatility * self.sigma * np.sign(order_size) * (abs(order_size) ** self.delta)


class LOBFeatureEngine:
    """
    Unified LOB Feature Engine.

    Combines all LOB-based features into single interface.
    """

    def __init__(self, n_levels: int = 5):
        self.lee_ready = LeeReadyClassifier()
        self.book_imbalance = BookImbalance(n_levels=n_levels)
        self.queue_imbalance = QueueImbalance()
        self.spread_decomp = SpreadDecomposition()
        self.lob_slope = LOBSlope(n_levels=n_levels)
        self.resilience = Resilience()
        self.price_impact = PriceImpactCurve()

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all LOB features.

        Args:
            df: DataFrame with bid, ask, volume columns

        Returns:
            DataFrame with LOB feature columns added
        """
        result = df.copy()

        # Trade direction (Lee-Ready)
        result['trade_direction'] = self.lee_ready.classify_series(df)

        # Trade imbalance
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        result['trade_imbalance'] = (result['trade_direction'] * volume).rolling(20).sum()

        # Book imbalance (L1)
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            result['book_imbalance_l1'] = df.apply(
                lambda row: self.book_imbalance.calculate_level1(
                    row['bid_size'], row['ask_size']
                ), axis=1
            )

        # Queue imbalance
        if 'bid' in df.columns and 'ask' in df.columns:
            bid_size = df['bid_size'] if 'bid_size' in df.columns else pd.Series(1, index=df.index)
            ask_size = df['ask_size'] if 'ask_size' in df.columns else pd.Series(1, index=df.index)

            queue_imb = []
            for bid, ask, bs, ask_s in zip(df['bid'], df['ask'], bid_size, ask_size):
                queue_imb.append(self.queue_imbalance.calculate(bid, ask, bs, ask_s))
            result['queue_imbalance'] = queue_imb

        # Spread decomposition
        if 'bid' in df.columns and 'ask' in df.columns:
            spread_features = self.spread_decomp.decompose(df)
            for name, series in spread_features.items():
                result[f'spread_{name}'] = series

        # Resilience
        if 'bid' in df.columns and 'ask' in df.columns:
            result['resilience'] = self.resilience.calculate(df)

        # Calibrate price impact
        self.price_impact.calibrate(df)
        result['impact_sigma'] = self.price_impact.sigma

        logger.info(f"Generated {len([c for c in result.columns if c not in df.columns])} LOB features")

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'trade_direction', 'trade_imbalance',
            'book_imbalance_l1', 'queue_imbalance',
            'spread_quoted_spread', 'spread_effective_spread',
            'spread_realized_spread', 'spread_adverse_selection',
            'resilience', 'impact_sigma'
        ]

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all LOB features.
        Interface compatible with HFT Feature Engine.
        """
        return self.generate_all_features(df)
