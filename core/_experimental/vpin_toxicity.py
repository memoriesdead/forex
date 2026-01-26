"""
VPIN: Volume-Synchronized Probability of Informed Trading
==========================================================
Source: Easley, Lopez de Prado, O'Hara (2012)
"Flow Toxicity and Liquidity in a High-frequency World"

VPIN measures order flow toxicity - probability that trades are from
informed traders who know something the market doesn't.

High VPIN = Informed traders are active = Volatility incoming
- Flash Crash (May 6, 2010): VPIN spiked before crash
- Useful for: Risk warning, mean reversion fade timing

Integration:
- Complements OFI/OEI in order_flow_features.py
- Use as regime indicator with HMM
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VPINResult:
    """VPIN calculation result."""
    vpin: float  # Current VPIN estimate (0-1)
    toxicity_regime: int  # 0=low, 1=normal, 2=high
    buy_volume: float  # Classified buy volume
    sell_volume: float  # Classified sell volume
    imbalance: float  # Volume imbalance ratio


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN = |V_buy - V_sell| / (V_buy + V_sell)

    Where volumes are classified using:
    1. Tick Rule: Trade at uptick = buy, downtick = sell
    2. Bulk Volume Classification (BVC): More robust

    Parameters:
    -----------
    bucket_size : int
        Number of contracts per volume bucket
    n_buckets : int
        Number of buckets for rolling VPIN
    """

    def __init__(self, bucket_size: int = 50000, n_buckets: int = 50):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

        # State
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0

        # History of completed buckets
        self.bucket_buys = []
        self.bucket_sells = []

        # Thresholds for regime classification
        self.low_threshold = 0.3
        self.high_threshold = 0.7

    def classify_trade(self, price: float, prev_price: float,
                      volume: float) -> Tuple[float, float]:
        """
        Classify trade as buy or sell using tick rule.

        Returns:
            (buy_volume, sell_volume)
        """
        if price > prev_price:
            return volume, 0.0  # Buy
        elif price < prev_price:
            return 0.0, volume  # Sell
        else:
            # Unchanged - split 50/50
            return volume / 2, volume / 2

    def bulk_classify(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[float, float]:
        """
        Bulk Volume Classification (BVC) - more robust than tick rule.

        Uses: V_buy = V * CDF(Z) where Z = (P - VWAP) / sigma
        """
        if len(prices) < 2:
            return 0.0, 0.0

        # VWAP
        vwap = np.average(prices, weights=volumes)

        # Standard deviation
        sigma = np.std(prices) + 1e-10

        # Z-scores
        z = (prices - vwap) / sigma

        # CDF approximation (normal)
        from scipy.stats import norm
        buy_fractions = norm.cdf(z)

        buy_volume = np.sum(volumes * buy_fractions)
        sell_volume = np.sum(volumes * (1 - buy_fractions))

        return buy_volume, sell_volume

    def update(self, price: float, prev_price: float, volume: float) -> Optional[VPINResult]:
        """
        Update VPIN with new trade.

        Returns VPINResult when a new bucket is completed.
        """
        # Classify trade
        buy_vol, sell_vol = self.classify_trade(price, prev_price, volume)

        # Add to current bucket
        self.current_bucket_volume += volume
        self.current_bucket_buy += buy_vol
        self.current_bucket_sell += sell_vol

        # Check if bucket is complete
        if self.current_bucket_volume >= self.bucket_size:
            # Store bucket
            self.bucket_buys.append(self.current_bucket_buy)
            self.bucket_sells.append(self.current_bucket_sell)

            # Keep only last n_buckets
            if len(self.bucket_buys) > self.n_buckets:
                self.bucket_buys.pop(0)
                self.bucket_sells.pop(0)

            # Reset bucket
            self.current_bucket_volume = 0.0
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0

            # Calculate VPIN
            return self.calculate_vpin()

        return None

    def calculate_vpin(self) -> VPINResult:
        """Calculate current VPIN from bucket history."""
        if not self.bucket_buys:
            return VPINResult(0.5, 1, 0, 0, 0)

        total_buy = sum(self.bucket_buys)
        total_sell = sum(self.bucket_sells)
        total_volume = total_buy + total_sell

        if total_volume < 1e-10:
            return VPINResult(0.5, 1, 0, 0, 0)

        # VPIN = average absolute imbalance
        vpin = 0.0
        for buy, sell in zip(self.bucket_buys, self.bucket_sells):
            bucket_total = buy + sell
            if bucket_total > 0:
                vpin += abs(buy - sell) / bucket_total
        vpin /= len(self.bucket_buys)

        # Imbalance ratio
        imbalance = (total_buy - total_sell) / total_volume

        # Regime
        if vpin < self.low_threshold:
            regime = 0  # Low toxicity
        elif vpin > self.high_threshold:
            regime = 2  # High toxicity
        else:
            regime = 1  # Normal

        return VPINResult(
            vpin=vpin,
            toxicity_regime=regime,
            buy_volume=total_buy,
            sell_volume=total_sell,
            imbalance=imbalance
        )

    def get_current_vpin(self) -> float:
        """Get current VPIN estimate."""
        result = self.calculate_vpin()
        return result.vpin


class VPINFeatures:
    """
    Generate VPIN-based features for HFT.
    """

    def __init__(self, bucket_sizes: list = None):
        """
        Initialize with multiple bucket sizes for multi-scale analysis.
        """
        if bucket_sizes is None:
            bucket_sizes = [10000, 50000, 100000]  # Small, medium, large

        self.calculators = {
            size: VPINCalculator(bucket_size=size)
            for size in bucket_sizes
        }

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute VPIN features from tick data.

        Args:
            df: DataFrame with columns [timestamp, bid, ask, volume] or [price, volume]

        Returns:
            DataFrame with VPIN features
        """
        if 'mid' in df.columns:
            prices = df['mid'].values
        elif 'bid' in df.columns and 'ask' in df.columns:
            prices = (df['bid'].values + df['ask'].values) / 2
        else:
            prices = df['close'].values if 'close' in df.columns else df['price'].values

        volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(prices))

        n = len(prices)

        # Initialize output arrays
        vpin_values = {size: np.zeros(n) for size in self.calculators}
        toxicity_regime = {size: np.zeros(n) for size in self.calculators}
        imbalance = {size: np.zeros(n) for size in self.calculators}

        # Process each tick
        for i in range(1, n):
            prev_price = prices[i-1]
            price = prices[i]
            volume = volumes[i]

            for size, calc in self.calculators.items():
                result = calc.update(price, prev_price, volume)
                if result:
                    vpin_values[size][i] = result.vpin
                    toxicity_regime[size][i] = result.toxicity_regime
                    imbalance[size][i] = result.imbalance
                else:
                    # Use previous value
                    vpin_values[size][i] = vpin_values[size][i-1]
                    toxicity_regime[size][i] = toxicity_regime[size][i-1]
                    imbalance[size][i] = imbalance[size][i-1]

        # Create output DataFrame
        result = pd.DataFrame(index=df.index)

        for size in self.calculators:
            result[f'vpin_{size}'] = vpin_values[size]
            result[f'toxicity_{size}'] = toxicity_regime[size]
            result[f'vol_imbalance_{size}'] = imbalance[size]

        # Aggregate features
        result['vpin_mean'] = np.mean([vpin_values[s] for s in self.calculators], axis=0)
        result['vpin_max'] = np.max([vpin_values[s] for s in self.calculators], axis=0)
        result['toxicity_alert'] = (result['vpin_max'] > 0.7).astype(int)

        return result


def calculate_vpin_simple(prices: np.ndarray, volumes: np.ndarray,
                         n_buckets: int = 50) -> np.ndarray:
    """
    Simple VPIN calculation for feature engineering.

    Args:
        prices: Mid prices
        volumes: Trade volumes
        n_buckets: Rolling window for VPIN

    Returns:
        Array of VPIN values
    """
    n = len(prices)
    vpin = np.zeros(n)

    # Classify trades using tick rule
    buy_volume = np.zeros(n)
    sell_volume = np.zeros(n)

    for i in range(1, n):
        if prices[i] > prices[i-1]:
            buy_volume[i] = volumes[i]
        elif prices[i] < prices[i-1]:
            sell_volume[i] = volumes[i]
        else:
            buy_volume[i] = volumes[i] / 2
            sell_volume[i] = volumes[i] / 2

    # Rolling VPIN
    for i in range(n_buckets, n):
        window_buy = np.sum(buy_volume[i-n_buckets:i])
        window_sell = np.sum(sell_volume[i-n_buckets:i])
        total = window_buy + window_sell

        if total > 0:
            vpin[i] = abs(window_buy - window_sell) / total

    return vpin
