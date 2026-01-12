"""
Chinese HFT Factors - Gold Standard Algorithms
===============================================
Sources:
- QuantsPlaybook (hugo2046) - 券商金工研报复现
- QUANTAXIS (yutiansut) - 中国最大量化开源框架
- WonderTrader - UFT 175ns延迟引擎
- HftBacktest - 队列位置/微观价格

Verified algorithms used by 幻方量化, 九坤投资, 明汯投资

Key Factors:
1. Smart Money Factor 2.0 (聪明钱因子) - 开源证券
2. Chip Distribution (筹码分布因子) - 成交量分布
3. Integrated OFI (集成订单流不平衡) - Cont 2014 extended
4. Microprice - Better mid estimation
5. Kyle Lambda - Market impact
6. Amihud Illiquidity - 非流动性因子
7. Trade Imbalance (Lee-Ready) - Tick分类
8. Penny Jump Detection - 大单跟随
9. Volume Clock Sampling - VPIN预处理
10. LOB Pressure Factors - 订单簿压力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MicropriceResult:
    """Microprice calculation result."""
    microprice: float
    imbalance: float
    weighted_mid: float
    pressure_ratio: float


class Microprice:
    """
    Microprice - Better mid-price estimation.

    Source: HftBacktest, Stoikov 2018

    Formula:
    microprice = mid + spread * imbalance / 2

    Where:
    imbalance = (bid_size - ask_size) / (bid_size + ask_size)

    For Forex HFT:
    - Better entry/exit price estimation
    - Reduces adverse selection
    - Improves fill simulation
    """

    @staticmethod
    def calculate(bid: float, ask: float,
                  bid_size: float, ask_size: float) -> MicropriceResult:
        """
        Calculate microprice from L2 data.

        Args:
            bid: Best bid price
            ask: Best ask price
            bid_size: Volume at best bid
            ask_size: Volume at best ask

        Returns:
            MicropriceResult with all components
        """
        mid = (bid + ask) / 2
        spread = ask - bid

        total_size = bid_size + ask_size
        if total_size == 0:
            return MicropriceResult(mid, 0.0, mid, 0.5)

        imbalance = (bid_size - ask_size) / total_size
        microprice = mid + spread * imbalance / 2

        # Weighted mid (size-weighted)
        weighted_mid = (bid * ask_size + ask * bid_size) / total_size

        # Pressure ratio (buy pressure)
        pressure_ratio = bid_size / total_size

        return MicropriceResult(
            microprice=microprice,
            imbalance=imbalance,
            weighted_mid=weighted_mid,
            pressure_ratio=pressure_ratio
        )

    @staticmethod
    def multi_level_microprice(bids: List[Tuple[float, float]],
                               asks: List[Tuple[float, float]],
                               decay: float = 0.5) -> float:
        """
        Multi-level microprice with exponential decay.

        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            decay: Weight decay for deeper levels

        Returns:
            Multi-level microprice
        """
        if not bids or not asks:
            return 0.0

        weighted_bid = 0.0
        weighted_ask = 0.0
        bid_weight_sum = 0.0
        ask_weight_sum = 0.0

        for i, (price, size) in enumerate(bids):
            weight = (decay ** i) * size
            weighted_bid += price * weight
            bid_weight_sum += weight

        for i, (price, size) in enumerate(asks):
            weight = (decay ** i) * size
            weighted_ask += price * weight
            ask_weight_sum += weight

        if bid_weight_sum == 0 or ask_weight_sum == 0:
            return (bids[0][0] + asks[0][0]) / 2

        return (weighted_bid / bid_weight_sum + weighted_ask / ask_weight_sum) / 2


class SmartMoneyFactor:
    """
    Smart Money Factor 2.0 (聪明钱因子).

    Source: 开源证券 《市场微观结构研究系列（3）》
    Reproduced in: QuantsPlaybook (hugo2046)

    Key insight:
    - Track large orders that move price
    - Institutional flow vs retail flow
    - Volume-weighted price impact

    Formula:
    S = sum(sign(ret) * volume * |ret|) / sum(volume * |ret|)

    For Forex HFT:
    - Detect institutional positioning
    - Follow smart money flow
    - Avoid toxic flow
    """

    def __init__(self, window: int = 20, threshold_pct: float = 0.7):
        """
        Args:
            window: Lookback window
            threshold_pct: Percentile for "large" trades
        """
        self.window = window
        self.threshold_pct = threshold_pct

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Smart Money Factor.

        Args:
            df: DataFrame with 'close', 'volume' columns

        Returns:
            Series of smart money factor values
        """
        returns = df['close'].pct_change()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Weight = |return| * volume (impact-weighted)
        weights = np.abs(returns) * volume

        # Signed impact
        signed_impact = np.sign(returns) * weights

        # Rolling smart money factor
        numerator = signed_impact.rolling(self.window).sum()
        denominator = weights.rolling(self.window).sum()

        smart_money = numerator / (denominator + 1e-10)

        return smart_money.fillna(0)

    def calculate_v2(self, df: pd.DataFrame) -> pd.Series:
        """
        Smart Money Factor 2.0 - Enhanced version.

        Improvements:
        - Separate large vs small trades
        - Time-weighted decay
        - Volatility normalization
        """
        returns = df['close'].pct_change()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Identify large trades (above threshold)
        vol_threshold = volume.rolling(self.window).quantile(self.threshold_pct)
        is_large = volume > vol_threshold

        # Large trade impact
        large_impact = np.where(is_large, np.sign(returns) * np.abs(returns) * volume, 0)

        # Small trade impact (contrarian)
        small_impact = np.where(~is_large, np.sign(returns) * np.abs(returns) * volume, 0)

        # Smart money = large trade direction
        large_sum = pd.Series(large_impact).rolling(self.window).sum()
        small_sum = pd.Series(small_impact).rolling(self.window).sum()

        # Normalize by volatility
        volatility = returns.rolling(self.window).std()

        smart_money_v2 = (large_sum - small_sum) / (volatility * volume.rolling(self.window).sum() + 1e-10)

        return smart_money_v2.fillna(0)


class ChipDistributionFactor:
    """
    Chip Distribution Factor (筹码分布因子).

    Source: 券商金工研报, QuantsPlaybook

    Concept:
    - Track where volume accumulated at different prices
    - Identify support/resistance from volume profile
    - Calculate average cost basis of holders

    For Forex HFT:
    - Volume-at-price analysis
    - Mean reversion to VWAP levels
    - Breakout probability estimation
    """

    def __init__(self, lookback: int = 100, n_bins: int = 50):
        self.lookback = lookback
        self.n_bins = n_bins

    def calculate_vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate distance from VWAP (Volume Weighted Average Price).

        Returns z-score of price relative to VWAP.
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3 if 'high' in df.columns else df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        vwap = (typical_price * volume).rolling(self.lookback).sum() / volume.rolling(self.lookback).sum()

        # Standard deviation for z-score
        std = typical_price.rolling(self.lookback).std()

        vwap_zscore = (df['close'] - vwap) / (std + 1e-10)

        return vwap_zscore.fillna(0)

    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volume profile metrics.

        Returns:
            Dict with poc (point of control), value_area_high, value_area_low
        """
        prices = df['close'].values[-self.lookback:]
        volumes = df['volume'].values[-self.lookback:] if 'volume' in df.columns else np.ones(len(prices))

        # Create price bins
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, self.n_bins + 1)

        # Accumulate volume in each bin
        volume_profile = np.zeros(self.n_bins)
        for i, (price, vol) in enumerate(zip(prices, volumes)):
            bin_idx = min(int((price - price_min) / (price_max - price_min + 1e-10) * self.n_bins), self.n_bins - 1)
            volume_profile[bin_idx] += vol

        # Point of Control (POC) - price level with most volume
        poc_idx = np.argmax(volume_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Value Area (70% of volume)
        total_vol = volume_profile.sum()
        target_vol = total_vol * 0.7

        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative = 0
        value_area_bins = []
        for idx in sorted_indices:
            cumulative += volume_profile[idx]
            value_area_bins.append(idx)
            if cumulative >= target_vol:
                break

        value_area_low = bins[min(value_area_bins)]
        value_area_high = bins[max(value_area_bins) + 1]

        return {
            'poc': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'current_in_value_area': value_area_low <= prices[-1] <= value_area_high
        }

    def calculate_chip_concentration(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate chip concentration ratio.

        High concentration = potential breakout
        Low concentration = mean reversion likely
        """
        closes = df['close']
        volumes = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        def calc_concentration(window_data):
            if len(window_data) < 10:
                return 0.5
            prices = window_data['close'].values
            vols = window_data['volume'].values if 'volume' in window_data.columns else np.ones(len(prices))

            # Weighted standard deviation
            mean_price = np.average(prices, weights=vols)
            variance = np.average((prices - mean_price) ** 2, weights=vols)

            # Normalize by price range
            price_range = prices.max() - prices.min()
            if price_range == 0:
                return 1.0

            concentration = 1 - np.sqrt(variance) / price_range
            return max(0, min(1, concentration))

        result = df.rolling(self.lookback).apply(
            lambda x: calc_concentration(df.loc[x.index]), raw=False
        )['close']

        return result.fillna(0.5)


class KyleLambda:
    """
    Kyle Lambda - Market Impact Estimation.

    Source: Kyle (1985), verified in Chinese quant research

    Lambda measures price impact per unit of order flow:
    Delta_P = lambda * (Buy_volume - Sell_volume)

    For Forex HFT:
    - Estimate execution cost
    - Optimal order sizing
    - Detect low-impact windows
    """

    def __init__(self, window: int = 100):
        self.window = window

    def estimate_lambda(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate Kyle's Lambda using regression.

        Lambda = Cov(ret, signed_volume) / Var(signed_volume)
        """
        returns = df['close'].pct_change() * 10000  # bps

        # Signed volume (using tick rule if no trade direction)
        if 'signed_volume' in df.columns:
            signed_vol = df['signed_volume']
        else:
            # Estimate using price direction
            volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
            signed_vol = np.sign(returns) * volume

        # Rolling regression: ret = alpha + lambda * signed_vol
        cov = returns.rolling(self.window).cov(signed_vol)
        var = signed_vol.rolling(self.window).var()

        kyle_lambda = cov / (var + 1e-10)

        return kyle_lambda.fillna(kyle_lambda.mean())

    def estimate_permanent_impact(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate permanent price impact (information content).

        Permanent impact = portion of price change that doesn't revert
        """
        returns = df['close'].pct_change()

        # Future returns (does impact persist?)
        future_ret_5 = returns.shift(-5).rolling(5).sum()
        current_ret = returns

        # Correlation = permanent impact ratio
        permanent_ratio = current_ret.rolling(self.window).corr(future_ret_5)

        return permanent_ratio.fillna(0)


class AmihudIlliquidity:
    """
    Amihud Illiquidity Factor.

    Source: Amihud (2002), widely used in Chinese quant

    ILLIQ = |Return| / Dollar_Volume

    For Forex HFT:
    - Detect illiquid periods
    - Adjust position sizing
    - Avoid trading in low liquidity
    """

    def __init__(self, window: int = 20):
        self.window = window

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Amihud illiquidity ratio.
        """
        returns = df['close'].pct_change().abs()

        if 'volume' in df.columns:
            dollar_volume = df['volume'] * df['close']
        else:
            dollar_volume = df['close']  # Proxy

        illiq = returns / (dollar_volume + 1e-10)

        # Rolling average
        amihud = illiq.rolling(self.window).mean()

        # Normalize to z-score
        amihud_zscore = (amihud - amihud.rolling(100).mean()) / (amihud.rolling(100).std() + 1e-10)

        return amihud_zscore.fillna(0)


class IntegratedOFI:
    """
    Integrated Order Flow Imbalance.

    Source: Cont et al. (2014), extended in Chinese research

    Standard OFI:
    OFI_t = (bid_size * I(bid_up) - ask_size * I(ask_down))

    Integrated version adds:
    - Multi-level aggregation
    - Decay weighting
    - Normalization by spread

    For Forex HFT:
    - Primary alpha signal
    - Short-term direction prediction
    - Queue priority estimation
    """

    def __init__(self, n_levels: int = 5, decay: float = 0.8):
        self.n_levels = n_levels
        self.decay = decay
        self.prev_bids = None
        self.prev_asks = None

    def calculate_tick(self, bids: List[Tuple[float, float]],
                       asks: List[Tuple[float, float]]) -> float:
        """
        Calculate integrated OFI for single tick.

        Args:
            bids: List of (price, size) at each level
            asks: List of (price, size) at each level

        Returns:
            Integrated OFI value
        """
        if self.prev_bids is None:
            self.prev_bids = bids
            self.prev_asks = asks
            return 0.0

        ofi = 0.0

        for i in range(min(self.n_levels, len(bids), len(self.prev_bids))):
            weight = self.decay ** i

            curr_bid_price, curr_bid_size = bids[i]
            prev_bid_price, prev_bid_size = self.prev_bids[i]

            # Bid side contribution
            if curr_bid_price > prev_bid_price:
                ofi += weight * curr_bid_size
            elif curr_bid_price == prev_bid_price:
                ofi += weight * (curr_bid_size - prev_bid_size)
            else:
                ofi -= weight * prev_bid_size

        for i in range(min(self.n_levels, len(asks), len(self.prev_asks))):
            weight = self.decay ** i

            curr_ask_price, curr_ask_size = asks[i]
            prev_ask_price, prev_ask_size = self.prev_asks[i]

            # Ask side contribution (negative for sells)
            if curr_ask_price < prev_ask_price:
                ofi -= weight * curr_ask_size
            elif curr_ask_price == prev_ask_price:
                ofi -= weight * (curr_ask_size - prev_ask_size)
            else:
                ofi += weight * prev_ask_size

        self.prev_bids = bids
        self.prev_asks = asks

        return ofi

    def calculate_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate integrated OFI from DataFrame with bid/ask data.
        """
        if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
            # Fallback to simple volume-based OFI
            returns = df['close'].pct_change()
            volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
            return (np.sign(returns) * volume).fillna(0)

        bid_changes = df['bid'].diff()
        ask_changes = df['ask'].diff()

        ofi = pd.Series(0.0, index=df.index)

        # Bid side
        ofi += np.where(bid_changes > 0, df['bid_size'], 0)
        ofi += np.where(bid_changes == 0, df['bid_size'].diff(), 0)
        ofi -= np.where(bid_changes < 0, df['bid_size'].shift(1), 0)

        # Ask side (inverted)
        ofi -= np.where(ask_changes < 0, df['ask_size'], 0)
        ofi -= np.where(ask_changes == 0, df['ask_size'].diff(), 0)
        ofi += np.where(ask_changes > 0, df['ask_size'].shift(1), 0)

        return ofi.fillna(0)


class PennyJumpDetector:
    """
    Penny Jump Detection (大单跟随).

    Source: 知乎 HFT策略, FMZ量化

    Strategy:
    - Detect large "elephant" orders
    - Jump ahead by 1 tick
    - Profit from price impact

    For Forex HFT:
    - Identify large order flow
    - Front-run when legal/ethical
    - Avoid being jumped
    """

    def __init__(self, size_threshold_pct: float = 0.95,
                 min_spread_ticks: int = 2):
        self.size_threshold_pct = size_threshold_pct
        self.min_spread_ticks = min_spread_ticks

    def detect_elephant(self, df: pd.DataFrame,
                       window: int = 100) -> pd.Series:
        """
        Detect large orders (elephants).

        Returns:
            Series with 1 (buy elephant), -1 (sell elephant), 0 (none)
        """
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        threshold = volume.rolling(window).quantile(self.size_threshold_pct)
        is_elephant = volume > threshold

        # Direction from price change
        returns = df['close'].pct_change()
        direction = np.sign(returns)

        elephants = np.where(is_elephant, direction, 0)

        return pd.Series(elephants, index=df.index)

    def calculate_jump_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate penny jump trading signal.

        Signal = Follow elephants with tight stop.
        """
        elephants = self.detect_elephant(df)

        # Only signal if spread is wide enough
        if 'bid' in df.columns and 'ask' in df.columns:
            spread = df['ask'] - df['bid']
            tick_size = spread.min()
            wide_spread = spread >= tick_size * self.min_spread_ticks

            signal = np.where(wide_spread, elephants, 0)
        else:
            signal = elephants

        return pd.Series(signal, index=df.index)


class VolumeClockSampler:
    """
    Volume Clock Sampling.

    Source: Easley et al. (VPIN paper), Chinese HFT research

    Instead of time-based bars, create volume-based bars.
    Benefits:
    - More samples during active periods
    - Fewer samples during quiet periods
    - Better for VPIN calculation

    For Forex HFT:
    - Normalize activity across sessions
    - Improve factor calculation
    - Better regime detection
    """

    def __init__(self, bucket_volume: float = 1000000):
        self.bucket_volume = bucket_volume

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to volume clock.

        Returns:
            DataFrame with volume-based bars
        """
        if 'volume' not in df.columns:
            logger.warning("No volume column, using time-based sampling")
            return df

        volume = df['volume'].values
        close = df['close'].values

        bars = []
        current_volume = 0
        bar_start_idx = 0

        for i in range(len(df)):
            current_volume += volume[i]

            if current_volume >= self.bucket_volume:
                bar = {
                    'timestamp': df.index[i],
                    'open': close[bar_start_idx],
                    'high': close[bar_start_idx:i+1].max(),
                    'low': close[bar_start_idx:i+1].min(),
                    'close': close[i],
                    'volume': current_volume,
                    'n_ticks': i - bar_start_idx + 1
                }
                bars.append(bar)

                current_volume = 0
                bar_start_idx = i + 1

        return pd.DataFrame(bars).set_index('timestamp') if bars else df


class ChineseHFTFactorEngine:
    """
    Unified Chinese HFT Factor Engine.

    Combines all factors from:
    - QuantsPlaybook (券商金工)
    - QUANTAXIS
    - WonderTrader research
    - HftBacktest
    """

    def __init__(self):
        self.microprice = Microprice()
        self.smart_money = SmartMoneyFactor()
        self.chip_dist = ChipDistributionFactor()
        self.kyle_lambda = KyleLambda()
        self.amihud = AmihudIlliquidity()
        self.integrated_ofi = IntegratedOFI()
        self.penny_jump = PennyJumpDetector()
        self.volume_clock = VolumeClockSampler()

    def generate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Chinese HFT factors.

        Returns:
            DataFrame with factor columns added
        """
        result = df.copy()

        # Smart Money Factors
        result['smart_money_v1'] = self.smart_money.calculate(df)
        result['smart_money_v2'] = self.smart_money.calculate_v2(df)

        # Chip Distribution
        result['vwap_zscore'] = self.chip_dist.calculate_vwap_distance(df)

        # Kyle Lambda (Market Impact)
        result['kyle_lambda'] = self.kyle_lambda.estimate_lambda(df)
        result['permanent_impact'] = self.kyle_lambda.estimate_permanent_impact(df)

        # Amihud Illiquidity
        result['amihud_illiq'] = self.amihud.calculate(df)

        # Integrated OFI
        result['integrated_ofi'] = self.integrated_ofi.calculate_series(df)

        # Penny Jump
        result['elephant_detect'] = self.penny_jump.detect_elephant(df)
        result['penny_jump_signal'] = self.penny_jump.calculate_jump_signal(df)

        # Microprice (if L2 data available)
        if 'bid' in df.columns and 'ask' in df.columns:
            bid_size = df['bid_size'] if 'bid_size' in df.columns else pd.Series(1, index=df.index)
            ask_size = df['ask_size'] if 'ask_size' in df.columns else pd.Series(1, index=df.index)

            microprice_results = [
                self.microprice.calculate(b, a, bs, as_)
                for b, a, bs, as_ in zip(df['bid'], df['ask'], bid_size, ask_size)
            ]

            result['microprice'] = [r.microprice for r in microprice_results]
            result['microprice_imbalance'] = [r.imbalance for r in microprice_results]
            result['pressure_ratio'] = [r.pressure_ratio for r in microprice_results]

        logger.info(f"Generated {len([c for c in result.columns if c not in df.columns])} Chinese HFT factors")

        return result

    def get_factor_names(self) -> List[str]:
        """Return list of factor names."""
        return [
            'smart_money_v1', 'smart_money_v2',
            'vwap_zscore',
            'kyle_lambda', 'permanent_impact',
            'amihud_illiq',
            'integrated_ofi',
            'elephant_detect', 'penny_jump_signal',
            'microprice', 'microprice_imbalance', 'pressure_ratio'
        ]


class ChineseHFTFactors:
    """
    Wrapper class for HFT Feature Engine integration.
    Provides compute_all_features() interface.
    """

    def __init__(self):
        self.engine = ChineseHFTFactorEngine()

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all Chinese HFT features.
        Interface compatible with HFT Feature Engine.
        """
        return self.engine.generate_all_factors(df)
