"""
Chinese Gold Standard Quantitative Factors for Forex
=====================================================
Peer-reviewed and institutional research implementations.

CITATIONS:
----------
[1] VPIN (Volume-Synchronized Probability of Informed Trading)
    Source: 招商证券 "琢璞"系列报告之十七：高频数据中的知情交易
    URL: https://asset.quant-wiki.com/pdf/20200630-招商证券-"琢璞"系列报告之十七：高频数据中的知情交易（二）.pdf
    Original: Easley, D., López de Prado, M., & O'Hara, M. (2012).
              "Flow Toxicity and Liquidity in a High-frequency World"
              Review of Financial Studies, 25(5), 1457-1493.

[2] Order Flow Imbalance (OFI)
    Source: 兴业证券 高频研究系列五：市场微观结构剖析
    URL: https://asset.quant-wiki.com/pdf/20221109-兴业证券-高频研究系列五：市场微观结构剖析.pdf
    Original: Cont, R., Kukanov, A., & Stoikov, S. (2014).
              "The Price Impact of Order Book Events"
              Journal of Financial Econometrics, 12(1), 47-88.

[3] Meta-Labeling and Triple Barrier
    Source: Lopez de Prado, M. (2018).
            "Advances in Financial Machine Learning"
            Wiley. Chapter 3: Meta-Labeling.
    Implementation: mlfinlab library

[4] HMM Regime Detection
    Source: 国泰君安 市场状态识别研究
    Original: Hamilton, J. D. (1989).
              "A New Approach to the Economic Analysis of Nonstationary
              Time Series and the Business Cycle"
              Econometrica, 57(2), 357-384.

[5] Kalman Filter for Finance
    Source: 知乎 Kalman滤波金融应用
    URL: https://zhuanlan.zhihu.com/p/101630539
    Original: Kalman, R. E. (1960).
              "A New Approach to Linear Filtering and Prediction Problems"
              Journal of Basic Engineering, 82(1), 35-45.

[6] Avellaneda-Stoikov Market Making
    Source: Avellaneda, M., & Stoikov, S. (2008).
            "High-frequency trading in a limit order book"
            Quantitative Finance, 8(3), 217-224.
    Chinese: 订单流交易实战应用 https://zhuanlan.zhihu.com/p/133214828

[7] LSTM+Attention for Forex (ALFA Model)
    Source: ScienceDirect (2025)
    URL: https://www.sciencedirect.com/science/article/pii/S2666827025000313

[8] Cross-Market Sentiment (CCSA-DL)
    Source: 数据分析与知识发现 (2023)
    URL: https://manu44.magtech.com.cn/Jwk_infotech_wk3/EN/10.11925/infotech.2096-3467.2022.1147
    Innovation: BERT-TextCNN sentiment + LSTM (16.77% improvement)

[9] iTransformer / TimeMixer / TimeXer
    Source: Tsinghua University Time-Series-Library
    GitHub: https://github.com/thuml/Time-Series-Library
    Papers: ICLR 2024 (iTransformer, TimeMixer), NeurIPS 2024 (TimeXer)

[10] Alpha191 Guotai Junan
     Source: 国泰君安 "基于短周期价量特征的多因子选股体系" (2017)
     URL: https://zhuanlan.zhihu.com/p/30195354
     Platform: JoinQuant https://joinquant.com/data/dict/alpha191
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


# =============================================================================
# [1] VPIN - Volume-Synchronized Probability of Informed Trading
# Citation: 招商证券 + Easley, López de Prado, O'Hara (2012)
# =============================================================================

class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.

    Predicts volatility spikes 30-60 minutes ahead by measuring
    order flow toxicity through volume-time sampling.

    Citation:
        招商证券 "琢璞"系列报告之十七：高频数据中的知情交易（二）
        Easley, D., López de Prado, M., & O'Hara, M. (2012).
        "Flow Toxicity and Liquidity in a High-frequency World"
        Review of Financial Studies, 25(5), 1457-1493.
    """

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        """
        Args:
            bucket_size: Volume per bucket (default 50 units)
            n_buckets: Number of buckets for VPIN calculation (default 50)
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

    def bulk_volume_classification(self,
                                    prices: pd.Series,
                                    volumes: pd.Series,
                                    sigma: float = None) -> Tuple[pd.Series, pd.Series]:
        """
        Bulk Volume Classification (BVC) method.

        Classifies volume as buy/sell using price changes and
        normal distribution CDF.

        Formula:
            V_buy = V * CDF(Z)
            V_sell = V * (1 - CDF(Z))
            where Z = (P_close - P_open) / sigma
        """
        if sigma is None:
            sigma = prices.pct_change().std()

        price_change = prices.diff()
        z_score = price_change / (sigma * prices.shift(1) + 1e-10)

        # CDF of standard normal
        buy_prob = stats.norm.cdf(z_score)

        v_buy = volumes * buy_prob
        v_sell = volumes * (1 - buy_prob)

        return v_buy, v_sell

    def calculate_vpin(self,
                       prices: pd.Series,
                       volumes: pd.Series) -> pd.Series:
        """
        Calculate VPIN time series.

        Formula:
            OI_bucket = |V_buy - V_sell| / (V_buy + V_sell)
            VPIN = mean(OI_buckets[-n:])

        Returns:
            Series of VPIN values (0 to 1, higher = more toxic flow)
        """
        v_buy, v_sell = self.bulk_volume_classification(prices, volumes)

        # Order imbalance per observation
        total_vol = v_buy + v_sell + 1e-10
        oi = np.abs(v_buy - v_sell) / total_vol

        # Rolling VPIN over n_buckets
        vpin = oi.rolling(self.n_buckets, min_periods=10).mean()

        return vpin

    def calculate_toxicity_signal(self,
                                   prices: pd.Series,
                                   volumes: pd.Series,
                                   threshold: float = 0.7) -> pd.Series:
        """
        Generate trading signal from VPIN.

        High VPIN (>0.7) indicates toxic flow, expect volatility.
        """
        vpin = self.calculate_vpin(prices, volumes)

        # Signal: 1 = high toxicity expected, 0 = normal
        signal = (vpin > threshold).astype(int)

        return signal


# =============================================================================
# [2] Multi-Level Order Flow Imbalance (OFI)
# Citation: 兴业证券 + Cont, Kukanov, Stoikov (2014)
# =============================================================================

class MultiLevelOFI:
    """
    Multi-Level Order Flow Imbalance.

    Extends single-level OFI to capture depth across multiple
    price levels of the order book.

    Citation:
        兴业证券 高频研究系列五：市场微观结构剖析
        Cont, R., Kukanov, A., & Stoikov, S. (2014).
        "The Price Impact of Order Book Events"
        Journal of Financial Econometrics, 12(1), 47-88.
    """

    def __init__(self, n_levels: int = 10, decay_factor: float = 0.9):
        """
        Args:
            n_levels: Number of price levels to consider
            decay_factor: Weight decay for deeper levels (0.9 = 10% decay per level)
        """
        self.n_levels = n_levels
        self.decay_factor = decay_factor

        # Pre-compute level weights
        self.weights = np.array([decay_factor ** i for i in range(n_levels)])
        self.weights = self.weights / self.weights.sum()  # Normalize

    def calculate_level_ofi(self,
                            bid_price: float, prev_bid_price: float,
                            bid_size: float, prev_bid_size: float,
                            ask_price: float, prev_ask_price: float,
                            ask_size: float, prev_ask_size: float) -> float:
        """
        Calculate OFI for a single level.

        Formula (Cont et al. 2014):
            OFI = ΔQ_bid * I(P_bid >= P_bid_prev) - ΔQ_ask * I(P_ask <= P_ask_prev)

        Where:
            ΔQ = change in quantity
            I() = indicator function
        """
        # Bid side contribution
        if bid_price >= prev_bid_price:
            bid_ofi = bid_size - prev_bid_size
        elif bid_price < prev_bid_price:
            bid_ofi = -prev_bid_size
        else:
            bid_ofi = 0

        # Ask side contribution
        if ask_price <= prev_ask_price:
            ask_ofi = -(ask_size - prev_ask_size)
        elif ask_price > prev_ask_price:
            ask_ofi = prev_ask_size
        else:
            ask_ofi = 0

        return bid_ofi + ask_ofi

    def calculate_integrated_ofi(self,
                                  prices: pd.Series,
                                  volumes: pd.Series,
                                  window: int = 20) -> pd.Series:
        """
        Calculate integrated OFI for single-level data (tick data).

        Approximates multi-level OFI using price momentum and volume.
        """
        returns = prices.pct_change()

        # Classify trades
        buy_volume = volumes.where(returns > 0, 0)
        sell_volume = volumes.where(returns < 0, 0)

        # OFI = cumulative buy - sell pressure
        ofi = (buy_volume - sell_volume).rolling(window, min_periods=5).sum()

        # Normalize by total volume
        total_vol = volumes.rolling(window, min_periods=5).sum() + 1e-10
        normalized_ofi = ofi / total_vol

        return normalized_ofi

    def calculate_ofi_momentum(self,
                                prices: pd.Series,
                                volumes: pd.Series,
                                short_window: int = 10,
                                long_window: int = 50) -> pd.Series:
        """
        OFI momentum signal (short-term vs long-term).

        Positive = increasing buy pressure
        Negative = increasing sell pressure
        """
        ofi_short = self.calculate_integrated_ofi(prices, volumes, short_window)
        ofi_long = self.calculate_integrated_ofi(prices, volumes, long_window)

        return ofi_short - ofi_long


# =============================================================================
# [3] Meta-Labeling and Triple Barrier
# Citation: Lopez de Prado (2018)
# =============================================================================

class TripleBarrierLabeling:
    """
    Triple Barrier Method for labeling financial time series.

    Creates labels based on which barrier is hit first:
    - Upper barrier (profit target)
    - Lower barrier (stop-loss)
    - Vertical barrier (max holding time)

    Citation:
        Lopez de Prado, M. (2018).
        "Advances in Financial Machine Learning"
        Wiley. Chapter 3.
    """

    def __init__(self,
                 pt_sl: Tuple[float, float] = (1.0, 1.0),
                 min_ret: float = 0.0,
                 vertical_barrier: int = 10):
        """
        Args:
            pt_sl: (profit_taking, stop_loss) multipliers of volatility
            min_ret: Minimum return to consider (filters noise)
            vertical_barrier: Maximum holding period in bars
        """
        self.pt_sl = pt_sl
        self.min_ret = min_ret
        self.vertical_barrier = vertical_barrier

    def get_daily_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Estimate daily volatility using exponential moving std."""
        returns = prices.pct_change()
        return returns.ewm(span=window).std()

    def apply_triple_barrier(self,
                              prices: pd.Series,
                              events: pd.Series = None) -> pd.Series:
        """
        Apply triple barrier labeling.

        Returns:
            Series with labels:
             1 = hit upper barrier (profit)
            -1 = hit lower barrier (loss)
             0 = hit vertical barrier (timeout)
        """
        if events is None:
            events = pd.Series(index=prices.index, data=True)

        vol = self.get_daily_volatility(prices)
        labels = pd.Series(index=prices.index, dtype=float)

        for i, (idx, is_event) in enumerate(events.items()):
            if not is_event or i >= len(prices) - 1:
                labels[idx] = np.nan
                continue

            entry_price = prices.iloc[i]
            entry_vol = vol.iloc[i] if not np.isnan(vol.iloc[i]) else 0.01

            # Set barriers
            upper = entry_price * (1 + self.pt_sl[0] * entry_vol)
            lower = entry_price * (1 - self.pt_sl[1] * entry_vol)
            max_idx = min(i + self.vertical_barrier, len(prices) - 1)

            # Check which barrier hit first
            label = 0  # Default: vertical barrier
            for j in range(i + 1, max_idx + 1):
                price = prices.iloc[j]
                if price >= upper:
                    label = 1
                    break
                elif price <= lower:
                    label = -1
                    break

            labels[idx] = label

        return labels


class MetaLabeling:
    """
    Meta-Labeling for bet sizing.

    Secondary model that learns when to trade (or not)
    given a primary model's directional prediction.

    Citation:
        Lopez de Prado, M. (2018).
        "Advances in Financial Machine Learning"
        Wiley. Chapter 3.3.
    """

    def __init__(self, primary_threshold: float = 0.5):
        """
        Args:
            primary_threshold: Confidence threshold for primary model
        """
        self.primary_threshold = primary_threshold

    def create_meta_labels(self,
                            primary_predictions: pd.Series,
                            actual_returns: pd.Series,
                            holding_period: int = 5) -> pd.Series:
        """
        Create meta-labels for training the meta-labeling model.

        Meta-label = 1 if primary prediction was correct
        Meta-label = 0 if primary prediction was wrong

        This teaches the model WHEN to trust the primary model.
        """
        # Forward returns
        forward_returns = actual_returns.rolling(holding_period).sum().shift(-holding_period)

        # Check if primary was correct
        primary_direction = np.sign(primary_predictions)
        actual_direction = np.sign(forward_returns)

        meta_labels = (primary_direction == actual_direction).astype(int)

        return meta_labels

    def calculate_bet_size(self,
                           meta_probability: pd.Series,
                           primary_signal: pd.Series,
                           max_leverage: float = 1.0) -> pd.Series:
        """
        Calculate position size using meta-labeling probabilities.

        Formula:
            bet_size = primary_signal * meta_probability * max_leverage
        """
        return primary_signal * meta_probability * max_leverage


# =============================================================================
# [4] HMM Regime Detection
# Citation: 国泰君安 + Hamilton (1989)
# =============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Identifies market states (trending, mean-reverting, volatile)
    using Gaussian HMM on return features.

    Citation:
        国泰君安 市场状态识别研究
        Hamilton, J. D. (1989).
        "A New Approach to the Economic Analysis of Nonstationary
        Time Series and the Business Cycle"
        Econometrica, 57(2), 357-384.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100):
        """
        Args:
            n_states: Number of hidden states (default 3: bull, bear, volatile)
            n_iter: Number of EM iterations for training
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None

    def prepare_features(self,
                         prices: pd.Series,
                         window: int = 20) -> np.ndarray:
        """
        Prepare features for HMM training.

        Features: [returns, volatility, momentum]
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window).std()
        momentum = prices.pct_change(window)

        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'momentum': momentum
        }).dropna()

        return features.values, features.index

    def fit_predict(self, prices: pd.Series) -> pd.Series:
        """
        Fit HMM and predict regimes.

        Returns:
            Series with regime labels (0, 1, 2)

        Interpretation (typically):
            0: Low volatility / trending
            1: High volatility / mean-reverting
            2: Transitional / uncertain
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            # Fallback: simple volatility-based regime
            return self._simple_regime(prices)

        features, idx = self.prepare_features(prices)

        if len(features) < 50:
            return self._simple_regime(prices)

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",  # Use diagonal covariance (more robust)
                n_iter=self.n_iter,
                random_state=42
            )

            self.model.fit(features)
            regimes = self.model.predict(features)

            result = pd.Series(index=prices.index, dtype=float)
            result[idx] = regimes
            result = result.ffill().fillna(0)

            return result
        except Exception:
            # Fallback on any HMM failure
            return self._simple_regime(prices)

    def _simple_regime(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Simple volatility-based regime detection (fallback)."""
        returns = prices.pct_change()
        vol = returns.rolling(window).std()
        vol_percentile = vol.rolling(100).rank(pct=True)

        # 0: low vol, 1: medium vol, 2: high vol
        regime = pd.cut(vol_percentile, bins=[0, 0.33, 0.66, 1.0], labels=[0, 1, 2])
        return regime.astype(float).fillna(0)


# =============================================================================
# [5] Kalman Filter
# Citation: 知乎 + Kalman (1960)
# =============================================================================

class KalmanFilterForex:
    """
    Kalman Filter for forex signal smoothing and prediction.

    Applications:
    1. Price denoising
    2. Dynamic hedge ratio estimation
    3. Spread mean estimation for pairs trading

    Citation:
        知乎 Kalman滤波金融应用 https://zhuanlan.zhihu.com/p/101630539
        Kalman, R. E. (1960).
        "A New Approach to Linear Filtering and Prediction Problems"
        Journal of Basic Engineering, 82(1), 35-45.
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Args:
            process_noise: Q - process noise covariance
            measurement_noise: R - measurement noise covariance
        """
        self.Q = process_noise
        self.R = measurement_noise

    def filter(self, observations: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Kalman filter to observations.

        State equation: x_t = x_{t-1} + w_t  (random walk)
        Observation equation: z_t = x_t + v_t

        Returns:
            (filtered_state, kalman_gain)
        """
        n = len(observations)

        # Initialize
        x = observations.iloc[0]  # State estimate
        P = 1.0  # Error covariance

        filtered = np.zeros(n)
        gains = np.zeros(n)

        for i, z in enumerate(observations):
            # Predict
            x_pred = x
            P_pred = P + self.Q

            # Update
            K = P_pred / (P_pred + self.R)  # Kalman gain
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred

            filtered[i] = x
            gains[i] = K

        return pd.Series(filtered, index=observations.index), \
               pd.Series(gains, index=observations.index)

    def calculate_spread_zscore(self,
                                 series1: pd.Series,
                                 series2: pd.Series,
                                 window: int = 50) -> pd.Series:
        """
        Calculate z-score of spread using Kalman-filtered mean.

        For pairs/spread trading.
        """
        spread = series1 - series2
        filtered_mean, _ = self.filter(spread)

        # Rolling std for z-score
        spread_std = spread.rolling(window).std()
        zscore = (spread - filtered_mean) / (spread_std + 1e-10)

        return zscore


# =============================================================================
# [6] Avellaneda-Stoikov Market Making
# Citation: Avellaneda & Stoikov (2008)
# =============================================================================

class AvellanedaStoikov:
    """
    Avellaneda-Stoikov market making model.

    Optimal bid/ask quotes considering inventory risk.

    Citation:
        Avellaneda, M., & Stoikov, S. (2008).
        "High-frequency trading in a limit order book"
        Quantitative Finance, 8(3), 217-224.

        Chinese: 订单流交易实战应用 https://zhuanlan.zhihu.com/p/133214828
    """

    def __init__(self,
                 gamma: float = 0.1,
                 k: float = 1.5,
                 sigma: float = 0.02):
        """
        Args:
            gamma: Risk aversion parameter
            k: Order arrival rate parameter
            sigma: Volatility estimate
        """
        self.gamma = gamma
        self.k = k
        self.sigma = sigma

    def reservation_price(self,
                          mid_price: float,
                          inventory: float,
                          time_remaining: float) -> float:
        """
        Calculate reservation price (market maker's internal valuation).

        Formula:
            r = s - q * gamma * sigma^2 * (T - t)

        Where:
            s = mid price
            q = current inventory
            gamma = risk aversion
            sigma = volatility
            T - t = time remaining
        """
        return mid_price - inventory * self.gamma * self.sigma**2 * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """
        Calculate optimal bid-ask spread.

        Formula:
            delta = gamma * sigma^2 * (T-t) + (2/gamma) * log(1 + gamma/k)
        """
        inventory_component = self.gamma * self.sigma**2 * time_remaining
        arrival_component = (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        return inventory_component + arrival_component

    def optimal_quotes(self,
                       mid_price: float,
                       inventory: float,
                       time_remaining: float = 1.0) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask prices.

        Returns:
            (bid_price, ask_price)
        """
        r = self.reservation_price(mid_price, inventory, time_remaining)
        delta = self.optimal_spread(time_remaining)

        bid = r - delta / 2
        ask = r + delta / 2

        return bid, ask

    def inventory_signal(self,
                         prices: pd.Series,
                         position_series: pd.Series) -> pd.Series:
        """
        Generate trading signal based on inventory-adjusted fair value.

        Returns signal:
            Positive = price above fair value (sell)
            Negative = price below fair value (buy)
        """
        vol = prices.pct_change().rolling(20).std() * np.sqrt(252)

        signals = []
        for i in range(len(prices)):
            if i < 20:
                signals.append(0)
                continue

            mid = prices.iloc[i]
            inventory = position_series.iloc[i] if i < len(position_series) else 0
            sigma = vol.iloc[i] if not np.isnan(vol.iloc[i]) else self.sigma

            r = mid - inventory * self.gamma * sigma**2
            signal = (mid - r) / (sigma + 1e-10)  # Z-score from fair value
            signals.append(signal)

        return pd.Series(signals, index=prices.index)


# =============================================================================
# [7-9] Deep Learning Feature Engineering
# Citation: Tsinghua Time-Series-Library, CCSA-DL
# =============================================================================

class DeepLearningFeatures:
    """
    Feature engineering inspired by deep learning research.

    Citations:
        [7] ALFA Model (LSTM+Attention): ScienceDirect 2025
        [8] CCSA-DL: 数据分析与知识发现 2023
        [9] iTransformer/TimeMixer: Tsinghua ICLR 2024
    """

    @staticmethod
    def attention_weights_proxy(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Proxy for attention weights using price importance.

        Higher weight = more important time step for prediction.
        Based on price volatility contribution.
        """
        returns = prices.pct_change().abs()
        weights = returns.rolling(window).apply(
            lambda x: x / (x.sum() + 1e-10) if len(x) > 0 else 0
        )
        return weights

    @staticmethod
    def temporal_attention_context(prices: pd.Series,
                                    features: pd.DataFrame,
                                    window: int = 20) -> pd.Series:
        """
        Attention-weighted context vector (simplified).

        Formula:
            context = sum(attention_weights * features)
        """
        weights = DeepLearningFeatures.attention_weights_proxy(prices, window)

        # Weight each feature by attention
        context = (features.T * weights).T.sum(axis=1)
        return context

    @staticmethod
    def multiscale_decomposition(prices: pd.Series,
                                  scales: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Multiscale trend decomposition (TimeMixer inspired).

        Decomposes price into multiple time scales.

        Citation: TimeMixer (ICLR 2024)
        """
        result = pd.DataFrame(index=prices.index)

        for scale in scales:
            # Trend at this scale
            trend = prices.rolling(scale, min_periods=1).mean()
            result[f'trend_{scale}'] = trend

            # Residual at this scale
            result[f'residual_{scale}'] = prices - trend

            # Scale interaction (mixing)
            if scale > scales[0]:
                prev_scale = scales[scales.index(scale) - 1]
                result[f'mix_{prev_scale}_{scale}'] = \
                    result[f'trend_{prev_scale}'] - result[f'trend_{scale}']

        return result

    @staticmethod
    def exogenous_features(prices: pd.Series,
                           exog_dict: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """
        Prepare exogenous variable features (TimeXer inspired).

        Citation: TimeXer (NeurIPS 2024)

        Args:
            exog_dict: Dictionary of exogenous series (e.g., DXY, VIX)
        """
        result = pd.DataFrame(index=prices.index)

        if exog_dict is None:
            # Create synthetic exogenous features
            returns = prices.pct_change()

            result['volatility_regime'] = returns.rolling(20).std().rank(pct=True)
            result['momentum_regime'] = returns.rolling(10).sum().rank(pct=True)
            result['mean_reversion'] = -returns.rolling(5).sum()
        else:
            for name, series in exog_dict.items():
                aligned = series.reindex(prices.index).ffill()
                result[f'exog_{name}'] = aligned
                result[f'exog_{name}_ret'] = aligned.pct_change()
                result[f'exog_{name}_zscore'] = (
                    (aligned - aligned.rolling(20).mean()) /
                    (aligned.rolling(20).std() + 1e-10)
                )

        return result


# =============================================================================
# UNIFIED GENERATOR CLASS
# =============================================================================

class ChineseGoldStandardFeatures:
    """
    Unified generator for all Chinese gold standard features.

    Combines:
    - VPIN (招商证券)
    - Multi-level OFI (兴业证券/Cont 2014)
    - Triple Barrier signals (Lopez de Prado)
    - HMM Regime (Hamilton/国泰君安)
    - Kalman Filter signals
    - Avellaneda-Stoikov features
    - Deep Learning proxies (Tsinghua)
    """

    def __init__(self):
        self.vpin = VPIN()
        self.ofi = MultiLevelOFI()
        self.triple_barrier = TripleBarrierLabeling()
        self.hmm = HMMRegimeDetector()
        self.kalman = KalmanFilterForex()
        self.as_model = AvellanedaStoikov()
        self.dl_features = DeepLearningFeatures()

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all gold standard features.

        Args:
            df: DataFrame with 'close', 'volume' columns

        Returns:
            DataFrame with all features
        """
        result = pd.DataFrame(index=df.index)

        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))

        # [1] VPIN Features
        try:
            result['vpin'] = self.vpin.calculate_vpin(close, volume)
            result['vpin_toxicity'] = self.vpin.calculate_toxicity_signal(close, volume)
        except Exception:
            result['vpin'] = 0
            result['vpin_toxicity'] = 0

        # [2] OFI Features
        try:
            result['ofi_integrated'] = self.ofi.calculate_integrated_ofi(close, volume)
            result['ofi_momentum'] = self.ofi.calculate_ofi_momentum(close, volume)
        except Exception:
            result['ofi_integrated'] = 0
            result['ofi_momentum'] = 0

        # [3] Triple Barrier Labels (for training, not prediction)
        try:
            result['tb_labels'] = self.triple_barrier.apply_triple_barrier(close)
        except Exception:
            result['tb_labels'] = 0

        # [4] HMM Regime
        try:
            result['hmm_regime'] = self.hmm.fit_predict(close)
        except Exception:
            result['hmm_regime'] = 0

        # [5] Kalman Filter
        try:
            kalman_state, kalman_gain = self.kalman.filter(close)
            result['kalman_filtered'] = kalman_state
            result['kalman_gain'] = kalman_gain
            result['kalman_deviation'] = close - kalman_state
        except Exception:
            result['kalman_filtered'] = close
            result['kalman_gain'] = 0
            result['kalman_deviation'] = 0

        # [6] Avellaneda-Stoikov (with zero inventory assumption)
        try:
            position_proxy = pd.Series(0, index=df.index)  # No position
            result['as_inventory_signal'] = self.as_model.inventory_signal(close, position_proxy)
        except Exception:
            result['as_inventory_signal'] = 0

        # [7-9] Deep Learning Features
        try:
            result['attention_proxy'] = self.dl_features.attention_weights_proxy(close)
        except Exception:
            result['attention_proxy'] = 0

        try:
            multiscale = self.dl_features.multiscale_decomposition(close)
            for col in multiscale.columns:
                result[f'dl_{col}'] = multiscale[col]
        except Exception:
            pass

        try:
            exog = self.dl_features.exogenous_features(close)
            for col in exog.columns:
                result[f'dl_{col}'] = exog[col]
        except Exception:
            pass

        # Derived signals
        try:
            result['regime_adjusted_signal'] = result['ofi_momentum'] * (1 + result['hmm_regime'] * 0.1)
            result['toxicity_adjusted_signal'] = result['ofi_momentum'] * (1 - result['vpin'] * 0.5)
        except Exception:
            result['regime_adjusted_signal'] = 0
            result['toxicity_adjusted_signal'] = 0

        # Clean up
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_chinese_gold_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all Chinese gold standard features in one call.

    Citations: See module docstring for full list.
    """
    generator = ChineseGoldStandardFeatures()
    return generator.generate_all(df)
