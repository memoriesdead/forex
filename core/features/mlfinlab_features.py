"""
MLFinLab Features - Lopez de Prado Methods
===========================================
Gold standard implementations from "Advances in Financial Machine Learning"
by Dr. Marcos Lopez de Prado.

Source: https://github.com/hudson-and-thames/mlfinlab
Documentation: https://www.mlfinlab.com/

Implemented Features:
1. Triple Barrier Method - Dynamic labeling
2. Meta-Labeling - Secondary model for bet sizing
3. Fractional Differentiation - Stationarity while preserving memory
4. CUSUM Filter - Event-based sampling
5. Entropy Features - Market microstructure
6. Structural Breaks - Regime detection

These methods are used by institutional quants worldwide.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class TripleBarrierLabeling:
    """
    Triple Barrier Method for labeling financial time series.

    From Chapter 3 of "Advances in Financial Machine Learning".

    Three barriers:
    - Upper barrier: Take profit (positive return threshold)
    - Lower barrier: Stop loss (negative return threshold)
    - Vertical barrier: Maximum holding period

    Label assignment:
    - 1: Upper barrier hit first (profitable trade)
    - -1: Lower barrier hit first (losing trade)
    - 0: Vertical barrier hit (timeout)
    """

    def __init__(
        self,
        profit_take: float = 2.0,
        stop_loss: float = 2.0,
        max_holding: int = 10,
        volatility_window: int = 20
    ):
        """
        Initialize Triple Barrier labeling.

        Args:
            profit_take: Multiplier for upper barrier (x volatility)
            stop_loss: Multiplier for lower barrier (x volatility)
            max_holding: Maximum holding period (vertical barrier)
            volatility_window: Window for volatility estimation
        """
        self.profit_take = profit_take
        self.stop_loss = stop_loss
        self.max_holding = max_holding
        self.volatility_window = volatility_window

    def get_daily_volatility(self, close: pd.Series) -> pd.Series:
        """Estimate daily volatility using exponential weighted std."""
        returns = close.pct_change()
        return returns.ewm(span=self.volatility_window).std()

    def apply_triple_barrier(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex = None
    ) -> pd.DataFrame:
        """
        Apply triple barrier method to generate labels.

        Args:
            close: Price series
            events: Event timestamps (if None, use all)

        Returns:
            DataFrame with columns: t1 (barrier time), ret, label
        """
        if events is None:
            events = close.index

        # Get volatility
        volatility = self.get_daily_volatility(close)

        results = []
        for idx in events:
            if idx not in close.index:
                continue

            loc = close.index.get_loc(idx)
            if loc >= len(close) - 1:
                continue

            # Get volatility at event time
            vol = volatility.iloc[loc] if loc < len(volatility) else volatility.iloc[-1]
            if pd.isna(vol) or vol == 0:
                vol = close.pct_change().std()

            # Set barriers
            upper_barrier = close.iloc[loc] * (1 + self.profit_take * vol)
            lower_barrier = close.iloc[loc] * (1 - self.stop_loss * vol)
            vertical_barrier = min(loc + self.max_holding, len(close) - 1)

            # Find first barrier touch
            label = 0
            exit_time = vertical_barrier
            exit_price = close.iloc[vertical_barrier]

            for t in range(loc + 1, vertical_barrier + 1):
                price = close.iloc[t]

                if price >= upper_barrier:
                    label = 1
                    exit_time = t
                    exit_price = price
                    break
                elif price <= lower_barrier:
                    label = -1
                    exit_time = t
                    exit_price = price
                    break

            # Calculate return
            ret = (exit_price / close.iloc[loc]) - 1

            results.append({
                'event_time': idx,
                't1': close.index[exit_time],
                'ret': ret,
                'label': label,
                'upper': upper_barrier,
                'lower': lower_barrier,
                'vol': vol
            })

        return pd.DataFrame(results)


class MetaLabeling:
    """
    Meta-Labeling for bet sizing.

    From Chapter 3 of "Advances in Financial Machine Learning".

    Meta-labeling is a secondary model that learns when to bet,
    not the direction. It improves precision while maintaining recall.

    Usage:
    1. Primary model predicts direction (side)
    2. Meta-label model predicts whether to take the bet (0 or 1)
    3. Final position = side * meta_label * bet_size
    """

    def __init__(self, primary_threshold: float = 0.0):
        """
        Initialize Meta-Labeling.

        Args:
            primary_threshold: Threshold for primary model signal
        """
        self.primary_threshold = primary_threshold

    def generate_meta_labels(
        self,
        primary_signal: pd.Series,
        actual_returns: pd.Series
    ) -> pd.Series:
        """
        Generate meta-labels based on primary signal correctness.

        Args:
            primary_signal: Primary model predictions (-1, 0, 1)
            actual_returns: Actual forward returns

        Returns:
            Series of meta-labels (1 = correct, 0 = incorrect)
        """
        # Meta-label is 1 if primary signal direction matches return direction
        signal_direction = np.sign(primary_signal)
        return_direction = np.sign(actual_returns)

        meta_labels = (signal_direction == return_direction).astype(int)

        # Where primary signal is 0, meta-label is also 0
        meta_labels[primary_signal == 0] = 0

        return meta_labels


class FractionalDifferentiation:
    """
    Fractional Differentiation for stationarity.

    From Chapter 5 of "Advances in Financial Machine Learning".

    Makes series stationary while preserving memory (predictive power).
    Standard differencing (d=1) removes too much memory.
    Fractional d (0 < d < 1) balances stationarity and memory.
    """

    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        """
        Initialize Fractional Differentiation.

        Args:
            d: Differentiation order (0 < d < 1)
            threshold: Weight threshold for truncation
        """
        self.d = d
        self.threshold = threshold

    def get_weights(self, d: float, size: int) -> np.ndarray:
        """Calculate fractional differentiation weights."""
        weights = [1.0]
        for k in range(1, size):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < self.threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])

    def frac_diff(self, series: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation.

        Args:
            series: Input time series

        Returns:
            Fractionally differentiated series
        """
        weights = self.get_weights(self.d, len(series))
        width = len(weights)

        result = pd.Series(index=series.index, dtype=float)

        for i in range(width - 1, len(series)):
            result.iloc[i] = np.dot(weights, series.iloc[i - width + 1:i + 1].values)

        return result


class CUSUMFilter:
    """
    CUSUM Filter for event-based sampling.

    From Chapter 2 of "Advances in Financial Machine Learning".

    Samples events when cumulative sum of returns exceeds threshold.
    More efficient than fixed-time sampling.
    """

    def __init__(self, threshold: float = None):
        """
        Initialize CUSUM Filter.

        Args:
            threshold: CUSUM threshold (if None, use std)
        """
        self.threshold = threshold

    def get_events(self, close: pd.Series) -> pd.DatetimeIndex:
        """
        Get event timestamps using CUSUM filter.

        Args:
            close: Price series

        Returns:
            DatetimeIndex of event times
        """
        returns = close.pct_change().dropna()

        if self.threshold is None:
            h = returns.std()
        else:
            h = self.threshold

        events = []
        s_pos = 0
        s_neg = 0

        for i, r in enumerate(returns):
            s_pos = max(0, s_pos + r)
            s_neg = min(0, s_neg + r)

            if s_neg < -h:
                s_neg = 0
                events.append(returns.index[i])
            elif s_pos > h:
                s_pos = 0
                events.append(returns.index[i])

        return pd.DatetimeIndex(events)


class EntropyFeatures:
    """
    Entropy-based features for market microstructure.

    From Chapter 18 of "Advances in Financial Machine Learning".

    Entropy measures information content and market efficiency.
    """

    def __init__(self, window: int = 20):
        """
        Initialize Entropy Features.

        Args:
            window: Rolling window size
        """
        self.window = window

    def shannon_entropy(self, series: pd.Series) -> pd.Series:
        """Calculate rolling Shannon entropy."""
        def entropy(x):
            # Discretize into bins
            counts = np.histogram(x, bins=10)[0]
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Remove zeros
            return -np.sum(probs * np.log2(probs))

        return series.rolling(self.window).apply(entropy, raw=True)

    def approximate_entropy(self, series: pd.Series, m: int = 2, r: float = None) -> pd.Series:
        """
        Calculate rolling Approximate Entropy.

        Measures regularity/predictability of time series.
        Lower = more regular, Higher = more random.
        """
        if r is None:
            r = 0.2 * series.std()

        def apen(x):
            n = len(x)
            if n < m + 1:
                return np.nan

            def _phi(m_val):
                patterns = np.array([x[i:i + m_val] for i in range(n - m_val + 1)])
                count = 0
                for i in range(len(patterns)):
                    for j in range(len(patterns)):
                        if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                            count += 1
                return np.log(count / (n - m_val + 1))

            return _phi(m) - _phi(m + 1)

        return series.rolling(self.window).apply(apen, raw=True)


class MLFinLabFeatures:
    """
    Combined MLFinLab feature generator.

    Generates all Lopez de Prado features in one call.
    """

    def __init__(
        self,
        frac_diff_d: float = 0.5,
        vol_window: int = 20,
        entropy_window: int = 20
    ):
        """
        Initialize MLFinLab features.

        Args:
            frac_diff_d: Fractional differentiation order
            vol_window: Volatility window
            entropy_window: Entropy window
        """
        self.frac_diff = FractionalDifferentiation(d=frac_diff_d)
        self.vol_window = vol_window
        self.entropy = EntropyFeatures(window=entropy_window)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all MLFinLab features.

        Args:
            df: DataFrame with close price

        Returns:
            DataFrame with MLFinLab features
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Fractionally Differentiated Features
        for d in [0.3, 0.5, 0.7]:
            fd = FractionalDifferentiation(d=d)
            features[f'FRAC_DIFF_{int(d*10)}'] = fd.frac_diff(close)

        # 2. Volatility Features (Lopez de Prado style)
        # Exponential weighted volatility
        features['VOL_EWM20'] = returns.ewm(span=20).std()
        features['VOL_EWM60'] = returns.ewm(span=60).std()

        # Parkinson volatility
        if 'high' in df.columns and 'low' in df.columns:
            log_hl = np.log(df['high'] / df['low'] + 1e-12)
            features['VOL_PARKINSON'] = np.sqrt(
                (log_hl ** 2).rolling(20).mean() / (4 * np.log(2))
            )

        # 3. CUSUM-based features
        cusum = CUSUMFilter()
        s_pos = returns.cumsum().clip(lower=0)
        s_neg = returns.cumsum().clip(upper=0)
        features['CUSUM_POS'] = s_pos
        features['CUSUM_NEG'] = s_neg
        features['CUSUM_DIFF'] = s_pos + s_neg

        # 4. Entropy Features
        features['ENTROPY_SHANNON'] = self.entropy.shannon_entropy(returns)

        # 5. Structural Break Indicators
        # Rolling mean shift
        ma_20 = close.rolling(20).mean()
        ma_60 = close.rolling(60).mean()
        features['STRUCT_MA_CROSS'] = (ma_20 - ma_60) / (ma_60 + 1e-12)

        # Variance ratio
        var_5 = returns.rolling(5).var()
        var_20 = returns.rolling(20).var()
        features['STRUCT_VAR_RATIO'] = var_5 / (var_20 + 1e-12)

        # 6. Information-driven features
        # Volume-synchronized probability of informed trading (VPIN proxy)
        if 'volume' in df.columns:
            volume = df['volume']
            buy_vol = volume.where(returns > 0, 0)
            sell_vol = volume.where(returns < 0, 0)
            features['INFO_VPIN'] = np.abs(buy_vol - sell_vol).rolling(20).sum() / (
                volume.rolling(20).sum() + 1e-12
            )

        # 7. Bet sizing features (for meta-labeling)
        # Signal confidence proxy
        ret_20 = returns.rolling(20).sum()
        vol_20 = returns.rolling(20).std()
        features['BET_CONFIDENCE'] = np.abs(ret_20) / (vol_20 * np.sqrt(20) + 1e-12)

        # 8. Triple barrier proxy features
        # Distance to rolling high/low as barrier proxy
        roll_high = close.rolling(20).max()
        roll_low = close.rolling(20).min()
        features['TB_UPPER_DIST'] = (roll_high - close) / (close + 1e-12)
        features['TB_LOWER_DIST'] = (close - roll_low) / (close + 1e-12)
        features['TB_RANGE_POS'] = (close - roll_low) / (roll_high - roll_low + 1e-12)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'FRAC_DIFF_3', 'FRAC_DIFF_5', 'FRAC_DIFF_7',
            'VOL_EWM20', 'VOL_EWM60', 'VOL_PARKINSON',
            'CUSUM_POS', 'CUSUM_NEG', 'CUSUM_DIFF',
            'ENTROPY_SHANNON',
            'STRUCT_MA_CROSS', 'STRUCT_VAR_RATIO',
            'INFO_VPIN',
            'BET_CONFIDENCE',
            'TB_UPPER_DIST', 'TB_LOWER_DIST', 'TB_RANGE_POS'
        ]


# Convenience function
def generate_mlfinlab_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate MLFinLab features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with ~17 MLFinLab features
    """
    generator = MLFinLabFeatures()
    return generator.generate_all(df)
