"""
Time-Series-Library Inspired Features
=====================================
Feature engineering inspired by state-of-the-art time series models.

Source: https://github.com/thuml/Time-Series-Library (11k+ stars)

Inspired by:
1. iTransformer (ICLR 2024 Spotlight) - Inverted attention on variates
2. TimeMixer (ICLR 2024) - Multi-scale decomposition
3. PatchTST (ICLR 2023) - Patch-based representation
4. TimesNet (ICLR 2023) - 2D temporal variation

These features capture patterns that deep learning models learn,
but in a form usable by gradient boosting models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft
import warnings

warnings.filterwarnings('ignore')


class MultiScaleDecomposition:
    """
    Multi-scale decomposition inspired by TimeMixer.

    TimeMixer decomposes series into multiple scales for
    Past-Decomposable-Mixing (PDM).

    We implement similar decomposition using moving averages
    at multiple scales.
    """

    def __init__(self, scales: List[int] = None):
        """
        Initialize multi-scale decomposition.

        Args:
            scales: List of scales (window sizes)
        """
        self.scales = scales or [5, 10, 20, 40, 80]

    def decompose(self, series: pd.Series) -> pd.DataFrame:
        """
        Decompose series into trend and seasonal at multiple scales.

        Returns DataFrame with trend_N and seasonal_N for each scale.
        """
        features = pd.DataFrame(index=series.index)

        for scale in self.scales:
            # Trend component (moving average)
            trend = series.rolling(scale, min_periods=1).mean()
            features[f'TREND_{scale}'] = trend / (series + 1e-12)

            # Seasonal/residual component
            seasonal = series - trend
            features[f'SEASONAL_{scale}'] = seasonal / (series.rolling(20).std() + 1e-12)

        return features


class PatchFeatures:
    """
    Patch-based features inspired by PatchTST.

    PatchTST divides time series into patches and processes them
    independently before aggregating.

    We create statistics over non-overlapping patches.
    """

    def __init__(self, patch_size: int = 8, num_patches: int = 4):
        """
        Initialize patch features.

        Args:
            patch_size: Size of each patch
            num_patches: Number of patches to use
        """
        self.patch_size = patch_size
        self.num_patches = num_patches

    def extract_patch_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Extract statistics from patches.

        Returns features for each patch: mean, std, trend.
        """
        features = pd.DataFrame(index=series.index)
        total_window = self.patch_size * self.num_patches

        for i in range(self.num_patches):
            start_offset = i * self.patch_size
            end_offset = (i + 1) * self.patch_size

            # Patch mean (relative to current value)
            patch_mean = series.rolling(total_window).apply(
                lambda x: x[-end_offset:-start_offset].mean() if start_offset > 0
                else x[-end_offset:].mean(),
                raw=True
            )
            features[f'PATCH{i}_MEAN'] = patch_mean / (series + 1e-12)

            # Patch std
            patch_std = series.rolling(total_window).apply(
                lambda x: x[-end_offset:-start_offset].std() if start_offset > 0
                else x[-end_offset:].std(),
                raw=True
            )
            features[f'PATCH{i}_STD'] = patch_std / (series.rolling(20).std() + 1e-12)

        return features


class TemporalVariation2D:
    """
    2D temporal variation features inspired by TimesNet.

    TimesNet reshapes 1D time series into 2D based on
    detected periods, then applies 2D convolutions.

    We extract period-based features using FFT.
    """

    def __init__(self, top_k_periods: int = 3):
        """
        Initialize temporal variation features.

        Args:
            top_k_periods: Number of top periods to extract
        """
        self.top_k_periods = top_k_periods

    def detect_periods(self, series: np.ndarray) -> List[int]:
        """Detect dominant periods using FFT."""
        n = len(series)
        if n < 10:
            return [5, 10, 20]

        # Compute FFT
        fft_vals = np.abs(fft(series - series.mean()))
        freqs = np.fft.fftfreq(n)

        # Get positive frequencies only
        pos_mask = freqs > 0
        fft_pos = fft_vals[pos_mask]
        freqs_pos = freqs[pos_mask]

        # Find top-k peaks
        if len(fft_pos) < self.top_k_periods:
            return [5, 10, 20]

        top_indices = np.argsort(fft_pos)[-self.top_k_periods:]
        periods = [int(1 / freqs_pos[i]) if freqs_pos[i] > 0 else 10
                   for i in top_indices]

        # Filter valid periods
        periods = [p for p in periods if 2 < p < 100]
        if not periods:
            periods = [5, 10, 20]

        return sorted(periods)

    def extract_period_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Extract features based on detected periods.
        """
        features = pd.DataFrame(index=series.index)

        # Use fixed periods (more stable for trading)
        periods = [5, 10, 20, 40]

        for period in periods:
            # Autocorrelation at period lag
            autocorr = series.rolling(period * 2).apply(
                lambda x: pd.Series(x).autocorr(lag=period) if len(x) > period else 0,
                raw=False
            )
            features[f'PERIOD{period}_AUTOCORR'] = autocorr

            # Period-aligned returns
            features[f'PERIOD{period}_RET'] = series / series.shift(period) - 1

        return features


class InvertedVariateFeatures:
    """
    Inverted variate features inspired by iTransformer.

    iTransformer applies attention across variates (features)
    instead of time steps, treating each variate as a token.

    We create cross-variate interaction features.
    """

    def __init__(self):
        """Initialize inverted variate features."""
        pass

    def extract_variate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cross-variate interaction features.

        For OHLCV data, creates interactions between price components.
        """
        features = pd.DataFrame(index=df.index)

        if 'open' in df.columns and 'close' in df.columns:
            # Open-Close relationship
            features['VAR_OC_RATIO'] = df['open'] / (df['close'] + 1e-12)
            features['VAR_OC_DIFF'] = (df['close'] - df['open']) / (df['open'] + 1e-12)

        if 'high' in df.columns and 'low' in df.columns:
            # High-Low relationship
            features['VAR_HL_RATIO'] = df['high'] / (df['low'] + 1e-12)
            features['VAR_HL_RANGE'] = (df['high'] - df['low']) / (df['close'] + 1e-12)

            # Body vs wick ratios
            if 'open' in df.columns and 'close' in df.columns:
                body = np.abs(df['close'] - df['open'])
                upper_wick = df['high'] - np.maximum(df['open'], df['close'])
                lower_wick = np.minimum(df['open'], df['close']) - df['low']
                total_range = df['high'] - df['low'] + 1e-12

                features['VAR_BODY_PCT'] = body / total_range
                features['VAR_UPPER_WICK'] = upper_wick / total_range
                features['VAR_LOWER_WICK'] = lower_wick / total_range

        if 'volume' in df.columns and 'close' in df.columns:
            # Volume-price relationship
            returns = df['close'].pct_change()
            features['VAR_VOL_RET_CORR'] = returns.rolling(20).corr(
                df['volume'].pct_change()
            )

        return features


class TimeSeriesLibraryFeatures:
    """
    Combined Time-Series-Library inspired features.

    Captures patterns from:
    - TimeMixer (multi-scale)
    - PatchTST (patch-based)
    - TimesNet (period-based)
    - iTransformer (variate interactions)
    """

    def __init__(
        self,
        scales: List[int] = None,
        patch_size: int = 8,
        num_patches: int = 4
    ):
        """
        Initialize Time-Series-Library features.

        Args:
            scales: Scales for multi-scale decomposition
            patch_size: Size of patches
            num_patches: Number of patches
        """
        self.multi_scale = MultiScaleDecomposition(scales)
        self.patch = PatchFeatures(patch_size, num_patches)
        self.temporal = TemporalVariation2D()
        self.inverted = InvertedVariateFeatures()

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Time-Series-Library inspired features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with ~45 features
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # 1. Multi-scale decomposition (TimeMixer style)
        ms_features = self.multi_scale.decompose(close)
        features = pd.concat([features, ms_features], axis=1)

        # 2. Patch features (PatchTST style)
        patch_features = self.patch.extract_patch_features(close)
        features = pd.concat([features, patch_features], axis=1)

        # 3. Period features (TimesNet style)
        period_features = self.temporal.extract_period_features(close)
        features = pd.concat([features, period_features], axis=1)

        # 4. Variate interactions (iTransformer style)
        variate_features = self.inverted.extract_variate_interactions(df)
        features = pd.concat([features, variate_features], axis=1)

        # 5. Additional deep learning inspired features

        # Attention-like features: softmax of returns
        returns = close.pct_change()
        for window in [5, 10, 20]:
            # Softmax attention weights proxy
            ret_std = returns.rolling(window).std()
            normalized_ret = returns / (ret_std + 1e-12)
            exp_ret = np.exp(normalized_ret.clip(-5, 5))  # Clip to prevent overflow
            features[f'ATTN_WEIGHT_{window}'] = exp_ret / (exp_ret.rolling(window).sum() + 1e-12)

        # Positional encoding proxy: sin/cos of position in trend
        trend_pos = (close - close.rolling(60).min()) / (
            close.rolling(60).max() - close.rolling(60).min() + 1e-12
        )
        features['POS_SIN'] = np.sin(2 * np.pi * trend_pos)
        features['POS_COS'] = np.cos(2 * np.pi * trend_pos)

        # Layer normalization proxy: z-score
        features['LAYER_NORM'] = (close - close.rolling(20).mean()) / (
            close.rolling(20).std() + 1e-12
        )

        # Residual connection proxy: current vs smoothed
        features['RESIDUAL'] = close / close.rolling(10).mean() - 1

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        names = []

        # Multi-scale
        for scale in self.multi_scale.scales:
            names.extend([f'TREND_{scale}', f'SEASONAL_{scale}'])

        # Patch
        for i in range(self.patch.num_patches):
            names.extend([f'PATCH{i}_MEAN', f'PATCH{i}_STD'])

        # Period
        for period in [5, 10, 20, 40]:
            names.extend([f'PERIOD{period}_AUTOCORR', f'PERIOD{period}_RET'])

        # Variate
        names.extend([
            'VAR_OC_RATIO', 'VAR_OC_DIFF', 'VAR_HL_RATIO', 'VAR_HL_RANGE',
            'VAR_BODY_PCT', 'VAR_UPPER_WICK', 'VAR_LOWER_WICK', 'VAR_VOL_RET_CORR'
        ])

        # Attention/position
        names.extend([
            'ATTN_WEIGHT_5', 'ATTN_WEIGHT_10', 'ATTN_WEIGHT_20',
            'POS_SIN', 'POS_COS', 'LAYER_NORM', 'RESIDUAL'
        ])

        return names


# Convenience function
def generate_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Time-Series-Library inspired features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with ~45 features
    """
    generator = TimeSeriesLibraryFeatures()
    return generator.generate_all(df)
