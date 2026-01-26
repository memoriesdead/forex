"""
Academic Deep Learning Models for Time Series Forecasting
==========================================================

References to state-of-the-art deep learning architectures for
time series forecasting from top ML conferences (ICLR, NeurIPS).

This module provides configuration classes and integration points
for models from the Time-Series-Library (https://github.com/thuml/Time-Series-Library).

CITATIONS:
----------

1. TEMPORAL FUSION TRANSFORMER (TFT)
   Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021).
   "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
   International Journal of Forecasting, 37(4), 1748-1764.
   DOI: 10.1016/j.ijforecast.2021.01.012
   arXiv: https://arxiv.org/abs/1912.09363
   Google Research: https://research.google/pubs/temporal-fusion-transformers/

2. iTRANSFORMER (ICLR 2024 Spotlight)
   Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
   "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
   ICLR 2024.
   arXiv: https://arxiv.org/abs/2310.06625
   GitHub: https://github.com/thuml/iTransformer

3. TimeXer (NeurIPS 2024)
   Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., & Long, M. (2024).
   "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
   NeurIPS 2024.
   arXiv: https://arxiv.org/abs/2402.19072
   GitHub: https://github.com/thuml/TimeXer

4. N-BEATS (ICLR 2020)
   Oreshkin, B.N., Carpov, D., Chapados, N., & Bengio, Y. (2020).
   "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting"
   ICLR 2020.
   arXiv: https://arxiv.org/abs/1905.10437
   GitHub: https://github.com/ServiceNow/N-BEATS

5. TimeMixer (ICLR 2024)
   Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., & Long, M. (2024).
   "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting"
   ICLR 2024 Spotlight.
   GitHub: https://github.com/thuml/Time-Series-Library

6. PatchTST (ICLR 2023)
   Nie, Y., Nguyen, N.H., Sinthong, P., & Kalagnanam, J. (2023).
   "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
   ICLR 2023.
   arXiv: https://arxiv.org/abs/2211.14730

KEY ARCHITECTURAL INSIGHTS:
---------------------------

TEMPORAL FUSION TRANSFORMER (TFT):
    - Variable Selection Network: Learns which features are important
    - Gated Residual Network: Nonlinear feature processing
    - Interpretable Multi-Head Attention: Temporal patterns
    - Quantile Outputs: Probabilistic forecasting

    "TFT proposes a novel interpretable Multi-Head Attention mechanism,
    which contrary to the standard implementation, provides feature
    interpretability."
    - Lim et al. (2021)

iTRANSFORMER:
    - Inverted Tokenization: Embed entire time series per variate as token
    - Cross-Variate Attention: Captures multivariate correlations
    - Per-Variate FFN: Learns temporal dynamics

    "iTransformer simply applies the attention and feed-forward network
    on the inverted dimensions. Specifically, the time points of individual
    series are embedded into variate tokens which are utilized by the
    attention mechanism to capture multivariate correlations."
    - Liu et al. (2024)

N-BEATS:
    - Neural Basis Expansion: Trend and seasonality decomposition
    - Doubly Residual Architecture: Forecast and backcast
    - Interpretable Stacks: Polynomial (trend) + Fourier (seasonality)

    "N-BEATS demonstrated state-of-the-art performance, improving forecast
    accuracy by 11% over a statistical benchmark and by 3% over the winner
    of the M4 competition."
    - Oreshkin et al. (2020)

APPLICABILITY:
--------------
All models are ADAPTABLE for forex:
- TFT: Best for multi-horizon forecasting with known future inputs (time features)
- iTransformer: Best for multivariate (multi-pair) forecasting
- TimeXer: Best when using exogenous variables (interest rates, sentiment)
- N-BEATS: Best for univariate, interpretable forecasting
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class TFTConfig:
    """
    Temporal Fusion Transformer Configuration.

    Reference: Lim et al. (2021), International Journal of Forecasting

    The TFT architecture handles:
    - Static covariates (e.g., currency pair identifier)
    - Known future inputs (e.g., day of week, hour)
    - Unknown future inputs (e.g., price, volume)
    """
    # Model dimensions
    hidden_size: int = 64
    attention_head_size: int = 4
    num_attention_heads: int = 4
    hidden_continuous_size: int = 16
    dropout: float = 0.1

    # Sequence lengths
    encoder_length: int = 168  # Lookback (1 week hourly)
    prediction_length: int = 24  # Forecast horizon

    # Training
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100

    # Quantiles for probabilistic forecasting
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class iTransformerConfig:
    """
    iTransformer Configuration.

    Reference: Liu et al. (2024), ICLR 2024

    Key innovation: Invert the standard Transformer to apply attention
    across variates instead of time points.
    """
    # Model dimensions
    d_model: int = 512
    d_ff: int = 512
    n_heads: int = 8
    e_layers: int = 3
    dropout: float = 0.1

    # Sequence lengths
    seq_len: int = 96  # Lookback
    pred_len: int = 24  # Forecast horizon

    # Number of variates (e.g., multiple currency pairs)
    enc_in: int = 7  # Input variates
    dec_in: int = 7  # Decoder variates
    c_out: int = 7  # Output variates

    # Training
    learning_rate: float = 0.0001
    batch_size: int = 32


@dataclass
class TimeXerConfig:
    """
    TimeXer Configuration.

    Reference: Wang et al. (2024), NeurIPS 2024

    Designed for forecasting with exogenous variables.
    """
    # Model dimensions
    d_model: int = 256
    d_ff: int = 256
    n_heads: int = 8
    e_layers: int = 2
    dropout: float = 0.1

    # Patch settings
    patch_len: int = 16
    stride: int = 8

    # Sequence lengths
    seq_len: int = 96
    pred_len: int = 24

    # Variates
    enc_in: int = 1  # Endogenous (target)
    exo_in: int = 5  # Exogenous (features)


@dataclass
class NBEATSConfig:
    """
    N-BEATS Configuration.

    Reference: Oreshkin et al. (2020), ICLR 2020

    Architecture:
    - Stack of blocks with trend and seasonality basis functions
    - Doubly residual learning (forecast + backcast)
    """
    # Architecture
    num_stacks: int = 2
    num_blocks: int = 3
    num_layers: int = 4
    layer_size: int = 512

    # Basis functions
    degree_of_polynomial: int = 3  # For trend stack
    num_harmonics: int = 1  # For seasonality stack

    # Sequence lengths
    backcast_length: int = 10  # Lookback multiplier
    forecast_length: int = 1  # Forecast multiplier

    # Generic vs Interpretable
    generic_architecture: bool = False  # True = generic, False = interpretable


# ============================================================================
# TIME-SERIES-LIBRARY INTEGRATION
# ============================================================================

class TimeSeriesLibraryModels:
    """
    Integration with Tsinghua Time-Series-Library.

    GitHub: https://github.com/thuml/Time-Series-Library

    This class provides utilities to prepare data and configurations
    for models in the Time-Series-Library.

    Supported models:
    - iTransformer
    - TimeXer
    - TimeMixer
    - PatchTST
    - DLinear
    - And many more...
    """

    AVAILABLE_MODELS = [
        'iTransformer',
        'TimeXer',
        'TimeMixer',
        'PatchTST',
        'DLinear',
        'FEDformer',
        'Autoformer',
        'Informer',
        'Transformer',
    ]

    @staticmethod
    def prepare_forex_data(
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: List[str] = None,
        seq_len: int = 96,
        pred_len: int = 24
    ) -> Dict:
        """
        Prepare forex data for Time-Series-Library models.

        Args:
            df: DataFrame with OHLCV data
            target_col: Target column name
            feature_cols: Feature column names
            seq_len: Lookback length
            pred_len: Prediction length

        Returns:
            Dictionary with prepared data arrays
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Create sequences
        data = df[feature_cols].values
        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len, target_idx])

        return {
            'X': np.array(X),
            'y': np.array(y),
            'feature_cols': feature_cols,
            'target_col': target_col,
            'seq_len': seq_len,
            'pred_len': pred_len
        }

    @staticmethod
    def get_model_config(
        model_name: str,
        seq_len: int = 96,
        pred_len: int = 24,
        enc_in: int = 7
    ) -> Dict:
        """
        Get default configuration for a Time-Series-Library model.

        Args:
            model_name: Model name
            seq_len: Lookback length
            pred_len: Prediction length
            enc_in: Number of input variates

        Returns:
            Configuration dictionary
        """
        base_config = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'enc_in': enc_in,
            'dec_in': enc_in,
            'c_out': enc_in,
            'd_model': 512,
            'd_ff': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
            'factor': 3,
            'output_attention': False,
        }

        # Model-specific overrides
        if model_name == 'iTransformer':
            base_config['class_strategy'] = 'projection'
        elif model_name == 'TimeXer':
            base_config['patch_len'] = 16
            base_config['stride'] = 8
        elif model_name == 'PatchTST':
            base_config['patch_len'] = 16
            base_config['stride'] = 8
            base_config['padding_patch'] = 'end'

        return base_config


# ============================================================================
# FEATURE GENERATION FOR DEEP LEARNING
# ============================================================================

class DeepLearningFeatures:
    """
    Generate features optimized for deep learning models.

    These features are designed to work well with Transformer-based
    architectures like TFT and iTransformer.
    """

    @staticmethod
    def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-based features for TFT.

        These are "known future inputs" that the model can use
        for forecasting.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            DataFrame with time features
        """
        features = pd.DataFrame(index=df.index)

        if isinstance(df.index, pd.DatetimeIndex):
            # Cyclical encoding (better for neural networks)
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

            features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

            features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

            # Forex session indicators
            hour = df.index.hour
            features['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(float)
            features['is_london_session'] = ((hour >= 8) & (hour < 16)).astype(float)
            features['is_ny_session'] = ((hour >= 13) & (hour < 22)).astype(float)

            # Weekend proximity (forex closes Friday, opens Sunday)
            features['day_of_week'] = df.index.dayofweek
            features['is_friday'] = (df.index.dayofweek == 4).astype(float)
            features['is_monday'] = (df.index.dayofweek == 0).astype(float)

        return features

    @staticmethod
    def normalize_for_dl(
        df: pd.DataFrame,
        method: str = 'zscore',
        window: int = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize data for deep learning.

        Args:
            df: DataFrame to normalize
            method: 'zscore', 'minmax', or 'robust'
            window: Rolling window (None = global)

        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        params = {}

        if method == 'zscore':
            if window:
                mean = df.rolling(window).mean()
                std = df.rolling(window).std()
            else:
                mean = df.mean()
                std = df.std()

            normalized = (df - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std}

        elif method == 'minmax':
            if window:
                min_val = df.rolling(window).min()
                max_val = df.rolling(window).max()
            else:
                min_val = df.min()
                max_val = df.max()

            normalized = (df - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val}

        elif method == 'robust':
            if window:
                median = df.rolling(window).median()
                q75 = df.rolling(window).quantile(0.75)
                q25 = df.rolling(window).quantile(0.25)
                iqr = q75 - q25
            else:
                median = df.median()
                q75 = df.quantile(0.75)
                q25 = df.quantile(0.25)
                iqr = q75 - q25

            normalized = (df - median) / (iqr + 1e-8)
            params = {'median': median, 'iqr': iqr}

        else:
            raise ValueError(f"Unknown method: {method}")

        return normalized.fillna(0), params


class AcademicDeepLearningFeatures:
    """
    Generate features for academic deep learning models.

    This class prepares data and configurations for state-of-the-art
    time series forecasting models.
    """

    def __init__(self):
        self.dl_features = DeepLearningFeatures()
        self.tsl_models = TimeSeriesLibraryModels()

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features suitable for deep learning.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with DL-ready features
        """
        features = pd.DataFrame(index=df.index)

        # Time features (for TFT known future inputs)
        time_feats = self.dl_features.generate_time_features(df)
        features = pd.concat([features, time_feats], axis=1)

        # Price features
        close = df['close']
        returns = close.pct_change()

        # Log returns (better for DL)
        features['log_return'] = np.log(close / close.shift(1))

        # Normalized price (relative to rolling mean)
        for w in [20, 50, 100]:
            features[f'price_norm_{w}'] = close / close.rolling(w).mean() - 1

        # Volatility features (normalized)
        features['volatility_20'] = returns.rolling(20).std()
        features['volatility_zscore'] = (
            features['volatility_20'] -
            features['volatility_20'].rolling(100).mean()
        ) / (features['volatility_20'].rolling(100).std() + 1e-8)

        # Volume features (if available)
        if 'volume' in df.columns:
            volume = df['volume']
            features['volume_norm'] = volume / volume.rolling(20).mean() - 1

        # Range features
        if 'high' in df.columns and 'low' in df.columns:
            features['range_pct'] = (df['high'] - df['low']) / close

        return features.fillna(0)


def generate_deep_learning_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate deep learning features.

    Citations:
    - Lim et al. (2021) - Temporal Fusion Transformer
    - Liu et al. (2024) - iTransformer
    - Wang et al. (2024) - TimeXer
    - Oreshkin et al. (2020) - N-BEATS

    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional arguments

    Returns:
        DataFrame with DL-ready features
    """
    generator = AcademicDeepLearningFeatures()
    return generator.generate_all(df)


# Module-level exports
__all__ = [
    # Configurations
    'TFTConfig',
    'iTransformerConfig',
    'TimeXerConfig',
    'NBEATSConfig',
    # Utilities
    'TimeSeriesLibraryModels',
    'DeepLearningFeatures',
    'AcademicDeepLearningFeatures',
    'generate_deep_learning_features',
]
