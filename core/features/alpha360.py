"""
Alpha360 - Microsoft Qlib 360 Raw Price Features
=================================================
Source: Microsoft Research Asia - Qlib
Official: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

360 features from 60-day normalized OHLCV lookback:
- CLOSE0-CLOSE59 (60 factors)
- OPEN0-OPEN59 (60 factors)
- HIGH0-HIGH59 (60 factors)
- LOW0-LOW59 (60 factors)
- VWAP0-VWAP59 (60 factors)
- VOLUME0-VOLUME59 (60 factors)

All prices normalized by current close.
All volumes normalized by current volume.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


class Alpha360:
    """
    Microsoft Qlib Alpha360 - 360 Raw Price Features.

    60-day lookback of normalized OHLCV data.
    No feature engineering - raw temporal patterns for deep learning.

    Usage:
        alpha360 = Alpha360()
        features = alpha360.generate_all(ohlcv_df)
    """

    def __init__(self, lookback: int = 60):
        """
        Initialize Alpha360.

        Args:
            lookback: Number of days to look back (default: 60)
        """
        self.lookback = lookback

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 360 Alpha360 factors.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Optional: vwap (will be calculated if missing)

        Returns:
            DataFrame with 360 factor columns
        """
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame(index=df.index)

        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # VWAP (calculate if not present)
        if 'vwap' in df.columns:
            vwap = df['vwap']
        else:
            # Approximate VWAP as (high + low + close) / 3
            vwap = (high + low + close) / 3

        # Generate CLOSE features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'CLOSE{i}'] = close.shift(i) / (close + 1e-12)
        features['CLOSE0'] = close / (close + 1e-12)  # Always 1

        # Generate OPEN features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'OPEN{i}'] = open_.shift(i) / (close + 1e-12)
        features['OPEN0'] = open_ / (close + 1e-12)

        # Generate HIGH features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'HIGH{i}'] = high.shift(i) / (close + 1e-12)
        features['HIGH0'] = high / (close + 1e-12)

        # Generate LOW features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'LOW{i}'] = low.shift(i) / (close + 1e-12)
        features['LOW0'] = low / (close + 1e-12)

        # Generate VWAP features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'VWAP{i}'] = vwap.shift(i) / (close + 1e-12)
        features['VWAP0'] = vwap / (close + 1e-12)

        # Generate VOLUME features (60)
        for i in range(self.lookback - 1, 0, -1):
            features[f'VOLUME{i}'] = volume.shift(i) / (volume + 1e-12)
        features['VOLUME0'] = volume / (volume + 1e-12)  # Always 1

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # Reorder columns properly
        ordered_cols = []
        for field in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VWAP', 'VOLUME']:
            for i in range(self.lookback - 1, -1, -1):
                ordered_cols.append(f'{field}{i}')

        features = features[ordered_cols]

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []
        for field in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VWAP', 'VOLUME']:
            for i in range(self.lookback - 1, -1, -1):
                names.append(f'{field}{i}')
        return names


class Alpha360Compact:
    """
    Compact version of Alpha360 for HFT with shorter lookback.

    Default: 20-day lookback = 120 features (vs 360 for 60-day)
    """

    def __init__(self, lookback: int = 20):
        """
        Initialize compact Alpha360.

        Args:
            lookback: Number of periods to look back (default: 20)
        """
        self.lookback = lookback

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate compact Alpha360 features."""
        alpha = Alpha360(lookback=self.lookback)
        return alpha.generate_all(df)

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        for field in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VWAP', 'VOLUME']:
            for i in range(self.lookback - 1, -1, -1):
                names.append(f'{field}{i}')
        return names


# Convenience functions
def generate_alpha360(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Generate Alpha360 factors.

    Args:
        df: OHLCV DataFrame
        lookback: Days to look back (default: 60)

    Returns:
        DataFrame with 360 factors (6 fields x 60 days)
    """
    alpha = Alpha360(lookback=lookback)
    return alpha.generate_all(df)


def generate_alpha360_compact(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Generate compact Alpha360 for HFT.

    Args:
        df: OHLCV DataFrame
        lookback: Periods to look back (default: 20)

    Returns:
        DataFrame with 120 factors (6 fields x 20 periods)
    """
    alpha = Alpha360Compact(lookback=lookback)
    return alpha.generate_all(df)
