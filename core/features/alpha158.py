"""
Alpha158 - Microsoft Qlib 158 Technical Factors
================================================
Source: Microsoft Research Asia - Qlib
Official: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

158 handcrafted technical indicators organized into:
- KBAR features (9 factors)
- Price features (20 factors)
- Volume features (5 factors)
- Rolling statistics (124 factors across 5 windows)

Windows: [5, 10, 20, 30, 60] days
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class Alpha158:
    """
    Microsoft Qlib Alpha158 - 158 Technical Factors.

    Usage:
        alpha158 = Alpha158()
        features = alpha158.generate_all(ohlcv_df)
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize Alpha158.

        Args:
            windows: Rolling windows for technical indicators.
                     Default: [5, 10, 20, 30, 60]
        """
        self.windows = windows or [5, 10, 20, 30, 60]

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    @staticmethod
    def _ref(series: pd.Series, n: int) -> pd.Series:
        """Reference value n periods ago."""
        return series.shift(n)

    @staticmethod
    def _delta(series: pd.Series, n: int) -> pd.Series:
        """Difference from n periods ago."""
        return series.diff(n)

    @staticmethod
    def _mean(series: pd.Series, n: int) -> pd.Series:
        """Rolling mean."""
        return series.rolling(n, min_periods=1).mean()

    @staticmethod
    def _std(series: pd.Series, n: int) -> pd.Series:
        """Rolling standard deviation."""
        return series.rolling(n, min_periods=2).std()

    @staticmethod
    def _max(series: pd.Series, n: int) -> pd.Series:
        """Rolling maximum."""
        return series.rolling(n, min_periods=1).max()

    @staticmethod
    def _min(series: pd.Series, n: int) -> pd.Series:
        """Rolling minimum."""
        return series.rolling(n, min_periods=1).min()

    @staticmethod
    def _sum(series: pd.Series, n: int) -> pd.Series:
        """Rolling sum."""
        return series.rolling(n, min_periods=1).sum()

    @staticmethod
    def _corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """Rolling correlation."""
        return x.rolling(n, min_periods=2).corr(y)

    @staticmethod
    def _rank(series: pd.Series, n: int) -> pd.Series:
        """Rolling rank (percentile position)."""
        def rank_pct(arr):
            if len(arr) < 2:
                return 0.5
            return (arr[-1] > arr[:-1]).sum() / (len(arr) - 1)
        return series.rolling(n, min_periods=2).apply(rank_pct, raw=True)

    @staticmethod
    def _slope(series: pd.Series, n: int) -> pd.Series:
        """Rolling slope (linear regression)."""
        def calc_slope(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]
        return series.rolling(n, min_periods=2).apply(calc_slope, raw=True)

    @staticmethod
    def _rsquare(series: pd.Series, n: int) -> pd.Series:
        """Rolling R-squared of linear regression."""
        def calc_rsq(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            corr = np.corrcoef(x, arr)[0, 1]
            return corr ** 2 if not np.isnan(corr) else 0
        return series.rolling(n, min_periods=2).apply(calc_rsq, raw=True)

    @staticmethod
    def _resi(series: pd.Series, n: int) -> pd.Series:
        """Rolling residual from linear regression."""
        def calc_resi(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            coef = np.polyfit(x, arr, 1)
            pred = coef[0] * (len(arr) - 1) + coef[1]
            return arr[-1] - pred
        return series.rolling(n, min_periods=2).apply(calc_resi, raw=True)

    @staticmethod
    def _quantile(series: pd.Series, n: int, q: float) -> pd.Series:
        """Rolling quantile."""
        return series.rolling(n, min_periods=1).quantile(q)

    @staticmethod
    def _idxmax(series: pd.Series, n: int) -> pd.Series:
        """Days since rolling max."""
        def idx_max(arr):
            return (len(arr) - 1 - np.argmax(arr)) / len(arr)
        return series.rolling(n, min_periods=1).apply(idx_max, raw=True)

    @staticmethod
    def _idxmin(series: pd.Series, n: int) -> pd.Series:
        """Days since rolling min."""
        def idx_min(arr):
            return (len(arr) - 1 - np.argmin(arr)) / len(arr)
        return series.rolling(n, min_periods=1).apply(idx_min, raw=True)

    # =========================================================================
    # KBAR FEATURES (9 factors)
    # =========================================================================

    def _kbar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate KBAR features from OHLC data."""
        o, h, l, c = df['open'], df['high'], df['low'], df['close']

        features = pd.DataFrame(index=df.index)

        # KMID: (close-open)/open
        features['KMID'] = (c - o) / (o + 1e-12)

        # KLEN: (high-low)/open
        features['KLEN'] = (h - l) / (o + 1e-12)

        # KMID2: (close-open)/(high-low)
        features['KMID2'] = (c - o) / (h - l + 1e-12)

        # KUP: (high-max(open,close))/open
        features['KUP'] = (h - np.maximum(o, c)) / (o + 1e-12)

        # KUP2: (high-max(open,close))/(high-low)
        features['KUP2'] = (h - np.maximum(o, c)) / (h - l + 1e-12)

        # KLOW: (min(open,close)-low)/open
        features['KLOW'] = (np.minimum(o, c) - l) / (o + 1e-12)

        # KLOW2: (min(open,close)-low)/(high-low)
        features['KLOW2'] = (np.minimum(o, c) - l) / (h - l + 1e-12)

        # KSFT: (2*close-high-low)/open
        features['KSFT'] = (2 * c - h - l) / (o + 1e-12)

        # KSFT2: (2*close-high-low)/(high-low)
        features['KSFT2'] = (2 * c - h - l) / (h - l + 1e-12)

        return features

    # =========================================================================
    # PRICE FEATURES (20 factors: 4 prices x 5 windows)
    # =========================================================================

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate normalized price features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        for field in ['open', 'high', 'low', 'close']:
            for i in range(5):  # Windows 0-4
                if i == 0:
                    features[f'{field.upper()}0'] = df[field] / (close + 1e-12)
                else:
                    features[f'{field.upper()}{i}'] = self._ref(df[field], i) / (close + 1e-12)

        return features

    # =========================================================================
    # VOLUME FEATURES (5 factors)
    # =========================================================================

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate normalized volume features."""
        features = pd.DataFrame(index=df.index)
        vol = df['volume']

        for i in range(5):  # Windows 0-4
            if i == 0:
                features[f'VOLUME0'] = vol / (vol + 1e-12)
            else:
                features[f'VOLUME{i}'] = self._ref(vol, i) / (vol + 1e-12)

        return features

    # =========================================================================
    # ROLLING FEATURES (124 factors: 28 operators x ~5 windows)
    # =========================================================================

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        for d in self.windows:
            # ROC: Rate of change
            features[f'ROC{d}'] = self._ref(close, d) / (close + 1e-12)

            # MA: Moving average
            features[f'MA{d}'] = self._mean(close, d) / (close + 1e-12)

            # STD: Standard deviation
            features[f'STD{d}'] = self._std(close, d) / (close + 1e-12)

            # BETA: Slope
            features[f'BETA{d}'] = self._slope(close, d) / (close + 1e-12)

            # RSQR: R-squared
            features[f'RSQR{d}'] = self._rsquare(close, d)

            # RESI: Residual
            features[f'RESI{d}'] = self._resi(close, d) / (close + 1e-12)

            # MAX: Rolling max
            features[f'MAX{d}'] = self._max(high, d) / (close + 1e-12)

            # MIN: Rolling min
            features[f'MIN{d}'] = self._min(low, d) / (close + 1e-12)

            # QTLU: 80th percentile
            features[f'QTLU{d}'] = self._quantile(close, d, 0.8) / (close + 1e-12)

            # QTLD: 20th percentile
            features[f'QTLD{d}'] = self._quantile(close, d, 0.2) / (close + 1e-12)

            # RANK: Percentile rank
            features[f'RANK{d}'] = self._rank(close, d)

            # RSV: Raw stochastic value
            max_high = self._max(high, d)
            min_low = self._min(low, d)
            features[f'RSV{d}'] = (close - min_low) / (max_high - min_low + 1e-12)

            # IMAX: Days since max
            features[f'IMAX{d}'] = self._idxmax(high, d)

            # IMIN: Days since min
            features[f'IMIN{d}'] = self._idxmin(low, d)

            # IMXD: Difference between max and min positions
            features[f'IMXD{d}'] = features[f'IMAX{d}'] - features[f'IMIN{d}']

            # CORR: Correlation between close and log volume
            features[f'CORR{d}'] = self._corr(close, np.log(volume + 1), d)

            # CORD: Correlation of returns and volume changes
            ret = close / self._ref(close, 1) - 1
            vol_ret = np.log(volume / self._ref(volume, 1) + 1)
            features[f'CORD{d}'] = self._corr(ret, vol_ret, d)

            # CNTP: Percentage of up days
            up = (close > self._ref(close, 1)).astype(float)
            features[f'CNTP{d}'] = self._mean(up, d)

            # CNTN: Percentage of down days
            down = (close < self._ref(close, 1)).astype(float)
            features[f'CNTN{d}'] = self._mean(down, d)

            # CNTD: Up days minus down days percentage
            features[f'CNTD{d}'] = features[f'CNTP{d}'] - features[f'CNTN{d}']

            # SUMP: Sum of gains / total absolute change (RSI-like)
            gains = np.maximum(close - self._ref(close, 1), 0)
            abs_change = np.abs(close - self._ref(close, 1))
            features[f'SUMP{d}'] = self._sum(gains, d) / (self._sum(abs_change, d) + 1e-12)

            # SUMN: Sum of losses / total absolute change
            losses = np.maximum(self._ref(close, 1) - close, 0)
            features[f'SUMN{d}'] = self._sum(losses, d) / (self._sum(abs_change, d) + 1e-12)

            # SUMD: Difference (similar to RSI)
            features[f'SUMD{d}'] = features[f'SUMP{d}'] - features[f'SUMN{d}']

            # VMA: Volume moving average
            features[f'VMA{d}'] = self._mean(volume, d) / (volume + 1e-12)

            # VSTD: Volume standard deviation
            features[f'VSTD{d}'] = self._std(volume, d) / (volume + 1e-12)

            # WVMA: Weighted volume-price volatility
            weighted = np.abs(ret) * volume
            features[f'WVMA{d}'] = self._std(weighted, d) / (self._mean(weighted, d) + 1e-12)

            # VSUMP: Volume increase ratio
            vol_gains = np.maximum(volume - self._ref(volume, 1), 0)
            vol_abs = np.abs(volume - self._ref(volume, 1))
            features[f'VSUMP{d}'] = self._sum(vol_gains, d) / (self._sum(vol_abs, d) + 1e-12)

            # VSUMN: Volume decrease ratio
            vol_losses = np.maximum(self._ref(volume, 1) - volume, 0)
            features[f'VSUMN{d}'] = self._sum(vol_losses, d) / (self._sum(vol_abs, d) + 1e-12)

            # VSUMD: Volume RSI
            features[f'VSUMD{d}'] = features[f'VSUMP{d}'] - features[f'VSUMN{d}']

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 158 Alpha158 factors.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 158 factor columns
        """
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Generate all feature groups
        kbar = self._kbar_features(df)
        price = self._price_features(df)
        volume = self._volume_features(df)
        rolling = self._rolling_features(df)

        # Combine all features
        features = pd.concat([kbar, price, volume, rolling], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []

        # KBAR (9)
        names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2',
                      'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

        # Price (20)
        for field in ['OPEN', 'HIGH', 'LOW', 'CLOSE']:
            for i in range(5):
                names.append(f'{field}{i}')

        # Volume (5)
        for i in range(5):
            names.append(f'VOLUME{i}')

        # Rolling (28 operators x 5 windows = 140)
        operators = ['ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN',
                    'QTLU', 'QTLD', 'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD',
                    'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN',
                    'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD']

        for op in operators:
            for d in self.windows:
                names.append(f'{op}{d}')

        return names


# Convenience function
def generate_alpha158(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Generate Alpha158 factors.

    Args:
        df: OHLCV DataFrame
        windows: Rolling windows (default: [5, 10, 20, 30, 60])

    Returns:
        DataFrame with 158 factors
    """
    alpha = Alpha158(windows=windows)
    return alpha.generate_all(df)
