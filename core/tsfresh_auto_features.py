"""
TSFresh Automatic Feature Extraction
=====================================
Source:
- tsfresh library (https://tsfresh.readthedocs.io/)
- 知乎: 量化小白也能自动化挖掘出6万+因子

Key Innovation:
- Transforms 84 base features into 65,000+ factors
- Automatic feature selection using FRESH algorithm
- Statistical significance filtering

For Forex HFT:
- Extract microstructure patterns automatically
- Discover hidden relationships in tick data
- Generate massive feature space for ML

Usage:
    extractor = TSFreshExtractor()
    features = extractor.extract(df, target)
    # Returns DataFrame with 1000s of features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Check for tsfresh
try:
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import roll_time_series, impute
    from tsfresh.feature_extraction import (
        ComprehensiveFCParameters,
        MinimalFCParameters,
        EfficientFCParameters
    )
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False
    logger.warning("tsfresh not available - install with: pip install tsfresh")


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    # Rolling window parameters
    min_timeshift: int = 5
    max_timeshift: int = 20

    # Feature complexity
    extraction_mode: str = 'efficient'  # 'minimal', 'efficient', 'comprehensive'

    # Feature selection
    fdr_level: float = 0.05  # False discovery rate
    n_jobs: int = -1  # Parallel jobs

    # Memory management
    chunksize: int = 10000


class TSFreshExtractor:
    """
    TSFresh-based automatic feature extraction.

    Source: 知乎 "量化小白也能自动化挖掘出6万+因子"

    Modes:
    - minimal: ~10 features per time series (fast)
    - efficient: ~100 features per time series (balanced)
    - comprehensive: ~750 features per time series (slow but thorough)
    """

    def __init__(self, config: FeatureExtractionConfig = None):
        self.config = config or FeatureExtractionConfig()
        self.selected_features: List[str] = []
        self.feature_params = None

    def _get_fc_parameters(self):
        """Get feature calculation parameters based on mode."""
        if not HAS_TSFRESH:
            return None

        mode = self.config.extraction_mode

        if mode == 'minimal':
            return MinimalFCParameters()
        elif mode == 'efficient':
            return EfficientFCParameters()
        elif mode == 'comprehensive':
            return ComprehensiveFCParameters()
        else:
            return EfficientFCParameters()

    def prepare_data(
        self,
        df: pd.DataFrame,
        id_col: str = None,
        time_col: str = 'timestamp',
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare data in tsfresh long format.

        Args:
            df: Input DataFrame
            id_col: Column identifying different time series (optional)
            time_col: Time column name
            value_cols: Columns to extract features from

        Returns:
            DataFrame in tsfresh format
        """
        if value_cols is None:
            value_cols = [c for c in df.columns if c not in [time_col, id_col, 'id']]

        # Create id column if not present
        if id_col is None:
            df = df.copy()
            df['id'] = 0
            id_col = 'id'

        # Melt to long format
        long_df = df.melt(
            id_vars=[id_col, time_col] if time_col in df.columns else [id_col],
            value_vars=value_cols,
            var_name='kind',
            value_name='value'
        )

        return long_df

    def extract(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        value_cols: Optional[List[str]] = None,
        use_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from time series data.

        Args:
            df: Input DataFrame with time series columns
            target: Target variable for feature selection (optional)
            value_cols: Columns to extract features from
            use_rolling: Whether to use rolling windows

        Returns:
            DataFrame with extracted features
        """
        if not HAS_TSFRESH:
            logger.warning("tsfresh not available, using fallback")
            return self._extract_fallback(df, value_cols)

        if value_cols is None:
            value_cols = [c for c in df.columns if c in
                         ['open', 'high', 'low', 'close', 'volume', 'returns', 'spread']]

        if not value_cols:
            logger.warning("No value columns found")
            return pd.DataFrame()

        logger.info(f"Extracting features from {len(value_cols)} columns using {self.config.extraction_mode} mode")

        # Prepare data
        df_prepared = df[value_cols].copy()
        df_prepared = df_prepared.reset_index(drop=True)

        # Add id and time columns
        df_prepared['id'] = 0
        df_prepared['time'] = range(len(df_prepared))

        if use_rolling:
            # Roll time series for sliding window features
            logger.info(f"Rolling time series with window [{self.config.min_timeshift}, {self.config.max_timeshift}]")

            try:
                df_rolled = roll_time_series(
                    df_prepared,
                    column_id='id',
                    column_sort='time',
                    max_timeshift=self.config.max_timeshift,
                    min_timeshift=self.config.min_timeshift
                )
            except Exception as e:
                logger.warning(f"Rolling failed: {e}, using non-rolled")
                df_rolled = df_prepared

        else:
            df_rolled = df_prepared

        # Get feature calculation parameters
        fc_params = self._get_fc_parameters()

        # Extract features
        logger.info("Extracting features...")

        try:
            features = extract_features(
                df_rolled,
                column_id='id',
                column_sort='time',
                default_fc_parameters=fc_params,
                n_jobs=self.config.n_jobs,
                disable_progressbar=False,
                chunksize=self.config.chunksize
            )

            # Impute missing values
            impute(features)

            logger.info(f"Extracted {features.shape[1]} features")

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._extract_fallback(df, value_cols)

        # Feature selection if target provided
        if target is not None:
            logger.info("Selecting relevant features...")

            try:
                # Align target with features
                target_aligned = target.iloc[:len(features)]
                target_aligned = target_aligned[~target_aligned.isna()]

                features_aligned = features.iloc[:len(target_aligned)]

                selected = select_features(
                    features_aligned,
                    target_aligned,
                    fdr_level=self.config.fdr_level
                )

                self.selected_features = selected.columns.tolist()
                logger.info(f"Selected {len(self.selected_features)} relevant features")

                return selected

            except Exception as e:
                logger.warning(f"Feature selection failed: {e}")

        return features

    def _extract_fallback(
        self,
        df: pd.DataFrame,
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fallback feature extraction without tsfresh."""
        logger.info("Using fallback feature extraction")

        if value_cols is None:
            value_cols = [c for c in df.columns if c in
                         ['open', 'high', 'low', 'close', 'volume', 'returns', 'spread']]

        features = pd.DataFrame(index=df.index)

        for col in value_cols:
            if col not in df.columns:
                continue

            x = df[col].values

            # Basic statistics
            for window in [5, 10, 20, 50]:
                if len(x) > window:
                    features[f'{col}__mean_{window}'] = pd.Series(x).rolling(window).mean().values
                    features[f'{col}__std_{window}'] = pd.Series(x).rolling(window).std().values
                    features[f'{col}__min_{window}'] = pd.Series(x).rolling(window).min().values
                    features[f'{col}__max_{window}'] = pd.Series(x).rolling(window).max().values
                    features[f'{col}__median_{window}'] = pd.Series(x).rolling(window).median().values

                    # Quantiles
                    features[f'{col}__q25_{window}'] = pd.Series(x).rolling(window).quantile(0.25).values
                    features[f'{col}__q75_{window}'] = pd.Series(x).rolling(window).quantile(0.75).values

            # Differences
            for lag in [1, 5, 10]:
                if len(x) > lag:
                    features[f'{col}__diff_{lag}'] = pd.Series(x).diff(lag).values
                    features[f'{col}__pct_change_{lag}'] = pd.Series(x).pct_change(lag).values

            # Autocorrelation
            for lag in [1, 5, 10]:
                if len(x) > lag + 20:
                    autocorr = pd.Series(x).autocorr(lag)
                    features[f'{col}__autocorr_{lag}'] = autocorr

            # Skewness and Kurtosis
            for window in [20, 50]:
                if len(x) > window:
                    features[f'{col}__skew_{window}'] = pd.Series(x).rolling(window).skew().values
                    features[f'{col}__kurt_{window}'] = pd.Series(x).rolling(window).kurt().values

        logger.info(f"Fallback extracted {features.shape[1]} features")

        return features

    def transform(
        self,
        df: pd.DataFrame,
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform new data using selected features.

        Args:
            df: New data to transform
            value_cols: Value columns

        Returns:
            DataFrame with selected features
        """
        features = self.extract(df, value_cols=value_cols, use_rolling=False)

        if self.selected_features:
            available = [f for f in self.selected_features if f in features.columns]
            return features[available]

        return features


class QuickFeatureExtractor:
    """
    Quick feature extraction for HFT.
    Optimized for speed over comprehensiveness.
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20, 50]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract quick features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        # Price columns
        price_cols = [c for c in df.columns if c in ['open', 'high', 'low', 'close', 'mid', 'price']]

        for col in price_cols:
            x = df[col].values

            for w in self.windows:
                # Rolling statistics
                s = pd.Series(x)
                features[f'{col}_mean_{w}'] = s.rolling(w).mean().values
                features[f'{col}_std_{w}'] = s.rolling(w).std().values
                features[f'{col}_zscore_{w}'] = ((s - s.rolling(w).mean()) / s.rolling(w).std()).values

                # Range position
                roll_max = s.rolling(w).max()
                roll_min = s.rolling(w).min()
                range_ = roll_max - roll_min
                features[f'{col}_range_pos_{w}'] = ((s - roll_min) / range_.replace(0, np.nan)).values

                # Momentum
                features[f'{col}_mom_{w}'] = (s / s.shift(w) - 1).values * 10000

        # Volume features
        if 'volume' in df.columns:
            v = pd.Series(df['volume'].values)
            for w in self.windows:
                features[f'volume_mean_{w}'] = v.rolling(w).mean().values
                features[f'volume_ratio_{w}'] = (v / v.rolling(w).mean()).values

        # Returns
        if 'close' in df.columns:
            ret = pd.Series(df['close'].values).pct_change()
            for w in self.windows:
                features[f'ret_mean_{w}'] = ret.rolling(w).mean().values * 10000
                features[f'ret_std_{w}'] = ret.rolling(w).std().values * 10000
                features[f'ret_skew_{w}'] = ret.rolling(w).skew().values
                features[f'ret_kurt_{w}'] = ret.rolling(w).kurt().values

        # Cross-column features
        if 'high' in df.columns and 'low' in df.columns:
            h = df['high'].values
            l = df['low'].values
            features['hl_range'] = (h - l)
            features['hl_range_pct'] = (h - l) / ((h + l) / 2) * 10000

        if 'open' in df.columns and 'close' in df.columns:
            o = df['open'].values
            c = df['close'].values
            features['oc_change'] = (c - o)
            features['oc_direction'] = np.sign(c - o)

        return features


class TSFreshFactorEngine:
    """
    Unified TSFresh Factor Engine.
    Wrapper for HFT Feature Engine integration.
    """

    def __init__(self, mode: str = 'quick'):
        """
        Initialize engine.

        Args:
            mode: 'quick', 'efficient', or 'comprehensive'
        """
        self.mode = mode

        if mode == 'quick':
            self.extractor = QuickFeatureExtractor()
        else:
            config = FeatureExtractionConfig(extraction_mode=mode)
            self.extractor = TSFreshExtractor(config)

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all TSFresh features.
        Interface compatible with HFT Feature Engine.
        """
        return self.extractor.extract(df)


if __name__ == '__main__':
    print("TSFresh Auto Feature Extraction Test")
    print("=" * 50)

    # Generate test data
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=n, freq='1min'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.1) + np.abs(np.random.randn(n) * 0.1),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.1) - np.abs(np.random.randn(n) * 0.1),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.exponential(1000, n)
    })

    print(f"Input data: {df.shape}")

    # Test quick extractor
    print("\n--- Quick Feature Extraction ---")
    quick = QuickFeatureExtractor()
    quick_features = quick.extract(df)
    print(f"Quick features: {quick_features.shape[1]}")
    print(f"Sample features: {list(quick_features.columns[:10])}")

    # Test TSFresh extractor (fallback mode)
    print("\n--- TSFresh Feature Extraction ---")
    config = FeatureExtractionConfig(extraction_mode='efficient')
    extractor = TSFreshExtractor(config)
    tsfresh_features = extractor.extract(df, use_rolling=False)
    print(f"TSFresh features: {tsfresh_features.shape[1]}")

    print("\n" + "=" * 50)
    print("TSFresh Test Complete")

    if HAS_TSFRESH:
        print("tsfresh library: AVAILABLE")
    else:
        print("tsfresh library: NOT INSTALLED (using fallback)")
        print("Install with: pip install tsfresh")
