"""
Fast Feature Engine for Batch Training
=======================================
Vectorized feature generation - 10x faster than HFTFeatureEngine.
Uses pandas/numpy operations instead of tick-by-tick processing.

Target: Generate 100+ features in <30 seconds per 100k ticks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FastFeatureEngine:
    """
    Fast vectorized feature generation for batch training.

    Features categories:
    - Returns (multi-horizon)
    - Volatility (rolling)
    - Spread/Microstructure
    - Technical indicators
    - Statistical moments
    """

    def __init__(self):
        self.windows = [5, 10, 20, 50, 100, 200]
        self.return_lags = [1, 2, 3, 5, 10, 20, 50]

    def process(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Generate all features for a dataframe.

        Args:
            df: DataFrame with columns: timestamp, bid, ask (or close)
            symbol: Optional symbol name

        Returns:
            DataFrame with all features added
        """
        result = df.copy()

        # Ensure we have mid price
        if 'bid' in result.columns and 'ask' in result.columns:
            result['mid'] = (result['bid'] + result['ask']) / 2
            result['spread'] = result['ask'] - result['bid']
            result['spread_bps'] = result['spread'] / result['mid'] * 10000
        elif 'close' in result.columns:
            result['mid'] = result['close']
            result['spread'] = 0.0
            result['spread_bps'] = 0.0
        else:
            raise ValueError("Need bid/ask or close columns")

        # Returns features
        self._add_returns(result)

        # Volatility features
        self._add_volatility(result)

        # Spread/microstructure features
        self._add_microstructure(result)

        # Technical indicators
        self._add_technical(result)

        # Statistical features
        self._add_statistical(result)

        # Cross-sectional features
        self._add_cross_sectional(result)

        # Drop intermediate columns
        result = result.drop(columns=['mid'], errors='ignore')

        # Count features
        exclude = ['timestamp', 'bid', 'ask', 'volume', 'close', 'open', 'high', 'low', 'spread']
        feature_cols = [c for c in result.columns
                       if not c.startswith('target_')
                       and c not in exclude]

        logger.info(f"Generated {len(feature_cols)} features for {len(result):,} samples")

        return result

    def _add_returns(self, df: pd.DataFrame):
        """Add return-based features."""
        mid = df['mid']

        # Log returns at various lags
        for lag in self.return_lags:
            df[f'ret_{lag}'] = np.log(mid / mid.shift(lag)) * 10000  # bps

        # Cumulative returns
        df['ret_cum_5'] = df['ret_1'].rolling(5).sum()
        df['ret_cum_10'] = df['ret_1'].rolling(10).sum()
        df['ret_cum_20'] = df['ret_1'].rolling(20).sum()
        df['ret_cum_50'] = df['ret_1'].rolling(50).sum()

        # Return acceleration
        df['ret_accel'] = df['ret_1'] - df['ret_1'].shift(1)
        df['ret_accel_5'] = df['ret_5'] - df['ret_5'].shift(5)

    def _add_volatility(self, df: pd.DataFrame):
        """Add volatility features."""
        ret = df['ret_1']

        for w in self.windows:
            # Standard deviation
            df[f'vol_{w}'] = ret.rolling(w).std()

            # Realized variance
            df[f'rvar_{w}'] = (ret ** 2).rolling(w).sum()

            # High-low range if available
            if 'high' in df.columns and 'low' in df.columns:
                df[f'range_{w}'] = (df['high'].rolling(w).max() -
                                   df['low'].rolling(w).min()) / df['mid'] * 10000

        # Volatility ratios
        df['vol_ratio_5_20'] = df['vol_5'] / df['vol_20'].replace(0, np.nan)
        df['vol_ratio_10_50'] = df['vol_10'] / df['vol_50'].replace(0, np.nan)
        df['vol_ratio_20_100'] = df['vol_20'] / df['vol_100'].replace(0, np.nan)

        # Volatility z-score
        vol_mean = df['vol_20'].rolling(100).mean()
        vol_std = df['vol_20'].rolling(100).std()
        df['vol_zscore'] = (df['vol_20'] - vol_mean) / vol_std.replace(0, np.nan)

    def _add_microstructure(self, df: pd.DataFrame):
        """Add microstructure/spread features."""
        spread = df['spread_bps']

        # Rolling spread stats
        for w in [10, 20, 50, 100]:
            df[f'spread_mean_{w}'] = spread.rolling(w).mean()
            df[f'spread_std_{w}'] = spread.rolling(w).std()

        # Spread z-score
        df['spread_zscore'] = ((spread - df['spread_mean_50']) /
                              df['spread_std_50'].replace(0, np.nan))

        # Spread change
        df['spread_change'] = spread - spread.shift(1)
        df['spread_change_pct'] = df['spread_change'] / spread.shift(1).replace(0, np.nan)

    def _add_technical(self, df: pd.DataFrame):
        """Add technical indicator features."""
        mid = df['mid']

        # Moving averages
        for w in self.windows:
            df[f'sma_{w}'] = mid.rolling(w).mean()
            df[f'ema_{w}'] = mid.ewm(span=w, adjust=False).mean()

        # MA crossovers (as difference)
        df['ma_cross_5_20'] = (df['sma_5'] - df['sma_20']) / df['sma_20'] * 10000
        df['ma_cross_10_50'] = (df['sma_10'] - df['sma_50']) / df['sma_50'] * 10000
        df['ma_cross_20_100'] = (df['sma_20'] - df['sma_100']) / df['sma_100'] * 10000

        # Price position relative to MA
        for w in [20, 50, 100]:
            df[f'price_vs_sma_{w}'] = (mid - df[f'sma_{w}']) / df[f'sma_{w}'] * 10000

        # RSI approximation (simple momentum)
        delta = mid.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        for w in [10, 20, 50]:
            avg_gain = gain.rolling(w).mean()
            avg_loss = loss.rolling(w).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f'rsi_{w}'] = 100 - (100 / (1 + rs))

        # Bollinger bands position
        for w in [20, 50]:
            bb_mid = df[f'sma_{w}']
            bb_std = mid.rolling(w).std()
            df[f'bb_pos_{w}'] = (mid - bb_mid) / (2 * bb_std).replace(0, np.nan)

        # MACD-like
        df['macd'] = df['ema_10'] - df['ema_20']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Drop intermediate MA columns
        for w in self.windows:
            df.drop(columns=[f'sma_{w}', f'ema_{w}'], inplace=True, errors='ignore')

    def _add_statistical(self, df: pd.DataFrame):
        """Add statistical moment features."""
        ret = df['ret_1']

        for w in [20, 50, 100]:
            # Skewness
            mean = ret.rolling(w).mean()
            std = ret.rolling(w).std()
            skew = ((ret - mean) ** 3).rolling(w).mean() / (std ** 3).replace(0, np.nan)
            df[f'skew_{w}'] = skew

            # Kurtosis (excess)
            kurt = ((ret - mean) ** 4).rolling(w).mean() / (std ** 4).replace(0, np.nan) - 3
            df[f'kurt_{w}'] = kurt

        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = ret.rolling(50).apply(
                lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False
            )

    def _add_cross_sectional(self, df: pd.DataFrame):
        """Add cross-sectional/relative features."""
        mid = df['mid']
        ret = df['ret_1']

        # Z-scores of returns
        for w in [20, 50, 100]:
            mean = ret.rolling(w).mean()
            std = ret.rolling(w).std()
            df[f'ret_zscore_{w}'] = (ret - mean) / std.replace(0, np.nan)

        # Price percentile in range
        for w in [50, 100, 200]:
            roll_min = mid.rolling(w).min()
            roll_max = mid.rolling(w).max()
            df[f'price_pctl_{w}'] = (mid - roll_min) / (roll_max - roll_min).replace(0, np.nan)

        # Momentum percentile
        for w in [20, 50]:
            ret_rank = ret.rolling(w).rank(pct=True)
            df[f'mom_pctl_{w}'] = ret_rank


def create_targets(df: pd.DataFrame, horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Create prediction targets.

    Args:
        df: DataFrame with mid price
        horizons: List of forward horizons in ticks

    Returns:
        DataFrame with target columns added
    """
    result = df.copy()

    # Get mid price
    if 'mid' in result.columns:
        mid = result['mid']
    elif 'bid' in result.columns and 'ask' in result.columns:
        mid = (result['bid'] + result['ask']) / 2
    elif 'close' in result.columns:
        mid = result['close']
    else:
        raise ValueError("Need price columns")

    for h in horizons:
        future = mid.shift(-h)
        returns = (np.log(future) - np.log(mid)) * 10000  # bps
        result[f'target_return_{h}'] = returns
        result[f'target_direction_{h}'] = (returns > 0).astype(int)

    return result


def batch_process_symbol(symbol: str, data_dir: str = "data/dukascopy") -> Optional[pd.DataFrame]:
    """
    Load and process a symbol's data.

    Args:
        symbol: Currency pair symbol
        data_dir: Directory with CSV files

    Returns:
        Processed DataFrame or None if failed
    """
    from pathlib import Path

    data_path = Path(data_dir)
    files = sorted(data_path.glob(f"{symbol}_*.csv"))

    if not files:
        logger.warning(f"No data files for {symbol}")
        return None

    # Load all files
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]

            # Handle timestamp
            for col in ['timestamp', 'time', 'datetime']:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col])
                    if col != 'timestamp':
                        df = df.drop(columns=[col])
                    break

            frames.append(df)
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")

    if not frames:
        return None

    # Combine and sort
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(combined):,} ticks for {symbol}")

    # Generate features
    engine = FastFeatureEngine()
    featured = engine.process(combined, symbol)

    # Create targets
    with_targets = create_targets(featured)

    # Drop rows with NaN in targets
    target_cols = [c for c in with_targets.columns if c.startswith('target_')]
    with_targets = with_targets.dropna(subset=target_cols)

    logger.info(f"Final dataset: {len(with_targets):,} samples")

    return with_targets


if __name__ == "__main__":
    # Test with EURUSD
    import time

    start = time.time()
    df = batch_process_symbol("EURUSD")
    elapsed = time.time() - start

    if df is not None:
        print(f"\nProcessed {len(df):,} samples in {elapsed:.1f}s")

        # Show feature columns
        exclude = ['timestamp', 'bid', 'ask', 'volume', 'close', 'open', 'high', 'low']
        features = [c for c in df.columns
                   if not c.startswith('target_') and c not in exclude]
        print(f"Features: {len(features)}")
        print(f"Sample features: {features[:20]}")
