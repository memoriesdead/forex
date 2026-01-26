"""
Barra CNE6 - MSCI Risk Factor Model (Adapted for Forex)
========================================================
Source: MSCI Barra CNE6 Model (China A-shares, 2019)
Reference: 清华大学 Barra白皮书, MSCI Factor Models

Original CNE6 has 46 risk factors for equities:
- 9 Style factors (SIZE, BETA, MOMENTUM, etc.)
- 1 Market factor
- 36 Industry factors

Forex Adaptation:
Since forex doesn't have market cap, book value, or industries,
we adapt the concepts using analogous price/volume metrics:

- SIZE → Volatility magnitude (proxy for "size" of moves)
- BETA → Correlation to USD index (DXY proxy)
- MOMENTUM → Multi-period price momentum
- VOLATILITY → Realized and implied volatility
- LIQUIDITY → Volume-based metrics
- VALUE → Mean reversion signals
- GROWTH → Trend strength
- QUALITY → Signal consistency
- LEVERAGE → Position concentration
- SENTIMENT → Order flow imbalance
- DIVIDEND → Carry (interest rate differential proxy)

Total: 46 factors maintaining the spirit of Barra CNE6
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class BarraCNE6Forex:
    """
    Barra CNE6 Risk Factors Adapted for Forex.

    46 factors organized into:
    - Size factors (5)
    - Beta factors (4)
    - Momentum factors (6)
    - Volatility factors (6)
    - Liquidity factors (5)
    - Value factors (5)
    - Growth factors (4)
    - Quality factors (4)
    - Leverage factors (3)
    - Sentiment factors (4)

    Usage:
        barra = BarraCNE6Forex()
        features = barra.generate_all(ohlcv_df)
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize Barra CNE6 Forex.

        Args:
            windows: Rolling windows (default: [5, 10, 20, 60])
        """
        self.windows = windows or [5, 10, 20, 60]

    # =========================================================================
    # SIZE FACTORS (5) - Volatility magnitude as proxy for "size"
    # =========================================================================

    def _size_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SIZE factors - magnitude of price movements.
        In equities: market cap. In forex: volatility scale.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # LNCAP equivalent: Log of ATR (Average True Range)
        tr = np.maximum(high - low,
                       np.abs(high - close.shift(1)),
                       np.abs(low - close.shift(1)))
        atr_20 = tr.rolling(20, min_periods=1).mean()
        features['SIZE_LNCAP'] = np.log(atr_20 + 1e-12)

        # MIDCAP: Nonlinear size (squared deviation from mean)
        features['SIZE_MIDCAP'] = (atr_20 - atr_20.rolling(60, min_periods=1).mean()) ** 2

        # Size relative to long-term average
        atr_60 = tr.rolling(60, min_periods=1).mean()
        features['SIZE_RELATIVE'] = atr_20 / (atr_60 + 1e-12)

        # Log range as size proxy
        daily_range = (high - low) / (close + 1e-12)
        features['SIZE_RANGE'] = np.log(daily_range.rolling(20, min_periods=1).mean() + 1e-12)

        # Size percentile rank
        features['SIZE_RANK'] = atr_20.rolling(60, min_periods=1).apply(
            lambda x: (x[-1] > x[:-1]).sum() / len(x) if len(x) > 1 else 0.5, raw=True
        )

        return features

    # =========================================================================
    # BETA FACTORS (4) - Market sensitivity
    # =========================================================================

    def _beta_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BETA factors - sensitivity to market movements.
        Uses rolling correlation/regression with lagged returns.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # Historical beta: regression coefficient on market (use lagged self as proxy)
        market_proxy = returns.shift(1)  # Lagged return as market proxy

        for d in [20, 60]:
            # Rolling beta (correlation * vol ratio)
            corr = returns.rolling(d, min_periods=5).corr(market_proxy)
            vol_ret = returns.rolling(d, min_periods=2).std()
            vol_mkt = market_proxy.rolling(d, min_periods=2).std()
            features[f'BETA_HIST{d}'] = corr * (vol_ret / (vol_mkt + 1e-12))

        # Daily standard deviation of returns (DASTD)
        features['BETA_DASTD'] = returns.rolling(20, min_periods=2).std()

        # Cumulative range (CMRA)
        log_ret = np.log(close / close.shift(1) + 1e-12)
        features['BETA_CMRA'] = (
            log_ret.rolling(12, min_periods=1).max() -
            log_ret.rolling(12, min_periods=1).min()
        )

        return features

    # =========================================================================
    # MOMENTUM FACTORS (6) - Price trend persistence
    # =========================================================================

    def _momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MOMENTUM factors - trend strength and persistence.
        Classic Barra momentum uses 12-1 month return.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # RSTR: Relative strength (cumulative return)
        for d in [5, 10, 20, 60]:
            features[f'MOM_RSTR{d}'] = close / close.shift(d) - 1

        # Short-term reversal (STREV)
        features['MOM_STREV'] = close.shift(1) / close.shift(5) - 1

        # Momentum quality: consistency of direction
        returns = close.pct_change()
        up_ratio = (returns > 0).rolling(20, min_periods=1).mean()
        features['MOM_QUALITY'] = 2 * up_ratio - 1  # -1 to 1 scale

        return features

    # =========================================================================
    # VOLATILITY FACTORS (6) - Return dispersion
    # =========================================================================

    def _volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VOLATILITY factors - various volatility measures.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()

        # Realized volatility at different horizons
        for d in [5, 20, 60]:
            features[f'VOL_RVOL{d}'] = returns.rolling(d, min_periods=2).std()

        # Parkinson volatility (uses high-low)
        log_hl = np.log(high / low + 1e-12)
        features['VOL_PARKINSON'] = np.sqrt(
            (log_hl ** 2).rolling(20, min_periods=1).mean() / (4 * np.log(2))
        )

        # Volatility of volatility
        vol_20 = returns.rolling(20, min_periods=2).std()
        features['VOL_VOLVOL'] = vol_20.rolling(20, min_periods=2).std()

        # Volatility skew (asymmetry)
        pos_ret = returns.where(returns > 0, 0)
        neg_ret = returns.where(returns < 0, 0)
        up_vol = pos_ret.rolling(20, min_periods=2).std()
        down_vol = np.abs(neg_ret).rolling(20, min_periods=2).std()
        features['VOL_SKEW'] = (up_vol - down_vol) / (up_vol + down_vol + 1e-12)

        return features

    # =========================================================================
    # LIQUIDITY FACTORS (5) - Trading activity
    # =========================================================================

    def _liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LIQUIDITY factors - volume and turnover metrics.
        """
        features = pd.DataFrame(index=df.index)
        volume = df['volume']
        close = df['close']

        # Share turnover (STOM)
        for d in [5, 20, 60]:
            avg_vol = volume.rolling(d, min_periods=1).mean()
            features[f'LIQ_STOM{d}'] = np.log(avg_vol + 1)

        # Volume coefficient of variation
        vol_std = volume.rolling(20, min_periods=2).std()
        vol_mean = volume.rolling(20, min_periods=1).mean()
        features['LIQ_VOLCOV'] = vol_std / (vol_mean + 1e-12)

        # Amihud illiquidity (return/volume ratio)
        returns = close.pct_change()
        amihud = np.abs(returns) / (volume + 1e-12)
        features['LIQ_AMIHUD'] = -np.log(amihud.rolling(20, min_periods=1).mean() + 1e-12)

        return features

    # =========================================================================
    # VALUE FACTORS (5) - Mean reversion signals
    # =========================================================================

    def _value_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VALUE factors - mean reversion and valuation proxies.
        In equities: P/E, P/B. In forex: deviation from MA.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Book-to-price proxy: distance from moving average
        for d in [20, 60]:
            ma = close.rolling(d, min_periods=1).mean()
            features[f'VAL_BTOP{d}'] = (ma - close) / (close + 1e-12)

        # Z-score (standardized deviation)
        ma_60 = close.rolling(60, min_periods=1).mean()
        std_60 = close.rolling(60, min_periods=2).std()
        features['VAL_ZSCORE'] = (close - ma_60) / (std_60 + 1e-12)

        # Relative strength index as value indicator
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        features['VAL_RSI'] = (100 - (100 / (1 + rs))) / 100 - 0.5  # Centered

        # Bollinger position
        bb_upper = ma_60 + 2 * std_60
        bb_lower = ma_60 - 2 * std_60
        features['VAL_BBPOS'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)

        return features

    # =========================================================================
    # GROWTH FACTORS (4) - Trend strength
    # =========================================================================

    def _growth_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        GROWTH factors - trend strength and acceleration.
        In equities: earnings growth. In forex: price trend strength.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Earnings growth proxy: trend slope
        def calc_slope(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]

        for d in [20, 60]:
            features[f'GRO_SLOPE{d}'] = close.rolling(d, min_periods=2).apply(calc_slope, raw=True)

        # Sales growth proxy: rate of change acceleration
        roc_20 = close / close.shift(20) - 1
        features['GRO_ACC'] = roc_20 - roc_20.shift(20)

        # R-squared of trend (quality of fit)
        def calc_rsq(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            corr = np.corrcoef(x, arr)[0, 1]
            return corr ** 2 if not np.isnan(corr) else 0

        features['GRO_RSQ'] = close.rolling(20, min_periods=2).apply(calc_rsq, raw=True)

        return features

    # =========================================================================
    # QUALITY FACTORS (4) - Signal consistency
    # =========================================================================

    def _quality_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        QUALITY factors - consistency and reliability.
        In equities: ROE, profit margins. In forex: signal quality.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # ROE proxy: Sharpe-like metric (return/risk)
        for d in [20, 60]:
            ret_sum = returns.rolling(d, min_periods=1).sum()
            ret_std = returns.rolling(d, min_periods=2).std()
            features[f'QUA_SHARPE{d}'] = ret_sum / (ret_std * np.sqrt(d) + 1e-12)

        # Profit margin proxy: positive return ratio
        features['QUA_WINRATE'] = (returns > 0).rolling(20, min_periods=1).mean()

        # Accruals proxy: difference between recent and longer-term performance
        ret_5 = returns.rolling(5, min_periods=1).sum()
        ret_20 = returns.rolling(20, min_periods=1).sum()
        features['QUA_ACCRUAL'] = ret_5 - (ret_20 / 4)  # Short vs normalized long

        return features

    # =========================================================================
    # LEVERAGE FACTORS (3) - Position concentration
    # =========================================================================

    def _leverage_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LEVERAGE factors - concentration and drawdown risk.
        In equities: debt/equity. In forex: drawdown metrics.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Debt-to-equity proxy: drawdown depth
        rolling_max = close.rolling(60, min_periods=1).max()
        drawdown = (close - rolling_max) / (rolling_max + 1e-12)
        features['LEV_DRAWDOWN'] = drawdown

        # Book leverage proxy: distance from range midpoint
        rolling_min = close.rolling(60, min_periods=1).min()
        range_pos = (close - rolling_min) / (rolling_max - rolling_min + 1e-12)
        features['LEV_RANGEPOS'] = range_pos - 0.5  # Centered at 0

        # Market leverage: recent volatility vs historical
        returns = close.pct_change()
        vol_5 = returns.rolling(5, min_periods=2).std()
        vol_60 = returns.rolling(60, min_periods=2).std()
        features['LEV_VOLRATIO'] = vol_5 / (vol_60 + 1e-12)

        return features

    # =========================================================================
    # SENTIMENT FACTORS (4) - Order flow and market mood
    # =========================================================================

    def _sentiment_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SENTIMENT factors - market mood and positioning.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        returns = close.pct_change()

        # Analyst sentiment proxy: recent price trend strength
        ma_5 = close.rolling(5, min_periods=1).mean()
        ma_20 = close.rolling(20, min_periods=1).mean()
        features['SENT_TREND'] = (ma_5 - ma_20) / (ma_20 + 1e-12)

        # On-balance volume proxy
        obv_sign = np.sign(returns)
        obv = (obv_sign * volume).rolling(20, min_periods=1).sum()
        features['SENT_OBV'] = obv / (volume.rolling(20, min_periods=1).sum() + 1e-12)

        # Money flow proxy (using price position in range)
        typical_price = (high + low + close) / 3
        mf_ratio = typical_price / close.shift(1)
        features['SENT_MFI'] = mf_ratio.rolling(14, min_periods=1).mean() - 1

        # Short interest proxy: down volume ratio
        down_vol = volume.where(returns < 0, 0)
        features['SENT_SHORT'] = (
            down_vol.rolling(10, min_periods=1).sum() /
            (volume.rolling(10, min_periods=1).sum() + 1e-12)
        )

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 46 Barra CNE6 factors.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 46 factor columns
        """
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Generate all factor groups
        size = self._size_factors(df)
        beta = self._beta_factors(df)
        momentum = self._momentum_factors(df)
        volatility = self._volatility_factors(df)
        liquidity = self._liquidity_factors(df)
        value = self._value_factors(df)
        growth = self._growth_factors(df)
        quality = self._quality_factors(df)
        leverage = self._leverage_factors(df)
        sentiment = self._sentiment_factors(df)

        # Combine all features
        features = pd.concat([
            size, beta, momentum, volatility, liquidity,
            value, growth, quality, leverage, sentiment
        ], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []

        # Size (5)
        names.extend(['SIZE_LNCAP', 'SIZE_MIDCAP', 'SIZE_RELATIVE',
                      'SIZE_RANGE', 'SIZE_RANK'])

        # Beta (4)
        names.extend(['BETA_HIST20', 'BETA_HIST60', 'BETA_DASTD', 'BETA_CMRA'])

        # Momentum (6)
        names.extend(['MOM_RSTR5', 'MOM_RSTR10', 'MOM_RSTR20',
                      'MOM_RSTR60', 'MOM_STREV', 'MOM_QUALITY'])

        # Volatility (6)
        names.extend(['VOL_RVOL5', 'VOL_RVOL20', 'VOL_RVOL60',
                      'VOL_PARKINSON', 'VOL_VOLVOL', 'VOL_SKEW'])

        # Liquidity (5)
        names.extend(['LIQ_STOM5', 'LIQ_STOM20', 'LIQ_STOM60',
                      'LIQ_VOLCOV', 'LIQ_AMIHUD'])

        # Value (5)
        names.extend(['VAL_BTOP20', 'VAL_BTOP60', 'VAL_ZSCORE',
                      'VAL_RSI', 'VAL_BBPOS'])

        # Growth (4)
        names.extend(['GRO_SLOPE20', 'GRO_SLOPE60', 'GRO_ACC', 'GRO_RSQ'])

        # Quality (4)
        names.extend(['QUA_SHARPE20', 'QUA_SHARPE60', 'QUA_WINRATE', 'QUA_ACCRUAL'])

        # Leverage (3)
        names.extend(['LEV_DRAWDOWN', 'LEV_RANGEPOS', 'LEV_VOLRATIO'])

        # Sentiment (4)
        names.extend(['SENT_TREND', 'SENT_OBV', 'SENT_MFI', 'SENT_SHORT'])

        return names

    def get_factor_category(self, factor_name: str) -> str:
        """Get the category of a factor by name."""
        categories = {
            'SIZE': 'Size',
            'BETA': 'Beta',
            'MOM': 'Momentum',
            'VOL': 'Volatility',
            'LIQ': 'Liquidity',
            'VAL': 'Value',
            'GRO': 'Growth',
            'QUA': 'Quality',
            'LEV': 'Leverage',
            'SENT': 'Sentiment'
        }
        prefix = factor_name.split('_')[0]
        return categories.get(prefix, 'Unknown')


# Convenience function
def generate_barra_cne6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Barra CNE6 factors for forex.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 46 Barra factors
    """
    barra = BarraCNE6Forex()
    return barra.generate_all(df)
