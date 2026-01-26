"""
US Academic Forex Factors
=========================
Gold standard peer-reviewed factors from US academia.

Sources:
1. Lustig, Roussanov, Verdelhan (2011) - "Common Risk Factors in Currency Markets"
   Review of Financial Studies - DOL (Dollar), HML_FX (Carry)

2. Menkhoff et al (2012) - "Currency Momentum Strategies"
   BIS Working Papers No 366 - MOM (Momentum)

3. Asness, Moskowitz, Pedersen (2013) - "Value and Momentum Everywhere"
   Journal of Financial Economics - VAL (Value)

4. Bybee, Gomes, Valente (2023) - "Macro-Based Factors for Currency Returns"
   SSRN - Macro factors

5. AQR Research - Time Series Momentum
   TSM (Time Series Momentum)

Total: 50+ factors organized into:
- Carry factors (8)
- Momentum factors (10)
- Value factors (8)
- Volatility factors (8)
- Macro-proxy factors (8)
- Technical factors (8)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class USAcademicFactors:
    """
    US Academic Forex Factors - Gold Standard.

    Based on peer-reviewed research from:
    - Review of Financial Studies
    - Journal of Financial Economics
    - BIS Working Papers
    - NBER Working Papers
    - SSRN

    Usage:
        factors = USAcademicFactors()
        features = factors.generate_all(ohlcv_df)
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize US Academic Factors.

        Args:
            windows: Rolling windows (default: [5, 10, 20, 60, 120])
        """
        self.windows = windows or [5, 10, 20, 60, 120]

    # =========================================================================
    # CARRY FACTORS (8) - Lustig, Roussanov, Verdelhan (2011)
    # =========================================================================

    def _carry_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CARRY factors - Interest rate differential proxies.

        In real carry trades: Long high yield, short low yield.
        For single pair: Use forward discount (interest rate differential proxy).

        We approximate using price momentum as carry proxy.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # Forward discount proxy: rolling mean return (approximates carry)
        for d in [5, 20, 60]:
            features[f'CARRY_FWD{d}'] = returns.rolling(d, min_periods=1).mean()

        # Carry momentum: trend in carry
        carry_proxy = returns.rolling(20, min_periods=1).mean()
        features['CARRY_MOM5'] = carry_proxy - carry_proxy.shift(5)
        features['CARRY_MOM20'] = carry_proxy - carry_proxy.shift(20)

        # Carry reversal signal
        features['CARRY_REV'] = -returns.rolling(5, min_periods=1).mean()

        # Carry volatility adjusted (Sharpe-like)
        for d in [20, 60]:
            ret_mean = returns.rolling(d, min_periods=1).mean()
            ret_std = returns.rolling(d, min_periods=2).std()
            features[f'CARRY_SR{d}'] = ret_mean / (ret_std + 1e-12)

        return features

    # =========================================================================
    # MOMENTUM FACTORS (10) - Menkhoff et al (2012), AQR TSM
    # =========================================================================

    def _momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MOMENTUM factors - Cross-sectional and time-series momentum.

        MOM(1,1) = 1-month formation, 1-month holding
        TSM = Time series momentum (Moskowitz et al 2012)
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # Cross-sectional momentum: past returns
        for d in [5, 10, 20, 60, 120]:
            features[f'MOM_XS{d}'] = close / close.shift(d) - 1

        # Time-series momentum (TSM): sign of past return * magnitude
        for d in [20, 60, 120]:
            past_ret = close / close.shift(d) - 1
            features[f'MOM_TS{d}'] = np.sign(past_ret) * np.abs(past_ret)

        # Momentum acceleration
        mom_20 = close / close.shift(20) - 1
        mom_60 = close / close.shift(60) - 1
        features['MOM_ACC'] = mom_20 - (mom_60 / 3)

        # Momentum quality: consistency
        up_ratio = (returns > 0).rolling(20, min_periods=1).mean()
        features['MOM_QUAL'] = 2 * up_ratio - 1

        return features

    # =========================================================================
    # VALUE FACTORS (8) - Asness, Moskowitz, Pedersen (2013)
    # =========================================================================

    def _value_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VALUE factors - PPP deviation proxies.

        Real forex value = PPP deviation.
        We proxy using price deviation from long-term average.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # PPP deviation proxy: distance from moving average
        for d in [20, 60, 120, 250]:
            ma = close.rolling(d, min_periods=1).mean()
            features[f'VAL_PPP{d}'] = (ma - close) / (close + 1e-12)

        # Real exchange rate proxy: cumulative return deviation
        cum_ret = (close / close.iloc[0] - 1) if len(close) > 0 else close
        features['VAL_RER'] = cum_ret - cum_ret.rolling(120, min_periods=1).mean()

        # Value momentum: change in value signal
        val_20 = (close.rolling(20, min_periods=1).mean() - close) / (close + 1e-12)
        features['VAL_MOM'] = val_20 - val_20.shift(20)

        # Bollinger value (statistical value)
        ma_60 = close.rolling(60, min_periods=1).mean()
        std_60 = close.rolling(60, min_periods=2).std()
        features['VAL_BB'] = (ma_60 - close) / (2 * std_60 + 1e-12)

        # Mean reversion signal
        features['VAL_MEAN_REV'] = (close.rolling(5, min_periods=1).mean() -
                                    close.rolling(60, min_periods=1).mean()) / (close + 1e-12)

        return features

    # =========================================================================
    # VOLATILITY FACTORS (8) - Volatility Risk Premium Research
    # =========================================================================

    def _volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        VOLATILITY factors - FX volatility risk premium.

        Based on research showing vol risk premium predicts returns.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()

        # Realized volatility at different horizons
        for d in [5, 20, 60]:
            features[f'VOL_RV{d}'] = returns.rolling(d, min_periods=2).std() * np.sqrt(252)

        # Parkinson volatility (high-low based)
        log_hl = np.log(high / low + 1e-12)
        features['VOL_PARK'] = np.sqrt(
            (log_hl ** 2).rolling(20, min_periods=1).mean() / (4 * np.log(2))
        ) * np.sqrt(252)

        # Volatility risk premium proxy: short-term vs long-term vol
        vol_5 = returns.rolling(5, min_periods=2).std()
        vol_60 = returns.rolling(60, min_periods=2).std()
        features['VOL_VRP'] = vol_5 - vol_60

        # Volatility trend
        features['VOL_TREND'] = vol_5 / (vol_60 + 1e-12) - 1

        # Volatility of volatility
        features['VOL_VOV'] = vol_5.rolling(20, min_periods=2).std()

        # Volatility skew: up vol vs down vol
        pos_ret = returns.where(returns > 0, 0)
        neg_ret = returns.where(returns < 0, 0)
        up_vol = pos_ret.rolling(20, min_periods=2).std()
        down_vol = np.abs(neg_ret).rolling(20, min_periods=2).std()
        features['VOL_SKEW'] = (up_vol - down_vol) / (up_vol + down_vol + 1e-12)

        return features

    # =========================================================================
    # MACRO-PROXY FACTORS (8) - Bybee, Gomes, Valente (2023)
    # =========================================================================

    def _macro_proxy_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACRO-PROXY factors - Macroeconomic indicator proxies.

        Without actual macro data, we proxy using price patterns that
        correlate with macro conditions.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # GDP growth proxy: trend strength
        def calc_slope(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]

        for d in [20, 60]:
            features[f'MACRO_GDP{d}'] = close.rolling(d, min_periods=2).apply(calc_slope, raw=True)

        # Inflation proxy: price level change
        features['MACRO_INF20'] = close / close.shift(20) - 1
        features['MACRO_INF60'] = close / close.shift(60) - 1

        # Economic uncertainty proxy: volatility regime
        vol_20 = returns.rolling(20, min_periods=2).std()
        vol_60 = returns.rolling(60, min_periods=2).std()
        features['MACRO_UNC'] = vol_20 / (vol_60 + 1e-12)

        # Risk appetite proxy: drawdown from high
        rolling_max = close.rolling(60, min_periods=1).max()
        features['MACRO_RISK'] = close / rolling_max - 1

        # Growth momentum
        growth_proxy = close / close.shift(20) - 1
        features['MACRO_GMOM'] = growth_proxy - growth_proxy.shift(20)

        # Cycle indicator: deviation from trend
        trend = close.rolling(120, min_periods=1).mean()
        features['MACRO_CYCLE'] = (close - trend) / (trend + 1e-12)

        return features

    # =========================================================================
    # TECHNICAL FACTORS (8) - Standard academic technical analysis
    # =========================================================================

    def _technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TECHNICAL factors - Academic technical analysis.

        Based on established technical indicators used in forex research.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change()

        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        features['TECH_RSI'] = (100 - (100 / (1 + rs))) / 100 - 0.5

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['TECH_MACD'] = (macd - signal) / (close + 1e-12)

        # Stochastic Oscillator
        low_14 = low.rolling(14, min_periods=1).min()
        high_14 = high.rolling(14, min_periods=1).max()
        features['TECH_STOCH'] = (close - low_14) / (high_14 - low_14 + 1e-12) - 0.5

        # Williams %R
        features['TECH_WILLR'] = (high_14 - close) / (high_14 - low_14 + 1e-12) - 0.5

        # Average True Range (normalized)
        tr = np.maximum(high - low,
                       np.abs(high - close.shift(1)),
                       np.abs(low - close.shift(1)))
        features['TECH_ATR'] = tr.rolling(14, min_periods=1).mean() / (close + 1e-12)

        # Bollinger Band position
        ma_20 = close.rolling(20, min_periods=1).mean()
        std_20 = close.rolling(20, min_periods=2).std()
        features['TECH_BB'] = (close - ma_20) / (2 * std_20 + 1e-12)

        # Moving average crossover signal
        ma_10 = close.rolling(10, min_periods=1).mean()
        ma_50 = close.rolling(50, min_periods=1).mean()
        features['TECH_MAXO'] = (ma_10 - ma_50) / (ma_50 + 1e-12)

        # Price rate of change
        features['TECH_ROC'] = close / close.shift(10) - 1

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all US Academic factors.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 50 factor columns
        """
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Generate all factor groups
        carry = self._carry_factors(df)
        momentum = self._momentum_factors(df)
        value = self._value_factors(df)
        volatility = self._volatility_factors(df)
        macro = self._macro_proxy_factors(df)
        technical = self._technical_factors(df)

        # Combine all features
        features = pd.concat([
            carry, momentum, value, volatility, macro, technical
        ], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []

        # Carry (8)
        names.extend(['CARRY_FWD5', 'CARRY_FWD20', 'CARRY_FWD60',
                      'CARRY_MOM5', 'CARRY_MOM20', 'CARRY_REV',
                      'CARRY_SR20', 'CARRY_SR60'])

        # Momentum (10)
        names.extend(['MOM_XS5', 'MOM_XS10', 'MOM_XS20', 'MOM_XS60', 'MOM_XS120',
                      'MOM_TS20', 'MOM_TS60', 'MOM_TS120', 'MOM_ACC', 'MOM_QUAL'])

        # Value (8)
        names.extend(['VAL_PPP20', 'VAL_PPP60', 'VAL_PPP120', 'VAL_PPP250',
                      'VAL_RER', 'VAL_MOM', 'VAL_BB', 'VAL_MEAN_REV'])

        # Volatility (8)
        names.extend(['VOL_RV5', 'VOL_RV20', 'VOL_RV60', 'VOL_PARK',
                      'VOL_VRP', 'VOL_TREND', 'VOL_VOV', 'VOL_SKEW'])

        # Macro (8)
        names.extend(['MACRO_GDP20', 'MACRO_GDP60', 'MACRO_INF20', 'MACRO_INF60',
                      'MACRO_UNC', 'MACRO_RISK', 'MACRO_GMOM', 'MACRO_CYCLE'])

        # Technical (8)
        names.extend(['TECH_RSI', 'TECH_MACD', 'TECH_STOCH', 'TECH_WILLR',
                      'TECH_ATR', 'TECH_BB', 'TECH_MAXO', 'TECH_ROC'])

        return names

    def get_factor_category(self, factor_name: str) -> str:
        """Get the category of a factor by name."""
        categories = {
            'CARRY': 'Carry (Interest Rate)',
            'MOM': 'Momentum',
            'VAL': 'Value (PPP)',
            'VOL': 'Volatility',
            'MACRO': 'Macro-Proxy',
            'TECH': 'Technical'
        }
        prefix = factor_name.split('_')[0]
        return categories.get(prefix, 'Unknown')


# Convenience function
def generate_us_academic_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate US Academic forex factors.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 50 academic factors
    """
    factors = USAcademicFactors()
    return factors.generate_all(df)
