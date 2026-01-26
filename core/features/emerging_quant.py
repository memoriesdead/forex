"""
Emerging Markets Quantitative Finance Factors
=============================================
Quantitative formulas specific to emerging market currencies.

Regions Covered:
- Brazil (BRL)
- Russia (RUB) - historical research
- Southeast Asia (SGD, HKD, CNH)
- Mexico (MXN)
- Turkey (TRY)
- South Africa (ZAR)

Sources:
1. B3 Exchange (Brasil Bolsa Balcao) Research
   - Quantitative Brokers algorithms: Bolt, Strobe, Closer
   - PUMA trading platform research
   - ScienceDirect: "Algorithmic trading impact on spreads"

2. MOEX (Moscow Exchange) Historical Research
   - Pre-2022 FX market research
   - Petrocurrency dynamics (RUB-oil correlation)

3. Asian Central Banks Research
   - MAS (Singapore): SGD NEER band management
   - HKMA: HKD peg dynamics
   - PBOC: CNH-CNY spread research

4. Mexican Central Bank (Banxico)
   - MXN carry trade analysis
   - Remittance flow impact

5. SARB (South African Reserve Bank)
   - ZAR commodity currency dynamics

Total: 20 factors organized into:
- BRL Carry (4): Brazil high-rate environment
- Petrocurrency (4): Oil-correlated currencies
- Asian Pegs (4): Managed currency dynamics
- EM Volatility (4): Emerging market risk
- Commodity FX (4): Commodity-linked currencies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


class EmergingMarketsFeatures:
    """
    Emerging Markets Quantitative Features.

    Based on research from:
    - B3 Exchange (Brazil)
    - Asian central banks (MAS, HKMA, PBOC)
    - Banxico (Mexico)
    - SARB (South Africa)
    - Historical MOEX research (Russia)

    Specialized for EM currencies with unique characteristics:
    - High carry environments
    - Commodity linkages
    - Peg/band management
    - Political risk premiums

    Usage:
        features = EmergingMarketsFeatures()
        df_features = features.generate_all(ohlcv_df)
    """

    def __init__(
        self,
        em_rate_differential: float = 0.10,  # Assumed EM-DM rate differential
        oil_correlation_window: int = 60,     # Window for oil correlation proxy
    ):
        """
        Initialize Emerging Markets Features.

        Args:
            em_rate_differential: Assumed EM vs DM rate differential (default 10%)
            oil_correlation_window: Window for commodity correlation calc
        """
        self.em_rate = em_rate_differential
        self.oil_window = oil_correlation_window

    # =========================================================================
    # BRL CARRY (4) - Brazil High-Rate Environment
    # =========================================================================

    def _brl_carry_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Brazilian Real Carry Features.

        Brazil maintains one of highest real interest rates globally:
        - SELIC rate often 10%+ nominal
        - Strong carry trade flows
        - DI (interbank deposit) rate correlation

        B3 Exchange characteristics:
        - Quantitative Brokers algorithms
        - High algorithmic trading penetration
        - Spread dynamics research

        References:
        - B3 Exchange: "Algorithmic trading research"
        - ScienceDirect: "BRL carry decomposition"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. High-yield carry proxy: return + rate differential
        for d in [10, 30]:
            carry_return = returns.rolling(d, min_periods=1).mean()
            # High EM rates = larger carry component
            features[f'EM_BRL_CARRY_{d}d'] = carry_return + (self.em_rate / 252)

        # 2. DI rate proxy: volatility-adjusted momentum
        # DI rates correlate with BRL strength
        vol = returns.rolling(20, min_periods=2).std()
        mom = returns.rolling(20, min_periods=1).mean()
        features['EM_BRL_DI'] = mom / (vol + 1e-12)

        # 3. Carry unwind risk: sharp reversals in high-carry environment
        # EM currencies prone to sudden stop episodes
        cumret = returns.rolling(5, min_periods=1).sum()
        sudden_stop = np.where(cumret < -3 * vol, 1, 0)
        features['EM_BRL_UNWIND'] = sudden_stop

        return features

    # =========================================================================
    # PETROCURRENCY (4) - Oil-Correlated Currencies
    # =========================================================================

    def _petrocurrency_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Petrocurrency Dynamics (RUB, NOK, CAD, MXN).

        Oil-exporting countries' currencies correlate with oil prices:
        - RUB: Historically strong oil correlation (pre-2022)
        - NOK: Norwegian krone as petrocurrency
        - CAD: Canadian dollar oil sensitivity
        - MXN: Mexican peso oil income component

        We proxy oil correlation using:
        - Momentum patterns similar to commodity cycles
        - Volatility clustering (commodity style)
        - Trend-following signals

        References:
        - IMF: "Commodity currencies and exchange rates"
        - BIS: "Oil prices and exchange rates"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Commodity momentum proxy: trend-following signal
        # Petrocurrencies follow commodity super-cycles
        for d in [20, 60]:
            trend = close / close.shift(d) - 1
            features[f'EM_PETRO_MOM_{d}d'] = trend

        # 2. Commodity volatility clustering
        # Oil price shocks create clustered volatility
        vol = returns.rolling(20, min_periods=2).std()
        vol_lag = vol.shift(5)
        features['EM_PETRO_VOL_CLUST'] = vol / (vol_lag + 1e-12)

        # 3. Oil shock proxy: large price moves
        # Petrocurrencies react strongly to oil shocks
        vol_60 = returns.rolling(60, min_periods=10).std()
        oil_shock = np.where(returns.abs() > 2 * vol_60, 1, 0)
        features['EM_PETRO_SHOCK'] = oil_shock

        return features

    # =========================================================================
    # ASIAN PEGS (4) - Managed Currency Dynamics
    # =========================================================================

    def _asian_peg_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Asian Managed Currency Features (SGD, HKD, CNH).

        Key characteristics:
        - SGD: MAS manages via NEER band (undisclosed width)
        - HKD: Hard peg to USD (7.75-7.85 band)
        - CNH: Offshore RMB, managed float

        Trading strategies:
        - Band edge detection
        - Mean reversion within bands
        - Intervention anticipation

        References:
        - MAS: "Exchange rate management in Singapore"
        - HKMA: "Linked exchange rate system"
        - PBOC: "CNH market development"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Band position: where price is relative to recent range
        # Pegged currencies mean-revert within bands
        rolling_min = close.rolling(60, min_periods=10).min()
        rolling_max = close.rolling(60, min_periods=10).max()
        band_pos = (close - rolling_min) / (rolling_max - rolling_min + 1e-12)
        features['EM_ASIAN_BAND_POS'] = band_pos

        # 2. Band edge signal: extreme positions trigger mean reversion
        features['EM_ASIAN_BAND_EDGE'] = np.where(
            (band_pos < 0.1) | (band_pos > 0.9), 1, 0
        )

        # 3. Intervention proxy: unusual low volatility after large move
        vol_5 = returns.rolling(5, min_periods=2).std()
        vol_20 = returns.rolling(20, min_periods=2).std()
        vol_ratio = vol_5 / (vol_20 + 1e-12)
        # Low recent vol after being at band edge = intervention
        features['EM_ASIAN_INTERV'] = np.where(
            (band_pos < 0.1) | (band_pos > 0.9),
            1 - vol_ratio,  # Positive when vol drops
            0
        )

        # 4. CNH-CNY spread proxy: offshore premium
        # Use momentum divergence as proxy for CNH premium
        mom_5 = returns.rolling(5, min_periods=1).mean()
        mom_20 = returns.rolling(20, min_periods=1).mean()
        features['EM_ASIAN_OFFSHORE'] = mom_5 - mom_20

        return features

    # =========================================================================
    # EM VOLATILITY (4) - Emerging Market Risk
    # =========================================================================

    def _em_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Emerging Market Volatility Features.

        EM currencies exhibit unique volatility patterns:
        - Higher baseline volatility than DM
        - Fat tails (kurtosis)
        - Volatility asymmetry (larger on depreciation)
        - Contagion during crises

        References:
        - BIS: "Volatility patterns in EM currencies"
        - IMF: "Currency crisis early warning indicators"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. EM volatility premium: vol relative to historical
        vol = returns.rolling(20, min_periods=2).std() * np.sqrt(252)
        vol_hist = returns.rolling(252, min_periods=60).std() * np.sqrt(252)
        features['EM_VOL_PREM'] = vol / (vol_hist + 1e-12)

        # 2. Fat tails indicator: excess kurtosis
        kurt = returns.rolling(60, min_periods=20).apply(
            lambda x: stats.kurtosis(x, nan_policy='omit') if len(x) > 10 else 0,
            raw=True
        )
        features['EM_FAT_TAILS'] = kurt

        # 3. Volatility asymmetry: depreciation vol vs appreciation vol
        pos_ret = returns.where(returns > 0, 0)
        neg_ret = returns.where(returns < 0, 0)
        vol_pos = pos_ret.rolling(30, min_periods=5).std()
        vol_neg = neg_ret.abs().rolling(30, min_periods=5).std()
        features['EM_VOL_ASYM'] = (vol_neg - vol_pos) / (vol_neg + vol_pos + 1e-12)

        # 4. Crisis indicator: simultaneous high vol and depreciation
        high_vol = vol > vol.rolling(60, min_periods=20).quantile(0.9)
        depreciation = returns.rolling(5, min_periods=1).sum() < -0.02  # 2% drop
        features['EM_CRISIS_IND'] = np.where(high_vol & depreciation, 1, 0)

        return features

    # =========================================================================
    # COMMODITY FX (4) - Commodity-Linked Currencies
    # =========================================================================

    def _commodity_fx_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Commodity-Linked Currency Features (AUD, NZD, ZAR, CLP).

        Currencies of commodity exporters:
        - AUD: Iron ore, coal
        - NZD: Dairy, agricultural
        - ZAR: Gold, platinum, coal
        - CLP: Copper

        Common characteristics:
        - Terms of trade sensitivity
        - Risk-on/risk-off correlation
        - Global growth sensitivity

        References:
        - RBA: "AUD and commodity prices"
        - SARB: "ZAR as a commodity currency"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Terms of trade proxy: trend strength
        # Commodity currencies follow global growth trends
        trend_60 = close / close.shift(60) - 1
        trend_20 = close / close.shift(20) - 1
        features['EM_TOT_TREND'] = trend_60

        # 2. Global growth sensitivity: beta to market moves
        # Rolling beta to market proxy (own momentum as proxy)
        market_proxy = returns.rolling(60, min_periods=20).mean()
        cov = returns.rolling(60, min_periods=20).cov(market_proxy)
        var_mkt = market_proxy.rolling(60, min_periods=20).var()
        features['EM_GROWTH_BETA'] = cov / (var_mkt + 1e-12)

        # 3. Risk-on indicator: positive when risk appetite high
        # Use volatility as inverse risk appetite
        vol = returns.rolling(20, min_periods=2).std()
        vol_zscore = (vol - vol.rolling(120, min_periods=30).mean()) / (vol.rolling(120, min_periods=30).std() + 1e-12)
        features['EM_RISK_ON'] = -vol_zscore  # Positive in low vol (risk-on)

        # 4. Commodity cycle position: where in the cycle
        ma_20 = close.rolling(20, min_periods=1).mean()
        ma_100 = close.rolling(100, min_periods=20).mean()
        features['EM_COMM_CYCLE'] = (ma_20 - ma_100) / (ma_100 + 1e-12)

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Emerging Markets features.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 20 factor columns
        """
        # Ensure required columns
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        # Fill missing OHLC from close
        df = df.copy()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        # Generate all factor groups
        brl = self._brl_carry_factors(df)
        petro = self._petrocurrency_factors(df)
        asian = self._asian_peg_factors(df)
        vol = self._em_volatility_factors(df)
        commodity = self._commodity_fx_factors(df)

        # Combine all features
        features = pd.concat([
            brl, petro, asian, vol, commodity
        ], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []

        # BRL Carry (4)
        names.extend(['EM_BRL_CARRY_10d', 'EM_BRL_CARRY_30d', 'EM_BRL_DI',
                      'EM_BRL_UNWIND'])

        # Petrocurrency (4)
        names.extend(['EM_PETRO_MOM_20d', 'EM_PETRO_MOM_60d', 'EM_PETRO_VOL_CLUST',
                      'EM_PETRO_SHOCK'])

        # Asian Pegs (4)
        names.extend(['EM_ASIAN_BAND_POS', 'EM_ASIAN_BAND_EDGE', 'EM_ASIAN_INTERV',
                      'EM_ASIAN_OFFSHORE'])

        # EM Volatility (4)
        names.extend(['EM_VOL_PREM', 'EM_FAT_TAILS', 'EM_VOL_ASYM',
                      'EM_CRISIS_IND'])

        # Commodity FX (4)
        names.extend(['EM_TOT_TREND', 'EM_GROWTH_BETA', 'EM_RISK_ON',
                      'EM_COMM_CYCLE'])

        return names

    def get_factor_category(self, factor_name: str) -> str:
        """Get the category of a factor by name."""
        if 'BRL' in factor_name:
            return 'BRL High-Yield Carry'
        elif 'PETRO' in factor_name:
            return 'Petrocurrency Dynamics'
        elif 'ASIAN' in factor_name:
            return 'Asian Managed Currencies'
        elif any(x in factor_name for x in ['VOL_PREM', 'FAT_TAILS', 'VOL_ASYM', 'CRISIS']):
            return 'EM Volatility'
        elif any(x in factor_name for x in ['TOT', 'GROWTH_BETA', 'RISK_ON', 'COMM_CYCLE']):
            return 'Commodity FX'
        return 'Unknown'


# Convenience function
def generate_emerging_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Emerging Markets features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 20 emerging market factors
    """
    features = EmergingMarketsFeatures()
    return features.generate_all(df)
