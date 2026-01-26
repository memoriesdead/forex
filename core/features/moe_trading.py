"""
Mixture of Experts (MOE) Trading Module
=======================================
Gold standard MOE architecture for trading from Chinese/Asian research.

Citation:
    Zou, W. et al. (2024). "Trading like Gurus: Mixture of Instructions
    for Profitable Intelligent Trading." arXiv.
    - 24% excess annual return demonstrated
    - Multiple expert networks (momentum, mean-rev, volatility, trend)
    - Gating network selects experts based on market regime

Key Innovation:
    - Each expert specializes in a market condition
    - Gating network learns optimal expert weights
    - Ensemble output combines expert predictions

Implementation:
    - MomentumExpert: Trend-following signals
    - MeanReversionExpert: Mean-reversion signals
    - VolatilityExpert: Volatility regime signals
    - TrendExpert: Trend strength signals
    - GatingNetwork: Regime-based expert selection

Total: 20 features

Usage:
    from core.features.moe_trading import MixtureOfExpertsFeatures
    moe = MixtureOfExpertsFeatures()
    features = moe.generate_all(df)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
from scipy import stats
from scipy.special import softmax

warnings.filterwarnings('ignore')


class MomentumExpert:
    """
    Momentum Expert Network.

    Specializes in trending market conditions.
    Uses multiple momentum indicators with different horizons.

    Reference: Jegadeesh & Titman (1993) "Returns to Buying Winners"
    """

    def __init__(self, horizons: List[int] = [5, 10, 20, 50]):
        self.horizons = horizons

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum expert signals."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # Price momentum at multiple horizons
        for h in self.horizons:
            # Simple momentum
            features[f'MOM_EXPERT_ret_{h}'] = returns.rolling(h, min_periods=1).sum()

            # Momentum strength (normalized)
            ret_h = returns.rolling(h, min_periods=1).sum()
            vol_h = returns.rolling(h, min_periods=2).std() + 1e-10
            features[f'MOM_EXPERT_strength_{h}'] = ret_h / vol_h

        # Momentum acceleration (second derivative)
        mom_20 = returns.rolling(20, min_periods=1).sum()
        features['MOM_EXPERT_accel'] = mom_20.diff(5)

        # Cross-sectional momentum rank (within series)
        for h in [10, 20]:
            ret_h = returns.rolling(h, min_periods=1).sum()
            features[f'MOM_EXPERT_rank_{h}'] = ret_h.rolling(60, min_periods=10).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 1 else 0.5,
                raw=False
            )

        # Trend following signal
        ma_short = close.rolling(10, min_periods=1).mean()
        ma_long = close.rolling(50, min_periods=1).mean()
        features['MOM_EXPERT_trend'] = (ma_short - ma_long) / (ma_long + 1e-10)

        # Expert confidence (based on trend consistency)
        pos_returns = (returns > 0).rolling(20, min_periods=1).mean()
        features['MOM_EXPERT_confidence'] = 2 * pos_returns - 1  # -1 to 1

        return features


class MeanReversionExpert:
    """
    Mean Reversion Expert Network.

    Specializes in range-bound markets.
    Uses mean-reversion indicators with Ornstein-Uhlenbeck dynamics.

    Reference: Poterba & Summers (1988) "Mean Reversion in Stock Prices"
    """

    def __init__(self, windows: List[int] = [10, 20, 50]):
        self.windows = windows

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-reversion expert signals."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # Z-score from moving average
        for w in self.windows:
            ma = close.rolling(w, min_periods=1).mean()
            std = close.rolling(w, min_periods=2).std()
            features[f'MR_EXPERT_zscore_{w}'] = -(close - ma) / (std + 1e-10)

        # Bollinger Band position
        ma_20 = close.rolling(20, min_periods=1).mean()
        std_20 = close.rolling(20, min_periods=2).std()
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        bb_pos = (close - lower) / (upper - lower + 1e-10)
        features['MR_EXPERT_bb_pos'] = -(2 * bb_pos - 1)  # -1 at upper, +1 at lower

        # RSI-based reversion
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features['MR_EXPERT_rsi_rev'] = (50 - rsi) / 50  # Contrarian: -1 at 100, +1 at 0

        # Half-life estimation (OU process)
        # ln(2) / lambda where lambda is mean-reversion speed
        lag_price = close.shift(1)
        diff_price = close - lag_price

        # Simple regression estimate
        features['MR_EXPERT_halflife'] = self._estimate_halflife(close, 60)

        # Expert confidence (low volatility = range-bound)
        vol_ratio = returns.rolling(10, min_periods=2).std() / (returns.rolling(50, min_periods=5).std() + 1e-10)
        features['MR_EXPERT_confidence'] = 1 - vol_ratio.clip(0, 2) / 2

        return features

    def _estimate_halflife(self, prices: pd.Series, window: int) -> pd.Series:
        """Estimate mean-reversion half-life using rolling regression."""
        halflife = pd.Series(index=prices.index, dtype=float)

        for i in range(window, len(prices)):
            y = prices.iloc[i-window:i].diff().dropna().values
            x = prices.iloc[i-window:i-1].values
            if len(x) > 2 and np.std(x) > 0:
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    if slope < 0:
                        halflife.iloc[i] = -np.log(2) / slope
                    else:
                        halflife.iloc[i] = np.nan
                except:
                    halflife.iloc[i] = np.nan
            else:
                halflife.iloc[i] = np.nan

        return halflife.fillna(halflife.median())


class VolatilityExpert:
    """
    Volatility Expert Network.

    Specializes in volatility regime detection and trading.
    Uses volatility clustering and regime detection.

    Reference: Engle (1982) "Autoregressive Conditional Heteroskedasticity"
    """

    def __init__(self):
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility expert signals."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        returns = close.pct_change()

        # Realized volatility at multiple scales
        vol_5 = returns.rolling(5, min_periods=2).std()
        vol_20 = returns.rolling(20, min_periods=2).std()
        vol_60 = returns.rolling(60, min_periods=5).std()

        # Volatility term structure
        features['VOL_EXPERT_term_short'] = vol_5 / (vol_20 + 1e-10)
        features['VOL_EXPERT_term_long'] = vol_20 / (vol_60 + 1e-10)

        # Volatility z-score
        vol_ma = vol_20.rolling(60, min_periods=10).mean()
        vol_std = vol_20.rolling(60, min_periods=10).std()
        features['VOL_EXPERT_zscore'] = (vol_20 - vol_ma) / (vol_std + 1e-10)

        # Parkinson volatility (range-based)
        log_hl = np.log(high / (low + 1e-10) + 1e-10)
        parkinson = np.sqrt(log_hl ** 2 / (4 * np.log(2))).rolling(20, min_periods=2).mean()
        features['VOL_EXPERT_parkinson'] = parkinson

        # Volatility clustering (autocorrelation of squared returns)
        sq_returns = returns ** 2
        vol_cluster = sq_returns.rolling(20, min_periods=2).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
            raw=True
        )
        features['VOL_EXPERT_cluster'] = vol_cluster

        # Regime indicator (high/low vol)
        vol_pct = vol_20.rolling(252, min_periods=20).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 1 else 0.5,
            raw=False
        )
        features['VOL_EXPERT_regime'] = np.where(vol_pct > 0.8, 1,
                                                  np.where(vol_pct < 0.2, -1, 0))

        # Expert confidence (clear regime = high confidence)
        features['VOL_EXPERT_confidence'] = np.abs(vol_pct - 0.5) * 2

        return features


class TrendExpert:
    """
    Trend Expert Network.

    Specializes in identifying and trading trends.
    Uses ADX, moving average alignment, and trend persistence.

    Reference: Wilder (1978) "New Concepts in Technical Trading Systems"
    """

    def __init__(self):
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend expert signals."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        returns = close.pct_change()

        # ADX (Average Directional Index)
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - close.shift(1)),
                                   np.abs(low - close.shift(1))))

        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                           np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                            np.maximum(low.shift(1) - low, 0), 0)

        dm_plus = pd.Series(dm_plus, index=df.index)
        dm_minus = pd.Series(dm_minus, index=df.index)
        tr = pd.Series(tr, index=df.index)

        atr_14 = tr.rolling(14, min_periods=1).mean()
        di_plus = 100 * dm_plus.rolling(14, min_periods=1).mean() / (atr_14 + 1e-10)
        di_minus = 100 * dm_minus.rolling(14, min_periods=1).mean() / (atr_14 + 1e-10)

        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx, index=df.index).rolling(14, min_periods=1).mean()

        features['TREND_EXPERT_adx'] = adx / 100  # Normalize to 0-1
        features['TREND_EXPERT_direction'] = np.sign(di_plus - di_minus)

        # Moving average alignment
        ma_10 = close.rolling(10, min_periods=1).mean()
        ma_20 = close.rolling(20, min_periods=1).mean()
        ma_50 = close.rolling(50, min_periods=1).mean()

        # Alignment score: +1 if all MAs aligned bullish, -1 if bearish
        bull_align = (ma_10 > ma_20) & (ma_20 > ma_50)
        bear_align = (ma_10 < ma_20) & (ma_20 < ma_50)
        features['TREND_EXPERT_align'] = np.where(bull_align, 1, np.where(bear_align, -1, 0))

        # Trend persistence (runs test)
        signs = np.sign(returns)
        run_length = signs.groupby((signs != signs.shift()).cumsum()).cumcount() + 1
        features['TREND_EXPERT_persist'] = run_length.rolling(20, min_periods=1).mean() / 10

        # Trend strength (slope of regression)
        def calc_slope(x):
            if len(x) < 2:
                return 0
            return stats.linregress(range(len(x)), x)[0]

        features['TREND_EXPERT_slope'] = close.rolling(20, min_periods=2).apply(calc_slope, raw=True)
        features['TREND_EXPERT_slope'] = features['TREND_EXPERT_slope'] / (close + 1e-10)

        # Expert confidence (strong ADX = confident)
        features['TREND_EXPERT_confidence'] = (adx / 50).clip(0, 1)

        return features


class GatingNetwork:
    """
    Gating Network for Expert Selection.

    Learns optimal weights for combining experts based on market state.
    Uses softmax attention over expert outputs.

    Reference: Jacobs et al. (1991) "Adaptive Mixtures of Local Experts"
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize gating network.

        Args:
            temperature: Softmax temperature (lower = more selective)
        """
        self.temperature = temperature

    def compute_gates(self, expert_confidences: pd.DataFrame) -> pd.DataFrame:
        """
        Compute gating weights from expert confidences.

        Args:
            expert_confidences: DataFrame with confidence columns from each expert

        Returns:
            DataFrame with normalized gate weights
        """
        # Softmax over expert confidences
        confidences = expert_confidences.fillna(0.5).values
        gates = softmax(confidences / self.temperature, axis=1)

        return pd.DataFrame(
            gates,
            index=expert_confidences.index,
            columns=[c.replace('confidence', 'gate') for c in expert_confidences.columns]
        )


class MixtureOfExpertsFeatures:
    """
    Mixture of Experts Feature Generator.

    Combines multiple expert networks with gating mechanism.
    Based on MIGA paper demonstrating 24% excess returns.

    Citations:
    [1] Zou et al. (2024). "Trading like Gurus: Mixture of Instructions
        for Profitable Intelligent Trading." arXiv.
    [2] Jacobs et al. (1991). "Adaptive Mixtures of Local Experts"
        Neural Computation.
    [3] Shazeer et al. (2017). "Outrageously Large Neural Networks:
        The Sparsely-Gated Mixture-of-Experts Layer" ICLR.

    Total: 20 features
    - Momentum Expert: 4 features
    - Mean Reversion Expert: 3 features
    - Volatility Expert: 3 features
    - Trend Expert: 3 features
    - Gating: 4 features
    - Combined: 3 features
    """

    def __init__(self, temperature: float = 1.0):
        self.momentum_expert = MomentumExpert(horizons=[5, 10, 20])
        self.mr_expert = MeanReversionExpert(windows=[10, 20])
        self.vol_expert = VolatilityExpert()
        self.trend_expert = TrendExpert()
        self.gating = GatingNetwork(temperature=temperature)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all MOE features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 20 MOE features
        """
        # Validate input
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        df = df.copy()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        # Generate expert signals
        mom_features = self.momentum_expert.generate_signals(df)
        mr_features = self.mr_expert.generate_signals(df)
        vol_features = self.vol_expert.generate_signals(df)
        trend_features = self.trend_expert.generate_signals(df)

        # Extract key signals from each expert (for final output)
        result = pd.DataFrame(index=df.index)

        # Momentum Expert summary (4 features)
        result['MOE_MOM_signal'] = mom_features['MOM_EXPERT_trend']
        result['MOE_MOM_strength'] = mom_features.get('MOM_EXPERT_strength_20', mom_features['MOM_EXPERT_trend'])
        result['MOE_MOM_accel'] = mom_features['MOM_EXPERT_accel']
        result['MOE_MOM_conf'] = mom_features['MOM_EXPERT_confidence']

        # Mean Reversion Expert summary (3 features)
        result['MOE_MR_signal'] = mr_features['MR_EXPERT_zscore_20']
        result['MOE_MR_bb'] = mr_features['MR_EXPERT_bb_pos']
        result['MOE_MR_conf'] = mr_features['MR_EXPERT_confidence']

        # Volatility Expert summary (3 features)
        result['MOE_VOL_term'] = vol_features['VOL_EXPERT_term_short']
        result['MOE_VOL_zscore'] = vol_features['VOL_EXPERT_zscore']
        result['MOE_VOL_conf'] = vol_features['VOL_EXPERT_confidence']

        # Trend Expert summary (3 features)
        result['MOE_TREND_adx'] = trend_features['TREND_EXPERT_adx']
        result['MOE_TREND_align'] = trend_features['TREND_EXPERT_align']
        result['MOE_TREND_conf'] = trend_features['TREND_EXPERT_confidence']

        # Gating weights (4 features)
        confidences = pd.DataFrame({
            'MOM_confidence': result['MOE_MOM_conf'],
            'MR_confidence': result['MOE_MR_conf'],
            'VOL_confidence': result['MOE_VOL_conf'],
            'TREND_confidence': result['MOE_TREND_conf']
        })
        gates = self.gating.compute_gates(confidences)
        result['MOE_GATE_momentum'] = gates.iloc[:, 0]
        result['MOE_GATE_meanrev'] = gates.iloc[:, 1]
        result['MOE_GATE_volatility'] = gates.iloc[:, 2]
        result['MOE_GATE_trend'] = gates.iloc[:, 3]

        # Combined MOE output (3 features)
        # Weighted sum of expert signals
        result['MOE_COMBINED_signal'] = (
            result['MOE_GATE_momentum'] * result['MOE_MOM_signal'] +
            result['MOE_GATE_meanrev'] * result['MOE_MR_signal'] +
            result['MOE_GATE_trend'] * result['MOE_TREND_align']
        )

        # Expert agreement (high when experts align)
        expert_signs = np.sign(pd.DataFrame({
            'mom': result['MOE_MOM_signal'],
            'mr': result['MOE_MR_signal'],
            'trend': result['MOE_TREND_align']
        }))
        result['MOE_AGREEMENT'] = expert_signs.mean(axis=1).abs()

        # Dominant expert
        result['MOE_DOMINANT'] = gates.idxmax(axis=1).map({
            'MOM_gate': 1, 'MR_gate': 2, 'VOL_gate': 3, 'TREND_gate': 4
        }).fillna(0)

        # Clean up
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            # Momentum (4)
            'MOE_MOM_signal', 'MOE_MOM_strength', 'MOE_MOM_accel', 'MOE_MOM_conf',
            # Mean Reversion (3)
            'MOE_MR_signal', 'MOE_MR_bb', 'MOE_MR_conf',
            # Volatility (3)
            'MOE_VOL_term', 'MOE_VOL_zscore', 'MOE_VOL_conf',
            # Trend (3)
            'MOE_TREND_adx', 'MOE_TREND_align', 'MOE_TREND_conf',
            # Gating (4)
            'MOE_GATE_momentum', 'MOE_GATE_meanrev', 'MOE_GATE_volatility', 'MOE_GATE_trend',
            # Combined (3)
            'MOE_COMBINED_signal', 'MOE_AGREEMENT', 'MOE_DOMINANT'
        ]

    @staticmethod
    def get_citations() -> Dict[str, str]:
        """Get academic citations for MOE trading."""
        return {
            'MIGA': """Zou, W. et al. (2024). "Trading like Gurus: Mixture of Instructions
                       for Profitable Intelligent Trading." arXiv.
                       Key finding: 24% excess annual return with MOE architecture.""",
            'MOE_Original': """Jacobs, R.A. et al. (1991). "Adaptive Mixtures of Local Experts"
                              Neural Computation, 3(1), 79-87.
                              Foundation paper for mixture of experts.""",
            'Sparse_MOE': """Shazeer, N. et al. (2017). "Outrageously Large Neural Networks:
                            The Sparsely-Gated Mixture-of-Experts Layer" ICLR.
                            Modern sparse gating mechanisms."""
        }


# Convenience function
def generate_moe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate MOE trading features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 20 MOE features
    """
    generator = MixtureOfExpertsFeatures()
    return generator.generate_all(df)
