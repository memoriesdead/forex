"""
MegaFeatureGenerator - Unified 1300+ Feature Generator
======================================================
Combines ALL feature libraries from:
- Chinese Quant: Alpha101 (62), Alpha158 (179), Alpha360 (200), Barra CNE6 (46)
- Chinese Quant: Alpha191 Guotai Junan (191), Chinese HFT Factors (50+)
- USA Academic: Carry, Momentum, Value, Volatility, Macro, Technical (50)
- GitHub/Gitee: MLFinLab (17), Time-Series-Library (41)
- Renaissance: Weak Signals (51)
- Chinese HFT Additions: RSRS, Yang-Zhang Vol, Hawkes OFI (25+)
- GLOBAL EXPANSION (2026-01-17):
  - India Quant: IIQF, NSE Academy, RBI research (25)
  - Japan Quant: JPX, J-Quants, TSE Arrowhead, HAR-RV, BOJ (30) - UPDATED
  - Europe Quant: ETH, Imperial, Heston, SABR (15)
  - Emerging Markets: Brazil, Russia, SE Asia (20)
  - Universal Math: OU, Kelly, Cointegration (30)
- REINFORCEMENT LEARNING (2026-01-17):
  - Pure Mathematical RL: Q-Learning, SARSA, TD(λ), Dyna-Q (35)
  - Almgren-Chriss Optimal Execution
  - Thompson Sampling Bandits
  - Kelly-RL Integration
- EASTERN ASIA GOLD STANDARD (2026-01-17):
  - MOE (Mixture of Experts): MIGA architecture (20) - 24% excess return
  - GNN Temporal: Graph attention, message passing (15)
  - Korea Quant: MS-GARCH, CGMY Lévy, VKOSPI (20)
  - Asia FX Spread: CNH-CNY, HKD peg (15)
  - MARL Trading: Multi-agent RL dynamics (15)

Total: 1350+ features for institutional-grade ML

Usage:
    from core.features.mega_generator import MegaFeatureGenerator

    generator = MegaFeatureGenerator()
    features = generator.generate_all(ohlcv_df)
    print(f"Generated {len(features.columns)} features")
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from scipy import stats


class ChineseHFTAdditions:
    """
    Additional Chinese HFT formulas not in existing implementations.

    Includes:
    - RSRS (支撑阻力相对强度) - Regime detection from Chinese quants
    - Yang-Zhang Volatility - More efficient vol estimation
    - Volume Clock Acceleration - Second derivative of volume
    - Hawkes-inspired OFI - Order flow with decay
    - Session Reversal - T+1 effect adapted for forex
    """

    @staticmethod
    def rsrs(high: pd.Series, low: pd.Series, window: int = 18) -> pd.Series:
        """
        RSRS (相对强度指标) - Relative Strength Rating System.
        Chinese quant standard for regime detection.

        Calculates slope of high vs low regression, then z-scores it.
        High RSRS = bullish regime, Low RSRS = bearish regime.
        """
        slopes = []
        for i in range(len(high)):
            if i < window:
                slopes.append(np.nan)
            else:
                y = high.iloc[i-window:i].values
                x = low.iloc[i-window:i].values
                try:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                except:
                    slopes.append(np.nan)

        slope_series = pd.Series(slopes, index=high.index)
        # Standardize over 250 periods (or available data)
        lookback = min(250, len(slope_series) - 1)
        zscore = (slope_series - slope_series.rolling(lookback, min_periods=20).mean()) / \
                 (slope_series.rolling(lookback, min_periods=20).std() + 1e-10)
        return zscore

    @staticmethod
    def yang_zhang_volatility(open_: pd.Series, high: pd.Series,
                               low: pd.Series, close: pd.Series,
                               window: int = 20) -> pd.Series:
        """
        Yang-Zhang volatility estimator - more efficient than close-to-close.
        Combines overnight, open-close, and Rogers-Satchell components.
        """
        # Overnight volatility
        overnight = np.log(open_ / close.shift(1))
        overnight_var = overnight.rolling(window, min_periods=5).var()

        # Open-to-close volatility
        open_close = np.log(close / open_)
        oc_var = open_close.rolling(window, min_periods=5).var()

        # Rogers-Satchell volatility
        rs = np.log(high / close) * np.log(high / open_) + \
             np.log(low / close) * np.log(low / open_)
        rs_var = rs.rolling(window, min_periods=5).mean()

        # Combine (Yang-Zhang weights)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var.abs()

        return np.sqrt(yz_var.clip(lower=0) * 252)  # Annualized

    @staticmethod
    def volume_clock_acceleration(volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Volume clock acceleration (成交量时钟加速度).
        Second derivative of volume - detects volume regime changes.
        """
        vol_ma = volume.rolling(window, min_periods=5).mean()
        vol_velocity = vol_ma.diff()
        vol_acceleration = vol_velocity.diff()
        return vol_acceleration / (vol_ma + 1e-10)

    @staticmethod
    def hawkes_ofi(returns: pd.Series, volume: pd.Series,
                   alpha: float = 0.5, beta: float = 1.0,
                   window: int = 50) -> pd.Series:
        """
        Hawkes-inspired Order Flow Imbalance.
        Decaying impact of past order flow on current intensity.
        """
        # Classify trades as buy/sell based on returns
        buy_volume = volume.where(returns > 0, 0)
        sell_volume = volume.where(returns < 0, 0)

        # Exponential decay weights
        weights = np.array([alpha * np.exp(-beta * i) for i in range(window)])
        weights = weights / weights.sum()

        # Weighted imbalance
        def weighted_imbalance(buy, sell):
            if len(buy) < window:
                return np.nan
            buy_arr = buy[-window:].values
            sell_arr = sell[-window:].values
            buy_weighted = np.sum(buy_arr * weights[::-1])
            sell_weighted = np.sum(sell_arr * weights[::-1])
            total = buy_weighted + sell_weighted + 1e-10
            return (buy_weighted - sell_weighted) / total

        ofi = pd.Series(index=returns.index, dtype=float)
        for i in range(window, len(returns)):
            ofi.iloc[i] = weighted_imbalance(
                buy_volume.iloc[i-window:i+1],
                sell_volume.iloc[i-window:i+1]
            )
        return ofi

    @staticmethod
    def session_reversal(returns: pd.Series, window: int = 5) -> pd.Series:
        """
        Session reversal effect (T+1 adapted for forex).
        Contrarian signal based on recent session returns.
        """
        session_return = returns.rolling(window, min_periods=1).sum()
        return -session_return  # Contrarian

    @staticmethod
    def trade_intensity(volume: pd.Series, window: int = 20) -> pd.Series:
        """
        Trade intensity - volume relative to recent average.
        High intensity suggests institutional activity.
        """
        vol_ma = volume.rolling(window, min_periods=5).mean()
        vol_std = volume.rolling(window, min_periods=5).std()
        return (volume - vol_ma) / (vol_std + 1e-10)

    @staticmethod
    def price_pressure(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Price pressure indicator.
        Measures where close is relative to range.
        """
        range_ = high - low + 1e-10
        upper_pressure = (high - close) / range_
        lower_pressure = (close - low) / range_
        return lower_pressure - upper_pressure  # Positive = bullish pressure

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all Chinese HFT addition features."""
        result = pd.DataFrame(index=df.index)

        open_ = df.get('open', df['close'].shift(1).fillna(df['close']))
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        returns = close.pct_change().fillna(0)

        # RSRS - Multiple windows
        result['rsrs_18'] = self.rsrs(high, low, 18)
        result['rsrs_10'] = self.rsrs(high, low, 10)
        result['rsrs_25'] = self.rsrs(high, low, 25)

        # Yang-Zhang Volatility - Multiple windows
        result['yz_vol_10'] = self.yang_zhang_volatility(open_, high, low, close, 10)
        result['yz_vol_20'] = self.yang_zhang_volatility(open_, high, low, close, 20)
        result['yz_vol_30'] = self.yang_zhang_volatility(open_, high, low, close, 30)

        # Volume Clock Acceleration
        result['vol_accel_10'] = self.volume_clock_acceleration(volume, 10)
        result['vol_accel_20'] = self.volume_clock_acceleration(volume, 20)

        # Hawkes OFI - Multiple decay rates
        result['hawkes_ofi_fast'] = self.hawkes_ofi(returns, volume, alpha=0.7, beta=1.5, window=30)
        result['hawkes_ofi_slow'] = self.hawkes_ofi(returns, volume, alpha=0.3, beta=0.5, window=50)

        # Session Reversal
        result['session_rev_5'] = self.session_reversal(returns, 5)
        result['session_rev_10'] = self.session_reversal(returns, 10)
        result['session_rev_20'] = self.session_reversal(returns, 20)

        # Trade Intensity
        result['trade_intensity_10'] = self.trade_intensity(volume, 10)
        result['trade_intensity_20'] = self.trade_intensity(volume, 20)

        # Price Pressure
        result['price_pressure'] = self.price_pressure(high, low, close)

        # Derived signals
        result['rsrs_momentum'] = result['rsrs_18'].diff(5)  # RSRS momentum
        result['yz_vol_ratio'] = result['yz_vol_10'] / (result['yz_vol_30'] + 1e-10)  # Vol term structure

        # RSRS regime signal (discrete)
        result['rsrs_regime'] = np.where(result['rsrs_18'] > 1, 1,
                                          np.where(result['rsrs_18'] < -1, -1, 0))

        # Combined signals
        result['chinese_composite'] = (
            result['rsrs_18'].fillna(0) * 0.3 +
            result['hawkes_ofi_fast'].fillna(0) * 0.3 +
            result['session_rev_5'].fillna(0) * 0.2 +
            result['price_pressure'].fillna(0) * 0.2
        )

        return result


class MegaFeatureGenerator:
    """
    Unified generator for ALL 1200+ features.

    Feature breakdown:
    - Alpha101: 62 WorldQuant alphas
    - Alpha158: 179 Microsoft Qlib factors
    - Alpha360: 200 raw price features (compact version)
    - Barra CNE6: 46 risk factors
    - Alpha191: 191 Guotai Junan factors (Chinese institutional)
    - Chinese HFT: 50+ HFT microstructure factors
    - Chinese HFT Additions: 25+ RSRS, Yang-Zhang, Hawkes
    - US Academic: 50 peer-reviewed factors
    - MLFinLab: 17 Lopez de Prado features
    - Time-Series-Library: 41 deep learning inspired
    - Renaissance: 51 weak signals
    - India Quant: 25 IIQF/NSE/RBI factors (NEW 2026-01-17)
    - Japan Quant: 20 JPX/J-Quants factors (NEW 2026-01-17)
    - Europe Quant: 15 Heston/SABR/Rough Vol factors (NEW 2026-01-17)
    - Emerging Markets: 20 Brazil/Asia factors (NEW 2026-01-17)
    - Universal Math: 30 OU/Kelly/Cointegration factors (NEW 2026-01-17)
    - RL Algorithms: 35 Q-Learning/SARSA/TD(λ)/Dyna-Q/Thompson (NEW 2026-01-17)
    """

    def __init__(
        self,
        enable_alpha101: bool = True,
        enable_alpha158: bool = True,
        enable_alpha360: bool = True,
        enable_barra: bool = True,
        enable_alpha191: bool = True,
        enable_chinese_hft: bool = True,
        enable_chinese_additions: bool = True,
        enable_chinese_gold: bool = True,
        enable_us_academic: bool = True,
        enable_mlfinlab: bool = True,
        enable_timeseries: bool = True,
        enable_renaissance: bool = True,
        # Global expansion (2026-01-17)
        enable_india: bool = True,
        enable_japan: bool = True,
        enable_europe: bool = True,
        enable_emerging: bool = True,
        enable_universal_math: bool = True,
        # Reinforcement Learning (2026-01-17)
        enable_rl: bool = True,
        # Eastern Asia Gold Standard (2026-01-17)
        enable_moe: bool = True,           # Mixture of Experts (MIGA - 24% excess return)
        enable_gnn: bool = True,           # Graph Neural Network features
        enable_korea: bool = True,         # Korea Quant (MS-GARCH, CGMY)
        enable_asia_fx: bool = True,       # Asia FX Spread (CNH-CNY, HKD peg)
        enable_marl: bool = True,          # Multi-Agent RL
        alpha360_lookback: int = 20,
        verbose: bool = False
    ):
        """
        Initialize MegaFeatureGenerator.

        Args:
            enable_*: Enable/disable specific feature sets
            alpha360_lookback: Lookback for Alpha360 (20=compact, 60=full)
            verbose: Print progress
        """
        self.verbose = verbose
        self.enable_flags = {
            'alpha101': enable_alpha101,
            'alpha158': enable_alpha158,
            'alpha360': enable_alpha360,
            'barra': enable_barra,
            'alpha191': enable_alpha191,
            'chinese_hft': enable_chinese_hft,
            'chinese_additions': enable_chinese_additions,
            'chinese_gold': enable_chinese_gold,
            'us_academic': enable_us_academic,
            'mlfinlab': enable_mlfinlab,
            'timeseries': enable_timeseries,
            'renaissance': enable_renaissance,
            # Global expansion (2026-01-17)
            'india': enable_india,
            'japan': enable_japan,
            'europe': enable_europe,
            'emerging': enable_emerging,
            'universal_math': enable_universal_math,
            # Reinforcement Learning (2026-01-17)
            'rl': enable_rl,
            # Eastern Asia Gold Standard (2026-01-17)
            'moe': enable_moe,
            'gnn': enable_gnn,
            'korea': enable_korea,
            'asia_fx': enable_asia_fx,
            'marl': enable_marl,
        }
        self.alpha360_lookback = alpha360_lookback

        # Lazy load generators (avoids import errors if not enabled)
        self._generators = {}

    def _get_generator(self, name: str):
        """Lazy load generator to avoid import issues."""
        if name not in self._generators:
            if name == 'alpha101':
                from .alpha101 import Alpha101Complete
                self._generators[name] = Alpha101Complete()
            elif name == 'alpha158':
                from .alpha158 import Alpha158
                self._generators[name] = Alpha158()
            elif name == 'alpha360':
                from .alpha360 import Alpha360Compact
                self._generators[name] = Alpha360Compact(lookback=self.alpha360_lookback)
            elif name == 'barra':
                from .barra_cne6 import BarraCNE6Forex
                self._generators[name] = BarraCNE6Forex()
            elif name == 'alpha191':
                from core._experimental.alpha191_guotaijunan import Alpha191GuotaiJunan
                self._generators[name] = Alpha191GuotaiJunan()
            elif name == 'chinese_hft':
                from core._experimental.chinese_hft_factors import ChineseHFTFactorEngine
                self._generators[name] = ChineseHFTFactorEngine()
            elif name == 'chinese_additions':
                self._generators[name] = ChineseHFTAdditions()
            elif name == 'chinese_gold':
                from .chinese_gold_standard import ChineseGoldStandardFeatures
                self._generators[name] = ChineseGoldStandardFeatures()
            elif name == 'us_academic':
                from .us_academic_factors import USAcademicFactors
                self._generators[name] = USAcademicFactors()
            elif name == 'mlfinlab':
                from .mlfinlab_features import MLFinLabFeatures
                self._generators[name] = MLFinLabFeatures()
            elif name == 'timeseries':
                from .timeseries_features import TimeSeriesLibraryFeatures
                self._generators[name] = TimeSeriesLibraryFeatures()
            elif name == 'renaissance':
                from .renaissance import RenaissanceSignalGenerator
                self._generators[name] = RenaissanceSignalGenerator()
            # Global expansion (2026-01-17)
            elif name == 'india':
                from .india_quant import IndiaQuantFeatures
                self._generators[name] = IndiaQuantFeatures()
            elif name == 'japan':
                from .japan_quant import JapanQuantFeatures
                self._generators[name] = JapanQuantFeatures()
            elif name == 'europe':
                from .europe_quant import EuropeQuantFeatures
                self._generators[name] = EuropeQuantFeatures()
            elif name == 'emerging':
                from .emerging_quant import EmergingMarketsFeatures
                self._generators[name] = EmergingMarketsFeatures()
            elif name == 'universal_math':
                from .universal_math import UniversalMathFeatures
                self._generators[name] = UniversalMathFeatures()
            # Reinforcement Learning (2026-01-17)
            elif name == 'rl':
                from .rl_algorithms import RLFeatureGenerator
                self._generators[name] = RLFeatureGenerator()
            # Eastern Asia Gold Standard (2026-01-17)
            elif name == 'moe':
                from .moe_trading import MixtureOfExpertsFeatures
                self._generators[name] = MixtureOfExpertsFeatures()
            elif name == 'gnn':
                from .gnn_temporal import TemporalGNNFeatures
                self._generators[name] = TemporalGNNFeatures()
            elif name == 'korea':
                from .korea_quant import KoreaQuantFeatures
                self._generators[name] = KoreaQuantFeatures()
            elif name == 'asia_fx':
                from .asia_fx_spread import AsiaFXSpreadFeatures
                self._generators[name] = AsiaFXSpreadFeatures()
            elif name == 'marl':
                from .marl_trading import MultiAgentRLFeatures
                self._generators[name] = MultiAgentRLFeatures()
        return self._generators.get(name)

    def _log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[MegaFeatureGenerator] {msg}", flush=True)

    def generate_all(
        self,
        df: pd.DataFrame,
        prefix_features: bool = True
    ) -> pd.DataFrame:
        """
        Generate ALL 806+ features.

        Args:
            df: DataFrame with OHLCV data
                Required columns: open, high, low, close, volume
                Optional: vwap, returns, bid, ask, price
            prefix_features: Add source prefix to feature names (e.g., ALPHA101_*)

        Returns:
            DataFrame with all features (NaN/inf cleaned)
        """
        # Validate input
        df = self._prepare_dataframe(df)

        feature_sets = []
        feature_counts = {}

        # 1. Chinese Quant - Alpha101 (62 features)
        if self.enable_flags['alpha101']:
            self._log("Generating Alpha101 (62 features)...")
            try:
                gen = self._get_generator('alpha101')
                alpha101_features = gen.generate_all_alphas(df)
                if prefix_features:
                    alpha101_features.columns = [f'ALPHA101_{c}' for c in alpha101_features.columns]
                feature_sets.append(alpha101_features)
                feature_counts['alpha101'] = len(alpha101_features.columns)
            except Exception as e:
                logger.warning(f"Alpha101 failed: {e}")
                feature_counts['alpha101'] = 0

        # 2. Chinese Quant - Alpha158 (179 features)
        if self.enable_flags['alpha158']:
            self._log("Generating Alpha158 (179 features)...")
            try:
                gen = self._get_generator('alpha158')
                alpha158_features = gen.generate_all(df)
                if prefix_features:
                    alpha158_features.columns = [f'ALPHA158_{c}' for c in alpha158_features.columns]
                feature_sets.append(alpha158_features)
                feature_counts['alpha158'] = len(alpha158_features.columns)
            except Exception as e:
                logger.warning(f"Alpha158 failed: {e}")
                feature_counts['alpha158'] = 0

        # 3. Chinese Quant - Alpha360 Compact (200 features with lookback=20, 360 with lookback=60)
        if self.enable_flags['alpha360']:
            expected = self.alpha360_lookback * 6  # 6 price fields
            self._log(f"Generating Alpha360 ({expected} features)...")
            try:
                gen = self._get_generator('alpha360')
                alpha360_features = gen.generate_all(df)
                if prefix_features:
                    alpha360_features.columns = [f'ALPHA360_{c}' for c in alpha360_features.columns]
                feature_sets.append(alpha360_features)
                feature_counts['alpha360'] = len(alpha360_features.columns)
            except Exception as e:
                logger.warning(f"Alpha360 failed: {e}")
                feature_counts['alpha360'] = 0

        # 4. Chinese Quant - Barra CNE6 (46 features)
        if self.enable_flags['barra']:
            self._log("Generating Barra CNE6 (46 features)...")
            try:
                gen = self._get_generator('barra')
                barra_features = gen.generate_all(df)
                if prefix_features:
                    barra_features.columns = [f'BARRA_{c}' for c in barra_features.columns]
                feature_sets.append(barra_features)
                feature_counts['barra'] = len(barra_features.columns)
            except Exception as e:
                logger.warning(f"Barra CNE6 failed: {e}")
                feature_counts['barra'] = 0

        # 5. Chinese Quant - Alpha191 Guotai Junan (191 features)
        if self.enable_flags['alpha191']:
            self._log("Generating Alpha191 Guotai Junan (191 features)...")
            try:
                gen = self._get_generator('alpha191')
                alpha191_features = gen.generate_all_alphas(df)
                if prefix_features:
                    alpha191_features.columns = [f'ALPHA191_{c}' for c in alpha191_features.columns]
                feature_sets.append(alpha191_features)
                feature_counts['alpha191'] = len(alpha191_features.columns)
            except Exception as e:
                logger.warning(f"Alpha191 failed: {e}")
                feature_counts['alpha191'] = 0

        # 6. Chinese HFT Factors (50+ features)
        if self.enable_flags['chinese_hft']:
            self._log("Generating Chinese HFT Factors (50+ features)...")
            try:
                gen = self._get_generator('chinese_hft')
                chinese_hft_features = gen.generate_all_factors(df)
                if prefix_features:
                    chinese_hft_features.columns = [f'CNHFT_{c}' for c in chinese_hft_features.columns]
                feature_sets.append(chinese_hft_features)
                feature_counts['chinese_hft'] = len(chinese_hft_features.columns)
            except Exception as e:
                logger.warning(f"Chinese HFT failed: {e}")
                feature_counts['chinese_hft'] = 0

        # 7. Chinese HFT Additions - RSRS, Yang-Zhang, Hawkes (25+ features)
        if self.enable_flags['chinese_additions']:
            self._log("Generating Chinese HFT Additions (25+ features)...")
            try:
                gen = self._get_generator('chinese_additions')
                additions_features = gen.generate_all(df)
                if prefix_features:
                    additions_features.columns = [f'CNADD_{c}' for c in additions_features.columns]
                feature_sets.append(additions_features)
                feature_counts['chinese_additions'] = len(additions_features.columns)
            except Exception as e:
                logger.warning(f"Chinese HFT Additions failed: {e}")
                feature_counts['chinese_additions'] = 0

        # 8. Chinese Gold Standard - VPIN, OFI, Meta-Labeling, HMM, Kalman (35+ features)
        if self.enable_flags['chinese_gold']:
            self._log("Generating Chinese Gold Standard (35+ features)...")
            try:
                gen = self._get_generator('chinese_gold')
                gold_features = gen.generate_all(df)
                if prefix_features:
                    gold_features.columns = [f'CNGOLD_{c}' for c in gold_features.columns]
                feature_sets.append(gold_features)
                feature_counts['chinese_gold'] = len(gold_features.columns)
            except Exception as e:
                logger.warning(f"Chinese Gold Standard failed: {e}")
                feature_counts['chinese_gold'] = 0

        # 9. US Academic Factors (50 features)
        if self.enable_flags['us_academic']:
            self._log("Generating US Academic Factors (50 features)...")
            try:
                gen = self._get_generator('us_academic')
                us_features = gen.generate_all(df)
                if prefix_features:
                    us_features.columns = [f'USACAD_{c}' for c in us_features.columns]
                feature_sets.append(us_features)
                feature_counts['us_academic'] = len(us_features.columns)
            except Exception as e:
                logger.warning(f"US Academic failed: {e}")
                feature_counts['us_academic'] = 0

        # 10. MLFinLab Features (17 features)
        if self.enable_flags['mlfinlab']:
            self._log("Generating MLFinLab (17 features)...")
            try:
                gen = self._get_generator('mlfinlab')
                mlfinlab_features = gen.generate_all(df)
                if prefix_features:
                    mlfinlab_features.columns = [f'MLFINLAB_{c}' for c in mlfinlab_features.columns]
                feature_sets.append(mlfinlab_features)
                feature_counts['mlfinlab'] = len(mlfinlab_features.columns)
            except Exception as e:
                logger.warning(f"MLFinLab failed: {e}")
                feature_counts['mlfinlab'] = 0

        # 11. Time-Series-Library Features (41 features)
        if self.enable_flags['timeseries']:
            self._log("Generating Time-Series-Library (41 features)...")
            try:
                gen = self._get_generator('timeseries')
                ts_features = gen.generate_all(df)
                if prefix_features:
                    ts_features.columns = [f'TSLIB_{c}' for c in ts_features.columns]
                feature_sets.append(ts_features)
                feature_counts['timeseries'] = len(ts_features.columns)
            except Exception as e:
                logger.warning(f"Time-Series-Library failed: {e}")
                feature_counts['timeseries'] = 0

        # 12. Renaissance Signals (51 features)
        if self.enable_flags['renaissance']:
            self._log("Generating Renaissance Signals (51 features)...")
            try:
                gen = self._get_generator('renaissance')
                # Renaissance adds signal_ columns to the dataframe
                df_with_signals = gen.generate_all_signals(df.copy())
                signal_cols = [c for c in df_with_signals.columns if c.startswith('signal_')]
                ren_features = df_with_signals[signal_cols].copy()
                if prefix_features:
                    ren_features.columns = [f'REN_{c}' for c in ren_features.columns]
                feature_sets.append(ren_features)
                feature_counts['renaissance'] = len(ren_features.columns)
            except Exception as e:
                logger.warning(f"Renaissance failed: {e}")
                feature_counts['renaissance'] = 0

        # =========================================================================
        # GLOBAL EXPANSION (2026-01-17) - 110 NEW FEATURES
        # =========================================================================

        # 13. India Quant Features (25 features)
        if self.enable_flags['india']:
            self._log("Generating India Quant (25 features)...")
            try:
                gen = self._get_generator('india')
                india_features = gen.generate_all(df)
                if prefix_features:
                    india_features.columns = [f'INDIA_{c}' for c in india_features.columns]
                feature_sets.append(india_features)
                feature_counts['india'] = len(india_features.columns)
            except Exception as e:
                logger.warning(f"India Quant failed: {e}")
                feature_counts['india'] = 0

        # 14. Japan Quant Features (20 features)
        if self.enable_flags['japan']:
            self._log("Generating Japan Quant (20 features)...")
            try:
                gen = self._get_generator('japan')
                japan_features = gen.generate_all(df)
                if prefix_features:
                    japan_features.columns = [f'JAPAN_{c}' for c in japan_features.columns]
                feature_sets.append(japan_features)
                feature_counts['japan'] = len(japan_features.columns)
            except Exception as e:
                logger.warning(f"Japan Quant failed: {e}")
                feature_counts['japan'] = 0

        # 15. Europe Quant Features (15 features)
        if self.enable_flags['europe']:
            self._log("Generating Europe Quant (15 features)...")
            try:
                gen = self._get_generator('europe')
                europe_features = gen.generate_all(df)
                if prefix_features:
                    europe_features.columns = [f'EUROPE_{c}' for c in europe_features.columns]
                feature_sets.append(europe_features)
                feature_counts['europe'] = len(europe_features.columns)
            except Exception as e:
                logger.warning(f"Europe Quant failed: {e}")
                feature_counts['europe'] = 0

        # 16. Emerging Markets Features (20 features)
        if self.enable_flags['emerging']:
            self._log("Generating Emerging Markets (20 features)...")
            try:
                gen = self._get_generator('emerging')
                emerging_features = gen.generate_all(df)
                if prefix_features:
                    emerging_features.columns = [f'EM_{c}' for c in emerging_features.columns]
                feature_sets.append(emerging_features)
                feature_counts['emerging'] = len(emerging_features.columns)
            except Exception as e:
                logger.warning(f"Emerging Markets failed: {e}")
                feature_counts['emerging'] = 0

        # 17. Universal Math Features (30 features)
        if self.enable_flags['universal_math']:
            self._log("Generating Universal Math (30 features)...")
            try:
                gen = self._get_generator('universal_math')
                math_features = gen.generate_all(df)
                if prefix_features:
                    math_features.columns = [f'MATH_{c}' for c in math_features.columns]
                feature_sets.append(math_features)
                feature_counts['universal_math'] = len(math_features.columns)
            except Exception as e:
                logger.warning(f"Universal Math failed: {e}")
                feature_counts['universal_math'] = 0

        # =========================================================================
        # REINFORCEMENT LEARNING ALGORITHMS (2026-01-17) - 35 NEW FEATURES
        # Pure mathematical RL without neural networks
        # =========================================================================

        # 18. RL Algorithm Features (35 features)
        if self.enable_flags['rl']:
            self._log("Generating RL Algorithms (35 features)...")
            try:
                gen = self._get_generator('rl')
                rl_features = gen.generate_all(df)
                if prefix_features:
                    # RL features already prefixed with RL_, just add module prefix
                    rl_features.columns = [f'RLALGO_{c}' if not c.startswith('RL_') else c for c in rl_features.columns]
                feature_sets.append(rl_features)
                feature_counts['rl'] = len(rl_features.columns)
            except Exception as e:
                logger.warning(f"RL Algorithms failed: {e}")
                feature_counts['rl'] = 0

        # =========================================================================
        # EASTERN ASIA GOLD STANDARD (2026-01-17) - 85 NEW FEATURES
        # MOE (24% excess return), GNN, Korea Quant, Asia FX, MARL
        # =========================================================================

        # 19. MOE (Mixture of Experts) Features (20 features)
        if self.enable_flags['moe']:
            self._log("Generating MOE Trading (20 features - 24% excess return)...")
            try:
                gen = self._get_generator('moe')
                moe_features = gen.generate_all(df)
                if prefix_features:
                    moe_features.columns = [f'MOE_{c}' if not c.startswith('MOE_') else c for c in moe_features.columns]
                feature_sets.append(moe_features)
                feature_counts['moe'] = len(moe_features.columns)
            except Exception as e:
                logger.warning(f"MOE Trading failed: {e}")
                feature_counts['moe'] = 0

        # 20. GNN Temporal Features (15 features)
        if self.enable_flags['gnn']:
            self._log("Generating GNN Temporal (15 features)...")
            try:
                gen = self._get_generator('gnn')
                gnn_features = gen.generate_all(df)
                if prefix_features:
                    gnn_features.columns = [f'GNN_{c}' if not c.startswith('GNN_') else c for c in gnn_features.columns]
                feature_sets.append(gnn_features)
                feature_counts['gnn'] = len(gnn_features.columns)
            except Exception as e:
                logger.warning(f"GNN Temporal failed: {e}")
                feature_counts['gnn'] = 0

        # 21. Korea Quant Features (20 features)
        if self.enable_flags['korea']:
            self._log("Generating Korea Quant (20 features)...")
            try:
                gen = self._get_generator('korea')
                korea_features = gen.generate_all(df)
                if prefix_features:
                    korea_features.columns = [f'KOREA_{c}' if not c.startswith('KR_') else c for c in korea_features.columns]
                feature_sets.append(korea_features)
                feature_counts['korea'] = len(korea_features.columns)
            except Exception as e:
                logger.warning(f"Korea Quant failed: {e}")
                feature_counts['korea'] = 0

        # 22. Asia FX Spread Features (15 features)
        if self.enable_flags['asia_fx']:
            self._log("Generating Asia FX Spread (15 features)...")
            try:
                gen = self._get_generator('asia_fx')
                asia_features = gen.generate_all(df)
                if prefix_features:
                    asia_features.columns = [f'ASIAFX_{c}' if not c.startswith('ASIA_') else c for c in asia_features.columns]
                feature_sets.append(asia_features)
                feature_counts['asia_fx'] = len(asia_features.columns)
            except Exception as e:
                logger.warning(f"Asia FX Spread failed: {e}")
                feature_counts['asia_fx'] = 0

        # 23. MARL (Multi-Agent RL) Features (15 features)
        if self.enable_flags['marl']:
            self._log("Generating MARL Trading (15 features)...")
            try:
                gen = self._get_generator('marl')
                marl_features = gen.generate_all(df)
                if prefix_features:
                    marl_features.columns = [f'MARL_{c}' if not c.startswith('MARL_') else c for c in marl_features.columns]
                feature_sets.append(marl_features)
                feature_counts['marl'] = len(marl_features.columns)
            except Exception as e:
                logger.warning(f"MARL Trading failed: {e}")
                feature_counts['marl'] = 0

        # Combine all features
        if not feature_sets:
            raise ValueError("No features generated. Check enable flags.")

        combined = pd.concat(feature_sets, axis=1)

        # Clean up
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(0)

        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # Log summary
        total = len(combined.columns)
        self._log(f"Total features generated: {total}")
        if self.verbose:
            for name, count in feature_counts.items():
                print(f"  - {name}: {count}")

        return combined

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame with required columns.

        Ensures: open, high, low, close, volume, returns, vwap, price
        Returns ONLY the required OHLCV columns - excludes any target/feature columns.
        """
        # Required column
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        # Create minimal OHLCV dataframe (exclude all other columns to avoid leakage)
        ohlcv = pd.DataFrame(index=df.index)
        ohlcv['close'] = df['close'].values

        # Create open/high/low from close if not in original
        if 'open' in df.columns:
            ohlcv['open'] = df['open'].values
        else:
            ohlcv['open'] = df['close'].shift(1).fillna(df['close']).values

        if 'high' in df.columns:
            ohlcv['high'] = df['high'].values
        else:
            ohlcv['high'] = df['close'].values

        if 'low' in df.columns:
            ohlcv['low'] = df['low'].values
        else:
            ohlcv['low'] = df['close'].values

        # Volume
        if 'volume' in df.columns:
            ohlcv['volume'] = df['volume'].values
        else:
            ohlcv['volume'] = 1  # Tick count proxy

        # Returns
        ohlcv['returns'] = ohlcv['close'].pct_change()

        # VWAP
        if 'vwap' in df.columns:
            ohlcv['vwap'] = df['vwap'].values
        else:
            ohlcv['vwap'] = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3

        # Price (for Renaissance signals)
        if 'bid' in df.columns and 'ask' in df.columns:
            ohlcv['price'] = (df['bid'].values + df['ask'].values) / 2
            ohlcv['bid'] = df['bid'].values
            ohlcv['ask'] = df['ask'].values
        else:
            ohlcv['price'] = ohlcv['close']

        return ohlcv

    def get_feature_counts(self) -> Dict[str, int]:
        """Get expected feature counts for each library."""
        counts = {
            'alpha101': 62,
            'alpha158': 179,
            'alpha360': self.alpha360_lookback * 6,  # 6 price fields
            'barra': 46,
            'alpha191': 191,
            'chinese_hft': 50,
            'chinese_additions': 25,
            'chinese_gold': 35,  # VPIN, OFI, HMM, Kalman, etc.
            'us_academic': 50,
            'mlfinlab': 17,
            'timeseries': 41,
            'renaissance': 51,
            # Global expansion (2026-01-17)
            'india': 25,
            'japan': 30,  # Updated: +10 (HAR-RV, Power-Law, BOJ)
            'europe': 15,
            'emerging': 20,
            'universal_math': 30,
            # Reinforcement Learning (2026-01-17)
            'rl': 35,
            # Eastern Asia Gold Standard (2026-01-17)
            'moe': 20,     # Mixture of Experts (MIGA - 24% excess return)
            'gnn': 15,     # Graph Neural Network temporal
            'korea': 20,   # MS-GARCH, CGMY Lévy, VKOSPI
            'asia_fx': 15, # CNH-CNY spread, HKD peg
            'marl': 15,    # Multi-Agent RL
        }

        # Filter by enabled
        enabled_counts = {k: v for k, v in counts.items() if self.enable_flags.get(k, False)}
        enabled_counts['total'] = sum(enabled_counts.values())

        return enabled_counts

    def get_feature_names(self, prefix: bool = True) -> List[str]:
        """Get all feature names (without generating them)."""
        names = []

        if self.enable_flags['alpha101']:
            alpha_names = [f'alpha_{i:03d}' for i in range(1, 102)]
            if prefix:
                alpha_names = [f'ALPHA101_{n}' for n in alpha_names]
            names.extend(alpha_names)

        # Note: Exact names depend on generator implementation
        # This is an approximation

        return names


# Convenience functions

def generate_mega_features(
    df: pd.DataFrame,
    alpha360_lookback: int = 20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate all 1100+ features in one call.

    Args:
        df: OHLCV DataFrame
        alpha360_lookback: 20 for compact (120 features), 60 for full (360 features)
        verbose: Print progress

    Returns:
        DataFrame with 1100+ features
    """
    generator = MegaFeatureGenerator(
        alpha360_lookback=alpha360_lookback,
        verbose=verbose
    )
    return generator.generate_all(df)


def generate_chinese_quant_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Generate only Chinese Quant features (800+)."""
    generator = MegaFeatureGenerator(
        enable_alpha101=True,
        enable_alpha158=True,
        enable_alpha360=True,
        enable_barra=True,
        enable_alpha191=True,
        enable_chinese_hft=True,
        enable_chinese_additions=True,
        enable_chinese_gold=True,
        enable_us_academic=False,
        enable_mlfinlab=False,
        enable_timeseries=False,
        enable_renaissance=False,
        verbose=verbose
    )
    return generator.generate_all(df)


def generate_academic_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Generate USA Academic + MLFinLab features (67)."""
    generator = MegaFeatureGenerator(
        enable_alpha101=False,
        enable_alpha158=False,
        enable_alpha360=False,
        enable_barra=False,
        enable_alpha191=False,
        enable_chinese_hft=False,
        enable_chinese_additions=False,
        enable_chinese_gold=False,
        enable_us_academic=True,
        enable_mlfinlab=True,
        enable_timeseries=False,
        enable_renaissance=False,
        verbose=verbose
    )
    return generator.generate_all(df)


def generate_renaissance_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Generate Renaissance weak signals (51)."""
    generator = MegaFeatureGenerator(
        enable_alpha101=False,
        enable_alpha158=False,
        enable_alpha360=False,
        enable_barra=False,
        enable_alpha191=False,
        enable_chinese_hft=False,
        enable_chinese_additions=False,
        enable_chinese_gold=False,
        enable_us_academic=False,
        enable_mlfinlab=False,
        enable_timeseries=False,
        enable_renaissance=True,
        verbose=verbose
    )
    return generator.generate_all(df)
