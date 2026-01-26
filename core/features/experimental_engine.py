"""
Experimental Feature Engine - Integrates ALL experimental modules
=================================================================
Combines 150+ features from core/_experimental/ modules:

1. Kalman Filters (10 features)
   - Extended Kalman Filter
   - Unscented Kalman Filter
   - Adaptive Kalman

2. GARCH Models (5 features)
   - GARCH(1,1)
   - EGARCH
   - GJR-GARCH

3. Chinese HFT Factors (12 features)
   - Microprice
   - Smart Money
   - Kyle lambda

4. Attention Factors (7 features)
   - Stat arb (Sharpe 4.0+)
   - Cross-sectional attention

5. Market Impact (5 features)
   - Kyle model
   - Glosten-Milgrom
   - Hasbrouck

6. Order Flow (8 features)
   - OFI (Order Flow Imbalance)
   - OEI (Order Execution Imbalance)
   - VPIN
   - Hawkes

7. Jump Detection (3 features)
   - Lee-Mykland
   - Hawkes intensity

8. HAR-RV Volatility (3 features)
   - Daily, weekly, monthly components

9. Alpha191 Guotai Junan (30 top selections)
   - Chinese institutional factors

10. LOB Features (8 features)
    - Book imbalance
    - Depth ratio

11. Regime Features (5 features)
    - HMM states
    - Markov switching

12. Range Volatility (4 features)
    - Parkinson
    - Garman-Klass

13. Elite Quant Factors (20 features)
    - Selected from research papers

14. MyTT China Formulas (30+ features)
    - MACD, KDJ, RSI, BOLL, WR, BIAS
    - DMI, TRIX, VR, EMV, DPO, BRAR
    - ATR, CCI, PSY, TAQ, MTM, ROC

Total: 180+ experimental features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentalConfig:
    """Configuration for experimental feature engine."""
    enable_kalman: bool = True
    enable_garch: bool = True
    enable_chinese_hft: bool = True
    enable_attention: bool = True
    enable_market_impact: bool = True
    enable_order_flow: bool = True
    enable_jump_detection: bool = True
    enable_har_rv: bool = True
    enable_alpha191: bool = True
    enable_lob: bool = True
    enable_regime: bool = True
    enable_range_vol: bool = True
    enable_elite: bool = True
    enable_mytt: bool = True
    verbose: bool = False


class ExperimentalFeatureEngine:
    """
    Unified engine for all experimental features.

    Usage:
        engine = ExperimentalFeatureEngine()
        features = engine.generate_all(df)
    """

    def __init__(
        self,
        enable_kalman: bool = True,
        enable_garch: bool = True,
        enable_chinese_hft: bool = True,
        enable_attention: bool = True,
        enable_market_impact: bool = True,
        enable_order_flow: bool = True,
        enable_jump: bool = True,
        enable_har_rv: bool = True,
        enable_alpha191: bool = True,
        enable_lob: bool = True,
        enable_regime: bool = True,
        enable_range_vol: bool = True,
        enable_elite: bool = True,
        enable_mytt: bool = True,
        verbose: bool = False,
    ):
        self.enable_flags = {
            'kalman': enable_kalman,
            'garch': enable_garch,
            'chinese_hft': enable_chinese_hft,
            'attention': enable_attention,
            'market_impact': enable_market_impact,
            'order_flow': enable_order_flow,
            'jump': enable_jump,
            'har_rv': enable_har_rv,
            'alpha191': enable_alpha191,
            'lob': enable_lob,
            'regime': enable_regime,
            'range_vol': enable_range_vol,
            'elite': enable_elite,
            'mytt': enable_mytt,
        }
        self.verbose = verbose
        self._generators = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ExperimentalEngine] {msg}")

    def _get_generator(self, name: str):
        """Lazy load experimental generators."""
        if name not in self._generators:
            try:
                if name == 'kalman':
                    from core._experimental.advanced_kalman_filters import KalmanFilterFeatures
                    self._generators[name] = KalmanFilterFeatures()
                elif name == 'garch':
                    from core._experimental.arima_garch_models import GARCHFeatures
                    self._generators[name] = GARCHFeatures()
                elif name == 'chinese_hft':
                    from core._experimental.chinese_hft_factors import ChineseHFTFactorEngine
                    self._generators[name] = ChineseHFTFactorEngine()
                elif name == 'attention':
                    from core._experimental.attention_factors import AttentionFactorEngine
                    self._generators[name] = AttentionFactorEngine()
                elif name == 'market_impact':
                    from core._experimental.market_impact_models import MarketImpactFeatures
                    self._generators[name] = MarketImpactFeatures()
                elif name == 'order_flow':
                    from core._experimental.order_flow_features import OrderFlowEngine
                    self._generators[name] = OrderFlowEngine()
                elif name == 'jump':
                    from core._experimental.jump_detection import JumpDetectionFeatures
                    self._generators[name] = JumpDetectionFeatures()
                elif name == 'har_rv':
                    from core._experimental.har_rv_volatility import HARRVFeatures
                    self._generators[name] = HARRVFeatures()
                elif name == 'alpha191':
                    from core._experimental.alpha191_guotaijunan import Alpha191GuotaiJunan
                    self._generators[name] = Alpha191GuotaiJunan()
                elif name == 'lob':
                    from core._experimental.lob_features import LOBFeatures
                    self._generators[name] = LOBFeatures()
                elif name == 'regime':
                    from core._experimental.regime_features import RegimeFeatures
                    self._generators[name] = RegimeFeatures()
                elif name == 'range_vol':
                    from core._experimental.range_volatility import RangeVolatilityFeatures
                    self._generators[name] = RangeVolatilityFeatures()
                elif name == 'elite':
                    from core._experimental.elite_quant_factors import EliteQuantFactors
                    self._generators[name] = EliteQuantFactors()
            except ImportError as e:
                logger.warning(f"Could not import {name}: {e}")
                self._generators[name] = None
            except Exception as e:
                logger.warning(f"Error loading {name}: {e}")
                self._generators[name] = None

        return self._generators.get(name)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all experimental features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with experimental features
        """
        feature_sets = []
        feature_counts = {}

        # Ensure required columns
        df = self._prepare_dataframe(df)

        # 1. Kalman Filters
        if self.enable_flags['kalman']:
            self._log("Generating Kalman Filter features...")
            features = self._generate_kalman(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['kalman'] = len(features.columns)

        # 2. GARCH Models
        if self.enable_flags['garch']:
            self._log("Generating GARCH features...")
            features = self._generate_garch(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['garch'] = len(features.columns)

        # 3. Chinese HFT Factors
        if self.enable_flags['chinese_hft']:
            self._log("Generating Chinese HFT features...")
            features = self._generate_chinese_hft(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['chinese_hft'] = len(features.columns)

        # 4. Attention Factors
        if self.enable_flags['attention']:
            self._log("Generating Attention features...")
            features = self._generate_attention(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['attention'] = len(features.columns)

        # 5. Market Impact
        if self.enable_flags['market_impact']:
            self._log("Generating Market Impact features...")
            features = self._generate_market_impact(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['market_impact'] = len(features.columns)

        # 6. Order Flow
        if self.enable_flags['order_flow']:
            self._log("Generating Order Flow features...")
            features = self._generate_order_flow(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['order_flow'] = len(features.columns)

        # 7. Jump Detection
        if self.enable_flags['jump']:
            self._log("Generating Jump Detection features...")
            features = self._generate_jump(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['jump'] = len(features.columns)

        # 8. HAR-RV
        if self.enable_flags['har_rv']:
            self._log("Generating HAR-RV features...")
            features = self._generate_har_rv(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['har_rv'] = len(features.columns)

        # 9. Alpha191
        if self.enable_flags['alpha191']:
            self._log("Generating Alpha191 features...")
            features = self._generate_alpha191(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['alpha191'] = len(features.columns)

        # 10. LOB Features
        if self.enable_flags['lob']:
            self._log("Generating LOB features...")
            features = self._generate_lob(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['lob'] = len(features.columns)

        # 11. Regime Features
        if self.enable_flags['regime']:
            self._log("Generating Regime features...")
            features = self._generate_regime(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['regime'] = len(features.columns)

        # 12. Range Volatility
        if self.enable_flags['range_vol']:
            self._log("Generating Range Volatility features...")
            features = self._generate_range_vol(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['range_vol'] = len(features.columns)

        # 13. Elite Quant Factors
        if self.enable_flags['elite']:
            self._log("Generating Elite Quant features...")
            features = self._generate_elite(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['elite'] = len(features.columns)

        # 14. MyTT China Formulas
        if self.enable_flags['mytt']:
            self._log("Generating MyTT China Formula features...")
            features = self._generate_mytt(df)
            if features is not None and len(features.columns) > 0:
                feature_sets.append(features)
                feature_counts['mytt'] = len(features.columns)

        # Combine all
        if not feature_sets:
            logger.warning("No experimental features generated")
            return pd.DataFrame(index=df.index)

        combined = pd.concat(feature_sets, axis=1)

        # Clean up
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(0)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        total = len(combined.columns)
        self._log(f"Total experimental features: {total}")
        if self.verbose:
            for name, count in feature_counts.items():
                print(f"  - {name}: {count}")

        return combined

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with required columns."""
        ohlcv = pd.DataFrame(index=df.index)

        ohlcv['close'] = df['close'].values
        ohlcv['open'] = df.get('open', df['close'].shift(1).fillna(df['close'])).values
        ohlcv['high'] = df.get('high', df['close']).values
        ohlcv['low'] = df.get('low', df['close']).values
        ohlcv['volume'] = df.get('volume', pd.Series(1, index=df.index)).values
        ohlcv['returns'] = ohlcv['close'].pct_change().fillna(0)

        return ohlcv

    def _generate_kalman(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate Kalman filter features."""
        try:
            gen = self._get_generator('kalman')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_KAL_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Kalman generation failed: {e}")

        # Fallback: simple Kalman-like filter
        return self._kalman_fallback(df)

    def _kalman_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback Kalman-like features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']

        # Simple exponential smoothing as Kalman approximation
        for alpha in [0.1, 0.3, 0.5]:
            smoothed = close.ewm(alpha=alpha).mean()
            result[f'EXP_KAL_smooth_{int(alpha*10)}'] = smoothed
            result[f'EXP_KAL_error_{int(alpha*10)}'] = close - smoothed

        # Trend estimation
        result['EXP_KAL_trend'] = close.diff().ewm(span=20).mean()
        result['EXP_KAL_trend_var'] = close.diff().ewm(span=20).var()

        return result

    def _generate_garch(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate GARCH features."""
        try:
            gen = self._get_generator('garch')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_GARCH_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"GARCH generation failed: {e}")

        # Fallback: GARCH-like volatility
        return self._garch_fallback(df)

    def _garch_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback GARCH-like features."""
        result = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change().fillna(0)

        # EWMA volatility (GARCH-like)
        for span in [10, 20, 60]:
            vol = returns.ewm(span=span).std()
            result[f'EXP_GARCH_vol_{span}'] = vol

        # Volatility of volatility
        result['EXP_GARCH_vol_of_vol'] = result['EXP_GARCH_vol_20'].rolling(10).std()

        # Asymmetric volatility (leverage effect)
        neg_returns = returns.where(returns < 0, 0)
        result['EXP_GARCH_neg_vol'] = neg_returns.ewm(span=20).std()

        return result

    def _generate_chinese_hft(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate Chinese HFT features."""
        try:
            gen = self._get_generator('chinese_hft')
            if gen is not None:
                features = gen.generate_all_factors(df)
                features.columns = [f'EXP_CNHFT_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Chinese HFT generation failed: {e}")

        return self._chinese_hft_fallback(df)

    def _chinese_hft_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback Chinese HFT-like features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Microprice
        result['EXP_CNHFT_microprice'] = (high + low + close) / 3

        # Volume clock
        vol_ma = volume.rolling(20).mean()
        result['EXP_CNHFT_vol_clock'] = volume / (vol_ma + 1e-10)

        # Smart money indicator
        range_ = high - low + 1e-10
        result['EXP_CNHFT_smart_money'] = (close - low) / range_ - (high - close) / range_

        # Kyle lambda approximation
        returns = close.pct_change().fillna(0)
        vol_change = volume.pct_change().fillna(0)
        result['EXP_CNHFT_kyle_lambda'] = returns.rolling(20).std() / (vol_change.rolling(20).std() + 1e-10)

        return result

    def _generate_attention(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate attention-based features."""
        try:
            gen = self._get_generator('attention')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_ATT_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Attention generation failed: {e}")

        return self._attention_fallback(df)

    def _attention_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback attention-like features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change().fillna(0)

        # Self-attention proxy (correlation with lagged self)
        for lag in [1, 5, 10]:
            result[f'EXP_ATT_self_corr_{lag}'] = returns.rolling(20).corr(returns.shift(lag))

        # Cross-window attention
        short_mean = close.rolling(5).mean()
        long_mean = close.rolling(20).mean()
        result['EXP_ATT_cross_window'] = short_mean / (long_mean + 1e-10) - 1

        return result

    def _generate_market_impact(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate market impact features."""
        try:
            gen = self._get_generator('market_impact')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_MI_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Market impact generation failed: {e}")

        return self._market_impact_fallback(df)

    def _market_impact_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback market impact features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']
        returns = close.pct_change().fillna(0)

        # Kyle lambda (price impact per unit volume)
        vol_ma = volume.rolling(20).mean()
        result['EXP_MI_kyle_lambda'] = returns.abs() / (volume / (vol_ma + 1e-10) + 1e-10)

        # Amihud illiquidity
        result['EXP_MI_amihud'] = returns.abs() / (volume + 1e-10)
        result['EXP_MI_amihud_20'] = result['EXP_MI_amihud'].rolling(20).mean()

        return result

    def _generate_order_flow(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate order flow features."""
        try:
            gen = self._get_generator('order_flow')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_OF_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Order flow generation failed: {e}")

        return self._order_flow_fallback(df)

    def _order_flow_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback order flow features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        returns = close.pct_change().fillna(0)

        # Order Flow Imbalance (OFI)
        buy_vol = volume.where(returns > 0, 0)
        sell_vol = volume.where(returns < 0, 0)
        result['EXP_OF_ofi'] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)
        result['EXP_OF_ofi_5'] = result['EXP_OF_ofi'].rolling(5).mean()
        result['EXP_OF_ofi_20'] = result['EXP_OF_ofi'].rolling(20).mean()

        # Trade direction (tick rule)
        tick_dir = np.sign(close.diff())
        result['EXP_OF_tick_dir'] = tick_dir.rolling(10).sum() / 10

        return result

    def _generate_jump(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate jump detection features."""
        try:
            gen = self._get_generator('jump')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_JUMP_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Jump detection failed: {e}")

        return self._jump_fallback(df)

    def _jump_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback jump detection features."""
        result = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change().fillna(0)

        # Z-score of returns
        mu = returns.rolling(20).mean()
        sigma = returns.rolling(20).std()
        z_score = (returns - mu) / (sigma + 1e-10)
        result['EXP_JUMP_zscore'] = z_score

        # Jump indicator
        result['EXP_JUMP_indicator'] = (z_score.abs() > 3).astype(float)
        result['EXP_JUMP_count'] = result['EXP_JUMP_indicator'].rolling(20).sum()

        return result

    def _generate_har_rv(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate HAR-RV features."""
        try:
            gen = self._get_generator('har_rv')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_HARRV_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"HAR-RV generation failed: {e}")

        return self._har_rv_fallback(df)

    def _har_rv_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback HAR-RV features."""
        result = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change().fillna(0)

        # Realized variance at different horizons
        rv_d = returns ** 2  # Daily component
        rv_w = rv_d.rolling(5).mean()  # Weekly component
        rv_m = rv_d.rolling(20).mean()  # Monthly component

        result['EXP_HARRV_daily'] = rv_d
        result['EXP_HARRV_weekly'] = rv_w
        result['EXP_HARRV_monthly'] = rv_m

        return result

    def _generate_alpha191(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate Alpha191 features."""
        try:
            gen = self._get_generator('alpha191')
            if gen is not None:
                features = gen.generate_all_alphas(df)
                # Take top 30
                features = features.iloc[:, :30]
                features.columns = [f'EXP_A191_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Alpha191 generation failed: {e}")

        return None

    def _generate_lob(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate LOB features."""
        try:
            gen = self._get_generator('lob')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_LOB_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"LOB generation failed: {e}")

        return self._lob_fallback(df)

    def _lob_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback LOB-like features."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # Price level imbalance
        range_ = high - low + 1e-10
        result['EXP_LOB_upper_ratio'] = (high - close) / range_
        result['EXP_LOB_lower_ratio'] = (close - low) / range_
        result['EXP_LOB_imbalance'] = result['EXP_LOB_lower_ratio'] - result['EXP_LOB_upper_ratio']

        return result

    def _generate_regime(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate regime features."""
        try:
            gen = self._get_generator('regime')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_REG_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Regime generation failed: {e}")

        return self._regime_fallback(df)

    def _regime_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback regime features."""
        result = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change().fillna(0)

        # Simple regime based on volatility
        vol = returns.rolling(20).std()
        vol_mean = vol.rolling(60).mean()
        result['EXP_REG_vol_regime'] = np.where(vol > vol_mean, 1, 0)

        # Trend regime
        ma_short = df['close'].rolling(10).mean()
        ma_long = df['close'].rolling(50).mean()
        result['EXP_REG_trend_regime'] = np.where(ma_short > ma_long, 1, 0)

        return result

    def _generate_range_vol(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate range volatility features."""
        try:
            gen = self._get_generator('range_vol')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_RV_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Range volatility generation failed: {e}")

        return self._range_vol_fallback(df)

    def _range_vol_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback range volatility features."""
        result = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']
        open_ = df['open']

        # Parkinson volatility
        log_hl = np.log(high / low)
        result['EXP_RV_parkinson'] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2)))

        # Garman-Klass volatility
        log_oc = np.log(close / open_)
        gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_oc ** 2)
        result['EXP_RV_garman_klass'] = np.sqrt(gk.rolling(20).mean())

        return result

    def _generate_elite(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate elite quant features."""
        try:
            gen = self._get_generator('elite')
            if gen is not None:
                features = gen.generate_all(df)
                features.columns = [f'EXP_ELITE_{c}' for c in features.columns]
                return features
        except Exception as e:
            logger.warning(f"Elite quant generation failed: {e}")

        return None

    def _generate_mytt(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate MyTT (麦语言) Chinese technical indicator features.

        MyTT is a popular Chinese technical analysis library implementing
        classic indicators from 通达信/同花顺 trading platforms.
        """
        try:
            # Try to import MyTT from libs/china_formulas/
            import sys
            from pathlib import Path
            mytt_path = Path(__file__).parent.parent.parent / 'libs' / 'china_formulas'
            if str(mytt_path) not in sys.path:
                sys.path.insert(0, str(mytt_path))

            from MyTT import (
                MACD, KDJ, RSI, BOLL, WR, BIAS, DMI, TRIX, VR, EMV,
                DPO, BRAR, DMA, MTM, ROC, CCI, ATR, BBI, PSY, TAQ,
                MA, EMA, HHV, LLV, STD
            )

            result = pd.DataFrame(index=df.index)
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_ = df['open'].values
            volume = df['volume'].values

            # MACD
            try:
                dif, dea, macd = MACD(close)
                result['EXP_MYTT_MACD_DIF'] = dif
                result['EXP_MYTT_MACD_DEA'] = dea
                result['EXP_MYTT_MACD_HIST'] = macd
            except:
                pass

            # KDJ
            try:
                k, d, j = KDJ(close, high, low)
                result['EXP_MYTT_KDJ_K'] = k
                result['EXP_MYTT_KDJ_D'] = d
                result['EXP_MYTT_KDJ_J'] = j
            except:
                pass

            # RSI at different periods
            for period in [6, 12, 24]:
                try:
                    rsi = RSI(close, period)
                    result[f'EXP_MYTT_RSI_{period}'] = rsi
                except:
                    pass

            # Bollinger Bands
            try:
                upper, mid, lower = BOLL(close)
                result['EXP_MYTT_BOLL_UPPER'] = upper
                result['EXP_MYTT_BOLL_MID'] = mid
                result['EXP_MYTT_BOLL_LOWER'] = lower
                # Normalized position within bands
                result['EXP_MYTT_BOLL_POS'] = (close - lower) / (upper - lower + 1e-10)
            except:
                pass

            # Williams %R
            try:
                wr, wr1 = WR(close, high, low)
                result['EXP_MYTT_WR'] = wr
                result['EXP_MYTT_WR1'] = wr1
            except:
                pass

            # BIAS
            try:
                bias1, bias2, bias3 = BIAS(close)
                result['EXP_MYTT_BIAS1'] = bias1
                result['EXP_MYTT_BIAS2'] = bias2
                result['EXP_MYTT_BIAS3'] = bias3
            except:
                pass

            # DMI
            try:
                pdi, mdi, adx, adxr = DMI(close, high, low)
                result['EXP_MYTT_DMI_PDI'] = pdi
                result['EXP_MYTT_DMI_MDI'] = mdi
                result['EXP_MYTT_DMI_ADX'] = adx
                result['EXP_MYTT_DMI_ADXR'] = adxr
            except:
                pass

            # TRIX
            try:
                trix, trma = TRIX(close)
                result['EXP_MYTT_TRIX'] = trix
                result['EXP_MYTT_TRMA'] = trma
            except:
                pass

            # VR
            try:
                vr = VR(close, volume)
                result['EXP_MYTT_VR'] = vr
            except:
                pass

            # EMV
            try:
                emv, maemv = EMV(high, low, volume)
                result['EXP_MYTT_EMV'] = emv
                result['EXP_MYTT_MAEMV'] = maemv
            except:
                pass

            # DPO
            try:
                dpo, madpo = DPO(close)
                result['EXP_MYTT_DPO'] = dpo
                result['EXP_MYTT_MADPO'] = madpo
            except:
                pass

            # BRAR
            try:
                ar, br = BRAR(open_, close, high, low)
                result['EXP_MYTT_AR'] = ar
                result['EXP_MYTT_BR'] = br
            except:
                pass

            # DMA
            try:
                dif_dma, difma = DMA(close)
                result['EXP_MYTT_DMA_DIF'] = dif_dma
                result['EXP_MYTT_DMA_DIFMA'] = difma
            except:
                pass

            # MTM
            try:
                mtm, mtmma = MTM(close)
                result['EXP_MYTT_MTM'] = mtm
                result['EXP_MYTT_MTMMA'] = mtmma
            except:
                pass

            # ROC
            try:
                roc, maroc = ROC(close)
                result['EXP_MYTT_ROC'] = roc
                result['EXP_MYTT_MAROC'] = maroc
            except:
                pass

            # CCI
            try:
                cci = CCI(close, high, low)
                result['EXP_MYTT_CCI'] = cci
            except:
                pass

            # ATR
            try:
                atr = ATR(close, high, low)
                result['EXP_MYTT_ATR'] = atr
            except:
                pass

            # BBI
            try:
                bbi = BBI(close)
                result['EXP_MYTT_BBI'] = bbi
            except:
                pass

            # PSY
            try:
                psy, psyma = PSY(close)
                result['EXP_MYTT_PSY'] = psy
                result['EXP_MYTT_PSYMA'] = psyma
            except:
                pass

            # TAQ (唐安奇通道)
            try:
                up, mid_taq, down = TAQ(high, low, 20)
                result['EXP_MYTT_TAQ_UP'] = up
                result['EXP_MYTT_TAQ_MID'] = mid_taq
                result['EXP_MYTT_TAQ_DOWN'] = down
            except:
                pass

            # Additional derived features
            # Moving averages
            for period in [5, 10, 20, 60]:
                try:
                    ma = MA(close, period)
                    result[f'EXP_MYTT_MA_{period}'] = ma
                    # Distance from MA
                    result[f'EXP_MYTT_MA_{period}_DIST'] = (close - ma) / (ma + 1e-10)
                except:
                    pass

            # EMA
            for period in [12, 26]:
                try:
                    ema = EMA(close, period)
                    result[f'EXP_MYTT_EMA_{period}'] = ema
                except:
                    pass

            # HHV/LLV (Highest/Lowest)
            for period in [5, 10, 20]:
                try:
                    hhv = HHV(close, period)
                    llv = LLV(close, period)
                    result[f'EXP_MYTT_HHV_{period}'] = hhv
                    result[f'EXP_MYTT_LLV_{period}'] = llv
                    # Normalized position
                    result[f'EXP_MYTT_RANGE_POS_{period}'] = (close - llv) / (hhv - llv + 1e-10)
                except:
                    pass

            logger.info(f"Generated {len(result.columns)} MyTT features")
            return result

        except ImportError as e:
            logger.warning(f"MyTT import failed: {e}")
            return self._mytt_fallback(df)
        except Exception as e:
            logger.warning(f"MyTT generation failed: {e}")
            return self._mytt_fallback(df)

    def _mytt_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback MyTT-like features when MyTT module unavailable."""
        result = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # Simple MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        result['EXP_MYTT_MACD_DIF'] = dif
        result['EXP_MYTT_MACD_DEA'] = dea
        result['EXP_MYTT_MACD_HIST'] = (dif - dea) * 2

        # Simple RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        result['EXP_MYTT_RSI_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        result['EXP_MYTT_BOLL_UPPER'] = sma20 + 2 * std20
        result['EXP_MYTT_BOLL_MID'] = sma20
        result['EXP_MYTT_BOLL_LOWER'] = sma20 - 2 * std20

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        result['EXP_MYTT_ATR'] = tr.rolling(14).mean()

        # Williams %R
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        result['EXP_MYTT_WR'] = -100 * (hh - close) / (hh - ll + 1e-10)

        return result


def generate_experimental_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Convenience function to generate all experimental features."""
    engine = ExperimentalFeatureEngine(verbose=verbose)
    return engine.generate_all(df)


def create_experimental_engine(
    config: Optional[ExperimentalConfig] = None,
    verbose: bool = False
) -> ExperimentalFeatureEngine:
    """
    Factory function to create experimental feature engine.

    Args:
        config: Experimental config (uses defaults if None)
        verbose: Enable verbose logging

    Returns:
        ExperimentalFeatureEngine instance
    """
    if config is None:
        config = ExperimentalConfig(verbose=verbose)

    engine = ExperimentalFeatureEngine(verbose=config.verbose)

    # Apply config settings
    engine.enable_kalman = config.enable_kalman
    engine.enable_garch = config.enable_garch
    engine.enable_chinese_hft = config.enable_chinese_hft
    engine.enable_attention = config.enable_attention
    engine.enable_market_impact = config.enable_market_impact
    engine.enable_order_flow = config.enable_order_flow
    engine.enable_jump_detection = config.enable_jump_detection
    engine.enable_har_rv = config.enable_har_rv
    engine.enable_alpha191 = config.enable_alpha191
    engine.enable_lob = config.enable_lob
    engine.enable_regime = config.enable_regime
    engine.enable_range_vol = config.enable_range_vol
    engine.enable_elite = config.enable_elite
    engine.enable_mytt = config.enable_mytt

    return engine


# Export
__all__ = [
    'ExperimentalFeatureEngine',
    'ExperimentalConfig',
    'create_experimental_engine',
    'generate_experimental_features',
]
