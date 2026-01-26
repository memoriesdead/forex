"""
HFT Feature Engineering Pipeline
================================
Unified feature generation combining all signal sources for HFT.

Signal Sources (Chinese RenTech + USA Renaissance):
- Alpha101 (WorldQuant) - 62+ alphas
- Alpha191 (国泰君安 Guotai Junan) - 191 short-period factors
- Renaissance Signals - 50+ weak signals
- Order Flow (OFI/OEI/VPIN) - Microstructure signals
- Cross-Asset - DXY, VIX, correlations
- Microstructure Vol - TSRV, noise filtering
- HAR-RV Volatility - Multi-scale volatility (Corsi 2009)
- Range-Based Volatility - Parkinson (1980), Garman-Klass (1980),
                           Rogers-Satchell (1991), Yang-Zhang (2000)
- Market Impact Models - Kyle (1985), Glosten-Milgrom (1985),
                         Roll (1984), Huang-Stoll (1997)
- Technical - Fast indicators for HFT

Renaissance Technologies Inferred Methods:
- Advanced Kalman Filters (EKF/UKF) - State estimation
- GARCH Family - Volatility forecasting
- Jump Detection (Bipower Variation) - News vs continuous trading
- Regime Features (HMM) - Market state detection
- Spectral/Wavelet Analysis - Cycle detection

Chinese Gold Standard Algorithms (幻方量化, 九坤投资, 明汯投资):
- Microprice (Stoikov 2018) - Better mid-price estimation
- Smart Money Factor 2.0 (开源证券) - Institutional flow detection
- Kyle Lambda - Market impact estimation
- Amihud Illiquidity - Non-liquidity factor
- Integrated OFI (Cont 2014) - Order flow imbalance
- Lee-Ready Classification - Trade direction (85% accuracy)
- Book/Queue Imbalance - LOB pressure signals
- Higher-Order Moments - Skewness, kurtosis, tail risk
- Intraday Momentum - Session effects
- IC/ICIR Engine - Factor evaluation

Target: 500+ features per tick for ML ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Warning deduplication - only show each warning once
_warned_modules: Set[str] = set()


def _warn_once(module_name: str, message: str) -> None:
    """Log a warning only once per module to prevent spam."""
    if module_name not in _warned_modules:
        _warned_modules.add(module_name)
        logger.warning(message)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Time windows for rolling features
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Feature groups to enable
    enable_alpha101: bool = True
    enable_alpha191: bool = True  # 国泰君安 191 factors [GuotaiJunan 2017]
    enable_renaissance: bool = True
    enable_order_flow: bool = True
    enable_hawkes: bool = True  # Hawkes self-exciting process [Hawkes 1971; Bacry 2015]
    enable_cross_asset: bool = True
    enable_microstructure: bool = True
    enable_har_rv: bool = True  # HAR-RV volatility
    enable_technical: bool = True
    enable_range_vol: bool = True  # Parkinson, Garman-Klass, Yang-Zhang
    enable_market_impact: bool = True  # Kyle, Glosten-Milgrom, Roll

    # Renaissance Technologies Inferred Methods
    enable_kalman: bool = True  # EKF/UKF state estimation
    enable_garch: bool = True  # GARCH volatility forecasting
    enable_jump_detection: bool = True  # Bipower variation + jump test
    enable_regime: bool = True  # HMM regime detection
    enable_spectral: bool = True  # FFT/Wavelet cycle detection

    # Chinese Gold Standard Algorithms (幻方量化, 九坤投资, 明汯投资)
    enable_chinese_hft: bool = True  # Microprice, Smart Money, Kyle Lambda
    enable_lob_features: bool = True  # Lee-Ready, Book Imbalance, Queue Imbalance
    enable_elite_quant: bool = True  # IC Engine, PCA, Elastic Net, Higher Moments

    # Automatic Factor Discovery (华泰金工, 东方金工)
    enable_genetic_mining: bool = True  # gplearn/DEAP genetic programming
    enable_tsfresh: bool = True  # Automatic feature extraction (65,000+ factors)

    # Normalization
    normalize: bool = True
    zscore_window: int = 100

    # Feature selection
    min_variance: float = 0.001
    max_correlation: float = 0.95


class FastTechnicalFeatures:
    """
    Ultra-fast technical indicators for HFT.
    Optimized for tick-by-tick computation.
    """

    def __init__(self, lookback: int = 200):
        self.lookback = lookback
        self.prices = []
        self.volumes = []
        self.timestamps = []

    def update(self, price: float, volume: float = 0.0,
               timestamp: datetime = None) -> Dict[str, float]:
        """Update with new tick and return features."""
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp or datetime.now())

        # Keep only lookback
        if len(self.prices) > self.lookback:
            self.prices = self.prices[-self.lookback:]
            self.volumes = self.volumes[-self.lookback:]
            self.timestamps = self.timestamps[-self.lookback:]

        return self.compute_features()

    def compute_features(self) -> Dict[str, float]:
        """Compute all fast technical features."""
        if len(self.prices) < 10:
            return {}

        prices = np.array(self.prices)
        volumes = np.array(self.volumes)

        features = {}

        # Returns at different lags
        for lag in [1, 5, 10, 20]:
            if len(prices) > lag:
                ret = (prices[-1] / prices[-lag-1] - 1) * 10000  # bps
                features[f'return_{lag}'] = ret

        # Volatility estimates
        if len(prices) > 20:
            returns = np.diff(np.log(prices[-21:]))
            features['vol_20'] = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized

        if len(prices) > 5:
            returns = np.diff(np.log(prices[-6:]))
            features['vol_5'] = np.std(returns) * np.sqrt(252 * 24 * 60)

        # Momentum indicators
        if len(prices) > 20:
            features['momentum_20'] = (prices[-1] / prices[-20] - 1) * 10000

        if len(prices) > 10:
            features['momentum_10'] = (prices[-1] / prices[-10] - 1) * 10000

        # Mean reversion signals
        for window in [10, 20, 50]:
            if len(prices) > window:
                ma = np.mean(prices[-window:])
                std = np.std(prices[-window:])
                if std > 0:
                    features[f'zscore_{window}'] = (prices[-1] - ma) / std

        # Price acceleration
        if len(prices) > 10:
            ret_5 = prices[-1] / prices[-6] - 1 if len(prices) > 5 else 0
            ret_10 = prices[-1] / prices[-11] - 1
            features['acceleration'] = (ret_5 - ret_10 / 2) * 10000

        # High/Low range position
        for window in [20, 50]:
            if len(prices) > window:
                high = np.max(prices[-window:])
                low = np.min(prices[-window:])
                rng = high - low
                if rng > 0:
                    features[f'range_position_{window}'] = (prices[-1] - low) / rng

        # Volume features
        if len(volumes) > 20 and np.sum(volumes) > 0:
            avg_vol = np.mean(volumes[-20:])
            if avg_vol > 0:
                features['volume_ratio'] = volumes[-1] / avg_vol

            # Volume-weighted price
            vwap = np.sum(prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
            features['vwap_deviation'] = (prices[-1] - vwap) / prices[-1] * 10000

        # Tick direction
        if len(prices) > 1:
            features['tick_direction'] = 1 if prices[-1] > prices[-2] else (-1 if prices[-1] < prices[-2] else 0)

        # Consecutive moves
        if len(prices) > 5:
            directions = np.sign(np.diff(prices[-5:]))
            features['consecutive_ups'] = np.sum(directions > 0)
            features['consecutive_downs'] = np.sum(directions < 0)

        return features


class HFTFeatureEngine:
    """
    Unified HFT Feature Engineering Engine.

    Combines all signal sources into a single feature vector.
    Optimized for real-time tick-by-tick processing.

    Usage:
        engine = HFTFeatureEngine()
        engine.initialize(historical_data)

        for tick in live_ticks:
            features = engine.process_tick(tick)
            prediction = model.predict(features)
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

        # Initialize feature generators
        self.fast_tech = FastTechnicalFeatures()

        # Lazy load heavy modules
        self._alpha101 = None
        self._alpha191 = None  # 国泰君安 191 factors
        self._renaissance = None
        self._order_flow = None
        self._microstructure = None
        self._har_rv = None  # HAR-RV volatility
        self._range_vol = None  # Parkinson, Garman-Klass, Yang-Zhang
        self._market_impact = None  # Kyle, Glosten-Milgrom, Roll

        # Renaissance Technologies modules
        self._kalman = None  # EKF/UKF filters
        self._garch = None  # GARCH volatility
        self._jump_detection = None  # Bipower variation
        self._regime = None  # HMM regime
        self._spectral = None  # FFT/Wavelet

        # Chinese Gold Standard modules (幻方量化, 九坤投资, 明汯投资)
        self._chinese_hft = None  # Microprice, Smart Money, Kyle Lambda
        self._lob_features = None  # Lee-Ready, Book Imbalance
        self._elite_quant = None  # IC Engine, PCA, Elastic Net

        # Automatic Factor Discovery (华泰金工, 东方金工)
        self._genetic_mining = None  # gplearn/DEAP genetic programming
        self._tsfresh = None  # TSFresh auto features

        # State for incremental updates
        self.price_buffer: Dict[str, List[float]] = {}
        self.volume_buffer: Dict[str, List[float]] = {}
        self.tick_buffer: Dict[str, pd.DataFrame] = {}

        # Feature cache for efficiency
        self.feature_cache: Dict[str, Dict[str, float]] = {}
        self.last_feature_time: Dict[str, datetime] = {}

        # Feature names for consistency
        self.feature_names: List[str] = []

    def _lazy_load_alpha101(self):
        """Lazy load Alpha101 module."""
        if self._alpha101 is None:
            try:
                from core.features.alpha101 import Alpha101Complete
                self._alpha101 = Alpha101Complete()
            except ImportError:
                _warn_once("alpha101", "Alpha101 module not available")
                self._alpha101 = None
        return self._alpha101

    def _lazy_load_renaissance(self):
        """Lazy load Renaissance signals module."""
        if self._renaissance is None:
            try:
                from core.features.renaissance import RenaissanceSignalGenerator
                self._renaissance = RenaissanceSignalGenerator()
            except ImportError:
                _warn_once("renaissance", "Renaissance signals module not available")
                self._renaissance = None
        return self._renaissance

    def _lazy_load_order_flow(self):
        """Lazy load Order Flow features module."""
        if self._order_flow is None:
            try:
                from core._experimental.order_flow_features import OrderFlowFeatures
                self._order_flow = OrderFlowFeatures()
            except ImportError:
                _warn_once("order_flow", "Order Flow features module not available")
                self._order_flow = None
        return self._order_flow

    def _lazy_load_microstructure(self):
        """Lazy load Microstructure volatility module."""
        if self._microstructure is None:
            try:
                from core._experimental.microstructure_vol import MicrostructureVolatility
                self._microstructure = MicrostructureVolatility()
            except ImportError:
                _warn_once("microstructure", "Microstructure volatility module not available")
                self._microstructure = None
        return self._microstructure

    def _lazy_load_alpha191(self):
        """
        Lazy load 国泰君安 Alpha191 module. [国泰君安 2017]

        Reference:
            国泰君安证券. (2017). "Alpha191: 191 Short-Period Alpha Factors."
            "基于短周期价量特征的多因子选股体系——数量化专题之九十三"
            Qlib implementation: https://github.com/microsoft/qlib
        """
        if self._alpha191 is None:
            try:
                # Production location [Qlib integration pattern]
                from core.features.alpha191 import Alpha191GuotaiJunan
                self._alpha191 = Alpha191GuotaiJunan()
            except ImportError:
                try:
                    # Fallback to experimental
                    from core._experimental.alpha191_guotaijunan import Alpha191GuotaiJunan
                    self._alpha191 = Alpha191GuotaiJunan()
                except ImportError:
                    _warn_once("alpha191", "Alpha191 module not available")
                    self._alpha191 = None
        return self._alpha191

    def _lazy_load_hawkes(self):
        """
        Lazy load Hawkes Order Flow module. [Hawkes 1971; Bacry 2015]

        Reference:
            Hawkes, A.G. (1971). "Spectra of some self-exciting and mutually
            exciting point processes." Biometrika, 58(1), 83-90.

            Bacry, E., Mastromatteo, I., & Muzy, J.F. (2015). "Hawkes
            processes in finance." Market Microstructure and Liquidity.
        """
        if not hasattr(self, '_hawkes') or self._hawkes is None:
            try:
                from core.features.hawkes_order_flow import HawkesOrderFlow
                self._hawkes = HawkesOrderFlow()
            except ImportError:
                _warn_once("hawkes", "Hawkes Order Flow module not available")
                self._hawkes = None
        return self._hawkes

    def _lazy_load_har_rv(self):
        """Lazy load HAR-RV volatility module."""
        if self._har_rv is None:
            try:
                from core._experimental.har_rv_volatility import HARRVVolatility
                self._har_rv = HARRVVolatility()
            except ImportError:
                _warn_once("har_rv", "HAR-RV volatility module not available")
                self._har_rv = None
        return self._har_rv

    def _lazy_load_range_vol(self):
        """
        Lazy load Range-Based Volatility module.
        Includes: Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991), Yang-Zhang (2000)
        """
        if self._range_vol is None:
            try:
                from core._experimental.range_volatility import RangeVolatility
                self._range_vol = RangeVolatility()
            except ImportError:
                _warn_once("range_vol", "Range volatility module not available")
                self._range_vol = None
        return self._range_vol

    def _lazy_load_market_impact(self):
        """
        Lazy load Market Impact models module.
        Includes: Kyle (1985), Glosten-Milgrom (1985), Roll (1984), Huang-Stoll (1997)
        """
        if self._market_impact is None:
            try:
                from core._experimental.market_impact_models import MarketImpactFeatures
                self._market_impact = MarketImpactFeatures()
            except ImportError:
                _warn_once("market_impact", "Market impact module not available")
                self._market_impact = None
        return self._market_impact

    def _lazy_load_kalman(self):
        """
        Lazy load Advanced Kalman Filter module.
        Includes: EKF, UKF, Adaptive Kalman, Kalman Ensemble
        Source: Bar-Shalom "Estimation with Applications to Tracking"
        """
        if self._kalman is None:
            try:
                from core._experimental.advanced_kalman_filters import AdaptiveKalmanFilter
                self._kalman = AdaptiveKalmanFilter(base_filter='ukf')
            except ImportError:
                _warn_once("kalman", "Advanced Kalman filters module not available")
                self._kalman = None
        return self._kalman

    def _lazy_load_garch(self):
        """
        Lazy load GARCH volatility module.
        Includes: GARCH, EGARCH, GJR-GARCH, Regime-Switching GARCH
        Source: Bollerslev (1986), Nelson (1991)
        """
        if self._garch is None:
            try:
                from core._experimental.arima_garch_models import VolatilityForecaster
                self._garch = VolatilityForecaster()
            except ImportError:
                _warn_once("garch", "GARCH module not available")
                self._garch = None
        return self._garch

    def _lazy_load_jump_detection(self):
        """
        Lazy load Jump Detection module.
        Includes: Bipower Variation, BNS Test, Lee-Mykland
        Source: Barndorff-Nielsen & Shephard (2004)
        """
        if self._jump_detection is None:
            try:
                from core._experimental.jump_detection import JumpVolatilityModel
                self._jump_detection = JumpVolatilityModel()
            except ImportError:
                _warn_once("jump_detection", "Jump detection module not available")
                self._jump_detection = None
        return self._jump_detection

    def _lazy_load_regime(self):
        """
        Lazy load Regime Detection module.
        Includes: HMM 3-state, Regime-Scaled Features
        Source: Hamilton (1989)
        """
        if self._regime is None:
            try:
                from core._experimental.regime_features import RegimeDependentFeatureEngine
                self._regime = RegimeDependentFeatureEngine()
            except ImportError:
                _warn_once("regime", "Regime detection module not available")
                self._regime = None
        return self._regime

    def _lazy_load_spectral(self):
        """
        Lazy load Spectral/Wavelet Analysis module.
        Includes: FFT, Welch PSD, DWT, CWT, Hilbert Transform
        Source: Mallat (1989), Oppenheim & Schafer
        """
        if self._spectral is None:
            try:
                from core._experimental.spectral_analysis import SpectralFeatureEngine
                self._spectral = SpectralFeatureEngine()
            except ImportError:
                _warn_once("spectral", "Spectral analysis module not available")
                self._spectral = None
        return self._spectral

    def _lazy_load_chinese_hft(self):
        """
        Lazy load Chinese HFT Factors module.
        Includes: Microprice (Stoikov 2018), Smart Money Factor (开源证券),
                  Kyle Lambda, Amihud Illiquidity, Integrated OFI (Cont 2014)
        Source: QUANTAXIS, QuantsPlaybook, HftBacktest (Gitee/GitHub)
        """
        if self._chinese_hft is None:
            try:
                from core._experimental.chinese_hft_factors import ChineseHFTFactors
                self._chinese_hft = ChineseHFTFactors()
            except ImportError:
                _warn_once("chinese_hft", "Chinese HFT factors module not available")
                self._chinese_hft = None
        return self._chinese_hft

    def _lazy_load_lob_features(self):
        """
        Lazy load LOB Features module.
        Includes: Lee-Ready Classification (85% accuracy), Book Imbalance,
                  Queue Imbalance, Spread Decomposition, LOB Slope
        Source: HftBacktest, LOBSTER, Academic literature
        """
        if self._lob_features is None:
            try:
                from core._experimental.lob_features import LOBFeatureEngine
                self._lob_features = LOBFeatureEngine()
            except ImportError:
                _warn_once("lob", "LOB features module not available")
                self._lob_features = None
        return self._lob_features

    def _lazy_load_elite_quant(self):
        """
        Lazy load Elite Quant Factors module.
        Includes: IC/ICIR Engine, PCA Orthogonalization, Elastic Net Combination,
                  Higher-Order Moments (skew, kurtosis, tail risk), Intraday Momentum
        Source: 幻方量化, 九坤投资, 明汯投资 (Chinese hedge funds)
        """
        if self._elite_quant is None:
            try:
                from core._experimental.elite_quant_factors import EliteQuantEngine
                self._elite_quant = EliteQuantEngine()
            except ImportError:
                _warn_once("elite_quant", "Elite quant factors module not available")
                self._elite_quant = None
        return self._elite_quant

    def _lazy_load_genetic_mining(self):
        """
        Lazy load Genetic Factor Mining module.
        Includes: gplearn genetic programming, DEAP evolutionary algorithms
        Source: 华泰金工遗传规划因子挖掘, 东方金工 DFQ系统
        """
        if self._genetic_mining is None:
            try:
                from core._experimental.genetic_factor_mining import GeneticFactorEngine
                self._genetic_mining = GeneticFactorEngine()
            except ImportError:
                _warn_once("genetic_mining", "Genetic factor mining module not available")
                self._genetic_mining = None
        return self._genetic_mining

    def _lazy_load_tsfresh(self):
        """
        Lazy load TSFresh Automatic Feature Extraction module.
        Includes: 65,000+ automatic time series features
        Source: 知乎 量化小白也能自动化挖掘出6万+因子
        """
        if self._tsfresh is None:
            try:
                from core._experimental.tsfresh_auto_features import TSFreshFactorEngine
                self._tsfresh = TSFreshFactorEngine(mode='quick')
            except ImportError:
                _warn_once("tsfresh", "TSFresh module not available")
                self._tsfresh = None
        return self._tsfresh

    def initialize(self, historical_data: pd.DataFrame, symbol: str = "EURUSD"):
        """
        Initialize engine with historical data.

        Args:
            historical_data: DataFrame with columns [timestamp, bid, ask, volume]
            symbol: Trading symbol
        """
        logger.info(f"Initializing HFT Feature Engine for {symbol}")

        # Store in buffers
        if 'bid' in historical_data.columns and 'ask' in historical_data.columns:
            prices = (historical_data['bid'] + historical_data['ask']) / 2
        elif 'close' in historical_data.columns:
            prices = historical_data['close']
        elif 'price' in historical_data.columns:
            prices = historical_data['price']
        else:
            raise ValueError("No price column found (bid/ask, close, or price)")

        self.price_buffer[symbol] = prices.tolist()[-1000:]

        if 'volume' in historical_data.columns:
            self.volume_buffer[symbol] = historical_data['volume'].tolist()[-1000:]
        else:
            self.volume_buffer[symbol] = [0.0] * len(self.price_buffer[symbol])

        self.tick_buffer[symbol] = historical_data.tail(1000).copy()

        # Pre-compute features on historical data
        logger.info("Pre-computing features on historical data...")

        # Fast technical
        for price, vol in zip(self.price_buffer[symbol], self.volume_buffer[symbol]):
            self.fast_tech.update(price, vol)

        # Build feature names from first computation
        features = self._compute_all_features(symbol)
        self.feature_names = sorted(features.keys())

        logger.info(f"Initialized with {len(self.feature_names)} features")

    def process_tick(self, symbol: str, bid: float, ask: float,
                     volume: float = 0.0, timestamp: datetime = None) -> Dict[str, float]:
        """
        Process a single tick and return features.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            volume: Trade volume
            timestamp: Tick timestamp

        Returns:
            Dict of feature_name -> feature_value
        """
        mid = (bid + ask) / 2
        spread = ask - bid

        # Update buffers
        if symbol not in self.price_buffer:
            self.price_buffer[symbol] = []
            self.volume_buffer[symbol] = []

        self.price_buffer[symbol].append(mid)
        self.volume_buffer[symbol].append(volume)

        # Keep buffer size manageable
        if len(self.price_buffer[symbol]) > 1000:
            self.price_buffer[symbol] = self.price_buffer[symbol][-1000:]
            self.volume_buffer[symbol] = self.volume_buffer[symbol][-1000:]

        # Update fast technical
        tech_features = self.fast_tech.update(mid, volume, timestamp)

        # Add spread features
        tech_features['spread_bps'] = spread / mid * 10000
        tech_features['mid_price'] = mid

        # Compute full features periodically (every 10 ticks for efficiency)
        tick_count = len(self.price_buffer[symbol])

        if tick_count % 10 == 0 or symbol not in self.feature_cache:
            all_features = self._compute_all_features(symbol)
            all_features.update(tech_features)
            self.feature_cache[symbol] = all_features
            self.last_feature_time[symbol] = timestamp or datetime.now()
        else:
            # Use cached features with updated tick-level features
            all_features = self.feature_cache.get(symbol, {}).copy()
            all_features.update(tech_features)

        return all_features

    def _compute_all_features(self, symbol: str) -> Dict[str, float]:
        """Compute all feature groups."""
        features = {}
        prices = np.array(self.price_buffer.get(symbol, []))
        volumes = np.array(self.volume_buffer.get(symbol, []))

        if len(prices) < 20:
            return features

        # 1. Alpha101 features
        if self.config.enable_alpha101:
            alpha_features = self._compute_alpha101_features(prices, volumes)
            features.update(alpha_features)

        # 2. Renaissance signals
        if self.config.enable_renaissance:
            ren_features = self._compute_renaissance_features(prices, volumes)
            features.update(ren_features)

        # 3. Order flow features
        if self.config.enable_order_flow:
            of_features = self._compute_order_flow_features(prices, volumes)
            features.update(of_features)

        # 3.5. Hawkes Order Flow features [Hawkes 1971; Bacry 2015]
        if self.config.enable_hawkes:
            hawkes_features = self._compute_hawkes_features(prices, volumes)
            features.update(hawkes_features)

        # 4. Microstructure features
        if self.config.enable_microstructure:
            micro_features = self._compute_microstructure_features(prices)
            features.update(micro_features)

        # 5. Cross-asset features (if available)
        if self.config.enable_cross_asset:
            cross_features = self._compute_cross_asset_features(symbol)
            features.update(cross_features)

        # 6. 国泰君安 Alpha191 features
        if self.config.enable_alpha191:
            alpha191_features = self._compute_alpha191_features(prices, volumes)
            features.update(alpha191_features)

        # 7. HAR-RV volatility features
        if self.config.enable_har_rv:
            har_rv_features = self._compute_har_rv_features(prices)
            features.update(har_rv_features)

        # 8. Range-based volatility features (Parkinson, Garman-Klass, Yang-Zhang)
        if self.config.enable_range_vol:
            range_vol_features = self._compute_range_vol_features(prices, volumes)
            features.update(range_vol_features)

        # 9. Market impact features (Kyle, Glosten-Milgrom, Roll)
        if self.config.enable_market_impact:
            market_impact_features = self._compute_market_impact_features(prices, volumes)
            features.update(market_impact_features)

        # 10. Advanced Kalman Filter features (EKF/UKF)
        if self.config.enable_kalman:
            kalman_features = self._compute_kalman_features(prices)
            features.update(kalman_features)

        # 11. GARCH volatility features
        if self.config.enable_garch:
            garch_features = self._compute_garch_features(prices)
            features.update(garch_features)

        # 12. Jump detection features (Bipower Variation)
        if self.config.enable_jump_detection:
            jump_features = self._compute_jump_features(prices)
            features.update(jump_features)

        # 13. Regime features (HMM)
        if self.config.enable_regime:
            regime_features = self._compute_regime_features(prices)
            features.update(regime_features)

        # 14. Spectral/Wavelet features
        if self.config.enable_spectral:
            spectral_features = self._compute_spectral_features(prices)
            features.update(spectral_features)

        # 15. Chinese HFT Factors (幻方量化, 九坤投资 methods)
        if self.config.enable_chinese_hft:
            chinese_hft_features = self._compute_chinese_hft_features(prices, volumes)
            features.update(chinese_hft_features)

        # 16. LOB Features (Lee-Ready, Book Imbalance)
        if self.config.enable_lob_features:
            lob_features = self._compute_lob_features(prices, volumes)
            features.update(lob_features)

        # 17. Elite Quant Factors (IC Engine, PCA, Higher Moments)
        if self.config.enable_elite_quant:
            elite_quant_features = self._compute_elite_quant_features(prices, volumes)
            features.update(elite_quant_features)

        # 18. Genetic Factor Mining (华泰金工, 东方金工 DFQ)
        if self.config.enable_genetic_mining:
            genetic_features = self._compute_genetic_mining_features(prices, volumes)
            features.update(genetic_features)

        # 19. TSFresh Automatic Features (65,000+ factors)
        if self.config.enable_tsfresh:
            tsfresh_features = self._compute_tsfresh_features(prices, volumes)
            features.update(tsfresh_features)

        # Normalize if configured
        if self.config.normalize:
            features = self._normalize_features(features, prices)

        return features

    def _compute_alpha101_features(self, prices: np.ndarray,
                                   volumes: np.ndarray) -> Dict[str, float]:
        """Compute Alpha101 features."""
        features = {}

        alpha101 = self._lazy_load_alpha101()
        if alpha101 is None:
            return features

        try:
            # Create minimal DataFrame for Alpha101
            df = pd.DataFrame({
                'open': prices * 0.999,  # Approximate open
                'high': prices * 1.0002,  # Approximate high
                'low': prices * 0.9998,   # Approximate low
                'close': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices))
            })

            # Use generate_all_alphas for efficient computation
            result = alpha101.generate_all_alphas(df)

            # Extract last values for each alpha column
            for col in result.columns:
                if col.startswith('alpha') and col not in ['open', 'high', 'low', 'close', 'volume']:
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'alpha101_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Alpha101 computation error: {e}")

        return features

    def _compute_renaissance_features(self, prices: np.ndarray,
                                      volumes: np.ndarray) -> Dict[str, float]:
        """Compute Renaissance-style signals."""
        features = {}

        renaissance = self._lazy_load_renaissance()
        if renaissance is None:
            # Fallback: compute basic Renaissance-style signals
            return self._compute_basic_renaissance(prices, volumes)

        try:
            df = pd.DataFrame({
                'open': prices,
                'high': prices,
                'low': prices,
                'close': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices))
            })

            result = renaissance.generate_all_signals(df)

            # Extract last values
            for col in result.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'ren_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Renaissance computation error: {e}")
            return self._compute_basic_renaissance(prices, volumes)

        return features

    def _compute_basic_renaissance(self, prices: np.ndarray,
                                   volumes: np.ndarray) -> Dict[str, float]:
        """Basic Renaissance-style signals without full module."""
        features = {}

        # Trend signals
        for window in [5, 10, 20, 50]:
            if len(prices) > window:
                ma = np.mean(prices[-window:])
                features[f'ren_trend_{window}'] = (prices[-1] / ma - 1) * 10000

                # MA slope
                if len(prices) > window + 5:
                    ma_prev = np.mean(prices[-window-5:-5])
                    features[f'ren_slope_{window}'] = (ma / ma_prev - 1) * 10000

        # Mean reversion
        for window in [10, 20, 50]:
            if len(prices) > window:
                mean = np.mean(prices[-window:])
                std = np.std(prices[-window:])
                if std > 0:
                    features[f'ren_mr_{window}'] = (prices[-1] - mean) / std

        # Momentum
        for lag in [5, 10, 20]:
            if len(prices) > lag:
                features[f'ren_mom_{lag}'] = (prices[-1] / prices[-lag-1] - 1) * 10000

        # Volatility regime
        if len(prices) > 20:
            vol_short = np.std(np.diff(np.log(prices[-10:])))
            vol_long = np.std(np.diff(np.log(prices[-50:]))) if len(prices) > 50 else vol_short
            if vol_long > 0:
                features['ren_vol_regime'] = vol_short / vol_long

        return features

    def _compute_order_flow_features(self, prices: np.ndarray,
                                     volumes: np.ndarray) -> Dict[str, float]:
        """Compute order flow features."""
        features = {}

        order_flow = self._lazy_load_order_flow()
        if order_flow is None:
            return self._compute_basic_order_flow(prices, volumes)

        try:
            # Update order flow with recent trades
            for i in range(-min(20, len(prices)), 0):
                price = prices[i]
                vol = volumes[i] if len(volumes) > abs(i) else 1.0
                # Infer direction from price movement
                if i > -len(prices):
                    direction = 1 if prices[i] > prices[i-1] else -1
                else:
                    direction = 1
                order_flow.on_trade(price, vol, direction)

            signals = order_flow.get_signals()
            for key, val in signals.items():
                if not np.isnan(val) and not np.isinf(val):
                    features[f'of_{key}'] = float(val)

        except Exception as e:
            logger.debug(f"Order flow computation error: {e}")
            return self._compute_basic_order_flow(prices, volumes)

        return features

    def _compute_basic_order_flow(self, prices: np.ndarray,
                                  volumes: np.ndarray) -> Dict[str, float]:
        """Basic order flow features without full module."""
        features = {}

        if len(prices) < 10:
            return features

        # Price direction
        directions = np.sign(np.diff(prices[-50:]))

        # Order flow imbalance (simple)
        vol = volumes[-50:] if len(volumes) >= 50 else np.ones(len(directions))
        if len(vol) > len(directions):
            vol = vol[-len(directions):]

        buy_vol = np.sum(vol[directions > 0])
        sell_vol = np.sum(vol[directions < 0])
        total = buy_vol + sell_vol

        if total > 0:
            features['of_imbalance'] = (buy_vol - sell_vol) / total

        # Trade intensity
        features['of_trade_count'] = len(directions)

        # Volume-weighted direction
        if len(vol) == len(directions):
            vw_dir = np.sum(directions * vol) / np.sum(vol) if np.sum(vol) > 0 else 0
            features['of_vw_direction'] = vw_dir

        return features

    def _compute_hawkes_features(self, prices: np.ndarray,
                                  volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute Hawkes process order flow features. [Hawkes 1971; Bacry 2015]

        The intensity function λ(t) follows: [Bacry et al. 2015, Eq. 2.1]
            λ(t) = μ + ∫ φ(t-s) dN(s)

        Where:
            μ = baseline intensity [Hawkes 1971]
            φ(t) = exponential kernel exp(-βt) [Bacry et al. 2015]
            N(s) = counting process of past events [Cont et al. 2014]

        References:
            Hawkes, A.G. (1971). Biometrika, 58(1), 83-90.
            Bacry, E., Mastromatteo, I., & Muzy, J.F. (2015). arXiv:1502.04592
            Cont, R., Kukanov, A., & Stoikov, S. (2014). J. Fin. Econometrics.
        """
        features = {}

        if len(prices) < 30:
            return features

        hawkes = self._lazy_load_hawkes()
        if hawkes is None:
            return self._compute_basic_hawkes(prices, volumes)

        try:
            # Infer trade directions from price changes
            price_changes = np.diff(prices[-100:])
            directions = np.sign(price_changes)
            volumes_use = volumes[-99:] if len(volumes) >= 99 else np.ones(len(directions))

            # Update Hawkes model with recent events
            for i, (direction, vol) in enumerate(zip(directions, volumes_use)):
                side = 'bid' if direction > 0 else 'ask'
                hawkes.on_event(side, float(vol))

            # Get analysis
            analysis = hawkes.analyze()

            # Extract features
            features['hawkes_bid_intensity'] = float(analysis.bid_intensity)
            features['hawkes_ask_intensity'] = float(analysis.ask_intensity)
            features['hawkes_bid_baseline'] = float(analysis.bid_baseline)
            features['hawkes_ask_baseline'] = float(analysis.ask_baseline)
            features['hawkes_branching_ratio'] = float(analysis.branching_ratio)
            features['hawkes_is_stable'] = 1.0 if analysis.is_stable else 0.0
            features['hawkes_bid_pressure'] = float(analysis.bid_pressure)
            features['hawkes_direction'] = float(analysis.predicted_direction)

            # Certainty check for VPIN-style validation [Easley 2012]
            features['hawkes_certainty'] = 1.0 if analysis.certainty_check_passed else 0.0

        except Exception as e:
            logger.debug(f"Hawkes computation error: {e}")
            return self._compute_basic_hawkes(prices, volumes)

        return features

    def _compute_basic_hawkes(self, prices: np.ndarray,
                               volumes: np.ndarray) -> Dict[str, float]:
        """
        Basic Hawkes-style features without full module. [Hawkes 1971]

        Uses exponential decay for event intensity estimation.
        """
        features = {}

        if len(prices) < 30:
            return features

        # Compute price changes
        returns = np.diff(prices[-50:])
        directions = np.sign(returns)

        # Simple exponential kernel intensity estimation [Bacry et al. 2015]
        decay = 0.94  # Exponential decay rate β ≈ 0.94
        bid_intensity = 0.0
        ask_intensity = 0.0

        for i, d in enumerate(directions):
            weight = decay ** (len(directions) - i - 1)
            if d > 0:
                bid_intensity += weight
            else:
                ask_intensity += weight

        # Normalize
        total = bid_intensity + ask_intensity
        if total > 0:
            features['hawkes_bid_intensity'] = bid_intensity / total
            features['hawkes_ask_intensity'] = ask_intensity / total
            features['hawkes_bid_pressure'] = (bid_intensity - ask_intensity) / total
            features['hawkes_direction'] = 1.0 if bid_intensity > ask_intensity else -1.0
        else:
            features['hawkes_bid_intensity'] = 0.5
            features['hawkes_ask_intensity'] = 0.5
            features['hawkes_bid_pressure'] = 0.0
            features['hawkes_direction'] = 0.0

        # Branching ratio estimate [Hawkes 1971]
        # α/β should be < 1 for stability
        features['hawkes_branching_ratio'] = 0.5  # Assume stable
        features['hawkes_is_stable'] = 1.0

        return features

    def _compute_microstructure_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute microstructure features."""
        features = {}

        micro = self._lazy_load_microstructure()
        if micro is None:
            return self._compute_basic_microstructure(prices)

        try:
            # TSRV
            tsrv = micro.tsrv(prices)
            features['micro_tsrv'] = float(tsrv) if not np.isnan(tsrv) else 0.0

            # Noise variance
            noise_var = micro.estimate_noise_variance(prices)
            features['micro_noise_var'] = float(noise_var) if not np.isnan(noise_var) else 0.0

            # Signal-to-noise ratio
            snr = micro.signal_to_noise_ratio(prices)
            if not np.isinf(snr):
                features['micro_snr'] = float(snr)

        except Exception as e:
            logger.debug(f"Microstructure computation error: {e}")
            return self._compute_basic_microstructure(prices)

        return features

    def _compute_basic_microstructure(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic microstructure features without full module."""
        features = {}

        if len(prices) < 20:
            return features

        returns = np.diff(np.log(prices))

        # Realized variance
        features['micro_rv'] = np.sum(returns[-100:]**2) if len(returns) >= 100 else np.sum(returns**2)

        # First-order autocorrelation (noise indicator)
        if len(returns) > 2:
            autocov = np.cov(returns[:-1], returns[1:])[0, 1]
            features['micro_autocov'] = float(autocov) if not np.isnan(autocov) else 0.0

        return features

    def _compute_cross_asset_features(self, symbol: str) -> Dict[str, float]:
        """Compute cross-asset correlation features."""
        features = {}

        # Note: In production, this would fetch DXY, VIX, etc.
        # For now, return empty as these require external data feeds

        return features

    def _compute_alpha191_features(self, prices: np.ndarray,
                                   volumes: np.ndarray) -> Dict[str, float]:
        """Compute 国泰君安 Alpha191 features."""
        features = {}

        alpha191 = self._lazy_load_alpha191()
        if alpha191 is None:
            return features

        try:
            # Create minimal DataFrame for Alpha191
            df = pd.DataFrame({
                'open': prices * 0.999,   # Approximate open
                'high': prices * 1.0002,  # Approximate high
                'low': prices * 0.9998,   # Approximate low
                'close': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices)),
                'vwap': prices  # Use close as VWAP approximation
            })

            # Use generate_all_alphas for efficient computation
            result = alpha191.generate_all_alphas(df)

            # Extract last values for each alpha column (only include alpha191_* columns)
            for col in result.columns:
                if col.startswith('alpha191_'):
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[col] = float(val)

        except Exception as e:
            logger.debug(f"Alpha191 computation error: {e}")

        return features

    def _compute_har_rv_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute HAR-RV volatility features."""
        features = {}

        har_rv = self._lazy_load_har_rv()
        if har_rv is None:
            return self._compute_basic_har_rv(prices)

        try:
            # Compute log returns
            if len(prices) < 30:
                return features

            returns = np.diff(np.log(prices))
            returns_series = pd.Series(returns)

            # Compute RV at different scales
            rv = har_rv.compute_realized_volatility(returns_series, window=1)
            if len(rv) > 0 and not np.isnan(rv.iloc[-1]):
                features['har_rv_daily'] = float(rv.iloc[-1])

            # HAR components
            components = har_rv.compute_har_components(rv)
            if components is not None and len(components) > 0:
                for col in components.columns:
                    val = components[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'har_{col}'] = float(val)

            # Volatility regime (get last value from Series)
            regime_series = har_rv.get_volatility_regime(returns_series)
            if regime_series is not None and len(regime_series) > 0:
                regime = regime_series.iloc[-1]
                features['har_regime'] = {'low': 0, 'normal': 1, 'high': 2}.get(regime, 1)

        except Exception as e:
            logger.debug(f"HAR-RV computation error: {e}")
            return self._compute_basic_har_rv(prices)

        return features

    def _compute_basic_har_rv(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic HAR-RV features without full module."""
        features = {}

        if len(prices) < 30:
            return features

        returns = np.diff(np.log(prices))

        # Daily RV (sum of squared returns)
        rv_daily = np.sum(returns[-5:]**2)
        features['har_rv_daily'] = float(rv_daily) if not np.isnan(rv_daily) else 0.0

        # Weekly RV (average of last 5 daily)
        if len(returns) >= 25:
            rv_weekly = np.mean([np.sum(returns[i:i+5]**2) for i in range(-25, -4, 5)])
            features['har_rv_weekly'] = float(rv_weekly) if not np.isnan(rv_weekly) else 0.0

        # Monthly RV (average of last 22 daily)
        if len(returns) >= 110:
            rv_monthly = np.mean([np.sum(returns[i:i+5]**2) for i in range(-110, -4, 5)])
            features['har_rv_monthly'] = float(rv_monthly) if not np.isnan(rv_monthly) else 0.0

        # Simple regime based on RV percentile
        if len(returns) >= 50:
            rv_history = pd.Series([np.sum(returns[i:i+5]**2) for i in range(0, len(returns)-4, 5)])
            percentile = (rv_history < rv_daily).mean()
            if percentile < 0.3:
                features['har_regime'] = 0  # Low vol
            elif percentile > 0.7:
                features['har_regime'] = 2  # High vol
            else:
                features['har_regime'] = 1  # Normal

        return features

    def _compute_range_vol_features(self, prices: np.ndarray,
                                    volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute range-based volatility features.

        Sources:
        - Parkinson (1980): "The Extreme Value Method for Estimating the Variance of the Rate of Return"
        - Garman-Klass (1980): "On the Estimation of Security Price Volatilities from Historical Data"
        - Rogers-Satchell (1991): "Estimating Variance from High, Low and Closing Prices"
        - Yang-Zhang (2000): "Drift Independent Volatility Estimation"
        """
        features = {}

        range_vol = self._lazy_load_range_vol()
        if range_vol is None:
            return self._compute_basic_range_vol(prices)

        try:
            # Need OHLC - approximate from mid prices
            if len(prices) < 30:
                return features

            # Create approximate OHLC (for tick data)
            window = 5  # Aggregate 5 ticks into one "bar"
            n_bars = len(prices) // window

            if n_bars < 10:
                return features

            opens = []
            highs = []
            lows = []
            closes = []

            for i in range(n_bars):
                start_idx = i * window
                end_idx = start_idx + window
                bar_prices = prices[start_idx:end_idx]
                opens.append(bar_prices[0])
                highs.append(np.max(bar_prices))
                lows.append(np.min(bar_prices))
                closes.append(bar_prices[-1])

            o = pd.Series(opens)
            h = pd.Series(highs)
            l = pd.Series(lows)
            c = pd.Series(closes)

            # Compute volatility estimators
            vol_window = min(20, len(o) - 1)

            # Parkinson (1980) - uses only high/low
            parkinson = range_vol.parkinson(h, l, window=vol_window)
            if len(parkinson) > 0 and not np.isnan(parkinson.iloc[-1]):
                features['range_vol_parkinson'] = float(parkinson.iloc[-1])

            # Garman-Klass (1980) - uses OHLC
            gk = range_vol.garman_klass(o, h, l, c, window=vol_window)
            if len(gk) > 0 and not np.isnan(gk.iloc[-1]):
                features['range_vol_gk'] = float(gk.iloc[-1])

            # Rogers-Satchell (1991) - zero-drift estimator
            rs = range_vol.rogers_satchell(o, h, l, c, window=vol_window)
            if len(rs) > 0 and not np.isnan(rs.iloc[-1]):
                features['range_vol_rs'] = float(rs.iloc[-1])

            # Yang-Zhang (2000) - handles overnight jumps
            yz = range_vol.yang_zhang(o, h, l, c, window=vol_window)
            if len(yz) > 0 and not np.isnan(yz.iloc[-1]):
                features['range_vol_yz'] = float(yz.iloc[-1])

            # Ensemble volatility
            ensemble = range_vol.ensemble(o, h, l, c, window=vol_window)
            if len(ensemble) > 0 and not np.isnan(ensemble.iloc[-1]):
                features['range_vol_ensemble'] = float(ensemble.iloc[-1])

            # Volatility regime
            regime = range_vol.volatility_regime(c)
            if regime:
                features['range_vol_regime'] = {'low': 0, 'normal': 1, 'high': 2}.get(regime, 1)

        except Exception as e:
            logger.debug(f"Range volatility computation error: {e}")
            return self._compute_basic_range_vol(prices)

        return features

    def _compute_basic_range_vol(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic range volatility features without full module."""
        features = {}

        if len(prices) < 30:
            return features

        # Simple Parkinson-style from tick range
        window = 5
        n_bars = len(prices) // window

        if n_bars < 10:
            return features

        # Compute high-low range volatility
        hl_vols = []
        for i in range(n_bars):
            bar = prices[i*window:(i+1)*window]
            hl_range = np.log(np.max(bar) / np.min(bar))
            # Parkinson constant: 1 / (4 * ln(2))
            hl_vols.append(hl_range**2 / (4 * np.log(2)))

        if hl_vols:
            features['range_vol_parkinson'] = np.sqrt(np.mean(hl_vols[-20:])) * np.sqrt(252 * 24 * 12)

        return features

    def _compute_market_impact_features(self, prices: np.ndarray,
                                        volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute market impact features.

        Sources:
        - Kyle (1985): "Continuous Auctions and Insider Trading" - lambda estimation
        - Glosten-Milgrom (1985): "Bid, Ask and Transaction Prices" - adverse selection
        - Roll (1984): "A Simple Implicit Measure of the Effective Bid-Ask Spread"
        - Huang-Stoll (1997): "The Components of the Bid-Ask Spread"
        """
        features = {}

        market_impact = self._lazy_load_market_impact()
        if market_impact is None:
            return self._compute_basic_market_impact(prices, volumes)

        try:
            if len(prices) < 30:
                return features

            prices_series = pd.Series(prices)
            volumes_series = pd.Series(volumes) if len(volumes) == len(prices) else pd.Series(np.ones(len(prices)))

            # Compute all features
            result = market_impact.compute_all_features(prices_series, volumes_series)

            # Extract features
            for key, val in result.items():
                if isinstance(val, (pd.Series, np.ndarray)):
                    if len(val) > 0:
                        last_val = val.iloc[-1] if hasattr(val, 'iloc') else val[-1]
                        if not np.isnan(last_val) and not np.isinf(last_val):
                            features[f'mi_{key}'] = float(last_val)
                elif isinstance(val, (int, float)):
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'mi_{key}'] = float(val)

        except Exception as e:
            logger.debug(f"Market impact computation error: {e}")
            return self._compute_basic_market_impact(prices, volumes)

        return features

    def _compute_basic_market_impact(self, prices: np.ndarray,
                                     volumes: np.ndarray) -> Dict[str, float]:
        """Basic market impact features without full module."""
        features = {}

        if len(prices) < 30:
            return features

        # Roll (1984) effective spread estimate
        # Roll spread = 2 * sqrt(-cov(r_t, r_{t-1}))
        returns = np.diff(np.log(prices))
        if len(returns) > 2:
            autocov = np.cov(returns[:-1], returns[1:])[0, 1]
            if autocov < 0:
                features['mi_roll_spread'] = 2 * np.sqrt(-autocov)

        # Simple Kyle lambda proxy
        # lambda ≈ |ΔP| / V (price impact per unit volume)
        if len(volumes) == len(prices) and np.sum(volumes) > 0:
            price_changes = np.abs(np.diff(prices))
            vol = volumes[1:]
            vol_mask = vol > 0
            if np.sum(vol_mask) > 0:
                lambda_est = np.mean(price_changes[vol_mask] / vol[vol_mask])
                features['mi_kyle_lambda'] = float(lambda_est) if not np.isnan(lambda_est) else 0.0

        # Amihud illiquidity
        # ILLIQ = |r| / V (return per dollar volume)
        if len(volumes) == len(prices) and np.sum(volumes) > 0:
            abs_returns = np.abs(returns)
            dollar_vol = volumes[1:] * prices[1:]
            vol_mask = dollar_vol > 0
            if np.sum(vol_mask) > 0:
                illiq = np.mean(abs_returns[vol_mask] / dollar_vol[vol_mask])
                features['mi_amihud_illiq'] = float(illiq) * 1e6 if not np.isnan(illiq) else 0.0

        return features

    def _compute_kalman_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute Advanced Kalman Filter features.

        Sources:
        - Bar-Shalom "Estimation with Applications to Tracking"
        - Julier & Uhlmann (1997) "UKF for Nonlinear Estimation"

        Why Renaissance Uses This:
        - Price = TrueValue + MicrostructureNoise
        - Filters out bid-ask bounce
        - Estimates latent price velocity (momentum)
        """
        features = {}

        kalman = self._lazy_load_kalman()
        if kalman is None:
            return self._compute_basic_kalman(prices)

        try:
            if len(prices) < 30:
                return features

            # Update Kalman filter with recent prices
            for price in prices[-50:]:
                state = kalman.update(float(price))

            # Extract features
            features['kalman_price'] = kalman.get_filtered_price()
            features['kalman_velocity'] = kalman.get_velocity()

            # Price deviation from Kalman estimate (mean reversion signal)
            deviation = prices[-1] - features['kalman_price']
            features['kalman_deviation'] = deviation * 10000  # In bps

            # Regime indicator from innovation sequence
            if hasattr(kalman, 'get_regime_indicator'):
                features['kalman_regime'] = kalman.get_regime_indicator()

        except Exception as e:
            logger.debug(f"Kalman computation error: {e}")
            return self._compute_basic_kalman(prices)

        return features

    def _compute_basic_kalman(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic Kalman-like features without full module."""
        features = {}

        if len(prices) < 20:
            return features

        # Simple exponential smoothing as Kalman proxy
        alpha = 0.1
        filtered = prices[0]
        for p in prices[1:]:
            filtered = alpha * p + (1 - alpha) * filtered

        features['kalman_price'] = filtered
        features['kalman_deviation'] = (prices[-1] - filtered) * 10000

        # Velocity from differenced filtered price
        if len(prices) > 5:
            velocity = (filtered - prices[-5]) / 5
            features['kalman_velocity'] = velocity * 10000

        return features

    def _compute_garch_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute GARCH volatility features.

        Sources:
        - Bollerslev (1986) "GARCH"
        - Nelson (1991) "EGARCH"

        Why Renaissance Uses This:
        - Volatility clustering (high vol follows high vol)
        - Position sizing with Kelly criterion
        - Leverage effect (neg returns increase vol more)
        """
        features = {}

        if len(prices) < 50:
            return self._compute_basic_garch(prices)

        garch = self._lazy_load_garch()
        if garch is None:
            return self._compute_basic_garch(prices)

        try:
            # Compute returns
            returns = np.diff(np.log(prices))

            # Fit if not already (expensive, so cache)
            if not garch.fitted:
                garch.fit(returns)

            # Get current volatility
            features['garch_vol'] = garch.get_current_volatility() * np.sqrt(252 * 24 * 60)

            # Volatility forecast
            if garch.fitted:
                forecast = garch.forecast(horizon=5)
                features['garch_vol_forecast'] = float(forecast[-1]) * np.sqrt(252 * 24 * 60)

        except Exception as e:
            logger.debug(f"GARCH computation error: {e}")
            return self._compute_basic_garch(prices)

        return features

    def _compute_basic_garch(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic GARCH-like features without full module."""
        features = {}

        if len(prices) < 20:
            return features

        returns = np.diff(np.log(prices))

        # Simple EWMA volatility as GARCH proxy
        alpha = 0.06  # Decay factor
        ewma_var = returns[0]**2
        for r in returns[1:]:
            ewma_var = alpha * r**2 + (1 - alpha) * ewma_var

        features['garch_vol'] = np.sqrt(ewma_var) * np.sqrt(252 * 24 * 60)

        return features

    def _compute_jump_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute Jump Detection features using Bipower Variation.

        Sources:
        - Barndorff-Nielsen & Shephard (2004) "Power and Multipower Variations"

        Why Renaissance Uses This:
        - Distinguish news events from continuous trading
        - GARCH fails on jumps
        - Adjust position sizing during jump periods
        """
        features = {}

        if len(prices) < 50:
            return self._compute_basic_jump(prices)

        jump_model = self._lazy_load_jump_detection()
        if jump_model is None:
            return self._compute_basic_jump(prices)

        try:
            returns = np.diff(np.log(prices))
            returns_series = pd.Series(returns)

            # Compute decomposition
            decomp = jump_model.decompose_volatility(returns_series)

            # Latest values
            features['jump_rv'] = float(decomp.realized_variance[-1])
            features['jump_bpv'] = float(decomp.bipower_variance[-1])
            features['jump_var'] = float(decomp.jump_variance[-1])
            features['jump_ratio'] = float(decomp.relative_jump[-1])
            features['jump_indicator'] = float(decomp.jump_indicator[-1])

            # Continuous vs jump vol
            features['continuous_vol'] = np.sqrt(float(decomp.bipower_variance[-1]))
            features['jump_vol'] = np.sqrt(float(decomp.jump_variance[-1]))

        except Exception as e:
            logger.debug(f"Jump detection error: {e}")
            return self._compute_basic_jump(prices)

        return features

    def _compute_basic_jump(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic jump detection without full module."""
        features = {}

        if len(prices) < 20:
            return features

        returns = np.diff(np.log(prices))

        # Simple threshold-based jump detection
        mad = np.median(np.abs(returns - np.median(returns)))
        sigma = mad * 1.4826
        threshold = 3 * sigma

        jumps = np.abs(returns) > threshold
        features['jump_indicator'] = float(jumps[-1]) if len(jumps) > 0 else 0.0
        features['jump_ratio'] = float(np.sum(jumps)) / len(jumps)

        return features

    def _compute_regime_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute Regime-Dependent features using HMM.

        Sources:
        - Hamilton (1989) "Markov Switching Models"

        Why Renaissance Uses This:
        - Different strategies for different regimes
        - Position sizing based on regime
        - Avoid trading during regime transitions
        """
        features = {}

        if len(prices) < 50:
            return self._compute_basic_regime(prices)

        regime_engine = self._lazy_load_regime()
        if regime_engine is None:
            return self._compute_basic_regime(prices)

        try:
            returns = np.diff(np.log(prices))

            # Fit if not already
            if not regime_engine.fitted and len(returns) > 100:
                regime_engine.fit(returns)

            # Detect current regime
            state = regime_engine.detect_regime(returns)

            features['regime'] = float(state.regime)
            features['regime_duration'] = float(state.duration)
            features['regime_transition_prob'] = float(state.transition_prob)
            features['regime_prob_low'] = float(state.probability[0])
            features['regime_prob_normal'] = float(state.probability[1])
            features['regime_prob_high'] = float(state.probability[2])

            # Position and stop multipliers
            features['regime_position_mult'] = regime_engine.config.position_mult.get(state.regime, 1.0)
            features['regime_stop_mult'] = regime_engine.config.stop_mult.get(state.regime, 1.0)

        except Exception as e:
            logger.debug(f"Regime detection error: {e}")
            return self._compute_basic_regime(prices)

        return features

    def _compute_basic_regime(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic regime detection without full module."""
        features = {}

        if len(prices) < 20:
            return features

        # Simple volatility-based regime
        returns = np.diff(np.log(prices))
        vol = np.std(returns[-20:])

        if vol < 0.0002:
            regime = 0  # Low
        elif vol > 0.0006:
            regime = 2  # High
        else:
            regime = 1  # Normal

        features['regime'] = float(regime)
        features['regime_position_mult'] = [1.5, 1.0, 0.5][regime]
        features['regime_stop_mult'] = [1.0, 1.5, 2.5][regime]

        return features

    def _compute_spectral_features(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Compute Spectral/Wavelet Analysis features.

        Sources:
        - Mallat (1989) "Wavelet Decomposition"
        - Oppenheim & Schafer "Signal Processing"

        Why Renaissance Uses This:
        - Identify dominant market cycles
        - Filter noise from signal
        - Detect phase shifts
        """
        features = {}

        if len(prices) < 100:
            return self._compute_basic_spectral(prices)

        spectral = self._lazy_load_spectral()
        if spectral is None:
            return self._compute_basic_spectral(prices)

        try:
            prices_series = pd.Series(prices)
            result = spectral.compute_features(prices_series, window=min(100, len(prices) - 10))

            # Get last row
            for col in result.columns:
                val = result[col].iloc[-1]
                if not np.isnan(val) and not np.isinf(val):
                    features[f'spectral_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Spectral analysis error: {e}")
            return self._compute_basic_spectral(prices)

        return features

    def _compute_basic_spectral(self, prices: np.ndarray) -> Dict[str, float]:
        """Basic spectral features without full module."""
        features = {}

        if len(prices) < 50:
            return features

        returns = np.diff(np.log(prices))

        # Simple FFT-based dominant frequency
        from scipy.fft import fft, fftfreq

        spectrum = np.abs(fft(returns))**2
        freqs = fftfreq(len(returns))

        # Positive frequencies only
        pos_mask = freqs > 0
        if np.sum(pos_mask) > 0:
            pos_freqs = freqs[pos_mask]
            pos_power = spectrum[pos_mask]

            dominant_idx = np.argmax(pos_power)
            dominant_freq = pos_freqs[dominant_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else len(returns)

            features['spectral_dominant_period'] = dominant_period

            # Spectral entropy
            power_norm = pos_power / (pos_power.sum() + 1e-10)
            entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
            features['spectral_entropy'] = entropy / np.log(len(power_norm))

        return features

    def _compute_chinese_hft_features(self, prices: np.ndarray,
                                      volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute Chinese HFT factors.

        Sources:
        - Stoikov (2018) "Microprice"
        - 开源证券 "Smart Money Factor 2.0"
        - QUANTAXIS, QuantsPlaybook (Gitee)
        - Cont et al. (2014) "Integrated OFI"

        Includes: Microprice, Smart Money, Kyle Lambda, Amihud Illiquidity
        """
        features = {}

        if len(prices) < 30:
            return self._compute_basic_chinese_hft(prices, volumes)

        chinese_hft = self._lazy_load_chinese_hft()
        if chinese_hft is None:
            return self._compute_basic_chinese_hft(prices, volumes)

        try:
            # Create DataFrame for Chinese HFT
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices)),
                'high': prices * 1.0002,
                'low': prices * 0.9998,
                'bid': prices * 0.9999,
                'ask': prices * 1.0001,
                'bid_size': np.ones(len(prices)) * 100,
                'ask_size': np.ones(len(prices)) * 100
            })

            result = chinese_hft.compute_all_features(df)

            # Extract last values
            for col in result.columns:
                if col not in ['close', 'volume', 'high', 'low', 'bid', 'ask', 'bid_size', 'ask_size']:
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'cn_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Chinese HFT computation error: {e}")
            return self._compute_basic_chinese_hft(prices, volumes)

        return features

    def _compute_basic_chinese_hft(self, prices: np.ndarray,
                                   volumes: np.ndarray) -> Dict[str, float]:
        """Basic Chinese HFT features without full module."""
        features = {}

        if len(prices) < 20:
            return features

        returns = np.diff(np.log(prices))

        # Microprice approximation (using equal bid/ask sizes)
        mid = prices[-1]
        spread = mid * 0.0002  # Assume 2 pip spread
        features['cn_microprice'] = mid  # Equal to mid when sizes equal

        # Kyle Lambda approximation
        if len(volumes) == len(prices) and np.sum(volumes) > 0:
            price_changes = np.abs(np.diff(prices))
            vol = volumes[1:]
            vol_mask = vol > 0
            if np.sum(vol_mask) > 0:
                lambda_est = np.mean(price_changes[vol_mask] / vol[vol_mask])
                features['cn_kyle_lambda'] = float(lambda_est) if not np.isnan(lambda_est) else 0.0

        # Amihud illiquidity
        if len(volumes) == len(prices) and np.sum(volumes) > 0:
            abs_returns = np.abs(returns)
            dollar_vol = volumes[1:] * prices[1:]
            vol_mask = dollar_vol > 0
            if np.sum(vol_mask) > 0:
                illiq = np.mean(abs_returns[vol_mask] / dollar_vol[vol_mask])
                features['cn_amihud_illiq'] = float(illiq) * 1e6 if not np.isnan(illiq) else 0.0

        # Smart money factor approximation
        if len(volumes) == len(prices):
            signed_vol = np.sign(returns) * volumes[1:] * np.abs(returns)
            vol_ret = volumes[1:] * np.abs(returns)
            if np.sum(vol_ret) > 0:
                smart_money = np.sum(signed_vol) / np.sum(vol_ret)
                features['cn_smart_money'] = float(smart_money) if not np.isnan(smart_money) else 0.0

        return features

    def _compute_lob_features(self, prices: np.ndarray,
                              volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute Limit Order Book features.

        Sources:
        - Lee & Ready (1991) "Trade Direction Classification"
        - HftBacktest (GitHub)
        - LOBSTER dataset methodology

        Includes: Lee-Ready, Book Imbalance, Queue Imbalance, Spread Decomposition
        """
        features = {}

        if len(prices) < 30:
            return self._compute_basic_lob(prices, volumes)

        lob_features = self._lazy_load_lob_features()
        if lob_features is None:
            return self._compute_basic_lob(prices, volumes)

        try:
            # Create DataFrame for LOB
            df = pd.DataFrame({
                'trade_price': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices)),
                'bid': prices * 0.9999,
                'ask': prices * 1.0001,
                'bid_size': np.ones(len(prices)) * 100,
                'ask_size': np.ones(len(prices)) * 100
            })

            result = lob_features.compute_all_features(df)

            # Extract last values
            for col in result.columns:
                if col not in ['trade_price', 'volume', 'bid', 'ask', 'bid_size', 'ask_size']:
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'lob_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"LOB features computation error: {e}")
            return self._compute_basic_lob(prices, volumes)

        return features

    def _compute_basic_lob(self, prices: np.ndarray,
                           volumes: np.ndarray) -> Dict[str, float]:
        """Basic LOB features without full module."""
        features = {}

        if len(prices) < 20:
            return features

        # Simple Lee-Ready direction inference
        price_changes = np.diff(prices)
        directions = np.sign(price_changes)

        # Buy/sell pressure
        if len(directions) > 0:
            buy_pressure = np.sum(directions > 0) / len(directions)
            features['lob_buy_pressure'] = float(buy_pressure)

        # Book imbalance approximation (assume equal sizes)
        features['lob_book_imbalance'] = 0.0  # Equal sizes = no imbalance

        # Spread approximation
        spread_bps = 0.02  # 2 pip approximation
        features['lob_spread_bps'] = spread_bps * 100

        # Roll spread estimate
        returns = np.diff(np.log(prices))
        if len(returns) > 2:
            autocov = np.cov(returns[:-1], returns[1:])[0, 1]
            if autocov < 0:
                features['lob_roll_spread'] = 2 * np.sqrt(-autocov)

        return features

    def _compute_elite_quant_features(self, prices: np.ndarray,
                                      volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute Elite Quant factors from top Chinese funds.

        Sources:
        - 幻方量化 (Huanfang Quant)
        - 九坤投资 (Nine Kun Investment)
        - 明汯投资 (Minghui Investment)

        Includes: IC/ICIR evaluation, PCA orthogonalization, Higher-order moments,
                  Intraday momentum, Adaptive factor weighting
        """
        features = {}

        if len(prices) < 50:
            return self._compute_basic_elite_quant(prices, volumes)

        elite_quant = self._lazy_load_elite_quant()
        if elite_quant is None:
            return self._compute_basic_elite_quant(prices, volumes)

        try:
            # Create DataFrame for Elite Quant
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes if len(volumes) == len(prices) else np.ones(len(prices)),
                'high': prices * 1.0002,
                'low': prices * 0.9998,
                'open': np.roll(prices, 1)
            })
            df['open'].iloc[0] = df['close'].iloc[0]

            result = elite_quant.compute_all_features(df)

            # Extract last values
            for col in result.columns:
                if col not in ['close', 'volume', 'high', 'low', 'open']:
                    val = result[col].iloc[-1]
                    if not np.isnan(val) and not np.isinf(val):
                        features[f'elite_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Elite quant computation error: {e}")
            return self._compute_basic_elite_quant(prices, volumes)

        return features

    def _compute_basic_elite_quant(self, prices: np.ndarray,
                                   volumes: np.ndarray) -> Dict[str, float]:
        """Basic elite quant features without full module."""
        features = {}

        if len(prices) < 30:
            return features

        returns = np.diff(np.log(prices))

        # Higher-order moments
        if len(returns) >= 20:
            # Skewness
            mean_ret = np.mean(returns[-20:])
            std_ret = np.std(returns[-20:])
            if std_ret > 0:
                skew = np.mean(((returns[-20:] - mean_ret) / std_ret) ** 3)
                features['elite_skewness'] = float(skew) if not np.isnan(skew) else 0.0

            # Kurtosis
            if std_ret > 0:
                kurt = np.mean(((returns[-20:] - mean_ret) / std_ret) ** 4) - 3
                features['elite_kurtosis'] = float(kurt) if not np.isnan(kurt) else 0.0

        # Tail risk (CVaR approximation)
        if len(returns) >= 50:
            sorted_returns = np.sort(returns[-50:])
            var_5 = np.percentile(sorted_returns, 5)
            cvar = np.mean(sorted_returns[sorted_returns <= var_5]) if np.sum(sorted_returns <= var_5) > 0 else var_5
            features['elite_cvar_5'] = float(cvar) if not np.isnan(cvar) else 0.0

        # Intraday momentum (first half vs second half)
        if len(returns) >= 20:
            half = len(returns) // 2
            first_half_ret = np.sum(returns[:half])
            second_half_ret = np.sum(returns[half:])
            features['elite_intraday_mom'] = float(first_half_ret - second_half_ret)

        # Cross-sectional momentum proxy (deviation from rolling mean)
        if len(returns) >= 20:
            rolling_mean = np.mean(returns[-20:])
            current_ret = returns[-1]
            features['elite_cs_momentum'] = float(current_ret - rolling_mean) if not np.isnan(current_ret) else 0.0

        return features

    def _compute_genetic_mining_features(self, prices: np.ndarray,
                                         volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute Genetic Factor Mining features.
        Source: 华泰金工遗传规划因子挖掘, 东方金工 DFQ系统
        """
        features = {}

        if len(prices) < 20:
            return features

        genetic_mining = self._lazy_load_genetic_mining()

        if genetic_mining is not None:
            try:
                df = pd.DataFrame({
                    'close': prices,
                    'volume': volumes if len(volumes) == len(prices) else np.zeros_like(prices)
                })
                df['open'] = np.roll(prices, 1)
                df['open'][0] = prices[0]
                df['high'] = np.maximum(prices, df['open'])
                df['low'] = np.minimum(prices, df['open'])

                result = genetic_mining.compute_all_features(df)

                if isinstance(result, pd.DataFrame) and not result.empty:
                    for col in result.columns:
                        val = result[col].iloc[-1]
                        if not np.isnan(val) and not np.isinf(val):
                            features[f'gp_{col}'] = float(val)
            except Exception as e:
                logger.debug(f"Genetic mining feature extraction failed: {e}")
        else:
            # Fallback: Basic genetic-inspired features
            returns = np.diff(np.log(prices + 1e-10))
            if len(returns) >= 10:
                features['gp_momentum_10'] = float(np.sum(returns[-10:]))
                features['gp_volatility_10'] = float(np.std(returns[-10:]))
                features['gp_range_ratio'] = float(
                    (np.max(prices[-10:]) - np.min(prices[-10:])) / (np.mean(prices[-10:]) + 1e-10)
                )

        return features

    def _compute_tsfresh_features(self, prices: np.ndarray,
                                  volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute TSFresh automatic features.
        Source: 知乎 量化小白也能自动化挖掘出6万+因子
        """
        features = {}

        if len(prices) < 20:
            return features

        tsfresh = self._lazy_load_tsfresh()

        if tsfresh is not None:
            try:
                df = pd.DataFrame({
                    'close': prices,
                    'volume': volumes if len(volumes) == len(prices) else np.zeros_like(prices)
                })
                df['open'] = np.roll(prices, 1)
                df['open'][0] = prices[0]
                df['high'] = np.maximum(prices, df['open'])
                df['low'] = np.minimum(prices, df['open'])

                result = tsfresh.compute_all_features(df)

                if isinstance(result, pd.DataFrame) and not result.empty:
                    for col in result.columns:
                        val = result[col].iloc[-1]
                        if not np.isnan(val) and not np.isinf(val):
                            features[f'ts_{col}'] = float(val)
            except Exception as e:
                logger.debug(f"TSFresh feature extraction failed: {e}")
        else:
            # Fallback: Basic rolling features
            for window in [5, 10, 20]:
                if len(prices) >= window:
                    features[f'ts_mean_{window}'] = float(np.mean(prices[-window:]))
                    features[f'ts_std_{window}'] = float(np.std(prices[-window:]))
                    features[f'ts_min_{window}'] = float(np.min(prices[-window:]))
                    features[f'ts_max_{window}'] = float(np.max(prices[-window:]))
                    features[f'ts_median_{window}'] = float(np.median(prices[-window:]))

        return features

    def _normalize_features(self, features: Dict[str, float],
                           prices: np.ndarray) -> Dict[str, float]:
        """Normalize features using z-score."""
        # Simple clipping for extreme values
        normalized = {}
        for key, val in features.items():
            if np.isnan(val) or np.isinf(val):
                normalized[key] = 0.0
            else:
                # Clip to [-10, 10] for z-scores, [-1, 1] for ratios
                if 'zscore' in key or 'mr_' in key:
                    normalized[key] = np.clip(val, -10, 10)
                elif 'ratio' in key or 'position' in key:
                    normalized[key] = np.clip(val, -1, 1)
                else:
                    normalized[key] = np.clip(val, -1000, 1000)

        return normalized

    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        if not self.feature_names:
            self.feature_names = sorted(features.keys())

        vector = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            vector[i] = features.get(name, 0.0)

        return vector

    def get_feature_dataframe(self, features: Dict[str, float]) -> pd.DataFrame:
        """Convert feature dict to single-row DataFrame."""
        return pd.DataFrame([features])

    def batch_process(self, data: pd.DataFrame, symbol: str = "EURUSD") -> pd.DataFrame:
        """
        Process entire DataFrame and add all features.

        Args:
            data: DataFrame with price data
            symbol: Trading symbol

        Returns:
            DataFrame with all features added
        """
        logger.info(f"Batch processing {len(data)} rows for {symbol}")

        # Initialize with data
        self.initialize(data.head(100), symbol)

        # Process each row
        all_features = []

        for idx, row in data.iterrows():
            if 'bid' in row and 'ask' in row:
                bid, ask = row['bid'], row['ask']
            elif 'close' in row:
                mid = row['close']
                bid = mid - 0.00005  # Assume 1 pip spread
                ask = mid + 0.00005
            else:
                continue

            volume = row.get('volume', 0.0)
            timestamp = row.get('timestamp', None)

            features = self.process_tick(symbol, bid, ask, volume, timestamp)
            features['timestamp'] = timestamp
            all_features.append(features)

        result = pd.DataFrame(all_features)

        # Merge with original data
        if 'timestamp' in data.columns:
            result = result.set_index('timestamp')
            data = data.set_index('timestamp')
            result = pd.concat([data, result], axis=1)
            result = result.reset_index()

        logger.info(f"Generated {len(result.columns)} total columns")

        return result


class FeatureSelector:
    """
    Feature selection for ML training.
    Removes low-variance and highly correlated features.
    """

    def __init__(self, min_variance: float = 0.001,
                 max_correlation: float = 0.95):
        self.min_variance = min_variance
        self.max_correlation = max_correlation
        self.selected_features: List[str] = []

    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """Fit selector to training data."""
        # Remove low variance
        variances = X.var()
        high_variance = variances[variances > self.min_variance].index.tolist()

        # Remove highly correlated
        X_filtered = X[high_variance]
        corr_matrix = X_filtered.corr().abs()

        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [col for col in upper.columns
                   if any(upper[col] > self.max_correlation)]

        self.selected_features = [f for f in high_variance if f not in to_drop]

        logger.info(f"Selected {len(self.selected_features)} features "
                   f"from {len(X.columns)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features only."""
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def create_hft_feature_engine(config: FeatureConfig = None) -> HFTFeatureEngine:
    """Factory function to create feature engine."""
    return HFTFeatureEngine(config)


if __name__ == '__main__':
    print("HFT Feature Engine Test")
    print("=" * 50)

    # Generate synthetic test data
    np.random.seed(42)
    n = 1000

    # Simulate forex tick data
    base_price = 1.1000
    returns = np.random.randn(n) * 0.0001
    prices = base_price * np.exp(np.cumsum(returns))

    spread = 0.00005  # 0.5 pip spread
    bids = prices - spread / 2
    asks = prices + spread / 2

    data = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-10', periods=n, freq='100ms'),
        'bid': bids,
        'ask': asks,
        'volume': np.random.exponential(100, n)
    })

    print(f"Test data: {len(data)} ticks")
    print(f"Price range: {bids.min():.5f} - {asks.max():.5f}")

    # Create engine
    engine = HFTFeatureEngine()

    # Batch process
    result = engine.batch_process(data, "EURUSD")

    print(f"\nGenerated features: {len(engine.feature_names)}")
    print(f"Feature names: {engine.feature_names[:20]}...")

    # Show sample features
    print("\nSample feature values (last row):")
    feature_cols = [c for c in result.columns if c not in ['timestamp', 'bid', 'ask', 'volume']]
    for col in feature_cols[:10]:
        val = result[col].iloc[-1]
        print(f"  {col}: {val:.6f}")

    # Real-time processing test
    print("\n" + "=" * 50)
    print("Real-time processing test")

    engine2 = HFTFeatureEngine()
    engine2.initialize(data.head(100), "EURUSD")

    import time
    start = time.time()

    for i in range(100, 200):
        features = engine2.process_tick(
            "EURUSD",
            data['bid'].iloc[i],
            data['ask'].iloc[i],
            data['volume'].iloc[i],
            data['timestamp'].iloc[i]
        )

    elapsed = time.time() - start
    print(f"Processed 100 ticks in {elapsed*1000:.2f}ms")
    print(f"Per-tick latency: {elapsed*10:.3f}ms")
