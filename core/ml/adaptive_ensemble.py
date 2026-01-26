"""
Adaptive Ensemble with Chinese Quant Online Learning
=====================================================

Integrates Chinese quant-style online learning with the trading bot.

Features:
1. Load pre-trained models as base
2. Enable incremental online updates
3. Regime-adaptive ensemble weighting
4. Drift detection with auto-retrain
5. Hot-swap models without stopping trading

Usage:
    from core.ml.adaptive_ensemble import AdaptiveMLEnsemble

    ensemble = AdaptiveMLEnsemble(model_dir=Path("models/production"))
    ensemble.load_models(["EURUSD", "GBPUSD"])
    ensemble.enable_online_learning()

    # In trading loop
    signal = ensemble.predict(symbol, features)
    ensemble.add_observation(symbol, features, actual_direction, price)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Import Chinese online learning
from core.ml.chinese_online_learning import (
    ChineseQuantOnlineLearner,
    IncrementalXGBoost,
    IncrementalLightGBM,
    IncrementalCatBoost,
    DriftDetector,
    RegimeDetector,
    DriftMetrics,
    RegimeState,
)

# Import DoubleAdapt for two-stage adaptation [Zhang et al. KDD 2023]
try:
    from core.ml.double_adapt import DoubleAdapt, AdaptationResult
    DOUBLE_ADAPT_AVAILABLE = True
except ImportError:
    DOUBLE_ADAPT_AVAILABLE = False
    DoubleAdapt = None
    AdaptationResult = None

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from ML models."""
    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0-1
    predicted_return: float  # Expected return in bps
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_votes: Dict[str, int] = field(default_factory=dict)
    regime: str = "unknown"
    drift_detected: bool = False


class AdaptiveMLEnsemble:
    """
    ML Ensemble with Chinese Quant-style Online Learning.

    Extends the basic MLEnsemble with:
    1. Incremental learning (增量学习)
    2. Regime detection (市场状态识别)
    3. Drift detection (概念漂移检测)
    4. Adaptive ensemble weights
    5. Hot model updates
    """

    def __init__(
        self,
        model_dir: Path = None,
        online_model_dir: Path = None,
        enable_online_learning: bool = True,
        update_interval: int = 60,  # seconds
        min_samples_for_update: int = 500,
        live_weight: float = 3.0,
    ):
        self.model_dir = model_dir or Path("models/production")
        self.online_model_dir = online_model_dir or Path("models/production/online")

        # Settings
        self.enable_online = enable_online_learning
        self.update_interval = update_interval
        self.min_samples = min_samples_for_update
        self.live_weight = live_weight

        # Static models (from training)
        self.static_models: Dict[str, Dict] = {}
        self.feature_names: List[str] = []

        # Online learners (one per symbol)
        self.online_learners: Dict[str, ChineseQuantOnlineLearner] = {}

        # Shared detectors
        self.regime_detector = RegimeDetector()
        self.current_regime = RegimeState()

        # Data buffers
        self.observation_buffer: Dict[str, deque] = {}

        # State
        self.loaded = False
        self.online_enabled = False
        self.last_update_times: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._running = False

        # Metrics
        self.prediction_count = 0
        self.update_count = 0

        # DoubleAdapt meta-learning [Zhang et al. KDD 2023]
        # Two-layer adaptation: data + model [Zhang et al. 2023, Section 3.2]
        # inner_lr = task-specific learning rate [Finn et al. ICML 2017]
        # Triggers on regime change for fast adaptation
        self.double_adapt: Optional[DoubleAdapt] = None
        self._last_regime: Optional[str] = None
        self._adaptation_results: List[Any] = []
        if DOUBLE_ADAPT_AVAILABLE:
            self.double_adapt = DoubleAdapt(
                n_reference_samples=500,
                n_historical_samples=5000,
                adaptation_threshold=0.3,
                inner_lr=0.01,  # α in MAML [Finn et al. 2017, Eq. 1]
                n_inner_steps=5,  # Fast adaptation steps
            )
            logger.info("DoubleAdapt meta-learning enabled [Zhang et al. KDD 2023]")

    def load_models(self, symbols: List[str] = None):
        """Load pre-trained static models."""
        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            model_path = self.model_dir / f"{symbol}_models.pkl"

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)

                    # Support multiple model formats
                    adapted = None

                    # Format 1: train_models.py format (target_direction_N/xgboost/etc)
                    # Supports target_direction_1, target_direction_5, target_direction_10, etc.
                    target_keys = [k for k in data.keys() if k.startswith('target_direction_')]
                    if target_keys:
                        # Use first available target
                        td_key = target_keys[0]
                        td = data[td_key]
                        adapted = {
                            'models': {
                                'xgboost': td.get('xgboost'),
                                'lightgbm': td.get('lightgbm'),
                                'catboost': td.get('catboost')
                            },
                            'feature_names': td.get('features', []),
                            'n_features': len(td.get('features', [])),
                        }

                    # Format 2: New train_parallel_max.py format (xgb/lgb/cb)
                    elif 'xgb' in data or 'lgb' in data or 'cb' in data:
                        adapted = {
                            'models': {
                                'xgboost': data.get('xgb'),
                                'lightgbm': data.get('lgb'),
                                'catboost': data.get('cb')
                            },
                            'feature_names': [],
                            'n_features': 0  # Track expected feature count
                        }
                        # Try to get feature count from XGBoost Booster
                        xgb_model = data.get('xgb')
                        if xgb_model and hasattr(xgb_model, 'num_features'):
                            adapted['n_features'] = xgb_model.num_features()
                        # Try to get feature names from CatBoost
                        cb = data.get('cb')
                        if cb and hasattr(cb, 'feature_names_'):
                            adapted['feature_names'] = list(cb.feature_names_)
                            adapted['n_features'] = len(cb.feature_names_)

                    if adapted:
                        # Filter out None models
                        adapted['models'] = {k: v for k, v in adapted['models'].items() if v is not None}
                        self.static_models[symbol] = adapted

                        if not self.feature_names and adapted['feature_names']:
                            self.feature_names = adapted['feature_names']

                        n_feat = adapted.get('n_features', len(adapted['feature_names']))
                        logger.info(f"[{symbol}] Loaded static models: {list(adapted['models'].keys())} "
                                   f"({n_feat} features)")
                    else:
                        logger.warning(f"[{symbol}] Unknown model format: {list(data.keys())}")

                except Exception as e:
                    logger.error(f"[{symbol}] Failed to load: {e}")
            else:
                logger.warning(f"[{symbol}] No models at {model_path}")

            # Initialize observation buffer
            self.observation_buffer[symbol] = deque(maxlen=10000)
            self.last_update_times[symbol] = 0

        self.loaded = len(self.static_models) > 0
        logger.info(f"Loaded {len(self.static_models)} static model sets")

    def init_online_learning(self, symbols: List[str] = None):
        """
        Initialize online learners from static models.

        This transfers the pre-trained models to incremental learners.
        """
        if symbols is None:
            symbols = list(self.static_models.keys())

        self.online_model_dir.mkdir(parents=True, exist_ok=True)

        for symbol in symbols:
            if symbol not in self.static_models:
                logger.warning(f"[{symbol}] No static model to initialize from")
                continue

            # Create online learner
            learner = ChineseQuantOnlineLearner(
                symbol=symbol,
                model_dir=self.online_model_dir,
                update_interval=self.update_interval,
                min_samples_for_update=self.min_samples,
                live_weight=self.live_weight,
            )

            # Transfer static models to incremental learners
            static = self.static_models[symbol]['models']

            if 'xgboost' in static and static['xgboost'] is not None:
                learner.xgb.model = static['xgboost']
                learner.xgb.version.version = 1
                logger.info(f"[{symbol}] Transferred XGBoost to online learner")

            if 'lightgbm' in static and static['lightgbm'] is not None:
                learner.lgb.model = static['lightgbm']
                learner.lgb.version.version = 1
                logger.info(f"[{symbol}] Transferred LightGBM to online learner")

            if 'catboost' in static and static['catboost'] is not None:
                learner.cb.model = static['catboost']
                learner.cb.version.version = 1
                logger.info(f"[{symbol}] Transferred CatBoost to online learner")

            self.online_learners[symbol] = learner

        self.online_enabled = len(self.online_learners) > 0
        logger.info(f"Initialized {len(self.online_learners)} online learners")

    def start_background_updates(self):
        """Start background thread for periodic model updates."""
        if not self.online_enabled:
            logger.warning("Online learning not enabled, skipping background updates")
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Started background update thread")

    def stop_background_updates(self):
        """Stop background update thread."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Stopped background update thread")

    def _update_loop(self):
        """Background loop for periodic model updates."""
        while self._running:
            try:
                # Update each symbol
                for symbol, learner in self.online_learners.items():
                    if learner.should_update():
                        result = learner.incremental_update()
                        if result.get("status") == "success":
                            self.update_count += 1
                            logger.info(f"[{symbol}] Online update #{result['total_updates']}: "
                                       f"acc={result['accuracy']:.4f}, regime={result['regime']}")

                # Sleep before next check
                time.sleep(10)

            except Exception as e:
                logger.error(f"Update loop error: {e}")
                time.sleep(30)

    def add_observation(
        self,
        symbol: str,
        features: np.ndarray,
        actual_direction: Optional[int],
        price: float,
    ):
        """
        Add observation for online learning.

        Called after each tick with the actual outcome.
        """
        with self._lock:
            # Add to buffer
            if symbol in self.observation_buffer:
                self.observation_buffer[symbol].append({
                    'features': features,
                    'label': actual_direction,
                    'price': price,
                    'timestamp': datetime.now(),
                })

            # Update online learner
            if self.online_enabled and symbol in self.online_learners:
                self.online_learners[symbol].add_tick(features, actual_direction, price)

            # Update regime detector
            self.regime_detector.update(price, self._last_price.get(symbol, price))
            self._last_price[symbol] = price

            # Update DoubleAdapt reference distribution [Zhang et al. KDD 2023]
            if self.double_adapt is not None:
                self.double_adapt.dist_adapter.update_reference(features)

            # Check for regime change and trigger adaptation
            current_regime_name = self.regime_detector.detect_regime().state_name
            if self._last_regime is not None and current_regime_name != self._last_regime:
                logger.info(f"[{symbol}] Regime change detected: {self._last_regime} → {current_regime_name}")
                self._trigger_regime_adaptation(symbol, features)
            self._last_regime = current_regime_name

    _last_price: Dict[str, float] = {}

    def _trigger_regime_adaptation(self, symbol: str, features: np.ndarray):
        """
        Trigger DoubleAdapt on regime change. [Zhang et al. KDD 2023]

        DoubleAdapt performs: [Zhang et al. 2023, Algorithm 1]
        1. Data adapter: transforms features to new distribution
        2. Model adapter: fine-tunes weights with meta-gradient
        """
        if self.double_adapt is None or not DOUBLE_ADAPT_AVAILABLE:
            return

        # Check if adaptation is needed based on distribution shift
        if not self.double_adapt.should_adapt():
            return

        # Get validation data from buffer
        buffer = self.observation_buffer.get(symbol, deque())
        if len(buffer) < 100:
            return

        # Extract recent samples for validation
        recent = list(buffer)[-100:]
        X_val = np.array([s['features'] for s in recent])
        y_val = np.array([s['label'] for s in recent if s['label'] is not None])

        if len(y_val) < 50:
            return

        # Adapt each model in the ensemble
        if symbol in self.static_models:
            static = self.static_models[symbol]['models']
            for model_name, model in static.items():
                if model is None:
                    continue
                try:
                    adapted_model, result = self.double_adapt.adapt(model, X_val, y_val)
                    if result.should_deploy:
                        static[model_name] = adapted_model
                        self._adaptation_results.append(result)
                        logger.info(f"[{symbol}] {model_name} adapted: "
                                   f"acc {result.val_accuracy_before:.3f} → {result.val_accuracy_after:.3f}")
                except Exception as e:
                    logger.warning(f"[{symbol}] {model_name} adaptation failed: {e}")

    def on_regime_change(self, symbol: str, new_data: np.ndarray):
        """
        Called by drift detector on regime shift. [Gama et al. 2014]

        DoubleAdapt performs: [Zhang et al. 2023, Algorithm 1]
        1. Data adapter: transforms features to new distribution
        2. Model adapter: fine-tunes weights with meta-gradient

        Args:
            symbol: Trading symbol
            new_data: Recent feature data from new regime
        """
        if self.double_adapt is None:
            return

        self._trigger_regime_adaptation(symbol, new_data)

    def predict(self, symbol: str, features: Dict[str, float]) -> Optional[Signal]:
        """
        Generate trading signal with adaptive ensemble.

        Uses online learners if enabled, falls back to static models.
        """
        self.prediction_count += 1

        # Convert features to array
        if self.feature_names:
            X = np.array([features.get(f, 0.0) for f in self.feature_names])
        else:
            X = np.array(list(features.values()))

        # Debug: log feature count periodically
        if self.prediction_count % 500 == 1:
            logger.info(f"[{symbol}] Predict: features={len(X)}, feature_names={len(self.feature_names)}")

        # Get current regime
        self.current_regime = self.regime_detector.detect_regime()

        predictions = {}
        probabilities = {}
        drift_detected = False

        # Try online learners first
        if self.online_enabled and symbol in self.online_learners:
            learner = self.online_learners[symbol]
            try:
                prob, conf = learner.predict(X)
                probabilities['online_ensemble'] = prob
                predictions['online_ensemble'] = 1 if prob > 0.5 else 0

                # Check for drift
                if learner.drift_detector:
                    drift_metrics = learner.drift_detector.check_drift(X.reshape(1, -1))
                    drift_detected = drift_metrics.should_retrain

            except Exception as e:
                logger.warning(f"[{symbol}] Online prediction failed: {e}")

        # Fallback or supplement with static models
        if symbol in self.static_models:
            static_data = self.static_models[symbol]
            static = static_data['models']
            n_features = static_data.get('n_features', 0)

            # Truncate features to match model's expected count
            X_static = X
            if n_features > 0 and len(X) > n_features:
                X_static = X[:n_features]
            elif n_features > 0 and len(X) < n_features:
                # Pad with zeros if we have fewer features
                X_static = np.pad(X, (0, n_features - len(X)), 'constant')

            for name, model in static.items():
                try:
                    X_input = X_static.reshape(1, -1)
                    proba = 0.5

                    # Handle different model types
                    if isinstance(model, xgb.Booster):
                        # XGBoost Booster - use DMatrix
                        dmat = xgb.DMatrix(X_input)
                        proba = model.predict(dmat)[0]
                    elif hasattr(model, 'predict_proba'):
                        # Sklearn-style classifier (XGBClassifier, LGBMClassifier, CatBoost)
                        proba = model.predict_proba(X_input)[0]
                        proba = proba[1] if len(proba) > 1 else proba[0]
                    elif isinstance(model, lgb.Booster):
                        # LightGBM Booster
                        proba = model.predict(X_input)[0]
                    elif hasattr(model, 'predict'):
                        # Generic predict
                        pred = model.predict(X_input)
                        proba = pred[0] if hasattr(pred, '__len__') else pred

                    probabilities[f'static_{name}'] = float(proba)
                    predictions[f'static_{name}'] = 1 if proba > 0.5 else 0

                except Exception as e:
                    logger.warning(f"[{symbol}] Static {name} prediction failed: {e}")

        if not probabilities:
            if self.prediction_count % 500 == 1:
                logger.warning(f"[{symbol}] No probabilities generated")
            return None

        # Debug: log probabilities periodically
        if self.prediction_count % 500 == 1:
            logger.info(f"[{symbol}] Probabilities: {probabilities}")

        # Weighted ensemble (favor online if available)
        weights = {}
        if 'online_ensemble' in probabilities:
            weights['online_ensemble'] = 0.6  # Trust online more
            for k in probabilities:
                if k != 'online_ensemble':
                    weights[k] = 0.4 / (len(probabilities) - 1)
        else:
            for k in probabilities:
                weights[k] = 1.0 / len(probabilities)

        # Calculate weighted average
        avg_prob = sum(probabilities[k] * weights.get(k, 0) for k in probabilities)

        # Direction and confidence
        direction = 1 if avg_prob > 0.5 else -1
        raw_confidence = abs(avg_prob - 0.5) * 2
        confidence = raw_confidence

        # Adjust confidence based on regime
        if self.current_regime.state == 1:  # Bear market - be more cautious
            confidence *= 0.8
        elif self.current_regime.state == 2:  # Sideways - be very cautious
            confidence *= 0.7

        # Debug: log confidence periodically
        if self.prediction_count % 100 == 1:
            logger.info(f"[{symbol}] Signal gen: avg_prob={avg_prob:.4f}, raw_conf={raw_confidence:.4f}, "
                       f"regime={self.current_regime.state_name}, final_conf={confidence:.4f}, dir={direction}")

        # Predicted return
        predicted_return = direction * confidence * 10

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            features=features,
            model_votes={k: v for k, v in predictions.items()},
            regime=self.current_regime.state_name,
            drift_detected=drift_detected,
        )

    def force_update(self, symbol: str) -> Dict[str, Any]:
        """Force an immediate model update for a symbol."""
        if symbol not in self.online_learners:
            return {"status": "error", "reason": "no online learner"}

        return self.online_learners[symbol].incremental_update(force=True)

    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status."""
        status = {
            "loaded": self.loaded,
            "online_enabled": self.online_enabled,
            "static_models": list(self.static_models.keys()),
            "online_learners": list(self.online_learners.keys()),
            "predictions": self.prediction_count,
            "updates": self.update_count,
            "current_regime": self.current_regime.state_name,
            # DoubleAdapt status [Zhang et al. KDD 2023]
            "double_adapt_enabled": self.double_adapt is not None,
            "double_adapt_adaptations": len(self._adaptation_results),
        }

        # Per-symbol stats
        for symbol, learner in self.online_learners.items():
            status[f"{symbol}_updates"] = learner.total_updates
            status[f"{symbol}_buffer_size"] = len(learner.feature_buffer)

        # DoubleAdapt distribution shift [Zhang et al. KDD 2023]
        if self.double_adapt is not None:
            try:
                status["double_adapt_distribution_shift"] = self.double_adapt.dist_adapter.get_distribution_shift()
            except Exception:
                pass

        return status

    def save_all(self):
        """Save all online learners."""
        for symbol, learner in self.online_learners.items():
            learner.save_models()
        logger.info(f"Saved {len(self.online_learners)} online learners")


# Factory function
def create_adaptive_ensemble(
    model_dir: str = "models/production",
    enable_online: bool = True,
    **kwargs
) -> AdaptiveMLEnsemble:
    """Create an adaptive ensemble with online learning."""
    ensemble = AdaptiveMLEnsemble(
        model_dir=Path(model_dir),
        enable_online_learning=enable_online,
        **kwargs
    )
    return ensemble


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create ensemble
    ensemble = create_adaptive_ensemble()

    # Load static models
    ensemble.load_models(["EURUSD", "GBPUSD", "USDJPY"])

    # Initialize online learning
    ensemble.init_online_learning()

    # Start background updates
    ensemble.start_background_updates()

    # Simulate trading
    import random
    for i in range(100):
        # Fake features
        features = {f"feature_{j}": random.random() for j in range(100)}

        # Predict
        for symbol in ["EURUSD", "GBPUSD"]:
            signal = ensemble.predict(symbol, features)
            if signal:
                print(f"[{i}] {symbol}: dir={signal.direction}, conf={signal.confidence:.3f}, "
                      f"regime={signal.regime}, drift={signal.drift_detected}")

        # Add observation
        for symbol in ["EURUSD", "GBPUSD"]:
            X = np.array(list(features.values()))
            actual = random.choice([0, 1])
            price = 1.1 + random.random() * 0.01
            ensemble.add_observation(symbol, X, actual, price)

        time.sleep(0.1)

    # Status
    print("\nStatus:", ensemble.get_status())

    # Stop
    ensemble.stop_background_updates()
    ensemble.save_all()
