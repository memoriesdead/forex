"""
Signal Generator
================
ML-based signal generation for trading decisions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal with direction and confidence."""
    symbol: str
    direction: int           # 1 = long, -1 = short, 0 = neutral
    confidence: float        # 0.0 to 1.0
    probability: float       # Raw model probability
    model_votes: Dict[str, int]  # Per-model votes

    @property
    def is_valid(self) -> bool:
        return self.direction != 0 and self.confidence > 0


class SignalGenerator:
    """
    Generate trading signals using ML ensemble.

    Uses lazy-loaded models via ModelLoader.
    Ensemble voting: XGBoost + LightGBM + CatBoost majority.
    """

    def __init__(self, model_loader=None):
        """
        Initialize signal generator.

        Args:
            model_loader: ModelLoader instance (created if None)
        """
        if model_loader is None:
            from core.models import ModelLoader
            model_loader = ModelLoader()

        self.model_loader = model_loader
        self._feature_cache: Dict[str, List[str]] = {}

    def predict(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[Signal]:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Trading symbol
            features: Feature dictionary from feature engine

        Returns:
            Signal or None if prediction fails
        """
        # Load model
        model_data = self.model_loader.load(symbol)
        if model_data is None:
            logger.debug(f"No model for {symbol}")
            return None

        models = model_data.get('models', {})
        feature_names = model_data.get('feature_names', [])

        if not models or not feature_names:
            logger.warning(f"Invalid model data for {symbol}")
            return None

        # Prepare feature vector
        X = self._prepare_features(features, feature_names)
        if X is None:
            return None

        # Get predictions from each model
        predictions = {}
        probabilities = []

        # XGBoost
        if 'xgboost' in models and models['xgboost'] is not None:
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X.reshape(1, -1))
                prob = float(models['xgboost'].predict(dmatrix)[0])
                predictions['xgboost'] = 1 if prob > 0.5 else -1
                probabilities.append(prob)
            except Exception as e:
                logger.debug(f"XGBoost prediction error: {e}")

        # LightGBM
        if 'lightgbm' in models and models['lightgbm'] is not None:
            try:
                prob = float(models['lightgbm'].predict(X.reshape(1, -1))[0])
                predictions['lightgbm'] = 1 if prob > 0.5 else -1
                probabilities.append(prob)
            except Exception as e:
                logger.debug(f"LightGBM prediction error: {e}")

        # CatBoost
        if 'catboost' in models and models['catboost'] is not None:
            try:
                prob = float(models['catboost'].predict_proba(X.reshape(1, -1))[0, 1])
                predictions['catboost'] = 1 if prob > 0.5 else -1
                probabilities.append(prob)
            except Exception as e:
                logger.debug(f"CatBoost prediction error: {e}")

        if not predictions:
            logger.warning(f"No valid predictions for {symbol}")
            return None

        # Ensemble voting
        votes = list(predictions.values())
        avg_prob = np.mean(probabilities) if probabilities else 0.5

        # Majority vote
        long_votes = sum(1 for v in votes if v == 1)
        short_votes = sum(1 for v in votes if v == -1)

        if long_votes > short_votes:
            direction = 1
        elif short_votes > long_votes:
            direction = -1
        else:
            direction = 0  # Tie = no signal

        # Confidence from probability distance from 0.5
        confidence = abs(avg_prob - 0.5) * 2  # Scale to 0-1

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            probability=avg_prob,
            model_votes=predictions
        )

    def _prepare_features(
        self,
        features: Dict[str, float],
        feature_names: List[str]
    ) -> Optional[np.ndarray]:
        """Prepare feature vector in correct order."""
        if not feature_names:
            return None

        X = np.zeros(len(feature_names))

        for i, name in enumerate(feature_names):
            if name in features:
                X[i] = features[name]
            else:
                X[i] = 0.0  # Missing features default to 0

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def get_available_symbols(self) -> List[str]:
        """Get symbols with available models."""
        return self.model_loader.get_available()

    def preload_models(self, symbols: List[str]):
        """Preload models for faster prediction."""
        self.model_loader.preload(symbols)
