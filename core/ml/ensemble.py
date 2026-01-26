"""
Hot Swappable Ensemble Manager
==============================
Atomic model replacement for live trading without interruption.
"""

import threading
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Tracks a model version."""
    version: int
    timestamp: datetime
    accuracy: float
    auc: float
    n_samples: int
    models: Dict[str, Any]  # xgboost, lightgbm, catboost
    feature_names: list


class HotSwappableEnsemble:
    """
    Thread-safe ensemble manager with atomic model swapping.

    Allows retraining in background while predictions continue
    using current models. New models are swapped in atomically.
    """

    def __init__(self, symbol: str, model_dir: Path = None):
        """
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            model_dir: Directory to save/load models
        """
        self.symbol = symbol
        self.model_dir = model_dir or Path("models/production")

        # Current active models (read by predictions)
        self._current: Optional[ModelVersion] = None

        # Previous version (fallback)
        self._previous: Optional[ModelVersion] = None

        # Version counter
        self._version = 0

        # Lock for atomic swaps
        self._lock = threading.RLock()

        # Load initial models if available
        self._load_initial_models()

    def _load_initial_models(self):
        """Load models from disk if available."""
        model_file = self.model_dir / f"{self.symbol}_models.pkl"
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    models_dict = pickle.load(f)

                # Find the first target's models
                for target, model_data in models_dict.items():
                    self._current = ModelVersion(
                        version=0,
                        timestamp=datetime.now(),
                        accuracy=0.0,  # Unknown for loaded models
                        auc=0.0,
                        n_samples=0,
                        models=model_data,
                        feature_names=model_data.get('features', [])
                    )
                    self._version = 1
                    logger.info(f"Loaded initial models for {self.symbol}")
                    break
            except Exception as e:
                logger.error(f"Failed to load models: {e}")

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Make ensemble prediction using current models.

        Args:
            features: Feature dict from HFTFeatureEngine

        Returns:
            (probability, confidence) - probability of up move, confidence
        """
        with self._lock:
            if self._current is None:
                return 0.5, 0.0  # No model = no prediction

            models = self._current.models
            feature_names = self._current.feature_names

        # Extract features in correct order
        try:
            X = np.array([[features.get(f, 0.0) for f in feature_names]])
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return 0.5, 0.0

        # Get predictions from each model
        predictions = []

        try:
            import xgboost as xgb
            if 'xgboost' in models:
                dmatrix = xgb.DMatrix(X)
                pred = models['xgboost'].predict(dmatrix)[0]
                predictions.append(pred)
        except Exception as e:
            logger.debug(f"XGBoost prediction error: {e}")

        try:
            if 'lightgbm' in models:
                pred = models['lightgbm'].predict(X)[0]
                predictions.append(pred)
        except Exception as e:
            logger.debug(f"LightGBM prediction error: {e}")

        try:
            if 'catboost' in models:
                pred = models['catboost'].predict_proba(X)[0, 1]
                predictions.append(pred)
        except Exception as e:
            logger.debug(f"CatBoost prediction error: {e}")

        if not predictions:
            return 0.5, 0.0

        # Ensemble average
        prob = np.mean(predictions)
        # Confidence = agreement between models
        confidence = 1.0 - np.std(predictions) * 2 if len(predictions) > 1 else 0.5

        return float(prob), float(confidence)

    def hot_swap(self, new_models: Dict[str, Any], feature_names: list,
                 accuracy: float, auc: float, n_samples: int) -> bool:
        """
        Atomically swap in new models.

        Args:
            new_models: Dict with 'xgboost', 'lightgbm', 'catboost' models
            feature_names: List of feature names
            accuracy: Validation accuracy
            auc: Validation AUC
            n_samples: Number of training samples

        Returns:
            True if swap successful
        """
        with self._lock:
            # Check if new models are better
            if self._current is not None:
                improvement = accuracy - self._current.accuracy
                if improvement < 0.005:  # Require 0.5% improvement
                    logger.info(f"[{self.symbol}] No improvement: {accuracy:.4f} vs {self._current.accuracy:.4f}")
                    return False

            # Create new version
            new_version = ModelVersion(
                version=self._version,
                timestamp=datetime.now(),
                accuracy=accuracy,
                auc=auc,
                n_samples=n_samples,
                models=new_models,
                feature_names=feature_names
            )

            # Atomic swap
            self._previous = self._current
            self._current = new_version
            self._version += 1

            logger.info(f"[{self.symbol}] HOT-SWAP: v{new_version.version} "
                       f"acc={accuracy:.4f} auc={auc:.4f} samples={n_samples}")

            # Save to disk
            self._save_models()

            return True

    def _save_models(self):
        """Save current models to disk."""
        if self._current is None:
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Save in format compatible with existing code
        model_file = self.model_dir / f"{self.symbol}_models.pkl"
        models_dict = {
            'target_direction_10': {
                'xgboost': self._current.models.get('xgboost'),
                'lightgbm': self._current.models.get('lightgbm'),
                'catboost': self._current.models.get('catboost'),
                'features': self._current.feature_names
            }
        }

        try:
            with open(model_file, 'wb') as f:
                pickle.dump(models_dict, f)
            logger.info(f"Saved models to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def rollback(self) -> bool:
        """Rollback to previous model version."""
        with self._lock:
            if self._previous is None:
                logger.warning("No previous version to rollback to")
                return False

            self._current = self._previous
            self._previous = None
            logger.info(f"[{self.symbol}] Rolled back to v{self._current.version}")
            return True

    def get_stats(self) -> Dict:
        """Get current model stats."""
        with self._lock:
            if self._current is None:
                return {'status': 'no_model'}

            return {
                'symbol': self.symbol,
                'version': self._current.version,
                'accuracy': self._current.accuracy,
                'auc': self._current.auc,
                'n_samples': self._current.n_samples,
                'timestamp': self._current.timestamp.isoformat(),
                'has_fallback': self._previous is not None
            }


# ==============================================================================
# GAP 5: Walk-Forward Stacking Ensemble
# ==============================================================================
# Citations:
# - van der Laan et al. (2007) "Super Learner" - https://www.degruyter.com/document/doi/10.2202/1544-6115.1309
# - Wolpert (1992) "Stacked Generalization" - https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231
# - de Prado (2018) "Advances in Financial ML" Chapter 7: Cross-Validation in Finance
# ==============================================================================


class WalkForwardStackingEnsemble:
    """
    Walk-Forward Stacking Ensemble for time series.

    Uses TimeSeriesSplit to generate out-of-sample predictions from base models,
    then trains a meta-learner on these OOS predictions. This prevents information
    leakage that occurs with standard stacking on time series data.

    Based on:
    - Super Learner (van der Laan et al. 2007)
    - Stacked Generalization (Wolpert 1992)
    - Financial ML best practices (de Prado 2018)
    """

    def __init__(
        self,
        base_models: list = None,
        meta_learner = None,
        n_splits: int = 5,
        min_train_size: int = 1000,
        gap: int = 10,  # Purging gap to prevent leakage
    ):
        """
        Args:
            base_models: List of (name, model) tuples for base learners
            meta_learner: Model for meta-learning (default: XGBoost)
            n_splits: Number of time series splits
            min_train_size: Minimum samples in first training fold
            gap: Gap between train and validation to prevent leakage
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap = gap

        # Default base models (GPU-accelerated)
        if base_models is None:
            self.base_models = self._create_default_base_models()
        else:
            self.base_models = base_models

        # Default meta-learner
        if meta_learner is None:
            self.meta_learner = self._create_default_meta_learner()
        else:
            self.meta_learner = meta_learner

        # Fitted models
        self._fitted_base_models = {}
        self._fitted_meta_learner = None
        self._is_fitted = False

        # Calibration scores
        self.oos_accuracy = None
        self.oos_auc = None
        self.meta_confidence = None

    def _create_default_base_models(self) -> list:
        """Create default GPU-accelerated base models."""
        models = []

        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                tree_method='hist',
                device='cuda',
                random_state=42,
                n_jobs=-1,
            )
            models.append(('xgboost', xgb_model))
        except ImportError:
            pass

        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                device='gpu',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            models.append(('lightgbm', lgb_model))
        except ImportError:
            pass

        try:
            from catboost import CatBoostClassifier
            cb_model = CatBoostClassifier(
                iterations=200,
                depth=8,
                learning_rate=0.05,
                task_type='GPU',
                random_state=42,
                verbose=False,
            )
            models.append(('catboost', cb_model))
        except ImportError:
            pass

        if not models:
            # Fallback to sklearn
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ]

        return models

    def _create_default_meta_learner(self):
        """Create default meta-learner (XGBoost)."""
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                tree_method='hist',
                device='cuda',
                random_state=42,
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42)

    def _time_series_split(self, n_samples: int):
        """
        Generate time series split indices with purging gap.

        Yields (train_idx, val_idx) tuples where train always comes before val
        and there's a gap between them to prevent information leakage.
        """
        fold_size = (n_samples - self.min_train_size) // self.n_splits

        for i in range(self.n_splits):
            train_end = self.min_train_size + i * fold_size
            val_start = train_end + self.gap
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(val_idx) > 0:
                yield train_idx, val_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WalkForwardStackingEnsemble':
        """
        Fit the stacking ensemble using walk-forward validation.

        1. Generate OOS predictions from each base model using time series CV
        2. Train meta-learner on OOS predictions
        3. Retrain base models on full data

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            self
        """
        n_samples = len(X)
        n_base = len(self.base_models)

        logger.info(f"[WalkForwardStacking] Fitting with {n_samples} samples, "
                   f"{n_base} base models, {self.n_splits} splits")

        # Initialize OOS predictions matrix
        oos_predictions = np.zeros((n_samples, n_base))
        oos_valid = np.zeros(n_samples, dtype=bool)

        # Generate OOS predictions for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(self._time_series_split(n_samples)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                try:
                    # Clone model for this fold
                    import copy
                    fold_model = copy.deepcopy(model)

                    # Fit on training data
                    fold_model.fit(X_train, y_train)

                    # Get OOS predictions
                    if hasattr(fold_model, 'predict_proba'):
                        preds = fold_model.predict_proba(X_val)[:, 1]
                    else:
                        preds = fold_model.predict(X_val)

                    oos_predictions[val_idx, model_idx] = preds

                except Exception as e:
                    logger.warning(f"Fold {fold_idx} {name} error: {e}")
                    oos_predictions[val_idx, model_idx] = 0.5

            oos_valid[val_idx] = True

            logger.debug(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

        # Train meta-learner on OOS predictions
        valid_mask = oos_valid & (oos_predictions.sum(axis=1) > 0)
        n_valid = valid_mask.sum()

        if n_valid < 100:
            logger.warning(f"Only {n_valid} valid OOS samples, meta-learner may be unstable")

        logger.info(f"[WalkForwardStacking] Training meta-learner on {n_valid} OOS samples")

        try:
            self.meta_learner.fit(oos_predictions[valid_mask], y[valid_mask])
            self._fitted_meta_learner = self.meta_learner

            # Calculate OOS metrics
            meta_preds = self.meta_learner.predict_proba(oos_predictions[valid_mask])[:, 1]
            self.oos_accuracy = np.mean((meta_preds > 0.5) == y[valid_mask])

            from sklearn.metrics import roc_auc_score
            self.oos_auc = roc_auc_score(y[valid_mask], meta_preds)

            # Meta confidence = mean probability for correct predictions
            correct = (meta_preds > 0.5) == y[valid_mask]
            self.meta_confidence = np.mean(np.abs(meta_preds - 0.5)[correct]) * 2

            logger.info(f"[WalkForwardStacking] OOS acc={self.oos_accuracy:.4f}, "
                       f"auc={self.oos_auc:.4f}, confidence={self.meta_confidence:.4f}")

        except Exception as e:
            logger.error(f"Meta-learner training failed: {e}")
            return self

        # Retrain base models on full data for production
        logger.info("[WalkForwardStacking] Retraining base models on full data")

        for name, model in self.base_models:
            try:
                import copy
                final_model = copy.deepcopy(model)
                final_model.fit(X, y)
                self._fitted_base_models[name] = final_model
            except Exception as e:
                logger.warning(f"Final {name} training error: {e}")

        self._is_fitted = True
        logger.info("[WalkForwardStacking] Fit complete")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the stacking ensemble.

        Args:
            X: Feature matrix

        Returns:
            Probability array (n_samples, 2) for binary classification
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        n_samples = X.shape[0]
        base_preds = np.zeros((n_samples, len(self.base_models)))

        for i, (name, _) in enumerate(self.base_models):
            if name in self._fitted_base_models:
                model = self._fitted_base_models[name]
                if hasattr(model, 'predict_proba'):
                    base_preds[:, i] = model.predict_proba(X)[:, 1]
                else:
                    base_preds[:, i] = model.predict(X)
            else:
                base_preds[:, i] = 0.5

        # Meta-learner prediction
        meta_proba = self._fitted_meta_learner.predict_proba(base_preds)

        return meta_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def get_meta_confidence(self) -> float:
        """
        Get meta-learner confidence score.

        Returns:
            Confidence score (0-1) or None if not fitted
        """
        return self.meta_confidence if self._is_fitted else None

    def passes_confidence_threshold(self, threshold: float = 0.7) -> bool:
        """
        Check if meta-learner confidence exceeds threshold.

        Args:
            threshold: Minimum confidence required

        Returns:
            True if confidence > threshold
        """
        if self.meta_confidence is None:
            return False
        return self.meta_confidence > threshold

    def get_stats(self) -> Dict:
        """Get ensemble statistics."""
        return {
            'is_fitted': self._is_fitted,
            'n_base_models': len(self.base_models),
            'n_splits': self.n_splits,
            'oos_accuracy': self.oos_accuracy,
            'oos_auc': self.oos_auc,
            'meta_confidence': self.meta_confidence,
            'base_models': [name for name, _ in self.base_models],
            'fitted_models': list(self._fitted_base_models.keys()),
        }


def create_walk_forward_stacking(
    n_splits: int = 5,
    min_train_size: int = 1000,
    gap: int = 10,
) -> WalkForwardStackingEnsemble:
    """
    Factory function to create WalkForwardStackingEnsemble.

    Args:
        n_splits: Number of time series splits
        min_train_size: Minimum training samples in first fold
        gap: Purging gap between train and validation

    Returns:
        Configured WalkForwardStackingEnsemble
    """
    return WalkForwardStackingEnsemble(
        n_splits=n_splits,
        min_train_size=min_train_size,
        gap=gap,
    )


# Global registry of ensemble managers
_ensembles: Dict[str, HotSwappableEnsemble] = {}
_registry_lock = threading.Lock()


def get_ensemble(symbol: str) -> HotSwappableEnsemble:
    """Get or create ensemble manager for symbol."""
    global _ensembles
    with _registry_lock:
        if symbol not in _ensembles:
            _ensembles[symbol] = HotSwappableEnsemble(symbol)
        return _ensembles[symbol]
