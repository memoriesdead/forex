"""
Live Retrainer - Background GPU Training
========================================
Continuously retrains models on RTX 5080 using live tick data.
"""

import threading
import time
import logging
from typing import Dict, Optional, List
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class LiveRetrainer:
    """
    Background retraining engine that maxes out GPU utilization.

    Runs in a separate thread, periodically:
    1. Extracts data from tick buffer
    2. Trains XGBoost + LightGBM + CatBoost on GPU
    3. Hot-swaps models if accuracy improves
    """

    def __init__(self, symbols: List[str], retrain_interval: int = 60,
                 min_samples: int = 1000):
        """
        Args:
            symbols: List of symbols to retrain
            retrain_interval: Seconds between retrains
            min_samples: Minimum samples required to retrain
        """
        self.symbols = symbols
        self.retrain_interval = retrain_interval
        self.min_samples = min_samples

        # Import here to avoid circular imports
        from core.data.buffer import get_tick_buffer
        from core.ml.ensemble import get_ensemble
        from core.ml.gpu_config import (get_xgb_gpu_params, get_lgb_gpu_params,
                                        get_catboost_gpu_params, configure_gpu)

        self.get_tick_buffer = get_tick_buffer
        self.get_ensemble = get_ensemble
        self.get_xgb_params = get_xgb_gpu_params
        self.get_lgb_params = get_lgb_gpu_params
        self.get_cb_params = get_catboost_gpu_params

        # Configure GPU once
        configure_gpu()

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Stats
        self.retrain_count = 0
        self.last_retrain_time = None
        self.last_gpu_util = 0

    def start(self):
        """Start background retraining thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._retrain_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info("[RETRAIN] Background retraining started")

    def stop(self):
        """Stop background retraining."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("[RETRAIN] Background retraining stopped")

    def _retrain_loop(self):
        """Main retraining loop."""
        while not self._stop_event.is_set():
            try:
                for symbol in self.symbols:
                    if self._stop_event.is_set():
                        break
                    self._retrain_symbol(symbol)

                self.retrain_count += 1
                self.last_retrain_time = time.time()

            except Exception as e:
                logger.error(f"[RETRAIN] Error: {e}")

            # Wait for next interval
            self._stop_event.wait(timeout=self.retrain_interval)

    def _retrain_symbol(self, symbol: str):
        """Retrain models for a single symbol."""
        tick_buffer = self.get_tick_buffer()
        ensemble = self.get_ensemble(symbol)

        # Get training data
        X, y, feature_names = tick_buffer.get_training_data(
            n_samples=5000, symbol=symbol
        )

        if X is None or len(X) < self.min_samples:
            logger.debug(f"[RETRAIN] {symbol}: Not enough data ({len(X) if X is not None else 0} samples)")
            return

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep time order
        )

        logger.info(f"[RETRAIN] {symbol}: Training on {len(X_train)} samples...")

        # Train models on GPU
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb

        models = {}

        # XGBoost
        try:
            xgb_params = self.get_xgb_params()
            xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc'})

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            models['xgboost'] = xgb.train(
                xgb_params, dtrain, num_boost_round=200,
                evals=[(dval, 'val')], early_stopping_rounds=20,
                verbose_eval=False
            )
        except Exception as e:
            logger.error(f"[RETRAIN] XGBoost error: {e}")

        # LightGBM
        try:
            lgb_params = self.get_lgb_params()
            lgb_params.update({'objective': 'binary', 'metric': 'auc'})

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            models['lightgbm'] = lgb.train(
                lgb_params, train_data, num_boost_round=200,
                valid_sets=[val_data], callbacks=[lgb.early_stopping(20, verbose=False)]
            )
        except Exception as e:
            logger.error(f"[RETRAIN] LightGBM error: {e}")

        # CatBoost
        try:
            cb_params = self.get_cb_params()

            model = cb.CatBoostClassifier(
                **cb_params,
                iterations=200,
                loss_function='Logloss',
                eval_metric='AUC',
                early_stopping_rounds=20,
                verbose=False
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            models['catboost'] = model
        except Exception as e:
            logger.error(f"[RETRAIN] CatBoost error: {e}")

        if not models:
            logger.error(f"[RETRAIN] {symbol}: All models failed")
            return

        # Evaluate ensemble
        predictions = []
        if 'xgboost' in models:
            predictions.append(models['xgboost'].predict(xgb.DMatrix(X_val)))
        if 'lightgbm' in models:
            predictions.append(models['lightgbm'].predict(X_val))
        if 'catboost' in models:
            predictions.append(models['catboost'].predict_proba(X_val)[:, 1])

        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_class = (ensemble_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_val, ensemble_class)
        auc = roc_auc_score(y_val, ensemble_pred)

        logger.info(f"[RETRAIN] {symbol}: acc={accuracy:.4f} auc={auc:.4f}")

        # Hot-swap if improved
        ensemble.hot_swap(
            new_models=models,
            feature_names=feature_names,
            accuracy=accuracy,
            auc=auc,
            n_samples=len(X_train)
        )

    def get_stats(self) -> Dict:
        """Get retrainer statistics."""
        return {
            'running': self._running,
            'retrain_count': self.retrain_count,
            'last_retrain': self.last_retrain_time,
            'interval': self.retrain_interval,
            'symbols': self.symbols
        }


# Singleton instance
_retrainer: Optional[LiveRetrainer] = None
_retrainer_lock = threading.Lock()


def get_retrainer(symbols: List[str] = None) -> LiveRetrainer:
    """Get or create the global retrainer."""
    global _retrainer
    with _retrainer_lock:
        if _retrainer is None:
            if symbols is None:
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            _retrainer = LiveRetrainer(symbols=symbols)
        return _retrainer
