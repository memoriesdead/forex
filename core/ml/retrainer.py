"""
Hybrid Retrainer - Historical + Live Data Combined
===================================================
Uses massive historical data as foundation, adds live data for adaptation.
Target: 65-70%+ accuracy (up from 53-62%)
"""

import numpy as np
import pandas as pd
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class HybridRetrainer:
    """
    Combines historical data (200k+ samples) with live data for maximum accuracy.

    Strategy:
    1. Load historical data as foundation (high volume, diverse conditions)
    2. Append live data with higher weight (recent market behavior)
    3. Train on GPU with full RTX 5080 power
    4. Hot-swap if accuracy improves
    """

    def __init__(self, symbols: List[str] = None,
                 tier: str = None,
                 historical_dir: Path = None,
                 retrain_interval: int = 120,  # 2 min (more data per cycle)
                 live_weight: float = 3.0):    # Live data weighted 3x
        """
        Args:
            symbols: Currency pairs to train (optional if tier is provided)
            tier: Symbol tier from registry (majors, crosses, exotics, all)
            historical_dir: Path to historical parquet files
            retrain_interval: Seconds between retrains
            live_weight: Weight multiplier for live samples
        """
        # Get symbols from registry if not provided
        if symbols is None:
            from core.symbol.registry import SymbolRegistry
            registry = SymbolRegistry.get()
            if tier:
                pairs = registry.get_enabled(tier=tier if tier != 'all' else None)
            else:
                pairs = registry.get_enabled(tier='majors')  # Default to majors
            symbols = [p.symbol for p in pairs]

        self.symbols = symbols
        self.historical_dir = historical_dir or Path("training_package")
        self.retrain_interval = retrain_interval
        self.live_weight = live_weight

        # Load historical data once
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self._load_historical_data()

        # Import components
        from core.data.buffer import get_tick_buffer
        from core.ml.ensemble import get_ensemble
        from core.ml.gpu_config import (get_xgb_gpu_params, get_lgb_gpu_params,
                                        get_catboost_gpu_params, configure_gpu)

        self.get_tick_buffer = get_tick_buffer
        self.get_ensemble = get_ensemble
        self.get_xgb_params = get_xgb_gpu_params
        self.get_lgb_params = get_lgb_gpu_params
        self.get_cb_params = get_catboost_gpu_params

        # Configure GPU
        configure_gpu()

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Stats
        self.retrain_count = 0
        self.best_accuracy: Dict[str, float] = {}

    def _load_historical_data(self):
        """Load historical training data for each symbol."""
        for symbol in self.symbols:
            symbol_dir = self.historical_dir / symbol
            if not symbol_dir.exists():
                logger.warning(f"No historical data for {symbol}")
                continue

            try:
                train = pd.read_parquet(symbol_dir / "train.parquet")
                val = pd.read_parquet(symbol_dir / "val.parquet")

                # Combine train + val for more data
                combined = pd.concat([train, val], ignore_index=True)
                self.historical_data[symbol] = combined

                logger.info(f"[HYBRID] Loaded {len(combined):,} historical samples for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load historical data for {symbol}: {e}")

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (non-target, non-timestamp columns)."""
        exclude_patterns = ['target', 'timestamp', 'time', 'date', 'datetime']
        return [c for c in df.columns
                if not any(p in c.lower() for p in exclude_patterns)
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'bool']]

    def _get_combined_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Combine historical + live data with weighting.

        Returns:
            X: Feature matrix
            y: Labels
            feature_names: Column names
        """
        # Get historical data
        if symbol not in self.historical_data:
            return None, None, None

        hist_df = self.historical_data[symbol]
        feature_cols = self._get_feature_columns(hist_df)

        # Find target column (prefer direction_10 or direction_1)
        target_col = None
        for t in ['target_direction_10', 'target_direction_5', 'target_direction_1']:
            if t in hist_df.columns:
                target_col = t
                break

        if target_col is None:
            logger.error(f"No target column found for {symbol}")
            return None, None, None

        # Get live data from buffer
        tick_buffer = self.get_tick_buffer()
        X_live, y_live, live_features = tick_buffer.get_training_data(
            n_samples=10000, symbol=symbol
        )

        # Start with historical
        X_hist = hist_df[feature_cols].values
        y_hist = hist_df[target_col].values

        # Sample historical to keep training fast (use 50k random samples)
        n_hist_samples = min(50000, len(X_hist))
        hist_idx = np.random.choice(len(X_hist), n_hist_samples, replace=False)
        X_hist = X_hist[hist_idx]
        y_hist = y_hist[hist_idx]

        # If we have live data, combine with weighting
        if X_live is not None and len(X_live) > 100:
            # Match features between historical and live
            common_features = [f for f in feature_cols if f in live_features]

            if len(common_features) > 50:  # Need enough common features
                # Re-extract with common features only
                hist_feat_idx = [feature_cols.index(f) for f in common_features]
                live_feat_idx = [live_features.index(f) for f in common_features]

                X_hist_common = X_hist[:, hist_feat_idx]
                X_live_common = X_live[:, live_feat_idx]

                # Duplicate live data to increase its weight
                n_live_copies = int(self.live_weight)
                X_live_weighted = np.tile(X_live_common, (n_live_copies, 1))
                y_live_weighted = np.tile(y_live, n_live_copies)

                # Combine
                X_combined = np.vstack([X_hist_common, X_live_weighted])
                y_combined = np.hstack([y_hist, y_live_weighted])

                logger.info(f"[HYBRID] {symbol}: {n_hist_samples:,} hist + {len(X_live)*n_live_copies:,} live (weighted) = {len(X_combined):,} total")

                return X_combined, y_combined, common_features

        # Fallback to historical only
        logger.info(f"[HYBRID] {symbol}: Using {n_hist_samples:,} historical samples only")
        return X_hist, y_hist, feature_cols

    def start(self):
        """Start background hybrid retraining."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._retrain_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info("[HYBRID] Background hybrid retraining started (historical + live)")

    def stop(self):
        """Stop hybrid retraining."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("[HYBRID] Stopped")

    def _retrain_loop(self):
        """Main retraining loop."""
        # Initial delay to let live data accumulate
        self._stop_event.wait(timeout=30)

        while not self._stop_event.is_set():
            try:
                for symbol in self.symbols:
                    if self._stop_event.is_set():
                        break
                    self._retrain_symbol(symbol)

                self.retrain_count += 1

            except Exception as e:
                logger.error(f"[HYBRID] Error: {e}")

            self._stop_event.wait(timeout=self.retrain_interval)

    def _retrain_symbol(self, symbol: str):
        """Retrain models for a symbol using hybrid data."""
        ensemble = self.get_ensemble(symbol)

        # Get combined historical + live data
        X, y, feature_names = self._get_combined_data(symbol)

        if X is None or len(X) < 1000:
            return

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Split data (keep time order for recent data)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True  # Shuffle since we mixed data
        )

        logger.info(f"[HYBRID] {symbol}: Training on {len(X_train):,} samples...")

        # Train models on GPU
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb

        models = {}

        # XGBoost with more trees for larger dataset
        try:
            xgb_params = self.get_xgb_params()
            xgb_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 14,  # Deeper for more data
            })

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            models['xgboost'] = xgb.train(
                xgb_params, dtrain,
                num_boost_round=500,  # More trees
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
        except Exception as e:
            logger.error(f"[HYBRID] XGBoost error: {e}")

        # LightGBM
        try:
            lgb_params = self.get_lgb_params()
            lgb_params.update({
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 14,
                'num_leaves': 1023,  # More leaves for more data
            })

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            models['lightgbm'] = lgb.train(
                lgb_params, train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        except Exception as e:
            logger.error(f"[HYBRID] LightGBM error: {e}")

        # CatBoost
        try:
            cb_params = self.get_cb_params()
            # Override for deeper training
            cb_params['depth'] = 12
            cb_params['iterations'] = 500
            # Remove params that conflict with explicit kwargs
            cb_params.pop('early_stopping_rounds', None)
            cb_params.pop('verbose', None)

            model = cb.CatBoostClassifier(
                **cb_params,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                     early_stopping_rounds=30, verbose=False)
            models['catboost'] = model
        except Exception as e:
            logger.error(f"[HYBRID] CatBoost error: {e}")

        if not models:
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

        # Track best accuracy
        prev_best = self.best_accuracy.get(symbol, 0.5)

        logger.info(f"[HYBRID] {symbol}: acc={accuracy:.4f} auc={auc:.4f} (prev best: {prev_best:.4f})")

        # Hot-swap if improved
        if accuracy > prev_best:
            self.best_accuracy[symbol] = accuracy
            ensemble.hot_swap(
                new_models=models,
                feature_names=feature_names,
                accuracy=accuracy,
                auc=auc,
                n_samples=len(X_train)
            )
            logger.info(f"[HYBRID] {symbol}: NEW BEST! {prev_best:.4f} -> {accuracy:.4f}")


# Singleton
_hybrid_retrainer: Optional[HybridRetrainer] = None
_lock = threading.Lock()


def get_hybrid_retrainer(symbols: List[str] = None, tier: str = None) -> HybridRetrainer:
    """
    Get or create hybrid retrainer.

    Args:
        symbols: Specific symbols to train (optional)
        tier: Symbol tier from registry (majors, crosses, exotics, all)

    Returns:
        HybridRetrainer instance
    """
    global _hybrid_retrainer
    with _lock:
        if _hybrid_retrainer is None:
            _hybrid_retrainer = HybridRetrainer(symbols=symbols, tier=tier)
        return _hybrid_retrainer
