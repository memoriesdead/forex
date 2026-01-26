"""
Stacking Ensemble - 6-Model Stacking with Meta-Learner
=======================================================
World-class ensemble using:
- Level 1 Base Learners:
  - XGBoost (GPU)
  - LightGBM (GPU)
  - CatBoost (GPU)
  - TabNet (Attention-based)
  - 1D-CNN (Convolutional)
  - MLP (Dense)

- Level 2 Meta-Learner:
  - XGBoost on stacked predictions + top features

Target: 72%+ accuracy, 0.80+ AUC
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import pickle
import json
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble."""

    # XGBoost parameters (GPU-accelerated)
    xgb_params: Dict = field(default_factory=lambda: {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 12,
        'max_bin': 256,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 50,
    })

    # LightGBM parameters (GPU-accelerated)
    lgb_params: Dict = field(default_factory=lambda: {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_depth': 12,
        'num_leaves': 2047,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
    })

    # CatBoost parameters (GPU-accelerated)
    cb_params: Dict = field(default_factory=lambda: {
        'task_type': 'GPU',
        'devices': '0',
        'depth': 10,
        'border_count': 32,
        'learning_rate': 0.03,
        'iterations': 1500,
        'boosting_type': 'Plain',
        'l2_leaf_reg': 3.0,
        'early_stopping_rounds': 50,
        'verbose': 100,
    })

    # TabNet parameters
    tabnet_params: Dict = field(default_factory=lambda: {
        'n_d': 64,
        'n_a': 64,
        'n_steps': 5,
        'gamma': 1.5,
        'n_independent': 2,
        'n_shared': 2,
        'lambda_sparse': 1e-4,
        'max_epochs': 100,
        'patience': 10,
        'batch_size': 1024,
        'virtual_batch_size': 128,
    })

    # CNN parameters
    cnn_params: Dict = field(default_factory=lambda: {
        'conv_layers': [64, 128, 256],
        'kernel_size': 3,
        'dropout': 0.3,
        'fc_layers': [256, 128],
        'max_epochs': 50,
        'batch_size': 1024,
        'learning_rate': 0.001,
    })

    # MLP parameters
    mlp_params: Dict = field(default_factory=lambda: {
        'hidden_layers': [512, 256, 128, 64],
        'dropout': 0.3,
        'batch_norm': True,
        'max_epochs': 50,
        'batch_size': 1024,
        'learning_rate': 0.001,
    })

    # Meta-learner parameters
    meta_learner_params: Dict = field(default_factory=lambda: {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    })

    # Stacking settings
    n_folds: int = 5  # For out-of-fold predictions
    top_features_for_meta: int = 50  # Number of original features to add to meta
    enable_tabnet: bool = True  # TabNet can be slow, allow disabling
    enable_cnn: bool = True  # CNN can be slow, allow disabling
    enable_mlp: bool = True  # MLP can be slow, allow disabling
    parallel_base_learners: bool = False  # Train base learners in parallel


class StackingEnsemble:
    """
    6-Model Stacking Ensemble with Meta-Learner.

    Architecture:
        Level 1 - Base Learners (trained on original features):
            1. XGBoost (GPU) - Gradient boosting
            2. LightGBM (GPU) - Gradient boosting (faster)
            3. CatBoost (GPU) - Gradient boosting (categorical)
            4. TabNet - Attention-based neural network
            5. 1D-CNN - Temporal patterns
            6. MLP - Dense neural network

        Level 2 - Meta-Learner:
            XGBoost trained on:
            - 6 stacked base learner predictions
            - Top 50 original features

    Usage:
        ensemble = StackingEnsemble()
        ensemble.fit(X_train, y_train, X_val, y_val, feature_names)
        predictions = ensemble.predict(X_test)
        proba = ensemble.predict_proba(X_test)
    """

    def __init__(self, config: Optional[StackingConfig] = None):
        """Initialize stacking ensemble."""
        self.config = config or StackingConfig()

        # Base learners
        self.xgb_model = None
        self.lgb_model = None
        self.cb_model = None
        self.tabnet_model = None
        self.cnn_model = None
        self.mlp_model = None

        # Meta-learner
        self.meta_learner = None

        # Feature selection for meta-learner
        self.top_feature_indices = None
        self.feature_names = None

        # Training state
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Optional feature names

        Returns:
            self
        """
        logger.info("Fitting Stacking Ensemble...")
        logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

        self.feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # ===== Level 1: Base Learners =====
        logger.info("Level 1: Training Base Learners...")

        if self.config.parallel_base_learners:
            self._fit_base_learners_parallel(X_train, y_train, X_val, y_val)
        else:
            self._fit_base_learners_sequential(X_train, y_train, X_val, y_val)

        # Generate out-of-fold predictions for meta-learner training
        logger.info("Generating OOF predictions for meta-learner...")
        oof_predictions = self._generate_oof_predictions(X_train, y_train)
        val_predictions = self._generate_val_predictions(X_val)

        # ===== Level 2: Meta-Learner =====
        logger.info("Level 2: Training Meta-Learner...")

        # Select top features based on XGBoost importance
        self._select_top_features(X_train, y_train)

        # Build meta-features
        X_meta_train = self._build_meta_features(X_train, oof_predictions)
        X_meta_val = self._build_meta_features(X_val, val_predictions)

        # Train meta-learner
        self._fit_meta_learner(X_meta_train, y_train, X_meta_val, y_val)

        self.is_fitted = True
        logger.info("Stacking Ensemble training complete!")

        return self

    def _fit_base_learners_sequential(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train base learners sequentially."""
        # 1. XGBoost
        logger.info("  [1/6] Training XGBoost...")
        self._fit_xgboost(X_train, y_train, X_val, y_val)
        gc.collect()

        # 2. LightGBM
        logger.info("  [2/6] Training LightGBM...")
        self._fit_lightgbm(X_train, y_train, X_val, y_val)
        gc.collect()

        # 3. CatBoost
        logger.info("  [3/6] Training CatBoost...")
        self._fit_catboost(X_train, y_train, X_val, y_val)
        gc.collect()

        # 4. TabNet (optional)
        if self.config.enable_tabnet:
            logger.info("  [4/6] Training TabNet...")
            self._fit_tabnet(X_train, y_train, X_val, y_val)
            gc.collect()
        else:
            logger.info("  [4/6] TabNet disabled")

        # 5. CNN (optional)
        if self.config.enable_cnn:
            logger.info("  [5/6] Training 1D-CNN...")
            self._fit_cnn(X_train, y_train, X_val, y_val)
            gc.collect()
        else:
            logger.info("  [5/6] CNN disabled")

        # 6. MLP (optional)
        if self.config.enable_mlp:
            logger.info("  [6/6] Training MLP...")
            self._fit_mlp(X_train, y_train, X_val, y_val)
            gc.collect()
        else:
            logger.info("  [6/6] MLP disabled")

    def _fit_base_learners_parallel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train base learners in parallel (experimental)."""
        # Note: GPU models can't truly parallelize on single GPU
        # This is more useful for CPU models or multi-GPU setups
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self._fit_xgboost, X_train, y_train, X_val, y_val),
                executor.submit(self._fit_lightgbm, X_train, y_train, X_val, y_val),
                executor.submit(self._fit_catboost, X_train, y_train, X_val, y_val),
            ]
            for f in futures:
                f.result()

        # Sequential for neural models (GPU memory)
        if self.config.enable_tabnet:
            self._fit_tabnet(X_train, y_train, X_val, y_val)
        if self.config.enable_cnn:
            self._fit_cnn(X_train, y_train, X_val, y_val)
        if self.config.enable_mlp:
            self._fit_mlp(X_train, y_train, X_val, y_val)

    def _fit_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit XGBoost model."""
        import xgboost as xgb

        params = self.config.xgb_params.copy()
        n_estimators = params.pop('n_estimators', 1500)
        early_stopping = params.pop('early_stopping_rounds', 50)

        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )

        # Log validation score
        val_pred = self.xgb_model.predict(dval)
        auc = self._compute_auc(y_val, val_pred)
        logger.info(f"    XGBoost validation AUC: {auc:.4f}")

    def _fit_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit LightGBM model."""
        import lightgbm as lgb

        params = self.config.lgb_params.copy()
        n_estimators = params.pop('n_estimators', 1500)

        params['objective'] = 'binary'
        params['metric'] = 'auc'

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Log validation score
        val_pred = self.lgb_model.predict(X_val)
        auc = self._compute_auc(y_val, val_pred)
        logger.info(f"    LightGBM validation AUC: {auc:.4f}")

    def _fit_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit CatBoost model."""
        import catboost as cb

        params = self.config.cb_params.copy()

        self.cb_model = cb.CatBoostClassifier(**params)
        self.cb_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        # Log validation score
        val_pred = self.cb_model.predict_proba(X_val)[:, 1]
        auc = self._compute_auc(y_val, val_pred)
        logger.info(f"    CatBoost validation AUC: {auc:.4f}")

    def _fit_tabnet(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit TabNet model."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier

            params = self.config.tabnet_params.copy()
            max_epochs = params.pop('max_epochs', 100)
            patience = params.pop('patience', 10)
            batch_size = params.pop('batch_size', 1024)
            virtual_batch_size = params.pop('virtual_batch_size', 128)

            self.tabnet_model = TabNetClassifier(**params, verbose=0)

            self.tabnet_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
            )

            # Log validation score
            val_pred = self.tabnet_model.predict_proba(X_val)[:, 1]
            auc = self._compute_auc(y_val, val_pred)
            logger.info(f"    TabNet validation AUC: {auc:.4f}")

        except ImportError:
            logger.warning("TabNet not installed. Skipping. Install with: pip install pytorch-tabnet")
            self.tabnet_model = None

    def _fit_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit 1D-CNN model."""
        try:
            from .neural_models import CNN1DClassifier

            params = self.config.cnn_params.copy()

            self.cnn_model = CNN1DClassifier(
                input_dim=X_train.shape[1],
                **params
            )

            self.cnn_model.fit(X_train, y_train, X_val, y_val)

            # Log validation score
            val_pred = self.cnn_model.predict_proba(X_val)
            auc = self._compute_auc(y_val, val_pred)
            logger.info(f"    CNN validation AUC: {auc:.4f}")

        except ImportError as e:
            logger.warning(f"CNN model failed to import: {e}")
            self.cnn_model = None

    def _fit_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit MLP model."""
        try:
            from .neural_models import MLPClassifier

            params = self.config.mlp_params.copy()

            self.mlp_model = MLPClassifier(
                input_dim=X_train.shape[1],
                **params
            )

            self.mlp_model.fit(X_train, y_train, X_val, y_val)

            # Log validation score
            val_pred = self.mlp_model.predict_proba(X_val)
            auc = self._compute_auc(y_val, val_pred)
            logger.info(f"    MLP validation AUC: {auc:.4f}")

        except ImportError as e:
            logger.warning(f"MLP model failed to import: {e}")
            self.mlp_model = None

    def _select_top_features(self, X: np.ndarray, y: np.ndarray):
        """Select top features for meta-learner."""
        # Use XGBoost feature importance
        if self.xgb_model is not None:
            import xgboost as xgb

            importance = self.xgb_model.get_score(importance_type='gain')

            # Map feature names to indices
            feature_importance = []
            for i in range(X.shape[1]):
                feat_name = f'f{i}'
                imp = importance.get(feat_name, 0.0)
                feature_importance.append((i, imp))

            # Sort by importance and take top K
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            self.top_feature_indices = [
                idx for idx, _ in feature_importance[:self.config.top_features_for_meta]
            ]
        else:
            # Fallback: use first K features
            self.top_feature_indices = list(range(min(self.config.top_features_for_meta, X.shape[1])))

    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions for meta-learner training."""
        from sklearn.model_selection import KFold
        import xgboost as xgb

        n_samples = len(X)
        n_models = self._count_active_models()
        oof_preds = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=self.config.n_folds, shuffle=False)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]

            col_idx = 0

            # XGBoost
            if self.xgb_model is not None:
                dval = xgb.DMatrix(X_va)
                oof_preds[val_idx, col_idx] = self.xgb_model.predict(dval)
                col_idx += 1

            # LightGBM
            if self.lgb_model is not None:
                oof_preds[val_idx, col_idx] = self.lgb_model.predict(X_va)
                col_idx += 1

            # CatBoost
            if self.cb_model is not None:
                oof_preds[val_idx, col_idx] = self.cb_model.predict_proba(X_va)[:, 1]
                col_idx += 1

            # TabNet
            if self.tabnet_model is not None:
                oof_preds[val_idx, col_idx] = self.tabnet_model.predict_proba(X_va)[:, 1]
                col_idx += 1

            # CNN
            if self.cnn_model is not None:
                oof_preds[val_idx, col_idx] = self.cnn_model.predict_proba(X_va)
                col_idx += 1

            # MLP
            if self.mlp_model is not None:
                oof_preds[val_idx, col_idx] = self.mlp_model.predict_proba(X_va)
                col_idx += 1

        return oof_preds

    def _generate_val_predictions(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from all base learners for validation set."""
        import xgboost as xgb

        predictions = []

        # XGBoost
        if self.xgb_model is not None:
            dval = xgb.DMatrix(X)
            predictions.append(self.xgb_model.predict(dval))

        # LightGBM
        if self.lgb_model is not None:
            predictions.append(self.lgb_model.predict(X))

        # CatBoost
        if self.cb_model is not None:
            predictions.append(self.cb_model.predict_proba(X)[:, 1])

        # TabNet
        if self.tabnet_model is not None:
            predictions.append(self.tabnet_model.predict_proba(X)[:, 1])

        # CNN
        if self.cnn_model is not None:
            predictions.append(self.cnn_model.predict_proba(X))

        # MLP
        if self.mlp_model is not None:
            predictions.append(self.mlp_model.predict_proba(X))

        return np.column_stack(predictions)

    def _build_meta_features(self, X: np.ndarray, stacked_preds: np.ndarray) -> np.ndarray:
        """Build meta-features for level 2."""
        # Combine stacked predictions with top original features
        top_features = X[:, self.top_feature_indices]
        return np.hstack([stacked_preds, top_features])

    def _fit_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Fit the meta-learner."""
        import xgboost as xgb

        params = self.config.meta_learner_params.copy()
        n_estimators = params.pop('n_estimators', 500)

        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.meta_learner = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        # Log final score
        val_pred = self.meta_learner.predict(dval)
        auc = self._compute_auc(y_val, val_pred)
        acc = ((val_pred > 0.5) == y_val).mean()
        logger.info(f"    Meta-Learner validation AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    def _count_active_models(self) -> int:
        """Count number of active base learners."""
        count = 0
        if self.xgb_model is not None:
            count += 1
        if self.lgb_model is not None:
            count += 1
        if self.cb_model is not None:
            count += 1
        if self.tabnet_model is not None:
            count += 1
        if self.cnn_model is not None:
            count += 1
        if self.mlp_model is not None:
            count += 1
        return count

    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC score."""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Probability of positive class
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        import xgboost as xgb

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get base learner predictions
        stacked_preds = self._generate_val_predictions(X)

        # Build meta-features
        meta_features = self._build_meta_features(X, stacked_preds)

        # Predict with meta-learner
        dmeta = xgb.DMatrix(meta_features)
        return self.meta_learner.predict(dmeta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features
            threshold: Classification threshold

        Returns:
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_base_learner_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each base learner."""
        import xgboost as xgb

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = {}

        if self.xgb_model is not None:
            dval = xgb.DMatrix(X)
            predictions['xgboost'] = self.xgb_model.predict(dval)

        if self.lgb_model is not None:
            predictions['lightgbm'] = self.lgb_model.predict(X)

        if self.cb_model is not None:
            predictions['catboost'] = self.cb_model.predict_proba(X)[:, 1]

        if self.tabnet_model is not None:
            predictions['tabnet'] = self.tabnet_model.predict_proba(X)[:, 1]

        if self.cnn_model is not None:
            predictions['cnn'] = self.cnn_model.predict_proba(X)

        if self.mlp_model is not None:
            predictions['mlp'] = self.mlp_model.predict_proba(X)

        return predictions

    def save(self, path: Path):
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save gradient boosting models
        if self.xgb_model is not None:
            self.xgb_model.save_model(str(path / 'xgb_model.json'))

        if self.lgb_model is not None:
            self.lgb_model.save_model(str(path / 'lgb_model.txt'))

        if self.cb_model is not None:
            self.cb_model.save_model(str(path / 'cb_model.cbm'))

        if self.meta_learner is not None:
            self.meta_learner.save_model(str(path / 'meta_learner.json'))

        # Save neural models
        if self.tabnet_model is not None:
            self.tabnet_model.save_model(str(path / 'tabnet_model'))

        if self.cnn_model is not None:
            self.cnn_model.save(path / 'cnn_model.pt')

        if self.mlp_model is not None:
            self.mlp_model.save(path / 'mlp_model.pt')

        # Save state
        state = {
            'top_feature_indices': self.top_feature_indices,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted,
        }
        with open(path / 'state.pkl', 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved ensemble to {path}")

    @classmethod
    def load(cls, path: Path) -> 'StackingEnsemble':
        """Load ensemble from disk."""
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb

        path = Path(path)

        # Load state
        with open(path / 'state.pkl', 'rb') as f:
            state = pickle.load(f)

        ensemble = cls(config=state['config'])
        ensemble.top_feature_indices = state['top_feature_indices']
        ensemble.feature_names = state['feature_names']
        ensemble.is_fitted = state['is_fitted']

        # Load gradient boosting models
        if (path / 'xgb_model.json').exists():
            ensemble.xgb_model = xgb.Booster()
            ensemble.xgb_model.load_model(str(path / 'xgb_model.json'))

        if (path / 'lgb_model.txt').exists():
            ensemble.lgb_model = lgb.Booster(model_file=str(path / 'lgb_model.txt'))

        if (path / 'cb_model.cbm').exists():
            ensemble.cb_model = cb.CatBoostClassifier()
            ensemble.cb_model.load_model(str(path / 'cb_model.cbm'))

        if (path / 'meta_learner.json').exists():
            ensemble.meta_learner = xgb.Booster()
            ensemble.meta_learner.load_model(str(path / 'meta_learner.json'))

        # Load neural models
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            if (path / 'tabnet_model.zip').exists():
                ensemble.tabnet_model = TabNetClassifier()
                ensemble.tabnet_model.load_model(str(path / 'tabnet_model.zip'))
        except:
            pass

        try:
            from .neural_models import CNN1DClassifier, MLPClassifier
            if (path / 'cnn_model.pt').exists():
                ensemble.cnn_model = CNN1DClassifier.load(path / 'cnn_model.pt')
            if (path / 'mlp_model.pt').exists():
                ensemble.mlp_model = MLPClassifier.load(path / 'mlp_model.pt')
        except:
            pass

        return ensemble


def create_stacking_ensemble(
    config: Optional[StackingConfig] = None,
    use_gpu: bool = True,
    enable_tabnet: bool = True,
    enable_neural: bool = True
) -> StackingEnsemble:
    """
    Factory function to create stacking ensemble.

    Args:
        config: Stacking config (uses defaults if None)
        use_gpu: Whether to use GPU acceleration
        enable_tabnet: Whether to include TabNet
        enable_neural: Whether to include CNN/MLP

    Returns:
        StackingEnsemble instance
    """
    if config is None:
        config = StackingConfig()

    # Disable GPU if requested
    if not use_gpu:
        config.xgb_params['device'] = 'cpu'
        config.xgb_params['tree_method'] = 'hist'
        config.lgb_params['device'] = 'cpu'
        config.cb_params['task_type'] = 'CPU'

    # Disable optional models
    if not enable_tabnet:
        config.tabnet_params = None
    if not enable_neural:
        config.cnn_params = None
        config.mlp_params = None

    return StackingEnsemble(config)


# Export
__all__ = ['StackingEnsemble', 'StackingConfig', 'create_stacking_ensemble']
