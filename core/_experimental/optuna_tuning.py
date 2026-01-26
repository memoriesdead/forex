"""
Optuna Hyperparameter Optimization for HFT Models
==================================================
Source: Optuna (13k+ stars) - https://github.com/optuna/optuna

Tree-Parzen Estimator (TPE) for efficient hyperparameter search.
Finds good parameters in 50-100 trials vs 1000s for grid search.

Use cases:
1. XGBoost/LightGBM/CatBoost hyperparameters
2. Signal weight optimization
3. Risk parameter tuning (Kelly fraction, stop loss)
4. Feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not available")


@dataclass
class OptimizationResult:
    """Optimization result."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str


class XGBoostOptimizer:
    """
    Optimize XGBoost hyperparameters using Optuna.
    """

    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                objective: str = 'binary:logistic',
                metric: str = 'auc') -> OptimizationResult:
        """
        Optimize XGBoost hyperparameters.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            objective: XGBoost objective
            metric: Optimization metric

        Returns:
            OptimizationResult with best parameters
        """
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, returning defaults")
            return self._default_xgb_params()

        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not available")
            return self._default_xgb_params()

        def objective_fn(trial):
            params = {
                'objective': objective,
                'eval_metric': metric,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            }

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.pop('n_estimators'),
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            preds = model.predict(dval)
            if metric == 'auc':
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val, preds)
            else:
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, (preds > 0.5).astype(int))

            return score

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='xgboost_optimization'
        )

        self.study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"Best XGBoost params: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value:.4f}")

        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study_name='xgboost_optimization'
        )

    def _default_xgb_params(self) -> OptimizationResult:
        """Return default XGBoost parameters."""
        return OptimizationResult(
            best_params={
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 1.0,
                'gamma': 0.01
            },
            best_value=0.0,
            n_trials=0,
            study_name='default'
        )


class LightGBMOptimizer:
    """Optimize LightGBM hyperparameters."""

    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                objective: str = 'binary',
                metric: str = 'auc') -> OptimizationResult:
        """Optimize LightGBM hyperparameters."""
        if not HAS_OPTUNA:
            return self._default_lgb_params()

        try:
            import lightgbm as lgb
        except ImportError:
            return self._default_lgb_params()

        def objective_fn(trial):
            params = {
                'objective': objective,
                'metric': metric,
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=params.pop('n_estimators'),
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            preds = model.predict(X_val)
            if metric == 'auc':
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val, preds)
            else:
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, (preds > 0.5).astype(int))

            return score

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='lightgbm_optimization'
        )

        self.study.optimize(objective_fn, n_trials=self.n_trials, timeout=self.timeout)

        return OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study_name='lightgbm_optimization'
        )

    def _default_lgb_params(self) -> OptimizationResult:
        return OptimizationResult(
            best_params={
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            best_value=0.0,
            n_trials=0,
            study_name='default'
        )


class SignalWeightOptimizer:
    """
    Optimize signal weights in ensemble.

    Finds optimal combination of:
    - Alpha101 weight
    - Alpha191 weight
    - Renaissance weight
    - Technical weight
    - Order flow weight
    """

    def __init__(self, n_trials: int = 200):
        self.n_trials = n_trials
        self.study = None

    def optimize(self, signals: Dict[str, np.ndarray],
                returns: np.ndarray) -> OptimizationResult:
        """
        Optimize signal weights to maximize Sharpe ratio.

        Args:
            signals: Dict of signal name -> signal array
            returns: Forward returns to predict

        Returns:
            OptimizationResult with optimal weights
        """
        if not HAS_OPTUNA:
            return self._equal_weights(list(signals.keys()))

        signal_names = list(signals.keys())
        signal_matrix = np.column_stack([signals[name] for name in signal_names])

        def objective_fn(trial):
            # Suggest weights
            weights = []
            for name in signal_names:
                w = trial.suggest_float(f'w_{name}', 0.0, 1.0)
                weights.append(w)

            weights = np.array(weights)
            weights = weights / (weights.sum() + 1e-10)  # Normalize

            # Combined signal
            combined = signal_matrix @ weights

            # Calculate Sharpe (assuming signals predict direction)
            position = np.sign(combined)
            pnl = position * returns

            # Sharpe ratio
            sharpe = np.mean(pnl) / (np.std(pnl) + 1e-10) * np.sqrt(252 * 24 * 12)  # Annualized for 5-min bars

            return sharpe

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='signal_weight_optimization'
        )

        self.study.optimize(objective_fn, n_trials=self.n_trials)

        # Extract normalized weights
        best_weights = {}
        total = 0
        for name in signal_names:
            w = self.study.best_params[f'w_{name}']
            best_weights[name] = w
            total += w

        for name in signal_names:
            best_weights[name] /= (total + 1e-10)

        logger.info(f"Optimal signal weights: {best_weights}")
        logger.info(f"Best Sharpe: {self.study.best_value:.2f}")

        return OptimizationResult(
            best_params=best_weights,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study_name='signal_weight_optimization'
        )

    def _equal_weights(self, signal_names: List[str]) -> OptimizationResult:
        n = len(signal_names)
        return OptimizationResult(
            best_params={name: 1.0/n for name in signal_names},
            best_value=0.0,
            n_trials=0,
            study_name='default'
        )


class RiskParameterOptimizer:
    """
    Optimize risk parameters using historical performance.

    Parameters to optimize:
    - Kelly fraction (0.1 to 0.5)
    - Stop loss (in ATR multiples)
    - Take profit (in ATR multiples)
    - Max position size
    """

    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials

    def optimize(self, signals: np.ndarray, returns: np.ndarray,
                volatility: np.ndarray) -> OptimizationResult:
        """
        Optimize risk parameters.

        Args:
            signals: Trading signals (-1, 0, 1)
            returns: Realized returns
            volatility: Volatility estimates (for position sizing)

        Returns:
            OptimizationResult with optimal risk parameters
        """
        if not HAS_OPTUNA:
            return self._default_risk_params()

        def objective_fn(trial):
            kelly_frac = trial.suggest_float('kelly_fraction', 0.1, 0.5)
            stop_atr = trial.suggest_float('stop_loss_atr', 1.0, 5.0)
            tp_atr = trial.suggest_float('take_profit_atr', 1.0, 5.0)

            # Simulate trading with these parameters
            pnl = []
            position = 0
            entry_price = 0
            cumulative_return = 1.0

            for t in range(len(signals)):
                if signals[t] != 0 and position == 0:
                    # Enter position
                    position = signals[t]
                    entry_price = 1.0  # Normalized
                    size = kelly_frac / (volatility[t] + 1e-10)
                    size = min(size, 1.0)  # Cap at 100%

                elif position != 0:
                    # Check exit conditions
                    move = returns[t] * position * size
                    stop = -stop_atr * volatility[t]
                    tp = tp_atr * volatility[t]

                    if move <= stop or move >= tp or signals[t] == -position:
                        pnl.append(move)
                        cumulative_return *= (1 + move)
                        position = 0

            if len(pnl) < 10:
                return -10.0  # Not enough trades

            # Objective: Maximize risk-adjusted return
            total_return = cumulative_return - 1
            vol = np.std(pnl)
            sharpe = (np.mean(pnl) / (vol + 1e-10)) * np.sqrt(252)
            max_dd = self._max_drawdown(np.cumprod(1 + np.array(pnl)))

            # Penalize high drawdown
            score = sharpe - max_dd * 2

            return score

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        self.study.optimize(objective_fn, n_trials=self.n_trials)

        return OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            n_trials=len(self.study.trials),
            study_name='risk_optimization'
        )

    def _max_drawdown(self, cumulative: np.ndarray) -> float:
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)

    def _default_risk_params(self) -> OptimizationResult:
        return OptimizationResult(
            best_params={
                'kelly_fraction': 0.25,
                'stop_loss_atr': 2.0,
                'take_profit_atr': 3.0
            },
            best_value=0.0,
            n_trials=0,
            study_name='default'
        )


def save_optimization_results(results: Dict[str, OptimizationResult],
                             filepath: Path):
    """Save optimization results to JSON."""
    data = {
        name: {
            'best_params': result.best_params,
            'best_value': result.best_value,
            'n_trials': result.n_trials
        }
        for name, result in results.items()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved optimization results to {filepath}")


def load_optimization_results(filepath: Path) -> Dict[str, Dict]:
    """Load optimization results from JSON."""
    with open(filepath) as f:
        return json.load(f)
