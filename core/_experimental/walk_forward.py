"""
Walk-Forward Cross-Validation for Financial Time Series
========================================================
Prevents overfitting by using proper time-series validation.

Methods:
1. Expanding Window - Train on all historical, test on next period
2. Rolling Window - Fixed training window, slides forward
3. Purged K-Fold - K-fold with gap to prevent leakage
4. Combinatorial Purged CV - Multiple test periods

Source: Lopez de Prado "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from one walk-forward fold."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    train_score: float
    test_score: float
    predictions: np.ndarray
    actuals: np.ndarray


class WalkForwardCV:
    """
    Walk-Forward Cross-Validation for time series.

    Usage:
        wf = WalkForwardCV(n_splits=5, train_ratio=0.8)
        for train_idx, test_idx in wf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(self,
                 n_splits: int = 5,
                 train_ratio: float = 0.8,
                 gap: int = 0,
                 expanding: bool = True):
        """
        Initialize walk-forward CV.

        Args:
            n_splits: Number of folds
            train_ratio: Ratio of data for training (if not expanding)
            gap: Number of periods to skip between train/test (prevents leakage)
            expanding: If True, training window expands; if False, rolls
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap
        self.expanding = expanding

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Features array
            y: Labels array (optional)

        Yields:
            (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window: train on all data up to test start
                train_end = test_size * (i + 1)
                train_start = 0
            else:
                # Rolling window: fixed training size
                train_size = int(test_size * self.train_ratio / (1 - self.train_ratio))
                train_end = test_size * (i + 1)
                train_start = max(0, train_end - train_size)

            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_start >= n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Adds embargo period between train and test to prevent information leakage
    from overlapping labels or features.

    Source: Lopez de Prado "Advances in Financial Machine Learning", Chapter 7
    """

    def __init__(self,
                 n_splits: int = 5,
                 embargo_pct: float = 0.01):
        """
        Initialize purged K-fold.

        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of data to embargo after each test set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              times: Optional[pd.DatetimeIndex] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test splits.

        Args:
            X: Features
            y: Labels (optional)
            times: Timestamps for embargo calculation

        Yields:
            (train_indices, test_indices)
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Embargo: exclude data right after test period
            embargo_end = min(test_end + embargo_size, n_samples)

            # Training indices: everything except test + embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:embargo_end] = False

            train_idx = indices[train_mask]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Creates multiple paths through the data, each treated as a potential backtest.
    More robust than single walk-forward.

    Source: Lopez de Prado "Advances in Financial Machine Learning", Chapter 12
    """

    def __init__(self,
                 n_splits: int = 6,
                 n_test_splits: int = 2,
                 embargo_pct: float = 0.01):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of groups
            n_test_splits: Number of groups to use as test per path
            embargo_pct: Embargo percentage
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate all combinatorial paths."""
        from itertools import combinations

        n_samples = len(X)
        group_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        # All combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = []
            embargo_idx = []

            for g in test_groups:
                start = g * group_size
                end = (g + 1) * group_size if g < self.n_splits - 1 else n_samples
                test_idx.extend(range(start, end))
                embargo_idx.extend(range(end, min(end + embargo_size, n_samples)))

            # Training: everything except test + embargo
            all_excluded = set(test_idx) | set(embargo_idx)
            train_idx = [i for i in range(n_samples) if i not in all_excluded]

            yield np.array(train_idx), np.array(test_idx)


def run_walk_forward_backtest(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: WalkForwardCV,
    timestamps: Optional[pd.DatetimeIndex] = None
) -> List[WalkForwardResult]:
    """
    Run complete walk-forward backtest.

    Args:
        model: Sklearn-compatible model with fit/predict
        X: Features
        y: Labels
        cv: Cross-validation splitter
        timestamps: Optional timestamps for reporting

    Returns:
        List of WalkForwardResult for each fold
    """
    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Fit model
        model.fit(X[train_idx], y[train_idx])

        # Predict
        train_pred = model.predict(X[train_idx])
        test_pred = model.predict(X[test_idx])

        # Score (accuracy for classification)
        train_score = (train_pred == y[train_idx]).mean()
        test_score = (test_pred == y[test_idx]).mean()

        # Get timestamps if available
        if timestamps is not None:
            train_start = timestamps[train_idx[0]]
            train_end = timestamps[train_idx[-1]]
            test_start = timestamps[test_idx[0]]
            test_end = timestamps[test_idx[-1]]
        else:
            train_start = train_end = test_start = test_end = datetime.now()

        result = WalkForwardResult(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_score=train_score,
            test_score=test_score,
            predictions=test_pred,
            actuals=y[test_idx]
        )
        results.append(result)

        logger.info(f"Fold {fold}: Train={train_score:.4f}, Test={test_score:.4f}")

    return results


def walk_forward_analysis(results: List[WalkForwardResult]) -> Dict[str, float]:
    """
    Analyze walk-forward results.

    Returns:
        Dict with summary statistics
    """
    train_scores = [r.train_score for r in results]
    test_scores = [r.test_score for r in results]

    return {
        'mean_train_score': np.mean(train_scores),
        'std_train_score': np.std(train_scores),
        'mean_test_score': np.mean(test_scores),
        'std_test_score': np.std(test_scores),
        'min_test_score': np.min(test_scores),
        'max_test_score': np.max(test_scores),
        'overfit_ratio': np.mean(train_scores) / (np.mean(test_scores) + 1e-8),
        'n_folds': len(results),
        'total_predictions': sum(r.test_size for r in results),
    }


def detect_overfitting(
    train_score: float,
    test_score: float,
    threshold: float = 1.15
) -> bool:
    """
    Detect if model is overfitting.

    Args:
        train_score: Training accuracy
        test_score: Test accuracy
        threshold: Max acceptable ratio (e.g., 1.15 = 15% higher train)

    Returns:
        True if overfitting detected
    """
    if test_score < 0.01:
        return True

    ratio = train_score / test_score
    return ratio > threshold


class WalkForwardOptimizer:
    """
    Optimize model hyperparameters using walk-forward validation.

    Prevents selecting hyperparameters that overfit to a single test set.
    """

    def __init__(self,
                 model_class,
                 param_grid: Dict[str, List],
                 cv: WalkForwardCV,
                 scoring: str = 'accuracy'):
        """
        Initialize optimizer.

        Args:
            model_class: Model class to instantiate
            param_grid: Dict of parameter names to lists of values
            cv: Walk-forward CV splitter
            scoring: Metric to optimize
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WalkForwardOptimizer':
        """
        Find best parameters using walk-forward CV.
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        best_score = -np.inf
        best_params = None

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # Evaluate this parameter set
            scores = []
            for train_idx, test_idx in self.cv.split(X, y):
                model = self.model_class(**params)
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                score = (pred == y[test_idx]).mean()
                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score

        return self


# Convenience functions

def expanding_window_cv(n_splits: int = 5, gap: int = 0) -> WalkForwardCV:
    """Create expanding window CV."""
    return WalkForwardCV(n_splits=n_splits, gap=gap, expanding=True)


def rolling_window_cv(n_splits: int = 5, train_ratio: float = 0.8, gap: int = 0) -> WalkForwardCV:
    """Create rolling window CV."""
    return WalkForwardCV(n_splits=n_splits, train_ratio=train_ratio, gap=gap, expanding=False)


def purged_cv(n_splits: int = 5, embargo_pct: float = 0.01) -> PurgedKFold:
    """Create purged K-fold CV."""
    return PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)


if __name__ == '__main__':
    # Test walk-forward CV
    print("Walk-Forward Cross-Validation Test")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Test expanding window
    print("\nExpanding Window CV:")
    cv = expanding_window_cv(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Fold {fold}: Train {len(train_idx)}, Test {len(test_idx)}")

    # Test rolling window
    print("\nRolling Window CV:")
    cv = rolling_window_cv(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Fold {fold}: Train {len(train_idx)}, Test {len(test_idx)}")

    # Test purged K-fold
    print("\nPurged K-Fold CV:")
    cv = purged_cv(n_splits=5, embargo_pct=0.02)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"  Fold {fold}: Train {len(train_idx)}, Test {len(test_idx)}")

    # Test with actual model
    print("\nWith XGBoost:")
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
        cv = expanding_window_cv(n_splits=5, gap=10)

        results = run_walk_forward_backtest(model, X, y, cv)
        analysis = walk_forward_analysis(results)

        print(f"  Mean test score: {analysis['mean_test_score']:.4f}")
        print(f"  Overfit ratio: {analysis['overfit_ratio']:.2f}")

    except ImportError:
        print("  XGBoost not available")
