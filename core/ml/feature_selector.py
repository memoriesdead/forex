"""
Feature Selector - Intelligent Feature Selection for World-Class ML
====================================================================
Implements multi-stage feature selection:
1. Variance Filter - Remove low-variance features
2. Correlation Filter - Remove highly correlated features
3. Mutual Information - Rank by information gain
4. Tree Importance - Use LightGBM for importance ranking

Target: 1,500+ features -> 400 gold features per pair
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection pipeline."""
    # Variance filter
    variance_threshold: float = 0.01  # Remove features with var < this

    # Correlation filter
    correlation_threshold: float = 0.95  # Remove if correlation > this

    # Mutual information
    mi_percentile: float = 60.0  # Keep top 40% by MI

    # Tree importance
    n_estimators: int = 200  # LightGBM estimators for importance
    top_k_features: int = 400  # Final number of gold features

    # Random seed for reproducibility
    random_state: int = 42


class FeatureSelector:
    """
    Multi-stage feature selection pipeline.

    Pipeline:
        1. Variance Filter: Remove constant/near-constant features
        2. Correlation Filter: Remove highly correlated redundant features
        3. Mutual Information: Rank features by information gain with target
        4. Tree Importance: Final ranking using LightGBM feature importance

    Usage:
        selector = FeatureSelector()
        gold_features = selector.fit_transform(X_train, y_train, feature_names)
        X_train_selected = X_train[:, selector.selected_indices_]
        X_test_selected = X_test[:, selector.selected_indices_]
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        """Initialize feature selector with configuration."""
        self.config = config or FeatureSelectionConfig()

        # State (set after fit)
        self.selected_features_: Optional[List[str]] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self.stage_results_: Optional[Dict[str, Dict]] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> 'FeatureSelector':
        """
        Fit the feature selector.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            self
        """
        logger.info(f"Starting feature selection on {X.shape[1]} features...")

        self.stage_results_ = {}
        current_features = np.arange(len(feature_names))
        current_names = list(feature_names)

        # Stage 1: Variance Filter
        logger.info("Stage 1: Variance Filter")
        var_mask = self._variance_filter(X[:, current_features])
        current_features = current_features[var_mask]
        current_names = [n for n, m in zip(current_names, var_mask) if m]
        self.stage_results_['variance'] = {
            'input_features': len(feature_names),
            'output_features': len(current_features),
            'removed': len(feature_names) - len(current_features)
        }
        logger.info(f"  {len(feature_names)} -> {len(current_features)} features")

        # Stage 2: Correlation Filter
        logger.info("Stage 2: Correlation Filter")
        corr_mask = self._correlation_filter(X[:, current_features])
        current_features = current_features[corr_mask]
        current_names = [n for n, m in zip(current_names, corr_mask) if m]
        self.stage_results_['correlation'] = {
            'input_features': self.stage_results_['variance']['output_features'],
            'output_features': len(current_features),
            'removed': self.stage_results_['variance']['output_features'] - len(current_features)
        }
        logger.info(f"  {self.stage_results_['variance']['output_features']} -> {len(current_features)} features")

        # Stage 3: Mutual Information
        logger.info("Stage 3: Mutual Information Ranking")
        mi_scores = self._mutual_information(X[:, current_features], y)
        mi_threshold = np.percentile(mi_scores, 100 - self.config.mi_percentile)
        mi_mask = mi_scores >= mi_threshold
        current_features = current_features[mi_mask]
        current_names = [n for n, m in zip(current_names, mi_mask) if m]
        mi_scores = mi_scores[mi_mask]
        self.stage_results_['mutual_info'] = {
            'input_features': self.stage_results_['correlation']['output_features'],
            'output_features': len(current_features),
            'removed': self.stage_results_['correlation']['output_features'] - len(current_features)
        }
        logger.info(f"  {self.stage_results_['correlation']['output_features']} -> {len(current_features)} features")

        # Stage 4: Tree Importance
        logger.info("Stage 4: Tree Importance (LightGBM)")
        tree_scores = self._tree_importance(X[:, current_features], y)

        # Combine MI and tree scores
        combined_scores = 0.5 * self._normalize(mi_scores) + 0.5 * self._normalize(tree_scores)

        # Select top K
        top_k = min(self.config.top_k_features, len(current_features))
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        self.selected_indices_ = current_features[top_indices]
        self.selected_features_ = [current_names[i] for i in top_indices]
        self.feature_scores_ = {
            name: float(combined_scores[top_indices[i]])
            for i, name in enumerate(self.selected_features_)
        }

        self.stage_results_['tree_importance'] = {
            'input_features': self.stage_results_['mutual_info']['output_features'],
            'output_features': len(self.selected_features_),
            'removed': self.stage_results_['mutual_info']['output_features'] - len(self.selected_features_)
        }

        logger.info(f"  {self.stage_results_['mutual_info']['output_features']} -> {len(self.selected_features_)} GOLD features")
        logger.info(f"Final: Selected {len(self.selected_features_)} gold features from {len(feature_names)} original")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted selector.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Selected features (n_samples, n_selected)
        """
        if self.selected_indices_ is None:
            raise RuntimeError("FeatureSelector not fitted. Call fit() first.")

        return X[:, self.selected_indices_]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _variance_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Remove features with low variance.

        Features with variance below threshold are constant or near-constant
        and provide no discriminative power.
        """
        variances = np.nanvar(X, axis=0)
        mask = variances >= self.config.variance_threshold
        return mask

    def _correlation_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Remove highly correlated features.

        When two features are highly correlated (>0.95), we keep the first
        and remove the second to reduce redundancy.
        """
        n_features = X.shape[1]

        # Handle NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute correlation matrix efficiently
        # Standardize first
        means = np.mean(X_clean, axis=0)
        stds = np.std(X_clean, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        X_std = (X_clean - means) / stds

        # Correlation matrix
        corr_matrix = np.abs(np.corrcoef(X_std.T))

        # Find highly correlated pairs
        mask = np.ones(n_features, dtype=bool)
        for i in range(n_features):
            if not mask[i]:
                continue
            for j in range(i + 1, n_features):
                if not mask[j]:
                    continue
                if abs(corr_matrix[i, j]) > self.config.correlation_threshold:
                    mask[j] = False  # Remove the second feature

        return mask

    def _mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mutual information between features and target.

        Uses sklearn's mutual_info_classif for classification tasks.
        """
        try:
            from sklearn.feature_selection import mutual_info_classif

            # Handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            mi_scores = mutual_info_classif(
                X_clean, y,
                discrete_features=False,
                random_state=self.config.random_state,
                n_neighbors=5
            )

            return mi_scores

        except Exception as e:
            logger.warning(f"MI computation failed: {e}. Using variance as fallback.")
            return np.nanvar(X, axis=0)

    def _tree_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute feature importance using LightGBM.

        Tree-based importance captures non-linear relationships
        that linear methods miss.
        """
        try:
            import lightgbm as lgb

            # Handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Train a quick LightGBM
            model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbose=-1,
                random_state=self.config.random_state
            )

            model.fit(X_clean, y)

            return model.feature_importances_

        except Exception as e:
            logger.warning(f"Tree importance failed: {e}. Using variance as fallback.")
            return np.nanvar(X, axis=0)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val < 1e-10:
            return np.ones_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def save(self, path: Path):
        """Save fitted selector to disk."""
        path = Path(path)

        state = {
            'config': self.config,
            'selected_features': self.selected_features_,
            'selected_indices': self.selected_indices_,
            'feature_scores': self.feature_scores_,
            'stage_results': self.stage_results_,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Also save human-readable summary
        summary_path = path.with_suffix('.json')
        summary = {
            'n_selected': len(self.selected_features_) if self.selected_features_ else 0,
            'top_20_features': (
                list(self.selected_features_[:20]) if self.selected_features_ else []
            ),
            'stage_results': self.stage_results_,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved selector to {path}")

    @classmethod
    def load(cls, path: Path) -> 'FeatureSelector':
        """Load fitted selector from disk."""
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        selector = cls(config=state['config'])
        selector.selected_features_ = state['selected_features']
        selector.selected_indices_ = state['selected_indices']
        selector.feature_scores_ = state['feature_scores']
        selector.stage_results_ = state['stage_results']

        return selector

    def get_summary(self) -> Dict:
        """Get summary of feature selection."""
        if self.selected_features_ is None:
            return {'error': 'Selector not fitted'}

        return {
            'n_selected': len(self.selected_features_),
            'stage_results': self.stage_results_,
            'top_10_features': self.selected_features_[:10],
            'top_10_scores': [
                self.feature_scores_[f] for f in self.selected_features_[:10]
            ],
        }


class AdaptiveFeatureSelector:
    """
    Adaptive feature selector that maintains separate feature sets per symbol.

    Each symbol may have different optimal features due to:
    - Different liquidity patterns
    - Different correlation structures
    - Different information sources (e.g., EUR pairs vs JPY pairs)

    Usage:
        adaptive = AdaptiveFeatureSelector()

        # Fit for each symbol
        for symbol in symbols:
            adaptive.fit(symbol, X_train, y_train, feature_names)

        # Transform
        X_selected = adaptive.transform(symbol, X)
    """

    def __init__(
        self,
        base_config: Optional[FeatureSelectionConfig] = None,
        shared_features_pct: float = 0.5
    ):
        """
        Initialize adaptive feature selector.

        Args:
            base_config: Base configuration for all selectors
            shared_features_pct: Percentage of features that should be
                                 shared across all symbols (0-1)
        """
        self.base_config = base_config or FeatureSelectionConfig()
        self.shared_features_pct = shared_features_pct

        self.selectors_: Dict[str, FeatureSelector] = {}
        self.shared_features_: Optional[List[str]] = None

    def fit(
        self,
        symbol: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> 'AdaptiveFeatureSelector':
        """Fit selector for a specific symbol."""
        selector = FeatureSelector(self.base_config)
        selector.fit(X, y, feature_names)
        self.selectors_[symbol] = selector

        # Update shared features
        self._update_shared_features()

        return self

    def transform(self, symbol: str, X: np.ndarray) -> np.ndarray:
        """Transform features for a specific symbol."""
        if symbol not in self.selectors_:
            raise ValueError(f"Symbol {symbol} not fitted. Call fit() first.")

        return self.selectors_[symbol].transform(X)

    def _update_shared_features(self):
        """Update shared features across all fitted symbols."""
        if len(self.selectors_) < 2:
            return

        # Find features that appear in most selectors
        feature_counts = {}
        for selector in self.selectors_.values():
            if selector.selected_features_:
                for f in selector.selected_features_:
                    feature_counts[f] = feature_counts.get(f, 0) + 1

        # Sort by count
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top shared_features_pct as shared
        n_shared = int(len(sorted_features) * self.shared_features_pct)
        self.shared_features_ = [f for f, _ in sorted_features[:n_shared]]

    def get_shared_features(self) -> List[str]:
        """Get features that are important across all symbols."""
        return self.shared_features_ or []

    def save(self, path: Path):
        """Save all selectors to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for symbol, selector in self.selectors_.items():
            selector.save(path / f"{symbol}_selector.pkl")

        # Save shared features
        with open(path / "shared_features.json", 'w') as f:
            json.dump({
                'shared_features': self.shared_features_,
                'symbols': list(self.selectors_.keys()),
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'AdaptiveFeatureSelector':
        """Load all selectors from disk."""
        path = Path(path)

        adaptive = cls()

        # Load shared features
        with open(path / "shared_features.json", 'r') as f:
            data = json.load(f)
            adaptive.shared_features_ = data['shared_features']

        # Load individual selectors
        for pkl_file in path.glob("*_selector.pkl"):
            symbol = pkl_file.stem.replace("_selector", "")
            adaptive.selectors_[symbol] = FeatureSelector.load(pkl_file)

        return adaptive


def select_features_fast(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 400
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Fast feature selection for quick experimentation.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        top_k: Number of features to select

    Returns:
        (selected_indices, selected_names, scores)
    """
    config = FeatureSelectionConfig(top_k_features=top_k)
    selector = FeatureSelector(config)
    selector.fit(X, y, feature_names)

    return (
        selector.selected_indices_,
        selector.selected_features_,
        np.array([selector.feature_scores_[f] for f in selector.selected_features_])
    )


def create_feature_selector(
    config: Optional[FeatureSelectionConfig] = None,
    adaptive: bool = False,
    shared_features: int = 200
) -> Union[FeatureSelector, AdaptiveFeatureSelector]:
    """
    Factory function to create feature selectors.

    Args:
        config: Feature selection config (uses defaults if None)
        adaptive: If True, creates AdaptiveFeatureSelector
        shared_features: Min shared features for adaptive selector

    Returns:
        FeatureSelector or AdaptiveFeatureSelector instance
    """
    if config is None:
        config = FeatureSelectionConfig()

    if adaptive:
        return AdaptiveFeatureSelector(config, min_shared_features=shared_features)
    return FeatureSelector(config)


# Export
__all__ = [
    'FeatureSelector',
    'FeatureSelectionConfig',
    'AdaptiveFeatureSelector',
    'select_features_fast',
    'create_feature_selector',
]
