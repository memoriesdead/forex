"""
Feature Importance Methods (Lopez de Prado AFML)
=================================================
Implementation of feature importance methods from "Advances in Financial Machine Learning"

Sources (Gold Standard Citations):
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
  Chapter 8: Feature Importance

Methods Implemented:
1. MDA (Mean Decrease Accuracy) - Permutation importance
2. MDI (Mean Decrease Impurity) - Tree-based importance
3. SFI (Single Feature Importance) - Univariate importance
4. Clustered Feature Importance - Correlation-aware importance
5. SHAP Integration - Shapley values wrapper

These methods are critical for:
- Identifying truly predictive features vs noise
- Detecting overfitting (features with high MDI but low MDA)
- Regulatory compliance (model explainability)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results."""
    importance: pd.Series
    std: Optional[pd.Series] = None
    method: str = "unknown"
    n_samples: int = 0
    n_features: int = 0


class FeatureImportance:
    """
    Feature Importance Analysis (AFML Methods)

    Implements permutation-based and tree-based feature importance
    with proper cross-validation to avoid overfitting.

    Usage:
        fi = FeatureImportance()
        mda = fi.mean_decrease_accuracy(model, X, y, cv=5)
        mdi = fi.mean_decrease_impurity(model, X)
    """

    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize feature importance calculator.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state

    # =========================================================================
    # MDA: Mean Decrease Accuracy (Permutation Importance)
    # =========================================================================

    def mean_decrease_accuracy(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_repeats: int = 10,
        sample_weight: Optional[np.ndarray] = None
    ) -> FeatureImportanceResult:
        """
        Mean Decrease Accuracy (MDA) - Permutation-based feature importance.

        For each feature:
        1. Fit model on training data
        2. Measure baseline score on OOS data
        3. Permute feature values in OOS data
        4. Measure score drop after permutation
        5. Importance = mean score drop across CV folds

        Reference:
            Lopez de Prado (2018), AFML Chapter 8.3

        Args:
            model: Fitted sklearn-compatible model
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            scoring: Scoring metric ('accuracy', 'f1', 'neg_log_loss', etc.)
            n_repeats: Number of permutation repeats per feature
            sample_weight: Optional sample weights

        Returns:
            FeatureImportanceResult with importance scores
        """
        from sklearn.model_selection import KFold

        n_samples, n_features = X.shape
        feature_names = X.columns.tolist()

        # Initialize importance arrays
        importance_matrix = np.zeros((cv, n_features))

        # Cross-validation
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            model_clone = clone(model)
            if sample_weight is not None:
                model_clone.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
            else:
                model_clone.fit(X_train, y_train)

            # Baseline score
            baseline_score = self._score(model_clone, X_test, y_test, scoring)

            # Permute each feature
            for feat_idx, feature in enumerate(feature_names):
                scores_after_permute = []

                for _ in range(n_repeats):
                    X_permuted = X_test.copy()
                    X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                    score_permuted = self._score(model_clone, X_permuted, y_test, scoring)
                    scores_after_permute.append(baseline_score - score_permuted)

                importance_matrix[fold_idx, feat_idx] = np.mean(scores_after_permute)

        # Aggregate across folds
        importance = pd.Series(
            np.mean(importance_matrix, axis=0),
            index=feature_names,
            name='mda_importance'
        )
        std = pd.Series(
            np.std(importance_matrix, axis=0),
            index=feature_names,
            name='mda_std'
        )

        return FeatureImportanceResult(
            importance=importance.sort_values(ascending=False),
            std=std,
            method='MDA',
            n_samples=n_samples,
            n_features=n_features
        )

    def _score(self, model, X, y, scoring: str) -> float:
        """Calculate model score."""
        if scoring == 'accuracy':
            return model.score(X, y)
        elif scoring == 'neg_log_loss':
            from sklearn.metrics import log_loss
            return -log_loss(y, model.predict_proba(X))
        elif scoring == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y, model.predict(), average='weighted')
        elif scoring == 'roc_auc':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y, model.predict_proba(X)[:, 1])
        else:
            return model.score(X, y)

    # =========================================================================
    # MDI: Mean Decrease Impurity (Tree-based)
    # =========================================================================

    def mean_decrease_impurity(
        self,
        model: Any,
        X: pd.DataFrame,
        normalize: bool = True
    ) -> FeatureImportanceResult:
        """
        Mean Decrease Impurity (MDI) - Tree-based feature importance.

        Uses the built-in feature_importances_ from tree-based models.
        Fast but can be biased toward high-cardinality features.

        Reference:
            Lopez de Prado (2018), AFML Chapter 8.2

        Args:
            model: Fitted tree-based model (RF, XGBoost, LightGBM, etc.)
            X: Feature DataFrame
            normalize: Whether to normalize importances to sum to 1

        Returns:
            FeatureImportanceResult with importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns,
            name='mdi_importance'
        )

        if normalize:
            importance = importance / importance.sum()

        return FeatureImportanceResult(
            importance=importance.sort_values(ascending=False),
            std=None,
            method='MDI',
            n_samples=len(X),
            n_features=len(X.columns)
        )

    # =========================================================================
    # SFI: Single Feature Importance
    # =========================================================================

    def single_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> FeatureImportanceResult:
        """
        Single Feature Importance (SFI) - Univariate importance.

        Trains model on each feature individually and measures OOS score.
        Simple baseline that ignores feature interactions.

        Reference:
            Lopez de Prado (2018), AFML Chapter 8.4

        Args:
            model: sklearn-compatible model
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            FeatureImportanceResult with importance scores
        """
        feature_names = X.columns.tolist()
        scores = {}

        for feature in feature_names:
            X_single = X[[feature]]
            model_clone = clone(model)
            cv_scores = cross_val_score(
                model_clone, X_single, y,
                cv=cv, scoring=scoring, n_jobs=self.n_jobs
            )
            scores[feature] = np.mean(cv_scores)

        importance = pd.Series(scores, name='sfi_importance')

        return FeatureImportanceResult(
            importance=importance.sort_values(ascending=False),
            std=None,
            method='SFI',
            n_samples=len(X),
            n_features=len(X.columns)
        )

    # =========================================================================
    # CLUSTERED FEATURE IMPORTANCE
    # =========================================================================

    def clustered_mda(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_clusters: int = 10,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Tuple[FeatureImportanceResult, Dict[int, List[str]]]:
        """
        Clustered MDA - Correlation-aware feature importance.

        Clusters correlated features together and permutes entire clusters.
        Avoids redundant importance across correlated features.

        Reference:
            Lopez de Prado (2018), AFML Chapter 8.5

        Args:
            model: sklearn-compatible model
            X: Feature DataFrame
            y: Target Series
            n_clusters: Number of feature clusters
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Tuple of (importance result, cluster assignments)
        """
        from sklearn.model_selection import KFold

        # Cluster features by correlation
        corr = X.corr()
        dist = ((1 - corr) / 2).values  # Convert correlation to distance
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        dist_condensed = squareform(dist, checks=False)
        Z = linkage(dist_condensed, method='ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # Map features to clusters
        feature_names = X.columns.tolist()
        cluster_map = {}
        for feat, cluster in zip(feature_names, clusters):
            if cluster not in cluster_map:
                cluster_map[cluster] = []
            cluster_map[cluster].append(feat)

        # Calculate clustered importance
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cluster_importance = {c: [] for c in cluster_map.keys()}

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            baseline = self._score(model_clone, X_test, y_test, scoring)

            for cluster_id, features in cluster_map.items():
                X_permuted = X_test.copy()
                for feat in features:
                    X_permuted[feat] = np.random.permutation(X_permuted[feat].values)
                score_permuted = self._score(model_clone, X_permuted, y_test, scoring)
                cluster_importance[cluster_id].append(baseline - score_permuted)

        # Average importance per cluster
        importance = pd.Series(
            {f"cluster_{c}": np.mean(scores) for c, scores in cluster_importance.items()},
            name='clustered_mda_importance'
        )

        result = FeatureImportanceResult(
            importance=importance.sort_values(ascending=False),
            std=None,
            method='Clustered_MDA',
            n_samples=len(X),
            n_features=len(X.columns)
        )

        return result, cluster_map

    # =========================================================================
    # SHAP INTEGRATION
    # =========================================================================

    def shap_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        n_samples: int = 100,
        check_additivity: bool = False
    ) -> FeatureImportanceResult:
        """
        SHAP (SHapley Additive exPlanations) feature importance.

        Uses Shapley values for theoretically sound feature attribution.
        Requires 'shap' package to be installed.

        Reference:
            Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to
            Interpreting Model Predictions". NeurIPS.

        Args:
            model: Fitted model
            X: Feature DataFrame
            n_samples: Number of samples for SHAP calculation
            check_additivity: Whether to check SHAP additivity

        Returns:
            FeatureImportanceResult with SHAP importance
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP package required. Install with: pip install shap")

        # Subsample for efficiency
        if len(X) > n_samples:
            X_sample = X.sample(n_samples, random_state=self.random_state)
        else:
            X_sample = X

        # Create explainer based on model type
        model_type = type(model).__name__.lower()

        if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type or 'catboost' in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample, check_additivity=check_additivity)
        else:
            # Use KernelExplainer for other models
            background = shap.sample(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # Multi-class: use absolute mean across classes
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Mean absolute SHAP value per feature
        importance = pd.Series(
            shap_values.mean(axis=0),
            index=X.columns,
            name='shap_importance'
        )

        return FeatureImportanceResult(
            importance=importance.sort_values(ascending=False),
            std=pd.Series(shap_values.std(axis=0), index=X.columns),
            method='SHAP',
            n_samples=len(X_sample),
            n_features=len(X.columns)
        )

    # =========================================================================
    # COMPARISON & ANALYSIS
    # =========================================================================

    def compare_methods(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Compare all feature importance methods.

        Useful for identifying potential overfitting:
        - High MDI + Low MDA = likely overfit feature
        - High MDA + Low MDI = complex feature interaction

        Returns:
            DataFrame with importance from all methods
        """
        results = {}

        # MDI (if tree-based)
        if hasattr(model, 'feature_importances_'):
            mdi = self.mean_decrease_impurity(model, X)
            results['MDI'] = mdi.importance

        # MDA
        mda = self.mean_decrease_accuracy(model, X, y, cv=cv)
        results['MDA'] = mda.importance

        # SFI
        try:
            sfi = self.single_feature_importance(model, X, y, cv=cv)
            results['SFI'] = sfi.importance
        except Exception:
            pass

        comparison = pd.DataFrame(results)

        # Normalize for comparison
        comparison_norm = comparison.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10))

        # Add overfit indicator: high MDI but low MDA
        if 'MDI' in comparison_norm.columns and 'MDA' in comparison_norm.columns:
            comparison['overfit_risk'] = comparison_norm['MDI'] - comparison_norm['MDA']

        return comparison.sort_values('MDA', ascending=False)

    def detect_overfit_features(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Detect features likely to cause overfitting.

        Features with high MDI but low MDA are likely memorizing noise.

        Args:
            model: Fitted tree-based model
            X: Feature DataFrame
            y: Target Series
            threshold: MDI-MDA difference threshold

        Returns:
            List of potentially overfit feature names
        """
        comparison = self.compare_methods(model, X, y)

        if 'overfit_risk' not in comparison.columns:
            return []

        overfit_features = comparison[comparison['overfit_risk'] > threshold].index.tolist()

        return overfit_features


# =============================================================================
# SAMPLE WEIGHTS BY UNIQUENESS (AFML Ch. 4)
# =============================================================================

class SampleWeightsByUniqueness:
    """
    Sample Weights by Label Uniqueness (AFML Ch. 4)

    Assigns sample weights based on how unique each sample's label is.
    Samples with overlapping labels get lower weights to avoid overweighting.

    Reference:
        Lopez de Prado (2018), AFML Chapter 4.5
    """

    @staticmethod
    def get_indicator_matrix(
        t1: pd.Series,
        close_idx: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Create indicator matrix showing which samples span each time point.

        Args:
            t1: Series with sample end times (index = start time)
            close_idx: Full datetime index of the price series

        Returns:
            DataFrame with samples as columns, times as rows
        """
        indicator = pd.DataFrame(0, index=close_idx, columns=range(len(t1)))

        for i, (t0, t1_val) in enumerate(t1.items()):
            indicator.loc[t0:t1_val, i] = 1

        return indicator

    @staticmethod
    def get_average_uniqueness(indicator_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate average uniqueness for each sample.

        Uniqueness = 1 / (number of concurrent samples)

        Args:
            indicator_matrix: Output from get_indicator_matrix

        Returns:
            Series of uniqueness values per sample
        """
        # Concurrency at each time point
        concurrency = indicator_matrix.sum(axis=1)

        # Average uniqueness per sample
        uniqueness = indicator_matrix.div(concurrency, axis=0)
        avg_uniqueness = uniqueness.sum() / indicator_matrix.sum()

        return avg_uniqueness

    @staticmethod
    def get_sample_weights(
        t1: pd.Series,
        close_idx: pd.DatetimeIndex,
        num_co_events: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get sample weights based on uniqueness.

        Args:
            t1: Series with sample end times (index = start time)
            close_idx: Full datetime index
            num_co_events: Optional pre-computed concurrency

        Returns:
            Series of sample weights
        """
        # Get indicator matrix
        indicator = SampleWeightsByUniqueness.get_indicator_matrix(t1, close_idx)

        # Get uniqueness
        uniqueness = SampleWeightsByUniqueness.get_average_uniqueness(indicator)

        # Normalize weights to sum to number of samples
        weights = uniqueness * len(uniqueness) / uniqueness.sum()

        return weights

    @staticmethod
    def get_time_decay_weights(
        t1: pd.Series,
        decay_factor: float = 0.5
    ) -> pd.Series:
        """
        Apply time decay to sample weights (more recent = higher weight).

        Args:
            t1: Series with sample end times
            decay_factor: Decay rate (0 = uniform, 1 = linear decay)

        Returns:
            Time-decayed sample weights
        """
        # Linear decay from oldest to newest
        n = len(t1)
        decay = np.linspace(1 - decay_factor, 1, n)

        return pd.Series(decay, index=t1.index)

    @staticmethod
    def combine_weights(
        uniqueness_weights: pd.Series,
        time_decay_weights: pd.Series
    ) -> pd.Series:
        """
        Combine uniqueness and time decay weights.

        Args:
            uniqueness_weights: From get_sample_weights
            time_decay_weights: From get_time_decay_weights

        Returns:
            Combined normalized weights
        """
        combined = uniqueness_weights * time_decay_weights
        return combined * len(combined) / combined.sum()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_feature_importance(n_jobs: int = -1) -> FeatureImportance:
    """Factory function for FeatureImportance."""
    return FeatureImportance(n_jobs=n_jobs)


def get_sample_weights(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex
) -> pd.Series:
    """Convenience function for sample weights."""
    return SampleWeightsByUniqueness.get_sample_weights(t1, close_idx)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Feature Importance & Sample Weights Test")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Generate test data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=5,
        n_redundant=5, n_clusters_per_class=2, random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Test feature importance
    fi = FeatureImportance()

    print("\n1. MDI (Mean Decrease Impurity):")
    mdi = fi.mean_decrease_impurity(model, X)
    print(mdi.importance.head(5))

    print("\n2. MDA (Mean Decrease Accuracy):")
    mda = fi.mean_decrease_accuracy(model, X, y, cv=3, n_repeats=3)
    print(mda.importance.head(5))

    print("\n3. SFI (Single Feature Importance):")
    sfi = fi.single_feature_importance(model, X, y, cv=3)
    print(sfi.importance.head(5))

    print("\n4. Method Comparison:")
    comparison = fi.compare_methods(model, X, y, cv=3)
    print(comparison.head(5))

    print("\n5. Overfit Features:")
    overfit = fi.detect_overfit_features(model, X, y)
    print(f"Potentially overfit features: {overfit}")

    # Test sample weights
    print("\n6. Sample Weights by Uniqueness:")
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    t1 = pd.Series(dates[5:], index=dates[:-5])  # 5-day holding periods
    weights = SampleWeightsByUniqueness.get_sample_weights(t1, dates)
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"Mean weight: {weights.mean():.3f}")

    print("\nTest PASSED")
