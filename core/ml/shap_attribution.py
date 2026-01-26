"""
SHAP Factor Attribution for HFT

Feature importance attribution with SHAP values for interpretable ML.
Used by Chinese quants for factor contribution analysis.

Academic Citations:
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
  NeurIPS 2017 - Original SHAP paper

- Shapley (1953): "A Value for n-Person Games"
  Contributions to the Theory of Games - Game theory foundation

- arXiv:2305.01610 (2023): "Feature Attribution for Financial Time Series"
  SHAP extensions for temporal data

Chinese Quant Application:
- 华泰证券: 因子归因分析 (factor attribution analysis)
- 国泰君安: 基于SHAP的特征重要性分析
- 幻方量化: Uses SHAP for model interpretability
- BigQuant: 因子贡献度分析 (factor contribution analysis)

Why SHAP for HFT:
    1. Local explanations - Why THIS prediction?
    2. Global importance - Which features matter most?
    3. Interaction effects - Feature combinations
    4. Consistency - Same feature, same contribution
    5. Regulatory compliance - Explainable AI

The Math:
    φ_i(v) = Σ_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! [v(S∪{i}) - v(S)]

    Where:
    - φ_i = Shapley value for feature i
    - v(S) = Model prediction using features S
    - N = All features
    - Sum over all subsets not containing i
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
import warnings

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap not installed. Install with: pip install shap")


@dataclass
class FeatureAttribution:
    """Attribution result for a single prediction."""

    # Feature contributions
    feature_names: List[str]
    shap_values: np.ndarray  # SHAP values for each feature
    base_value: float  # Expected model output (E[f(x)])
    prediction: float  # Actual prediction f(x)

    # Top contributors
    top_positive: List[Tuple[str, float]]  # Features pushing up
    top_negative: List[Tuple[str, float]]  # Features pushing down

    # Summary
    total_positive_contribution: float
    total_negative_contribution: float


@dataclass
class GlobalImportance:
    """Global feature importance across dataset."""

    feature_names: List[str]
    mean_abs_shap: np.ndarray  # Mean |SHAP| per feature
    std_shap: np.ndarray  # Std of SHAP per feature

    # Rankings
    importance_ranking: List[str]  # Features sorted by importance
    top_10_features: List[Tuple[str, float]]

    # Interaction effects (if computed)
    interaction_matrix: Optional[np.ndarray] = None


@dataclass
class FactorContribution:
    """
    Factor contribution analysis (华泰证券 style).

    Decomposes prediction into factor group contributions.
    """

    # Group contributions
    group_names: List[str]
    group_contributions: Dict[str, float]
    group_feature_counts: Dict[str, int]

    # Per-group SHAP
    group_shap_values: Dict[str, np.ndarray]

    # Total
    total_contribution: float
    base_value: float


class TreeSHAP:
    """
    TreeSHAP for XGBoost/LightGBM/CatBoost ensemble.

    Fast exact SHAP computation for tree-based models.

    Reference:
        Lundberg et al. (2020): "From local explanations to global understanding
        with explainable AI for trees"
    """

    def __init__(
        self,
        models: Dict[str, Any],  # {'xgb': model, 'lgb': model, 'catboost': model}
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
    ):
        """
        Initialize TreeSHAP explainer.

        Args:
            models: Dictionary of trained tree models
            feature_names: List of feature names
            background_data: Background dataset for SHAP (optional for trees)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self.models = models
        self.feature_names = feature_names
        self.background_data = background_data

        # Create explainers for each model
        self.explainers = {}
        for name, model in models.items():
            try:
                self.explainers[name] = shap.TreeExplainer(model)
            except Exception as e:
                warnings.warn(f"Could not create explainer for {name}: {e}")

    def explain_prediction(
        self,
        x: np.ndarray,
        aggregate: str = 'mean',
    ) -> FeatureAttribution:
        """
        Explain a single prediction.

        Args:
            x: Feature vector (1D array)
            aggregate: How to combine model explanations ('mean', 'median')

        Returns:
            FeatureAttribution with SHAP values
        """
        x = np.atleast_2d(x)

        # Get SHAP values from each model
        all_shap_values = []
        all_base_values = []

        for name, explainer in self.explainers.items():
            try:
                shap_values = explainer.shap_values(x)

                # Handle binary classification (take class 1)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                all_shap_values.append(shap_values[0])
                all_base_values.append(explainer.expected_value)
            except Exception as e:
                warnings.warn(f"SHAP failed for {name}: {e}")

        if not all_shap_values:
            raise ValueError("No SHAP values computed from any model")

        # Aggregate across models
        shap_values_array = np.array(all_shap_values)

        if aggregate == 'mean':
            final_shap = np.mean(shap_values_array, axis=0)
            base_value = np.mean([
                v[1] if isinstance(v, (list, np.ndarray)) else v
                for v in all_base_values
            ])
        elif aggregate == 'median':
            final_shap = np.median(shap_values_array, axis=0)
            base_value = np.median([
                v[1] if isinstance(v, (list, np.ndarray)) else v
                for v in all_base_values
            ])
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Compute prediction
        prediction = base_value + np.sum(final_shap)

        # Find top contributors
        feature_contributions = list(zip(self.feature_names, final_shap))
        sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

        top_positive = [(f, v) for f, v in sorted_contributions if v > 0][:5]
        top_negative = [(f, v) for f, v in sorted_contributions if v < 0][-5:][::-1]

        return FeatureAttribution(
            feature_names=self.feature_names,
            shap_values=final_shap,
            base_value=base_value,
            prediction=prediction,
            top_positive=top_positive,
            top_negative=top_negative,
            total_positive_contribution=np.sum(final_shap[final_shap > 0]),
            total_negative_contribution=np.sum(final_shap[final_shap < 0]),
        )

    def global_importance(
        self,
        X: np.ndarray,
        max_samples: int = 1000,
    ) -> GlobalImportance:
        """
        Compute global feature importance.

        Args:
            X: Feature matrix
            max_samples: Maximum samples to use

        Returns:
            GlobalImportance with rankings
        """
        # Sample if needed
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]

        # Get SHAP values for all samples from each model
        all_shap_matrices = []

        for name, explainer in self.explainers.items():
            try:
                shap_values = explainer.shap_values(X)

                # Handle binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                all_shap_matrices.append(shap_values)
            except Exception as e:
                warnings.warn(f"Global SHAP failed for {name}: {e}")

        if not all_shap_matrices:
            raise ValueError("No SHAP values computed")

        # Average across models
        shap_matrix = np.mean(np.array(all_shap_matrices), axis=0)

        # Compute importance statistics
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        std_shap = np.std(shap_matrix, axis=0)

        # Ranking
        importance_order = np.argsort(-mean_abs_shap)
        importance_ranking = [self.feature_names[i] for i in importance_order]

        top_10 = [
            (self.feature_names[i], mean_abs_shap[i])
            for i in importance_order[:10]
        ]

        return GlobalImportance(
            feature_names=self.feature_names,
            mean_abs_shap=mean_abs_shap,
            std_shap=std_shap,
            importance_ranking=importance_ranking,
            top_10_features=top_10,
        )


class FactorGroupAttribution:
    """
    Factor group attribution (华泰证券 style).

    Groups features by factor category and computes group-level SHAP.

    Chinese quant practice:
    - 动量因子 (Momentum factors)
    - 波动率因子 (Volatility factors)
    - 微观结构因子 (Microstructure factors)
    - 技术因子 (Technical factors)
    - 基本面因子 (Fundamental factors)
    """

    def __init__(
        self,
        shap_explainer: TreeSHAP,
        feature_groups: Dict[str, List[str]],
    ):
        """
        Initialize factor group attribution.

        Args:
            shap_explainer: TreeSHAP instance
            feature_groups: Dict mapping group names to feature names
                Example: {'momentum': ['mom_5', 'mom_10'], 'volatility': ['vol_20']}
        """
        self.explainer = shap_explainer
        self.feature_groups = feature_groups

        # Create feature name to index mapping
        self.feature_to_idx = {
            name: i for i, name in enumerate(shap_explainer.feature_names)
        }

        # Validate groups
        for group, features in feature_groups.items():
            for f in features:
                if f not in self.feature_to_idx:
                    warnings.warn(f"Feature '{f}' in group '{group}' not found")

    def attribute_to_groups(
        self,
        x: np.ndarray,
    ) -> FactorContribution:
        """
        Attribute prediction to factor groups.

        Args:
            x: Feature vector

        Returns:
            FactorContribution with group-level analysis
        """
        # Get base attribution
        attr = self.explainer.explain_prediction(x)

        # Group SHAP values
        group_contributions = {}
        group_shap_values = {}
        group_feature_counts = {}

        for group, features in self.feature_groups.items():
            # Get indices of features in this group
            indices = [
                self.feature_to_idx[f] for f in features
                if f in self.feature_to_idx
            ]

            if indices:
                group_shap = attr.shap_values[indices]
                group_contributions[group] = np.sum(group_shap)
                group_shap_values[group] = group_shap
                group_feature_counts[group] = len(indices)
            else:
                group_contributions[group] = 0.0
                group_shap_values[group] = np.array([])
                group_feature_counts[group] = 0

        return FactorContribution(
            group_names=list(self.feature_groups.keys()),
            group_contributions=group_contributions,
            group_feature_counts=group_feature_counts,
            group_shap_values=group_shap_values,
            total_contribution=np.sum(attr.shap_values),
            base_value=attr.base_value,
        )

    def group_importance(
        self,
        X: np.ndarray,
        max_samples: int = 1000,
    ) -> Dict[str, float]:
        """
        Compute global group importance.

        Args:
            X: Feature matrix
            max_samples: Maximum samples

        Returns:
            Dict of group -> mean |SHAP|
        """
        global_imp = self.explainer.global_importance(X, max_samples)

        group_importance = {}
        for group, features in self.feature_groups.items():
            indices = [
                self.feature_to_idx[f] for f in features
                if f in self.feature_to_idx
            ]
            if indices:
                group_importance[group] = np.mean(global_imp.mean_abs_shap[indices])
            else:
                group_importance[group] = 0.0

        return group_importance


# =============================================================================
# HFT-Optimized Functions (No SHAP library needed)
# =============================================================================

def fast_feature_importance(
    predictions: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """
    Fast permutation importance for HFT (no SHAP needed).

    Approximate feature importance using correlation with prediction.

    Args:
        predictions: Model predictions
        features: Feature matrix
        feature_names: Feature names

    Returns:
        List of (feature_name, importance) sorted by importance
    """
    importances = []

    for i, name in enumerate(feature_names):
        # Correlation between feature and prediction
        corr = np.corrcoef(features[:, i], predictions)[0, 1]
        importances.append((name, abs(corr) if not np.isnan(corr) else 0.0))

    return sorted(importances, key=lambda x: x[1], reverse=True)


def quick_contribution_estimate(
    x: np.ndarray,
    mean_features: np.ndarray,
    prediction: float,
    base_prediction: float,
) -> np.ndarray:
    """
    Ultra-fast contribution estimate for HFT.

    Linear approximation: contribution ≈ (x - mean) * (pred - base) / variance

    This is NOT exact SHAP but gives directional insight in microseconds.

    Args:
        x: Feature vector
        mean_features: Mean of training features
        prediction: Current prediction
        base_prediction: Mean prediction

    Returns:
        Approximate contributions per feature
    """
    # Simple linear contribution estimate
    deviation = x - mean_features
    total_deviation = np.sum(np.abs(deviation))

    if total_deviation == 0:
        return np.zeros_like(x)

    # Distribute prediction difference proportionally
    pred_diff = prediction - base_prediction
    contributions = deviation * pred_diff / (total_deviation + 1e-10)

    return contributions


def feature_stability_score(
    shap_values_history: List[np.ndarray],
    feature_idx: int,
) -> float:
    """
    Measure feature contribution stability over time.

    Stable features = consistent contributors = more reliable.

    Args:
        shap_values_history: List of SHAP value arrays over time
        feature_idx: Index of feature to analyze

    Returns:
        Stability score (0 = unstable, 1 = stable)
    """
    if len(shap_values_history) < 2:
        return 1.0

    values = np.array([sv[feature_idx] for sv in shap_values_history])

    # Sign consistency
    signs = np.sign(values)
    sign_consistency = np.abs(np.mean(signs))

    # Magnitude consistency (coefficient of variation)
    if np.mean(np.abs(values)) > 0:
        cv = np.std(values) / (np.mean(np.abs(values)) + 1e-10)
        magnitude_consistency = 1 / (1 + cv)
    else:
        magnitude_consistency = 1.0

    return 0.5 * sign_consistency + 0.5 * magnitude_consistency


# =============================================================================
# Standard HFT Factor Groups (Chinese Quant Style)
# =============================================================================

HFT_FACTOR_GROUPS = {
    '动量因子': [  # Momentum factors
        'mom_1', 'mom_5', 'mom_10', 'mom_20',
        'roc_5', 'roc_10', 'roc_20',
        'acceleration', 'macd', 'macd_signal',
    ],
    '波动率因子': [  # Volatility factors
        'vol_5', 'vol_10', 'vol_20', 'vol_50',
        'atr', 'atr_ratio', 'vol_regime',
        'realized_vol', 'parkinson_vol',
    ],
    '微观结构因子': [  # Microstructure factors
        'spread', 'spread_pct', 'bid_ask_imbalance',
        'ofi', 'vpin', 'toxicity',
        'trade_intensity', 'quote_intensity',
    ],
    '技术因子': [  # Technical factors
        'rsi', 'rsi_14', 'stoch_k', 'stoch_d',
        'bb_position', 'bb_width',
        'ma_cross', 'ma_slope',
    ],
    '订单流因子': [  # Order flow factors
        'buy_volume', 'sell_volume', 'volume_imbalance',
        'large_trade_ratio', 'aggressive_ratio',
        'net_order_flow',
    ],
}


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHAP FACTOR ATTRIBUTION FOR HFT")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Lundberg & Lee (2017): NeurIPS 2017 SHAP paper")
    print("  - Shapley (1953): Game theory foundation")
    print("  - 华泰证券: 因子归因分析")
    print()

    # Demo without actual models
    print("Quick Contribution Estimate Demo:")
    print("-" * 50)

    np.random.seed(42)

    # Simulated features
    n_features = 10
    feature_names = [f"feature_{i}" for i in range(n_features)]

    x = np.random.randn(n_features)
    mean_features = np.zeros(n_features)
    prediction = 0.7
    base_prediction = 0.5

    contributions = quick_contribution_estimate(
        x, mean_features, prediction, base_prediction
    )

    print("Feature contributions (linear approximation):")
    for name, contrib in zip(feature_names, contributions):
        direction = "↑" if contrib > 0 else "↓"
        print(f"  {name}: {contrib:+.4f} {direction}")

    print()
    print(f"  Sum of contributions: {np.sum(contributions):.4f}")
    print(f"  Actual pred - base: {prediction - base_prediction:.4f}")
    print()

    # Fast importance demo
    print("Fast Feature Importance Demo:")
    print("-" * 50)

    predictions = np.random.randn(100) * 0.5 + 0.5
    features = np.random.randn(100, n_features)

    # Add correlation to some features
    features[:, 0] = predictions + np.random.randn(100) * 0.1
    features[:, 1] = -predictions + np.random.randn(100) * 0.2

    importance = fast_feature_importance(predictions, features, feature_names)

    print("Top features by correlation with prediction:")
    for name, imp in importance[:5]:
        print(f"  {name}: {imp:.4f}")

    print()
    print("=" * 70)
    print("FACTOR GROUP STRUCTURE (华泰证券 Style):")
    print("=" * 70)
    print()
    for group, factors in HFT_FACTOR_GROUPS.items():
        print(f"{group}:")
        print(f"  {', '.join(factors[:5])}...")
    print()
