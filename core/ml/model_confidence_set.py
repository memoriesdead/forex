"""
Model Confidence Set (MCS) - Statistical Model Selection

Identifies which models are statistically indistinguishable from the best.
Don't assume all 3 ensemble models are good - MCS tests this rigorously.

Academic Citations:
- Hansen, Lunde, Nason (2011): "The Model Confidence Set"
  Econometrica - Original MCS paper

- White (2000): "A Reality Check for Data Snooping"
  Econometrica - Foundation for multiple model comparison

- Romano & Wolf (2005): "Stepwise Multiple Testing as Formalized Data Snooping"
  Econometrica - Stepdown procedures

- Hansen & Lunde (2005): "A Forecast Comparison of Volatility Models"
  Journal of Applied Econometrics - MCS application

Chinese Quant Application:
- 华泰证券: 模型置信集在因子选择中的应用
- 中金量化: Uses MCS for strategy selection
- 招商证券: "统计显著性检验避免过拟合"

The MCS Procedure:
    1. Start with set M of all candidate models
    2. Test null hypothesis: all models in M are equally good
    3. If rejected, remove worst model
    4. Repeat until null is not rejected
    5. Remaining set M* = Model Confidence Set

    MCS contains the best model(s) with probability 1-α

Why it matters:
    - We have XGBoost, LightGBM, CatBoost
    - Are they all equally good? Or is one clearly worse?
    - MCS tells us which models belong in the "elite set"
    - Only use models in MCS for trading decisions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from scipy import stats
import warnings


@dataclass
class MCSResult:
    """Result of Model Confidence Set procedure."""

    # Included models
    included_models: List[str]  # Models in the confidence set
    excluded_models: List[str]  # Models eliminated

    # Statistics
    n_models_initial: int
    n_models_final: int
    elimination_order: List[str]  # Order models were eliminated

    # P-values
    final_pvalue: float  # P-value for remaining set
    model_pvalues: Dict[str, float]  # P-value when each model was tested

    # Performance comparison
    model_losses: Dict[str, float]  # Average loss for each model
    best_model: str
    worst_model: str

    # Confidence level
    alpha: float
    confidence_level: float


@dataclass
class ModelComparison:
    """Pairwise model comparison result."""

    model_a: str
    model_b: str
    loss_diff: float  # Loss(A) - Loss(B), negative means A is better
    t_statistic: float
    p_value: float
    significant: bool  # Is difference significant?


class ModelConfidenceSet:
    """
    Model Confidence Set for statistical model selection.

    Reference:
        Hansen, Lunde, Nason (2011) Econometrica
    """

    def __init__(
        self,
        alpha: float = 0.10,  # 90% confidence set
        bootstrap_reps: int = 1000,
        block_size: Optional[int] = None,  # For time series
    ):
        """
        Initialize MCS.

        Args:
            alpha: Significance level (1 - confidence)
            bootstrap_reps: Number of bootstrap replications
            block_size: Block size for block bootstrap (time series)
        """
        self.alpha = alpha
        self.bootstrap_reps = bootstrap_reps
        self.block_size = block_size

    def _compute_loss_differences(
        self,
        losses: Dict[str, np.ndarray],
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Compute pairwise loss differences.

        Args:
            losses: Dictionary mapping model names to loss series

        Returns:
            Dictionary mapping (model_i, model_j) to d_ij = loss_i - loss_j
        """
        model_names = list(losses.keys())
        n_models = len(model_names)
        diffs = {}

        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_i, name_j = model_names[i], model_names[j]
                diffs[(name_i, name_j)] = losses[name_i] - losses[name_j]
                diffs[(name_j, name_i)] = losses[name_j] - losses[name_i]

        return diffs

    def _bootstrap_variance(
        self,
        d: np.ndarray,
    ) -> float:
        """
        Bootstrap variance estimate for loss difference.

        Args:
            d: Loss difference series

        Returns:
            Bootstrap variance estimate
        """
        n = len(d)
        block = self.block_size or max(1, int(np.sqrt(n)))

        # Block bootstrap
        boot_means = []
        for _ in range(self.bootstrap_reps):
            n_blocks = int(np.ceil(n / block))
            indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n - block + 1)
                indices.extend(range(start, start + block))
            indices = indices[:n]
            boot_means.append(np.mean(d[indices]))

        return np.var(boot_means)

    def _tmax_statistic(
        self,
        losses: Dict[str, np.ndarray],
        model_set: List[str],
    ) -> Tuple[float, str]:
        """
        Compute T_max statistic for MCS procedure.

        T_max = max_i (d_i_bar / sqrt(var(d_i)))

        where d_i = loss_i - (1/M) * sum_j loss_j

        Args:
            losses: Loss series for each model
            model_set: Current model set

        Returns:
            (T_max value, model with max statistic)
        """
        M = len(model_set)
        n = len(losses[model_set[0]])

        # Compute average loss across models
        avg_loss = np.mean([losses[m] for m in model_set], axis=0)

        # Compute relative performance for each model
        t_stats = {}
        for model in model_set:
            d = losses[model] - avg_loss
            d_bar = np.mean(d)
            var_d = self._bootstrap_variance(d)

            if var_d > 0:
                t_stats[model] = d_bar / np.sqrt(var_d)
            else:
                t_stats[model] = 0.0

        # Find max
        max_model = max(t_stats.keys(), key=lambda m: t_stats[m])
        return t_stats[max_model], max_model

    def _bootstrap_tmax_distribution(
        self,
        losses: Dict[str, np.ndarray],
        model_set: List[str],
    ) -> np.ndarray:
        """
        Bootstrap distribution of T_max under null hypothesis.

        Args:
            losses: Loss series
            model_set: Current model set

        Returns:
            Array of bootstrapped T_max values
        """
        n = len(losses[model_set[0]])
        block = self.block_size or max(1, int(np.sqrt(n)))

        tmax_boot = []
        for _ in range(self.bootstrap_reps):
            # Block bootstrap
            n_blocks = int(np.ceil(n / block))
            indices = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n - block + 1)
                indices.extend(range(start, start + block))
            indices = indices[:n]

            # Resample losses
            losses_boot = {m: losses[m][indices] for m in model_set}

            # Center the bootstrapped losses (under null, all models equal)
            for m in model_set:
                losses_boot[m] = losses_boot[m] - np.mean(losses_boot[m]) + np.mean(losses[m])

            # Compute T_max on bootstrap sample
            tmax, _ = self._tmax_statistic(losses_boot, model_set)
            tmax_boot.append(tmax)

        return np.array(tmax_boot)

    def compute(
        self,
        losses: Dict[str, np.ndarray],
    ) -> MCSResult:
        """
        Compute Model Confidence Set.

        Args:
            losses: Dictionary mapping model names to loss series.
                    Each series should be the same length (matched samples).

        Returns:
            MCSResult with included/excluded models

        Example:
            >>> mcs = ModelConfidenceSet(alpha=0.10)
            >>> losses = {
            ...     'XGBoost': 1 - xgb_accuracy,
            ...     'LightGBM': 1 - lgb_accuracy,
            ...     'CatBoost': 1 - cat_accuracy,
            ... }
            >>> result = mcs.compute(losses)
            >>> print(f"Models in MCS: {result.included_models}")
        """
        model_names = list(losses.keys())
        n_models = len(model_names)

        # Verify equal length
        n_samples = len(losses[model_names[0]])
        for m in model_names:
            assert len(losses[m]) == n_samples, f"Model {m} has different length"

        # Average losses for ranking
        avg_losses = {m: np.mean(losses[m]) for m in model_names}
        best_model = min(avg_losses.keys(), key=lambda m: avg_losses[m])
        worst_model = max(avg_losses.keys(), key=lambda m: avg_losses[m])

        # MCS procedure
        current_set = model_names.copy()
        eliminated = []
        model_pvalues = {}

        while len(current_set) > 1:
            # Compute T_max statistic
            tmax_obs, worst_in_set = self._tmax_statistic(losses, current_set)

            # Bootstrap distribution
            tmax_boot = self._bootstrap_tmax_distribution(losses, current_set)

            # P-value: proportion of bootstrap values >= observed
            pvalue = np.mean(tmax_boot >= tmax_obs)
            model_pvalues[worst_in_set] = pvalue

            if pvalue < self.alpha:
                # Reject null: worst model is significantly worse
                eliminated.append(worst_in_set)
                current_set.remove(worst_in_set)
            else:
                # Cannot reject: remaining models are statistically equivalent
                break

        # Final p-value for remaining set
        if len(current_set) > 1:
            _, _ = self._tmax_statistic(losses, current_set)
            final_pvalue = model_pvalues.get(eliminated[-1], 1.0) if eliminated else 1.0
        else:
            final_pvalue = 1.0

        return MCSResult(
            included_models=current_set,
            excluded_models=eliminated,
            n_models_initial=n_models,
            n_models_final=len(current_set),
            elimination_order=eliminated,
            final_pvalue=final_pvalue,
            model_pvalues=model_pvalues,
            model_losses=avg_losses,
            best_model=best_model,
            worst_model=worst_model,
            alpha=self.alpha,
            confidence_level=1 - self.alpha,
        )


class PairwiseModelComparison:
    """
    Pairwise model comparison with Diebold-Mariano test.

    Reference:
        Diebold & Mariano (1995) "Comparing Predictive Accuracy"
        Journal of Business & Economic Statistics
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize pairwise comparison.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def compare(
        self,
        losses_a: np.ndarray,
        losses_b: np.ndarray,
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> ModelComparison:
        """
        Compare two models using Diebold-Mariano test.

        H0: E[loss_A] = E[loss_B]
        H1: E[loss_A] != E[loss_B]

        Args:
            losses_a: Loss series for model A
            losses_b: Loss series for model B
            model_a: Name of model A
            model_b: Name of model B

        Returns:
            ModelComparison result
        """
        d = losses_a - losses_b
        n = len(d)

        d_bar = np.mean(d)

        # HAC variance estimate (Newey-West)
        max_lag = int(np.ceil(n ** (1/3)))
        gamma_0 = np.var(d)

        gamma_sum = 0
        for h in range(1, max_lag + 1):
            gamma_h = np.mean((d[h:] - d_bar) * (d[:-h] - d_bar))
            weight = 1 - h / (max_lag + 1)  # Bartlett kernel
            gamma_sum += 2 * weight * gamma_h

        var_d = (gamma_0 + gamma_sum) / n

        if var_d > 0:
            t_stat = d_bar / np.sqrt(var_d)
        else:
            t_stat = 0.0

        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        significant = p_value < self.alpha

        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            loss_diff=d_bar,
            t_statistic=t_stat,
            p_value=p_value,
            significant=significant,
        )


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_model_ranking(
    losses: Dict[str, np.ndarray],
    recent_n: int = 100,
) -> List[str]:
    """
    Quick model ranking based on recent performance.

    Args:
        losses: Model loss series
        recent_n: Number of recent samples to use

    Returns:
        Models ranked from best to worst
    """
    recent_losses = {m: np.mean(losses[m][-recent_n:]) for m in losses}
    return sorted(recent_losses.keys(), key=lambda m: recent_losses[m])


def should_exclude_model(
    model_losses: np.ndarray,
    best_losses: np.ndarray,
    threshold: float = 0.02,
) -> bool:
    """
    Quick check if model should be excluded from ensemble.

    Args:
        model_losses: Loss series for candidate model
        best_losses: Loss series for best model
        threshold: Minimum loss difference to exclude

    Returns:
        True if model is significantly worse
    """
    diff = np.mean(model_losses) - np.mean(best_losses)
    return diff > threshold


def adaptive_ensemble_weights(
    losses: Dict[str, np.ndarray],
    window: int = 200,
    softmax_temp: float = 10.0,
) -> Dict[str, float]:
    """
    Compute adaptive ensemble weights based on recent performance.

    Args:
        losses: Model loss series
        window: Rolling window size
        softmax_temp: Softmax temperature (higher = more uniform)

    Returns:
        Weights for each model
    """
    recent_losses = {m: np.mean(losses[m][-window:]) for m in losses}

    # Negative loss for softmax (lower loss = higher weight)
    neg_losses = np.array([-recent_losses[m] for m in losses])

    # Softmax
    exp_losses = np.exp(softmax_temp * (neg_losses - np.max(neg_losses)))
    weights_arr = exp_losses / np.sum(exp_losses)

    return dict(zip(losses.keys(), weights_arr))


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL CONFIDENCE SET - STATISTICAL MODEL SELECTION")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Hansen, Lunde, Nason (2011): Econometrica")
    print("  - White (2000): A Reality Check for Data Snooping")
    print()

    # Simulate model performances
    np.random.seed(42)
    n = 500

    # Generate losses (0-1 loss from predictions)
    # XGBoost and LightGBM are similar, CatBoost is slightly worse
    losses = {
        'XGBoost': 0.18 + 0.05 * np.random.randn(n),
        'LightGBM': 0.19 + 0.05 * np.random.randn(n),
        'CatBoost': 0.22 + 0.05 * np.random.randn(n),  # Worse
    }

    # Compute MCS
    print("Computing Model Confidence Set...")
    print("-" * 50)

    mcs = ModelConfidenceSet(alpha=0.10, bootstrap_reps=500)
    result = mcs.compute(losses)

    print(f"Initial models: {result.n_models_initial}")
    print(f"Final models:   {result.n_models_final}")
    print()
    print(f"Models in MCS (90% confidence):")
    for m in result.included_models:
        print(f"  - {m}: avg loss = {result.model_losses[m]:.4f}")
    print()
    if result.excluded_models:
        print(f"Excluded models:")
        for m in result.excluded_models:
            pval = result.model_pvalues.get(m, 0)
            print(f"  - {m}: avg loss = {result.model_losses[m]:.4f}, p-value = {pval:.4f}")
    print()
    print(f"Best model: {result.best_model}")
    print()

    # Pairwise comparison
    print("PAIRWISE COMPARISONS (Diebold-Mariano)")
    print("-" * 50)

    comparator = PairwiseModelComparison(alpha=0.05)

    for m1 in losses:
        for m2 in losses:
            if m1 < m2:
                comp = comparator.compare(losses[m1], losses[m2], m1, m2)
                sig = "*" if comp.significant else ""
                print(f"  {m1} vs {m2}: diff = {comp.loss_diff:+.4f}, "
                      f"p = {comp.p_value:.4f} {sig}")

    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("  MCS tells us which models are statistically indistinguishable")
    print("  from the best. Only use models IN the MCS for trading.")
    print()
    print("  In this example:")
    print(f"    - {result.included_models} are in the MCS")
    print(f"    - {result.excluded_models} are significantly worse")
    print("=" * 70)
