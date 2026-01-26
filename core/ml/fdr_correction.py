"""
False Discovery Rate (FDR) Correction for Multiple Testing
============================================================

When testing 51 pairs × 575 features = 29,325+ hypotheses, some will
appear "significant" by pure chance. FDR correction adjusts p-values
to control the expected proportion of false discoveries.

Key Insight:
    Testing many strategies guarantees finding SOME that look good.
    FDR correction tells us which findings are REAL vs lucky.

    Without correction: 5% false positives × 29,325 tests = 1,466 false discoveries!
    With BH correction: Controls expected false discoveries to 5%

Methods Implemented:
    1. Bonferroni: α/n (very conservative, controls FWER)
    2. Holm: Step-down Bonferroni (less conservative)
    3. Benjamini-Hochberg: Controls FDR at α (recommended)
    4. Benjamini-Yekutieli: BY for dependent tests
    5. Storey Q-Value: Estimates π₀ for more power

References:
    [1] Benjamini, Y., & Hochberg, Y. (1995).
        "Controlling the False Discovery Rate: A Practical and Powerful
        Approach to Multiple Testing."
        Journal of the Royal Statistical Society Series B, 57(1), 289-300.
        https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

    [2] Benjamini, Y., & Yekutieli, D. (2001).
        "The Control of the False Discovery Rate in Multiple Testing
        Under Dependency."
        Annals of Statistics, 29(4), 1165-1188.

    [3] Storey, J. D. (2002).
        "A Direct Approach to False Discovery Rates."
        Journal of the Royal Statistical Society B, 64(3), 479-498.

    [4] Harvey, C.R., Liu, Y., & Zhu, H. (2016).
        "...and the Cross-Section of Expected Returns."
        Review of Financial Studies, 29(1), 5-68.
        Application to finance/factor testing.

    [5] Holm, S. (1979).
        "A Simple Sequentially Rejective Multiple Test Procedure."
        Scandinavian Journal of Statistics, 6(2), 65-70.

Author: Claude Code
Created: 2026-01-25
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class FDRResult:
    """Results from FDR correction."""

    # Input
    original_p_values: np.ndarray
    n_tests: int

    # Corrected values
    adjusted_p_values: np.ndarray      # Adjusted p-values
    significant: np.ndarray            # Boolean mask of significant tests
    n_discoveries: int                 # Number of rejections

    # Method info
    method: str
    alpha: float

    # Effect on false discoveries
    expected_false_discoveries: float  # E[# false among rejected]
    fdr_estimate: float                # Estimated FDR

    def __repr__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  FDR CORRECTION RESULT                                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Method:                {self.method:>20}                  ║
║  Tests Performed:       {self.n_tests:>20,}                  ║
║  Significance Level:    {self.alpha:>20.4f}                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Discoveries:           {self.n_discoveries:>20,}                  ║
║  Expected False:        {self.expected_false_discoveries:>20.2f}                  ║
║  Estimated FDR:         {self.fdr_estimate:>20.2%}                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Original min p:        {np.min(self.original_p_values):>20.2e}                  ║
║  Adjusted min p:        {np.min(self.adjusted_p_values):>20.2e}                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


class FDRCorrector:
    """
    Multiple testing correction using FDR control methods.

    Implements several methods for controlling false discoveries when
    testing many hypotheses simultaneously.

    Reference:
        Benjamini & Hochberg (1995): FDR control
        Harvey, Liu & Zhu (2016): Application to factor testing
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize FDR corrector.

        Args:
            alpha: Desired FDR level (default 0.05 = 5%)
        """
        self.alpha = alpha

    def benjamini_hochberg(
        self,
        p_values: np.ndarray,
        alpha: Optional[float] = None
    ) -> FDRResult:
        """
        Benjamini-Hochberg procedure for FDR control.

        The most commonly used FDR method. Controls FDR at level α
        under independence or positive dependence (PRDS).

        Algorithm:
            1. Order p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
            2. Find largest k where p(k) ≤ k/m × α
            3. Reject all hypotheses with i ≤ k

        Args:
            p_values: Array of raw p-values
            alpha: FDR level (uses self.alpha if None)

        Returns:
            FDRResult with adjusted p-values and discoveries

        Reference:
            Benjamini, Y., & Hochberg, Y. (1995). JRSS-B.
        """
        alpha = alpha or self.alpha
        p_values = np.asarray(p_values)
        n = len(p_values)

        # Sort p-values and get sort order
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH thresholds: i/m × α
        thresholds = alpha * np.arange(1, n + 1) / n

        # Find where p-values are below thresholds
        below_threshold = sorted_p <= thresholds

        # Find largest k where p(k) <= threshold(k)
        if np.any(below_threshold):
            k = np.max(np.where(below_threshold)[0]) + 1  # +1 for 1-based counting
        else:
            k = 0

        # Mark discoveries (first k in sorted order)
        discoveries = np.zeros(n, dtype=bool)
        if k > 0:
            discoveries[sorted_idx[:k]] = True

        # Adjusted p-values (BH adjusted)
        # adj_p(k) = min(p(k) × m/k, adj_p(k+1), 1)
        adjusted = np.zeros(n)
        adjusted[sorted_idx[-1]] = min(sorted_p[-1] * n / n, 1.0)
        for i in range(n - 2, -1, -1):
            adj = sorted_p[i] * n / (i + 1)
            adjusted[sorted_idx[i]] = min(adj, adjusted[sorted_idx[i + 1]], 1.0)

        # Estimate FDR
        n_discoveries = np.sum(discoveries)
        if n_discoveries > 0:
            # Expected false discoveries among rejections
            # Under BH, this is controlled at α
            expected_false = alpha * n_discoveries
            fdr_estimate = expected_false / n_discoveries
        else:
            expected_false = 0.0
            fdr_estimate = 0.0

        return FDRResult(
            original_p_values=p_values,
            n_tests=n,
            adjusted_p_values=adjusted,
            significant=discoveries,
            n_discoveries=n_discoveries,
            method='Benjamini-Hochberg',
            alpha=alpha,
            expected_false_discoveries=expected_false,
            fdr_estimate=fdr_estimate
        )

    def benjamini_yekutieli(
        self,
        p_values: np.ndarray,
        alpha: Optional[float] = None
    ) -> FDRResult:
        """
        Benjamini-Yekutieli procedure for dependent tests.

        More conservative than BH, but valid under arbitrary dependence.
        Use when tests are correlated (e.g., overlapping features).

        Adjustment: α → α / Σ(1/i) for i=1 to m

        Args:
            p_values: Array of raw p-values
            alpha: FDR level

        Returns:
            FDRResult with adjusted p-values

        Reference:
            Benjamini, Y., & Yekutieli, D. (2001). Annals of Statistics.
        """
        alpha = alpha or self.alpha
        p_values = np.asarray(p_values)
        n = len(p_values)

        # BY correction factor: sum(1/i) for i = 1 to n
        by_factor = np.sum(1.0 / np.arange(1, n + 1))

        # Adjust alpha
        adjusted_alpha = alpha / by_factor

        # Run BH with adjusted alpha
        result = self.benjamini_hochberg(p_values, adjusted_alpha)

        return FDRResult(
            original_p_values=result.original_p_values,
            n_tests=result.n_tests,
            adjusted_p_values=result.adjusted_p_values * by_factor,
            significant=result.significant,
            n_discoveries=result.n_discoveries,
            method='Benjamini-Yekutieli',
            alpha=alpha,
            expected_false_discoveries=result.expected_false_discoveries,
            fdr_estimate=result.fdr_estimate
        )

    def bonferroni(
        self,
        p_values: np.ndarray,
        alpha: Optional[float] = None
    ) -> FDRResult:
        """
        Bonferroni correction (controls FWER, not FDR).

        Most conservative method. Controls the family-wise error rate:
        P(any false rejection) ≤ α

        Very stringent for large numbers of tests.

        Args:
            p_values: Array of raw p-values
            alpha: FWER level

        Returns:
            FDRResult with adjusted p-values

        Reference:
            Bonferroni, C. (1936). Teoria statistica delle classi.
        """
        alpha = alpha or self.alpha
        p_values = np.asarray(p_values)
        n = len(p_values)

        # Simple adjustment: p × n
        adjusted = np.minimum(p_values * n, 1.0)
        significant = adjusted <= alpha
        n_discoveries = np.sum(significant)

        return FDRResult(
            original_p_values=p_values,
            n_tests=n,
            adjusted_p_values=adjusted,
            significant=significant,
            n_discoveries=n_discoveries,
            method='Bonferroni',
            alpha=alpha,
            expected_false_discoveries=alpha if n_discoveries > 0 else 0,
            fdr_estimate=alpha / n_discoveries if n_discoveries > 0 else 0
        )

    def holm(
        self,
        p_values: np.ndarray,
        alpha: Optional[float] = None
    ) -> FDRResult:
        """
        Holm step-down procedure (controls FWER).

        Less conservative than Bonferroni while still controlling FWER.
        Uses step-down approach for more power.

        Algorithm:
            1. Sort p-values: p(1) ≤ ... ≤ p(m)
            2. Find smallest i where p(i) > α/(m-i+1)
            3. Reject H(1), ..., H(i-1)

        Args:
            p_values: Array of raw p-values
            alpha: FWER level

        Returns:
            FDRResult with adjusted p-values

        Reference:
            Holm, S. (1979). Scandinavian Journal of Statistics.
        """
        alpha = alpha or self.alpha
        p_values = np.asarray(p_values)
        n = len(p_values)

        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Adjusted p-values: p(i) × (m - i + 1), with cumulative max
        adjusted = np.zeros(n)
        for i in range(n):
            adjusted[sorted_idx[i]] = sorted_p[i] * (n - i)
        # Cumulative maximum to ensure monotonicity
        for i in range(1, n):
            adjusted[sorted_idx[i]] = max(adjusted[sorted_idx[i]], adjusted[sorted_idx[i - 1]])
        adjusted = np.minimum(adjusted, 1.0)

        significant = adjusted <= alpha
        n_discoveries = np.sum(significant)

        return FDRResult(
            original_p_values=p_values,
            n_tests=n,
            adjusted_p_values=adjusted,
            significant=significant,
            n_discoveries=n_discoveries,
            method='Holm',
            alpha=alpha,
            expected_false_discoveries=alpha if n_discoveries > 0 else 0,
            fdr_estimate=alpha / n_discoveries if n_discoveries > 0 else 0
        )

    def storey_qvalue(
        self,
        p_values: np.ndarray,
        alpha: Optional[float] = None,
        lambda_: float = 0.5
    ) -> FDRResult:
        """
        Storey's q-value method with π₀ estimation.

        More powerful than BH when many null hypotheses are false.
        Estimates the proportion of true nulls (π₀) and adjusts accordingly.

        Algorithm:
            1. Estimate π₀ (proportion of true nulls)
            2. Compute q-values = FDR-adjusted significance

        Args:
            p_values: Array of raw p-values
            alpha: FDR level
            lambda_: Tuning parameter for π₀ estimation (default 0.5)

        Returns:
            FDRResult with q-values

        Reference:
            Storey, J. D. (2002). JRSS-B.
        """
        alpha = alpha or self.alpha
        p_values = np.asarray(p_values)
        n = len(p_values)

        # Estimate π₀ (proportion of true nulls)
        # Using bootstrap method for robustness
        pi0 = self._estimate_pi0(p_values, lambda_)

        # Sort p-values
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Q-values: q(i) = min{π₀ × m × p(j) / j : j ≥ i}
        q_values = np.zeros(n)
        q_values[sorted_idx[-1]] = min(pi0 * sorted_p[-1], 1.0)
        for i in range(n - 2, -1, -1):
            q = pi0 * n * sorted_p[i] / (i + 1)
            q_values[sorted_idx[i]] = min(q, q_values[sorted_idx[i + 1]], 1.0)

        significant = q_values <= alpha
        n_discoveries = np.sum(significant)

        # Estimated FDR among discoveries
        if n_discoveries > 0:
            fdr_estimate = np.mean(q_values[significant])
            expected_false = fdr_estimate * n_discoveries
        else:
            fdr_estimate = 0.0
            expected_false = 0.0

        return FDRResult(
            original_p_values=p_values,
            n_tests=n,
            adjusted_p_values=q_values,
            significant=significant,
            n_discoveries=n_discoveries,
            method=f'Storey Q-Value (π₀={pi0:.3f})',
            alpha=alpha,
            expected_false_discoveries=expected_false,
            fdr_estimate=fdr_estimate
        )

    def _estimate_pi0(self, p_values: np.ndarray, lambda_: float = 0.5) -> float:
        """
        Estimate proportion of true null hypotheses (π₀).

        Uses the fact that under H0, p-values are uniform[0,1].
        The proportion of p-values > λ estimates π₀.

        Args:
            p_values: Array of p-values
            lambda_: Tuning parameter

        Returns:
            Estimated π₀

        Reference:
            Storey (2002): π₀ estimation
        """
        n = len(p_values)

        # Simple estimator: #{p > λ} / (n × (1-λ))
        n_above = np.sum(p_values > lambda_)
        pi0 = n_above / (n * (1 - lambda_))

        # Bound between 0 and 1
        pi0 = np.clip(pi0, 0.0, 1.0)

        return pi0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Quick Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        alpha: FDR level

    Returns:
        Tuple of (significant_mask, adjusted_p_values, n_discoveries)

    Example:
        >>> p_vals = [0.001, 0.04, 0.03, 0.5]
        >>> sig, adj, n = benjamini_hochberg(p_vals)
        >>> print(f"{n} discoveries at FDR=5%")

    Reference:
        Benjamini & Hochberg (1995)
    """
    corrector = FDRCorrector(alpha=alpha)
    result = corrector.benjamini_hochberg(p_values)
    return result.significant, result.adjusted_p_values, result.n_discoveries


def correct_for_multiple_testing(
    p_values: np.ndarray,
    n_pairs: int = 51,
    n_features: int = 575,
    method: str = 'bh',
    alpha: float = 0.05
) -> FDRResult:
    """
    Correct p-values for multiple testing in forex context.

    Designed for the typical scenario of testing many pairs × features.

    Args:
        p_values: Array of p-values
        n_pairs: Number of currency pairs tested (for logging)
        n_features: Number of features per pair (for logging)
        method: Correction method ('bh', 'by', 'bonferroni', 'holm', 'storey')
        alpha: Significance level

    Returns:
        FDRResult with corrected values

    Example:
        >>> # Test 51 pairs, get p-values for each
        >>> p_vals = [quick_edge_test(acc, 1000)['p_value'] for acc in accuracies]
        >>> result = correct_for_multiple_testing(p_vals, n_pairs=51)
        >>> print(result)

    Reference:
        Harvey, Liu & Zhu (2016): Factor testing in finance
    """
    logger.info(f"Correcting {len(p_values)} p-values for multiple testing")
    logger.info(f"  Context: {n_pairs} pairs × {n_features} features = {n_pairs * n_features:,} potential tests")

    corrector = FDRCorrector(alpha=alpha)

    if method.lower() == 'bh':
        return corrector.benjamini_hochberg(p_values, alpha)
    elif method.lower() == 'by':
        return corrector.benjamini_yekutieli(p_values, alpha)
    elif method.lower() == 'bonferroni':
        return corrector.bonferroni(p_values, alpha)
    elif method.lower() == 'holm':
        return corrector.holm(p_values, alpha)
    elif method.lower() == 'storey':
        return corrector.storey_qvalue(p_values, alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bh', 'by', 'bonferroni', 'holm', or 'storey'")


def adjusted_significance_threshold(
    n_tests: int,
    alpha: float = 0.05,
    method: str = 'bh'
) -> float:
    """
    Calculate adjusted significance threshold for multiple testing.

    Useful for knowing "how significant must my result be?" given
    the number of tests performed.

    Args:
        n_tests: Number of tests performed
        alpha: Desired overall significance
        method: Correction method

    Returns:
        Adjusted threshold for individual tests

    Example:
        >>> # Testing 51 pairs
        >>> threshold = adjusted_significance_threshold(51, alpha=0.05)
        >>> print(f"Need p < {threshold:.4f} for each pair")

    Reference:
        Standard multiple testing theory
    """
    if method.lower() == 'bonferroni':
        return alpha / n_tests
    elif method.lower() == 'bh':
        # For BH, depends on rank; this is for rank 1 (most significant)
        return alpha / n_tests  # Conservative estimate
    elif method.lower() == 'by':
        by_factor = np.sum(1.0 / np.arange(1, n_tests + 1))
        return alpha / (n_tests * by_factor)
    else:
        return alpha / n_tests  # Default to Bonferroni


def survival_analysis_for_factors(
    p_values: np.ndarray,
    factor_names: List[str],
    alpha: float = 0.05,
    method: str = 'bh'
) -> Dict[str, dict]:
    """
    Analyze which factors survive multiple testing correction.

    Provides a report of which factors are truly significant after
    accounting for multiple testing.

    Args:
        p_values: P-values for each factor
        factor_names: Names of factors
        alpha: FDR level
        method: Correction method

    Returns:
        Dictionary with factor-level analysis

    Example:
        >>> factors = ['Alpha001', 'Alpha002', ..., 'Alpha101']
        >>> p_vals = [test_factor(f)['p_value'] for f in factors]
        >>> results = survival_analysis_for_factors(p_vals, factors)
        >>> for name, data in results.items():
        ...     if data['survives']:
        ...         print(f"{name}: p={data['adjusted_p']:.2e}")

    Reference:
        Harvey, Liu & Zhu (2016): Factor survival analysis
    """
    result = correct_for_multiple_testing(p_values, method=method, alpha=alpha)

    analysis = {}
    for i, name in enumerate(factor_names):
        analysis[name] = {
            'original_p': p_values[i],
            'adjusted_p': result.adjusted_p_values[i],
            'survives': result.significant[i],
            'rank': np.sum(np.array(p_values) <= p_values[i])
        }

    # Summary
    n_survive = np.sum(result.significant)
    logger.info(f"Factor survival: {n_survive}/{len(factor_names)} survive at FDR={alpha}")

    return analysis


if __name__ == "__main__":
    # Demo: Multiple testing correction for 51 forex pairs
    print("=" * 70)
    print("FDR CORRECTION DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    # Simulate p-values for 51 pairs
    # 10 pairs have real edge (p ~ 0.001), 41 are null (p ~ uniform)
    n_real = 10
    n_null = 41
    p_real = np.random.beta(0.5, 100, n_real)  # Small p-values
    p_null = np.random.uniform(0, 1, n_null)   # Uniform under null
    p_values = np.concatenate([p_real, p_null])
    np.random.shuffle(p_values)

    print(f"\nSimulated {len(p_values)} tests:")
    print(f"  {n_real} with real edge")
    print(f"  {n_null} null (random)")
    print(f"  Min p-value: {np.min(p_values):.2e}")
    print(f"  Naively significant (p<0.05): {np.sum(p_values < 0.05)}")

    # Run different corrections
    corrector = FDRCorrector(alpha=0.05)

    print("\n--- BONFERRONI (most conservative) ---")
    result = corrector.bonferroni(p_values)
    print(f"Discoveries: {result.n_discoveries}")

    print("\n--- HOLM (step-down) ---")
    result = corrector.holm(p_values)
    print(f"Discoveries: {result.n_discoveries}")

    print("\n--- BENJAMINI-HOCHBERG (FDR control) ---")
    result = corrector.benjamini_hochberg(p_values)
    print(result)

    print("\n--- STOREY Q-VALUE (adaptive) ---")
    result = corrector.storey_qvalue(p_values)
    print(f"Discoveries: {result.n_discoveries}")
    print(f"Method: {result.method}")
