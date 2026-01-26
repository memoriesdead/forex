"""
Tail Risk and Black Swan Detection

Implements power law analysis, extreme value theory, and Knightian uncertainty
quantification for understanding the fundamentally unpredictable component of
financial markets.

Key Insight (Taleb 2007):
    Black swans contribute 30-50% of total variance in financial markets.
    These events are fundamentally unpredictable by definition.

The 94% Rule (Silver Market):
    One day (Hunt brothers' collapse) contributed 94% of total excess kurtosis
    over 32 years of trading. Extreme events dominate.

Key Formulas:
    Power Law:           P(|r| > x) ∝ x^(-α),  α ≈ 3 for markets
    Hill Estimator:      α = k / Σ[ln(X_i / X_k)]
    Pareto VaR:          VaR_α = x_m · (1/p)^(1/α)
    Extreme Value:       ξ (shape parameter), μ (location), σ (scale)

References:
    [1] Taleb, N.N. (2007). "The Black Swan." Random House.
    [2] Knight, F.H. (1921). "Risk, Uncertainty and Profit."
    [3] Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices."
    [4] Gabaix, X. (2009). "Power Laws in Economics and Finance."
        Annual Review of Economics.
    [5] McNeil, A.J., Frey, R., & Embrechts, P. (2005).
        "Quantitative Risk Management." Princeton University Press.
    [6] Hill, B.M. (1975). "A Simple General Approach to Inference About
        the Tail of a Distribution." Annals of Statistics.
    [7] Pickands, J. (1975). "Statistical Inference Using Extreme Order
        Statistics." Annals of Statistics.

Author: Claude Code + Kevin
Created: 2026-01-22
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class TailRiskAnalysis:
    """Complete tail risk analysis results."""
    tail_exponent: float
    is_heavy_tailed: bool
    is_infinite_variance: bool
    variance_from_extremes: float
    kurtosis: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    knightian_uncertainty_ratio: float
    max_predictable_accuracy: float

    def __repr__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  TAIL RISK ANALYSIS                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Tail Exponent α:         {self.tail_exponent:6.2f}                           ║
║  Heavy-Tailed:            {"YES" if self.is_heavy_tailed else "NO":3}                              ║
║  Infinite Variance:       {"YES" if self.is_infinite_variance else "NO":3}                              ║
║  Variance from Extremes:  {self.variance_from_extremes*100:5.1f}%                           ║
║  Excess Kurtosis:         {self.kurtosis:6.2f}                           ║
╠══════════════════════════════════════════════════════════════╣
║  VaR 95%:                 {self.var_95*100:6.2f}%                           ║
║  VaR 99%:                 {self.var_99*100:6.2f}%                           ║
║  Expected Shortfall 95%:  {self.expected_shortfall_95*100:6.2f}%                           ║
╠══════════════════════════════════════════════════════════════╣
║  Knightian Uncertainty:   {self.knightian_uncertainty_ratio*100:5.1f}%                           ║
║  Max Predictable Acc:     {self.max_predictable_accuracy:5.1f}%                           ║
╚══════════════════════════════════════════════════════════════╝
"""


class TailRiskAnalyzer:
    """
    Tail risk analyzer for financial returns.

    Quantifies the fundamental unpredictability from extreme events.
    Black swans (>4σ events) typically account for 30-50% of variance
    and are by definition unpredictable.

    Knightian Uncertainty (Knight 1921):
        Risk = Known probability distribution (modelable)
        Uncertainty = Unknown distribution (NOT modelable)

    Power Law Implications (Gabaix 2009):
        α < 2: Infinite variance
        α < 3: Infinite skewness
        α < 4: Infinite kurtosis
        Financial markets: α ≈ 3 (borderline infinite variance)

    Reference:
        [1] Taleb (2007): Black swan theory
        [2] Knight (1921): Risk vs uncertainty
        [3] Mandelbrot (1963): Power laws in finance
        [4] Gabaix (2009): Power laws review
    """

    def __init__(
        self,
        black_swan_threshold: float = 4.0,
        extreme_percentile: float = 5.0
    ):
        """
        Initialize tail risk analyzer.

        Args:
            black_swan_threshold: Std deviations for black swan classification
            extreme_percentile: Percentile for extreme value analysis
        """
        self.black_swan_threshold = black_swan_threshold
        self.extreme_percentile = extreme_percentile

    def analyze(self, returns: np.ndarray) -> TailRiskAnalysis:
        """
        Complete tail risk analysis.

        Args:
            returns: Array of returns

        Returns:
            TailRiskAnalysis dataclass
        """
        # Tail exponent estimation
        tail_result = self.estimate_tail_exponent(returns)
        alpha = tail_result['alpha']

        # Black swan contribution
        bs_result = self.black_swan_contribution(returns)

        # Heavy tail tests
        is_heavy = alpha < 4  # Finite kurtosis requires α > 4
        is_infinite_var = alpha < 2

        # VaR and ES
        var_95 = self.pareto_var(returns, alpha, 0.95)
        var_99 = self.pareto_var(returns, alpha, 0.99)
        es_95 = self.expected_shortfall(returns, alpha, 0.95)

        # Kurtosis
        kurtosis = stats.kurtosis(returns, fisher=True)

        # Knightian uncertainty: unpredictable fraction
        knightian = bs_result['variance_from_black_swans']

        # Max predictable accuracy accounting for tail risk
        # Black swans are unpredictable, reducing effective ceiling
        base_ceiling = 0.85  # Information theory limit
        max_acc = base_ceiling * (1 - knightian * 0.5)  # Partial adjustment

        return TailRiskAnalysis(
            tail_exponent=alpha,
            is_heavy_tailed=is_heavy,
            is_infinite_variance=is_infinite_var,
            variance_from_extremes=bs_result['variance_from_black_swans'],
            kurtosis=kurtosis,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            knightian_uncertainty_ratio=knightian,
            max_predictable_accuracy=max_acc * 100
        )

    def estimate_tail_exponent(
        self,
        returns: np.ndarray,
        method: str = 'hill'
    ) -> Dict:
        """
        Estimate power law tail exponent.

        Power Law Distribution:
            P(|r| > x) ∝ x^(-α)

        Interpretation:
            α ≈ 3: Typical for financial markets
            α < 2: Infinite variance
            α < 1: Infinite mean
            α > 4: Near-Gaussian

        Hill Estimator (Hill 1975):
            α̂ = k / Σᵢ[ln(X_{(i)} / X_{(k)})]

        Where X_{(i)} are order statistics.

        Reference:
            [6] Hill, B.M. (1975). "A Simple General Approach to Inference
                About the Tail of a Distribution."
            [4] Gabaix (2009): Power laws in economics

        Args:
            returns: Return series
            method: 'hill' or 'mle'

        Returns:
            Dictionary with alpha and diagnostics
        """
        abs_returns = np.abs(returns - np.mean(returns))

        if method == 'hill':
            return self._hill_estimator(abs_returns)
        else:
            return self._mle_pareto(abs_returns)

    def _hill_estimator(self, abs_returns: np.ndarray) -> Dict:
        """
        Hill estimator for tail exponent.

        α̂ = k / Σᵢ₌₁ᵏ [ln(X_{(i)} / X_{(k+1)})]

        Reference:
            [6] Hill (1975)
        """
        n = len(abs_returns)
        sorted_returns = np.sort(abs_returns)[::-1]  # Descending

        # Use top k% (typically 5-10%)
        k = max(10, n // 20)

        threshold = sorted_returns[k]
        if threshold <= 0:
            return {'alpha': 3.0, 'std_error': 1.0, 'k': k}

        # Hill estimator
        log_excesses = np.log(sorted_returns[:k] / threshold)
        alpha = k / np.sum(log_excesses)

        # Standard error
        std_error = alpha / np.sqrt(k)

        # Alternative estimates at different k
        alphas_different_k = []
        for k_test in [k // 2, k, k * 2]:
            if k_test < n and k_test > 5:
                thresh = sorted_returns[k_test]
                if thresh > 0:
                    log_exc = np.log(sorted_returns[:k_test] / thresh)
                    if np.sum(log_exc) > 0:
                        alphas_different_k.append(k_test / np.sum(log_exc))

        # Stability: variance across different k values
        if len(alphas_different_k) > 1:
            stability = 1 - np.std(alphas_different_k) / (np.mean(alphas_different_k) + 1e-6)
        else:
            stability = 0.5

        return {
            'alpha': alpha,
            'std_error': std_error,
            'k': k,
            'threshold': threshold,
            'stability': stability
        }

    def _mle_pareto(self, abs_returns: np.ndarray) -> Dict:
        """
        MLE estimation for Pareto tail.

        For Pareto: f(x) = α x_m^α / x^(α+1)
        MLE: α̂ = n / Σ ln(x_i / x_m)

        Reference:
            [5] McNeil et al. (2005): Quantitative Risk Management
        """
        n = len(abs_returns)
        threshold_pct = 100 - self.extreme_percentile
        x_m = np.percentile(abs_returns, threshold_pct)

        if x_m <= 0:
            return {'alpha': 3.0, 'std_error': 1.0}

        exceedances = abs_returns[abs_returns > x_m]
        if len(exceedances) < 10:
            return {'alpha': 3.0, 'std_error': 1.0}

        log_excesses = np.log(exceedances / x_m)
        alpha = len(exceedances) / np.sum(log_excesses)
        std_error = alpha / np.sqrt(len(exceedances))

        return {
            'alpha': alpha,
            'std_error': std_error,
            'threshold': x_m,
            'n_exceedances': len(exceedances)
        }

    def black_swan_contribution(
        self,
        returns: np.ndarray,
        threshold_sigma: Optional[float] = None
    ) -> Dict:
        """
        Quantify variance contribution from black swan events.

        Black Swan Definition (Taleb):
            - Extreme outlier (rare)
            - Massive impact
            - Retrospective predictability illusion

        The 94% Rule:
            In silver market 1979-2011, ONE DAY (Hunt brothers' collapse)
            contributed 94% of total excess kurtosis.

        Reference:
            [1] Taleb (2007): "The Black Swan"
            [3] Mandelbrot (1963): Fat tails in finance

        Args:
            returns: Return series
            threshold_sigma: Std devs for black swan (default: self.black_swan_threshold)

        Returns:
            Dictionary with black swan analysis
        """
        if threshold_sigma is None:
            threshold_sigma = self.black_swan_threshold

        mu = np.mean(returns)
        sigma = np.std(returns)

        if sigma < 1e-10:
            return {
                'black_swan_count': 0,
                'black_swan_pct': 0,
                'variance_from_black_swans': 0,
                'kurtosis_from_extremes': 0
            }

        # Z-scores
        z_scores = np.abs((returns - mu) / sigma)
        is_black_swan = z_scores > threshold_sigma

        n_black_swans = np.sum(is_black_swan)
        pct_black_swans = n_black_swans / len(returns)

        # Variance contribution
        total_variance = np.var(returns)
        if n_black_swans > 0:
            bs_variance = np.sum((returns[is_black_swan] - mu) ** 2) / len(returns)
            variance_contribution = bs_variance / total_variance
        else:
            variance_contribution = 0

        # Kurtosis contribution from extremes
        fourth_moment_total = np.mean(((returns - mu) / sigma) ** 4)
        if n_black_swans > 0:
            fourth_moment_extremes = np.sum(z_scores[is_black_swan] ** 4) / len(returns)
            kurtosis_contribution = fourth_moment_extremes / fourth_moment_total
        else:
            kurtosis_contribution = 0

        # Top 1 event contribution (like the 94% rule)
        sorted_squared = np.sort((returns - mu) ** 2)[::-1]
        top1_contribution = sorted_squared[0] / np.sum(sorted_squared)

        return {
            'black_swan_count': int(n_black_swans),
            'black_swan_pct': pct_black_swans * 100,
            'variance_from_black_swans': variance_contribution,
            'kurtosis_from_extremes': kurtosis_contribution,
            'top1_variance_contribution': top1_contribution,
            'threshold_sigma': threshold_sigma
        }

    def pareto_var(
        self,
        returns: np.ndarray,
        alpha: float,
        confidence: float = 0.99
    ) -> float:
        """
        Calculate VaR using Pareto tail assumption.

        Pareto VaR:
            VaR_p = x_m · (n_u / (n · (1 - p)))^(1/α)

        Where:
            x_m = threshold
            n_u = exceedances above threshold
            n = total observations

        Reference:
            [5] McNeil et al. (2005): EVT-based VaR
            [7] Pickands (1975): POT method

        Args:
            returns: Return series
            alpha: Tail exponent
            confidence: VaR confidence level

        Returns:
            VaR as positive number (loss magnitude)
        """
        n = len(returns)
        p = 1 - confidence

        # Use negative returns for loss distribution
        losses = -returns

        # Threshold at extreme_percentile
        threshold = np.percentile(losses, 100 - self.extreme_percentile)
        n_exceed = np.sum(losses > threshold)

        if n_exceed < 5 or alpha <= 0:
            return np.percentile(losses, confidence * 100)

        # Pareto VaR
        var = threshold * ((n_exceed / (n * p)) ** (1 / alpha))

        return var

    def expected_shortfall(
        self,
        returns: np.ndarray,
        alpha: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR) using Pareto tail.

        ES Formula (for Pareto):
            ES_p = VaR_p · α / (α - 1)   [for α > 1]

        ES is the expected loss given that loss exceeds VaR.

        Reference:
            [5] McNeil et al. (2005): Coherent risk measures
            Artzner et al. (1999): Coherent measures of risk

        Args:
            returns: Return series
            alpha: Tail exponent
            confidence: Confidence level

        Returns:
            Expected Shortfall as positive number
        """
        var = self.pareto_var(returns, alpha, confidence)

        if alpha > 1:
            es = var * alpha / (alpha - 1)
        else:
            # For α ≤ 1, ES is infinite (heavy tails)
            es = var * 10  # Practical cap

        # Sanity check with empirical
        losses = -returns
        var_empirical = np.percentile(losses, confidence * 100)
        es_empirical = np.mean(losses[losses > var_empirical])

        # Return maximum of theoretical and empirical
        return max(es, es_empirical)

    def extreme_value_fit(self, returns: np.ndarray) -> Dict:
        """
        Fit Generalized Extreme Value (GEV) distribution to block maxima.

        GEV Distribution:
            F(x) = exp{-[1 + ξ(x-μ)/σ]^(-1/ξ)}

        Shape parameter ξ:
            ξ > 0: Fréchet (heavy tail, power law)
            ξ = 0: Gumbel (light tail, exponential)
            ξ < 0: Weibull (bounded)

        Reference:
            [7] Pickands (1975): EVT
            Coles, S. (2001): "An Introduction to Statistical Modeling
                of Extreme Values"

        Args:
            returns: Return series

        Returns:
            Dictionary with GEV parameters
        """
        # Block maxima approach
        block_size = 20  # ~1 month of daily data
        n_blocks = len(returns) // block_size

        if n_blocks < 10:
            return {
                'xi': 0.3,
                'mu': np.mean(returns),
                'sigma': np.std(returns),
                'tail_type': 'frechet'
            }

        block_maxima = []
        for i in range(n_blocks):
            block = returns[i * block_size:(i + 1) * block_size]
            block_maxima.append(np.max(np.abs(block)))

        block_maxima = np.array(block_maxima)

        # Fit GEV using scipy
        try:
            shape, loc, scale = stats.genextreme.fit(block_maxima)
            xi = -shape  # scipy uses opposite sign convention
        except:
            xi = 0.3
            loc = np.mean(block_maxima)
            scale = np.std(block_maxima)

        # Classify tail type
        if xi > 0.1:
            tail_type = 'frechet'  # Heavy tail
        elif xi < -0.1:
            tail_type = 'weibull'  # Bounded
        else:
            tail_type = 'gumbel'  # Light tail

        return {
            'xi': xi,
            'mu': loc,
            'sigma': scale,
            'tail_type': tail_type,
            'is_heavy_tailed': xi > 0
        }

    def compute_features(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute all tail risk features for ML.

        Args:
            returns: Return series

        Returns:
            Dictionary of features
        """
        analysis = self.analyze(returns)

        # Additional features
        tail_result = self.estimate_tail_exponent(returns)
        bs_result = self.black_swan_contribution(returns)
        gev = self.extreme_value_fit(returns)

        return {
            'tail_exponent': analysis.tail_exponent,
            'tail_is_heavy': float(analysis.is_heavy_tailed),
            'tail_is_infinite_var': float(analysis.is_infinite_variance),
            'tail_variance_from_extremes': analysis.variance_from_extremes,
            'tail_kurtosis': analysis.kurtosis,
            'tail_var_95': analysis.var_95,
            'tail_var_99': analysis.var_99,
            'tail_es_95': analysis.expected_shortfall_95,
            'tail_knightian_ratio': analysis.knightian_uncertainty_ratio,
            'tail_hill_stability': tail_result.get('stability', 0.5),
            'tail_gev_xi': gev['xi'],
            'tail_gev_heavy': float(gev['is_heavy_tailed']),
            'tail_top1_contribution': bs_result['top1_variance_contribution'],
            'tail_black_swan_pct': bs_result['black_swan_pct']
        }


def calculate_knightian_uncertainty(returns: np.ndarray) -> Dict:
    """
    Quantify Knightian uncertainty in return series.

    Knight (1921) Distinguished:
        Risk: Measurable uncertainty (known distribution)
        Uncertainty: Unmeasurable (unknown distribution)

    In practice, Knightian uncertainty manifests as:
        1. Black swan events with no prior
        2. Regime changes
        3. Structural breaks

    Reference:
        [2] Knight, F.H. (1921). "Risk, Uncertainty and Profit."

    Args:
        returns: Return series

    Returns:
        Dictionary with uncertainty decomposition
    """
    analyzer = TailRiskAnalyzer()

    # Black swan contribution (true unknowns)
    bs = analyzer.black_swan_contribution(returns)
    black_swan_uncertainty = bs['variance_from_black_swans']

    # Regime change detection (structural unknowns)
    # Simple: variance ratio test
    n = len(returns)
    mid = n // 2
    var_first = np.var(returns[:mid])
    var_second = np.var(returns[mid:])
    regime_ratio = max(var_first, var_second) / (min(var_first, var_second) + 1e-10)
    regime_uncertainty = min(0.2, (regime_ratio - 1) / 10)  # Cap at 20%

    # Tail risk (distribution unknowns)
    tail = analyzer.estimate_tail_exponent(returns)
    if tail['alpha'] < 3:
        tail_uncertainty = 0.1  # Heavy tails = more unknowable
    else:
        tail_uncertainty = 0.05

    # Total Knightian uncertainty
    total_uncertainty = black_swan_uncertainty + regime_uncertainty + tail_uncertainty
    total_uncertainty = min(0.5, total_uncertainty)  # Cap at 50%

    # Implied accuracy reduction
    base_ceiling = 85  # Information theory maximum
    adjusted_ceiling = base_ceiling * (1 - total_uncertainty)

    return {
        'black_swan_uncertainty': black_swan_uncertainty,
        'regime_uncertainty': regime_uncertainty,
        'tail_uncertainty': tail_uncertainty,
        'total_knightian_uncertainty': total_uncertainty,
        'risk_component': 1 - total_uncertainty,
        'implied_max_accuracy': adjusted_ceiling,
        'accuracy_reduction': base_ceiling - adjusted_ceiling
    }


# Convenience function
def create_tail_risk_features(returns: np.ndarray) -> Dict[str, float]:
    """
    Create all tail risk features for a time series.

    Reference:
        [1-7] See module docstring for full citations

    Args:
        returns: Return series

    Returns:
        Dictionary of tail risk features
    """
    analyzer = TailRiskAnalyzer()
    return analyzer.compute_features(returns)


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    n = 5000

    # Generate heavy-tailed returns (Student-t with df=3)
    heavy_tailed = np.random.standard_t(df=3, size=n) * 0.01

    # Normal returns for comparison
    normal_returns = np.random.randn(n) * 0.01

    print("=" * 60)
    print("TAIL RISK ANALYSIS - HEAVY-TAILED (Student-t, df=3)")
    print("=" * 60)

    analyzer = TailRiskAnalyzer()
    analysis = analyzer.analyze(heavy_tailed)
    print(analysis)

    print("\n" + "=" * 60)
    print("TAIL RISK ANALYSIS - NORMAL")
    print("=" * 60)

    analysis_normal = analyzer.analyze(normal_returns)
    print(analysis_normal)

    print("\n" + "=" * 60)
    print("KNIGHTIAN UNCERTAINTY DECOMPOSITION")
    print("=" * 60)

    uncertainty = calculate_knightian_uncertainty(heavy_tailed)
    for name, value in uncertainty.items():
        print(f"{name:35} {value:.4f}")

    print("\n" + "=" * 60)
    print("FEATURES FOR ML")
    print("=" * 60)

    features = create_tail_risk_features(heavy_tailed)
    for name, value in features.items():
        print(f"{name:35} {value:.6f}")
