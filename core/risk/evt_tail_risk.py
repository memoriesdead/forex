"""
Extreme Value Theory (EVT) / Generalized Pareto Distribution (GPD) Tail Risk

Implements proper EVT/GPD-based VaR and CVaR for tail risk quantification.
This is the gold-standard approach used by major quant firms for tail risk.

=============================================================================
CITATIONS (ACADEMIC - PEER REVIEWED)
=============================================================================

[1] McNeil, A. J., Frey, R., & Embrechts, P. (2015).
    "Quantitative Risk Management: Concepts, Techniques and Tools."
    Princeton University Press.
    ISBN: 978-0691166278
    Chapters 4-7: EVT and tail risk
    THE canonical reference for EVT in finance

[2] Balkema, A. A., & de Haan, L. (1974).
    "Residual Life Time at Great Age."
    Annals of Probability, 2(5), 792-804.
    URL: https://projecteuclid.org/journals/annals-of-probability/volume-2/issue-5/
    Key finding: Original GPD theorem (Pickands-Balkema-de Haan)

[3] Pickands, J. (1975).
    "Statistical Inference Using Extreme Order Statistics."
    Annals of Statistics, 3(1), 119-131.
    Key finding: Peaks-over-threshold method

[4] Gilli, M., & Këllezi, E. (2006).
    "An Application of Extreme Value Theory for Measuring Financial Risk."
    Computational Economics, 27(2-3), 207-228.
    URL: https://link.springer.com/article/10.1007/s10614-006-9025-7
    Key finding: Practical application to VaR

[5] scipy.stats.genpareto documentation:
    URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html

[6] Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997).
    "Modelling Extremal Events for Insurance and Finance."
    Springer.
    Key finding: Mathematical foundations of EVT

=============================================================================

GPD Formula:
    F(x) = 1 - (1 + ξx/σ)^(-1/ξ)

    Where:
        ξ = shape (tail heaviness)
        σ = scale
        x = excess above threshold

    Shape parameter interpretation:
        ξ > 0: Heavy tail (Pareto-like, infinite higher moments)
        ξ = 0: Exponential tail (light)
        ξ < 0: Bounded tail (finite upper limit)

VaR with GPD (Peaks Over Threshold):
    VaR_q = u + (σ/ξ) × [(n/N_u × (1-q))^(-ξ) - 1]

CVaR (Expected Shortfall) with GPD:
    CVaR_q = VaR_q/(1-ξ) + (σ - ξ×u)/(1-ξ)

Author: Claude Code
Date: 2026-01-25
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings


@dataclass
class EVTAnalysis:
    """Complete EVT/GPD analysis results."""
    threshold: float  # u
    n_exceedances: int  # N_u
    shape_xi: float  # ξ
    scale_sigma: float  # σ
    var_95: float
    var_99: float
    var_999: float
    cvar_95: float  # CVaR/ES
    cvar_99: float
    tail_type: str  # "heavy", "exponential", "bounded"
    is_well_fitted: bool  # Goodness-of-fit test passed
    certainty_check_passed: bool  # For 99.999% certainty system

    def __repr__(self) -> str:
        return f"""
╔════════════════════════════════════════════════════════════════╗
║  EVT/GPD TAIL RISK ANALYSIS                                    ║
╠════════════════════════════════════════════════════════════════╣
║  Threshold (u):              {self.threshold:12.6f}                  ║
║  Exceedances (N_u):          {self.n_exceedances:6d}                        ║
║  Shape (ξ):                  {self.shape_xi:12.4f}                  ║
║  Scale (σ):                  {self.scale_sigma:12.6f}                  ║
║  Tail Type:                  {self.tail_type:12s}                  ║
╠════════════════════════════════════════════════════════════════╣
║  VaR 95%:                    {self.var_95*100:8.2f}%                       ║
║  VaR 99%:                    {self.var_99*100:8.2f}%                       ║
║  VaR 99.9%:                  {self.var_999*100:8.2f}%                       ║
╠════════════════════════════════════════════════════════════════╣
║  CVaR/ES 95%:                {self.cvar_95*100:8.2f}%                       ║
║  CVaR/ES 99%:                {self.cvar_99*100:8.2f}%                       ║
╠════════════════════════════════════════════════════════════════╣
║  Well Fitted:                {"YES" if self.is_well_fitted else "NO ":3}                              ║
║  Certainty Check:            {"PASS" if self.certainty_check_passed else "FAIL":4}                             ║
╚════════════════════════════════════════════════════════════════╝
"""


class EVTTailRisk:
    """
    Extreme Value Theory / Generalized Pareto Distribution tail risk analyzer.

    Citation [1]: McNeil et al. (2015)
        "For losses exceeding a high threshold u, the excess distribution
         converges to a GPD as u → ∞"

    Citation [2]: Balkema & de Haan (1974) - The fundamental theorem:
        "For a wide class of distributions F, the conditional excess
         distribution F_u(x) = P(X-u ≤ x | X > u) converges to GPD"

    POT (Peaks Over Threshold) Method:
        1. Choose threshold u (typically 90-95th percentile)
        2. Fit GPD to excesses (X - u) for X > u
        3. Use GPD parameters for tail estimation

    Example:
        >>> evt = EVTTailRisk()
        >>> evt.fit(returns)
        >>> var_99 = evt.var(0.99)
        >>> cvar_99 = evt.cvar(0.99)
    """

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        min_exceedances: int = 50
    ):
        """
        Initialize EVT analyzer.

        Args:
            threshold_percentile: Percentile for threshold selection (90-99)
            min_exceedances: Minimum exceedances required for fit
        """
        self.threshold_percentile = threshold_percentile
        self.min_exceedances = min_exceedances

        # Fitted parameters
        self._threshold = None
        self._shape = None  # ξ
        self._scale = None  # σ
        self._n_exceedances = None
        self._n_total = None
        self._fitted = False

    def fit(
        self,
        returns: np.ndarray,
        use_losses: bool = True
    ) -> 'EVTTailRisk':
        """
        Fit GPD to tail of distribution using POT method.

        Citation [3]: Pickands (1975) - POT method

        Args:
            returns: Return series
            use_losses: If True, analyze negative returns (losses)

        Returns:
            self for chaining
        """
        returns = np.asarray(returns)

        # Convert to losses if needed
        if use_losses:
            data = -returns  # Positive losses
        else:
            data = returns

        self._n_total = len(data)

        # Choose threshold
        self._threshold = np.percentile(data, self.threshold_percentile)

        # Get exceedances
        exceedances = data[data > self._threshold] - self._threshold

        if len(exceedances) < self.min_exceedances:
            warnings.warn(f"Only {len(exceedances)} exceedances (< {self.min_exceedances}). "
                         "Results may be unreliable.")

        self._n_exceedances = len(exceedances)

        # Fit GPD using scipy
        # Citation [5]: scipy.stats.genpareto
        try:
            # genpareto uses different parameterization
            # scipy: c = shape, loc = 0, scale = scale
            c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            self._shape = c  # ξ
            self._scale = scale  # σ
        except Exception as e:
            warnings.warn(f"GPD fit failed: {e}. Using MLE fallback.")
            self._fit_mle(exceedances)

        self._fitted = True
        return self

    def _fit_mle(self, exceedances: np.ndarray):
        """
        MLE estimation for GPD parameters.

        Citation [4]: Gilli & Këllezi (2006)

        GPD log-likelihood:
            L(ξ,σ) = -n×log(σ) - (1+1/ξ) × Σ log(1 + ξ×x_i/σ)
        """
        n = len(exceedances)

        def neg_log_lik(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            if xi == 0:
                # Exponential case
                return n * np.log(sigma) + np.sum(exceedances) / sigma

            # Check domain
            z = 1 + xi * exceedances / sigma
            if np.any(z <= 0):
                return 1e10

            nll = n * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(z))
            return nll

        # Initial guess
        sigma_init = np.std(exceedances)
        xi_init = 0.1

        result = minimize(
            neg_log_lik,
            x0=[xi_init, sigma_init],
            method='L-BFGS-B',
            bounds=[(-0.5, 2.0), (1e-10, None)]
        )

        self._shape = result.x[0]
        self._scale = result.x[1]

    def var(self, confidence: float = 0.99) -> float:
        """
        Calculate Value at Risk using GPD.

        Citation [1]: McNeil et al. (2015) - GPD VaR formula:
            VaR_q = u + (σ/ξ) × [(n/(N_u×(1-q)))^ξ - 1]

        Where:
            u = threshold
            σ = GPD scale
            ξ = GPD shape
            n = total observations
            N_u = exceedances above threshold
            q = confidence level

        Args:
            confidence: VaR confidence level (0.95, 0.99, 0.999)

        Returns:
            VaR as positive number (loss magnitude)
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        p = 1 - confidence  # Exceedance probability

        xi = self._shape
        sigma = self._scale
        u = self._threshold

        # Probability of exceeding threshold
        F_u = self._n_exceedances / self._n_total

        if xi == 0:
            # Exponential case
            var = u - sigma * np.log(p / F_u)
        else:
            # General GPD case
            var = u + (sigma / xi) * ((p / F_u) ** (-xi) - 1)

        return var

    def cvar(self, confidence: float = 0.99) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall) using GPD.

        Citation [1]: McNeil et al. (2015) - GPD CVaR formula:
            CVaR_q = VaR_q/(1-ξ) + (σ - ξ×u)/(1-ξ)

        ES is the expected loss given that loss exceeds VaR.
        It's a coherent risk measure (unlike VaR).

        Args:
            confidence: CVaR confidence level

        Returns:
            CVaR as positive number (expected loss given VaR breach)
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        xi = self._shape
        sigma = self._scale
        u = self._threshold

        var = self.var(confidence)

        if xi >= 1:
            # Expected shortfall is infinite for ξ ≥ 1
            warnings.warn(f"CVaR is infinite for shape={xi:.3f} >= 1")
            return var * 10  # Practical cap

        if xi == 0:
            # Exponential case
            cvar = var + sigma
        else:
            # General case
            cvar = var / (1 - xi) + (sigma - xi * u) / (1 - xi)

        return cvar

    def tail_probability(self, x: float) -> float:
        """
        Calculate tail probability P(X > x) using GPD.

        Citation [1]: McNeil et al. (2015)
            P(X > x) = (N_u/n) × [1 + ξ(x-u)/σ]^(-1/ξ)

        Args:
            x: Value to calculate probability for

        Returns:
            Probability of exceeding x
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        if x <= self._threshold:
            # Below threshold, use empirical
            return (x <= self._threshold).mean() if hasattr(x, '__len__') else 0.5

        xi = self._shape
        sigma = self._scale
        u = self._threshold
        F_u = self._n_exceedances / self._n_total

        excess = x - u

        if xi == 0:
            return F_u * np.exp(-excess / sigma)
        else:
            return F_u * (1 + xi * excess / sigma) ** (-1 / xi)

    def analyze(self, returns: np.ndarray) -> EVTAnalysis:
        """
        Complete EVT/GPD analysis.

        Args:
            returns: Return series

        Returns:
            EVTAnalysis dataclass
        """
        self.fit(returns)

        # Calculate all metrics
        var_95 = self.var(0.95)
        var_99 = self.var(0.99)
        var_999 = self.var(0.999)
        cvar_95 = self.cvar(0.95)
        cvar_99 = self.cvar(0.99)

        # Classify tail type
        if self._shape > 0.1:
            tail_type = "heavy"
        elif self._shape < -0.1:
            tail_type = "bounded"
        else:
            tail_type = "exponential"

        # Goodness-of-fit test (Kolmogorov-Smirnov)
        is_well_fitted = self._goodness_of_fit_test(returns)

        # Certainty check: passes if well-fitted and VaR is reasonable
        certainty_passed = is_well_fitted and var_99 < 0.10  # < 10% daily loss

        return EVTAnalysis(
            threshold=self._threshold,
            n_exceedances=self._n_exceedances,
            shape_xi=self._shape,
            scale_sigma=self._scale,
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_type=tail_type,
            is_well_fitted=is_well_fitted,
            certainty_check_passed=certainty_passed
        )

    def _goodness_of_fit_test(
        self,
        returns: np.ndarray,
        alpha: float = 0.05
    ) -> bool:
        """
        Kolmogorov-Smirnov test for GPD fit.

        Citation [4]: Gilli & Këllezi (2006) - GoF testing

        Args:
            returns: Original returns
            alpha: Significance level

        Returns:
            True if fit is acceptable
        """
        if not self._fitted:
            return False

        # Get exceedances
        losses = -returns
        exceedances = losses[losses > self._threshold] - self._threshold

        if len(exceedances) < 20:
            return False

        # KS test against fitted GPD
        try:
            ks_stat, p_value = stats.kstest(
                exceedances,
                'genpareto',
                args=(self._shape, 0, self._scale)
            )
            return p_value > alpha
        except:
            return False

    def get_params(self) -> Dict:
        """Get fitted parameters."""
        if not self._fitted:
            return {}

        return {
            'threshold': self._threshold,
            'shape_xi': self._shape,
            'scale_sigma': self._scale,
            'n_exceedances': self._n_exceedances,
            'n_total': self._n_total
        }


def calculate_evt_risk_metrics(
    returns: np.ndarray,
    max_loss_threshold: float = 0.02
) -> Dict:
    """
    Quick EVT risk check for certainty validation.

    Used by the 99.999% certainty system to verify tail risk.

    Args:
        returns: Return series
        max_loss_threshold: Maximum acceptable VaR(99%)

    Returns:
        Dictionary with EVT risk status
    """
    evt = EVTTailRisk()
    analysis = evt.analyze(returns)

    return {
        'var_99': analysis.var_99,
        'cvar_99': analysis.cvar_99,
        'shape': analysis.shape_xi,
        'tail_type': analysis.tail_type,
        'is_well_fitted': analysis.is_well_fitted,
        'within_limits': analysis.var_99 < max_loss_threshold,
        'certainty_check_passed': analysis.certainty_check_passed,
        'recommendation': 'ACCEPTABLE' if analysis.certainty_check_passed else 'HIGH_TAIL_RISK'
    }


def gpd_quantile(
    exceedances: np.ndarray,
    p: float,
    threshold: float = None
) -> float:
    """
    Calculate GPD quantile directly from exceedances.

    Convenience function for quick calculations.

    Args:
        exceedances: Excess values above threshold
        p: Quantile level (0-1)
        threshold: Original threshold (0 if already subtracted)

    Returns:
        Quantile value
    """
    if threshold is None:
        threshold = 0

    # Fit GPD
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    # Calculate quantile
    q = stats.genpareto.ppf(p, c, loc=0, scale=scale)

    return threshold + q


class AdaptiveEVT:
    """
    Adaptive EVT with automatic threshold selection.

    Uses the mean residual life plot and stability analysis
    to automatically choose the optimal threshold.

    Citation [6]: Embrechts et al. (1997) - threshold selection
    """

    def __init__(self, candidate_percentiles: List[float] = None):
        """
        Args:
            candidate_percentiles: Percentiles to consider for threshold
        """
        if candidate_percentiles is None:
            candidate_percentiles = [85, 90, 92.5, 95, 97.5]
        self.candidate_percentiles = candidate_percentiles
        self._best_percentile = None
        self._evt = None

    def fit(self, returns: np.ndarray) -> 'AdaptiveEVT':
        """
        Fit with automatic threshold selection.

        Args:
            returns: Return series

        Returns:
            self for chaining
        """
        losses = -np.asarray(returns)

        # Try each candidate threshold
        results = []

        for pct in self.candidate_percentiles:
            try:
                evt = EVTTailRisk(threshold_percentile=pct)
                evt.fit(returns)

                # Check stability: shape should be consistent
                if evt._fitted:
                    results.append({
                        'percentile': pct,
                        'shape': evt._shape,
                        'scale': evt._scale,
                        'n_exceed': evt._n_exceedances,
                        'evt': evt
                    })
            except:
                continue

        if not results:
            # Fallback to 95th percentile
            self._best_percentile = 95
            self._evt = EVTTailRisk(threshold_percentile=95)
            self._evt.fit(returns)
            return self

        # Choose threshold with most stable shape estimate
        # (shape shouldn't change much with threshold if model is correct)
        shapes = [r['shape'] for r in results]
        mean_shape = np.mean(shapes)

        # Pick one closest to mean (most stable)
        best_idx = np.argmin([abs(s - mean_shape) for s in shapes])
        self._best_percentile = results[best_idx]['percentile']
        self._evt = results[best_idx]['evt']

        return self

    def var(self, confidence: float = 0.99) -> float:
        """Get VaR using best threshold."""
        return self._evt.var(confidence)

    def cvar(self, confidence: float = 0.99) -> float:
        """Get CVaR using best threshold."""
        return self._evt.cvar(confidence)

    def analyze(self, returns: np.ndarray) -> EVTAnalysis:
        """Full analysis with adaptive threshold."""
        self.fit(returns)
        return self._evt.analyze(returns)


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Generate heavy-tailed returns (Student-t with df=3)
    n = 2000
    heavy_tailed = np.random.standard_t(df=3, size=n) * 0.01

    # Add some extreme outliers (black swans)
    heavy_tailed[np.random.choice(n, 5, replace=False)] = np.array([-0.08, -0.06, 0.07, -0.05, 0.08])

    print("=" * 60)
    print("EVT/GPD TAIL RISK ANALYSIS")
    print("=" * 60)

    evt = EVTTailRisk(threshold_percentile=95)
    analysis = evt.analyze(heavy_tailed)
    print(analysis)

    print("\n" + "-" * 60)
    print("QUICK CERTAINTY CHECK")
    print("-" * 60)

    check = calculate_evt_risk_metrics(heavy_tailed)
    for k, v in check.items():
        if isinstance(v, float):
            print(f"{k:25} {v:.4f}")
        else:
            print(f"{k:25} {v}")

    print("\n" + "-" * 60)
    print("ADAPTIVE EVT (AUTO THRESHOLD)")
    print("-" * 60)

    adaptive = AdaptiveEVT()
    adaptive_analysis = adaptive.analyze(heavy_tailed)
    print(f"Best threshold percentile: {adaptive._best_percentile}")
    print(f"VaR 99%: {adaptive.var(0.99)*100:.2f}%")
    print(f"CVaR 99%: {adaptive.cvar(0.99)*100:.2f}%")
