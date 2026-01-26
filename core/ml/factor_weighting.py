"""
Factor Weighting Paradigm for HFT

Equal-weight vs momentum-based factor weighting strategies.
Based on BigQuant 2024/25 research showing equal-weight often beats optimization.

Academic Citations:
- DeMiguel, Garlappi & Uppal (2009): "Optimal Versus Naive Diversification"
  Review of Financial Studies - 1/N beats optimization

- Kirby & Ostdiek (2012): "It's All in the Timing"
  Journal of Financial Economics - Timing vs estimation error

- arXiv:2305.15910 (2023): "Simple Factor Weighting in Asset Pricing"
  Equal-weight robustness across markets

- BigQuant (2024): "等权 vs 优化权重：因子组合的实证研究"
  等权因子优于复杂优化 (equal-weight beats complex optimization)

Chinese Quant Application:
- BigQuant (2024/25): 等权因子组合研究
- 华泰证券: 因子权重优化方法比较
- 国泰君安: 因子组合配置策略
- 中信证券: 多因子模型权重设计

The Key Insight:
    Complex factor weight optimization often UNDERPERFORMS simple equal-weight!

    Why?
    1. Estimation error in covariance matrices
    2. Overfitting to historical data
    3. Transaction costs from frequent rebalancing

    Equal-weight gives:
    - No estimation error
    - Maximum diversification
    - Minimum turnover

    This is why Renaissance might use simpler weighting than you think.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class WeightingResult:
    """Result of factor weighting calculation."""

    weights: np.ndarray
    weight_dict: Dict[str, float]

    # Performance metrics (if backtested)
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Method used
    method: str = "equal_weight"
    estimation_error: float = 0.0


@dataclass
class FactorPerformance:
    """Performance metrics for a single factor."""

    factor_name: str
    ic: float  # Information coefficient
    icir: float  # IC information ratio
    returns: float  # Factor returns
    turnover: float  # Turnover rate
    sharpe: float  # Sharpe ratio


class EqualWeightCombiner:
    """
    Equal-weight factor combination (1/N rule).

    Reference:
        DeMiguel et al. (2009): "Optimal Versus Naive Diversification"
        Shows 1/N beats mean-variance in many settings

    BigQuant (2024):
        "等权因子组合在A股市场表现优于大多数优化方法"
        (Equal-weight beats most optimization in A-share market)
    """

    def __init__(
        self,
        factor_names: List[str],
        min_weight: float = 0.0,  # Minimum weight per factor
        max_weight: float = 1.0,  # Maximum weight per factor
    ):
        """
        Initialize equal-weight combiner.

        Args:
            factor_names: Names of factors
            min_weight: Minimum weight (for constraints)
            max_weight: Maximum weight (for constraints)
        """
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_weights(self) -> WeightingResult:
        """
        Compute equal weights.

        Returns:
            WeightingResult with 1/N weights
        """
        weights = np.ones(self.n_factors) / self.n_factors

        weight_dict = {name: w for name, w in zip(self.factor_names, weights)}

        return WeightingResult(
            weights=weights,
            weight_dict=weight_dict,
            method="equal_weight",
            estimation_error=0.0,  # No estimation error!
        )


class ICWeightedCombiner:
    """
    IC-weighted factor combination.

    Weights factors by their Information Coefficient.
    Simple but accounts for factor quality.

    Reference:
        国泰君安: 基于IC的因子权重配置
    """

    def __init__(
        self,
        factor_names: List[str],
        lookback_window: int = 20,
        decay_factor: float = 0.9,  # Exponential decay for recent IC
    ):
        """
        Initialize IC-weighted combiner.

        Args:
            factor_names: Names of factors
            lookback_window: Window for IC calculation
            decay_factor: Decay for exponential weighting
        """
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        self.lookback = lookback_window
        self.decay = decay_factor

        self._ic_history: Dict[str, List[float]] = {
            name: [] for name in factor_names
        }

    def record_ic(
        self,
        factor_name: str,
        ic: float,
    ) -> None:
        """
        Record IC observation for a factor.

        Args:
            factor_name: Factor name
            ic: Information coefficient
        """
        if factor_name in self._ic_history:
            self._ic_history[factor_name].append(ic)
            # Trim to window
            if len(self._ic_history[factor_name]) > self.lookback * 2:
                self._ic_history[factor_name] = self._ic_history[factor_name][-self.lookback:]

    def compute_weights(self) -> WeightingResult:
        """
        Compute IC-weighted weights.

        Returns:
            WeightingResult with IC-proportional weights
        """
        # Compute average IC for each factor
        avg_ics = []
        for name in self.factor_names:
            history = self._ic_history[name]
            if len(history) > 0:
                # Exponential weighted average
                weights = self.decay ** np.arange(len(history))[::-1]
                avg_ic = np.average(history, weights=weights)
            else:
                avg_ic = 0.0
            avg_ics.append(max(avg_ic, 0))  # Only positive IC

        avg_ics = np.array(avg_ics)

        # Normalize to weights
        total = np.sum(avg_ics)
        if total > 0:
            weights = avg_ics / total
        else:
            weights = np.ones(self.n_factors) / self.n_factors

        weight_dict = {name: w for name, w in zip(self.factor_names, weights)}

        # Estimation error based on IC variance
        ic_stds = []
        for name in self.factor_names:
            history = self._ic_history[name]
            if len(history) > 5:
                ic_stds.append(np.std(history))
        estimation_error = np.mean(ic_stds) if ic_stds else 0.1

        return WeightingResult(
            weights=weights,
            weight_dict=weight_dict,
            method="ic_weighted",
            estimation_error=estimation_error,
        )


class MomentumWeightedCombiner:
    """
    Momentum-weighted factor combination.

    Weights factors by recent performance (factor momentum).

    Reference:
        Kirby & Ostdiek (2012): Factor timing
        BigQuant (2024): 因子动量效应研究
    """

    def __init__(
        self,
        factor_names: List[str],
        lookback: int = 20,
        min_weight: float = 0.05,  # Floor on weights
    ):
        """
        Initialize momentum-weighted combiner.

        Args:
            factor_names: Names of factors
            lookback: Lookback window for momentum
            min_weight: Minimum weight floor
        """
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        self.lookback = lookback
        self.min_weight = min_weight

        self._return_history: Dict[str, List[float]] = {
            name: [] for name in factor_names
        }

    def record_return(
        self,
        factor_name: str,
        factor_return: float,
    ) -> None:
        """
        Record return for a factor.

        Args:
            factor_name: Factor name
            factor_return: Factor return
        """
        if factor_name in self._return_history:
            self._return_history[factor_name].append(factor_return)
            if len(self._return_history[factor_name]) > self.lookback * 2:
                self._return_history[factor_name] = self._return_history[factor_name][-self.lookback:]

    def compute_weights(self) -> WeightingResult:
        """
        Compute momentum-weighted weights.

        Returns:
            WeightingResult with momentum-based weights
        """
        # Compute momentum (cumulative return) for each factor
        momentums = []
        for name in self.factor_names:
            history = self._return_history[name]
            if len(history) > 0:
                # Use recent returns
                recent = history[-min(self.lookback, len(history)):]
                momentum = np.sum(recent)
            else:
                momentum = 0.0
            momentums.append(momentum)

        momentums = np.array(momentums)

        # Softmax to convert to weights
        # Shift to positive for softmax
        shifted = momentums - np.max(momentums)
        exp_mom = np.exp(shifted)
        weights = exp_mom / np.sum(exp_mom)

        # Apply floor
        weights = np.maximum(weights, self.min_weight)
        weights = weights / np.sum(weights)

        weight_dict = {name: w for name, w in zip(self.factor_names, weights)}

        return WeightingResult(
            weights=weights,
            weight_dict=weight_dict,
            method="momentum_weighted",
            estimation_error=0.05,  # Some estimation error
        )


class RiskParityCombiner:
    """
    Risk parity factor combination.

    Equal risk contribution from each factor.

    Reference:
        Maillard, Roncalli, Teiletche (2010): "On the properties of equally-weighted risk contribution portfolios"
    """

    def __init__(
        self,
        factor_names: List[str],
        lookback: int = 50,
    ):
        """
        Initialize risk parity combiner.

        Args:
            factor_names: Names of factors
            lookback: Lookback for volatility estimation
        """
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        self.lookback = lookback

        self._return_history: Dict[str, List[float]] = {
            name: [] for name in factor_names
        }

    def record_return(
        self,
        factor_name: str,
        factor_return: float,
    ) -> None:
        """Record factor return."""
        if factor_name in self._return_history:
            self._return_history[factor_name].append(factor_return)
            if len(self._return_history[factor_name]) > self.lookback * 2:
                self._return_history[factor_name] = self._return_history[factor_name][-self.lookback:]

    def compute_weights(self) -> WeightingResult:
        """
        Compute risk parity weights.

        Returns:
            WeightingResult with risk parity weights
        """
        # Estimate volatility for each factor
        vols = []
        for name in self.factor_names:
            history = self._return_history[name]
            if len(history) > 10:
                vol = np.std(history[-self.lookback:])
            else:
                vol = 0.01  # Default vol
            vols.append(max(vol, 0.001))

        vols = np.array(vols)

        # Inverse volatility weighting (simplified risk parity)
        inv_vols = 1 / vols
        weights = inv_vols / np.sum(inv_vols)

        weight_dict = {name: w for name, w in zip(self.factor_names, weights)}

        # Estimation error from vol estimation
        estimation_error = np.std(vols) / np.mean(vols)

        return WeightingResult(
            weights=weights,
            weight_dict=weight_dict,
            method="risk_parity",
            estimation_error=estimation_error,
        )


class AdaptiveWeightCombiner:
    """
    Adaptive factor weighting that switches between methods.

    Uses equal-weight by default, switches to IC-weighted
    when IC estimates are stable.

    Reference:
        BigQuant (2025): 自适应因子权重 (adaptive factor weights)
    """

    def __init__(
        self,
        factor_names: List[str],
        ic_stability_threshold: float = 0.3,  # Max IC std to use IC weights
        min_observations: int = 20,
    ):
        """
        Initialize adaptive combiner.

        Args:
            factor_names: Factor names
            ic_stability_threshold: Max IC volatility to use IC weights
            min_observations: Minimum observations before adapting
        """
        self.factor_names = factor_names
        self.ic_threshold = ic_stability_threshold
        self.min_obs = min_observations

        self.equal_combiner = EqualWeightCombiner(factor_names)
        self.ic_combiner = ICWeightedCombiner(factor_names)
        self.momentum_combiner = MomentumWeightedCombiner(factor_names)

        self._observation_count = 0

    def record_observation(
        self,
        factor_ics: Dict[str, float],
        factor_returns: Dict[str, float],
    ) -> None:
        """
        Record IC and return observations.

        Args:
            factor_ics: Dict of factor -> IC
            factor_returns: Dict of factor -> return
        """
        for name, ic in factor_ics.items():
            self.ic_combiner.record_ic(name, ic)

        for name, ret in factor_returns.items():
            self.momentum_combiner.record_return(name, ret)

        self._observation_count += 1

    def compute_weights(self) -> WeightingResult:
        """
        Compute adaptive weights.

        Returns:
            WeightingResult from best method
        """
        if self._observation_count < self.min_obs:
            # Not enough data - use equal weight
            return self.equal_combiner.compute_weights()

        # Check IC stability
        ic_result = self.ic_combiner.compute_weights()

        if ic_result.estimation_error < self.ic_threshold:
            # IC estimates are stable - use IC weights
            return ic_result
        else:
            # IC too unstable - fall back to equal weight
            return self.equal_combiner.compute_weights()


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_equal_weight(n_factors: int) -> np.ndarray:
    """
    Ultra-fast equal weight computation.

    Args:
        n_factors: Number of factors

    Returns:
        Equal weight array
    """
    return np.ones(n_factors) / n_factors


def quick_ic_weight(
    ics: np.ndarray,
    min_weight: float = 0.05,
) -> np.ndarray:
    """
    Fast IC-based weighting.

    Args:
        ics: Array of IC values per factor
        min_weight: Minimum weight floor

    Returns:
        IC-weighted array
    """
    # Only positive IC
    positive_ics = np.maximum(ics, 0)

    total = np.sum(positive_ics)
    if total > 0:
        weights = positive_ics / total
    else:
        weights = np.ones(len(ics)) / len(ics)

    # Apply floor
    weights = np.maximum(weights, min_weight)
    return weights / np.sum(weights)


def compare_weighting_methods(
    factor_returns: np.ndarray,  # (n_periods, n_factors)
    factor_ics: np.ndarray,  # (n_periods, n_factors)
) -> Dict[str, float]:
    """
    Compare performance of different weighting methods.

    Args:
        factor_returns: Return matrix
        factor_ics: IC matrix

    Returns:
        Dict of method -> Sharpe ratio
    """
    n_periods, n_factors = factor_returns.shape
    results = {}

    # Equal weight
    ew_weights = np.ones(n_factors) / n_factors
    ew_portfolio_returns = np.sum(factor_returns * ew_weights, axis=1)
    if np.std(ew_portfolio_returns) > 0:
        results['equal_weight'] = np.mean(ew_portfolio_returns) / np.std(ew_portfolio_returns) * np.sqrt(252)
    else:
        results['equal_weight'] = 0

    # IC-weighted (using average IC)
    avg_ics = np.mean(factor_ics, axis=0)
    ic_weights = quick_ic_weight(avg_ics)
    ic_portfolio_returns = np.sum(factor_returns * ic_weights, axis=1)
    if np.std(ic_portfolio_returns) > 0:
        results['ic_weighted'] = np.mean(ic_portfolio_returns) / np.std(ic_portfolio_returns) * np.sqrt(252)
    else:
        results['ic_weighted'] = 0

    # Inverse volatility
    factor_vols = np.std(factor_returns, axis=0)
    if np.all(factor_vols > 0):
        vol_weights = (1 / factor_vols) / np.sum(1 / factor_vols)
        vol_portfolio_returns = np.sum(factor_returns * vol_weights, axis=1)
        if np.std(vol_portfolio_returns) > 0:
            results['inverse_vol'] = np.mean(vol_portfolio_returns) / np.std(vol_portfolio_returns) * np.sqrt(252)
        else:
            results['inverse_vol'] = 0
    else:
        results['inverse_vol'] = 0

    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FACTOR WEIGHTING PARADIGM")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - DeMiguel et al. (2009): 1/N beats optimization")
    print("  - BigQuant (2024): 等权因子优于复杂优化")
    print("  - 华泰证券: 因子权重优化方法比较")
    print()

    # Simulated factor data
    np.random.seed(42)
    factor_names = ['momentum', 'value', 'volatility', 'microstructure', 'sentiment']
    n_factors = len(factor_names)
    n_periods = 100

    # Simulated returns with different alphas
    alphas = [0.001, 0.0005, 0.0008, 0.0012, 0.0003]
    factor_returns = np.random.randn(n_periods, n_factors) * 0.01
    for i, alpha in enumerate(alphas):
        factor_returns[:, i] += alpha

    # Simulated ICs
    factor_ics = 0.05 + 0.02 * np.random.randn(n_periods, n_factors)

    # Compare methods
    print("COMPARISON OF WEIGHTING METHODS")
    print("-" * 50)

    comparison = compare_weighting_methods(factor_returns, factor_ics)

    for method, sharpe in sorted(comparison.items(), key=lambda x: -x[1]):
        print(f"  {method}: Sharpe = {sharpe:.2f}")

    print()

    # Equal weight
    print("EQUAL WEIGHT (1/N) APPROACH")
    print("-" * 50)

    ew_combiner = EqualWeightCombiner(factor_names)
    ew_result = ew_combiner.compute_weights()

    print(f"Weights:")
    for name, w in ew_result.weight_dict.items():
        print(f"  {name}: {w:.1%}")
    print(f"Estimation error: {ew_result.estimation_error:.4f}")
    print()

    # IC-weighted
    print("IC-WEIGHTED APPROACH")
    print("-" * 50)

    ic_combiner = ICWeightedCombiner(factor_names)

    # Record some IC observations
    for t in range(50):
        for i, name in enumerate(factor_names):
            ic_combiner.record_ic(name, factor_ics[t, i])

    ic_result = ic_combiner.compute_weights()

    print(f"Weights:")
    for name, w in ic_result.weight_dict.items():
        print(f"  {name}: {w:.1%}")
    print(f"Estimation error: {ic_result.estimation_error:.4f}")
    print()

    print("=" * 70)
    print("KEY INSIGHT (BigQuant 2024):")
    print("  Equal-weight often BEATS complex optimization!")
    print()
    print("  Why? Because estimation error in optimization")
    print("  outweighs any theoretical optimal gains.")
    print()
    print("  'It is better to be roughly right than precisely wrong.'")
    print("  - Keynes (and Renaissance, probably)")
    print("=" * 70)
