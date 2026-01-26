"""
Reinforcement Learning Research Module for Forex Trading

USA Quant-Level Peer-Reviewed RL Methods with Full Academic Citations.

══════════════════════════════════════════════════════════════════════════════
ACADEMIC CITATIONS (22 Papers):
══════════════════════════════════════════════════════════════════════════════

FOUNDATIONAL RL ALGORITHMS:
1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   "Proximal Policy Optimization Algorithms." arXiv:1707.06347. OpenAI.

2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
   "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor."
   ICML 2018. UC Berkeley.

3. Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015).
   "Continuous Control with Deep Reinforcement Learning." arXiv:1509.02971. DeepMind.

4. Fujimoto, S., Hoof, H., & Meger, D. (2018).
   "Addressing Function Approximation Error in Actor-Critic Methods." ICML 2018. McGill.

REWARD SHAPING:
5. Moody, J., & Saffell, M. (2001).
   "Learning to Trade via Direct Reinforcement."
   IEEE Transactions on Neural Networks, 12(4), 875-889. [1,400+ citations]

6. Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998).
   "Performance Functions and Reinforcement Learning for Trading Systems."
   Journal of Forecasting, 17(5-6), 441-470.

RISK-SENSITIVE RL:
7. Tamar, A., Glassner, Y., & Mannor, S. (2015).
   "Optimizing the CVaR via Sampling." AAAI 2015.

8. Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2017).
   "Risk-Constrained Reinforcement Learning with Percentile Risk Criteria."
   Journal of Machine Learning Research, 18(1), 6070-6120.

9. Greenberg, I., Chow, Y., & Ghavamzadeh, M. (2024).
   "A Reductions Approach to Risk-Sensitive RL with Optimized Certainty Equivalents."
   arXiv:2403.06323.

10. Lim, S. H., & Malik, A. (2024).
    "Risk-Seeking RL via Multi-Timescale EVaR Optimization." OpenReview.

PORTFOLIO OPTIMIZATION:
11. Zhang, Z., Zohren, S., & Roberts, S. (2020).
    "Deep Reinforcement Learning for Trading." Journal of Financial Data Science.

12. Jiang, Z., Xu, D., & Liang, J. (2017).
    "A Deep Reinforcement Learning Framework for the Financial Portfolio Management."
    arXiv:1706.10059. [800+ citations]

13. Yang, H., Liu, X. Y., Zhong, S., & Walid, A. (2020).
    "Deep Reinforcement Learning for Automated Stock Trading."
    ACM ICAIF 2020. [FinRL Framework]

14. International Journal of Computational Intelligence Systems (2025).
    "Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization."
    Springer. DOI: 10.1007/s44196-025-00875-8

DISTRIBUTIONAL RL:
15. Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018).
    "Distributional Reinforcement Learning with Quantile Regression." AAAI 2018.

16. Dabney, W., Ostrovski, G., Silver, D., & Munos, R. (2018).
    "Implicit Quantile Networks for Distributional RL." ICML 2018. DeepMind.

17. Zhou, F., et al. (2020).
    "Non-Crossing Quantile Regression for Distributional RL." NeurIPS 2020.

OPTIMAL EXECUTION:
18. Almgren, R., & Chriss, N. (2001).
    "Optimal Execution of Portfolio Transactions."
    Journal of Risk, 3(2), 5-39. [2,500+ citations]

19. Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006).
    "Reinforcement Learning for Optimized Trade Execution." ICML 2006.

MARKET MAKING:
20. Avellaneda, M., & Stoikov, S. (2008).
    "High-frequency Trading in a Limit Order Book."
    Quantitative Finance, 8(3), 217-224. [1,800+ citations]

21. Spooner, T., Fearnley, J., Savani, R., & Koutsoupias, E. (2018).
    "Market Making via Reinforcement Learning." AAMAS 2018.

META-RL:
22. Finn, C., Abbeel, P., & Levine, S. (2017).
    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
    ICML 2017. [10,000+ citations]

══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import Enum
import warnings

try:
    from scipy.optimize import minimize_scalar, minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, some functions will be limited")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: REWARD SHAPING
# ══════════════════════════════════════════════════════════════════════════════

class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio for Online RL Trading.

    Reference:
        Moody, J., & Saffell, M. (2001).
        "Learning to Trade via Direct Reinforcement."
        IEEE Transactions on Neural Networks, 12(4), 875-889.

    The DSR provides an incremental update to the Sharpe ratio that can be
    used as a reward signal in reinforcement learning. Unlike batch Sharpe,
    it can be computed online as new returns arrive.

    Mathematical Formulation:
        A_t = A_{t-1} + η(R_t - A_{t-1})     [EMA of returns]
        B_t = B_{t-1} + η(R_t² - B_{t-1})   [EMA of squared returns]

        D_t = (B_{t-1} · ΔA_t - 0.5 · A_{t-1} · ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

    Where:
        η = adaptation rate (typically 0.01-0.1)
        R_t = return at time t
        D_t = differential Sharpe ratio at time t
    """

    def __init__(self, eta: float = 0.01):
        """
        Initialize Differential Sharpe Ratio calculator.

        Args:
            eta: Adaptation rate for exponential moving averages (0.01-0.1)
                 Lower = more stable, slower adaptation
                 Higher = faster adaptation, more noise
        """
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self._initialized = False

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self.A = 0.0
        self.B = 0.0
        self._initialized = False

    def update(self, return_t: float) -> float:
        """
        Compute differential Sharpe ratio for new return.

        Args:
            return_t: Return at current timestep (can be log return or simple)

        Returns:
            Differential Sharpe ratio (reward signal)
        """
        if not self._initialized:
            self.A = return_t
            self.B = return_t ** 2
            self._initialized = True
            return 0.0

        # Compute deltas
        delta_A = return_t - self.A
        delta_B = return_t ** 2 - self.B

        # Compute variance term (avoid division by zero)
        var_term = self.B - self.A ** 2

        if var_term > 1e-10:
            # Differential Sharpe formula (Moody & Saffell 2001, Eq. 7)
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (var_term ** 1.5)
        else:
            dsr = 0.0

        # Update EMAs
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return dsr

    def compute_batch(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute DSR for a batch of returns.

        Args:
            returns: Array of returns

        Returns:
            Array of differential Sharpe ratios
        """
        self.reset()
        dsr_values = np.zeros(len(returns))
        for i, r in enumerate(returns):
            dsr_values[i] = self.update(r)
        return dsr_values

    @staticmethod
    def compute_static(returns: np.ndarray, eta: float = 0.01) -> float:
        """
        Static method to compute final DSR value.

        Args:
            returns: Array of returns
            eta: Adaptation rate

        Returns:
            Final cumulative differential Sharpe ratio
        """
        dsr = DifferentialSharpeRatio(eta=eta)
        total_dsr = 0.0
        for r in returns:
            total_dsr += dsr.update(r)
        return total_dsr


class DifferentialDownsideDeviation:
    """
    Differential Downside Deviation for Sortino-like RL rewards.

    Reference:
        Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998).
        "Performance Functions and Reinforcement Learning for Trading Systems."
        Journal of Forecasting, 17(5-6), 441-470.

    Extension of DSR that only penalizes downside volatility,
    aligned with Sortino ratio philosophy.
    """

    def __init__(self, eta: float = 0.01, mar: float = 0.0):
        """
        Args:
            eta: Adaptation rate
            mar: Minimum Acceptable Return (threshold for downside)
        """
        self.eta = eta
        self.mar = mar
        self.A = 0.0  # EMA of returns
        self.D = 0.0  # EMA of downside squared returns
        self._initialized = False

    def reset(self) -> None:
        self.A = 0.0
        self.D = 0.0
        self._initialized = False

    def update(self, return_t: float) -> float:
        """Compute differential downside deviation reward."""
        if not self._initialized:
            self.A = return_t
            downside = min(0, return_t - self.mar) ** 2
            self.D = downside
            self._initialized = True
            return 0.0

        delta_A = return_t - self.A
        downside = min(0, return_t - self.mar) ** 2
        delta_D = downside - self.D

        if self.D > 1e-10:
            # Sortino-like differential reward
            ddd = (np.sqrt(self.D) * delta_A - 0.5 * self.A * delta_D / np.sqrt(self.D)) / self.D
        else:
            ddd = delta_A  # No downside yet, just use return

        self.A += self.eta * delta_A
        self.D += self.eta * delta_D

        return ddd


class MaxDrawdownPenalty:
    """
    Maximum Drawdown Penalty for RL Reward Shaping.

    Reference:
        International Journal of Computational Intelligence Systems (2025).
        "Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization."

    Penalizes the agent when portfolio value drops from peak,
    encouraging drawdown-aware trading strategies.
    """

    def __init__(self, penalty_scale: float = 1.0):
        """
        Args:
            penalty_scale: Multiplier for drawdown penalty (higher = more risk-averse)
        """
        self.penalty_scale = penalty_scale
        self.peak_value = 1.0
        self.current_value = 1.0

    def reset(self, initial_value: float = 1.0) -> None:
        self.peak_value = initial_value
        self.current_value = initial_value

    def update(self, return_t: float) -> float:
        """
        Compute drawdown-penalized reward.

        Args:
            return_t: Return at current timestep

        Returns:
            Return minus drawdown penalty
        """
        # Update portfolio value
        self.current_value *= (1 + return_t)

        # Update peak
        if self.current_value > self.peak_value:
            self.peak_value = self.current_value

        # Compute drawdown
        drawdown = (self.peak_value - self.current_value) / self.peak_value

        # Penalized reward
        penalty = self.penalty_scale * drawdown
        return return_t - penalty

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        return (self.peak_value - self.current_value) / self.peak_value


@dataclass
class RewardConfig:
    """Configuration for combined reward function."""
    use_dsr: bool = True
    use_sortino: bool = False
    use_drawdown_penalty: bool = True
    dsr_weight: float = 1.0
    sortino_weight: float = 1.0
    drawdown_weight: float = 0.5
    eta: float = 0.01
    drawdown_scale: float = 2.0


class CombinedRewardFunction:
    """
    Combined Reward Function for Trading RL.

    Combines multiple reward signals:
    - Differential Sharpe Ratio (Moody & Saffell 2001)
    - Differential Downside Deviation (Sortino-like)
    - Maximum Drawdown Penalty

    Reference:
        International Journal of Computational Intelligence Systems (2025).
        "Risk-Adjusted Deep Reinforcement Learning for Portfolio Optimization:
        A Multi-reward Approach."
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

        self.dsr = DifferentialSharpeRatio(eta=self.config.eta)
        self.ddd = DifferentialDownsideDeviation(eta=self.config.eta)
        self.mdd = MaxDrawdownPenalty(penalty_scale=self.config.drawdown_scale)

    def reset(self) -> None:
        self.dsr.reset()
        self.ddd.reset()
        self.mdd.reset()

    def compute_reward(self, return_t: float) -> float:
        """
        Compute combined reward from multiple signals.

        Args:
            return_t: Return at current timestep

        Returns:
            Combined reward signal
        """
        reward = 0.0

        if self.config.use_dsr:
            reward += self.config.dsr_weight * self.dsr.update(return_t)

        if self.config.use_sortino:
            reward += self.config.sortino_weight * self.ddd.update(return_t)

        if self.config.use_drawdown_penalty:
            reward += self.config.drawdown_weight * self.mdd.update(return_t)

        return reward


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: RISK-SENSITIVE RL (CVaR, VaR, EVaR)
# ══════════════════════════════════════════════════════════════════════════════

class ValueAtRisk:
    """
    Value at Risk (VaR) Calculator.

    Reference:
        Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing
        Financial Risk." McGraw-Hill.

    VaR_α = inf{x : P(X ≤ x) ≥ α}

    Interpretation: The maximum loss at confidence level (1-α)
    """

    @staticmethod
    def historical(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Historical VaR using empirical quantile.

        Args:
            returns: Array of returns (can be negative for losses)
            alpha: Significance level (0.05 = 95% confidence)

        Returns:
            VaR value (will be negative for losses)
        """
        return np.percentile(returns, alpha * 100)

    @staticmethod
    def parametric(mean: float, std: float, alpha: float = 0.05) -> float:
        """
        Parametric VaR assuming normal distribution.

        Args:
            mean: Expected return
            std: Standard deviation of returns
            alpha: Significance level

        Returns:
            VaR value
        """
        if not SCIPY_AVAILABLE:
            # Approximate z-scores for common alpha values
            z_scores = {0.01: -2.326, 0.05: -1.645, 0.10: -1.282}
            z = z_scores.get(alpha, -1.645)
        else:
            z = norm.ppf(alpha)
        return mean + z * std

    @staticmethod
    def cornish_fisher(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Cornish-Fisher VaR with skewness and kurtosis adjustment.

        More accurate for non-normal return distributions (fat tails).

        Args:
            returns: Array of returns
            alpha: Significance level

        Returns:
            Adjusted VaR value
        """
        if not SCIPY_AVAILABLE:
            return ValueAtRisk.historical(returns, alpha)

        mean = np.mean(returns)
        std = np.std(returns)
        skew = ((returns - mean) ** 3).mean() / std ** 3
        kurt = ((returns - mean) ** 4).mean() / std ** 4 - 3  # Excess kurtosis

        z = norm.ppf(alpha)

        # Cornish-Fisher expansion
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        return mean + z_cf * std


class ConditionalValueAtRisk:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall.

    References:
        1. Rockafellar, R. T., & Uryasev, S. (2000).
           "Optimization of Conditional Value-at-Risk."
           Journal of Risk, 2(3), 21-42.

        2. Tamar, A., Glassner, Y., & Mannor, S. (2015).
           "Optimizing the CVaR via Sampling." AAAI 2015.

    CVaR_α = E[X | X ≤ VaR_α]

    Interpretation: Expected loss given we're in worst α% of scenarios.
    CVaR is coherent (satisfies subadditivity) unlike VaR.
    """

    @staticmethod
    def historical(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        Historical CVaR using empirical distribution.

        Args:
            returns: Array of returns
            alpha: Significance level (0.05 = worst 5%)

        Returns:
            CVaR value (expected loss in tail)
        """
        var = np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var
        return tail_returns.mean()

    @staticmethod
    def parametric(mean: float, std: float, alpha: float = 0.05) -> float:
        """
        Parametric CVaR assuming normal distribution.

        Args:
            mean: Expected return
            std: Standard deviation
            alpha: Significance level

        Returns:
            CVaR value
        """
        if not SCIPY_AVAILABLE:
            # Approximate for normal distribution
            # CVaR ≈ mean - std * phi(z) / alpha where z = Phi^{-1}(alpha)
            z_scores = {0.01: -2.326, 0.05: -1.645, 0.10: -1.282}
            phi_values = {0.01: 0.0267, 0.05: 0.1031, 0.10: 0.1755}
            z = z_scores.get(alpha, -1.645)
            phi_z = phi_values.get(alpha, 0.1031)
            return mean - std * phi_z / alpha

        z = norm.ppf(alpha)
        phi_z = norm.pdf(z)
        return mean - std * phi_z / alpha

    @staticmethod
    def from_quantiles(quantiles: np.ndarray, probs: np.ndarray, alpha: float = 0.05) -> float:
        """
        Compute CVaR from quantile distribution (for Distributional RL).

        Args:
            quantiles: Quantile values from distributional RL
            probs: Probability weights for each quantile
            alpha: Significance level

        Returns:
            CVaR value
        """
        # Sort by quantile value
        sorted_idx = np.argsort(quantiles)
        sorted_q = quantiles[sorted_idx]
        sorted_p = probs[sorted_idx]

        # Accumulate until alpha
        cum_prob = 0.0
        cvar = 0.0

        for q, p in zip(sorted_q, sorted_p):
            if cum_prob + p <= alpha:
                cvar += q * p
                cum_prob += p
            else:
                # Partial weight for last quantile
                remaining = alpha - cum_prob
                cvar += q * remaining
                break

        return cvar / alpha if alpha > 0 else 0.0


class EntropicValueAtRisk:
    """
    Entropic Value at Risk (EVaR).

    Reference:
        Lim, S. H., & Malik, A. (2024).
        "Risk-Seeking RL via Multi-Timescale EVaR Optimization."
        OpenReview.

    EVaR is tighter than CVaR and has better mathematical properties.

    EVaR_α = inf_{t>0} { (1/t) · log(E[exp(t·X)]) + (1/t)·log(1/α) }

    EVaR ≥ CVaR ≥ VaR (EVaR is most conservative)
    """

    @staticmethod
    def compute(returns: np.ndarray, alpha: float = 0.05,
                t_range: Tuple[float, float] = (1e-6, 100)) -> float:
        """
        Compute EVaR via optimization.

        Args:
            returns: Array of returns (losses should be negative)
            alpha: Significance level
            t_range: Range for optimization parameter t

        Returns:
            EVaR value
        """
        if not SCIPY_AVAILABLE:
            # Fall back to CVaR
            return ConditionalValueAtRisk.historical(returns, alpha)

        def objective(t):
            if t <= 0:
                return np.inf
            try:
                moment = np.mean(np.exp(t * returns))
                if moment <= 0:
                    return np.inf
                return (1/t) * np.log(moment) + (1/t) * np.log(1/alpha)
            except (OverflowError, RuntimeWarning):
                return np.inf

        result = minimize_scalar(objective, bounds=t_range, method='bounded')
        return result.fun if result.success else ConditionalValueAtRisk.historical(returns, alpha)


class CVaRConstrainedKelly:
    """
    Kelly Criterion with CVaR Constraint.

    References:
        1. Kelly, J. L. (1956). "A New Interpretation of Information Rate."
           Bell System Technical Journal.

        2. Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2017).
           "Risk-Constrained RL with Percentile Risk Criteria."
           JMLR, 18(1), 6070-6120.

    Combines optimal growth (Kelly) with tail risk constraint (CVaR).
    """

    def __init__(self, cvar_limit: float = -0.02, alpha: float = 0.05,
                 kelly_fraction: float = 0.25):
        """
        Args:
            cvar_limit: Maximum acceptable CVaR (e.g., -0.02 = max 2% expected tail loss)
            alpha: CVaR significance level
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.cvar_limit = cvar_limit
        self.alpha = alpha
        self.kelly_fraction = kelly_fraction

    def optimal_position(self, win_prob: float, win_loss_ratio: float,
                         return_std: float) -> float:
        """
        Compute CVaR-constrained Kelly position.

        Args:
            win_prob: Probability of winning trade
            win_loss_ratio: Average win / Average loss
            return_std: Standard deviation of returns

        Returns:
            Optimal position size (fraction of capital)
        """
        # Standard Kelly
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly = max(0, kelly) * self.kelly_fraction

        # CVaR constraint (assuming normal for simplicity)
        cvar_position = ConditionalValueAtRisk.parametric(0, return_std, self.alpha)

        if cvar_position != 0:
            # Scale position so CVaR meets limit
            cvar_constrained = abs(self.cvar_limit / cvar_position)
        else:
            cvar_constrained = kelly

        # Return minimum of Kelly and CVaR-constrained
        return min(kelly, cvar_constrained)


@dataclass
class RiskMetricsResult:
    """Results from risk metrics calculation."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    evar_95: Optional[float]
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float


def compute_risk_metrics(returns: np.ndarray, risk_free: float = 0.0) -> RiskMetricsResult:
    """
    Compute comprehensive risk metrics.

    Args:
        returns: Array of returns
        risk_free: Risk-free rate for Sortino

    Returns:
        RiskMetricsResult with all metrics
    """
    # VaR
    var_95 = ValueAtRisk.historical(returns, 0.05)
    var_99 = ValueAtRisk.historical(returns, 0.01)

    # CVaR
    cvar_95 = ConditionalValueAtRisk.historical(returns, 0.05)
    cvar_99 = ConditionalValueAtRisk.historical(returns, 0.01)

    # EVaR (if scipy available)
    try:
        evar_95 = EntropicValueAtRisk.compute(returns, 0.05)
    except:
        evar_95 = None

    # Max Drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    max_dd = drawdowns.max()

    # Calmar Ratio
    annualized_return = np.mean(returns) * 252
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # Sortino Ratio
    excess_returns = returns - risk_free / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino = annualized_return / downside_std if downside_std > 0 else 0.0

    return RiskMetricsResult(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        evar_95=evar_95,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        sortino_ratio=sortino
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OPTIMAL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionConfig:
    """Configuration for optimal execution."""
    permanent_impact: float = 0.1  # gamma - permanent price impact
    temporary_impact: float = 0.01  # eta - temporary price impact
    risk_aversion: float = 1e-6  # lambda - risk aversion parameter


class AlmgrenChriss:
    """
    Almgren-Chriss Optimal Execution Model.

    Reference:
        Almgren, R., & Chriss, N. (2001).
        "Optimal Execution of Portfolio Transactions."
        Journal of Risk, 3(2), 5-39. [2,500+ citations]

    Finds optimal trading trajectory that minimizes:
    E[Cost] + λ · Var[Cost]

    Where cost includes:
    - Permanent impact: γ · σ² · X² · T
    - Temporary impact: η · X² / T
    - Risk penalty: λ · σ² · ∫ x(t)² dt
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def optimal_trajectory(self, shares: float, time_horizon: float,
                          volatility: float, n_steps: int = 100) -> np.ndarray:
        """
        Compute optimal execution trajectory.

        Args:
            shares: Total shares to execute
            time_horizon: Time to complete execution (in trading days)
            volatility: Asset volatility (daily)
            n_steps: Number of time steps

        Returns:
            Array of remaining shares at each time step
        """
        gamma = self.config.permanent_impact
        eta = self.config.temporary_impact
        lam = self.config.risk_aversion
        sigma = volatility

        # Time discretization
        tau = time_horizon / n_steps
        t = np.linspace(0, time_horizon, n_steps + 1)

        # Almgren-Chriss parameter
        kappa_sq = lam * sigma**2 / eta
        kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 0.0

        if kappa > 1e-10:
            # Optimal trajectory (Almgren-Chriss Eq. 20)
            trajectory = shares * np.sinh(kappa * (time_horizon - t)) / np.sinh(kappa * time_horizon)
        else:
            # Linear trajectory (risk-neutral)
            trajectory = shares * (1 - t / time_horizon)

        return trajectory

    def expected_cost(self, shares: float, time_horizon: float,
                     volatility: float) -> Tuple[float, float, float]:
        """
        Compute expected execution cost.

        Args:
            shares: Total shares to execute
            time_horizon: Time to complete execution
            volatility: Asset volatility

        Returns:
            Tuple of (total_cost, permanent_cost, temporary_cost)
        """
        gamma = self.config.permanent_impact
        eta = self.config.temporary_impact
        sigma = volatility

        # Permanent impact cost
        permanent_cost = 0.5 * gamma * sigma**2 * shares**2 * time_horizon

        # Temporary impact cost (for linear trajectory)
        temporary_cost = eta * shares**2 / time_horizon

        total_cost = permanent_cost + temporary_cost

        return total_cost, permanent_cost, temporary_cost

    def optimal_time_horizon(self, shares: float, volatility: float,
                            urgency: float = 1.0) -> float:
        """
        Compute optimal execution time given urgency parameter.

        Args:
            shares: Total shares to execute
            volatility: Asset volatility
            urgency: Urgency multiplier (higher = faster execution)

        Returns:
            Optimal time horizon
        """
        gamma = self.config.permanent_impact
        eta = self.config.temporary_impact
        lam = self.config.risk_aversion * urgency
        sigma = volatility

        # Optimal T* minimizes cost + risk
        # T* = sqrt(eta / (lambda * sigma^2))
        if lam * sigma**2 > 1e-10:
            return np.sqrt(eta / (lam * sigma**2))
        else:
            return 1.0  # Default to 1 day


class RLOptimalExecution:
    """
    RL-Enhanced Optimal Execution.

    Reference:
        Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006).
        "Reinforcement Learning for Optimized Trade Execution."
        ICML 2006.

    Combines Almgren-Chriss baseline with RL adaptations for:
    - Time-varying volatility
    - Order book state
    - Intraday patterns
    """

    def __init__(self, base_config: Optional[ExecutionConfig] = None):
        self.ac = AlmgrenChriss(base_config)
        self.state_history = []

    def compute_state(self, remaining_shares: float, time_remaining: float,
                     volatility: float, spread: float,
                     order_imbalance: float) -> np.ndarray:
        """
        Compute state vector for RL agent.

        Args:
            remaining_shares: Shares left to execute
            time_remaining: Time left (fraction of total)
            volatility: Current volatility estimate
            spread: Current bid-ask spread
            order_imbalance: Current order book imbalance

        Returns:
            State vector for RL
        """
        # Normalize features
        state = np.array([
            remaining_shares,  # Will be normalized by initial shares
            time_remaining,
            volatility * np.sqrt(252),  # Annualized
            spread * 10000,  # In bps
            order_imbalance  # Already in [-1, 1]
        ])
        return state

    def compute_reward(self, execution_price: float, mid_price: float,
                      shares_executed: float, time_elapsed: float) -> float:
        """
        Compute execution reward (negative cost).

        Args:
            execution_price: Price at which trade executed
            mid_price: Mid price at execution time
            shares_executed: Number of shares executed this step
            time_elapsed: Time elapsed this step

        Returns:
            Reward (negative of implementation shortfall)
        """
        # Implementation shortfall
        slippage = (execution_price - mid_price) * shares_executed

        # Add time penalty (opportunity cost)
        time_penalty = 0.0001 * time_elapsed * shares_executed

        return -(slippage + time_penalty)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MARKET MAKING WITH RL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketMakingConfig:
    """Configuration for market making."""
    gamma: float = 0.1  # Risk aversion
    sigma: float = 0.02  # Volatility (daily)
    kappa: float = 1.5  # Order arrival intensity
    max_inventory: int = 100  # Maximum inventory
    tick_size: float = 0.0001  # Minimum price increment


class AvellanedaStoikovRL:
    """
    Avellaneda-Stoikov Market Making with RL Enhancement.

    References:
        1. Avellaneda, M., & Stoikov, S. (2008).
           "High-frequency Trading in a Limit Order Book."
           Quantitative Finance, 8(3), 217-224. [1,800+ citations]

        2. Spooner, T., Fearnley, J., Savani, R., & Koutsoupias, E. (2018).
           "Market Making via Reinforcement Learning." AAMAS 2018.

    Computes optimal bid/ask quotes given inventory risk.

    Key formulas:
        Reservation price: r = s - q · γ · σ² · (T - t)
        Optimal spread: δ = γ · σ² · (T - t) + (2/γ) · log(1 + γ/κ)
    """

    def __init__(self, config: Optional[MarketMakingConfig] = None):
        self.config = config or MarketMakingConfig()
        self.inventory = 0
        self.pnl = 0.0
        self.mid_price_history = []

    def reset(self) -> None:
        """Reset market maker state."""
        self.inventory = 0
        self.pnl = 0.0
        self.mid_price_history = []

    def reservation_price(self, mid_price: float, time_remaining: float) -> float:
        """
        Compute reservation price (Avellaneda-Stoikov Eq. 8).

        The reservation price is where the market maker is indifferent
        to holding inventory. It adjusts based on current inventory.

        Args:
            mid_price: Current mid price
            time_remaining: Time until end of session (fraction)

        Returns:
            Reservation price
        """
        gamma = self.config.gamma
        sigma = self.config.sigma
        q = self.inventory

        # r = s - q * gamma * sigma^2 * (T - t)
        return mid_price - q * gamma * sigma**2 * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """
        Compute optimal bid-ask spread (Avellaneda-Stoikov Eq. 10).

        Args:
            time_remaining: Time until end of session

        Returns:
            Optimal spread (full, not half)
        """
        gamma = self.config.gamma
        sigma = self.config.sigma
        kappa = self.config.kappa

        # delta = gamma * sigma^2 * (T - t) + (2/gamma) * log(1 + gamma/kappa)
        time_component = gamma * sigma**2 * time_remaining
        intensity_component = (2 / gamma) * np.log(1 + gamma / kappa)

        return time_component + intensity_component

    def optimal_quotes(self, mid_price: float, time_remaining: float) -> Tuple[float, float]:
        """
        Compute optimal bid and ask prices.

        Args:
            mid_price: Current mid price
            time_remaining: Time until end of session

        Returns:
            Tuple of (bid_price, ask_price)
        """
        r = self.reservation_price(mid_price, time_remaining)
        delta = self.optimal_spread(time_remaining)

        bid = r - delta / 2
        ask = r + delta / 2

        # Round to tick size
        tick = self.config.tick_size
        bid = np.floor(bid / tick) * tick
        ask = np.ceil(ask / tick) * tick

        return bid, ask

    def compute_state(self, mid_price: float, time_remaining: float,
                     volatility: float, order_flow: float) -> np.ndarray:
        """
        Compute state for RL agent.

        Args:
            mid_price: Current mid price
            time_remaining: Time until session end
            volatility: Current volatility estimate
            order_flow: Recent order flow imbalance

        Returns:
            State vector
        """
        return np.array([
            self.inventory / self.config.max_inventory,  # Normalized inventory
            time_remaining,
            volatility / self.config.sigma,  # Relative volatility
            order_flow,  # Order flow imbalance
            mid_price  # For price-aware policies
        ])

    def step(self, mid_price: float, time_remaining: float,
            bid_filled: bool, ask_filled: bool) -> float:
        """
        Process one time step.

        Args:
            mid_price: Current mid price
            time_remaining: Time remaining
            bid_filled: Whether bid was hit
            ask_filled: Whether ask was lifted

        Returns:
            Reward for this step
        """
        bid, ask = self.optimal_quotes(mid_price, time_remaining)

        reward = 0.0

        if bid_filled:
            # Bought at bid
            self.inventory += 1
            self.pnl -= bid
            reward += mid_price - bid  # Positive if bought below mid

        if ask_filled:
            # Sold at ask
            self.inventory -= 1
            self.pnl += ask
            reward += ask - mid_price  # Positive if sold above mid

        # Inventory penalty (quadratic)
        inventory_penalty = 0.001 * self.inventory**2
        reward -= inventory_penalty

        self.mid_price_history.append(mid_price)

        return reward

    def terminal_value(self, final_price: float) -> float:
        """
        Compute terminal value including inventory liquidation.

        Args:
            final_price: Price at session end

        Returns:
            Total PnL including mark-to-market
        """
        return self.pnl + self.inventory * final_price


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DISTRIBUTIONAL RL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class QuantileDistribution:
    """
    Quantile Distribution for Distributional RL.

    Reference:
        Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018).
        "Distributional Reinforcement Learning with Quantile Regression."
        AAAI 2018.

    Instead of learning E[R], learn the full distribution of returns
    via quantile regression. Enables risk-sensitive decision making.
    """

    def __init__(self, n_quantiles: int = 51):
        """
        Args:
            n_quantiles: Number of quantiles to estimate
        """
        self.n_quantiles = n_quantiles
        # Quantile midpoints (τ)
        self.taus = (np.arange(n_quantiles) + 0.5) / n_quantiles
        self.quantiles = None

    def set_quantiles(self, quantiles: np.ndarray) -> None:
        """Set quantile values from model output."""
        assert len(quantiles) == self.n_quantiles
        self.quantiles = np.sort(quantiles)  # Ensure sorted

    def mean(self) -> float:
        """Expected value (mean of distribution)."""
        if self.quantiles is None:
            return 0.0
        return self.quantiles.mean()

    def var(self, alpha: float = 0.05) -> float:
        """Value at Risk at level alpha."""
        if self.quantiles is None:
            return 0.0
        idx = int(alpha * self.n_quantiles)
        return self.quantiles[idx]

    def cvar(self, alpha: float = 0.05) -> float:
        """Conditional Value at Risk."""
        if self.quantiles is None:
            return 0.0
        idx = int(alpha * self.n_quantiles)
        return self.quantiles[:idx+1].mean()

    def std(self) -> float:
        """Standard deviation."""
        if self.quantiles is None:
            return 0.0
        return self.quantiles.std()

    def iqr(self) -> float:
        """Interquartile range."""
        if self.quantiles is None:
            return 0.0
        q25 = self.quantiles[int(0.25 * self.n_quantiles)]
        q75 = self.quantiles[int(0.75 * self.n_quantiles)]
        return q75 - q25

    def risk_adjusted_value(self, risk_aversion: float = 1.0) -> float:
        """
        Risk-adjusted value: mean - risk_aversion * std.

        Args:
            risk_aversion: Risk aversion coefficient

        Returns:
            Risk-adjusted expected value
        """
        return self.mean() - risk_aversion * self.std()


class QuantileHuberLoss:
    """
    Quantile Huber Loss for QR-DQN training.

    Reference:
        Dabney et al. (2018). "Distributional RL with Quantile Regression."

    Combines quantile regression loss with Huber loss for robustness.

    ρ_τ^κ(u) = |τ - 𝟙(u < 0)| · L_κ(u) / κ

    Where L_κ is Huber loss with threshold κ.
    """

    def __init__(self, kappa: float = 1.0):
        """
        Args:
            kappa: Huber loss threshold
        """
        self.kappa = kappa

    def __call__(self, predictions: np.ndarray, targets: np.ndarray,
                taus: np.ndarray) -> float:
        """
        Compute quantile Huber loss.

        Args:
            predictions: Predicted quantiles [batch, n_quantiles]
            targets: Target values [batch, 1] or [batch, n_quantiles]
            taus: Quantile levels [n_quantiles]

        Returns:
            Loss value
        """
        # Ensure correct shapes
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        # TD errors
        u = targets - predictions

        # Huber loss
        huber = np.where(
            np.abs(u) <= self.kappa,
            0.5 * u**2,
            self.kappa * (np.abs(u) - 0.5 * self.kappa)
        )

        # Quantile weights
        tau_weights = np.abs(taus - (u < 0).astype(float))

        # Combined loss
        loss = tau_weights * huber / self.kappa

        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: META-RL FOR REGIME CHANGES
# ══════════════════════════════════════════════════════════════════════════════

class MAMLTrader:
    """
    Model-Agnostic Meta-Learning for Trading.

    Reference:
        Finn, C., Abbeel, P., & Levine, S. (2017).
        "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
        ICML 2017. [10,000+ citations]

    MAML learns initialization that enables fast adaptation to new market regimes
    with only a few gradient steps.

    θ* = argmin_θ Σ_i L_i(θ - α∇L_i(θ))

    Inner loop: Adapt to regime with few samples
    Outer loop: Learn good initialization across regimes
    """

    def __init__(self, inner_lr: float = 0.01, outer_lr: float = 0.001,
                 inner_steps: int = 5):
        """
        Args:
            inner_lr: Learning rate for inner (adaptation) loop
            outer_lr: Learning rate for outer (meta) loop
            inner_steps: Number of gradient steps for adaptation
        """
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_parameters = None  # θ - would be neural network weights

    def adapt(self, support_data: np.ndarray, support_labels: np.ndarray) -> Dict:
        """
        Adapt to new regime using support set.

        Args:
            support_data: Support set features [n_support, feature_dim]
            support_labels: Support set labels [n_support]

        Returns:
            Adapted parameters (task-specific)
        """
        # In practice, this would:
        # 1. Clone meta_parameters
        # 2. Take inner_steps gradient steps on support data
        # 3. Return adapted parameters

        # Placeholder for actual implementation
        adapted_params = {
            'inner_lr': self.inner_lr,
            'steps': self.inner_steps,
            'support_size': len(support_data)
        }
        return adapted_params

    def predict(self, query_data: np.ndarray, adapted_params: Dict) -> np.ndarray:
        """
        Make predictions using adapted parameters.

        Args:
            query_data: Query set features
            adapted_params: Task-adapted parameters

        Returns:
            Predictions
        """
        # In practice, forward pass with adapted_params
        return np.zeros(len(query_data))

    def meta_update(self, tasks: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Meta-update across multiple tasks (regimes).

        Args:
            tasks: List of (support_data, query_data) tuples

        Returns:
            Meta-loss
        """
        # In practice:
        # 1. For each task, compute adapted params
        # 2. Evaluate on query set
        # 3. Compute gradients through adaptation
        # 4. Update meta_parameters

        return 0.0  # Placeholder


class RegimeAwareRL:
    """
    Regime-Aware RL that combines HMM regime detection with policy switching.

    References:
        1. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis
           of Nonstationary Time Series and the Business Cycle."
           Econometrica, 57(2), 357-384.

        2. Context-based Meta-RL literature (2020-2024).

    Uses latent context to identify market regime and selects appropriate policy.
    """

    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: Number of market regimes (e.g., trending, mean-reverting, volatile)
        """
        self.n_regimes = n_regimes
        self.regime_policies = {}  # Policy per regime
        self.current_regime = 0
        self.regime_probs = np.ones(n_regimes) / n_regimes

    def update_regime(self, regime_probs: np.ndarray) -> int:
        """
        Update regime belief.

        Args:
            regime_probs: Probability distribution over regimes from HMM

        Returns:
            Most likely regime
        """
        self.regime_probs = regime_probs
        self.current_regime = np.argmax(regime_probs)
        return self.current_regime

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from regime-appropriate policy.

        Args:
            state: Current state

        Returns:
            Action
        """
        # Ensemble across regimes weighted by probability
        # action = Σ_i p(regime=i) * π_i(state)

        # Placeholder - would use actual policies
        return np.zeros(1)

    def compute_context_vector(self, recent_returns: np.ndarray,
                               recent_volatility: np.ndarray) -> np.ndarray:
        """
        Compute latent context for regime identification.

        Args:
            recent_returns: Recent return history
            recent_volatility: Recent volatility estimates

        Returns:
            Context vector
        """
        return np.array([
            np.mean(recent_returns),
            np.std(recent_returns),
            np.mean(recent_volatility),
            np.std(recent_volatility),
            np.corrcoef(recent_returns[:-1], recent_returns[1:])[0, 1]  # Autocorrelation
        ])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: POSITION SIZING RL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSizerConfig:
    """Configuration for RL position sizer."""
    max_position: float = 1.0  # Maximum position (fraction of capital)
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    cvar_limit: float = -0.02  # Maximum acceptable CVaR
    use_regime: bool = True  # Use regime in state


class RLPositionSizer:
    """
    RL-Based Position Sizing.

    References:
        1. Jiang, Z., Xu, D., & Liang, J. (2017).
           "A Deep Reinforcement Learning Framework for Portfolio Management."

        2. Yang, H., Liu, X. Y., Zhong, S., & Walid, A. (2020).
           "Deep Reinforcement Learning for Automated Stock Trading."
           ACM ICAIF 2020.

    Uses RL to optimize position sizing based on:
    - Prediction confidence
    - Current regime
    - Risk constraints (CVaR, drawdown)
    - Recent performance
    """

    def __init__(self, config: Optional[PositionSizerConfig] = None):
        self.config = config or PositionSizerConfig()
        self.kelly = CVaRConstrainedKelly(
            cvar_limit=self.config.cvar_limit,
            kelly_fraction=self.config.kelly_fraction
        )
        self.recent_returns = []
        self.max_recent = 100

    def compute_state(self, prediction_confidence: float, regime_probs: np.ndarray,
                     current_position: float, recent_pnl: float) -> np.ndarray:
        """
        Compute state for position sizing RL.

        Args:
            prediction_confidence: ML model confidence [0, 1]
            regime_probs: Probability of each regime
            current_position: Current position (fraction)
            recent_pnl: Recent PnL (normalized)

        Returns:
            State vector
        """
        state = [
            prediction_confidence,
            current_position / self.config.max_position,
            recent_pnl
        ]

        if self.config.use_regime:
            state.extend(regime_probs.tolist())

        return np.array(state)

    def kelly_baseline(self, win_prob: float, win_loss_ratio: float,
                      return_std: float) -> float:
        """
        Compute CVaR-constrained Kelly as baseline.

        Args:
            win_prob: Probability of winning
            win_loss_ratio: Win/loss ratio
            return_std: Return volatility

        Returns:
            Baseline position size
        """
        return self.kelly.optimal_position(win_prob, win_loss_ratio, return_std)

    def compute_reward(self, position: float, realized_return: float,
                      transaction_cost: float = 0.0001) -> float:
        """
        Compute reward for position sizing decision.

        Args:
            position: Position taken
            realized_return: Actual return
            transaction_cost: Cost per unit traded

        Returns:
            Reward (risk-adjusted return)
        """
        # Realized PnL
        pnl = position * realized_return

        # Transaction cost
        cost = abs(position) * transaction_cost

        # Track for risk metrics
        self.recent_returns.append(pnl - cost)
        if len(self.recent_returns) > self.max_recent:
            self.recent_returns.pop(0)

        # Risk-adjusted reward (use DSR if enough history)
        if len(self.recent_returns) >= 10:
            returns_arr = np.array(self.recent_returns)
            cvar = ConditionalValueAtRisk.historical(returns_arr, 0.05)

            # Penalize if CVaR exceeds limit
            if cvar < self.config.cvar_limit:
                penalty = 10 * (self.config.cvar_limit - cvar)
            else:
                penalty = 0

            return pnl - cost - penalty
        else:
            return pnl - cost


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: FEATURE GENERATION FOR RL
# ══════════════════════════════════════════════════════════════════════════════

class RLFeatureGenerator:
    """
    Generate features specifically designed for RL trading agents.

    Combines:
    - Risk metrics (VaR, CVaR, drawdown)
    - Reward statistics (DSR, rolling Sharpe)
    - Execution metrics (slippage, fill rate)
    - Market state (regime, volatility)
    """

    def __init__(self, lookback: int = 100):
        """
        Args:
            lookback: Lookback period for rolling metrics
        """
        self.lookback = lookback
        self.dsr = DifferentialSharpeRatio(eta=0.01)

    def generate(self, returns: np.ndarray, prices: np.ndarray,
                volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Generate RL-specific features.

        Args:
            returns: Recent returns
            prices: Recent prices
            volumes: Recent volumes (optional)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        if len(returns) < 10:
            return features

        # Risk metrics
        risk = compute_risk_metrics(returns[-self.lookback:])
        features['var_95'] = risk.var_95
        features['cvar_95'] = risk.cvar_95
        features['max_drawdown'] = risk.max_drawdown
        features['sortino'] = risk.sortino_ratio
        features['calmar'] = risk.calmar_ratio

        # Reward statistics
        dsr_values = self.dsr.compute_batch(returns[-self.lookback:])
        features['dsr_current'] = dsr_values[-1] if len(dsr_values) > 0 else 0
        features['dsr_mean'] = np.mean(dsr_values)
        features['dsr_std'] = np.std(dsr_values)

        # Rolling Sharpe
        if len(returns) >= 20:
            rolling_mean = np.mean(returns[-20:])
            rolling_std = np.std(returns[-20:])
            features['sharpe_20'] = rolling_mean / rolling_std if rolling_std > 0 else 0

        # Volatility features
        features['volatility'] = np.std(returns[-20:]) * np.sqrt(252)
        features['volatility_ratio'] = (
            np.std(returns[-10:]) / np.std(returns[-50:])
            if len(returns) >= 50 and np.std(returns[-50:]) > 0
            else 1.0
        )

        # Price momentum
        if len(prices) >= 20:
            features['momentum_20'] = (prices[-1] / prices[-20]) - 1

        # Volume features (if available)
        if volumes is not None and len(volumes) >= 20:
            features['volume_ratio'] = np.mean(volumes[-5:]) / np.mean(volumes[-20:])

        return features


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_reward_function(reward_type: str = 'combined',
                          **kwargs) -> Union[DifferentialSharpeRatio,
                                            DifferentialDownsideDeviation,
                                            MaxDrawdownPenalty,
                                            CombinedRewardFunction]:
    """
    Factory function to create reward functions.

    Args:
        reward_type: One of 'dsr', 'sortino', 'drawdown', 'combined'
        **kwargs: Arguments passed to reward function

    Returns:
        Reward function instance
    """
    if reward_type == 'dsr':
        return DifferentialSharpeRatio(**kwargs)
    elif reward_type == 'sortino':
        return DifferentialDownsideDeviation(**kwargs)
    elif reward_type == 'drawdown':
        return MaxDrawdownPenalty(**kwargs)
    elif reward_type == 'combined':
        config = RewardConfig(**kwargs) if kwargs else None
        return CombinedRewardFunction(config)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def create_execution_model(model_type: str = 'almgren_chriss',
                          **kwargs) -> Union[AlmgrenChriss, RLOptimalExecution]:
    """
    Factory function to create execution models.

    Args:
        model_type: One of 'almgren_chriss', 'rl_execution'
        **kwargs: Arguments passed to model

    Returns:
        Execution model instance
    """
    config = ExecutionConfig(**kwargs) if kwargs else None

    if model_type == 'almgren_chriss':
        return AlmgrenChriss(config)
    elif model_type == 'rl_execution':
        return RLOptimalExecution(config)
    else:
        raise ValueError(f"Unknown execution model: {model_type}")


def create_market_maker(mm_type: str = 'avellaneda_stoikov',
                       **kwargs) -> AvellanedaStoikovRL:
    """
    Factory function to create market maker.

    Args:
        mm_type: Currently only 'avellaneda_stoikov'
        **kwargs: Arguments passed to model

    Returns:
        Market maker instance
    """
    config = MarketMakingConfig(**kwargs) if kwargs else None
    return AvellanedaStoikovRL(config)


def generate_rl_features(returns: np.ndarray, prices: np.ndarray,
                        volumes: Optional[np.ndarray] = None,
                        lookback: int = 100) -> Dict[str, float]:
    """
    Convenience function to generate RL features.

    Args:
        returns: Return series
        prices: Price series
        volumes: Volume series (optional)
        lookback: Lookback period

    Returns:
        Dictionary of RL features
    """
    generator = RLFeatureGenerator(lookback=lookback)
    return generator.generate(returns, prices, volumes)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Reward Shaping (Moody & Saffell 2001)
    'DifferentialSharpeRatio',
    'DifferentialDownsideDeviation',
    'MaxDrawdownPenalty',
    'RewardConfig',
    'CombinedRewardFunction',

    # Risk-Sensitive RL (Tamar 2015, Chow 2017)
    'ValueAtRisk',
    'ConditionalValueAtRisk',
    'EntropicValueAtRisk',
    'CVaRConstrainedKelly',
    'RiskMetricsResult',
    'compute_risk_metrics',

    # Optimal Execution (Almgren-Chriss 2001)
    'ExecutionConfig',
    'AlmgrenChriss',
    'RLOptimalExecution',

    # Market Making (Avellaneda-Stoikov 2008)
    'MarketMakingConfig',
    'AvellanedaStoikovRL',

    # Distributional RL (Dabney 2018)
    'QuantileDistribution',
    'QuantileHuberLoss',

    # Meta-RL (Finn 2017)
    'MAMLTrader',
    'RegimeAwareRL',

    # Position Sizing
    'PositionSizerConfig',
    'RLPositionSizer',

    # Feature Generation
    'RLFeatureGenerator',

    # Factory Functions
    'create_reward_function',
    'create_execution_model',
    'create_market_maker',
    'generate_rl_features',
]
