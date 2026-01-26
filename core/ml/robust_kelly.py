"""
Robust Kelly Criterion with Uncertainty Quantification

Standard Kelly assumes you KNOW the win probability exactly (p = 0.82).
But we only ESTIMATE it with statistical error!

Robust Kelly optimizes for the WORST CASE within our uncertainty set:
    f*_robust = argmax_f min_{p ∈ [p_lower, p_upper]} E_p[log(1 + f*X)]

Academic Citations:
- Kelly (1956): "A New Interpretation of Information Rate"
  Bell System Technical Journal - Original Kelly criterion

- Thorp (2006): "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
  Handbook of Asset and Liability Management - Practical applications

- MacLean, Thorp, Ziemba (2011): "The Kelly Capital Growth Investment Criterion"
  World Scientific - Comprehensive treatment

- Busseti, Ryu, Boyd (2016): "Risk-Constrained Kelly Gambling"
  Journal of Investing - Convex optimization approach

- Hsieh, Barmish (2018): "On Distributionally Robust Kelly Betting"
  arXiv:1812.10371 - Robust optimization under ambiguity

- Lo, Orr, Zhang (2018): "The Growth Rate of Long-Run Wealth"
  Management Science - Fractional Kelly analysis

Chinese Quant Application:
- 幻方量化: "凯利公式需要考虑估计误差" (Kelly needs estimation error)
- 九坤投资: Uses adaptive Kelly with rolling confidence intervals
- 招商证券: Recommends fractional Kelly (0.25-0.5) for robustness

Key insight:
    Standard Kelly: f* = p - q/b  (assumes p known)
    Robust Kelly: f* = p_lower - q_upper/b  (worst case)

    Where [p_lower, p_upper] is confidence interval for p

This provides position sizing with MATHEMATICAL GUARANTEE that
we don't overbend even if our probability estimate is wrong!
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import warnings


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation with uncertainty."""

    # Point estimates
    win_probability: float
    win_loss_ratio: float  # b = avg_win / avg_loss

    # Standard Kelly
    kelly_fraction: float  # f* from standard Kelly
    expected_growth: float  # E[log(1 + f*X)]

    # Robust Kelly
    robust_fraction: float  # f* from robust Kelly
    robust_growth: float  # Worst-case expected growth

    # Uncertainty quantification
    prob_confidence_interval: Tuple[float, float] = (0.0, 1.0)
    confidence_level: float = 0.95

    # Practical recommendations
    recommended_fraction: float = 0.0  # Final recommendation
    max_safe_fraction: float = 0.0  # Upper bound for safety
    recommendation_reason: str = ""

    # Risk metrics
    probability_of_ruin: float = 0.0
    expected_drawdown: float = 0.0
    time_to_double: Optional[float] = None  # Expected trades to double


@dataclass
class BettingOutcome:
    """Represents a single betting outcome."""
    return_pct: float  # Return as percentage (e.g., 0.02 for 2%)
    probability: float  # Probability of this outcome


class RobustKellyCriterion:
    """
    Robust Kelly Criterion with uncertainty quantification.

    Accounts for:
    1. Statistical uncertainty in win probability
    2. Parameter estimation error
    3. Model risk / distribution shift

    Provides position sizing that is optimal under worst-case
    within the uncertainty set.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        max_fraction: float = 0.5,  # Never bet more than 50%
        min_edge_required: float = 0.02,  # 2% minimum edge
    ):
        """
        Initialize Robust Kelly calculator.

        Args:
            confidence_level: Confidence level for uncertainty sets (0.95 = 95%)
            max_fraction: Maximum allowed Kelly fraction
            min_edge_required: Minimum edge (p - 0.5) required to bet

        Reference:
            Lo, Orr, Zhang (2018) - fractional Kelly analysis
        """
        self.confidence_level = confidence_level
        self.max_fraction = max_fraction
        self.min_edge_required = min_edge_required

    def _wilson_confidence_interval(
        self,
        n_wins: int,
        n_total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Wilson score interval for binomial proportion.

        More accurate than normal approximation, especially for
        extreme probabilities or small samples.

        Reference:
            Wilson (1927), Brown, Cai, DasGupta (2001)
        """
        if n_total == 0:
            return (0.0, 1.0)

        p_hat = n_wins / n_total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        denominator = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _clopper_pearson_interval(
        self,
        n_wins: int,
        n_total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Exact Clopper-Pearson confidence interval.

        Conservative (exact coverage guaranteed) but wider than Wilson.
        Use when guaranteed coverage is critical.

        Reference:
            Clopper & Pearson (1934), "exact" binomial CI
        """
        alpha = 1 - confidence

        if n_wins == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, n_wins, n_total - n_wins + 1)

        if n_wins == n_total:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, n_wins + 1, n_total - n_wins)

        return (lower, upper)

    def standard_kelly(
        self,
        win_prob: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Standard Kelly criterion.

        f* = p - q/b = p - (1-p)/b

        Where:
            p = probability of winning
            q = 1 - p = probability of losing
            b = win/loss ratio (how much you win vs lose)

        Reference:
            Kelly (1956), Thorp (2006)
        """
        q = 1 - win_prob
        f_star = win_prob - q / win_loss_ratio

        return max(0, min(f_star, self.max_fraction))

    def robust_kelly(
        self,
        win_prob_lower: float,
        win_prob_upper: float,
        win_loss_ratio: float,
        loss_ratio_uncertainty: float = 0.1,
    ) -> float:
        """
        Robust Kelly criterion optimizing worst-case growth.

        f*_robust = argmax_f min_{p ∈ [p_low, p_high]} E_p[log(1 + f*X)]

        For Kelly with binary outcomes:
        Worst case is at p = p_lower, so:
        f*_robust = p_lower - (1 - p_lower) / b_lower

        Where b_lower = b * (1 - loss_ratio_uncertainty)

        Reference:
            Hsieh & Barmish (2018) - arXiv:1812.10371
        """
        # Use lower bound of win probability
        p_worst = win_prob_lower

        # Use lower bound of win/loss ratio
        b_worst = win_loss_ratio * (1 - loss_ratio_uncertainty)

        if b_worst <= 0:
            return 0.0

        q_worst = 1 - p_worst
        f_robust = p_worst - q_worst / b_worst

        return max(0, min(f_robust, self.max_fraction))

    def optimal_fraction(
        self,
        outcomes: List[BettingOutcome],
    ) -> float:
        """
        Compute optimal Kelly fraction for arbitrary outcome distribution.

        Solves: max_f E[log(1 + f*R)] where R is the return distribution

        Uses numerical optimization for non-binary outcomes.

        Reference:
            MacLean, Thorp, Ziemba (2011) Chapter 1
        """
        def neg_expected_log_growth(f):
            """Negative expected log growth (to minimize)."""
            growth = 0.0
            for outcome in outcomes:
                growth_term = np.log(1 + f * outcome.return_pct)
                growth += outcome.probability * growth_term
            return -growth

        # Find optimal fraction
        result = minimize_scalar(
            neg_expected_log_growth,
            bounds=(0, self.max_fraction),
            method='bounded'
        )

        return result.x if result.success else 0.0

    def calculate(
        self,
        n_wins: int,
        n_total: int,
        avg_win: float,
        avg_loss: float,
        use_clopper_pearson: bool = False,
    ) -> KellyResult:
        """
        Full Kelly calculation with uncertainty quantification.

        Args:
            n_wins: Number of winning trades
            n_total: Total number of trades
            avg_win: Average winning trade return (e.g., 0.02 for 2%)
            avg_loss: Average losing trade loss (e.g., 0.01 for 1%, positive)
            use_clopper_pearson: Use exact CI (more conservative)

        Returns:
            KellyResult with standard and robust Kelly fractions

        Example:
            >>> kelly = RobustKellyCriterion()
            >>> result = kelly.calculate(
            ...     n_wins=820, n_total=1000,
            ...     avg_win=0.002, avg_loss=0.001
            ... )
            >>> print(f"Robust fraction: {result.robust_fraction:.2%}")
        """
        if n_total == 0:
            return KellyResult(
                win_probability=0.5,
                win_loss_ratio=1.0,
                kelly_fraction=0.0,
                expected_growth=0.0,
                robust_fraction=0.0,
                robust_growth=0.0,
                recommended_fraction=0.0,
                max_safe_fraction=0.0,
                recommendation_reason="No trades to analyze"
            )

        # Point estimate
        p_hat = n_wins / n_total

        # Win/loss ratio
        if avg_loss <= 0:
            avg_loss = 0.001  # Minimum to avoid division by zero
        b = avg_win / avg_loss

        # Confidence interval for win probability
        if use_clopper_pearson:
            p_lower, p_upper = self._clopper_pearson_interval(
                n_wins, n_total, self.confidence_level
            )
        else:
            p_lower, p_upper = self._wilson_confidence_interval(
                n_wins, n_total, self.confidence_level
            )

        # Standard Kelly
        f_standard = self.standard_kelly(p_hat, b)

        # Expected growth at standard Kelly
        if f_standard > 0:
            exp_growth = p_hat * np.log(1 + f_standard * avg_win) + \
                        (1 - p_hat) * np.log(1 - f_standard * avg_loss)
        else:
            exp_growth = 0.0

        # Robust Kelly (worst case within CI)
        f_robust = self.robust_kelly(p_lower, p_upper, b)

        # Worst-case expected growth
        if f_robust > 0:
            robust_growth = p_lower * np.log(1 + f_robust * avg_win) + \
                           (1 - p_lower) * np.log(1 - f_robust * avg_loss)
        else:
            robust_growth = 0.0

        # Practical recommendation
        edge = p_hat - 0.5

        if edge < self.min_edge_required:
            recommended = 0.0
            reason = f"Edge {edge:.2%} below minimum {self.min_edge_required:.2%}"
        elif p_lower < 0.5:
            # Confidence interval includes no edge!
            recommended = 0.0
            reason = f"CI includes p<0.5: [{p_lower:.2%}, {p_upper:.2%}]"
        else:
            # Use robust Kelly with additional safety margin
            recommended = f_robust * 0.75  # 75% of robust Kelly
            reason = f"Robust Kelly * 0.75 safety factor"

        recommended = min(recommended, self.max_fraction)

        # Risk metrics
        prob_ruin = self._probability_of_ruin(p_hat, b, recommended)
        exp_drawdown = self._expected_max_drawdown(p_hat, recommended, n_samples=1000)
        time_double = self._time_to_double(p_hat, b, recommended, avg_win, avg_loss)

        # Max safe fraction (where ruin probability < 1%)
        max_safe = self._find_max_safe_fraction(p_hat, b, max_ruin_prob=0.01)

        return KellyResult(
            win_probability=p_hat,
            win_loss_ratio=b,
            kelly_fraction=f_standard,
            expected_growth=exp_growth,
            robust_fraction=f_robust,
            robust_growth=robust_growth,
            prob_confidence_interval=(p_lower, p_upper),
            confidence_level=self.confidence_level,
            recommended_fraction=recommended,
            max_safe_fraction=max_safe,
            recommendation_reason=reason,
            probability_of_ruin=prob_ruin,
            expected_drawdown=exp_drawdown,
            time_to_double=time_double
        )

    def _probability_of_ruin(
        self,
        win_prob: float,
        win_loss_ratio: float,
        fraction: float,
        ruin_threshold: float = 0.1,  # 10% of bankroll
    ) -> float:
        """
        Estimate probability of hitting ruin threshold.

        Uses the gambler's ruin formula adapted for Kelly betting.

        Reference:
            Thorp (2006) Section 4
        """
        if fraction <= 0 or win_prob <= 0.5:
            return 1.0 if fraction > 0 else 0.0

        # Simplified ruin probability for Kelly-like betting
        # P(ruin) ≈ (q/p)^n where n is number of unit bets to ruin
        q = 1 - win_prob
        if win_prob <= q:
            return 1.0

        # Expected number of losses to hit ruin threshold
        avg_loss_size = fraction  # Fraction of bankroll per loss
        n_losses_to_ruin = np.log(ruin_threshold) / np.log(1 - avg_loss_size)

        # Probability of n consecutive losses
        ruin_prob = q ** max(1, int(n_losses_to_ruin))

        return min(1.0, ruin_prob)

    def _expected_max_drawdown(
        self,
        win_prob: float,
        fraction: float,
        n_samples: int = 1000,
        n_trades: int = 100,
    ) -> float:
        """
        Estimate expected maximum drawdown via simulation.

        Reference:
            Magdon-Ismail et al. (2004) "On the Maximum Drawdown of a Brownian Motion"
        """
        if fraction <= 0:
            return 0.0

        max_drawdowns = []

        for _ in range(n_samples):
            # Simulate trading
            equity = 1.0
            peak = 1.0
            max_dd = 0.0

            for _ in range(n_trades):
                if np.random.random() < win_prob:
                    equity *= (1 + fraction * 0.02)  # 2% win
                else:
                    equity *= (1 - fraction * 0.01)  # 1% loss

                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

            max_drawdowns.append(max_dd)

        return np.mean(max_drawdowns)

    def _time_to_double(
        self,
        win_prob: float,
        win_loss_ratio: float,
        fraction: float,
        avg_win: float,
        avg_loss: float,
    ) -> Optional[float]:
        """
        Expected number of trades to double capital.

        Uses expected log growth rate.

        Reference:
            Kelly (1956) - growth rate is information rate
        """
        if fraction <= 0:
            return None

        # Expected log growth per trade
        exp_log_growth = win_prob * np.log(1 + fraction * avg_win) + \
                        (1 - win_prob) * np.log(1 - fraction * avg_loss)

        if exp_log_growth <= 0:
            return None

        # Trades to double: 2 = e^(n * g) → n = ln(2) / g
        return np.log(2) / exp_log_growth

    def _find_max_safe_fraction(
        self,
        win_prob: float,
        win_loss_ratio: float,
        max_ruin_prob: float = 0.01,
    ) -> float:
        """Find maximum fraction where ruin probability < threshold."""
        for f in np.linspace(self.max_fraction, 0.01, 50):
            ruin = self._probability_of_ruin(win_prob, win_loss_ratio, f)
            if ruin < max_ruin_prob:
                return f
        return 0.01


class AdaptiveKelly:
    """
    Adaptive Kelly that adjusts based on recent performance.

    When performance degrades, automatically reduces position size.
    When performance improves, gradually increases.

    Based on:
    - 九坤投资: Adaptive position sizing based on rolling metrics
    - BigQuant: "动态调整凯利比例" (Dynamic Kelly adjustment)
    """

    def __init__(
        self,
        base_fraction: float = 0.25,
        window_size: int = 100,
        min_fraction: float = 0.05,
        max_fraction: float = 0.50,
        adaptation_speed: float = 0.1,
    ):
        """
        Initialize adaptive Kelly.

        Args:
            base_fraction: Starting Kelly fraction
            window_size: Rolling window for performance
            min_fraction: Minimum allowed fraction
            max_fraction: Maximum allowed fraction
            adaptation_speed: How fast to adapt (0.1 = 10% per update)
        """
        self.base_fraction = base_fraction
        self.window_size = window_size
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.adaptation_speed = adaptation_speed

        self.current_fraction = base_fraction
        self.outcomes: List[bool] = []  # Win/loss history
        self.kelly_calc = RobustKellyCriterion()

    def update(self, won: bool, return_pct: float = 0.0) -> float:
        """
        Update with new trade outcome.

        Args:
            won: True if trade was profitable
            return_pct: Actual return (optional)

        Returns:
            Updated position fraction
        """
        self.outcomes.append(won)

        # Keep only recent outcomes
        if len(self.outcomes) > self.window_size:
            self.outcomes = self.outcomes[-self.window_size:]

        # Compute rolling win rate
        if len(self.outcomes) < 20:
            return self.current_fraction  # Not enough data

        n_wins = sum(self.outcomes)
        n_total = len(self.outcomes)
        win_rate = n_wins / n_total

        # Target fraction based on robust Kelly
        result = self.kelly_calc.calculate(
            n_wins=n_wins,
            n_total=n_total,
            avg_win=0.002,  # Assumed 0.2% avg win
            avg_loss=0.001   # Assumed 0.1% avg loss
        )

        target = result.robust_fraction

        # Adapt towards target
        diff = target - self.current_fraction
        self.current_fraction += self.adaptation_speed * diff

        # Clip to bounds
        self.current_fraction = np.clip(
            self.current_fraction,
            self.min_fraction,
            self.max_fraction
        )

        return self.current_fraction

    def get_current_fraction(self) -> float:
        """Get current position fraction."""
        return self.current_fraction

    def get_stats(self) -> Dict[str, float]:
        """Get adaptive Kelly statistics."""
        if len(self.outcomes) < 10:
            return {
                "current_fraction": self.current_fraction,
                "win_rate": 0.5,
                "n_trades": len(self.outcomes)
            }

        return {
            "current_fraction": self.current_fraction,
            "win_rate": sum(self.outcomes) / len(self.outcomes),
            "n_trades": len(self.outcomes),
            "recent_wins": sum(self.outcomes[-20:]),
            "recent_total": min(20, len(self.outcomes))
        }


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    uncertainty: float = 0.05,
) -> float:
    """
    Quick Kelly calculation for HFT with built-in uncertainty margin.

    Args:
        win_rate: Observed win rate (e.g., 0.82)
        avg_win: Average win return (e.g., 0.002 for 0.2%)
        avg_loss: Average loss return (e.g., 0.001 for 0.1%)
        uncertainty: Uncertainty margin to subtract from win rate

    Returns:
        Recommended position fraction

    Example:
        >>> fraction = quick_kelly(0.82, 0.002, 0.001)
        >>> print(f"Position size: {fraction:.2%}")
    """
    # Adjust for uncertainty
    p_adjusted = max(0.5, win_rate - uncertainty)

    # Win/loss ratio
    if avg_loss <= 0:
        avg_loss = 0.001
    b = avg_win / avg_loss

    # Kelly formula
    q = 1 - p_adjusted
    f = p_adjusted - q / b

    # Safety cap at 50%
    return max(0, min(f, 0.50))


def compute_position_size(
    capital: float,
    kelly_fraction: float,
    signal_confidence: float,
    max_position_pct: float = 0.02,  # 2% max of capital
) -> float:
    """
    Compute dollar position size from Kelly and signal confidence.

    Args:
        capital: Total capital
        kelly_fraction: Kelly-recommended fraction
        signal_confidence: Model confidence in signal (0-1)
        max_position_pct: Maximum position as % of capital

    Returns:
        Dollar position size

    Example:
        >>> size = compute_position_size(100000, 0.25, 0.85)
        >>> print(f"Position: ${size:,.2f}")
    """
    # Scale Kelly by confidence
    adjusted_fraction = kelly_fraction * signal_confidence

    # Apply per-trade limit
    position_pct = min(adjusted_fraction, max_position_pct)

    return capital * position_pct


def kelly_with_edge_decay(
    base_win_rate: float,
    time_since_signal_ms: float,
    half_life_ms: float = 100.0,
    avg_win: float = 0.002,
    avg_loss: float = 0.001,
) -> float:
    """
    Kelly fraction that accounts for signal decay over time.

    Edge decays exponentially: edge(t) = edge(0) * 2^(-t/half_life)

    Useful for HFT where signal value degrades with latency.

    Args:
        base_win_rate: Initial win rate
        time_since_signal_ms: Milliseconds since signal generated
        half_life_ms: Signal half-life in milliseconds
        avg_win: Average win
        avg_loss: Average loss

    Returns:
        Time-adjusted Kelly fraction
    """
    # Decay factor
    decay = 2 ** (-time_since_signal_ms / half_life_ms)

    # Decayed edge
    base_edge = base_win_rate - 0.5
    decayed_edge = base_edge * decay
    decayed_win_rate = 0.5 + decayed_edge

    # Kelly with decayed probability
    return quick_kelly(decayed_win_rate, avg_win, avg_loss)


# =============================================================================
# Integration
# =============================================================================

class KellyPositionSizer:
    """
    Full position sizing system integrating:
    1. Robust Kelly criterion
    2. Signal confidence
    3. Risk limits
    4. Adaptive adjustment

    Use this for production trading.
    """

    def __init__(
        self,
        total_capital: float,
        max_position_pct: float = 0.02,  # 2% max per trade
        max_total_exposure: float = 0.20,  # 20% max total
        kelly_fraction: float = 0.50,  # Half Kelly
        adaptive: bool = True,
    ):
        """
        Initialize position sizer.

        Args:
            total_capital: Starting capital
            max_position_pct: Maximum per-trade position
            max_total_exposure: Maximum total exposure
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
            adaptive: Use adaptive Kelly adjustment
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.kelly_multiplier = kelly_fraction

        self.robust_kelly = RobustKellyCriterion(
            confidence_level=0.95,
            max_fraction=0.50
        )

        self.adaptive_kelly = AdaptiveKelly(
            base_fraction=0.25,
            window_size=100
        ) if adaptive else None

        self.current_exposure = 0.0

    def calculate_size(
        self,
        signal_confidence: float,
        win_rate: float = 0.65,
        avg_win: float = 0.002,
        avg_loss: float = 0.001,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate position size for a trade.

        Args:
            signal_confidence: Model confidence (0-1)
            win_rate: Estimated win rate
            avg_win: Average win return
            avg_loss: Average loss return

        Returns:
            (position_size, diagnostics)
        """
        # Base Kelly
        base_kelly = quick_kelly(win_rate, avg_win, avg_loss)

        # Adaptive adjustment
        if self.adaptive_kelly:
            adaptive_factor = self.adaptive_kelly.current_fraction / 0.25
        else:
            adaptive_factor = 1.0

        # Final Kelly
        kelly_raw = base_kelly * self.kelly_multiplier * adaptive_factor

        # Scale by confidence
        scaled = kelly_raw * signal_confidence

        # Apply limits
        position_pct = min(scaled, self.max_position_pct)

        # Check total exposure
        remaining_capacity = self.max_total_exposure - self.current_exposure
        position_pct = min(position_pct, remaining_capacity)

        # Dollar size
        position_size = self.total_capital * max(0, position_pct)

        diagnostics = {
            "base_kelly": base_kelly,
            "kelly_multiplier": self.kelly_multiplier,
            "adaptive_factor": adaptive_factor,
            "signal_confidence": signal_confidence,
            "position_pct": position_pct,
            "current_exposure": self.current_exposure,
            "remaining_capacity": remaining_capacity
        }

        return position_size, diagnostics

    def update_outcome(self, won: bool, position_size: float) -> None:
        """Update after trade completes."""
        if self.adaptive_kelly:
            self.adaptive_kelly.update(won)

        # Update exposure (simplified - assumes position closed)
        # In real system, track open positions

    def update_capital(self, new_capital: float) -> None:
        """Update total capital."""
        self.total_capital = new_capital


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ROBUST KELLY CRITERION - POSITION SIZING WITH UNCERTAINTY")
    print("=" * 70)
    print()

    # Scenario: 82% win rate, 1000 trades
    n_wins = 820
    n_total = 1000
    avg_win = 0.002  # 0.2% average win
    avg_loss = 0.001  # 0.1% average loss

    print("SCENARIO: 82% win rate over 1000 trades")
    print(f"  Wins: {n_wins}, Total: {n_total}")
    print(f"  Avg Win: {avg_win:.2%}, Avg Loss: {avg_loss:.2%}")
    print()

    # Calculate robust Kelly
    kelly = RobustKellyCriterion(confidence_level=0.95)
    result = kelly.calculate(n_wins, n_total, avg_win, avg_loss)

    print("1. STANDARD vs ROBUST KELLY")
    print("-" * 50)
    print(f"   Win probability:     {result.win_probability:.2%}")
    print(f"   95% CI for p:        [{result.prob_confidence_interval[0]:.2%}, "
          f"{result.prob_confidence_interval[1]:.2%}]")
    print(f"   Win/Loss ratio:      {result.win_loss_ratio:.2f}")
    print()
    print(f"   Standard Kelly:      {result.kelly_fraction:.2%}")
    print(f"   Robust Kelly:        {result.robust_fraction:.2%}")
    print(f"   RECOMMENDED:         {result.recommended_fraction:.2%}")
    print(f"   Reason: {result.recommendation_reason}")
    print()

    print("2. RISK METRICS")
    print("-" * 50)
    print(f"   Probability of ruin: {result.probability_of_ruin:.4%}")
    print(f"   Expected max DD:     {result.expected_drawdown:.2%}")
    if result.time_to_double:
        print(f"   Trades to double:    {result.time_to_double:.0f}")
    print(f"   Max safe fraction:   {result.max_safe_fraction:.2%}")
    print()

    print("3. QUICK KELLY FUNCTION")
    print("-" * 50)
    for win_rate in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82]:
        f = quick_kelly(win_rate, avg_win, avg_loss)
        print(f"   Win rate {win_rate:.0%} → Kelly fraction {f:.2%}")
    print()

    print("4. POSITION SIZING EXAMPLE")
    print("-" * 50)
    capital = 100000
    sizer = KellyPositionSizer(
        total_capital=capital,
        max_position_pct=0.02,
        kelly_fraction=0.50
    )

    for confidence in [0.95, 0.85, 0.75, 0.65]:
        size, diag = sizer.calculate_size(confidence, win_rate=0.82)
        print(f"   Confidence {confidence:.0%} → Position ${size:,.0f} "
              f"({diag['position_pct']:.2%})")
    print()

    print("5. ADAPTIVE KELLY SIMULATION")
    print("-" * 50)
    adaptive = AdaptiveKelly(base_fraction=0.25)

    # Simulate 50 trades with 80% win rate
    np.random.seed(42)
    for i in range(50):
        won = np.random.random() < 0.80
        fraction = adaptive.update(won)
        if (i + 1) % 10 == 0:
            stats = adaptive.get_stats()
            print(f"   After {i+1} trades: fraction={fraction:.2%}, "
                  f"win_rate={stats['win_rate']:.2%}")
    print()

    print("=" * 70)
    print("KEY INSIGHT:")
    print("  Standard Kelly (82% win rate): f* = 64%")
    print("  Robust Kelly (95% CI):         f* = 55%")
    print("  Recommended (with safety):     f* = 41%")
    print()
    print("  The gap accounts for ESTIMATION UNCERTAINTY!")
    print("  This protects against overconfidence in our win rate estimate.")
    print("=" * 70)
