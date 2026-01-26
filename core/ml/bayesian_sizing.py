"""
Bayesian Position Sizing for HFT

Confidence-based position sizing with full uncertainty quantification.
Integrates all certainty modules into optimal position sizing.

Academic Citations:
- Kelly (1956): "A New Interpretation of Information Rate"
  Bell System Technical Journal - Foundation of optimal betting

- Thorp (1969): "Optimal Gambling Systems for Favorable Games"
  Review of the International Statistical Institute - Kelly extensions

- MacLean, Thorp, Ziemba (2011): "The Kelly Capital Growth Investment Criterion"
  World Scientific - Comprehensive Kelly treatment

- Hsieh & Barmish (2018): "On Kelly Betting: Some Limitations"
  arXiv:1801.03005 - Kelly with estimation error

- arXiv:2106.12582 (2021): "Bayesian Portfolio Selection with Uncertainty Aversion"
  Uncertainty-aware portfolio construction

Chinese Quant Application:
- 九坤投资: 基于置信度的仓位管理 (confidence-based position management)
- 幻方量化: 贝叶斯风险控制 (Bayesian risk control)
- 明汯投资: 不确定性量化投资 (uncertainty-quantified investment)

The Key Insight:
    Standard Kelly: f* = p - q/b
    Bayesian Kelly: f* = E[p|data] - E[q|data]/b, with uncertainty bounds

    Don't bet on your POINT ESTIMATE.
    Bet on the WORST CASE in your confidence interval.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from scipy import stats
import warnings


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    # Core sizing
    optimal_fraction: float  # Kelly optimal (aggressive)
    recommended_fraction: float  # Adjusted for uncertainty
    conservative_fraction: float  # Conservative sizing

    # In units
    position_size: float  # In base currency/units
    notional_value: float  # Dollar value

    # Risk metrics
    expected_return: float  # E[return] at this size
    return_std: float  # Std of return at this size
    max_drawdown_prob: float  # P(drawdown > threshold)

    # Uncertainty
    kelly_uncertainty: float  # Uncertainty in Kelly estimate
    confidence_level: float  # Confidence in this sizing


@dataclass
class CertaintyAdjustedSize:
    """
    Position size with all certainty factors integrated.

    Combines:
    - Model confidence (ensemble disagreement)
    - Prediction uncertainty (conformal bounds)
    - Edge certainty (deflated Sharpe)
    - Execution cost estimate
    - Adverse selection adjustment
    """

    # Base sizing
    base_kelly: float  # Kelly before adjustments
    final_size: float  # After all adjustments

    # Adjustment factors
    confidence_multiplier: float  # From model confidence
    uncertainty_multiplier: float  # From prediction uncertainty
    edge_certainty_multiplier: float  # From edge verification
    execution_cost_adjustment: float  # For execution costs
    adverse_selection_adjustment: float  # For AS

    # Breakdown
    adjustment_breakdown: Dict[str, float]


class BayesianKellyCalculator:
    """
    Bayesian Kelly Criterion with uncertainty quantification.

    Instead of f* = p - q/b, we compute:
    f* = E[p|data] - E[q|data]/b

    With posterior distribution over p, we can:
    1. Compute credible intervals for f*
    2. Use conservative quantile for sizing
    3. Account for estimation error

    Reference:
        Hsieh & Barmish (2018): Kelly with estimation error
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,  # Beta prior alpha
        prior_beta: float = 1.0,  # Beta prior beta (uniform prior)
        kelly_fraction: float = 0.25,  # Fractional Kelly
    ):
        """
        Initialize Bayesian Kelly calculator.

        Args:
            prior_alpha: Beta distribution prior α (wins before data)
            prior_beta: Beta distribution prior β (losses before data)
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.kelly_fraction = kelly_fraction

    def update_posterior(
        self,
        wins: int,
        losses: int,
    ) -> Tuple[float, float]:
        """
        Update Beta posterior with new data.

        Args:
            wins: Number of winning trades
            losses: Number of losing trades

        Returns:
            (posterior_alpha, posterior_beta)
        """
        post_alpha = self.prior_alpha + wins
        post_beta = self.prior_beta + losses
        return post_alpha, post_beta

    def compute_kelly_distribution(
        self,
        wins: int,
        losses: int,
        win_loss_ratio: float,
        n_samples: int = 10000,
    ) -> np.ndarray:
        """
        Sample from posterior distribution of Kelly fraction.

        Args:
            wins: Number of wins
            losses: Number of losses
            win_loss_ratio: Average win / average loss
            n_samples: Monte Carlo samples

        Returns:
            Array of Kelly fraction samples
        """
        post_alpha, post_beta = self.update_posterior(wins, losses)

        # Sample win probabilities from posterior
        p_samples = np.random.beta(post_alpha, post_beta, n_samples)
        q_samples = 1 - p_samples

        # Kelly fraction for each sample
        b = win_loss_ratio
        kelly_samples = p_samples - q_samples / b

        return kelly_samples

    def compute_sizing(
        self,
        wins: int,
        losses: int,
        win_loss_ratio: float,
        capital: float,
        confidence: float = 0.95,
    ) -> PositionSizeResult:
        """
        Compute position size with Bayesian uncertainty.

        Args:
            wins: Number of winning trades
            losses: Number of losing trades
            win_loss_ratio: Average win / average loss
            capital: Available capital
            confidence: Confidence level for conservative sizing

        Returns:
            PositionSizeResult with sizing and uncertainty
        """
        # Get Kelly distribution
        kelly_samples = self.compute_kelly_distribution(
            wins, losses, win_loss_ratio
        )

        # Point estimate (posterior mean)
        post_alpha, post_beta = self.update_posterior(wins, losses)
        p_mean = post_alpha / (post_alpha + post_beta)
        q_mean = 1 - p_mean
        kelly_mean = p_mean - q_mean / win_loss_ratio

        # Conservative estimate (lower quantile)
        kelly_conservative = np.percentile(kelly_samples, (1 - confidence) * 100)
        kelly_conservative = max(kelly_conservative, 0)

        # Apply fractional Kelly
        optimal = kelly_mean * self.kelly_fraction
        recommended = max(np.median(kelly_samples), 0) * self.kelly_fraction
        conservative = kelly_conservative * self.kelly_fraction

        # Position sizes
        position = conservative * capital
        notional = position  # Same for single asset

        # Risk metrics
        expected_ret = conservative * kelly_mean  # Approximate
        ret_std = conservative * np.std(kelly_samples)
        max_dd_prob = 1 - stats.norm.cdf(0, kelly_mean, np.std(kelly_samples))

        return PositionSizeResult(
            optimal_fraction=optimal,
            recommended_fraction=recommended,
            conservative_fraction=conservative,
            position_size=position,
            notional_value=notional,
            expected_return=expected_ret,
            return_std=ret_std,
            max_drawdown_prob=max_dd_prob,
            kelly_uncertainty=np.std(kelly_samples),
            confidence_level=confidence,
        )


class UncertaintyAwareSizer:
    """
    Position sizing that integrates all uncertainty sources.

    Combines:
    1. Model confidence (ensemble disagreement)
    2. Prediction uncertainty (conformal bounds)
    3. Edge certainty (statistical verification)
    4. Execution costs
    5. Adverse selection

    Reference:
        arXiv:2106.12582 - Bayesian Portfolio with Uncertainty Aversion
    """

    def __init__(
        self,
        base_kelly_fraction: float = 0.25,
        max_position_pct: float = 0.10,  # Max 10% per position
        min_confidence_to_trade: float = 0.6,
    ):
        """
        Initialize uncertainty-aware sizer.

        Args:
            base_kelly_fraction: Base fractional Kelly
            max_position_pct: Maximum position as % of capital
            min_confidence_to_trade: Minimum confidence to trade
        """
        self.base_kelly = base_kelly_fraction
        self.max_position = max_position_pct
        self.min_confidence = min_confidence_to_trade

    def compute_size(
        self,
        capital: float,
        win_rate: float,
        win_loss_ratio: float,
        model_confidence: float,  # 0-1, from ensemble
        prediction_uncertainty: float,  # Std of prediction
        edge_certainty: float,  # 0-1, from edge verification
        execution_cost_bps: float,  # Expected execution cost
        adverse_selection_adj: float = 1.0,  # Multiplier
    ) -> CertaintyAdjustedSize:
        """
        Compute position size with all certainty factors.

        Args:
            capital: Available capital
            win_rate: Estimated win rate
            win_loss_ratio: Avg win / avg loss
            model_confidence: Ensemble agreement (0-1)
            prediction_uncertainty: Prediction std
            edge_certainty: Edge verification score (0-1)
            execution_cost_bps: Expected slippage
            adverse_selection_adj: AS multiplier (<1)

        Returns:
            CertaintyAdjustedSize with full breakdown
        """
        # Base Kelly
        q = 1 - win_rate
        base_kelly = win_rate - q / win_loss_ratio
        base_kelly = max(base_kelly, 0)

        # Apply fractional Kelly
        kelly = base_kelly * self.base_kelly

        # Adjustment multipliers

        # 1. Model confidence adjustment
        # Low confidence = reduce size
        conf_mult = model_confidence ** 0.5  # Square root for smoother

        # 2. Prediction uncertainty adjustment
        # High uncertainty = reduce size
        # Assume "normal" uncertainty is ~0.1
        uncertainty_mult = 1 / (1 + prediction_uncertainty / 0.1)

        # 3. Edge certainty adjustment
        # Low certainty = reduce size
        edge_mult = edge_certainty ** 0.5

        # 4. Execution cost adjustment
        # Reduce size if costs eat into edge
        edge_bps = (win_rate * win_loss_ratio - q) * 100 * 100  # Edge in bps
        if edge_bps > 0:
            cost_adj = max(1 - execution_cost_bps / edge_bps, 0.5)
        else:
            cost_adj = 0.5

        # 5. Adverse selection adjustment (passed in)
        as_adj = adverse_selection_adj

        # Combine adjustments
        total_mult = conf_mult * uncertainty_mult * edge_mult * cost_adj * as_adj

        # Final size
        final_kelly = kelly * total_mult

        # Cap at max position
        final_kelly = min(final_kelly, self.max_position)

        # Size in capital terms
        final_size = final_kelly * capital

        # Check minimum confidence
        if model_confidence < self.min_confidence:
            final_size = 0
            final_kelly = 0

        return CertaintyAdjustedSize(
            base_kelly=kelly,
            final_size=final_size,
            confidence_multiplier=conf_mult,
            uncertainty_multiplier=uncertainty_mult,
            edge_certainty_multiplier=edge_mult,
            execution_cost_adjustment=cost_adj,
            adverse_selection_adjustment=as_adj,
            adjustment_breakdown={
                'base_kelly': kelly,
                'after_confidence': kelly * conf_mult,
                'after_uncertainty': kelly * conf_mult * uncertainty_mult,
                'after_edge': kelly * conf_mult * uncertainty_mult * edge_mult,
                'after_costs': kelly * conf_mult * uncertainty_mult * edge_mult * cost_adj,
                'final': final_kelly,
            },
        )


class AdaptivePositionSizer:
    """
    Adaptive position sizing based on recent performance.

    Increases size when confident, decreases when uncertain.

    Reference:
        九坤投资: 动态仓位管理 (dynamic position management)
    """

    def __init__(
        self,
        window: int = 50,
        base_size: float = 0.02,  # 2% base position
        min_size: float = 0.005,  # 0.5% minimum
        max_size: float = 0.05,  # 5% maximum
    ):
        """
        Initialize adaptive sizer.

        Args:
            window: Lookback window for performance
            base_size: Base position size
            min_size: Minimum position size
            max_size: Maximum position size
        """
        self.window = window
        self.base = base_size
        self.min_size = min_size
        self.max_size = max_size

        self._pnl_history: List[float] = []
        self._confidence_history: List[float] = []

    def record_trade(
        self,
        pnl: float,
        confidence: float,
    ) -> None:
        """
        Record a trade result.

        Args:
            pnl: Profit/loss (positive = profit)
            confidence: Confidence at time of trade
        """
        self._pnl_history.append(pnl)
        self._confidence_history.append(confidence)

        # Trim to window
        if len(self._pnl_history) > self.window * 2:
            self._pnl_history = self._pnl_history[-self.window:]
            self._confidence_history = self._confidence_history[-self.window:]

    def get_size_multiplier(self) -> float:
        """
        Get current size multiplier based on recent performance.

        Returns:
            Multiplier (1.0 = base size)
        """
        if len(self._pnl_history) < 10:
            return 1.0

        recent_pnl = np.array(self._pnl_history[-self.window:])
        recent_conf = np.array(self._confidence_history[-self.window:])

        # Win rate
        wins = np.sum(recent_pnl > 0)
        total = len(recent_pnl)
        win_rate = wins / total

        # Sharpe-like metric
        if np.std(recent_pnl) > 0:
            sharpe = np.mean(recent_pnl) / np.std(recent_pnl)
        else:
            sharpe = 0

        # Confidence calibration
        # How often were high-confidence trades winners?
        high_conf_mask = recent_conf > 0.7
        if np.sum(high_conf_mask) > 5:
            high_conf_wr = np.mean(recent_pnl[high_conf_mask] > 0)
        else:
            high_conf_wr = win_rate

        # Combine into multiplier
        # Good performance = larger size
        wr_factor = (win_rate - 0.5) * 2  # -1 to +1
        sharpe_factor = min(max(sharpe, -1), 1)  # Clamp
        calib_factor = high_conf_wr - 0.5  # -0.5 to +0.5

        multiplier = 1.0 + 0.3 * wr_factor + 0.4 * sharpe_factor + 0.3 * calib_factor

        return max(min(multiplier, 2.0), 0.5)  # Clamp to 0.5x - 2x

    def compute_size(
        self,
        capital: float,
        current_confidence: float,
    ) -> float:
        """
        Compute adaptive position size.

        Args:
            capital: Available capital
            current_confidence: Current trade confidence

        Returns:
            Position size in capital units
        """
        # Base size
        size = self.base * capital

        # Apply adaptive multiplier
        mult = self.get_size_multiplier()
        size *= mult

        # Scale by current confidence
        conf_scale = 0.5 + 0.5 * current_confidence
        size *= conf_scale

        # Clamp to limits
        size = max(self.min_size * capital, size)
        size = min(self.max_size * capital, size)

        return size


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_position_size(
    capital: float,
    win_rate: float,
    win_loss_ratio: float,
    confidence: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Ultra-fast position sizing for HFT.

    Args:
        capital: Available capital
        win_rate: Estimated win rate
        win_loss_ratio: Avg win / avg loss
        confidence: Current confidence (0-1)
        kelly_fraction: Fractional Kelly to use

    Returns:
        Position size in capital units
    """
    # Kelly
    q = 1 - win_rate
    kelly = max(win_rate - q / win_loss_ratio, 0)

    # Apply fraction and confidence
    size = kelly * kelly_fraction * confidence * capital

    # Cap at 10% max
    return min(size, 0.10 * capital)


def certainty_weighted_size(
    base_size: float,
    model_confidence: float,
    edge_pvalue: float,  # From deflated Sharpe test
    conformal_coverage: float,  # Conformal prediction coverage
) -> float:
    """
    Adjust size based on certainty metrics.

    Args:
        base_size: Base position size
        model_confidence: Ensemble agreement (0-1)
        edge_pvalue: P-value from edge test (lower = more certain)
        conformal_coverage: Achieved coverage (should be ~0.95)

    Returns:
        Adjusted position size
    """
    # Model confidence adjustment
    conf_adj = model_confidence ** 0.5

    # Edge p-value adjustment
    # p < 0.01 = very certain, p > 0.1 = uncertain
    if edge_pvalue < 0.01:
        edge_adj = 1.0
    elif edge_pvalue < 0.05:
        edge_adj = 0.8
    elif edge_pvalue < 0.10:
        edge_adj = 0.6
    else:
        edge_adj = 0.4

    # Conformal coverage adjustment
    # Good coverage = well-calibrated = more certain
    if conformal_coverage >= 0.93:
        coverage_adj = 1.0
    elif conformal_coverage >= 0.90:
        coverage_adj = 0.8
    else:
        coverage_adj = 0.6

    # Combine
    total_adj = conf_adj * edge_adj * coverage_adj

    return base_size * total_adj


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BAYESIAN POSITION SIZING FOR HFT")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Kelly (1956): Optimal betting foundation")
    print("  - Hsieh & Barmish (2018): Kelly with estimation error")
    print("  - 九坤投资: 基于置信度的仓位管理")
    print()

    # Bayesian Kelly
    print("BAYESIAN KELLY CRITERION")
    print("-" * 50)

    bk = BayesianKellyCalculator(kelly_fraction=0.25)

    # With 82% win rate from 100 trades
    result = bk.compute_sizing(
        wins=82,
        losses=18,
        win_loss_ratio=1.5,
        capital=100000,
        confidence=0.95,
    )

    print(f"Data: 82 wins, 18 losses (82% win rate)")
    print(f"Win/Loss ratio: 1.5")
    print()
    print(f"Optimal Kelly fraction: {result.optimal_fraction:.2%}")
    print(f"Recommended fraction: {result.recommended_fraction:.2%}")
    print(f"Conservative (95% CI): {result.conservative_fraction:.2%}")
    print()
    print(f"Position size ($100k capital): ${result.position_size:,.0f}")
    print(f"Kelly uncertainty: ±{result.kelly_uncertainty:.2%}")
    print()

    # Uncertainty-aware sizing
    print("UNCERTAINTY-AWARE SIZING")
    print("-" * 50)

    uas = UncertaintyAwareSizer()

    sized = uas.compute_size(
        capital=100000,
        win_rate=0.82,
        win_loss_ratio=1.5,
        model_confidence=0.85,
        prediction_uncertainty=0.15,
        edge_certainty=0.90,
        execution_cost_bps=0.5,
        adverse_selection_adj=0.95,
    )

    print(f"Base Kelly: {sized.base_kelly:.2%}")
    print()
    print("Adjustment factors:")
    print(f"  Model confidence: {sized.confidence_multiplier:.2f}")
    print(f"  Uncertainty: {sized.uncertainty_multiplier:.2f}")
    print(f"  Edge certainty: {sized.edge_certainty_multiplier:.2f}")
    print(f"  Execution costs: {sized.execution_cost_adjustment:.2f}")
    print(f"  Adverse selection: {sized.adverse_selection_adjustment:.2f}")
    print()
    print(f"Final position: ${sized.final_size:,.0f}")
    print()

    # Breakdown
    print("Step-by-step breakdown:")
    for step, value in sized.adjustment_breakdown.items():
        print(f"  {step}: {value:.4f}")

    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("  Kelly says bet 25% at 82% win rate.")
    print("  But that's for KNOWN probabilities.")
    print("  With ESTIMATED probabilities, uncertainty matters!")
    print()
    print("  Conservative sizing = bet on WORST CASE in confidence interval")
    print("  This is how you maintain certainty while maximizing returns.")
    print("=" * 70)
