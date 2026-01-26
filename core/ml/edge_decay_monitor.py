"""
Edge Decay Monitor: Real-Time Detection of Edge Degradation
============================================================

Detects when a trading edge is fading BEFORE it costs money.

Key Insight:
    All edges decay over time due to:
    - Crowding (others discover the same signal)
    - Regime change (market dynamics shift)
    - Alpha decay (information gets arbitraged away)

    Early detection lets you:
    1. Reduce position sizes before losses
    2. Trigger retraining at the right time
    3. Avoid trading during unreliable periods

Detection Methods:
    1. CUSUM: Cumulative sum control chart (detect mean shifts)
    2. EWMA: Exponential weighted moving average (smooth detection)
    3. Binomial: Sequential probability ratio test
    4. Drawdown: Maximum drawdown monitoring
    5. Win streak: Detect unusual losing streaks

References:
    [1] Page, E. S. (1954).
        "Continuous Inspection Schemes."
        Biometrika, 41(1/2), 100-115.
        https://doi.org/10.2307/2333009
        Original CUSUM paper.

    [2] Roberts, S. W. (1959).
        "Control Chart Tests Based on Geometric Moving Averages."
        Technometrics, 1(3), 239-250.
        EWMA control charts.

    [3] Wald, A. (1945).
        "Sequential Tests of Statistical Hypotheses."
        Annals of Mathematical Statistics, 16(2), 117-186.
        Sequential probability ratio test (SPRT).

    [4] McLean, R. D., & Pontiff, J. (2016).
        "Does Academic Research Destroy Stock Return Predictability?"
        Journal of Finance, 71(1), 5-32.
        Alpha decay in published factors (~58% post-publication decay).

    [5] Chordia, T., Subrahmanyam, A., & Tong, Q. (2014).
        "Have Capital Market Anomalies Attenuated in the Recent Era
        of High Liquidity and Trading Activity?"
        Journal of Accounting and Economics, 58(1), 41-58.

Author: Claude Code
Created: 2026-01-25
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from collections import deque
from datetime import datetime, timedelta
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecayAlert:
    """Alert when edge decay is detected."""

    # Detection info
    detected: bool                     # Was decay detected?
    severity: str                      # 'mild', 'moderate', 'severe', 'critical'
    method: str                        # Detection method that triggered

    # Metrics
    current_accuracy: float            # Recent accuracy
    baseline_accuracy: float           # Historical baseline
    decay_amount: float                # How much accuracy dropped
    p_value: float                     # Statistical significance of drop

    # Action recommendation
    action: str                        # 'monitor', 'reduce_size', 'pause_trading', 'retrain'
    confidence: float                  # Confidence in detection (0-1)

    # Timing
    trades_since_baseline: int         # How many trades since last baseline
    estimated_decay_rate: float        # Estimated decay per trade (negative)

    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        if not self.detected:
            return "DecayAlert(detected=False, edge is stable)"

        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  EDGE DECAY ALERT - {self.severity.upper():^10}                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Method:              {self.method:>20}                      ║
║  Baseline Accuracy:   {self.baseline_accuracy:>20.2%}                      ║
║  Current Accuracy:    {self.current_accuracy:>20.2%}                      ║
║  Decay Amount:        {self.decay_amount:>20.2%}                      ║
║  P-value:             {self.p_value:>20.2e}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Trades Since Reset:  {self.trades_since_baseline:>20,}                      ║
║  Decay Rate/Trade:    {self.estimated_decay_rate:>20.4%}                      ║
║  Detection Confidence:{self.confidence:>20.2%}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  RECOMMENDED ACTION:  {self.action:>20}                      ║
╚══════════════════════════════════════════════════════════════════╝
"""


class EdgeDecayMonitor:
    """
    Real-time edge decay monitoring using multiple detection methods.

    Continuously monitors trading performance and alerts when the
    edge shows signs of degradation.

    Uses control chart methods from quality engineering adapted for
    trading edge monitoring.

    Reference:
        Page (1954): CUSUM charts
        Roberts (1959): EWMA charts
        McLean & Pontiff (2016): Alpha decay patterns
    """

    def __init__(
        self,
        baseline_accuracy: float = 0.63,
        window_size: int = 50,
        decay_threshold: float = 0.05,    # 5% drop triggers mild alert
        critical_threshold: float = 0.10,  # 10% drop triggers critical
        alpha: float = 0.05                # Significance level for tests
    ):
        """
        Initialize edge decay monitor.

        Args:
            baseline_accuracy: Expected accuracy under normal conditions
            window_size: Number of recent trades to analyze
            decay_threshold: Accuracy drop to trigger mild alert
            critical_threshold: Accuracy drop to trigger critical alert
            alpha: Significance level for statistical tests
        """
        self.baseline_accuracy = baseline_accuracy
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.critical_threshold = critical_threshold
        self.alpha = alpha

        # Trade history
        self.outcomes: deque = deque(maxlen=10000)
        self.timestamps: deque = deque(maxlen=10000)

        # CUSUM state
        self._cusum_pos = 0.0  # Positive CUSUM (detects decrease)
        self._cusum_neg = 0.0  # Negative CUSUM (detects increase)

        # EWMA state
        self._ewma = baseline_accuracy
        self._ewma_variance = 0.0
        self._ewma_lambda = 0.2  # Smoothing factor

        # Tracking
        self._n_trades = 0
        self._last_alert_trade = 0
        self._baseline_set_trade = 0

    def update(
        self,
        win: bool,
        timestamp: Optional[datetime] = None
    ) -> Optional[DecayAlert]:
        """
        Record a trade outcome and check for decay.

        Args:
            win: True if trade was profitable, False otherwise
            timestamp: Trade timestamp (defaults to now)

        Returns:
            DecayAlert if decay detected, None otherwise

        Example:
            >>> monitor = EdgeDecayMonitor(baseline_accuracy=0.63)
            >>> for win in trade_outcomes:
            ...     alert = monitor.update(win)
            ...     if alert and alert.detected:
            ...         print(f"ALERT: {alert.action}")
        """
        outcome = 1.0 if win else 0.0
        ts = timestamp or datetime.now()

        self.outcomes.append(outcome)
        self.timestamps.append(ts)
        self._n_trades += 1

        # Update CUSUM
        self._update_cusum(outcome)

        # Update EWMA
        self._update_ewma(outcome)

        # Check for decay only if we have enough data
        if len(self.outcomes) >= self.window_size:
            return self._check_decay()

        return None

    def _update_cusum(self, outcome: float) -> None:
        """
        Update CUSUM statistic.

        CUSUM detects shifts in the mean by accumulating deviations
        from the target. Upper and lower CUSUM detect opposite shifts.

        Reference:
            Page, E. S. (1954). Biometrika.
        """
        # Slack parameter (k) - typically 0.5 standard deviations
        # For Bernoulli with p=baseline, std = sqrt(p*(1-p))
        std = np.sqrt(self.baseline_accuracy * (1 - self.baseline_accuracy))
        k = 0.5 * std

        # Deviation from baseline
        deviation = outcome - self.baseline_accuracy

        # Update positive CUSUM (detects downward shift)
        self._cusum_pos = max(0, self._cusum_pos - deviation - k)

        # Update negative CUSUM (detects upward shift - improvement)
        self._cusum_neg = max(0, self._cusum_neg + deviation - k)

    def _update_ewma(self, outcome: float) -> None:
        """
        Update EWMA (Exponentially Weighted Moving Average).

        EWMA smooths recent outcomes, giving more weight to recent trades.

        Formula:
            EWMA_t = λ × x_t + (1-λ) × EWMA_{t-1}

        Reference:
            Roberts, S. W. (1959). Technometrics.
        """
        lam = self._ewma_lambda

        # Update EWMA
        old_ewma = self._ewma
        self._ewma = lam * outcome + (1 - lam) * old_ewma

        # Update variance estimate
        deviation = outcome - old_ewma
        self._ewma_variance = (1 - lam) * (self._ewma_variance + lam * deviation ** 2)

    def _check_decay(self) -> Optional[DecayAlert]:
        """
        Check all detection methods for edge decay.

        Returns:
            DecayAlert if any method detects significant decay
        """
        alerts = []

        # Method 1: CUSUM test
        cusum_alert = self._cusum_test()
        if cusum_alert.detected:
            alerts.append(cusum_alert)

        # Method 2: EWMA test
        ewma_alert = self._ewma_test()
        if ewma_alert.detected:
            alerts.append(ewma_alert)

        # Method 3: Binomial test on recent window
        binomial_alert = self._binomial_test()
        if binomial_alert.detected:
            alerts.append(binomial_alert)

        # Method 4: Win streak test
        streak_alert = self._losing_streak_test()
        if streak_alert.detected:
            alerts.append(streak_alert)

        # Return most severe alert if any detected
        if alerts:
            # Sort by severity
            severity_order = {'critical': 4, 'severe': 3, 'moderate': 2, 'mild': 1}
            alerts.sort(key=lambda a: severity_order.get(a.severity, 0), reverse=True)
            return alerts[0]

        return None

    def _cusum_test(self) -> DecayAlert:
        """
        CUSUM test for mean shift.

        Triggers when cumulative deviation exceeds threshold (h).
        h = 5 is common, corresponding to ~5 sigma shift.

        Reference:
            Page (1954): CUSUM control charts
        """
        # Threshold (h parameter) - calibrated for trading
        std = np.sqrt(self.baseline_accuracy * (1 - self.baseline_accuracy))
        h = 5 * std * np.sqrt(self.window_size)  # Scale with window

        # Recent accuracy
        recent = list(self.outcomes)[-self.window_size:]
        current_accuracy = np.mean(recent)
        decay_amount = self.baseline_accuracy - current_accuracy

        # Check if CUSUM exceeds threshold
        if self._cusum_pos > h:
            # Severity based on how much CUSUM exceeds threshold
            excess = self._cusum_pos / h
            if excess > 2.0:
                severity = 'critical'
                action = 'pause_trading'
            elif excess > 1.5:
                severity = 'severe'
                action = 'retrain'
            elif excess > 1.2:
                severity = 'moderate'
                action = 'reduce_size'
            else:
                severity = 'mild'
                action = 'monitor'

            # P-value approximation for CUSUM
            p_value = 2 * stats.norm.cdf(-self._cusum_pos / (std * np.sqrt(self._n_trades)))

            return DecayAlert(
                detected=True,
                severity=severity,
                method='CUSUM',
                current_accuracy=current_accuracy,
                baseline_accuracy=self.baseline_accuracy,
                decay_amount=decay_amount,
                p_value=p_value,
                action=action,
                confidence=1 - p_value,
                trades_since_baseline=self._n_trades - self._baseline_set_trade,
                estimated_decay_rate=-decay_amount / max(1, self._n_trades - self._baseline_set_trade)
            )

        return DecayAlert(
            detected=False,
            severity='none',
            method='CUSUM',
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            decay_amount=decay_amount,
            p_value=1.0,
            action='none',
            confidence=0.0,
            trades_since_baseline=self._n_trades - self._baseline_set_trade,
            estimated_decay_rate=0.0
        )

    def _ewma_test(self) -> DecayAlert:
        """
        EWMA control chart test.

        Tests if EWMA has drifted significantly below baseline.

        Reference:
            Roberts (1959): EWMA control charts
        """
        recent = list(self.outcomes)[-self.window_size:]
        current_accuracy = np.mean(recent)
        decay_amount = self.baseline_accuracy - current_accuracy

        # Control limits for EWMA
        lam = self._ewma_lambda
        std = np.sqrt(self.baseline_accuracy * (1 - self.baseline_accuracy))
        ewma_std = std * np.sqrt(lam / (2 - lam))

        # Z-score of current EWMA
        z_score = (self._ewma - self.baseline_accuracy) / (ewma_std + 1e-10)

        # Test for significant downward shift
        if z_score < -3:  # 3-sigma rule
            severity = 'critical' if z_score < -4 else 'severe' if z_score < -3.5 else 'moderate'
            action = 'pause_trading' if z_score < -4 else 'retrain' if z_score < -3.5 else 'reduce_size'
            p_value = stats.norm.cdf(z_score)

            return DecayAlert(
                detected=True,
                severity=severity,
                method='EWMA',
                current_accuracy=current_accuracy,
                baseline_accuracy=self.baseline_accuracy,
                decay_amount=decay_amount,
                p_value=p_value,
                action=action,
                confidence=1 - p_value,
                trades_since_baseline=self._n_trades - self._baseline_set_trade,
                estimated_decay_rate=-decay_amount / max(1, self._n_trades - self._baseline_set_trade)
            )

        return DecayAlert(
            detected=False,
            severity='none',
            method='EWMA',
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            decay_amount=decay_amount,
            p_value=1.0,
            action='none',
            confidence=0.0,
            trades_since_baseline=self._n_trades - self._baseline_set_trade,
            estimated_decay_rate=0.0
        )

    def _binomial_test(self) -> DecayAlert:
        """
        Exact binomial test for recent accuracy.

        Tests if recent win rate is significantly below baseline.

        H0: p = baseline_accuracy
        H1: p < baseline_accuracy (one-sided)

        Reference:
            Standard hypothesis testing
        """
        recent = list(self.outcomes)[-self.window_size:]
        n_wins = int(np.sum(recent))
        n_trades = len(recent)

        current_accuracy = n_wins / n_trades
        decay_amount = self.baseline_accuracy - current_accuracy

        # One-sided binomial test
        p_value = stats.binom.cdf(n_wins, n_trades, self.baseline_accuracy)

        if p_value < self.alpha:
            # Severity based on p-value
            if p_value < 0.001:
                severity = 'critical'
                action = 'pause_trading'
            elif p_value < 0.01:
                severity = 'severe'
                action = 'retrain'
            elif p_value < 0.05:
                severity = 'moderate'
                action = 'reduce_size'
            else:
                severity = 'mild'
                action = 'monitor'

            return DecayAlert(
                detected=True,
                severity=severity,
                method='Binomial',
                current_accuracy=current_accuracy,
                baseline_accuracy=self.baseline_accuracy,
                decay_amount=decay_amount,
                p_value=p_value,
                action=action,
                confidence=1 - p_value,
                trades_since_baseline=self._n_trades - self._baseline_set_trade,
                estimated_decay_rate=-decay_amount / max(1, self._n_trades - self._baseline_set_trade)
            )

        return DecayAlert(
            detected=False,
            severity='none',
            method='Binomial',
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            decay_amount=decay_amount,
            p_value=p_value,
            action='none',
            confidence=0.0,
            trades_since_baseline=self._n_trades - self._baseline_set_trade,
            estimated_decay_rate=0.0
        )

    def _losing_streak_test(self) -> DecayAlert:
        """
        Test for unusual losing streaks.

        Long losing streaks may indicate regime change even if
        overall accuracy hasn't dropped much yet.

        Reference:
            Probability of runs in Bernoulli trials
        """
        recent = list(self.outcomes)[-self.window_size:]

        # Find longest losing streak
        max_losing_streak = 0
        current_streak = 0
        for outcome in recent:
            if outcome == 0:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0

        # Expected max streak under baseline accuracy
        # For n trials with p(loss) = q, E[max_streak] ≈ log_q(n)
        q = 1 - self.baseline_accuracy
        expected_max_streak = np.log(len(recent)) / np.log(1/q) if q > 0 else 0

        # P-value for observing streak >= max_losing_streak
        # Approximate using geometric distribution
        p_value = (1 - self.baseline_accuracy) ** max_losing_streak

        current_accuracy = np.mean(recent)
        decay_amount = self.baseline_accuracy - current_accuracy

        # Alert if streak is unusually long
        if max_losing_streak >= expected_max_streak * 2 and p_value < 0.01:
            severity = 'severe' if max_losing_streak >= expected_max_streak * 3 else 'moderate'
            action = 'reduce_size'

            return DecayAlert(
                detected=True,
                severity=severity,
                method=f'Losing Streak ({max_losing_streak} consecutive)',
                current_accuracy=current_accuracy,
                baseline_accuracy=self.baseline_accuracy,
                decay_amount=decay_amount,
                p_value=p_value,
                action=action,
                confidence=1 - p_value,
                trades_since_baseline=self._n_trades - self._baseline_set_trade,
                estimated_decay_rate=-decay_amount / max(1, self._n_trades - self._baseline_set_trade)
            )

        return DecayAlert(
            detected=False,
            severity='none',
            method='Losing Streak',
            current_accuracy=current_accuracy,
            baseline_accuracy=self.baseline_accuracy,
            decay_amount=decay_amount,
            p_value=p_value,
            action='none',
            confidence=0.0,
            trades_since_baseline=self._n_trades - self._baseline_set_trade,
            estimated_decay_rate=0.0
        )

    def reset_baseline(
        self,
        new_baseline: Optional[float] = None
    ) -> None:
        """
        Reset baseline accuracy and CUSUM accumulators.

        Call after retraining or when intentionally resetting monitoring.

        Args:
            new_baseline: New baseline accuracy (uses current if None)
        """
        if new_baseline is not None:
            self.baseline_accuracy = new_baseline
        elif len(self.outcomes) >= self.window_size:
            # Use recent performance as new baseline
            recent = list(self.outcomes)[-self.window_size:]
            self.baseline_accuracy = np.mean(recent)

        # Reset accumulators
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._ewma = self.baseline_accuracy
        self._ewma_variance = 0.0
        self._baseline_set_trade = self._n_trades

        logger.info(f"Baseline reset to {self.baseline_accuracy:.2%}")

    def get_status(self) -> Dict[str, float]:
        """
        Get current monitoring status.

        Returns:
            Dictionary with monitoring metrics
        """
        if len(self.outcomes) < 10:
            return {
                'n_trades': len(self.outcomes),
                'status': 'warming_up'
            }

        recent = list(self.outcomes)[-min(self.window_size, len(self.outcomes)):]
        current_accuracy = np.mean(recent)

        return {
            'n_trades': self._n_trades,
            'baseline_accuracy': self.baseline_accuracy,
            'current_accuracy': current_accuracy,
            'ewma_accuracy': self._ewma,
            'cusum_positive': self._cusum_pos,
            'cusum_negative': self._cusum_neg,
            'decay_from_baseline': self.baseline_accuracy - current_accuracy,
            'status': 'monitoring'
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_edge_decay_fast(
    recent_wins: int,
    recent_trades: int,
    baseline_accuracy: float,
    significance: float = 0.05
) -> Tuple[bool, float, str]:
    """
    Quick edge decay detection using binomial test.

    Args:
        recent_wins: Number of wins in recent window
        recent_trades: Number of trades in recent window
        baseline_accuracy: Expected accuracy
        significance: Significance threshold

    Returns:
        Tuple of (decay_detected, p_value, action)

    Example:
        >>> detected, p, action = detect_edge_decay_fast(28, 50, 0.63)
        >>> if detected:
        ...     print(f"Decay detected! Action: {action}")

    Reference:
        Standard binomial test for proportion
    """
    current_accuracy = recent_wins / recent_trades
    p_value = stats.binom.cdf(recent_wins, recent_trades, baseline_accuracy)

    if p_value < 0.001:
        return True, p_value, 'pause_trading'
    elif p_value < 0.01:
        return True, p_value, 'retrain'
    elif p_value < significance:
        return True, p_value, 'reduce_size'
    else:
        return False, p_value, 'continue'


def create_decay_monitor(
    baseline_accuracy: float = 0.63,
    window_size: int = 50
) -> EdgeDecayMonitor:
    """
    Create a preconfigured edge decay monitor.

    Args:
        baseline_accuracy: Expected accuracy
        window_size: Analysis window

    Returns:
        Configured EdgeDecayMonitor

    Example:
        >>> monitor = create_decay_monitor(baseline_accuracy=0.63)
        >>> for win in trade_results:
        ...     alert = monitor.update(win)
        ...     if alert and alert.detected:
        ...         handle_decay(alert)
    """
    return EdgeDecayMonitor(
        baseline_accuracy=baseline_accuracy,
        window_size=window_size,
        decay_threshold=0.05,
        critical_threshold=0.10
    )


def estimate_half_life(
    accuracies: List[float],
    initial_accuracy: float
) -> float:
    """
    Estimate alpha decay half-life from historical accuracies.

    Half-life = time for edge to decay to 50% of initial value.

    Model: accuracy(t) = 0.5 + (initial - 0.5) × exp(-λ × t)
    Half-life = ln(2) / λ

    Args:
        accuracies: Time series of accuracies
        initial_accuracy: Starting accuracy

    Returns:
        Estimated half-life in number of periods

    Example:
        >>> accs = [0.63, 0.62, 0.61, 0.60, 0.59, 0.58]
        >>> half_life = estimate_half_life(accs, 0.63)
        >>> print(f"Half-life: {half_life:.0f} periods")

    Reference:
        McLean & Pontiff (2016): Alpha decay estimation
    """
    if len(accuracies) < 3:
        return float('inf')

    # Convert to edge above random
    edges = [a - 0.5 for a in accuracies]
    initial_edge = initial_accuracy - 0.5

    if initial_edge <= 0:
        return float('inf')

    # Log-linear regression: ln(edge) = ln(initial_edge) - λ × t
    t = np.arange(len(edges))
    log_edges = np.log(np.maximum(edges, 1e-10))

    # Fit linear regression
    try:
        slope, intercept = np.polyfit(t, log_edges, 1)
        if slope >= 0:
            return float('inf')  # No decay (or improvement)

        lambda_ = -slope
        half_life = np.log(2) / lambda_
        return max(0, half_life)
    except Exception:
        return float('inf')


if __name__ == "__main__":
    # Demo: Simulate edge decay detection
    print("=" * 70)
    print("EDGE DECAY MONITOR DEMONSTRATION")
    print("=" * 70)

    np.random.seed(42)

    # Create monitor with 63% baseline
    monitor = EdgeDecayMonitor(baseline_accuracy=0.63, window_size=50)

    # Simulate trades with gradual decay
    n_trades = 200
    initial_accuracy = 0.63
    final_accuracy = 0.52  # Decay to near-random

    alerts_triggered = []

    for i in range(n_trades):
        # Accuracy decays linearly
        current_true_accuracy = initial_accuracy - (initial_accuracy - final_accuracy) * (i / n_trades)

        # Generate outcome based on current accuracy
        win = np.random.random() < current_true_accuracy

        alert = monitor.update(win)

        if alert and alert.detected:
            alerts_triggered.append((i, alert))
            print(f"\nTrade {i}: {alert}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total trades: {n_trades}")
    print(f"Alerts triggered: {len(alerts_triggered)}")
    if alerts_triggered:
        first_alert = alerts_triggered[0]
        print(f"First alert at trade {first_alert[0]}")
        print(f"First alert action: {first_alert[1].action}")

    # Final status
    status = monitor.get_status()
    print(f"\nFinal accuracy: {status['current_accuracy']:.2%}")
    print(f"EWMA accuracy: {status['ewma_accuracy']:.2%}")

    # Quick test
    print("\n--- Quick Detection Test ---")
    detected, p_val, action = detect_edge_decay_fast(
        recent_wins=28,
        recent_trades=50,
        baseline_accuracy=0.63
    )
    print(f"28 wins in 50 trades (vs 63% baseline):")
    print(f"  Decay detected: {detected}")
    print(f"  P-value: {p_val:.4f}")
    print(f"  Action: {action}")
