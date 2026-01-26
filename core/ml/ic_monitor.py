"""
IC/ICIR Factor Monitoring Module
================================

Real-time Information Coefficient monitoring for factor health tracking.
Standard technique used by Chinese quant firms (国泰君安, 中信, 华泰).

Key Insight (Chinese Quant Industry Standard):
    IC (Information Coefficient) directly measures factor-return predictability.
    ICIR (IC Information Ratio) measures stability of that predictability.
    When IC decays, your edge is dying - act IMMEDIATELY.

Key Formulas:
    IC (Information Coefficient):
        IC_t = corr(factor_value_t, return_{t+1})
        IC_t = Σ[(f_i - f̄)(r_i - r̄)] / √[Σ(f_i - f̄)² × Σ(r_i - r̄)²]

    ICIR (Information Coefficient to Information Ratio):
        ICIR = mean(IC) / std(IC)
        ICIR = IC̄ / σ_IC

    Rank IC (Spearman):
        RankIC_t = corr(rank(factor_value_t), rank(return_{t+1}))

    Factor Effectiveness Thresholds (Industry Standard):
        |IC| > 0.02: Marginally effective
        |IC| > 0.03: Moderately effective
        |IC| > 0.05: Strongly effective
        |IC| > 0.10: Exceptional factor
        ICIR > 0.5: Stable alpha generation
        ICIR > 1.0: Excellent stability

References:
    [1] Qian, E. & Hua, R. (2004). "Active Risk and Information Ratio."
        Journal of Investment Management, 2(3), 1-15.

    [2] Grinold, R.C. & Kahn, R.N. (2000). "Active Portfolio Management."
        McGraw-Hill. Chapter on Information Ratio.

    [3] 国泰君安证券研究所 (2019). "因子有效性检验方法论."
        Guotai Junan Securities Research.

    [4] 华泰证券金工团队 (2020). "多因子模型IC分析框架."
        Huatai Securities Quantitative Research.

    [5] BigQuant Research (2024). "量化因子IC/ICIR监控体系."
        https://bigquant.com/wiki/doc/

    [6] Clarke, R., de Silva, H., & Thorley, S. (2002).
        "Portfolio Constraints and the Fundamental Law of Active Management."
        Financial Analysts Journal, 58(5), 48-66.

Author: Claude Code + Kevin
Created: 2025-01-22
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import warnings


class FactorHealth(Enum):
    """Factor health status based on IC/ICIR."""
    EXCELLENT = "EXCELLENT"      # IC > 0.05, ICIR > 1.0
    GOOD = "GOOD"                # IC > 0.03, ICIR > 0.5
    MARGINAL = "MARGINAL"        # IC > 0.02, ICIR > 0.3
    WEAK = "WEAK"                # IC > 0.01
    DEAD = "DEAD"                # IC <= 0.01 or negative trend


@dataclass
class ICResult:
    """Result of IC analysis for a single period."""
    ic: float                    # Pearson IC
    rank_ic: float               # Spearman Rank IC
    ic_abs: float                # Absolute IC
    t_stat: float                # T-statistic for IC significance
    p_value: float               # P-value for IC significance
    is_significant: bool         # IC significantly different from 0


@dataclass
class ICIRResult:
    """Comprehensive IC/ICIR analysis result."""

    # Core metrics
    mean_ic: float               # Average IC over period
    std_ic: float                # Standard deviation of IC
    icir: float                  # Information Coefficient IR
    mean_rank_ic: float          # Average Rank IC

    # Health assessment
    health: FactorHealth
    is_effective: bool           # Factor worth using?

    # Trend analysis
    ic_trend: float              # Slope of IC over time (decay detection)
    is_decaying: bool            # IC trending downward?
    decay_rate: float            # % decay per period

    # Statistical significance
    ic_t_stat: float             # T-stat for mean IC
    ic_p_value: float            # P-value for mean IC
    pct_significant: float       # % of periods with significant IC

    # Thresholds
    pct_above_002: float         # % of periods with |IC| > 0.02
    pct_above_003: float         # % of periods with |IC| > 0.03
    pct_above_005: float         # % of periods with |IC| > 0.05

    # History
    n_periods: int               # Number of periods analyzed
    ic_series: List[float]       # IC time series

    def __repr__(self) -> str:
        trend = "↓ DECAYING" if self.is_decaying else "→ STABLE"
        return f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  IC/ICIR FACTOR HEALTH REPORT                                                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Health Status:        {self.health.value:^15}                                        ║
║  Factor Effective:     {"YES" if self.is_effective else "NO":^15}                                        ║
║  Trend:                {trend:^15}                                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  CORE METRICS:                                                                   ║
║    Mean IC:            {self.mean_ic:>+8.4f}  (target: |IC| > 0.03)                   ║
║    Std IC:             {self.std_ic:>8.4f}                                           ║
║    ICIR:               {self.icir:>+8.4f}  (target: ICIR > 0.5)                      ║
║    Mean Rank IC:       {self.mean_rank_ic:>+8.4f}                                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  EFFECTIVENESS:                                                                  ║
║    % periods |IC|>0.02: {self.pct_above_002:>6.1%}                                          ║
║    % periods |IC|>0.03: {self.pct_above_003:>6.1%}                                          ║
║    % periods |IC|>0.05: {self.pct_above_005:>6.1%}                                          ║
║    % significant:       {self.pct_significant:>6.1%}                                          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  TREND ANALYSIS:                                                                 ║
║    IC Trend:           {self.ic_trend:>+8.6f}/period                                   ║
║    Decay Rate:         {self.decay_rate:>+8.2%}/period                                   ║
║    Periods Analyzed:   {self.n_periods:>8}                                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  STATISTICAL SIGNIFICANCE:                                                       ║
║    IC t-statistic:     {self.ic_t_stat:>+8.3f}                                           ║
║    IC p-value:         {self.ic_p_value:>8.2e}                                           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""


class ICMonitor:
    """
    Real-time IC/ICIR monitoring for factor health.

    Tracks factor effectiveness over time and provides early warning
    when factors begin to decay.

    Reference:
        [1] Grinold & Kahn (2000): Active Portfolio Management
        [2] 国泰君安/华泰 Quant Research: Factor IC Analysis
    """

    def __init__(
        self,
        window_size: int = 100,
        decay_threshold: float = -0.001,
        min_ic_threshold: float = 0.02,
        significance_level: float = 0.05
    ):
        """
        Initialize IC Monitor.

        Args:
            window_size: Rolling window for IC calculation
            decay_threshold: IC trend threshold for decay warning
            min_ic_threshold: Minimum |IC| to consider factor effective
            significance_level: P-value threshold for significance
        """
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.min_ic_threshold = min_ic_threshold
        self.significance_level = significance_level

        # Storage for real-time monitoring
        self.ic_history: deque = deque(maxlen=1000)
        self.rank_ic_history: deque = deque(maxlen=1000)
        self.factor_history: deque = deque(maxlen=window_size + 10)
        self.return_history: deque = deque(maxlen=window_size + 10)

    def compute_ic(
        self,
        factor_values: np.ndarray,
        forward_returns: np.ndarray
    ) -> ICResult:
        """
        Compute Information Coefficient for a single period.

        Args:
            factor_values: Factor values at time t
            forward_returns: Returns at time t+1

        Returns:
            ICResult with IC metrics

        Formula:
            IC = corr(factor, return)
            RankIC = corr(rank(factor), rank(return))
        """
        # Remove NaN values
        mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
        f = factor_values[mask]
        r = forward_returns[mask]

        if len(f) < 10:
            return ICResult(
                ic=0.0, rank_ic=0.0, ic_abs=0.0,
                t_stat=0.0, p_value=1.0, is_significant=False
            )

        # Pearson IC
        ic, p_value = stats.pearsonr(f, r)

        # Spearman Rank IC (more robust to outliers)
        rank_ic, _ = stats.spearmanr(f, r)

        # T-statistic for IC
        n = len(f)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

        # Handle NaN
        if np.isnan(ic):
            ic = 0.0
        if np.isnan(rank_ic):
            rank_ic = 0.0

        return ICResult(
            ic=ic,
            rank_ic=rank_ic,
            ic_abs=abs(ic),
            t_stat=t_stat,
            p_value=p_value,
            is_significant=p_value < self.significance_level
        )

    def update(
        self,
        factor_value: float,
        realized_return: float
    ):
        """
        Update monitor with new observation (real-time streaming).

        Args:
            factor_value: Current factor value
            realized_return: Realized return for previous prediction
        """
        self.factor_history.append(factor_value)
        self.return_history.append(realized_return)

        # Compute rolling IC if enough data
        if len(self.factor_history) >= self.window_size:
            factors = np.array(list(self.factor_history)[-self.window_size:])
            returns = np.array(list(self.return_history)[-self.window_size:])

            result = self.compute_ic(factors[:-1], returns[1:])
            self.ic_history.append(result.ic)
            self.rank_ic_history.append(result.rank_ic)

    def analyze(
        self,
        factor_values: Optional[np.ndarray] = None,
        forward_returns: Optional[np.ndarray] = None,
        rolling_window: int = 20
    ) -> ICIRResult:
        """
        Comprehensive IC/ICIR analysis.

        Args:
            factor_values: Factor value series (if not using streaming)
            forward_returns: Forward return series (if not using streaming)
            rolling_window: Window for rolling IC calculation

        Returns:
            ICIRResult with full analysis
        """
        # Use provided data or history
        if factor_values is not None and forward_returns is not None:
            ic_series = []
            rank_ic_series = []

            for i in range(len(factor_values) - rolling_window):
                f = factor_values[i:i+rolling_window]
                r = forward_returns[i:i+rolling_window]
                result = self.compute_ic(f, r)
                ic_series.append(result.ic)
                rank_ic_series.append(result.rank_ic)

            ic_series = np.array(ic_series)
            rank_ic_series = np.array(rank_ic_series)
        else:
            ic_series = np.array(list(self.ic_history))
            rank_ic_series = np.array(list(self.rank_ic_history))

        if len(ic_series) < 5:
            return self._empty_result()

        # Core metrics
        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series, ddof=1)
        icir = mean_ic / (std_ic + 1e-10)
        mean_rank_ic = np.mean(rank_ic_series)

        # Trend analysis (linear regression on IC)
        x = np.arange(len(ic_series))
        slope, intercept, r_value, p_value_trend, std_err = stats.linregress(x, ic_series)
        ic_trend = slope
        is_decaying = slope < self.decay_threshold

        # Decay rate as percentage
        if abs(mean_ic) > 1e-6:
            decay_rate = slope / abs(mean_ic)
        else:
            decay_rate = 0.0

        # Statistical significance of mean IC
        ic_t_stat = mean_ic / (std_ic / np.sqrt(len(ic_series)) + 1e-10)
        ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), len(ic_series) - 1))

        # Percentage above thresholds
        ic_abs = np.abs(ic_series)
        pct_above_002 = np.mean(ic_abs > 0.02)
        pct_above_003 = np.mean(ic_abs > 0.03)
        pct_above_005 = np.mean(ic_abs > 0.05)

        # Percentage significant
        # Approximate: assume IC ~ N(0, 1/sqrt(n)) under null
        n_per_period = rolling_window
        pct_significant = np.mean(ic_abs > 2 / np.sqrt(n_per_period))

        # Health assessment
        health, is_effective = self._assess_health(
            mean_ic, icir, is_decaying, pct_above_003
        )

        return ICIRResult(
            mean_ic=mean_ic,
            std_ic=std_ic,
            icir=icir,
            mean_rank_ic=mean_rank_ic,
            health=health,
            is_effective=is_effective,
            ic_trend=ic_trend,
            is_decaying=is_decaying,
            decay_rate=decay_rate,
            ic_t_stat=ic_t_stat,
            ic_p_value=ic_p_value,
            pct_significant=pct_significant,
            pct_above_002=pct_above_002,
            pct_above_003=pct_above_003,
            pct_above_005=pct_above_005,
            n_periods=len(ic_series),
            ic_series=ic_series.tolist()
        )

    def _assess_health(
        self,
        mean_ic: float,
        icir: float,
        is_decaying: bool,
        pct_above_003: float
    ) -> Tuple[FactorHealth, bool]:
        """
        Assess factor health based on IC metrics.

        Industry standard thresholds from Chinese quant research.
        """
        abs_ic = abs(mean_ic)

        if is_decaying and abs_ic < 0.02:
            return FactorHealth.DEAD, False

        if abs_ic > 0.05 and icir > 1.0 and not is_decaying:
            return FactorHealth.EXCELLENT, True

        if abs_ic > 0.03 and icir > 0.5:
            return FactorHealth.GOOD, True

        if abs_ic > 0.02 and icir > 0.3:
            return FactorHealth.MARGINAL, pct_above_003 > 0.3

        if abs_ic > 0.01:
            return FactorHealth.WEAK, False

        return FactorHealth.DEAD, False

    def _empty_result(self) -> ICIRResult:
        """Return empty result when insufficient data."""
        return ICIRResult(
            mean_ic=0.0, std_ic=0.0, icir=0.0, mean_rank_ic=0.0,
            health=FactorHealth.DEAD, is_effective=False,
            ic_trend=0.0, is_decaying=False, decay_rate=0.0,
            ic_t_stat=0.0, ic_p_value=1.0, pct_significant=0.0,
            pct_above_002=0.0, pct_above_003=0.0, pct_above_005=0.0,
            n_periods=0, ic_series=[]
        )

    def get_decay_warning(self) -> Optional[str]:
        """
        Get decay warning if factor is degrading.

        Returns:
            Warning message or None
        """
        if len(self.ic_history) < 20:
            return None

        recent_ic = list(self.ic_history)[-20:]
        x = np.arange(len(recent_ic))
        slope, _, _, _, _ = stats.linregress(x, recent_ic)

        if slope < self.decay_threshold:
            return f"WARNING: Factor IC decaying at {slope:.4f}/period. Consider retraining."

        return None


class MultiFactorICMonitor:
    """
    Monitor IC for multiple factors simultaneously.

    Tracks which factors are healthy and which are dying.
    """

    def __init__(self, factor_names: List[str], **kwargs):
        """
        Initialize multi-factor monitor.

        Args:
            factor_names: List of factor names to monitor
            **kwargs: Arguments passed to individual ICMonitor instances
        """
        self.factor_names = factor_names
        self.monitors = {name: ICMonitor(**kwargs) for name in factor_names}

    def update(
        self,
        factor_values: Dict[str, float],
        realized_return: float
    ):
        """Update all monitors with new observation."""
        for name, value in factor_values.items():
            if name in self.monitors:
                self.monitors[name].update(value, realized_return)

    def analyze_all(self) -> Dict[str, ICIRResult]:
        """Analyze all factors."""
        return {name: monitor.analyze() for name, monitor in self.monitors.items()}

    def get_healthy_factors(self) -> List[str]:
        """Get list of healthy factors."""
        results = self.analyze_all()
        return [name for name, result in results.items() if result.is_effective]

    def get_dying_factors(self) -> List[str]:
        """Get list of factors with decaying IC."""
        results = self.analyze_all()
        return [name for name, result in results.items() if result.is_decaying]


# Convenience functions
def compute_ic(
    factor_values: np.ndarray,
    forward_returns: np.ndarray
) -> float:
    """
    Quick IC calculation.

    Reference: Standard Pearson correlation
    """
    mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    if np.sum(mask) < 10:
        return 0.0
    return stats.pearsonr(factor_values[mask], forward_returns[mask])[0]


def compute_icir(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    window: int = 20
) -> float:
    """
    Quick ICIR calculation.

    ICIR = mean(rolling_IC) / std(rolling_IC)
    """
    ic_series = []
    for i in range(len(factor_values) - window):
        ic = compute_ic(
            factor_values[i:i+window],
            forward_returns[i:i+window]
        )
        ic_series.append(ic)

    if len(ic_series) < 5:
        return 0.0

    return np.mean(ic_series) / (np.std(ic_series) + 1e-10)


def is_factor_effective(ic: float, icir: float) -> bool:
    """
    Quick check if factor is effective.

    Industry standard thresholds:
        |IC| > 0.03 AND ICIR > 0.5
    """
    return abs(ic) > 0.03 and icir > 0.5


if __name__ == "__main__":
    print("=" * 70)
    print("IC/ICIR MONITORING DEMONSTRATION")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n = 500

    # Factor with predictive power (IC ~ 0.04)
    true_signal = np.random.randn(n)
    noise = np.random.randn(n) * 5
    factor_values = true_signal + noise * 0.5
    forward_returns = true_signal * 0.1 + noise * 0.1

    # Create monitor and analyze
    monitor = ICMonitor()
    result = monitor.analyze(factor_values, forward_returns, rolling_window=20)

    print(result)

    # Quick functions
    print("\n[Quick IC/ICIR Calculation]")
    ic = compute_ic(factor_values, forward_returns)
    icir = compute_icir(factor_values, forward_returns)
    print(f"IC: {ic:.4f}")
    print(f"ICIR: {icir:.4f}")
    print(f"Effective: {is_factor_effective(ic, icir)}")
