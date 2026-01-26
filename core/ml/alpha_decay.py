"""
Alpha Decay and Reflexivity Analysis

Implements Soros's reflexivity theory, Goodhart's Law effects, and alpha decay
estimation based on McLean & Pontiff (2016) findings.

Key Insight (McLean & Pontiff 2016):
    Anomaly returns decay 58% post-publication, 74% post-academic-paper.
    Alpha decay is a fundamental property of markets, not a bug.

Key Formulas:
    Exponential Decay:    α(t) = α₀ · e^(-λt)
    Half-Life:            t_{1/2} = ln(2) / λ
    Reflexivity Path:     r = corr(pred, action) · corr(action, outcome)

References:
    [1] McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy
        Stock Return Predictability?" Journal of Finance.
    [2] Soros, G. (2003). "The Alchemy of Finance." Wiley.
    [3] Goodhart, C.A.E. (1975). "Problems of Monetary Management."
    [4] Harvey, C.R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section
        of Expected Returns." Review of Financial Studies.
    [5] Chordia, T., Subrahmanyam, A., & Tong, Q. (2014). "Have Capital
        Market Anomalies Attenuated?" Journal of Financial Economics.
    [6] Schwert, G.W. (2003). "Anomalies and Market Efficiency."
        Handbook of the Economics of Finance.

Author: Claude Code + Kevin
Created: 2026-01-22
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class AlphaDecayAnalysis:
    """Results of alpha decay analysis."""
    initial_edge: float
    current_edge: float
    decay_rate: float
    half_life_months: float
    projected_edge_1y: float
    projected_edge_3y: float
    time_to_zero: float
    mclean_pontiff_benchmark: float
    strategy_quality: str

    def __repr__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  ALPHA DECAY ANALYSIS                                        ║
╠══════════════════════════════════════════════════════════════╣
║  Initial Edge:          {self.initial_edge:6.2f}%                           ║
║  Current Edge:          {self.current_edge:6.2f}%                           ║
║  Monthly Decay Rate:    {self.decay_rate*100:6.2f}%                           ║
║  Half-Life:             {self.half_life_months:6.1f} months                       ║
╠══════════════════════════════════════════════════════════════╣
║  1-Year Projection:     {self.projected_edge_1y:6.2f}%                           ║
║  3-Year Projection:     {self.projected_edge_3y:6.2f}%                           ║
║  Time to Zero Edge:     {self.time_to_zero:6.1f} months                       ║
╠══════════════════════════════════════════════════════════════╣
║  McLean-Pontiff Decay:  {self.mclean_pontiff_benchmark*100:5.1f}% (benchmark)                  ║
║  Strategy Quality:      {self.strategy_quality:20}              ║
╚══════════════════════════════════════════════════════════════╝
"""


class AlphaDecayEstimator:
    """
    Estimate and project alpha decay for trading strategies.

    Alpha Decay Phenomenon (McLean & Pontiff 2016):
        - 97 anomalies studied (1926-2013)
        - Post-publication: 42% of original return (58% decay)
        - Post-academic: 26% of original (74% decay)

    Harvey et al. (2016) Half-Lives:
        - Price-based technical: 6-18 months
        - Fundamental value: 2-5 years
        - Alternative data: 3-12 months
        - HFT microstructure: Days-weeks

    Goodhart's Law:
        "When a measure becomes a target, it ceases to be a good measure."
        Published strategies attract capital → arbitrage away returns.

    Reference:
        [1] McLean & Pontiff (2016): Academic alpha decay
        [4] Harvey et al. (2016): Factor decay rates
        [3] Goodhart (1975): Goodhart's Law
    """

    # Benchmark decay rates from literature
    BENCHMARK_DECAY = {
        'hft': 0.15,           # 15%/month (days-weeks half-life)
        'technical': 0.06,     # 6%/month (6-18 months half-life)
        'alternative': 0.08,   # 8%/month (3-12 months half-life)
        'fundamental': 0.02,   # 2%/month (2-5 years half-life)
        'unknown': 0.05        # 5%/month (14 months half-life)
    }

    def __init__(self, strategy_type: str = 'unknown'):
        """
        Initialize alpha decay estimator.

        Args:
            strategy_type: One of 'hft', 'technical', 'alternative', 'fundamental'
        """
        self.strategy_type = strategy_type
        self.benchmark_decay = self.BENCHMARK_DECAY.get(strategy_type, 0.05)

    def estimate_decay(
        self,
        backtest_accuracy: float,
        live_accuracy: float,
        months_live: float
    ) -> AlphaDecayAnalysis:
        """
        Estimate alpha decay from backtest vs live performance.

        Exponential Decay Model:
            edge(t) = edge₀ · e^(-λt)

        Reference:
            [1] McLean & Pontiff (2016)

        Args:
            backtest_accuracy: Accuracy in backtest (%)
            live_accuracy: Current live accuracy (%)
            months_live: Months since strategy went live

        Returns:
            AlphaDecayAnalysis with projections
        """
        # Calculate edges (above random)
        initial_edge = backtest_accuracy - 50
        current_edge = max(0, live_accuracy - 50)

        if initial_edge <= 0:
            return AlphaDecayAnalysis(
                initial_edge=0,
                current_edge=0,
                decay_rate=0,
                half_life_months=float('inf'),
                projected_edge_1y=0,
                projected_edge_3y=0,
                time_to_zero=float('inf'),
                mclean_pontiff_benchmark=0.58,
                strategy_quality='no_edge'
            )

        # Estimate decay rate
        if months_live > 0 and current_edge > 0:
            # Empirical decay rate: λ = -ln(current/initial) / t
            ratio = current_edge / initial_edge
            decay_rate = -np.log(ratio) / months_live
        else:
            # Use benchmark
            decay_rate = self.benchmark_decay

        # Ensure reasonable bounds
        decay_rate = np.clip(decay_rate, 0.001, 0.3)

        # Half-life: t_{1/2} = ln(2) / λ
        half_life = np.log(2) / decay_rate

        # Projections
        projected_1y = current_edge * np.exp(-decay_rate * 12)
        projected_3y = current_edge * np.exp(-decay_rate * 36)

        # Time to negligible edge (<1%)
        if decay_rate > 0 and current_edge > 1:
            time_to_zero = -np.log(1 / current_edge) / decay_rate
        else:
            time_to_zero = float('inf')

        # Strategy quality assessment
        if half_life > 24:
            quality = 'excellent_durability'
        elif half_life > 12:
            quality = 'good_durability'
        elif half_life > 6:
            quality = 'moderate_durability'
        else:
            quality = 'fast_decay'

        return AlphaDecayAnalysis(
            initial_edge=initial_edge,
            current_edge=current_edge,
            decay_rate=decay_rate,
            half_life_months=half_life,
            projected_edge_1y=projected_1y,
            projected_edge_3y=projected_3y,
            time_to_zero=time_to_zero,
            mclean_pontiff_benchmark=0.58,
            strategy_quality=quality
        )

    def estimate_from_time_series(
        self,
        accuracy_history: List[Tuple[float, float]]
    ) -> Dict:
        """
        Estimate decay from historical accuracy measurements.

        Fits exponential decay model to multiple observations.

        Args:
            accuracy_history: List of (months_since_start, accuracy) tuples

        Returns:
            Dictionary with decay parameters
        """
        if len(accuracy_history) < 2:
            return {'decay_rate': self.benchmark_decay, 'r_squared': 0}

        times = np.array([t for t, _ in accuracy_history])
        accuracies = np.array([a for _, a in accuracy_history])
        edges = np.maximum(0, accuracies - 50)

        # Fit: ln(edge) = ln(edge₀) - λt
        # Filter out zeros
        valid = edges > 0.1
        if np.sum(valid) < 2:
            return {'decay_rate': self.benchmark_decay, 'r_squared': 0}

        log_edges = np.log(edges[valid])
        times_valid = times[valid]

        slope, intercept, r_value, _, _ = stats.linregress(times_valid, log_edges)

        decay_rate = -slope
        initial_edge = np.exp(intercept)

        return {
            'decay_rate': max(0, decay_rate),
            'initial_edge': initial_edge,
            'r_squared': r_value ** 2,
            'half_life_months': np.log(2) / max(decay_rate, 1e-6)
        }

    def strategy_capacity(
        self,
        current_edge: float,
        current_capital: float,
        market_volume: float
    ) -> Dict:
        """
        Estimate strategy capacity before alpha decays significantly.

        Capacity Limit:
            As capital increases, market impact grows:
            effective_edge = edge - k · √(capital/volume)

        Reference:
            Almgren & Chriss (2000): Market impact models
            [5] Chordia et al. (2014): Capacity constraints

        Args:
            current_edge: Current edge (percentage points)
            current_capital: Capital deployed ($)
            market_volume: Market daily volume ($)

        Returns:
            Dictionary with capacity analysis
        """
        # Simple market impact model
        # Impact ≈ k · √(participation rate)
        k = 10  # Impact coefficient (bps per 1% participation)

        participation_rate = current_capital / market_volume

        # Current market impact
        current_impact = k * np.sqrt(participation_rate) / 100  # Convert to percentage

        # Effective edge
        effective_edge = current_edge - current_impact

        # Maximum capital before edge goes to zero
        # edge = k · √(cap/vol) → cap = (edge/k)² · vol
        if current_edge > 0:
            max_capital = ((current_edge * 100 / k) ** 2) * market_volume
        else:
            max_capital = 0

        # Optimal capital (maximize edge * capital)
        # Derivative: d(edge·cap)/d(cap) = edge - 1.5k√(cap/vol) = 0
        # → optimal_cap = (2·edge/(3k))² · vol
        optimal_capital = ((2 * current_edge * 100 / (3 * k)) ** 2) * market_volume

        return {
            'current_participation': participation_rate,
            'current_impact': current_impact,
            'effective_edge': effective_edge,
            'max_capital': max_capital,
            'optimal_capital': optimal_capital,
            'capacity_utilization': current_capital / max_capital if max_capital > 0 else 1
        }


class ReflexivityAnalyzer:
    """
    Analyze reflexivity in trading predictions.

    Soros's Reflexivity Theory (Soros 2003):
        Participant Beliefs → Market Prices → Fundamentals → Beliefs...

    Key Insight:
        Predictions can become self-fulfilling or self-defeating.
        A truly accurate model would eventually destroy its own accuracy.

    The Prediction Paradox:
        1. Model predicts "price will rise" with 99% accuracy
        2. Traders buy based on prediction
        3. Price rises (confirming prediction)
        4. But rise was CAUSED by prediction
        5. Remove prediction → price doesn't rise
        6. Model wasn't predicting, it was CREATING

    Reference:
        [2] Soros, G. (2003). "The Alchemy of Finance"
        [3] Goodhart (1975): Goodhart's Law
    """

    def analyze_reflexivity(
        self,
        predictions: np.ndarray,
        positions: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict:
        """
        Analyze reflexivity in prediction-position-outcome chain.

        Reflexivity Path:
            prediction → position → outcome

        If correlation(prediction, position) × correlation(position, outcome)
        is high, predictions may be self-fulfilling.

        Reference:
            [2] Soros (2003)

        Args:
            predictions: Model predictions
            positions: Positions taken (can be same as predictions)
            outcomes: Actual outcomes

        Returns:
            Dictionary with reflexivity analysis
        """
        # Correlation: prediction → position
        pred_pos_corr, _ = stats.pearsonr(predictions, positions)

        # Correlation: position → outcome
        pos_out_corr, _ = stats.pearsonr(positions, outcomes)

        # Correlation: prediction → outcome (direct)
        pred_out_corr, _ = stats.pearsonr(predictions, outcomes)

        # Reflexivity coefficient: indirect path
        reflexivity_path = pred_pos_corr * pos_out_corr

        # Decomposition of prediction-outcome correlation
        # Total = Direct (true prediction) + Indirect (reflexivity)
        direct_component = pred_out_corr - reflexivity_path
        indirect_component = reflexivity_path

        # Is reflexivity dominant?
        is_reflexive = abs(reflexivity_path) > abs(direct_component)

        # Accuracy decomposition
        observed_accuracy = np.mean(
            (predictions > np.median(predictions)) ==
            (outcomes > np.median(outcomes))
        )

        # True accuracy estimate (removing reflexive component)
        if is_reflexive:
            reflexivity_boost = reflexivity_path * (observed_accuracy - 0.5)
            true_accuracy = observed_accuracy - reflexivity_boost
        else:
            true_accuracy = observed_accuracy

        return {
            'prediction_position_corr': pred_pos_corr,
            'position_outcome_corr': pos_out_corr,
            'prediction_outcome_corr': pred_out_corr,
            'reflexivity_coefficient': reflexivity_path,
            'direct_component': direct_component,
            'indirect_component': indirect_component,
            'is_reflexivity_dominant': is_reflexive,
            'observed_accuracy': observed_accuracy,
            'estimated_true_accuracy': max(0.5, true_accuracy),
            'accuracy_inflation': observed_accuracy - true_accuracy
        }

    def feedback_stability_test(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        window_size: int = 50
    ) -> Dict:
        """
        Test feedback loop stability over time.

        Unstable feedback: prediction accuracy degrades as strategy trades
        Stable feedback: prediction accuracy remains constant

        Reference:
            [2] Soros (2003): Boom-bust cycles
            [3] Goodhart (1975): Policy invalidation

        Args:
            predictions: Time series of predictions
            outcomes: Time series of outcomes
            window_size: Rolling window for accuracy

        Returns:
            Dictionary with stability analysis
        """
        n = len(predictions)
        if n < window_size * 2:
            return {'is_stable': True, 'degradation_rate': 0}

        # Calculate rolling accuracy
        binary_pred = (predictions > np.median(predictions)).astype(int)
        binary_out = (outcomes > np.median(outcomes)).astype(int)
        correct = (binary_pred == binary_out).astype(float)

        rolling_accuracy = []
        for i in range(window_size, n):
            window_acc = np.mean(correct[i - window_size:i])
            rolling_accuracy.append(window_acc)

        rolling_accuracy = np.array(rolling_accuracy)

        # Test for degradation
        times = np.arange(len(rolling_accuracy))
        slope, intercept, r_value, p_value, _ = stats.linregress(
            times, rolling_accuracy
        )

        # Degradation rate (percentage points per observation)
        degradation_rate = -slope * window_size

        # Is degradation significant?
        is_degrading = slope < 0 and p_value < 0.05

        # Stability score (1 = perfectly stable, 0 = rapidly degrading)
        stability_score = max(0, 1 - abs(degradation_rate) * 10)

        return {
            'is_stable': not is_degrading,
            'is_degrading': is_degrading,
            'degradation_rate': degradation_rate,
            'degradation_pvalue': p_value,
            'stability_score': stability_score,
            'initial_accuracy': rolling_accuracy[0],
            'final_accuracy': rolling_accuracy[-1],
            'accuracy_change': rolling_accuracy[-1] - rolling_accuracy[0]
        }

    def crowding_risk(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> Dict:
        """
        Estimate crowding risk (too many similar strategies).

        When strategies crowd:
            1. Entry signals correlate
            2. Market impact increases
            3. Exit stampedes cause crashes

        Reference:
            [5] Chordia et al. (2014): Anomaly attenuation
            Khandani & Lo (2011): Quant meltdown analysis

        Args:
            strategy_returns: Returns from the strategy
            market_returns: Overall market returns

        Returns:
            Dictionary with crowding analysis
        """
        # Correlation with market (high = less unique)
        market_corr, _ = stats.pearsonr(strategy_returns, market_returns)

        # Beta (systematic risk)
        beta = (
            np.cov(strategy_returns, market_returns)[0, 1] /
            (np.var(market_returns) + 1e-10)
        )

        # Drawdown correlation (do drawdowns occur together?)
        strat_dd = np.minimum.accumulate(np.cumsum(strategy_returns))
        strat_dd = np.cumsum(strategy_returns) - strat_dd
        market_dd = np.minimum.accumulate(np.cumsum(market_returns))
        market_dd = np.cumsum(market_returns) - market_dd

        dd_corr, _ = stats.pearsonr(strat_dd, market_dd)

        # Crowding score (0-1, higher = more crowded)
        crowding_score = (abs(market_corr) + abs(dd_corr)) / 2

        # Crowding risk level
        if crowding_score > 0.7:
            risk_level = 'high'
        elif crowding_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'market_correlation': market_corr,
            'beta': beta,
            'drawdown_correlation': dd_corr,
            'crowding_score': crowding_score,
            'crowding_risk_level': risk_level,
            'diversification_benefit': 1 - crowding_score
        }


def estimate_accuracy_ceiling_with_decay(
    current_accuracy: float,
    strategy_type: str = 'technical',
    months_deployed: float = 0,
    market_share_pct: float = 0.01
) -> Dict:
    """
    Estimate accuracy ceiling accounting for alpha decay.

    Combines:
        1. Information theory limits (fundamental)
        2. Alpha decay (temporal)
        3. Capacity constraints (scale)
        4. Reflexivity (feedback)

    Reference:
        [1-6] See module docstring

    Args:
        current_accuracy: Current accuracy (%)
        strategy_type: Type for decay benchmark
        months_deployed: Months since deployment
        market_share_pct: Percent of market volume

    Returns:
        Dictionary with ceiling analysis
    """
    # Information theory ceiling
    info_ceiling = 85.0  # Fundamental limit

    # Decay effect
    estimator = AlphaDecayEstimator(strategy_type)
    decay_rate = estimator.benchmark_decay

    # Project forward decay
    decay_factor_1y = np.exp(-decay_rate * 12)
    decay_factor_3y = np.exp(-decay_rate * 36)

    current_edge = current_accuracy - 50
    projected_edge_1y = current_edge * decay_factor_1y
    projected_edge_3y = current_edge * decay_factor_3y

    # Capacity constraint (simplified)
    # Larger market share = more impact = less effective edge
    capacity_penalty = min(10, market_share_pct * 100)  # 1% share = 1% penalty

    # Practical ceiling
    practical_ceiling_now = min(info_ceiling, 50 + current_edge)
    practical_ceiling_1y = min(info_ceiling, 50 + projected_edge_1y - capacity_penalty)
    practical_ceiling_3y = min(info_ceiling, 50 + projected_edge_3y - capacity_penalty)

    return {
        'information_ceiling': info_ceiling,
        'current_accuracy': current_accuracy,
        'current_edge': current_edge,
        'decay_rate_monthly': decay_rate,
        'projected_accuracy_1y': 50 + projected_edge_1y,
        'projected_accuracy_3y': 50 + projected_edge_3y,
        'capacity_penalty': capacity_penalty,
        'practical_ceiling_now': practical_ceiling_now,
        'practical_ceiling_1y': practical_ceiling_1y,
        'practical_ceiling_3y': practical_ceiling_3y,
        'sustainable_edge': projected_edge_3y - capacity_penalty
    }


# Convenience functions
def create_alpha_decay_features(
    backtest_accuracy: float,
    live_accuracy: float,
    months_live: float,
    strategy_type: str = 'technical'
) -> Dict[str, float]:
    """
    Create alpha decay features for ML integration.

    Args:
        backtest_accuracy: Backtest accuracy (%)
        live_accuracy: Live accuracy (%)
        months_live: Months deployed
        strategy_type: Strategy type

    Returns:
        Dictionary of features
    """
    estimator = AlphaDecayEstimator(strategy_type)
    analysis = estimator.estimate_decay(backtest_accuracy, live_accuracy, months_live)

    return {
        'decay_initial_edge': analysis.initial_edge,
        'decay_current_edge': analysis.current_edge,
        'decay_rate': analysis.decay_rate,
        'decay_half_life': min(analysis.half_life_months, 120),
        'decay_projected_1y': analysis.projected_edge_1y,
        'decay_projected_3y': analysis.projected_edge_3y,
        'decay_ratio': analysis.current_edge / (analysis.initial_edge + 1e-6)
    }


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("ALPHA DECAY ANALYSIS")
    print("=" * 60)

    estimator = AlphaDecayEstimator(strategy_type='technical')
    analysis = estimator.estimate_decay(
        backtest_accuracy=70.0,
        live_accuracy=64.0,
        months_live=6
    )
    print(analysis)

    print("\n" + "=" * 60)
    print("REFLEXIVITY ANALYSIS")
    print("=" * 60)

    np.random.seed(42)
    n = 1000

    # Simulate reflexive market
    predictions = np.random.randn(n)
    positions = predictions + np.random.randn(n) * 0.3
    # Outcomes partially driven by positions (reflexivity)
    outcomes = 0.5 * positions + 0.5 * np.random.randn(n)

    analyzer = ReflexivityAnalyzer()
    reflexivity = analyzer.analyze_reflexivity(predictions, positions, outcomes)

    for name, value in reflexivity.items():
        if isinstance(value, float):
            print(f"{name:35} {value:.4f}")
        else:
            print(f"{name:35} {value}")

    print("\n" + "=" * 60)
    print("ACCURACY CEILING WITH DECAY")
    print("=" * 60)

    ceiling = estimate_accuracy_ceiling_with_decay(
        current_accuracy=64.0,
        strategy_type='technical',
        months_deployed=6,
        market_share_pct=0.1
    )

    for name, value in ceiling.items():
        print(f"{name:35} {value:.4f}")
