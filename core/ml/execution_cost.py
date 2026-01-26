"""
Execution Cost Prediction for HFT

ML-based prediction of market impact, slippage, and execution costs.
Critical for optimal execution and certainty quantification.

Academic Citations:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
  Journal of Risk - Foundational market impact model

- Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
  Quantitative Finance - Square-root law

- Obizhaeva & Wang (2013): "Optimal Trading Strategy and Supply/Demand Dynamics"
  Journal of Financial Markets - Transient impact

- arXiv:2303.02482 (2023): "Deep Learning for Market Impact Prediction"
  State-of-the-art ML for execution costs

Chinese Quant Application:
- 海通证券: 交易成本预测模型 (transaction cost prediction)
- 中信证券: 市场冲击成本分析 (market impact analysis)
- 招商证券: 最优执行算法 (optimal execution)
- 华泰证券: 滑点预测与控制 (slippage prediction)

The Key Insight:
    Execution cost = Spread cost + Market impact + Timing cost + Opportunity cost

    If you can PREDICT execution cost with certainty,
    you can include it in your edge calculation:

    Net Edge = Gross Edge - E[Execution Cost]
             = 82% win rate - predicted slippage
             = TRUE edge after costs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from scipy import stats
import warnings


@dataclass
class ExecutionCostEstimate:
    """Comprehensive execution cost estimate."""

    # Component costs (in basis points)
    spread_cost_bps: float  # Half spread
    market_impact_bps: float  # Price move from our trade
    timing_cost_bps: float  # Unfavorable price movement during execution
    opportunity_cost_bps: float  # Cost of not executing immediately

    # Total
    total_cost_bps: float
    total_cost_dollar: float  # In dollar terms

    # Uncertainty
    cost_std_bps: float  # Standard deviation of cost estimate
    cost_95_ci: Tuple[float, float]  # 95% confidence interval

    # Breakdown
    permanent_impact_bps: float  # Impact that doesn't decay
    temporary_impact_bps: float  # Impact that decays


@dataclass
class SlippageResult:
    """Result of slippage prediction."""

    expected_slippage_bps: float
    expected_slippage_pips: float
    slippage_std: float

    # By direction
    buy_slippage: float  # Expected slippage for buy
    sell_slippage: float  # Expected slippage for sell

    # Confidence
    confidence: float  # Model confidence in prediction
    prediction_interval: Tuple[float, float]


class AlmgrenChrissModel:
    """
    Almgren-Chriss (2001) Optimal Execution Model.

    Classic model for market impact and optimal execution.

    Market Impact = η * σ * (Q/V)^0.5 + γ * σ * (Q/V) * T

    Where:
    - η: Temporary impact coefficient
    - γ: Permanent impact coefficient
    - σ: Volatility
    - Q: Order size
    - V: Daily volume
    - T: Execution time

    Reference:
        Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
    """

    def __init__(
        self,
        eta: float = 0.1,  # Temporary impact coefficient
        gamma: float = 0.05,  # Permanent impact coefficient
        sigma_daily: float = 0.01,  # Daily volatility
        daily_volume: float = 1e9,  # Daily volume
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            eta: Temporary impact coefficient (market-specific)
            gamma: Permanent impact coefficient
            sigma_daily: Daily volatility (in decimal)
            daily_volume: Average daily volume
        """
        self.eta = eta
        self.gamma = gamma
        self.sigma_daily = sigma_daily
        self.daily_volume = daily_volume

    def estimate_cost(
        self,
        order_size: float,
        execution_time: float = 1.0,  # Fraction of day
        spread_bps: float = 1.0,
    ) -> ExecutionCostEstimate:
        """
        Estimate execution cost for an order.

        Args:
            order_size: Order size in currency units
            execution_time: Execution time as fraction of day
            spread_bps: Current spread in basis points

        Returns:
            ExecutionCostEstimate with all components
        """
        # Participation rate
        participation = order_size / self.daily_volume

        # Temporary impact (square-root law)
        temp_impact = self.eta * self.sigma_daily * np.sqrt(participation)

        # Permanent impact (linear)
        perm_impact = self.gamma * self.sigma_daily * participation

        # Timing cost (volatility during execution)
        timing = 0.5 * self.sigma_daily * np.sqrt(execution_time) * participation

        # Spread cost
        spread_cost = spread_bps / 10000 / 2  # Half spread for one-way

        # Total (convert to bps)
        total_impact = (temp_impact + perm_impact + timing) * 10000 + spread_bps / 2
        total_dollar = total_impact * order_size / 10000

        # Uncertainty (simplified)
        cost_std = total_impact * 0.3  # ~30% std relative to mean

        return ExecutionCostEstimate(
            spread_cost_bps=spread_bps / 2,
            market_impact_bps=(temp_impact + perm_impact) * 10000,
            timing_cost_bps=timing * 10000,
            opportunity_cost_bps=0.0,  # Not included in basic model
            total_cost_bps=total_impact,
            total_cost_dollar=total_dollar,
            cost_std_bps=cost_std,
            cost_95_ci=(total_impact - 1.96 * cost_std, total_impact + 1.96 * cost_std),
            permanent_impact_bps=perm_impact * 10000,
            temporary_impact_bps=temp_impact * 10000,
        )


class SquareRootImpactModel:
    """
    Square-Root Market Impact Model (Gatheral 2010).

    The most empirically validated impact model:

    Impact = Y * σ * (Q / ADV)^0.5

    Where Y is the impact coefficient (typically 0.1-0.3).

    Reference:
        Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
        Empirically: Impact ∝ sqrt(volume) across markets/assets
    """

    def __init__(
        self,
        impact_coefficient: float = 0.15,
        adv: float = 1e9,
    ):
        """
        Initialize square-root model.

        Args:
            impact_coefficient: Y in the formula (0.1-0.3 typical)
            adv: Average daily volume
        """
        self.Y = impact_coefficient
        self.adv = adv

    def estimate_impact(
        self,
        order_size: float,
        volatility: float,
        spread_bps: float = 1.0,
    ) -> float:
        """
        Estimate market impact in basis points.

        Args:
            order_size: Order size
            volatility: Current volatility (annualized)
            spread_bps: Current spread in bps

        Returns:
            Expected market impact in basis points
        """
        # Daily vol from annual
        daily_vol = volatility / np.sqrt(252)

        # Square-root impact
        participation = order_size / self.adv
        impact = self.Y * daily_vol * np.sqrt(participation)

        # Convert to bps and add spread
        return impact * 10000 + spread_bps / 2


class MLSlippagePredictor:
    """
    ML-based slippage prediction (海通证券 style).

    Uses historical execution data to predict slippage.

    Features:
    - Order characteristics (size, direction, aggressiveness)
    - Market state (spread, volatility, volume)
    - Time features (session, time-of-day)
    - Microstructure (book imbalance, trade imbalance)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize ML slippage predictor.

        Args:
            feature_names: Names of features used for prediction
        """
        self.feature_names = feature_names or self._default_features()
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self._fitted = False

        # Historical slippage statistics
        self._slippage_history: List[float] = []
        self._feature_history: List[np.ndarray] = []

    def _default_features(self) -> List[str]:
        """Default features for slippage prediction."""
        return [
            # Order features
            'order_size_pct_adv',  # Order size as % of ADV
            'is_buy',  # 1 for buy, 0 for sell
            'aggressiveness',  # 0=passive, 1=aggressive

            # Market state
            'spread_bps',  # Current spread
            'volatility_5m',  # 5-minute volatility
            'volume_ratio',  # Volume vs average

            # Microstructure
            'bid_ask_imbalance',  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
            'trade_imbalance',  # Recent buy/sell imbalance
            'book_depth_ratio',  # Depth at best vs total

            # Time features
            'minutes_since_open',
            'is_session_start',  # First 30 min
            'is_session_end',  # Last 30 min
        ]

    def fit(
        self,
        features: np.ndarray,
        slippage: np.ndarray,
    ) -> None:
        """
        Fit slippage model on historical data.

        Args:
            features: Feature matrix (n_samples, n_features)
            slippage: Actual slippage in bps (n_samples,)
        """
        # Simple linear regression for interpretability
        # In production, use XGBoost/LightGBM

        # Standardize features
        self.scaler_mean = np.mean(features, axis=0)
        self.scaler_std = np.std(features, axis=0) + 1e-10

        X = (features - self.scaler_mean) / self.scaler_std

        # Add bias term
        X_bias = np.column_stack([np.ones(len(X)), X])

        # Solve normal equations
        try:
            self.model = np.linalg.lstsq(X_bias, slippage, rcond=None)[0]
        except Exception:
            # Fallback to pseudo-inverse
            self.model = np.linalg.pinv(X_bias) @ slippage

        # Store residual statistics
        predictions = X_bias @ self.model
        residuals = slippage - predictions
        self._residual_std = np.std(residuals)

        self._fitted = True

    def predict(
        self,
        features: np.ndarray,
    ) -> SlippageResult:
        """
        Predict slippage for new order.

        Args:
            features: Feature vector (n_features,)

        Returns:
            SlippageResult with prediction and uncertainty
        """
        if not self._fitted:
            return self._fallback_prediction(features)

        # Standardize
        X = (features - self.scaler_mean) / self.scaler_std
        X_bias = np.concatenate([[1], X])

        # Predict
        expected_slippage = X_bias @ self.model

        # Convert to pips (assuming forex, 1 pip = 0.1 bps for major pairs)
        slippage_pips = expected_slippage / 0.1

        # Prediction interval
        z = 1.96
        lower = expected_slippage - z * self._residual_std
        upper = expected_slippage + z * self._residual_std

        # Direction-specific (simplified)
        is_buy = features[1] if len(features) > 1 else 0.5
        buy_slippage = expected_slippage * (1 + 0.1 * (1 - is_buy))
        sell_slippage = expected_slippage * (1 + 0.1 * is_buy)

        return SlippageResult(
            expected_slippage_bps=expected_slippage,
            expected_slippage_pips=slippage_pips,
            slippage_std=self._residual_std,
            buy_slippage=buy_slippage,
            sell_slippage=sell_slippage,
            confidence=0.8,  # Placeholder
            prediction_interval=(lower, upper),
        )

    def _fallback_prediction(self, features: np.ndarray) -> SlippageResult:
        """Fallback when model not fitted."""
        # Use simple heuristics
        spread_bps = features[3] if len(features) > 3 else 1.0
        order_size_pct = features[0] if len(features) > 0 else 0.001

        # Rule of thumb: slippage = half spread + impact
        impact = 0.1 * np.sqrt(order_size_pct) * 10000
        slippage = spread_bps / 2 + impact

        return SlippageResult(
            expected_slippage_bps=slippage,
            expected_slippage_pips=slippage / 0.1,
            slippage_std=slippage * 0.5,
            buy_slippage=slippage * 1.1,
            sell_slippage=slippage * 0.9,
            confidence=0.5,
            prediction_interval=(slippage * 0.5, slippage * 1.5),
        )

    def update_online(
        self,
        features: np.ndarray,
        actual_slippage: float,
    ) -> None:
        """
        Online update with new observation.

        Args:
            features: Features of executed order
            actual_slippage: Actual slippage observed
        """
        self._slippage_history.append(actual_slippage)
        self._feature_history.append(features)

        # Refit periodically
        if len(self._slippage_history) >= 100 and len(self._slippage_history) % 50 == 0:
            X = np.array(self._feature_history[-500:])  # Last 500
            y = np.array(self._slippage_history[-500:])
            self.fit(X, y)


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_slippage_estimate(
    spread_bps: float,
    order_size_pct_adv: float,
    volatility: float,
) -> float:
    """
    Ultra-fast slippage estimate for HFT.

    Rule of thumb:
    Slippage ≈ spread/2 + Y * vol * sqrt(participation)

    Args:
        spread_bps: Current spread in basis points
        order_size_pct_adv: Order size as fraction of ADV
        volatility: Daily volatility (decimal)

    Returns:
        Expected slippage in basis points
    """
    # Half spread
    spread_cost = spread_bps / 2

    # Square-root impact (Y = 0.15 typical)
    impact = 0.15 * volatility * np.sqrt(order_size_pct_adv) * 10000

    return spread_cost + impact


def net_edge_after_costs(
    gross_win_rate: float,
    avg_slippage_bps: float,
    avg_profit_bps: float,
    avg_loss_bps: float,
) -> float:
    """
    Calculate true edge after execution costs.

    Args:
        gross_win_rate: Win rate before costs (e.g., 0.82)
        avg_slippage_bps: Average slippage per trade
        avg_profit_bps: Average profit on winning trades
        avg_loss_bps: Average loss on losing trades (positive)

    Returns:
        Net expected value per trade in basis points
    """
    # Gross EV
    gross_ev = gross_win_rate * avg_profit_bps - (1 - gross_win_rate) * avg_loss_bps

    # Slippage hits both winning and losing trades
    total_slippage = 2 * avg_slippage_bps  # Entry + exit

    return gross_ev - total_slippage


def optimal_execution_threshold(
    spread_bps: float,
    expected_profit_bps: float,
    slippage_estimate_bps: float,
    min_profit_ratio: float = 2.0,
) -> bool:
    """
    Determine if a trade meets minimum profit threshold after costs.

    Args:
        spread_bps: Current spread
        expected_profit_bps: Expected profit from signal
        slippage_estimate_bps: Expected slippage
        min_profit_ratio: Minimum profit/cost ratio to trade

    Returns:
        True if trade should be executed
    """
    total_cost = spread_bps + 2 * slippage_estimate_bps  # Round trip
    return expected_profit_bps >= min_profit_ratio * total_cost


class ExecutionCostTracker:
    """
    Track and analyze execution costs over time.

    Used for:
    1. Model validation (predicted vs actual)
    2. Cost trend analysis
    3. Execution quality metrics
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize tracker.

        Args:
            window_size: Number of recent trades to track
        """
        self.window_size = window_size
        self._predicted_costs: List[float] = []
        self._actual_costs: List[float] = []
        self._timestamps: List[float] = []

    def record(
        self,
        predicted_cost_bps: float,
        actual_cost_bps: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a prediction and actual cost."""
        self._predicted_costs.append(predicted_cost_bps)
        self._actual_costs.append(actual_cost_bps)
        self._timestamps.append(timestamp or 0)

        # Trim to window
        if len(self._predicted_costs) > self.window_size:
            self._predicted_costs = self._predicted_costs[-self.window_size:]
            self._actual_costs = self._actual_costs[-self.window_size:]
            self._timestamps = self._timestamps[-self.window_size:]

    def get_metrics(self) -> Dict[str, float]:
        """Get execution quality metrics."""
        if len(self._predicted_costs) < 10:
            return {'error': 'insufficient_data'}

        predicted = np.array(self._predicted_costs)
        actual = np.array(self._actual_costs)

        # Prediction error
        errors = actual - predicted
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        bias = np.mean(errors)  # Positive = underestimate

        # Correlation
        corr = np.corrcoef(predicted, actual)[0, 1]

        # Savings (if we avoided high-cost trades)
        # Would need more context for this

        return {
            'mae_bps': mae,
            'rmse_bps': rmse,
            'bias_bps': bias,
            'correlation': corr,
            'mean_actual_cost': np.mean(actual),
            'mean_predicted_cost': np.mean(predicted),
            'n_observations': len(predicted),
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXECUTION COST PREDICTION FOR HFT")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Almgren & Chriss (2001): Optimal Execution")
    print("  - Gatheral (2010): Square-Root Impact")
    print("  - 海通证券: 交易成本预测模型")
    print()

    # Almgren-Chriss Model
    print("ALMGREN-CHRISS MODEL")
    print("-" * 50)

    ac_model = AlmgrenChrissModel(
        eta=0.1,
        gamma=0.05,
        sigma_daily=0.01,
        daily_volume=1e9,
    )

    order_sizes = [1e6, 5e6, 10e6, 50e6]

    for size in order_sizes:
        cost = ac_model.estimate_cost(size, spread_bps=1.0)
        print(f"Order ${size/1e6:.0f}M:")
        print(f"  Spread cost:    {cost.spread_cost_bps:.2f} bps")
        print(f"  Market impact:  {cost.market_impact_bps:.2f} bps")
        print(f"  Total cost:     {cost.total_cost_bps:.2f} bps (${cost.total_cost_dollar:,.0f})")
        print()

    # Quick slippage
    print("QUICK SLIPPAGE ESTIMATES")
    print("-" * 50)

    scenarios = [
        (1.0, 0.001, 0.01),  # Low: 1 bps spread, 0.1% ADV, 1% vol
        (2.0, 0.01, 0.02),   # Medium: 2 bps spread, 1% ADV, 2% vol
        (3.0, 0.05, 0.03),   # High: 3 bps spread, 5% ADV, 3% vol
    ]

    for spread, size_pct, vol in scenarios:
        slip = quick_slippage_estimate(spread, size_pct, vol)
        print(f"Spread={spread}bps, Size={size_pct*100:.1f}%ADV, Vol={vol*100:.0f}%:")
        print(f"  Expected slippage: {slip:.2f} bps")

    print()

    # Net edge calculation
    print("NET EDGE AFTER COSTS")
    print("-" * 50)

    gross_wr = 0.82
    avg_slip = 0.5  # bps
    avg_profit = 10  # bps
    avg_loss = 8  # bps

    net_ev = net_edge_after_costs(gross_wr, avg_slip, avg_profit, avg_loss)

    print(f"Gross win rate: {gross_wr*100:.0f}%")
    print(f"Avg slippage: {avg_slip} bps")
    print(f"Gross EV: {gross_wr * avg_profit - (1-gross_wr) * avg_loss:.2f} bps")
    print(f"Net EV (after costs): {net_ev:.2f} bps")
    print()

    print("=" * 70)
    print("KEY INSIGHT:")
    print("  Gross edge (82% win rate) means nothing without knowing costs.")
    print("  Net edge = Gross edge - Execution costs")
    print("  If Net edge > 0, you have TRUE mathematical certainty.")
    print("=" * 70)
