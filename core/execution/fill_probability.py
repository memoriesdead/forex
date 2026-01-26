"""
Fill Probability and Market Impact Models
==========================================
Advanced models for estimating order fill probability and execution quality.

Models included:
1. Queue-Based Fill Probability
2. Poisson Arrival Model
3. Market Impact Model (Almgren-Chriss style)
4. Slippage Estimation

Source: Aït-Sahalia (2017) "High Frequency Market Making"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.optimize import minimize
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = 1
    SELL = -1


@dataclass
class FillEstimate:
    """Fill probability and execution estimate."""
    fill_probability: float
    expected_time_to_fill: float  # seconds
    expected_slippage_bps: float
    market_impact_bps: float
    adverse_selection_bps: float
    confidence: float


@dataclass
class ExecutionResult:
    """Actual execution result for calibration."""
    order_id: str
    side: Side
    limit_price: float
    quantity: float
    filled: bool
    fill_price: Optional[float]
    fill_time: Optional[datetime]
    time_in_queue: float  # seconds
    queue_position_at_submit: float
    market_mid_at_fill: Optional[float]


class PoissonFillModel:
    """
    Poisson Arrival Model for Fill Probability.

    Models trade arrivals as Poisson process, estimates probability
    of filling based on queue position and arrival rate.

    P(fill within T) = P(arrivals >= queue_position within T)
    """

    def __init__(self, calibration_window: int = 1000):
        """
        Initialize model.

        Args:
            calibration_window: Number of trades to use for rate estimation
        """
        self.calibration_window = calibration_window
        self.trade_times: List[datetime] = []
        self.trade_volumes: List[float] = []

        # Calibrated parameters
        self.arrival_rate: float = 1.0  # Trades per second
        self.avg_trade_size: float = 100.0

    def record_trade(self, timestamp: datetime, volume: float) -> None:
        """Record a trade for rate estimation."""
        self.trade_times.append(timestamp)
        self.trade_volumes.append(volume)

        if len(self.trade_times) > self.calibration_window:
            self.trade_times.pop(0)
            self.trade_volumes.pop(0)

        self._recalibrate()

    def _recalibrate(self) -> None:
        """Recalibrate arrival rate from trade history."""
        if len(self.trade_times) < 10:
            return

        # Time span
        time_span = (self.trade_times[-1] - self.trade_times[0]).total_seconds()
        if time_span <= 0:
            return

        # Arrival rate (trades per second)
        self.arrival_rate = len(self.trade_times) / time_span

        # Average trade size
        self.avg_trade_size = np.mean(self.trade_volumes)

    def fill_probability(self, queue_position: float, time_horizon: float = 60.0) -> float:
        """
        Calculate probability of fill within time horizon.

        Args:
            queue_position: Quantity ahead in queue
            time_horizon: Time window in seconds

        Returns:
            Probability of fill [0, 1]
        """
        if queue_position <= 0:
            return 1.0

        # Expected volume to trade
        expected_volume = self.arrival_rate * self.avg_trade_size * time_horizon

        if expected_volume <= 0:
            return 0.0

        # Number of "average trades" needed to fill position
        trades_needed = queue_position / self.avg_trade_size

        # Poisson probability P(X >= trades_needed) where X ~ Poisson(rate * T)
        lambda_param = self.arrival_rate * time_horizon

        if lambda_param > 50:
            # Normal approximation for large lambda
            z = (trades_needed - lambda_param) / np.sqrt(lambda_param)
            return 1 - stats.norm.cdf(z)
        else:
            # Exact Poisson
            return 1 - stats.poisson.cdf(int(trades_needed) - 1, lambda_param)

    def expected_time_to_fill(self, queue_position: float) -> float:
        """
        Estimate expected time to fill.

        Args:
            queue_position: Quantity ahead in queue

        Returns:
            Expected time in seconds
        """
        if queue_position <= 0:
            return 0.0

        if self.arrival_rate <= 0 or self.avg_trade_size <= 0:
            return float('inf')

        # Expected time = queue_position / (arrival_rate * avg_trade_size)
        volume_rate = self.arrival_rate * self.avg_trade_size
        return queue_position / volume_rate


class MarketImpactModel:
    """
    Market Impact Model.

    Estimates temporary and permanent price impact of orders.

    Temporary impact: Price moves during execution, reverts after
    Permanent impact: Information content, price doesn't fully revert

    Based on: Almgren-Chriss optimal execution model
    """

    def __init__(self,
                 daily_volume: float = 1e6,
                 daily_volatility: float = 0.01,
                 temporary_impact_coef: float = 0.1,
                 permanent_impact_coef: float = 0.1):
        """
        Initialize market impact model.

        Args:
            daily_volume: Average daily volume
            daily_volatility: Daily volatility (as decimal)
            temporary_impact_coef: Temporary impact coefficient
            permanent_impact_coef: Permanent impact coefficient
        """
        self.daily_volume = daily_volume
        self.daily_volatility = daily_volatility
        self.temp_coef = temporary_impact_coef
        self.perm_coef = permanent_impact_coef

    def temporary_impact(self, order_size: float, execution_time: float) -> float:
        """
        Calculate temporary market impact in basis points.

        Args:
            order_size: Order size
            execution_time: Execution time in seconds

        Returns:
            Temporary impact in basis points
        """
        # Participation rate
        volume_per_second = self.daily_volume / (6.5 * 3600)  # Assuming 6.5 hour trading day
        participation = order_size / (volume_per_second * execution_time) if execution_time > 0 else 1.0

        # Square-root impact model
        impact = self.temp_coef * self.daily_volatility * np.sqrt(participation) * 10000

        return impact

    def permanent_impact(self, order_size: float) -> float:
        """
        Calculate permanent market impact in basis points.

        Args:
            order_size: Order size

        Returns:
            Permanent impact in basis points
        """
        # Linear impact model
        volume_fraction = order_size / self.daily_volume
        impact = self.perm_coef * self.daily_volatility * volume_fraction * 10000

        return impact

    def total_impact(self, order_size: float, execution_time: float = 60.0) -> float:
        """
        Calculate total market impact.

        Args:
            order_size: Order size
            execution_time: Expected execution time in seconds

        Returns:
            Total impact in basis points
        """
        return self.temporary_impact(order_size, execution_time) + self.permanent_impact(order_size)

    def optimal_execution_time(self, order_size: float, urgency: float = 0.5) -> float:
        """
        Calculate optimal execution time.

        Args:
            order_size: Order size
            urgency: Urgency parameter (0 = patient, 1 = aggressive)

        Returns:
            Optimal execution time in seconds
        """
        # Almgren-Chriss optimal trading rate
        # Balance between temporary impact (faster = worse) and timing risk (slower = worse)

        if urgency >= 1.0:
            return 1.0  # Execute immediately

        # Simple heuristic: larger orders need more time
        volume_fraction = order_size / self.daily_volume
        base_time = 60 * (1 + 100 * volume_fraction)  # Base time in seconds

        # Adjust for urgency
        return base_time * (1 - urgency)


class SlippageEstimator:
    """
    Slippage Estimation Model.

    Estimates expected slippage based on:
    - Order size vs available liquidity
    - Spread
    - Market impact
    - Adverse selection
    """

    def __init__(self):
        """Initialize slippage estimator."""
        self.historical_slippages: List[Dict] = []
        self.max_history = 1000

        # Calibrated parameters
        self.base_slippage_bps: float = 0.5
        self.size_impact_coef: float = 0.1
        self.spread_coef: float = 0.5

    def record_execution(self, result: ExecutionResult) -> None:
        """Record execution for calibration."""
        if not result.filled or result.fill_price is None:
            return

        # Calculate actual slippage
        if result.side == Side.BUY:
            slippage_bps = (result.fill_price - result.limit_price) / result.limit_price * 10000
        else:
            slippage_bps = (result.limit_price - result.fill_price) / result.limit_price * 10000

        self.historical_slippages.append({
            'slippage_bps': slippage_bps,
            'quantity': result.quantity,
            'queue_position': result.queue_position_at_submit,
            'time_in_queue': result.time_in_queue
        })

        if len(self.historical_slippages) > self.max_history:
            self.historical_slippages.pop(0)

        self._recalibrate()

    def _recalibrate(self) -> None:
        """Recalibrate from historical data."""
        if len(self.historical_slippages) < 20:
            return

        slippages = [h['slippage_bps'] for h in self.historical_slippages]
        self.base_slippage_bps = np.median(slippages)

    def estimate_slippage(self,
                          order_size: float,
                          spread_bps: float,
                          book_depth: float,
                          side: Side) -> float:
        """
        Estimate expected slippage.

        Args:
            order_size: Order quantity
            spread_bps: Current spread in basis points
            book_depth: Available liquidity at best level
            side: Order side

        Returns:
            Expected slippage in basis points
        """
        # Base slippage
        slippage = self.base_slippage_bps

        # Spread component (crossing = half spread)
        slippage += spread_bps * self.spread_coef

        # Size component (larger orders = more slippage)
        if book_depth > 0:
            size_ratio = order_size / book_depth
            slippage += self.size_impact_coef * size_ratio * 10  # Scale factor

        return slippage

    def estimate_execution_cost(self,
                                order_size: float,
                                mid_price: float,
                                spread_bps: float,
                                book_depth: float,
                                side: Side) -> Dict[str, float]:
        """
        Estimate total execution cost breakdown.

        Returns:
            Dict with cost components in basis points
        """
        # Slippage
        slippage = self.estimate_slippage(order_size, spread_bps, book_depth, side)

        # Commission (assume 0.1 bps for institutional)
        commission = 0.1

        # Spread cost (half-spread for market orders)
        spread_cost = spread_bps / 2

        # Total
        total = slippage + commission + spread_cost

        return {
            'slippage_bps': slippage,
            'commission_bps': commission,
            'spread_cost_bps': spread_cost,
            'total_cost_bps': total,
            'total_cost_dollars': total / 10000 * mid_price * order_size
        }


class FillProbabilityEngine:
    """
    Unified Fill Probability Engine.

    Combines all models to provide comprehensive fill estimates.
    """

    def __init__(self):
        """Initialize engine with all models."""
        self.poisson_model = PoissonFillModel()
        self.impact_model = MarketImpactModel()
        self.slippage_estimator = SlippageEstimator()

        # Adverse selection estimate
        self.adverse_selection_bps = 1.0

    def estimate_fill(self,
                      side: Side,
                      limit_price: float,
                      quantity: float,
                      queue_position: float,
                      mid_price: float,
                      spread_bps: float,
                      book_depth: float,
                      time_horizon: float = 60.0) -> FillEstimate:
        """
        Generate comprehensive fill estimate.

        Args:
            side: Order side
            limit_price: Limit order price
            quantity: Order quantity
            queue_position: Current queue position
            mid_price: Current mid price
            spread_bps: Current spread in bps
            book_depth: Depth at limit price level
            time_horizon: Time window in seconds

        Returns:
            FillEstimate with all components
        """
        # Fill probability
        fill_prob = self.poisson_model.fill_probability(queue_position, time_horizon)

        # Time to fill
        time_to_fill = self.poisson_model.expected_time_to_fill(queue_position)

        # Slippage (for limit orders that fill, minimal)
        slippage = self.slippage_estimator.estimate_slippage(
            quantity, spread_bps, book_depth, side
        )

        # Market impact (if we were to use market order)
        impact = self.impact_model.total_impact(quantity, time_to_fill)

        # Confidence based on data quality
        confidence = min(1.0, len(self.poisson_model.trade_times) / 100)

        return FillEstimate(
            fill_probability=fill_prob,
            expected_time_to_fill=time_to_fill,
            expected_slippage_bps=slippage,
            market_impact_bps=impact,
            adverse_selection_bps=self.adverse_selection_bps,
            confidence=confidence
        )

    def should_use_market_order(self,
                                limit_fill_estimate: FillEstimate,
                                urgency: float = 0.5) -> Tuple[bool, str]:
        """
        Decide whether to use market order instead of limit.

        Args:
            limit_fill_estimate: Fill estimate for limit order
            urgency: How urgent is the fill (0-1)

        Returns:
            (use_market, reason)
        """
        # Cost of waiting (opportunity cost)
        wait_cost = limit_fill_estimate.expected_time_to_fill * urgency * 0.01  # bps per second

        # Cost of limit order
        limit_cost = limit_fill_estimate.expected_slippage_bps * limit_fill_estimate.fill_probability

        # Cost of market order
        market_cost = limit_fill_estimate.market_impact_bps + limit_fill_estimate.adverse_selection_bps

        # Probability-adjusted comparison
        if limit_fill_estimate.fill_probability < 0.5:
            # Low fill prob, consider market
            if market_cost < limit_cost + wait_cost:
                return True, "Low fill probability, market order cheaper"

        if urgency > 0.8 and limit_fill_estimate.expected_time_to_fill > 30:
            return True, "High urgency, long expected wait"

        if market_cost < limit_cost * 0.5:
            return True, "Market order significantly cheaper"

        return False, "Limit order preferred"

    def record_trade(self, timestamp: datetime, volume: float) -> None:
        """Record trade for model calibration."""
        self.poisson_model.record_trade(timestamp, volume)

    def record_execution(self, result: ExecutionResult) -> None:
        """Record execution for model calibration."""
        self.slippage_estimator.record_execution(result)


class MakerTakerDecision:
    """
    Maker vs Taker Decision Model.

    Decides optimal order type based on:
    - Urgency
    - Spread
    - Fill probability
    - Fees (maker rebate vs taker fee)
    """

    def __init__(self,
                 maker_rebate_bps: float = 0.02,
                 taker_fee_bps: float = 0.05):
        """
        Initialize decision model.

        Args:
            maker_rebate_bps: Rebate for providing liquidity
            taker_fee_bps: Fee for taking liquidity
        """
        self.maker_rebate = maker_rebate_bps
        self.taker_fee = taker_fee_bps

    def optimal_aggressiveness(self,
                               fill_estimate: FillEstimate,
                               spread_bps: float,
                               urgency: float = 0.5) -> Dict[str, any]:
        """
        Calculate optimal order aggressiveness.

        Returns:
            Dict with recommendation and analysis
        """
        # Cost of different strategies
        strategies = {}

        # Passive maker (at bid/ask)
        maker_cost = (
            fill_estimate.expected_slippage_bps
            - self.maker_rebate
            + fill_estimate.adverse_selection_bps * fill_estimate.fill_probability
        )
        maker_expected = maker_cost * fill_estimate.fill_probability

        # Aggressive taker (cross spread)
        taker_cost = spread_bps / 2 + self.taker_fee + fill_estimate.market_impact_bps
        taker_expected = taker_cost  # 100% fill

        # Mid-price pegging
        mid_fill_prob = fill_estimate.fill_probability * 0.7  # Lower fill rate at mid
        mid_cost = spread_bps / 4 - self.maker_rebate / 2
        mid_expected = mid_cost * mid_fill_prob

        strategies = {
            'maker': {'cost': maker_expected, 'fill_prob': fill_estimate.fill_probability},
            'taker': {'cost': taker_expected, 'fill_prob': 1.0},
            'mid': {'cost': mid_expected, 'fill_prob': mid_fill_prob}
        }

        # Adjust for urgency
        for strat, data in strategies.items():
            wait_penalty = (1 - data['fill_prob']) * urgency * 10
            data['adjusted_cost'] = data['cost'] + wait_penalty

        # Find optimal
        optimal = min(strategies.items(), key=lambda x: x[1]['adjusted_cost'])

        return {
            'recommendation': optimal[0],
            'expected_cost_bps': optimal[1]['adjusted_cost'],
            'strategies': strategies,
            'urgency': urgency,
            'spread_bps': spread_bps
        }


if __name__ == '__main__':
    print("Fill Probability Engine Test")
    print("=" * 50)

    engine = FillProbabilityEngine()

    # Simulate some trades
    now = datetime.now()
    for i in range(100):
        engine.record_trade(now + timedelta(seconds=i * 0.5), 50 + np.random.randn() * 20)

    # Estimate fill for a limit order
    estimate = engine.estimate_fill(
        side=Side.BUY,
        limit_price=1.1000,
        quantity=100,
        queue_position=300,
        mid_price=1.1001,
        spread_bps=1.0,
        book_depth=500,
        time_horizon=60
    )

    print(f"\nFill Estimate:")
    print(f"  Fill Probability: {estimate.fill_probability:.2%}")
    print(f"  Expected Time: {estimate.expected_time_to_fill:.1f}s")
    print(f"  Expected Slippage: {estimate.expected_slippage_bps:.2f} bps")
    print(f"  Market Impact: {estimate.market_impact_bps:.2f} bps")
    print(f"  Adverse Selection: {estimate.adverse_selection_bps:.2f} bps")
    print(f"  Confidence: {estimate.confidence:.2%}")

    # Market order decision
    use_market, reason = engine.should_use_market_order(estimate, urgency=0.7)
    print(f"\nUse Market Order: {use_market}")
    print(f"Reason: {reason}")

    # Maker/Taker decision
    print("\n" + "=" * 50)
    print("Maker/Taker Decision")

    decision_model = MakerTakerDecision()
    decision = decision_model.optimal_aggressiveness(estimate, spread_bps=1.0, urgency=0.5)

    print(f"\nRecommendation: {decision['recommendation'].upper()}")
    print(f"Expected Cost: {decision['expected_cost_bps']:.2f} bps")

    print("\nStrategy Analysis:")
    for strat, data in decision['strategies'].items():
        print(f"  {strat}: cost={data['adjusted_cost']:.2f} bps, fill={data['fill_prob']:.0%}")
