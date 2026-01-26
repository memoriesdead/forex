"""
Almgren-Chriss Optimal Execution for FX
=========================================
Implementation of the Almgren-Chriss optimal execution framework
adapted for 24-hour FX markets with session-based liquidity.

The AC model minimizes: E[Cost] + λ·Var[Cost]

Where:
- E[Cost] = Expected execution shortfall
- λ = Risk aversion parameter
- Var[Cost] = Variance of execution cost

Key adaptations for FX:
1. Session-based volatility and liquidity parameters
2. No market close (continuous trading)
3. Dealer-based spread model instead of order book

References:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Almgren (2003): "Optimal Execution with Nonlinear Impact Functions"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging

from .config import (
    ExecutionConfig, ExecutionSlice, ExecutionSchedule, ExecutionStrategy,
    FXSession, get_current_session, get_session_config, get_symbol_config
)
from .market_impact_fx import FXMarketImpactModel

logger = logging.getLogger(__name__)


@dataclass
class ACParameters:
    """Almgren-Chriss model parameters."""
    # Position and time
    X: float              # Total shares to execute
    T: float              # Time horizon in seconds
    N: int                # Number of trading periods

    # Market parameters
    sigma: float          # Volatility (per second)
    eta: float            # Temporary impact coefficient
    gamma: float          # Permanent impact coefficient

    # Risk preferences
    lambda_: float        # Risk aversion coefficient

    @property
    def tau(self) -> float:
        """Length of each trading period."""
        return self.T / self.N

    @property
    def kappa(self) -> float:
        """
        Optimal trading rate parameter.
        kappa = sqrt(lambda * sigma^2 / eta)
        """
        if self.eta <= 0:
            return 1.0
        return np.sqrt(self.lambda_ * self.sigma ** 2 / self.eta)


@dataclass
class ACTrajectory:
    """Result of Almgren-Chriss optimal trajectory calculation."""
    # Time points (0, tau, 2*tau, ..., T)
    time_points: np.ndarray

    # Holdings at each time point (starts at X, ends at 0)
    holdings: np.ndarray

    # Trade sizes at each period
    trade_sizes: np.ndarray

    # Trade rates (shares per second)
    trade_rates: np.ndarray

    # Expected costs
    expected_cost: float          # Total expected cost
    expected_permanent_cost: float
    expected_temporary_cost: float
    variance_cost: float          # Variance of cost

    # Parameters used
    params: ACParameters

    @property
    def utility(self) -> float:
        """Expected cost + risk penalty."""
        return self.expected_cost + self.params.lambda_ * self.variance_cost


class AlmgrenChrissOptimizer:
    """
    Almgren-Chriss optimal execution trajectory calculator.

    The optimal trajectory for risk-averse execution is:

    x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    where kappa = sqrt(lambda * sigma^2 / eta)
    """

    def __init__(self,
                 config: Optional[ExecutionConfig] = None,
                 impact_model: Optional[FXMarketImpactModel] = None):
        self.config = config or ExecutionConfig()
        self.impact_model = impact_model or FXMarketImpactModel(self.config)

    def calculate_trajectory(self,
                            symbol: str,
                            total_quantity: float,
                            horizon_seconds: float,
                            num_periods: int,
                            mid_price: float,
                            volatility: Optional[float] = None,
                            risk_aversion: Optional[float] = None,
                            session: Optional[FXSession] = None) -> ACTrajectory:
        """
        Calculate optimal execution trajectory.

        Args:
            symbol: Currency pair
            total_quantity: Total position to liquidate
            horizon_seconds: Time horizon in seconds
            num_periods: Number of trading periods
            mid_price: Current mid price
            volatility: Price volatility (estimated if None)
            risk_aversion: Risk aversion parameter (from config if None)
            session: FX session (auto-detected if None)

        Returns:
            ACTrajectory with optimal trading schedule
        """
        if session is None:
            session = get_current_session()

        session_cfg = get_session_config(session)
        symbol_cfg = get_symbol_config(symbol)

        # Estimate volatility if not provided (bps per second)
        if volatility is None:
            # Default: 10 bps/day = ~0.0006 bps/second
            volatility = 10.0 / 86400 * session_cfg.volatility_multiplier

        # Get risk aversion
        if risk_aversion is None:
            risk_aversion = self.config.risk_aversion

        # Estimate impact coefficients from market model
        eta = self.config.temp_impact_coef * session_cfg.spread_multiplier
        gamma = self.config.perm_impact_coef * session_cfg.spread_multiplier

        # Build parameters
        params = ACParameters(
            X=total_quantity,
            T=horizon_seconds,
            N=num_periods,
            sigma=volatility,
            eta=eta,
            gamma=gamma,
            lambda_=risk_aversion
        )

        # Calculate optimal trajectory
        trajectory = self._solve_trajectory(params)

        return trajectory

    def _solve_trajectory(self, params: ACParameters) -> ACTrajectory:
        """
        Solve for optimal trajectory using closed-form solution.

        The optimal holdings trajectory is:
        x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)

        For kappa -> 0 (risk-neutral): x_k = X * (1 - t_k/T) (TWAP)
        For kappa -> inf (infinitely risk-averse): immediate execution
        """
        tau = params.tau
        kappa = params.kappa

        # Time points
        time_points = np.array([k * tau for k in range(params.N + 1)])

        # Calculate holdings at each time point
        if kappa * params.T < 0.01:
            # Risk-neutral case (TWAP limit)
            holdings = params.X * (1 - time_points / params.T)
        elif kappa * params.T > 100:
            # Extremely risk-averse (immediate execution)
            holdings = np.zeros(params.N + 1)
            holdings[0] = params.X
        else:
            # General case
            sinh_kT = np.sinh(kappa * params.T)
            holdings = params.X * np.sinh(kappa * (params.T - time_points)) / sinh_kT

        # Ensure boundary conditions
        holdings[0] = params.X
        holdings[-1] = 0

        # Calculate trade sizes (difference in holdings)
        trade_sizes = -np.diff(holdings)  # Positive = selling

        # Trade rates (shares per second in each period)
        trade_rates = trade_sizes / tau

        # Calculate expected costs
        expected_perm = self._calculate_permanent_cost(holdings, trade_sizes, params)
        expected_temp = self._calculate_temporary_cost(trade_rates, params)
        variance = self._calculate_variance(holdings, params)

        return ACTrajectory(
            time_points=time_points,
            holdings=holdings,
            trade_sizes=trade_sizes,
            trade_rates=trade_rates,
            expected_cost=expected_perm + expected_temp,
            expected_permanent_cost=expected_perm,
            expected_temporary_cost=expected_temp,
            variance_cost=variance,
            params=params
        )

    def _calculate_permanent_cost(self,
                                 holdings: np.ndarray,
                                 trade_sizes: np.ndarray,
                                 params: ACParameters) -> float:
        """
        Calculate expected permanent impact cost.

        E[Permanent Cost] = gamma * sum(x_k * n_k)
        """
        # Permanent cost = gamma * sum over trades of (holdings * trade_size)
        cost = 0.0
        for k in range(len(trade_sizes)):
            cost += holdings[k] * trade_sizes[k]

        return params.gamma * cost

    def _calculate_temporary_cost(self,
                                 trade_rates: np.ndarray,
                                 params: ACParameters) -> float:
        """
        Calculate expected temporary impact cost.

        E[Temporary Cost] = eta * tau * sum(v_k^2)
        where v_k = n_k / tau is the trade rate
        """
        # Temporary cost = eta * tau * sum(rate^2)
        cost = params.eta * params.tau * np.sum(trade_rates ** 2)
        return cost

    def _calculate_variance(self,
                           holdings: np.ndarray,
                           params: ACParameters) -> float:
        """
        Calculate variance of execution cost.

        Var[Cost] = sigma^2 * tau * sum(x_k^2)
        """
        # Variance from price movements while holding
        # Skip the last point (holdings = 0)
        variance = params.sigma ** 2 * params.tau * np.sum(holdings[:-1] ** 2)
        return variance

    def create_schedule(self,
                       order_id: str,
                       symbol: str,
                       direction: int,
                       total_quantity: float,
                       mid_price: float,
                       volatility: Optional[float] = None,
                       horizon_seconds: Optional[int] = None,
                       session: Optional[FXSession] = None) -> ExecutionSchedule:
        """
        Create an executable schedule from AC trajectory.

        Args:
            order_id: Unique order identifier
            symbol: Currency pair
            direction: 1 = buy, -1 = sell
            total_quantity: Total to execute
            mid_price: Current mid price
            volatility: Price volatility
            horizon_seconds: Execution window
            session: FX session

        Returns:
            ExecutionSchedule with slices
        """
        if horizon_seconds is None:
            horizon_seconds = self.config.default_horizon_seconds

        if session is None:
            session = get_current_session()

        # Calculate number of periods
        slice_interval = self.config.slice_interval_seconds
        num_periods = max(2, horizon_seconds // slice_interval)

        # Calculate optimal trajectory
        trajectory = self.calculate_trajectory(
            symbol=symbol,
            total_quantity=total_quantity,
            horizon_seconds=horizon_seconds,
            num_periods=num_periods,
            mid_price=mid_price,
            volatility=volatility,
            session=session
        )

        # Convert to execution slices
        now = datetime.now(timezone.utc)
        slices = []

        for i, (time_offset, qty) in enumerate(zip(trajectory.time_points[:-1],
                                                    trajectory.trade_sizes)):
            target_time = now + timedelta(seconds=float(time_offset))

            # Determine strategy for slice based on size and urgency
            if i == 0 and trajectory.params.kappa * trajectory.params.T > 5:
                # High urgency, use market for first slice
                strategy = ExecutionStrategy.MARKET
            else:
                strategy = ExecutionStrategy.LIMIT

            slice_obj = ExecutionSlice(
                slice_id=i,
                target_time=target_time,
                target_quantity=float(qty),
                strategy=strategy
            )
            slices.append(slice_obj)

        # Create schedule
        schedule = ExecutionSchedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            slices=slices,
            strategy=ExecutionStrategy.ALMGREN_CHRISS,
            horizon_seconds=horizon_seconds,
            expected_cost_bps=trajectory.expected_cost * 10000  # Convert to bps
        )

        return schedule


class AdaptiveAlmgrenChriss:
    """
    Adaptive Almgren-Chriss with real-time parameter updates.

    Updates trajectory based on:
    1. Actual fills vs expected
    2. Market condition changes
    3. Volatility regime shifts
    """

    def __init__(self,
                 base_optimizer: Optional[AlmgrenChrissOptimizer] = None):
        self.optimizer = base_optimizer or AlmgrenChrissOptimizer()
        self.execution_history: List[dict] = []

    def update_and_reoptimize(self,
                             current_trajectory: ACTrajectory,
                             time_elapsed: float,
                             quantity_remaining: float,
                             current_volatility: float,
                             current_session: FXSession) -> ACTrajectory:
        """
        Re-optimize trajectory based on current state.

        Args:
            current_trajectory: Original trajectory
            time_elapsed: Seconds since start
            quantity_remaining: Remaining position
            current_volatility: Current volatility estimate
            current_session: Current FX session

        Returns:
            New optimized trajectory
        """
        params = current_trajectory.params

        # Time remaining
        time_remaining = params.T - time_elapsed
        if time_remaining <= params.tau:
            # Not enough time to re-optimize
            return current_trajectory

        # Periods remaining
        periods_remaining = max(2, int(time_remaining / params.tau))

        # New parameters with updated volatility
        session_cfg = get_session_config(current_session)

        new_params = ACParameters(
            X=quantity_remaining,
            T=time_remaining,
            N=periods_remaining,
            sigma=current_volatility,
            eta=params.eta * session_cfg.spread_multiplier,
            gamma=params.gamma * session_cfg.spread_multiplier,
            lambda_=params.lambda_
        )

        # Solve new trajectory
        return self.optimizer._solve_trajectory(new_params)

    def record_fill(self,
                   target_qty: float,
                   filled_qty: float,
                   target_price: float,
                   fill_price: float,
                   slippage_bps: float):
        """Record a fill for future learning."""
        self.execution_history.append({
            'target_qty': target_qty,
            'filled_qty': filled_qty,
            'target_price': target_price,
            'fill_price': fill_price,
            'slippage_bps': slippage_bps,
            'timestamp': datetime.now(timezone.utc)
        })

        # Keep limited history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]


def get_ac_optimizer(config: Optional[ExecutionConfig] = None) -> AlmgrenChrissOptimizer:
    """Factory function to get AC optimizer."""
    return AlmgrenChrissOptimizer(config)


if __name__ == '__main__':
    print("Almgren-Chriss Optimal Execution Test")
    print("=" * 70)

    optimizer = AlmgrenChrissOptimizer()

    # Test case: Liquidate 1M EURUSD over 5 minutes
    trajectory = optimizer.calculate_trajectory(
        symbol='EURUSD',
        total_quantity=1_000_000,
        horizon_seconds=300,
        num_periods=10,
        mid_price=1.0850,
        volatility=0.0001,  # ~10 bps/day
        risk_aversion=1e-6,
        session=FXSession.LONDON
    )

    print("\nTrajectory for 1M EURUSD, 5 minutes, London Session")
    print("-" * 70)
    print(f"{'Period':<8} {'Time (s)':<10} {'Holdings':<15} {'Trade Size':<15} {'Rate (units/s)':<15}")
    print("-" * 70)

    for i in range(len(trajectory.trade_sizes)):
        print(f"{i:<8} {trajectory.time_points[i]:<10.0f} "
              f"{trajectory.holdings[i]:<15,.0f} "
              f"{trajectory.trade_sizes[i]:<15,.0f} "
              f"{trajectory.trade_rates[i]:<15,.0f}")

    print(f"\n{'Final':<8} {trajectory.time_points[-1]:<10.0f} "
          f"{trajectory.holdings[-1]:<15,.0f}")

    print("\n" + "=" * 70)
    print("Cost Analysis")
    print("-" * 70)
    print(f"Expected Permanent Cost: {trajectory.expected_permanent_cost * 10000:.2f} bps")
    print(f"Expected Temporary Cost: {trajectory.expected_temporary_cost * 10000:.2f} bps")
    print(f"Total Expected Cost:     {trajectory.expected_cost * 10000:.2f} bps")
    print(f"Cost Variance:           {trajectory.variance_cost * 10000:.4f} bps^2")
    print(f"Risk-Adjusted Utility:   {trajectory.utility * 10000:.2f} bps")

    # Compare different risk aversion levels
    print("\n" + "=" * 70)
    print("Impact of Risk Aversion (λ)")
    print("-" * 70)

    lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    for lam in lambdas:
        traj = optimizer.calculate_trajectory(
            symbol='EURUSD',
            total_quantity=1_000_000,
            horizon_seconds=300,
            num_periods=10,
            mid_price=1.0850,
            risk_aversion=lam,
            session=FXSession.LONDON
        )
        # How front-loaded is the execution?
        first_trade_pct = traj.trade_sizes[0] / traj.params.X * 100
        print(f"λ = {lam:.0e} | "
              f"First trade: {first_trade_pct:5.1f}% | "
              f"E[Cost]: {traj.expected_cost * 10000:5.2f} bps | "
              f"Var: {traj.variance_cost * 10000:.4f}")

    # Create executable schedule
    print("\n" + "=" * 70)
    print("Executable Schedule")
    print("-" * 70)

    schedule = optimizer.create_schedule(
        order_id='AC_001',
        symbol='EURUSD',
        direction=-1,  # Sell
        total_quantity=1_000_000,
        mid_price=1.0850,
        horizon_seconds=300
    )

    print(f"Order ID: {schedule.order_id}")
    print(f"Strategy: {schedule.strategy.value}")
    print(f"Total Qty: {schedule.total_quantity:,.0f}")
    print(f"Slices: {schedule.num_slices}")
    print(f"Expected Cost: {schedule.expected_cost_bps:.2f} bps")
    print("\nSlice Details:")
    for s in schedule.slices[:5]:
        print(f"  Slice {s.slice_id}: {s.target_quantity:,.0f} units at {s.target_time.strftime('%H:%M:%S')} ({s.strategy.value})")
    if schedule.num_slices > 5:
        print(f"  ... and {schedule.num_slices - 5} more slices")
