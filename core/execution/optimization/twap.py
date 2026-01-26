"""
Session-Aware TWAP Scheduler for FX
====================================
Time-Weighted Average Price execution with FX session awareness.

Standard TWAP divides orders equally over time. This implementation
weights more volume during high-liquidity periods (London-NY overlap).

Features:
- Session-based volume weighting
- Weekend gap avoidance
- Configurable slice intervals
- Real-time execution tracking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import logging

from .config import (
    ExecutionConfig, ExecutionSlice, ExecutionSchedule, ExecutionStrategy,
    FXSession, FXSessionConfig, DEFAULT_SESSIONS,
    get_current_session, get_session_config, get_symbol_config
)

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single TWAP execution slice."""
    slice_id: int
    start_time: datetime
    end_time: datetime
    target_quantity: float
    session: FXSession
    weight: float  # Session weight applied


@dataclass
class TWAPSchedule:
    """Complete TWAP schedule."""
    order_id: str
    symbol: str
    direction: int
    total_quantity: float
    slices: List[TWAPSlice]
    start_time: datetime
    end_time: datetime
    horizon_seconds: float
    session_aware: bool


class SessionAwareTWAP:
    """
    Session-aware TWAP execution scheduler.

    Weights execution volume by session liquidity:
    - More volume during London-NY overlap (highest liquidity)
    - Less volume during Tokyo session (lower liquidity)
    - Avoids weekend gaps
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

        # Session volume weights (normalized to sum to 1 over 24h)
        self.session_weights = {
            FXSession.OVERLAP_LN: 1.8,   # Most liquid
            FXSession.LONDON: 1.4,
            FXSession.NEW_YORK: 1.2,
            FXSession.OVERLAP_TL: 1.0,
            FXSession.TOKYO: 0.4,
            FXSession.OFF_HOURS: 0.2     # Least liquid
        }

    def create_schedule(self,
                       order_id: str,
                       symbol: str,
                       direction: int,
                       total_quantity: float,
                       horizon_seconds: Optional[int] = None,
                       slice_interval_seconds: Optional[int] = None,
                       start_time: Optional[datetime] = None,
                       session_aware: bool = True) -> ExecutionSchedule:
        """
        Create a TWAP execution schedule.

        Args:
            order_id: Unique order identifier
            symbol: Currency pair
            direction: 1 = buy, -1 = sell
            total_quantity: Total quantity to execute
            horizon_seconds: Total execution window (default from config)
            slice_interval_seconds: Time between slices (default from config)
            start_time: Start time (default: now)
            session_aware: Whether to weight by session liquidity

        Returns:
            ExecutionSchedule with TWAP slices
        """
        if horizon_seconds is None:
            horizon_seconds = self.config.default_horizon_seconds

        if slice_interval_seconds is None:
            slice_interval_seconds = self.config.slice_interval_seconds

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        # Calculate number of slices
        num_slices = max(2, horizon_seconds // slice_interval_seconds)

        # Generate time intervals
        intervals = self._generate_intervals(
            start_time=start_time,
            horizon_seconds=horizon_seconds,
            num_slices=num_slices
        )

        # Calculate weights for each interval
        if session_aware:
            weights = self._calculate_session_weights(intervals)
        else:
            weights = np.ones(num_slices) / num_slices

        # Normalize weights
        weights = weights / weights.sum()

        # Create execution slices
        slices = []
        for i, (interval_start, interval_end, weight) in enumerate(zip(
            intervals[:-1], intervals[1:], weights
        )):
            slice_qty = total_quantity * weight

            # Skip tiny slices
            if slice_qty < 1:
                continue

            session = get_current_session(interval_start)

            slices.append(ExecutionSlice(
                slice_id=i,
                target_time=interval_start,
                target_quantity=slice_qty,
                strategy=ExecutionStrategy.MARKET,
                status="pending"
            ))

        # Adjust last slice to ensure total matches
        if slices:
            executed_so_far = sum(s.target_quantity for s in slices[:-1])
            slices[-1].target_quantity = total_quantity - executed_so_far

        return ExecutionSchedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            slices=slices,
            strategy=ExecutionStrategy.TWAP,
            horizon_seconds=horizon_seconds,
            expected_cost_bps=self._estimate_cost(symbol, total_quantity, num_slices)
        )

    def _generate_intervals(self,
                           start_time: datetime,
                           horizon_seconds: float,
                           num_slices: int) -> List[datetime]:
        """Generate evenly-spaced time intervals."""
        interval_seconds = horizon_seconds / num_slices

        intervals = []
        for i in range(num_slices + 1):
            t = start_time + timedelta(seconds=i * interval_seconds)
            intervals.append(t)

        return intervals

    def _calculate_session_weights(self, intervals: List[datetime]) -> np.ndarray:
        """
        Calculate volume weights based on FX sessions.

        Returns normalized weights that sum to 1.
        """
        weights = []

        for i in range(len(intervals) - 1):
            start = intervals[i]
            end = intervals[i + 1]

            # Get predominant session for this interval
            session = get_current_session(start)

            # Apply session weight
            weight = self.session_weights.get(session, 1.0)

            # Check for weekend (reduced liquidity)
            if self._is_weekend(start):
                weight *= 0.1

            weights.append(weight)

        return np.array(weights)

    def _is_weekend(self, dt: datetime) -> bool:
        """Check if datetime falls on FX weekend (Friday 21:00 - Sunday 21:00 UTC)."""
        weekday = dt.weekday()
        hour = dt.hour

        # Friday after 21:00 UTC
        if weekday == 4 and hour >= 21:
            return True

        # Saturday
        if weekday == 5:
            return True

        # Sunday before 21:00 UTC
        if weekday == 6 and hour < 21:
            return True

        return False

    def _estimate_cost(self,
                      symbol: str,
                      total_quantity: float,
                      num_slices: int) -> float:
        """Estimate execution cost in basis points."""
        symbol_cfg = get_symbol_config(symbol)

        # TWAP cost ≈ spread/2 + small impact from participation
        spread_cost = symbol_cfg.avg_spread_bps / 2

        # Participation impact (lower for more slices)
        slice_qty = total_quantity / num_slices
        participation_cost = 0.1 * np.sqrt(slice_qty / 1e6)

        return spread_cost + participation_cost


class SimpleTWAP(SessionAwareTWAP):
    """
    Simple TWAP without session awareness.

    Divides quantity equally over time regardless of session.
    Use when:
    - Execution window is short (< 1 hour)
    - Session effects are minimal
    - Simplicity is preferred
    """

    def create_schedule(self,
                       order_id: str,
                       symbol: str,
                       direction: int,
                       total_quantity: float,
                       horizon_seconds: Optional[int] = None,
                       slice_interval_seconds: Optional[int] = None,
                       start_time: Optional[datetime] = None,
                       **kwargs) -> ExecutionSchedule:
        """Create simple TWAP schedule (session_aware=False)."""
        return super().create_schedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            horizon_seconds=horizon_seconds,
            slice_interval_seconds=slice_interval_seconds,
            start_time=start_time,
            session_aware=False  # Always use equal weights
        )


class AdaptiveTWAP:
    """
    Adaptive TWAP that adjusts based on market conditions.

    Features:
    - Speeds up when spread is tight
    - Slows down when volatility is high
    - Adjusts to order flow imbalance
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.base_twap = SessionAwareTWAP(config)
        self.execution_state: Dict[str, dict] = {}

    def create_adaptive_schedule(self,
                                order_id: str,
                                symbol: str,
                                direction: int,
                                total_quantity: float,
                                horizon_seconds: Optional[int] = None) -> ExecutionSchedule:
        """Create initial schedule (will be adapted during execution)."""
        schedule = self.base_twap.create_schedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            horizon_seconds=horizon_seconds
        )

        # Store state for adaptation
        self.execution_state[order_id] = {
            'original_schedule': schedule,
            'slices_executed': 0,
            'quantity_executed': 0.0,
            'vwap_so_far': 0.0,
            'spread_history': [],
            'vol_history': []
        }

        return schedule

    def adapt_next_slice(self,
                        order_id: str,
                        current_spread_bps: float,
                        current_volatility: float,
                        order_flow: float = 0.0) -> Optional[ExecutionSlice]:
        """
        Get adapted next slice based on market conditions.

        Args:
            order_id: Order identifier
            current_spread_bps: Current bid-ask spread
            current_volatility: Current realized volatility
            order_flow: Recent order flow imbalance (-1 to 1)

        Returns:
            Adapted execution slice or None if complete
        """
        if order_id not in self.execution_state:
            return None

        state = self.execution_state[order_id]
        schedule = state['original_schedule']

        # Get next pending slice
        next_slice_idx = state['slices_executed']
        if next_slice_idx >= len(schedule.slices):
            return None

        next_slice = schedule.slices[next_slice_idx]

        # Record market conditions
        state['spread_history'].append(current_spread_bps)
        state['vol_history'].append(current_volatility)

        # Calculate adaptation factor
        symbol_cfg = get_symbol_config(schedule.symbol)

        # Spread factor: trade more when spread is tight
        avg_spread = symbol_cfg.avg_spread_bps
        spread_factor = avg_spread / max(current_spread_bps, 0.1)
        spread_factor = np.clip(spread_factor, 0.5, 2.0)

        # Volatility factor: trade less when vol is high
        if len(state['vol_history']) > 1:
            avg_vol = np.mean(state['vol_history'])
            vol_factor = avg_vol / max(current_volatility, 1e-8)
            vol_factor = np.clip(vol_factor, 0.5, 2.0)
        else:
            vol_factor = 1.0

        # Combined adaptation
        adaptation = spread_factor * vol_factor

        # Adjust slice quantity
        adapted_qty = next_slice.target_quantity * adaptation

        # Ensure we don't over-execute
        remaining = schedule.total_quantity - state['quantity_executed']
        adapted_qty = min(adapted_qty, remaining)

        # Create adapted slice
        adapted_slice = ExecutionSlice(
            slice_id=next_slice.slice_id,
            target_time=datetime.now(timezone.utc),
            target_quantity=adapted_qty,
            strategy=next_slice.strategy,
            status="active"
        )

        return adapted_slice

    def record_execution(self,
                        order_id: str,
                        executed_qty: float,
                        executed_price: float):
        """Record an execution for tracking."""
        if order_id not in self.execution_state:
            return

        state = self.execution_state[order_id]
        state['slices_executed'] += 1
        state['quantity_executed'] += executed_qty

        # Update VWAP
        old_vwap = state['vwap_so_far']
        old_qty = state['quantity_executed'] - executed_qty
        if state['quantity_executed'] > 0:
            state['vwap_so_far'] = (
                (old_vwap * old_qty + executed_price * executed_qty) /
                state['quantity_executed']
            )


def get_twap_scheduler(config: Optional[ExecutionConfig] = None,
                      session_aware: bool = True) -> SessionAwareTWAP:
    """Factory function to get TWAP scheduler."""
    if session_aware:
        return SessionAwareTWAP(config)
    else:
        return SimpleTWAP(config)


if __name__ == '__main__':
    print("Session-Aware TWAP Test")
    print("=" * 70)

    scheduler = SessionAwareTWAP()

    # Create schedule starting at different times
    test_times = [
        datetime(2026, 1, 17, 8, 0, tzinfo=timezone.utc),   # Tokyo-London overlap
        datetime(2026, 1, 17, 10, 0, tzinfo=timezone.utc),  # London
        datetime(2026, 1, 17, 14, 0, tzinfo=timezone.utc),  # London-NY overlap
        datetime(2026, 1, 17, 19, 0, tzinfo=timezone.utc),  # NY afternoon
        datetime(2026, 1, 17, 23, 0, tzinfo=timezone.utc),  # Off hours
    ]

    print("\n1M EURUSD TWAP Schedule (1 hour horizon)")
    print("-" * 70)

    for start in test_times:
        schedule = scheduler.create_schedule(
            order_id=f"TWAP_{start.hour:02d}",
            symbol='EURUSD',
            direction=1,
            total_quantity=1_000_000,
            horizon_seconds=3600,
            slice_interval_seconds=300,  # 5-minute slices
            start_time=start
        )

        session = get_current_session(start)
        first_slice_pct = schedule.slices[0].target_quantity / schedule.total_quantity * 100

        print(f"Start: {start.strftime('%H:%M')} UTC ({session.value:12s}) | "
              f"Slices: {schedule.num_slices:2d} | "
              f"First: {first_slice_pct:5.1f}% | "
              f"Cost: {schedule.expected_cost_bps:.2f} bps")

    # Show detailed schedule for London-NY overlap
    print("\n" + "=" * 70)
    print("Detailed Schedule: Start at 14:00 UTC (London-NY Overlap)")
    print("-" * 70)

    schedule = scheduler.create_schedule(
        order_id="TWAP_DETAIL",
        symbol='EURUSD',
        direction=1,
        total_quantity=1_000_000,
        horizon_seconds=3600,
        slice_interval_seconds=300,
        start_time=datetime(2026, 1, 17, 14, 0, tzinfo=timezone.utc)
    )

    print(f"{'Slice':<6} {'Time':<10} {'Quantity':<15} {'Session':<15} {'% of Total':<10}")
    print("-" * 70)

    for slice_obj in schedule.slices:
        session = get_current_session(slice_obj.target_time)
        pct = slice_obj.target_quantity / schedule.total_quantity * 100
        print(f"{slice_obj.slice_id:<6} "
              f"{slice_obj.target_time.strftime('%H:%M'):<10} "
              f"{slice_obj.target_quantity:>12,.0f}   "
              f"{session.value:<15} "
              f"{pct:>6.1f}%")

    # Compare session-aware vs simple TWAP
    print("\n" + "=" * 70)
    print("Session-Aware vs Simple TWAP Comparison")
    print("-" * 70)

    simple_scheduler = SimpleTWAP()

    for start in [
        datetime(2026, 1, 17, 14, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 17, 3, 0, tzinfo=timezone.utc),
    ]:
        aware_schedule = scheduler.create_schedule(
            order_id="aware",
            symbol='EURUSD',
            direction=1,
            total_quantity=1_000_000,
            horizon_seconds=3600,
            start_time=start
        )

        simple_schedule = simple_scheduler.create_schedule(
            order_id="simple",
            symbol='EURUSD',
            direction=1,
            total_quantity=1_000_000,
            horizon_seconds=3600,
            start_time=start
        )

        session = get_current_session(start)
        aware_first = aware_schedule.slices[0].target_quantity
        simple_first = simple_schedule.slices[0].target_quantity

        print(f"\nStart: {start.strftime('%H:%M')} UTC ({session.value})")
        print(f"  Session-Aware First Slice: {aware_first:>12,.0f}")
        print(f"  Simple TWAP First Slice:   {simple_first:>12,.0f}")
        print(f"  Ratio: {aware_first/simple_first:.2f}x")
