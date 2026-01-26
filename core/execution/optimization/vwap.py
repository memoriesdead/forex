"""
FX Volume-Profile VWAP Scheduler
================================
Volume-Weighted Average Price execution with FX-specific volume profiles.

Unlike equities where volume data is public, FX volume must be estimated.
This implementation uses:
- BIS Triennial Survey data for session volumes
- Intraday volume patterns by session
- Symbol-specific volume characteristics

Features:
- Estimated FX volume profiles
- Participation rate caps
- Real-time volume tracking
- Adaptive execution based on actual vs expected volume
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone, timedelta
import logging

from .config import (
    ExecutionConfig, ExecutionSlice, ExecutionSchedule, ExecutionStrategy,
    FXSession, FXSessionConfig, DEFAULT_SESSIONS,
    get_current_session, get_session_config, get_symbol_config
)

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """
    Intraday volume profile for FX.

    Represents expected volume distribution across 24 hours.
    """
    # Hourly volume weights (24 values, one per hour UTC)
    hourly_weights: np.ndarray

    # Session-level weights
    session_weights: Dict[FXSession, float]

    # Symbol this profile is for
    symbol: str

    def get_weight_at(self, dt: datetime) -> float:
        """Get volume weight at a specific time."""
        hour = dt.hour
        return self.hourly_weights[hour]

    def get_period_weight(self,
                         start: datetime,
                         end: datetime,
                         interval_minutes: int = 5) -> float:
        """Get total volume weight for a time period."""
        total_weight = 0.0
        current = start

        while current < end:
            # Weight is proportional to time in this hour
            minutes_in_period = min(
                (end - current).total_seconds() / 60,
                interval_minutes
            )
            weight = self.hourly_weights[current.hour] * (minutes_in_period / 60)
            total_weight += weight
            current += timedelta(minutes=interval_minutes)

        return total_weight


# FX Volume profiles based on BIS data and market microstructure
def create_volume_profile(symbol: str) -> VolumeProfile:
    """
    Create FX volume profile for a symbol.

    Based on BIS Triennial Survey and market structure:
    - London dominates (37% of volume)
    - NY second (19% of volume)
    - Tokyo third (6% of volume)
    - Overlap periods have highest volume
    """
    # Base hourly profile (normalized, peaks during London-NY overlap)
    # Hours are UTC
    base_profile = np.array([
        0.3,   # 00:00 - Tokyo session start
        0.4,   # 01:00 - Tokyo
        0.5,   # 02:00 - Tokyo
        0.5,   # 03:00 - Tokyo peak
        0.4,   # 04:00 - Tokyo
        0.4,   # 05:00 - Tokyo winding down
        0.5,   # 06:00 - Tokyo-London transition
        0.8,   # 07:00 - London open, TL overlap
        1.0,   # 08:00 - London ramp up
        1.2,   # 09:00 - London peak
        1.2,   # 10:00 - London
        1.1,   # 11:00 - London
        1.5,   # 12:00 - London-NY overlap start
        1.8,   # 13:00 - LN overlap peak
        1.8,   # 14:00 - LN overlap peak
        1.6,   # 15:00 - LN overlap
        1.2,   # 16:00 - London close
        1.0,   # 17:00 - NY afternoon
        0.9,   # 18:00 - NY
        0.7,   # 19:00 - NY winding down
        0.5,   # 20:00 - NY close
        0.3,   # 21:00 - Off hours
        0.2,   # 22:00 - Off hours
        0.2,   # 23:00 - Off hours
    ])

    # Normalize to sum to 1
    base_profile = base_profile / base_profile.sum()

    # Symbol-specific adjustments
    if 'JPY' in symbol:
        # More Tokyo volume for JPY pairs
        tokyo_boost = np.zeros(24)
        tokyo_boost[0:9] = 0.3  # Boost Tokyo hours
        base_profile = base_profile + tokyo_boost * base_profile
        base_profile = base_profile / base_profile.sum()

    elif 'EUR' in symbol or 'GBP' in symbol:
        # More London volume for EUR/GBP pairs
        london_boost = np.zeros(24)
        london_boost[7:16] = 0.2  # Boost London hours
        base_profile = base_profile + london_boost * base_profile
        base_profile = base_profile / base_profile.sum()

    # Session weights
    session_weights = {
        FXSession.TOKYO: 0.15,
        FXSession.LONDON: 0.37,
        FXSession.NEW_YORK: 0.19,
        FXSession.OVERLAP_LN: 0.20,
        FXSession.OVERLAP_TL: 0.05,
        FXSession.OFF_HOURS: 0.04
    }

    return VolumeProfile(
        hourly_weights=base_profile,
        session_weights=session_weights,
        symbol=symbol
    )


class FXVWAPScheduler:
    """
    Volume-Weighted Average Price scheduler for FX.

    Executes proportionally to estimated volume profile to
    minimize market impact and achieve VWAP benchmark.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.volume_profiles: Dict[str, VolumeProfile] = {}

    def get_volume_profile(self, symbol: str) -> VolumeProfile:
        """Get or create volume profile for symbol."""
        if symbol not in self.volume_profiles:
            self.volume_profiles[symbol] = create_volume_profile(symbol)
        return self.volume_profiles[symbol]

    def create_schedule(self,
                       order_id: str,
                       symbol: str,
                       direction: int,
                       total_quantity: float,
                       horizon_seconds: Optional[int] = None,
                       slice_interval_seconds: Optional[int] = None,
                       start_time: Optional[datetime] = None,
                       max_participation_rate: Optional[float] = None) -> ExecutionSchedule:
        """
        Create a VWAP execution schedule.

        Args:
            order_id: Unique order identifier
            symbol: Currency pair
            direction: 1 = buy, -1 = sell
            total_quantity: Total quantity to execute
            horizon_seconds: Total execution window
            slice_interval_seconds: Time between slices
            start_time: Start time (default: now)
            max_participation_rate: Max percentage of estimated volume

        Returns:
            ExecutionSchedule with VWAP slices
        """
        if horizon_seconds is None:
            horizon_seconds = self.config.default_horizon_seconds

        if slice_interval_seconds is None:
            slice_interval_seconds = self.config.slice_interval_seconds

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if max_participation_rate is None:
            max_participation_rate = self.config.max_participation_rate

        # Get volume profile
        profile = self.get_volume_profile(symbol)
        symbol_cfg = get_symbol_config(symbol)

        # Calculate slices
        num_slices = max(2, horizon_seconds // slice_interval_seconds)
        end_time = start_time + timedelta(seconds=horizon_seconds)

        # Generate time intervals
        intervals = []
        for i in range(num_slices + 1):
            t = start_time + timedelta(seconds=i * slice_interval_seconds)
            intervals.append(t)

        # Calculate volume weights for each interval
        weights = []
        for i in range(len(intervals) - 1):
            weight = profile.get_period_weight(
                intervals[i],
                intervals[i + 1],
                interval_minutes=slice_interval_seconds // 60
            )
            weights.append(weight)

        weights = np.array(weights)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(num_slices) / num_slices

        # Calculate slice quantities
        slice_quantities = total_quantity * weights

        # Apply participation rate cap
        slice_quantities = self._apply_participation_cap(
            slice_quantities=slice_quantities,
            intervals=intervals,
            symbol_cfg=symbol_cfg,
            max_rate=max_participation_rate
        )

        # Ensure we execute full quantity (redistribute any capped amounts)
        total_allocated = slice_quantities.sum()
        if total_allocated < total_quantity:
            # Redistribute to uncapped slices
            shortfall = total_quantity - total_allocated
            uncapped_mask = slice_quantities < (total_quantity / num_slices * 1.5)
            if uncapped_mask.any():
                slice_quantities[uncapped_mask] += shortfall / uncapped_mask.sum()

        # Create execution slices
        slices = []
        for i, (interval_start, qty) in enumerate(zip(intervals[:-1], slice_quantities)):
            if qty < 1:  # Skip tiny slices
                continue

            slices.append(ExecutionSlice(
                slice_id=i,
                target_time=interval_start,
                target_quantity=float(qty),
                strategy=ExecutionStrategy.MARKET,
                status="pending"
            ))

        # Estimate execution cost
        expected_cost = self._estimate_cost(
            symbol=symbol,
            total_quantity=total_quantity,
            weights=weights,
            intervals=intervals
        )

        return ExecutionSchedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            slices=slices,
            strategy=ExecutionStrategy.VWAP,
            horizon_seconds=horizon_seconds,
            expected_cost_bps=expected_cost
        )

    def _apply_participation_cap(self,
                                slice_quantities: np.ndarray,
                                intervals: List[datetime],
                                symbol_cfg,
                                max_rate: float) -> np.ndarray:
        """
        Apply participation rate cap to slice quantities.

        Ensures we don't exceed max_rate of estimated volume.
        """
        capped = slice_quantities.copy()

        for i, qty in enumerate(slice_quantities):
            if i >= len(intervals) - 1:
                continue

            interval_start = intervals[i]
            interval_end = intervals[i + 1]

            # Estimate volume during this interval
            session = get_current_session(interval_start)
            session_cfg = get_session_config(session)

            interval_seconds = (interval_end - interval_start).total_seconds()
            daily_volume = symbol_cfg.daily_volume_estimate

            # Volume in this interval
            session_fraction = session_cfg.volume_weight
            interval_fraction = interval_seconds / (24 * 3600)
            estimated_volume = daily_volume * session_fraction * interval_fraction

            # Apply cap
            max_qty = estimated_volume * max_rate
            capped[i] = min(qty, max_qty)

        return capped

    def _estimate_cost(self,
                      symbol: str,
                      total_quantity: float,
                      weights: np.ndarray,
                      intervals: List[datetime]) -> float:
        """
        Estimate VWAP execution cost.

        VWAP cost = spread/2 + impact from participation
        """
        symbol_cfg = get_symbol_config(symbol)

        # Base spread cost
        spread_cost = symbol_cfg.avg_spread_bps / 2

        # Impact from participation (weighted by session)
        total_impact = 0.0
        for i, (weight, interval_start) in enumerate(zip(weights, intervals[:-1])):
            if weight <= 0:
                continue

            session = get_current_session(interval_start)
            session_cfg = get_session_config(session)

            # Slice quantity
            slice_qty = total_quantity * weight

            # Participation-based impact (lower in liquid sessions)
            session_impact = 0.1 * np.sqrt(slice_qty / 1e6)
            session_impact *= session_cfg.spread_multiplier

            total_impact += session_impact * weight

        return spread_cost + total_impact


class AdaptiveVWAP:
    """
    Adaptive VWAP that tracks actual volume and adjusts execution.

    Features:
    - Tracks actual vs expected volume
    - Speeds up/slows down based on volume realization
    - Maintains target VWAP tracking
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.base_vwap = FXVWAPScheduler(config)
        self.execution_state: Dict[str, dict] = {}

    def start_execution(self,
                       order_id: str,
                       symbol: str,
                       direction: int,
                       total_quantity: float,
                       horizon_seconds: int) -> ExecutionSchedule:
        """Start VWAP execution with tracking."""
        schedule = self.base_vwap.create_schedule(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            total_quantity=total_quantity,
            horizon_seconds=horizon_seconds
        )

        # Initialize tracking state
        self.execution_state[order_id] = {
            'schedule': schedule,
            'profile': self.base_vwap.get_volume_profile(symbol),
            'start_time': datetime.now(timezone.utc),
            'expected_volume': 0.0,
            'actual_volume': 0.0,
            'executed_qty': 0.0,
            'slices_executed': 0,
            'vwap': 0.0,
            'market_vwap': 0.0
        }

        return schedule

    def update_volume(self,
                     order_id: str,
                     observed_volume: float,
                     market_vwap: float) -> float:
        """
        Update with observed market volume.

        Args:
            order_id: Order identifier
            observed_volume: Actual volume observed since last update
            market_vwap: Current market VWAP

        Returns:
            Recommended execution rate adjustment (1.0 = on track)
        """
        if order_id not in self.execution_state:
            return 1.0

        state = self.execution_state[order_id]
        state['actual_volume'] += observed_volume
        state['market_vwap'] = market_vwap

        # Calculate expected volume up to now
        elapsed = (datetime.now(timezone.utc) - state['start_time']).total_seconds()
        schedule = state['schedule']
        profile = state['profile']

        expected = profile.get_period_weight(
            state['start_time'],
            datetime.now(timezone.utc)
        ) * schedule.total_quantity

        state['expected_volume'] = expected

        # Volume ratio
        if expected > 0:
            volume_ratio = state['actual_volume'] / expected
        else:
            volume_ratio = 1.0

        # If volume is higher than expected, we can execute faster
        # If volume is lower, we should slow down
        adjustment = np.clip(volume_ratio, 0.5, 2.0)

        return adjustment

    def get_next_slice(self,
                      order_id: str,
                      volume_adjustment: float = 1.0) -> Optional[ExecutionSlice]:
        """Get next execution slice with optional adjustment."""
        if order_id not in self.execution_state:
            return None

        state = self.execution_state[order_id]
        schedule = state['schedule']

        idx = state['slices_executed']
        if idx >= len(schedule.slices):
            return None

        base_slice = schedule.slices[idx]

        # Adjust quantity
        adjusted_qty = base_slice.target_quantity * volume_adjustment

        # Don't exceed remaining
        remaining = schedule.total_quantity - state['executed_qty']
        adjusted_qty = min(adjusted_qty, remaining)

        return ExecutionSlice(
            slice_id=base_slice.slice_id,
            target_time=datetime.now(timezone.utc),
            target_quantity=adjusted_qty,
            strategy=base_slice.strategy,
            status="active"
        )

    def record_fill(self,
                   order_id: str,
                   filled_qty: float,
                   fill_price: float):
        """Record a fill."""
        if order_id not in self.execution_state:
            return

        state = self.execution_state[order_id]

        # Update VWAP
        old_value = state['vwap'] * state['executed_qty']
        state['executed_qty'] += filled_qty
        if state['executed_qty'] > 0:
            state['vwap'] = (old_value + fill_price * filled_qty) / state['executed_qty']

        state['slices_executed'] += 1

    def get_tracking_error(self, order_id: str) -> float:
        """
        Calculate VWAP tracking error.

        Returns difference between our VWAP and market VWAP in bps.
        """
        if order_id not in self.execution_state:
            return 0.0

        state = self.execution_state[order_id]

        if state['vwap'] <= 0 or state['market_vwap'] <= 0:
            return 0.0

        # Tracking error in bps
        error = (state['vwap'] - state['market_vwap']) / state['market_vwap'] * 10000

        return error


def get_vwap_scheduler(config: Optional[ExecutionConfig] = None) -> FXVWAPScheduler:
    """Factory function to get VWAP scheduler."""
    return FXVWAPScheduler(config)


if __name__ == '__main__':
    print("FX VWAP Scheduler Test")
    print("=" * 70)

    scheduler = FXVWAPScheduler()

    # Show volume profile for EURUSD
    print("\nEURUSD Hourly Volume Profile")
    print("-" * 70)

    profile = scheduler.get_volume_profile('EURUSD')

    print("Hour | Weight | Bar")
    print("-" * 70)
    for hour in range(24):
        weight = profile.hourly_weights[hour]
        bar = "#" * int(weight * 200)
        print(f" {hour:02d}  | {weight:.3f}  | {bar}")

    # Create VWAP schedule
    print("\n" + "=" * 70)
    print("1M EURUSD VWAP Schedule (1 hour at 14:00 UTC)")
    print("-" * 70)

    schedule = scheduler.create_schedule(
        order_id="VWAP_001",
        symbol='EURUSD',
        direction=1,
        total_quantity=1_000_000,
        horizon_seconds=3600,
        slice_interval_seconds=300,
        start_time=datetime(2026, 1, 17, 14, 0, tzinfo=timezone.utc)
    )

    print(f"Order ID: {schedule.order_id}")
    print(f"Strategy: {schedule.strategy.value}")
    print(f"Slices: {schedule.num_slices}")
    print(f"Expected Cost: {schedule.expected_cost_bps:.2f} bps")
    print(f"\n{'Slice':<6} {'Time':<10} {'Quantity':<15} {'% Total':<10}")
    print("-" * 50)

    for s in schedule.slices:
        pct = s.target_quantity / schedule.total_quantity * 100
        print(f"{s.slice_id:<6} {s.target_time.strftime('%H:%M'):<10} "
              f"{s.target_quantity:>12,.0f}   {pct:>6.1f}%")

    # Compare VWAP for different symbols
    print("\n" + "=" * 70)
    print("VWAP Volume Distribution by Symbol (1 hour at 03:00 UTC - Tokyo)")
    print("-" * 70)

    symbols = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDJPY']
    for symbol in symbols:
        schedule = scheduler.create_schedule(
            order_id=f"VWAP_{symbol}",
            symbol=symbol,
            direction=1,
            total_quantity=1_000_000,
            horizon_seconds=3600,
            start_time=datetime(2026, 1, 17, 3, 0, tzinfo=timezone.utc)
        )

        first_slice_pct = schedule.slices[0].target_quantity / schedule.total_quantity * 100

        print(f"{symbol}: First slice {first_slice_pct:5.1f}% "
              f"(JPY pairs higher during Tokyo)")

    # Compare start times
    print("\n" + "=" * 70)
    print("VWAP Cost by Start Time (1M EURUSD, 1 hour)")
    print("-" * 70)

    start_times = [
        (3, "Tokyo"),
        (9, "London Open"),
        (14, "LN Overlap"),
        (20, "NY Close"),
        (23, "Off Hours")
    ]

    for hour, label in start_times:
        schedule = scheduler.create_schedule(
            order_id=f"VWAP_{hour}",
            symbol='EURUSD',
            direction=1,
            total_quantity=1_000_000,
            horizon_seconds=3600,
            start_time=datetime(2026, 1, 17, hour, 0, tzinfo=timezone.utc)
        )
        print(f"{hour:02d}:00 UTC ({label:12s}): Cost = {schedule.expected_cost_bps:.2f} bps")
