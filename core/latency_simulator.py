"""
Latency Simulation for HFT Backtesting
=======================================
Models realistic latency in order execution.

Components:
- Feed latency (market data delay)
- Order latency (submission to exchange)
- Fill latency (order acceptance to fill)
- Network jitter

Source: NautilusTrader latency documentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LatencyType(Enum):
    """Types of latency."""
    FEED = "feed"  # Market data arrival
    ORDER = "order"  # Order submission
    FILL = "fill"  # Order execution
    CANCEL = "cancel"  # Cancellation


@dataclass
class LatencyConfig:
    """Latency configuration for a venue."""
    venue_name: str = "default"

    # Mean latencies (milliseconds)
    feed_latency_ms: float = 10.0
    order_latency_ms: float = 50.0
    fill_latency_ms: float = 5.0
    cancel_latency_ms: float = 30.0

    # Standard deviations (for jitter)
    feed_jitter_ms: float = 2.0
    order_jitter_ms: float = 10.0
    fill_jitter_ms: float = 1.0
    cancel_jitter_ms: float = 5.0

    # Tail behavior (for occasional spikes)
    spike_probability: float = 0.01
    spike_multiplier: float = 5.0


# Pre-configured venues
VENUE_CONFIGS = {
    'retail_forex': LatencyConfig(
        venue_name='retail_forex',
        feed_latency_ms=50.0,
        order_latency_ms=100.0,
        fill_latency_ms=20.0,
        cancel_latency_ms=80.0,
        order_jitter_ms=30.0
    ),
    'institutional_forex': LatencyConfig(
        venue_name='institutional_forex',
        feed_latency_ms=5.0,
        order_latency_ms=10.0,
        fill_latency_ms=2.0,
        cancel_latency_ms=8.0,
        order_jitter_ms=3.0
    ),
    'exchange_colocated': LatencyConfig(
        venue_name='exchange_colocated',
        feed_latency_ms=0.1,
        order_latency_ms=0.5,
        fill_latency_ms=0.1,
        cancel_latency_ms=0.3,
        order_jitter_ms=0.05
    ),
    'ib_gateway': LatencyConfig(
        venue_name='ib_gateway',
        feed_latency_ms=30.0,
        order_latency_ms=80.0,
        fill_latency_ms=15.0,
        cancel_latency_ms=60.0,
        order_jitter_ms=20.0
    )
}


class LatencySimulator:
    """
    Latency Simulation Engine.

    Simulates realistic network and execution latencies.

    Usage:
        sim = LatencySimulator(venue='retail_forex')
        delayed_time = sim.apply_feed_latency(tick_time)
        order_accepted_time = sim.apply_order_latency(submit_time)
    """

    def __init__(self, venue: str = 'retail_forex', config: LatencyConfig = None):
        """
        Initialize latency simulator.

        Args:
            venue: Venue name from VENUE_CONFIGS
            config: Custom LatencyConfig (overrides venue)
        """
        if config:
            self.config = config
        elif venue in VENUE_CONFIGS:
            self.config = VENUE_CONFIGS[venue]
        else:
            self.config = LatencyConfig()

        # For reproducibility
        self.rng = np.random.default_rng(42)

        # Track latency statistics
        self.latency_history: Dict[LatencyType, List[float]] = {
            lt: [] for lt in LatencyType
        }

    def _sample_latency(self, mean: float, std: float) -> float:
        """
        Sample latency from distribution.

        Uses log-normal for realistic latency distribution.
        """
        # Log-normal parameters from mean and std
        if mean <= 0:
            return 0.0

        # Check for spike
        if self.rng.random() < self.config.spike_probability:
            mean *= self.config.spike_multiplier
            std *= self.config.spike_multiplier

        # Log-normal sampling (always positive, right-skewed)
        if std <= 0:
            return mean

        mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
        sigma = np.sqrt(np.log(1 + (std ** 2) / (mean ** 2)))

        latency = self.rng.lognormal(mu, sigma)

        return max(0, latency)

    def apply_feed_latency(self, timestamp: datetime) -> datetime:
        """
        Apply feed latency to market data timestamp.

        Args:
            timestamp: Original tick timestamp

        Returns:
            Delayed timestamp (when we receive the tick)
        """
        latency_ms = self._sample_latency(
            self.config.feed_latency_ms,
            self.config.feed_jitter_ms
        )

        self.latency_history[LatencyType.FEED].append(latency_ms)

        return timestamp + timedelta(milliseconds=latency_ms)

    def apply_order_latency(self, submit_time: datetime) -> datetime:
        """
        Apply order submission latency.

        Args:
            submit_time: Time order was submitted

        Returns:
            Time order is received by exchange
        """
        latency_ms = self._sample_latency(
            self.config.order_latency_ms,
            self.config.order_jitter_ms
        )

        self.latency_history[LatencyType.ORDER].append(latency_ms)

        return submit_time + timedelta(milliseconds=latency_ms)

    def apply_fill_latency(self, match_time: datetime) -> datetime:
        """
        Apply fill confirmation latency.

        Args:
            match_time: Time order was matched

        Returns:
            Time we receive fill confirmation
        """
        latency_ms = self._sample_latency(
            self.config.fill_latency_ms,
            self.config.fill_jitter_ms
        )

        self.latency_history[LatencyType.FILL].append(latency_ms)

        return match_time + timedelta(milliseconds=latency_ms)

    def apply_cancel_latency(self, cancel_time: datetime) -> datetime:
        """
        Apply cancel request latency.

        Args:
            cancel_time: Time cancel was requested

        Returns:
            Time cancel is processed
        """
        latency_ms = self._sample_latency(
            self.config.cancel_latency_ms,
            self.config.cancel_jitter_ms
        )

        self.latency_history[LatencyType.CANCEL].append(latency_ms)

        return cancel_time + timedelta(milliseconds=latency_ms)

    def get_total_roundtrip(self) -> float:
        """
        Get total expected roundtrip latency (ms).

        Order submit -> fill confirmation received
        """
        return (
            self.config.order_latency_ms +
            self.config.fill_latency_ms
        )

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get latency statistics.

        Returns:
            Dict with statistics per latency type
        """
        stats = {}

        for lt, history in self.latency_history.items():
            if history:
                stats[lt.value] = {
                    'mean_ms': np.mean(history),
                    'std_ms': np.std(history),
                    'min_ms': np.min(history),
                    'max_ms': np.max(history),
                    'p50_ms': np.percentile(history, 50),
                    'p99_ms': np.percentile(history, 99),
                    'count': len(history)
                }
            else:
                stats[lt.value] = {
                    'mean_ms': 0, 'std_ms': 0, 'min_ms': 0,
                    'max_ms': 0, 'p50_ms': 0, 'p99_ms': 0, 'count': 0
                }

        return stats

    def reset_statistics(self) -> None:
        """Reset latency statistics."""
        for lt in LatencyType:
            self.latency_history[lt] = []


class LatencyAwareBacktest:
    """
    Latency-aware backtest wrapper.

    Applies latency to all market data and orders.
    """

    def __init__(self, simulator: LatencySimulator):
        """
        Initialize latency-aware backtest.

        Args:
            simulator: LatencySimulator instance
        """
        self.simulator = simulator

        # Pending events queue (sorted by actual arrival time)
        self.pending_events: List[Tuple[datetime, str, dict]] = []

    def delay_tick(self, tick_time: datetime, tick_data: dict) -> Tuple[datetime, dict]:
        """
        Delay tick data by feed latency.

        Args:
            tick_time: Original tick timestamp
            tick_data: Tick data dict

        Returns:
            (delayed_time, tick_data)
        """
        delayed = self.simulator.apply_feed_latency(tick_time)
        return delayed, tick_data

    def delay_order(self, submit_time: datetime, order: dict) -> Tuple[datetime, dict]:
        """
        Delay order submission.

        Args:
            submit_time: Order submission time
            order: Order data dict

        Returns:
            (exchange_received_time, order)
        """
        received = self.simulator.apply_order_latency(submit_time)
        return received, order

    def process_events(self, current_time: datetime) -> List[Tuple[str, dict]]:
        """
        Process events that have arrived by current time.

        Args:
            current_time: Current simulation time

        Returns:
            List of (event_type, event_data) that have arrived
        """
        arrived = []
        remaining = []

        for arrival_time, event_type, event_data in self.pending_events:
            if arrival_time <= current_time:
                arrived.append((event_type, event_data))
            else:
                remaining.append((arrival_time, event_type, event_data))

        self.pending_events = remaining
        return arrived

    def add_pending_event(self, arrival_time: datetime,
                          event_type: str, event_data: dict) -> None:
        """Add event to pending queue."""
        self.pending_events.append((arrival_time, event_type, event_data))
        self.pending_events.sort(key=lambda x: x[0])


class LatencyImpactAnalyzer:
    """
    Analyze impact of latency on trading performance.
    """

    def __init__(self):
        """Initialize analyzer."""
        self.trades_with_latency: List[Dict] = []
        self.trades_without_latency: List[Dict] = []

    def record_trade_with_latency(self,
                                  intended_price: float,
                                  actual_price: float,
                                  latency_ms: float,
                                  side: str) -> None:
        """Record trade with latency impact."""
        if side == 'buy':
            slippage = actual_price - intended_price
        else:
            slippage = intended_price - actual_price

        self.trades_with_latency.append({
            'intended_price': intended_price,
            'actual_price': actual_price,
            'latency_ms': latency_ms,
            'slippage': slippage,
            'slippage_bps': slippage / intended_price * 10000
        })

    def record_trade_without_latency(self,
                                     intended_price: float,
                                     actual_price: float,
                                     side: str) -> None:
        """Record trade without latency (ideal case)."""
        if side == 'buy':
            slippage = actual_price - intended_price
        else:
            slippage = intended_price - actual_price

        self.trades_without_latency.append({
            'intended_price': intended_price,
            'actual_price': actual_price,
            'slippage': slippage,
            'slippage_bps': slippage / intended_price * 10000
        })

    def get_latency_impact(self) -> Dict[str, float]:
        """
        Calculate latency impact on trading.

        Returns:
            Dict with impact metrics
        """
        if not self.trades_with_latency:
            return {}

        with_latency = pd.DataFrame(self.trades_with_latency)
        without_latency = pd.DataFrame(self.trades_without_latency) if self.trades_without_latency else None

        impact = {
            'avg_slippage_with_latency_bps': with_latency['slippage_bps'].mean(),
            'total_slippage_with_latency': with_latency['slippage'].sum(),
            'avg_latency_ms': with_latency['latency_ms'].mean(),
            'latency_correlation': with_latency['latency_ms'].corr(with_latency['slippage_bps'])
        }

        if without_latency is not None and len(without_latency) > 0:
            impact['avg_slippage_without_latency_bps'] = without_latency['slippage_bps'].mean()
            impact['latency_cost_bps'] = (
                impact['avg_slippage_with_latency_bps'] -
                impact['avg_slippage_without_latency_bps']
            )
        else:
            impact['latency_cost_bps'] = impact['avg_slippage_with_latency_bps']

        return impact


if __name__ == '__main__':
    print("Latency Simulator Test")
    print("=" * 50)

    # Test different venue configurations
    for venue in ['retail_forex', 'institutional_forex', 'exchange_colocated']:
        print(f"\n{venue.upper()}:")
        sim = LatencySimulator(venue=venue)

        # Simulate some latencies
        for _ in range(100):
            sim.apply_feed_latency(datetime.now())
            sim.apply_order_latency(datetime.now())
            sim.apply_fill_latency(datetime.now())

        stats = sim.get_statistics()
        for lt, s in stats.items():
            if s['count'] > 0:
                print(f"  {lt}: mean={s['mean_ms']:.2f}ms, p99={s['p99_ms']:.2f}ms")

    # Test latency impact
    print("\n" + "=" * 50)
    print("Latency Impact Analysis")

    analyzer = LatencyImpactAnalyzer()
    sim = LatencySimulator(venue='retail_forex')

    # Simulate trades
    np.random.seed(42)
    for i in range(100):
        intended_price = 1.1000 + np.random.randn() * 0.001
        latency = sim._sample_latency(sim.config.order_latency_ms, sim.config.order_jitter_ms)

        # Price moves during latency
        price_move = np.random.randn() * 0.0001 * (latency / 50)
        actual_price = intended_price + price_move

        analyzer.record_trade_with_latency(
            intended_price, actual_price, latency, 'buy'
        )
        analyzer.record_trade_without_latency(
            intended_price, intended_price, 'buy'
        )

    impact = analyzer.get_latency_impact()
    print(f"\nLatency Impact:")
    print(f"  Avg Slippage (with latency): {impact['avg_slippage_with_latency_bps']:.2f} bps")
    print(f"  Avg Slippage (without): {impact['avg_slippage_without_latency_bps']:.2f} bps")
    print(f"  Latency Cost: {impact['latency_cost_bps']:.2f} bps")
    print(f"  Correlation: {impact['latency_correlation']:.3f}")
