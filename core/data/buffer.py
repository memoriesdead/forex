"""
Live Tick Buffer for Continuous Retraining
==========================================
Thread-safe ring buffer that stores tick data + features + outcomes
for live model retraining on RTX 5080.
"""

import numpy as np
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class TickRecord:
    """Single tick with features and outcome."""
    timestamp: float
    symbol: str
    bid: float
    ask: float
    mid: float
    features: Dict[str, float]
    # Outcomes filled in after N ticks
    outcome_direction_10: Optional[int] = None
    outcome_direction_50: Optional[int] = None
    outcome_return_10: Optional[float] = None
    outcome_return_50: Optional[float] = None


class LiveTickBuffer:
    """
    Thread-safe ring buffer for live tick data.

    Stores ticks with features, then fills in outcomes after N ticks
    have passed (so we know the actual direction).
    """

    def __init__(self, max_size: int = 50000, horizons: List[int] = None):
        """
        Args:
            max_size: Max ticks to store (default 50k = ~8 hours)
            horizons: Outcome horizons in ticks (default [10, 50])
        """
        self.max_size = max_size
        self.horizons = horizons or [10, 50]
        self.max_horizon = max(self.horizons)

        # Main buffer - stores TickRecords
        self._buffer: deque = deque(maxlen=max_size)

        # Pending outcomes - ticks waiting for outcome calculation
        self._pending: deque = deque(maxlen=max_size)

        # Lock for thread safety
        self._lock = threading.RLock()

        # Stats
        self.total_ticks = 0
        self.labeled_ticks = 0

    def add_tick(self, symbol: str, bid: float, ask: float,
                 features: Dict[str, float], timestamp: float = None):
        """
        Add a new tick to the buffer.

        Args:
            symbol: Currency pair
            bid: Bid price
            ask: Ask price
            features: Feature dict from HFTFeatureEngine
            timestamp: Unix timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()

        mid = (bid + ask) / 2

        record = TickRecord(
            timestamp=timestamp,
            symbol=symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            features=features.copy()
        )

        with self._lock:
            self._buffer.append(record)
            self._pending.append(record)
            self.total_ticks += 1

            # Calculate outcomes for old ticks
            self._calculate_outcomes(mid)

    def _calculate_outcomes(self, current_mid: float):
        """Calculate outcomes for ticks that are old enough."""
        with self._lock:
            while len(self._pending) > self.max_horizon:
                old_tick = self._pending.popleft()

                # Calculate direction outcomes
                if 10 in self.horizons:
                    future_mid = self._get_mid_at_offset(10)
                    if future_mid is not None:
                        old_tick.outcome_direction_10 = 1 if future_mid > old_tick.mid else 0
                        old_tick.outcome_return_10 = (future_mid - old_tick.mid) / old_tick.mid * 10000  # bps

                if 50 in self.horizons:
                    future_mid = self._get_mid_at_offset(50)
                    if future_mid is not None:
                        old_tick.outcome_direction_50 = 1 if future_mid > old_tick.mid else 0
                        old_tick.outcome_return_50 = (future_mid - old_tick.mid) / old_tick.mid * 10000  # bps

                self.labeled_ticks += 1

    def _get_mid_at_offset(self, offset: int) -> Optional[float]:
        """Get mid price at offset from pending head."""
        with self._lock:
            if len(self._buffer) > offset:
                # Get the tick at offset from the oldest pending tick
                idx = len(self._buffer) - len(self._pending) + offset
                if 0 <= idx < len(self._buffer):
                    return self._buffer[idx].mid
        return None

    def get_training_data(self, n_samples: int = 5000,
                          symbol: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get labeled training data for retraining.

        Args:
            n_samples: Number of samples to return
            symbol: Filter by symbol (None = all)

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
        """
        with self._lock:
            # Get labeled ticks
            labeled = [t for t in self._buffer
                      if t.outcome_direction_10 is not None
                      and (symbol is None or t.symbol == symbol)]

            if len(labeled) < 100:
                return None, None, None

            # Take most recent n_samples
            samples = labeled[-n_samples:] if len(labeled) > n_samples else labeled

            # Extract features and labels
            feature_names = list(samples[0].features.keys())
            X = np.array([[t.features[f] for f in feature_names] for t in samples])
            y = np.array([t.outcome_direction_10 for t in samples])

            return X, y, feature_names

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                'total_ticks': self.total_ticks,
                'buffer_size': len(self._buffer),
                'pending_labels': len(self._pending),
                'labeled_ticks': self.labeled_ticks,
                'buffer_capacity': self.max_size,
                'fill_pct': len(self._buffer) / self.max_size * 100
            }

    def clear(self):
        """Clear all data."""
        with self._lock:
            self._buffer.clear()
            self._pending.clear()
            self.total_ticks = 0
            self.labeled_ticks = 0


# Singleton instance for global access
_global_buffer: Optional[LiveTickBuffer] = None
_buffer_lock = threading.Lock()


def get_tick_buffer(max_size: int = 50000) -> LiveTickBuffer:
    """Get or create the global tick buffer."""
    global _global_buffer
    with _buffer_lock:
        if _global_buffer is None:
            _global_buffer = LiveTickBuffer(max_size=max_size)
        return _global_buffer
