"""
Execution Optimization Configuration
=====================================
FX-specific configuration for session-aware execution optimization.

Sessions:
- Tokyo: 00:00-09:00 UTC (lower liquidity)
- London: 07:00-16:00 UTC (high liquidity)
- New York: 12:00-21:00 UTC (high liquidity)
- Overlap (LN): 12:00-16:00 UTC (highest liquidity)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
from datetime import datetime, time, timezone
import numpy as np


class FXSession(Enum):
    """FX trading sessions with liquidity characteristics."""
    TOKYO = "tokyo"           # 00:00-09:00 UTC
    LONDON = "london"         # 07:00-16:00 UTC
    NEW_YORK = "new_york"     # 12:00-21:00 UTC
    OVERLAP_LN = "overlap_ln" # 12:00-16:00 UTC (London-NY overlap)
    OVERLAP_TL = "overlap_tl" # 07:00-09:00 UTC (Tokyo-London overlap)
    OFF_HOURS = "off_hours"   # 21:00-00:00 UTC (weekend, low liquidity)


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ALMGREN_CHRISS = "almgren_chriss"
    ADAPTIVE = "adaptive"  # RL-driven


@dataclass
class FXSessionConfig:
    """Configuration for each FX trading session."""
    session: FXSession
    start_hour: int  # UTC hour
    end_hour: int    # UTC hour
    liquidity_multiplier: float  # 1.0 = baseline
    spread_multiplier: float     # 1.0 = baseline
    volatility_multiplier: float # 1.0 = baseline
    volume_weight: float         # Relative volume in this session

    def is_active(self, dt: Optional[datetime] = None) -> bool:
        """Check if session is active at given time."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour

        # Handle sessions crossing midnight
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:
            return hour >= self.start_hour or hour < self.end_hour


# Default session configurations based on BIS FX survey data
DEFAULT_SESSIONS: Dict[FXSession, FXSessionConfig] = {
    FXSession.TOKYO: FXSessionConfig(
        session=FXSession.TOKYO,
        start_hour=0,
        end_hour=9,
        liquidity_multiplier=0.4,  # Low liquidity
        spread_multiplier=1.3,      # Higher spreads
        volatility_multiplier=0.8,
        volume_weight=0.15
    ),
    FXSession.LONDON: FXSessionConfig(
        session=FXSession.LONDON,
        start_hour=7,
        end_hour=16,
        liquidity_multiplier=1.4,  # High liquidity
        spread_multiplier=0.8,      # Tighter spreads
        volatility_multiplier=1.2,
        volume_weight=0.37
    ),
    FXSession.NEW_YORK: FXSessionConfig(
        session=FXSession.NEW_YORK,
        start_hour=12,
        end_hour=21,
        liquidity_multiplier=1.2,  # Good liquidity
        spread_multiplier=0.9,
        volatility_multiplier=1.1,
        volume_weight=0.35
    ),
    FXSession.OVERLAP_LN: FXSessionConfig(
        session=FXSession.OVERLAP_LN,
        start_hour=12,
        end_hour=16,
        liquidity_multiplier=1.8,  # Highest liquidity
        spread_multiplier=0.7,      # Tightest spreads
        volatility_multiplier=1.0,
        volume_weight=0.20
    ),
    FXSession.OVERLAP_TL: FXSessionConfig(
        session=FXSession.OVERLAP_TL,
        start_hour=7,
        end_hour=9,
        liquidity_multiplier=0.9,
        spread_multiplier=1.0,
        volatility_multiplier=1.0,
        volume_weight=0.08
    ),
    FXSession.OFF_HOURS: FXSessionConfig(
        session=FXSession.OFF_HOURS,
        start_hour=21,
        end_hour=0,
        liquidity_multiplier=0.2,  # Very low liquidity
        spread_multiplier=1.8,      # Wide spreads
        volatility_multiplier=0.5,
        volume_weight=0.05
    )
}


@dataclass
class ExecutionConfig:
    """
    Configuration for execution optimization.

    Based on Almgren-Chriss optimal execution framework adapted for FX.
    """

    # Almgren-Chriss parameters
    risk_aversion: float = 1e-6           # Lambda: risk aversion coefficient
    temp_impact_coef: float = 1e-4        # Eta: temporary impact coefficient
    perm_impact_coef: float = 1e-5        # Gamma: permanent impact coefficient

    # Execution timing
    default_horizon_seconds: int = 300    # 5 minutes default execution window
    slice_interval_seconds: int = 30      # Slice every 30 seconds
    min_slice_interval_seconds: int = 5   # Minimum 5 seconds between slices
    max_horizon_seconds: int = 3600       # Maximum 1 hour execution window

    # Urgency thresholds
    urgency_high_threshold: float = 0.8   # High urgency -> more market orders
    urgency_low_threshold: float = 0.3    # Low urgency -> more passive

    # RL adaptation
    use_rl_adaptation: bool = True        # Use DDPG for adaptive execution
    rl_exploration_rate: float = 0.1      # Epsilon for exploration
    rl_update_frequency: int = 100        # Update RL model every N executions

    # Session awareness
    session_aware: bool = True            # Adjust execution based on FX session

    # Limit order parameters
    limit_order_timeout_seconds: int = 60 # Cancel limit orders after 60s
    limit_offset_bps_default: float = 0.5 # Default limit price offset in bps
    max_limit_offset_bps: float = 3.0     # Maximum limit offset

    # Risk limits
    max_participation_rate: float = 0.10  # Max 10% of estimated volume
    max_market_impact_bps: float = 5.0    # Max acceptable market impact

    # Fill rate thresholds
    min_fill_probability: float = 0.50    # Minimum fill prob for limit orders

    # Fallback behavior
    fallback_to_market: bool = True       # If strategy fails, use market order
    execution_timeout_seconds: int = 60   # Max time for entire execution


@dataclass
class SymbolExecutionConfig:
    """Per-symbol execution configuration."""
    symbol: str
    pip_value: float = 0.0001           # Value of 1 pip (0.01 for JPY pairs)
    min_lot_size: float = 1000.0        # Minimum position size
    lot_step: float = 1000.0            # Position size increment
    daily_volume_estimate: float = 1e9  # Estimated daily volume in base currency
    avg_spread_bps: float = 1.0         # Average spread in basis points
    tick_size: float = 0.00001          # Minimum price increment


# Default symbol configs for major pairs
DEFAULT_SYMBOL_CONFIGS: Dict[str, SymbolExecutionConfig] = {
    'EURUSD': SymbolExecutionConfig(
        symbol='EURUSD',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=1.2e9,
        avg_spread_bps=0.8,
        tick_size=0.00001
    ),
    'GBPUSD': SymbolExecutionConfig(
        symbol='GBPUSD',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.6e9,
        avg_spread_bps=1.2,
        tick_size=0.00001
    ),
    'USDJPY': SymbolExecutionConfig(
        symbol='USDJPY',
        pip_value=0.01,
        min_lot_size=1000,
        daily_volume_estimate=0.8e9,
        avg_spread_bps=1.0,
        tick_size=0.001
    ),
    'USDCHF': SymbolExecutionConfig(
        symbol='USDCHF',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.2e9,
        avg_spread_bps=1.5,
        tick_size=0.00001
    ),
    'AUDUSD': SymbolExecutionConfig(
        symbol='AUDUSD',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.3e9,
        avg_spread_bps=1.3,
        tick_size=0.00001
    ),
    'USDCAD': SymbolExecutionConfig(
        symbol='USDCAD',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.3e9,
        avg_spread_bps=1.4,
        tick_size=0.00001
    ),
    'NZDUSD': SymbolExecutionConfig(
        symbol='NZDUSD',
        pip_value=0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.15e9,
        avg_spread_bps=1.8,
        tick_size=0.00001
    ),
    'EURJPY': SymbolExecutionConfig(
        symbol='EURJPY',
        pip_value=0.01,
        min_lot_size=1000,
        daily_volume_estimate=0.4e9,
        avg_spread_bps=1.5,
        tick_size=0.001
    ),
    'GBPJPY': SymbolExecutionConfig(
        symbol='GBPJPY',
        pip_value=0.01,
        min_lot_size=1000,
        daily_volume_estimate=0.25e9,
        avg_spread_bps=2.0,
        tick_size=0.001
    ),
}


def get_current_session(dt: Optional[datetime] = None) -> FXSession:
    """
    Get the current FX trading session.

    Priority: OVERLAP_LN > OVERLAP_TL > LONDON > NEW_YORK > TOKYO > OFF_HOURS
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    hour = dt.hour

    # Check overlaps first (highest priority)
    if 12 <= hour < 16:
        return FXSession.OVERLAP_LN
    if 7 <= hour < 9:
        return FXSession.OVERLAP_TL

    # Check major sessions
    if 7 <= hour < 16:
        return FXSession.LONDON
    if 12 <= hour < 21:
        return FXSession.NEW_YORK
    if 0 <= hour < 9:
        return FXSession.TOKYO

    # Off hours
    return FXSession.OFF_HOURS


def get_session_config(session: Optional[FXSession] = None,
                       dt: Optional[datetime] = None) -> FXSessionConfig:
    """Get configuration for a session."""
    if session is None:
        session = get_current_session(dt)
    return DEFAULT_SESSIONS.get(session, DEFAULT_SESSIONS[FXSession.OFF_HOURS])


def get_symbol_config(symbol: str) -> SymbolExecutionConfig:
    """Get execution configuration for a symbol."""
    if symbol in DEFAULT_SYMBOL_CONFIGS:
        return DEFAULT_SYMBOL_CONFIGS[symbol]

    # Default config for unknown symbols
    is_jpy = 'JPY' in symbol
    return SymbolExecutionConfig(
        symbol=symbol,
        pip_value=0.01 if is_jpy else 0.0001,
        min_lot_size=1000,
        daily_volume_estimate=0.1e9,
        avg_spread_bps=2.0,
        tick_size=0.001 if is_jpy else 0.00001
    )


@dataclass
class ExecutionSlice:
    """A single execution slice within a larger order."""
    slice_id: int
    target_time: datetime
    target_quantity: float
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    strategy: ExecutionStrategy = ExecutionStrategy.MARKET
    limit_price: Optional[float] = None
    status: str = "pending"  # pending, active, filled, cancelled, expired


@dataclass
class ExecutionSchedule:
    """Complete execution schedule for an order."""
    order_id: str
    symbol: str
    direction: int  # 1 = buy, -1 = sell
    total_quantity: float
    slices: list = field(default_factory=list)
    strategy: ExecutionStrategy = ExecutionStrategy.TWAP
    horizon_seconds: int = 300
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expected_cost_bps: float = 0.0
    actual_cost_bps: float = 0.0

    @property
    def num_slices(self) -> int:
        return len(self.slices)

    @property
    def executed_quantity(self) -> float:
        return sum(s.executed_quantity for s in self.slices)

    @property
    def remaining_quantity(self) -> float:
        return self.total_quantity - self.executed_quantity

    @property
    def is_complete(self) -> bool:
        return all(s.status in ('filled', 'cancelled') for s in self.slices)

    @property
    def vwap(self) -> float:
        """Calculate volume-weighted average price of executed slices."""
        total_value = sum(s.executed_quantity * s.executed_price
                        for s in self.slices if s.executed_price > 0)
        total_qty = sum(s.executed_quantity for s in self.slices if s.executed_price > 0)
        return total_value / total_qty if total_qty > 0 else 0.0


@dataclass
class ExecutionDecision:
    """Decision output from the execution engine."""
    strategy: ExecutionStrategy
    schedule: Optional[ExecutionSchedule]
    expected_cost_bps: float
    expected_time_seconds: float
    use_limit: bool
    limit_offset_bps: float
    aggressiveness: float  # 0 = passive, 1 = aggressive
    confidence: float
    reasoning: str

    # Alternative strategies considered
    alternatives: Dict[str, float] = field(default_factory=dict)


if __name__ == '__main__':
    # Test session detection
    from datetime import datetime, timezone

    print("FX Session Detection Test")
    print("=" * 50)

    test_hours = [0, 3, 7, 8, 12, 14, 17, 20, 22, 23]
    for hour in test_hours:
        dt = datetime(2026, 1, 17, hour, 30, tzinfo=timezone.utc)
        session = get_current_session(dt)
        config = get_session_config(session)
        print(f"{hour:02d}:30 UTC -> {session.value:12s} "
              f"(liq: {config.liquidity_multiplier:.1f}x, "
              f"spread: {config.spread_multiplier:.1f}x)")

    print("\n" + "=" * 50)
    print("Symbol Configs")
    for symbol in ['EURUSD', 'USDJPY', 'GBPUSD', 'UNKNOWN']:
        cfg = get_symbol_config(symbol)
        print(f"{symbol}: pip={cfg.pip_value}, spread={cfg.avg_spread_bps}bps")
