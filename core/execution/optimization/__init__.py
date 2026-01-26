"""
Execution Optimization Module
==============================
FX-specific execution optimization for minimizing market impact.

Components:
- ExecutionEngine: Strategy selection and scheduling
- ExecutionInterceptor: Drop-in replacement for order submission
- AlmgrenChrissOptimizer: Optimal execution trajectories
- SessionAwareTWAP: Time-weighted execution
- FXVWAPScheduler: Volume-weighted execution
- DDPGExecutor: RL-based adaptive execution
- FXMarketImpactModel: Market impact estimation

Quick Start:
    from core.execution.optimization import (
        get_execution_engine,
        get_execution_interceptor
    )

    # Create engine
    engine = get_execution_engine()

    # Get optimal execution decision
    decision = engine.optimize(
        symbol='EURUSD',
        direction=1,  # buy
        quantity=1_000_000,
        mid_price=1.0850,
        spread_bps=1.0,
        urgency=0.5,
        signal_confidence=0.7
    )

    # Or use interceptor with existing bot
    interceptor = get_execution_interceptor(ib_connector)
    exec_id = await interceptor.intercept(
        symbol='EURUSD',
        direction=1,
        quantity=1_000_000,
        signal_strength=0.7
    )
"""

# Configuration
from .config import (
    ExecutionConfig,
    ExecutionSlice,
    ExecutionSchedule,
    ExecutionStrategy,
    ExecutionDecision,
    FXSession,
    FXSessionConfig,
    SymbolExecutionConfig,
    get_current_session,
    get_session_config,
    get_symbol_config,
    DEFAULT_SESSIONS,
    DEFAULT_SYMBOL_CONFIGS,
)

# Market Impact Model
from .market_impact_fx import (
    FXMarketImpactModel,
    MarketImpactEstimate,
    TemporaryImpactModel,
    PermanentImpactModel,
    DealerSpreadModel,
    get_market_impact_model,
)

# Almgren-Chriss Optimal Execution
from .almgren_chriss import (
    AlmgrenChrissOptimizer,
    AdaptiveAlmgrenChriss,
    ACParameters,
    ACTrajectory,
    get_ac_optimizer,
)

# TWAP Scheduler
from .twap import (
    SessionAwareTWAP,
    SimpleTWAP,
    AdaptiveTWAP,
    TWAPSlice,
    TWAPSchedule,
    get_twap_scheduler,
)

# VWAP Scheduler
from .vwap import (
    FXVWAPScheduler,
    AdaptiveVWAP,
    VolumeProfile,
    create_volume_profile,
    get_vwap_scheduler,
)

# RL Executor
from .rl_executor import (
    DDPGExecutor,
    RLExecutionOptimizer,
    ExecutionState,
    ExecutionAction,
    Experience,
    ReplayBuffer,
    get_rl_executor,
)

# Main Engine
from .engine import (
    ExecutionEngine,
    ExecutionInterceptor,
    ExecutionStatus,
    ExecutionResult,
    get_execution_engine,
    get_execution_interceptor,
)

__all__ = [
    # Config
    'ExecutionConfig',
    'ExecutionSlice',
    'ExecutionSchedule',
    'ExecutionStrategy',
    'ExecutionDecision',
    'FXSession',
    'FXSessionConfig',
    'SymbolExecutionConfig',
    'get_current_session',
    'get_session_config',
    'get_symbol_config',
    'DEFAULT_SESSIONS',
    'DEFAULT_SYMBOL_CONFIGS',

    # Market Impact
    'FXMarketImpactModel',
    'MarketImpactEstimate',
    'TemporaryImpactModel',
    'PermanentImpactModel',
    'DealerSpreadModel',
    'get_market_impact_model',

    # Almgren-Chriss
    'AlmgrenChrissOptimizer',
    'AdaptiveAlmgrenChriss',
    'ACParameters',
    'ACTrajectory',
    'get_ac_optimizer',

    # TWAP
    'SessionAwareTWAP',
    'SimpleTWAP',
    'AdaptiveTWAP',
    'TWAPSlice',
    'TWAPSchedule',
    'get_twap_scheduler',

    # VWAP
    'FXVWAPScheduler',
    'AdaptiveVWAP',
    'VolumeProfile',
    'create_volume_profile',
    'get_vwap_scheduler',

    # RL
    'DDPGExecutor',
    'RLExecutionOptimizer',
    'ExecutionState',
    'ExecutionAction',
    'Experience',
    'ReplayBuffer',
    'get_rl_executor',

    # Engine
    'ExecutionEngine',
    'ExecutionInterceptor',
    'ExecutionStatus',
    'ExecutionResult',
    'get_execution_engine',
    'get_execution_interceptor',
]
