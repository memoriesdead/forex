# Execution Module
# Order execution and backtesting for forex trading

from .order_book import OrderBookL3, PriceLevel, Order, OrderBookSignals
from .queue_position import (
    RiskAverseQueueModel,
    ProbabilisticQueueModel,
    L3FIFOQueueModel,
    QueuePositionTracker,
)
from .fill_probability import PoissonFillModel, MarketImpactModel, FillProbabilityEngine
from .latency import LatencySimulator, LatencyConfig
from .backtest import TickBacktestEngine, BacktestConfig

# Execution Optimization (2026-01-17)
from .optimization import (
    # Core engine
    ExecutionEngine,
    ExecutionInterceptor,
    get_execution_engine,
    get_execution_interceptor,
    # Config
    ExecutionConfig,
    ExecutionStrategy,
    ExecutionDecision,
    ExecutionSchedule,
    ExecutionSlice,
    FXSession,
    get_current_session,
    get_session_config,
    # Strategies
    AlmgrenChrissOptimizer,
    SessionAwareTWAP,
    FXVWAPScheduler,
    FXMarketImpactModel,
    # RL
    DDPGExecutor,
    RLExecutionOptimizer,
)

__all__ = [
    # Order Book
    'OrderBookL3',
    'PriceLevel',
    'Order',
    'OrderBookSignals',
    # Queue Models
    'RiskAverseQueueModel',
    'ProbabilisticQueueModel',
    'L3FIFOQueueModel',
    'QueuePositionTracker',
    # Fill Probability
    'PoissonFillModel',
    'MarketImpactModel',
    'FillProbabilityEngine',
    # Latency
    'LatencySimulator',
    'LatencyConfig',
    # Backtest
    'TickBacktestEngine',
    'BacktestConfig',
    # Execution Optimization
    'ExecutionEngine',
    'ExecutionInterceptor',
    'get_execution_engine',
    'get_execution_interceptor',
    'ExecutionConfig',
    'ExecutionStrategy',
    'ExecutionDecision',
    'ExecutionSchedule',
    'ExecutionSlice',
    'FXSession',
    'get_current_session',
    'get_session_config',
    'AlmgrenChrissOptimizer',
    'SessionAwareTWAP',
    'FXVWAPScheduler',
    'FXMarketImpactModel',
    'DDPGExecutor',
    'RLExecutionOptimizer',
]
