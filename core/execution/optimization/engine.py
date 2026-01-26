"""
Unified Execution Engine
=========================
Central execution optimization engine that:
1. Receives trading signals
2. Estimates market impact for each strategy
3. Selects optimal execution approach
4. Manages execution schedule
5. Provides interceptor for existing trading bots

Components:
- ExecutionEngine: Strategy selection and scheduling
- ExecutionInterceptor: Drop-in replacement for direct order submission
- ExecutionManager: Background execution of schedules
"""

import asyncio
import uuid
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np

from .config import (
    ExecutionConfig, ExecutionSchedule, ExecutionSlice, ExecutionStrategy,
    ExecutionDecision, FXSession, get_current_session, get_session_config,
    get_symbol_config
)
from .market_impact_fx import FXMarketImpactModel, get_market_impact_model
from .almgren_chriss import AlmgrenChrissOptimizer, get_ac_optimizer
from .twap import SessionAwareTWAP, get_twap_scheduler
from .vwap import FXVWAPScheduler, get_vwap_scheduler
from .rl_executor import RLExecutionOptimizer, get_rl_executor, ExecutionAction

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"
    ACTIVE = "active"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionResult:
    """Result of a completed execution."""
    order_id: str
    symbol: str
    direction: int
    requested_quantity: float
    executed_quantity: float
    vwap: float
    expected_cost_bps: float
    actual_cost_bps: float
    strategy_used: ExecutionStrategy
    num_slices: int
    duration_seconds: float
    status: ExecutionStatus


class ExecutionEngine:
    """
    Unified execution optimization engine.

    Selects optimal execution strategy based on:
    - Order size and urgency
    - Market conditions (spread, volatility)
    - Session liquidity
    - Signal confidence

    Strategies available:
    - MARKET: Immediate execution (high urgency)
    - LIMIT: Passive execution (tight spread, low urgency)
    - TWAP: Time-weighted (medium orders)
    - VWAP: Volume-weighted (larger orders)
    - ALMGREN_CHRISS: Optimal trajectory (institutional)
    - ADAPTIVE: RL-driven (learned optimization)
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

        # Initialize strategy components
        self.impact_model = get_market_impact_model(self.config)
        self.ac_optimizer = get_ac_optimizer(self.config)
        self.twap_scheduler = get_twap_scheduler(self.config)
        self.vwap_scheduler = get_vwap_scheduler(self.config)
        self.rl_optimizer = get_rl_executor() if self.config.use_rl_adaptation else None

        # Strategy thresholds
        self.small_order_threshold = 100_000    # Below this -> MARKET/LIMIT
        self.medium_order_threshold = 1_000_000  # Below this -> TWAP
        self.large_order_threshold = 10_000_000  # Above this -> AC

    def optimize(self,
                symbol: str,
                direction: int,
                quantity: float,
                mid_price: float,
                spread_bps: float,
                volatility: float = 0.0001,
                urgency: float = 0.5,
                signal_confidence: float = 0.5,
                session: Optional[FXSession] = None) -> ExecutionDecision:
        """
        Determine optimal execution strategy.

        Args:
            symbol: Currency pair
            direction: 1 = buy, -1 = sell
            quantity: Order size
            mid_price: Current mid price
            spread_bps: Current spread in basis points
            volatility: Current volatility
            urgency: Urgency level 0-1 (higher = faster)
            signal_confidence: ML signal confidence 0-1
            session: FX session (auto-detected if None)

        Returns:
            ExecutionDecision with strategy and schedule
        """
        if session is None:
            session = get_current_session()

        session_cfg = get_session_config(session)

        # Estimate costs for each strategy
        costs = self._estimate_strategy_costs(
            symbol=symbol,
            quantity=quantity,
            mid_price=mid_price,
            spread_bps=spread_bps,
            volatility=volatility,
            session=session
        )

        # Select strategy based on order characteristics
        strategy, reasoning = self._select_strategy(
            quantity=quantity,
            urgency=urgency,
            signal_confidence=signal_confidence,
            spread_bps=spread_bps,
            session=session,
            costs=costs
        )

        # Generate schedule for selected strategy
        schedule = self._create_schedule(
            strategy=strategy,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            mid_price=mid_price,
            volatility=volatility,
            urgency=urgency,
            session=session
        )

        # Apply RL adjustment if enabled
        if self.rl_optimizer and self.config.use_rl_adaptation:
            rl_action = self._get_rl_adjustment(
                symbol=symbol,
                quantity=quantity,
                spread_bps=spread_bps,
                volatility=volatility,
                session=session,
                urgency=urgency
            )
        else:
            rl_action = None

        # Build decision
        expected_cost = costs.get(strategy, spread_bps / 2)
        use_limit = strategy == ExecutionStrategy.LIMIT or (
            spread_bps < 1.0 and urgency < 0.5
        )
        limit_offset = min(spread_bps * 0.3, self.config.max_limit_offset_bps) if use_limit else 0

        aggressiveness = 0.3 + 0.4 * urgency + 0.3 * signal_confidence

        return ExecutionDecision(
            strategy=strategy,
            schedule=schedule,
            expected_cost_bps=expected_cost,
            expected_time_seconds=schedule.horizon_seconds if schedule else 0,
            use_limit=use_limit,
            limit_offset_bps=limit_offset,
            aggressiveness=aggressiveness,
            confidence=0.7 + 0.3 * signal_confidence,
            reasoning=reasoning,
            alternatives=costs
        )

    def _estimate_strategy_costs(self,
                                symbol: str,
                                quantity: float,
                                mid_price: float,
                                spread_bps: float,
                                volatility: float,
                                session: FXSession) -> Dict[ExecutionStrategy, float]:
        """Estimate execution cost for each strategy."""
        costs = {}

        # MARKET: immediate, pay full spread + impact
        impact = self.impact_model.estimate_impact(
            symbol=symbol,
            quantity=quantity,
            direction=1,
            mid_price=mid_price,
            spread_bps=spread_bps,
            session=session
        )
        costs[ExecutionStrategy.MARKET] = impact.total_impact_bps

        # LIMIT: pay less spread but risk non-fill
        # Cost = (1 - fill_prob) * market_cost + fill_prob * (spread/4)
        fill_prob = 0.7  # Estimated
        costs[ExecutionStrategy.LIMIT] = (
            (1 - fill_prob) * impact.total_impact_bps +
            fill_prob * spread_bps / 4
        )

        # TWAP: distributed execution
        num_slices = max(5, int(quantity / 200_000))
        twap_cost = self.impact_model.estimate_execution_cost(
            symbol=symbol,
            quantity=quantity,
            horizon_seconds=300,
            num_slices=num_slices,
            session=session
        )
        costs[ExecutionStrategy.TWAP] = twap_cost

        # VWAP: volume-weighted
        # Similar to TWAP but slightly better in liquid sessions
        session_cfg = get_session_config(session)
        vwap_cost = twap_cost * (0.9 + 0.1 / session_cfg.liquidity_multiplier)
        costs[ExecutionStrategy.VWAP] = vwap_cost

        # AC: optimal trajectory
        # Best for large orders, accounts for risk
        ac_cost = twap_cost * 0.85  # AC typically 15% better than naive
        costs[ExecutionStrategy.ALMGREN_CHRISS] = ac_cost

        return costs

    def _select_strategy(self,
                        quantity: float,
                        urgency: float,
                        signal_confidence: float,
                        spread_bps: float,
                        session: FXSession,
                        costs: Dict[ExecutionStrategy, float]) -> Tuple[ExecutionStrategy, str]:
        """Select optimal execution strategy."""

        # High urgency -> MARKET
        if urgency > self.config.urgency_high_threshold:
            return ExecutionStrategy.MARKET, "High urgency requires immediate execution"

        # Small orders -> MARKET or LIMIT
        if quantity < self.small_order_threshold:
            if spread_bps < 1.0 and urgency < 0.5:
                return ExecutionStrategy.LIMIT, "Small order with tight spread, use limit"
            return ExecutionStrategy.MARKET, "Small order, direct market execution"

        # Very low urgency + tight spread -> LIMIT
        if urgency < self.config.urgency_low_threshold and spread_bps < 1.2:
            return ExecutionStrategy.LIMIT, "Low urgency with tight spread, passive execution"

        # Large institutional orders -> Almgren-Chriss
        if quantity > self.large_order_threshold:
            return ExecutionStrategy.ALMGREN_CHRISS, "Large order, optimal trajectory minimizes impact"

        # Medium orders -> TWAP or VWAP based on session
        session_cfg = get_session_config(session)

        if session_cfg.liquidity_multiplier > 1.2:
            # High liquidity session -> VWAP (track volume)
            return ExecutionStrategy.VWAP, f"Medium order in liquid session ({session.value}), use VWAP"
        else:
            # Lower liquidity -> TWAP (spread evenly)
            return ExecutionStrategy.TWAP, f"Medium order in {session.value} session, use TWAP"

    def _create_schedule(self,
                        strategy: ExecutionStrategy,
                        symbol: str,
                        direction: int,
                        quantity: float,
                        mid_price: float,
                        volatility: float,
                        urgency: float,
                        session: FXSession) -> Optional[ExecutionSchedule]:
        """Create execution schedule for selected strategy."""
        order_id = str(uuid.uuid4())

        # Calculate horizon based on urgency
        base_horizon = self.config.default_horizon_seconds
        horizon = int(base_horizon * (1 - 0.7 * urgency))
        horizon = max(60, min(horizon, self.config.max_horizon_seconds))

        if strategy == ExecutionStrategy.MARKET:
            # Single immediate slice
            return ExecutionSchedule(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                total_quantity=quantity,
                slices=[ExecutionSlice(
                    slice_id=0,
                    target_time=datetime.now(timezone.utc),
                    target_quantity=quantity,
                    strategy=ExecutionStrategy.MARKET,
                    status="pending"
                )],
                strategy=strategy,
                horizon_seconds=0,
                expected_cost_bps=0
            )

        elif strategy == ExecutionStrategy.LIMIT:
            # Single limit slice
            return ExecutionSchedule(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                total_quantity=quantity,
                slices=[ExecutionSlice(
                    slice_id=0,
                    target_time=datetime.now(timezone.utc),
                    target_quantity=quantity,
                    strategy=ExecutionStrategy.LIMIT,
                    status="pending"
                )],
                strategy=strategy,
                horizon_seconds=self.config.limit_order_timeout_seconds,
                expected_cost_bps=0
            )

        elif strategy == ExecutionStrategy.TWAP:
            return self.twap_scheduler.create_schedule(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                total_quantity=quantity,
                horizon_seconds=horizon
            )

        elif strategy == ExecutionStrategy.VWAP:
            return self.vwap_scheduler.create_schedule(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                total_quantity=quantity,
                horizon_seconds=horizon
            )

        elif strategy == ExecutionStrategy.ALMGREN_CHRISS:
            return self.ac_optimizer.create_schedule(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                total_quantity=quantity,
                mid_price=mid_price,
                volatility=volatility,
                horizon_seconds=horizon
            )

        return None

    def _get_rl_adjustment(self,
                          symbol: str,
                          quantity: float,
                          spread_bps: float,
                          volatility: float,
                          session: FXSession,
                          urgency: float) -> Optional[ExecutionAction]:
        """Get RL-based adjustment to execution parameters."""
        if not self.rl_optimizer:
            return None

        from .rl_executor import ExecutionState

        session_cfg = get_session_config(session)

        state = ExecutionState(
            remaining_qty_pct=1.0,
            time_remaining_pct=1.0,
            spread_bps=spread_bps,
            volatility=volatility,
            session_liquidity=session_cfg.liquidity_multiplier,
            order_flow=0.0,
            fill_rate=0.8,
            slippage_so_far_bps=0.0
        )

        return self.rl_optimizer.agent.select_action(state, explore=False)


class ExecutionInterceptor:
    """
    Drop-in interceptor for trading bot order flow.

    Replaces direct order submission with optimized execution.

    Usage:
        # Instead of:
        await executor.submit_order(order)

        # Use:
        interceptor = ExecutionInterceptor(executor)
        exec_id = await interceptor.intercept(...)
    """

    def __init__(self,
                 executor: Any,
                 config: Optional[ExecutionConfig] = None,
                 enabled: bool = True):
        """
        Initialize interceptor.

        Args:
            executor: Original order executor (e.g., IBConnector)
            config: Execution configuration
            enabled: Whether optimization is enabled
        """
        self.executor = executor
        self.config = config or ExecutionConfig()
        self.engine = ExecutionEngine(self.config)
        self.enabled = enabled

        # Active executions
        self.active_executions: Dict[str, dict] = {}

        # Background execution thread
        self._executor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    async def intercept(self,
                       symbol: str,
                       direction: int,
                       quantity: float,
                       signal_strength: float = 0.5,
                       mid_price: Optional[float] = None,
                       spread_bps: Optional[float] = None,
                       volatility: Optional[float] = None,
                       urgency: Optional[float] = None) -> str:
        """
        Intercept order and execute optimally.

        Args:
            symbol: Currency pair
            direction: 1 = buy, -1 = sell
            quantity: Order quantity
            signal_strength: ML signal confidence (0-1)
            mid_price: Current mid price (required for optimization)
            spread_bps: Current spread
            volatility: Current volatility
            urgency: Override urgency (auto-calculated if None)

        Returns:
            Execution ID for tracking
        """
        if not self.enabled:
            # Pass through to original executor
            return await self._execute_immediately(symbol, direction, quantity)

        # Default values
        if mid_price is None:
            mid_price = 1.0  # Will be updated

        if spread_bps is None:
            symbol_cfg = get_symbol_config(symbol)
            spread_bps = symbol_cfg.avg_spread_bps

        if volatility is None:
            volatility = 0.0001

        if urgency is None:
            # Calculate urgency from signal strength
            urgency = 0.3 + 0.5 * signal_strength

        # Get optimization decision
        decision = self.engine.optimize(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            mid_price=mid_price,
            spread_bps=spread_bps,
            volatility=volatility,
            urgency=urgency,
            signal_confidence=signal_strength
        )

        logger.info(f"[EXEC] {symbol} {quantity:,.0f}: Strategy={decision.strategy.value}, "
                   f"Cost={decision.expected_cost_bps:.2f}bps, Reason: {decision.reasoning}")

        # Execute based on decision
        if decision.schedule is None or decision.strategy == ExecutionStrategy.MARKET:
            # Immediate execution
            return await self._execute_immediately(symbol, direction, quantity)

        # Start scheduled execution
        exec_id = decision.schedule.order_id
        self.active_executions[exec_id] = {
            'decision': decision,
            'schedule': decision.schedule,
            'symbol': symbol,
            'direction': direction,
            'executed_qty': 0.0,
            'executed_value': 0.0,
            'slices_completed': 0,
            'status': ExecutionStatus.ACTIVE,
            'start_time': datetime.now(timezone.utc)
        }

        # Execute in background
        asyncio.create_task(self._execute_schedule(exec_id))

        return exec_id

    async def _execute_immediately(self,
                                  symbol: str,
                                  direction: int,
                                  quantity: float) -> str:
        """Execute order immediately via original executor."""
        from scripts.hft_trading_bot import Order, Side

        exec_id = str(uuid.uuid4())

        order = Order(
            symbol=symbol,
            side=Side.BUY if direction > 0 else Side.SELL,
            quantity=quantity,
            order_type="MARKET"
        )

        try:
            await self.executor.submit_order(order)
        except Exception as e:
            logger.error(f"Immediate execution failed: {e}")

        return exec_id

    async def _execute_schedule(self, exec_id: str):
        """Execute scheduled slices in background."""
        if exec_id not in self.active_executions:
            return

        exec_state = self.active_executions[exec_id]
        schedule = exec_state['schedule']

        for slice_obj in schedule.slices:
            if self._stop_event.is_set():
                break

            # Wait until target time
            now = datetime.now(timezone.utc)
            wait_seconds = (slice_obj.target_time - now).total_seconds()

            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            # Execute slice
            try:
                await self._execute_slice(exec_id, slice_obj)
            except Exception as e:
                logger.error(f"Slice execution failed: {e}")
                if self.config.fallback_to_market:
                    # Execute remaining as market
                    remaining = schedule.total_quantity - exec_state['executed_qty']
                    if remaining > 0:
                        await self._execute_immediately(
                            schedule.symbol,
                            schedule.direction,
                            remaining
                        )
                break

        # Mark complete
        exec_state['status'] = ExecutionStatus.COMPLETE
        logger.info(f"[EXEC] {exec_id} completed: "
                   f"Executed {exec_state['executed_qty']:,.0f} / {schedule.total_quantity:,.0f}")

    async def _execute_slice(self, exec_id: str, slice_obj: ExecutionSlice):
        """Execute a single slice."""
        from scripts.hft_trading_bot import Order, Side

        exec_state = self.active_executions[exec_id]
        schedule = exec_state['schedule']

        order = Order(
            symbol=schedule.symbol,
            side=Side.BUY if schedule.direction > 0 else Side.SELL,
            quantity=slice_obj.target_quantity,
            order_type="MARKET" if slice_obj.strategy == ExecutionStrategy.MARKET else "LIMIT",
            limit_price=slice_obj.limit_price
        )

        trade = await self.executor.submit_order(order)

        if trade:
            exec_state['executed_qty'] += trade.quantity
            exec_state['executed_value'] += trade.quantity * trade.fill_price
            exec_state['slices_completed'] += 1
            slice_obj.executed_quantity = trade.quantity
            slice_obj.executed_price = trade.fill_price
            slice_obj.status = "filled"

    def get_execution_status(self, exec_id: str) -> Optional[dict]:
        """Get status of an execution."""
        return self.active_executions.get(exec_id)

    def cancel_execution(self, exec_id: str) -> bool:
        """Cancel an active execution."""
        if exec_id not in self.active_executions:
            return False

        exec_state = self.active_executions[exec_id]
        exec_state['status'] = ExecutionStatus.CANCELLED
        self._stop_event.set()

        return True

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return [
            eid for eid, state in self.active_executions.items()
            if state['status'] == ExecutionStatus.ACTIVE
        ]


def get_execution_engine(config: Optional[ExecutionConfig] = None) -> ExecutionEngine:
    """Factory function to get execution engine."""
    return ExecutionEngine(config)


def get_execution_interceptor(executor: Any,
                             config: Optional[ExecutionConfig] = None) -> ExecutionInterceptor:
    """Factory function to get execution interceptor."""
    return ExecutionInterceptor(executor, config)


if __name__ == '__main__':
    print("Execution Engine Test")
    print("=" * 70)

    engine = ExecutionEngine()

    # Test different scenarios
    scenarios = [
        # (symbol, quantity, urgency, confidence, spread, description)
        ('EURUSD', 50_000, 0.9, 0.8, 0.8, "Small urgent order"),
        ('EURUSD', 50_000, 0.2, 0.6, 0.6, "Small patient order"),
        ('EURUSD', 500_000, 0.5, 0.7, 1.0, "Medium order"),
        ('EURUSD', 5_000_000, 0.5, 0.7, 1.0, "Large order"),
        ('EURUSD', 50_000_000, 0.3, 0.8, 1.2, "Institutional order"),
        ('USDJPY', 1_000_000, 0.6, 0.7, 1.5, "JPY medium order"),
    ]

    print("\nStrategy Selection by Scenario")
    print("-" * 70)

    for symbol, qty, urgency, conf, spread, desc in scenarios:
        decision = engine.optimize(
            symbol=symbol,
            direction=1,
            quantity=qty,
            mid_price=1.0850 if 'JPY' not in symbol else 150.0,
            spread_bps=spread,
            volatility=0.0001,
            urgency=urgency,
            signal_confidence=conf
        )

        print(f"\n{desc}")
        print(f"  Size: {qty:>12,} | Urgency: {urgency:.1f} | Spread: {spread:.1f}bps")
        print(f"  Strategy: {decision.strategy.value:15s} | Cost: {decision.expected_cost_bps:.2f}bps")
        print(f"  Reason: {decision.reasoning}")

    # Test schedule generation
    print("\n" + "=" * 70)
    print("Schedule Generation: 1M EURUSD, Medium Urgency")
    print("-" * 70)

    decision = engine.optimize(
        symbol='EURUSD',
        direction=1,
        quantity=1_000_000,
        mid_price=1.0850,
        spread_bps=1.0,
        urgency=0.5,
        signal_confidence=0.7
    )

    if decision.schedule:
        sched = decision.schedule
        print(f"Order ID: {sched.order_id}")
        print(f"Strategy: {sched.strategy.value}")
        print(f"Horizon: {sched.horizon_seconds}s")
        print(f"Slices: {sched.num_slices}")
        print(f"\nFirst 5 slices:")
        for s in sched.slices[:5]:
            print(f"  {s.target_time.strftime('%H:%M:%S')}: "
                  f"{s.target_quantity:>10,.0f} ({s.strategy.value})")

    # Test cost comparison
    print("\n" + "=" * 70)
    print("Strategy Cost Comparison: 5M EURUSD")
    print("-" * 70)

    costs = engine._estimate_strategy_costs(
        symbol='EURUSD',
        quantity=5_000_000,
        mid_price=1.0850,
        spread_bps=1.0,
        volatility=0.0001,
        session=FXSession.LONDON
    )

    for strategy, cost in sorted(costs.items(), key=lambda x: x[1]):
        bar = "#" * int(cost * 10)
        print(f"{strategy.value:18s}: {cost:5.2f} bps  {bar}")
