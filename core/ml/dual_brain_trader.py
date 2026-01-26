"""
Dual Brain Trader - Orchestrates Formula + Trading Brains
==========================================================
Chinese Quant Style architecture combining specialized models.

Architecture:
┌─────────────────┐     ┌─────────────────┐
│  Formula Brain  │ ──► │  Trading Brain  │ ──► Execute
│  (Local Ollama) │     │  (Kimi-K2 API)  │
│  ~500ms         │     │  ~1-2s          │
└─────────────────┘     └─────────────────┘

Only calls Trading Brain when signal_strength > threshold,
keeping API costs minimal (~$0.25/day).
"""

import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .formula_brain import FormulaBrain, FormulaAnalysis, create_formula_brain
from .trading_brain import TradingBrain, TradingDecision, create_trading_brain

logger = logging.getLogger(__name__)


@dataclass
class DualBrainResult:
    """Combined result from both brains."""
    symbol: str
    formula_analysis: FormulaAnalysis
    trading_decision: Optional[TradingDecision]
    should_trade: bool
    latency_ms: float
    timestamp: datetime


class DualBrainTrader:
    """
    Orchestrates Formula Brain + Trading Brain for optimal trading.

    Strategy:
    1. Always run Formula Brain (fast, free, local)
    2. Only call Trading Brain if signal is strong enough
    3. Cache decisions to avoid duplicate API calls
    4. Fallback to local-only if API unavailable
    """

    def __init__(
        self,
        formula_model: str = "forex-finetuned",
        trading_provider: str = "moonshot",
        trading_api_key: Optional[str] = None,
        signal_threshold: float = 1.5,  # Minimum sigma to call Trading Brain
        cache_ttl_seconds: int = 60,
        use_local_fallback: bool = True
    ):
        # Brains
        self.formula_brain = create_formula_brain(formula_model)
        self.trading_brain = create_trading_brain(
            provider=trading_provider,
            api_key=trading_api_key,
            use_local_fallback=use_local_fallback
        )

        # Config
        self.signal_threshold = signal_threshold
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.use_local_fallback = use_local_fallback

        # State
        self._decision_cache: Dict[str, Tuple[TradingDecision, datetime]] = {}
        self._stats = {
            "formula_calls": 0,
            "trading_calls": 0,
            "cache_hits": 0,
            "trades_executed": 0
        }
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize both brains."""
        if self._initialized:
            return True

        logger.info("Initializing Dual Brain Trader...")

        # Warm up Formula Brain
        formula_ok = await self.formula_brain.warmup()
        if not formula_ok:
            logger.error("Formula Brain failed to initialize")
            return False

        logger.info("Dual Brain Trader ready")
        self._initialized = True
        return True

    async def analyze_and_decide(
        self,
        symbol: str,
        ohlcv: Dict[str, float],
        account_state: Dict[str, Any],
        tick_features: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        force_trading_brain: bool = False
    ) -> DualBrainResult:
        """
        Full analysis and trading decision.

        Args:
            symbol: Currency pair
            ohlcv: Price data
            account_state: Account info
            tick_features: Pre-calculated features
            ml_prediction: ML ensemble prediction
            market_context: Additional context
            force_trading_brain: Always call Trading Brain

        Returns:
            DualBrainResult with analysis and decision
        """
        start_time = datetime.now()

        # Step 1: Formula Brain (always, local, fast)
        self._stats["formula_calls"] += 1
        formula_analysis = await self.formula_brain.analyze(
            symbol=symbol,
            ohlcv=ohlcv,
            tick_features=tick_features,
            ml_prediction=ml_prediction
        )

        trading_decision = None
        should_trade = False

        # Step 2: Decide if we should call Trading Brain
        signal_strong = abs(formula_analysis.signal_strength) >= self.signal_threshold
        high_confidence = formula_analysis.confidence >= 0.6

        if force_trading_brain or (signal_strong and high_confidence):
            # Check cache first
            cached = self._get_cached_decision(symbol, formula_analysis)
            if cached:
                self._stats["cache_hits"] += 1
                trading_decision = cached
            else:
                # Call Trading Brain
                self._stats["trading_calls"] += 1
                trading_decision = await self.trading_brain.decide(
                    symbol=symbol,
                    formula_analysis={
                        "alpha_signals": formula_analysis.alpha_signals,
                        "volatility": formula_analysis.volatility,
                        "microstructure": formula_analysis.microstructure,
                        "signal_strength": formula_analysis.signal_strength,
                        "direction": formula_analysis.direction,
                        "confidence": formula_analysis.confidence,
                        "reasoning": formula_analysis.reasoning
                    },
                    account_state=account_state,
                    market_context=market_context
                )

                # Cache the decision
                self._cache_decision(symbol, formula_analysis, trading_decision)

            should_trade = trading_decision.action != "HOLD"
            if should_trade:
                self._stats["trades_executed"] += 1

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return DualBrainResult(
            symbol=symbol,
            formula_analysis=formula_analysis,
            trading_decision=trading_decision,
            should_trade=should_trade,
            latency_ms=latency,
            timestamp=datetime.now()
        )

    def _get_cached_decision(
        self,
        symbol: str,
        analysis: FormulaAnalysis
    ) -> Optional[TradingDecision]:
        """Get cached decision if still valid."""
        cache_key = f"{symbol}_{analysis.direction}_{int(analysis.signal_strength)}"

        if cache_key in self._decision_cache:
            decision, cached_at = self._decision_cache[cache_key]
            if datetime.now() - cached_at < self.cache_ttl:
                return decision

        return None

    def _cache_decision(
        self,
        symbol: str,
        analysis: FormulaAnalysis,
        decision: TradingDecision
    ):
        """Cache a trading decision."""
        cache_key = f"{symbol}_{analysis.direction}_{int(analysis.signal_strength)}"
        self._decision_cache[cache_key] = (decision, datetime.now())

        # Clean old entries
        now = datetime.now()
        self._decision_cache = {
            k: v for k, v in self._decision_cache.items()
            if now - v[1] < self.cache_ttl
        }

    async def quick_analyze(
        self,
        symbol: str,
        ohlcv: Dict[str, float],
        tick_features: Optional[Dict[str, float]] = None
    ) -> FormulaAnalysis:
        """
        Quick analysis using only Formula Brain (no API call).
        Use for screening or when you don't need a trading decision.
        """
        self._stats["formula_calls"] += 1
        return await self.formula_brain.analyze(
            symbol=symbol,
            ohlcv=ohlcv,
            tick_features=tick_features
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_calls = self._stats["formula_calls"]
        api_calls = self._stats["trading_calls"]

        return {
            **self._stats,
            "api_call_rate": api_calls / max(1, total_calls),
            "cache_hit_rate": self._stats["cache_hits"] / max(1, api_calls + self._stats["cache_hits"]),
            "estimated_api_cost": api_calls * 0.0005  # ~$0.50 per 1000 calls
        }

    async def close(self):
        """Cleanup resources."""
        if hasattr(self.trading_brain, 'close'):
            await self.trading_brain.close()


# Factory function
def create_dual_brain_trader(
    formula_model: str = "forex-finetuned",
    trading_provider: str = "moonshot",
    signal_threshold: float = 1.5
) -> DualBrainTrader:
    """
    Create a Dual Brain Trader.

    Args:
        formula_model: Ollama model name for Formula Brain
        trading_provider: API provider for Trading Brain
        signal_threshold: Minimum signal strength to call Trading Brain

    Returns:
        Configured DualBrainTrader
    """
    return DualBrainTrader(
        formula_model=formula_model,
        trading_provider=trading_provider,
        signal_threshold=signal_threshold
    )


# Example usage
async def example():
    """Example usage of Dual Brain Trader."""
    trader = create_dual_brain_trader()
    await trader.initialize()

    # Analyze EURUSD
    result = await trader.analyze_and_decide(
        symbol="EURUSD",
        ohlcv={
            "open": 1.0850,
            "high": 1.0865,
            "low": 1.0845,
            "close": 1.0860,
            "volume": 1000000,
            "spread": 0.00008
        },
        account_state={
            "balance": 10000,
            "open_positions": 0,
            "daily_pnl": 0,
            "drawdown_pct": 0,
            "trades_today": 0
        }
    )

    print(f"Symbol: {result.symbol}")
    print(f"Signal strength: {result.formula_analysis.signal_strength:.2f}σ")
    print(f"Direction: {result.formula_analysis.direction}")

    if result.trading_decision:
        print(f"Action: {result.trading_decision.action}")
        print(f"Position: {result.trading_decision.position_pct}%")
        print(f"Reasoning: {result.trading_decision.reasoning}")

    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Stats: {trader.get_stats()}")

    await trader.close()


if __name__ == "__main__":
    asyncio.run(example())
