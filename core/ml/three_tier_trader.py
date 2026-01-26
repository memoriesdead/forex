"""
Three-Tier Forex Trading Architecture
=====================================
Opus 4.5 recommended architecture adapted for forex.

Tier 1: ML Ensemble (XGB/LGB/CB) - <50ms, screens ALL ticks
Tier 2: Forex-R1 (7B local) - ~2s, formula reasoning for strong signals
Tier 3: Kimi-K2 (1T API) - ~2s, complex decisions only

Cost: ~$0.05/day (99% local processing)
"""

import asyncio
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TierUsed(Enum):
    SKIP = "skip"
    TIER1_DIRECT = "tier1"
    TIER2_FORMULA = "tier2"
    TIER3_KIMI = "tier3"


@dataclass
class ThreeTierResult:
    """Result from three-tier analysis."""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD", "SKIP"]
    tier_used: TierUsed
    position_pct: float
    stop_loss_pips: float
    take_profit_pips: float
    confidence: float
    reasoning: str
    latency_ms: float
    timestamp: datetime

    # Tier-specific data
    ml_prediction: Optional[Dict] = None
    formula_analysis: Optional[Dict] = None
    kimi_decision: Optional[Dict] = None


class ThreeTierTrader:
    """
    Three-tier forex trading system.

    Flow:
    Tick → Tier1 (ML, <50ms) → Tier2 (Formula, ~2s) → Tier3 (Kimi, ~2s)

    Most ticks stop at Tier 1 (no trade).
    Strong signals go to Tier 2 (formula reasoning).
    Complex situations go to Tier 3 (Kimi-K2 API).
    """

    # Tier 1 thresholds
    TIER1_SKIP_THRESHOLD = 0.55      # Below this = no trade
    TIER1_DIRECT_THRESHOLD = 0.80    # Above this = direct execute
    TIER1_TIER2_RANGE = (0.55, 0.80) # Between = go to Tier 2

    # Tier 2 thresholds
    TIER2_SKIP_SIGMA = 1.0           # Below 1σ = no trade
    TIER2_EXECUTE_SIGMA = 1.5        # Above 1.5σ = execute
    TIER2_TIER3_SIGMA = 2.5          # Above 2.5σ = consider Tier 3

    # Tier 3 triggers
    LARGE_POSITION_PCT = 20          # >20% capital = Tier 3
    CORRELATION_THRESHOLD = 0.7      # High correlation = Tier 3

    def __init__(
        self,
        formula_model: str = "forex-finetuned",
        trading_provider: str = "moonshot",
        enable_tier3: bool = True
    ):
        # Lazy imports to avoid circular dependencies
        self._ml_ensemble = None
        self._formula_brain = None
        self._trading_brain = None

        self.formula_model = formula_model
        self.trading_provider = trading_provider
        self.enable_tier3 = enable_tier3

        # Stats
        self._stats = {
            "total_ticks": 0,
            "tier1_skips": 0,
            "tier1_directs": 0,
            "tier2_calls": 0,
            "tier2_executes": 0,
            "tier3_calls": 0,
            "trades_executed": 0
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all tiers."""
        if self._initialized:
            return True

        logger.info("Initializing Three-Tier Trader...")

        try:
            # Tier 1: ML Ensemble (already exists)
            from .adaptive_ensemble import AdaptiveMLEnsemble
            self._ml_ensemble = AdaptiveMLEnsemble()
            logger.info("Tier 1 (ML Ensemble) ready")

            # Tier 2: Formula Brain
            from .formula_brain import FormulaBrain
            self._formula_brain = FormulaBrain(model_name=self.formula_model)
            await self._formula_brain.warmup()
            logger.info("Tier 2 (Formula Brain) ready")

            # Tier 3: Trading Brain (optional)
            if self.enable_tier3:
                from .trading_brain import TradingBrain
                self._trading_brain = TradingBrain(provider=self.trading_provider)
                logger.info("Tier 3 (Kimi-K2) ready")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def process_tick(
        self,
        symbol: str,
        features: Dict[str, float],
        ohlcv: Dict[str, float],
        account_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> ThreeTierResult:
        """
        Process a tick through the three-tier system.

        Args:
            symbol: Currency pair (e.g., "EURUSD")
            features: Pre-calculated 575 features
            ohlcv: OHLCV data
            account_state: Account balance, positions, etc.
            market_context: Session, volatility regime, etc.

        Returns:
            ThreeTierResult with action and reasoning
        """
        start_time = datetime.now()
        self._stats["total_ticks"] += 1

        # =====================================================================
        # TIER 1: ML Ensemble (<50ms)
        # =====================================================================
        ml_result = self._tier1_ml(symbol, features)

        if ml_result["confidence"] < self.TIER1_SKIP_THRESHOLD:
            # No edge - skip
            self._stats["tier1_skips"] += 1
            return ThreeTierResult(
                symbol=symbol,
                action="SKIP",
                tier_used=TierUsed.SKIP,
                position_pct=0,
                stop_loss_pips=0,
                take_profit_pips=0,
                confidence=ml_result["confidence"],
                reasoning=f"Tier 1: Low confidence ({ml_result['confidence']:.2%})",
                latency_ms=self._latency_ms(start_time),
                timestamp=datetime.now(),
                ml_prediction=ml_result
            )

        if ml_result["confidence"] > self.TIER1_DIRECT_THRESHOLD:
            # Very high confidence - direct execute
            self._stats["tier1_directs"] += 1
            self._stats["trades_executed"] += 1

            action = "BUY" if ml_result["direction"] > 0 else "SELL"
            position = self._calculate_position(ml_result["confidence"], account_state)

            return ThreeTierResult(
                symbol=symbol,
                action=action,
                tier_used=TierUsed.TIER1_DIRECT,
                position_pct=position,
                stop_loss_pips=25,
                take_profit_pips=50,
                confidence=ml_result["confidence"],
                reasoning=f"Tier 1 Direct: {ml_result['confidence']:.2%} confidence",
                latency_ms=self._latency_ms(start_time),
                timestamp=datetime.now(),
                ml_prediction=ml_result
            )

        # =====================================================================
        # TIER 2: Formula Brain (~2s)
        # =====================================================================
        self._stats["tier2_calls"] += 1

        formula_result = await self._tier2_formula(symbol, ohlcv, features, ml_result)

        if formula_result.signal_strength < self.TIER2_SKIP_SIGMA:
            # Weak signal after formula analysis
            return ThreeTierResult(
                symbol=symbol,
                action="SKIP",
                tier_used=TierUsed.TIER2_FORMULA,
                position_pct=0,
                stop_loss_pips=0,
                take_profit_pips=0,
                confidence=formula_result.confidence,
                reasoning=f"Tier 2: Signal only {formula_result.signal_strength:.1f}σ",
                latency_ms=self._latency_ms(start_time),
                timestamp=datetime.now(),
                ml_prediction=ml_result,
                formula_analysis=self._formula_to_dict(formula_result)
            )

        # Check if we need Tier 3
        needs_tier3 = self._needs_tier3(
            formula_result.signal_strength,
            account_state,
            market_context
        )

        if not needs_tier3 or not self.enable_tier3:
            # Execute with formula sizing
            self._stats["tier2_executes"] += 1
            self._stats["trades_executed"] += 1

            action = "BUY" if formula_result.direction > 0 else "SELL"
            position = self._kelly_position(
                formula_result.confidence,
                formula_result.signal_strength,
                account_state
            )

            return ThreeTierResult(
                symbol=symbol,
                action=action,
                tier_used=TierUsed.TIER2_FORMULA,
                position_pct=position,
                stop_loss_pips=self._dynamic_stop(formula_result),
                take_profit_pips=self._dynamic_tp(formula_result),
                confidence=formula_result.confidence,
                reasoning=f"Tier 2: {formula_result.signal_strength:.1f}σ - {formula_result.reasoning}",
                latency_ms=self._latency_ms(start_time),
                timestamp=datetime.now(),
                ml_prediction=ml_result,
                formula_analysis=self._formula_to_dict(formula_result)
            )

        # =====================================================================
        # TIER 3: Kimi-K2 API (~2s)
        # =====================================================================
        self._stats["tier3_calls"] += 1

        kimi_result = await self._tier3_kimi(
            symbol, formula_result, account_state, market_context
        )

        if kimi_result.action != "HOLD":
            self._stats["trades_executed"] += 1

        return ThreeTierResult(
            symbol=symbol,
            action=kimi_result.action,
            tier_used=TierUsed.TIER3_KIMI,
            position_pct=kimi_result.position_pct,
            stop_loss_pips=kimi_result.stop_loss_pips,
            take_profit_pips=kimi_result.take_profit_pips,
            confidence=kimi_result.confidence,
            reasoning=f"Tier 3: {kimi_result.reasoning}",
            latency_ms=self._latency_ms(start_time),
            timestamp=datetime.now(),
            ml_prediction=ml_result,
            formula_analysis=self._formula_to_dict(formula_result),
            kimi_decision=self._kimi_to_dict(kimi_result)
        )

    def _tier1_ml(self, symbol: str, features: Dict[str, float]) -> Dict:
        """Tier 1: Fast ML prediction."""
        try:
            import numpy as np
            feature_array = np.array(list(features.values())).reshape(1, -1)
            prediction = self._ml_ensemble.predict(symbol, feature_array)
            return {
                "direction": prediction.get("direction", 0),
                "probability": prediction.get("probability", 0.5),
                "confidence": prediction.get("confidence", 0.5)
            }
        except Exception as e:
            logger.warning(f"Tier 1 error: {e}")
            return {"direction": 0, "probability": 0.5, "confidence": 0.0}

    async def _tier2_formula(
        self,
        symbol: str,
        ohlcv: Dict[str, float],
        features: Dict[str, float],
        ml_result: Dict
    ):
        """Tier 2: Formula analysis."""
        return await self._formula_brain.analyze(
            symbol=symbol,
            ohlcv=ohlcv,
            tick_features=features,
            ml_prediction=ml_result
        )

    async def _tier3_kimi(
        self,
        symbol: str,
        formula_result,
        account_state: Dict,
        market_context: Optional[Dict]
    ):
        """Tier 3: Kimi-K2 decision."""
        return await self._trading_brain.decide(
            symbol=symbol,
            formula_analysis=self._formula_to_dict(formula_result),
            account_state=account_state,
            market_context=market_context
        )

    def _needs_tier3(
        self,
        signal_strength: float,
        account_state: Dict,
        market_context: Optional[Dict]
    ) -> bool:
        """Determine if Tier 3 is needed."""
        # Very strong signal
        if signal_strength > self.TIER2_TIER3_SIGMA:
            return True

        # Large position
        if account_state.get("open_position_pct", 0) > self.LARGE_POSITION_PCT:
            return True

        # Friday afternoon (weekend gap risk)
        now = datetime.now()
        if now.weekday() == 4 and now.hour >= 14:  # Friday after 2 PM
            return True

        # High correlation with existing positions
        if market_context:
            if market_context.get("correlation_with_open", 0) > self.CORRELATION_THRESHOLD:
                return True

            # Unusual volatility regime
            if market_context.get("volatility_regime") == "extreme":
                return True

        return False

    def _calculate_position(self, confidence: float, account_state: Dict) -> float:
        """Simple position sizing based on confidence."""
        base = 10  # Base 10% position
        return min(50, base * (confidence / 0.6))

    def _kelly_position(
        self,
        confidence: float,
        signal_strength: float,
        account_state: Dict
    ) -> float:
        """Kelly-based position sizing."""
        # Simplified Kelly: f = (p * b - q) / b
        # where p = win prob, b = win/loss ratio, q = 1-p
        win_prob = 0.5 + (confidence - 0.5) * 0.5  # Convert confidence to win prob
        win_loss_ratio = 1.5 + (signal_strength - 1.5) * 0.2  # Higher sigma = better ratio

        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio
        kelly = max(0, kelly)

        # Use half Kelly for safety
        half_kelly = kelly * 0.5

        # Cap at 25% of capital
        return min(25, half_kelly * 100)

    def _dynamic_stop(self, formula_result) -> float:
        """Dynamic stop loss based on volatility."""
        vol = formula_result.volatility.get("har_rv", 0.001)
        # Higher vol = wider stop
        return max(15, min(50, 25 * (1 + vol * 100)))

    def _dynamic_tp(self, formula_result) -> float:
        """Dynamic take profit based on signal strength."""
        # Stronger signal = wider TP (let winners run)
        return max(30, min(100, 50 * (formula_result.signal_strength / 1.5)))

    def _formula_to_dict(self, formula_result) -> Dict:
        """Convert FormulaAnalysis to dict."""
        return {
            "alpha_signals": formula_result.alpha_signals,
            "volatility": formula_result.volatility,
            "microstructure": formula_result.microstructure,
            "signal_strength": formula_result.signal_strength,
            "direction": formula_result.direction,
            "confidence": formula_result.confidence,
            "reasoning": formula_result.reasoning
        }

    def _kimi_to_dict(self, kimi_result) -> Dict:
        """Convert TradingDecision to dict."""
        return {
            "action": kimi_result.action,
            "position_pct": kimi_result.position_pct,
            "stop_loss_pips": kimi_result.stop_loss_pips,
            "take_profit_pips": kimi_result.take_profit_pips,
            "confidence": kimi_result.confidence,
            "reasoning": kimi_result.reasoning
        }

    def _latency_ms(self, start: datetime) -> float:
        """Calculate latency in milliseconds."""
        return (datetime.now() - start).total_seconds() * 1000

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        total = self._stats["total_ticks"]
        return {
            **self._stats,
            "tier1_skip_rate": self._stats["tier1_skips"] / max(1, total),
            "tier2_call_rate": self._stats["tier2_calls"] / max(1, total),
            "tier3_call_rate": self._stats["tier3_calls"] / max(1, total),
            "trade_rate": self._stats["trades_executed"] / max(1, total)
        }

    async def close(self):
        """Cleanup."""
        if self._trading_brain and hasattr(self._trading_brain, 'close'):
            await self._trading_brain.close()


# Factory function
def create_three_tier_trader(
    formula_model: str = "forex-finetuned",
    trading_provider: str = "moonshot",
    enable_tier3: bool = True
) -> ThreeTierTrader:
    """Create a Three-Tier Trader."""
    return ThreeTierTrader(
        formula_model=formula_model,
        trading_provider=trading_provider,
        enable_tier3=enable_tier3
    )
