"""
LLM Reasoner for Forex Trading - Chinese Quant Style Integration
================================================================
Integrates DeepSeek-R1 using multi-agent architecture inspired by:
- 幻方量化 (High-Flyer Quant) - DeepSeek's parent company
- 九坤投资 (Ubiquant) - Chinese quant LLM research
- TradingAgents (UCLA/MIT) - Multi-agent debate framework
- arxiv:2409.06289 - Automate Strategy Finding with LLM

Architecture:
- Async inference via Ollama for low latency (<100ms for simple queries)
- Multi-agent perspectives: Bull, Bear, Risk Manager
- Chain-of-thought reasoning for explainable decisions
- Prompt caching for repeated patterns

Sources:
- https://github.com/TauricResearch/TradingAgents
- https://arxiv.org/html/2409.06289v3
- https://github.com/Open-Finance-Lab/AgenticTrading
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

# Ollama path for Windows
OLLAMA_PATH = r"C:\Users\kevin\AppData\Local\Programs\Ollama\ollama.exe"


class AgentRole(Enum):
    """Agent roles in the multi-agent trading system."""
    TECHNICAL_ANALYST = "technical"
    FUNDAMENTAL_ANALYST = "fundamental"
    SENTIMENT_ANALYST = "sentiment"
    BULL_RESEARCHER = "bull"
    BEAR_RESEARCHER = "bear"
    RISK_MANAGER = "risk"
    TRADER = "trader"


@dataclass
class MarketContext:
    """Current market state for LLM context."""
    symbol: str
    current_price: float
    spread_pips: float
    direction_prediction: int  # 1=up, -1=down, 0=neutral
    confidence: float
    regime: str  # "trending", "mean_reverting", "volatile"
    features: Dict[str, float] = field(default_factory=dict)
    recent_trades: List[Dict] = field(default_factory=list)
    account_balance: float = 0.0
    current_position: float = 0.0
    daily_pnl: float = 0.0


@dataclass
class TradingDecision:
    """LLM-reasoned trading decision."""
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    position_size_pct: float
    reasoning: str  # Chain-of-thought explanation
    bull_argument: str
    bear_argument: str
    risk_assessment: str
    latency_ms: float
    should_override_ml: bool = False


class LRUCache:
    """Simple LRU cache for prompt responses."""

    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: str) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = value


class OllamaClient:
    """Async client for Ollama API with model fallback."""

    # Model preference order: forex-r1-v3 (SFT+DPO on H100), then fallbacks
    FALLBACK_MODELS = [
        "forex-r1-v3:latest",  # SFT+DPO trained on 8x H100 with 1,183 formulas + YOUR 51 pairs
        "forex-r1-v2:latest",  # GRPO-trained model with 806+ formulas + trade reasoning
        "forex-r1:latest",  # Custom model with 806+ formulas embedded
        "forex-expert:latest",
        "deepseek-r1:8b",
        "deepseek-r1:7b",
        "qwen2.5:7b",
    ]

    def __init__(self, model: str = "forex-r1-v3:latest", ollama_path: str = OLLAMA_PATH):
        self.model = model
        self.ollama_path = ollama_path
        self.cache = LRUCache(max_size=200)
        self._warm = False
        self._fallback_tried = False

    async def _check_model_available(self, model: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            import subprocess
            process = await asyncio.create_subprocess_exec(
                self.ollama_path, "list",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
            return model.split(":")[0] in stdout.decode('utf-8', errors='ignore')
        except Exception:
            return False

    async def _try_fallback(self) -> None:
        """Try fallback models if primary model fails."""
        if self._fallback_tried:
            return

        self._fallback_tried = True
        for fallback_model in self.FALLBACK_MODELS:
            if fallback_model != self.model:
                if await self._check_model_available(fallback_model):
                    logger.info(f"Falling back from {self.model} to {fallback_model}")
                    self.model = fallback_model
                    return

        logger.warning(f"No fallback model available, staying with {self.model}")

    async def generate(self, prompt: str, use_cache: bool = True,
                       max_tokens: int = 500, temperature: float = 0.3) -> Tuple[str, float]:
        """Generate response from Ollama with timing."""
        start = time.perf_counter()

        # Check cache first
        if use_cache:
            cache_key = hashlib.md5(f"{prompt}:{max_tokens}:{temperature}".encode()).hexdigest()
            cached = self.cache.get(cache_key)
            if cached:
                latency = (time.perf_counter() - start) * 1000
                return cached, latency

        try:
            # Use subprocess for async execution
            import subprocess
            process = await asyncio.create_subprocess_exec(
                self.ollama_path, "run", self.model, prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0  # 30 second timeout
            )
            response = stdout.decode('utf-8', errors='ignore').strip()

            # Cache the response
            if use_cache and response:
                self.cache.set(cache_key, response)

            latency = (time.perf_counter() - start) * 1000
            return response, latency

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start) * 1000
            logger.warning(f"Ollama timeout after {latency:.0f}ms")
            return "TIMEOUT: Unable to complete reasoning", latency
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error(f"Ollama error: {e}")
            return f"ERROR: {str(e)}", latency

    async def warmup(self) -> float:
        """Warm up the model with a simple query, using fallback if needed."""
        if self._warm:
            return 0.0

        # Check if primary model is available, try fallback if not
        if not await self._check_model_available(self.model):
            logger.info(f"Model {self.model} not found, trying fallbacks...")
            await self._try_fallback()

        response, latency = await self.generate("What is 2+2?", use_cache=False)

        # If response contains ERROR and we haven't tried fallback yet, try now
        if "ERROR" in response and not self._fallback_tried:
            await self._try_fallback()
            response, latency = await self.generate("What is 2+2?", use_cache=False)

        self._warm = True
        logger.info(f"Ollama warmup complete: {latency:.0f}ms (model: {self.model})")
        return latency


class ForexTradingAgent:
    """Single agent with a specific role in the trading decision process."""

    PROMPTS = {
        AgentRole.TECHNICAL_ANALYST: """You are a Technical Analyst for forex trading.
Analyze these indicators for {symbol}:
- Price: {price}, Direction Prediction: {direction} ({confidence:.1%} confidence)
- Regime: {regime}
- Key features: {features}

Provide a brief technical analysis (2-3 sentences). Focus on momentum, support/resistance, and trend strength.""",

        AgentRole.BULL_RESEARCHER: """You are a Bullish Researcher arguing FOR a {direction_word} position in {symbol}.
Current analysis:
{technical_analysis}

Make your BULL case in 2-3 sentences. Why should we be optimistic about this trade?""",

        AgentRole.BEAR_RESEARCHER: """You are a Bearish Researcher arguing AGAINST a {direction_word} position in {symbol}.
Current analysis:
{technical_analysis}

Make your BEAR case in 2-3 sentences. What are the risks and why might this trade fail?""",

        AgentRole.RISK_MANAGER: """You are a Risk Manager evaluating a potential trade.
Symbol: {symbol}
Account Balance: ${balance:,.2f}
Current Position: {position:,.0f} units
Daily P&L: ${daily_pnl:,.2f}
Proposed Direction: {direction_word}
ML Confidence: {confidence:.1%}

BULL case: {bull_case}
BEAR case: {bear_case}

Evaluate the risk. Recommend position size as percentage of account (0-50%). Be conservative.
Format: "RISK_SCORE: X/10, POSITION_SIZE: Y%, REASON: ..."
""",

        AgentRole.TRADER: """You are the Head Trader making the final decision for {symbol}.

MARKET STATE:
- Price: {price}, Spread: {spread} pips
- ML Prediction: {direction_word} ({confidence:.1%})
- Regime: {regime}

ANALYSIS:
Technical: {technical_analysis}
Bull Case: {bull_case}
Bear Case: {bear_case}
Risk Assessment: {risk_assessment}

DECISION REQUIRED:
1. ACTION: BUY, SELL, or HOLD
2. If trading, position size recommendation
3. Brief reasoning (1-2 sentences)

Think step by step. Consider:
- Is the ML confidence high enough (>60%)?
- Does the bull or bear case win the debate?
- Is the risk acceptable?

Format your response as:
ACTION: [BUY/SELL/HOLD]
SIZE: [percentage]%
REASONING: [your explanation]
"""
    }

    def __init__(self, role: AgentRole, client: OllamaClient):
        self.role = role
        self.client = client

    async def analyze(self, context: MarketContext, **kwargs) -> Tuple[str, float]:
        """Run analysis for this agent's role."""
        prompt_template = self.PROMPTS.get(self.role, "Analyze: {symbol}")

        # Build prompt with context
        direction_word = "LONG" if context.direction_prediction == 1 else "SHORT" if context.direction_prediction == -1 else "NEUTRAL"

        # Select top 5 features by absolute value
        top_features = dict(sorted(
            context.features.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )[:5])

        prompt = prompt_template.format(
            symbol=context.symbol,
            price=context.current_price,
            spread=context.spread_pips,
            direction=context.direction_prediction,
            direction_word=direction_word,
            confidence=context.confidence,
            regime=context.regime,
            features=json.dumps(top_features, indent=2),
            balance=context.account_balance,
            position=context.current_position,
            daily_pnl=context.daily_pnl,
            **kwargs
        )

        return await self.client.generate(prompt)


class MultiAgentTradingReasoner:
    """
    Multi-agent LLM system for forex trading decisions.

    Architecture inspired by:
    - TradingAgents (UCLA/MIT): Bull/Bear debate mechanism
    - High-Flyer Quant: Deep learning trading since 2017
    - AgenticTrading: Memory-augmented alpha discovery

    Flow:
    1. Technical Analyst provides market analysis
    2. Bull Researcher argues for the trade
    3. Bear Researcher argues against the trade
    4. Risk Manager evaluates and sizes the position
    5. Trader makes final decision
    """

    def __init__(self, model: str = "forex-r1-v3:latest"):
        self.client = OllamaClient(model=model)
        self.agents = {
            role: ForexTradingAgent(role, self.client)
            for role in AgentRole
        }
        self.decision_history: List[TradingDecision] = []
        self._initialized = False

    async def initialize(self) -> float:
        """Initialize and warmup the model."""
        if self._initialized:
            return 0.0
        latency = await self.client.warmup()
        self._initialized = True
        return latency

    async def analyze_trade(self, context: MarketContext,
                            fast_mode: bool = False) -> TradingDecision:
        """
        Run full multi-agent analysis on a potential trade.

        Args:
            context: Current market state and ML predictions
            fast_mode: If True, skip debate and use cached patterns

        Returns:
            TradingDecision with action, reasoning, and confidence
        """
        start = time.perf_counter()

        if fast_mode:
            return await self._fast_analysis(context)

        # Stage 1: Technical Analysis
        technical_analysis, ta_latency = await self.agents[AgentRole.TECHNICAL_ANALYST].analyze(context)

        # Stage 2: Bull/Bear Debate (parallel)
        bull_task = self.agents[AgentRole.BULL_RESEARCHER].analyze(
            context, technical_analysis=technical_analysis
        )
        bear_task = self.agents[AgentRole.BEAR_RESEARCHER].analyze(
            context, technical_analysis=technical_analysis
        )

        (bull_case, bull_latency), (bear_case, bear_latency) = await asyncio.gather(
            bull_task, bear_task
        )

        # Stage 3: Risk Assessment
        risk_assessment, risk_latency = await self.agents[AgentRole.RISK_MANAGER].analyze(
            context,
            technical_analysis=technical_analysis,
            bull_case=bull_case,
            bear_case=bear_case
        )

        # Stage 4: Final Trading Decision
        trader_response, trader_latency = await self.agents[AgentRole.TRADER].analyze(
            context,
            technical_analysis=technical_analysis,
            bull_case=bull_case,
            bear_case=bear_case,
            risk_assessment=risk_assessment
        )

        total_latency = (time.perf_counter() - start) * 1000

        # Parse trader response
        decision = self._parse_trader_response(
            trader_response,
            context,
            bull_case,
            bear_case,
            risk_assessment,
            total_latency
        )

        self.decision_history.append(decision)
        return decision

    async def _fast_analysis(self, context: MarketContext) -> TradingDecision:
        """Fast analysis using simplified prompts and caching."""
        start = time.perf_counter()

        direction_word = "LONG" if context.direction_prediction == 1 else "SHORT"

        prompt = f"""Quick forex decision for {context.symbol}:
ML says {direction_word} with {context.confidence:.0%} confidence.
Regime: {context.regime}, Spread: {context.spread_pips} pips.

Reply in one line: ACTION SIZE% REASON
Example: BUY 5% Strong momentum with tight spread"""

        response, latency = await self.client.generate(prompt, use_cache=True, max_tokens=100)

        total_latency = (time.perf_counter() - start) * 1000

        # Parse simple response
        action = "HOLD"
        size = 0.0
        if "BUY" in response.upper():
            action = "BUY"
        elif "SELL" in response.upper():
            action = "SELL"

        # Extract percentage
        import re
        size_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if size_match:
            size = float(size_match.group(1))

        return TradingDecision(
            action=action,
            confidence=context.confidence,
            position_size_pct=min(size, 50.0),
            reasoning=response,
            bull_argument="(fast mode)",
            bear_argument="(fast mode)",
            risk_assessment="(fast mode)",
            latency_ms=total_latency,
            should_override_ml=False
        )

    def _parse_trader_response(self, response: str, context: MarketContext,
                               bull_case: str, bear_case: str,
                               risk_assessment: str, latency: float) -> TradingDecision:
        """Parse the trader agent's response into a TradingDecision."""
        import re

        # Default values
        action = "HOLD"
        size = 0.0
        reasoning = response

        # Parse ACTION
        action_match = re.search(r'ACTION:\s*(BUY|SELL|HOLD)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()

        # Parse SIZE
        size_match = re.search(r'SIZE:\s*(\d+(?:\.\d+)?)\s*%', response, re.IGNORECASE)
        if size_match:
            size = float(size_match.group(1))

        # Parse REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Check if LLM wants to override ML prediction
        override = False
        if action == "HOLD" and context.confidence > 0.7:
            override = True  # LLM is being more conservative than ML
        elif action != "HOLD" and context.confidence < 0.55:
            override = True  # LLM is more aggressive than ML

        return TradingDecision(
            action=action,
            confidence=context.confidence,
            position_size_pct=min(size, 50.0),
            reasoning=reasoning,
            bull_argument=bull_case[:500],  # Truncate for storage
            bear_argument=bear_case[:500],
            risk_assessment=risk_assessment[:500],
            latency_ms=latency,
            should_override_ml=override
        )

    async def validate_trade_entry(self, context: MarketContext) -> Tuple[bool, str, float]:
        """
        Quick validation before trade entry.

        Returns: (should_trade, reason, latency_ms)
        """
        start = time.perf_counter()

        direction_word = "LONG" if context.direction_prediction == 1 else "SHORT"

        prompt = f"""Quick trade validation for {context.symbol}:
- Direction: {direction_word}
- Confidence: {context.confidence:.0%}
- Spread: {context.spread_pips} pips
- Regime: {context.regime}

Should we trade? Reply: YES or NO, then brief reason (10 words max)."""

        response, _ = await self.client.generate(prompt, use_cache=True, max_tokens=50)

        latency = (time.perf_counter() - start) * 1000

        should_trade = "YES" in response.upper()
        return should_trade, response, latency

    async def explain_regime(self, regime: str, features: Dict[str, float]) -> str:
        """Get LLM explanation of current market regime."""
        top_features = dict(sorted(features.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)[:5])

        prompt = f"""Explain this forex market regime in 2 sentences:
Regime: {regime}
Key indicators: {json.dumps(top_features)}

What does this mean for trading?"""

        response, _ = await self.client.generate(prompt, use_cache=True)
        return response

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about LLM decisions."""
        if not self.decision_history:
            return {"total_decisions": 0}

        actions = [d.action for d in self.decision_history]
        latencies = [d.latency_ms for d in self.decision_history]
        overrides = sum(1 for d in self.decision_history if d.should_override_ml)

        return {
            "total_decisions": len(self.decision_history),
            "actions": {
                "BUY": actions.count("BUY"),
                "SELL": actions.count("SELL"),
                "HOLD": actions.count("HOLD")
            },
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "ml_overrides": overrides,
            "override_rate": overrides / len(self.decision_history)
        }


class TradingBotLLMIntegration:
    """
    Integration layer between trading bot and LLM reasoner.

    Usage modes:
    1. ADVISORY: LLM provides reasoning but doesn't change ML decisions
    2. VALIDATION: LLM can veto trades that ML suggests
    3. AUTONOMOUS: LLM can override ML decisions (use with caution)
    """

    class Mode(Enum):
        ADVISORY = "advisory"
        VALIDATION = "validation"
        AUTONOMOUS = "autonomous"

    def __init__(self, mode: Mode = Mode.VALIDATION):
        self.reasoner = MultiAgentTradingReasoner()
        self.mode = mode
        self.enabled = True
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM reasoner."""
        if not self._initialized:
            await self.reasoner.initialize()
            self._initialized = True
            logger.info(f"LLM Integration initialized in {self.mode.value} mode")

    async def process_signal(
        self,
        symbol: str,
        ml_signal: int,
        ml_confidence: float,
        current_price: float,
        spread_pips: float,
        regime: str,
        features: Dict[str, float],
        account_balance: float,
        current_position: float,
        daily_pnl: float,
        fast_mode: bool = True
    ) -> Tuple[int, float, str]:
        """
        Process ML signal through LLM reasoning.

        Args:
            symbol: Currency pair
            ml_signal: ML prediction (1=buy, -1=sell, 0=hold)
            ml_confidence: ML confidence (0-1)
            current_price: Current price
            spread_pips: Current spread
            regime: Market regime
            features: Feature dict
            account_balance: Account balance
            current_position: Current position size
            daily_pnl: Daily P&L
            fast_mode: Use fast analysis (recommended for HFT)

        Returns:
            (final_signal, position_size_pct, reasoning)
        """
        if not self.enabled:
            return ml_signal, 10.0, "LLM disabled"

        context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            spread_pips=spread_pips,
            direction_prediction=ml_signal,
            confidence=ml_confidence,
            regime=regime,
            features=features,
            account_balance=account_balance,
            current_position=current_position,
            daily_pnl=daily_pnl
        )

        if self.mode == self.Mode.ADVISORY:
            # Just get reasoning, don't change signal
            decision = await self.reasoner.analyze_trade(context, fast_mode=fast_mode)
            return ml_signal, 10.0, decision.reasoning

        elif self.mode == self.Mode.VALIDATION:
            # LLM can veto but not initiate trades
            should_trade, reason, _ = await self.reasoner.validate_trade_entry(context)

            if ml_signal != 0 and not should_trade:
                return 0, 0.0, f"VETOED: {reason}"
            return ml_signal, 10.0, reason

        elif self.mode == self.Mode.AUTONOMOUS:
            # LLM has full control
            decision = await self.reasoner.analyze_trade(context, fast_mode=fast_mode)

            if decision.action == "BUY":
                return 1, decision.position_size_pct, decision.reasoning
            elif decision.action == "SELL":
                return -1, decision.position_size_pct, decision.reasoning
            else:
                return 0, 0.0, decision.reasoning

        return ml_signal, 10.0, "Unknown mode"

    def set_mode(self, mode: Mode) -> None:
        """Change the integration mode."""
        self.mode = mode
        logger.info(f"LLM Integration mode changed to: {mode.value}")

    def disable(self) -> None:
        """Disable LLM integration."""
        self.enabled = False

    def enable(self) -> None:
        """Enable LLM integration."""
        self.enabled = True


# Convenience functions
def create_llm_integration(mode: str = "validation") -> TradingBotLLMIntegration:
    """Create LLM integration with specified mode."""
    mode_map = {
        "advisory": TradingBotLLMIntegration.Mode.ADVISORY,
        "validation": TradingBotLLMIntegration.Mode.VALIDATION,
        "autonomous": TradingBotLLMIntegration.Mode.AUTONOMOUS
    }
    return TradingBotLLMIntegration(mode=mode_map.get(mode, TradingBotLLMIntegration.Mode.VALIDATION))


async def test_reasoner():
    """Test the LLM reasoner."""
    print("=" * 60)
    print("Testing Multi-Agent LLM Trading Reasoner")
    print("=" * 60)

    reasoner = MultiAgentTradingReasoner()
    await reasoner.initialize()

    context = MarketContext(
        symbol="EURUSD",
        current_price=1.0850,
        spread_pips=0.8,
        direction_prediction=1,
        confidence=0.68,
        regime="trending",
        features={
            "rsi_14": 62.5,
            "macd_signal": 0.0012,
            "atr_14": 0.0045,
            "volume_ratio": 1.3,
            "momentum_10": 0.0023
        },
        account_balance=10000.0,
        current_position=0,
        daily_pnl=125.50
    )

    print("\n--- Fast Mode Analysis ---")
    fast_decision = await reasoner.analyze_trade(context, fast_mode=True)
    print(f"Action: {fast_decision.action}")
    print(f"Size: {fast_decision.position_size_pct}%")
    print(f"Latency: {fast_decision.latency_ms:.0f}ms")
    print(f"Reasoning: {fast_decision.reasoning}")

    print("\n--- Full Multi-Agent Analysis ---")
    full_decision = await reasoner.analyze_trade(context, fast_mode=False)
    print(f"Action: {full_decision.action}")
    print(f"Size: {full_decision.position_size_pct}%")
    print(f"Latency: {full_decision.latency_ms:.0f}ms")
    print(f"\nBull Case: {full_decision.bull_argument[:200]}...")
    print(f"\nBear Case: {full_decision.bear_argument[:200]}...")
    print(f"\nRisk Assessment: {full_decision.risk_assessment[:200]}...")
    print(f"\nFinal Reasoning: {full_decision.reasoning}")

    print("\n--- Statistics ---")
    stats = reasoner.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(test_reasoner())
