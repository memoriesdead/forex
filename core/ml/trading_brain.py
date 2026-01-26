"""
Trading Brain - Kimi-K2 API for Final Trading Decisions
========================================================
Uses Kimi-K2-Instruct (1T MoE, 32B active) for trading decisions.

Part of Dual-Brain Architecture:
- Formula Brain: Math & signal calculation (local)
- Trading Brain (this): Final trading decisions via API

API Options:
- Moonshot AI: https://platform.moonshot.cn/
- Together.ai: https://www.together.ai/
- Self-hosted: vLLM/SGLang
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Output from Trading Brain."""
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    position_pct: float  # % of capital
    stop_loss_pips: float
    take_profit_pips: float
    confidence: float
    reasoning: str
    timestamp: datetime


class TradingBrain:
    """
    API-based LLM for trading decisions.

    Uses Kimi-K2-Instruct for:
    - Final BUY/SELL/HOLD decisions
    - Position sizing
    - Risk management
    - Multi-asset correlation
    """

    SYSTEM_PROMPT = """You are an expert forex trader with deep knowledge of:
- Risk management (Kelly Criterion, VaR, position sizing)
- Market microstructure and execution
- Multi-timeframe analysis
- Correlation between currency pairs
- Market regimes (trending, ranging, volatile)

You receive analyzed signals from a Formula Brain that has calculated:
- Alpha signals (Alpha101, Alpha191)
- Volatility forecasts (HAR-RV, GARCH)
- Microstructure signals (VPIN, Kyle Lambda)
- Signal strength in sigma units

Your job is to make the FINAL trading decision:
- Consider the formula analysis
- Apply risk management rules
- Account for current market conditions
- Determine optimal position size
- Set appropriate stop loss and take profit

RULES:
1. Never risk more than 2% of capital per trade
2. Minimum signal strength of 1.5σ to trade
3. Use tighter stops in high volatility
4. Reduce position size when correlation is high
5. HOLD if uncertain - no trade is better than bad trade

Always respond in JSON format."""

    def __init__(
        self,
        provider: Literal["moonshot", "together", "openai"] = "moonshot",
        api_key: Optional[str] = None,
        model: str = "kimi-k2-instruct",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

        # API endpoints
        self.endpoints = {
            "moonshot": "https://api.moonshot.cn/v1/chat/completions",
            "together": "https://api.together.xyz/v1/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions"
        }

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        key_names = {
            "moonshot": "MOONSHOT_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        key = os.getenv(key_names.get(self.provider, ""))
        if not key:
            logger.warning(f"No API key found for {self.provider}")
        return key or ""

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def decide(
        self,
        symbol: str,
        formula_analysis: Dict[str, Any],
        account_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> TradingDecision:
        """
        Make a trading decision based on formula analysis.

        Args:
            symbol: Currency pair
            formula_analysis: Output from Formula Brain
            account_state: Current account info (balance, positions, etc.)
            market_context: Optional additional context

        Returns:
            TradingDecision with action, size, stops, etc.
        """
        prompt = self._build_decision_prompt(
            symbol, formula_analysis, account_state, market_context
        )

        response = await self._query(prompt)
        return self._parse_decision(symbol, response)

    def _build_decision_prompt(
        self,
        symbol: str,
        formula_analysis: Dict[str, Any],
        account_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the decision prompt."""
        prompt = f"""Make a trading decision for {symbol}.

FORMULA BRAIN ANALYSIS:
{json.dumps(formula_analysis, indent=2)}

ACCOUNT STATE:
- Balance: ${account_state.get('balance', 0):,.2f}
- Open positions: {account_state.get('open_positions', 0)}
- Daily P&L: ${account_state.get('daily_pnl', 0):,.2f}
- Max drawdown today: {account_state.get('drawdown_pct', 0):.1%}
- Trades today: {account_state.get('trades_today', 0)}
"""

        if market_context:
            prompt += f"""
MARKET CONTEXT:
- Session: {market_context.get('session', 'unknown')}
- Volatility regime: {market_context.get('volatility_regime', 'normal')}
- Trend: {market_context.get('trend', 'neutral')}
- Correlated positions: {market_context.get('correlated_positions', [])}
"""

        prompt += """
PROVIDE YOUR DECISION in JSON format:
{
  "action": "BUY" | "SELL" | "HOLD",
  "position_pct": 0.0,  // % of capital (0-50)
  "stop_loss_pips": 0.0,
  "take_profit_pips": 0.0,
  "confidence": 0.0,  // 0-1
  "reasoning": "Brief explanation"
}
"""
        return prompt

    async def _query(self, prompt: str) -> str:
        """Query the API with retries."""
        session = await self._get_session()
        endpoint = self.endpoints.get(self.provider, self.endpoints["moonshot"])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }

        for attempt in range(self.max_retries):
            try:
                async with session.post(endpoint, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error = await resp.text()
                        logger.error(f"API error {resp.status}: {error}")

            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Query failed: {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

        return ""

    def _parse_decision(self, symbol: str, response: str) -> TradingDecision:
        """Parse the API response into a decision."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                action = data.get("action", "HOLD").upper()
                if action not in ["BUY", "SELL", "HOLD"]:
                    action = "HOLD"

                return TradingDecision(
                    action=action,
                    symbol=symbol,
                    position_pct=min(50, max(0, float(data.get("position_pct", 0)))),
                    stop_loss_pips=max(0, float(data.get("stop_loss_pips", 25))),
                    take_profit_pips=max(0, float(data.get("take_profit_pips", 50))),
                    confidence=min(1, max(0, float(data.get("confidence", 0)))),
                    reasoning=data.get("reasoning", ""),
                    timestamp=datetime.now()
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decision: {e}")

        # Fallback: HOLD
        return TradingDecision(
            action="HOLD",
            symbol=symbol,
            position_pct=0,
            stop_loss_pips=0,
            take_profit_pips=0,
            confidence=0,
            reasoning="Failed to parse response - defaulting to HOLD",
            timestamp=datetime.now()
        )

    async def explain_regime(
        self,
        symbols: list[str],
        market_data: Dict[str, Any]
    ) -> str:
        """Get market regime explanation."""
        prompt = f"""Analyze the current market regime for {', '.join(symbols)}.

MARKET DATA:
{json.dumps(market_data, indent=2)}

Explain:
1. Current regime (trending/ranging/volatile)
2. Key drivers
3. Expected duration
4. Trading implications
"""
        return await self._query(prompt)


class TradingBrainLocal:
    """
    Local fallback using Ollama (for when API is unavailable).
    Uses same model as Formula Brain but with trading prompts.
    """

    def __init__(self, model_name: str = "forex-finetuned"):
        self.model_name = model_name
        # Reuse Formula Brain's Ollama logic
        from .formula_brain import FormulaBrain
        self._brain = FormulaBrain(model_name=model_name)

    async def decide(
        self,
        symbol: str,
        formula_analysis: Dict[str, Any],
        account_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> TradingDecision:
        """Make decision using local model."""
        prompt = f"""As a forex trader, make a decision for {symbol}.

SIGNALS: {json.dumps(formula_analysis)}
ACCOUNT: Balance ${account_state.get('balance', 0):.2f}

Decision (JSON): {{"action": "BUY/SELL/HOLD", "position_pct": 0-50, "stop_loss_pips": 0, "take_profit_pips": 0, "confidence": 0-1, "reasoning": "..."}}
"""
        response = await self._brain._query(prompt)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return TradingDecision(
                    action=data.get("action", "HOLD"),
                    symbol=symbol,
                    position_pct=float(data.get("position_pct", 0)),
                    stop_loss_pips=float(data.get("stop_loss_pips", 25)),
                    take_profit_pips=float(data.get("take_profit_pips", 50)),
                    confidence=float(data.get("confidence", 0)),
                    reasoning=data.get("reasoning", ""),
                    timestamp=datetime.now()
                )
        except:
            pass

        return TradingDecision(
            action="HOLD", symbol=symbol, position_pct=0,
            stop_loss_pips=0, take_profit_pips=0, confidence=0,
            reasoning="Parse error", timestamp=datetime.now()
        )


# Factory function
def create_trading_brain(
    provider: str = "moonshot",
    api_key: Optional[str] = None,
    use_local_fallback: bool = True
) -> TradingBrain:
    """
    Create a Trading Brain instance.

    Args:
        provider: "moonshot", "together", or "openai"
        api_key: API key (or set env var)
        use_local_fallback: Use local Ollama if API fails
    """
    brain = TradingBrain(provider=provider, api_key=api_key)

    if use_local_fallback and not brain.api_key:
        logger.info("No API key - using local fallback")
        return TradingBrainLocal()

    return brain
