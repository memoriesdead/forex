"""
Formula Brain - Local DeepSeek-R1 Fine-tuned on 806+ Forex Formulas
====================================================================
Runs locally via Ollama for fast, free mathematical analysis.

Part of Dual-Brain Architecture:
- Formula Brain (this): Math & signal calculation
- Trading Brain: Final trading decisions via API
"""

import json
import subprocess
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FormulaAnalysis:
    """Output from Formula Brain analysis."""
    alpha_signals: Dict[str, float]
    volatility: Dict[str, float]
    microstructure: Dict[str, float]
    signal_strength: float  # In sigma units
    direction: int  # -1, 0, 1
    confidence: float  # 0-1
    reasoning: str
    raw_response: str


class FormulaBrain:
    """
    Local LLM for mathematical forex analysis.

    Uses fine-tuned DeepSeek-R1 model with 806+ embedded formulas:
    - Alpha101 (WorldQuant)
    - Alpha191 (Guotai Junan)
    - Volatility (HAR-RV, GARCH, etc.)
    - Microstructure (VPIN, Kyle Lambda, etc.)
    - Risk (Kelly Criterion, VaR, etc.)
    """

    def __init__(
        self,
        model_name: str = "forex-finetuned",
        ollama_path: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.model_name = model_name
        self.ollama_path = ollama_path or self._find_ollama()
        self.timeout = timeout
        self._warmed_up = False

    def _find_ollama(self) -> str:
        """Find Ollama executable."""
        paths = [
            Path.home() / "AppData/Local/Programs/Ollama/ollama.exe",
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
        ]
        for p in paths:
            if p.exists():
                return str(p)
        return "ollama"  # Hope it's in PATH

    async def warmup(self) -> bool:
        """Warm up the model for faster first inference."""
        if self._warmed_up:
            return True

        try:
            logger.info(f"Warming up Formula Brain ({self.model_name})...")
            result = await self._query("What is Alpha001?", max_tokens=50)
            self._warmed_up = bool(result)
            logger.info("Formula Brain ready")
            return self._warmed_up
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return False

    async def analyze(
        self,
        symbol: str,
        ohlcv: Dict[str, float],
        tick_features: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> FormulaAnalysis:
        """
        Analyze market data using formula knowledge.

        Args:
            symbol: Currency pair (e.g., "EURUSD")
            ohlcv: Dict with open, high, low, close, volume
            tick_features: Optional pre-calculated features
            ml_prediction: Optional ML ensemble prediction

        Returns:
            FormulaAnalysis with signals, confidence, and reasoning
        """
        prompt = self._build_analysis_prompt(symbol, ohlcv, tick_features, ml_prediction)
        response = await self._query(prompt)
        return self._parse_response(response)

    def _build_analysis_prompt(
        self,
        symbol: str,
        ohlcv: Dict[str, float],
        tick_features: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the analysis prompt."""
        prompt = f"""Analyze {symbol} using your forex formula knowledge.

CURRENT DATA:
- Open: {ohlcv.get('open', 0):.5f}
- High: {ohlcv.get('high', 0):.5f}
- Low: {ohlcv.get('low', 0):.5f}
- Close: {ohlcv.get('close', 0):.5f}
- Volume: {ohlcv.get('volume', 0):.0f}
- Spread: {ohlcv.get('spread', 0):.5f}
"""

        if tick_features:
            prompt += f"""
KEY FEATURES:
- Returns (1m): {tick_features.get('return_1m', 0):.6f}
- Returns (5m): {tick_features.get('return_5m', 0):.6f}
- Volatility: {tick_features.get('volatility', 0):.6f}
- RSI: {tick_features.get('rsi', 50):.1f}
- MACD: {tick_features.get('macd', 0):.6f}
"""

        if ml_prediction:
            prompt += f"""
ML ENSEMBLE PREDICTION:
- Direction: {ml_prediction.get('direction', 0)}
- Probability: {ml_prediction.get('probability', 0.5):.2%}
- Confidence: {ml_prediction.get('confidence', 0):.2f}
"""

        prompt += """
CALCULATE AND PROVIDE:
1. Key alpha signals (Alpha101_042, Alpha191_015, etc.)
2. Volatility forecast (HAR-RV or GARCH)
3. Microstructure signals (VPIN, Kyle Lambda if data available)
4. Overall signal strength in sigma units
5. Direction (-1 sell, 0 neutral, +1 buy)
6. Confidence (0-1)
7. Brief reasoning (1-2 sentences)

Respond in JSON format:
{
  "alpha_signals": {"alpha101_042": 0.0, ...},
  "volatility": {"har_rv": 0.0, "garch": 0.0},
  "microstructure": {"vpin": 0.0, "kyle_lambda": 0.0},
  "signal_strength": 0.0,
  "direction": 0,
  "confidence": 0.0,
  "reasoning": "..."
}
"""
        return prompt

    async def _query(self, prompt: str, max_tokens: int = 500) -> str:
        """Query the local Ollama model."""
        cmd = [
            self.ollama_path, "run", self.model_name,
            "--nowordwrap"
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=self.timeout
            )

            if proc.returncode != 0:
                logger.error(f"Ollama error: {stderr.decode()}")
                return ""

            return stdout.decode().strip()

        except asyncio.TimeoutError:
            logger.error(f"Ollama timeout after {self.timeout}s")
            proc.kill()
            return ""
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return ""

    def _parse_response(self, response: str) -> FormulaAnalysis:
        """Parse the LLM response into structured output."""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                return FormulaAnalysis(
                    alpha_signals=data.get("alpha_signals", {}),
                    volatility=data.get("volatility", {}),
                    microstructure=data.get("microstructure", {}),
                    signal_strength=float(data.get("signal_strength", 0)),
                    direction=int(data.get("direction", 0)),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    raw_response=response
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON: {e}")

        # Fallback: return neutral analysis
        return FormulaAnalysis(
            alpha_signals={},
            volatility={},
            microstructure={},
            signal_strength=0.0,
            direction=0,
            confidence=0.0,
            reasoning="Failed to parse response",
            raw_response=response
        )

    async def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> Dict[str, float]:
        """Calculate Kelly Criterion position sizing."""
        prompt = f"""Calculate Kelly Criterion for:
- Win rate: {win_rate:.2%}
- Average win: {avg_win:.4f}
- Average loss: {avg_loss:.4f}

Provide:
1. Full Kelly fraction
2. Half Kelly (recommended)
3. Quarter Kelly (conservative)

Respond in JSON: {{"full_kelly": 0.0, "half_kelly": 0.0, "quarter_kelly": 0.0, "formula": "..."}}
"""
        response = await self._query(prompt, max_tokens=200)

        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        # Fallback calculation
        if avg_loss > 0:
            b = avg_win / avg_loss
            q = 1 - win_rate
            kelly = (win_rate * b - q) / b
            kelly = max(0, kelly)
        else:
            kelly = 0

        return {
            "full_kelly": kelly,
            "half_kelly": kelly / 2,
            "quarter_kelly": kelly / 4,
            "formula": "kelly = (p*b - q) / b"
        }


# Convenience function
def create_formula_brain(model_name: str = "forex-finetuned") -> FormulaBrain:
    """Create a Formula Brain instance."""
    return FormulaBrain(model_name=model_name)
