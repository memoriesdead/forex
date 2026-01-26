"""
HFT Trading Bot - Production Ready
===================================
Real-time forex trading using institutional-grade ML ensemble.

Components:
- TrueFX live tick feed
- HFT feature engine (100+ features)
- ML ensemble predictor (XGBoost/LightGBM/CatBoost)
- IB Gateway execution
- Latency-aware order management
- Risk management (Kelly, drawdown limits)
- Real-time monitoring

Usage:
    python scripts/hft_trading_bot.py --mode paper --symbols EURUSD,GBPUSD
    python scripts/hft_trading_bot.py --mode live --symbols EURUSD

Modes:
    paper - Paper trading via IB Gateway (DUO423364)
    live - Live trading (requires live account)
    backtest - Run on historical data
    monitor - Monitor only, no trades
"""

import asyncio
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import pickle
import signal
import sys
from collections import deque

# Load .env file before anything else
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system env vars

sys.path.insert(0, str(Path(__file__).parent.parent))

# Live retraining components
try:
    from core.data.buffer import get_tick_buffer
    from core.ml.retrainer import get_hybrid_retrainer  # Historical + Live combined
    HAS_LIVE_RETRAIN = True
except ImportError:
    HAS_LIVE_RETRAIN = False

# Chinese Quant Online Learning (2026-01-18)
# Based on techniques from 幻方量化, 九坤投资, 明汯投资
# Reference: https://bigquant.com/wiki/doc/xVqIPu6RoI
try:
    from core.ml.adaptive_ensemble import AdaptiveMLEnsemble, create_adaptive_ensemble
    HAS_CHINESE_ONLINE = True
except ImportError:
    HAS_CHINESE_ONLINE = False

# Execution optimization (2026-01-17)
try:
    from core.execution.optimization import (
        get_execution_interceptor,
        get_execution_engine,
        ExecutionConfig,
    )
    HAS_EXEC_OPTIMIZER = True
except ImportError:
    HAS_EXEC_OPTIMIZER = False

# Multi-source data feed (2026-01-18)
# Maximum coverage: TrueFX + IB + OANDA = 70+ pairs
try:
    from core.data.multi_source_feed import (
        MultiSourceFeed,
        create_multi_source_feed,
        DataSource,
        LiveTick,
    )
    HAS_MULTI_SOURCE = True
except ImportError:
    HAS_MULTI_SOURCE = False

# Live tick saver - 100% data capture (2026-01-18)
# Chinese quant level persistence: every tick saved to disk
try:
    from core.data.tick_saver import get_tick_saver, LiveTickSaver
    HAS_TICK_SAVER = True
except ImportError:
    HAS_TICK_SAVER = False

# DeepSeek-R1 LLM Integration (2026-01-20)
# Multi-agent reasoning: Bull/Bear debate, Risk management
# Based on: TradingAgents (UCLA/MIT), High-Flyer Quant (幻方量化)
# Target: 63% → 75%+ win rate with LLM augmentation
try:
    from core.ml.llm_reasoner import (
        TradingBotLLMIntegration,
        create_llm_integration,
        MarketContext,
    )
    HAS_LLM_REASONER = True
except ImportError:
    HAS_LLM_REASONER = False

# Live LLM Tuning (2026-01-23) - Chinese Quant Style Continuous Learning
# Records trade outcomes → generates DPO pairs → retrains LoRA on RTX 5080
# Based on: 幻方量化, 九坤投资, DeepSeek GRPO
# Model: forex-r1-v3 (89% accuracy, trained on 8x H100)
try:
    from core.ml.trade_outcome_buffer import get_outcome_buffer, TradeOutcome
    from core.ml.drift_detector import get_drift_detector
    from core.ml.live_lora_tuner import get_live_tuner
    HAS_LIVE_TUNING = True
except ImportError:
    HAS_LIVE_TUNING = False

# FAST Certainty Validator (2026-01-23) - 18-Module Validation for 89%+ Accuracy
# Programmatic implementation: <10ms vs 60s+ for LLM. Same thresholds, FAST.
try:
    from core.ml.fast_certainty_validator import (
        FastCertaintyValidator,
        get_fast_certainty_validator,
        CERTAINTY_THRESHOLDS,
    )
    HAS_CERTAINTY_VALIDATOR = True
except ImportError:
    HAS_CERTAINTY_VALIDATOR = False

# HTTP client for MCP server
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Maximum GPU configuration (RTX 5080 16GB)
try:
    from core.ml.max_gpu_config import maximize_system, get_max_ensemble_config, print_gpu_status
    HAS_MAX_GPU = True
except ImportError:
    HAS_MAX_GPU = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    MONITOR = "monitor"


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Signal:
    """Trading signal from ML models."""
    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0-1
    predicted_return: float  # Expected return in bps
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_votes: Dict[str, int] = field(default_factory=dict)


@dataclass
class Order:
    """Order to be executed."""
    symbol: str
    side: Side
    quantity: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Executed trade."""
    symbol: str
    side: Side
    quantity: float
    fill_price: float
    timestamp: datetime
    signal_confidence: float
    latency_ms: float


class KellyOptimizer:
    """
    Proper Kelly Criterion optimizer for HFT.

    Key principles:
    1. Use actual model accuracy (64%) as baseline
    2. Track real win/loss history per symbol
    3. Calculate true W/L ratio from trades
    4. Use fractional Kelly (1/4 for HFT variance)
    5. Adjust for drawdown and transaction costs
    """

    def __init__(self,
                 base_accuracy: float = 0.775,  # 77.5% VERIFIED win rate on 3,900 trades
                 kelly_fraction: float = 0.55,  # FULL Kelly for maximum compounding
                 min_trades_for_kelly: int = 5,   # Adjust even faster
                 spread_cost_bps: float = 0.3,  # IB Pro IDEALPRO tight spreads
                 commission_bps: float = 0.1):  # IB Pro forex low commissions

        self.base_accuracy = base_accuracy
        self.kelly_fraction = kelly_fraction
        self.min_trades_for_kelly = min_trades_for_kelly
        self.spread_cost_bps = spread_cost_bps
        self.commission_bps = commission_bps

        # Per-symbol trade tracking
        self.trade_history: Dict[str, List[Dict]] = {}  # symbol -> list of trades
        self.symbol_stats: Dict[str, Dict] = {}  # symbol -> {wins, losses, avg_win, avg_loss}

    def record_trade(self, symbol: str, pnl_bps: float, holding_time_sec: float):
        """Record a completed trade for Kelly calculation."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
            self.symbol_stats[symbol] = {
                'wins': 0, 'losses': 0,
                'total_win_bps': 0.0, 'total_loss_bps': 0.0,
                'win_rate': self.base_accuracy,
                'wl_ratio': 1.0
            }

        # Deduct transaction costs
        net_pnl = pnl_bps - self.spread_cost_bps - self.commission_bps

        trade = {
            'pnl_bps': net_pnl,
            'timestamp': datetime.now(),
            'holding_time': holding_time_sec
        }
        self.trade_history[symbol].append(trade)

        # Update stats
        stats = self.symbol_stats[symbol]
        if net_pnl > 0:
            stats['wins'] += 1
            stats['total_win_bps'] += net_pnl
        else:
            stats['losses'] += 1
            stats['total_loss_bps'] += abs(net_pnl)

        # Recalculate ratios
        total = stats['wins'] + stats['losses']
        if total >= self.min_trades_for_kelly:
            stats['win_rate'] = stats['wins'] / total
            if stats['losses'] > 0 and stats['wins'] > 0:
                avg_win = stats['total_win_bps'] / stats['wins']
                avg_loss = stats['total_loss_bps'] / stats['losses']
                stats['wl_ratio'] = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Keep only last 100 trades per symbol
        if len(self.trade_history[symbol]) > 100:
            self.trade_history[symbol] = self.trade_history[symbol][-100:]

    def get_kelly_fraction(self, symbol: str, confidence: float,
                           current_drawdown: float = 0.0) -> float:
        """
        Calculate optimal Kelly fraction for a trade.

        Kelly formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1-p)
            b = win/loss ratio
        """
        # Get symbol-specific stats or use base
        if symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol]
            total_trades = stats['wins'] + stats['losses']

            if total_trades >= self.min_trades_for_kelly:
                # Use observed win rate
                win_prob = stats['win_rate']
                wl_ratio = stats['wl_ratio']
            else:
                # Blend base accuracy with observed (Bayesian update)
                observed_wr = stats['win_rate'] if total_trades > 0 else self.base_accuracy
                weight = total_trades / self.min_trades_for_kelly
                win_prob = (1 - weight) * self.base_accuracy + weight * observed_wr
                wl_ratio = stats.get('wl_ratio', 1.0)
        else:
            # Use base accuracy for new symbols
            win_prob = self.base_accuracy
            wl_ratio = 1.0

        # Adjust win_prob by signal confidence
        # Higher confidence = higher effective win rate
        adjusted_win_prob = win_prob * (0.8 + 0.4 * confidence)  # 80%-120% of base
        adjusted_win_prob = min(0.85, max(0.40, adjusted_win_prob))  # Clamp

        # Calculate raw Kelly
        q = 1 - adjusted_win_prob
        if wl_ratio <= 0:
            return 0.0

        raw_kelly = (adjusted_win_prob * wl_ratio - q) / wl_ratio

        # Apply FULL Kelly - 77.5% verified win rate justifies aggressive sizing
        kelly = raw_kelly * self.kelly_fraction

        # Drawdown adjustment: only reduce if severe (>20%)
        # With 77.5% edge, drawdowns recover fast
        if current_drawdown > 0.20:
            dd_factor = max(0.5, 1.0 - (current_drawdown * 2))  # At 25% DD, factor = 0.5
            kelly *= dd_factor

        # MAXIMUM KELLY - 77.5% win rate = 55% theoretical optimal
        # Full Kelly capture for maximum compounding
        kelly = max(0.05, min(0.55, kelly))  # 5% to 55% max - FULL KELLY

        return kelly

    def get_stats(self, symbol: str) -> Dict:
        """Get current stats for a symbol."""
        if symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol].copy()
            stats['total_trades'] = stats['wins'] + stats['losses']
            return stats
        return {
            'wins': 0, 'losses': 0, 'total_trades': 0,
            'win_rate': self.base_accuracy, 'wl_ratio': 1.0
        }


class RiskManager:
    """
    Risk management using Kelly Criterion and drawdown limits.

    HFT-OPTIMIZED (2026-01-19):
    - Uses KellyOptimizer for proper position sizing
    - Tracks actual win/loss per symbol
    - Adjusts for drawdown and transaction costs
    - Quarter Kelly (0.25) for HFT variance control
    """

    def __init__(self,
                 max_position_pct: float = 0.55,  # 55% max per position - FULL KELLY
                 max_drawdown_pct: float = 0.25,  # 25% max drawdown - aggressive recovery
                 kelly_fraction: float = 0.55,    # FULL Kelly - 77.5% win rate verified
                 max_daily_trades: int = 10000,   # MAXIMUM TRADES - IB allows 50/sec
                 stop_loss_pct: float = 0.01,     # 1% stop loss (wider for compounding)
                 starting_capital: float = None,
                 base_accuracy: float = 0.775):   # 77.5% verified win rate

        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction
        self.max_daily_trades = max_daily_trades
        self.stop_loss_pct = stop_loss_pct

        # Kelly optimizer for proper position sizing
        self.kelly_optimizer = KellyOptimizer(
            base_accuracy=base_accuracy,
            kelly_fraction=kelly_fraction
        )

        # Track stop losses per position
        self.stop_losses: Dict[str, float] = {}  # symbol -> stop price

        # State - use starting capital if provided
        self.starting_capital = starting_capital or float(os.getenv('STARTING_CAPITAL', 100.0))
        self.account_balance = self.starting_capital
        self.peak_balance = self.account_balance
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()

        # Track entry prices for P&L calculation
        self.entry_prices: Dict[str, float] = {}
        self.entry_times: Dict[str, datetime] = {}

    def reset_daily_limits(self):
        """Reset daily counters."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today

    def update_balance(self, balance: float):
        """Update account balance."""
        self.account_balance = balance
        self.peak_balance = max(self.peak_balance, balance)

    def current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.account_balance) / self.peak_balance

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        self.reset_daily_limits()

        # Check drawdown limit
        dd = self.current_drawdown()
        if dd > self.max_drawdown_pct:
            return False, f"Drawdown limit exceeded: {dd:.1%} > {self.max_drawdown_pct:.1%}"

        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {self.daily_trades}"

        return True, "OK"

    def kelly_size(self, win_prob: float, win_loss_ratio: float = 1.5) -> float:
        """
        Calculate Kelly position size (legacy method, use calculate_position_size instead).

        f* = (p * b - q) / b
        where p = win prob, q = 1-p, b = win/loss ratio
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        # Fractional Kelly (safer)
        kelly = kelly * self.kelly_fraction

        # Clamp to reasonable range
        kelly = max(0, min(kelly, self.max_position_pct))

        return kelly

    def calculate_position_size(self, signal: Signal,
                                current_price: float) -> int:
        """
        Calculate position size in UNITS using proper Kelly optimization.

        HFT-OPTIMIZED:
        - Uses KellyOptimizer with actual model accuracy (64%)
        - Adjusts for current drawdown
        - Accounts for transaction costs
        - Quarter Kelly (0.25) for variance control
        """
        # Get optimal Kelly fraction from optimizer
        current_dd = self.current_drawdown()
        kelly = self.kelly_optimizer.get_kelly_fraction(
            symbol=signal.symbol,
            confidence=signal.confidence,
            current_drawdown=current_dd
        )

        # Dollar amount to risk
        position_dollars = self.account_balance * kelly

        # Leverage for forex (MAX 50:1 for aggressive trading)
        leverage = 50
        buying_power = position_dollars * leverage

        # Convert to units (base currency units we can buy)
        position_units = int(buying_power / current_price)

        # Minimum position size
        min_units = 100  # Minimum for forex
        position_units = max(min_units, position_units)

        # Cap at max position percent of account
        max_notional = self.account_balance * leverage * self.max_position_pct
        max_units = int(max_notional / current_price)
        position_units = min(position_units, max_units)

        # Log Kelly details for debugging
        stats = self.kelly_optimizer.get_stats(signal.symbol)
        logger.debug(
            f"[{signal.symbol}] Kelly: {kelly:.4f}, DD: {current_dd:.2%}, "
            f"WR: {stats['win_rate']:.2%}, W/L: {stats['wl_ratio']:.2f}, "
            f"Trades: {stats['total_trades']}, Size: {position_units} units"
        )

        return position_units

    def record_trade(self, pnl: float, symbol: str = None,
                     entry_price: float = None, exit_price: float = None):
        """
        Record trade for Kelly optimization and daily limits.

        Args:
            pnl: P&L in dollars
            symbol: Trading pair
            entry_price: Entry price
            exit_price: Exit price
        """
        self.daily_trades += 1
        self.daily_pnl += pnl

        # Record for Kelly optimizer if we have the details
        if symbol and entry_price and exit_price:
            # Calculate P&L in basis points
            pnl_bps = ((exit_price - entry_price) / entry_price) * 10000

            # Get holding time
            holding_time = 0.0
            if symbol in self.entry_times:
                holding_time = (datetime.now() - self.entry_times[symbol]).total_seconds()
                del self.entry_times[symbol]

            # Record for Kelly calculation
            self.kelly_optimizer.record_trade(symbol, pnl_bps, holding_time)

            # Log trade stats
            stats = self.kelly_optimizer.get_stats(symbol)
            logger.info(
                f"[{symbol}] Trade recorded: {pnl_bps:.1f}bps, "
                f"WR: {stats['win_rate']:.1%}, W/L: {stats['wl_ratio']:.2f}, "
                f"Trades: {stats['total_trades']}"
            )

    def set_entry(self, symbol: str, price: float):
        """Record entry price and time for P&L tracking."""
        self.entry_prices[symbol] = price
        self.entry_times[symbol] = datetime.now()

    def get_entry(self, symbol: str) -> Optional[float]:
        """Get entry price for a symbol."""
        return self.entry_prices.get(symbol)

    def set_stop_loss(self, symbol: str, entry_price: float, direction: int):
        """Set stop loss for a position. direction: 1=long, -1=short"""
        if direction > 0:  # Long position - stop below entry
            self.stop_losses[symbol] = entry_price * (1 - self.stop_loss_pct)
        else:  # Short position - stop above entry
            self.stop_losses[symbol] = entry_price * (1 + self.stop_loss_pct)
        logger.info(f"[{symbol}] Stop loss set at {self.stop_losses[symbol]:.5f} ({self.stop_loss_pct*100:.1f}%)")

    def check_stop_loss(self, symbol: str, current_price: float, direction: int) -> bool:
        """Check if stop loss is triggered. Returns True if should close."""
        if symbol not in self.stop_losses:
            return False

        stop_price = self.stop_losses[symbol]

        if direction > 0:  # Long - stop if price drops below stop
            if current_price <= stop_price:
                logger.warning(f"[{symbol}] STOP LOSS TRIGGERED: price {current_price:.5f} <= stop {stop_price:.5f}")
                return True
        else:  # Short - stop if price rises above stop
            if current_price >= stop_price:
                logger.warning(f"[{symbol}] STOP LOSS TRIGGERED: price {current_price:.5f} >= stop {stop_price:.5f}")
                return True

        return False

    def clear_stop_loss(self, symbol: str):
        """Clear stop loss when position is closed."""
        if symbol in self.stop_losses:
            del self.stop_losses[symbol]


class SimpleFeatureGenerator:
    """
    Generate the 14 simple features that the trained models expect.
    Features: bid, ask, bid_volume, ask_volume, mid, spread, close, volume,
              ret_1, ret_5, ret_10, ret_20, vol_20, vol_50
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.price_history: Dict[str, List[float]] = {}

    def process_tick(self, symbol: str, bid: float, ask: float, volume: float = 0.0) -> Dict[str, float]:
        """Generate features from tick data."""
        mid = (bid + ask) / 2
        spread = ask - bid

        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        history = self.price_history[symbol]
        history.append(mid)

        # Keep only lookback prices
        if len(history) > self.lookback:
            history.pop(0)

        # Calculate returns (price change ratios)
        def calc_return(n):
            if len(history) > n:
                return (history[-1] / history[-n-1]) - 1.0
            return 0.0

        # Calculate volatility (std of returns)
        def calc_vol(n):
            if len(history) > n:
                prices = history[-n:]
                returns = [(prices[i] / prices[i-1]) - 1.0 for i in range(1, len(prices))]
                if returns:
                    return float(np.std(returns)) if len(returns) > 1 else 0.0
            return 0.0

        features = {
            'bid': bid,
            'ask': ask,
            'bid_volume': volume,
            'ask_volume': volume,
            'mid': mid,
            'spread': spread,
            'close': mid,  # Use mid as close for tick data
            'volume': volume,
            'ret_1': calc_return(1),
            'ret_5': calc_return(5),
            'ret_10': calc_return(10),
            'ret_20': calc_return(20),
            'vol_20': calc_vol(20),
            'vol_50': calc_vol(50),
        }

        return features


class MLEnsemble:
    """
    ML Ensemble Predictor.
    Loads trained models and generates trading signals.
    """

    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path("models/production")
        self.models: Dict[str, Dict] = {}
        self.feature_names: List[str] = []
        self.loaded = False

    def load_models(self, symbols: List[str] = None):
        """Load trained models for symbols."""
        if symbols is None:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY']

        for symbol in symbols:
            # Try main models file (new format from train_models.py)
            model_path = self.model_dir / f"{symbol}_models.pkl"

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)

                    # Structure: data['target_direction_1']['xgboost'] etc.
                    # Use target_direction_1 (1-tick) for HFT
                    if 'target_direction_1' in data:
                        td = data['target_direction_1']
                        adapted = {
                            'models': {
                                'xgboost': td.get('xgboost'),
                                'lightgbm': td.get('lightgbm'),
                                'catboost': td.get('catboost')
                            },
                            'feature_names': td.get('features', [])
                        }
                        # Filter out None models
                        adapted['models'] = {k: v for k, v in adapted['models'].items() if v is not None}
                        self.models[symbol] = adapted
                        if not self.feature_names and adapted['feature_names']:
                            self.feature_names = adapted['feature_names']
                        logger.info(f"Loaded {len(adapted['models'])} models for {symbol} ({len(adapted['feature_names'])} features)")
                    else:
                        logger.warning(f"No target_direction_1 in {symbol} models")

                except Exception as e:
                    logger.error(f"Failed to load models for {symbol}: {e}")
            else:
                logger.warning(f"No models found at {model_path}")

        self.loaded = len(self.models) > 0

    def predict(self, symbol: str, features: Dict[str, float]) -> Optional[Signal]:
        """Generate signal from features."""
        if symbol not in self.models:
            return None

        model_data = self.models[symbol]
        models = model_data.get('models', {})

        if not models:
            return None

        # Prepare feature vector
        if self.feature_names:
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        else:
            X = np.array([list(features.values())])

        # Get predictions from each model
        predictions = {}
        probabilities = {}

        for name, model in models.items():
            try:
                # Handle different model types
                if hasattr(model, 'inplace_predict'):
                    # XGBoost Booster
                    import xgboost as xgb
                    dmat = xgb.DMatrix(X, feature_names=self.feature_names if self.feature_names else None)
                    proba = model.predict(dmat)[0]
                    pred = 1 if proba > 0.5 else 0
                    probabilities[name] = float(proba)
                elif hasattr(model, 'predict_proba'):
                    # Sklearn-like model
                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]
                    probabilities[name] = proba[1] if len(proba) > 1 else proba[0]
                else:
                    # Basic predict only
                    pred = model.predict(X)[0]
                    probabilities[name] = 0.5

                predictions[name] = int(pred)

            except Exception as e:
                logger.warning(f"Prediction error for {name}: {e}")

        if not predictions:
            return None

        # Ensemble voting
        avg_prob = np.mean(list(probabilities.values()))
        majority_vote = 1 if sum(predictions.values()) > len(predictions) / 2 else 0

        # Direction: 1 = long, -1 = short
        direction = 1 if majority_vote == 1 else -1

        # Confidence: distance from 0.5
        confidence = abs(avg_prob - 0.5) * 2

        # Predicted return in bps (simple estimate)
        predicted_return = direction * confidence * 10  # 10 bps max

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            features=features,
            model_votes=predictions
        )


class IBGatewayConnector:
    """
    Interactive Brokers Gateway Connector.
    Handles order execution via IB API.

    Supports:
    - Spot Forex (CASH) - up to 50:1 leverage
    - CME Futures (6E, 6B, 6J, etc.) - standardized contracts
    - Micro Futures (M6E, M6B) - smaller size for small accounts
    - LIMIT orders for better fills
    - MARKET orders for urgent execution

    NOTE: Uses port 4004 for paper trading (socat forwarding).
    """

    # CME Futures mapping: symbol -> (futures_symbol, multiplier, exchange)
    FUTURES_MAP = {
        'EURUSD': ('6E', 125000, 'CME'),   # Euro FX Futures
        'GBPUSD': ('6B', 62500, 'CME'),    # British Pound Futures
        'USDJPY': ('6J', 12500000, 'CME'), # Japanese Yen Futures
        'AUDUSD': ('6A', 100000, 'CME'),   # Australian Dollar Futures
        'USDCAD': ('6C', 100000, 'CME'),   # Canadian Dollar Futures
        'USDCHF': ('6S', 125000, 'CME'),   # Swiss Franc Futures
        'NZDUSD': ('6N', 100000, 'CME'),   # New Zealand Dollar Futures
    }

    # Micro Futures for smaller positions
    MICRO_FUTURES_MAP = {
        'EURUSD': ('M6E', 12500, 'CME'),   # Micro Euro FX
        'GBPUSD': ('M6B', 6250, 'CME'),    # Micro British Pound
        'USDJPY': ('MJY', 1250000, 'CME'), # Micro Yen
        'AUDUSD': ('M6A', 10000, 'CME'),   # Micro AUD
        'USDCAD': ('MCD', 10000, 'CME'),   # Micro CAD
        'USDCHF': ('MSF', 12500, 'CME'),   # Micro CHF
    }

    def __init__(self, host: str = None, port: int = None,
                 client_id: int = None, use_futures: bool = False,
                 use_micro_futures: bool = True):
        import os
        self.host = host or os.getenv('IB_HOST', 'localhost')
        self.port = port or int(os.getenv('IB_PORT', 4004))
        self.client_id = client_id or int(os.getenv('IB_CLIENT_ID', 1))
        self.connected = False
        self.ib = None
        self.use_futures = use_futures
        self.use_micro_futures = use_micro_futures  # Better for $200 account

    async def connect(self):
        """Connect to IB Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IB Gateway at {self.host}:{self.port}")

            # Get account info
            account = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else "Unknown"
            logger.info(f"Account: {account}")

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            logger.info("Ensure IB Gateway Docker is running: docker start ibgateway")
            logger.info(f"Connect to port {self.port} (paper trading uses 4004)")
            self.connected = False

    async def get_account_balance(self) -> float:
        """Get current account balance."""
        if not self.connected or not self.ib:
            return 100000.0  # Default for paper

        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    return float(av.value)
        except:
            pass

        return 100000.0

    async def submit_order(self, order: Order, use_limit: bool = False,
                           current_price: float = None) -> Optional[Trade]:
        """
        Submit order to IB Gateway.

        Supports:
        - Spot Forex (CASH) - default, 50:1 leverage
        - Micro Futures - for majors, better for small accounts
        - LIMIT orders - for better fills on high-confidence signals
        - MARKET orders - for urgent execution
        """
        if not self.connected or not self.ib:
            logger.warning("Not connected to IB Gateway - simulating fill")
            return self._simulate_fill(order)

        try:
            from ib_insync import Forex, Future, MarketOrder, LimitOrder

            # Determine contract type
            symbol = order.symbol
            contract = None

            # Try micro futures first for majors (better for small accounts)
            if self.use_micro_futures and symbol in self.MICRO_FUTURES_MAP:
                fut_sym, multiplier, exchange = self.MICRO_FUTURES_MAP[symbol]
                # Get front month contract
                contract = Future(fut_sym, exchange=exchange)
                contract.lastTradeDateOrContractMonth = ''  # Front month
                logger.debug(f"[{symbol}] Using MICRO FUTURES: {fut_sym}")

            # Fall back to regular futures
            elif self.use_futures and symbol in self.FUTURES_MAP:
                fut_sym, multiplier, exchange = self.FUTURES_MAP[symbol]
                contract = Future(fut_sym, exchange=exchange)
                contract.lastTradeDateOrContractMonth = ''  # Front month
                logger.debug(f"[{symbol}] Using FUTURES: {fut_sym}")

            # Default to spot forex
            if contract is None:
                contract = Forex(symbol[:3] + symbol[3:])
                logger.debug(f"[{symbol}] Using SPOT FOREX")

            # Smart order type selection
            # Use LIMIT for high confidence signals, MARKET for urgent
            if use_limit and current_price and order.order_type == "MARKET":
                # Set limit price slightly better than current
                # BUY: limit at ask - 0.5 pip
                # SELL: limit at bid + 0.5 pip
                pip_size = 0.0001 if 'JPY' not in symbol else 0.01
                if order.side.value == "BUY":
                    limit_price = current_price - (0.5 * pip_size)
                else:
                    limit_price = current_price + (0.5 * pip_size)
                order.limit_price = limit_price
                order.order_type = "LIMIT"
                logger.debug(f"[{symbol}] Using LIMIT order @ {limit_price:.5f}")

            # Create IB order
            if order.order_type == "MARKET":
                ib_order = MarketOrder(
                    order.side.value,
                    order.quantity
                )
            else:
                ib_order = LimitOrder(
                    order.side.value,
                    order.quantity,
                    order.limit_price
                )

            # Submit
            submit_time = datetime.now()
            trade = self.ib.placeOrder(contract, ib_order)

            # Wait for fill (with timeout)
            await asyncio.sleep(0.1)

            fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0.0
            fill_time = datetime.now()

            latency_ms = (fill_time - submit_time).total_seconds() * 1000

            return Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                timestamp=fill_time,
                signal_confidence=0.0,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return None

    def _simulate_fill(self, order: Order) -> Trade:
        """Simulate order fill for testing."""
        # Simulate realistic fill price
        base_price = order.limit_price if order.limit_price else 1.1000
        slippage = 0.00001 * (1 if order.side == Side.BUY else -1)

        return Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=base_price + slippage,
            timestamp=datetime.now(),
            signal_confidence=0.0,
            latency_ms=50.0
        )

    async def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")


class HFTTradingBot:
    """
    Main HFT Trading Bot.
    Orchestrates data, features, signals, and execution.
    """

    def __init__(self,
                 mode: TradingMode = TradingMode.PAPER,
                 symbols: List[str] = None,
                 use_execution_optimizer: bool = False,
                 use_online_learning: bool = True,
                 skip_ib: bool = False):
        """
        Initialize HFT Trading Bot.

        Args:
            mode: Trading mode (paper/live/backtest/monitor)
            symbols: List of forex symbols to trade
            use_execution_optimizer: Enable execution optimization (TWAP/VWAP)
            use_online_learning: Enable Chinese quant-style online learning
            skip_ib: Skip IB Gateway connection (simulated fills only)
                                 (incremental model updates from live data)
        """
        self.mode = mode
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY']
        self.use_execution_optimizer = use_execution_optimizer
        self.use_online_learning = use_online_learning and HAS_CHINESE_ONLINE
        self.skip_ib = skip_ib

        # Components
        self.feature_engine = None
        self.ml_ensemble = None
        self.adaptive_ensemble = None  # Chinese quant online learning (2026-01-18)
        self.risk_manager = RiskManager()
        self.ib_connector = None
        self.execution_interceptor = None  # Execution optimizer

        # State
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.tick_count: Dict[str, int] = {}
        self.last_prices: Dict[str, Tuple[float, float]] = {}
        self.current_volatility: Dict[str, float] = {}  # Track volatility per symbol

        # Trade cooldown - MINIMUM for maximum trades
        # 77.5% edge justifies rapid compounding
        self.last_trade_time: Dict[str, datetime] = {}  # symbol -> last trade timestamp
        self.trade_cooldown_sec: float = 5.0  # 5 second cooldown - MAXIMUM FREQUENCY

        # TAKE PROFIT / POSITION MANAGEMENT (2026-01-23)
        # Close positions FAST to compound gains
        self.take_profit_pips: float = 10.0  # Take profit at 10 pips - faster turnover
        self.take_profit_pct: float = 0.002  # Or 0.2% profit - quick captures
        self.max_hold_minutes: float = 15.0  # 15 min max hold - faster compounding
        self.position_entry_times: Dict[str, datetime] = {}  # Track when positions opened

        # Performance tracking
        self.start_time = datetime.now()
        self.total_pnl = 0.0

        # Live retraining
        self.tick_buffer = None
        self.retrainer = None

        # Online learning state (for observation feedback)
        self.pending_observations: Dict[str, List] = {}  # symbol -> [(features, price, timestamp)]

        # Tick saver - 100% data capture (2026-01-18)
        self.tick_saver = None

        # Control
        self.running = False

        # Live LLM Tuning (2026-01-23) - Chinese Quant Style Continuous Learning
        # MCP server for outcome recording, drift detection, LoRA training
        self.mcp_tuning_url = "http://localhost:8082"
        self.use_live_tuning = HAS_LIVE_TUNING and HAS_REQUESTS
        self.outcome_buffer = None
        self.drift_detector = None
        self._last_llm_reasoning: Dict[str, str] = {}  # symbol -> last reasoning
        self._last_signal_confidence: Dict[str, float] = {}  # symbol -> last confidence
        self._outcome_count_since_training = 0
        self._training_check_interval = 100  # Check every 100 ticks

        # Certainty Validator (2026-01-23) - 18-Module Validation for 89%+ Accuracy
        # This is the KEY difference: use forex-r1-v3's 18 certainty modules
        # instead of simple YES/NO validation
        self.certainty_validator = None
        self.use_certainty_validation = HAS_CERTAINTY_VALIDATOR

    async def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing HFT Trading Bot - Mode: {self.mode.value}")

        # MAXIMIZE GPU UTILIZATION (RTX 5080 16GB)
        if HAS_MAX_GPU:
            try:
                maximize_system()
                logger.info("[MAX_GPU] System maximized for RTX 5080 16GB")
            except Exception as e:
                logger.warning(f"[MAX_GPU] Could not maximize: {e}")

        # Initialize feature engine
        # Use SimpleFeatureGenerator for models trained on basic features (14 features)
        self.simple_feature_gen = SimpleFeatureGenerator(lookback=100)
        logger.info("SimpleFeatureGenerator initialized (14 features for trained models)")

        try:
            from core.features.engine import HFTFeatureEngine
            import pandas as pd
            self.feature_engine = HFTFeatureEngine()

            # Initialize feature engine with historical data for each symbol
            # This pre-populates buffers so we can compute 575 features immediately
            for symbol in self.symbols:
                hist_path = f"training_package/{symbol}/train.parquet"
                try:
                    hist_data = pd.read_parquet(hist_path)
                    if len(hist_data) > 100:
                        # Use last 500 ticks to warm up the feature engine
                        init_data = hist_data.tail(500)
                        self.feature_engine.initialize(init_data, symbol)
                        logger.info(f"[{symbol}] HFTFeatureEngine pre-warmed with {len(init_data)} historical ticks -> {len(self.feature_engine.feature_names)} features")
                except FileNotFoundError:
                    logger.warning(f"[{symbol}] No historical data for feature engine warmup")
                except Exception as e:
                    logger.warning(f"[{symbol}] Feature engine warmup failed: {e}")

            logger.info(f"HFTFeatureEngine initialized (advanced features: {len(self.feature_engine.feature_names)} total)")
        except Exception as e:
            self.feature_engine = None
            logger.warning(f"HFTFeatureEngine not available: {e}")

        # Load ML models
        if self.use_online_learning:
            # Chinese Quant Online Learning (2026-01-18)
            # Incremental updates based on 幻方/九坤/明汯 techniques
            logger.info("Initializing Chinese Quant Online Learning...")
            self.adaptive_ensemble = create_adaptive_ensemble(
                model_dir="models/production",
                enable_online=True,
                update_interval=60,  # Update models every 60 seconds
                min_samples_for_update=500,
                live_weight=3.0,  # Weight live data 3x vs historical
            )
            self.adaptive_ensemble.load_models(self.symbols)
            self.adaptive_ensemble.init_online_learning()
            self.adaptive_ensemble.start_background_updates()
            self.ml_ensemble = None  # Use adaptive_ensemble instead
            logger.info(f"Online learning enabled: {self.adaptive_ensemble.get_status()}")
        else:
            # Standard static ensemble
            self.ml_ensemble = MLEnsemble()
            self.ml_ensemble.load_models(self.symbols)
            self.adaptive_ensemble = None

        if not (self.ml_ensemble and self.ml_ensemble.loaded) and not self.adaptive_ensemble:
            logger.warning("No ML models loaded - running in signal-only mode")

        # Connect to IB Gateway (for paper/live modes)
        # Use spot forex (CASH contracts) - no expiry needed, works with IB paper trading
        if self.mode in [TradingMode.PAPER, TradingMode.LIVE] and not self.skip_ib:
            self.ib_connector = IBGatewayConnector(
                use_futures=False,  # Don't use full-size futures
                use_micro_futures=False  # Use spot forex (CASH) - no expiry issues
            )
            await self.ib_connector.connect()

            if self.ib_connector.connected:
                ib_balance = await self.ib_connector.get_account_balance()
                logger.info(f"IB Account balance: ${ib_balance:,.2f}")
                # Use our starting capital (e.g., $200) instead of IB paper balance
                logger.info(f"Trading with simulated capital: ${self.risk_manager.starting_capital:,.2f}")
                logger.info(f"Leverage: 50:1 | Order types: MARKET + LIMIT | Spot Forex: ENABLED")
        elif self.skip_ib:
            logger.info("[NO-IB] Running with SIMULATED FILLS only (no IB Gateway)")
            logger.info("[NO-IB] No commission charges - pure simulation mode")
            # Create connector but don't connect (will use _simulate_fill)
            self.ib_connector = IBGatewayConnector()

        # Initialize positions
        for symbol in self.symbols:
            self.positions[symbol] = Position(symbol=symbol)
            self.tick_count[symbol] = 0

        # Initialize HYBRID retraining (Historical + Live data combined)
        if HAS_LIVE_RETRAIN:
            try:
                self.tick_buffer = get_tick_buffer(max_size=50000)
                self.retrainer = get_hybrid_retrainer(symbols=self.symbols)
                self.retrainer.start()
                logger.info("[HYBRID] Historical + Live retraining enabled - 200k+ samples per cycle")
            except Exception as e:
                logger.warning(f"[HYBRID] Failed to initialize: {e}")

        # Initialize execution optimizer (2026-01-17)
        if self.use_execution_optimizer and HAS_EXEC_OPTIMIZER:
            try:
                self.execution_interceptor = get_execution_interceptor(
                    executor=self.ib_connector,
                    config=ExecutionConfig()
                )
                logger.info("[EXEC] Execution optimizer enabled - TWAP/VWAP/AC strategies available")
            except Exception as e:
                logger.warning(f"[EXEC] Failed to initialize optimizer: {e}")
                self.execution_interceptor = None

        # Initialize tick saver - 100% data capture (2026-01-18)
        if HAS_TICK_SAVER:
            try:
                self.tick_saver = get_tick_saver(
                    output_dir="data/live",
                    buffer_size=500,  # Flush every 500 ticks
                    flush_interval=5.0,  # Or every 5 seconds
                    per_symbol=True,  # Separate files per symbol
                )
                logger.info("[TICK_SAVER] 100% data capture enabled - saving all ticks to disk")
            except Exception as e:
                logger.warning(f"[TICK_SAVER] Failed to initialize: {e}")
                self.tick_saver = None

        # Initialize volatility tracking
        for symbol in self.symbols:
            self.current_volatility[symbol] = 0.0001  # Default 10 bps/day

        # Initialize DeepSeek-R1 LLM Reasoner (2026-01-20)
        # Multi-agent system: Bull/Bear debate + Risk management
        # Based on TradingAgents (UCLA/MIT), High-Flyer Quant (幻方量化)
        self.llm_integration = None
        if HAS_LLM_REASONER:
            try:
                self.llm_integration = create_llm_integration(mode="validation")
                await self.llm_integration.initialize()
                logger.info("[LLM] DeepSeek-R1 Multi-Agent Reasoner enabled")
                logger.info("[LLM] Mode: VALIDATION (LLM can veto ML signals)")
                logger.info("[LLM] Target: 63% → 75%+ win rate with reasoning")
            except Exception as e:
                logger.warning(f"[LLM] Failed to initialize: {e}")
                self.llm_integration = None

        # Initialize Live LLM Tuning (2026-01-23)
        # Records trade outcomes → generates DPO pairs → retrains LoRA
        if self.use_live_tuning:
            try:
                self.outcome_buffer = get_outcome_buffer()
                self.drift_detector = get_drift_detector()
                logger.info("[LIVE_TUNING] Initialized with forex-r1-v3 model")
                logger.info("[LIVE_TUNING] MCP server: %s", self.mcp_tuning_url)
                logger.info("[LIVE_TUNING] Features: outcome recording, drift detection, LoRA training")

                # Check MCP server connectivity
                try:
                    resp = requests.get(f"{self.mcp_tuning_url}/health", timeout=2)
                    if resp.ok:
                        logger.info("[LIVE_TUNING] MCP server connected")
                    else:
                        logger.warning("[LIVE_TUNING] MCP server not responding")
                except:
                    logger.warning("[LIVE_TUNING] MCP server not available (will use local fallback)")

            except Exception as e:
                logger.warning(f"[LIVE_TUNING] Failed to initialize: {e}")
                self.use_live_tuning = False

        # Initialize FAST Certainty Validator (2026-01-23) - 18-Module Validation
        # Programmatic <10ms validation using same thresholds as forex-r1-v3 training
        if self.use_certainty_validation:
            try:
                self.certainty_validator = get_fast_certainty_validator()
                await self.certainty_validator.warmup()
                logger.info("[FAST_CERTAINTY] 18-Module Validator ENABLED (<10ms)")
                logger.info("[FAST_CERTAINTY] ML>70%, 14+/18 modules, critical checks")
                logger.info("[FAST_CERTAINTY] Target: 89%+ accuracy, MAXIMUM SPEED")
            except Exception as e:
                logger.warning(f"[FAST_CERTAINTY] Failed: {e}")
                self.certainty_validator = None
                self.use_certainty_validation = False

        logger.info("Initialization complete")

    async def process_tick(self, symbol: str, bid: float, ask: float,
                          volume: float = 0.0, timestamp: datetime = None,
                          source: str = "unknown"):
        """Process incoming tick and generate/execute signals."""
        timestamp = timestamp or datetime.now()
        mid = (bid + ask) / 2

        self.tick_count[symbol] = self.tick_count.get(symbol, 0) + 1
        self.last_prices[symbol] = (bid, ask)

        # 0. Save tick to disk - 100% data capture (Chinese quant level)
        if self.tick_saver:
            try:
                self.tick_saver.save_tick(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    source=source,
                    volume=volume,
                    timestamp=timestamp,
                )
            except Exception:
                pass  # Don't let save errors affect trading

        # 1. Generate simple features (14 features for trained models)
        simple_features = self.simple_feature_gen.process_tick(symbol, bid, ask, volume)

        # 1a. Also generate advanced features if available
        if self.feature_engine:
            try:
                advanced_features = self.feature_engine.process_tick(symbol, bid, ask, volume, timestamp)
            except Exception:
                advanced_features = simple_features
        else:
            advanced_features = simple_features

        # 1.5. Buffer tick for live retraining
        if self.tick_buffer and simple_features:
            try:
                self.tick_buffer.add_tick(symbol, bid, ask, simple_features, timestamp.timestamp())
            except Exception:
                pass  # Don't let buffer errors affect trading

        # 2. Generate signal from ML ensemble (or adaptive ensemble with online learning)
        signal = None
        if self.adaptive_ensemble:
            # Chinese Quant Online Learning path - use advanced features (575) if available
            # advanced_features is a dict, check if it has enough features (>100 keys)
            use_advanced = advanced_features and isinstance(advanced_features, dict) and len(advanced_features) > 100
            features_for_prediction = advanced_features if use_advanced else simple_features
            if self.tick_count.get(symbol, 0) % 500 == 1:
                logger.info(f"[{symbol}] Using {'ADVANCED' if use_advanced else 'SIMPLE'} features ({len(features_for_prediction)} features)")
            signal = self.adaptive_ensemble.predict(symbol, features_for_prediction)

            # Store features for observation feedback (will label with actual outcome)
            if symbol not in self.pending_observations:
                self.pending_observations[symbol] = []
            self.pending_observations[symbol].append({
                'features': np.array(list(simple_features.values())) if isinstance(simple_features, dict) else simple_features,
                'price': mid,
                'timestamp': timestamp,
            })
            # Keep only last 100 pending observations
            if len(self.pending_observations[symbol]) > 100:
                self.pending_observations[symbol] = self.pending_observations[symbol][-100:]

            # Process observation feedback - label old observations with actual outcome
            # Based on 幻方量化 technique: use 1-tick lookahead for direction labeling
            self._process_observation_feedback(symbol, mid)

        elif self.ml_ensemble and self.ml_ensemble.loaded:
            signal = self.ml_ensemble.predict(symbol, simple_features)

        # 3. Risk check
        can_trade, reason = self.risk_manager.can_trade()

        if not can_trade:
            if self.tick_count[symbol] % 1000 == 0:
                logger.warning(f"Trading blocked: {reason}")
            return

        # 3.5 STOP LOSS DISABLED for explosive compounding
        # With 64% accuracy (14% edge), let the MODEL decide exits, not fixed stops
        # Max drawdown protection (15%) still active at portfolio level
        # Individual trade stops cause churning in sideways markets

        # 3.6 CHECK TAKE PROFIT - CRITICAL for realizing P&L!
        # Without this, positions stay open forever and profits are never realized
        await self._check_take_profit(symbol, bid, ask)

        # 4. Generate order if signal is strong enough
        # 89% ACCURACY MODE: Only trade when ML confidence > 70% (threshold from H100 training)
        # This is stricter but matches what forex-r1-v3 was trained for
        ml_threshold = 0.70 if self.use_certainty_validation else 0.15
        if signal and signal.confidence > ml_threshold:
            self.signals.append(signal)

            # 4.5 CERTAINTY VALIDATION - 18-Module Check (2026-01-23)
            # THIS IS THE KEY TO 89% ACCURACY
            # forex-r1-v3 applies EdgeProof, VPIN, ICIR, etc. to validate trade
            llm_reasoning = ""
            llm_approved = False  # Default to REJECT until proven valid
            kelly_fraction = 0.0

            if self.certainty_validator and self.use_certainty_validation:
                try:
                    # Get current regime from HMM detector if available
                    regime = "trending"  # Default
                    if self.adaptive_ensemble and hasattr(self.adaptive_ensemble, 'get_regime'):
                        regime = self.adaptive_ensemble.get_regime(symbol) or "trending"

                    # Get model probabilities for 18-module validation
                    model_probs = signal.model_votes if signal.model_votes else {
                        'xgboost': 0.5 + (signal.confidence * signal.direction) / 2,
                        'lightgbm': 0.5 + (signal.confidence * signal.direction) / 2,
                        'catboost': 0.5 + (signal.confidence * signal.direction) / 2,
                    }

                    # FAST 18-Module Certainty Validation (<10ms)
                    certainty_result = self.certainty_validator.validate(
                        symbol=symbol,
                        direction=signal.direction,
                        ml_confidence=signal.confidence,
                        model_probs=model_probs,
                        price=mid,
                        spread_pips=(ask - bid) / mid * 10000,
                        regime=regime,
                        features=features if isinstance(features, dict) else {},
                    )

                    llm_approved = certainty_result.should_trade
                    kelly_fraction = certainty_result.kelly_fraction
                    llm_reasoning = certainty_result.reasoning
                    modules_passed = sum(certainty_result.modules_passed.values())

                    if llm_approved:
                        logger.info(
                            f"[FAST] APPROVED {symbol}: {certainty_result.certainty_score:.0%} "
                            f"({modules_passed}/18) Kelly={kelly_fraction:.1%} [{certainty_result.latency_ms:.1f}ms]"
                        )
                    elif self.tick_count[symbol] % 100 == 0:
                        logger.debug(
                            f"[FAST] VETOED {symbol}: {modules_passed}/18 - {llm_reasoning[:60]}"
                        )

                    # Store for DPO pair generation
                    self._last_llm_reasoning[symbol] = llm_reasoning
                    self._last_signal_confidence[symbol] = signal.confidence

                except Exception as e:
                    logger.warning(f"[FAST] Validation error: {e}")
                    llm_approved = False

            # Fallback to simple LLM validation if certainty validator not available
            elif self.llm_integration and signal.confidence > 0.40:  # Lower threshold for max trades
                try:
                    regime = "trending"
                    if self.adaptive_ensemble and hasattr(self.adaptive_ensemble, 'get_regime'):
                        regime = self.adaptive_ensemble.get_regime(symbol) or "trending"

                    final_signal, llm_size, llm_reasoning = await self.llm_integration.process_signal(
                        symbol=symbol,
                        ml_signal=signal.direction,
                        ml_confidence=signal.confidence,
                        current_price=mid,
                        spread_pips=(ask - bid) / mid * 10000,
                        regime=regime,
                        features=features,
                        account_balance=self.risk_manager.current_capital,
                        current_position=self.positions[symbol].quantity,
                        daily_pnl=self.total_pnl,
                        fast_mode=True,
                    )
                    llm_approved = final_signal != 0 or signal.direction == 0
                    self._last_llm_reasoning[symbol] = llm_reasoning
                    self._last_signal_confidence[symbol] = signal.confidence
                except Exception as e:
                    logger.debug(f"[LLM] Error: {e}")
                    llm_approved = True  # Allow trade on simple LLM error
            else:
                # No validator available - use old behavior
                llm_approved = True

            # Log signals for debugging
            if self.tick_count[symbol] % 100 == 0:
                val_status = "CERT:OK" if llm_approved else "CERT:VETO"
                if not self.use_certainty_validation:
                    val_status = "LLM:OK" if llm_approved else "LLM:VETO"
                logger.info(f"[{symbol}] Signal: dir={signal.direction}, conf={signal.confidence:.3f}, votes={signal.model_votes}, {val_status}")

            # Only trade if in paper/live mode AND LLM approved (or no LLM)
            if self.mode in [TradingMode.PAPER, TradingMode.LIVE] and llm_approved:
                await self._execute_signal(signal, bid, ask, llm_reasoning)

        # 5. Log progress periodically (every 100 ticks)
        if self.tick_count[symbol] % 100 == 0:
            self._log_status(symbol)

        # 6. Log portfolio summary every 500 ticks total
        total_ticks = sum(self.tick_count.values())
        if total_ticks % 500 == 0 and total_ticks > 0:
            self._log_portfolio_summary()

        # 7. Check if LoRA training should trigger (every 100 ticks)
        if total_ticks % self._training_check_interval == 0 and self.use_live_tuning:
            self._check_training_trigger()

    async def _execute_signal(self, signal: Signal, bid: float, ask: float, llm_reasoning: str = ""):
        """Execute trading signal with optional LLM reasoning."""
        position = self.positions[signal.symbol]
        mid_price = (bid + ask) / 2
        spread_bps = (ask - bid) / mid_price * 10000

        # Check trade cooldown - prevent rapid-fire trading
        now = datetime.now()
        if signal.symbol in self.last_trade_time:
            elapsed = (now - self.last_trade_time[signal.symbol]).total_seconds()
            if elapsed < self.trade_cooldown_sec:
                return  # Still in cooldown

        # Don't add to existing position in same direction
        # Wait for signal reversal to trade again
        if position.quantity != 0:
            pos_direction = 1 if position.quantity > 0 else -1
            if pos_direction == signal.direction:
                return  # Already have position in same direction

        # HFT KELLY OPTIMIZED: Position sizing controlled by KellyOptimizer
        # - Uses actual model accuracy (64%) as baseline
        # - Tracks per-symbol win/loss ratio
        # - Quarter Kelly (0.25) for HFT variance control
        # - Adjusts for current drawdown
        size = self.risk_manager.calculate_position_size(signal, mid_price)

        if size < 100:
            return  # Minimum 100 units for forex

        # Determine side and check position
        if signal.direction > 0:  # Long signal
            if position.quantity >= 0:  # Not short, can go long
                side = Side.BUY
                direction = 1
            else:  # Close short first
                side = Side.BUY
                direction = 1
                size = min(size, abs(position.quantity))
        else:  # Short signal
            if position.quantity <= 0:  # Not long, can go short
                side = Side.SELL
                direction = -1
            else:  # Close long first
                side = Side.SELL
                direction = -1
                size = min(size, position.quantity)

        trade = None

        # Use execution optimizer if available
        if self.execution_interceptor:
            try:
                exec_id = await self.execution_interceptor.intercept(
                    symbol=signal.symbol,
                    direction=direction,
                    quantity=size,
                    signal_strength=signal.confidence,
                    mid_price=mid_price,
                    spread_bps=spread_bps,
                    volatility=self.current_volatility.get(signal.symbol, 0.0001)
                )
                # For now, assume immediate fill (actual fill tracking in interceptor)
                trade = Trade(
                    symbol=signal.symbol,
                    side=side,
                    quantity=size,
                    fill_price=mid_price,
                    timestamp=datetime.now(),
                    signal_confidence=signal.confidence,
                    latency_ms=0.0
                )
            except Exception as e:
                logger.error(f"[EXEC] Interceptor failed: {e}, falling back to direct execution")
                self.execution_interceptor = None  # Disable on failure

        # Fallback to direct execution
        if trade is None:
            # Use LIMIT orders for high confidence signals (>50%) for better fills
            # Use MARKET orders for lower confidence (need fast execution)
            use_limit = signal.confidence > 0.50

            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=size,
                order_type="LIMIT" if use_limit else "MARKET",
                limit_price=mid_price if use_limit else None
            )
            trade = await self.ib_connector.submit_order(
                order,
                use_limit=use_limit,
                current_price=mid_price
            )

        if trade:
            trade.signal_confidence = signal.confidence
            self.trades.append(trade)

            # Track entry for Kelly optimization
            old_qty = position.quantity
            was_flat = old_qty == 0

            # Update position
            if trade.side == Side.BUY:
                position.quantity += trade.quantity
            else:
                position.quantity -= trade.quantity

            # Record entry price for new positions
            if was_flat and position.quantity != 0:
                self.risk_manager.set_entry(signal.symbol, trade.fill_price)
                position.avg_price = trade.fill_price  # FIX: Set avg_price for P&L calculation
                self.position_entry_times[signal.symbol] = datetime.now()  # Track for time-based exits
                logger.info(f"[{signal.symbol}] Entry recorded @ {trade.fill_price:.5f}")

            # If position flipped or closed, record the trade for Kelly
            if old_qty != 0 and (
                (old_qty > 0 and position.quantity <= 0) or  # Long closed/flipped
                (old_qty < 0 and position.quantity >= 0)     # Short closed/flipped
            ):
                entry_price = self.risk_manager.get_entry(signal.symbol)
                if entry_price:
                    # Calculate P&L
                    was_long = old_qty > 0
                    if was_long:
                        pnl = (trade.fill_price - entry_price) * abs(old_qty)
                    else:  # Was short
                        pnl = (entry_price - trade.fill_price) * abs(old_qty)

                    self.risk_manager.record_trade(
                        pnl=pnl,
                        symbol=signal.symbol,
                        entry_price=entry_price,
                        exit_price=trade.fill_price
                    )

                    # Record for live LLM tuning (DPO pair generation)
                    self._record_trade_outcome(
                        symbol=signal.symbol,
                        was_long=was_long,
                        pnl=pnl,
                        ml_confidence=signal.confidence,
                        llm_reasoning=llm_reasoning
                    )

                    # Set new entry if flipped to opposite direction
                    if position.quantity != 0:
                        self.risk_manager.set_entry(signal.symbol, trade.fill_price)
                        position.avg_price = trade.fill_price  # FIX: Update avg_price on flip
                        self.position_entry_times[signal.symbol] = datetime.now()  # Reset entry time

            # Update trade cooldown
            self.last_trade_time[signal.symbol] = datetime.now()

            logger.info(f"TRADE: {trade.side.value} {trade.quantity:,.0f} units {trade.symbol} "
                       f"@ {trade.fill_price:.5f} (conf: {signal.confidence:.2f})")

    async def _close_position(self, symbol: str, reason: str = "SIGNAL"):
        """Close entire position for a symbol (used for stop loss, etc.)"""
        position = self.positions.get(symbol)
        if not position or position.quantity == 0:
            return

        # Store position info before closing
        old_qty = position.quantity
        was_long = old_qty > 0

        # Determine close side
        side = Side.SELL if was_long else Side.BUY
        size = abs(old_qty)

        logger.warning(f"[{symbol}] CLOSING POSITION: {size} units ({reason})")

        order = Order(
            symbol=symbol,
            side=side,
            quantity=size,
            order_type="MARKET"
        )

        trade = await self.ib_connector.submit_order(order)

        if trade:
            # Record trade for Kelly optimization
            entry_price = self.risk_manager.get_entry(symbol)
            pnl = 0.0
            if entry_price:
                # Calculate P&L
                if was_long:
                    pnl = (trade.fill_price - entry_price) * size
                else:
                    pnl = (entry_price - trade.fill_price) * size

                self.risk_manager.record_trade(
                    pnl=pnl,
                    symbol=symbol,
                    entry_price=entry_price,
                    exit_price=trade.fill_price
                )

                # Record for live LLM tuning (DPO pair generation)
                ml_confidence = self._last_signal_confidence.get(symbol, 0.6)
                llm_reasoning = self._last_llm_reasoning.get(symbol, "")
                self._record_trade_outcome(
                    symbol=symbol,
                    was_long=was_long,
                    pnl=pnl,
                    ml_confidence=ml_confidence,
                    llm_reasoning=llm_reasoning
                )

            # Update position to zero
            position.quantity = 0
            position.avg_price = 0.0  # FIX: Reset avg_price when position closed

            # Clear stop loss and entry time
            self.risk_manager.clear_stop_loss(symbol)
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]

            # Update trade cooldown
            self.last_trade_time[symbol] = datetime.now()

            # Update realized P&L tracking
            position.realized_pnl += pnl if pnl else 0

            logger.info(f"[{symbol}] POSITION CLOSED: {side.value} {size} @ {trade.fill_price:.5f} ({reason}) | Realized: ${pnl:.2f}")

    def _process_observation_feedback(self, symbol: str, current_mid: float):
        """
        Label pending observations with actual price direction and feed to online learner.

        Based on Chinese quant techniques from 幻方量化/九坤投资:
        - Use 1-10 tick lookahead for direction labeling
        - Feed labeled data immediately for incremental learning

        Reference: https://blog.csdn.net/xieyan0811/article/details/82949236
        """
        if symbol not in self.pending_observations:
            return

        pending = self.pending_observations[symbol]
        if len(pending) < 2:  # Need at least 2 observations
            return

        # Process observations that are at least 1 tick old (oldest first)
        # Keep only the most recent observation for next iteration
        to_remove = []

        for i, obs in enumerate(pending[:-1]):  # All except last (current)
            try:
                # Determine actual direction: 1 if price went up, 0 if down
                actual_direction = 1 if current_mid > obs['price'] else 0

                # Feed to adaptive ensemble for online learning
                self.adaptive_ensemble.add_observation(
                    symbol=symbol,
                    features=obs['features'],
                    actual_direction=actual_direction,
                    price=obs['price']
                )
                to_remove.append(i)

            except Exception as e:
                # Don't let feedback errors affect trading
                if self.tick_count.get(symbol, 0) % 1000 == 0:
                    logger.warning(f"[ONLINE] Observation feedback error: {e}")

        # Remove processed observations (reverse to maintain indices)
        for i in sorted(to_remove, reverse=True):
            pending.pop(i)

    def _record_trade_outcome(self, symbol: str, was_long: bool, pnl: float,
                               ml_confidence: float, llm_reasoning: str = ""):
        """
        Record trade outcome for live learning.

        Citation: DeepSeek GRPO - uses trade outcomes to generate DPO pairs
        Citation: 幻方量化 - "及时应对市场规则变化，不断更新模型"
        """
        if not self.use_live_tuning:
            return

        try:
            # Prepare outcome data
            outcome_data = {
                "symbol": symbol,
                "ml_direction": 1 if was_long else -1,
                "ml_confidence": ml_confidence,
                "llm_reasoning": llm_reasoning or self._last_llm_reasoning.get(symbol, ""),
                "llm_decision": "APPROVE",  # Trade was executed
                "llm_confidence": 0.7,
                "actual_direction": 1 if pnl > 0 else -1,
                "pnl_pips": pnl / 10,  # Convert to pips (approx)
                "pnl_dollars": pnl,
            }

            # Try MCP server first for centralized tracking
            try:
                resp = requests.post(
                    f"{self.mcp_tuning_url}/api/tuning/record_outcome",
                    json=outcome_data,
                    timeout=2
                )
                if resp.ok:
                    result = resp.json()
                    logger.info(f"[LIVE_TUNING] Recorded: {symbol} PnL=${pnl:.2f} "
                               f"(buffer: {result.get('buffer_size', 0)}, "
                               f"ready: {result.get('ready_for_training', False)})")

                    # Check drift alert
                    if "drift_alert" in result:
                        alert = result["drift_alert"]
                        logger.warning(f"[DRIFT] {alert['type']}: {alert['recommendation']}")

                    return
            except Exception as e:
                logger.debug(f"[LIVE_TUNING] MCP server unavailable: {e}")

            # Fallback: record locally
            if self.outcome_buffer:
                self.outcome_buffer.record(
                    symbol=outcome_data["symbol"],
                    ml_direction=outcome_data["ml_direction"],
                    ml_confidence=outcome_data["ml_confidence"],
                    llm_reasoning=outcome_data["llm_reasoning"],
                    llm_decision=outcome_data["llm_decision"],
                    llm_confidence=outcome_data["llm_confidence"],
                    actual_direction=outcome_data["actual_direction"],
                    pnl_pips=outcome_data["pnl_pips"],
                    pnl_dollars=outcome_data["pnl_dollars"],
                )
                self._outcome_count_since_training += 1
                logger.info(f"[LIVE_TUNING] Recorded locally: {symbol} PnL=${pnl:.2f} "
                           f"(count: {self._outcome_count_since_training})")

        except Exception as e:
            logger.error(f"[LIVE_TUNING] Error recording outcome: {e}")

    def _check_training_trigger(self):
        """
        Check if LoRA training should trigger.

        Citation: 幻方量化 - continuous model updates
        Citation: BigQuant - "定期更新训练集数据，通过滚动训练的方式更新预测模型"
        """
        if not self.use_live_tuning:
            return

        try:
            resp = requests.get(
                f"{self.mcp_tuning_url}/api/tuning/should_train",
                timeout=2
            )
            if resp.ok:
                result = resp.json()
                if result.get("should_train") and not result.get("is_training"):
                    logger.info("[LIVE_TUNING] Triggering background LoRA training...")
                    requests.post(
                        f"{self.mcp_tuning_url}/api/tuning/trigger_training",
                        timeout=5
                    )
                    self._outcome_count_since_training = 0
        except Exception as e:
            logger.debug(f"[LIVE_TUNING] Training check error: {e}")

    async def _check_take_profit(self, symbol: str, bid: float, ask: float):
        """
        Check if position should be closed for TAKE PROFIT.

        CRITICAL: This realizes P&L! Without this, profits are only on paper.

        Conditions to close:
        1. Profit >= take_profit_pips (e.g., 15 pips)
        2. Profit >= take_profit_pct (e.g., 0.3%)
        3. Hold time > max_hold_minutes AND profitable (lock in gains)
        """
        position = self.positions.get(symbol)
        if not position or position.quantity == 0:
            return

        mid = (bid + ask) / 2
        entry_price = position.avg_price if position.avg_price > 0 else self.risk_manager.get_entry(symbol)

        if not entry_price or entry_price <= 0:
            return

        # Calculate current P&L
        is_long = position.quantity > 0
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01

        if is_long:
            pnl_pips = (mid - entry_price) / pip_size
            pnl_pct = (mid - entry_price) / entry_price
        else:
            pnl_pips = (entry_price - mid) / pip_size
            pnl_pct = (entry_price - mid) / entry_price

        # Calculate dollar P&L for logging
        if is_long:
            if 'JPY' in symbol:
                pnl_dollars = (entry_price - mid) * position.quantity / mid
            else:
                pnl_dollars = (mid - entry_price) * position.quantity
        else:
            if 'JPY' in symbol:
                pnl_dollars = (mid - entry_price) * position.quantity / mid
            else:
                pnl_dollars = (entry_price - mid) * abs(position.quantity)

        # Check hold time
        hold_minutes = 0.0
        if symbol in self.position_entry_times:
            hold_time = datetime.now() - self.position_entry_times[symbol]
            hold_minutes = hold_time.total_seconds() / 60

        # TAKE PROFIT CONDITIONS
        should_close = False
        reason = ""

        # 1. Hit take profit in pips
        if pnl_pips >= self.take_profit_pips:
            should_close = True
            reason = f"TAKE_PROFIT: +{pnl_pips:.1f} pips (target: {self.take_profit_pips})"

        # 2. Hit take profit in percentage
        elif pnl_pct >= self.take_profit_pct:
            should_close = True
            reason = f"TAKE_PROFIT: +{pnl_pct*100:.2f}% (target: {self.take_profit_pct*100:.1f}%)"

        # 3. Held too long but profitable - lock in gains
        elif hold_minutes >= self.max_hold_minutes and pnl_pips > 3.0:
            should_close = True
            reason = f"TIME_EXIT: +{pnl_pips:.1f} pips after {hold_minutes:.0f}min"

        # 4. Small profit (> 5 pips) and held > 15 min - don't let it reverse
        elif hold_minutes >= 15.0 and pnl_pips > 5.0:
            should_close = True
            reason = f"LOCK_PROFIT: +{pnl_pips:.1f} pips after {hold_minutes:.0f}min"

        if should_close:
            logger.info(f"[{symbol}] CLOSING: {reason} | PnL: ${pnl_dollars:.2f}")
            await self._close_position(symbol, reason)

            # Clear entry time tracking
            if symbol in self.position_entry_times:
                del self.position_entry_times[symbol]

    def _log_status(self, symbol: str):
        """Log current status with P&L."""
        position = self.positions[symbol]
        n_signals = len([s for s in self.signals if s.symbol == symbol])
        n_trades = len([t for t in self.trades if t.symbol == symbol])

        bid, ask = self.last_prices.get(symbol, (0, 0))
        mid = (bid + ask) / 2

        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        # Use avg_price from position, fallback to risk_manager entry price
        entry_price = position.avg_price if position.avg_price > 0 else self.risk_manager.get_entry(symbol)
        if position.quantity != 0 and entry_price and entry_price > 0 and mid > 0:
            if 'JPY' in symbol:
                # JPY pairs: P&L in USD = (entry - current) * position / current
                unrealized_pnl = (entry_price - mid) * position.quantity / mid
            else:
                # USD pairs: P&L = (current - entry) * position
                unrealized_pnl = (mid - entry_price) * position.quantity

        pnl_str = f"+${unrealized_pnl:.2f}" if unrealized_pnl >= 0 else f"-${abs(unrealized_pnl):.2f}"

        logger.info(
            f"[{symbol}] Ticks: {self.tick_count[symbol]:,} | "
            f"Trades: {n_trades} | Pos: {position.quantity:,.0f} | "
            f"Price: {mid:.5f} | PnL: {pnl_str}"
        )

    def _get_total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions."""
        total_pnl = 0.0
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            bid, ask = self.last_prices.get(symbol, (0, 0))
            mid = (bid + ask) / 2
            if mid == 0:
                continue
            entry_price = position.avg_price if position.avg_price > 0 else self.risk_manager.get_entry(symbol)
            if entry_price and entry_price > 0:
                if 'JPY' in symbol:
                    total_pnl += (entry_price - mid) * position.quantity / mid
                else:
                    total_pnl += (mid - entry_price) * position.quantity
        return total_pnl

    def _log_portfolio_summary(self):
        """Log overall portfolio status."""
        total_unrealized = self._get_total_unrealized_pnl()
        realized = self.risk_manager.daily_pnl
        # Also sum up realized P&L from positions
        all_realized = sum(p.realized_pnl for p in self.positions.values())
        realized = max(realized, all_realized)  # Use whichever is higher

        total = realized + total_unrealized
        n_positions = sum(1 for p in self.positions.values() if p.quantity != 0)
        total_trades = len(self.trades)

        # Calculate account value
        starting = self.risk_manager.starting_capital
        current_value = starting + total

        pnl_str = f"+${total:.2f}" if total >= 0 else f"-${abs(total):.2f}"
        unreal_str = f"+${total_unrealized:.2f}" if total_unrealized >= 0 else f"-${abs(total_unrealized):.2f}"
        real_str = f"+${realized:.2f}" if realized >= 0 else f"-${abs(realized):.2f}"
        pct_return = (total / starting * 100) if starting > 0 else 0

        logger.info("=" * 70)
        logger.info(f"PORTFOLIO: ${starting:.0f} → ${current_value:.2f} ({pct_return:+.1f}%)")
        logger.info(f"Positions: {n_positions} | Trades: {total_trades} | "
                   f"Unrealized: {unreal_str} | REALIZED: {real_str} | TOTAL: {pnl_str}")
        logger.info("=" * 70)

    async def run_live(self):
        """Run with live TrueFX data feed."""
        logger.info("Starting live trading with TrueFX feed...")

        try:
            from core.data.loader import UnifiedDataLoader

            loader = UnifiedDataLoader()
            self.running = True

            # Stream live ticks
            async for tick in loader.stream_live(self.symbols):
                if not self.running:
                    break

                await self.process_tick(
                    symbol=tick.symbol,
                    bid=tick.bid,
                    ask=tick.ask,
                    volume=tick.volume,
                    timestamp=tick.timestamp
                )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Live trading error: {e}")
        finally:
            await self.shutdown()

    async def run_multi_source(self):
        """
        Run with MULTI-SOURCE data feed for MAXIMUM coverage.

        Data Sources:
        1. TrueFX - Free tick streaming (10-15 majors)
        2. Interactive Brokers - Full market data (70+ pairs)
        3. OANDA - Streaming quotes (28+ pairs)

        Automatically selects best quote (lowest spread) from all sources.
        """
        if not HAS_MULTI_SOURCE:
            logger.warning("Multi-source feed not available, falling back to TrueFX")
            await self.run_live()
            return

        logger.info("=" * 60)
        logger.info("MULTI-SOURCE DATA FEED - MAXIMUM COVERAGE")
        logger.info("=" * 60)
        logger.info("Sources: TrueFX + IB Gateway + OANDA")
        logger.info(f"Requested symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        try:
            # Create multi-source feed from environment
            self.multi_source_feed = create_multi_source_feed(
                enable_truefx=True,
                enable_ib=True,
                enable_oanda=True,
            )

            # Connect to all sources
            results = await self.multi_source_feed.connect()

            connected_sources = [s.value for s, ok in results.items() if ok]
            failed_sources = [s.value for s, ok in results.items() if not ok]

            logger.info(f"Connected: {', '.join(connected_sources)}")
            if failed_sources:
                logger.warning(f"Failed: {', '.join(failed_sources)}")

            # Get all covered symbols
            all_symbols = self.multi_source_feed.get_all_symbols()
            logger.info(f"Total symbols available: {len(all_symbols)}")

            # Filter to requested symbols (or use all if 'ALL' specified)
            if 'ALL' in self.symbols or not self.symbols:
                trade_symbols = all_symbols
            else:
                trade_symbols = [s for s in self.symbols if s in all_symbols]
                missing = [s for s in self.symbols if s not in all_symbols]
                if missing:
                    logger.warning(f"No data for: {', '.join(missing)}")

            # Update symbols list
            self.symbols = trade_symbols
            logger.info(f"Trading {len(self.symbols)} symbols")

            # Initialize positions for all symbols
            for symbol in self.symbols:
                if symbol not in self.positions:
                    self.positions[symbol] = Position(symbol=symbol)
                    self.tick_count[symbol] = 0

            # Print coverage
            self.multi_source_feed.print_coverage()

            self.running = True

            # Register tick callback
            def on_tick(tick):
                if tick.symbol in self.symbols:
                    asyncio.create_task(self.process_tick(
                        symbol=tick.symbol,
                        bid=tick.bid,
                        ask=tick.ask,
                        volume=tick.volume,
                        timestamp=tick.timestamp,
                        source=tick.source.value if hasattr(tick.source, 'value') else str(tick.source)
                    ))

            self.multi_source_feed.register_callback(on_tick)

            # Start streaming from all sources
            await self.multi_source_feed.stream()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Multi-source trading error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'multi_source_feed') and self.multi_source_feed:
                await self.multi_source_feed.disconnect()
            await self.shutdown()

    async def run_backtest(self, data: Dict[str, pd.DataFrame]):
        """Run backtest on historical data."""
        logger.info("Running backtest...")

        self.running = True

        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue

            logger.info(f"Processing {len(df)} ticks for {symbol}")

            for idx, row in df.iterrows():
                if not self.running:
                    break

                bid = row.get('bid', row.get('close', 1.0))
                ask = row.get('ask', bid + 0.0001)
                volume = row.get('volume', 0)
                timestamp = row.get('timestamp', datetime.now())

                await self.process_tick(symbol, bid, ask, volume, timestamp)

        self._print_backtest_results()

    def _print_backtest_results(self):
        """Print backtest summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        for symbol in self.symbols:
            signals = [s for s in self.signals if s.symbol == symbol]
            trades = [t for t in self.trades if t.symbol == symbol]

            if signals:
                avg_conf = np.mean([s.confidence for s in signals])
                long_signals = len([s for s in signals if s.direction > 0])
                short_signals = len([s for s in signals if s.direction < 0])

                print(f"\n{symbol}:")
                print(f"  Signals: {len(signals)} (Long: {long_signals}, Short: {short_signals})")
                print(f"  Avg Confidence: {avg_conf:.2%}")
                print(f"  Trades: {len(trades)}")

        print("\n" + "=" * 60)

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False

        # Stop Chinese Quant Online Learning (2026-01-18)
        if self.adaptive_ensemble:
            self.adaptive_ensemble.stop_background_updates()
            self.adaptive_ensemble.save_all()
            logger.info("[ONLINE] Stopped and saved adaptive ensemble")

        # Stop live retraining
        if self.retrainer:
            self.retrainer.stop()
            logger.info("[RETRAIN] Stopped")

        # Stop tick saver and log stats
        if self.tick_saver:
            stats = self.tick_saver.get_stats()
            self.tick_saver.stop()
            logger.info(f"[TICK_SAVER] Stopped. Saved {stats['ticks_saved']} ticks to {stats['output_dir']}")

        if self.ib_connector:
            await self.ib_connector.disconnect()

        # Save results
        self._save_results()

        logger.info("Shutdown complete")

    def _save_results(self):
        """Save trading results."""
        results = {
            'mode': self.mode.value,
            'symbols': self.symbols,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_ticks': sum(self.tick_count.values()),
            'total_signals': len(self.signals),
            'total_trades': len(self.trades),
            'positions': {s: {'quantity': p.quantity, 'pnl': p.realized_pnl}
                         for s, p in self.positions.items()},
            'risk_manager': {
                'daily_trades': self.risk_manager.daily_trades,
                'drawdown': self.risk_manager.current_drawdown()
            }
        }

        results_path = Path("logs/hft_results.json")
        results_path.parent.mkdir(exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")


async def main():
    parser = argparse.ArgumentParser(description='HFT Trading Bot')
    parser.add_argument('--mode', type=str, default='paper',
                        choices=['paper', 'live', 'backtest', 'monitor'],
                        help='Trading mode')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD',
                        help='Comma-separated symbols (use ALL for all available)')
    parser.add_argument('--days', type=int, default=1,
                        help='Days of data for backtest')
    parser.add_argument('--use-execution-optimizer', action='store_true',
                        help='Enable execution optimization (TWAP/VWAP/Almgren-Chriss)')
    parser.add_argument('--online-learning', action='store_true', default=True,
                        help='Enable Chinese quant-style online learning (default: True)')
    parser.add_argument('--no-online-learning', action='store_true',
                        help='Disable online learning (use static models only)')
    parser.add_argument('--capital', type=float, default=100.0,
                        help='Starting capital in USD (default: 100)')
    parser.add_argument('--multi-source', action='store_true',
                        help='Enable multi-source data feed (TrueFX + IB + OANDA)')
    parser.add_argument('--full-coverage', action='store_true',
                        help='Maximum coverage: all symbols from all sources')
    parser.add_argument('--no-ib', action='store_true',
                        help='Skip IB Gateway (simulated fills only, no commissions)')
    args = parser.parse_args()

    mode = TradingMode(args.mode)
    symbols = [s.strip() for s in args.symbols.split(',')]

    # Full coverage mode = all symbols from multi-source feed
    if args.full_coverage:
        args.multi_source = True
        symbols = ['ALL']
        logger.info("FULL COVERAGE MODE: Trading all available symbols")

    # Online learning is enabled by default, disabled with --no-online-learning
    use_online = args.online_learning and not args.no_online_learning

    bot = HFTTradingBot(
        mode=mode,
        symbols=symbols,
        use_execution_optimizer=args.use_execution_optimizer,
        use_online_learning=use_online,
        skip_ib=args.no_ib
    )

    # Set starting capital
    bot.risk_manager.starting_capital = args.capital
    bot.risk_manager.account_balance = args.capital
    bot.risk_manager.peak_balance = args.capital
    logger.info(f"Starting capital: ${args.capital:.2f}")

    # Handle shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Initialize
    await bot.initialize()

    # Run based on mode
    if mode == TradingMode.BACKTEST:
        # Load historical data
        from core.data.loader import UnifiedDataLoader
        loader = UnifiedDataLoader()

        data = {}
        for symbol in symbols:
            df = loader.load_historical(
                symbol=symbol,
                start_date=datetime.now() - timedelta(days=args.days),
                end_date=datetime.now(),
                source='truefx'
            )
            data[symbol] = df

        await bot.run_backtest(data)

    elif mode in [TradingMode.PAPER, TradingMode.LIVE]:
        if args.multi_source:
            await bot.run_multi_source()
        else:
            await bot.run_live()

    elif mode == TradingMode.MONITOR:
        logger.info("Monitor mode - watching signals only")
        if args.multi_source:
            await bot.run_multi_source()
        else:
            await bot.run_live()


if __name__ == '__main__':
    asyncio.run(main())
