"""
Per-Symbol Risk Manager
=======================
Independent risk tracking for each trading symbol.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from datetime import datetime, date
import logging

from .limits import RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class SymbolMetrics:
    """Tracking metrics for a symbol."""
    daily_trades: int = 0
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    last_trade_time: Optional[datetime] = None
    last_reset_date: Optional[date] = None
    win_count: int = 0
    loss_count: int = 0


class PerSymbolRiskManager:
    """
    Risk manager for a single trading symbol.

    Tracks:
    - Daily trade count
    - Per-symbol drawdown
    - Win/loss statistics
    - Position sizing via Kelly

    Independent from other symbols - EURUSD hitting its limit
    doesn't affect GBPUSD trading.
    """

    def __init__(self, symbol: str, limits: RiskLimits = None):
        self.symbol = symbol
        self.limits = limits or RiskLimits()
        self.metrics = SymbolMetrics()
        self._position_size: float = 0.0

    def can_trade(self, spread_pips: float = 0.0) -> Tuple[bool, str]:
        """
        Check if trading is allowed for this symbol.

        Returns:
            (can_trade, reason)
        """
        # Reset daily counters if new day
        self._check_daily_reset()

        # Daily trade limit
        if self.metrics.daily_trades >= self.limits.daily_trade_limit:
            return False, f"{self.symbol}: Daily trade limit ({self.limits.daily_trade_limit})"

        # Drawdown limit
        if self.metrics.current_drawdown >= self.limits.max_drawdown_pct:
            return False, f"{self.symbol}: Max drawdown ({self.metrics.current_drawdown:.1%})"

        # Spread check
        if spread_pips > self.limits.spread_limit_pips:
            return False, f"{self.symbol}: Spread too wide ({spread_pips:.1f} > {self.limits.spread_limit_pips})"

        return True, "OK"

    def calculate_position_size(
        self,
        signal_strength: float,
        account_balance: float,
        win_rate: float = 0.55
    ) -> float:
        """
        Calculate position size using fractional Kelly.

        Args:
            signal_strength: Signal confidence (-1 to 1)
            account_balance: Total account balance
            win_rate: Historical win rate for this symbol

        Returns:
            Position size in base currency units
        """
        # Allocated capital for this symbol
        allocated = account_balance * self.limits.max_position_pct

        # Kelly criterion: f* = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = lose prob
        # Simplified: use signal strength as confidence multiplier
        kelly_bet = abs(signal_strength) * self.limits.kelly_fraction

        # Final position size
        position = allocated * kelly_bet

        self._position_size = position
        return position

    def record_trade(self, pnl: float = 0.0):
        """Record a trade execution."""
        self._check_daily_reset()
        self.metrics.daily_trades += 1
        self.metrics.last_trade_time = datetime.now()

        if pnl != 0:
            self.record_pnl(pnl)

    def record_pnl(self, pnl: float):
        """Record realized PnL."""
        self.metrics.daily_pnl += pnl
        self.metrics.current_equity += pnl

        if pnl > 0:
            self.metrics.win_count += 1
        else:
            self.metrics.loss_count += 1

        # Update peak and drawdown
        if self.metrics.current_equity > self.metrics.peak_equity:
            self.metrics.peak_equity = self.metrics.current_equity

        if self.metrics.peak_equity > 0:
            self.metrics.current_drawdown = (
                (self.metrics.peak_equity - self.metrics.current_equity)
                / self.metrics.peak_equity
            )

    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        today = date.today()
        if self.metrics.last_reset_date != today:
            self.metrics.daily_trades = 0
            self.metrics.daily_pnl = 0.0
            self.metrics.last_reset_date = today
            logger.debug(f"{self.symbol}: Daily counters reset")

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.metrics.win_count + self.metrics.loss_count
        if total == 0:
            return 0.5  # Default assumption
        return self.metrics.win_count / total

    @property
    def trades_remaining(self) -> int:
        """Get remaining trades for today."""
        return max(0, self.limits.daily_trade_limit - self.metrics.daily_trades)

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            'symbol': self.symbol,
            'daily_trades': self.metrics.daily_trades,
            'trades_remaining': self.trades_remaining,
            'daily_pnl': self.metrics.daily_pnl,
            'current_drawdown': self.metrics.current_drawdown,
            'win_rate': self.win_rate,
            'can_trade': self.can_trade()[0],
        }
