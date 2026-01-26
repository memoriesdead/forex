"""
Portfolio Risk Manager
======================
Manages risk across all trading symbols.
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import threading
import logging
import numpy as np

from .limits import RiskLimits
from .per_symbol import PerSymbolRiskManager

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Portfolio-level risk management across all symbols.

    Features:
    - Per-symbol risk managers (independent limits)
    - Correlation-based position limits
    - Total portfolio exposure limits
    - Daily P&L tracking
    """

    def __init__(self, total_capital: float, max_portfolio_risk: float = 0.10):
        """
        Initialize portfolio risk manager.

        Args:
            total_capital: Total trading capital
            max_portfolio_risk: Max % of capital at risk (default 10%)
        """
        self.total_capital = total_capital
        self.max_portfolio_risk = max_portfolio_risk

        self._symbol_managers: Dict[str, PerSymbolRiskManager] = {}
        self._lock = threading.RLock()

        # Portfolio metrics
        self.total_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.peak_equity: float = total_capital
        self.current_equity: float = total_capital

        # Correlation matrix (populated externally)
        self.correlations: Optional[Dict[str, Dict[str, float]]] = None

    def get_manager(self, symbol: str) -> PerSymbolRiskManager:
        """
        Get or create a risk manager for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            PerSymbolRiskManager for the symbol
        """
        with self._lock:
            if symbol not in self._symbol_managers:
                # Get config from registry
                try:
                    from core.symbol import SymbolRegistry
                    config = SymbolRegistry.get().get_config(symbol)
                    limits = RiskLimits.from_dict(config)
                except Exception:
                    limits = RiskLimits()

                self._symbol_managers[symbol] = PerSymbolRiskManager(symbol, limits)
                logger.debug(f"Created risk manager for {symbol}")

            return self._symbol_managers[symbol]

    def can_trade(self, symbol: str, spread_pips: float = 0.0) -> Tuple[bool, str]:
        """
        Check if trading is allowed for a symbol.

        Checks both symbol-level and portfolio-level limits.

        Returns:
            (can_trade, reason)
        """
        # Symbol-level check
        manager = self.get_manager(symbol)
        can, reason = manager.can_trade(spread_pips)
        if not can:
            return can, reason

        # Portfolio-level checks
        portfolio_check = self._check_portfolio_risk(symbol)
        if not portfolio_check[0]:
            return portfolio_check

        return True, "OK"

    def _check_portfolio_risk(self, symbol: str) -> Tuple[bool, str]:
        """Check portfolio-level risk limits."""
        # Total drawdown check
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown >= self.max_portfolio_risk:
                return False, f"Portfolio drawdown limit ({drawdown:.1%})"

        # Max open positions
        open_positions = len([m for m in self._symbol_managers.values()
                            if getattr(m, '_position_size', 0) > 0])
        max_positions = RiskLimits().max_open_positions

        if open_positions >= max_positions:
            return False, f"Max open positions ({max_positions})"

        # Correlation check (avoid concentrated exposure)
        if self.correlations and symbol in self.correlations:
            high_corr_exposure = self._check_correlation_risk(symbol)
            if not high_corr_exposure[0]:
                return high_corr_exposure

        return True, "OK"

    def _check_correlation_risk(self, symbol: str) -> Tuple[bool, str]:
        """Check if adding position would create correlation risk."""
        if not self.correlations or symbol not in self.correlations:
            return True, "OK"

        # Get symbols with open positions
        open_symbols = [s for s, m in self._symbol_managers.items()
                       if getattr(m, '_position_size', 0) > 0]

        # Check correlation with existing positions
        max_corr = RiskLimits().max_correlation
        for other in open_symbols:
            if other in self.correlations.get(symbol, {}):
                corr = self.correlations[symbol][other]
                if abs(corr) > max_corr:
                    return False, f"High correlation with {other} ({corr:.2f})"

        return True, "OK"

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        win_rate: float = 0.55
    ) -> float:
        """
        Calculate position size for a symbol.

        Args:
            symbol: Trading symbol
            signal_strength: Signal confidence (-1 to 1)
            win_rate: Historical win rate

        Returns:
            Position size
        """
        manager = self.get_manager(symbol)
        return manager.calculate_position_size(
            signal_strength,
            self.total_capital,
            win_rate
        )

    def record_trade(self, symbol: str, pnl: float = 0.0):
        """Record a trade for a symbol."""
        manager = self.get_manager(symbol)
        manager.record_trade(pnl)

        if pnl != 0:
            self._update_portfolio_pnl(pnl)

    def _update_portfolio_pnl(self, pnl: float):
        """Update portfolio-level P&L tracking."""
        with self._lock:
            self.total_pnl += pnl
            self.daily_pnl += pnl
            self.current_equity += pnl

            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity

    def reset_daily(self):
        """Reset daily metrics."""
        with self._lock:
            self.daily_pnl = 0.0
            for manager in self._symbol_managers.values():
                manager._check_daily_reset()

    def set_correlations(self, corr_matrix: Dict[str, Dict[str, float]]):
        """Set correlation matrix for risk checks."""
        self.correlations = corr_matrix

    def summary(self) -> Dict:
        """Get portfolio summary."""
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        symbol_summaries = {
            symbol: manager.summary()
            for symbol, manager in self._symbol_managers.items()
        }

        return {
            'total_capital': self.total_capital,
            'current_equity': self.current_equity,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'drawdown': drawdown,
            'active_symbols': len(self._symbol_managers),
            'symbols': symbol_summaries,
        }

    def get_tradeable_symbols(self) -> List[str]:
        """Get list of symbols that can currently trade."""
        tradeable = []
        for symbol in self._symbol_managers:
            can, _ = self.can_trade(symbol)
            if can:
                tradeable.append(symbol)
        return tradeable
