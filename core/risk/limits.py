"""
Risk Limits
===========
Dataclass for risk parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RiskLimits:
    """Risk limits for a trading symbol or portfolio."""

    # Position sizing
    max_position_pct: float = 0.02       # Max % of capital per trade
    kelly_fraction: float = 0.25          # Kelly criterion fraction

    # Loss limits
    max_drawdown_pct: float = 0.05        # Max drawdown before stopping
    max_loss_per_trade_pct: float = 0.01  # Max loss per single trade

    # Trade limits
    daily_trade_limit: int = 50           # Max trades per day
    max_open_positions: int = 10          # Max concurrent positions

    # Correlation risk
    max_correlation: float = 0.7          # Avoid highly correlated positions

    # Execution
    min_confidence: float = 0.15          # Minimum signal confidence to trade
    spread_limit_pips: float = 3.0        # Max spread to trade

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RiskLimits':
        """Create RiskLimits from a configuration dict."""
        return cls(
            max_position_pct=config.get('max_position_pct', 0.02),
            kelly_fraction=config.get('kelly_fraction', 0.25),
            max_drawdown_pct=config.get('max_drawdown_pct', 0.05),
            max_loss_per_trade_pct=config.get('max_loss_per_trade_pct', 0.01),
            daily_trade_limit=config.get('daily_trade_limit', 50),
            max_open_positions=config.get('max_open_positions', 10),
            max_correlation=config.get('max_correlation', 0.7),
            min_confidence=config.get('min_confidence', 0.15),
            spread_limit_pips=config.get('spread_limit_pips', 3.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_position_pct': self.max_position_pct,
            'kelly_fraction': self.kelly_fraction,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_loss_per_trade_pct': self.max_loss_per_trade_pct,
            'daily_trade_limit': self.daily_trade_limit,
            'max_open_positions': self.max_open_positions,
            'max_correlation': self.max_correlation,
            'min_confidence': self.min_confidence,
            'spread_limit_pips': self.spread_limit_pips,
        }
