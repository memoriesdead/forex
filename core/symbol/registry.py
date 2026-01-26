"""
Symbol Registry
===============
Singleton registry for all trading pairs with runtime enable/disable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """Represents a forex trading pair with its configuration."""
    symbol: str
    tier: str
    base_currency: str
    quote_currency: str
    pip_value: float
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def is_jpy_pair(self) -> bool:
        """Check if this is a JPY pair (different pip calculation)."""
        return 'JPY' in self.symbol


class SymbolRegistry:
    """
    Singleton registry for all trading symbols.

    Provides:
    - Central source of truth for symbols
    - Runtime enable/disable
    - Tier-based filtering
    - Per-symbol configuration

    Usage:
        registry = SymbolRegistry.get()
        majors = registry.get_enabled(tier='majors')
        config = registry.get_config('EURUSD')
    """

    _instance: Optional['SymbolRegistry'] = None
    _lock = threading.Lock()

    def __init__(self):
        if SymbolRegistry._instance is not None:
            raise RuntimeError("Use SymbolRegistry.get() instead")

        self._pairs: Dict[str, TradingPair] = {}
        self._config_path: Optional[Path] = None
        self._load_symbols()

    @classmethod
    def get(cls) -> 'SymbolRegistry':
        """Get the singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _load_symbols(self):
        """Load symbols from config."""
        from config.symbols import (
            SYMBOL_TIERS, get_symbol_config, get_pip_value, parse_symbol
        )

        for tier, symbols in SYMBOL_TIERS.items():
            for symbol in symbols:
                try:
                    base, quote = parse_symbol(symbol)
                    self._pairs[symbol] = TradingPair(
                        symbol=symbol,
                        tier=tier,
                        base_currency=base,
                        quote_currency=quote,
                        pip_value=get_pip_value(symbol),
                        config=get_symbol_config(symbol),
                        enabled=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to load symbol {symbol}: {e}")

        logger.info(f"Loaded {len(self._pairs)} symbols across {len(SYMBOL_TIERS)} tiers")

    def get_pair(self, symbol: str) -> Optional[TradingPair]:
        """Get a trading pair by symbol."""
        return self._pairs.get(symbol)

    def get_all(self) -> List[TradingPair]:
        """Get all trading pairs."""
        return list(self._pairs.values())

    def get_enabled(self, tier: str = None) -> List[TradingPair]:
        """
        Get enabled trading pairs, optionally filtered by tier.

        Args:
            tier: Optional tier filter ('majors', 'crosses', etc.)

        Returns:
            List of enabled TradingPair objects
        """
        pairs = [p for p in self._pairs.values() if p.enabled]
        if tier:
            pairs = [p for p in pairs if p.tier == tier]
        return pairs

    def get_symbols(self, tier: str = None, enabled_only: bool = True) -> List[str]:
        """Get symbol names as strings."""
        if enabled_only:
            pairs = self.get_enabled(tier)
        else:
            pairs = self.get_all()
            if tier:
                pairs = [p for p in pairs if p.tier == tier]
        return [p.symbol for p in pairs]

    def get_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for a symbol."""
        pair = self._pairs.get(symbol)
        if pair:
            return pair.config.copy()
        from config.symbols import DEFAULT_SYMBOL_CONFIG
        return DEFAULT_SYMBOL_CONFIG.copy()

    def enable(self, symbol: str) -> bool:
        """Enable a symbol for trading."""
        if symbol in self._pairs:
            self._pairs[symbol].enabled = True
            logger.info(f"Enabled {symbol}")
            return True
        return False

    def disable(self, symbol: str) -> bool:
        """Disable a symbol from trading."""
        if symbol in self._pairs:
            self._pairs[symbol].enabled = False
            logger.info(f"Disabled {symbol}")
            return True
        return False

    def enable_tier(self, tier: str):
        """Enable all symbols in a tier."""
        for pair in self._pairs.values():
            if pair.tier == tier:
                pair.enabled = True
        logger.info(f"Enabled tier: {tier}")

    def disable_tier(self, tier: str):
        """Disable all symbols in a tier."""
        for pair in self._pairs.values():
            if pair.tier == tier:
                pair.enabled = False
        logger.info(f"Disabled tier: {tier}")

    def set_config(self, symbol: str, key: str, value: Any) -> bool:
        """Update a configuration value for a symbol."""
        if symbol in self._pairs:
            self._pairs[symbol].config[key] = value
            return True
        return False

    def save_state(self, path: Path = None):
        """Save enabled/disabled state to file."""
        path = path or Path("config/symbol_state.json")
        state = {
            symbol: {'enabled': pair.enabled, 'config': pair.config}
            for symbol, pair in self._pairs.items()
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved symbol state to {path}")

    def load_state(self, path: Path = None):
        """Load enabled/disabled state from file."""
        path = path or Path("config/symbol_state.json")
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        for symbol, data in state.items():
            if symbol in self._pairs:
                self._pairs[symbol].enabled = data.get('enabled', True)
                self._pairs[symbol].config.update(data.get('config', {}))

        logger.info(f"Loaded symbol state from {path}")

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        tiers = {}
        for pair in self._pairs.values():
            if pair.tier not in tiers:
                tiers[pair.tier] = {'total': 0, 'enabled': 0}
            tiers[pair.tier]['total'] += 1
            if pair.enabled:
                tiers[pair.tier]['enabled'] += 1

        return {
            'total_symbols': len(self._pairs),
            'enabled_symbols': len([p for p in self._pairs.values() if p.enabled]),
            'tiers': tiers
        }

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._pairs

    def __iter__(self):
        return iter(self._pairs.values())
