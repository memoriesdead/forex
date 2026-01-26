"""
Symbol Configuration
====================
Centralized symbol definitions, tiers, and default parameters.
Single source of truth for all forex pairs.
"""

from typing import Dict, List, Any

# Symbol tiers by liquidity and trading priority
SYMBOL_TIERS: Dict[str, List[str]] = {
    'majors': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
        'AUDUSD', 'USDCAD', 'NZDUSD'
    ],
    'crosses': [
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF',
        'AUDJPY', 'EURAUD', 'GBPAUD'
    ],
    'exotics': [
        'EURNZD', 'GBPNZD', 'AUDNZD', 'NZDJPY',
        'AUDCAD', 'CADCHF', 'CADJPY'
    ],
    'minor_crosses': [
        'AUDCHF', 'GBPCAD', 'GBPCHF', 'EURCAD',
        'CHFJPY', 'CADNZD', 'NZDCHF'
    ],
    'emerging': [
        'USDMXN', 'USDZAR', 'USDTRY', 'USDSEK',
        'USDNOK', 'USDDKK', 'USDPLN', 'USDCZK',
        'USDHUF', 'USDSGD', 'USDHKD', 'USDCNH'
    ],
    'euro_emerging': [
        'EURTRY', 'EURZAR', 'EURMXN', 'EURSEK',
        'EURNOK', 'EURDKK', 'EURPLN', 'EURCZK',
        'EURHUF', 'EURSGD', 'EURHKD'
    ],
}

# Flattened list of all symbols
ALL_SYMBOLS: List[str] = []
for tier_symbols in SYMBOL_TIERS.values():
    ALL_SYMBOLS.extend(tier_symbols)
ALL_SYMBOLS = list(dict.fromkeys(ALL_SYMBOLS))  # Remove duplicates, preserve order

# Pip values for position sizing (in quote currency per pip per standard lot)
PIP_VALUES: Dict[str, float] = {
    # JPY pairs (pip = 0.01)
    'USDJPY': 1000.0,  # 1000 JPY per pip
    'EURJPY': 1000.0,
    'GBPJPY': 1000.0,
    'AUDJPY': 1000.0,
    'NZDJPY': 1000.0,
    'CADJPY': 1000.0,
    'CHFJPY': 1000.0,
    'SGDJPY': 1000.0,
    'HKDJPY': 1000.0,
    # Standard pairs (pip = 0.0001)
    'DEFAULT': 10.0,  # $10 per pip for XXX/USD pairs
}

# Default trading parameters for all symbols
DEFAULT_SYMBOL_CONFIG: Dict[str, Any] = {
    'max_position_pct': 0.02,       # 2% of capital per trade
    'max_drawdown_pct': 0.05,       # 5% max drawdown per symbol
    'daily_trade_limit': 50,        # Per-symbol daily limit
    'min_confidence': 0.15,         # Signal threshold
    'kelly_fraction': 0.25,         # Conservative Kelly (25%)
    'spread_limit_pips': 3.0,       # Max spread to trade
    'enabled': True,                # Default enabled
}

# Tier-specific overrides
TIER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'majors': {
        'max_position_pct': 0.03,   # Higher position for majors
        'spread_limit_pips': 2.0,   # Tighter spread requirement
    },
    'crosses': {
        'max_position_pct': 0.02,
        'spread_limit_pips': 3.0,
    },
    'exotics': {
        'max_position_pct': 0.015,  # Smaller positions
        'spread_limit_pips': 5.0,   # Allow wider spreads
    },
    'emerging': {
        'max_position_pct': 0.01,   # Even smaller
        'spread_limit_pips': 10.0,  # Much wider spreads OK
        'daily_trade_limit': 20,    # Fewer trades
    },
}


def get_symbol_tier(symbol: str) -> str:
    """Get the tier for a symbol."""
    for tier, symbols in SYMBOL_TIERS.items():
        if symbol in symbols:
            return tier
    return 'unknown'


def get_symbol_config(symbol: str) -> Dict[str, Any]:
    """Get merged config for a symbol (default + tier overrides)."""
    config = DEFAULT_SYMBOL_CONFIG.copy()
    tier = get_symbol_tier(symbol)
    if tier in TIER_OVERRIDES:
        config.update(TIER_OVERRIDES[tier])
    return config


def get_pip_value(symbol: str) -> float:
    """Get pip value for a symbol."""
    return PIP_VALUES.get(symbol, PIP_VALUES['DEFAULT'])


def parse_symbol(symbol: str) -> tuple:
    """Parse symbol into base and quote currencies."""
    if len(symbol) != 6:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol[:3], symbol[3:]
