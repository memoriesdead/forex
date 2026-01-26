# Data Module
# Data loading and buffering for forex trading

from .loader import UnifiedDataLoader, TrueFXLiveLoader, TrueFXHistoricalLoader
from .buffer import LiveTickBuffer, TickRecord, get_tick_buffer

# Multi-source data feed (2026-01-18)
try:
    from .multi_source_feed import (
        MultiSourceFeed,
        TrueFXFeed,
        IBMarketDataFeed,
        OANDAFeed,
        DataSource,
        LiveTick,
        SymbolCoverage,
        create_multi_source_feed,
    )
    HAS_MULTI_SOURCE = True
except ImportError:
    HAS_MULTI_SOURCE = False

# Live tick saver - 100% data capture (2026-01-18)
try:
    from .tick_saver import (
        LiveTickSaver,
        SavedTick,
        get_tick_saver,
        save_tick,
    )
    HAS_TICK_SAVER = True
except ImportError:
    HAS_TICK_SAVER = False

__all__ = [
    'UnifiedDataLoader',
    'TrueFXLiveLoader',
    'TrueFXHistoricalLoader',
    'LiveTickBuffer',
    'TickRecord',
    'get_tick_buffer',
    # Multi-source feed
    'MultiSourceFeed',
    'TrueFXFeed',
    'IBMarketDataFeed',
    'OANDAFeed',
    'DataSource',
    'LiveTick',
    'SymbolCoverage',
    'create_multi_source_feed',
    'HAS_MULTI_SOURCE',
    # Tick saver (2026-01-18)
    'LiveTickSaver',
    'SavedTick',
    'get_tick_saver',
    'save_tick',
    'HAS_TICK_SAVER',
]
