"""
Model Cache
===========
LRU cache for ML models to limit RAM usage.
"""

from collections import OrderedDict
from typing import Dict, Optional, Any
import threading
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """
    LRU (Least Recently Used) cache for ML models.

    With 78+ forex pairs and ~70MB per model, loading all would use 5.5GB RAM.
    This cache keeps only the N most recently used models in memory.

    Usage:
        cache = ModelCache(max_models=10)  # 700MB max
        cache.put('EURUSD', model_data)
        model = cache.get('EURUSD')  # Moves to most recent
    """

    def __init__(self, max_models: int = 10):
        """
        Initialize cache.

        Args:
            max_models: Maximum models to keep in RAM (default 10 = ~700MB)
        """
        self.max_models = max_models
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a model from cache.

        If found, moves to most recently used position.

        Args:
            symbol: Trading symbol

        Returns:
            Model data dict or None if not cached
        """
        with self._lock:
            if symbol in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(symbol)
                self._hits += 1
                return self._cache[symbol]

            self._misses += 1
            return None

    def put(self, symbol: str, model: Dict[str, Any]):
        """
        Add a model to cache.

        Evicts least recently used if at capacity.

        Args:
            symbol: Trading symbol
            model: Model data dictionary
        """
        with self._lock:
            # If already in cache, update and move to end
            if symbol in self._cache:
                self._cache[symbol] = model
                self._cache.move_to_end(symbol)
                return

            # Evict LRU if at capacity
            while len(self._cache) >= self.max_models:
                evicted_symbol, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted {evicted_symbol} from model cache")

            self._cache[symbol] = model
            logger.debug(f"Cached model for {symbol} ({len(self._cache)}/{self.max_models})")

    def remove(self, symbol: str) -> bool:
        """Remove a model from cache."""
        with self._lock:
            if symbol in self._cache:
                del self._cache[symbol]
                return True
            return False

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")

    def contains(self, symbol: str) -> bool:
        """Check if symbol is in cache."""
        with self._lock:
            return symbol in self._cache

    @property
    def size(self) -> int:
        """Current number of cached models."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_models,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'cached_symbols': list(self._cache.keys()),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, symbol: str) -> bool:
        return self.contains(symbol)
