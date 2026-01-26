"""
Model Loader
============
Lazy loading of ML models with caching.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pickle
import json
import logging
import threading

from .cache import ModelCache

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Lazy loader for ML models with LRU caching.

    Features:
    - Loads models on-demand (not at startup)
    - LRU cache limits RAM usage
    - Supports multiple model formats
    - Thread-safe

    Usage:
        loader = ModelLoader()
        model = loader.load('EURUSD')  # Loads from disk or cache
        available = loader.get_available()  # List symbols with models
    """

    def __init__(
        self,
        model_dir: Path = None,
        max_cached: int = 10
    ):
        """
        Initialize model loader.

        Args:
            model_dir: Directory containing model files
            max_cached: Maximum models to keep in RAM
        """
        self.model_dir = model_dir or Path("models/production")
        self.cache = ModelCache(max_models=max_cached)
        self._lock = threading.RLock()
        self._index: Dict[str, Dict] = {}
        self._load_index()

    def _load_index(self):
        """Load model index from disk."""
        index_path = self.model_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self._index = json.load(f)
                logger.info(f"Loaded model index: {len(self._index)} models")
            except Exception as e:
                logger.warning(f"Failed to load model index: {e}")

    def _save_index(self):
        """Save model index to disk."""
        index_path = self.model_dir / "index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model index: {e}")

    def load(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load a model for a symbol.

        First checks cache, then loads from disk if needed.

        Args:
            symbol: Trading symbol

        Returns:
            Model data dictionary or None if not found
        """
        # Check cache first
        cached = self.cache.get(symbol)
        if cached is not None:
            return cached

        # Load from disk
        with self._lock:
            model_data = self._load_from_disk(symbol)
            if model_data is not None:
                self.cache.put(symbol, model_data)
                return model_data

        return None

    def _load_from_disk(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load model from disk."""
        # Try different file patterns
        patterns = [
            f"{symbol}_models.pkl",
            f"{symbol}_target_direction_10_models.pkl",
        ]

        for pattern in patterns:
            model_path = self.model_dir / pattern
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)

                    # Normalize format
                    normalized = self._normalize_model_data(data, symbol)
                    logger.info(f"Loaded model for {symbol} from {model_path.name}")
                    return normalized

                except Exception as e:
                    logger.error(f"Failed to load {symbol} from {model_path}: {e}")

        logger.debug(f"No model found for {symbol}")
        return None

    def _normalize_model_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Normalize model data to consistent format."""
        # Handle old format: {'target_direction_1': {'xgboost': ...}}
        if 'target_direction_1' in data or 'target_direction_10' in data:
            # Find best target
            for target in ['target_direction_10', 'target_direction_5', 'target_direction_1']:
                if target in data:
                    td = data[target]
                    return {
                        'models': {
                            'xgboost': td.get('xgboost'),
                            'lightgbm': td.get('lightgbm'),
                            'catboost': td.get('catboost'),
                        },
                        'feature_names': td.get('features', []),
                        'target': target,
                        'symbol': symbol,
                    }

        # Handle new format: {'models': {...}, 'feature_names': [...]}
        if 'models' in data:
            data['symbol'] = symbol
            return data

        # Unknown format
        logger.warning(f"Unknown model format for {symbol}")
        return {'models': {}, 'feature_names': [], 'symbol': symbol}

    def get_available(self) -> List[str]:
        """
        Get list of symbols with available models.

        Returns:
            List of symbol names
        """
        if not self.model_dir.exists():
            return []

        symbols = set()
        for f in self.model_dir.glob("*_models.pkl"):
            # Extract symbol from filename
            name = f.stem.replace('_models', '')
            name = name.replace('_target_direction_10', '')
            name = name.replace('_target_direction_5', '')
            name = name.replace('_target_direction_1', '')
            symbols.add(name)

        return sorted(symbols)

    def is_available(self, symbol: str) -> bool:
        """Check if a model exists for a symbol."""
        return symbol in self.get_available()

    def unload(self, symbol: str) -> bool:
        """Remove a model from cache."""
        return self.cache.remove(symbol)

    def preload(self, symbols: List[str]):
        """Preload models for a list of symbols."""
        for symbol in symbols:
            self.load(symbol)
        logger.info(f"Preloaded {len(symbols)} models")

    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """Get model metadata without loading full model."""
        # Check index
        if symbol in self._index:
            return self._index[symbol]

        # Check results.json
        results_path = self.model_dir / f"{symbol}_results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    def stats(self) -> Dict:
        """Get loader statistics."""
        return {
            'model_dir': str(self.model_dir),
            'available_models': len(self.get_available()),
            'cache': self.cache.stats(),
        }

    def __len__(self) -> int:
        return len(self.get_available())

    def __contains__(self, symbol: str) -> bool:
        return self.is_available(symbol)
