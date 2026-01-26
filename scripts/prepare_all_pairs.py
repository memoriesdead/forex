#!/usr/bin/env python3
"""
Prepare All Pairs - Data Preparation for 78 Forex Pairs
========================================================
Prepares training data for all available forex pairs from Dukascopy.

Features:
- Loads raw tick data from data/ directory
- Generates OHLCV data at specified frequency
- Creates train/val/test splits (70/15/15)
- Generates target labels (direction at various horizons)
- Saves to training_package/ directory

Usage:
    # Prepare all pairs with available data
    python scripts/prepare_all_pairs.py --all

    # Prepare specific pairs
    python scripts/prepare_all_pairs.py --pairs EURUSD,GBPUSD,USDJPY

    # Check which pairs have data
    python scripts/prepare_all_pairs.py --status

    # Specify custom frequency
    python scripts/prepare_all_pairs.py --all --freq 5min
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# All 78 forex pairs available from Dukascopy
ALL_FOREX_PAIRS = [
    # Majors (7)
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',

    # EUR Crosses (10)
    'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD',
    'EURPLN', 'EURSEK', 'EURTRY',

    # GBP Crosses (10)
    'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPNOK', 'GBPPLN',
    'GBPSEK', 'GBPSGD', 'GBPTRY',

    # JPY Crosses (10)
    'AUDJPY', 'CADJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDJPY', 'SGDJPY',
    'TRYJPY', 'USDJPY', 'ZARJPY',

    # AUD Crosses (6)
    'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDSGD',

    # NZD Crosses (5)
    'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDSGD', 'NZDUSD',

    # CAD Crosses (3)
    'CADCHF', 'CADJPY',

    # CHF Crosses (3)
    'CHFJPY',

    # USD Crosses (15)
    'USDCNH', 'USDCZK', 'USDDKK', 'USDHKD', 'USDHUF', 'USDNOK', 'USDPLN',
    'USDSEK', 'USDSGD', 'USDTHB', 'USDTRY',

    # Exotic pairs (8)
    'EURDKK', 'EURHKD', 'EURHUF', 'EURSGD', 'HKDJPY', 'SGDJPY',
]

# Remove duplicates
ALL_FOREX_PAIRS = list(dict.fromkeys(ALL_FOREX_PAIRS))


class DataPreparer:
    """Prepare training data for forex pairs."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        freq: str = '1min',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        target_horizons: List[int] = [1, 5, 10, 20, 50, 100],
    ):
        """
        Initialize DataPreparer.

        Args:
            data_dir: Directory containing raw data
            output_dir: Directory to save prepared data
            freq: OHLCV frequency (default: 1min)
            train_ratio: Training data ratio
            val_ratio: Validation data ratio (test = 1 - train - val)
            target_horizons: List of forward horizons for direction targets
        """
        self.data_dir = data_dir or PROJECT_ROOT / 'data'
        self.output_dir = output_dir or PROJECT_ROOT / 'training_package'
        self.freq = freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_horizons = target_horizons

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_available_pairs(self) -> List[str]:
        """Get list of pairs that have raw data."""
        available = []

        for pair in ALL_FOREX_PAIRS:
            # Check various data locations
            pair_dir = self.data_dir / pair
            pair_file = self.data_dir / f"{pair}.parquet"
            pair_csv = self.data_dir / f"{pair}.csv"

            if pair_dir.exists() or pair_file.exists() or pair_csv.exists():
                available.append(pair)

        # Also check training_package for already prepared data
        for pair_dir in self.output_dir.iterdir():
            if pair_dir.is_dir() and (pair_dir / 'train.parquet').exists():
                if pair_dir.name not in available:
                    available.append(pair_dir.name)

        return sorted(list(set(available)))

    def get_prepared_pairs(self) -> List[str]:
        """Get list of pairs that have been prepared."""
        prepared = []

        for pair_dir in self.output_dir.iterdir():
            if pair_dir.is_dir() and (pair_dir / 'train.parquet').exists():
                prepared.append(pair_dir.name)

        return sorted(prepared)

    def get_status(self) -> Dict:
        """Get data preparation status."""
        available = self.get_available_pairs()
        prepared = self.get_prepared_pairs()

        return {
            'total_pairs': len(ALL_FOREX_PAIRS),
            'available_pairs': len(available),
            'prepared_pairs': len(prepared),
            'remaining': len(available) - len(prepared),
            'pairs': {
                'available': available,
                'prepared': prepared,
                'remaining': [p for p in available if p not in prepared],
            }
        }

    def print_status(self):
        """Print data preparation status."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("DATA PREPARATION STATUS")
        print("=" * 60)
        print(f"Total forex pairs: {status['total_pairs']}")
        print(f"Pairs with data: {status['available_pairs']}")
        print(f"Pairs prepared: {status['prepared_pairs']}")
        print(f"Remaining to prepare: {status['remaining']}")

        if status['remaining'] > 0:
            print(f"\nTo prepare remaining:")
            for pair in status['pairs']['remaining'][:10]:
                print(f"  - {pair}")
            if len(status['pairs']['remaining']) > 10:
                print(f"  ... and {len(status['pairs']['remaining']) - 10} more")

        print("=" * 60 + "\n")

    def load_raw_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Load raw tick data for a pair."""
        # Try different data sources
        sources = [
            self.data_dir / pair / f"{pair}_ticks.parquet",
            self.data_dir / pair / "ticks.parquet",
            self.data_dir / f"{pair}.parquet",
            self.data_dir / f"{pair}_ticks.csv",
            self.data_dir / f"{pair}.csv",
        ]

        for source in sources:
            if source.exists():
                try:
                    if source.suffix == '.parquet':
                        df = pd.read_parquet(source)
                    else:
                        df = pd.read_csv(source)

                    # Ensure datetime index
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp')
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.set_index('datetime')
                    elif not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)

                    return df.sort_index()

                except Exception as e:
                    logger.warning(f"Failed to load {source}: {e}")
                    continue

        # Check if already prepared in training_package
        train_file = self.output_dir / pair / 'train.parquet'
        if train_file.exists():
            return None  # Already prepared

        return None

    def create_ohlcv(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """Create OHLCV data from tick data."""
        # Determine price column
        if 'mid' in tick_data.columns:
            price_col = 'mid'
        elif 'close' in tick_data.columns:
            price_col = 'close'
        elif 'bid' in tick_data.columns and 'ask' in tick_data.columns:
            tick_data['mid'] = (tick_data['bid'] + tick_data['ask']) / 2
            price_col = 'mid'
        else:
            price_col = tick_data.columns[0]

        # Resample to OHLCV
        ohlcv = tick_data[price_col].resample(self.freq).ohlc()
        ohlcv.columns = ['open', 'high', 'low', 'close']

        # Add volume if available
        if 'volume' in tick_data.columns:
            ohlcv['volume'] = tick_data['volume'].resample(self.freq).sum()
        else:
            ohlcv['volume'] = tick_data[price_col].resample(self.freq).count()

        # Forward fill missing values
        ohlcv = ohlcv.ffill()

        return ohlcv.dropna()

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for supervised learning."""
        result = df.copy()

        for horizon in self.target_horizons:
            # Direction target: 1 if price goes up, 0 if down
            future_return = result['close'].shift(-horizon) / result['close'] - 1
            result[f'target_direction_{horizon}'] = (future_return > 0).astype(int)

            # Return target (log return in bps)
            result[f'target_return_{horizon}'] = np.log(
                result['close'].shift(-horizon) / result['close']
            ) * 10000  # bps

            # Volatility target
            result[f'target_vol_{horizon}'] = result['close'].pct_change().rolling(horizon).std().shift(-horizon)

        # Drop rows with NaN targets
        result = result.dropna()

        return result

    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets."""
        n = len(df)

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        return train, val, test

    def prepare_pair(self, pair: str, force: bool = False) -> bool:
        """
        Prepare training data for a single pair.

        Args:
            pair: Currency pair symbol
            force: Force re-preparation even if already done

        Returns:
            True if successful
        """
        output_dir = self.output_dir / pair

        # Check if already prepared
        if not force and (output_dir / 'train.parquet').exists():
            logger.info(f"{pair}: Already prepared")
            return True

        logger.info(f"{pair}: Loading raw data...")

        # Load raw data
        raw_data = self.load_raw_data(pair)
        if raw_data is None:
            logger.warning(f"{pair}: No raw data found")
            return False

        logger.info(f"{pair}: Creating OHLCV ({self.freq})...")

        # Create OHLCV
        ohlcv = self.create_ohlcv(raw_data)
        if len(ohlcv) < 1000:
            logger.warning(f"{pair}: Too few samples ({len(ohlcv)})")
            return False

        logger.info(f"{pair}: Creating targets...")

        # Create targets
        data = self.create_targets(ohlcv)

        logger.info(f"{pair}: Splitting data...")

        # Split
        train, val, test = self.split_data(data)

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)

        train.to_parquet(output_dir / 'train.parquet')
        val.to_parquet(output_dir / 'val.parquet')
        test.to_parquet(output_dir / 'test.parquet')

        # Save metadata
        metadata = {
            'pair': pair,
            'freq': self.freq,
            'samples': {
                'train': len(train),
                'val': len(val),
                'test': len(test),
            },
            'date_range': {
                'start': str(data.index[0]),
                'end': str(data.index[-1]),
            },
            'target_horizons': self.target_horizons,
            'prepared_at': datetime.now().isoformat(),
        }

        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"{pair}: Saved {len(train)} train, {len(val)} val, {len(test)} test samples")

        return True

    def prepare_all(self, pairs: Optional[List[str]] = None, force: bool = False) -> Dict:
        """
        Prepare training data for all pairs.

        Args:
            pairs: List of pairs to prepare (None = all available)
            force: Force re-preparation

        Returns:
            Results dict
        """
        if pairs is None:
            pairs = self.get_available_pairs()

        results = {
            'success': [],
            'failed': [],
            'skipped': [],
        }

        for i, pair in enumerate(pairs):
            logger.info(f"\n[{i+1}/{len(pairs)}] {pair}")

            try:
                if self.prepare_pair(pair, force=force):
                    results['success'].append(pair)
                else:
                    results['failed'].append(pair)
            except Exception as e:
                logger.error(f"{pair}: {e}")
                results['failed'].append(pair)

        logger.info(f"\n{'='*60}")
        logger.info(f"Preparation complete:")
        logger.info(f"  Success: {len(results['success'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*60}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare training data for forex pairs'
    )

    parser.add_argument('--all', action='store_true', help='Prepare all available pairs')
    parser.add_argument('--pairs', type=str, help='Comma-separated list of pairs')
    parser.add_argument('--status', action='store_true', help='Show preparation status')
    parser.add_argument('--force', action='store_true', help='Force re-preparation')

    parser.add_argument('--freq', type=str, default='1min', help='OHLCV frequency (default: 1min)')
    parser.add_argument('--train-ratio', type=float, default=0.70, help='Training ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio')

    args = parser.parse_args()

    preparer = DataPreparer(
        freq=args.freq,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    if args.status:
        preparer.print_status()

    elif args.all:
        preparer.prepare_all(force=args.force)

    elif args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
        preparer.prepare_all(pairs=pairs, force=args.force)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
