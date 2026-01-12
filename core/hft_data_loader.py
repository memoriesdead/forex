"""
HFT Data Loader - Historical and Live Data Pipeline
=====================================================
Unified data loading for:
- TrueFX historical tick data
- TrueFX live tick stream
- Dukascopy historical
- Oracle Cloud stored data

Seamlessly switches from historical to live when markets open.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
import gzip
import io
import logging
import os

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types."""
    TRUEFX_HISTORICAL = "truefx_historical"
    TRUEFX_LIVE = "truefx_live"
    DUKASCOPY = "dukascopy"
    ORACLE_CLOUD = "oracle_cloud"
    LOCAL_CSV = "local_csv"


@dataclass
class TickData:
    """Standardized tick data structure."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float = 1.0
    ask_size: float = 1.0
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    source: DataSource = DataSource.LOCAL_CSV


class TrueFXHistoricalLoader:
    """
    Load TrueFX historical tick data.

    Data format: SYMBOL,TIMESTAMP,BID,ASK
    """

    BASE_URL = "https://www.truefx.com/dev/data"

    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD',
        'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
    ]

    def __init__(self, data_dir: Path = None):
        """Initialize loader."""
        self.data_dir = data_dir or Path("data/truefx")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_month(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """
        Load one month of tick data.

        Args:
            symbol: Currency pair
            year: Year (e.g., 2024)
            month: Month (1-12)

        Returns:
            DataFrame with tick data
        """
        # Check local cache first
        cache_file = self.data_dir / f"{symbol}_{year}_{month:02d}.parquet"

        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        # Download from TrueFX
        url = f"{self.BASE_URL}/{year}/{month:02d}/{symbol}-{year}-{month:02d}.zip"

        try:
            import requests
            import zipfile

            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                logger.warning(f"Failed to download: {url}")
                return pd.DataFrame()

            # Extract and parse
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for name in zf.namelist():
                    if name.endswith('.csv'):
                        with zf.open(name) as f:
                            df = pd.read_csv(
                                f,
                                header=None,
                                names=['symbol', 'timestamp', 'bid', 'ask'],
                                parse_dates=['timestamp']
                            )
                            break

            # Cache locally
            df.to_parquet(cache_file)
            logger.info(f"Cached {len(df)} ticks to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Error loading TrueFX data: {e}")
            return pd.DataFrame()

    def load_range(self, symbol: str, start_date: datetime,
                   end_date: datetime) -> pd.DataFrame:
        """Load tick data for date range."""
        frames = []

        current = start_date.replace(day=1)
        while current <= end_date:
            df = self.load_month(symbol, current.year, current.month)
            if not df.empty:
                # Filter to date range
                df = df[(df['timestamp'] >= start_date) &
                        (df['timestamp'] <= end_date)]
                frames.append(df)

            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True).sort_values('timestamp')


class TrueFXLiveLoader:
    """
    TrueFX live tick stream.

    Connects to TrueFX API for real-time quotes.
    """

    API_URL = "https://webrates.truefx.com/rates/connect.html"

    def __init__(self, username: str = None, password: str = None):
        """Initialize live loader."""
        self.username = username or os.getenv('TRUEFX_USERNAME', 'demo')
        self.password = password or os.getenv('TRUEFX_PASSWORD', 'demo')
        self.session_id = None
        self.running = False
        self.callbacks: List[Callable[[TickData], None]] = []

    async def connect(self) -> bool:
        """Connect to TrueFX API."""
        try:
            params = {
                'u': self.username,
                'p': self.password,
                'q': 'ozrates',
                'c': ','.join(TrueFXHistoricalLoader.SYMBOLS),
                'f': 'csv',
                's': 'n'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL, params=params) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        if text.strip():
                            self.session_id = text.strip()
                            logger.info(f"Connected to TrueFX: {self.session_id[:20]}...")
                            return True

            return False

        except Exception as e:
            logger.error(f"TrueFX connection error: {e}")
            return False

    async def stream_ticks(self) -> Generator[TickData, None, None]:
        """
        Stream live ticks.

        Yields:
            TickData objects as they arrive
        """
        self.running = True

        while self.running:
            try:
                params = {
                    'id': self.session_id or 'demo',
                    'f': 'csv',
                    'c': ','.join(TrueFXHistoricalLoader.SYMBOLS)
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.API_URL, params=params) as resp:
                        if resp.status == 200:
                            text = await resp.text()

                            for line in text.strip().split('\n'):
                                if not line:
                                    continue

                                parts = line.split(',')
                                if len(parts) >= 4:
                                    try:
                                        # Parse: SYMBOL,TIMESTAMP_MS,BID,BID_POINT,ASK,ASK_POINT
                                        symbol = parts[0]
                                        ts_ms = int(parts[1])
                                        bid = float(parts[2]) + float(parts[3]) / 100000
                                        ask = float(parts[4]) + float(parts[5]) / 100000

                                        tick = TickData(
                                            timestamp=datetime.fromtimestamp(ts_ms / 1000),
                                            symbol=symbol,
                                            bid=bid,
                                            ask=ask,
                                            source=DataSource.TRUEFX_LIVE
                                        )

                                        yield tick

                                        # Notify callbacks
                                        for cb in self.callbacks:
                                            cb(tick)

                                    except (ValueError, IndexError) as e:
                                        continue

                await asyncio.sleep(0.1)  # 100ms polling

            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(1)

    def stop(self):
        """Stop streaming."""
        self.running = False

    def add_callback(self, callback: Callable[[TickData], None]):
        """Add tick callback."""
        self.callbacks.append(callback)


class OracleCloudLoader:
    """
    Load data from Oracle Cloud storage.

    Location: /home/ubuntu/market_data/
    """

    def __init__(self, ssh_key: str = None, host: str = "89.168.65.47"):
        """Initialize Oracle Cloud loader."""
        self.host = host
        self.ssh_key = ssh_key or str(Path.home() / "forex" / "ssh-key-2026-01-07 (1).key")
        self.remote_path = "/home/ubuntu/market_data"

    def list_available(self) -> Dict[str, List[str]]:
        """List available data on Oracle Cloud."""
        import subprocess

        try:
            cmd = f'ssh -i "{self.ssh_key}" ubuntu@{self.host} "find {self.remote_path} -name \'*.csv\' -o -name \'*.parquet\' | head -100"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            files = result.stdout.strip().split('\n')

            # Organize by type
            organized = {
                'forex': [],
                'hft_1sec': [],
                'historical': []
            }

            for f in files:
                if 'forex' in f.lower():
                    organized['forex'].append(f)
                elif 'hft_1sec' in f.lower():
                    organized['hft_1sec'].append(f)
                else:
                    organized['historical'].append(f)

            return organized

        except Exception as e:
            logger.error(f"Error listing Oracle Cloud data: {e}")
            return {}

    def download_file(self, remote_file: str, local_path: Path) -> bool:
        """Download file from Oracle Cloud."""
        import subprocess

        try:
            cmd = f'scp -i "{self.ssh_key}" ubuntu@{self.host}:{remote_file} "{local_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Download error: {e}")
            return False

    def load_forex_ticks(self, symbol: str, date: datetime = None) -> pd.DataFrame:
        """Load forex tick data from Oracle Cloud."""
        import subprocess

        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')
        remote_file = f"{self.remote_path}/forex/{symbol}_{date_str}.csv"

        try:
            cmd = f'ssh -i "{self.ssh_key}" ubuntu@{self.host} "cat {remote_file} 2>/dev/null"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

            if result.returncode != 0 or not result.stdout.strip():
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(result.stdout))

            # Standardize columns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error loading from Oracle Cloud: {e}")
            return pd.DataFrame()


class LocalDataLoader:
    """Load data from local files."""

    def __init__(self, data_dir: Path = None):
        """Initialize local loader."""
        self.data_dir = data_dir or Path("data")
        # Additional directories to search for tick data
        self.extra_dirs = [
            Path("data_cleaned/dukascopy_local"),
            Path("data_cleaned/hft_1min"),
            Path("data_cleaned/intraday_fast"),
            Path("data/truefx"),
        ]

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load CSV file."""
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)

        # Auto-detect and parse timestamp
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
                # Convert to timezone-naive for consistency
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                if col != 'timestamp':
                    df = df.rename(columns={col: 'timestamp'})
                break

        return df

    def load_parquet(self, filepath: Path) -> pd.DataFrame:
        """Load Parquet file."""
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        return pd.read_parquet(filepath)

    def find_tick_files(self, symbol: str) -> List[Path]:
        """Find tick data files for symbol."""
        patterns = [
            f"{symbol}*.csv",
            f"{symbol}*.parquet",
            f"*{symbol}*.csv",
            f"ticks/{symbol}*.csv"
        ]

        files = []
        # Search main data directory
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))
            files.extend(self.data_dir.glob(f"**/{pattern}"))

        # Search extra directories (Dukascopy, HFT, etc.)
        for extra_dir in self.extra_dirs:
            if extra_dir.exists():
                for pattern in patterns:
                    files.extend(extra_dir.glob(pattern))

        return sorted(set(files))


class UnifiedDataLoader:
    """
    Unified data loader for HFT pipeline.

    Handles:
    - Historical data loading
    - Live data streaming
    - Seamless historical-to-live transition
    - Data normalization

    Usage:
        loader = UnifiedDataLoader()

        # Historical backtest
        for tick in loader.load_historical('EURUSD', start, end):
            process(tick)

        # Live trading
        async for tick in loader.stream_live(['EURUSD', 'GBPUSD']):
            process(tick)
    """

    def __init__(self):
        """Initialize unified loader."""
        self.truefx_historical = TrueFXHistoricalLoader()
        self.truefx_live = TrueFXLiveLoader()
        self.oracle = OracleCloudLoader()
        self.local = LocalDataLoader()

        # Data cache
        self.cache: Dict[str, pd.DataFrame] = {}

    def load_historical(self, symbol: str, start_date: datetime,
                        end_date: datetime, source: DataSource = None) -> pd.DataFrame:
        """
        Load historical tick data.

        Tries sources in order:
        1. Local cache
        2. Oracle Cloud
        3. TrueFX download

        Returns:
            DataFrame with standardized tick data
        """
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Ensure start/end dates are timezone-naive
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)

        df = pd.DataFrame()

        # Try local first
        local_files = self.local.find_tick_files(symbol)
        if local_files:
            frames = []
            for f in local_files:
                if f.suffix == '.parquet':
                    frame = self.local.load_parquet(f)
                else:
                    frame = self.local.load_csv(f)

                if not frame.empty and 'timestamp' in frame.columns:
                    # Ensure timestamp is timezone-naive
                    if frame['timestamp'].dt.tz is not None:
                        frame['timestamp'] = frame['timestamp'].dt.tz_localize(None)
                    frame = frame[(frame['timestamp'] >= start_date) &
                                  (frame['timestamp'] <= end_date)]
                    if not frame.empty:
                        frames.append(frame)

            if frames:
                df = pd.concat(frames, ignore_index=True)

        # Try Oracle Cloud if local empty
        if df.empty:
            current = start_date
            frames = []
            while current <= end_date:
                frame = self.oracle.load_forex_ticks(symbol, current)
                if not frame.empty:
                    frames.append(frame)
                current += timedelta(days=1)

            if frames:
                df = pd.concat(frames, ignore_index=True)

        # Try TrueFX download
        if df.empty:
            df = self.truefx_historical.load_range(symbol, start_date, end_date)

        # Normalize
        if not df.empty:
            df = self._normalize_dataframe(df, symbol)
            self.cache[cache_key] = df

        return df

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        result = df.copy()

        # Ensure required columns
        if 'symbol' not in result.columns:
            result['symbol'] = symbol

        # Rename common columns
        rename_map = {
            'time': 'timestamp',
            'datetime': 'timestamp',
            'Bid': 'bid',
            'Ask': 'ask',
            'BidSize': 'bid_size',
            'AskSize': 'ask_size',
            'Last': 'last_price',
            'LastSize': 'last_size',
            'Volume': 'last_size'
        }

        for old, new in rename_map.items():
            if old in result.columns and new not in result.columns:
                result = result.rename(columns={old: new})

        # Add missing columns with defaults
        if 'bid_size' not in result.columns:
            result['bid_size'] = 1.0
        if 'ask_size' not in result.columns:
            result['ask_size'] = 1.0

        # Sort by timestamp
        if 'timestamp' in result.columns:
            result = result.sort_values('timestamp').reset_index(drop=True)

        return result

    async def stream_live(self, symbols: List[str] = None) -> Generator[TickData, None, None]:
        """
        Stream live tick data.

        Args:
            symbols: Symbols to stream (default: all)

        Yields:
            TickData objects
        """
        if symbols is None:
            symbols = TrueFXHistoricalLoader.SYMBOLS

        await self.truefx_live.connect()

        async for tick in self.truefx_live.stream_ticks():
            if tick.symbol in symbols:
                yield tick

    def historical_to_live_generator(self, symbol: str,
                                      historical_end: datetime) -> Generator[TickData, None, None]:
        """
        Generator that transitions from historical to live.

        First yields historical data up to historical_end,
        then switches to live stream.

        Args:
            symbol: Symbol to load
            historical_end: When to switch to live

        Yields:
            TickData objects
        """
        # Historical phase
        historical_start = historical_end - timedelta(days=30)
        df = self.load_historical(symbol, historical_start, historical_end)

        for _, row in df.iterrows():
            yield TickData(
                timestamp=row['timestamp'],
                symbol=symbol,
                bid=row['bid'],
                ask=row['ask'],
                bid_size=row.get('bid_size', 1.0),
                ask_size=row.get('ask_size', 1.0),
                source=DataSource.TRUEFX_HISTORICAL
            )

        logger.info(f"Switching to live data for {symbol}")

        # Live phase - would need async wrapper in real usage
        # This is a sync generator, live would need async for tick in self.truefx_live.stream_ticks()

    def get_data_summary(self) -> Dict:
        """Get summary of available data."""
        return {
            'local_files': len(self.local.find_tick_files('*')),
            'cache_keys': list(self.cache.keys()),
            'truefx_symbols': TrueFXHistoricalLoader.SYMBOLS,
            'oracle_cloud': self.oracle.list_available()
        }


if __name__ == '__main__':
    print("HFT Data Loader Test")
    print("=" * 50)

    loader = UnifiedDataLoader()

    # Test local loading
    print("\nSearching for local data...")
    for symbol in ['EURUSD', 'GBPUSD']:
        files = loader.local.find_tick_files(symbol)
        print(f"  {symbol}: {len(files)} files found")

    # Test Oracle Cloud listing
    print("\nOracle Cloud data:")
    try:
        available = loader.oracle.list_available()
        for category, files in available.items():
            print(f"  {category}: {len(files)} files")
    except Exception as e:
        print(f"  Not connected: {e}")

    # Summary
    print("\nData Summary:")
    summary = loader.get_data_summary()
    print(f"  Local files: {summary['local_files']}")
    print(f"  Symbols: {', '.join(summary['truefx_symbols'])}")
