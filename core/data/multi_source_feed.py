"""
Multi-Source Live Data Feed
============================
Aggregates real-time forex data from ALL available sources for maximum coverage.

Data Sources:
1. TrueFX - Free tick streaming (10 majors)
2. Interactive Brokers - Full market data (70+ pairs)
3. OANDA - Streaming quotes (28+ pairs)
4. Forex.com - REST quotes (12 pairs)
5. tastyfx/IG - Streaming quotes (14 pairs)

Architecture:
    MultiSourceFeed
    ├── TrueFXFeed (primary for majors - free)
    ├── IBFeed (backup + exotics)
    ├── OANDAFeed (backup + additional pairs)
    ├── ForexComFeed (backup)
    └── TastyFXFeed (backup)

    → BestQuoteAggregator → Trading Bot
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import threading
import time
import aiohttp

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source identifiers."""
    TRUEFX = "truefx"
    IB = "ib"
    OANDA = "oanda"
    FOREX_COM = "forex_com"
    TASTYFX = "tastyfx"
    IG = "ig"


@dataclass
class LiveTick:
    """Unified tick representation from any source."""
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    spread_pips: float
    timestamp: datetime
    source: DataSource
    volume: float = 0.0

    # Quality metrics
    latency_ms: float = 0.0
    stale: bool = False

    @property
    def is_valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask >= self.bid


@dataclass
class SymbolCoverage:
    """Track which sources cover which symbols."""
    symbol: str
    sources: Set[DataSource] = field(default_factory=set)
    primary_source: Optional[DataSource] = None
    last_tick: Optional[LiveTick] = None
    tick_count: int = 0


class TrueFXFeed:
    """
    TrueFX live tick feed.
    Free, covers 10 major pairs.
    """

    URL = "https://webrates.truefx.com/rates/connect.html"

    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD',
        'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
        'AUDJPY', 'CHFJPY', 'EURAUD', 'EURCHF', 'GBPCHF'
    ]

    PIP_VALUES = {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
        'USDCHF': 0.0001, 'USDCAD': 0.0001, 'AUDUSD': 0.0001,
        'NZDUSD': 0.0001, 'EURGBP': 0.0001, 'EURJPY': 0.01,
        'GBPJPY': 0.01, 'AUDJPY': 0.01, 'CHFJPY': 0.01,
        'EURAUD': 0.0001, 'EURCHF': 0.0001, 'GBPCHF': 0.0001,
    }

    def __init__(self, username: str = 'demo', password: str = 'demo'):
        self.username = username
        self.password = password
        self.session_id = None
        self._session = None
        self._running = False

    async def connect(self) -> bool:
        """Connect to TrueFX."""
        try:
            self._session = aiohttp.ClientSession()

            # Get session
            params = {
                'u': self.username,
                'p': self.password,
                'q': 'ozrates',
                'c': ','.join(f"{s[:3]}/{s[3:]}" for s in self.SYMBOLS),
                'f': 'csv',
                's': 'n'
            }

            async with self._session.get(self.URL, params=params) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    if text.strip():
                        self.session_id = text.strip().split(',')[0] if ',' in text else None
                        logger.info(f"TrueFX connected, session: {self.session_id}")
                        return True

            logger.warning("TrueFX connection failed")
            return False

        except Exception as e:
            logger.error(f"TrueFX connect error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from TrueFX."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    async def stream(self, callback: Callable[[LiveTick], None]):
        """Stream ticks to callback."""
        self._running = True

        while self._running:
            try:
                params = {'id': self.session_id} if self.session_id else {
                    'u': self.username,
                    'p': self.password,
                    'q': 'ozrates',
                    'c': ','.join(f"{s[:3]}/{s[3:]}" for s in self.SYMBOLS),
                    'f': 'csv'
                }

                async with self._session.get(self.URL, params=params) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.strip().split('\n'):
                            tick = self._parse_line(line)
                            if tick:
                                callback(tick)

                await asyncio.sleep(0.1)  # 100ms polling

            except Exception as e:
                logger.error(f"TrueFX stream error: {e}")
                await asyncio.sleep(1.0)

    def _parse_line(self, line: str) -> Optional[LiveTick]:
        """Parse TrueFX CSV line."""
        try:
            parts = line.strip().split(',')
            if len(parts) < 6:
                return None

            # Format: EUR/USD,1705432100000,1.0850,5,1.0851,5
            symbol = parts[0].replace('/', '')
            timestamp_ms = int(parts[1])
            bid = float(parts[2]) + float(parts[3]) / 100000
            ask = float(parts[4]) + float(parts[5]) / 100000

            pip_value = self.PIP_VALUES.get(symbol, 0.0001)
            spread = ask - bid
            spread_pips = spread / pip_value

            return LiveTick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2,
                spread=spread,
                spread_pips=spread_pips,
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                source=DataSource.TRUEFX
            )

        except Exception:
            return None

    def get_symbols(self) -> List[str]:
        return self.SYMBOLS.copy()


class IBMarketDataFeed:
    """
    Interactive Brokers market data feed.
    Requires IB Gateway running.
    Covers 70+ forex pairs.

    Uses a separate thread to avoid asyncio event loop conflicts.
    """

    # All forex pairs available on IB
    SYMBOLS = [
        # Majors
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        # Crosses
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD',
        'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURCAD', 'EURNZD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
        # Exotics
        'EURSEK', 'EURNOK', 'EURDKK', 'USDSEK', 'USDNOK', 'USDDKK',
        'USDSGD', 'USDMXN', 'USDZAR', 'USDTRY', 'USDHKD', 'USDCNH',
        'EURPLN', 'EURHUF', 'EURTRY', 'GBPTRY', 'SGDJPY', 'HKDJPY',
    ]

    PIP_VALUES = {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
        'USDCHF': 0.0001, 'AUDUSD': 0.0001, 'USDCAD': 0.0001,
        'NZDUSD': 0.0001, 'EURJPY': 0.01, 'GBPJPY': 0.01,
    }

    def __init__(self, host: str = 'localhost', port: int = 4004, client_id: int = 10):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None
        self._running = False
        self._subscriptions = {}
        self._callbacks = []
        self._thread = None
        self._connected = False
        self._tick_queue = []  # Thread-safe tick buffer
        self._queue_lock = threading.Lock()

    async def connect(self) -> bool:
        """Connect to IB Gateway."""
        try:
            from ib_insync import IB
            import nest_asyncio

            # Allow nested event loops (fixes "already running" error)
            try:
                nest_asyncio.apply()
            except Exception:
                pass  # Already applied or not needed

            self._ib = IB()

            # Try async connect first
            try:
                await self._ib.connectAsync(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    readonly=True
                )
                self._connected = True
                logger.info(f"IB Market Data connected to {self.host}:{self.port}")
                return True
            except Exception as e:
                # Fallback to sync connect in thread
                logger.debug(f"Async connect failed, trying sync: {e}")

                def _connect_sync():
                    try:
                        # Create new IB instance for thread
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        self._ib = IB()
                        self._ib.connect(
                            self.host,
                            self.port,
                            clientId=self.client_id,
                            readonly=True
                        )
                        self._connected = True
                        logger.info(f"IB Market Data connected to {self.host}:{self.port}")
                        return True
                    except Exception as e2:
                        logger.error(f"IB sync connect error: {e2}")
                        return False

                # Run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(_connect_sync)
                    result = future.result(timeout=10)
                return result

        except ImportError as e:
            logger.warning(f"ib_insync or nest_asyncio not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"IB connect error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from IB."""
        self._running = False
        if self._ib:
            try:
                # Cancel all subscriptions
                for contract in self._subscriptions.values():
                    try:
                        self._ib.cancelMktData(contract)
                    except Exception:
                        pass
                self._ib.disconnect()
            except Exception as e:
                logger.error(f"IB disconnect error: {e}")
            self._ib = None
        self._connected = False

    def _subscribe_sync(self, symbols: List[str]):
        """Subscribe to market data (runs in thread)."""
        if not self._ib or not self._connected:
            return

        from ib_insync import Forex

        for symbol in symbols:
            if symbol in self._subscriptions:
                continue

            try:
                # Create forex contract
                contract = Forex(symbol[:3] + symbol[3:])
                self._ib.qualifyContracts(contract)

                # Subscribe to market data
                ticker = self._ib.reqMktData(contract, '', False, False)
                self._subscriptions[symbol] = contract

                # Set up callback
                ticker.updateEvent += lambda t, sym=symbol: self._on_tick_sync(sym, t)

                logger.debug(f"IB subscribed to {symbol}")

            except Exception as e:
                logger.error(f"IB subscribe error for {symbol}: {e}")

    def _on_tick_sync(self, symbol: str, ticker):
        """Handle tick update from IB (runs in IB thread)."""
        try:
            if not hasattr(ticker, 'bid') or not hasattr(ticker, 'ask'):
                return
            if ticker.bid <= 0 or ticker.ask <= 0:
                return

            pip_value = self.PIP_VALUES.get(symbol, 0.0001 if 'JPY' not in symbol else 0.01)
            spread = ticker.ask - ticker.bid

            tick = LiveTick(
                symbol=symbol,
                bid=ticker.bid,
                ask=ticker.ask,
                mid=(ticker.bid + ticker.ask) / 2,
                spread=spread,
                spread_pips=spread / pip_value,
                timestamp=datetime.now(),
                source=DataSource.IB,
                volume=ticker.volume if hasattr(ticker, 'volume') and ticker.volume else 0
            )

            # Queue tick for async processing
            with self._queue_lock:
                self._tick_queue.append(tick)

        except Exception as e:
            logger.error(f"IB tick error for {symbol}: {e}")

    def _run_ib_loop(self, callback: Callable[[LiveTick], None]):
        """Run IB event loop in separate thread."""
        try:
            # Subscribe to all symbols
            self._subscribe_sync(self.SYMBOLS)
            logger.info(f"IB subscribed to {len(self._subscriptions)} symbols")

            # Run IB event loop
            while self._running and self._ib:
                try:
                    self._ib.sleep(0.05)  # Process IB events

                    # Process queued ticks
                    with self._queue_lock:
                        ticks = self._tick_queue.copy()
                        self._tick_queue.clear()

                    for tick in ticks:
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"IB callback error: {e}")

                except Exception as e:
                    if self._running:
                        logger.error(f"IB loop error: {e}")
                    break

        except Exception as e:
            logger.error(f"IB thread error: {e}")
        finally:
            logger.info("IB thread stopped")

    async def stream(self, callback: Callable[[LiveTick], None]):
        """Stream ticks from IB Gateway."""
        if not self._connected:
            logger.warning("IB not connected, cannot stream")
            return

        self._running = True

        # Start IB event loop in separate thread
        self._thread = threading.Thread(
            target=self._run_ib_loop,
            args=(callback,),
            daemon=True,
            name="IB-MarketData"
        )
        self._thread.start()
        logger.info("IB market data thread started")

        # Keep async task alive
        while self._running:
            await asyncio.sleep(0.5)

            # Check if thread is still alive
            if self._thread and not self._thread.is_alive():
                logger.warning("IB thread died, attempting reconnect...")
                break

    def get_symbols(self) -> List[str]:
        return self.SYMBOLS.copy()


class OANDAFeed:
    """
    OANDA streaming feed.
    Covers 28+ forex pairs with low latency.
    """

    STREAM_URL = "https://stream-fxpractice.oanda.com"
    LIVE_STREAM_URL = "https://stream-fxtrade.oanda.com"

    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD',
        'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURCAD', 'EURNZD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY',
    ]

    PIP_VALUES = {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
        'USDCHF': 0.0001, 'AUDUSD': 0.0001, 'USDCAD': 0.0001,
    }

    def __init__(self, api_key: str, account_id: str, paper: bool = False):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = self.STREAM_URL if paper else self.LIVE_STREAM_URL
        logger.info(f"OANDA feed using {'practice' if paper else 'LIVE'} endpoint")
        self._session = None
        self._running = False

    async def connect(self) -> bool:
        """Connect to OANDA streaming."""
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.warning("OANDA API key not configured")
            return False

        try:
            self._session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )
            logger.info("OANDA feed initialized")
            return True

        except Exception as e:
            logger.error(f"OANDA connect error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from OANDA."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    async def stream(self, callback: Callable[[LiveTick], None]):
        """Stream prices from OANDA."""
        self._running = True

        instruments = ','.join(f"{s[:3]}_{s[3:]}" for s in self.SYMBOLS)
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing/stream"

        while self._running:
            try:
                async with self._session.get(
                    url,
                    params={'instruments': instruments}
                ) as resp:
                    async for line in resp.content:
                        if not self._running:
                            break

                        tick = self._parse_line(line.decode())
                        if tick:
                            callback(tick)

            except Exception as e:
                logger.error(f"OANDA stream error: {e}")
                await asyncio.sleep(5.0)

    def _parse_line(self, line: str) -> Optional[LiveTick]:
        """Parse OANDA streaming line."""
        try:
            import json
            data = json.loads(line)

            if data.get('type') != 'PRICE':
                return None

            symbol = data['instrument'].replace('_', '')
            bids = data.get('bids', [{}])
            asks = data.get('asks', [{}])

            if not bids or not asks:
                return None

            bid = float(bids[0].get('price', 0))
            ask = float(asks[0].get('price', 0))

            pip_value = self.PIP_VALUES.get(symbol, 0.0001 if 'JPY' not in symbol else 0.01)
            spread = ask - bid

            return LiveTick(
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2,
                spread=spread,
                spread_pips=spread / pip_value,
                timestamp=datetime.now(),
                source=DataSource.OANDA
            )

        except Exception:
            return None

    def get_symbols(self) -> List[str]:
        return self.SYMBOLS.copy()


class MultiSourceFeed:
    """
    Aggregates live data from ALL available sources.

    Features:
    - Best quote selection (lowest spread)
    - Automatic failover
    - Symbol coverage maximization
    - Latency tracking
    """

    def __init__(
        self,
        truefx_username: str = 'demo',
        truefx_password: str = 'demo',
        ib_host: str = 'localhost',
        ib_port: int = 4004,
        oanda_api_key: str = '',
        oanda_account_id: str = '',
        oanda_paper: bool = False,
        enable_truefx: bool = True,
        enable_ib: bool = True,
        enable_oanda: bool = True,
    ):
        """
        Initialize multi-source feed.

        Args:
            truefx_username: TrueFX username
            truefx_password: TrueFX password
            ib_host: IB Gateway host
            ib_port: IB Gateway port
            oanda_api_key: OANDA API key
            oanda_account_id: OANDA account ID
            oanda_paper: Use OANDA practice/demo account
            enable_truefx: Enable TrueFX feed
            enable_ib: Enable IB feed
            enable_oanda: Enable OANDA feed
        """
        self.feeds: Dict[DataSource, Any] = {}
        self._running = False
        self._lock = threading.RLock()

        # Latest ticks by symbol
        self._latest_ticks: Dict[str, Dict[DataSource, LiveTick]] = defaultdict(dict)

        # Best quotes by symbol
        self._best_quotes: Dict[str, LiveTick] = {}

        # Coverage tracking
        self._coverage: Dict[str, SymbolCoverage] = {}

        # Callbacks
        self._callbacks: List[Callable[[LiveTick], None]] = []

        # Statistics
        self._tick_counts: Dict[DataSource, int] = defaultdict(int)
        self._last_tick_time: Dict[DataSource, datetime] = {}

        # Initialize feeds
        if enable_truefx:
            self.feeds[DataSource.TRUEFX] = TrueFXFeed(truefx_username, truefx_password)

        if enable_ib:
            self.feeds[DataSource.IB] = IBMarketDataFeed(ib_host, ib_port)

        if enable_oanda and oanda_api_key:
            self.feeds[DataSource.OANDA] = OANDAFeed(oanda_api_key, oanda_account_id, paper=oanda_paper)

    async def connect(self) -> Dict[DataSource, bool]:
        """Connect to all feeds."""
        results = {}

        for source, feed in self.feeds.items():
            try:
                success = await feed.connect()
                results[source] = success

                if success:
                    # Register symbols
                    for symbol in feed.get_symbols():
                        if symbol not in self._coverage:
                            self._coverage[symbol] = SymbolCoverage(symbol=symbol)
                        self._coverage[symbol].sources.add(source)

                    logger.info(f"{source.value} connected: {len(feed.get_symbols())} symbols")

            except Exception as e:
                logger.error(f"Failed to connect {source.value}: {e}")
                results[source] = False

        # Set primary sources (prefer TrueFX for majors, IB for exotics)
        self._set_primary_sources()

        return results

    def _set_primary_sources(self):
        """Set primary data source for each symbol."""
        # Priority: TrueFX > OANDA > IB (for latency)
        priority = [DataSource.TRUEFX, DataSource.OANDA, DataSource.IB]

        for symbol, coverage in self._coverage.items():
            for source in priority:
                if source in coverage.sources:
                    coverage.primary_source = source
                    break

    async def disconnect(self):
        """Disconnect all feeds."""
        self._running = False

        for source, feed in self.feeds.items():
            try:
                await feed.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {source.value}: {e}")

    def register_callback(self, callback: Callable[[LiveTick], None]):
        """Register callback for tick updates."""
        self._callbacks.append(callback)

    def _on_tick(self, tick: LiveTick):
        """Handle tick from any source."""
        with self._lock:
            # Store tick
            self._latest_ticks[tick.symbol][tick.source] = tick
            self._tick_counts[tick.source] += 1
            self._last_tick_time[tick.source] = datetime.now()

            # Update coverage
            if tick.symbol in self._coverage:
                self._coverage[tick.symbol].tick_count += 1
                self._coverage[tick.symbol].last_tick = tick

            # Determine best quote
            best = self._get_best_quote(tick.symbol)
            if best:
                self._best_quotes[tick.symbol] = best

                # Notify callbacks with best quote
                for callback in self._callbacks:
                    try:
                        callback(best)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

    def _get_best_quote(self, symbol: str) -> Optional[LiveTick]:
        """Get best quote (lowest spread) for symbol."""
        ticks = self._latest_ticks.get(symbol, {})
        if not ticks:
            return None

        valid_ticks = [t for t in ticks.values() if t.is_valid]
        if not valid_ticks:
            return None

        # Sort by spread
        return min(valid_ticks, key=lambda t: t.spread_pips)

    async def stream(self, symbols: Optional[List[str]] = None):
        """
        Stream ticks from all sources.

        Args:
            symbols: Optional list to filter. If None, streams all available.
        """
        self._running = True

        # Start all feed streams
        tasks = []
        for source, feed in self.feeds.items():
            task = asyncio.create_task(
                feed.stream(self._on_tick),
                name=f"feed_{source.value}"
            )
            tasks.append(task)

        logger.info(f"Started {len(tasks)} data feeds")

        try:
            # Wait for all feeds (they run forever until stopped)
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def get_quote(self, symbol: str) -> Optional[LiveTick]:
        """Get latest best quote for symbol."""
        return self._best_quotes.get(symbol)

    def get_all_quotes(self, symbol: str) -> Dict[DataSource, LiveTick]:
        """Get quotes from all sources for symbol."""
        return dict(self._latest_ticks.get(symbol, {}))

    def get_coverage(self) -> Dict[str, SymbolCoverage]:
        """Get symbol coverage information."""
        return dict(self._coverage)

    def get_all_symbols(self) -> List[str]:
        """Get all covered symbols."""
        return list(self._coverage.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get feed statistics."""
        return {
            'feeds_active': len([f for f in self.feeds.values()]),
            'total_symbols': len(self._coverage),
            'tick_counts': dict(self._tick_counts),
            'last_tick_times': {
                k.value: v.isoformat()
                for k, v in self._last_tick_time.items()
            },
            'coverage_by_source': {
                source.value: sum(
                    1 for c in self._coverage.values()
                    if source in c.sources
                )
                for source in DataSource
            }
        }

    def print_coverage(self):
        """Print coverage summary."""
        print("\n" + "=" * 70)
        print("MULTI-SOURCE DATA FEED COVERAGE")
        print("=" * 70)

        # By source
        print("\nSOURCES:")
        for source in [DataSource.TRUEFX, DataSource.IB, DataSource.OANDA]:
            if source in self.feeds:
                count = sum(1 for c in self._coverage.values() if source in c.sources)
                ticks = self._tick_counts.get(source, 0)
                status = "ACTIVE" if ticks > 0 else "CONNECTED"
                print(f"  {source.value:12} : {count:3} symbols ({status}, {ticks} ticks)")

        # Total coverage
        print(f"\nTOTAL SYMBOLS: {len(self._coverage)}")

        # List symbols by coverage
        multi_source = [s for s, c in self._coverage.items() if len(c.sources) > 1]
        single_source = [s for s, c in self._coverage.items() if len(c.sources) == 1]

        print(f"\nMulti-source coverage ({len(multi_source)}): {', '.join(sorted(multi_source)[:10])}...")
        print(f"Single-source coverage ({len(single_source)}): {', '.join(sorted(single_source)[:10])}...")

        print("=" * 70 + "\n")


def create_multi_source_feed(
    enable_truefx: bool = True,
    enable_ib: bool = True,
    enable_oanda: bool = True,
) -> MultiSourceFeed:
    """
    Factory function to create multi-source feed from environment.

    Reads credentials from environment variables:
    - TRUEFX_USERNAME, TRUEFX_PASSWORD
    - IB_HOST, IB_PORT
    - OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_PAPER
    """
    import os
    from dotenv import load_dotenv

    # Load .env file
    load_dotenv()

    # Check OANDA paper mode
    oanda_paper = os.getenv('OANDA_PAPER', 'true').lower() in ('true', '1', 'yes')

    return MultiSourceFeed(
        truefx_username=os.getenv('TRUEFX_USERNAME', 'demo'),
        truefx_password=os.getenv('TRUEFX_PASSWORD', 'demo'),
        ib_host=os.getenv('IB_HOST', 'localhost'),
        ib_port=int(os.getenv('IB_PORT', 4004)),
        oanda_api_key=os.getenv('OANDA_API_KEY', ''),
        oanda_account_id=os.getenv('OANDA_ACCOUNT_ID', ''),
        oanda_paper=oanda_paper,
        enable_truefx=enable_truefx,
        enable_ib=enable_ib,
        enable_oanda=enable_oanda,
    )
