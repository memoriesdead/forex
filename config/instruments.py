"""
Forex Trading Instruments Configuration
=======================================
Spot Forex, Currency Futures, Micro Futures, and FX Options

Updated: 2026-01-23
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class InstrumentType(Enum):
    SPOT_FOREX = "spot"
    MICRO_FUTURES = "micro_futures"
    CURRENCY_FUTURES = "futures"
    FX_OPTIONS = "options"
    WEEKLY_OPTIONS = "weekly_options"


@dataclass
class InstrumentConfig:
    """Configuration for a trading instrument"""
    symbol: str
    instrument_type: InstrumentType
    exchange: str
    sec_type: str
    currency: str
    contract_size: float
    tick_size: float
    tick_value: float
    margin_requirement: float  # Approximate initial margin
    leverage: float
    description: str


# =============================================================================
# SPOT FOREX (IDEALPRO)
# =============================================================================
SPOT_FOREX: Dict[str, InstrumentConfig] = {
    "EURUSD": InstrumentConfig(
        symbol="EUR", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="USD",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="Euro/US Dollar"
    ),
    "GBPUSD": InstrumentConfig(
        symbol="GBP", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="USD",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="British Pound/US Dollar"
    ),
    "USDJPY": InstrumentConfig(
        symbol="USD", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="JPY",
        contract_size=100000, tick_size=0.001, tick_value=1000,
        margin_requirement=2000, leverage=50, description="US Dollar/Japanese Yen"
    ),
    "AUDUSD": InstrumentConfig(
        symbol="AUD", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="USD",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="Australian Dollar/US Dollar"
    ),
    "USDCAD": InstrumentConfig(
        symbol="USD", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="CAD",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="US Dollar/Canadian Dollar"
    ),
    "USDCHF": InstrumentConfig(
        symbol="USD", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="CHF",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="US Dollar/Swiss Franc"
    ),
    "NZDUSD": InstrumentConfig(
        symbol="NZD", instrument_type=InstrumentType.SPOT_FOREX,
        exchange="IDEALPRO", sec_type="CASH", currency="USD",
        contract_size=100000, tick_size=0.00001, tick_value=1.0,
        margin_requirement=2000, leverage=50, description="New Zealand Dollar/US Dollar"
    ),
}

# =============================================================================
# MICRO FX FUTURES (CME) - RECOMMENDED FOR SMALL ACCOUNTS
# =============================================================================
MICRO_FUTURES: Dict[str, InstrumentConfig] = {
    "M6E": InstrumentConfig(
        symbol="M6E", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=12500, tick_size=0.0001, tick_value=1.25,
        margin_requirement=250, leverage=25, description="Micro EUR/USD Futures"
    ),
    "M6B": InstrumentConfig(
        symbol="M6B", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=6250, tick_size=0.0001, tick_value=0.625,
        margin_requirement=200, leverage=25, description="Micro GBP/USD Futures"
    ),
    "M6J": InstrumentConfig(
        symbol="M6J", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=1250000, tick_size=0.000001, tick_value=1.25,
        margin_requirement=200, leverage=25, description="Micro USD/JPY Futures"
    ),
    "M6A": InstrumentConfig(
        symbol="M6A", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=10000, tick_size=0.0001, tick_value=1.0,
        margin_requirement=150, leverage=25, description="Micro AUD/USD Futures"
    ),
    "M6C": InstrumentConfig(
        symbol="M6C", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=10000, tick_size=0.0001, tick_value=1.0,
        margin_requirement=150, leverage=25, description="Micro CAD/USD Futures"
    ),
    "M6S": InstrumentConfig(
        symbol="M6S", instrument_type=InstrumentType.MICRO_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=12500, tick_size=0.0001, tick_value=1.25,
        margin_requirement=150, leverage=25, description="Micro CHF/USD Futures"
    ),
}

# =============================================================================
# CURRENCY FUTURES (CME) - STANDARD SIZE
# =============================================================================
CURRENCY_FUTURES: Dict[str, InstrumentConfig] = {
    "6E": InstrumentConfig(
        symbol="6E", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=125000, tick_size=0.0001, tick_value=12.50,
        margin_requirement=2500, leverage=25, description="Euro FX Futures"
    ),
    "6B": InstrumentConfig(
        symbol="6B", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=62500, tick_size=0.0001, tick_value=6.25,
        margin_requirement=2000, leverage=25, description="British Pound Futures"
    ),
    "6J": InstrumentConfig(
        symbol="6J", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=12500000, tick_size=0.0000001, tick_value=12.50,
        margin_requirement=2500, leverage=25, description="Japanese Yen Futures"
    ),
    "6A": InstrumentConfig(
        symbol="6A", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=100000, tick_size=0.0001, tick_value=10.0,
        margin_requirement=1500, leverage=25, description="Australian Dollar Futures"
    ),
    "6C": InstrumentConfig(
        symbol="6C", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=100000, tick_size=0.0001, tick_value=10.0,
        margin_requirement=1500, leverage=25, description="Canadian Dollar Futures"
    ),
    "6S": InstrumentConfig(
        symbol="6S", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=125000, tick_size=0.0001, tick_value=12.50,
        margin_requirement=2500, leverage=25, description="Swiss Franc Futures"
    ),
    "6N": InstrumentConfig(
        symbol="6N", instrument_type=InstrumentType.CURRENCY_FUTURES,
        exchange="CME", sec_type="FUT", currency="USD",
        contract_size=100000, tick_size=0.0001, tick_value=10.0,
        margin_requirement=1500, leverage=25, description="New Zealand Dollar Futures"
    ),
}

# =============================================================================
# FUTURES CONTRACT MONTHS
# =============================================================================
FUTURES_MONTHS = {
    "H": "March",
    "M": "June",
    "U": "September",
    "Z": "December"
}

def get_front_month() -> str:
    """Get the front month code for futures"""
    from datetime import datetime
    month = datetime.now().month
    if month <= 3:
        return "H"
    elif month <= 6:
        return "M"
    elif month <= 9:
        return "U"
    else:
        return "Z"

def get_futures_expiry(year: int = None, month_code: str = None) -> str:
    """Get futures expiry string (e.g., '202503' for March 2025)"""
    from datetime import datetime
    if year is None:
        year = datetime.now().year
    if month_code is None:
        month_code = get_front_month()

    month_map = {"H": "03", "M": "06", "U": "09", "Z": "12"}
    return f"{year}{month_map[month_code]}"


# =============================================================================
# SPOT TO FUTURES MAPPING
# =============================================================================
SPOT_TO_MICRO_FUTURES = {
    "EURUSD": "M6E",
    "GBPUSD": "M6B",
    "USDJPY": "M6J",
    "AUDUSD": "M6A",
    "USDCAD": "M6C",
    "USDCHF": "M6S",
}

SPOT_TO_FUTURES = {
    "EURUSD": "6E",
    "GBPUSD": "6B",
    "USDJPY": "6J",
    "AUDUSD": "6A",
    "USDCAD": "6C",
    "USDCHF": "6S",
    "NZDUSD": "6N",
}


# =============================================================================
# INSTRUMENT SELECTION HELPERS
# =============================================================================
def get_best_instrument(symbol: str, capital: float, trade_type: str = "day") -> InstrumentConfig:
    """
    Select the best instrument based on capital and trade type

    Args:
        symbol: Forex pair (e.g., 'EURUSD')
        capital: Available trading capital
        trade_type: 'scalp', 'day', 'swing', 'event'

    Returns:
        InstrumentConfig for the recommended instrument
    """
    # For small capital, prefer micro futures for swing, spot for scalp
    if capital < 500:
        if trade_type in ["swing", "position"]:
            if symbol in SPOT_TO_MICRO_FUTURES:
                return MICRO_FUTURES[SPOT_TO_MICRO_FUTURES[symbol]]
        return SPOT_FOREX.get(symbol)

    # For medium capital, can use either
    elif capital < 5000:
        if trade_type == "swing":
            if symbol in SPOT_TO_MICRO_FUTURES:
                return MICRO_FUTURES[SPOT_TO_MICRO_FUTURES[symbol]]
        return SPOT_FOREX.get(symbol)

    # For larger capital, full futures available
    else:
        if trade_type == "swing":
            if symbol in SPOT_TO_FUTURES:
                return CURRENCY_FUTURES[SPOT_TO_FUTURES[symbol]]
        return SPOT_FOREX.get(symbol)


def get_all_instruments() -> Dict[str, InstrumentConfig]:
    """Get all available instruments"""
    all_instruments = {}
    all_instruments.update(SPOT_FOREX)
    all_instruments.update(MICRO_FUTURES)
    all_instruments.update(CURRENCY_FUTURES)
    return all_instruments


def print_instruments_summary():
    """Print a summary of all available instruments"""
    print("\n" + "="*70)
    print("FOREX TRADING INSTRUMENTS")
    print("="*70)

    print("\nSPOT FOREX (IDEALPRO):")
    print("-"*70)
    for symbol, config in SPOT_FOREX.items():
        print(f"  {symbol:10} | Size: {config.contract_size:>10,} | Margin: ${config.margin_requirement:>6,} | {config.description}")

    print("\nMICRO FX FUTURES (CME):")
    print("-"*70)
    for symbol, config in MICRO_FUTURES.items():
        print(f"  {symbol:10} | Size: {config.contract_size:>10,} | Margin: ${config.margin_requirement:>6,} | {config.description}")

    print("\nCURRENCY FUTURES (CME):")
    print("-"*70)
    for symbol, config in CURRENCY_FUTURES.items():
        print(f"  {symbol:10} | Size: {config.contract_size:>10,} | Margin: ${config.margin_requirement:>6,} | {config.description}")

    print("\n" + "="*70)


if __name__ == "__main__":
    print_instruments_summary()
