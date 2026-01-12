#!/usr/bin/env python3
"""
All available forex pairs from Dukascopy.
Dukascopy provides 100+ forex pairs.
"""

# Major pairs (G10 currencies)
MAJOR_PAIRS = [
    'EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
]

# Cross pairs (major currencies without USD)
CROSS_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY'
]

# Exotic pairs (major vs emerging market currencies)
EXOTIC_PAIRS = [
    # EUR exotics
    'EURHUF', 'EURPLN', 'EURCZK', 'EURNOK', 'EURSEK', 'EURDKK',
    'EURTRY', 'EURZAR', 'EURMXN', 'EURSGD', 'EURHKD',

    # USD exotics
    'USDHUF', 'USDPLN', 'USDCZK', 'USDNOK', 'USDSEK', 'USDDKK',
    'USDTRY', 'USDZAR', 'USDMXN', 'USDSGD', 'USDHKD', 'USDCNH',
    'USDRUB', 'USDINR', 'USDTHB', 'USDIDR', 'USDKRW',

    # GBP exotics
    'GBPNOK', 'GBPSEK', 'GBPDKK', 'GBPZAR', 'GBPSGD', 'GBPHKD',

    # AUD exotics
    'AUDSGD', 'AUDHKD',

    # NZD exotics
    'NZDSGD', 'NZDHKD',

    # CHF exotics
    'CHFNOK', 'CHFSEK', 'CHFDKK', 'CHFZAR', 'CHFSGD', 'CHFHKD',

    # JPY exotics
    'JPYNOK', 'JPYSEK', 'JPYSGD',

    # CAD exotics
    'CADNOK', 'CADSEK', 'CADSGD', 'CADHKD',

    # Scandinavian crosses
    'NOKSEK', 'NOKDKK', 'SEKDKK',

    # Other
    'SGDHKD', 'SGDJPY', 'HKDJPY'
]

# All pairs combined
ALL_PAIRS = MAJOR_PAIRS + CROSS_PAIRS + EXOTIC_PAIRS

# Organized by region
AMERICAS = [
    'USDCAD', 'USDMXN', 'EURCAD', 'GBPCAD', 'AUDCAD', 'NZDCAD',
    'CADJPY', 'CADCHF', 'CADNOK', 'CADSEK', 'CADSGD', 'CADHKD'
]

EUROPE = [
    'EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
    'EURHUF', 'EURPLN', 'EURCZK', 'EURNOK', 'EURSEK', 'EURDKK',
    'EURTRY', 'EURZAR', 'EURSGD', 'EURHKD', 'EURMXN',
    'GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
    'GBPNOK', 'GBPSEK', 'GBPDKK', 'GBPZAR', 'GBPSGD', 'GBPHKD',
    'USDCHF', 'CHFJPY', 'CHFNOK', 'CHFSEK', 'CHFDKK', 'CHFZAR', 'CHFSGD', 'CHFHKD',
    'NOKSEK', 'NOKDKK', 'SEKDKK',
    'USDPLN', 'USDHUF', 'USDCZK', 'USDNOK', 'USDSEK', 'USDDKK', 'USDTRY', 'USDRUB'
]

ASIA_PACIFIC = [
    'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCNH', 'USDSGD', 'USDHKD',
    'USDTHB', 'USDIDR', 'USDKRW', 'USDINR',
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'JPYNOK', 'JPYSEK', 'JPYSGD',
    'AUDCHF', 'AUDCAD', 'AUDNZD', 'AUDSGD', 'AUDHKD',
    'NZDCHF', 'NZDCAD', 'NZDSGD', 'NZDHKD',
    'SGDHKD', 'SGDJPY', 'HKDJPY'
]

AFRICA = [
    'USDZAR', 'EURZAR', 'GBPZAR', 'CHFZAR'
]

# Filter by liquidity
HIGH_LIQUIDITY = MAJOR_PAIRS  # 7 pairs
MEDIUM_LIQUIDITY = CROSS_PAIRS  # ~20 pairs
LOW_LIQUIDITY = EXOTIC_PAIRS  # ~60+ pairs

def get_pairs(category='all'):
    """Get list of pairs by category."""
    if category == 'all':
        return ALL_PAIRS
    elif category == 'major':
        return MAJOR_PAIRS
    elif category == 'cross':
        return CROSS_PAIRS
    elif category == 'exotic':
        return EXOTIC_PAIRS
    elif category == 'high_liquidity':
        return HIGH_LIQUIDITY
    elif category == 'medium_liquidity':
        return MEDIUM_LIQUIDITY
    elif category == 'low_liquidity':
        return LOW_LIQUIDITY
    elif category == 'americas':
        return AMERICAS
    elif category == 'europe':
        return EUROPE
    elif category == 'asia':
        return ASIA_PACIFIC
    elif category == 'africa':
        return AFRICA
    else:
        return ALL_PAIRS

if __name__ == "__main__":
    print(f"Total pairs: {len(ALL_PAIRS)}")
    print(f"Major: {len(MAJOR_PAIRS)}")
    print(f"Cross: {len(CROSS_PAIRS)}")
    print(f"Exotic: {len(EXOTIC_PAIRS)}")
    print(f"\nAll pairs:")
    for i, pair in enumerate(ALL_PAIRS, 1):
        print(f"  {i:3d}. {pair}")
