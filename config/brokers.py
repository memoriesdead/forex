"""
Multi-Broker Configuration
==========================
Professional-grade broker configuration for multi-exchange trading.

Supported Brokers:
1. Interactive Brokers (IB Gateway) - PRIMARY
2. OANDA v20 API
3. Forex.com (GAIN Capital)
4. tastyfx (IG Group)
5. IG Markets

Environment Variables Required:
- IB_HOST, IB_PORT, IB_ACCOUNT_ID, IB_CLIENT_ID
- OANDA_API_KEY, OANDA_ACCOUNT_ID
- FOREXCOM_USERNAME, FOREXCOM_PASSWORD, FOREXCOM_APP_KEY
- TASTYFX_API_KEY, TASTYFX_USERNAME, TASTYFX_PASSWORD
- IG_API_KEY, IG_USERNAME, IG_PASSWORD
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class BrokerName(Enum):
    """Broker identifiers."""
    IB = "ib"
    OANDA = "oanda"
    FOREX_COM = "forex_com"
    TASTYFX = "tastyfx"
    IG = "ig"


@dataclass
class BrokerCredentials:
    """Broker credentials container."""
    name: BrokerName
    enabled: bool = True
    paper: bool = True

    # Connection
    host: str = ""
    port: int = 0
    api_key: str = ""
    api_secret: str = ""
    account_id: str = ""
    username: str = ""
    password: str = ""
    client_id: int = 1

    # Trading
    symbols: List[str] = field(default_factory=list)
    priority: int = 0
    max_position_size: float = 100000.0
    max_daily_trades: int = 100

    # Rate limits
    max_orders_per_second: float = 10.0
    max_requests_per_minute: int = 120


# ==================== Default Configurations ====================

DEFAULT_IB_CONFIG = {
    'name': BrokerName.IB,
    'host': 'localhost',
    'port': 4004,  # Paper trading port
    'account_id': 'DUO423364',
    'client_id': 1,
    'paper': True,
    'enabled': True,
    'priority': 1,  # Highest priority
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD'
    ],
}

DEFAULT_OANDA_CONFIG = {
    'name': BrokerName.OANDA,
    'api_key': '',  # Set via OANDA_API_KEY
    'account_id': '',  # Set via OANDA_ACCOUNT_ID
    'paper': True,
    'enabled': False,  # Disabled until API key provided
    'priority': 2,
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD',
        'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURCAD', 'EURNZD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY'
    ],
}

DEFAULT_FOREX_COM_CONFIG = {
    'name': BrokerName.FOREX_COM,
    'username': '',  # Set via FOREXCOM_USERNAME
    'password': '',  # Set via FOREXCOM_PASSWORD
    'api_secret': '',  # App key via FOREXCOM_APP_KEY
    'paper': True,
    'enabled': False,
    'priority': 3,
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY'
    ],
}

DEFAULT_TASTYFX_CONFIG = {
    'name': BrokerName.TASTYFX,
    'api_key': '',  # Set via TASTYFX_API_KEY
    'username': '',  # Set via TASTYFX_USERNAME
    'password': '',  # Set via TASTYFX_PASSWORD
    'paper': True,
    'enabled': False,
    'priority': 4,
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD'
    ],
}

DEFAULT_IG_CONFIG = {
    'name': BrokerName.IG,
    'api_key': '',  # Set via IG_API_KEY
    'username': '',  # Set via IG_USERNAME
    'password': '',  # Set via IG_PASSWORD
    'paper': True,
    'enabled': False,
    'priority': 5,
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'EURGBP', 'EURCHF', 'AUDJPY', 'EURAUD', 'GBPAUD'
    ],
}


def load_broker_config_from_env() -> Dict[str, Dict[str, Any]]:
    """
    Load broker configuration from environment variables.

    Returns:
        Dict of broker configurations ready for BrokerRouter
    """
    configs = {}

    # Interactive Brokers
    ib_config = dict(DEFAULT_IB_CONFIG)
    ib_config['host'] = os.getenv('IB_HOST', 'localhost')
    ib_config['port'] = int(os.getenv('IB_PORT', 4004))
    ib_config['account_id'] = os.getenv('IB_ACCOUNT_ID', 'DUO423364')
    ib_config['client_id'] = int(os.getenv('IB_CLIENT_ID', 1))
    ib_config['paper'] = os.getenv('IB_PAPER', 'true').lower() == 'true'
    configs['ib'] = ib_config

    # OANDA
    oanda_key = os.getenv('OANDA_API_KEY', os.getenv('OANDA_PRACTICE_API_KEY', ''))
    oanda_account = os.getenv('OANDA_ACCOUNT_ID', os.getenv('OANDA_PRACTICE_ACCOUNT_ID', ''))
    if oanda_key and oanda_account and oanda_key != 'your_practice_key_here':
        oanda_config = dict(DEFAULT_OANDA_CONFIG)
        oanda_config['api_key'] = oanda_key
        oanda_config['account_id'] = oanda_account
        oanda_config['paper'] = os.getenv('OANDA_PAPER', 'true').lower() == 'true'
        oanda_config['enabled'] = True
        configs['oanda'] = oanda_config

    # Forex.com
    fc_username = os.getenv('FOREXCOM_USERNAME', '')
    fc_password = os.getenv('FOREXCOM_PASSWORD', '')
    fc_app_key = os.getenv('FOREXCOM_APP_KEY', '')
    if fc_username and fc_password and fc_app_key:
        fc_config = dict(DEFAULT_FOREX_COM_CONFIG)
        fc_config['username'] = fc_username
        fc_config['password'] = fc_password
        fc_config['app_key'] = fc_app_key
        fc_config['paper'] = os.getenv('FOREXCOM_PAPER', 'true').lower() == 'true'
        fc_config['enabled'] = True
        configs['forex_com'] = fc_config

    # tastyfx
    tfx_api_key = os.getenv('TASTYFX_API_KEY', '')
    tfx_username = os.getenv('TASTYFX_USERNAME', '')
    tfx_password = os.getenv('TASTYFX_PASSWORD', '')
    if tfx_api_key and tfx_username and tfx_password:
        tfx_config = dict(DEFAULT_TASTYFX_CONFIG)
        tfx_config['api_key'] = tfx_api_key
        tfx_config['username'] = tfx_username
        tfx_config['password'] = tfx_password
        tfx_config['paper'] = os.getenv('TASTYFX_PAPER', 'true').lower() == 'true'
        tfx_config['enabled'] = True
        configs['tastyfx'] = tfx_config

    # IG Markets
    ig_api_key = os.getenv('IG_API_KEY', '')
    ig_username = os.getenv('IG_USERNAME', '')
    ig_password = os.getenv('IG_PASSWORD', '')
    if ig_api_key and ig_username and ig_password:
        ig_config = dict(DEFAULT_IG_CONFIG)
        ig_config['api_key'] = ig_api_key
        ig_config['username'] = ig_username
        ig_config['password'] = ig_password
        ig_config['paper'] = os.getenv('IG_PAPER', 'true').lower() == 'true'
        ig_config['enabled'] = True
        configs['ig'] = ig_config

    return configs


def get_enabled_brokers() -> List[str]:
    """Get list of enabled broker names."""
    configs = load_broker_config_from_env()
    return [name for name, cfg in configs.items() if cfg.get('enabled', False)]


def get_broker_summary() -> Dict[str, str]:
    """Get summary of broker configuration status."""
    configs = load_broker_config_from_env()

    summary = {}
    for name, cfg in configs.items():
        enabled = cfg.get('enabled', False)
        paper = cfg.get('paper', True)

        if enabled:
            mode = 'PAPER' if paper else 'LIVE'
            summary[name] = f"ENABLED ({mode})"
        else:
            summary[name] = "DISABLED (no credentials)"

    return summary


def print_broker_status():
    """Print broker configuration status."""
    print("\n" + "=" * 60)
    print("MULTI-BROKER CONFIGURATION STATUS")
    print("=" * 60)

    configs = load_broker_config_from_env()

    for name, cfg in configs.items():
        enabled = cfg.get('enabled', False)
        paper = cfg.get('paper', True)
        priority = cfg.get('priority', 0)

        status = "ENABLED" if enabled else "DISABLED"
        mode = "PAPER" if paper else "LIVE"
        symbols = len(cfg.get('symbols', []))

        print(f"\n{name.upper()}")
        print(f"  Status:   {status}")
        print(f"  Mode:     {mode}")
        print(f"  Priority: {priority}")
        print(f"  Symbols:  {symbols} pairs")

        if name == 'ib':
            print(f"  Host:     {cfg.get('host')}:{cfg.get('port')}")
            print(f"  Account:  {cfg.get('account_id')}")

    print("\n" + "=" * 60)

    enabled_count = sum(1 for cfg in configs.values() if cfg.get('enabled', False))
    print(f"Total: {enabled_count}/{len(configs)} brokers enabled")
    print("=" * 60 + "\n")


# ==================== Environment Template ====================

ENV_TEMPLATE = """
# =============================================================================
# MULTI-BROKER FOREX TRADING CONFIGURATION
# =============================================================================
# Add these to your .env file

# -----------------------------------------------------------------------------
# Interactive Brokers (IB Gateway)
# -----------------------------------------------------------------------------
# Docker container: ibgateway
# Paper: port 4004, Live: port 4001
IB_HOST=localhost
IB_PORT=4004
IB_ACCOUNT_ID=DUO423364
IB_CLIENT_ID=1
IB_PAPER=true

# -----------------------------------------------------------------------------
# OANDA v20 API
# Get keys from: https://www.oanda.com/demo-account/
# -----------------------------------------------------------------------------
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=101-001-xxxxxxx-001
OANDA_PAPER=true

# -----------------------------------------------------------------------------
# Forex.com (GAIN Capital)
# Get keys from: https://developer.forex.com/
# -----------------------------------------------------------------------------
FOREXCOM_USERNAME=your_username
FOREXCOM_PASSWORD=your_password
FOREXCOM_APP_KEY=your_app_key
FOREXCOM_PAPER=true

# -----------------------------------------------------------------------------
# tastyfx (IG Group US)
# Get keys from: https://labs.ig.com/
# -----------------------------------------------------------------------------
TASTYFX_API_KEY=your_api_key
TASTYFX_USERNAME=your_username
TASTYFX_PASSWORD=your_password
TASTYFX_PAPER=true

# -----------------------------------------------------------------------------
# IG Markets (International)
# Get keys from: https://labs.ig.com/
# -----------------------------------------------------------------------------
IG_API_KEY=your_api_key
IG_USERNAME=your_username
IG_PASSWORD=your_password
IG_PAPER=true
"""


def print_env_template():
    """Print environment variable template."""
    print(ENV_TEMPLATE)


if __name__ == "__main__":
    print_broker_status()
