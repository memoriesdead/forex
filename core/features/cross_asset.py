"""
Cross-Asset Signals for Forex Trading
======================================
Signals derived from related markets that affect forex.

Key Cross-Asset Relationships:
1. DXY (Dollar Index) - Inverse to EUR/USD, GBP/USD
2. VIX - Risk-off = USD strength
3. SPX - Risk-on = USD weakness
4. Gold - Safe haven, correlates with JPY/CHF
5. Oil - CAD, NOK correlation
6. Bonds - Yield differentials drive rates

These are what institutional traders actually use.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetSignal:
    """Single cross-asset signal."""
    name: str
    value: float  # Signal strength (-1 to 1)
    confidence: float  # 0 to 1
    related_asset: str
    description: str


class CrossAssetSignalGenerator:
    """
    Generate forex signals from cross-asset relationships.

    Usage:
        generator = CrossAssetSignalGenerator()
        signals = generator.generate_all_signals(forex_data, cross_asset_data)
    """

    # Asset correlations (positive = same direction, negative = inverse)
    CORRELATIONS = {
        # EUR/USD correlations
        'EURUSD': {
            'DXY': -0.95,    # Strong inverse
            'SPX': 0.30,     # Risk-on positive
            'VIX': -0.35,    # Risk-off negative
            'GOLD': 0.40,    # Both anti-USD
            'EURGBP': 0.60,  # Related pair
        },
        # GBP/USD correlations
        'GBPUSD': {
            'DXY': -0.90,
            'SPX': 0.25,
            'VIX': -0.30,
            'GOLD': 0.35,
            'EURUSD': 0.85,
        },
        # USD/JPY correlations
        'USDJPY': {
            'DXY': 0.60,
            'SPX': 0.50,     # Risk-on, JPY weakens
            'VIX': -0.55,    # Risk-off, JPY strengthens
            'US10Y': 0.70,   # Rate differential
            'NIKKEI': 0.45,
        },
        # USD/CHF correlations
        'USDCHF': {
            'DXY': 0.85,
            'VIX': 0.40,     # CHF safe haven
            'GOLD': -0.50,   # Both safe havens
            'EURUSD': -0.90,
        },
        # USD/CAD correlations
        'USDCAD': {
            'DXY': 0.70,
            'OIL': -0.60,    # Canada is oil exporter
            'SPX': -0.25,
        },
        # AUD/USD correlations
        'AUDUSD': {
            'DXY': -0.80,
            'SPX': 0.45,     # Risk-on currency
            'VIX': -0.50,
            'COPPER': 0.55,  # Commodity exporter
            'CHINA_PMI': 0.40,
        },
    }

    def __init__(self):
        self.signals: Dict[str, CrossAssetSignal] = {}

    def generate_all_signals(self,
                             forex_data: pd.DataFrame,
                             cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate all cross-asset signals.

        Args:
            forex_data: Forex OHLCV data with pair column or pair-named columns
            cross_asset_data: Dict of DataFrames for DXY, VIX, SPX, etc.

        Returns:
            DataFrame with cross-asset signals added
        """
        df = forex_data.copy()

        # DXY signals
        if cross_asset_data and 'DXY' in cross_asset_data:
            df = self._add_dxy_signals(df, cross_asset_data['DXY'])

        # VIX signals
        if cross_asset_data and 'VIX' in cross_asset_data:
            df = self._add_vix_signals(df, cross_asset_data['VIX'])

        # SPX signals
        if cross_asset_data and 'SPX' in cross_asset_data:
            df = self._add_spx_signals(df, cross_asset_data['SPX'])

        # Gold signals
        if cross_asset_data and 'GOLD' in cross_asset_data:
            df = self._add_gold_signals(df, cross_asset_data['GOLD'])

        # Oil signals
        if cross_asset_data and 'OIL' in cross_asset_data:
            df = self._add_oil_signals(df, cross_asset_data['OIL'])

        # Bond yield signals
        if cross_asset_data and 'US10Y' in cross_asset_data:
            df = self._add_yield_signals(df, cross_asset_data['US10Y'])

        # Inter-pair signals (even without external data)
        df = self._add_inter_pair_signals(df)

        # Risk sentiment composite
        df = self._add_risk_sentiment(df)

        return df

    def _add_dxy_signals(self, df: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
        """Add Dollar Index signals."""

        # Align DXY to forex data
        if 'close' in dxy.columns:
            dxy_close = dxy['close']
        else:
            dxy_close = dxy.iloc[:, 0]

        # Resample/align if needed
        dxy_aligned = dxy_close.reindex(df.index, method='ffill')

        # DXY momentum
        dxy_mom = dxy_aligned.pct_change(20)
        df['signal_dxy_momentum'] = np.where(dxy_mom > 0.01, -1,  # DXY up = sell EUR/USD
                                             np.where(dxy_mom < -0.01, 1, 0))

        # DXY vs MA
        dxy_ma = dxy_aligned.rolling(50).mean()
        df['signal_dxy_trend'] = np.where(dxy_aligned > dxy_ma, -1, 1)

        # DXY breakout
        dxy_high = dxy_aligned.rolling(20).max()
        dxy_low = dxy_aligned.rolling(20).min()
        df['signal_dxy_breakout'] = np.where(dxy_aligned >= dxy_high * 0.99, -1,  # DXY breaking out = sell EUR
                                             np.where(dxy_aligned <= dxy_low * 1.01, 1, 0))

        # DXY RSI
        dxy_delta = dxy_aligned.diff()
        dxy_gain = dxy_delta.where(dxy_delta > 0, 0).rolling(14).mean()
        dxy_loss = (-dxy_delta.where(dxy_delta < 0, 0)).rolling(14).mean()
        dxy_rsi = 100 - (100 / (1 + dxy_gain / (dxy_loss + 1e-8)))
        df['signal_dxy_rsi'] = np.where(dxy_rsi > 70, 1,  # DXY overbought = buy EUR
                                        np.where(dxy_rsi < 30, -1, 0))

        return df

    def _add_vix_signals(self, df: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
        """Add VIX (volatility/fear) signals."""

        if 'close' in vix.columns:
            vix_close = vix['close']
        else:
            vix_close = vix.iloc[:, 0]

        vix_aligned = vix_close.reindex(df.index, method='ffill')

        # VIX level
        df['signal_vix_level'] = np.where(vix_aligned > 30, -1,  # High fear = risk-off = sell risk currencies
                                          np.where(vix_aligned < 15, 1, 0))

        # VIX spike (sudden fear)
        vix_change = vix_aligned.pct_change(5)
        df['signal_vix_spike'] = np.where(vix_change > 0.20, -1,  # VIX spike = risk-off
                                          np.where(vix_change < -0.15, 1, 0))

        # VIX vs historical
        vix_percentile = vix_aligned.rolling(252).apply(lambda x: (x.iloc[-1] > x).mean())
        df['signal_vix_percentile'] = np.where(vix_percentile > 0.8, -1,
                                               np.where(vix_percentile < 0.2, 1, 0))

        # VIX term structure (if available)
        # Contango = complacency, Backwardation = fear

        return df

    def _add_spx_signals(self, df: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
        """Add S&P 500 signals (risk appetite)."""

        if 'close' in spx.columns:
            spx_close = spx['close']
        else:
            spx_close = spx.iloc[:, 0]

        spx_aligned = spx_close.reindex(df.index, method='ffill')

        # SPX momentum (risk-on/off)
        spx_mom = spx_aligned.pct_change(20)
        df['signal_spx_momentum'] = np.where(spx_mom > 0.03, 1,  # Stocks up = risk-on = buy risk currencies
                                             np.where(spx_mom < -0.03, -1, 0))

        # SPX vs 200 MA (bull/bear market)
        spx_ma200 = spx_aligned.rolling(200).mean()
        df['signal_spx_trend'] = np.where(spx_aligned > spx_ma200, 1, -1)

        # SPX new high (euphoria)
        spx_high52 = spx_aligned.rolling(252).max()
        df['signal_spx_new_high'] = np.where(spx_aligned >= spx_high52 * 0.99, 1, 0)

        # SPX drawdown
        spx_drawdown = (spx_aligned / spx_high52 - 1)
        df['signal_spx_drawdown'] = np.where(spx_drawdown < -0.10, -1,  # Correction
                                             np.where(spx_drawdown < -0.20, -2, 0))  # Bear market

        return df

    def _add_gold_signals(self, df: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
        """Add Gold signals (safe haven)."""

        if 'close' in gold.columns:
            gold_close = gold['close']
        else:
            gold_close = gold.iloc[:, 0]

        gold_aligned = gold_close.reindex(df.index, method='ffill')

        # Gold momentum
        gold_mom = gold_aligned.pct_change(20)
        df['signal_gold_momentum'] = np.where(gold_mom > 0.02, 1,  # Gold up = buy safe havens (JPY, CHF)
                                              np.where(gold_mom < -0.02, -1, 0))

        # Gold vs USD divergence
        # If gold and USD both strong = uncertainty
        # If gold up, USD down = classic risk-off
        if 'close' in df.columns:
            forex_ret = df['close'].pct_change(20)
            gold_ret = gold_aligned.pct_change(20)

            # Correlation-based signal
            corr = forex_ret.rolling(60).corr(gold_ret)
            df['signal_gold_forex_corr'] = np.where(corr > 0.5, 1,
                                                    np.where(corr < -0.5, -1, 0))

        return df

    def _add_oil_signals(self, df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
        """Add Oil signals (commodity currencies)."""

        if 'close' in oil.columns:
            oil_close = oil['close']
        else:
            oil_close = oil.iloc[:, 0]

        oil_aligned = oil_close.reindex(df.index, method='ffill')

        # Oil momentum (affects CAD, NOK, RUB)
        oil_mom = oil_aligned.pct_change(20)
        df['signal_oil_momentum'] = np.where(oil_mom > 0.05, 1,  # Oil up = buy CAD (sell USD/CAD)
                                             np.where(oil_mom < -0.05, -1, 0))

        # Oil volatility
        oil_vol = oil_aligned.pct_change().rolling(20).std()
        oil_vol_percentile = oil_vol.rolling(252).apply(lambda x: (x.iloc[-1] > x).mean())
        df['signal_oil_volatility'] = np.where(oil_vol_percentile > 0.8, -1, 0)  # High oil vol = uncertainty

        return df

    def _add_yield_signals(self, df: pd.DataFrame, us10y: pd.DataFrame) -> pd.DataFrame:
        """Add bond yield signals."""

        if 'close' in us10y.columns:
            yield_close = us10y['close']
        else:
            yield_close = us10y.iloc[:, 0]

        yield_aligned = yield_close.reindex(df.index, method='ffill')

        # Yield direction (higher yields = USD strength)
        yield_change = yield_aligned.diff(20)
        df['signal_yield_direction'] = np.where(yield_change > 0.1, 1,  # Rising yields = USD strength
                                                np.where(yield_change < -0.1, -1, 0))

        # Yield level
        df['signal_yield_level'] = np.where(yield_aligned > 4.5, 1,  # High yields = USD attractive
                                            np.where(yield_aligned < 2.0, -1, 0))

        # Yield curve (if 2Y available)
        # Inversion = recession signal = risk-off

        return df

    def _add_inter_pair_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signals from related forex pairs."""

        # These work even without external data
        if 'close' in df.columns:
            close = df['close']

            # Momentum divergence within pair
            mom_5 = close.pct_change(5)
            mom_20 = close.pct_change(20)

            # Short-term vs long-term momentum divergence
            df['signal_momentum_divergence'] = np.where(
                (mom_5 > 0) & (mom_20 < 0), 1,  # Short-term reversal up
                np.where((mom_5 < 0) & (mom_20 > 0), -1, 0)
            )

            # Acceleration
            mom_accel = mom_5 - mom_5.shift(5)
            df['signal_momentum_accel'] = np.where(mom_accel > 0.005, 1,
                                                   np.where(mom_accel < -0.005, -1, 0))

        return df

    def _add_risk_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk sentiment signal."""

        # Combine available signals
        risk_signals = []

        if 'signal_vix_level' in df.columns:
            risk_signals.append(df['signal_vix_level'])
        if 'signal_spx_momentum' in df.columns:
            risk_signals.append(df['signal_spx_momentum'])
        if 'signal_dxy_momentum' in df.columns:
            risk_signals.append(-df['signal_dxy_momentum'])  # Inverse for EUR/USD

        if risk_signals:
            df['signal_risk_composite'] = np.sign(sum(risk_signals))
            df['signal_risk_strength'] = sum(np.abs(s) for s in risk_signals) / len(risk_signals)
        else:
            df['signal_risk_composite'] = 0
            df['signal_risk_strength'] = 0

        return df

    def get_signal_weights(self, pair: str) -> Dict[str, float]:
        """Get optimal signal weights for a pair based on correlations."""

        if pair not in self.CORRELATIONS:
            return {}

        correlations = self.CORRELATIONS[pair]

        # Convert correlations to weights
        weights = {}
        for asset, corr in correlations.items():
            signal_name = f'signal_{asset.lower()}_momentum'
            weights[signal_name] = abs(corr)  # Use absolute correlation as weight

        return weights


def create_cross_asset_features(forex_pair: str,
                                forex_data: pd.DataFrame,
                                related_pairs: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Quick function to create cross-asset features for a forex pair.

    Args:
        forex_pair: Target pair (e.g., 'EURUSD')
        forex_data: Target pair data
        related_pairs: Dict of related pair data

    Returns:
        DataFrame with cross-asset features
    """
    generator = CrossAssetSignalGenerator()

    # Convert related pairs to cross-asset format
    cross_asset_data = {}

    # Map forex pairs to cross-asset equivalent
    if related_pairs:
        for pair, data in related_pairs.items():
            if 'DXY' in pair.upper():
                cross_asset_data['DXY'] = data
            elif 'SPX' in pair.upper() or 'SPY' in pair.upper():
                cross_asset_data['SPX'] = data
            elif 'VIX' in pair.upper():
                cross_asset_data['VIX'] = data
            elif 'GOLD' in pair.upper() or 'XAU' in pair.upper():
                cross_asset_data['GOLD'] = data
            elif 'OIL' in pair.upper() or 'CL' in pair.upper() or 'WTI' in pair.upper():
                cross_asset_data['OIL'] = data

    return generator.generate_all_signals(forex_data, cross_asset_data)


if __name__ == '__main__':
    print("Cross-Asset Signals Test")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')

    forex_data = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 1.10,
        'high': np.random.randn(500).cumsum() + 1.11,
        'low': np.random.randn(500).cumsum() + 1.09,
        'close': np.random.randn(500).cumsum() + 1.10,
    }, index=dates)

    dxy_data = pd.DataFrame({
        'close': np.random.randn(500).cumsum() + 104
    }, index=dates)

    vix_data = pd.DataFrame({
        'close': np.abs(np.random.randn(500)) * 5 + 15
    }, index=dates)

    # Generate signals
    generator = CrossAssetSignalGenerator()
    result = generator.generate_all_signals(
        forex_data,
        cross_asset_data={'DXY': dxy_data, 'VIX': vix_data}
    )

    signal_cols = [c for c in result.columns if c.startswith('signal_')]
    print(f"\nGenerated {len(signal_cols)} cross-asset signals:")
    for col in signal_cols:
        print(f"  - {col}")

    # Show sample values
    print(f"\nSample signal values (last row):")
    for col in signal_cols:
        print(f"  {col}: {result[col].iloc[-1]}")
