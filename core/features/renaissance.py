"""
Renaissance Technologies - Weak Signals Generator
Implements 50+ weak trading signals based on Renaissance principles

Jim Simons approach:
- Multiple weak signals (each 51-55% accuracy)
- Combined into strong ensemble
- High-frequency execution
- Statistical rigor
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats
from scipy.signal import find_peaks


class RenaissanceSignalGenerator:
    """
    Generate 50+ weak signals following Renaissance methodology

    Philosophy:
    - No single signal is strong (each ~52-55% accuracy)
    - Combine many weak signals for robust predictions
    - Focus on statistical significance
    - Short holding periods
    """

    def __init__(self):
        self.signals = {}

    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all Renaissance-style signals"""
        df = data.copy()

        # Ensure we have required columns
        if 'price' not in df.columns and 'bid' in df.columns and 'ask' in df.columns:
            df['price'] = (df['bid'] + df['ask']) / 2

        # ========================================
        # TREND SIGNALS (10 signals)
        # ========================================
        df = self._trend_signals(df)

        # ========================================
        # MEAN REVERSION SIGNALS (10 signals)
        # ========================================
        df = self._mean_reversion_signals(df)

        # ========================================
        # MOMENTUM SIGNALS (10 signals)
        # ========================================
        df = self._momentum_signals(df)

        # ========================================
        # VOLATILITY SIGNALS (10 signals)
        # ========================================
        df = self._volatility_signals(df)

        # ========================================
        # MICROSTRUCTURE SIGNALS (10 signals)
        # ========================================
        df = self._microstructure_signals(df)

        # Store all signal columns
        signal_cols = [col for col in df.columns if col.startswith('signal_')]
        self.signals = signal_cols

        return df

    def _trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following signals (10)"""

        # Signal 1: MA crossover (multiple periods)
        df['signal_ma_cross_20_50'] = np.where(
            df['price'].rolling(20).mean() > df['price'].rolling(50).mean(), 1, -1
        )

        df['signal_ma_cross_10_30'] = np.where(
            df['price'].rolling(10).mean() > df['price'].rolling(30).mean(), 1, -1
        )

        df['signal_ma_cross_5_20'] = np.where(
            df['price'].rolling(5).mean() > df['price'].rolling(20).mean(), 1, -1
        )

        # Signal 2: Price vs MA
        ma_50 = df['price'].rolling(50).mean()
        df['signal_price_vs_ma50'] = np.where(df['price'] > ma_50, 1, -1)

        ma_100 = df['price'].rolling(100).mean()
        df['signal_price_vs_ma100'] = np.where(df['price'] > ma_100, 1, -1)

        # Signal 3: MA slope
        ma_20 = df['price'].rolling(20).mean()
        df['signal_ma_slope_20'] = np.where(ma_20.diff() > 0, 1, -1)

        df['signal_ma_slope_50'] = np.where(
            df['price'].rolling(50).mean().diff() > 0, 1, -1
        )

        # Signal 4: Multiple timeframe alignment
        ma_5 = df['price'].rolling(5).mean()
        ma_10 = df['price'].rolling(10).mean()
        ma_20 = df['price'].rolling(20).mean()

        bullish_align = (df['price'] > ma_5) & (ma_5 > ma_10) & (ma_10 > ma_20)
        bearish_align = (df['price'] < ma_5) & (ma_5 < ma_10) & (ma_10 < ma_20)

        df['signal_timeframe_align'] = np.where(bullish_align, 1,
                                                np.where(bearish_align, -1, 0))

        # Signal 5: Linear regression slope
        def rolling_linreg_slope(series, window=20):
            slopes = []
            for i in range(len(series)):
                if i < window:
                    slopes.append(0)
                else:
                    y = series.iloc[i-window:i].values
                    x = np.arange(window)
                    slope, _ = np.polyfit(x, y, 1)
                    slopes.append(slope)
            return pd.Series(slopes, index=series.index)

        df['signal_linreg_slope'] = np.where(
            rolling_linreg_slope(df['price']) > 0, 1, -1
        )

        # Signal 6: Exponential MA crossover
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['signal_ema_cross_12_26'] = np.where(ema_12 > ema_26, 1, -1)

        return df

    def _mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion signals (10)"""

        # Signal 1: Z-score reversion
        ma_20 = df['price'].rolling(20).mean()
        std_20 = df['price'].rolling(20).std()
        z_score = (df['price'] - ma_20) / std_20

        df['signal_zscore_revert'] = np.where(z_score < -2, 1,  # Oversold
                                              np.where(z_score > 2, -1, 0))  # Overbought

        df['signal_zscore_mild'] = np.where(z_score < -1, 1,
                                           np.where(z_score > 1, -1, 0))

        # Signal 2: Bollinger Bands
        bb_upper = ma_20 + 2 * std_20
        bb_lower = ma_20 - 2 * std_20

        df['signal_bb_revert'] = np.where(df['price'] < bb_lower, 1,
                                          np.where(df['price'] > bb_upper, -1, 0))

        # Signal 3: RSI divergence
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        df['signal_rsi'] = np.where(rsi < 30, 1,  # Oversold
                                    np.where(rsi > 70, -1, 0))  # Overbought

        df['signal_rsi_extreme'] = np.where(rsi < 20, 1,
                                           np.where(rsi > 80, -1, 0))

        # Signal 4: Distance from MA (percent)
        distance_pct = (df['price'] - ma_20) / ma_20

        df['signal_distance_revert'] = np.where(distance_pct < -0.01, 1,
                                               np.where(distance_pct > 0.01, -1, 0))

        # Signal 5: Return reversion
        returns_5 = df['price'].pct_change(5)
        df['signal_return_revert'] = np.where(returns_5 < -0.01, 1,
                                              np.where(returns_5 > 0.01, -1, 0))

        # Signal 6: Stochastic oscillator
        low_14 = df['price'].rolling(14).min()
        high_14 = df['price'].rolling(14).max()
        stoch = 100 * (df['price'] - low_14) / (high_14 - low_14)

        df['signal_stoch'] = np.where(stoch < 20, 1,
                                      np.where(stoch > 80, -1, 0))

        # Signal 7: Williams %R
        williams_r = -100 * (high_14 - df['price']) / (high_14 - low_14)
        df['signal_williams'] = np.where(williams_r < -80, 1,
                                        np.where(williams_r > -20, -1, 0))

        # Signal 8: Percent rank reversion
        def percent_rank(series, window=50):
            return series.rolling(window).apply(
                lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
                raw=True
            )

        pct_rank = percent_rank(df['price'])
        df['signal_pctrank_revert'] = np.where(pct_rank < 0.2, 1,
                                              np.where(pct_rank > 0.8, -1, 0))

        return df

    def _momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum signals (10)"""

        # Signal 1: Rate of change
        roc_10 = df['price'].pct_change(10)
        df['signal_roc_10'] = np.where(roc_10 > 0, 1, -1)

        roc_20 = df['price'].pct_change(20)
        df['signal_roc_20'] = np.where(roc_20 > 0, 1, -1)

        # Signal 2: Acceleration
        velocity = df['price'].diff()
        acceleration = velocity.diff()
        df['signal_acceleration'] = np.where(acceleration > 0, 1, -1)

        # Signal 3: MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()

        df['signal_macd'] = np.where(macd > signal_line, 1, -1)

        # Signal 4: Momentum strength
        momentum_20 = df['price'] - df['price'].shift(20)
        df['signal_momentum_strength'] = np.where(momentum_20 > 0, 1, -1)

        # Signal 5: Trix (triple EMA)
        ema1 = df['price'].ewm(span=15).mean()
        ema2 = ema1.ewm(span=15).mean()
        ema3 = ema2.ewm(span=15).mean()
        trix = ema3.pct_change()

        df['signal_trix'] = np.where(trix > 0, 1, -1)

        # Signal 6: Consecutive returns
        returns = df['price'].pct_change()
        consecutive_up = (returns > 0).rolling(3).sum()
        df['signal_consecutive'] = np.where(consecutive_up >= 2, 1,
                                           np.where(consecutive_up <= 1, -1, 0))

        # Signal 7: Volume-weighted momentum (if volume available)
        if 'volume' in df.columns:
            vwm = (df['price'].diff() * df['volume']).rolling(20).sum()
            df['signal_volume_momentum'] = np.where(vwm > 0, 1, -1)
        else:
            df['signal_volume_momentum'] = 0

        # Signal 8: Relative momentum
        returns_10 = df['price'].pct_change(10)
        returns_20 = df['price'].pct_change(20)
        df['signal_relative_momentum'] = np.where(returns_10 > returns_20, 1, -1)

        # Signal 9: Elder Ray Index
        ema_13 = df['price'].ewm(span=13).mean()
        bull_power = df['price'] - ema_13  # Using price as proxy for high
        df['signal_elder_ray'] = np.where(bull_power > 0, 1, -1)

        # Signal 10: Commodity Channel Index (CCI)
        typical_price = df['price']  # Simplified (usually (H+L+C)/3)
        sma_20 = typical_price.rolling(20).mean()
        mad = (typical_price - sma_20).abs().rolling(20).mean()
        cci = (typical_price - sma_20) / (0.015 * mad)

        df['signal_cci'] = np.where(cci > 100, 1,
                                    np.where(cci < -100, -1, 0))

        return df

    def _volatility_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based signals (10)"""

        returns = df['price'].pct_change()

        # Signal 1: Volatility regime
        vol_20 = returns.rolling(20).std()
        vol_50 = returns.rolling(50).std()

        df['signal_vol_regime'] = np.where(vol_20 < vol_50, 1, -1)  # Low vol = bullish

        # Signal 2: Volatility breakout
        vol_threshold = vol_20.rolling(50).quantile(0.75)
        df['signal_vol_breakout'] = np.where(vol_20 > vol_threshold, 1, 0)

        # Signal 3: ATR (Average True Range)
        tr = df['price'].diff().abs()
        atr = tr.rolling(14).mean()
        atr_signal = atr / df['price']  # Normalized

        df['signal_atr'] = np.where(atr_signal < atr_signal.rolling(50).mean(), 1, -1)

        # Signal 4: Bollinger Band width
        ma_20 = df['price'].rolling(20).mean()
        std_20 = df['price'].rolling(20).std()
        bb_width = (4 * std_20) / ma_20

        df['signal_bb_width'] = np.where(bb_width < bb_width.rolling(50).quantile(0.25), 1, 0)

        # Signal 5: Historical volatility percentile
        vol_percentile = vol_20.rolling(100).apply(
            lambda x: (x[-1] > x).sum() / len(x) if len(x) > 0 else 0.5,
            raw=True
        )
        df['signal_vol_percentile'] = np.where(vol_percentile < 0.3, 1,
                                              np.where(vol_percentile > 0.7, -1, 0))

        # Signal 6: Volatility mean reversion
        vol_zscore = (vol_20 - vol_20.rolling(50).mean()) / vol_20.rolling(50).std()
        df['signal_vol_revert'] = np.where(vol_zscore < -1, 1,
                                          np.where(vol_zscore > 1, -1, 0))

        # Signal 7: Keltner Channels
        ema_20 = df['price'].ewm(span=20).mean()
        kelt_upper = ema_20 + 2 * atr
        kelt_lower = ema_20 - 2 * atr

        df['signal_keltner'] = np.where(df['price'] < kelt_lower, 1,
                                       np.where(df['price'] > kelt_upper, -1, 0))

        # Signal 8: Donchian Channels
        high_20 = df['price'].rolling(20).max()
        low_20 = df['price'].rolling(20).min()

        df['signal_donchian'] = np.where(df['price'] == high_20, 1,
                                        np.where(df['price'] == low_20, -1, 0))

        # Signal 9: Chaikin Volatility
        high_low_range = (df['price'].rolling(10).max() - df['price'].rolling(10).min())
        chaikin_vol = high_low_range.pct_change(10)

        df['signal_chaikin_vol'] = np.where(chaikin_vol < 0, 1, -1)

        # Signal 10: Volatility skew
        recent_vol = returns.rolling(10).std()
        longer_vol = returns.rolling(30).std()

        df['signal_vol_skew'] = np.where(recent_vol < longer_vol, 1, -1)

        return df

    def _microstructure_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure signals (10)"""

        # Signal 1: Bid-ask spread
        if 'bid' in df.columns and 'ask' in df.columns:
            spread = df['ask'] - df['bid']
            spread_ma = spread.rolling(20).mean()

            df['signal_spread'] = np.where(spread < spread_ma, 1, -1)  # Tight spread = bullish

            # Signal 2: Spread volatility
            spread_vol = spread.rolling(20).std()
            df['signal_spread_vol'] = np.where(spread_vol < spread_vol.rolling(50).mean(), 1, -1)

            # Signal 3: Mid-price momentum
            mid = (df['bid'] + df['ask']) / 2
            mid_momentum = mid.diff()
            df['signal_mid_momentum'] = np.where(mid_momentum > 0, 1, -1)

            # Signal 4: Quote intensity (if available)
            # Placeholder: use price change frequency
            price_changes = (df['price'].diff() != 0).rolling(20).sum()
            df['signal_quote_intensity'] = np.where(
                price_changes > price_changes.rolling(50).mean(), 1, -1
            )

        else:
            # Fallback if bid/ask not available
            df['signal_spread'] = 0
            df['signal_spread_vol'] = 0
            df['signal_mid_momentum'] = 0
            df['signal_quote_intensity'] = 0

        # Signal 5: Price reversal detection
        # Find local peaks and troughs
        prices = df['price'].values
        peaks, _ = find_peaks(prices, distance=10)
        troughs, _ = find_peaks(-prices, distance=10)

        df['signal_reversal'] = 0
        if len(peaks) > 0:
            df.loc[df.index[peaks], 'signal_reversal'] = -1  # Sell at peaks
        if len(troughs) > 0:
            df.loc[df.index[troughs], 'signal_reversal'] = 1  # Buy at troughs

        # Signal 6: Time since last trade (tick frequency)
        try:
            if 'timestamp' in df.columns:
                time_series = pd.to_datetime(df['timestamp'])
                df['time_diff'] = time_series.diff().dt.total_seconds()
            elif pd.api.types.is_datetime64_any_dtype(df.index):
                df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
            else:
                df['time_diff'] = 1  # Default to constant interval
            time_diff_ma = df['time_diff'].rolling(20).mean()
            df['signal_tick_frequency'] = np.where(df['time_diff'] < time_diff_ma, 1, -1)
        except:
            df['signal_tick_frequency'] = 0

        # Signal 7: Price clustering
        # Check if price is at round numbers
        price_decimal = (df['price'] * 10000) % 10
        df['signal_price_cluster'] = np.where(price_decimal < 2, -1, 0)  # Resistance at round numbers

        # Signal 8: Order flow imbalance (simplified)
        # Use price momentum as proxy for buy/sell pressure
        flow = df['price'].diff().rolling(20).sum()
        df['signal_order_flow'] = np.where(flow > 0, 1, -1)

        # Signal 9: Roll return (overnight effect)
        # Not applicable for intraday, but included for completeness
        df['signal_roll'] = 0

        # Signal 10: Effective spread (realized cost)
        # Simplified: use recent price change volatility
        effective_spread = df['price'].diff().abs().rolling(20).mean()
        df['signal_effective_spread'] = np.where(
            effective_spread < effective_spread.rolling(50).mean(), 1, -1
        )

        return df

    def ensemble_signals(self, df: pd.DataFrame, method='average') -> pd.DataFrame:
        """
        Combine weak signals into ensemble prediction

        Args:
            method: 'average', 'weighted', 'vote'
        """
        signal_cols = [col for col in df.columns if col.startswith('signal_')]

        if not signal_cols:
            df['ensemble_signal'] = 0
            df['ensemble_confidence'] = 0
            return df

        # Get all signals as matrix
        signals_matrix = df[signal_cols].fillna(0).values

        if method == 'average':
            # Simple average
            ensemble = signals_matrix.mean(axis=1)

        elif method == 'weighted':
            # Weight by signal strength (simplified - equal weights for now)
            weights = np.ones(len(signal_cols)) / len(signal_cols)
            ensemble = (signals_matrix * weights).sum(axis=1)

        elif method == 'vote':
            # Majority vote
            ensemble = np.where(
                (signals_matrix == 1).sum(axis=1) > (signals_matrix == -1).sum(axis=1),
                1, -1
            )

        # Convert to discrete signals
        df['ensemble_signal'] = np.where(ensemble > 0.2, 1,
                                        np.where(ensemble < -0.2, -1, 0))

        # Calculate confidence (agreement among signals)
        agreement = np.abs(signals_matrix.sum(axis=1)) / len(signal_cols)
        df['ensemble_confidence'] = agreement

        return df


def test_renaissance_signals():
    """Test Renaissance signals generation"""
    print("="*70)
    print("RENAISSANCE SIGNALS TEST")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2026-01-01', periods=1000, freq='1min')
    price = 1.16 + np.cumsum(np.random.randn(1000) * 0.0001)

    data = pd.DataFrame({
        'timestamp': dates,
        'price': price,
        'bid': price - 0.0001,
        'ask': price + 0.0001
    })
    data.set_index('timestamp', inplace=True)

    # Generate signals
    generator = RenaissanceSignalGenerator()
    data_with_signals = generator.generate_all_signals(data)

    # Ensemble
    data_with_signals = generator.ensemble_signals(data_with_signals, method='average')

    # Print summary
    signal_cols = [col for col in data_with_signals.columns if col.startswith('signal_')]
    print(f"\nGenerated {len(signal_cols)} weak signals")

    ensemble_signals = data_with_signals['ensemble_signal']
    print(f"\nEnsemble signals:")
    print(f"  Buy:  {(ensemble_signals == 1).sum()} ({(ensemble_signals == 1).sum() / len(ensemble_signals) * 100:.1f}%)")
    print(f"  Sell: {(ensemble_signals == -1).sum()} ({(ensemble_signals == -1).sum() / len(ensemble_signals) * 100:.1f}%)")
    print(f"  Hold: {(ensemble_signals == 0).sum()} ({(ensemble_signals == 0).sum() / len(ensemble_signals) * 100:.1f}%)")

    print(f"\nAvg confidence: {data_with_signals['ensemble_confidence'].mean():.2f}")
    print()


if __name__ == "__main__":
    test_renaissance_signals()
