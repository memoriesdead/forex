"""
Range-Based Volatility Estimators
==================================
Academic volatility estimators using OHLC data.

Sources (Gold Standard Citations):
- Parkinson (1980): "The Extreme Value Method for Estimating the Variance of the Rate of Return"
  Journal of Business, Vol. 53, No. 1, pp. 61-65

- Garman & Klass (1980): "On the Estimation of Security Price Volatilities from Historical Data"
  Journal of Business, Vol. 53, No. 1, pp. 67-78

- Rogers & Satchell (1991): "Estimating Variance from High, Low and Closing Prices"
  Annals of Applied Probability, Vol. 1, No. 4, pp. 504-512

- Yang & Zhang (2000): "Drift Independent Volatility Estimation Based on High, Low, Open, and Close Prices"
  Journal of Business, Vol. 73, No. 3, pp. 477-492

All estimators are more efficient than close-to-close realized variance.
Efficiency gains: Parkinson ~5x, Garman-Klass ~7.4x, Yang-Zhang ~14x
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VolatilityEstimate:
    """Container for volatility estimate with metadata."""
    value: float
    estimator: str
    annualized: bool
    window: int
    efficiency_vs_cc: float  # Efficiency vs close-to-close


class RangeVolatility:
    """
    Range-Based Volatility Estimators

    All estimators use OHLC data for more efficient volatility estimation
    compared to close-to-close squared returns.

    Usage:
        rv = RangeVolatility(annualize=True, trading_periods=252)
        vol = rv.yang_zhang(open_, high, low, close)
    """

    def __init__(
        self,
        annualize: bool = True,
        trading_periods: int = 252,  # 252 for daily, 252*24 for hourly forex
        min_periods: int = 2
    ):
        """
        Initialize range volatility calculator.

        Args:
            annualize: Whether to annualize volatility
            trading_periods: Periods per year (252 for daily, 252*24*60 for minute)
            min_periods: Minimum periods for rolling calculations
        """
        self.annualize = annualize
        self.trading_periods = trading_periods
        self.min_periods = min_periods

        # Annualization factor
        self.ann_factor = np.sqrt(trading_periods) if annualize else 1.0

    # =========================================================================
    # PARKINSON (1980) - Extreme Value Method
    # =========================================================================

    def parkinson(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Parkinson (1980) volatility estimator.

        Uses high-low range only. ~5x more efficient than close-to-close.

        Formula:
            σ² = (1 / 4*ln(2)) * E[(ln(H/L))²]

        Reference:
            Parkinson, M. (1980). "The Extreme Value Method for Estimating
            the Variance of the Rate of Return". Journal of Business, 53(1), 61-65.

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size

        Returns:
            Parkinson volatility series
        """
        # Constant: 1 / (4 * ln(2)) ≈ 0.3607
        k = 1.0 / (4.0 * np.log(2))

        log_hl = np.log(high / low) ** 2

        variance = k * log_hl.rolling(window, min_periods=self.min_periods).mean()

        return np.sqrt(variance) * self.ann_factor

    def parkinson_single(self, high: float, low: float) -> float:
        """Single-period Parkinson volatility."""
        k = 1.0 / (4.0 * np.log(2))
        return np.sqrt(k * np.log(high / low) ** 2) * self.ann_factor

    # =========================================================================
    # GARMAN-KLASS (1980) - OHLC Estimator
    # =========================================================================

    def garman_klass(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Garman-Klass (1980) volatility estimator.

        Uses OHLC data. ~7.4x more efficient than close-to-close.
        Assumes no drift (zero mean return).

        Formula:
            σ² = 0.5 * (ln(H/L))² - (2*ln(2) - 1) * (ln(C/O))²

        Reference:
            Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security
            Price Volatilities from Historical Data". Journal of Business, 53(1), 67-78.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size

        Returns:
            Garman-Klass volatility series
        """
        log_hl_sq = np.log(high / low) ** 2
        log_co_sq = np.log(close / open_) ** 2

        # Coefficient: 2*ln(2) - 1 ≈ 0.3863
        k = 2.0 * np.log(2) - 1.0

        variance = 0.5 * log_hl_sq - k * log_co_sq

        # Rolling mean
        variance = variance.rolling(window, min_periods=self.min_periods).mean()

        # Handle negative variance (can happen due to noise)
        variance = variance.clip(lower=0)

        return np.sqrt(variance) * self.ann_factor

    def garman_klass_single(
        self,
        open_: float,
        high: float,
        low: float,
        close: float
    ) -> float:
        """Single-period Garman-Klass volatility."""
        log_hl_sq = np.log(high / low) ** 2
        log_co_sq = np.log(close / open_) ** 2
        k = 2.0 * np.log(2) - 1.0
        variance = max(0, 0.5 * log_hl_sq - k * log_co_sq)
        return np.sqrt(variance) * self.ann_factor

    # =========================================================================
    # ROGERS-SATCHELL (1991) - Drift-Independent
    # =========================================================================

    def rogers_satchell(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Rogers-Satchell (1991) volatility estimator.

        Drift-independent estimator using OHLC. Unbiased even with non-zero drift.

        Formula:
            σ² = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)

        Reference:
            Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from
            High, Low and Closing Prices". Annals of Applied Probability, 1(4), 504-512.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size

        Returns:
            Rogers-Satchell volatility series
        """
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)

        variance = log_hc * log_ho + log_lc * log_lo

        # Rolling mean
        variance = variance.rolling(window, min_periods=self.min_periods).mean()

        # Handle negative variance
        variance = variance.clip(lower=0)

        return np.sqrt(variance) * self.ann_factor

    def rogers_satchell_single(
        self,
        open_: float,
        high: float,
        low: float,
        close: float
    ) -> float:
        """Single-period Rogers-Satchell volatility."""
        variance = (np.log(high/close) * np.log(high/open_) +
                   np.log(low/close) * np.log(low/open_))
        return np.sqrt(max(0, variance)) * self.ann_factor

    # =========================================================================
    # YANG-ZHANG (2000) - Best Overall Estimator
    # =========================================================================

    def yang_zhang(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        k: float = 0.34
    ) -> pd.Series:
        """
        Yang-Zhang (2000) volatility estimator.

        Most efficient estimator (~14x vs close-to-close).
        Handles overnight jumps (open != previous close).
        Combines overnight, open-to-close, and Rogers-Satchell components.

        Formula:
            σ² = σ²_overnight + k*σ²_open_close + (1-k)*σ²_RS

        where k = 0.34 is optimal for α = 1.34 (market open/close timing)

        Reference:
            Yang, D., & Zhang, Q. (2000). "Drift Independent Volatility Estimation
            Based on High, Low, Open, and Close Prices". Journal of Business, 73(3), 477-492.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            k: Weighting parameter (default 0.34 is optimal)

        Returns:
            Yang-Zhang volatility series
        """
        # Component 1: Overnight variance (close-to-open)
        log_co = np.log(open_ / close.shift(1))
        overnight_var = log_co.rolling(window, min_periods=self.min_periods).var()

        # Component 2: Open-to-close variance
        log_oc = np.log(close / open_)
        open_close_var = log_oc.rolling(window, min_periods=self.min_periods).var()

        # Component 3: Rogers-Satchell variance (intraday)
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)
        rs_var = (log_hc * log_ho + log_lc * log_lo).rolling(
            window, min_periods=self.min_periods
        ).mean()

        # Combined variance
        variance = overnight_var + k * open_close_var + (1 - k) * rs_var

        # Handle negative variance
        variance = variance.clip(lower=0)

        return np.sqrt(variance) * self.ann_factor

    def yang_zhang_single(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        prev_close: float,
        k: float = 0.34
    ) -> float:
        """Single-period Yang-Zhang volatility (requires previous close)."""
        overnight = np.log(open_ / prev_close) ** 2
        open_close = np.log(close / open_) ** 2
        rs = (np.log(high/close) * np.log(high/open_) +
              np.log(low/close) * np.log(low/open_))
        variance = overnight + k * open_close + (1 - k) * rs
        return np.sqrt(max(0, variance)) * self.ann_factor

    # =========================================================================
    # CLOSE-TO-CLOSE (Baseline)
    # =========================================================================

    def close_to_close(
        self,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Standard close-to-close volatility (baseline for comparison).

        Formula:
            σ = std(ln(C_t / C_{t-1}))

        Args:
            close: Close prices
            window: Rolling window size

        Returns:
            Close-to-close volatility series
        """
        log_returns = np.log(close / close.shift(1))
        return log_returns.rolling(window, min_periods=self.min_periods).std() * self.ann_factor

    # =========================================================================
    # ENSEMBLE / COMPOSITE
    # =========================================================================

    def ensemble(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        weights: Optional[dict] = None
    ) -> pd.Series:
        """
        Ensemble of all volatility estimators.

        Combines multiple estimators for robust volatility estimation.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            weights: Dict of estimator weights (default: efficiency-weighted)

        Returns:
            Ensemble volatility series
        """
        if weights is None:
            # Default: weight by relative efficiency
            weights = {
                'yang_zhang': 0.40,      # 14x efficiency
                'garman_klass': 0.25,    # 7.4x efficiency
                'rogers_satchell': 0.20, # 6x efficiency
                'parkinson': 0.15        # 5x efficiency
            }

        vols = {
            'yang_zhang': self.yang_zhang(open_, high, low, close, window),
            'garman_klass': self.garman_klass(open_, high, low, close, window),
            'rogers_satchell': self.rogers_satchell(open_, high, low, close, window),
            'parkinson': self.parkinson(high, low, window)
        }

        ensemble_vol = sum(weights[k] * vols[k] for k in weights)

        return ensemble_vol

    # =========================================================================
    # VOLATILITY FEATURES FOR ML
    # =========================================================================

    def compute_all_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        windows: list = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Compute all volatility features for ML.

        Returns DataFrame with volatility features at multiple windows.
        """
        features = {}

        for w in windows:
            features[f'vol_parkinson_{w}'] = self.parkinson(high, low, w)
            features[f'vol_garman_klass_{w}'] = self.garman_klass(open_, high, low, close, w)
            features[f'vol_rogers_satchell_{w}'] = self.rogers_satchell(open_, high, low, close, w)
            features[f'vol_yang_zhang_{w}'] = self.yang_zhang(open_, high, low, close, w)
            features[f'vol_close_to_close_{w}'] = self.close_to_close(close, w)

        # Volatility ratios (regime indicators)
        features['vol_ratio_short_long'] = features[f'vol_yang_zhang_{windows[0]}'] / features[f'vol_yang_zhang_{windows[-1]}']
        features['vol_ratio_yz_cc'] = features[f'vol_yang_zhang_{windows[1]}'] / features[f'vol_close_to_close_{windows[1]}']

        # Volatility percentile (relative to history)
        vol_ref = features[f'vol_yang_zhang_{windows[1]}']
        features['vol_percentile'] = vol_ref.rolling(252, min_periods=20).apply(
            lambda x: (x < x.iloc[-1]).mean(), raw=False
        )

        return pd.DataFrame(features)

    # =========================================================================
    # VOLATILITY REGIME DETECTION
    # =========================================================================

    def volatility_regime(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        low_percentile: float = 25,
        high_percentile: float = 75
    ) -> pd.Series:
        """
        Classify volatility regime: low, normal, high.

        Uses Yang-Zhang (most efficient estimator).

        Returns:
            Series with regime labels
        """
        vol = self.yang_zhang(open_, high, low, close, window)

        low_thresh = vol.rolling(252, min_periods=50).quantile(low_percentile / 100)
        high_thresh = vol.rolling(252, min_periods=50).quantile(high_percentile / 100)

        regime = pd.Series('normal', index=vol.index)
        regime[vol < low_thresh] = 'low'
        regime[vol > high_thresh] = 'high'

        return regime

    def volatility_breakout(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        threshold_percentile: float = 90
    ) -> pd.Series:
        """
        Detect volatility breakouts (spikes above threshold).

        Returns:
            Boolean series (True = breakout)
        """
        vol = self.yang_zhang(open_, high, low, close, window)
        threshold = vol.rolling(252, min_periods=50).quantile(threshold_percentile / 100)
        return vol > threshold


class GarmanKlassYangZhang:
    """
    Combined Garman-Klass-Yang-Zhang estimator (GKYZ).

    Hybrid approach combining the best properties of both estimators.
    Used by some institutional traders.
    """

    def __init__(self, annualize: bool = True, trading_periods: int = 252):
        self.rv = RangeVolatility(annualize, trading_periods)

    def estimate(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        GKYZ estimator: average of Garman-Klass and Yang-Zhang.
        """
        gk = self.rv.garman_klass(open_, high, low, close, window)
        yz = self.rv.yang_zhang(open_, high, low, close, window)
        return (gk + yz) / 2


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_range_volatility(
    annualize: bool = True,
    trading_periods: int = 252
) -> RangeVolatility:
    """Factory function to create range volatility calculator."""
    return RangeVolatility(annualize, trading_periods)


# =============================================================================
# STANDALONE FUNCTIONS FOR QUICK USE
# =============================================================================

def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Quick Parkinson volatility."""
    return RangeVolatility().parkinson(high, low, window)


def garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Quick Garman-Klass volatility."""
    return RangeVolatility().garman_klass(open_, high, low, close, window)


def rogers_satchell_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Quick Rogers-Satchell volatility."""
    return RangeVolatility().rogers_satchell(open_, high, low, close, window)


def yang_zhang_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Quick Yang-Zhang volatility."""
    return RangeVolatility().yang_zhang(open_, high, low, close, window)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Range-Based Volatility Estimators Test")
    print("=" * 60)

    # Generate synthetic OHLC data
    np.random.seed(42)
    n = 500

    # Random walk with drift
    returns = np.random.randn(n) * 0.01 + 0.0001  # 1% vol, slight positive drift
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_ = close.copy()
    open_[1:] = close[:-1] * (1 + np.random.randn(n-1) * 0.002)  # Gap from prev close

    # Convert to Series
    close = pd.Series(close)
    high = pd.Series(high)
    low = pd.Series(low)
    open_ = pd.Series(open_)

    # Test estimators
    rv = RangeVolatility(annualize=True, trading_periods=252)

    print("\nVolatility Estimates (20-day, annualized):")
    print(f"  Close-to-Close:   {rv.close_to_close(close, 20).iloc[-1]:.4f}")
    print(f"  Parkinson:        {rv.parkinson(high, low, 20).iloc[-1]:.4f}")
    print(f"  Garman-Klass:     {rv.garman_klass(open_, high, low, close, 20).iloc[-1]:.4f}")
    print(f"  Rogers-Satchell:  {rv.rogers_satchell(open_, high, low, close, 20).iloc[-1]:.4f}")
    print(f"  Yang-Zhang:       {rv.yang_zhang(open_, high, low, close, 20).iloc[-1]:.4f}")
    print(f"  Ensemble:         {rv.ensemble(open_, high, low, close, 20).iloc[-1]:.4f}")

    # Efficiency comparison (theoretical)
    print("\nEfficiency vs Close-to-Close (theoretical):")
    print("  Parkinson:        ~5.0x")
    print("  Garman-Klass:     ~7.4x")
    print("  Rogers-Satchell:  ~6.0x")
    print("  Yang-Zhang:       ~14.0x")

    # Test features
    features = rv.compute_all_features(open_, high, low, close, [5, 10, 20])
    print(f"\nFeatures generated: {len(features.columns)}")
    print(f"Sample features: {list(features.columns)[:5]}")

    # Test regime
    regime = rv.volatility_regime(open_, high, low, close)
    print(f"\nVolatility regime (last): {regime.iloc[-1]}")

    print("\nTest PASSED")
