"""
Academic Volatility Estimators - Peer-Reviewed Methods
=======================================================

Implementations of state-of-the-art volatility estimation and forecasting
methods from top finance and econometrics journals.

CITATIONS:
----------

1. HAR-RV (Heterogeneous Autoregressive Realized Volatility)
   Corsi, F. (2009).
   "A Simple Approximate Long-Memory Model of Realized Volatility"
   Journal of Financial Econometrics, 7(2), 174-196.
   DOI: 10.1093/jjfinec/nbp001
   SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1365738

2. PARKINSON VOLATILITY
   Parkinson, M. (1980).
   "The Extreme Value Method for Estimating the Variance of the Rate of Return"
   Journal of Business, 53(1), 61-65.
   DOI: 10.1086/296071

3. GARMAN-KLASS VOLATILITY
   Garman, M.B., & Klass, M.J. (1980).
   "On the Estimation of Security Price Volatilities from Historical Data"
   Journal of Business, 53(1), 67-78.
   DOI: 10.1086/296072

4. ROGERS-SATCHELL VOLATILITY
   Rogers, L.C.G., & Satchell, S.E. (1991).
   "Estimating Variance from High, Low and Closing Prices"
   Annals of Applied Probability, 1(4), 504-512.

5. YANG-ZHANG VOLATILITY
   Yang, D., & Zhang, Q. (2000).
   "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices"
   Journal of Business, 73(3), 477-492.
   DOI: 10.1086/209650

6. REALIZED VOLATILITY
   Andersen, T.G., & Bollerslev, T. (1998).
   "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts"
   International Economic Review, 39(4), 885-905.

7. GARCH
   Bollerslev, T. (1986).
   "Generalized Autoregressive Conditional Heteroskedasticity"
   Journal of Econometrics, 31(3), 307-327.
   DOI: 10.1016/0304-4076(86)90063-1

8. EGARCH (Exponential GARCH)
   Nelson, D.B. (1991).
   "Conditional Heteroskedasticity in Asset Returns: A New Approach"
   Econometrica, 59(2), 347-370.

EFFICIENCY COMPARISON (Parkinson 1980, Garman-Klass 1980):
----------------------------------------------------------
- Parkinson: 5.2x more efficient than close-to-close
- Garman-Klass: 7.4x more efficient than close-to-close
- Rogers-Satchell: Handles drift
- Yang-Zhang: Handles overnight jumps + drift

APPLICABILITY: All methods are FOREX-NATIVE (24-hour market makes range-based
estimators particularly valuable)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VolatilityConfig:
    """Configuration for volatility calculations."""
    default_window: int = 20
    har_daily: int = 1
    har_weekly: int = 5
    har_monthly: int = 22
    annualization_factor: float = 252.0  # Trading days
    forex_annualization: float = 252.0 * 24  # Hourly data


class RealizedVolatility:
    """
    Realized Volatility Estimators.

    Based on Andersen & Bollerslev (1998):
    "Standard volatility models do provide accurate forecasts"

    Formula:
        RV_t = sqrt(Σ r²_{t,i})

    Where r_{t,i} are intraday returns.

    Citation:
        Andersen, T.G., & Bollerslev, T. (1998).
        International Economic Review, 39(4), 885-905.
    """

    def __init__(self, config: VolatilityConfig = None):
        self.config = config or VolatilityConfig()

    def compute_realized_vol(
        self,
        returns: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Compute realized volatility from returns.

        Args:
            returns: Return series
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Realized volatility series
        """
        window = window or self.config.default_window

        # Sum of squared returns
        rv = returns.rolling(window).apply(
            lambda x: np.sqrt(np.sum(x**2))
        )

        if annualize:
            rv = rv * np.sqrt(self.config.annualization_factor / window)

        return rv


class HARRV:
    """
    Heterogeneous Autoregressive Realized Volatility (HAR-RV).

    The HAR model captures multi-horizon volatility components through
    daily, weekly, and monthly aggregations, reflecting heterogeneous
    market participants with different time horizons.

    Formula:
        RV_{t+1}^{(d)} = β_0 + β_d·RV_t^{(d)} + β_w·RV_t^{(w)} + β_m·RV_t^{(m)} + ε

    Where:
        RV^{(d)} = daily realized volatility
        RV^{(w)} = weekly average (5-day)
        RV^{(m)} = monthly average (22-day)

    Key Finding:
        "The HAR model successfully achieves the purpose of reproducing
        the main empirical features of financial returns (long memory,
        fat tails, and self-similarity) in a very tractable and
        parsimonious way."
        - Corsi (2009)

    Citation:
        Corsi, F. (2009).
        Journal of Financial Econometrics, 7(2), 174-196.
    """

    def __init__(self, config: VolatilityConfig = None):
        self.config = config or VolatilityConfig()
        self.coefficients = None

    def compute_har_components(
        self,
        rv_daily: pd.Series
    ) -> pd.DataFrame:
        """
        Compute HAR components (daily, weekly, monthly).

        "Three primary volatility components can be identified: the
        short-term traders with daily or higher trading frequency,
        the medium-term investors who typically rebalance their
        positions weekly, and the long-term agents with a
        characteristic time of one or more months."
        - Corsi (2009)

        Args:
            rv_daily: Daily realized volatility series

        Returns:
            DataFrame with RV_d, RV_w, RV_m columns
        """
        df = pd.DataFrame(index=rv_daily.index)

        # Daily component
        df['RV_d'] = rv_daily

        # Weekly component (5-day average)
        df['RV_w'] = rv_daily.rolling(self.config.har_weekly).mean()

        # Monthly component (22-day average)
        df['RV_m'] = rv_daily.rolling(self.config.har_monthly).mean()

        return df

    def fit(
        self,
        rv_daily: pd.Series,
        forecast_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Fit HAR-RV model using OLS.

        Args:
            rv_daily: Daily realized volatility
            forecast_horizon: Forecast horizon (default: 1 day)

        Returns:
            Dictionary of fitted coefficients
        """
        # Compute components
        df = self.compute_har_components(rv_daily)

        # Target: h-day ahead RV
        df['RV_target'] = df['RV_d'].shift(-forecast_horizon)

        # Drop NaN
        df = df.dropna()

        if len(df) < 50:
            return {'beta_0': 0, 'beta_d': 1, 'beta_w': 0, 'beta_m': 0}

        # OLS regression
        X = df[['RV_d', 'RV_w', 'RV_m']].values
        y = df['RV_target'].values

        # Add constant
        X = np.column_stack([np.ones(len(X)), X])

        # OLS: β = (X'X)^(-1) X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = [0, 1, 0, 0]

        self.coefficients = {
            'beta_0': beta[0],
            'beta_d': beta[1],
            'beta_w': beta[2],
            'beta_m': beta[3]
        }

        return self.coefficients

    def predict(
        self,
        rv_daily: pd.Series
    ) -> pd.Series:
        """
        Forecast volatility using fitted HAR model.

        Args:
            rv_daily: Daily realized volatility

        Returns:
            Volatility forecast series
        """
        if self.coefficients is None:
            self.fit(rv_daily)

        df = self.compute_har_components(rv_daily)

        forecast = (
            self.coefficients['beta_0'] +
            self.coefficients['beta_d'] * df['RV_d'] +
            self.coefficients['beta_w'] * df['RV_w'] +
            self.coefficients['beta_m'] * df['RV_m']
        )

        return forecast


class RangeBasedVolatility:
    """
    Range-Based Volatility Estimators.

    These estimators use OHLC prices and are theoretically 5-7x more
    efficient than close-to-close estimators.

    Comparison (from Parkinson 1980, Garman-Klass 1980):
    - Close-to-close: efficiency = 1.0
    - Parkinson: efficiency = 5.2
    - Garman-Klass: efficiency = 7.4
    - Rogers-Satchell: handles drift
    - Yang-Zhang: handles overnight jumps

    Citations:
        Parkinson (1980), Garman & Klass (1980),
        Rogers & Satchell (1991), Yang & Zhang (2000)
    """

    def __init__(self, config: VolatilityConfig = None):
        self.config = config or VolatilityConfig()

    def parkinson(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Parkinson (1980) Range Volatility Estimator.

        Formula:
            σ²_P = (1/4ln(2)) * (H - L)²

        Where H, L are log high/low prices.

        "The extreme value method for estimating the variance of the
        rate of return of a common stock."
        - Parkinson (1980)

        Efficiency: 5.2x more efficient than close-to-close

        Citation:
            Parkinson, M. (1980).
            Journal of Business, 53(1), 61-65.

        Args:
            high: High prices
            low: Low prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Volatility series
        """
        window = window or self.config.default_window

        # Log high-low range
        log_hl = np.log(high / low)

        # Parkinson constant
        factor = 1.0 / (4.0 * np.log(2))

        # Rolling variance
        var = factor * (log_hl ** 2).rolling(window).mean()

        vol = np.sqrt(var)

        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)

        return vol

    def garman_klass(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Garman-Klass (1980) Volatility Estimator.

        Formula:
            σ²_GK = 0.5*(H-L)² - (2ln(2)-1)*(C-O)²

        Where H, L, O, C are log prices.

        "On the estimation of security price volatilities from
        historical data."
        - Garman & Klass (1980)

        Efficiency: 7.4x more efficient than close-to-close
        Assumption: No drift, continuous prices

        Citation:
            Garman, M.B., & Klass, M.J. (1980).
            Journal of Business, 53(1), 67-78.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Volatility series
        """
        window = window or self.config.default_window

        # Log prices
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        # Garman-Klass formula
        term1 = 0.5 * (log_hl ** 2)
        term2 = (2 * np.log(2) - 1) * (log_co ** 2)

        var = (term1 - term2).rolling(window).mean()

        # Handle negative variance (can happen with noisy data)
        var = var.clip(lower=0)
        vol = np.sqrt(var)

        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)

        return vol

    def rogers_satchell(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Rogers-Satchell (1991) Volatility Estimator.

        Formula:
            σ²_RS = (H-C)*(H-O) + (L-C)*(L-O)

        Where H, L, O, C are log prices.

        Key Advantage: Handles drift (non-zero mean returns)

        "Rogers and Satchell (1991) proposed a formula that allows
        for drifts."
        - Yang & Zhang (2000)

        Citation:
            Rogers, L.C.G., & Satchell, S.E. (1991).
            Annals of Applied Probability, 1(4), 504-512.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Volatility series
        """
        window = window or self.config.default_window

        # Log price ratios
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_)
        log_lc = np.log(low / close)

        # Rogers-Satchell formula
        rs_var = (log_ho * log_hc) + (log_lo * log_lc)

        var = rs_var.rolling(window).mean()
        var = var.clip(lower=0)
        vol = np.sqrt(var)

        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)

        return vol

    def yang_zhang(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Yang-Zhang (2000) Volatility Estimator.

        Formula:
            σ²_YZ = σ²_overnight + k*σ²_open + (1-k)*σ²_RS

        Where:
            k = 0.34 / (1.34 + (n+1)/(n-1))
            σ²_overnight = Var(log(O_t/C_{t-1}))
            σ²_open = Var(log(C_t/O_t))
            σ²_RS = Rogers-Satchell variance

        Key Advantage: Handles overnight jumps AND drift

        "Yang and Zhang (2000) published a formula that is unbiased,
        drift independent, and consistent in dealing with opening
        jumps; this latter feature is unique among the formulas examined."
        - Bali & Weinbaum (2005)

        Citation:
            Yang, D., & Zhang, Q. (2000).
            Journal of Business, 73(3), 477-492.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Volatility series
        """
        window = window or self.config.default_window

        # Yang-Zhang k parameter
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        # Overnight volatility: O_t vs C_{t-1}
        log_oc_overnight = np.log(open_ / close.shift(1))
        overnight_var = log_oc_overnight.rolling(window).var()

        # Open-to-close volatility
        log_co = np.log(close / open_)
        open_close_var = log_co.rolling(window).var()

        # Rogers-Satchell component
        rs_var = self.rogers_satchell(
            open_, high, low, close, window, annualize=False
        ) ** 2

        # Yang-Zhang formula
        var = overnight_var + k * open_close_var + (1 - k) * rs_var
        var = var.clip(lower=0)
        vol = np.sqrt(var)

        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)

        return vol

    def close_to_close(
        self,
        close: pd.Series,
        window: int = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Simple close-to-close volatility (baseline).

        Formula:
            σ = std(log(C_t/C_{t-1}))

        Used as baseline for efficiency comparison.

        Args:
            close: Close prices
            window: Rolling window
            annualize: Whether to annualize

        Returns:
            Volatility series
        """
        window = window or self.config.default_window

        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window).std()

        if annualize:
            vol = vol * np.sqrt(self.config.annualization_factor)

        return vol


class AcademicVolatilityFeatures:
    """
    Generate all academic volatility features.

    Combines HAR-RV, range-based estimators, and derived features
    into a unified feature generator.
    """

    def __init__(self, config: VolatilityConfig = None):
        self.config = config or VolatilityConfig()
        self.realized = RealizedVolatility(config)
        self.har = HARRV(config)
        self.range_based = RangeBasedVolatility(config)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)

        # Required columns
        close = df['close']

        # Optional columns (use close if not available)
        open_ = df.get('open', close.shift(1).fillna(close))
        high = df.get('high', close)
        low = df.get('low', close)

        # =========================================================
        # Close-to-Close (Baseline)
        # =========================================================
        for w in [5, 10, 20, 60]:
            features[f'VOL_CC_{w}'] = self.range_based.close_to_close(close, w)

        # =========================================================
        # Parkinson (1980) - 5.2x efficient
        # =========================================================
        for w in [5, 10, 20, 60]:
            features[f'VOL_PARKINSON_{w}'] = self.range_based.parkinson(high, low, w)

        # =========================================================
        # Garman-Klass (1980) - 7.4x efficient
        # =========================================================
        for w in [5, 10, 20, 60]:
            features[f'VOL_GK_{w}'] = self.range_based.garman_klass(
                open_, high, low, close, w
            )

        # =========================================================
        # Rogers-Satchell (1991) - Drift independent
        # =========================================================
        for w in [5, 10, 20, 60]:
            features[f'VOL_RS_{w}'] = self.range_based.rogers_satchell(
                open_, high, low, close, w
            )

        # =========================================================
        # Yang-Zhang (2000) - Handles overnight jumps
        # =========================================================
        for w in [10, 20, 60]:
            features[f'VOL_YZ_{w}'] = self.range_based.yang_zhang(
                open_, high, low, close, w
            )

        # =========================================================
        # Realized Volatility
        # =========================================================
        returns = close.pct_change()
        for w in [5, 10, 20, 60]:
            features[f'RV_{w}'] = self.realized.compute_realized_vol(returns, w)

        # =========================================================
        # HAR-RV Components - Corsi (2009)
        # =========================================================
        rv_daily = self.realized.compute_realized_vol(returns, 1, annualize=False)

        har_components = self.har.compute_har_components(rv_daily)
        features['HAR_RV_D'] = har_components['RV_d']
        features['HAR_RV_W'] = har_components['RV_w']
        features['HAR_RV_M'] = har_components['RV_m']

        # HAR forecast
        features['HAR_FORECAST'] = self.har.predict(rv_daily)

        # =========================================================
        # Volatility Ratios (Mean Reversion Signals)
        # =========================================================

        # Short-term vs Long-term ratio
        features['VOL_RATIO_5_20'] = features['VOL_GK_5'] / (features['VOL_GK_20'] + 1e-10)
        features['VOL_RATIO_10_60'] = features['VOL_GK_10'] / (features['VOL_GK_60'] + 1e-10)

        # Parkinson vs Close-to-Close (efficiency indicator)
        features['VOL_EFFICIENCY'] = features['VOL_PARKINSON_20'] / (features['VOL_CC_20'] + 1e-10)

        # =========================================================
        # Volatility of Volatility (VoV)
        # =========================================================
        for w in [20, 60]:
            features[f'VOV_{w}'] = features[f'VOL_GK_5'].rolling(w).std()

        # =========================================================
        # Volatility Z-Score (Regime Indicator)
        # =========================================================
        vol_mean = features['VOL_GK_20'].rolling(252).mean()
        vol_std = features['VOL_GK_20'].rolling(252).std()
        features['VOL_ZSCORE'] = (features['VOL_GK_20'] - vol_mean) / (vol_std + 1e-10)

        # =========================================================
        # Volatility Percentile
        # =========================================================
        features['VOL_PERCENTILE'] = features['VOL_GK_20'].rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )

        # =========================================================
        # ATR (Average True Range) - Wilder (1978)
        # =========================================================
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        for w in [5, 14, 20]:
            features[f'ATR_{w}'] = tr.rolling(w).mean()
            features[f'ATR_PCT_{w}'] = features[f'ATR_{w}'] / close

        return features.fillna(method='ffill').fillna(0)


def generate_volatility_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate volatility features.

    Citations:
    - Corsi (2009) - HAR-RV
    - Parkinson (1980) - Range volatility
    - Garman & Klass (1980) - OHLC volatility
    - Rogers & Satchell (1991) - Drift-independent
    - Yang & Zhang (2000) - Overnight jumps

    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional arguments

    Returns:
        DataFrame with volatility features
    """
    generator = AcademicVolatilityFeatures()
    return generator.generate_all(df)


# Module-level exports
__all__ = [
    'VolatilityConfig',
    'RealizedVolatility',
    'HARRV',
    'RangeBasedVolatility',
    'AcademicVolatilityFeatures',
    'generate_volatility_features',
]
