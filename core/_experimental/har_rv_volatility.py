"""
HAR-RV: Heterogeneous Autoregressive Realized Volatility
=========================================================
Source: Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
Verified from: CSDN Chinese quant sources (HAR-RV波动率)

The HAR-RV model captures multi-scale volatility dynamics:
- Daily (RV_d): Short-term volatility
- Weekly (RV_w): Medium-term volatility
- Monthly (RV_m): Long-term volatility

Extensions:
- HAR-RV-J: Adds jump component
- HAR-RV-CJ: Continuous + Jump decomposition
- HAR-RV-SJ: Signed jumps (positive/negative asymmetry)

Adapted for forex tick-level and minute-level data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class HARRVVolatility:
    """
    HAR-RV: Heterogeneous Autoregressive Realized Volatility Model

    Formula:
        RV_t+1 = β₀ + β_d * RV_d,t + β_w * RV_w,t + β_m * RV_m,t + ε_t+1

    Usage:
        har = HARRVVolatility()
        har.fit(returns)
        forecast = har.predict(returns)
        regime = har.get_volatility_regime(returns)
    """

    def __init__(
        self,
        daily_window: int = 1,
        weekly_window: int = 5,
        monthly_window: int = 22,
        jump_threshold: float = 3.0,
        annualization_factor: float = 252
    ):
        """
        Initialize HAR-RV model.

        Args:
            daily_window: Window for daily RV (default 1)
            weekly_window: Window for weekly RV (default 5)
            monthly_window: Window for monthly RV (default 22)
            jump_threshold: Z-score threshold for jump detection (default 3.0)
            annualization_factor: Trading days per year for annualization
        """
        self.daily_window = daily_window
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self.jump_threshold = jump_threshold
        self.annualization_factor = annualization_factor

        # Model coefficients
        self.beta0 = None  # Intercept
        self.beta_d = None  # Daily coefficient
        self.beta_w = None  # Weekly coefficient
        self.beta_m = None  # Monthly coefficient

        # Jump model coefficients (for HAR-RV-J)
        self.beta_j = None  # Jump coefficient
        self.beta_c = None  # Continuous component coefficient

        # Model statistics
        self.r_squared = None
        self.adjusted_r_squared = None
        self.residual_std = None

    # =========================================================================
    # REALIZED VOLATILITY COMPUTATIONS
    # =========================================================================

    def compute_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 1,
        squared: bool = True
    ) -> pd.Series:
        """
        Compute realized volatility from returns.

        Formula: RV = sqrt(sum(r_i^2))

        Args:
            returns: Return series
            window: Rolling window for aggregation
            squared: If True, return RV^2 (variance), else return RV (std)

        Returns:
            Realized volatility series
        """
        squared_returns = returns ** 2

        if window == 1:
            rv = squared_returns
        else:
            rv = squared_returns.rolling(window, min_periods=1).sum()

        if squared:
            return rv
        else:
            return np.sqrt(rv)

    def compute_bipower_variation(self, returns: pd.Series) -> pd.Series:
        """
        Compute Bipower Variation (BPV) - robust to jumps.

        Formula: BPV = (π/2) * sum(|r_i| * |r_{i-1}|)

        Used to separate continuous volatility from jumps.
        """
        mu1 = np.sqrt(2 / np.pi)
        abs_returns = returns.abs()
        bpv = (np.pi / 2) * (abs_returns * abs_returns.shift(1))
        return bpv.rolling(self.daily_window, min_periods=1).sum()

    def compute_jump_component(self, returns: pd.Series) -> pd.Series:
        """
        Compute jump component: RV - BPV (truncated at 0).

        Jumps are the difference between realized variance and bipower variation.
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        bpv = self.compute_bipower_variation(returns)
        jump = np.maximum(0, rv - bpv)
        return jump

    def compute_signed_jumps(self, returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Compute signed jump components (positive and negative).

        Returns:
            Tuple of (positive_jumps, negative_jumps)
        """
        jump = self.compute_jump_component(returns)

        # Detect jump direction based on return sign on jump days
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        bpv = self.compute_bipower_variation(returns)

        # Jump significance test
        is_jump = (rv - bpv) / (bpv + 1e-10) > self.jump_threshold

        # Positive jumps: up moves with significant jump
        pos_jump = np.where((returns > 0) & is_jump, jump, 0)
        # Negative jumps: down moves with significant jump
        neg_jump = np.where((returns < 0) & is_jump, jump, 0)

        return pd.Series(pos_jump, index=returns.index), pd.Series(neg_jump, index=returns.index)

    def compute_continuous_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Compute continuous component of volatility (BPV capped at RV).
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        bpv = self.compute_bipower_variation(returns)
        return np.minimum(rv, bpv)

    def compute_har_components(self, rv: pd.Series) -> pd.DataFrame:
        """
        Compute daily, weekly, monthly RV components for HAR model.

        Args:
            rv: Realized volatility series (squared returns)

        Returns:
            DataFrame with rv_daily, rv_weekly, rv_monthly columns
        """
        return pd.DataFrame({
            'rv_daily': rv,
            'rv_weekly': rv.rolling(self.weekly_window, min_periods=1).mean(),
            'rv_monthly': rv.rolling(self.monthly_window, min_periods=1).mean()
        }, index=rv.index)

    # =========================================================================
    # MODEL FITTING
    # =========================================================================

    def fit(self, returns: pd.Series, method: str = 'basic') -> 'HARRVVolatility':
        """
        Fit HAR-RV model using OLS.

        Args:
            returns: Return series
            method: 'basic' (HAR-RV), 'jump' (HAR-RV-J), 'cj' (HAR-RV-CJ), 'sj' (HAR-RV-SJ)

        Returns:
            Self for method chaining
        """
        # Compute realized volatility
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)

        # Get HAR components
        components = self.compute_har_components(rv)

        # Target: next-period RV
        y = rv.shift(-1)

        if method == 'basic':
            # HAR-RV: RV_t+1 ~ RV_d + RV_w + RV_m
            X = components.copy()
            X['intercept'] = 1
            X = X[['intercept', 'rv_daily', 'rv_weekly', 'rv_monthly']]

        elif method == 'jump':
            # HAR-RV-J: Add jump component
            jump = self.compute_jump_component(returns)
            X = components.copy()
            X['jump'] = jump
            X['intercept'] = 1
            X = X[['intercept', 'rv_daily', 'rv_weekly', 'rv_monthly', 'jump']]

        elif method == 'cj':
            # HAR-RV-CJ: Continuous + Jump decomposition
            continuous = self.compute_continuous_volatility(returns)
            jump = self.compute_jump_component(returns)
            X = pd.DataFrame({
                'intercept': 1,
                'continuous': continuous,
                'cont_weekly': continuous.rolling(self.weekly_window).mean(),
                'cont_monthly': continuous.rolling(self.monthly_window).mean(),
                'jump': jump,
                'jump_weekly': jump.rolling(self.weekly_window).mean(),
                'jump_monthly': jump.rolling(self.monthly_window).mean()
            }, index=returns.index)

        elif method == 'sj':
            # HAR-RV-SJ: Signed jumps
            pos_jump, neg_jump = self.compute_signed_jumps(returns)
            continuous = self.compute_continuous_volatility(returns)
            X = pd.DataFrame({
                'intercept': 1,
                'continuous': continuous,
                'cont_weekly': continuous.rolling(self.weekly_window).mean(),
                'cont_monthly': continuous.rolling(self.monthly_window).mean(),
                'pos_jump': pos_jump,
                'neg_jump': neg_jump
            }, index=returns.index)

        # Drop NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        if len(X_clean) < len(X.columns) + 1:
            # Not enough data points
            self.beta0 = 0
            self.beta_d = 0.3
            self.beta_w = 0.3
            self.beta_m = 0.3
            return self

        # OLS estimation: β = (X'X)^(-1) X'y
        try:
            XtX = X_clean.T @ X_clean
            Xty = X_clean.T @ y_clean
            betas = np.linalg.solve(XtX, Xty)

            # Store coefficients
            if method == 'basic':
                self.beta0, self.beta_d, self.beta_w, self.beta_m = betas
            elif method == 'jump':
                self.beta0, self.beta_d, self.beta_w, self.beta_m, self.beta_j = betas
            elif method == 'cj':
                self.beta0, self.beta_c = betas[0], betas[1]
                # Store weekly and monthly continuous as adjustments
                self.beta_d = betas[2]  # cont_weekly
                self.beta_w = betas[3]  # cont_monthly
                self.beta_j = betas[4]  # jump
                self.beta_m = betas[5] if len(betas) > 5 else 0
            elif method == 'sj':
                self.beta0, self.beta_c = betas[0], betas[1]
                self.beta_d = betas[2]
                self.beta_w = betas[3]
                self.beta_j = betas[4]  # pos_jump
                self.beta_m = betas[5]  # neg_jump (stored for reference)

            # Compute R-squared
            y_pred = X_clean @ betas
            ss_res = ((y_clean - y_pred) ** 2).sum()
            ss_tot = ((y_clean - y_clean.mean()) ** 2).sum()
            self.r_squared = 1 - ss_res / (ss_tot + 1e-10)

            n = len(y_clean)
            p = len(betas)
            self.adjusted_r_squared = 1 - (1 - self.r_squared) * (n - 1) / (n - p - 1 + 1e-10)
            self.residual_std = np.sqrt(ss_res / (n - p + 1e-10))

        except np.linalg.LinAlgError:
            # Fallback to simple weights
            self.beta0 = rv.mean() * 0.1
            self.beta_d = 0.35
            self.beta_w = 0.30
            self.beta_m = 0.25

        return self

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, returns: pd.Series, steps: int = 1) -> pd.Series:
        """
        Predict next-period volatility.

        Args:
            returns: Return series
            steps: Number of steps ahead (1 for next period)

        Returns:
            Predicted volatility series
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        components = self.compute_har_components(rv)

        # HAR prediction
        pred = (
            self.beta0 +
            self.beta_d * components['rv_daily'] +
            self.beta_w * components['rv_weekly'] +
            self.beta_m * components['rv_monthly']
        )

        # Multi-step ahead prediction (simple iteration)
        if steps > 1:
            for _ in range(steps - 1):
                # Update components with prediction
                rv_new = pred
                weekly_new = (components['rv_weekly'] * (self.weekly_window - 1) + rv_new) / self.weekly_window
                monthly_new = (components['rv_monthly'] * (self.monthly_window - 1) + rv_new) / self.monthly_window

                pred = (
                    self.beta0 +
                    self.beta_d * rv_new +
                    self.beta_w * weekly_new +
                    self.beta_m * monthly_new
                )

        return pred

    def predict_with_jumps(self, returns: pd.Series) -> pd.Series:
        """
        Predict volatility including jump component (HAR-RV-J).
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        components = self.compute_har_components(rv)
        jump = self.compute_jump_component(returns)

        pred = (
            self.beta0 +
            self.beta_d * components['rv_daily'] +
            self.beta_w * components['rv_weekly'] +
            self.beta_m * components['rv_monthly'] +
            (self.beta_j or 0) * jump
        )

        return pred

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1,
        annualize: bool = True
    ) -> float:
        """
        Get point forecast for future volatility.

        Args:
            returns: Return series
            horizon: Forecast horizon in periods
            annualize: Whether to annualize the volatility

        Returns:
            Volatility forecast
        """
        pred = self.predict(returns, steps=horizon)
        forecast = pred.iloc[-1] if len(pred) > 0 else 0

        # Convert from variance to volatility
        vol = np.sqrt(np.maximum(0, forecast))

        if annualize:
            vol = vol * np.sqrt(self.annualization_factor)

        return vol

    # =========================================================================
    # REGIME DETECTION
    # =========================================================================

    def get_volatility_regime(
        self,
        returns: pd.Series,
        low_percentile: float = 25,
        high_percentile: float = 75
    ) -> pd.Series:
        """
        Classify volatility regime: low, normal, high.

        Args:
            returns: Return series
            low_percentile: Percentile threshold for low volatility
            high_percentile: Percentile threshold for high volatility

        Returns:
            Series with regime labels ('low', 'normal', 'high')
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=False)

        low_thresh = rv.quantile(low_percentile / 100)
        high_thresh = rv.quantile(high_percentile / 100)

        regime = pd.Series('normal', index=returns.index)
        regime[rv < low_thresh] = 'low'
        regime[rv > high_thresh] = 'high'

        return regime

    def get_volatility_percentile(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Get rolling percentile rank of current volatility.

        Returns value between 0-100 indicating where current vol ranks historically.
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=False)
        return rv.rolling(window).apply(
            lambda x: stats.percentileofscore(x[:-1], x[-1]) if len(x) > 1 else 50,
            raw=False
        )

    def detect_vol_breakout(
        self,
        returns: pd.Series,
        lookback: int = 20,
        threshold: float = 2.0
    ) -> pd.Series:
        """
        Detect volatility breakouts (sudden vol spikes).

        Args:
            returns: Return series
            lookback: Lookback period for baseline
            threshold: Number of standard deviations for breakout

        Returns:
            Boolean series indicating breakout
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=False)
        rv_mean = rv.rolling(lookback).mean()
        rv_std = rv.rolling(lookback).std()

        z_score = (rv - rv_mean) / (rv_std + 1e-10)
        return z_score > threshold

    # =========================================================================
    # SIGNALS FOR TRADING
    # =========================================================================

    def vol_mean_reversion_signal(
        self,
        returns: pd.Series,
        z_entry: float = 2.0,
        z_exit: float = 0.5
    ) -> pd.Series:
        """
        Generate trading signals based on vol mean reversion.

        High vol = expect vol to drop = increase position
        Low vol = expect vol to spike = reduce position

        Args:
            returns: Return series
            z_entry: Z-score threshold for entry
            z_exit: Z-score threshold for exit

        Returns:
            Signal series: 1 (increase), -1 (reduce), 0 (neutral)
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=False)
        rv_mean = rv.rolling(22).mean()
        rv_std = rv.rolling(22).std()

        z_score = (rv - rv_mean) / (rv_std + 1e-10)

        signal = pd.Series(0, index=returns.index)
        signal[z_score > z_entry] = 1  # High vol, expect reversion down
        signal[z_score < -z_entry] = -1  # Low vol, expect spike up

        return signal

    def vol_momentum_signal(
        self,
        returns: pd.Series,
        short_window: int = 5,
        long_window: int = 22
    ) -> pd.Series:
        """
        Generate signals based on volatility momentum.

        Rising vol = reduce positions
        Falling vol = increase positions

        Returns:
            Signal series: 1 (increasing risk), -1 (decreasing risk), 0 (neutral)
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=False)

        rv_short = rv.rolling(short_window).mean()
        rv_long = rv.rolling(long_window).mean()

        # Momentum: short above long = rising vol
        signal = pd.Series(0, index=returns.index)
        signal[rv_short > rv_long * 1.1] = -1  # Rising vol, reduce
        signal[rv_short < rv_long * 0.9] = 1   # Falling vol, increase

        return signal

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_model_summary(self) -> Dict:
        """Get summary of fitted model."""
        return {
            'beta0': self.beta0,
            'beta_daily': self.beta_d,
            'beta_weekly': self.beta_w,
            'beta_monthly': self.beta_m,
            'beta_jump': self.beta_j,
            'r_squared': self.r_squared,
            'adj_r_squared': self.adjusted_r_squared,
            'residual_std': self.residual_std
        }

    def get_component_contributions(self, returns: pd.Series) -> pd.DataFrame:
        """
        Get contribution of each HAR component to forecast.

        Returns DataFrame with daily, weekly, monthly contributions.
        """
        rv = self.compute_realized_volatility(returns, self.daily_window, squared=True)
        components = self.compute_har_components(rv)

        return pd.DataFrame({
            'daily_contrib': self.beta_d * components['rv_daily'],
            'weekly_contrib': self.beta_w * components['rv_weekly'],
            'monthly_contrib': self.beta_m * components['rv_monthly'],
            'intercept': self.beta0,
            'total_forecast': self.predict(returns)
        }, index=returns.index)


class HARRVForex(HARRVVolatility):
    """
    HAR-RV specialized for forex tick data.

    Adjustments:
    - 24-hour market (no overnight gaps)
    - Tick-level realized volatility
    - Session-aware components (Asian/London/NY)
    """

    def __init__(
        self,
        tick_aggregation: str = '5min',
        asian_hours: Tuple[int, int] = (0, 8),
        london_hours: Tuple[int, int] = (8, 16),
        ny_hours: Tuple[int, int] = (16, 24)
    ):
        """
        Initialize forex-specific HAR-RV.

        Args:
            tick_aggregation: Time frequency for tick aggregation
            asian_hours: Hour range for Asian session
            london_hours: Hour range for London session
            ny_hours: Hour range for NY session
        """
        super().__init__(
            daily_window=1,
            weekly_window=5,
            monthly_window=22,
            annualization_factor=252 * 24  # 24-hour market
        )

        self.tick_aggregation = tick_aggregation
        self.asian_hours = asian_hours
        self.london_hours = london_hours
        self.ny_hours = ny_hours

    def compute_session_rv(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute realized volatility by trading session.

        Returns DataFrame with Asian, London, NY RV components.
        """
        if not hasattr(returns.index, 'hour'):
            # Not datetime index, return simple RV
            rv = self.compute_realized_volatility(returns)
            return pd.DataFrame({
                'rv_asian': rv / 3,
                'rv_london': rv / 3,
                'rv_ny': rv / 3
            }, index=returns.index)

        hour = returns.index.hour

        asian_mask = (hour >= self.asian_hours[0]) & (hour < self.asian_hours[1])
        london_mask = (hour >= self.london_hours[0]) & (hour < self.london_hours[1])
        ny_mask = (hour >= self.ny_hours[0]) | (hour < self.ny_hours[0])  # Wrap around

        rv_asian = (returns ** 2).where(asian_mask, 0).rolling(24).sum()
        rv_london = (returns ** 2).where(london_mask, 0).rolling(24).sum()
        rv_ny = (returns ** 2).where(ny_mask, 0).rolling(24).sum()

        return pd.DataFrame({
            'rv_asian': rv_asian,
            'rv_london': rv_london,
            'rv_ny': rv_ny
        }, index=returns.index)


if __name__ == '__main__':
    # Test
    print("HAR-RV Volatility - Testing")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.randn(n) * 0.01, name='returns')

    # Add some volatility clustering
    vol_process = np.ones(n)
    for i in range(1, n):
        vol_process[i] = 0.9 * vol_process[i-1] + 0.1 * np.random.rand()
    returns = returns * vol_process

    # Basic HAR-RV
    har = HARRVVolatility()
    har.fit(returns, method='basic')

    print("Model Summary:")
    summary = har.get_model_summary()
    for key, val in summary.items():
        if val is not None:
            print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")

    # Forecast
    forecast = har.forecast_volatility(returns, horizon=1, annualize=True)
    print(f"\nAnnualized Vol Forecast: {forecast:.4%}")

    # Regime
    regime = har.get_volatility_regime(returns)
    print(f"\nCurrent Regime: {regime.iloc[-1]}")
    print(f"Regime Distribution: {regime.value_counts().to_dict()}")

    # HAR-RV-J (with jumps)
    har_j = HARRVVolatility()
    har_j.fit(returns, method='jump')
    print(f"\nHAR-RV-J R-squared: {har_j.r_squared:.4f}")

    # Signals
    signal = har.vol_mean_reversion_signal(returns)
    print(f"\nVol Mean Reversion Signal: {signal.iloc[-1]}")
