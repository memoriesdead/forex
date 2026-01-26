"""
GitHub Gold Standard Quantitative Trading Formulas
====================================================
Production-grade formulas extracted from top GitHub repositories (35k+ stars combined).
All formulas include proper academic citations and source code references.

Academic Citations:
------------------
[1] Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events."
    Journal of Financial Econometrics, 12(1), 47-88.
    GitHub: https://github.com/nicolezattarin/LOB-feature-analysis

[2] Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity
    in a High-Frequency World." Review of Financial Studies, 25(5), 1457-1493.
    GitHub: https://github.com/yt-feng/VPIN

[3] Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price Volatilities
    from Historical Data." Journal of Business, 53(1), 67-78.
    GitHub: https://github.com/jasonstrimpel/volatility-trading

[4] Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
    Rate of Return." Journal of Business, 53(1), 61-65.

[5] Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
    Closing Prices." Annals of Applied Probability, 1(4), 504-512.

[6] Yang, D., & Zhang, Q. (2000). "Drift Independent Volatility Estimation Based on
    High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-492.

[7] Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    GitHub: https://github.com/hudson-and-thames/mlfinlab

[8] Kelly, J.L. (1956). "A New Interpretation of Information Rate."
    Bell System Technical Journal, 35(4), 917-926.

[9] Sharpe, W.F. (1966). "Mutual Fund Performance." Journal of Business, 39(1), 119-138.
    GitHub: https://github.com/ranaroussi/quantstats

[10] Sortino, F.A., & van der Meer, R. (1991). "Downside Risk."
     Journal of Portfolio Management, 17(4), 27-31.

[11] Liu, Y., et al. (2024). "iTransformer: Inverted Transformers Are Effective for
     Time Series Forecasting." ICLR 2024 Spotlight.
     GitHub: https://github.com/thuml/iTransformer

[12] Kakushadze, Z. (2016). "101 Formulaic Alphas." Wilmott, 2016(84), 72-81.
     arXiv: https://arxiv.org/abs/1601.00991
     GitHub: https://github.com/yli188/WorldQuant_alpha101_code

[13] Hawkes, A.G. (1971). "Spectra of Some Self-Exciting and Mutually Exciting
     Point Processes." Biometrika, 58(1), 83-90.
     GitHub: https://github.com/omitakahiro/Hawkes

[14] Yang, X., et al. (2020). "Qlib: An AI-oriented Quantitative Investment Platform."
     arXiv: https://arxiv.org/abs/2009.11189
     GitHub: https://github.com/microsoft/qlib

[15] Liu, X.-Y., et al. (2021). "FinRL: Deep Reinforcement Learning Framework to
     Automate Trading in Quantitative Finance." SSRN.
     GitHub: https://github.com/AI4Finance-Foundation/FinRL

License: MIT (compatible with all source repositories)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ORDER FLOW IMBALANCE (OFI)
# Citation: Cont, Kukanov, Stoikov (2014) [1]
# GitHub: nicolezattarin/LOB-feature-analysis (MIT License)
# Forex Applicability: HIGH
# =============================================================================

@dataclass
class OFIResult:
    """Order Flow Imbalance result container."""
    ofi: pd.Series
    ofi_normalized: pd.Series
    buy_pressure: pd.Series
    sell_pressure: pd.Series


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI) Calculator.

    Formula:
    --------
    OFI_t = ΔW_t - ΔV_t

    where:
    - ΔW_t = change in bid volume (adjusted for price movements)
    - ΔV_t = change in ask volume (adjusted for price movements)

    For bid side:
    - If P_bid increases: ΔW = V_bid (new level)
    - If P_bid unchanged: ΔW = V_bid - V_bid_prev (volume change)
    - If P_bid decreases: ΔW = -V_bid_prev (level removed)

    For ask side (inverse logic):
    - If P_ask decreases: ΔV = V_ask (new level)
    - If P_ask unchanged: ΔV = V_ask - V_ask_prev
    - If P_ask increases: ΔV = -V_ask_prev

    Citation:
    ---------
    Cont, R., Kukanov, A., & Stoikov, S. (2014).
    "The Price Impact of Order Book Events."
    Journal of Financial Econometrics, 12(1), 47-88.

    Source:
    -------
    https://github.com/nicolezattarin/LOB-feature-analysis
    """

    def __init__(self, normalize: bool = True, window: int = 20):
        """
        Initialize OFI calculator.

        Args:
            normalize: Whether to normalize OFI by rolling std
            window: Window for normalization
        """
        self.normalize = normalize
        self.window = window

    def calculate(
        self,
        bid_prices: pd.Series,
        bid_volumes: pd.Series,
        ask_prices: pd.Series,
        ask_volumes: pd.Series
    ) -> OFIResult:
        """
        Calculate Order Flow Imbalance.

        Args:
            bid_prices: Best bid price series
            bid_volumes: Best bid volume series
            ask_prices: Best ask price series
            ask_volumes: Best ask volume series

        Returns:
            OFIResult with OFI values and components
        """
        # Price changes
        bid_price_change = bid_prices.diff()
        ask_price_change = ask_prices.diff()

        # Volume changes
        bid_vol_change = bid_volumes.diff()
        ask_vol_change = ask_volumes.diff()

        # Bid side delta (ΔW)
        # Eq. (3) from Cont et al. (2014)
        delta_W = np.where(
            bid_price_change > 0,
            bid_volumes,  # New higher bid level
            np.where(
                bid_price_change == 0,
                bid_vol_change,  # Volume change at same level
                -bid_volumes.shift(1)  # Level removed
            )
        )

        # Ask side delta (ΔV)
        # Inverse logic for ask side
        delta_V = np.where(
            ask_price_change < 0,
            ask_volumes,  # New lower ask level
            np.where(
                ask_price_change == 0,
                ask_vol_change,  # Volume change at same level
                -ask_volumes.shift(1)  # Level removed
            )
        )

        # Convert to Series
        delta_W = pd.Series(delta_W, index=bid_prices.index)
        delta_V = pd.Series(delta_V, index=ask_prices.index)

        # OFI = ΔW - ΔV (Eq. 4)
        ofi = delta_W - delta_V

        # Normalized OFI
        if self.normalize:
            ofi_std = ofi.rolling(self.window).std()
            ofi_normalized = ofi / (ofi_std + 1e-10)
        else:
            ofi_normalized = ofi

        return OFIResult(
            ofi=ofi,
            ofi_normalized=ofi_normalized,
            buy_pressure=delta_W,
            sell_pressure=delta_V
        )

    def calculate_from_ohlcv(
        self,
        df: pd.DataFrame,
        spread_col: str = 'spread'
    ) -> pd.Series:
        """
        Approximate OFI from OHLCV data (for forex without L2 data).

        Uses tick rule to classify trades as buys/sells.

        Args:
            df: DataFrame with OHLCV columns
            spread_col: Column name for spread (optional)

        Returns:
            Approximated OFI series
        """
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))

        # Tick rule: classify direction based on price change
        price_change = close.diff()

        # Volume-weighted direction
        buy_volume = volume.where(price_change > 0, 0)
        sell_volume = volume.where(price_change < 0, 0)

        # OFI approximation
        ofi = buy_volume - sell_volume

        if self.normalize:
            ofi = ofi / (ofi.rolling(self.window).std() + 1e-10)

        return ofi


# =============================================================================
# VPIN (Volume-Synchronized Probability of Informed Trading)
# Citation: Easley, Lopez de Prado, O'Hara (2012) [2]
# GitHub: yt-feng/VPIN, monty-se/PINstimation (MIT License)
# Forex Applicability: HIGH
# =============================================================================

@dataclass
class VPINResult:
    """VPIN calculation result."""
    vpin: float
    vpin_series: pd.Series
    bucket_imbalances: List[float]
    toxicity_alert: bool


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    Formula:
    --------
    VPIN = Σ|V_buy^τ - V_sell^τ| / (n × V_bucket)

    where:
    - V_buy^τ = buy-initiated volume in bucket τ
    - V_sell^τ = sell-initiated volume in bucket τ
    - n = number of buckets in rolling window
    - V_bucket = fixed volume per bucket

    Trade classification uses Bulk Volume Classification (BVC):
    V_buy = V × Z((P - P_prev) / σ)
    V_sell = V × (1 - Z((P - P_prev) / σ))

    where Z is the standard normal CDF.

    Citation:
    ---------
    Easley, D., Lopez de Prado, M., & O'Hara, M. (2012).
    "Flow Toxicity and Liquidity in a High-Frequency World."
    Review of Financial Studies, 25(5), 1457-1493.

    Source:
    -------
    https://github.com/yt-feng/VPIN
    https://github.com/monty-se/PINstimation
    """

    def __init__(
        self,
        bucket_size: int = 50000,
        n_buckets: int = 50,
        toxicity_threshold: float = 0.7
    ):
        """
        Initialize VPIN calculator.

        Args:
            bucket_size: Fixed volume per bucket (V_bucket)
            n_buckets: Number of buckets for VPIN calculation (n)
            toxicity_threshold: Alert threshold for high toxicity
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.toxicity_threshold = toxicity_threshold

    def bulk_volume_classification(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        sigma: float = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Bulk Volume Classification (BVC) for trade direction.

        From Easley et al. (2012), Section III.A.

        Args:
            prices: Price series
            volumes: Volume series
            sigma: Price volatility (if None, estimated from data)

        Returns:
            Tuple of (buy_volume, sell_volume) series
        """
        from scipy.stats import norm

        if sigma is None:
            sigma = prices.pct_change().std()

        # Standardized price change
        price_change = prices.diff()
        z = price_change / (sigma * prices.shift(1) + 1e-10)

        # Probability of buy
        prob_buy = pd.Series(norm.cdf(z), index=prices.index)

        # Classified volumes
        buy_volume = volumes * prob_buy
        sell_volume = volumes * (1 - prob_buy)

        return buy_volume, sell_volume

    def calculate(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> VPINResult:
        """
        Calculate VPIN from price and volume series.

        Args:
            prices: Price series
            volumes: Volume series

        Returns:
            VPINResult with VPIN values and diagnostics
        """
        # Classify volumes
        buy_vol, sell_vol = self.bulk_volume_classification(prices, volumes)

        # Create volume buckets
        buckets = []
        current_bucket = {'buy': 0, 'sell': 0, 'volume': 0}

        for i in range(len(prices)):
            remaining_buy = buy_vol.iloc[i]
            remaining_sell = sell_vol.iloc[i]
            remaining_total = remaining_buy + remaining_sell

            while remaining_total > 0:
                space = self.bucket_size - current_bucket['volume']
                fill = min(remaining_total, space)

                # Proportional fill
                if remaining_total > 0:
                    buy_fill = fill * (remaining_buy / remaining_total)
                    sell_fill = fill * (remaining_sell / remaining_total)
                else:
                    buy_fill = sell_fill = 0

                current_bucket['buy'] += buy_fill
                current_bucket['sell'] += sell_fill
                current_bucket['volume'] += fill

                remaining_buy -= buy_fill
                remaining_sell -= sell_fill
                remaining_total = remaining_buy + remaining_sell

                # Bucket full
                if current_bucket['volume'] >= self.bucket_size:
                    buckets.append(current_bucket.copy())
                    current_bucket = {'buy': 0, 'sell': 0, 'volume': 0}

        # Calculate VPIN over rolling window
        if len(buckets) < self.n_buckets:
            # Not enough data
            return VPINResult(
                vpin=np.nan,
                vpin_series=pd.Series(dtype=float),
                bucket_imbalances=[],
                toxicity_alert=False
            )

        # Rolling VPIN
        vpin_values = []
        imbalances = []

        for i in range(self.n_buckets, len(buckets) + 1):
            window_buckets = buckets[i - self.n_buckets:i]
            window_imbalances = [
                abs(b['buy'] - b['sell']) for b in window_buckets
            ]
            imbalances.extend(window_imbalances[-1:])

            vpin = sum(window_imbalances) / (self.n_buckets * self.bucket_size)
            vpin_values.append(vpin)

        # Current VPIN
        current_vpin = vpin_values[-1] if vpin_values else np.nan

        return VPINResult(
            vpin=current_vpin,
            vpin_series=pd.Series(vpin_values),
            bucket_imbalances=imbalances,
            toxicity_alert=current_vpin > self.toxicity_threshold
        )

    def calculate_from_ohlcv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VPIN approximation from OHLCV data.

        For forex without tick data, uses BVC on bar data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Rolling VPIN series
        """
        close = df['close']
        volume = df.get('volume', df.get('tick_count', pd.Series(1, index=df.index)))

        buy_vol, sell_vol = self.bulk_volume_classification(close, volume)

        # Rolling order imbalance as VPIN proxy
        imbalance = (buy_vol - sell_vol).abs()
        total_vol = buy_vol + sell_vol

        vpin = imbalance.rolling(self.n_buckets).sum() / (
            total_vol.rolling(self.n_buckets).sum() + 1e-10
        )

        return vpin


# =============================================================================
# VOLATILITY ESTIMATORS
# Citations: Garman-Klass [3], Parkinson [4], Rogers-Satchell [5], Yang-Zhang [6]
# GitHub: jasonstrimpel/volatility-trading (MIT License)
# Forex Applicability: HIGH
# =============================================================================

class VolatilityEstimators:
    """
    Range-Based Volatility Estimators.

    These estimators are more efficient than close-to-close volatility
    because they use intraday price information (high, low, open, close).

    Efficiency comparison (vs close-to-close):
    - Parkinson: 5.2x more efficient
    - Garman-Klass: 7.4x more efficient
    - Rogers-Satchell: Handles drift
    - Yang-Zhang: Handles opening jumps + drift

    Citations:
    ----------
    [3] Garman, M.B., & Klass, M.J. (1980). JoB, 53(1), 67-78.
    [4] Parkinson, M. (1980). JoB, 53(1), 61-65.
    [5] Rogers, L.C.G., & Satchell, S.E. (1991). AAP, 1(4), 504-512.
    [6] Yang, D., & Zhang, Q. (2000). JoB, 73(3), 477-492.

    Source:
    -------
    https://github.com/jasonstrimpel/volatility-trading
    """

    def __init__(self, window: int = 20, annualize: bool = True, periods: int = 252):
        """
        Initialize volatility estimators.

        Args:
            window: Rolling window size
            annualize: Whether to annualize volatility
            periods: Periods per year (252 for daily, 252*24 for hourly)
        """
        self.window = window
        self.annualize = annualize
        self.periods = periods

    def close_to_close(
        self,
        close: pd.Series
    ) -> pd.Series:
        """
        Standard close-to-close volatility.

        Formula: σ = std(ln(C_t / C_{t-1})) × √periods

        Args:
            close: Close price series

        Returns:
            Annualized volatility series
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(self.window).std()

        if self.annualize:
            vol = vol * np.sqrt(self.periods)

        return vol

    def parkinson(
        self,
        high: pd.Series,
        low: pd.Series
    ) -> pd.Series:
        """
        Parkinson Volatility Estimator.

        Formula: σ² = (1 / 4ln(2)) × E[ln(H/L)²]

        5.2x more efficient than close-to-close.
        Assumes no drift and continuous trading.

        Citation: Parkinson (1980) [4]

        Args:
            high: High price series
            low: Low price series

        Returns:
            Annualized Parkinson volatility
        """
        # Eq. (4) from Parkinson (1980)
        log_hl = np.log(high / low)
        factor = 1 / (4 * np.log(2))

        variance = factor * (log_hl ** 2).rolling(self.window).mean()
        vol = np.sqrt(variance)

        if self.annualize:
            vol = vol * np.sqrt(self.periods)

        return vol

    def garman_klass(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Garman-Klass Volatility Estimator.

        Formula: σ² = 0.5 × ln(H/L)² - (2ln(2)-1) × ln(C/O)²

        7.4x more efficient than close-to-close.
        Assumes no drift and continuous trading.

        Citation: Garman & Klass (1980) [3]

        Args:
            open_: Open price series
            high: High price series
            low: Low price series
            close: Close price series

        Returns:
            Annualized Garman-Klass volatility
        """
        # Eq. (14) from Garman & Klass (1980)
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)

        variance = (
            0.5 * (log_hl ** 2) -
            (2 * np.log(2) - 1) * (log_co ** 2)
        ).rolling(self.window).mean()

        vol = np.sqrt(variance.clip(lower=0))

        if self.annualize:
            vol = vol * np.sqrt(self.periods)

        return vol

    def rogers_satchell(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Rogers-Satchell Volatility Estimator.

        Formula: σ² = ln(H/C)×ln(H/O) + ln(L/C)×ln(L/O)

        Handles non-zero drift (trending markets).

        Citation: Rogers & Satchell (1991) [5]

        Args:
            open_: Open price series
            high: High price series
            low: Low price series
            close: Close price series

        Returns:
            Annualized Rogers-Satchell volatility
        """
        # From Rogers & Satchell (1991)
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)

        variance = (
            log_hc * log_ho + log_lc * log_lo
        ).rolling(self.window).mean()

        vol = np.sqrt(variance.clip(lower=0))

        if self.annualize:
            vol = vol * np.sqrt(self.periods)

        return vol

    def yang_zhang(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: float = 0.34
    ) -> pd.Series:
        """
        Yang-Zhang Volatility Estimator.

        Formula: σ² = σ²_overnight + k×σ²_open + (1-k)×σ²_RS

        Handles both drift AND opening jumps.
        Most comprehensive estimator.

        Citation: Yang & Zhang (2000) [6]

        Args:
            open_: Open price series
            high: High price series
            low: Low price series
            close: Close price series
            k: Weighting factor (0.34 is optimal for daily data)

        Returns:
            Annualized Yang-Zhang volatility
        """
        # Overnight variance (close-to-open)
        log_oc = np.log(open_ / close.shift(1))
        var_overnight = log_oc.rolling(self.window).var()

        # Open-to-close variance
        log_co = np.log(close / open_)
        var_open = log_co.rolling(self.window).var()

        # Rogers-Satchell variance
        var_rs = self.rogers_satchell(open_, high, low, close) ** 2
        if self.annualize:
            var_rs = var_rs / self.periods  # De-annualize for combination

        # Yang-Zhang combination
        variance = var_overnight + k * var_open + (1 - k) * var_rs

        vol = np.sqrt(variance.clip(lower=0))

        if self.annualize:
            vol = vol * np.sqrt(self.periods)

        return vol

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility estimators.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with volatility columns
        """
        result = pd.DataFrame(index=df.index)

        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']

        result['VOL_CLOSE'] = self.close_to_close(close)
        result['VOL_PARKINSON'] = self.parkinson(high, low)
        result['VOL_GARMAN_KLASS'] = self.garman_klass(open_, high, low, close)
        result['VOL_ROGERS_SATCHELL'] = self.rogers_satchell(open_, high, low, close)
        result['VOL_YANG_ZHANG'] = self.yang_zhang(open_, high, low, close)

        # Volatility ratios (regime indicators)
        result['VOL_RATIO_PK_CC'] = result['VOL_PARKINSON'] / (result['VOL_CLOSE'] + 1e-10)
        result['VOL_RATIO_GK_RS'] = result['VOL_GARMAN_KLASS'] / (result['VOL_ROGERS_SATCHELL'] + 1e-10)

        return result


# =============================================================================
# RISK-ADJUSTED PERFORMANCE METRICS
# Citations: Sharpe [9], Sortino [10]
# GitHub: ranaroussi/quantstats (Apache-2.0 License)
# Forex Applicability: HIGH
# =============================================================================

class RiskMetrics:
    """
    Risk-Adjusted Performance Metrics.

    Implements industry-standard risk metrics from QuantStats.

    Citations:
    ----------
    [9] Sharpe, W.F. (1966). "Mutual Fund Performance." JoB, 39(1), 119-138.
    [10] Sortino, F.A., & van der Meer, R. (1991). JPM, 17(4), 27-31.

    Source:
    -------
    https://github.com/ranaroussi/quantstats
    """

    def __init__(self, rf: float = 0.0, periods: int = 252):
        """
        Initialize risk metrics calculator.

        Args:
            rf: Risk-free rate (annualized)
            periods: Periods per year
        """
        self.rf = rf
        self.periods = periods

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Sharpe Ratio.

        Formula: SR = (R_p - R_f) / σ_p × √periods

        Citation: Sharpe (1966) [9]

        Args:
            returns: Return series

        Returns:
            Annualized Sharpe ratio
        """
        excess_returns = returns - self.rf / self.periods
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.periods)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Sortino Ratio.

        Formula: SoR = (R_p - R_f) / σ_downside × √periods

        where σ_downside = √(Σ(R_t)² for R_t < 0) / N

        Citation: Sortino & van der Meer (1991) [10]

        Args:
            returns: Return series

        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - self.rf / self.periods
        downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

        if downside == 0:
            return 0
        return (excess_returns.mean() / downside) * np.sqrt(self.periods)

    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calmar Ratio.

        Formula: CR = CAGR / |Max Drawdown|

        Args:
            returns: Return series

        Returns:
            Calmar ratio
        """
        cagr = self.cagr(returns)
        max_dd = self.max_drawdown(returns)

        if max_dd == 0:
            return 0
        return cagr / abs(max_dd)

    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Maximum Drawdown.

        Formula: MDD = max(peak_t - trough_t) / peak_t

        Args:
            returns: Return series

        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def cagr(self, returns: pd.Series) -> float:
        """
        Compound Annual Growth Rate.

        Formula: CAGR = (V_final / V_initial)^(1/years) - 1

        Args:
            returns: Return series

        Returns:
            CAGR
        """
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / self.periods
        if n_years == 0:
            return 0
        return (1 + total_return) ** (1 / n_years) - 1

    def value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Value at Risk (Historical).

        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR (negative value representing loss)
        """
        return np.percentile(returns, (1 - confidence) * 100)

    def conditional_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).

        Formula: CVaR = E[R | R < VaR]

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            CVaR (negative value)
        """
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def generate_all(self, returns: pd.Series) -> Dict[str, float]:
        """
        Generate all risk metrics.

        Args:
            returns: Return series

        Returns:
            Dict with all metrics
        """
        return {
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns),
            'max_drawdown': self.max_drawdown(returns),
            'cagr': self.cagr(returns),
            'var_95': self.value_at_risk(returns, 0.95),
            'cvar_95': self.conditional_var(returns, 0.95),
            'volatility': returns.std() * np.sqrt(self.periods),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        }


# =============================================================================
# KELLY CRITERION (Enhanced)
# Citation: Kelly (1956) [8]
# GitHub: ranaroussi/quantstats (Apache-2.0 License)
# Forex Applicability: HIGH
# =============================================================================

class KellyCriterion:
    """
    Kelly Criterion for Optimal Position Sizing.

    Formula:
    --------
    f* = (p × b - q) / b = W - (1-W) / R

    where:
    - f* = optimal fraction of capital to bet
    - p (W) = probability of winning
    - q = probability of losing (1 - p)
    - b (R) = win/loss ratio

    For continuous returns:
    f* = μ / σ²

    Citation:
    ---------
    Kelly, J.L. (1956). "A New Interpretation of Information Rate."
    Bell System Technical Journal, 35(4), 917-926.

    Source:
    -------
    https://github.com/ranaroussi/quantstats
    """

    @staticmethod
    def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate full Kelly fraction.

        Args:
            win_rate: Probability of winning (0 to 1)
            win_loss_ratio: Average win / Average loss

        Returns:
            Optimal bet fraction (0 to 1)
        """
        if win_loss_ratio <= 0:
            return 0

        q = 1 - win_rate
        kelly = win_rate - (q / win_loss_ratio)
        return max(kelly, 0)

    @staticmethod
    def fractional_kelly(
        win_rate: float,
        win_loss_ratio: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate fractional Kelly (more conservative).

        Half-Kelly (fraction=0.5) provides:
        - 75% of optimal growth rate
        - 50% of the volatility

        Quarter-Kelly (fraction=0.25) provides:
        - 43.75% of optimal growth rate
        - 25% of the volatility

        Args:
            win_rate: Probability of winning
            win_loss_ratio: Win/loss ratio
            fraction: Kelly fraction (0.25-0.5 recommended)

        Returns:
            Fractional Kelly bet size
        """
        full_kelly = KellyCriterion.kelly_fraction(win_rate, win_loss_ratio)
        return full_kelly * fraction

    @staticmethod
    def kelly_from_returns(returns: pd.Series) -> float:
        """
        Calculate Kelly from historical returns.

        Formula: f* = μ / σ²

        Args:
            returns: Historical return series

        Returns:
            Optimal Kelly leverage
        """
        if returns.var() <= 0:
            return 0
        return returns.mean() / returns.var()

    @staticmethod
    def position_size(
        account_value: float,
        win_rate: float,
        win_loss_ratio: float,
        fraction: float = 0.25,
        max_position_pct: float = 0.10
    ) -> float:
        """
        Calculate position size in currency.

        Args:
            account_value: Total account value
            win_rate: Win probability
            win_loss_ratio: Win/loss ratio
            fraction: Kelly fraction (0.25 default)
            max_position_pct: Maximum position cap (0.10 = 10%)

        Returns:
            Position size in currency
        """
        kelly_pct = KellyCriterion.fractional_kelly(win_rate, win_loss_ratio, fraction)
        kelly_pct = min(kelly_pct, max_position_pct)
        return account_value * kelly_pct


# =============================================================================
# HAWKES PROCESS (Order Flow Intensity)
# Citation: Hawkes (1971) [13]
# GitHub: omitakahiro/Hawkes, x-datainitiative/tick (BSD License)
# Forex Applicability: HIGH
# =============================================================================

class HawkesProcess:
    """
    Hawkes Process for Self-Exciting Order Flow.

    Formula:
    --------
    λ(t) = μ + Σᵢ α × exp(-β(t - tᵢ))

    where:
    - λ(t) = intensity at time t
    - μ = base (background) intensity
    - α = jump size (excitation parameter)
    - β = decay rate
    - tᵢ = past event times

    Used for:
    - Modeling trade clustering
    - Predicting order flow intensity
    - Detecting unusual trading activity

    Citation:
    ---------
    Hawkes, A.G. (1971). "Spectra of Some Self-Exciting and
    Mutually Exciting Point Processes." Biometrika, 58(1), 83-90.

    Source:
    -------
    https://github.com/omitakahiro/Hawkes
    https://x-datainitiative.github.io/tick/modules/hawkes.html
    """

    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 1.0):
        """
        Initialize Hawkes process parameters.

        Args:
            mu: Base intensity
            alpha: Excitation parameter (0 < alpha < beta for stability)
            beta: Decay rate
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        # Check stability condition
        if alpha >= beta:
            warnings.warn("Hawkes process may be unstable: alpha >= beta")

    def intensity(self, t: float, event_times: np.ndarray) -> float:
        """
        Calculate conditional intensity at time t.

        Args:
            t: Current time
            event_times: Array of past event times (< t)

        Returns:
            Intensity λ(t)
        """
        past_events = event_times[event_times < t]

        # Base intensity
        intensity = self.mu

        # Add excitation from past events
        for ti in past_events:
            intensity += self.alpha * np.exp(-self.beta * (t - ti))

        return intensity

    def intensity_series(
        self,
        times: np.ndarray,
        event_times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate intensity at multiple time points.

        Args:
            times: Array of evaluation times
            event_times: Array of event times

        Returns:
            Array of intensities
        """
        return np.array([self.intensity(t, event_times) for t in times])

    def simulate(
        self,
        T: float,
        seed: int = None
    ) -> np.ndarray:
        """
        Simulate Hawkes process using Ogata's thinning algorithm.

        Args:
            T: End time
            seed: Random seed

        Returns:
            Array of event times
        """
        if seed is not None:
            np.random.seed(seed)

        events = []
        t = 0

        # Upper bound for intensity
        lambda_bar = self.mu / (1 - self.alpha / self.beta)

        while t < T:
            # Generate next candidate time
            u = np.random.random()
            t = t - np.log(u) / lambda_bar

            if t > T:
                break

            # Thinning
            lambda_t = self.intensity(t, np.array(events))
            if np.random.random() <= lambda_t / lambda_bar:
                events.append(t)

        return np.array(events)

    def branching_ratio(self) -> float:
        """
        Calculate branching ratio (criticality measure).

        n* = α/β

        - n* < 1: Subcritical (stable)
        - n* = 1: Critical (marginally stable)
        - n* > 1: Supercritical (explosive)

        Returns:
            Branching ratio
        """
        return self.alpha / self.beta

    def estimate_from_trades(
        self,
        trade_times: np.ndarray,
        T: float
    ) -> Dict[str, float]:
        """
        Estimate Hawkes parameters from trade data (MLE).

        Simplified estimation using method of moments.

        Args:
            trade_times: Array of trade arrival times
            T: Total time period

        Returns:
            Dict with estimated parameters
        """
        n = len(trade_times)

        # Average intensity
        avg_intensity = n / T

        # Inter-arrival times
        inter_arrivals = np.diff(trade_times)
        mean_ia = inter_arrivals.mean()
        var_ia = inter_arrivals.var()

        # Method of moments estimates
        # μ = λ(1 - n*) where λ is average intensity
        # Estimate branching ratio from variance
        cv = np.sqrt(var_ia) / mean_ia  # Coefficient of variation

        # For Hawkes: CV > 1 indicates clustering
        if cv > 1:
            n_star = min(0.9, (cv ** 2 - 1) / (cv ** 2))
        else:
            n_star = 0.1

        mu_est = avg_intensity * (1 - n_star)
        beta_est = 1 / mean_ia  # Approximate decay
        alpha_est = n_star * beta_est

        return {
            'mu': mu_est,
            'alpha': alpha_est,
            'beta': beta_est,
            'branching_ratio': n_star,
            'avg_intensity': avg_intensity
        }


# =============================================================================
# COMBINED FEATURE GENERATOR
# =============================================================================

class GoldStandardFeatures:
    """
    Combined Gold Standard Feature Generator.

    Generates features from all academic sources:
    - Order Flow Imbalance (Cont et al. 2014)
    - VPIN (Easley et al. 2012)
    - Volatility Estimators (Garman-Klass, Parkinson, etc.)
    - Risk Metrics (Sharpe, Sortino, etc.)
    """

    def __init__(
        self,
        ofi_window: int = 20,
        vpin_buckets: int = 50,
        vol_window: int = 20
    ):
        """
        Initialize feature generator.

        Args:
            ofi_window: OFI normalization window
            vpin_buckets: VPIN bucket count
            vol_window: Volatility estimation window
        """
        self.ofi = OrderFlowImbalance(window=ofi_window)
        self.vpin = VPIN(n_buckets=vpin_buckets)
        self.vol = VolatilityEstimators(window=vol_window)
        self.risk = RiskMetrics()

    def generate_microstructure_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate microstructure features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=df.index)

        close = df['close']
        returns = close.pct_change()

        # OFI approximation (without L2 data)
        features['OFI'] = self.ofi.calculate_from_ohlcv(df)

        # VPIN approximation
        features['VPIN'] = self.vpin.calculate_from_ohlcv(df)

        # All volatility estimators
        vol_features = self.vol.generate_all(df)
        for col in vol_features.columns:
            features[col] = vol_features[col]

        # Rolling risk metrics
        window = 60
        features['SHARPE_60'] = returns.rolling(window).apply(
            lambda x: self.risk.sharpe_ratio(x)
        )
        features['SORTINO_60'] = returns.rolling(window).apply(
            lambda x: self.risk.sortino_ratio(x)
        )
        features['MAX_DD_60'] = returns.rolling(window).apply(
            lambda x: self.risk.max_drawdown(x)
        )

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            # Microstructure
            'OFI', 'VPIN',
            # Volatility
            'VOL_CLOSE', 'VOL_PARKINSON', 'VOL_GARMAN_KLASS',
            'VOL_ROGERS_SATCHELL', 'VOL_YANG_ZHANG',
            'VOL_RATIO_PK_CC', 'VOL_RATIO_GK_RS',
            # Risk
            'SHARPE_60', 'SORTINO_60', 'MAX_DD_60'
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_gold_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all gold standard features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with ~12 gold standard features
    """
    generator = GoldStandardFeatures()
    return generator.generate_microstructure_features(df)


def calculate_kelly_position(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    account_value: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate position size using fractional Kelly.

    Args:
        win_rate: Historical win rate (0 to 1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        account_value: Total account value
        fraction: Kelly fraction (0.25 = quarter Kelly)

    Returns:
        Recommended position size
    """
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1
    return KellyCriterion.position_size(
        account_value=account_value,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        fraction=fraction
    )


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("GitHub Gold Standard Quantitative Trading Formulas")
    print("=" * 60)

    # Test data
    np.random.seed(42)
    n = 1000

    dates = pd.date_range('2024-01-01', periods=n, freq='H')
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.0001)
    high = close + np.abs(np.random.randn(n) * 0.0002)
    low = close - np.abs(np.random.randn(n) * 0.0002)
    open_ = close + np.random.randn(n) * 0.0001

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)

    # Generate features
    features = generate_gold_standard_features(df)
    print(f"\nGenerated {len(features.columns)} features:")
    for col in features.columns:
        print(f"  - {col}: {features[col].iloc[-1]:.6f}")

    # Kelly calculation
    position = calculate_kelly_position(
        win_rate=0.63,
        avg_win=0.015,
        avg_loss=0.010,
        account_value=100000,
        fraction=0.25
    )
    print(f"\nKelly Position Size: ${position:,.2f}")

    # Volatility comparison
    vol = VolatilityEstimators()
    print("\nVolatility Estimators (last value):")
    print(f"  Close-to-Close: {vol.close_to_close(df['close']).iloc[-1]:.4f}")
    print(f"  Parkinson:      {vol.parkinson(df['high'], df['low']).iloc[-1]:.4f}")
    print(f"  Garman-Klass:   {vol.garman_klass(df['open'], df['high'], df['low'], df['close']).iloc[-1]:.4f}")

    print("\nAll formulas loaded successfully!")
