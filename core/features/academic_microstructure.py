"""
Academic Microstructure Features - Order Flow & Market Impact
=============================================================

Peer-reviewed implementations of market microstructure measures from
top finance journals. These features capture informed trading, toxicity,
and market impact.

CITATIONS:
----------

1. ORDER FLOW IMBALANCE (OFI)
   Cont, R., Kukanov, A., & Stoikov, S. (2014).
   "The Price Impact of Order Book Events"
   Journal of Financial Econometrics, 12(1), 47-88.
   DOI: 10.1093/jjfinec/nbt003
   arXiv: https://arxiv.org/abs/1011.6402

2. VPIN (Volume-Synchronized Probability of Informed Trading)
   Easley, D., López de Prado, M., & O'Hara, M. (2012).
   "Flow Toxicity and Liquidity in a High-Frequency World"
   Review of Financial Studies, 25(5), 1457-1493.
   DOI: 10.1093/rfs/hhs053

3. KYLE'S LAMBDA (Market Impact Coefficient)
   Kyle, A. S. (1985).
   "Continuous Auctions and Insider Trading"
   Econometrica, 53(6), 1315-1335.
   DOI: 10.2307/1913210

4. AMIHUD ILLIQUIDITY
   Amihud, Y. (2002).
   "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects"
   Journal of Financial Markets, 5(1), 31-56.
   DOI: 10.1016/S1386-4181(01)00024-6

5. ROLL SPREAD ESTIMATOR
   Roll, R. (1984).
   "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market"
   Journal of Finance, 39(4), 1127-1139.
   DOI: 10.1111/j.1540-6261.1984.tb03897.x

APPLICABILITY:
--------------
- OFI: ADAPTABLE (equity-based, methodology applies to forex with order book data)
- VPIN: ADAPTABLE (widely used in forex for toxicity measurement)
- Kyle Lambda: FOREX-NATIVE (foundational model for all markets)
- Amihud: FOREX-NATIVE (liquidity measurement)
- Roll: FOREX-NATIVE (spread estimation)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from scipy.stats import norm
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure calculations."""
    vpin_bucket_size: int = 50000  # Volume per bucket
    vpin_n_buckets: int = 50  # Rolling window
    ofi_window: int = 10  # OFI aggregation window
    kyle_window: int = 100  # Kyle lambda estimation window
    amihud_window: int = 20  # Amihud illiquidity window


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI) per Cont, Kukanov & Stoikov (2014).

    The key finding from Cont et al. (2014):
    "Over short time intervals, price changes are mainly driven by
    order flow imbalance, defined as the imbalance between supply
    and demand at the best bid and ask prices."

    Formula:
        OFI_t = (V^B_t - V^B_{t-1}) * I{P^B_t >= P^B_{t-1}}
              - (V^A_t - V^A_{t-1}) * I{P^A_t <= P^A_{t-1}}

    Where:
        V^B, V^A = bid/ask volume at best prices
        P^B, P^A = best bid/ask prices
        I{} = indicator function

    Price Impact Model:
        ΔP_t = λ * OFI_t + ε_t

    Where λ = 1/market_depth (inverse of liquidity)

    Citation:
        Cont, R., Kukanov, A., & Stoikov, S. (2014).
        Journal of Financial Econometrics, 12(1), 47-88.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()

    def compute_ofi(
        self,
        bid_price: pd.Series,
        ask_price: pd.Series,
        bid_volume: pd.Series,
        ask_volume: pd.Series
    ) -> pd.Series:
        """
        Compute Order Flow Imbalance per Cont et al. (2014).

        Args:
            bid_price: Best bid price series
            ask_price: Best ask price series
            bid_volume: Volume at best bid
            ask_volume: Volume at best ask

        Returns:
            OFI series
        """
        # Changes in volume
        delta_bid_vol = bid_volume.diff()
        delta_ask_vol = ask_volume.diff()

        # Price improvement indicators
        bid_improve = (bid_price.diff() >= 0).astype(float)
        ask_improve = (ask_price.diff() <= 0).astype(float)

        # OFI = bid contribution - ask contribution
        # Eq (8) from Cont et al. (2014)
        ofi = (delta_bid_vol * bid_improve) - (delta_ask_vol * ask_improve)

        return ofi.fillna(0)

    def compute_ofi_from_trades(
        self,
        price: pd.Series,
        volume: pd.Series,
        bid: pd.Series,
        ask: pd.Series
    ) -> pd.Series:
        """
        Compute OFI from trade data using Lee-Ready classification.

        Lee, C.M.C., & Ready, M.J. (1991).
        "Inferring Trade Direction from Intraday Data"
        Journal of Finance, 46(2), 733-746.

        Args:
            price: Trade prices
            volume: Trade volumes
            bid: Best bid at trade time
            ask: Best ask at trade time

        Returns:
            OFI series (positive = buying pressure)
        """
        mid = (bid + ask) / 2

        # Lee-Ready classification
        # Trade at ask = buy, trade at bid = sell
        # Trade at mid = use tick rule
        buy_indicator = np.where(
            price > mid, 1,
            np.where(price < mid, -1, np.sign(price.diff()))
        )

        # Signed volume
        signed_volume = volume * buy_indicator

        # Rolling OFI
        ofi = pd.Series(signed_volume, index=price.index)

        return ofi

    def estimate_lambda(
        self,
        price_changes: pd.Series,
        ofi: pd.Series,
        window: int = None
    ) -> pd.Series:
        """
        Estimate Kyle's lambda using OFI regression.

        ΔP_t = λ * OFI_t + ε_t

        Cont et al. (2014) show λ is inversely proportional to market depth:
        "slope inversely proportional to the market depth"

        Args:
            price_changes: Price change series
            ofi: Order flow imbalance series
            window: Rolling window for estimation

        Returns:
            Rolling lambda estimates (price impact per unit OFI)
        """
        window = window or self.config.kyle_window

        lambdas = []
        for i in range(window, len(ofi)):
            X = ofi.iloc[i-window:i].values.reshape(-1, 1)
            y = price_changes.iloc[i-window:i].values

            # OLS: λ = (X'X)^(-1) X'y
            XtX = np.dot(X.T, X)
            if XtX > 1e-10:
                lambda_est = np.dot(X.T, y) / XtX
                lambdas.append(lambda_est[0])
            else:
                lambdas.append(np.nan)

        # Pad beginning with NaN
        result = pd.Series(
            [np.nan] * window + lambdas,
            index=ofi.index
        )

        return result


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    VPIN measures order flow toxicity - the probability that order flow
    adversely selects market makers. High VPIN indicates increased
    informed trading activity.

    Formula:
        VPIN = Σ|V^B_τ - V^S_τ| / (n * V_bucket)

    Where:
        τ = volume bucket index
        V^B_τ, V^S_τ = buy/sell volume in bucket τ
        n = number of buckets in rolling window
        V_bucket = fixed volume per bucket

    Key Findings:
        "VPIN reached historically high levels in the hours and days
        prior to the Flash Crash of May 6, 2010."
        - Easley, López de Prado, O'Hara (2012)

    Bulk Volume Classification (BVC):
        V^B = V * Φ((P_close - P_open) / σ)
        V^S = V - V^B

    Where Φ(·) = standard normal CDF

    Citation:
        Easley, D., López de Prado, M., & O'Hara, M. (2012).
        Review of Financial Studies, 25(5), 1457-1493.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()

    def bulk_volume_classification(
        self,
        open_price: pd.Series,
        close_price: pd.Series,
        volume: pd.Series,
        sigma: pd.Series = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Bulk Volume Classification (BVC) per Easley et al. (2012a).

        "They advocate a 'bulk volume' classification strategy,
        referred to as BV-VPIN."

        Args:
            open_price: Opening prices
            close_price: Closing prices
            volume: Total volume
            sigma: Volatility (if None, computed from returns)

        Returns:
            Tuple of (buy_volume, sell_volume)
        """
        # Compute returns
        if sigma is None:
            returns = np.log(close_price / open_price)
            sigma = returns.rolling(20).std()
            sigma = sigma.fillna(returns.std())

        # Standardized price change
        z = (close_price - open_price) / (sigma * open_price + 1e-10)

        # Buy probability from CDF
        buy_prob = pd.Series(norm.cdf(z), index=close_price.index)

        # Classify volume
        buy_volume = volume * buy_prob
        sell_volume = volume * (1 - buy_prob)

        return buy_volume, sell_volume

    def compute_vpin(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series,
        bucket_size: int = None,
        n_buckets: int = None
    ) -> pd.Series:
        """
        Compute VPIN metric.

        "VPIN captures the market dynamics in event time, i.e., equal
        increments of trading volume rather than calendar time."
        - Easley et al. (2012)

        Args:
            buy_volume: Classified buy volume
            sell_volume: Classified sell volume
            bucket_size: Volume per bucket
            n_buckets: Number of buckets in rolling window

        Returns:
            VPIN series (0-1, higher = more toxic)
        """
        bucket_size = bucket_size or self.config.vpin_bucket_size
        n_buckets = n_buckets or self.config.vpin_n_buckets

        total_volume = buy_volume + sell_volume

        # Create volume buckets
        cum_volume = total_volume.cumsum()
        bucket_idx = (cum_volume // bucket_size).astype(int)

        # Aggregate by bucket
        df = pd.DataFrame({
            'buy': buy_volume,
            'sell': sell_volume,
            'bucket': bucket_idx
        })

        bucket_agg = df.groupby('bucket').agg({
            'buy': 'sum',
            'sell': 'sum'
        })

        # Absolute imbalance per bucket
        imbalance = np.abs(bucket_agg['buy'] - bucket_agg['sell'])

        # Rolling VPIN
        vpin_buckets = imbalance.rolling(n_buckets).sum() / (n_buckets * bucket_size)

        # Map back to original index
        vpin = bucket_idx.map(vpin_buckets).fillna(method='ffill')

        return vpin

    def compute_vpin_fast(
        self,
        close: pd.Series,
        volume: pd.Series,
        window: int = 50
    ) -> pd.Series:
        """
        Fast VPIN approximation for tick data without OHLC.

        Uses tick rule for trade classification.

        Args:
            close: Price series
            volume: Volume series
            window: Rolling window

        Returns:
            VPIN approximation
        """
        # Tick rule classification
        price_change = close.diff()
        buy_indicator = np.sign(price_change).fillna(0)
        buy_indicator = np.where(buy_indicator == 0, 1, buy_indicator)  # Ties go to buyer

        buy_volume = volume * (buy_indicator == 1)
        sell_volume = volume * (buy_indicator == -1)

        # Rolling imbalance
        total_vol = volume.rolling(window).sum()
        net_vol = (buy_volume - sell_volume).rolling(window).sum()

        vpin = np.abs(net_vol) / (total_vol + 1e-10)

        return vpin


class KyleLambda:
    """
    Kyle's Lambda - Market Impact Coefficient.

    The foundational model of market microstructure. Kyle (1985) shows
    that informed traders optimally trade to minimize price impact,
    leading to a linear price impact function.

    Model:
        P_n = P_{n-1} + λ * (x_n + u_n) + ε_n

    Where:
        λ = σ_v / (2σ_u)  # Price impact coefficient
        σ_v = volatility of fundamental value
        σ_u = noise trader volatility
        x_n = informed trader order
        u_n = noise trader order

    Interpretation:
        "Kyle's lambda measures the dollar change in price due to
        a dollar change in order flow."

        Higher λ = less liquid market, higher price impact

    Citation:
        Kyle, A. S. (1985).
        Econometrica, 53(6), 1315-1335.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()

    def estimate_lambda_roll(
        self,
        price: pd.Series,
        order_flow: pd.Series,
        window: int = None
    ) -> pd.Series:
        """
        Rolling estimation of Kyle's lambda.

        Args:
            price: Price series
            order_flow: Signed order flow (positive = buying)
            window: Rolling window

        Returns:
            Rolling lambda estimates
        """
        window = window or self.config.kyle_window

        price_change = price.diff()

        # Rolling regression: ΔP = λ * OrderFlow
        lambdas = []
        for i in range(window, len(price)):
            X = order_flow.iloc[i-window:i].values
            y = price_change.iloc[i-window:i].values

            # Handle NaN
            mask = ~(np.isnan(X) | np.isnan(y))
            if mask.sum() < window // 2:
                lambdas.append(np.nan)
                continue

            X, y = X[mask], y[mask]

            # OLS
            XtX = np.dot(X, X)
            if XtX > 1e-10:
                lambda_est = np.dot(X, y) / XtX
                lambdas.append(lambda_est)
            else:
                lambdas.append(np.nan)

        return pd.Series(
            [np.nan] * window + lambdas,
            index=price.index
        )

    def compute_market_depth(self, lambda_: pd.Series) -> pd.Series:
        """
        Market depth = 1 / λ (inverse of price impact).

        "Market depth reflects the informativeness of the order flow
        and the size of trades by noise traders."
        - Kyle (1985)

        Args:
            lambda_: Kyle's lambda series

        Returns:
            Market depth series
        """
        return 1.0 / (lambda_.abs() + 1e-10)


class AmihudIlliquidity:
    """
    Amihud Illiquidity Measure.

    A simple and widely-used measure of price impact per unit volume.

    Formula:
        ILLIQ_t = (1/D) * Σ |r_d| / DVOL_d

    Where:
        r_d = daily return
        DVOL_d = daily dollar volume
        D = number of days

    Interpretation:
        Higher ILLIQ = less liquid, higher price impact

    Citation:
        Amihud, Y. (2002).
        Journal of Financial Markets, 5(1), 31-56.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()

    def compute_illiquidity(
        self,
        returns: pd.Series,
        dollar_volume: pd.Series,
        window: int = None
    ) -> pd.Series:
        """
        Compute Amihud illiquidity ratio.

        Args:
            returns: Return series
            dollar_volume: Dollar volume (price * volume)
            window: Rolling window

        Returns:
            Illiquidity series (higher = less liquid)
        """
        window = window or self.config.amihud_window

        # |Return| / Dollar Volume
        illiq_daily = returns.abs() / (dollar_volume + 1e-10)

        # Rolling average
        illiq = illiq_daily.rolling(window).mean()

        # Scale by 1e6 for interpretability
        return illiq * 1e6


class RollSpread:
    """
    Roll Spread Estimator - Implicit Bid-Ask Spread.

    Estimates effective spread from price autocovariance, based on
    the idea that bid-ask bounce creates negative serial correlation.

    Formula:
        Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))  if Cov < 0
               = 0                                otherwise

    Intuition:
        "The bid-ask spread induces negative serial correlation in
        transaction price changes."
        - Roll (1984)

    Citation:
        Roll, R. (1984).
        Journal of Finance, 39(4), 1127-1139.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()

    def compute_spread(
        self,
        price: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Compute Roll spread estimator.

        Args:
            price: Price series
            window: Rolling window for covariance

        Returns:
            Estimated spread series
        """
        price_change = price.diff()

        # Rolling covariance of consecutive price changes
        cov = price_change.rolling(window).cov(price_change.shift(1))

        # Spread = 2 * sqrt(-cov) if cov < 0
        spread = np.where(cov < 0, 2 * np.sqrt(-cov), 0)

        return pd.Series(spread, index=price.index)


class AcademicMicrostructureFeatures:
    """
    Generate all academic microstructure features.

    Combines OFI, VPIN, Kyle Lambda, Amihud, and Roll spread
    into a unified feature generator.
    """

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()
        self.ofi = OrderFlowImbalance(config)
        self.vpin = VPIN(config)
        self.kyle = KyleLambda(config)
        self.amihud = AmihudIlliquidity(config)
        self.roll = RollSpread(config)

    def generate_all(
        self,
        df: pd.DataFrame,
        has_orderbook: bool = False
    ) -> pd.DataFrame:
        """
        Generate all microstructure features.

        Args:
            df: DataFrame with OHLCV data
            has_orderbook: Whether L2 orderbook data is available

        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=df.index)

        # Required columns
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))

        # Optional columns
        open_price = df.get('open', close.shift(1).fillna(close))
        high = df.get('high', close)
        low = df.get('low', close)
        bid = df.get('bid', close - 0.00005)
        ask = df.get('ask', close + 0.00005)

        # =========================================================
        # VPIN Features - Easley, López de Prado, O'Hara (2012)
        # =========================================================
        buy_vol, sell_vol = self.vpin.bulk_volume_classification(
            open_price, close, volume
        )

        features['VPIN_50'] = self.vpin.compute_vpin_fast(close, volume, 50)
        features['VPIN_100'] = self.vpin.compute_vpin_fast(close, volume, 100)

        # Volume imbalance (simplified VPIN)
        features['VOL_IMBALANCE_20'] = (
            (buy_vol - sell_vol).rolling(20).sum() /
            (volume.rolling(20).sum() + 1e-10)
        )

        # =========================================================
        # OFI Features - Cont, Kukanov, Stoikov (2014)
        # =========================================================
        ofi = self.ofi.compute_ofi_from_trades(close, volume, bid, ask)

        for w in [5, 10, 20, 50]:
            features[f'OFI_{w}'] = ofi.rolling(w).sum()
            features[f'OFI_STD_{w}'] = ofi.rolling(w).std()

        # Normalized OFI
        features['OFI_NORM_20'] = (
            ofi.rolling(20).sum() / (ofi.rolling(20).std() + 1e-10)
        )

        # =========================================================
        # Kyle Lambda - Kyle (1985)
        # =========================================================
        order_flow = ofi.cumsum()  # Cumulative order flow

        for w in [50, 100]:
            lambda_est = self.kyle.estimate_lambda_roll(close, order_flow, w)
            features[f'KYLE_LAMBDA_{w}'] = lambda_est
            features[f'MARKET_DEPTH_{w}'] = self.kyle.compute_market_depth(lambda_est)

        # =========================================================
        # Amihud Illiquidity - Amihud (2002)
        # =========================================================
        returns = close.pct_change()
        dollar_volume = close * volume

        for w in [10, 20, 50]:
            features[f'AMIHUD_{w}'] = self.amihud.compute_illiquidity(
                returns, dollar_volume, w
            )

        # =========================================================
        # Roll Spread - Roll (1984)
        # =========================================================
        for w in [10, 20, 50]:
            features[f'ROLL_SPREAD_{w}'] = self.roll.compute_spread(close, w)

        # =========================================================
        # Derived Features
        # =========================================================

        # Spread relative to price
        spread = ask - bid
        features['SPREAD_PCT'] = spread / close
        features['SPREAD_VOLATILITY'] = spread.rolling(20).std() / spread.rolling(20).mean()

        # Toxicity indicator (VPIN > 0.5)
        features['HIGH_TOXICITY'] = (features['VPIN_50'] > 0.5).astype(float)

        # Price impact asymmetry
        features['IMPACT_ASYM'] = (
            features['OFI_20'].clip(lower=0).rolling(20).mean() -
            features['OFI_20'].clip(upper=0).abs().rolling(20).mean()
        )

        return features.fillna(0)


def generate_microstructure_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate microstructure features.

    Citations:
    - Cont et al. (2014) - Order Flow Imbalance
    - Easley et al. (2012) - VPIN
    - Kyle (1985) - Market Impact
    - Amihud (2002) - Illiquidity
    - Roll (1984) - Spread Estimation

    Args:
        df: DataFrame with price/volume data
        **kwargs: Additional arguments

    Returns:
        DataFrame with microstructure features
    """
    generator = AcademicMicrostructureFeatures()
    return generator.generate_all(df, **kwargs)


# Module-level exports
__all__ = [
    'MicrostructureConfig',
    'OrderFlowImbalance',
    'VPIN',
    'KyleLambda',
    'AmihudIlliquidity',
    'RollSpread',
    'AcademicMicrostructureFeatures',
    'generate_microstructure_features',
]
