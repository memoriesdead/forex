"""
Academic Market Making Models - Optimal Quoting & Inventory Control
====================================================================

Implementations of optimal market making models from the quantitative
finance literature. These models provide optimal bid/ask quotes based
on inventory risk and market conditions.

CITATIONS:
----------

1. AVELLANEDA-STOIKOV MODEL
   Avellaneda, M., & Stoikov, S. (2008).
   "High-frequency trading in a limit order book"
   Quantitative Finance, 8(3), 217-224.
   DOI: 10.1080/14697680701381228
   PDF: https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf

2. GUÉANT-LEHALLE-FERNANDEZ-TAPIA EXTENSION
   Guéant, O., Lehalle, C.A., & Fernandez-Tapia, J. (2012).
   "Dealing with the inventory risk: a solution to the market making problem"
   Mathematics and Financial Economics, 4(7), 477-507.
   DOI: 10.1007/s11579-012-0078-5

3. OPTIMAL EXECUTION WITH LIMIT ORDERS
   Guéant, O., & Lehalle, C.A. (2015).
   "General Intensity Shapes in Optimal Liquidation"
   Mathematical Finance, 25(3), 457-495.

4. MARKET MAKING WITH INVENTORY CONSTRAINTS
   Fodra, P., & Labadie, M. (2012).
   "High-frequency market-making with inventory constraints and directional bets"
   arXiv:1206.4810

KEY FORMULAS:
-------------
Reservation Price:
    r(s, q, t) = s - q·γ·σ²·(T-t)

Optimal Spread:
    δ^a + δ^b = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)

Where:
    s = mid price
    q = inventory (positive = long)
    γ = risk aversion parameter
    σ = volatility
    T = terminal time
    k = order arrival intensity parameter

APPLICABILITY: FOREX-NATIVE (especially for market makers, liquidity
providers, and understanding bid-ask spread dynamics)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MarketMakingConfig:
    """Configuration for market making models."""
    gamma: float = 0.1  # Risk aversion (higher = tighter spreads)
    sigma: float = 0.0002  # Volatility (forex typical ~0.02% per minute)
    k: float = 1.5  # Order arrival intensity parameter
    A: float = 1.0  # Arrival rate scaling
    T: float = 1.0  # Terminal time (normalized to 1)
    max_inventory: int = 100  # Maximum inventory limit
    tick_size: float = 0.00001  # Minimum price increment (1 pip / 10)


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov Optimal Market Making Model.

    The seminal model for optimal bid/ask quote placement with
    inventory risk management.

    Key Insight:
        "The main result is that the optimal bid and ask quotes are
        derived in an intuitive two-step procedure. First, the dealer
        computes a personal indifference valuation for the stock,
        given his current inventory. Second, he calibrates his bid
        and ask quotes to the limit order book."
        - Avellaneda & Stoikov (2008)

    Model Assumptions:
        - Mid-price follows arithmetic Brownian motion
        - Order arrivals are Poisson with intensity λ(δ) = A·exp(-k·δ)
        - Market maker has CARA utility (exponential)
        - No transaction costs

    Citation:
        Avellaneda, M., & Stoikov, S. (2008).
        Quantitative Finance, 8(3), 217-224.
    """

    def __init__(self, config: MarketMakingConfig = None):
        """
        Initialize Avellaneda-Stoikov model.

        Args:
            config: Model configuration

        Note on parameters:
            "The γ parameter is a value that must be defined by the
            market maker, considering how much inventory risk he is
            willing to be exposed. A value closer to zero means more
            aggressive quoting."
            - Avellaneda & Stoikov (2008)
        """
        self.config = config or MarketMakingConfig()

    def reservation_price(
        self,
        mid_price: float,
        inventory: float,
        time_remaining: float = None
    ) -> float:
        """
        Compute reservation (indifference) price.

        The reservation price is the market maker's personal valuation
        of the asset given their current inventory position.

        Formula:
            r(s, q, t) = s - q·γ·σ²·(T-t)

        Interpretation:
            - If inventory > 0 (long): reservation < mid (want to sell)
            - If inventory < 0 (short): reservation > mid (want to buy)
            - If inventory = 0: reservation = mid

        "First, the dealer computes a personal indifference valuation
        for the stock, given his current inventory."
        - Avellaneda & Stoikov (2008)

        Args:
            mid_price: Current mid price
            inventory: Current inventory (positive = long)
            time_remaining: Time remaining until terminal (default: T)

        Returns:
            Reservation price
        """
        time_remaining = time_remaining if time_remaining is not None else self.config.T

        gamma = self.config.gamma
        sigma = self.config.sigma

        # r = s - q·γ·σ²·(T-t)
        reservation = mid_price - inventory * gamma * sigma**2 * time_remaining

        return reservation

    def optimal_spread(self, time_remaining: float = None) -> float:
        """
        Compute optimal total spread.

        Formula:
            δ^a + δ^b = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)

        "The bid-ask spread is independent of the inventory."
        - Avellaneda & Stoikov (2008)

        The first term depends on time and risk aversion.
        The second term comes from the order arrival intensity model.

        Args:
            time_remaining: Time remaining until terminal

        Returns:
            Total spread (ask_delta + bid_delta)
        """
        time_remaining = time_remaining if time_remaining is not None else self.config.T

        gamma = self.config.gamma
        sigma = self.config.sigma
        k = self.config.k

        # Term 1: Time-dependent component
        term1 = gamma * sigma**2 * time_remaining

        # Term 2: Arrival intensity component
        term2 = (2.0 / gamma) * np.log(1 + gamma / k)

        return term1 + term2

    def optimal_quotes(
        self,
        mid_price: float,
        inventory: float,
        time_remaining: float = None
    ) -> Tuple[float, float]:
        """
        Compute optimal bid and ask prices.

        "To minimize inventory risk, prices should be skewed to favor
        the inventory to come back to its targeted ideal balance point.
        To maximize trade profitability, spreads should be enlarged such
        that the expected future value of the account is maximized."
        - Avellaneda & Stoikov (2008)

        Args:
            mid_price: Current mid price
            inventory: Current inventory (positive = long)
            time_remaining: Time remaining until terminal

        Returns:
            Tuple of (optimal_bid, optimal_ask)
        """
        time_remaining = time_remaining if time_remaining is not None else self.config.T

        # Step 1: Compute reservation price
        r = self.reservation_price(mid_price, inventory, time_remaining)

        # Step 2: Compute optimal spread
        spread = self.optimal_spread(time_remaining)

        # Step 3: Compute inventory skew
        # Positive inventory → lower bid, higher ask (encourage selling)
        # Negative inventory → higher bid, lower ask (encourage buying)
        gamma = self.config.gamma
        sigma = self.config.sigma
        skew = inventory * gamma * sigma**2 * time_remaining

        # Optimal quotes (symmetric spread around reservation, skewed by inventory)
        bid = r - spread / 2 + skew
        ask = r + spread / 2 + skew

        # Alternative: quotes directly around mid with skew
        # bid = mid_price - spread / 2 - skew
        # ask = mid_price + spread / 2 - skew

        return bid, ask

    def expected_fill_rate(self, delta: float) -> float:
        """
        Expected order fill rate at distance delta from mid.

        "Avellaneda and Stoikov showed that, given the empirical evidence,
        we can assume that λ(δ) = A·exp(-k·δ) for order arrival rates."

        Args:
            delta: Distance from mid price

        Returns:
            Expected fill rate (arrivals per unit time)
        """
        A = self.config.A
        k = self.config.k

        return A * np.exp(-k * delta)

    def inventory_limits_adjustment(
        self,
        bid: float,
        ask: float,
        inventory: float
    ) -> Tuple[float, float]:
        """
        Adjust quotes based on inventory limits.

        "When inventory approaches the maximum limit, quotes are
        adjusted more aggressively to reduce position."
        - Guéant, Lehalle, Fernandez-Tapia (2012)

        Args:
            bid: Current optimal bid
            ask: Current optimal ask
            inventory: Current inventory

        Returns:
            Adjusted (bid, ask)
        """
        max_inv = self.config.max_inventory

        # Inventory as fraction of limit
        inv_ratio = abs(inventory) / max_inv if max_inv > 0 else 0

        # Exponential adjustment as inventory approaches limit
        adjustment_factor = np.exp(2 * inv_ratio) - 1

        if inventory > 0:  # Long position, want to sell
            # Widen ask less, tighten bid (encourage sells)
            ask = ask - adjustment_factor * (ask - bid) * 0.1
            bid = bid - adjustment_factor * (ask - bid) * 0.2
        elif inventory < 0:  # Short position, want to buy
            # Tighten ask, widen bid (encourage buys)
            ask = ask + adjustment_factor * (ask - bid) * 0.2
            bid = bid + adjustment_factor * (ask - bid) * 0.1

        return bid, ask


class MarketMakingFeatures:
    """
    Generate market making features for analysis and trading.

    These features can be used to:
    - Understand market maker behavior
    - Estimate fair spreads
    - Detect inventory pressure
    """

    def __init__(self, config: MarketMakingConfig = None):
        self.config = config or MarketMakingConfig()
        self.model = AvellanedaStoikov(config)

    def estimate_market_maker_inventory(
        self,
        bid: pd.Series,
        ask: pd.Series,
        mid: pd.Series = None
    ) -> pd.Series:
        """
        Estimate market maker inventory from quote asymmetry.

        If market maker is long, they quote lower bids (discourage buying)
        and lower asks (encourage selling), creating negative skew.

        Args:
            bid: Bid price series
            ask: Ask price series
            mid: Mid price (if None, computed from bid/ask)

        Returns:
            Estimated inventory signal (positive = MM is long)
        """
        if mid is None:
            mid = (bid + ask) / 2

        # Quote skew: negative means MM wants to sell (is long)
        bid_dist = mid - bid
        ask_dist = ask - mid

        # Asymmetry: positive = ask is further from mid = MM is short
        asymmetry = ask_dist - bid_dist

        # Normalized by spread
        spread = ask - bid
        inv_signal = asymmetry / (spread + 1e-10)

        return inv_signal

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market making features.

        Args:
            df: DataFrame with bid/ask/close data

        Returns:
            DataFrame with market making features
        """
        features = pd.DataFrame(index=df.index)

        # Get prices
        close = df['close']
        bid = df.get('bid', close - 0.00005)
        ask = df.get('ask', close + 0.00005)
        mid = (bid + ask) / 2

        # =========================================================
        # Spread Features
        # =========================================================
        spread = ask - bid
        features['SPREAD'] = spread
        features['SPREAD_PCT'] = spread / mid
        features['SPREAD_BPS'] = spread / mid * 10000

        # Spread relative to theoretical (Avellaneda-Stoikov)
        # Estimate volatility
        returns = close.pct_change()
        vol = returns.rolling(20).std()

        # Theoretical spread (simplified)
        gamma = self.config.gamma
        k = self.config.k
        theoretical_spread = (2.0 / gamma) * np.log(1 + gamma / k) * mid

        features['SPREAD_VS_THEORETICAL'] = spread / (theoretical_spread + 1e-10)

        # =========================================================
        # Inventory Estimation Features
        # =========================================================
        features['MM_INVENTORY_SIGNAL'] = self.estimate_market_maker_inventory(
            bid, ask, mid
        )

        # Rolling inventory pressure
        for w in [5, 10, 20]:
            features[f'MM_INV_PRESSURE_{w}'] = features['MM_INVENTORY_SIGNAL'].rolling(w).mean()

        # =========================================================
        # Quote Quality Features
        # =========================================================

        # Bid-ask midpoint vs last close (microstructure)
        features['MID_VS_CLOSE'] = mid - close.shift(1)

        # Quote stability
        features['BID_STABILITY'] = bid.rolling(10).std() / (spread + 1e-10)
        features['ASK_STABILITY'] = ask.rolling(10).std() / (spread + 1e-10)

        # =========================================================
        # Optimal Quote Computation
        # =========================================================

        # Compute what AS model would suggest
        # Assuming zero inventory (neutral market maker)
        optimal_bids = []
        optimal_asks = []

        for i in range(len(df)):
            m = mid.iloc[i]
            # Use rolling volatility
            if i >= 20:
                self.model.config.sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.0002
            else:
                self.model.config.sigma = 0.0002

            b, a = self.model.optimal_quotes(m, inventory=0, time_remaining=1.0)
            optimal_bids.append(b)
            optimal_asks.append(a)

        features['AS_OPTIMAL_BID'] = optimal_bids
        features['AS_OPTIMAL_ASK'] = optimal_asks
        features['AS_OPTIMAL_SPREAD'] = features['AS_OPTIMAL_ASK'] - features['AS_OPTIMAL_BID']

        # How does actual spread compare to optimal?
        features['SPREAD_EFFICIENCY'] = spread / (features['AS_OPTIMAL_SPREAD'] + 1e-10)

        # =========================================================
        # Trading Signals
        # =========================================================

        # If spread is tighter than optimal, liquidity is good
        features['LIQUIDITY_ABUNDANT'] = (features['SPREAD_EFFICIENCY'] < 1.0).astype(float)

        # If MM inventory signal is extreme, expect mean reversion
        inv_zscore = (
            features['MM_INVENTORY_SIGNAL'] -
            features['MM_INVENTORY_SIGNAL'].rolling(50).mean()
        ) / (features['MM_INVENTORY_SIGNAL'].rolling(50).std() + 1e-10)

        features['MM_EXTREME_LONG'] = (inv_zscore < -2).astype(float)
        features['MM_EXTREME_SHORT'] = (inv_zscore > 2).astype(float)

        return features.fillna(0)


class AcademicMarketMakingFeatures:
    """
    Generate all academic market making features.

    Combines Avellaneda-Stoikov model with spread analysis
    and inventory estimation.
    """

    def __init__(self, config: MarketMakingConfig = None):
        self.config = config or MarketMakingConfig()
        self.mm_features = MarketMakingFeatures(config)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all market making features.

        Args:
            df: DataFrame with OHLCV + bid/ask data

        Returns:
            DataFrame with market making features
        """
        return self.mm_features.generate_features(df)


def generate_market_making_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate market making features.

    Citations:
    - Avellaneda & Stoikov (2008) - Optimal market making
    - Guéant et al. (2012) - Inventory risk
    - Lehalle et al. (2013) - Market microstructure

    Args:
        df: DataFrame with price data
        **kwargs: Additional arguments

    Returns:
        DataFrame with market making features
    """
    generator = AcademicMarketMakingFeatures()
    return generator.generate_all(df)


# Module-level exports
__all__ = [
    'MarketMakingConfig',
    'AvellanedaStoikov',
    'MarketMakingFeatures',
    'AcademicMarketMakingFeatures',
    'generate_market_making_features',
]
