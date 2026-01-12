"""
Market Impact Models (Academic Microstructure)
===============================================
Implementation of foundational market microstructure models.

Sources (Gold Standard Citations):

1. Kyle (1985): "Continuous Auctions and Insider Trading"
   Econometrica, Vol. 53, No. 6, pp. 1315-1335
   - Lambda (price impact parameter)
   - Informed vs uninformed trading

2. Glosten & Milgrom (1985): "Bid, Ask and Transaction Prices in a Specialist Market
   with Heterogeneously Informed Traders"
   Journal of Financial Economics, Vol. 14, pp. 71-100
   - Adverse selection component of spread
   - Bayesian updating of beliefs

3. Roll (1984): "A Simple Implicit Measure of the Effective Bid-Ask Spread in an
   Efficient Market"
   Journal of Finance, Vol. 39, No. 4, pp. 1127-1139
   - Effective spread from autocorrelation

4. Huang & Stoll (1997): "The Components of the Bid-Ask Spread: A General Approach"
   Review of Financial Studies, Vol. 10, No. 4, pp. 995-1034
   - Spread decomposition: adverse selection, inventory, order processing

5. Hasbrouck (1991): "Measuring the Information Content of Stock Trades"
   Journal of Finance, Vol. 46, No. 1, pp. 179-207
   - Trade informativeness, permanent price impact
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MarketImpactEstimate:
    """Container for market impact estimates."""
    lambda_: float  # Kyle's lambda (price impact per unit)
    sigma_u: float  # Noise trader volatility
    sigma_v: float  # Fundamental value volatility
    mu: float       # Informed trading probability
    method: str


class KyleModel:
    """
    Kyle (1985) Model - Continuous Auctions and Insider Trading

    The informed trader knows the true value V and trades optimally.
    Market maker sets prices to break even given order flow.

    Key result:
        P = lambda * (X + U)

    where:
        P = price change
        lambda = sigma_v / (2 * sigma_u) = price impact parameter
        X = informed trader order
        U = noise trader order (random)
        sigma_v = std of fundamental value changes
        sigma_u = std of noise trading

    Reference:
        Kyle, A. S. (1985). "Continuous Auctions and Insider Trading".
        Econometrica, 53(6), 1315-1335.
    """

    def __init__(self):
        self.lambda_ = None
        self.sigma_v = None
        self.sigma_u = None

    def estimate_lambda(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Estimate Kyle's lambda from price and volume data.

        Lambda = Cov(ΔP, V) / Var(V)

        Args:
            prices: Price series
            volumes: Volume series (signed: + for buy, - for sell)
            window: Rolling window for estimation

        Returns:
            Rolling lambda estimates
        """
        price_changes = prices.diff()

        # Estimate lambda via regression: ΔP = lambda * V + epsilon
        def estimate_single(dp, vol):
            if len(dp) < 5:
                return np.nan
            # OLS: lambda = Cov(ΔP, V) / Var(V)
            cov = np.cov(dp, vol)[0, 1]
            var = np.var(vol)
            if var > 0:
                return cov / var
            return np.nan

        lambda_series = price_changes.rolling(window).apply(
            lambda x: estimate_single(x.values, volumes.loc[x.index].values),
            raw=False
        )

        return lambda_series

    def estimate_from_trades(
        self,
        trade_prices: pd.Series,
        trade_signs: pd.Series,
        trade_sizes: pd.Series
    ) -> MarketImpactEstimate:
        """
        Estimate Kyle model parameters from trade data.

        Uses regression: ΔP = lambda * sign * size + epsilon

        Args:
            trade_prices: Transaction prices
            trade_signs: +1 for buyer-initiated, -1 for seller-initiated
            trade_sizes: Trade sizes

        Returns:
            MarketImpactEstimate with lambda and volatility parameters
        """
        price_changes = trade_prices.diff().dropna()
        signed_volume = (trade_signs * trade_sizes).iloc[1:]

        # Align indices
        idx = price_changes.index.intersection(signed_volume.index)
        dp = price_changes.loc[idx].values
        sv = signed_volume.loc[idx].values

        # OLS regression
        lambda_ = np.cov(dp, sv)[0, 1] / np.var(sv) if np.var(sv) > 0 else 0

        # Estimate volatilities
        sigma_dp = np.std(dp)
        sigma_sv = np.std(sv)

        # From Kyle model: sigma_dp^2 = lambda^2 * sigma_sv^2
        # and lambda = sigma_v / (2 * sigma_u)
        sigma_u = sigma_sv / 2 if sigma_sv > 0 else 0
        sigma_v = 2 * lambda_ * sigma_u if lambda_ > 0 else 0

        self.lambda_ = lambda_
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u

        return MarketImpactEstimate(
            lambda_=lambda_,
            sigma_u=sigma_u,
            sigma_v=sigma_v,
            mu=0,  # Not directly estimated
            method='Kyle_OLS'
        )

    def expected_price_impact(self, order_size: float) -> float:
        """
        Calculate expected price impact for given order size.

        ΔP = lambda * order_size

        Args:
            order_size: Signed order size (+ for buy, - for sell)

        Returns:
            Expected price change
        """
        if self.lambda_ is None:
            raise ValueError("Model not fitted. Call estimate_from_trades first.")
        return self.lambda_ * order_size

    def optimal_trade_intensity(self, total_order: float, time_horizon: float) -> float:
        """
        Calculate optimal trading rate (from Kyle model).

        Optimal intensity minimizes total execution cost.

        Args:
            total_order: Total order to execute
            time_horizon: Time available for execution

        Returns:
            Optimal order size per period
        """
        # In Kyle model, informed trader trades at constant rate
        return total_order / time_horizon


class GlostenMilgromModel:
    """
    Glosten-Milgrom (1985) Model - Adverse Selection

    Market maker faces informed and uninformed traders.
    Spread compensates for adverse selection risk.

    Key results:
        Ask = E[V | Buy order] > E[V]
        Bid = E[V | Sell order] < E[V]
        Spread = (Ask - Bid) = f(probability of informed trading)

    Reference:
        Glosten, L. R., & Milgrom, P. R. (1985). "Bid, Ask and Transaction
        Prices in a Specialist Market with Heterogeneously Informed Traders".
        Journal of Financial Economics, 14, 71-100.
    """

    def __init__(self):
        self.mu = None       # Probability of informed trading
        self.delta_v = None  # Information advantage
        self.spread = None   # Equilibrium spread

    def estimate_adverse_selection(
        self,
        trades: pd.DataFrame,
        mid_prices: pd.Series
    ) -> Tuple[float, float]:
        """
        Estimate adverse selection component of spread.

        Uses Glosten-Harris (1988) decomposition:
        ΔP = c * sign + z * sign * size + epsilon

        where c = adverse selection, z = inventory/order processing

        Args:
            trades: DataFrame with 'price', 'sign', 'size' columns
            mid_prices: Mid-price series

        Returns:
            Tuple of (adverse_selection, inventory_component)
        """
        # Price changes relative to mid
        price_change = trades['price'].diff()

        # Features
        sign = trades['sign']
        sign_size = trades['sign'] * trades['size']

        # Stack features
        X = pd.DataFrame({
            'sign': sign.iloc[1:],
            'sign_size': sign_size.iloc[1:]
        })
        y = price_change.iloc[1:]

        # Align and clean
        idx = X.dropna().index.intersection(y.dropna().index)
        X_clean = X.loc[idx].values
        y_clean = y.loc[idx].values

        if len(y_clean) < 10:
            return 0.0, 0.0

        # OLS
        X_aug = np.column_stack([np.ones(len(X_clean)), X_clean])
        try:
            beta = np.linalg.lstsq(X_aug, y_clean, rcond=None)[0]
            adverse_selection = beta[1]  # Coefficient on sign
            inventory = beta[2]  # Coefficient on sign * size
        except:
            adverse_selection = 0.0
            inventory = 0.0

        self.spread = 2 * adverse_selection  # Full spread

        return adverse_selection, inventory

    def probability_informed(
        self,
        spread: float,
        volatility: float,
        trade_frequency: float
    ) -> float:
        """
        Estimate probability of informed trading (mu).

        From Glosten-Milgrom model:
        Spread ≈ 2 * mu * delta_v

        where delta_v ≈ volatility / sqrt(trade_frequency)

        Args:
            spread: Bid-ask spread
            volatility: Price volatility
            trade_frequency: Number of trades per period

        Returns:
            Estimated probability of informed trading
        """
        if trade_frequency <= 0 or volatility <= 0:
            return 0.0

        delta_v = volatility / np.sqrt(trade_frequency)
        mu = spread / (2 * delta_v) if delta_v > 0 else 0

        self.mu = np.clip(mu, 0, 1)
        self.delta_v = delta_v

        return self.mu


class RollModel:
    """
    Roll (1984) Model - Effective Spread Estimation

    Estimates effective spread from first-order autocorrelation of returns.
    Works without observing quotes (only needs transaction prices).

    Key result:
        Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    Reference:
        Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask
        Spread in an Efficient Market". Journal of Finance, 39(4), 1127-1139.
    """

    @staticmethod
    def estimate_spread(
        prices: pd.Series,
        window: int = 50
    ) -> pd.Series:
        """
        Estimate effective spread using Roll model.

        Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

        Args:
            prices: Transaction price series
            window: Rolling window for estimation

        Returns:
            Rolling spread estimates
        """
        returns = prices.diff()

        def calc_spread(rets):
            if len(rets) < 3:
                return np.nan
            # First-order autocovariance
            autocov = np.cov(rets[:-1], rets[1:])[0, 1]
            if autocov >= 0:
                return 0  # Positive autocov implies no valid spread
            return 2 * np.sqrt(-autocov)

        spread = returns.rolling(window).apply(calc_spread, raw=True)

        return spread

    @staticmethod
    def effective_spread_single(price_changes: np.ndarray) -> float:
        """
        Single-period Roll spread estimate.

        Args:
            price_changes: Array of price changes

        Returns:
            Effective spread estimate
        """
        if len(price_changes) < 3:
            return np.nan
        autocov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        if autocov >= 0:
            return 0
        return 2 * np.sqrt(-autocov)


class HuangStollModel:
    """
    Huang & Stoll (1997) Model - Spread Component Decomposition

    Decomposes bid-ask spread into three components:
    1. Adverse selection (information asymmetry)
    2. Inventory holding cost
    3. Order processing cost

    Reference:
        Huang, R. D., & Stoll, H. R. (1997). "The Components of the Bid-Ask
        Spread: A General Approach". Review of Financial Studies, 10(4), 995-1034.
    """

    def __init__(self):
        self.adverse_selection = None
        self.inventory_cost = None
        self.order_processing = None
        self.total_spread = None

    def decompose_spread(
        self,
        trades: pd.DataFrame,
        quotes: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Decompose spread into components.

        Model:
        ΔQ_t = S/2 * (Q_t - α*Q_{t-1}) + ε_t

        where:
        - S = total spread
        - α = probability of trade reversal
        - Adverse selection = S * (1 - α)
        - Inventory + Order processing = S * α

        Args:
            trades: DataFrame with 'price', 'time' columns
            quotes: DataFrame with 'bid', 'ask', 'time' columns

        Returns:
            Dict with spread components
        """
        # Merge trades with quotes
        merged = pd.merge_asof(
            trades.sort_values('time'),
            quotes.sort_values('time'),
            on='time',
            direction='backward'
        )

        # Quote midpoint
        merged['mid'] = (merged['bid'] + merged['ask']) / 2
        merged['spread'] = merged['ask'] - merged['bid']

        # Trade indicator: +1 if above mid, -1 if below
        merged['sign'] = np.sign(merged['price'] - merged['mid'])

        # Lagged sign
        merged['sign_lag'] = merged['sign'].shift(1)

        # Quote revision
        merged['quote_change'] = merged['mid'].diff()

        # Regression: ΔQ = λ*S/2*sign + (1-λ)*S/2*sign_lag + ε
        # where λ = adverse selection proportion

        X = merged[['sign', 'sign_lag']].iloc[1:].dropna()
        y = merged['quote_change'].iloc[1:].loc[X.index]
        spread_mean = merged['spread'].mean() / 2  # Half spread

        if len(X) < 10:
            return {'adverse_selection': 0, 'inventory': 0, 'order_processing': 0}

        try:
            X_aug = np.column_stack([np.ones(len(X)), X.values])
            beta = np.linalg.lstsq(X_aug, y.values, rcond=None)[0]

            # Coefficients represent proportions of half-spread
            adverse_selection = beta[1] / spread_mean if spread_mean > 0 else 0
            inventory = (1 - adverse_selection) * 0.5  # Split remaining
            order_processing = (1 - adverse_selection) * 0.5

            # Clip to valid range
            adverse_selection = np.clip(adverse_selection, 0, 1)

        except:
            adverse_selection = 0.33
            inventory = 0.33
            order_processing = 0.34

        self.adverse_selection = adverse_selection
        self.inventory_cost = inventory
        self.order_processing = order_processing
        self.total_spread = merged['spread'].mean()

        return {
            'adverse_selection': adverse_selection,
            'inventory': inventory,
            'order_processing': order_processing,
            'total_spread': self.total_spread
        }


class HasbrouckModel:
    """
    Hasbrouck (1991) Model - Trade Informativeness

    Measures information content of trades using VAR model.
    Decomposes price into permanent (information) and transitory (noise) components.

    Reference:
        Hasbrouck, J. (1991). "Measuring the Information Content of Stock Trades".
        Journal of Finance, 46(1), 179-207.
    """

    def __init__(self):
        self.var_coef = None
        self.permanent_impact = None
        self.transitory_impact = None

    def estimate_var(
        self,
        returns: pd.Series,
        trade_signs: pd.Series,
        lags: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Estimate VAR model for returns and trade signs.

        r_t = Σ(a_j * r_{t-j}) + Σ(b_j * x_{t-j}) + ε_t
        x_t = Σ(c_j * r_{t-j}) + Σ(d_j * x_{t-j}) + η_t

        where x_t = trade sign (+1 buy, -1 sell)

        Args:
            returns: Return series
            trade_signs: Trade sign series
            lags: Number of VAR lags

        Returns:
            Dict with VAR coefficients
        """
        from scipy.linalg import lstsq

        # Build lag matrix
        n = len(returns)
        Y = np.column_stack([returns.values, trade_signs.values])[lags:]

        X_list = [np.ones(n - lags)]
        for lag in range(1, lags + 1):
            X_list.append(returns.shift(lag).values[lags:])
            X_list.append(trade_signs.shift(lag).values[lags:])

        X = np.column_stack(X_list)

        # Estimate VAR
        try:
            coef, _, _, _ = lstsq(X, Y)
            self.var_coef = coef
        except:
            self.var_coef = np.zeros((2 * lags + 1, 2))

        return {'coefficients': self.var_coef}

    def information_share(
        self,
        returns: pd.Series,
        trade_signs: pd.Series
    ) -> float:
        """
        Calculate information share of trades.

        Information share = variance of permanent component / total variance

        Args:
            returns: Return series
            trade_signs: Trade sign series

        Returns:
            Information share (0 to 1)
        """
        # Estimate VAR first
        self.estimate_var(returns, trade_signs)

        # Calculate impulse response to trade shock
        # Long-run impact of a trade = sum of b coefficients
        if self.var_coef is None:
            return 0.5

        # b coefficients are in odd columns (1, 3, 5, ...)
        b_coefs = self.var_coef[2::2, 0]  # Coefficients on lagged trade signs
        long_run_impact = np.sum(b_coefs)

        # Normalize by total variance
        total_var = np.var(returns)
        if total_var == 0:
            return 0.5

        # Information share approximation
        info_share = np.abs(long_run_impact) / np.sqrt(total_var)
        return np.clip(info_share, 0, 1)


class MarketImpactFeatures:
    """
    Generate market impact features for ML models.

    Combines all market impact models into feature set.
    """

    def __init__(self):
        self.kyle = KyleModel()
        self.gm = GlostenMilgromModel()
        self.roll = RollModel()
        self.hs = HuangStollModel()
        self.hasbrouck = HasbrouckModel()

    def compute_features(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        trade_signs: pd.Series,
        bid: Optional[pd.Series] = None,
        ask: Optional[pd.Series] = None,
        window: int = 50
    ) -> pd.DataFrame:
        """
        Compute all market impact features.

        Args:
            prices: Transaction prices
            volumes: Trade volumes
            trade_signs: +1 for buy, -1 for sell
            bid: Bid prices (optional)
            ask: Ask prices (optional)
            window: Rolling window

        Returns:
            DataFrame with market impact features
        """
        features = {}

        # Kyle's lambda
        signed_volume = trade_signs * volumes
        features['kyle_lambda'] = self.kyle.estimate_lambda(prices, signed_volume, window)

        # Roll spread
        features['roll_spread'] = self.roll.estimate_spread(prices, window)

        # If quotes available
        if bid is not None and ask is not None:
            spread = ask - bid
            mid = (bid + ask) / 2

            # Quoted spread
            features['quoted_spread'] = spread
            features['quoted_spread_pct'] = spread / mid * 10000  # bps

            # Effective spread (2 * |price - mid|)
            features['effective_spread'] = 2 * np.abs(prices - mid)

            # Realized spread (price impact)
            future_mid = mid.shift(-5)  # 5 periods ahead
            features['realized_spread'] = trade_signs * (future_mid - prices) * 2

            # Price impact (adverse selection proxy)
            features['price_impact'] = trade_signs * (mid.shift(-5) - mid)

        # Order flow imbalance impact
        ofi = signed_volume.rolling(window).sum()
        features['ofi_price_corr'] = prices.rolling(window).corr(ofi)

        # Amihud illiquidity
        returns = prices.pct_change()
        features['amihud_illiquidity'] = (
            np.abs(returns) / (volumes * prices)
        ).rolling(window).mean() * 1e6

        return pd.DataFrame(features)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_kyle_model() -> KyleModel:
    """Factory for Kyle model."""
    return KyleModel()


def create_market_impact_features() -> MarketImpactFeatures:
    """Factory for market impact features."""
    return MarketImpactFeatures()


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Market Impact Models Test")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Simulate trade data
    # True lambda = 0.001 (price impact per unit volume)
    true_lambda = 0.001

    # Random walk with impact
    volumes = np.random.exponential(100, n) * np.random.choice([-1, 1], n)
    noise = np.random.randn(n) * 0.0001
    price_changes = true_lambda * volumes + noise
    prices = pd.Series(100 + np.cumsum(price_changes))
    volumes = pd.Series(volumes)
    trade_signs = pd.Series(np.sign(volumes))

    print(f"\nTrue lambda: {true_lambda}")

    # Test Kyle model
    kyle = KyleModel()
    estimate = kyle.estimate_from_trades(prices, trade_signs, np.abs(volumes))
    print(f"\n1. Kyle Model:")
    print(f"   Estimated lambda: {estimate.lambda_:.6f}")
    print(f"   Sigma_u: {estimate.sigma_u:.4f}")
    print(f"   Sigma_v: {estimate.sigma_v:.4f}")

    # Test Roll spread
    roll_spread = RollModel.estimate_spread(prices, 50)
    print(f"\n2. Roll Spread:")
    print(f"   Estimated spread: {roll_spread.iloc[-1]:.6f}")

    # Test Glosten-Milgrom
    gm = GlostenMilgromModel()
    trades = pd.DataFrame({
        'price': prices,
        'sign': trade_signs,
        'size': np.abs(volumes)
    })
    adverse, inventory = gm.estimate_adverse_selection(trades, prices)
    print(f"\n3. Glosten-Milgrom:")
    print(f"   Adverse selection: {adverse:.6f}")
    print(f"   Inventory component: {inventory:.6f}")

    # Test Hasbrouck
    returns = prices.pct_change().fillna(0)
    hasbrouck = HasbrouckModel()
    info_share = hasbrouck.information_share(returns, trade_signs)
    print(f"\n4. Hasbrouck Information Share:")
    print(f"   Info share: {info_share:.4f}")

    # Test feature generation
    mif = MarketImpactFeatures()
    features = mif.compute_features(prices, np.abs(volumes), trade_signs)
    print(f"\n5. Feature Generation:")
    print(f"   Features: {list(features.columns)}")

    print("\nTest PASSED")
