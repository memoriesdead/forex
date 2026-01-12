"""
Gold Standard Quantitative Formulas
====================================
Mathematical methods verified from Chinese quants, WorldQuant, and academic research.

Sources:
- WorldQuant Alpha101 (101 Formulaic Alphas paper)
- Avellaneda-Stoikov (HFT Market Making, 2008)
- Marcos Lopez de Prado (Advances in Financial Machine Learning)
- Chinese quant research (Tsinghua, Peking University)
- Kelly Criterion (Edward Thorp)
- Almgren-Chriss (Optimal Execution)

These are PROVEN mathematical formulas used by billion-dollar quants.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# WORLDQUANT ALPHA101 FORMULAS (Adapted for Forex)
# =============================================================================

class Alpha101Forex:
    """
    WorldQuant Alpha101 formulas adapted for forex.

    Original paper: "101 Formulaic Alphas" by Zura Kakushadze (2016)
    These alphas use OHLCV data to generate trading signals.

    Adapted from: https://github.com/yli188/WorldQuant_alpha101_code
    """

    @staticmethod
    def rank(x: pd.Series) -> pd.Series:
        """Cross-sectional rank (percentile)."""
        return x.rank(pct=True)

    @staticmethod
    def delta(x: pd.Series, period: int = 1) -> pd.Series:
        """Difference from period ago."""
        return x.diff(period)

    @staticmethod
    def delay(x: pd.Series, period: int = 1) -> pd.Series:
        """Lag by period."""
        return x.shift(period)

    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling correlation."""
        return x.rolling(window).corr(y)

    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling covariance."""
        return x.rolling(window).cov(y)

    @staticmethod
    def ts_rank(x: pd.Series, window: int) -> pd.Series:
        """Time-series rank over window."""
        return x.rolling(window).apply(lambda arr: stats.rankdata(arr)[-1] / len(arr))

    @staticmethod
    def ts_max(x: pd.Series, window: int) -> pd.Series:
        """Rolling max."""
        return x.rolling(window).max()

    @staticmethod
    def ts_min(x: pd.Series, window: int) -> pd.Series:
        """Rolling min."""
        return x.rolling(window).min()

    @staticmethod
    def ts_argmax(x: pd.Series, window: int) -> pd.Series:
        """Position of max in window."""
        return x.rolling(window).apply(lambda arr: np.argmax(arr) + 1)

    @staticmethod
    def ts_argmin(x: pd.Series, window: int) -> pd.Series:
        """Position of min in window."""
        return x.rolling(window).apply(lambda arr: np.argmin(arr) + 1)

    @staticmethod
    def stddev(x: pd.Series, window: int) -> pd.Series:
        """Rolling standard deviation."""
        return x.rolling(window).std()

    @staticmethod
    def sum_rolling(x: pd.Series, window: int) -> pd.Series:
        """Rolling sum."""
        return x.rolling(window).sum()

    @staticmethod
    def product(x: pd.Series, window: int) -> pd.Series:
        """Rolling product."""
        return x.rolling(window).apply(np.prod)

    def alpha001(self, close: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        Momentum reversal signal.
        """
        cond = returns < 0
        inner = pd.Series(np.where(cond, self.stddev(returns, 20), close))
        return self.rank(self.ts_argmax(inner ** 2, 5)) - 0.5

    def alpha002(self, open_: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        Volume-price divergence.
        """
        return -1 * self.correlation(
            self.rank(self.delta(np.log(volume + 1), 2)),
            self.rank((close - open_) / (open_ + 0.0001)),
            6
        )

    def alpha003(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        Open-volume correlation.
        """
        return -1 * self.correlation(self.rank(open_), self.rank(volume), 10)

    def alpha004(self, low: pd.Series) -> pd.Series:
        """
        Alpha#4: (-1 * Ts_Rank(rank(low), 9))
        Low price momentum.
        """
        return -1 * self.ts_rank(self.rank(low), 9)

    def alpha006(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#6: (-1 * correlation(open, volume, 10))
        Simple open-volume correlation.
        """
        return -1 * self.correlation(open_, volume, 10)

    def alpha012(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        Volume-confirmed price reversal.
        """
        return np.sign(self.delta(volume, 1)) * (-1 * self.delta(close, 1))

    def alpha013(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
        Price-volume covariance.
        """
        return -1 * self.rank(self.covariance(self.rank(close), self.rank(volume), 5))

    def alpha014(self, open_: pd.Series, volume: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        Return momentum with volume confirmation.
        """
        return (-1 * self.rank(self.delta(returns, 3))) * self.correlation(open_, volume, 10)

    def alpha017(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        Complex momentum signal.
        """
        adv20 = volume.rolling(20).mean()
        return (
            (-1 * self.rank(self.ts_rank(close, 10))) *
            self.rank(self.delta(self.delta(close, 1), 1)) *
            self.rank(self.ts_rank(volume / (adv20 + 0.0001), 5))
        )

    def alpha020(self, open_: pd.Series, high: pd.Series, close: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        Gap analysis.
        """
        return (
            (-1 * self.rank(open_ - self.delay(high, 1))) *
            self.rank(open_ - self.delay(close, 1)) *
            self.rank(open_ - self.delay(low, 1))
        )

    def alpha033(self, open_: pd.Series, close: pd.Series) -> pd.Series:
        """
        Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        Open-close ratio.
        """
        return self.rank(-1 * (1 - (open_ / (close + 0.0001))))

    def alpha034(self, close: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        Volatility regime signal.
        """
        return self.rank(
            (1 - self.rank(self.stddev(returns, 2) / (self.stddev(returns, 5) + 0.0001))) +
            (1 - self.rank(self.delta(close, 1)))
        )

    def alpha038(self, close: pd.Series, open_: pd.Series) -> pd.Series:
        """
        Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        Close momentum with intraday ratio.
        """
        return (-1 * self.rank(self.ts_rank(close, 10))) * self.rank(close / (open_ + 0.0001))

    def alpha041(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Alpha#41: (((high * low)^0.5) - vwap)
        Geometric mean vs VWAP approximation.
        """
        vwap = (high + low + volume.rolling(5).mean()) / 3  # Simplified VWAP
        return np.sqrt(high * low) - vwap

    def alpha053(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Alpha#53: ((-1 * delta((((close - low) - (high - close)) / (close - low)), 9)))
        Price position within range.
        """
        inner = ((close - low) - (high - close)) / (close - low + 0.0001)
        return -1 * self.delta(inner, 9)

    def generate_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Alpha101 signals for forex data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume (or tick_count for forex)

        Returns:
            DataFrame with alpha signals added
        """
        result = df.copy()

        # Ensure required columns exist
        open_ = df['open'] if 'open' in df.columns else df['mid']
        high = df['high'] if 'high' in df.columns else df['ask']
        low = df['low'] if 'low' in df.columns else df['bid']
        close = df['close'] if 'close' in df.columns else df['mid']
        volume = df['volume'] if 'volume' in df.columns else df.get('tick_count', pd.Series(1, index=df.index))

        returns = close.pct_change()

        try:
            result['alpha001'] = self.alpha001(close, returns)
            result['alpha004'] = self.alpha004(low)
            result['alpha012'] = self.alpha012(volume, close)
            result['alpha033'] = self.alpha033(open_, close)
            result['alpha034'] = self.alpha034(close, returns)
            result['alpha038'] = self.alpha038(close, open_)
            result['alpha053'] = self.alpha053(close, high, low)

            # Only add volume-dependent alphas if volume is meaningful
            if volume.std() > 0:
                result['alpha002'] = self.alpha002(open_, close, volume)
                result['alpha003'] = self.alpha003(open_, volume)
                result['alpha006'] = self.alpha006(open_, volume)
                result['alpha013'] = self.alpha013(close, volume)
                result['alpha014'] = self.alpha014(open_, volume, returns)
                result['alpha017'] = self.alpha017(close, volume)

            result['alpha020'] = self.alpha020(open_, high, close, low)
            result['alpha041'] = self.alpha041(high, low, volume)

        except Exception as e:
            logger.warning(f"Error generating some alphas: {e}")

        return result


# =============================================================================
# AVELLANEDA-STOIKOV MARKET MAKING (HFT)
# =============================================================================

class AvellanedaStoikov:
    """
    Avellaneda-Stoikov HFT Market Making Model.

    Paper: "High-frequency trading in a limit order book" (2008)
    Source: https://github.com/fedecaccia/avellaneda-stoikov

    This calculates optimal bid/ask quotes for market making.
    """

    def __init__(self, gamma: float = 0.1, sigma: float = 0.02, k: float = 1.5):
        """
        Initialize Avellaneda-Stoikov parameters.

        Args:
            gamma: Risk aversion parameter (higher = more conservative)
            sigma: Volatility of the asset
            k: Order book liquidity parameter
        """
        self.gamma = gamma
        self.sigma = sigma
        self.k = k

    def reservation_price(self, mid_price: float, inventory: int, time_remaining: float) -> float:
        """
        Calculate the reservation (indifference) price.

        r(s,q,t) = s - q * gamma * sigma^2 * (T - t)

        The reservation price adjusts the mid price based on inventory:
        - Positive inventory (long) -> lower reservation price (encourage selling)
        - Negative inventory (short) -> higher reservation price (encourage buying)

        Args:
            mid_price: Current mid price
            inventory: Current inventory position (positive = long, negative = short)
            time_remaining: Time remaining until end of trading period (0 to 1)

        Returns:
            Reservation price
        """
        return mid_price - inventory * self.gamma * (self.sigma ** 2) * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """
        Calculate the optimal bid-ask spread.

        delta(t) = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

        Args:
            time_remaining: Time remaining until end of trading period

        Returns:
            Optimal spread (divide by 2 for each side)
        """
        term1 = self.gamma * (self.sigma ** 2) * time_remaining
        term2 = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        return term1 + term2

    def optimal_quotes(self, mid_price: float, inventory: int, time_remaining: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.

        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            time_remaining: Time remaining (0 to 1)

        Returns:
            Tuple of (bid_price, ask_price)
        """
        r = self.reservation_price(mid_price, inventory, time_remaining)
        spread = self.optimal_spread(time_remaining)
        half_spread = spread / 2

        bid = r - half_spread
        ask = r + half_spread

        return bid, ask

    def inventory_skew(self, inventory: int, max_inventory: int = 100) -> float:
        """
        Calculate inventory skew factor.

        Used to adjust spreads based on inventory levels:
        - At max inventory, widen the side we want to exit
        - At zero inventory, symmetric spreads

        Args:
            inventory: Current position
            max_inventory: Maximum allowed inventory

        Returns:
            Skew factor (-1 to 1)
        """
        return np.clip(inventory / max_inventory, -1, 1)

    def get_signals(self, df: pd.DataFrame, inventory: int = 0,
                    trading_hours: float = 24.0) -> pd.DataFrame:
        """
        Generate market making signals for a DataFrame.

        Args:
            df: DataFrame with 'mid' or 'close' price column
            inventory: Current inventory position
            trading_hours: Total trading hours in period

        Returns:
            DataFrame with bid, ask, reservation price columns
        """
        result = df.copy()

        mid = df['mid'] if 'mid' in df.columns else df['close']

        # Calculate time remaining (assuming uniform time steps)
        n = len(df)
        time_remaining = np.linspace(1, 0.01, n)  # Never reach exactly 0

        bids = []
        asks = []
        reservations = []

        for i, (price, t) in enumerate(zip(mid.values, time_remaining)):
            r = self.reservation_price(price, inventory, t)
            bid, ask = self.optimal_quotes(price, inventory, t)

            bids.append(bid)
            asks.append(ask)
            reservations.append(r)

        result['as_reservation'] = reservations
        result['as_bid'] = bids
        result['as_ask'] = asks
        result['as_spread'] = result['as_ask'] - result['as_bid']

        return result


# =============================================================================
# KELLY CRITERION & FRACTIONAL KELLY
# =============================================================================

class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    Source: Edward Thorp, "Beat the Dealer" (1962)

    f* = (p * b - q) / b

    Where:
        f* = optimal fraction of bankroll to bet
        p = probability of winning
        q = probability of losing (1 - p)
        b = odds (net return if win)

    For trading: f* = (expected_return) / (variance)
    """

    @staticmethod
    def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly fraction for binary outcomes.

        Args:
            win_prob: Probability of winning (0 to 1)
            win_loss_ratio: Average win / Average loss

        Returns:
            Optimal fraction to bet (0 to 1, can be negative for short)
        """
        q = 1 - win_prob
        b = win_loss_ratio

        kelly = (win_prob * b - q) / b
        return max(kelly, 0)  # Don't bet if edge is negative

    @staticmethod
    def fractional_kelly(win_prob: float, win_loss_ratio: float,
                         fraction: float = 0.5) -> float:
        """
        Calculate fractional Kelly (more conservative).

        Half-Kelly (fraction=0.5) gives:
        - 75% of optimal growth rate
        - 25% of the variance

        Quarter-Kelly (fraction=0.25) gives:
        - 43.75% of optimal growth rate
        - 6.25% of the variance

        Args:
            win_prob: Probability of winning
            win_loss_ratio: Average win / Average loss
            fraction: Fraction of Kelly to use (0.25 to 0.5 recommended)

        Returns:
            Fractional Kelly bet size
        """
        full_kelly = KellyCriterion.kelly_fraction(win_prob, win_loss_ratio)
        return full_kelly * fraction

    @staticmethod
    def kelly_from_returns(returns: pd.Series) -> float:
        """
        Calculate Kelly fraction from historical returns.

        f* = mean(returns) / var(returns)

        Args:
            returns: Series of historical returns

        Returns:
            Optimal Kelly fraction
        """
        mean_return = returns.mean()
        variance = returns.var()

        if variance <= 0:
            return 0

        return mean_return / variance

    @staticmethod
    def optimal_leverage(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate optimal leverage using Kelly.

        f* = (mu - r) / sigma^2

        Where:
            mu = expected return
            r = risk-free rate
            sigma = volatility

        Args:
            returns: Historical returns
            risk_free_rate: Risk-free rate (same frequency as returns)

        Returns:
            Optimal leverage (>1 means leverage, <1 means partial investment)
        """
        excess_return = returns.mean() - risk_free_rate
        variance = returns.var()

        if variance <= 0:
            return 0

        return excess_return / variance

    @staticmethod
    def position_size(account_value: float, win_prob: float,
                      win_loss_ratio: float, fraction: float = 0.5,
                      max_position_pct: float = 0.25) -> float:
        """
        Calculate actual position size in currency.

        Args:
            account_value: Total account value
            win_prob: Probability of winning
            win_loss_ratio: Average win / Average loss
            fraction: Kelly fraction (0.5 = half Kelly)
            max_position_pct: Maximum position as % of account

        Returns:
            Position size in currency units
        """
        kelly_pct = KellyCriterion.fractional_kelly(win_prob, win_loss_ratio, fraction)

        # Cap at maximum position
        kelly_pct = min(kelly_pct, max_position_pct)

        return account_value * kelly_pct


# =============================================================================
# TRIPLE BARRIER METHOD (Lopez de Prado)
# =============================================================================

class TripleBarrier:
    """
    Triple Barrier Method for labeling trades.

    Source: Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018)

    Three barriers:
    1. Upper barrier (profit taking)
    2. Lower barrier (stop loss)
    3. Vertical barrier (time limit)

    The first barrier touched determines the label.
    """

    @staticmethod
    def get_vertical_barrier(timestamps: pd.DatetimeIndex,
                             t_events: pd.DatetimeIndex,
                             num_bars: int) -> pd.Series:
        """
        Get vertical barrier timestamps.

        Args:
            timestamps: Full price bar timestamps
            t_events: Event timestamps (when signals occur)
            num_bars: Number of bars for vertical barrier

        Returns:
            Series with vertical barrier timestamps
        """
        t1 = pd.Series(pd.NaT, index=t_events)

        for i, t0 in enumerate(t_events):
            try:
                loc = timestamps.get_loc(t0)
                if loc + num_bars < len(timestamps):
                    t1.iloc[i] = timestamps[loc + num_bars]
            except KeyError:
                continue

        return t1

    @staticmethod
    def apply_triple_barrier(close: pd.Series,
                            t_events: pd.DatetimeIndex,
                            pt_sl: Tuple[float, float],
                            vertical_barrier: pd.Series,
                            min_return: float = 0.0) -> pd.DataFrame:
        """
        Apply triple barrier method.

        Args:
            close: Close price series
            t_events: Event timestamps
            pt_sl: Tuple of (profit_taking_mult, stop_loss_mult) relative to daily vol
            vertical_barrier: Vertical barrier timestamps
            min_return: Minimum return threshold

        Returns:
            DataFrame with barrier touch information
        """
        # Calculate daily volatility
        daily_vol = close.pct_change().rolling(20).std()

        results = []

        for t0 in t_events:
            if t0 not in close.index:
                continue

            entry_price = close.loc[t0]
            vol = daily_vol.loc[t0] if t0 in daily_vol.index else daily_vol.mean()

            # Set barriers
            pt = entry_price * (1 + pt_sl[0] * vol) if pt_sl[0] > 0 else np.inf
            sl = entry_price * (1 - pt_sl[1] * vol) if pt_sl[1] > 0 else 0

            # Get vertical barrier
            t1 = vertical_barrier.loc[t0] if t0 in vertical_barrier.index else pd.NaT

            # Find path after event
            if pd.isna(t1):
                t1 = close.index[-1]

            path = close.loc[t0:t1]

            # Find first barrier touch
            touch_pt = path[path >= pt].index.min() if (path >= pt).any() else pd.NaT
            touch_sl = path[path <= sl].index.min() if (path <= sl).any() else pd.NaT

            # Determine which barrier was touched first
            if pd.isna(touch_pt) and pd.isna(touch_sl):
                # Vertical barrier
                touch_time = t1
                label = 0  # Neutral
            elif pd.isna(touch_pt):
                touch_time = touch_sl
                label = -1  # Stop loss
            elif pd.isna(touch_sl):
                touch_time = touch_pt
                label = 1  # Profit taking
            else:
                if touch_pt <= touch_sl:
                    touch_time = touch_pt
                    label = 1
                else:
                    touch_time = touch_sl
                    label = -1

            # Calculate return
            if touch_time in close.index:
                ret = (close.loc[touch_time] - entry_price) / entry_price
            else:
                ret = 0

            results.append({
                't0': t0,
                't1': touch_time,
                'ret': ret,
                'label': label,
                'pt': pt,
                'sl': sl
            })

        return pd.DataFrame(results)

    @staticmethod
    def meta_labeling(primary_signal: pd.Series,
                      triple_barrier_labels: pd.DataFrame) -> pd.Series:
        """
        Apply meta-labeling: filter primary model signals.

        The meta-label is 1 if the primary signal would have been profitable,
        0 otherwise. This is used to train a secondary model that learns
        when to trust the primary model.

        Args:
            primary_signal: Primary model's directional signal (1, 0, -1)
            triple_barrier_labels: Output from apply_triple_barrier

        Returns:
            Meta-labels (1 = take the trade, 0 = skip)
        """
        meta_labels = pd.Series(0, index=primary_signal.index)

        for _, row in triple_barrier_labels.iterrows():
            t0 = row['t0']
            if t0 not in primary_signal.index:
                continue

            signal = primary_signal.loc[t0]
            label = row['label']

            # Meta-label is 1 if signal direction matches outcome
            if signal > 0 and label > 0:
                meta_labels.loc[t0] = 1
            elif signal < 0 and label < 0:
                meta_labels.loc[t0] = 1
            elif signal == 0:
                meta_labels.loc[t0] = 0

        return meta_labels


# =============================================================================
# FRACTIONAL DIFFERENTIATION (Lopez de Prado)
# =============================================================================

class FractionalDifferentiation:
    """
    Fractional Differentiation for stationarity while preserving memory.

    Source: Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018)

    Standard differentiation (d=1) removes all memory.
    Fractional differentiation (0 < d < 1) balances stationarity and memory.
    """

    @staticmethod
    def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
        """
        Calculate weights for fractional differentiation.

        w_k = -w_{k-1} * (d - k + 1) / k

        Args:
            d: Fractional differentiation order (0 to 1)
            size: Number of weights to compute
            threshold: Minimum weight to include

        Returns:
            Array of weights
        """
        weights = [1.0]
        k = 1

        while k < size:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1

        return np.array(weights[::-1])

    @staticmethod
    def frac_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
        """
        Apply fractional differentiation to a series.

        Args:
            series: Price series
            d: Fractional differentiation order
            threshold: Minimum weight threshold

        Returns:
            Fractionally differentiated series
        """
        weights = FractionalDifferentiation.get_weights(d, len(series), threshold)
        width = len(weights)

        result = pd.Series(index=series.index, dtype=float)

        for i in range(width, len(series)):
            result.iloc[i] = np.dot(weights, series.iloc[i-width+1:i+1].values)

        return result

    @staticmethod
    def get_optimal_d(series: pd.Series,
                      d_range: Tuple[float, float] = (0.1, 1.0),
                      adf_threshold: float = -2.86) -> float:
        """
        Find minimum d that achieves stationarity.

        Args:
            series: Price series
            d_range: Range of d values to test
            adf_threshold: ADF statistic threshold for stationarity

        Returns:
            Optimal d value
        """
        from statsmodels.tsa.stattools import adfuller

        for d in np.arange(d_range[0], d_range[1], 0.05):
            frac_series = FractionalDifferentiation.frac_diff(series, d)
            frac_series = frac_series.dropna()

            if len(frac_series) < 20:
                continue

            adf_stat = adfuller(frac_series)[0]

            if adf_stat < adf_threshold:
                return d

        return d_range[1]  # Return max if no stationarity found


# =============================================================================
# ALMGREN-CHRISS OPTIMAL EXECUTION
# =============================================================================

class AlmgrenChriss:
    """
    Almgren-Chriss Optimal Execution Model.

    Paper: "Optimal Execution of Portfolio Transactions" (2000)
    Source: https://github.com/joshuapjacob/almgren-chriss-optimal-execution

    Balances market impact vs timing risk for large orders.
    """

    def __init__(self, sigma: float, eta: float, gamma: float,
                 epsilon: float = 0.0, lambda_: float = 1e-6):
        """
        Initialize Almgren-Chriss parameters.

        Args:
            sigma: Volatility (daily)
            eta: Temporary market impact coefficient
            gamma: Permanent market impact coefficient
            epsilon: Fixed transaction cost
            lambda_: Risk aversion parameter
        """
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_

    def optimal_trajectory(self, X: float, T: int, n_steps: int) -> np.ndarray:
        """
        Calculate optimal execution trajectory.

        Args:
            X: Total shares to execute
            T: Total time horizon
            n_steps: Number of trading intervals

        Returns:
            Array of shares to trade at each step
        """
        tau = T / n_steps  # Time per interval

        # Calculate kappa
        kappa_sq = self.lambda_ * (self.sigma ** 2) / (self.eta * tau)
        kappa = np.sqrt(kappa_sq)

        # Time grid
        t = np.linspace(0, T, n_steps + 1)

        # Holdings at each time
        x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

        # Trades at each interval (negative of change in holdings)
        trades = -np.diff(x)

        return trades

    def execution_cost(self, X: float, T: int, n_steps: int) -> Dict[str, float]:
        """
        Calculate expected execution cost and variance.

        Args:
            X: Total shares to execute
            T: Total time horizon
            n_steps: Number of trading intervals

        Returns:
            Dict with 'expected_cost' and 'cost_variance'
        """
        trades = self.optimal_trajectory(X, T, n_steps)
        tau = T / n_steps

        # Permanent impact cost
        permanent_cost = 0.5 * self.gamma * X ** 2

        # Temporary impact cost
        temporary_cost = self.eta * np.sum(trades ** 2) / tau

        # Fixed costs
        fixed_cost = self.epsilon * n_steps

        # Variance (timing risk)
        t = np.linspace(0, T, n_steps + 1)
        x = X * np.sinh(np.sqrt(self.lambda_) * (T - t)) / np.sinh(np.sqrt(self.lambda_) * T)
        variance = (self.sigma ** 2) * tau * np.sum(x[:-1] ** 2)

        total_cost = permanent_cost + temporary_cost + fixed_cost

        return {
            'expected_cost': total_cost,
            'cost_variance': variance,
            'permanent_impact': permanent_cost,
            'temporary_impact': temporary_cost,
            'fixed_cost': fixed_cost
        }

    def twap_trajectory(self, X: float, n_steps: int) -> np.ndarray:
        """
        Calculate TWAP (Time-Weighted Average Price) trajectory.

        TWAP corresponds to zero risk aversion (lambda = 0).

        Args:
            X: Total shares to execute
            n_steps: Number of trading intervals

        Returns:
            Array of uniform trades
        """
        return np.full(n_steps, X / n_steps)

    def vwap_adjustment(self, trades: np.ndarray,
                        volume_profile: np.ndarray) -> np.ndarray:
        """
        Adjust trajectory to follow VWAP profile.

        Args:
            trades: Base trading trajectory
            volume_profile: Expected volume at each interval (should sum to 1)

        Returns:
            Volume-adjusted trades
        """
        total = trades.sum()
        return total * volume_profile / volume_profile.sum()


# =============================================================================
# STOCHASTIC VOLATILITY MODELS
# =============================================================================

class StochasticVolatility:
    """
    Stochastic Volatility Models.

    Sources:
    - Heston (1993)
    - SABR (Hagan et al., 2002)
    - Chinese research from Peking University
    """

    @staticmethod
    def heston_variance_path(v0: float, kappa: float, theta: float,
                             sigma_v: float, T: float, n_steps: int,
                             rng: np.random.Generator = None) -> np.ndarray:
        """
        Simulate Heston variance path.

        dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW_t

        Args:
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-run variance
            sigma_v: Vol of vol
            T: Time horizon
            n_steps: Number of steps
            rng: Random number generator

        Returns:
            Simulated variance path
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps
        v = np.zeros(n_steps + 1)
        v[0] = v0

        for i in range(n_steps):
            dW = rng.standard_normal() * np.sqrt(dt)
            v[i+1] = v[i] + kappa * (theta - v[i]) * dt + sigma_v * np.sqrt(max(v[i], 0)) * dW
            v[i+1] = max(v[i+1], 0)  # Ensure non-negative

        return v

    @staticmethod
    def sabr_implied_vol(F: float, K: float, T: float,
                         alpha: float, beta: float, rho: float,
                         nu: float) -> float:
        """
        SABR implied volatility approximation.

        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            alpha: Initial volatility
            beta: CEV exponent (0 to 1)
            rho: Correlation between price and vol
            nu: Vol of vol

        Returns:
            Implied volatility
        """
        if abs(F - K) < 1e-10:
            # ATM approximation
            logFK = 0
            FK_mid = F
        else:
            logFK = np.log(F / K)
            FK_mid = np.sqrt(F * K)

        FK_beta = (F * K) ** ((1 - beta) / 2)

        # z calculation
        z = (nu / alpha) * FK_beta * logFK

        # x(z) calculation
        if abs(z) < 1e-10:
            x_z = 1
        else:
            x_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

        # First factor
        factor1 = alpha / (FK_beta * (1 + ((1-beta)**2 / 24) * logFK**2 +
                                       ((1-beta)**4 / 1920) * logFK**4))

        # Second factor
        factor2 = 1 + T * (((1-beta)**2 / 24) * (alpha**2 / FK_beta**2) +
                           0.25 * rho * beta * nu * alpha / FK_beta +
                           (2 - 3*rho**2) * nu**2 / 24)

        return factor1 * x_z * factor2

    @staticmethod
    def estimate_heston_params(returns: pd.Series,
                               vol: pd.Series) -> Dict[str, float]:
        """
        Estimate Heston model parameters from data.

        Uses method of moments estimation.

        Args:
            returns: Return series
            vol: Realized volatility series

        Returns:
            Dict with kappa, theta, sigma_v estimates
        """
        variance = vol ** 2

        # Long-run variance
        theta = variance.mean()

        # Mean reversion (from AR(1) on variance)
        var_diff = variance.diff().dropna()
        var_lag = variance.shift(1).dropna()

        # Simple regression: dv = kappa * (theta - v) * dt
        # Approximate kappa from autocorrelation
        autocorr = variance.autocorr(lag=1)
        kappa = -np.log(autocorr) if autocorr > 0 else 1.0

        # Vol of vol
        sigma_v = var_diff.std() / np.sqrt(variance.mean())

        return {
            'kappa': kappa,
            'theta': theta,
            'sigma_v': sigma_v
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_quant_formulas() -> Dict:
    """
    Create all quant formula calculators.

    Returns:
        Dict with all formula calculators
    """
    return {
        'alpha101': Alpha101Forex(),
        'avellaneda_stoikov': AvellanedaStoikov(),
        'kelly': KellyCriterion(),
        'triple_barrier': TripleBarrier(),
        'frac_diff': FractionalDifferentiation(),
        'almgren_chriss': AlmgrenChriss(sigma=0.02, eta=2.5e-6, gamma=2.5e-7),
        'stochastic_vol': StochasticVolatility()
    }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Gold Standard Quant Formulas")
    print("=" * 50)

    # Kelly example
    kelly = KellyCriterion()
    win_prob = 0.55
    win_loss_ratio = 1.5

    full_kelly = kelly.kelly_fraction(win_prob, win_loss_ratio)
    half_kelly = kelly.fractional_kelly(win_prob, win_loss_ratio, 0.5)

    print(f"\nKelly Criterion:")
    print(f"  Win probability: {win_prob:.1%}")
    print(f"  Win/Loss ratio: {win_loss_ratio:.1f}")
    print(f"  Full Kelly: {full_kelly:.1%}")
    print(f"  Half Kelly: {half_kelly:.1%}")

    # Avellaneda-Stoikov example
    as_model = AvellanedaStoikov(gamma=0.1, sigma=0.02, k=1.5)
    bid, ask = as_model.optimal_quotes(mid_price=1.1000, inventory=10, time_remaining=0.5)

    print(f"\nAvellaneda-Stoikov Market Making:")
    print(f"  Mid price: 1.1000")
    print(f"  Inventory: 10 (long)")
    print(f"  Optimal bid: {bid:.5f}")
    print(f"  Optimal ask: {ask:.5f}")
    print(f"  Spread: {(ask-bid)*10000:.1f} pips")

    print("\nAll formulas loaded successfully!")
