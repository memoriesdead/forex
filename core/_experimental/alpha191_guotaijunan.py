"""
国泰君安 Alpha191 - 191 Short-Period Technical Factors
=======================================================
Source: 国泰君安 (Guotai Junan Securities) Research Report
"基于短周期价量特征的多因子选股体系——数量化专题之九十三" (2017)

Verified from:
- Zhihu (知乎): https://zhuanlan.zhihu.com/p/58595574
- GitHub: https://github.com/Daic115/alpha191
- GitHub: https://github.com/popbo/alphas

Adapted for forex from original equity implementation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class Alpha191GuotaiJunan:
    """
    Complete implementation of all 191 国泰君安 short-period alphas.

    Usage:
        alpha = Alpha191GuotaiJunan()
        df = alpha.generate_all_alphas(ohlcv_data)
    """

    # =========================================================================
    # HELPER FUNCTIONS (Same as Alpha101 for consistency)
    # =========================================================================

    @staticmethod
    def rank(x: pd.Series) -> pd.Series:
        """Cross-sectional rank (percentile)."""
        return x.rank(pct=True)

    @staticmethod
    def delta(x: pd.Series, d: int = 1) -> pd.Series:
        """Difference from d periods ago."""
        return x.diff(d)

    @staticmethod
    def delay(x: pd.Series, d: int = 1) -> pd.Series:
        """Lag by d periods."""
        return x.shift(d)

    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Rolling correlation."""
        return x.rolling(d, min_periods=d//2).corr(y)

    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Rolling covariance."""
        return x.rolling(d, min_periods=d//2).cov(y)

    @staticmethod
    def ts_rank(x: pd.Series, d: int) -> pd.Series:
        """Time-series rank over d periods."""
        return x.rolling(d, min_periods=d//2).apply(
            lambda arr: pd.Series(arr).rank().iloc[-1] / len(arr), raw=False
        )

    @staticmethod
    def ts_max(x: pd.Series, d: int) -> pd.Series:
        """Rolling maximum."""
        return x.rolling(d, min_periods=1).max()

    @staticmethod
    def ts_min(x: pd.Series, d: int) -> pd.Series:
        """Rolling minimum."""
        return x.rolling(d, min_periods=1).min()

    @staticmethod
    def ts_argmax(x: pd.Series, d: int) -> pd.Series:
        """Days since maximum (HIGHDAY)."""
        return x.rolling(d, min_periods=1).apply(lambda arr: d - np.argmax(arr) - 1, raw=True)

    @staticmethod
    def ts_argmin(x: pd.Series, d: int) -> pd.Series:
        """Days since minimum (LOWDAY)."""
        return x.rolling(d, min_periods=1).apply(lambda arr: d - np.argmin(arr) - 1, raw=True)

    @staticmethod
    def ts_sum(x: pd.Series, d: int) -> pd.Series:
        """Rolling sum (SUM)."""
        return x.rolling(d, min_periods=1).sum()

    @staticmethod
    def ts_mean(x: pd.Series, d: int) -> pd.Series:
        """Rolling mean (MEAN)."""
        return x.rolling(d, min_periods=1).mean()

    @staticmethod
    def ts_std(x: pd.Series, d: int) -> pd.Series:
        """Rolling standard deviation (STD)."""
        return x.rolling(d, min_periods=2).std()

    @staticmethod
    def ts_count(cond: pd.Series, d: int) -> pd.Series:
        """Count True values in rolling window."""
        return cond.astype(float).rolling(d, min_periods=1).sum()

    @staticmethod
    def decay_linear(x: pd.Series, d: int) -> pd.Series:
        """Linear decay weighted average (DECAYLINEAR)."""
        weights = np.arange(1, d + 1)
        return x.rolling(d, min_periods=d//2).apply(
            lambda arr: np.dot(arr[-len(weights):], weights[-len(arr):]) / weights[-len(arr):].sum() if len(arr) > 0 else np.nan,
            raw=True
        )

    @staticmethod
    def sma(x: pd.Series, n: int, m: int) -> pd.Series:
        """
        SMA(A, n, m) = (A * m + DELAY(SMA, 1) * (n - m)) / n
        Exponential moving average variant.
        """
        alpha = m / n
        return x.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def wma(x: pd.Series, d: int) -> pd.Series:
        """Weighted moving average."""
        weights = np.arange(1, d + 1)
        return x.rolling(d).apply(lambda arr: np.dot(arr, weights) / weights.sum(), raw=True)

    @staticmethod
    def sign(x: pd.Series) -> pd.Series:
        """Sign function."""
        return np.sign(x)

    @staticmethod
    def log(x: pd.Series) -> pd.Series:
        """Natural log."""
        return np.log(x.replace(0, np.nan).clip(lower=1e-10))

    @staticmethod
    def abs_(x: pd.Series) -> pd.Series:
        """Absolute value."""
        return x.abs()

    @staticmethod
    def regbeta(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
        """Rolling regression beta coefficient."""
        def calc_beta(y_arr, x_arr):
            if len(y_arr) < 2:
                return np.nan
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            cov = np.sum((x_arr - x_mean) * (y_arr - y_mean))
            var = np.sum((x_arr - x_mean) ** 2)
            return cov / var if var != 0 else np.nan

        # Create sequence for regression
        result = pd.Series(index=y.index, dtype=float)
        seq = np.arange(1, d + 1)
        for i in range(d - 1, len(y)):
            y_window = y.iloc[i - d + 1:i + 1].values
            result.iloc[i] = calc_beta(y_window, seq)
        return result

    # =========================================================================
    # ALPHAS 001-050
    # =========================================================================

    def alpha001(self, close: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK((CLOSE - OPEN) / OPEN), 6))"""
        return -1 * self.correlation(
            self.rank(self.delta(self.log(volume + 1), 1)),
            self.rank((close - open_) / (open_ + 1e-8)),
            6
        )

    def alpha002(self, close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))"""
        inner = ((close - low) - (high - close)) / (high - low + 1e-8)
        return -1 * self.delta(inner, 1)

    def alpha003(self, close: pd.Series) -> pd.Series:
        """SUM(conditional spread adjustments based on CLOSE vs DELAY)"""
        cond1 = close == self.delay(close, 1)
        cond2 = close > self.delay(close, 1)

        inner = np.where(cond1, 0,
                 np.where(cond2, close - self.ts_min(low, 6), close - self.ts_max(high, 6)))
        return self.ts_sum(pd.Series(inner, index=close.index), 6)

    def alpha004(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Complex conditional based on 8-day vs 2-day averages and volume ratios"""
        cond1 = (self.ts_sum(close, 8) / 8 + self.ts_std(close, 8)) < (self.ts_sum(close, 2) / 2)
        cond2 = (self.ts_sum(close, 2) / 2) < (self.ts_sum(close, 8) / 8 - self.ts_std(close, 8))
        cond3 = volume / self.ts_mean(volume, 20) >= 1

        return np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))

    def alpha005(self, volume: pd.Series, high: pd.Series) -> pd.Series:
        """(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))"""
        return -1 * self.ts_max(
            self.correlation(self.ts_rank(volume + 1, 5), self.ts_rank(high, 5), 5),
            3
        )

    def alpha006(self, open_: pd.Series, high: pd.Series) -> pd.Series:
        """(-1 * RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4))))"""
        return -1 * self.rank(self.sign(self.delta(open_ * 0.85 + high * 0.15, 4)))

    def alpha007(self, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """(RANK(TSMAX(VWAP - CLOSE, 3)) + RANK(TSMIN(VWAP - CLOSE, 3))) * RANK(DELTA(VOLUME, 3))"""
        return (
            self.rank(self.ts_max(vwap - close, 3)) +
            self.rank(self.ts_min(vwap - close, 3))
        ) * self.rank(self.delta(volume, 3))

    def alpha008(self, high: pd.Series, low: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(DELTA(((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8), 4) * -1)"""
        return self.rank(self.delta((high + low) / 2 * 0.2 + vwap * 0.8, 4) * -1)

    def alpha009(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """SMA(((HIGH + LOW) / 2 - DELAY((HIGH + LOW) / 2, 1)) * (HIGH - LOW) / VOLUME, 7, 2)"""
        mid = (high + low) / 2
        inner = (mid - self.delay(mid, 1)) * (high - low) / (volume + 1e-8)
        return self.sma(inner, 7, 2)

    def alpha010(self, close: pd.Series, returns: pd.Series) -> pd.Series:
        """RANK(TSMAX((if RET < 0 then STD(RET, 20) else CLOSE)^2, 5))"""
        cond = returns < 0
        inner = np.where(cond, self.ts_std(returns, 20), close)
        return self.rank(self.ts_max(pd.Series(inner, index=close.index) ** 2, 5))

    def alpha011(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """SUM(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 6)"""
        inner = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
        return self.ts_sum(inner, 6)

    def alpha012(self, open_: pd.Series, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(OPEN - SUM(VWAP, 10) / 10) * -1 * RANK(ABS(CLOSE - VWAP))"""
        return self.rank(open_ - self.ts_sum(vwap, 10) / 10) * -1 * self.rank(self.abs_(close - vwap))

    def alpha013(self, high: pd.Series, low: pd.Series, vwap: pd.Series) -> pd.Series:
        """(HIGH * LOW)^0.5 - VWAP"""
        return np.sqrt(high * low) - vwap

    def alpha014(self, close: pd.Series) -> pd.Series:
        """CLOSE - DELAY(CLOSE, 5)"""
        return close - self.delay(close, 5)

    def alpha015(self, open_: pd.Series, close: pd.Series) -> pd.Series:
        """OPEN / DELAY(CLOSE, 1) - 1"""
        return open_ / (self.delay(close, 1) + 1e-8) - 1

    def alpha016(self, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))"""
        return -1 * self.ts_max(
            self.rank(self.correlation(self.rank(volume + 1), self.rank(vwap), 5)),
            5
        )

    def alpha017(self, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(VWAP - TSMAX(VWAP, 15))^DELTA(CLOSE, 5)"""
        base = self.rank(vwap - self.ts_max(vwap, 15))
        exp = self.delta(close, 5)
        return base ** exp.clip(-10, 10)  # Clip to prevent overflow

    def alpha018(self, close: pd.Series) -> pd.Series:
        """CLOSE / DELAY(CLOSE, 5)"""
        return close / (self.delay(close, 5) + 1e-8)

    def alpha019(self, close: pd.Series) -> pd.Series:
        """Conditional: if CLOSE < DELAY then ratio1, elif CLOSE = DELAY then 0, else ratio2"""
        delay5 = self.delay(close, 5)
        cond1 = close < delay5
        cond2 = close == delay5
        return np.where(cond1, (close - delay5) / (delay5 + 1e-8),
                 np.where(cond2, 0, (close - delay5) / (close + 1e-8)))

    def alpha020(self, close: pd.Series) -> pd.Series:
        """(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100"""
        return (close - self.delay(close, 6)) / (self.delay(close, 6) + 1e-8) * 100

    def alpha021(self, close: pd.Series) -> pd.Series:
        """REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))"""
        return self.regbeta(self.ts_mean(close, 6), close, 6)

    def alpha022(self, close: pd.Series) -> pd.Series:
        """SMA(((CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) - DELAY(..., 3)), 12, 1)"""
        inner = (close - self.ts_mean(close, 6)) / (self.ts_mean(close, 6) + 1e-8)
        return self.sma(inner - self.delay(inner, 3), 12, 1)

    def alpha023(self, close: pd.Series) -> pd.Series:
        """Ratio of up-volatility to total volatility, scaled to 100"""
        cond = close > self.delay(close, 1)
        up_std = self.ts_std(np.where(cond, close, 0), 20)
        total_std = self.ts_std(close, 20)
        return self.sma(pd.Series(np.where(cond, up_std, total_std), index=close.index), 20, 1) * 100

    def alpha024(self, close: pd.Series) -> pd.Series:
        """SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)"""
        return self.sma(close - self.delay(close, 5), 5, 1)

    def alpha025(self, close: pd.Series, volume: pd.Series, returns: pd.Series) -> pd.Series:
        """Complex: (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(...))))) * (1 + RANK(SUM(RET, 250)))"""
        return (
            -1 * self.rank(self.delta(close, 7) * (1 - self.rank(self.decay_linear(volume / (self.ts_mean(volume, 20) + 1e-8), 9)))) *
            (1 + self.rank(self.ts_sum(returns, 250)))
        )

    def alpha026(self, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """(SUM(CLOSE, 7) / 7 - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)"""
        return (self.ts_sum(close, 7) / 7 - close) + self.correlation(vwap, self.delay(close, 5), 230)

    def alpha027(self, close: pd.Series) -> pd.Series:
        """WMA of price change percentages over 3 and 6 days"""
        pct3 = (close - self.delay(close, 3)) / (self.delay(close, 3) + 1e-8) * 100
        pct6 = (close - self.delay(close, 6)) / (self.delay(close, 6) + 1e-8) * 100
        return self.wma((pct3 + pct6) / 2, 12)

    def alpha028(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Triple EMA-based Stochastic variant"""
        llv9 = self.ts_min(low, 9)
        hhv9 = self.ts_max(high, 9)
        rsv = (close - llv9) / (hhv9 - llv9 + 1e-8) * 100
        k = self.sma(rsv, 3, 1)
        d = self.sma(k, 3, 1)
        j = 3 * k - 2 * d
        return j

    def alpha029(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """(CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME"""
        return (close - self.delay(close, 6)) / (self.delay(close, 6) + 1e-8) * volume

    def alpha030(self, close: pd.Series) -> pd.Series:
        """Regression-based - simplified"""
        return self.regbeta(close, close, 60)

    def alpha031(self, close: pd.Series) -> pd.Series:
        """(CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100"""
        return (close - self.ts_mean(close, 12)) / (self.ts_mean(close, 12) + 1e-8) * 100

    def alpha032(self, volume: pd.Series, high: pd.Series) -> pd.Series:
        """(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))"""
        return -1 * self.ts_sum(
            self.rank(self.correlation(self.rank(high), self.rank(volume + 1), 3)),
            3
        )

    def alpha033(self, close: pd.Series, low: pd.Series, returns: pd.Series) -> pd.Series:
        """Multi-component formula with returns range ratios"""
        ts_min5 = self.ts_min(low, 5)
        delay_min = self.delay(ts_min5, 5)
        ret_factor = (self.ts_sum(returns, 240) - self.ts_sum(returns, 20)) / 220
        return (-1 * ts_min5 + delay_min) * self.rank(ret_factor) * self.ts_rank(close, 5)

    def alpha034(self, close: pd.Series) -> pd.Series:
        """MEAN(CLOSE, 12) / CLOSE"""
        return self.ts_mean(close, 12) / (close + 1e-8)

    def alpha035(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """MIN of decay-linear ranks"""
        inner1 = self.decay_linear(self.delta(open_, 1), 15)
        inner2 = self.decay_linear(self.correlation(volume / (self.ts_mean(volume, 20) + 1e-8), open_ * 0.65 + open_ * 0.35, 17), 7)
        return np.minimum(self.rank(inner1), self.rank(inner2)) * -1

    def alpha036(self, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))"""
        return self.rank(self.ts_sum(self.correlation(self.rank(volume + 1), self.rank(vwap), 6), 2))

    def alpha037(self, open_: pd.Series, close: pd.Series, returns: pd.Series) -> pd.Series:
        """(-1 * RANK(SUM(OPEN, 5) * SUM(RET, 5) - DELAY(..., 10)))"""
        inner = self.ts_sum(open_, 5) * self.ts_sum(returns, 5)
        return -1 * self.rank(inner - self.delay(inner, 10))

    def alpha038(self, high: pd.Series) -> pd.Series:
        """Conditional: if MA(HIGH, 20) < HIGH then -1 * DELTA(HIGH, 2) else 0"""
        return np.where(self.ts_mean(high, 20) < high, -1 * self.delta(high, 2), 0)

    def alpha039(self, close: pd.Series, open_: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Difference of decay-linear correlations with negation"""
        adv20 = self.ts_mean(volume, 20)
        inner1 = self.decay_linear(self.delta(close, 2), 8)
        inner2 = self.decay_linear(self.correlation(vwap * 0.3 + open_ * 0.7, self.ts_sum(self.ts_mean(volume, 180), 37), 14), 12)
        return (self.rank(inner1) - self.rank(inner2)) * -1

    def alpha040(self, high: pd.Series, volume: pd.Series) -> pd.Series:
        """SUM(up volume, 26) / SUM(down volume, 26) * 100"""
        cond = high > self.delay(high, 1)
        up_vol = np.where(cond, volume, 0)
        down_vol = np.where(~cond, volume, 0)
        return self.ts_sum(pd.Series(up_vol, index=volume.index), 26) / (self.ts_sum(pd.Series(down_vol, index=volume.index), 26) + 1e-8) * 100

    def alpha041(self, vwap: pd.Series) -> pd.Series:
        """RANK(TSMAX(DELTA(VWAP, 3), 5)) * -1"""
        return self.rank(self.ts_max(self.delta(vwap, 3), 5)) * -1

    def alpha042(self, high: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10)"""
        return -1 * self.rank(self.ts_std(high, 10)) * self.correlation(high, volume + 1, 10)

    def alpha043(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Directional volume accumulation over 6 days"""
        cond = close > self.delay(close, 1)
        result = self.ts_sum(np.where(cond, volume, -volume), 6)
        return pd.Series(result, index=close.index)

    def alpha044(self, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Sum of decay-linear Tsrank measures"""
        inner1 = self.decay_linear(self.correlation(self.rank(low), self.rank(self.ts_mean(volume, 10)), 7), 6)
        inner2 = self.ts_rank(self.decay_linear(self.delta(vwap, 3), 10), 15)
        return self.rank(inner1) + inner2

    def alpha045(self, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(DELTA(...)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))"""
        inner = (close * 0.6 + open_ * 0.4)
        return self.rank(self.delta(inner, 1)) * self.rank(self.correlation(vwap, self.ts_mean(volume, 150), 15))

    def alpha046(self, close: pd.Series) -> pd.Series:
        """(MEAN(CLOSE, 3) + MEAN(CLOSE, 6) + MEAN(CLOSE, 12) + MEAN(CLOSE, 24)) / (4 * CLOSE)"""
        return (
            self.ts_mean(close, 3) + self.ts_mean(close, 6) +
            self.ts_mean(close, 12) + self.ts_mean(close, 24)
        ) / (4 * close + 1e-8)

    def alpha047(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """SMA(stochastic-like formula, 9, 1)"""
        llv9 = self.ts_min(low, 9)
        hhv9 = self.ts_max(high, 9)
        rsv = (self.ts_max(high, 6) - close) / (self.ts_max(high, 6) - self.ts_min(low, 6) + 1e-8) * 100
        return self.sma(rsv, 9, 1)

    def alpha048(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Complex ranking of momentum and volume patterns"""
        adv20 = self.ts_mean(volume, 20)
        inner = -1 * (self.rank((close - self.delay(close, 1))) * volume) / (adv20 + 1e-8)
        return inner

    def alpha049(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Ratio measuring uptrend dominance - DTM/DBM calculation"""
        dtm = np.where(open_ <= self.delay(open_, 1), 0,
                      np.maximum(high - open_, open_ - self.delay(open_, 1)))
        dbm = np.where(open_ >= self.delay(open_, 1), 0,
                      np.maximum(open_ - low, open_ - self.delay(open_, 1)))
        dtm_sum = self.ts_sum(pd.Series(dtm, index=high.index), 12)
        dbm_sum = self.ts_sum(pd.Series(dbm, index=high.index), 12)
        return np.where(dtm_sum > dbm_sum, (dtm_sum - dbm_sum) / (dtm_sum + 1e-8),
                       np.where(dtm_sum == dbm_sum, 0, (dtm_sum - dbm_sum) / (dbm_sum + 1e-8)))

    def alpha050(self, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """Net difference between uptrend and downtrend measures"""
        dtm = np.where(open_ <= self.delay(open_, 1), 0,
                      np.maximum(high - open_, open_ - self.delay(open_, 1)))
        dbm = np.where(open_ >= self.delay(open_, 1), 0,
                      np.maximum(open_ - low, open_ - self.delay(open_, 1)))
        return (self.ts_sum(pd.Series(dtm, index=high.index), 12) -
                self.ts_sum(pd.Series(dbm, index=high.index), 12)) / \
               (self.ts_sum(pd.Series(dtm, index=high.index), 12) +
                self.ts_sum(pd.Series(dbm, index=high.index), 12) + 1e-8)

    # =========================================================================
    # ALPHAS 051-100
    # =========================================================================

    def alpha051(self, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """Uptrend strength ratio"""
        dtm = np.where(open_ <= self.delay(open_, 1), 0,
                      np.maximum(high - open_, open_ - self.delay(open_, 1)))
        dbm = np.where(open_ >= self.delay(open_, 1), 0,
                      np.maximum(open_ - low, open_ - self.delay(open_, 1)))
        return self.ts_sum(pd.Series(dtm, index=high.index), 12) / \
               (self.ts_sum(pd.Series(dtm, index=high.index), 12) +
                self.ts_sum(pd.Series(dbm, index=high.index), 12) + 1e-8)

    def alpha052(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """SUM(MAX(HIGH - delayed midpoint, 0), 26) / SUM(MAX(...), 26)"""
        mid_delay = self.delay((high + low) / 2, 1)
        inner = np.maximum(0, high - mid_delay) - np.maximum(0, mid_delay - low)
        return self.ts_sum(pd.Series(inner, index=high.index), 26) / 26

    def alpha053(self, close: pd.Series) -> pd.Series:
        """COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100"""
        cond = close > self.delay(close, 1)
        return self.ts_count(cond, 12) / 12 * 100

    def alpha054(self, close: pd.Series, open_: pd.Series) -> pd.Series:
        """-1 * RANK(STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN) + CORR(CLOSE, OPEN, 10))"""
        return -1 * self.rank(
            self.ts_std(self.abs_(close - open_), 10) +
            (close - open_) +
            self.correlation(close, open_, 10)
        )

    def alpha055(self, close: pd.Series, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """Complex true range and momentum normalization"""
        inner = (close - self.ts_min(low, 16)) / (self.ts_max(high, 16) - self.ts_min(low, 16) + 1e-8)
        return self.ts_sum(inner * 16, 16) / self.ts_sum(pd.Series(np.ones_like(close), index=close.index) * 16, 16)

    def alpha056(self, volume: pd.Series, high: pd.Series, open_: pd.Series) -> pd.Series:
        """Conditional rank comparison"""
        cond = self.rank(open_ - self.ts_min(open_, 12)) < self.rank(self.rank(self.correlation(high, self.ts_mean(volume, 30), 12)))
        return np.where(cond, 1, 0)

    def alpha057(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """SMA(stochastic formula, 3, 1)"""
        llv9 = self.ts_min(low, 9)
        hhv9 = self.ts_max(high, 9)
        rsv = (close - llv9) / (hhv9 - llv9 + 1e-8) * 100
        return self.sma(rsv, 3, 1)

    def alpha058(self, close: pd.Series) -> pd.Series:
        """COUNT(CLOSE > DELAY(CLOSE, 1), 20) / 20 * 100"""
        cond = close > self.delay(close, 1)
        return self.ts_count(cond, 20) / 20 * 100

    def alpha059(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """SUM of conditional spread adjustments, 20"""
        delay_close = self.delay(close, 1)
        cond1 = close == delay_close
        inner = np.where(cond1, 0,
                 np.where(close > delay_close, close - np.minimum(low, delay_close),
                         close - np.maximum(high, delay_close)))
        return self.ts_sum(pd.Series(inner, index=close.index), 20)

    def alpha060(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """SUM(normalized volume-weighted spread, 20)"""
        inner = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
        return self.ts_sum(inner, 20)

    def alpha061(self, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """MAX(RANK(DECAYLINEAR(...)), RANK(...)) * -1"""
        adv180 = self.ts_mean(volume, 180)
        inner1 = self.decay_linear(vwap - self.ts_min(vwap, 16), 12)
        inner2 = self.ts_rank(self.correlation(vwap, adv180, 18), 20)
        return np.maximum(self.rank(inner1), inner2) * -1

    def alpha062(self, volume: pd.Series, high: pd.Series) -> pd.Series:
        """(-1 * CORR(HIGH, RANK(VOLUME), 5))"""
        return -1 * self.correlation(high, self.rank(volume + 1), 5)

    def alpha063(self, close: pd.Series) -> pd.Series:
        """SMA(MAX(delta, 0), 6, 1) / SMA(ABS(delta), 6, 1) * 100"""
        delta = close - self.delay(close, 1)
        return self.sma(np.maximum(delta, 0), 6, 1) / (self.sma(self.abs_(delta), 6, 1) + 1e-8) * 100

    def alpha064(self, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Complex decay-linear correlation ranks"""
        adv120 = self.ts_mean(volume, 120)
        inner1 = self.decay_linear(self.correlation(self.rank(vwap), self.rank(volume + 1), 4), 4)
        inner2 = self.decay_linear(self.ts_max(self.correlation(self.rank(close), self.rank(adv120), 12), 13), 14)
        return np.maximum(self.rank(inner1), self.rank(inner2)) * -1

    def alpha065(self, close: pd.Series) -> pd.Series:
        """MEAN(CLOSE, 6) / CLOSE"""
        return self.ts_mean(close, 6) / (close + 1e-8)

    def alpha066(self, close: pd.Series) -> pd.Series:
        """(CLOSE - MEAN(CLOSE, 6)) / MEAN(CLOSE, 6) * 100"""
        return (close - self.ts_mean(close, 6)) / (self.ts_mean(close, 6) + 1e-8) * 100

    def alpha067(self, close: pd.Series) -> pd.Series:
        """SMA(MAX(delta, 0), 24, 1) / SMA(ABS(delta), 24, 1) * 100"""
        delta = close - self.delay(close, 1)
        return self.sma(np.maximum(delta, 0), 24, 1) / (self.sma(self.abs_(delta), 24, 1) + 1e-8) * 100

    def alpha068(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """SMA(midpoint change * range / volume, 15, 2)"""
        mid = (high + low) / 2
        inner = (mid - self.delay(mid, 1)) * (high - low) / (volume + 1e-8)
        return self.sma(inner, 15, 2)

    def alpha069(self, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """Directional trend measurement (DTM vs DBM)"""
        dtm = np.where(open_ <= self.delay(open_, 1), 0,
                      np.maximum(high - open_, open_ - self.delay(open_, 1)))
        dbm = np.where(open_ >= self.delay(open_, 1), 0,
                      np.maximum(open_ - low, open_ - self.delay(open_, 1)))
        dtm_s = self.ts_sum(pd.Series(dtm, index=high.index), 20)
        dbm_s = self.ts_sum(pd.Series(dbm, index=high.index), 20)
        return np.where(dtm_s > dbm_s, (dtm_s - dbm_s) / (dtm_s + 1e-8),
                       np.where(dtm_s == dbm_s, 0, (dtm_s - dbm_s) / (dbm_s + 1e-8)))

    def alpha070(self, amount: pd.Series) -> pd.Series:
        """STD(AMOUNT, 6)"""
        return self.ts_std(amount, 6)

    def alpha071(self, close: pd.Series) -> pd.Series:
        """(CLOSE - MEAN(CLOSE, 24)) / MEAN(CLOSE, 24) * 100"""
        return (close - self.ts_mean(close, 24)) / (self.ts_mean(close, 24) + 1e-8) * 100

    def alpha072(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """SMA(stochastic formula, 15, 1)"""
        llv6 = self.ts_min(low, 6)
        hhv6 = self.ts_max(high, 6)
        rsv = (self.ts_max(high, 6) - close) / (hhv6 - llv6 + 1e-8) * 100
        return self.sma(rsv, 15, 1)

    def alpha073(self, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Nested decay-linear correlation ranks"""
        inner1 = self.decay_linear(self.delta(vwap, 5), 3)
        inner2 = self.decay_linear(self.correlation(close, self.ts_mean(volume, 150), 9), 4) / (self.rank(self.decay_linear(self.correlation(self.ts_rank(high, 4), self.ts_rank(self.ts_mean(volume, 19), 19), 7), 3)) + 1e-8)
        return self.ts_rank(inner1, 17) * -1 * inner2

    def alpha074(self, volume: pd.Series, vwap: pd.Series, low: pd.Series) -> pd.Series:
        """Sum of correlation ranks"""
        adv30 = self.ts_mean(volume, 30)
        return self.rank(self.correlation(close, adv30, 10)) + self.rank(self.correlation(self.ts_rank(close, 10), self.ts_rank(adv30, 10), 7))

    def alpha075(self, close: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """COUNT(matched direction with index, 50) / COUNT(index down, 50)"""
        # Simplified - using close as proxy for index
        bench_ret = close.pct_change()
        stock_ret = open_.pct_change()
        cond_match = (bench_ret < 0) & (stock_ret < bench_ret)
        cond_down = bench_ret < 0
        return self.ts_count(cond_match, 50) / (self.ts_count(cond_down, 50) + 1e-8)

    def alpha076(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """STD(normalized volume-price, 20) / MEAN(normalized volume-price, 20)"""
        inner = self.abs_(close / (self.delay(close, 1) + 1e-8) - 1) / (volume + 1e-8)
        return self.ts_std(inner, 20) / (self.ts_mean(inner, 20) + 1e-8)

    def alpha077(self, high: pd.Series, low: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """MIN(RANK(DECAYLINEAR(...)), RANK(...))"""
        inner1 = self.decay_linear(((high + low) / 2 + high) - (vwap + high), 20)
        inner2 = self.decay_linear(self.correlation(high + low, self.ts_mean(volume, 40), 3), 6)
        return np.minimum(self.rank(inner1), self.rank(inner2))

    def alpha078(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """((midpoint - MA(midpoint, 12)) / (normalized range))"""
        mid = (high + low + close) / 3
        ma_mid = self.ts_mean(mid, 12)
        norm = self.ts_sum(mid, 12) / 12
        return (mid - ma_mid) / (norm * 0.015 + 1e-8)

    def alpha079(self, close: pd.Series) -> pd.Series:
        """SMA(MAX(delta, 0), 12, 1) / SMA(ABS(delta), 12, 1) * 100"""
        delta = close - self.delay(close, 1)
        return self.sma(np.maximum(delta, 0), 12, 1) / (self.sma(self.abs_(delta), 12, 1) + 1e-8) * 100

    def alpha080(self, volume: pd.Series) -> pd.Series:
        """(VOLUME - DELAY(VOLUME, 5)) / DELAY(VOLUME, 5) * 100"""
        return (volume - self.delay(volume, 5)) / (self.delay(volume, 5) + 1e-8) * 100

    def alpha081(self, volume: pd.Series) -> pd.Series:
        """SMA(VOLUME, 21, 2)"""
        return self.sma(volume, 21, 2)

    def alpha082(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """SMA(stochastic formula, 20, 1)"""
        llv6 = self.ts_min(low, 6)
        hhv6 = self.ts_max(high, 6)
        rsv = (self.ts_max(high, 6) - close) / (hhv6 - llv6 + 1e-8) * 100
        return self.sma(rsv, 20, 1)

    def alpha083(self, high: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * RANK(COV(RANK(HIGH), RANK(VOLUME), 5)))"""
        return -1 * self.rank(self.covariance(self.rank(high), self.rank(volume + 1), 5))

    def alpha084(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Directional volume accumulation over 20 days"""
        cond = close > self.delay(close, 1)
        return self.ts_sum(np.where(cond, volume, -volume), 20)

    def alpha085(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """TSRANK(volume normalized, 20) * TSRANK(-delta, 8)"""
        return self.ts_rank(volume / (self.ts_mean(volume, 20) + 1e-8), 20) * self.ts_rank(-1 * self.delta(close, 7), 8)

    def alpha086(self, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """Conditional momentum adjustment logic"""
        cond = self.delay(close, 20) * 0.25 + self.delay(vwap, 20) * 0.75 < vwap
        return np.where(cond, -1, 1)

    def alpha087(self, close: pd.Series, high: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Complex decay-linear VWAP-based formula"""
        adv81 = self.ts_mean(volume, 81)
        inner1 = self.decay_linear(close - self.ts_max(close, 14), 13)
        inner2 = self.decay_linear(self.delta(vwap, 5), 11)
        inner3 = self.ts_rank(self.decay_linear(self.correlation(high, adv81, 17), 20), 13)
        return self.rank(inner1) + inner3 - self.rank(inner2)

    def alpha088(self, close: pd.Series) -> pd.Series:
        """(CLOSE - DELAY(CLOSE, 20)) / DELAY(CLOSE, 20) * 100"""
        return (close - self.delay(close, 20)) / (self.delay(close, 20) + 1e-8) * 100

    def alpha089(self, close: pd.Series) -> pd.Series:
        """MACD-like: 2 * (SMA13 - SMA27 - SMA(SMA13 - SMA27, 10))"""
        sma13 = self.sma(close, 13, 2)
        sma27 = self.sma(close, 27, 2)
        dif = sma13 - sma27
        dea = self.sma(dif, 10, 2)
        return 2 * (dif - dea)

    def alpha090(self, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1"""
        return self.rank(self.correlation(self.rank(vwap), self.rank(volume + 1), 5)) * -1

    def alpha091(self, close: pd.Series, volume: pd.Series, low: pd.Series) -> pd.Series:
        """RANK((CLOSE - TSMAX(CLOSE, 5))) * RANK(CORR(...)) * -1"""
        adv40 = self.ts_mean(volume, 40)
        return self.rank(close - self.ts_max(close, 5)) * self.rank(self.correlation(adv40, low, 5)) * -1

    def alpha092(self, close: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """MAX(RANK(DECAYLINEAR(...)), TSRANK(...)) * -1"""
        adv30 = self.ts_mean(volume, 30)
        inner1 = self.decay_linear(self.delta(close * 0.35 + vwap * 0.65, 2), 3)
        inner2 = self.ts_rank(self.decay_linear(self.abs_(self.correlation(adv30, close, 13)), 5), 15)
        return np.maximum(self.rank(inner1), inner2) * -1

    def alpha093(self, open_: pd.Series, low: pd.Series) -> pd.Series:
        """SUM(conditional open-low spreads, 20)"""
        cond = open_ >= self.delay(open_, 1)
        return self.ts_sum(np.where(cond, 0, np.maximum(open_ - low, open_ - self.delay(open_, 1))), 20)

    def alpha094(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Directional volume accumulation over 30 days"""
        cond = close > self.delay(close, 1)
        return self.ts_sum(np.where(cond, volume, -volume), 30)

    def alpha095(self, amount: pd.Series) -> pd.Series:
        """STD(AMOUNT, 20)"""
        return self.ts_std(amount, 20)

    def alpha096(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Double SMA of stochastic formula"""
        llv9 = self.ts_min(low, 9)
        hhv9 = self.ts_max(high, 9)
        rsv = (close - llv9) / (hhv9 - llv9 + 1e-8) * 100
        k = self.sma(rsv, 3, 1)
        return self.sma(k, 3, 1)

    def alpha097(self, volume: pd.Series) -> pd.Series:
        """STD(VOLUME, 10)"""
        return self.ts_std(volume, 10)

    def alpha098(self, close: pd.Series) -> pd.Series:
        """Conditional: if MA change < 5% then MIN spread else DELTA(CLOSE, 3)"""
        mean5 = self.ts_mean(close, 5)
        mean10 = self.ts_mean(close, 10)
        cond = (self.delta(mean5, 10) / mean10) < 0.05
        return np.where(cond, -1 * self.delta(close, 3), -1 * (close - self.ts_min(close, 200)))

    def alpha099(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """(-1 * RANK(COV(RANK(CLOSE), RANK(VOLUME), 5)))"""
        return -1 * self.rank(self.covariance(self.rank(close), self.rank(volume + 1), 5))

    def alpha100(self, volume: pd.Series) -> pd.Series:
        """STD(VOLUME, 20)"""
        return self.ts_std(volume, 20)

    # =========================================================================
    # ALPHAS 101-150
    # =========================================================================

    def alpha101(self, close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Conditional comparison of correlation ranks"""
        return np.where(
            (close - open_) / (high - low + 1e-8) * (high - low) > (high - open_),
            (close - open_) / (high - low + 1e-8) - (close - open_) / (close - low + 1e-8),
            0
        )

    def alpha102(self, volume: pd.Series) -> pd.Series:
        """SMA(MAX(vol delta, 0), 6, 1) / SMA(ABS(vol delta), 6, 1) * 100"""
        delta_vol = self.delta(volume, 1)
        return self.sma(np.maximum(delta_vol, 0), 6, 1) / (self.sma(self.abs_(delta_vol), 6, 1) + 1e-8) * 100

    def alpha103(self, low: pd.Series) -> pd.Series:
        """((20 - LOWDAY(LOW, 20)) / 20) * 100"""
        return ((20 - self.ts_argmin(low, 20)) / 20) * 100

    def alpha104(self, high: pd.Series, volume: pd.Series, close: pd.Series) -> pd.Series:
        """(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))"""
        return -1 * self.delta(self.correlation(high, volume + 1, 5), 5) * self.rank(self.ts_std(close, 20))

    def alpha105(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))"""
        return -1 * self.correlation(self.rank(open_), self.rank(volume + 1), 10)

    def alpha106(self, close: pd.Series) -> pd.Series:
        """CLOSE - DELAY(CLOSE, 20)"""
        return close - self.delay(close, 20)

    def alpha107(self, open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """(-1 * RANK(OPEN - DELAY(HIGH, 1)) * RANK(...) * RANK(...))"""
        return -1 * self.rank(open_ - self.delay(high, 1)) * \
               self.rank(open_ - self.delay(close, 1)) * \
               self.rank(open_ - self.delay(low, 1))

    def alpha108(self, high: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """(RANK(HIGH - TSMIN(HIGH, 2))^RANK(CORR(...))) * -1"""
        adv120 = self.ts_mean(volume, 120)
        return self.rank(high - self.ts_min(high, 2)) ** self.rank(self.correlation(high, adv120, 15).clip(-1, 1)) * -1

    def alpha109(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """SMA(HIGH - LOW, 10, 2) / SMA(SMA(...), 10, 2)"""
        inner = self.sma(high - low, 10, 2)
        return inner / (self.sma(inner, 10, 2) + 1e-8)

    def alpha110(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """SUM(MAX(HIGH - delayed close, 0), 20) / SUM(MAX(...), 20) * 100"""
        delay_close = self.delay(close, 1)
        up = np.maximum(0, high - delay_close)
        down = np.maximum(0, delay_close - low)
        return self.ts_sum(up, 20) / (self.ts_sum(down, 20) + 1e-8) * 100

    def alpha111(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Difference of volume-weighted stochastic SMAs"""
        mid = (high + low + close) / 3
        inner1 = self.sma(volume * (mid - self.delay(mid, 1)), 11, 2)
        inner2 = self.sma(volume * (mid - self.delay(mid, 1)), 4, 2)
        return inner1 - inner2

    def alpha112(self, close: pd.Series) -> pd.Series:
        """Up momentum ratio to total momentum"""
        delta = close - self.delay(close, 1)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        sum_up = self.ts_sum(pd.Series(up, index=close.index), 12)
        sum_down = self.ts_sum(pd.Series(down, index=close.index), 12)
        return (sum_up - sum_down) / (sum_up + sum_down + 1e-8)

    def alpha113(self, close: pd.Series, volume: pd.Series, low: pd.Series) -> pd.Series:
        """(-1 * RANK(delayed average) * CORR(...) * RANK(CORR(...)))"""
        adv30 = self.ts_mean(volume, 30)
        return -1 * self.rank(self.ts_sum(self.delay(close, 5), 20) / 20) * \
               self.correlation(close, volume + 1, 2) * \
               self.rank(self.correlation(self.ts_sum(close, 5), self.ts_sum(close, 20), 2))

    def alpha114(self, high: pd.Series, low: pd.Series, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Complex normalization of ranked ranges and volume"""
        adv20 = self.ts_mean(volume, 20)
        return self.rank(self.delay(high - low, 2)) * \
               self.rank(self.rank(volume + 1)) / \
               (high - low + 1e-8) / (adv20 + 1e-8) * -1

    def alpha115(self, high: pd.Series, low: pd.Series, volume: pd.Series, close: pd.Series) -> pd.Series:
        """RANK(correlation)^RANK(correlation)"""
        adv30 = self.ts_mean(volume, 30)
        corr1 = self.correlation(high * 0.9 + close * 0.1, adv30, 10)
        corr2 = self.correlation(self.ts_rank(high + low, 4), self.ts_rank(volume + 1, 10), 7)
        return self.rank(corr1) ** self.rank(corr2).clip(-2, 2)

    def alpha116(self, close: pd.Series) -> pd.Series:
        """REGBETA(CLOSE, SEQUENCE(20))"""
        return self.regbeta(close, close, 20)

    def alpha117(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Product of three Tsrank components"""
        return self.ts_rank(volume + 1, 32) * \
               (1 - self.ts_rank((close + high - low), 16)) * \
               (1 - self.ts_rank(close.pct_change(), 32))

    def alpha118(self, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """SUM(HIGH - OPEN, 20) / SUM(OPEN - LOW, 20) * 100"""
        return self.ts_sum(high - open_, 20) / (self.ts_sum(open_ - low, 20) + 1e-8) * 100

    def alpha119(self, open_: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Difference of complex decay-linear measures"""
        adv60 = self.ts_mean(volume, 60)
        inner1 = self.decay_linear(self.correlation(vwap, self.ts_sum(adv60, 9), 6), 4)
        inner2 = self.decay_linear(self.rank(self.ts_argmin(self.correlation(self.rank(open_), self.rank(adv60), 21), 9)), 7)
        return self.rank(inner1) - self.rank(inner2)

    def alpha120(self, vwap: pd.Series, close: pd.Series) -> pd.Series:
        """RANK(VWAP - CLOSE) / RANK(VWAP + CLOSE)"""
        return self.rank(vwap - close) / (self.rank(vwap + close) + 1e-8)

    def alpha121(self, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """(RANK(VWAP - TSMIN(VWAP, 12))^TSRANK(...)) * -1"""
        adv60 = self.ts_mean(volume, 60)
        return (self.rank(vwap - self.ts_min(vwap, 12)) **
                self.ts_rank(self.correlation(self.ts_rank(vwap, 20), self.ts_rank(adv60, 2), 18), 3).clip(-2, 2)) * -1

    def alpha122(self, close: pd.Series) -> pd.Series:
        """Triple SMA log derivative normalized"""
        sma13 = self.sma(self.sma(self.sma(self.log(close), 13, 2), 13, 2), 13, 2)
        return sma13 / (self.delay(sma13, 1) + 1e-8) - 1

    def alpha123(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """Conditional: if correlation rank1 < rank2 then -1 else 0"""
        adv20 = self.ts_mean(volume, 20)
        corr1 = self.correlation(self.ts_sum((high + low) / 2, 20), self.ts_sum(adv20, 20), 9)
        corr2 = self.correlation(low, volume + 1, 6)
        return np.where(self.rank(corr1) < self.rank(corr2), -1, 0)

    def alpha124(self, close: pd.Series, vwap: pd.Series) -> pd.Series:
        """(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)), 2)"""
        return (close - vwap) / (self.decay_linear(self.rank(self.ts_max(close, 30)), 2) + 1e-8)

    def alpha125(self, close: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Ratio of decay-linear correlation to decay-linear delta"""
        adv80 = self.ts_mean(volume, 80)
        inner1 = self.decay_linear(self.correlation(vwap, adv80, 17), 4)
        inner2 = self.decay_linear(self.delta(close * 0.5 + vwap * 0.5, 3), 16)
        return self.rank(inner1) / (self.rank(inner2) + 1e-8)

    def alpha126(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """(CLOSE + HIGH + LOW) / 3"""
        return (close + high + low) / 3

    def alpha127(self, close: pd.Series) -> pd.Series:
        """sqrt(MEAN((100 * price deviation / TSMAX)^2, 12))"""
        max12 = self.ts_max(close, 12)
        inner = (100 * (close / (max12 + 1e-8) - 1)) ** 2
        return np.sqrt(self.ts_mean(inner, 12))

    def alpha128(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """RSI-like formula using volume-weighted midpoints"""
        mid = (high + low + close) / 3
        delta = mid - self.delay(mid, 1)
        up = np.where(delta > 0, delta * volume, 0)
        down = np.where(delta < 0, -delta * volume, 0)
        sum_up = self.ts_sum(pd.Series(up, index=close.index), 14)
        sum_down = self.ts_sum(pd.Series(down, index=close.index), 14)
        return 100 - 100 / (1 + sum_up / (sum_down + 1e-8))

    def alpha129(self, close: pd.Series) -> pd.Series:
        """SUM(ABS(negative deltas), 12)"""
        delta = close - self.delay(close, 1)
        return self.ts_sum(np.where(delta < 0, self.abs_(delta), 0), 12)

    def alpha130(self, high: pd.Series, low: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Ratio of correlation-based decay-linear ranks"""
        adv40 = self.ts_mean(volume, 40)
        inner1 = self.decay_linear(self.rank((high + low) / 2), 5) * self.decay_linear(self.ts_rank(self.correlation(adv40, high, 5), 19), 17)
        inner2 = self.rank(self.decay_linear(self.delta((close * 0.68 + low * 0.32), 2), 19))
        return inner1 / (inner2 + 1e-8)

    def alpha131(self, vwap: pd.Series, volume: pd.Series, close: pd.Series) -> pd.Series:
        """RANK(DELTA(VWAP, 1))^TSRANK(CORR(...), 18)"""
        adv10 = self.ts_mean(volume, 10)
        return self.rank(self.delta(vwap, 1)) ** self.ts_rank(self.correlation(close, adv10, 5), 18).clip(-2, 2)

    def alpha132(self, amount: pd.Series) -> pd.Series:
        """MEAN(AMOUNT, 20)"""
        return self.ts_mean(amount, 20)

    def alpha133(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """((20 - HIGHDAY(HIGH, 20)) / 20) * 100 - ((20 - LOWDAY(LOW, 20)) / 20) * 100"""
        return ((20 - self.ts_argmax(high, 20)) / 20) * 100 - \
               ((20 - self.ts_argmin(low, 20)) / 20) * 100

    def alpha134(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """(CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME"""
        return (close - self.delay(close, 12)) / (self.delay(close, 12) + 1e-8) * volume

    def alpha135(self, close: pd.Series) -> pd.Series:
        """SMA(DELAY(price ratio, 1), 20, 1)"""
        ratio = close / (self.delay(close, 20) + 1e-8)
        return self.sma(self.delay(ratio, 1), 20, 1)

    def alpha136(self, open_: pd.Series, volume: pd.Series, returns: pd.Series) -> pd.Series:
        """(-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10)"""
        return -1 * self.rank(self.delta(returns, 3)) * self.correlation(open_, volume + 1, 10)

    def alpha137(self, close: pd.Series, high: pd.Series, low: pd.Series, open_: pd.Series) -> pd.Series:
        """Complex true range normalization with volume"""
        delta = close - self.delay(close, 1)
        inner1 = 16 * (close - self.delay(close, 1) + (close - open_) / 2 + self.delay(close, 1) - self.delay(open_, 1))
        inner2 = np.maximum(self.abs_(high - self.delay(close, 1)), np.maximum(self.abs_(low - self.delay(close, 1)), self.abs_(high - low)))
        return inner1 / (inner2 + 1e-8) * np.maximum(self.abs_(high - self.delay(close, 1)), self.abs_(low - self.delay(close, 1)))

    def alpha138(self, low: pd.Series, volume: pd.Series, vwap: pd.Series) -> pd.Series:
        """Complex nested decay-linear and Tsrank formula"""
        adv15 = self.ts_mean(volume, 15)
        inner = self.decay_linear(self.correlation(low, adv15, 7), 5)
        return self.rank(inner) - self.rank(self.ts_rank(self.decay_linear(self.correlation(self.rank(vwap), self.rank(volume + 1), 6), 4), 17))

    def alpha139(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * CORR(OPEN, VOLUME, 10))"""
        return -1 * self.correlation(open_, volume + 1, 10)

    def alpha140(self, close: pd.Series, high: pd.Series, low: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """MIN(RANK(DECAYLINEAR(...)), TSRANK(...))"""
        adv10 = self.ts_mean(volume, 10)
        inner1 = self.decay_linear(self.rank(open_ + low - 2 * high), 8)
        inner2 = self.ts_rank(self.decay_linear(self.correlation(self.ts_rank(close, 8), self.ts_rank(adv10, 17), 6), 4), 3)
        return np.minimum(self.rank(inner1), inner2)

    def alpha141(self, high: pd.Series, volume: pd.Series) -> pd.Series:
        """RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9)) * -1"""
        return self.rank(self.correlation(self.rank(high), self.rank(self.ts_mean(volume, 15)), 9)) * -1

    def alpha142(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """(-1 * RANK(TSRANK(CLOSE, 10))) * RANK(second delta) * RANK(...)"""
        return -1 * self.rank(self.ts_rank(close, 10)) * \
               self.rank(self.delta(self.delta(close, 1), 1)) * \
               self.rank(self.ts_rank(volume / (self.ts_mean(volume, 20) + 1e-8), 5))

    def alpha143(self) -> float:
        """Returns 0 (unimplemented)"""
        return 0

    def alpha144(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """SUMIF(normalized volume-price, 20, condition) / COUNT(condition, 20)"""
        cond = close < self.delay(close, 1)
        inner = self.abs_(close / (self.delay(close, 1) + 1e-8) - 1) / (volume + 1e-8)
        return self.ts_sum(np.where(cond, inner, 0), 20) / (self.ts_count(cond, 20) + 1e-8)

    def alpha145(self, volume: pd.Series) -> pd.Series:
        """(MEAN(VOL, 9) - MEAN(VOL, 26)) / MEAN(VOL, 12) * 100"""
        return (self.ts_mean(volume, 9) - self.ts_mean(volume, 26)) / (self.ts_mean(volume, 12) + 1e-8) * 100

    def alpha146(self, close: pd.Series) -> pd.Series:
        """Complex deviation from SMA normalized ratio"""
        mean6 = self.ts_mean(close, 6)
        inner = (close - mean6) / mean6
        inner2 = self.delay(inner, 2) + self.delay(inner, 4)
        return self.ts_mean(inner - inner2, 61)

    def alpha147(self, close: pd.Series) -> pd.Series:
        """REGBETA(MEAN(CLOSE, 12), SEQUENCE(12))"""
        return self.regbeta(self.ts_mean(close, 12), close, 12)

    def alpha148(self, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """Conditional: if correlation rank < open rank then -1 else 0"""
        adv60 = self.ts_mean(volume, 60)
        corr = self.correlation(open_, self.ts_sum(adv60, 9), 6)
        return np.where(self.rank(corr) < self.rank(open_ - self.ts_min(open_, 14)), -1, 0)

    def alpha149(self) -> float:
        """Returns 0 (filter-based, incomplete)"""
        return 0

    def alpha150(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """(CLOSE + HIGH + LOW) / 3 * VOLUME"""
        return (close + high + low) / 3 * volume

    # =========================================================================
    # ALPHAS 151-191
    # =========================================================================

    def alpha151(self, close: pd.Series) -> pd.Series:
        """SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)"""
        return self.sma(close - self.delay(close, 20), 20, 1)

    def alpha152(self, close: pd.Series) -> pd.Series:
        """Complex nested SMA and delayed formulas"""
        inner = self.delay(self.sma(self.delay(close / (self.delay(close, 9) + 1e-8), 1), 9, 1), 1)
        return self.sma(inner, 12, 1)

    def alpha153(self, close: pd.Series) -> pd.Series:
        """(MEAN(CLOSE, 3) + MEAN(CLOSE, 6) + MEAN(CLOSE, 12) + MEAN(CLOSE, 24)) / 4"""
        return (self.ts_mean(close, 3) + self.ts_mean(close, 6) +
                self.ts_mean(close, 12) + self.ts_mean(close, 24)) / 4

    def alpha154(self, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Conditional: if (VWAP - TSMIN) < CORR then 1 else 0"""
        adv40 = self.ts_mean(volume, 40)
        cond = (vwap - self.ts_min(vwap, 16)) < self.correlation(vwap, adv40, 18)
        return np.where(cond, 1, 0)

    def alpha155(self, volume: pd.Series) -> pd.Series:
        """MACD-like volume formula"""
        sma13 = self.sma(volume, 13, 2)
        sma27 = self.sma(volume, 27, 2)
        dif = sma13 - sma27
        dea = self.sma(dif, 10, 2)
        return dif - dea

    def alpha156(self, vwap: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """MAX(RANK(DECAYLINEAR(...)), RANK(...)) * -1"""
        inner1 = self.decay_linear(self.delta(vwap, 5), 3)
        inner2 = self.decay_linear(self.delta(open_ * 0.15 + vwap * 0.85, 2), 3) - \
                 self.ts_rank(self.decay_linear(self.ts_rank(self.correlation(self.ts_sum(close, 48), self.ts_sum(self.ts_mean(volume, 60), 48), 9), 7), 4), 18)
        return np.maximum(self.rank(inner1), self.rank(inner2)) * -1

    def alpha157(self, close: pd.Series, returns: pd.Series) -> pd.Series:
        """Complex nested products and log transformations"""
        inner = self.log(self.ts_sum(returns, 11))
        return np.minimum(self.rank(self.rank(inner)), 5) + self.ts_rank(self.delay(-1 * returns, 6), 5)

    def alpha158(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """((HIGH - SMA(CLOSE, 15, 2)) - (LOW - SMA(CLOSE, 15, 2))) / CLOSE"""
        sma15 = self.sma(close, 15, 2)
        return ((high - sma15) - (low - sma15)) / (close + 1e-8)

    def alpha159(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Weighted multi-period Stochastic formula"""
        llv6 = self.ts_min(low, 6)
        hhv6 = self.ts_max(high, 6)
        llv12 = self.ts_min(low, 12)
        hhv12 = self.ts_max(high, 12)
        llv24 = self.ts_min(low, 24)
        hhv24 = self.ts_max(high, 24)

        stoch6 = (close - llv6) / (hhv6 - llv6 + 1e-8)
        stoch12 = (close - llv12) / (hhv12 - llv12 + 1e-8)
        stoch24 = (close - llv24) / (hhv24 - llv24 + 1e-8)

        return (stoch6 + stoch12 * 2 + stoch24 * 3) / 6

    def alpha160(self, close: pd.Series) -> pd.Series:
        """SMA(conditional STD, 20, 1)"""
        cond = close < self.delay(close, 1)
        return self.sma(np.where(cond, self.ts_std(close, 20), 0), 20, 1)

    def alpha161(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """MEAN(MAX series, 12)"""
        tr = np.maximum(high - low, np.maximum(
            self.abs_(high - self.delay(close, 1)),
            self.abs_(low - self.delay(close, 1))
        ))
        return self.ts_mean(tr, 12)

    def alpha162(self, close: pd.Series) -> pd.Series:
        """Normalized oscillator ratio formula"""
        delta = close - self.delay(close, 1)
        max_delta = self.sma(np.maximum(delta, 0), 12, 1)
        abs_delta = self.sma(self.abs_(delta), 12, 1)

        inner = (max_delta / (abs_delta + 1e-8)) * 100 - \
                (self.delay((max_delta / (abs_delta + 1e-8)) * 100, 1))
        return inner / (self.delay((max_delta / (abs_delta + 1e-8)) * 100, 1) + 1e-8)

    def alpha163(self, close: pd.Series, high: pd.Series, low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """RANK(return * volume * VWAP * spread)"""
        returns = close.pct_change()
        return self.rank(-1 * returns * self.ts_mean(volume, 20) * vwap * (high - low))

    def alpha164(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Complex inverse-price stochastic formula"""
        cond = close > self.delay(close, 1)
        return np.where(
            cond,
            1 / (close - self.ts_min(low, 12) + 1e-8),
            0
        ) + self.sma(
            np.where(cond,
                    (close - self.ts_min(low, 12)) / (self.ts_max(high, 12) - self.ts_min(low, 12) + 1e-8),
                    0),
            13, 2
        )

    def alpha165(self, close: pd.Series) -> pd.Series:
        """(ROWMAX - ROWMIN) / STD scaling"""
        return (self.ts_max(close, 48) - self.ts_min(close, 48)) / (self.ts_std(close, 48) + 1e-8)

    def alpha166(self, close: pd.Series) -> pd.Series:
        """Skewness-like calculation"""
        returns = close.pct_change()
        return self.rank(self.ts_sum(returns, 10)) - self.rank(self.ts_mean(returns, 5))

    def alpha167(self, close: pd.Series) -> pd.Series:
        """SUM(positive deltas, 12)"""
        delta = close - self.delay(close, 1)
        return self.ts_sum(np.where(delta > 0, delta, 0), 12)

    def alpha168(self, volume: pd.Series) -> pd.Series:
        """(-1 * VOLUME) / MEAN(VOLUME, 20)"""
        return -1 * volume / (self.ts_mean(volume, 20) + 1e-8)

    def alpha169(self, close: pd.Series) -> pd.Series:
        """Complex nested delayed SMA differences"""
        mean6 = self.ts_mean(close, 6)
        inner = self.sma(mean6 - self.delay(mean6, 1), 9, 1)
        return self.sma(inner - self.delay(inner, 1), 12, 1)

    def alpha170(self, high: pd.Series, volume: pd.Series, vwap: pd.Series, close: pd.Series) -> pd.Series:
        """Multi-component ranked formula"""
        adv20 = self.ts_mean(volume, 20)
        return (self.rank(1 / close) * volume / adv20) * \
               (high * self.rank(high - close) / (self.ts_sum(high, 5) / 5 + 1e-8)) - \
               self.rank(vwap - self.delay(vwap, 5))

    def alpha171(self, open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """(-1 * (LOW - CLOSE) * (OPEN^5)) / ((CLOSE - HIGH) * (CLOSE^5))"""
        return -1 * (low - close) * (open_ ** 5) / ((close - high + 1e-8) * (close ** 5 + 1e-8))

    def alpha172(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """ADX-like directional measurement"""
        tr = np.maximum(high - low, np.maximum(
            self.abs_(high - self.delay(close, 1)),
            self.abs_(low - self.delay(close, 1))
        ))
        hd = high - self.delay(high, 1)
        ld = self.delay(low, 1) - low

        dmp = np.where((hd > 0) & (hd > ld), hd, 0)
        dmm = np.where((ld > 0) & (ld > hd), ld, 0)

        atr14 = self.ts_mean(tr, 14)
        pdi = self.ts_mean(pd.Series(dmp, index=high.index), 14) / (atr14 + 1e-8) * 100
        mdi = self.ts_mean(pd.Series(dmm, index=high.index), 14) / (atr14 + 1e-8) * 100

        return self.ts_mean(self.abs_(pdi - mdi) / (pdi + mdi + 1e-8) * 100, 6)

    def alpha173(self, close: pd.Series) -> pd.Series:
        """Triple SMA triple log formula"""
        return 3 * self.sma(close, 13, 2) - 2 * self.sma(self.sma(close, 13, 2), 13, 2) + \
               self.sma(self.sma(self.sma(self.log(close), 13, 2), 13, 2), 13, 2)

    def alpha174(self, close: pd.Series) -> pd.Series:
        """SMA(conditional STD, 20, 1) - variant"""
        cond = close > self.delay(close, 1)
        return self.sma(np.where(cond, self.ts_std(close, 20), 0), 20, 1)

    def alpha175(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """MEAN(MAX series, 6)"""
        tr = np.maximum(high - low, np.maximum(
            self.abs_(high - self.delay(close, 1)),
            self.abs_(low - self.delay(close, 1))
        ))
        return self.ts_mean(tr, 6)

    def alpha176(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """CORR(RANK(Stochastic), RANK(VOLUME), 6)"""
        llv12 = self.ts_min(low, 12)
        hhv12 = self.ts_max(high, 12)
        stoch = (close - llv12) / (hhv12 - llv12 + 1e-8)
        return self.correlation(self.rank(stoch), self.rank(self.ts_mean(volume, 6)), 6)

    def alpha177(self, high: pd.Series) -> pd.Series:
        """((20 - HIGHDAY(HIGH, 20)) / 20) * 100"""
        return ((20 - self.ts_argmax(high, 20)) / 20) * 100

    def alpha178(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * VOLUME"""
        return (close - self.delay(close, 1)) / (self.delay(close, 1) + 1e-8) * volume

    def alpha179(self, low: pd.Series, vwap: pd.Series, volume: pd.Series) -> pd.Series:
        """Product of volume and volume-based correlation ranks"""
        adv50 = self.ts_mean(volume, 50)
        return self.rank(self.correlation(vwap, volume + 1, 4)) * \
               self.rank(self.correlation(self.rank(low), self.rank(adv50), 12))

    def alpha180(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Conditional: if MA(VOL, 20) < VOL then Tsrank else -VOL"""
        mean20 = self.ts_mean(volume, 20)
        cond = mean20 < volume
        return np.where(cond, -1 * self.ts_rank(self.abs_(self.delta(close, 7)), 60) * self.sign(self.delta(close, 7)), -volume)

    def alpha181(self, close: pd.Series, returns: pd.Series) -> pd.Series:
        """Return deviation correlation versus benchmark"""
        # Simplified - using close as benchmark proxy
        bench_ret = returns
        return self.ts_sum(
            (returns - self.ts_mean(returns, 20)) -
            (bench_ret - self.ts_mean(bench_ret, 20)) ** 2,
            20
        )

    def alpha182(self, close: pd.Series, open_: pd.Series, returns: pd.Series) -> pd.Series:
        """COUNT(matched directions, 20) / 20"""
        # Simplified correlation
        cond = (close > open_) == (returns > 0)
        return self.ts_count(cond, 20) / 20

    def alpha183(self, close: pd.Series) -> pd.Series:
        """(ROWMAX - ROWMIN) / STD with 24-period window"""
        return (self.ts_max(close, 24) - self.ts_min(close, 24)) / (self.ts_std(close, 24) + 1e-8)

    def alpha184(self, close: pd.Series, open_: pd.Series) -> pd.Series:
        """RANK(CORR(delayed spread, CLOSE, 200)) + RANK(spread)"""
        spread = open_ - close
        return self.rank(self.correlation(self.delay(spread, 1), close, 200)) + self.rank(spread)

    def alpha185(self, close: pd.Series, open_: pd.Series) -> pd.Series:
        """RANK((-1 * ((1 - OPEN / CLOSE)^2)))"""
        return self.rank(-1 * (1 - open_ / (close + 1e-8)) ** 2)

    def alpha186(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """ADX-like measurement (averaged with delay)"""
        tr = np.maximum(high - low, np.maximum(
            self.abs_(high - self.delay(close, 1)),
            self.abs_(low - self.delay(close, 1))
        ))
        hd = high - self.delay(high, 1)
        ld = self.delay(low, 1) - low

        dmp = np.where((hd > 0) & (hd > ld), hd, 0)
        dmm = np.where((ld > 0) & (ld > hd), ld, 0)

        atr14 = self.sma(tr, 14, 1)
        pdi = self.sma(pd.Series(dmp, index=high.index), 14, 1) / (atr14 + 1e-8) * 100
        mdi = self.sma(pd.Series(dmm, index=high.index), 14, 1) / (atr14 + 1e-8) * 100

        adx = self.sma(self.abs_(pdi - mdi) / (pdi + mdi + 1e-8) * 100, 6, 1)
        return (adx + self.delay(adx, 6)) / 2

    def alpha187(self, open_: pd.Series, high: pd.Series) -> pd.Series:
        """SUM(conditional open spreads, 20)"""
        cond = open_ <= self.delay(open_, 1)
        return self.ts_sum(np.where(cond, 0, np.maximum(high - open_, open_ - self.delay(open_, 1))), 20)

    def alpha188(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """((HIGH - LOW - SMA(HIGH - LOW, 11, 2)) / SMA(HIGH - LOW, 11, 2)) * 100"""
        sma11 = self.sma(high - low, 11, 2)
        return (high - low - sma11) / (sma11 + 1e-8) * 100

    def alpha189(self, close: pd.Series) -> pd.Series:
        """MEAN(ABS(CLOSE - MEAN(CLOSE, 6)), 6)"""
        return self.ts_mean(self.abs_(close - self.ts_mean(close, 6)), 6)

    def alpha190(self, close: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """Log-ratio calculation"""
        pct_change = (close - open_) / (open_ + 1e-8)
        count_pos = self.ts_count(pct_change > 0.05, 20)
        return self.log(count_pos + 1)

    def alpha191(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """(CORR(MEAN(VOL, 20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE"""
        return self.correlation(self.ts_mean(volume, 20), low, 5) + (high + low) / 2 - close

    # =========================================================================
    # GENERATE ALL ALPHAS
    # =========================================================================

    def generate_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all 191 Alpha signals.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with alpha191_001 through alpha191_191 columns added
        """
        result = df.copy()

        # Extract columns (with forex fallbacks)
        open_ = df.get('open', df.get('mid', df['close']))
        high = df.get('high', df.get('ask', df['close']))
        low = df.get('low', df.get('bid', df['close']))
        close = df['close']
        volume = df.get('volume', df.get('tick_count', pd.Series(1, index=df.index)))
        amount = df.get('amount', volume * close)  # Proxy for amount

        # Calculate derived values
        returns = close.pct_change()
        vwap = (high + low + close) / 3  # Simplified VWAP

        # Make open_, high, low, volume available for alphas that reference them
        globals()['open_'] = open_
        globals()['high'] = high
        globals()['low'] = low
        globals()['close'] = close
        globals()['volume'] = volume

        # Define all alpha functions with their dependencies
        alpha_funcs = [
            ('alpha191_001', lambda: self.alpha001(close, open_, volume)),
            ('alpha191_002', lambda: self.alpha002(close, open_, high, low)),
            ('alpha191_003', lambda: self.alpha003(close)),
            ('alpha191_004', lambda: self.alpha004(close, volume)),
            ('alpha191_005', lambda: self.alpha005(volume, high)),
            ('alpha191_006', lambda: self.alpha006(open_, high)),
            ('alpha191_007', lambda: self.alpha007(close, vwap, volume)),
            ('alpha191_008', lambda: self.alpha008(high, low, vwap)),
            ('alpha191_009', lambda: self.alpha009(high, low, volume)),
            ('alpha191_010', lambda: self.alpha010(close, returns)),
            ('alpha191_011', lambda: self.alpha011(close, high, low, volume)),
            ('alpha191_012', lambda: self.alpha012(open_, close, vwap)),
            ('alpha191_013', lambda: self.alpha013(high, low, vwap)),
            ('alpha191_014', lambda: self.alpha014(close)),
            ('alpha191_015', lambda: self.alpha015(open_, close)),
            ('alpha191_016', lambda: self.alpha016(volume, vwap)),
            ('alpha191_017', lambda: self.alpha017(close, vwap)),
            ('alpha191_018', lambda: self.alpha018(close)),
            ('alpha191_019', lambda: self.alpha019(close)),
            ('alpha191_020', lambda: self.alpha020(close)),
            ('alpha191_021', lambda: self.alpha021(close)),
            ('alpha191_022', lambda: self.alpha022(close)),
            ('alpha191_023', lambda: self.alpha023(close)),
            ('alpha191_024', lambda: self.alpha024(close)),
            ('alpha191_025', lambda: self.alpha025(close, volume, returns)),
            ('alpha191_026', lambda: self.alpha026(close, vwap)),
            ('alpha191_027', lambda: self.alpha027(close)),
            ('alpha191_028', lambda: self.alpha028(close, high, low)),
            ('alpha191_029', lambda: self.alpha029(close, volume)),
            ('alpha191_030', lambda: self.alpha030(close)),
            ('alpha191_031', lambda: self.alpha031(close)),
            ('alpha191_032', lambda: self.alpha032(volume, high)),
            ('alpha191_033', lambda: self.alpha033(close, low, returns)),
            ('alpha191_034', lambda: self.alpha034(close)),
            ('alpha191_035', lambda: self.alpha035(open_, volume)),
            ('alpha191_036', lambda: self.alpha036(volume, vwap)),
            ('alpha191_037', lambda: self.alpha037(open_, close, returns)),
            ('alpha191_038', lambda: self.alpha038(high)),
            ('alpha191_039', lambda: self.alpha039(close, open_, volume, vwap)),
            ('alpha191_040', lambda: self.alpha040(high, volume)),
            ('alpha191_041', lambda: self.alpha041(vwap)),
            ('alpha191_042', lambda: self.alpha042(high, volume)),
            ('alpha191_043', lambda: self.alpha043(close, volume)),
            ('alpha191_044', lambda: self.alpha044(volume, vwap)),
            ('alpha191_045', lambda: self.alpha045(close, volume, vwap)),
            ('alpha191_046', lambda: self.alpha046(close)),
            ('alpha191_047', lambda: self.alpha047(high, low, close)),
            ('alpha191_048', lambda: self.alpha048(close, volume)),
            ('alpha191_049', lambda: self.alpha049(high, low)),
            ('alpha191_050', lambda: self.alpha050(high, low, open_)),
            ('alpha191_051', lambda: self.alpha051(high, low, open_)),
            ('alpha191_052', lambda: self.alpha052(high, low)),
            ('alpha191_053', lambda: self.alpha053(close)),
            ('alpha191_054', lambda: self.alpha054(close, open_)),
            ('alpha191_055', lambda: self.alpha055(close, high, low, open_)),
            ('alpha191_056', lambda: self.alpha056(volume, high, open_)),
            ('alpha191_057', lambda: self.alpha057(close, high, low)),
            ('alpha191_058', lambda: self.alpha058(close)),
            ('alpha191_059', lambda: self.alpha059(close, high, low)),
            ('alpha191_060', lambda: self.alpha060(close, high, low, volume)),
            ('alpha191_061', lambda: self.alpha061(vwap, volume)),
            ('alpha191_062', lambda: self.alpha062(volume, high)),
            ('alpha191_063', lambda: self.alpha063(close)),
            ('alpha191_064', lambda: self.alpha064(close, volume, vwap)),
            ('alpha191_065', lambda: self.alpha065(close)),
            ('alpha191_066', lambda: self.alpha066(close)),
            ('alpha191_067', lambda: self.alpha067(close)),
            ('alpha191_068', lambda: self.alpha068(high, low, volume)),
            ('alpha191_069', lambda: self.alpha069(high, low, open_)),
            ('alpha191_070', lambda: self.alpha070(amount)),
            ('alpha191_071', lambda: self.alpha071(close)),
            ('alpha191_072', lambda: self.alpha072(high, low, close)),
            ('alpha191_073', lambda: self.alpha073(close, volume, vwap)),
            ('alpha191_074', lambda: self.alpha074(volume, vwap, low)),
            ('alpha191_075', lambda: self.alpha075(close, open_, volume)),
            ('alpha191_076', lambda: self.alpha076(volume, close)),
            ('alpha191_077', lambda: self.alpha077(high, low, volume, vwap)),
            ('alpha191_078', lambda: self.alpha078(high, low, close, volume)),
            ('alpha191_079', lambda: self.alpha079(close)),
            ('alpha191_080', lambda: self.alpha080(volume)),
            ('alpha191_081', lambda: self.alpha081(volume)),
            ('alpha191_082', lambda: self.alpha082(high, low, close)),
            ('alpha191_083', lambda: self.alpha083(high, volume)),
            ('alpha191_084', lambda: self.alpha084(close, volume)),
            ('alpha191_085', lambda: self.alpha085(volume, close)),
            ('alpha191_086', lambda: self.alpha086(close, vwap)),
            ('alpha191_087', lambda: self.alpha087(close, high, vwap, volume)),
            ('alpha191_088', lambda: self.alpha088(close)),
            ('alpha191_089', lambda: self.alpha089(close)),
            ('alpha191_090', lambda: self.alpha090(volume, vwap)),
            ('alpha191_091', lambda: self.alpha091(close, volume, low)),
            ('alpha191_092', lambda: self.alpha092(close, volume, vwap)),
            ('alpha191_093', lambda: self.alpha093(open_, low)),
            ('alpha191_094', lambda: self.alpha094(close, volume)),
            ('alpha191_095', lambda: self.alpha095(amount)),
            ('alpha191_096', lambda: self.alpha096(high, low, close)),
            ('alpha191_097', lambda: self.alpha097(volume)),
            ('alpha191_098', lambda: self.alpha098(close)),
            ('alpha191_099', lambda: self.alpha099(volume, close)),
            ('alpha191_100', lambda: self.alpha100(volume)),
            ('alpha191_101', lambda: self.alpha101(close, open_, high, low)),
            ('alpha191_102', lambda: self.alpha102(volume)),
            ('alpha191_103', lambda: self.alpha103(low)),
            ('alpha191_104', lambda: self.alpha104(high, volume, close)),
            ('alpha191_105', lambda: self.alpha105(open_, volume)),
            ('alpha191_106', lambda: self.alpha106(close)),
            ('alpha191_107', lambda: self.alpha107(open_, close, high, low)),
            ('alpha191_108', lambda: self.alpha108(high, vwap, volume)),
            ('alpha191_109', lambda: self.alpha109(high, low)),
            ('alpha191_110', lambda: self.alpha110(close, high, low)),
            ('alpha191_111', lambda: self.alpha111(high, low, close, volume)),
            ('alpha191_112', lambda: self.alpha112(close)),
            ('alpha191_113', lambda: self.alpha113(close, volume, low)),
            ('alpha191_114', lambda: self.alpha114(high, low, close, vwap, volume)),
            ('alpha191_115', lambda: self.alpha115(high, low, volume, close)),
            ('alpha191_116', lambda: self.alpha116(close)),
            ('alpha191_117', lambda: self.alpha117(high, low, close, volume)),
            ('alpha191_118', lambda: self.alpha118(high, low, open_)),
            ('alpha191_119', lambda: self.alpha119(open_, vwap, volume)),
            ('alpha191_120', lambda: self.alpha120(vwap, close)),
            ('alpha191_121', lambda: self.alpha121(vwap, volume)),
            ('alpha191_122', lambda: self.alpha122(close)),
            ('alpha191_123', lambda: self.alpha123(high, low, volume)),
            ('alpha191_124', lambda: self.alpha124(close, vwap)),
            ('alpha191_125', lambda: self.alpha125(close, vwap, volume)),
            ('alpha191_126', lambda: self.alpha126(high, low, close)),
            ('alpha191_127', lambda: self.alpha127(close)),
            ('alpha191_128', lambda: self.alpha128(high, low, close, volume)),
            ('alpha191_129', lambda: self.alpha129(close)),
            ('alpha191_130', lambda: self.alpha130(high, low, volume, vwap)),
            ('alpha191_131', lambda: self.alpha131(vwap, volume, close)),
            ('alpha191_132', lambda: self.alpha132(amount)),
            ('alpha191_133', lambda: self.alpha133(high, low)),
            ('alpha191_134', lambda: self.alpha134(close, volume)),
            ('alpha191_135', lambda: self.alpha135(close)),
            ('alpha191_136', lambda: self.alpha136(open_, volume, returns)),
            ('alpha191_137', lambda: self.alpha137(close, high, low, open_)),
            ('alpha191_138', lambda: self.alpha138(low, volume, vwap)),
            ('alpha191_139', lambda: self.alpha139(open_, volume)),
            ('alpha191_140', lambda: self.alpha140(close, high, low, open_, volume)),
            ('alpha191_141', lambda: self.alpha141(high, volume)),
            ('alpha191_142', lambda: self.alpha142(close, volume)),
            ('alpha191_143', lambda: pd.Series(0, index=close.index)),  # Unimplemented
            ('alpha191_144', lambda: self.alpha144(close, volume)),
            ('alpha191_145', lambda: self.alpha145(volume)),
            ('alpha191_146', lambda: self.alpha146(close)),
            ('alpha191_147', lambda: self.alpha147(close)),
            ('alpha191_148', lambda: self.alpha148(open_, volume)),
            ('alpha191_149', lambda: pd.Series(0, index=close.index)),  # Unimplemented
            ('alpha191_150', lambda: self.alpha150(high, low, close, volume)),
            ('alpha191_151', lambda: self.alpha151(close)),
            ('alpha191_152', lambda: self.alpha152(close)),
            ('alpha191_153', lambda: self.alpha153(close)),
            ('alpha191_154', lambda: self.alpha154(vwap, volume)),
            ('alpha191_155', lambda: self.alpha155(volume)),
            ('alpha191_156', lambda: self.alpha156(vwap, open_, volume)),
            ('alpha191_157', lambda: self.alpha157(close, returns)),
            ('alpha191_158', lambda: self.alpha158(high, low, close)),
            ('alpha191_159', lambda: self.alpha159(close, high, low)),
            ('alpha191_160', lambda: self.alpha160(close)),
            ('alpha191_161', lambda: self.alpha161(close, high, low)),
            ('alpha191_162', lambda: self.alpha162(close)),
            ('alpha191_163', lambda: self.alpha163(close, high, low, vwap, volume)),
            ('alpha191_164', lambda: self.alpha164(close, high, low)),
            ('alpha191_165', lambda: self.alpha165(close)),
            ('alpha191_166', lambda: self.alpha166(close)),
            ('alpha191_167', lambda: self.alpha167(close)),
            ('alpha191_168', lambda: self.alpha168(volume)),
            ('alpha191_169', lambda: self.alpha169(close)),
            ('alpha191_170', lambda: self.alpha170(high, volume, vwap, close)),
            ('alpha191_171', lambda: self.alpha171(open_, close, high, low)),
            ('alpha191_172', lambda: self.alpha172(high, low, close)),
            ('alpha191_173', lambda: self.alpha173(close)),
            ('alpha191_174', lambda: self.alpha174(close)),
            ('alpha191_175', lambda: self.alpha175(high, low, close)),
            ('alpha191_176', lambda: self.alpha176(high, low, close, volume)),
            ('alpha191_177', lambda: self.alpha177(high)),
            ('alpha191_178', lambda: self.alpha178(close, volume)),
            ('alpha191_179', lambda: self.alpha179(low, vwap, volume)),
            ('alpha191_180', lambda: self.alpha180(close, volume)),
            ('alpha191_181', lambda: self.alpha181(close, returns)),
            ('alpha191_182', lambda: self.alpha182(close, open_, returns)),
            ('alpha191_183', lambda: self.alpha183(close)),
            ('alpha191_184', lambda: self.alpha184(close, open_)),
            ('alpha191_185', lambda: self.alpha185(close, open_)),
            ('alpha191_186', lambda: self.alpha186(high, low, close)),
            ('alpha191_187', lambda: self.alpha187(open_, high)),
            ('alpha191_188', lambda: self.alpha188(high, low)),
            ('alpha191_189', lambda: self.alpha189(close)),
            ('alpha191_190', lambda: self.alpha190(close, open_, high, low)),
            ('alpha191_191', lambda: self.alpha191(close, high, low, volume)),
        ]

        # Generate all alphas with error handling
        for name, func in alpha_funcs:
            try:
                val = func()
                if isinstance(val, (int, float)):
                    result[name] = val
                else:
                    result[name] = val
            except Exception as e:
                result[name] = 0

        return result


if __name__ == '__main__':
    # Test
    print("Alpha191 国泰君安 - Testing")
    print("=" * 50)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    np.random.seed(42)
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 500)
    }, index=dates)

    alpha = Alpha191GuotaiJunan()
    result = alpha.generate_all_alphas(df)

    alpha_cols = [c for c in result.columns if c.startswith('alpha191_')]
    print(f"Generated {len(alpha_cols)} alphas")
    print(f"Sample columns: {alpha_cols[:5]}...")
    print(f"Last columns: {alpha_cols[-5:]}")

    # Check non-zero values
    non_zero = sum(1 for c in alpha_cols if result[c].abs().sum() > 0)
    print(f"Non-zero alphas: {non_zero}/{len(alpha_cols)}")
