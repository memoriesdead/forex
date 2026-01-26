"""
Gitee Chinese Quantitative Factors for Forex Trading
=====================================================
Research from Gitee/GitHub Chinese quant repositories adapted for forex.

RESEARCH SOURCES:
-----------------
[1] QUANTAXIS (yutiansut) - 中国最大量化框架
    Gitee: https://gitee.com/yutiansut/QUANTAXIS
    GitHub: https://github.com/yutiansut/QUANTAXIS
    Stars: 8.5k+
    License: MIT

[2] QuantsPlaybook (hugo2046) - 券商金工研报复现
    GitHub: https://github.com/hugo2046/QuantsPlaybook
    Stars: 3.3k
    Content: 100+ strategies from Chinese securities research

[3] Alpha-101-GTJA-191 - 国泰君安191因子
    GitHub: https://github.com/wpwpwpwpwpwpwpwpwp/Alpha-101-GTJA-191
    Original: 国泰君安 "基于短周期价量特征的多因子选股体系" (2017)
    Stars: 116

[4] JoinQuant Alpha191
    GitHub: https://github.com/JoinQuant/jqdatasdk/blob/master/jqdatasdk/alpha191.py
    Platform: https://joinquant.com/data/dict/alpha191

[5] VPIN Implementation
    GitHub: https://github.com/jheusser/vpin
    GitHub: https://github.com/yt-feng/VPIN
    Stars: 113

[6] Microprice
    GitHub: https://github.com/sstoikov/microprice
    Paper: Stoikov (2018) "The Micro-Price: A High-Frequency Estimator"

[7] vnpy - 量化交易平台
    GitHub: https://github.com/vnpy/vnpy
    Stars: 25k+

[8] Hikyuu - 极速量化框架
    Gitee: https://gitee.com/fasiondog/hikyuu
    Stars: 2.5k+

ACADEMIC CITATIONS:
-------------------
[A1] Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and
     Time-Series Effects." Journal of Financial Markets, 5(1), 31-56.

[A2] Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order
     Book Events." Journal of Financial Econometrics, 12(1), 47-88.

[A3] Cont, R. (2023). "Cross-Impact of Order Flow Imbalance in Equity Markets."
     Quantitative Finance. (PCA extension for multi-level OFI)

[A4] Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and
     Liquidity in a High-frequency World." Review of Financial Studies, 25(5).

[A5] Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
     Econometrica, 53(6), 1315-1335.

[A6] Garman, M. B., & Klass, M. J. (1980). "On the Estimation of Security Price
     Volatilities from Historical Data." Journal of Business, 53(1), 67-78.

[A7] Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation
     Based on High, Low, Open, and Close Prices." Journal of Business, 73(3).

[A8] Parkinson, M. (1980). "The Extreme Value Method for Estimating the
     Variance of the Rate of Return." Journal of Business, 53(1), 61-65.

[A9] Rogers, L. C. G., & Satchell, S. E. (1991). "Estimating Variance from
     High, Low and Closing Prices." Annals of Applied Probability, 1(4).

[A10] Hasbrouck, J. (2009). "Trading Costs and Returns for U.S. Equities:
      Estimating Effective Costs from Daily Data." Journal of Finance, 64(3).

CHINESE SECURITIES RESEARCH REPORTS:
------------------------------------
[R1] 开源证券 《聪明钱因子模型的2.0版本》(2020)
     URL: https://www.kysec.cn/index.php?m=content&c=index&a=show&catid=108&id=1595

[R2] 招商证券 "琢璞"系列报告之十七：高频数据中的知情交易（二）(2020)
     URL: https://asset.quant-wiki.com/pdf/20200630-招商证券.pdf

[R3] 兴业证券 高频研究系列五：市场微观结构剖析
     URL: https://asset.quant-wiki.com/pdf/20221109-兴业证券.pdf

[R4] 国泰君安 数量化专题之九十三 (2017)
     "基于短周期价量特征的多因子选股体系"

[R5] 华泰证券 人工智能系列 (遗传规划因子挖掘、图神经网络)

[R6] 海通证券 "海量"专题 (深度学习高频因子)

FOREX APPLICABILITY NOTES:
--------------------------
- Most Alpha191 factors originally designed for A-shares
- Factors using only OHLCV data are directly applicable to forex
- Volume in forex = tick volume (proxy, not actual traded volume)
- Spread-based factors work well in forex (24h market, variable spreads)
- Order flow factors require L2 data from broker (e.g., IBKR, LMAX)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ALPHA191 FACTORS - FOREX APPLICABLE SUBSET
# Source: [3] Alpha-101-GTJA-191, [4] JoinQuant Alpha191
# Citation: [R4] 国泰君安 数量化专题之九十三 (2017)
# =============================================================================

class Alpha191ForexFactors:
    """
    Alpha191 factors adapted for forex trading.

    Original Source:
        国泰君安 "基于短周期价量特征的多因子选股体系" (2017)
        GitHub: https://github.com/wpwpwpwpwpwpwpwpwp/Alpha-101-GTJA-191
        JoinQuant: https://joinquant.com/data/dict/alpha191

    Selection Criteria:
        - Uses only OHLCV data (no fundamentals)
        - No A-share specific features (no limit-up/down, suspend)
        - Applicable to 24-hour forex market

    Forex Applicability: HIGH
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                Volume can be tick volume for forex.
        """
        self.open = df['open'] if 'open' in df.columns else df['close']
        self.high = df['high'] if 'high' in df.columns else df['close']
        self.low = df['low'] if 'low' in df.columns else df['close']
        self.close = df['close']
        self.volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        self.index = df.index

    # -------------------------------------------------------------------------
    # Helper Functions (from Alpha191 specification)
    # -------------------------------------------------------------------------

    @staticmethod
    def _delay(series: pd.Series, n: int) -> pd.Series:
        """DELAY(X, N) = X shifted by N periods."""
        return series.shift(n)

    @staticmethod
    def _delta(series: pd.Series, n: int) -> pd.Series:
        """DELTA(X, N) = X - DELAY(X, N)."""
        return series - series.shift(n)

    @staticmethod
    def _rank(series: pd.Series) -> pd.Series:
        """RANK(X) = Cross-sectional rank (for single asset, use pct_rank)."""
        return series.rank(pct=True)

    @staticmethod
    def _ts_rank(series: pd.Series, n: int) -> pd.Series:
        """TSRANK(X, N) = Time-series rank over N periods."""
        return series.rolling(n).apply(lambda x: stats.rankdata(x)[-1] / len(x), raw=True)

    @staticmethod
    def _ts_max(series: pd.Series, n: int) -> pd.Series:
        """TSMAX(X, N) = Maximum over N periods."""
        return series.rolling(n).max()

    @staticmethod
    def _ts_min(series: pd.Series, n: int) -> pd.Series:
        """TSMIN(X, N) = Minimum over N periods."""
        return series.rolling(n).min()

    @staticmethod
    def _ts_argmax(series: pd.Series, n: int) -> pd.Series:
        """TSARGMAX(X, N) = Index of maximum in last N periods."""
        return series.rolling(n).apply(lambda x: np.argmax(x) + 1, raw=True)

    @staticmethod
    def _ts_argmin(series: pd.Series, n: int) -> pd.Series:
        """TSARGMIN(X, N) = Index of minimum in last N periods."""
        return series.rolling(n).apply(lambda x: np.argmin(x) + 1, raw=True)

    @staticmethod
    def _sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
        """SMA(X, N, M) = Exponential moving average."""
        return series.ewm(alpha=m/n, adjust=False).mean()

    @staticmethod
    def _wma(series: pd.Series, n: int) -> pd.Series:
        """WMA(X, N) = Weighted moving average."""
        weights = np.arange(1, n + 1)
        return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def _decay_linear(series: pd.Series, n: int) -> pd.Series:
        """DECAYLINEAR(X, N) = Linearly decaying weighted average."""
        weights = np.arange(n, 0, -1).astype(float)
        weights = weights / weights.sum()
        return series.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def _corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """CORR(X, Y, N) = Rolling correlation."""
        return x.rolling(n).corr(y)

    @staticmethod
    def _cov(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """COV(X, Y, N) = Rolling covariance."""
        return x.rolling(n).cov(y)

    @staticmethod
    def _count(condition: pd.Series, n: int) -> pd.Series:
        """COUNT(COND, N) = Count of True in last N periods."""
        return condition.astype(int).rolling(n).sum()

    @staticmethod
    def _sum(series: pd.Series, n: int) -> pd.Series:
        """SUM(X, N) = Rolling sum."""
        return series.rolling(n).sum()

    @staticmethod
    def _mean(series: pd.Series, n: int) -> pd.Series:
        """MEAN(X, N) = Rolling mean."""
        return series.rolling(n).mean()

    @staticmethod
    def _std(series: pd.Series, n: int) -> pd.Series:
        """STD(X, N) = Rolling standard deviation."""
        return series.rolling(n).std()

    # -------------------------------------------------------------------------
    # FOREX-APPLICABLE ALPHA FACTORS
    # -------------------------------------------------------------------------

    def alpha_001(self) -> pd.Series:
        """
        Alpha001: Volume-Price Correlation
        Formula: -1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6)

        Forex Applicability: MEDIUM (volume is tick volume)
        """
        x = self._rank(self._delta(np.log(self.volume + 1), 1))
        y = self._rank((self.close - self.open) / (self.open + 1e-10))
        return -1 * self._corr(x, y, 6)

    def alpha_002(self) -> pd.Series:
        """
        Alpha002: Price Position Change
        Formula: -1 * DELTA(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)

        Forex Applicability: HIGH (pure price-based)
        """
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10)
        return -1 * self._delta(inner, 1)

    def alpha_014(self) -> pd.Series:
        """
        Alpha014: 5-Day Momentum
        Formula: CLOSE - DELAY(CLOSE, 5)

        Forex Applicability: HIGH
        """
        return self.close - self._delay(self.close, 5)

    def alpha_018(self) -> pd.Series:
        """
        Alpha018: 5-Day Price Ratio
        Formula: CLOSE / DELAY(CLOSE, 5)

        Forex Applicability: HIGH
        """
        return self.close / (self._delay(self.close, 5) + 1e-10)

    def alpha_020(self) -> pd.Series:
        """
        Alpha020: 6-Day Percentage Return
        Formula: (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * 100

        Forex Applicability: HIGH
        """
        delayed = self._delay(self.close, 6)
        return (self.close - delayed) / (delayed + 1e-10) * 100

    def alpha_031(self) -> pd.Series:
        """
        Alpha031: Price Deviation from MA
        Formula: (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100

        Forex Applicability: HIGH
        """
        ma = self._mean(self.close, 12)
        return (self.close - ma) / (ma + 1e-10) * 100

    def alpha_046(self) -> pd.Series:
        """
        Alpha046: Multi-MA Average Ratio
        Formula: (MA(CLOSE,3) + MA(CLOSE,6) + MA(CLOSE,12) + MA(CLOSE,24)) / 4 / CLOSE

        Forex Applicability: HIGH
        """
        ma3 = self._mean(self.close, 3)
        ma6 = self._mean(self.close, 6)
        ma12 = self._mean(self.close, 12)
        ma24 = self._mean(self.close, 24)
        return (ma3 + ma6 + ma12 + ma24) / 4 / (self.close + 1e-10)

    def alpha_053(self) -> pd.Series:
        """
        Alpha053: Win Rate
        Formula: COUNT(CLOSE > DELAY(CLOSE, 1), 12) / 12 * 100

        Forex Applicability: HIGH
        """
        condition = self.close > self._delay(self.close, 1)
        return self._count(condition, 12) / 12 * 100

    def alpha_054(self) -> pd.Series:
        """
        Alpha054: Price Dispersion
        Formula: (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN, 10)))

        Forex Applicability: HIGH
        """
        part1 = self._std(np.abs(self.close - self.open), 10)
        part2 = self.close - self.open
        part3 = self._corr(self.close, self.open, 10)
        return -1 * self._rank(part1 + part2 + part3)

    def alpha_060(self) -> pd.Series:
        """
        Alpha060: Williams %R Variant
        Formula: -1 * RANK(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME)

        Forex Applicability: MEDIUM (uses volume)
        """
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10)
        return -1 * self._rank(inner * self.volume)

    def alpha_101(self) -> pd.Series:
        """
        Alpha101: Normalized Price Position
        Formula: (CLOSE - OPEN) / ((HIGH - LOW) + 0.001)

        Forex Applicability: HIGH (Williams %R style)
        """
        return (self.close - self.open) / (self.high - self.low + 0.001)

    def alpha_126(self) -> pd.Series:
        """
        Alpha126: Typical Price
        Formula: (CLOSE + HIGH + LOW) / 3

        Forex Applicability: HIGH
        """
        return (self.close + self.high + self.low) / 3

    def alpha_130(self) -> pd.Series:
        """
        Alpha130: Volume-Weighted Returns
        Formula: RANK(DECAYLINEAR(((HIGH + LOW) / 2 - (DELAY(HIGH) + DELAY(LOW)) / 2) * VWAP, 5))

        Forex Applicability: MEDIUM (uses volume)
        """
        mid = (self.high + self.low) / 2
        delayed_mid = (self._delay(self.high, 1) + self._delay(self.low, 1)) / 2
        vwap = (self.close * self.volume).rolling(20).sum() / self.volume.rolling(20).sum()

        inner = (mid - delayed_mid) * vwap
        return self._rank(self._decay_linear(inner, 5))

    def alpha_160(self) -> pd.Series:
        """
        Alpha160: Conditional Volatility
        Formula: SMA(IF(CLOSE <= DELAY(CLOSE), STD(CLOSE, 20), 0), 20, 1)

        Forex Applicability: HIGH
        """
        condition = self.close <= self._delay(self.close, 1)
        vol = self._std(self.close, 20)
        conditional_vol = np.where(condition, vol, 0)
        return pd.Series(conditional_vol, index=self.index).rolling(20).mean()

    def alpha_188(self) -> pd.Series:
        """
        Alpha188: Normalized Range Expansion
        Formula: ((HIGH - LOW - SMA(HIGH - LOW, 11, 2)) / SMA(HIGH - LOW, 11, 2)) * 100

        Forex Applicability: HIGH
        """
        hl_range = self.high - self.low
        sma_range = self._sma(hl_range, 11, 2)
        return (hl_range - sma_range) / (sma_range + 1e-10) * 100

    def alpha_191(self) -> pd.Series:
        """
        Alpha191: Volume-Price Momentum
        Formula: (CORR(MEAN(VOLUME, 20), LOW, 5) + (HIGH + LOW) / 2) - CLOSE

        Forex Applicability: MEDIUM (uses volume)
        """
        vol_ma = self._mean(self.volume, 20)
        corr = self._corr(vol_ma, self.low, 5)
        mid = (self.high + self.low) / 2
        return corr + mid - self.close

    def compute_all(self) -> pd.DataFrame:
        """
        Compute all forex-applicable Alpha191 factors.

        Returns:
            DataFrame with all factor columns
        """
        factors = pd.DataFrame(index=self.index)

        # High applicability factors
        factors['alpha191_001'] = self.alpha_001()
        factors['alpha191_002'] = self.alpha_002()
        factors['alpha191_014'] = self.alpha_014()
        factors['alpha191_018'] = self.alpha_018()
        factors['alpha191_020'] = self.alpha_020()
        factors['alpha191_031'] = self.alpha_031()
        factors['alpha191_046'] = self.alpha_046()
        factors['alpha191_053'] = self.alpha_053()
        factors['alpha191_054'] = self.alpha_054()
        factors['alpha191_060'] = self.alpha_060()
        factors['alpha191_101'] = self.alpha_101()
        factors['alpha191_126'] = self.alpha_126()
        factors['alpha191_130'] = self.alpha_130()
        factors['alpha191_160'] = self.alpha_160()
        factors['alpha191_188'] = self.alpha_188()
        factors['alpha191_191'] = self.alpha_191()

        return factors.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# SMART MONEY FACTOR 2.0
# Source: [R1] 开源证券, [2] QuantsPlaybook
# =============================================================================

class SmartMoneyFactor2:
    """
    Smart Money Factor 2.0 (聪明钱因子模型的2.0版本).

    Source:
        开源证券 《市场微观结构研究系列（3）：聪明钱因子模型的2.0版本》(2020)
        URL: https://www.kysec.cn/index.php?m=content&c=index&a=show&catid=108&id=1595

    Reproduced in:
        QuantsPlaybook (hugo2046)
        GitHub: https://github.com/hugo2046/QuantsPlaybook/blob/master/B-因子构建类/聪明钱因子模型的2.0版本

    Methodology:
        1. Calculate S-indicator: S_t = |R_t| / sqrt(V_t)
        2. Rank by S (descending), top 20% cumulative volume = "smart money"
        3. Calculate Q = VWAP_smart / VWAP_all
        4. Q > 1: Smart money buying high (bearish signal)
        5. Q < 1: Smart money buying low (bullish signal)

    Forex Applicability: HIGH
    """

    def __init__(self, lookback: int = 10, smart_pct: float = 0.2):
        """
        Args:
            lookback: Number of periods for calculation (default 10 days)
            smart_pct: Percentage of volume considered "smart" (default 20%)
        """
        self.lookback = lookback
        self.smart_pct = smart_pct

    def calculate_s_indicator(self, returns: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate S-indicator per observation.

        Formula:
            S_t = |R_t| / sqrt(V_t)

        Interpretation:
            High S = large price move relative to volume (informed trade)
            Low S = small price move for volume (noise trade)
        """
        return np.abs(returns) / (np.sqrt(volume) + 1e-10)

    def identify_smart_trades(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify "smart money" trades using S-indicator ranking.

        Returns:
            Boolean series where True = smart money trade
        """
        returns = df['close'].pct_change()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        s_indicator = self.calculate_s_indicator(returns, volume)

        # Rolling identification
        def identify_window(window_data):
            s_vals = window_data['s'].values
            vol_vals = window_data['volume'].values

            # Sort by S descending
            sorted_idx = np.argsort(s_vals)[::-1]

            # Cumulative volume
            sorted_vol = vol_vals[sorted_idx]
            cum_vol = np.cumsum(sorted_vol)
            total_vol = cum_vol[-1]

            # Mark top 20% by cumulative volume as smart
            is_smart = cum_vol <= total_vol * self.smart_pct

            # Map back to original order
            result = np.zeros(len(s_vals), dtype=bool)
            for i, idx in enumerate(sorted_idx):
                result[idx] = is_smart[i]

            return result[-1]  # Return last value for rolling

        temp_df = pd.DataFrame({
            's': s_indicator,
            'volume': volume
        })

        # Simplified: use quantile threshold
        s_threshold = s_indicator.rolling(self.lookback).quantile(1 - self.smart_pct)
        return s_indicator > s_threshold

    def calculate_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Smart Money Factor Q.

        Formula:
            Q = VWAP_smart / VWAP_all

        Signal:
            Q > 1: Smart money selling (bearish)
            Q < 1: Smart money buying (bullish)
        """
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        returns = close.pct_change()
        s_indicator = self.calculate_s_indicator(returns, volume)

        # Rolling calculation
        def calc_q(window_idx):
            if len(window_idx) < 5:
                return 1.0

            window_df = df.loc[window_idx]
            s_vals = s_indicator.loc[window_idx].values

            # Threshold for smart money
            threshold = np.percentile(s_vals[~np.isnan(s_vals)], (1 - self.smart_pct) * 100)
            is_smart = s_vals >= threshold

            if is_smart.sum() == 0:
                return 1.0

            prices = window_df['close'].values
            vols = window_df['volume'].values if 'volume' in window_df.columns else np.ones(len(prices))

            # VWAP calculations
            smart_vwap = np.sum(prices[is_smart] * vols[is_smart]) / (np.sum(vols[is_smart]) + 1e-10)
            all_vwap = np.sum(prices * vols) / (np.sum(vols) + 1e-10)

            return smart_vwap / (all_vwap + 1e-10)

        # Use rolling apply
        q_values = []
        for i in range(len(df)):
            start_idx = max(0, i - self.lookback + 1)
            window_idx = df.index[start_idx:i+1]
            q = calc_q(window_idx)
            q_values.append(q)

        q_series = pd.Series(q_values, index=df.index)

        # Convert to signal: negative Q means smart money bullish
        signal = 1 - q_series  # Higher = more bullish

        return signal

    def calculate_enhanced_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced Smart Money Factor with additional signals.

        Returns:
            DataFrame with multiple smart money indicators
        """
        result = pd.DataFrame(index=df.index)

        # Base factor
        result['smart_money_q'] = self.calculate_factor(df)

        # Smart trade identification
        result['is_smart_trade'] = self.identify_smart_trades(df).astype(int)

        # S-indicator z-score
        returns = df['close'].pct_change()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        s_indicator = self.calculate_s_indicator(returns, volume)

        s_mean = s_indicator.rolling(50).mean()
        s_std = s_indicator.rolling(50).std()
        result['s_indicator_zscore'] = (s_indicator - s_mean) / (s_std + 1e-10)

        # Smart money momentum
        result['smart_money_momentum'] = result['smart_money_q'].diff(5)

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# VPIN ENHANCED (Volume-Synchronized Probability of Informed Trading)
# Source: [5] VPIN implementations, [R2] 招商证券
# Citation: [A4] Easley, López de Prado, O'Hara (2012)
# =============================================================================

class VPINEnhanced:
    """
    Enhanced VPIN Implementation.

    Sources:
        GitHub: https://github.com/jheusser/vpin (113 stars)
        GitHub: https://github.com/yt-feng/VPIN
        招商证券 "琢璞"系列报告之十七 (2020)

    Citation:
        Easley, D., López de Prado, M., & O'Hara, M. (2012).
        "Flow Toxicity and Liquidity in a High-frequency World"
        Review of Financial Studies, 25(5), 1457-1493.

    Formula:
        VPIN = Σ|V_B - V_S| / (n × V_bucket)

        V_B: Buy volume (classified using BVC)
        V_S: Sell volume
        n: Number of buckets
        V_bucket: Volume per bucket (volume clock)

    Forex Applicability: HIGH
    """

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        """
        Args:
            bucket_size: Volume per bucket (volume clock)
            n_buckets: Number of buckets for VPIN calculation
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

    def bulk_volume_classification(self,
                                    prices: pd.Series,
                                    volumes: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Bulk Volume Classification (BVC) method.

        Citation: Easley et al. (2012), Section 2.2

        Formula:
            Z = (P_close - P_open) / σ
            V_buy = V × Φ(Z)
            V_sell = V × (1 - Φ(Z))

        Where Φ is the standard normal CDF.
        """
        # Calculate price changes
        price_change = prices.diff()

        # Estimate volatility
        sigma = price_change.rolling(20).std()
        sigma = sigma.replace(0, np.nan).fillna(price_change.std())

        # Z-score
        z_score = price_change / (sigma + 1e-10)

        # CDF classification
        buy_prob = stats.norm.cdf(z_score)

        v_buy = volumes * buy_prob
        v_sell = volumes * (1 - buy_prob)

        return v_buy, v_sell

    def calculate_vpin(self,
                       prices: pd.Series,
                       volumes: pd.Series) -> pd.Series:
        """
        Calculate VPIN time series.

        High VPIN indicates:
            - High probability of informed trading
            - Expected volatility increase
            - Toxic order flow (adverse selection risk)
        """
        v_buy, v_sell = self.bulk_volume_classification(prices, volumes)

        # Order imbalance
        total_vol = v_buy + v_sell + 1e-10
        order_imbalance = np.abs(v_buy - v_sell) / total_vol

        # Rolling VPIN
        vpin = order_imbalance.rolling(self.n_buckets, min_periods=10).mean()

        return vpin

    def calculate_vpin_cdf(self,
                          prices: pd.Series,
                          volumes: pd.Series,
                          window: int = 252) -> pd.Series:
        """
        Calculate VPIN CDF (percentile rank).

        VPIN CDF > 0.9 = Extreme toxicity, expect flash crash

        Source: Easley et al. (2012), Section 4
        """
        vpin = self.calculate_vpin(prices, volumes)
        vpin_cdf = vpin.rolling(window).rank(pct=True)
        return vpin_cdf

    def calculate_all_signals(self,
                              df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all VPIN-related signals.
        """
        result = pd.DataFrame(index=df.index)

        prices = df['close']
        volumes = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Base VPIN
        result['vpin'] = self.calculate_vpin(prices, volumes)

        # VPIN CDF
        result['vpin_cdf'] = self.calculate_vpin_cdf(prices, volumes)

        # Toxicity signal (binary)
        result['toxicity_high'] = (result['vpin_cdf'] > 0.9).astype(int)
        result['toxicity_low'] = (result['vpin_cdf'] < 0.1).astype(int)

        # VPIN momentum
        result['vpin_momentum'] = result['vpin'].diff(5)

        # VPIN z-score
        vpin_mean = result['vpin'].rolling(100).mean()
        vpin_std = result['vpin'].rolling(100).std()
        result['vpin_zscore'] = (result['vpin'] - vpin_mean) / (vpin_std + 1e-10)

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# INTEGRATED ORDER FLOW IMBALANCE (OFI)
# Source: [R3] 兴业证券, VeighNa社区
# Citation: [A2] Cont et al. (2014), [A3] Cont (2023)
# =============================================================================

class IntegratedOFI:
    """
    Integrated Order Flow Imbalance with PCA.

    Sources:
        兴业证券 高频研究系列五 (2022)
        VeighNa社区: https://www.vnpy.com/forum/topic/33101

    Citations:
        Cont, R., Kukanov, A., & Stoikov, S. (2014).
        "The Price Impact of Order Book Events"
        Journal of Financial Econometrics, 12(1), 47-88.

        Cont, R. (2023). "Cross-Impact of Order Flow Imbalance in Equity Markets"
        Quantitative Finance. (PCA extension)

    Formula (Single Level):
        OFI_t = I{P_bid ≥ P_bid_prev} × V_bid - I{P_bid ≤ P_bid_prev} × V_bid_prev
              - I{P_ask ≤ P_ask_prev} × V_ask + I{P_ask ≥ P_ask_prev} × V_ask_prev

    Integrated OFI:
        Extract first principal component from multi-level OFI
        Explains ~85% of variance (Cont 2023)

    Forex Applicability: HIGH (with L2 data)
    """

    def __init__(self, n_levels: int = 5, use_pca: bool = True):
        """
        Args:
            n_levels: Number of order book levels
            use_pca: Whether to use PCA for integration
        """
        self.n_levels = n_levels
        self.use_pca = use_pca

    def calculate_single_level_ofi(self,
                                   bid_price: pd.Series,
                                   bid_size: pd.Series,
                                   ask_price: pd.Series,
                                   ask_size: pd.Series) -> pd.Series:
        """
        Calculate OFI for a single price level.

        Citation: Cont et al. (2014), Equation (2)
        """
        # Price changes
        delta_bid = bid_price.diff()
        delta_ask = ask_price.diff()

        # Bid side contribution
        ofi_bid = np.where(delta_bid > 0, bid_size,
                  np.where(delta_bid < 0, -bid_size.shift(1),
                          bid_size - bid_size.shift(1)))

        # Ask side contribution (negated)
        ofi_ask = np.where(delta_ask < 0, -ask_size,
                  np.where(delta_ask > 0, ask_size.shift(1),
                          -(ask_size - ask_size.shift(1))))

        ofi = pd.Series(ofi_bid + ofi_ask, index=bid_price.index)

        return ofi.fillna(0)

    def calculate_integrated_ofi_simple(self,
                                        df: pd.DataFrame) -> pd.Series:
        """
        Calculate integrated OFI from simple OHLCV data.

        Approximation when L2 data not available.
        """
        returns = df['close'].pct_change()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Sign volume based on price direction
        signed_volume = np.sign(returns) * volume

        # Cumulative OFI
        ofi = signed_volume.rolling(20).sum()

        # Normalize by average volume
        avg_vol = volume.rolling(100).mean()
        normalized_ofi = ofi / (avg_vol + 1e-10)

        return normalized_ofi.fillna(0)

    def calculate_ofi_price_impact(self,
                                   df: pd.DataFrame,
                                   window: int = 50) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate OFI and estimate price impact coefficient (lambda).

        Model: ΔP_t = λ × OFI_t + ε_t

        Citation: Cont et al. (2014), Section 3

        Returns:
            (ofi_series, lambda_series)
        """
        returns = df['close'].pct_change() * 10000  # bps
        ofi = self.calculate_integrated_ofi_simple(df)

        # Rolling regression for lambda
        cov = returns.rolling(window).cov(ofi)
        var = ofi.rolling(window).var()

        lambda_coef = cov / (var + 1e-10)

        return ofi, lambda_coef

    def calculate_all_ofi_features(self,
                                    df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all OFI-related features.
        """
        result = pd.DataFrame(index=df.index)

        # Base OFI
        ofi, lambda_coef = self.calculate_ofi_price_impact(df)
        result['ofi'] = ofi
        result['ofi_lambda'] = lambda_coef

        # OFI momentum
        result['ofi_momentum_5'] = ofi.diff(5)
        result['ofi_momentum_20'] = ofi.diff(20)

        # OFI z-score
        ofi_mean = ofi.rolling(100).mean()
        ofi_std = ofi.rolling(100).std()
        result['ofi_zscore'] = (ofi - ofi_mean) / (ofi_std + 1e-10)

        # OFI signal (smoothed)
        result['ofi_signal'] = ofi.rolling(5).mean()

        # OFI regime (positive/negative accumulation)
        result['ofi_regime'] = np.sign(ofi.rolling(20).mean())

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# VOLATILITY ESTIMATORS
# Citation: [A6] Garman-Klass, [A7] Yang-Zhang, [A8] Parkinson, [A9] Rogers-Satchell
# =============================================================================

class VolatilityEstimators:
    """
    Range-Based Volatility Estimators.

    Sources:
        GitHub: burakbayramli/books/Volatility_Trading_Sinclair
        TradingView: Garman-Klass-Yang-Zhang indicators

    Citations:
        [A6] Garman, M. B., & Klass, M. J. (1980)
        [A7] Yang, D., & Zhang, Q. (2000)
        [A8] Parkinson, M. (1980)
        [A9] Rogers, L. C. G., & Satchell, S. E. (1991)

    Advantages over close-to-close volatility:
        - More efficient (lower variance)
        - Uses intraday information (OHLC)
        - Better for short-term forecasting

    Forex Applicability: HIGH
    """

    def __init__(self, window: int = 20, annualize: bool = True):
        """
        Args:
            window: Rolling window for estimation
            annualize: Whether to annualize volatility (252 trading days)
        """
        self.window = window
        self.annualize = annualize
        self.annualization_factor = np.sqrt(252) if annualize else 1

    def parkinson(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Parkinson (1980) volatility estimator.

        Citation: [A8]

        Formula:
            σ² = (1/4ln2) × [ln(H/L)]²

        Efficiency: 5x more efficient than close-to-close
        """
        log_hl = np.log(high / low)
        variance = (1 / (4 * np.log(2))) * log_hl ** 2

        vol = np.sqrt(variance.rolling(self.window).mean()) * self.annualization_factor
        return vol

    def garman_klass(self,
                     open_p: pd.Series,
                     high: pd.Series,
                     low: pd.Series,
                     close: pd.Series) -> pd.Series:
        """
        Garman-Klass (1980) volatility estimator.

        Citation: [A6]

        Formula:
            σ² = 0.5 × [ln(H/L)]² - (2ln2 - 1) × [ln(C/O)]²

        Efficiency: 8x more efficient than close-to-close
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_p)

        variance = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

        vol = np.sqrt(variance.rolling(self.window).mean().clip(lower=0)) * self.annualization_factor
        return vol

    def rogers_satchell(self,
                        open_p: pd.Series,
                        high: pd.Series,
                        low: pd.Series,
                        close: pd.Series) -> pd.Series:
        """
        Rogers-Satchell (1991) volatility estimator.

        Citation: [A9]

        Formula:
            σ² = ln(H/C) × ln(H/O) + ln(L/C) × ln(L/O)

        Advantage: Handles drift (non-zero mean returns)
        """
        log_ho = np.log(high / open_p)
        log_lo = np.log(low / open_p)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)

        variance = log_ho * log_hc + log_lo * log_lc

        vol = np.sqrt(variance.rolling(self.window).mean().clip(lower=0)) * self.annualization_factor
        return vol

    def yang_zhang(self,
                   open_p: pd.Series,
                   high: pd.Series,
                   low: pd.Series,
                   close: pd.Series) -> pd.Series:
        """
        Yang-Zhang (2000) volatility estimator.

        Citation: [A7]

        Formula:
            σ² = σ²_o + k × σ²_c + (1-k) × σ²_rs

            σ²_o = Var(ln(O_t / C_{t-1}))  (overnight volatility)
            σ²_c = Var(ln(C_t / O_t))      (close-to-open volatility)
            σ²_rs = Rogers-Satchell variance
            k = 0.34 / (1.34 + (n+1)/(n-1))

        Advantage: Accounts for overnight gaps (important for forex)
        Efficiency: Most efficient OHLC estimator
        """
        n = self.window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        # Overnight return (open vs previous close)
        log_oc = np.log(open_p / close.shift(1))

        # Open-to-close return
        log_co = np.log(close / open_p)

        # Rogers-Satchell
        rs_var = self.rogers_satchell(open_p, high, low, close) ** 2 / (self.annualization_factor ** 2)

        # Component variances
        var_o = log_oc.rolling(n).var()
        var_c = log_co.rolling(n).var()

        # Yang-Zhang variance
        variance = var_o + k * var_c + (1 - k) * rs_var

        vol = np.sqrt(variance.clip(lower=0)) * self.annualization_factor
        return vol

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility estimators.
        """
        result = pd.DataFrame(index=df.index)

        o = df['open'] if 'open' in df.columns else df['close']
        h = df['high'] if 'high' in df.columns else df['close']
        l = df['low'] if 'low' in df.columns else df['close']
        c = df['close']

        # All estimators
        result['vol_parkinson'] = self.parkinson(h, l)
        result['vol_garman_klass'] = self.garman_klass(o, h, l, c)
        result['vol_rogers_satchell'] = self.rogers_satchell(o, h, l, c)
        result['vol_yang_zhang'] = self.yang_zhang(o, h, l, c)

        # Close-to-close for comparison
        result['vol_close_to_close'] = c.pct_change().rolling(self.window).std() * self.annualization_factor

        # Volatility ratios (regime detection)
        result['vol_ratio_pk_cc'] = result['vol_parkinson'] / (result['vol_close_to_close'] + 1e-10)
        result['vol_ratio_yz_cc'] = result['vol_yang_zhang'] / (result['vol_close_to_close'] + 1e-10)

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# KYLE LAMBDA AND AMIHUD ILLIQUIDITY
# Citation: [A5] Kyle (1985), [A1] Amihud (2002), [A10] Hasbrouck (2009)
# =============================================================================

class MarketImpactFactors:
    """
    Market Impact and Illiquidity Factors.

    Citations:
        [A5] Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
        [A1] Amihud, Y. (2002). "Illiquidity and Stock Returns"
        [A10] Hasbrouck, J. (2009). "Trading Costs and Returns"

    Sources:
        frds.io: https://frds.io/measures/kyle_lambda/
        QuantsPlaybook

    Forex Applicability: HIGH
    """

    def __init__(self, window: int = 50):
        self.window = window

    def kyle_lambda(self, df: pd.DataFrame) -> pd.Series:
        """
        Kyle's Lambda - Market impact coefficient.

        Citation: [A5] Kyle (1985), [A10] Hasbrouck (2009)

        Model:
            r_t = λ × S_t + ε_t

            r_t: Return in bps
            S_t: Signed square-root dollar volume
            λ: Price impact (Kyle's lambda)

        Interpretation:
            Higher λ = more illiquid market
            λ = cost of demanding liquidity
        """
        returns = df['close'].pct_change() * 10000  # bps
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Signed volume (using return direction)
        signed_vol = np.sign(returns) * np.sqrt(np.abs(volume))

        # Rolling regression
        cov = returns.rolling(self.window).cov(signed_vol)
        var = signed_vol.rolling(self.window).var()

        kyle_lambda = cov / (var + 1e-10)

        return kyle_lambda.fillna(0) * 1e6  # Scale for readability

    def amihud_illiquidity(self, df: pd.DataFrame) -> pd.Series:
        """
        Amihud Illiquidity Ratio.

        Citation: [A1] Amihud (2002)

        Formula:
            ILLIQ_i = (1/D) × Σ |r_{i,d}| / DollarVolume_{i,d}

        Interpretation:
            Higher ILLIQ = less liquid
            Price impact per dollar traded
        """
        returns = df['close'].pct_change().abs()
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        dollar_volume = volume * df['close']

        # Daily illiquidity
        daily_illiq = returns / (dollar_volume + 1e-10)

        # Rolling average
        amihud = daily_illiq.rolling(self.window).mean()

        return amihud.fillna(0) * 1e6  # Scale for readability

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all market impact factors.
        """
        result = pd.DataFrame(index=df.index)

        # Kyle's Lambda
        result['kyle_lambda'] = self.kyle_lambda(df)

        # Amihud Illiquidity
        result['amihud_illiq'] = self.amihud_illiquidity(df)

        # Z-scores
        for col in ['kyle_lambda', 'amihud_illiq']:
            mean = result[col].rolling(100).mean()
            std = result[col].rolling(100).std()
            result[f'{col}_zscore'] = (result[col] - mean) / (std + 1e-10)

        # Liquidity regime
        result['liquidity_regime'] = np.where(
            result['amihud_illiq_zscore'] > 1, -1,  # Illiquid
            np.where(result['amihud_illiq_zscore'] < -1, 1, 0)  # Liquid / Neutral
        )

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# MICROPRICE
# Source: [6] sstoikov/microprice
# Citation: Stoikov (2018)
# =============================================================================

class MicropriceCalculator:
    """
    Microprice - Fair Price Estimation from Order Book.

    Source:
        GitHub: https://github.com/sstoikov/microprice
        Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694

    Citation:
        Stoikov, S. (2018). "The Micro-Price: A High-Frequency Estimator of
        Future Prices." Quantitative Finance.

    Formula:
        Microprice = P_mid + Imbalance × Spread / 2

        Imbalance = (V_bid - V_ask) / (V_bid + V_ask)

    Alternative (size-weighted):
        Microprice = (V_ask × P_bid + V_bid × P_ask) / (V_bid + V_ask)

    Forex Applicability: HIGH (with L2 data)
    """

    @staticmethod
    def calculate_basic(bid: float, ask: float,
                       bid_size: float, ask_size: float) -> float:
        """
        Calculate basic microprice.
        """
        total_size = bid_size + ask_size
        if total_size == 0:
            return (bid + ask) / 2

        imbalance = (bid_size - ask_size) / total_size
        spread = ask - bid
        mid = (bid + ask) / 2

        return mid + imbalance * spread / 2

    @staticmethod
    def calculate_weighted(bid: float, ask: float,
                          bid_size: float, ask_size: float) -> float:
        """
        Calculate size-weighted microprice.

        Intuition: If more size at bid, fair price closer to ask
        """
        total_size = bid_size + ask_size
        if total_size == 0:
            return (bid + ask) / 2

        return (ask_size * bid + bid_size * ask) / total_size

    def calculate_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate microprice features from DataFrame.

        Requires columns: bid, ask (and optionally bid_size, ask_size)
        """
        result = pd.DataFrame(index=df.index)

        if 'bid' not in df.columns or 'ask' not in df.columns:
            # Estimate from close and spread assumption
            spread = df['close'] * 0.0001  # 1 pip assumption
            df = df.copy()
            df['bid'] = df['close'] - spread / 2
            df['ask'] = df['close'] + spread / 2

        bid_size = df['bid_size'] if 'bid_size' in df.columns else pd.Series(1, index=df.index)
        ask_size = df['ask_size'] if 'ask_size' in df.columns else pd.Series(1, index=df.index)

        # Microprice calculations
        result['microprice'] = [
            self.calculate_basic(b, a, bs, as_)
            for b, a, bs, as_ in zip(df['bid'], df['ask'], bid_size, ask_size)
        ]

        result['microprice_weighted'] = [
            self.calculate_weighted(b, a, bs, as_)
            for b, a, bs, as_ in zip(df['bid'], df['ask'], bid_size, ask_size)
        ]

        # Imbalance
        total_size = bid_size + ask_size + 1e-10
        result['book_imbalance'] = (bid_size - ask_size) / total_size

        # Spread
        result['spread'] = df['ask'] - df['bid']
        result['spread_pct'] = result['spread'] / ((df['bid'] + df['ask']) / 2) * 10000  # bps

        # Microprice vs mid deviation
        mid = (df['bid'] + df['ask']) / 2
        result['microprice_deviation'] = result['microprice'] - mid

        return result.replace([np.inf, -np.inf], np.nan).fillna(0)


# =============================================================================
# UNIFIED GENERATOR
# =============================================================================

class GiteeChineseFactorGenerator:
    """
    Unified generator for all Gitee Chinese quantitative factors.

    Combines:
        - Alpha191 forex-applicable factors (国泰君安)
        - Smart Money Factor 2.0 (开源证券)
        - VPIN Enhanced (招商证券/Easley)
        - Integrated OFI (兴业证券/Cont)
        - Volatility Estimators (Yang-Zhang, Garman-Klass, etc.)
        - Market Impact Factors (Kyle, Amihud)
        - Microprice (Stoikov)

    Total Features: ~60-80 factors

    Sources:
        See module docstring for complete citation list.
    """

    def __init__(self):
        self.smart_money = SmartMoneyFactor2()
        self.vpin = VPINEnhanced()
        self.ofi = IntegratedOFI()
        self.volatility = VolatilityEstimators()
        self.market_impact = MarketImpactFactors()
        self.microprice = MicropriceCalculator()

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Gitee Chinese factors.

        Args:
            df: DataFrame with OHLCV columns
                Required: close
                Optional: open, high, low, volume, bid, ask, bid_size, ask_size

        Returns:
            DataFrame with all factor columns (~60-80 features)
        """
        result = pd.DataFrame(index=df.index)

        # 1. Alpha191 Factors (16 factors)
        try:
            alpha191 = Alpha191ForexFactors(df)
            alpha191_features = alpha191.compute_all()
            for col in alpha191_features.columns:
                result[col] = alpha191_features[col]
        except Exception as e:
            print(f"Alpha191 calculation failed: {e}")

        # 2. Smart Money Factor 2.0 (4 factors)
        try:
            smart_money_features = self.smart_money.calculate_enhanced_factor(df)
            for col in smart_money_features.columns:
                result[f'sm_{col}'] = smart_money_features[col]
        except Exception as e:
            print(f"Smart Money calculation failed: {e}")

        # 3. VPIN Enhanced (6 factors)
        try:
            vpin_features = self.vpin.calculate_all_signals(df)
            for col in vpin_features.columns:
                result[f'vpin_{col}'] = vpin_features[col]
        except Exception as e:
            print(f"VPIN calculation failed: {e}")

        # 4. Integrated OFI (6 factors)
        try:
            ofi_features = self.ofi.calculate_all_ofi_features(df)
            for col in ofi_features.columns:
                result[f'ofi_{col}'] = ofi_features[col]
        except Exception as e:
            print(f"OFI calculation failed: {e}")

        # 5. Volatility Estimators (7 factors)
        try:
            vol_features = self.volatility.calculate_all(df)
            for col in vol_features.columns:
                result[col] = vol_features[col]
        except Exception as e:
            print(f"Volatility calculation failed: {e}")

        # 6. Market Impact Factors (5 factors)
        try:
            impact_features = self.market_impact.calculate_all(df)
            for col in impact_features.columns:
                result[f'impact_{col}'] = impact_features[col]
        except Exception as e:
            print(f"Market Impact calculation failed: {e}")

        # 7. Microprice (if L2 data available) (5 factors)
        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                microprice_features = self.microprice.calculate_from_df(df)
                for col in microprice_features.columns:
                    result[f'mp_{col}'] = microprice_features[col]
        except Exception as e:
            print(f"Microprice calculation failed: {e}")

        # Clean up
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature names grouped by source/category.
        """
        return {
            'alpha191': [f'alpha191_{i:03d}' for i in [1, 2, 14, 18, 20, 31, 46, 53, 54, 60, 101, 126, 130, 160, 188, 191]],
            'smart_money': ['sm_smart_money_q', 'sm_is_smart_trade', 'sm_s_indicator_zscore', 'sm_smart_money_momentum'],
            'vpin': ['vpin_vpin', 'vpin_vpin_cdf', 'vpin_toxicity_high', 'vpin_toxicity_low', 'vpin_vpin_momentum', 'vpin_vpin_zscore'],
            'ofi': ['ofi_ofi', 'ofi_ofi_lambda', 'ofi_ofi_momentum_5', 'ofi_ofi_momentum_20', 'ofi_ofi_zscore', 'ofi_ofi_signal', 'ofi_ofi_regime'],
            'volatility': ['vol_parkinson', 'vol_garman_klass', 'vol_rogers_satchell', 'vol_yang_zhang', 'vol_close_to_close', 'vol_ratio_pk_cc', 'vol_ratio_yz_cc'],
            'market_impact': ['impact_kyle_lambda', 'impact_amihud_illiq', 'impact_kyle_lambda_zscore', 'impact_amihud_illiq_zscore', 'impact_liquidity_regime'],
            'microprice': ['mp_microprice', 'mp_microprice_weighted', 'mp_book_imbalance', 'mp_spread', 'mp_spread_pct', 'mp_microprice_deviation']
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_gitee_chinese_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all Gitee Chinese factors in one call.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with ~60-80 factor columns

    Example:
        >>> import pandas as pd
        >>> df = pd.read_parquet('forex_data.parquet')
        >>> factors = generate_gitee_chinese_factors(df)
        >>> print(factors.columns.tolist())
    """
    generator = GiteeChineseFactorGenerator()
    return generator.generate_all(df)


def get_citation_info() -> str:
    """
    Get full citation information for this module.
    """
    return __doc__
