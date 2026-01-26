"""
Universal Mathematical Finance Factors
======================================
Cross-regional mathematical formulas used across all quantitative finance.

These are foundational mathematical models that transcend regional boundaries
and are used by quantitative traders worldwide.

Sources:
1. arXiv:1811.09312 - "Statistical Arbitrage with Pairs Trading"
   - Ornstein-Uhlenbeck process for mean reversion
   - Optimal entry/exit thresholds

2. Engle-Granger (1987) - "Cointegration and Error Correction"
   Nobel Prize-winning methodology for pairs trading

3. Avellaneda & Stoikov (2008) - "High-frequency trading in a limit order book"
   - Market making optimal quotes
   - Inventory risk management

4. Kelly (1956) - "A New Interpretation of Information Rate"
   - Optimal position sizing
   - Growth rate maximization

5. arXiv:2404.00424 - "Quantformer: Transformer-based Trading" (2024)
   - Attention mechanisms for time series
   - Feature importance weighting

6. Lopez de Prado (2018) - "Advances in Financial Machine Learning"
   - Triple barrier labeling
   - Feature importance (MDA, MDI)
   - Structural breaks detection

7. Hamilton (1989) - "Regime Switching Models"
   - Markov regime detection
   - State probability estimation

Total: 30 factors organized into:
- Ornstein-Uhlenbeck (5): Mean reversion spread trading
- Cointegration (5): Statistical arbitrage signals
- Market Making (5): Avellaneda-Stoikov framework
- Kelly Criterion (5): Position sizing signals
- Attention Weights (5): Transformer-inspired features
- Regime Detection (5): Markov switching signals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
from scipy import stats, optimize
from scipy.special import expit  # Sigmoid function

warnings.filterwarnings('ignore')


class UniversalMathFeatures:
    """
    Universal Mathematical Finance Features.

    Foundational quantitative models used worldwide:
    - Ornstein-Uhlenbeck mean reversion
    - Engle-Granger cointegration
    - Avellaneda-Stoikov market making
    - Kelly criterion position sizing
    - Attention-based feature weighting
    - Markov regime detection

    These models form the mathematical backbone of systematic trading
    across all asset classes and regions.

    Usage:
        features = UniversalMathFeatures()
        df_features = features.generate_all(ohlcv_df)
    """

    def __init__(
        self,
        ou_half_life: int = 20,        # OU process half-life in periods
        kelly_fraction: float = 0.25,   # Kelly fraction for position sizing
        regime_lookback: int = 60,      # Lookback for regime detection
    ):
        """
        Initialize Universal Math Features.

        Args:
            ou_half_life: Half-life for OU process (default 20)
            kelly_fraction: Fractional Kelly for sizing (default 0.25)
            regime_lookback: Lookback for regime detection (default 60)
        """
        self.ou_half_life = ou_half_life
        self.kelly_frac = kelly_fraction
        self.regime_lookback = regime_lookback

    # =========================================================================
    # ORNSTEIN-UHLENBECK (5) - Mean Reversion Spread Trading
    # =========================================================================

    def _ou_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ornstein-Uhlenbeck Process Features.

        The OU process models mean-reverting spreads:
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t

        Parameters:
        - theta: Speed of mean reversion
        - mu: Long-run mean
        - sigma: Volatility

        Trading signals:
        - Deviation from mean (z-score)
        - Expected time to reversion
        - Optimal entry/exit thresholds

        References:
        - arXiv:1811.09312: "Statistical Arbitrage with Pairs Trading"
        - Hudson & Thames: "Mean Reversion Module"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. OU spread: deviation from rolling mean
        # This is the "spread" in pairs trading terminology
        mu = close.rolling(self.ou_half_life * 2, min_periods=10).mean()
        spread = close - mu
        std = close.rolling(self.ou_half_life * 2, min_periods=10).std()
        features['UNI_OU_ZSCORE'] = spread / (std + 1e-12)

        # 2. Mean reversion speed (theta) estimate
        # theta = -log(autocorr) / dt
        # Higher theta = faster mean reversion
        def estimate_theta(series, window):
            result = np.zeros(len(series))
            for i in range(window, len(series)):
                subseries = series.iloc[i-window:i]
                if len(subseries) > 10:
                    autocorr = subseries.autocorr(lag=1)
                    if autocorr > 0 and autocorr < 1:
                        result[i] = -np.log(autocorr)
                    else:
                        result[i] = 0
            return pd.Series(result, index=series.index)

        theta = estimate_theta(spread, 30)
        features['UNI_OU_THETA'] = theta

        # 3. Half-life of mean reversion
        # t_half = ln(2) / theta
        features['UNI_OU_HALFLIFE'] = np.log(2) / (theta + 1e-12)
        features['UNI_OU_HALFLIFE'] = features['UNI_OU_HALFLIFE'].clip(-100, 100)

        # 4. OU trading signal: entry when z-score is extreme
        # Standard entry: |z| > 2, exit when |z| < 0.5
        z = features['UNI_OU_ZSCORE']
        features['UNI_OU_ENTRY'] = np.where(z > 2, -1,
                                            np.where(z < -2, 1, 0))

        # 5. Expected profit from mean reversion
        # E[profit] = spread * (1 - exp(-theta * t))
        expected_reversion = spread * (1 - np.exp(-theta.clip(0, 1) * 5))
        features['UNI_OU_EXPECTED'] = expected_reversion / (std + 1e-12)

        return features

    # =========================================================================
    # COINTEGRATION (5) - Engle-Granger Framework
    # =========================================================================

    def _cointegration_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cointegration-Based Features.

        Engle-Granger (1987) cointegration for single series:
        - Error correction model
        - Long-run equilibrium deviation
        - Speed of adjustment

        For single series, we use self-cointegration with lagged values.

        References:
        - Engle & Granger (1987): "Co-integration and Error Correction"
        - Nobel Prize in Economics 2003
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Log price for cointegration analysis
        log_price = np.log(close + 1)

        # 2. Error correction term: deviation from trend
        # Using Hodrick-Prescott-like decomposition
        trend = log_price.rolling(120, min_periods=20).mean()
        error = log_price - trend
        features['UNI_COINT_ERROR'] = error

        # 3. Speed of adjustment: how fast errors correct
        # Delta(price) = alpha * error(-1) + ...
        error_lag = error.shift(1)
        delta_price = log_price.diff()

        def rolling_regression_coef(y, x, window):
            result = np.zeros(len(y))
            for i in range(window, len(y)):
                y_sub = y.iloc[i-window:i].dropna()
                x_sub = x.iloc[i-window:i].dropna()
                if len(y_sub) > 10 and len(x_sub) > 10:
                    try:
                        slope, _, _, _, _ = stats.linregress(x_sub, y_sub)
                        result[i] = slope
                    except:
                        result[i] = 0
            return pd.Series(result, index=y.index)

        alpha = rolling_regression_coef(delta_price, error_lag, 60)
        features['UNI_COINT_ALPHA'] = alpha

        # 4. Cointegration strength: R-squared of error correction
        def rolling_r2(y, x, window):
            result = np.zeros(len(y))
            for i in range(window, len(y)):
                y_sub = y.iloc[i-window:i].dropna()
                x_sub = x.iloc[i-window:i].dropna()
                if len(y_sub) > 10 and len(x_sub) > 10:
                    try:
                        _, _, r, _, _ = stats.linregress(x_sub, y_sub)
                        result[i] = r ** 2
                    except:
                        result[i] = 0
            return pd.Series(result, index=y.index)

        r2 = rolling_r2(delta_price, error_lag, 60)
        features['UNI_COINT_R2'] = r2

        # 5. Disequilibrium signal: large errors predict correction
        error_zscore = error / (error.rolling(60, min_periods=10).std() + 1e-12)
        features['UNI_COINT_SIGNAL'] = -error_zscore  # Contrarian

        return features

    # =========================================================================
    # MARKET MAKING (5) - Avellaneda-Stoikov Framework
    # =========================================================================

    def _market_making_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Avellaneda-Stoikov Market Making Features.

        Optimal market making quotes:
        bid = S - spread/2, ask = S + spread/2
        spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)

        Parameters:
        - gamma: Risk aversion
        - sigma: Volatility
        - k: Order arrival rate
        - T: Time remaining

        References:
        - Avellaneda & Stoikov (2008): "HFT in a limit order book"
        - Cartea et al (2015): "Algorithmic and HF Trading"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        returns = close.pct_change()

        # 1. Volatility for spread calculation
        sigma = returns.rolling(20, min_periods=5).std()
        features['UNI_MM_VOL'] = sigma

        # 2. Optimal spread proxy: volatility-based
        # spread ~ gamma * sigma^2
        gamma = 0.1  # Risk aversion parameter
        optimal_spread = gamma * (sigma ** 2) * 252  # Annualized
        features['UNI_MM_SPREAD'] = optimal_spread

        # 3. Inventory imbalance: accumulated signed flow
        signed_returns = np.sign(returns) * returns.abs()
        inventory = signed_returns.rolling(20, min_periods=1).sum()
        features['UNI_MM_INV'] = inventory

        # 4. Inventory skew adjustment: quotes adjusted for inventory
        # When long inventory, lower bid (reduce buying)
        skew = -inventory / (sigma + 1e-12)
        features['UNI_MM_SKEW'] = skew

        # 5. Market making profitability: realized spread proxy
        # Half-spread = (high - low) / (2 * close)
        half_spread = (high - low) / (2 * close + 1e-12)
        # Subtract volatility cost
        mm_pnl = half_spread - sigma
        features['UNI_MM_PNL'] = mm_pnl.rolling(20, min_periods=1).mean()

        return features

    # =========================================================================
    # KELLY CRITERION (5) - Optimal Position Sizing
    # =========================================================================

    def _kelly_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kelly Criterion Position Sizing Features.

        Kelly (1956) optimal bet sizing:
        f* = (p * b - q) / b = edge / odds

        Where:
        - p: Win probability
        - q: Loss probability (1 - p)
        - b: Win/loss ratio (odds)
        - f*: Optimal fraction of capital

        Fractional Kelly (25%) is used in practice for smoother equity curves.

        References:
        - Kelly (1956): "A New Interpretation of Information Rate"
        - Thorp (2006): "The Kelly Criterion in Blackjack"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Win probability: fraction of positive returns
        win_prob = (returns > 0).rolling(60, min_periods=20).mean()
        features['UNI_KELLY_WINP'] = win_prob

        # 2. Win/loss ratio: average win / average loss
        wins = returns.where(returns > 0, np.nan)
        losses = returns.where(returns < 0, np.nan).abs()
        avg_win = wins.rolling(60, min_periods=10).mean()
        avg_loss = losses.rolling(60, min_periods=10).mean()
        win_loss_ratio = avg_win / (avg_loss + 1e-12)
        features['UNI_KELLY_RATIO'] = win_loss_ratio

        # 3. Full Kelly fraction: f* = (p * b - q) / b
        q = 1 - win_prob
        full_kelly = (win_prob * win_loss_ratio - q) / (win_loss_ratio + 1e-12)
        features['UNI_KELLY_FULL'] = full_kelly

        # 4. Fractional Kelly: conservative sizing
        frac_kelly = full_kelly * self.kelly_frac
        frac_kelly = frac_kelly.clip(-1, 1)  # Cap at 100% of capital
        features['UNI_KELLY_FRAC'] = frac_kelly

        # 5. Kelly edge: expected edge per trade
        edge = win_prob * avg_win - q * avg_loss
        features['UNI_KELLY_EDGE'] = edge

        return features

    # =========================================================================
    # ATTENTION WEIGHTS (5) - Transformer-Inspired Features
    # =========================================================================

    def _attention_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attention-Based Feature Weighting.

        Inspired by Transformer architecture (Vaswani et al 2017) and
        Quantformer (arXiv:2404.00424):
        - Self-attention: relative importance of past observations
        - Feature importance: which features matter most recently

        Simplified implementation without full transformer:
        - Attention weights based on similarity to recent pattern
        - Weighted moving averages

        References:
        - arXiv:2404.00424: "Quantformer: Transformer-based Trading" (2024)
        - Vaswani et al (2017): "Attention Is All You Need"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Self-attention: similarity of current return to past returns
        def attention_weights(series, query_len=5, key_len=20):
            """Compute attention-weighted average."""
            result = np.zeros(len(series))
            for i in range(key_len + query_len, len(series)):
                query = series.iloc[i-query_len:i].values
                keys = series.iloc[i-key_len-query_len:i-query_len].values

                # Reshape for attention
                if len(query) < query_len or len(keys) < key_len:
                    continue

                # Simple dot-product attention
                # Score = softmax(Q . K / sqrt(d))
                scores = np.zeros(key_len)
                for j in range(key_len - query_len + 1):
                    key_segment = keys[j:j+query_len]
                    score = np.dot(query, key_segment) / (np.linalg.norm(query) * np.linalg.norm(key_segment) + 1e-12)
                    scores[j] = score

                # Softmax
                scores = np.exp(scores - np.max(scores))
                weights = scores / (scores.sum() + 1e-12)

                # Weighted value
                values = series.iloc[i-key_len:i].values[-len(weights):]
                if len(values) == len(weights):
                    result[i] = np.dot(weights, values)

            return pd.Series(result, index=series.index)

        attention_ret = attention_weights(returns.fillna(0), 5, 20)
        features['UNI_ATTN_RET'] = attention_ret

        # 2. Attention to volatility regime
        vol = returns.rolling(5, min_periods=2).std()
        attention_vol = attention_weights(vol.fillna(0), 5, 20)
        features['UNI_ATTN_VOL'] = attention_vol

        # 3. Momentum attention: weighted by recency
        # Exponential weighting (recent matters more)
        weights = np.exp(-0.1 * np.arange(20))[::-1]
        weights = weights / weights.sum()

        def exp_weighted_ma(series, w):
            result = np.zeros(len(series))
            for i in range(len(w), len(series)):
                result[i] = np.dot(series.iloc[i-len(w):i].values, w)
            return pd.Series(result, index=series.index)

        features['UNI_ATTN_MOM'] = exp_weighted_ma(returns.fillna(0), weights)

        # 4. Price level attention: focus on extreme prices
        price_zscore = (close - close.rolling(60, min_periods=10).mean()) / (close.rolling(60, min_periods=10).std() + 1e-12)
        # Attention higher at extremes
        extreme_attention = np.abs(price_zscore)
        features['UNI_ATTN_EXTREME'] = extreme_attention

        # 5. Combined attention signal
        features['UNI_ATTN_SIGNAL'] = (
            features['UNI_ATTN_RET'] * 0.4 +
            features['UNI_ATTN_MOM'] * 0.4 +
            features['UNI_ATTN_VOL'] * 0.2
        )

        return features

    # =========================================================================
    # REGIME DETECTION (5) - Markov Switching
    # =========================================================================

    def _regime_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Markov Regime Switching Features.

        Hamilton (1989) regime switching model:
        - Multiple market regimes (bull/bear, high/low vol)
        - Probability of being in each regime
        - Transition probabilities

        Simplified implementation using:
        - Volatility regime detection
        - Trend regime detection
        - Combined regime indicator

        References:
        - Hamilton (1989): "A New Approach to Economic Analysis
          of Nonstationary Time Series and the Business Cycle"
        - Guidolin (2011): "Markov Switching Models in Empirical Finance"
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Volatility regime: high vs low vol
        vol = returns.rolling(20, min_periods=5).std()
        vol_median = vol.rolling(self.regime_lookback, min_periods=20).median()
        vol_regime = np.where(vol > vol_median, 1, 0)  # 1 = high vol
        features['UNI_REG_VOL'] = vol_regime

        # 2. Trend regime: bull vs bear
        ma_short = close.rolling(10, min_periods=1).mean()
        ma_long = close.rolling(50, min_periods=10).mean()
        trend_regime = np.where(ma_short > ma_long, 1, -1)  # 1 = bull, -1 = bear
        features['UNI_REG_TREND'] = trend_regime

        # 3. Regime probability: smooth transition
        # Use sigmoid of z-score as probability proxy
        vol_zscore = (vol - vol.rolling(60, min_periods=10).mean()) / (vol.rolling(60, min_periods=10).std() + 1e-12)
        vol_prob = expit(vol_zscore)  # P(high vol regime)
        features['UNI_REG_VOL_PROB'] = vol_prob

        # 4. Regime persistence: how long in current regime
        def regime_duration(regime_series):
            """Count consecutive periods in same regime."""
            result = np.zeros(len(regime_series))
            count = 1
            for i in range(1, len(regime_series)):
                if regime_series.iloc[i] == regime_series.iloc[i-1]:
                    count += 1
                else:
                    count = 1
                result[i] = count
            return pd.Series(result, index=regime_series.index)

        duration = regime_duration(pd.Series(trend_regime, index=df.index))
        features['UNI_REG_DURATION'] = duration

        # 5. Combined regime indicator
        # 2 = bull + high vol (risky rally)
        # 1 = bull + low vol (calm uptrend)
        # -1 = bear + low vol (calm downtrend)
        # -2 = bear + high vol (crisis)
        combined = trend_regime * (1 + vol_regime)
        features['UNI_REG_COMBINED'] = combined

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all Universal Math features.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with 30 factor columns
        """
        # Ensure required columns
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        # Fill missing OHLC from close
        df = df.copy()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        # Generate all factor groups
        ou = self._ou_factors(df)
        coint = self._cointegration_factors(df)
        mm = self._market_making_factors(df)
        kelly = self._kelly_factors(df)
        attention = self._attention_factors(df)
        regime = self._regime_factors(df)

        # Combine all features
        features = pd.concat([
            ou, coint, mm, kelly, attention, regime
        ], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []

        # OU (5)
        names.extend(['UNI_OU_ZSCORE', 'UNI_OU_THETA', 'UNI_OU_HALFLIFE',
                      'UNI_OU_ENTRY', 'UNI_OU_EXPECTED'])

        # Cointegration (5)
        names.extend(['UNI_COINT_ERROR', 'UNI_COINT_ALPHA', 'UNI_COINT_R2',
                      'UNI_COINT_SIGNAL', 'UNI_COINT_SIGNAL'])

        # Market Making (5)
        names.extend(['UNI_MM_VOL', 'UNI_MM_SPREAD', 'UNI_MM_INV',
                      'UNI_MM_SKEW', 'UNI_MM_PNL'])

        # Kelly (5)
        names.extend(['UNI_KELLY_WINP', 'UNI_KELLY_RATIO', 'UNI_KELLY_FULL',
                      'UNI_KELLY_FRAC', 'UNI_KELLY_EDGE'])

        # Attention (5)
        names.extend(['UNI_ATTN_RET', 'UNI_ATTN_VOL', 'UNI_ATTN_MOM',
                      'UNI_ATTN_EXTREME', 'UNI_ATTN_SIGNAL'])

        # Regime (5)
        names.extend(['UNI_REG_VOL', 'UNI_REG_TREND', 'UNI_REG_VOL_PROB',
                      'UNI_REG_DURATION', 'UNI_REG_COMBINED'])

        return names

    def get_factor_category(self, factor_name: str) -> str:
        """Get the category of a factor by name."""
        if 'OU_' in factor_name:
            return 'Ornstein-Uhlenbeck'
        elif 'COINT' in factor_name:
            return 'Cointegration'
        elif 'MM_' in factor_name:
            return 'Market Making'
        elif 'KELLY' in factor_name:
            return 'Kelly Criterion'
        elif 'ATTN' in factor_name:
            return 'Attention Weights'
        elif 'REG_' in factor_name:
            return 'Regime Detection'
        return 'Unknown'


# Convenience function
def generate_universal_math_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Universal Mathematical features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 30 mathematical finance factors
    """
    features = UniversalMathFeatures()
    return features.generate_all(df)
