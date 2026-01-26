"""
Ornstein-Uhlenbeck Process for Pairs/Spread Trading
====================================================
Source: Verified via CSDN Chinese quant (九坤投资 style)

The OU process models mean-reverting spreads:
    dS_t = θ(μ - S_t)dt + σdW_t

Where:
    S_t = Spread at time t
    θ (theta) = Mean reversion speed (kappa)
    μ (mu) = Long-term mean
    σ (sigma) = Volatility
    W_t = Wiener process

Key metrics:
    - Half-life: ln(2) / θ (time to revert halfway to mean)
    - Z-score: (S_t - μ) / σ (standardized deviation from mean)

Applications for forex:
    - EURUSD vs GBPUSD (European correlation)
    - AUDUSD vs NZDUSD (Oceania correlation)
    - USDCAD vs USDNOK (Commodity currencies)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class OUSpreadTrading:
    """
    Ornstein-Uhlenbeck Process for Pairs Trading.

    Usage:
        ou = OUSpreadTrading()
        ou.estimate_parameters(spread)
        if ou.half_life() < 30:  # Mean reverts within 30 days
            signals = ou.generate_signals(spread)
    """

    def __init__(self):
        """Initialize OU model."""
        self.theta = None   # Mean reversion speed
        self.mu = None      # Long-term mean
        self.sigma = None   # Volatility

        # Cointegration results
        self.hedge_ratio = None
        self.is_cointegrated = False
        self.adf_pvalue = None

    # =========================================================================
    # PARAMETER ESTIMATION
    # =========================================================================

    def estimate_parameters(
        self,
        spread: pd.Series,
        dt: float = 1/252,
        method: str = 'mle'
    ) -> Dict[str, float]:
        """
        Estimate OU parameters using Maximum Likelihood Estimation.

        The MLE estimates for OU process:
            θ = -log(ρ) / dt  where ρ is AR(1) coefficient
            μ = mean(S)
            σ = std(dS) / sqrt(dt)

        Args:
            spread: Spread time series
            dt: Time step (default 1/252 for daily data)
            method: 'mle' or 'ols'

        Returns:
            Dict with theta, mu, sigma
        """
        spread = spread.dropna()

        if len(spread) < 10:
            self.theta = 0.1
            self.mu = spread.mean() if len(spread) > 0 else 0
            self.sigma = spread.std() if len(spread) > 0 else 1
            return self._get_params_dict()

        # Long-term mean
        self.mu = spread.mean()

        if method == 'mle':
            # MLE estimation via AR(1) regression
            # S_t - S_{t-1} = θ(μ - S_{t-1})dt + σ√dt * ε
            # Rearranging: S_t = (1 - θdt)S_{t-1} + θμdt + σ√dt * ε

            y = spread.values[1:]
            x = spread.values[:-1]

            # AR(1) coefficient estimation
            x_mean = x.mean()
            y_mean = y.mean()

            cov_xy = np.sum((x - x_mean) * (y - y_mean))
            var_x = np.sum((x - x_mean) ** 2)

            if var_x == 0:
                rho = 0.99
            else:
                rho = cov_xy / var_x

            # Bound rho to prevent numerical issues
            rho = np.clip(rho, 0.001, 0.999)

            # theta from rho
            self.theta = -np.log(rho) / dt

            # Residual standard deviation
            residuals = y - rho * x
            self.sigma = residuals.std() / np.sqrt(dt)

        elif method == 'ols':
            # OLS on differences
            dS = spread.diff().dropna()
            S_lag = spread.shift(1).dropna()

            # Align
            min_len = min(len(dS), len(S_lag))
            dS = dS.iloc[:min_len]
            S_lag = S_lag.iloc[:min_len]

            # Regression: dS = θ(μ - S)dt + σε
            #             dS = θμdt - θS*dt + σε
            # As: dS = a + bS + ε where a = θμdt, b = -θdt

            X = np.column_stack([np.ones(len(S_lag)), S_lag.values])
            y = dS.values

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                a, b = beta

                if b < 0:
                    self.theta = -b / dt
                    self.mu = a / (-b) if b != 0 else spread.mean()
                else:
                    # Not mean-reverting
                    self.theta = 0.01
                    self.mu = spread.mean()

                residuals = y - X @ beta
                self.sigma = residuals.std() / np.sqrt(dt)

            except np.linalg.LinAlgError:
                self.theta = 0.1
                self.sigma = spread.std()

        return self._get_params_dict()

    def _get_params_dict(self) -> Dict[str, float]:
        """Get parameters as dictionary."""
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'half_life': self.half_life()
        }

    def fit_mle_optimizer(self, spread: pd.Series, dt: float = 1/252) -> Dict[str, float]:
        """
        Fit OU parameters using numerical MLE optimization.

        More accurate but slower than closed-form estimates.
        """
        spread = spread.dropna().values

        if len(spread) < 10:
            return self._get_params_dict()

        def neg_log_likelihood(params):
            theta, mu, sigma = params
            if theta <= 0 or sigma <= 0:
                return 1e10

            n = len(spread)
            S = spread

            # OU transition density is Normal
            # Mean: S_t * exp(-θdt) + μ(1 - exp(-θdt))
            # Var: σ² / (2θ) * (1 - exp(-2θdt))

            exp_neg_theta_dt = np.exp(-theta * dt)
            exp_neg_2theta_dt = np.exp(-2 * theta * dt)

            mean_next = S[:-1] * exp_neg_theta_dt + mu * (1 - exp_neg_theta_dt)
            var_next = (sigma ** 2) / (2 * theta) * (1 - exp_neg_2theta_dt)

            if var_next <= 0:
                return 1e10

            # Log-likelihood
            ll = -0.5 * (n - 1) * np.log(2 * np.pi * var_next)
            ll -= 0.5 * np.sum((S[1:] - mean_next) ** 2) / var_next

            return -ll

        # Initial guesses
        mu0 = spread.mean()
        sigma0 = np.std(np.diff(spread))
        theta0 = 0.1

        try:
            result = minimize(
                neg_log_likelihood,
                x0=[theta0, mu0, sigma0],
                bounds=[(0.001, 100), (-np.inf, np.inf), (0.001, np.inf)],
                method='L-BFGS-B'
            )

            if result.success:
                self.theta, self.mu, self.sigma = result.x
        except Exception:
            pass

        return self._get_params_dict()

    # =========================================================================
    # HALF-LIFE AND Z-SCORE
    # =========================================================================

    def half_life(self) -> float:
        """
        Compute half-life of mean reversion: ln(2) / θ.

        The half-life is the expected time for the spread to move
        halfway back to its long-term mean.

        Returns:
            Half-life in same units as dt (typically days)
        """
        if self.theta is None or self.theta <= 0:
            return np.inf
        return np.log(2) / self.theta

    def expected_reversion_time(self, target_pct: float = 0.9) -> float:
        """
        Time to revert given percentage toward mean.

        Args:
            target_pct: Percentage reversion (e.g., 0.9 = 90% reversion)

        Returns:
            Expected time in periods
        """
        if self.theta is None or self.theta <= 0:
            return np.inf
        return -np.log(1 - target_pct) / self.theta

    def zscore(self, spread: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Compute z-score for entry/exit signals.

        Z-score = (S - μ) / σ

        Args:
            spread: Spread series
            window: Rolling window for dynamic mean/std. If None, use fitted params.

        Returns:
            Z-score series
        """
        if window is not None:
            # Rolling z-score
            roll_mean = spread.rolling(window, min_periods=1).mean()
            roll_std = spread.rolling(window, min_periods=1).std()
            return (spread - roll_mean) / (roll_std + 1e-10)
        else:
            # Use fitted parameters
            if self.mu is None or self.sigma is None:
                return (spread - spread.mean()) / (spread.std() + 1e-10)
            return (spread - self.mu) / (self.sigma + 1e-10)

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_signals(
        self,
        spread: pd.Series,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0
    ) -> pd.Series:
        """
        Generate trading signals based on z-score thresholds.

        Strategy:
            - Enter short when z > entry_z (spread above mean)
            - Enter long when z < -entry_z (spread below mean)
            - Exit when |z| < exit_z (spread near mean)
            - Stop loss when |z| > stop_z

        Args:
            spread: Spread series
            entry_z: Z-score threshold for entry (default 2.0)
            exit_z: Z-score threshold for exit (default 0.5)
            stop_z: Z-score threshold for stop loss (default 4.0)

        Returns:
            Signal series: 1 (long spread), -1 (short spread), 0 (flat)
        """
        z = self.zscore(spread)

        signals = pd.Series(0, index=spread.index)
        position = 0

        for i in range(len(z)):
            z_val = z.iloc[i]

            if position == 0:
                # No position, look for entry
                if z_val > entry_z:
                    position = -1  # Short spread (expect reversion down)
                elif z_val < -entry_z:
                    position = 1  # Long spread (expect reversion up)

            elif position == 1:
                # Long position, look for exit
                if abs(z_val) < exit_z:
                    position = 0  # Exit near mean
                elif z_val < -stop_z:
                    position = 0  # Stop loss

            elif position == -1:
                # Short position, look for exit
                if abs(z_val) < exit_z:
                    position = 0  # Exit near mean
                elif z_val > stop_z:
                    position = 0  # Stop loss

            signals.iloc[i] = position

        return signals

    def generate_continuous_signals(
        self,
        spread: pd.Series,
        max_position: float = 1.0
    ) -> pd.Series:
        """
        Generate continuous position signals based on z-score.

        Position size proportional to z-score deviation.
        Larger deviation = larger position (up to max_position).

        Args:
            spread: Spread series
            max_position: Maximum position size

        Returns:
            Continuous signal series between -max_position and +max_position
        """
        z = self.zscore(spread)

        # Sigmoid-like scaling: position = -tanh(z) * max_position
        # Negative because we short when z is high (spread above mean)
        signals = -np.tanh(z) * max_position

        return signals

    # =========================================================================
    # COINTEGRATION AND HEDGE RATIO
    # =========================================================================

    def compute_hedge_ratio(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        method: str = 'ols'
    ) -> float:
        """
        Compute optimal hedge ratio for spread construction.

        Spread = asset1 - hedge_ratio * asset2

        Args:
            asset1: First asset price series
            asset2: Second asset price series
            method: 'ols', 'tls' (total least squares), or 'rolling'

        Returns:
            Optimal hedge ratio
        """
        # Align series
        asset1 = asset1.dropna()
        asset2 = asset2.dropna()
        common_idx = asset1.index.intersection(asset2.index)
        asset1 = asset1.loc[common_idx]
        asset2 = asset2.loc[common_idx]

        if len(asset1) < 10:
            self.hedge_ratio = 1.0
            return self.hedge_ratio

        if method == 'ols':
            # OLS: asset1 = alpha + beta * asset2
            X = np.column_stack([np.ones(len(asset2)), asset2.values])
            y = asset1.values

            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                self.hedge_ratio = beta[1]
            except np.linalg.LinAlgError:
                self.hedge_ratio = 1.0

        elif method == 'tls':
            # Total Least Squares (Deming regression)
            # More robust when both series have measurement error
            x = asset2.values
            y = asset1.values

            x_mean = x.mean()
            y_mean = y.mean()

            sxx = np.sum((x - x_mean) ** 2)
            syy = np.sum((y - y_mean) ** 2)
            sxy = np.sum((x - x_mean) * (y - y_mean))

            # TLS solution
            self.hedge_ratio = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy + 1e-10)

        elif method == 'rolling':
            # Use last 60 periods for rolling estimate
            window = min(60, len(asset1) // 2)
            x = asset2.iloc[-window:].values
            y = asset1.iloc[-window:].values

            X = np.column_stack([np.ones(len(x)), x])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                self.hedge_ratio = beta[1]
            except:
                self.hedge_ratio = 1.0

        return self.hedge_ratio

    def construct_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Construct spread from two asset series.

        Spread = asset1 - hedge_ratio * asset2

        Args:
            asset1: First asset series
            asset2: Second asset series
            hedge_ratio: If None, compute optimal ratio

        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self.compute_hedge_ratio(asset1, asset2)

        # Align series
        common_idx = asset1.index.intersection(asset2.index)
        return asset1.loc[common_idx] - hedge_ratio * asset2.loc[common_idx]

    def adf_test(self, spread: pd.Series, regression: str = 'c') -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity (mean-reversion).

        Null hypothesis: spread has a unit root (not stationary)
        Low p-value = reject null = spread is stationary = mean-reverting

        Args:
            spread: Spread series
            regression: 'c' (constant), 'ct' (constant+trend), 'ctt', 'n'

        Returns:
            Dict with test statistic, p-value, critical values
        """
        spread = spread.dropna()

        if len(spread) < 20:
            return {
                'statistic': 0,
                'pvalue': 1.0,
                'critical_values': {},
                'is_stationary': False
            }

        # ADF test implementation (simplified)
        # Full version would use statsmodels.tsa.stattools.adfuller

        # Regression: ΔS_t = α + βS_{t-1} + Σγ_i ΔS_{t-i} + ε_t

        dS = spread.diff().dropna()
        S_lag = spread.shift(1).dropna()

        # Align
        min_len = min(len(dS), len(S_lag))
        dS = dS.iloc[:min_len].values
        S_lag = S_lag.iloc[:min_len].values

        # Simple OLS for ADF
        if regression == 'c':
            X = np.column_stack([np.ones(len(S_lag)), S_lag])
        else:
            X = np.column_stack([np.ones(len(S_lag)), np.arange(len(S_lag)), S_lag])

        try:
            beta = np.linalg.lstsq(X, dS, rcond=None)[0]
            residuals = dS - X @ beta

            # t-statistic for beta coefficient on S_lag
            if regression == 'c':
                beta_S = beta[1]
                idx = 1
            else:
                beta_S = beta[2]
                idx = 2

            XtX_inv = np.linalg.inv(X.T @ X)
            sigma_sq = np.sum(residuals ** 2) / (len(dS) - len(beta))
            se = np.sqrt(sigma_sq * XtX_inv[idx, idx])

            adf_stat = beta_S / (se + 1e-10)

            # Critical values (approximate for n=250)
            critical_values = {
                '1%': -3.43,
                '5%': -2.86,
                '10%': -2.57
            }

            # Approximate p-value
            if adf_stat < -3.43:
                pvalue = 0.01
            elif adf_stat < -2.86:
                pvalue = 0.05
            elif adf_stat < -2.57:
                pvalue = 0.10
            else:
                pvalue = 0.5

            self.adf_pvalue = pvalue
            self.is_cointegrated = pvalue < 0.05

            return {
                'statistic': adf_stat,
                'pvalue': pvalue,
                'critical_values': critical_values,
                'is_stationary': pvalue < 0.05
            }

        except Exception:
            return {
                'statistic': 0,
                'pvalue': 1.0,
                'critical_values': {},
                'is_stationary': False
            }

    def johansen_test(
        self,
        asset1: pd.Series,
        asset2: pd.Series
    ) -> Dict:
        """
        Simplified Johansen cointegration test for two series.

        Tests for cointegrating relationship between asset1 and asset2.

        Returns:
            Dict with test results
        """
        # For two series, we test if there's a cointegrating vector
        # This is a simplified version - full Johansen requires matrix eigenvalue analysis

        # Method: Engle-Granger two-step approach
        # 1. Regress asset1 on asset2 to get residuals
        # 2. Test residuals for stationarity

        hedge_ratio = self.compute_hedge_ratio(asset1, asset2)
        spread = self.construct_spread(asset1, asset2, hedge_ratio)
        adf_result = self.adf_test(spread)

        return {
            'hedge_ratio': hedge_ratio,
            'spread_adf_stat': adf_result['statistic'],
            'spread_pvalue': adf_result['pvalue'],
            'is_cointegrated': adf_result['is_stationary']
        }

    # =========================================================================
    # SIMULATION AND BACKTESTING
    # =========================================================================

    def simulate_ou(
        self,
        n_steps: int,
        dt: float = 1/252,
        S0: Optional[float] = None
    ) -> pd.Series:
        """
        Simulate OU process path.

        Uses exact discretization:
            S_t = S_{t-1} * exp(-θdt) + μ(1 - exp(-θdt)) + σ√((1-exp(-2θdt))/(2θ)) * Z

        Args:
            n_steps: Number of time steps
            dt: Time step size
            S0: Initial value (default: mu)

        Returns:
            Simulated spread series
        """
        if self.theta is None or self.mu is None or self.sigma is None:
            raise ValueError("Model not fitted. Call estimate_parameters first.")

        if S0 is None:
            S0 = self.mu

        S = np.zeros(n_steps)
        S[0] = S0

        exp_neg_theta_dt = np.exp(-self.theta * dt)
        mean_reversion = self.mu * (1 - exp_neg_theta_dt)
        vol_factor = self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta))

        for t in range(1, n_steps):
            Z = np.random.randn()
            S[t] = S[t-1] * exp_neg_theta_dt + mean_reversion + vol_factor * Z

        return pd.Series(S)

    def backtest_strategy(
        self,
        spread: pd.Series,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        transaction_cost: float = 0.0001
    ) -> Dict:
        """
        Backtest pairs trading strategy.

        Args:
            spread: Spread series
            entry_z: Z-score entry threshold
            exit_z: Z-score exit threshold
            transaction_cost: Cost per trade as fraction

        Returns:
            Dict with performance metrics
        """
        signals = self.generate_signals(spread, entry_z, exit_z)

        # Calculate returns
        spread_returns = spread.pct_change().fillna(0)

        # Strategy returns
        strat_returns = signals.shift(1) * spread_returns

        # Transaction costs
        trades = signals.diff().abs().fillna(0)
        costs = trades * transaction_cost

        net_returns = strat_returns - costs

        # Metrics
        total_return = (1 + net_returns).prod() - 1
        sharpe = net_returns.mean() / (net_returns.std() + 1e-10) * np.sqrt(252)
        max_dd = (net_returns.cumsum() - net_returns.cumsum().cummax()).min()
        win_rate = (net_returns > 0).sum() / ((net_returns != 0).sum() + 1e-10)
        num_trades = (trades > 0).sum()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'avg_return_per_trade': total_return / (num_trades + 1e-10)
        }

    # =========================================================================
    # FOREX-SPECIFIC PAIRS
    # =========================================================================

    @staticmethod
    def get_forex_pairs_candidates() -> List[Tuple[str, str, str]]:
        """
        Get list of candidate forex pairs for pairs trading.

        Returns:
            List of (pair1, pair2, rationale) tuples
        """
        return [
            ('EURUSD', 'GBPUSD', 'European correlation'),
            ('AUDUSD', 'NZDUSD', 'Oceania correlation'),
            ('USDCAD', 'USDNOK', 'Commodity currencies'),
            ('EURUSD', 'EURGBP', 'EUR base pairs'),
            ('USDJPY', 'EURJPY', 'JPY cross correlation'),
            ('AUDUSD', 'AUDCAD', 'AUD cross correlation'),
            ('GBPUSD', 'GBPJPY', 'GBP base pairs'),
            ('USDCHF', 'USDJPY', 'Safe haven correlation'),
        ]


class PairsTradingSystem:
    """
    Complete pairs trading system using OU process.

    Manages multiple pairs, automatic pair selection, and portfolio allocation.
    """

    def __init__(
        self,
        max_half_life: float = 30,
        min_half_life: float = 1,
        adf_threshold: float = 0.05
    ):
        """
        Initialize pairs trading system.

        Args:
            max_half_life: Maximum acceptable half-life (days)
            min_half_life: Minimum half-life (avoid noise)
            adf_threshold: P-value threshold for cointegration
        """
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.adf_threshold = adf_threshold

        self.pairs = {}  # Dict of pair_name -> OUSpreadTrading

    def scan_pairs(
        self,
        price_data: Dict[str, pd.Series],
        candidate_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Scan candidate pairs for tradeable relationships.

        Args:
            price_data: Dict of symbol -> price series
            candidate_pairs: List of (symbol1, symbol2) pairs to test

        Returns:
            DataFrame with pair statistics
        """
        if candidate_pairs is None:
            # Generate all combinations
            symbols = list(price_data.keys())
            candidate_pairs = [
                (symbols[i], symbols[j])
                for i in range(len(symbols))
                for j in range(i + 1, len(symbols))
            ]

        results = []

        for sym1, sym2 in candidate_pairs:
            if sym1 not in price_data or sym2 not in price_data:
                continue

            asset1 = price_data[sym1]
            asset2 = price_data[sym2]

            ou = OUSpreadTrading()

            try:
                # Test cointegration
                coint_result = ou.johansen_test(asset1, asset2)

                # Construct spread and estimate OU params
                spread = ou.construct_spread(asset1, asset2, coint_result['hedge_ratio'])
                ou.estimate_parameters(spread)

                half_life = ou.half_life()

                results.append({
                    'pair': f"{sym1}/{sym2}",
                    'symbol1': sym1,
                    'symbol2': sym2,
                    'hedge_ratio': coint_result['hedge_ratio'],
                    'adf_stat': coint_result['spread_adf_stat'],
                    'pvalue': coint_result['spread_pvalue'],
                    'is_cointegrated': coint_result['is_cointegrated'],
                    'theta': ou.theta,
                    'mu': ou.mu,
                    'sigma': ou.sigma,
                    'half_life': half_life,
                    'is_tradeable': (
                        coint_result['is_cointegrated'] and
                        self.min_half_life < half_life < self.max_half_life
                    )
                })

                if results[-1]['is_tradeable']:
                    self.pairs[f"{sym1}/{sym2}"] = ou

            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_signals(
        self,
        price_data: Dict[str, pd.Series],
        entry_z: float = 2.0,
        exit_z: float = 0.5
    ) -> pd.DataFrame:
        """
        Get trading signals for all registered pairs.

        Returns:
            DataFrame with signals for each pair
        """
        signals = {}

        for pair_name, ou in self.pairs.items():
            sym1, sym2 = pair_name.split('/')

            if sym1 in price_data and sym2 in price_data:
                spread = ou.construct_spread(price_data[sym1], price_data[sym2])
                signals[pair_name] = ou.generate_signals(spread, entry_z, exit_z)

        return pd.DataFrame(signals)


if __name__ == '__main__':
    # Test
    print("OU Spread Trading - Testing")
    print("=" * 50)

    # Create sample correlated series
    np.random.seed(42)
    n = 500

    # Simulate two cointegrated series
    # Common factor + idiosyncratic noise
    common = np.cumsum(np.random.randn(n) * 0.01)
    asset1 = pd.Series(100 + common + np.random.randn(n) * 0.5)
    asset2 = pd.Series(80 + common * 0.8 + np.random.randn(n) * 0.5)

    # Initialize OU model
    ou = OUSpreadTrading()

    # Compute hedge ratio and construct spread
    hedge_ratio = ou.compute_hedge_ratio(asset1, asset2)
    print(f"Hedge Ratio: {hedge_ratio:.4f}")

    spread = ou.construct_spread(asset1, asset2)

    # Test cointegration
    adf_result = ou.adf_test(spread)
    print(f"ADF Statistic: {adf_result['statistic']:.4f}")
    print(f"P-value: {adf_result['pvalue']:.4f}")
    print(f"Is Stationary: {adf_result['is_stationary']}")

    # Estimate OU parameters
    params = ou.estimate_parameters(spread)
    print(f"\nOU Parameters:")
    print(f"  Theta (mean reversion speed): {params['theta']:.4f}")
    print(f"  Mu (long-term mean): {params['mu']:.4f}")
    print(f"  Sigma (volatility): {params['sigma']:.4f}")
    print(f"  Half-life: {params['half_life']:.2f} periods")

    # Generate signals
    signals = ou.generate_signals(spread)
    print(f"\nSignal Distribution:")
    print(signals.value_counts())

    # Backtest
    backtest = ou.backtest_strategy(spread)
    print(f"\nBacktest Results:")
    print(f"  Total Return: {backtest['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest['max_drawdown']:.2%}")
    print(f"  Win Rate: {backtest['win_rate']:.2%}")
    print(f"  Num Trades: {backtest['num_trades']}")

    # Simulate OU path
    sim_spread = ou.simulate_ou(100)
    print(f"\nSimulated spread stats:")
    print(f"  Mean: {sim_spread.mean():.4f} (expected: {ou.mu:.4f})")
    print(f"  Std: {sim_spread.std():.4f}")
