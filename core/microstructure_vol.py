"""
Microstructure Volatility and Noise Filtering
==============================================
Implements volatility estimation that accounts for market microstructure noise.

Key methods:
- Two Scales Realized Volatility (TSRV)
- Bid-ask bounce filtering
- Optimal sampling interval
- Noise variance estimation

Source: Aït-Sahalia et al. "Ultra High Frequency Volatility Estimation"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MicrostructureVolatility:
    """
    Microstructure-Aware Volatility Estimator.

    Standard volatility measures are biased at high frequencies due to:
    - Bid-ask bounce
    - Price discreteness
    - Asynchronous quotes
    - Quote stuffing

    This class implements noise-robust volatility estimation.

    Usage:
        mv = MicrostructureVolatility()
        true_vol = mv.tsrv(prices)
        noise_var = mv.estimate_noise_variance(prices)
        clean_prices = mv.filter_bidask_bounce(prices)
    """

    def __init__(self):
        """Initialize volatility estimator."""
        pass

    def realized_variance(self, prices: np.ndarray) -> float:
        """
        Standard Realized Variance (biased at high frequency).

        RV = Σ(r_i)^2

        Args:
            prices: Price series

        Returns:
            Realized variance
        """
        if len(prices) < 2:
            return 0.0

        returns = np.diff(np.log(prices))
        return np.sum(returns ** 2)

    def realized_volatility(self, prices: np.ndarray, annualize: bool = True) -> float:
        """
        Standard Realized Volatility.

        Args:
            prices: Price series
            annualize: If True, annualize (assumes daily data)

        Returns:
            Realized volatility
        """
        rv = self.realized_variance(prices)
        vol = np.sqrt(rv)

        if annualize:
            vol *= np.sqrt(252)  # Trading days

        return vol

    def tsrv(self, prices: np.ndarray, K: int = None) -> float:
        """
        Two Scales Realized Variance.

        Unbiased estimator that accounts for microstructure noise.

        TSRV = (1/K) Σ RV_k - (n_bar / n) * RV_all

        where:
        - RV_k is realized variance on subsampled grid k
        - n_bar is average sample size per grid
        - n is total observations

        Args:
            prices: Price series
            K: Number of subsampling grids (default: optimal)

        Returns:
            Noise-robust realized variance
        """
        n = len(prices)
        if n < 10:
            return self.realized_variance(prices)

        # Optimal K (Zhang et al. 2005)
        if K is None:
            K = int(np.ceil(n ** (2/3)))

        K = min(K, n // 2)
        if K < 2:
            K = 2

        # Subsampled RVs
        rv_subsampled = 0.0
        counts = []

        for k in range(K):
            # Subsample starting at offset k
            subsample = prices[k::K]
            if len(subsample) > 1:
                rv_k = self.realized_variance(subsample)
                rv_subsampled += rv_k
                counts.append(len(subsample))

        rv_subsampled /= K

        # Full sample RV (for bias correction)
        rv_full = self.realized_variance(prices)

        # Bias correction
        n_bar = np.mean(counts) if counts else n / K
        bias_correction = (n_bar / n) * rv_full

        tsrv = rv_subsampled - bias_correction

        return max(0, tsrv)  # Ensure non-negative

    def tsrv_volatility(self, prices: np.ndarray, K: int = None,
                        annualize: bool = True) -> float:
        """
        Two Scales Realized Volatility.

        Args:
            prices: Price series
            K: Subsampling grids
            annualize: Annualize volatility

        Returns:
            TSRV-based volatility
        """
        tsrv = self.tsrv(prices, K)
        vol = np.sqrt(tsrv)

        if annualize:
            vol *= np.sqrt(252)

        return vol

    def estimate_noise_variance(self, prices: np.ndarray) -> float:
        """
        Estimate microstructure noise variance.

        Uses first-order autocorrelation of returns.

        σ²_noise ≈ -Cov(r_t, r_{t-1})

        Args:
            prices: Price series

        Returns:
            Estimated noise variance
        """
        if len(prices) < 3:
            return 0.0

        returns = np.diff(np.log(prices))

        # Autocovariance at lag 1
        if len(returns) < 2:
            return 0.0

        autocov = np.cov(returns[:-1], returns[1:])[0, 1]

        # Noise variance is negative of first autocovariance
        noise_var = -autocov

        return max(0, noise_var)

    def signal_to_noise_ratio(self, prices: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio.

        SNR = true_variance / noise_variance

        Args:
            prices: Price series

        Returns:
            SNR (higher = less noisy)
        """
        true_var = self.tsrv(prices)
        noise_var = self.estimate_noise_variance(prices)

        if noise_var <= 0:
            return float('inf')

        return true_var / noise_var

    def optimal_sampling_interval(self, prices: np.ndarray,
                                  sampling_intervals: List[int] = None) -> int:
        """
        Find optimal sampling interval to minimize MSE.

        Trades off:
        - More samples = more information
        - Fewer samples = less noise

        Args:
            prices: High-frequency price series
            sampling_intervals: Intervals to test

        Returns:
            Optimal sampling interval
        """
        n = len(prices)

        if sampling_intervals is None:
            # Test intervals from 1 to n/10
            max_interval = max(10, n // 10)
            sampling_intervals = list(range(1, min(max_interval, 100)))

        if not sampling_intervals:
            return 1

        mse_estimates = []

        for interval in sampling_intervals:
            if interval >= n // 2:
                continue

            # Subsample
            subsampled = prices[::interval]

            if len(subsampled) < 3:
                continue

            # TSRV on subsampled data
            var_estimate = self.tsrv(subsampled)

            # Noise on subsampled data (should be lower)
            noise_var = self.estimate_noise_variance(subsampled)

            # Approximate MSE = variance of estimator + noise impact
            # More samples = lower variance, but more noise
            n_sub = len(subsampled)
            estimator_variance = 2 * var_estimate ** 2 / n_sub if var_estimate > 0 else 0

            mse = estimator_variance + noise_var
            mse_estimates.append((interval, mse))

        if not mse_estimates:
            return 1

        # Return interval with lowest MSE
        optimal = min(mse_estimates, key=lambda x: x[1])
        return optimal[0]

    def filter_bidask_bounce(self, prices: np.ndarray,
                             method: str = 'ma') -> np.ndarray:
        """
        Filter bid-ask bounce from prices.

        Args:
            prices: Price series with bid-ask bounce
            method: 'ma' (moving average) or 'kalman'

        Returns:
            Filtered price series
        """
        if len(prices) < 3:
            return prices

        if method == 'ma':
            # Simple moving average filter
            window = 3
            filtered = pd.Series(prices).rolling(window=window, center=True).mean()
            filtered = filtered.fillna(method='bfill').fillna(method='ffill')
            return filtered.values

        elif method == 'kalman':
            # Simple Kalman filter
            return self._kalman_filter(prices)

        return prices

    def _kalman_filter(self, prices: np.ndarray) -> np.ndarray:
        """Simple Kalman filter for price smoothing."""
        n = len(prices)
        filtered = np.zeros(n)

        # Initialize
        x_hat = prices[0]  # State estimate
        P = 1.0  # Error covariance

        # Process and measurement noise
        Q = 0.0001  # Process noise
        R = self.estimate_noise_variance(prices)
        if R <= 0:
            R = 0.0001

        for i in range(n):
            # Predict
            x_hat_minus = x_hat
            P_minus = P + Q

            # Update
            K = P_minus / (P_minus + R)  # Kalman gain
            x_hat = x_hat_minus + K * (prices[i] - x_hat_minus)
            P = (1 - K) * P_minus

            filtered[i] = x_hat

        return filtered

    def multi_scale_variance(self, prices: np.ndarray,
                             scales: List[int] = None) -> Dict[str, float]:
        """
        Calculate variance at multiple time scales.

        Args:
            prices: Price series
            scales: Time scales (sampling intervals)

        Returns:
            Dict with variance at each scale
        """
        if scales is None:
            scales = [1, 5, 15, 30, 60]

        result = {}
        for scale in scales:
            if scale >= len(prices) // 2:
                continue

            subsampled = prices[::scale]
            if len(subsampled) < 3:
                continue

            rv = self.realized_variance(subsampled)
            tsrv = self.tsrv(subsampled)

            result[f'rv_scale_{scale}'] = rv
            result[f'tsrv_scale_{scale}'] = tsrv

        return result

    def volatility_signature(self, prices: np.ndarray,
                             max_scale: int = 100) -> pd.DataFrame:
        """
        Generate volatility signature plot data.

        Shows how volatility estimate varies with sampling frequency.
        Flat signature = unbiased estimator.

        Args:
            prices: Price series
            max_scale: Maximum sampling interval

        Returns:
            DataFrame with scale and volatility columns
        """
        scales = list(range(1, min(max_scale, len(prices) // 3)))

        data = []
        for scale in scales:
            subsampled = prices[::scale]
            if len(subsampled) < 3:
                continue

            rv = self.realized_volatility(subsampled, annualize=False)
            tsrv = self.tsrv_volatility(subsampled, annualize=False)

            data.append({
                'scale': scale,
                'rv': rv,
                'tsrv': tsrv
            })

        return pd.DataFrame(data)


class JumpRobustVolatility:
    """
    Jump-Robust Volatility Estimation.

    Separates continuous and jump components.
    """

    def __init__(self, threshold: float = 4.0):
        """
        Initialize estimator.

        Args:
            threshold: Jump detection threshold (in std deviations)
        """
        self.threshold = threshold

    def bipower_variation(self, prices: np.ndarray) -> float:
        """
        Bipower Variation - robust to jumps.

        BV = (π/2) Σ |r_i| |r_{i-1}|

        Args:
            prices: Price series

        Returns:
            Bipower variation
        """
        if len(prices) < 3:
            return 0.0

        returns = np.diff(np.log(prices))
        abs_returns = np.abs(returns)

        bv = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])
        return bv

    def detect_jumps(self, prices: np.ndarray) -> np.ndarray:
        """
        Detect jumps in price series.

        Args:
            prices: Price series

        Returns:
            Boolean array indicating jumps
        """
        if len(prices) < 3:
            return np.zeros(len(prices), dtype=bool)

        returns = np.diff(np.log(prices))

        # Local volatility estimate (rolling std)
        window = min(20, len(returns) // 3)
        local_vol = pd.Series(returns).rolling(window=window, center=True).std()
        local_vol = local_vol.fillna(method='bfill').fillna(method='ffill').values

        # Threshold
        jumps = np.abs(returns) > self.threshold * local_vol

        # Pad to match price length
        return np.concatenate([[False], jumps])

    def remove_jumps(self, prices: np.ndarray) -> np.ndarray:
        """
        Remove jumps from price series.

        Args:
            prices: Price series with jumps

        Returns:
            Price series with jumps removed
        """
        jumps = self.detect_jumps(prices)
        filtered = prices.copy()

        for i in np.where(jumps)[0]:
            if i > 0:
                filtered[i] = filtered[i-1]

        return filtered

    def continuous_variation(self, prices: np.ndarray) -> float:
        """
        Estimate continuous variation (excluding jumps).

        Args:
            prices: Price series

        Returns:
            Continuous variation
        """
        filtered = self.remove_jumps(prices)
        mv = MicrostructureVolatility()
        return mv.tsrv(filtered)

    def jump_variation(self, prices: np.ndarray) -> float:
        """
        Estimate jump variation.

        Args:
            prices: Price series

        Returns:
            Jump variation (RV - continuous variation)
        """
        mv = MicrostructureVolatility()
        total_rv = mv.realized_variance(prices)
        continuous = self.continuous_variation(prices)

        return max(0, total_rv - continuous)


def add_microstructure_features(df: pd.DataFrame,
                                price_col: str = 'close') -> pd.DataFrame:
    """
    Add microstructure volatility features to DataFrame.

    Args:
        df: DataFrame with price data
        price_col: Name of price column

    Returns:
        DataFrame with additional volatility features
    """
    result = df.copy()
    mv = MicrostructureVolatility()
    jrv = JumpRobustVolatility()

    prices = df[price_col].values

    # Calculate features on rolling window
    window = 100

    result['tsrv'] = np.nan
    result['noise_var'] = np.nan
    result['snr'] = np.nan
    result['bipower_var'] = np.nan
    result['jump_var'] = np.nan

    for i in range(window, len(df)):
        window_prices = prices[i-window:i]

        result.iloc[i, result.columns.get_loc('tsrv')] = mv.tsrv(window_prices)
        result.iloc[i, result.columns.get_loc('noise_var')] = mv.estimate_noise_variance(window_prices)
        result.iloc[i, result.columns.get_loc('snr')] = mv.signal_to_noise_ratio(window_prices)
        result.iloc[i, result.columns.get_loc('bipower_var')] = jrv.bipower_variation(window_prices)
        result.iloc[i, result.columns.get_loc('jump_var')] = jrv.jump_variation(window_prices)

    return result


if __name__ == '__main__':
    print("Microstructure Volatility Test")
    print("=" * 50)

    # Generate synthetic high-frequency data with noise
    np.random.seed(42)
    n = 1000

    # True price process
    true_vol = 0.01
    true_prices = 100 * np.exp(np.cumsum(np.random.randn(n) * true_vol / np.sqrt(n)))

    # Add microstructure noise (bid-ask bounce)
    noise_std = 0.001
    noisy_prices = true_prices * (1 + np.random.randn(n) * noise_std)

    mv = MicrostructureVolatility()

    print("\nVolatility Estimates:")
    print(f"  True daily vol: {true_vol * np.sqrt(252):.4f}")
    print(f"  RV (noisy): {mv.realized_volatility(noisy_prices):.4f}")
    print(f"  RV (true): {mv.realized_volatility(true_prices):.4f}")
    print(f"  TSRV (noisy): {mv.tsrv_volatility(noisy_prices):.4f}")

    print(f"\nNoise Estimates:")
    print(f"  True noise std: {noise_std:.6f}")
    print(f"  Estimated noise var: {mv.estimate_noise_variance(noisy_prices):.8f}")
    print(f"  SNR: {mv.signal_to_noise_ratio(noisy_prices):.2f}")

    print(f"\nOptimal Sampling:")
    opt_interval = mv.optimal_sampling_interval(noisy_prices)
    print(f"  Optimal interval: {opt_interval}")

    # Test jump detection
    print("\n" + "=" * 50)
    print("Jump Detection Test")

    # Add a jump
    prices_with_jump = noisy_prices.copy()
    prices_with_jump[500] *= 1.05  # 5% jump

    jrv = JumpRobustVolatility(threshold=3.0)
    jumps = jrv.detect_jumps(prices_with_jump)
    print(f"  Jumps detected: {np.sum(jumps)}")
    print(f"  Jump locations: {np.where(jumps)[0]}")
    print(f"  Continuous var: {jrv.continuous_variation(prices_with_jump):.6f}")
    print(f"  Jump var: {jrv.jump_variation(prices_with_jump):.6f}")
