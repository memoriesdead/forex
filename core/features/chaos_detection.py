"""
Chaos Detection and Nonlinear Dynamics Features

Implements Lyapunov exponents, BDS test, correlation dimension, and other
chaos indicators for financial time series analysis.

Key Insight (Lorenz 1963):
    Small perturbations grow exponentially: |δZ(t)| ≈ |δZ(0)| · e^(λt)
    Prediction horizon is fundamentally limited by positive Lyapunov exponent.

Key Formulas:
    Lyapunov Exponent:        λ = lim_{t→∞} (1/t) ln|δZ(t)/δZ(0)|
    Prediction Horizon:       T_pred ≈ (1/λ) · ln(Δ/δ₀)
    Correlation Dimension:    D₂ = lim_{r→0} [ln C(r) / ln r]
    BDS Statistic:            W = √n [C_m(ε) - C_1(ε)^m] / σ_m

References:
    [1] Lorenz, E.N. (1963). "Deterministic Nonperiodic Flow."
        Journal of the Atmospheric Sciences, 20(2), 130-141.
    [2] Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993).
        "A practical method for calculating largest Lyapunov exponents
        from small data sets." Physica D, 65(1-2), 117-134.
    [3] Brock, W.A., Dechert, W.D., & Scheinkman, J.A. (1996).
        "A Test for Independence Based on the Correlation Dimension."
        Econometric Reviews, 15(3), 197-235.
    [4] Grassberger, P. & Procaccia, I. (1983). "Characterization of
        Strange Attractors." Physical Review Letters, 50(5), 346.
    [5] Peters, E.E. (1994). "Fractal Market Analysis." Wiley.
    [6] Takens, F. (1981). "Detecting strange attractors in turbulence."
        Lecture Notes in Mathematics, 898, 366-381.

Author: Claude Code + Kevin
Created: 2026-01-22
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist
from scipy import stats
import warnings


class ChaosDetector:
    """
    Chaos detection for financial time series.

    Detects chaotic dynamics that fundamentally limit prediction accuracy.
    Markets show chaos ~40-70% of the time, with positive Lyapunov exponents
    limiting prediction horizons to seconds-minutes at tick level.

    The Butterfly Effect in Finance:
        - Single large order
        - News tweet
        - Algorithmic glitch
        → Can cascade into large, unpredictable price movements

    Reference:
        [1] Lorenz (1963): Original chaos discovery
        [2] Rosenstein et al. (1993): Lyapunov estimation
        [3] Brock et al. (1996): BDS test for markets
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        time_delay: int = 1,
        theiler_window: Optional[int] = None
    ):
        """
        Initialize chaos detector.

        Args:
            embedding_dim: Dimension for phase space reconstruction (Takens)
            time_delay: Time delay for embedding
            theiler_window: Temporal separation to avoid correlated pairs
        """
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.theiler_window = theiler_window or (time_delay * embedding_dim)

    def compute_all_features(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute all chaos-related features.

        Args:
            returns: Time series of returns

        Returns:
            Dictionary of chaos features
        """
        result = {}

        # Lyapunov exponent (prediction horizon)
        lyap = self.estimate_lyapunov_exponent(returns)
        result.update({
            'chaos_lyapunov': lyap['lyapunov_exponent'],
            'chaos_is_chaotic': float(lyap['is_chaotic']),
            'chaos_pred_horizon': min(lyap['prediction_horizon'], 1000)
        })

        # BDS test (nonlinearity)
        bds = self.bds_test(returns)
        result.update({
            'chaos_bds_stat': bds['statistic'],
            'chaos_bds_pvalue': bds['p_value'],
            'chaos_is_nonlinear': float(bds['is_nonlinear'])
        })

        # Correlation dimension
        corr_dim = self.correlation_dimension(returns)
        result.update({
            'chaos_correlation_dim': corr_dim['dimension'],
            'chaos_is_low_dim': float(corr_dim['is_low_dimensional'])
        })

        # Hurst exponent (long memory vs anti-persistence)
        hurst = self.hurst_exponent(returns)
        result.update({
            'chaos_hurst': hurst['H'],
            'chaos_regime': hurst['regime_code']
        })

        # Recurrence quantification
        rqa = self.recurrence_quantification(returns)
        result.update({
            'chaos_recurrence_rate': rqa['recurrence_rate'],
            'chaos_determinism': rqa['determinism'],
            'chaos_laminarity': rqa['laminarity']
        })

        return result

    def estimate_lyapunov_exponent(
        self,
        returns: np.ndarray,
        max_iterations: int = 100
    ) -> Dict:
        """
        Estimate maximal Lyapunov exponent using Rosenstein method.

        The Lyapunov exponent λ characterizes exponential divergence:
            |δZ(t)| ≈ |δZ(0)| · e^(λt)

        Interpretation:
            λ > 0: Chaotic (sensitive dependence on initial conditions)
            λ = 0: Periodic or quasiperiodic
            λ < 0: Fixed point attractor

        Prediction Horizon:
            T_pred ≈ (1/λ) · ln(Δ/δ₀)
            For forex tick data: typically seconds to minutes

        Reference:
            [2] Rosenstein, M.T. et al. (1993). "A practical method for
                calculating largest Lyapunov exponents from small data sets."

        Args:
            returns: Time series
            max_iterations: Max divergence tracking iterations

        Returns:
            Dictionary with lyapunov_exponent, is_chaotic, prediction_horizon
        """
        N = len(returns)
        m = self.embedding_dim
        tau = self.time_delay

        if N < m * tau + max_iterations:
            return {
                'lyapunov_exponent': 0.1,
                'is_chaotic': True,
                'prediction_horizon': 10.0
            }

        # Phase space reconstruction (Takens' embedding theorem)
        # Reference: [6] Takens (1981)
        n_vectors = N - (m - 1) * tau
        embedded = np.zeros((n_vectors, m))
        for i in range(n_vectors):
            for j in range(m):
                embedded[i, j] = returns[i + j * tau]

        # Compute distance matrix
        distances = cdist(embedded, embedded, metric='euclidean')

        # Apply Theiler window (exclude temporally close neighbors)
        for i in range(len(distances)):
            start = max(0, i - self.theiler_window)
            end = min(len(distances), i + self.theiler_window + 1)
            distances[i, start:end] = np.inf

        # Find nearest neighbor for each point
        nearest_idx = np.argmin(distances, axis=1)
        nearest_dist = distances[np.arange(len(distances)), nearest_idx]

        # Track divergence over time
        divergence_log = []
        for k in range(1, min(max_iterations, n_vectors // 4)):
            div_k = []
            for i in range(len(embedded) - k):
                j = nearest_idx[i]
                if j + k < len(embedded) and nearest_dist[i] < np.inf:
                    d_t = np.linalg.norm(embedded[i + k] - embedded[j + k])
                    if d_t > 1e-12:
                        # ln(d(t)/d(0))
                        div_k.append(np.log(d_t / max(nearest_dist[i], 1e-12)))

            if len(div_k) > 10:
                divergence_log.append(np.mean(div_k))

        if len(divergence_log) < 5:
            return {
                'lyapunov_exponent': 0.05,
                'is_chaotic': True,
                'prediction_horizon': 20.0
            }

        # Fit linear slope = Lyapunov exponent
        x = np.arange(len(divergence_log))
        slope, _, _, _, _ = stats.linregress(x, divergence_log)
        lyapunov = max(0, slope)  # λ should be non-negative for this method

        # Prediction horizon: time to double error
        if lyapunov > 1e-6:
            pred_horizon = np.log(2) / lyapunov
        else:
            pred_horizon = float('inf')

        return {
            'lyapunov_exponent': lyapunov,
            'is_chaotic': lyapunov > 0.01,
            'prediction_horizon': min(pred_horizon, 10000)
        }

    def bds_test(
        self,
        returns: np.ndarray,
        embedding_dim: int = 2,
        epsilon: Optional[float] = None
    ) -> Dict:
        """
        BDS test for nonlinear dependence.

        Tests whether the time series is IID (independent and identically
        distributed) vs having nonlinear structure.

        BDS Statistic:
            W = √n [C_m(ε) - C_1(ε)^m] / σ_m

        Where C_m(ε) is the correlation integral at dimension m.

        Interpretation:
            Reject IID → nonlinear dependence exists
            Accept IID → linear or no dependence

        Reference:
            [3] Brock, W.A., Dechert, W.D., & Scheinkman, J.A. (1996).
                "A Test for Independence Based on the Correlation Dimension."

        Args:
            returns: Time series
            embedding_dim: Embedding dimension for test
            epsilon: Distance threshold (default: 0.5 * std)

        Returns:
            Dictionary with statistic, p_value, is_nonlinear
        """
        n = len(returns)
        if n < 50:
            return {'statistic': 0, 'p_value': 1.0, 'is_nonlinear': False}

        # Standardize
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)

        # Default epsilon: 0.5 standard deviations
        if epsilon is None:
            epsilon = 0.5

        m = embedding_dim

        # Embed time series
        embedded = np.zeros((n - m + 1, m))
        for i in range(n - m + 1):
            embedded[i] = returns[i:i + m]

        # Correlation integral at dimension m
        # C_m(ε) = (2 / (N(N-1))) Σ_{i<j} I(||x_i - x_j|| < ε)
        distances = cdist(embedded, embedded, metric='chebyshev')
        np.fill_diagonal(distances, np.inf)
        C_m = np.mean(distances < epsilon)

        # Correlation integral at dimension 1
        C_1 = np.mean(np.abs(returns[:, np.newaxis] - returns) < epsilon)

        # BDS statistic
        # Under null (IID): C_m ≈ C_1^m
        expected = C_1 ** m

        # Variance under null (simplified)
        k = 2  # depends on C_1, approximation
        sigma_squared = 4 * (C_1 ** (2 * m - 2)) * (1 - C_1) ** 2 / n
        sigma = np.sqrt(max(sigma_squared, 1e-10))

        statistic = (C_m - expected) / sigma

        # Two-sided p-value (asymptotically normal)
        p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))

        return {
            'statistic': statistic,
            'p_value': p_value,
            'C_m': C_m,
            'C_1_power_m': expected,
            'is_nonlinear': p_value < 0.05
        }

    def correlation_dimension(
        self,
        returns: np.ndarray,
        r_values: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.

        Correlation Dimension:
            D₂ = lim_{r→0} [ln C(r) / ln r]

        Where C(r) is the correlation integral:
            C(r) = lim_{N→∞} (1/N²) Σ_{i≠j} I(||x_i - x_j|| < r)

        For chaotic systems: D₂ is typically non-integer
        For random data: D₂ grows with embedding dimension

        Reference:
            [4] Grassberger, P. & Procaccia, I. (1983). "Characterization
                of Strange Attractors." Physical Review Letters.

        Args:
            returns: Time series
            r_values: Radius values for correlation integral

        Returns:
            Dictionary with dimension and analysis
        """
        N = len(returns)
        m = self.embedding_dim
        tau = self.time_delay

        if N < m * tau + 50:
            return {'dimension': 5.0, 'is_low_dimensional': False}

        # Embed
        n_vectors = N - (m - 1) * tau
        embedded = np.zeros((n_vectors, m))
        for i in range(n_vectors):
            for j in range(m):
                embedded[i, j] = returns[i + j * tau]

        # Normalize
        embedded = (embedded - np.mean(embedded, axis=0)) / (np.std(embedded, axis=0) + 1e-10)

        # Distance matrix
        distances = cdist(embedded, embedded, metric='euclidean')
        np.fill_diagonal(distances, np.inf)

        # Range of r values
        if r_values is None:
            d_sorted = np.sort(distances.flatten())
            d_sorted = d_sorted[d_sorted < np.inf]
            if len(d_sorted) < 10:
                return {'dimension': 5.0, 'is_low_dimensional': False}

            r_min = np.percentile(d_sorted, 1)
            r_max = np.percentile(d_sorted, 50)
            r_values = np.logspace(np.log10(max(r_min, 1e-6)), np.log10(r_max), 20)

        # Correlation integral C(r)
        C_r = []
        for r in r_values:
            count = np.sum(distances < r)
            C = count / (n_vectors * (n_vectors - 1))
            if C > 0:
                C_r.append((r, C))

        if len(C_r) < 5:
            return {'dimension': 5.0, 'is_low_dimensional': False}

        # Linear fit in log-log space: ln C(r) = D₂ · ln r + const
        log_r = np.log([x[0] for x in C_r])
        log_C = np.log([x[1] for x in C_r])

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_C)

        return {
            'dimension': slope,
            'r_squared': r_value ** 2,
            'std_error': std_err,
            'is_low_dimensional': slope < m / 2,  # Much less than embedding dim
            'embedding_dim': m
        }

    def hurst_exponent(self, returns: np.ndarray) -> Dict:
        """
        Estimate Hurst exponent using R/S analysis.

        Hurst Exponent:
            E[R(n)/S(n)] = C · n^H

        Interpretation:
            H = 0.5: Random walk (Brownian motion)
            H > 0.5: Persistent (trending)
            H < 0.5: Anti-persistent (mean-reverting)

        Reference:
            [5] Peters, E.E. (1994). "Fractal Market Analysis." Wiley.
            Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."

        Args:
            returns: Time series

        Returns:
            Dictionary with H, regime, predictability_boost
        """
        N = len(returns)
        if N < 20:
            return {'H': 0.5, 'regime': 'random', 'regime_code': 0}

        # R/S analysis at different scales
        rs_values = []
        n_values = []

        for n in [10, 20, 40, 80, 160, 320]:
            if n > N // 4:
                break

            rs_list = []
            for start in range(0, N - n, n // 2):
                segment = returns[start:start + n]
                mean = np.mean(segment)
                std = np.std(segment)
                if std < 1e-10:
                    continue

                # Cumulative deviation from mean
                cum_dev = np.cumsum(segment - mean)
                R = np.max(cum_dev) - np.min(cum_dev)
                S = std

                rs_list.append(R / S)

            if len(rs_list) > 0:
                rs_values.append(np.mean(rs_list))
                n_values.append(n)

        if len(rs_values) < 3:
            return {'H': 0.5, 'regime': 'random', 'regime_code': 0}

        # Linear fit: log(R/S) = H * log(n) + const
        log_n = np.log(n_values)
        log_rs = np.log(rs_values)
        slope, _, r_value, _, _ = stats.linregress(log_n, log_rs)

        H = np.clip(slope, 0, 1)

        # Regime classification
        if H > 0.6:
            regime = 'persistent'
            regime_code = 1
        elif H < 0.4:
            regime = 'antipersistent'
            regime_code = -1
        else:
            regime = 'random'
            regime_code = 0

        # Predictability boost from Hurst
        # H far from 0.5 means more predictable
        predictability_boost = abs(H - 0.5) * 0.2  # Max ~10% accuracy boost

        return {
            'H': H,
            'regime': regime,
            'regime_code': regime_code,
            'r_squared': r_value ** 2,
            'predictability_boost': predictability_boost
        }

    def recurrence_quantification(
        self,
        returns: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Recurrence Quantification Analysis (RQA).

        Analyzes recurrence plot statistics:
            - Recurrence Rate: fraction of recurrent points
            - Determinism: fraction of recurrent points forming lines
            - Laminarity: fraction forming vertical structures

        High determinism → more predictable (deterministic chaos)
        High laminarity → more intermittent dynamics

        Reference:
            Marwan, N. et al. (2007). "Recurrence plots for the analysis
            of complex systems." Physics Reports.
            Eckmann, J.P. et al. (1987). "Recurrence Plots of Dynamical Systems."

        Args:
            returns: Time series
            threshold: Distance threshold for recurrence (default: 10% of max distance)

        Returns:
            Dictionary with RQA measures
        """
        N = len(returns)
        m = min(self.embedding_dim, 5)
        tau = self.time_delay

        if N < m * tau + 20:
            return {
                'recurrence_rate': 0.5,
                'determinism': 0.5,
                'laminarity': 0.5
            }

        # Embed
        n_vectors = N - (m - 1) * tau
        embedded = np.zeros((n_vectors, m))
        for i in range(n_vectors):
            for j in range(m):
                embedded[i, j] = returns[i + j * tau]

        # Distance matrix
        distances = cdist(embedded, embedded, metric='euclidean')

        # Threshold
        if threshold is None:
            threshold = 0.1 * np.max(distances)

        # Recurrence matrix
        recurrence = (distances < threshold).astype(int)
        np.fill_diagonal(recurrence, 0)

        # Recurrence rate
        rr = np.sum(recurrence) / (n_vectors * (n_vectors - 1))

        # Determinism: fraction of recurrent points forming diagonal lines
        # Count diagonal lines of length >= 2
        diag_line_points = 0
        for k in range(-n_vectors + 2, n_vectors - 1):
            diag = np.diagonal(recurrence, k)
            # Find runs of 1s with length >= 2
            in_line = False
            line_length = 0
            for val in diag:
                if val == 1:
                    line_length += 1
                    if line_length >= 2:
                        in_line = True
                else:
                    if in_line:
                        diag_line_points += line_length
                    in_line = False
                    line_length = 0
            if in_line:
                diag_line_points += line_length

        total_recurrent = np.sum(recurrence)
        determinism = diag_line_points / max(total_recurrent, 1)

        # Laminarity: fraction in vertical lines
        vert_line_points = 0
        for col in range(n_vectors):
            column = recurrence[:, col]
            in_line = False
            line_length = 0
            for val in column:
                if val == 1:
                    line_length += 1
                    if line_length >= 2:
                        in_line = True
                else:
                    if in_line:
                        vert_line_points += line_length
                    in_line = False
                    line_length = 0
            if in_line:
                vert_line_points += line_length

        laminarity = vert_line_points / max(total_recurrent, 1)

        return {
            'recurrence_rate': rr,
            'determinism': np.clip(determinism, 0, 1),
            'laminarity': np.clip(laminarity, 0, 1),
            'threshold': threshold
        }


def estimate_prediction_horizon(returns: np.ndarray) -> Dict:
    """
    Estimate practical prediction horizon from chaos analysis.

    The horizon is limited by:
        1. Lyapunov exponent (exponential error growth)
        2. Noise level (signal-to-noise ratio)
        3. Market regime (trending vs ranging)

    Reference:
        [1] Lorenz (1963): Chaos and prediction limits
        [2] Rosenstein et al. (1993): Lyapunov estimation

    Args:
        returns: Time series

    Returns:
        Dictionary with horizon estimates
    """
    detector = ChaosDetector()

    # Lyapunov-based horizon
    lyap = detector.estimate_lyapunov_exponent(returns)
    lyap_horizon = lyap['prediction_horizon']

    # Hurst-based regime
    hurst = detector.hurst_exponent(returns)

    # Adjust horizon based on regime
    if hurst['regime'] == 'persistent':
        regime_multiplier = 1.5
    elif hurst['regime'] == 'antipersistent':
        regime_multiplier = 0.7
    else:
        regime_multiplier = 1.0

    # Practical horizon
    practical_horizon = lyap_horizon * regime_multiplier

    # Confidence (based on how chaotic)
    confidence = 1.0 if lyap['lyapunov_exponent'] > 0.05 else 0.5

    return {
        'lyapunov_horizon': lyap_horizon,
        'practical_horizon': practical_horizon,
        'regime_multiplier': regime_multiplier,
        'is_chaotic': lyap['is_chaotic'],
        'hurst_exponent': hurst['H'],
        'regime': hurst['regime'],
        'confidence': confidence
    }


# Convenience function
def create_chaos_features(returns: np.ndarray) -> Dict[str, float]:
    """
    Create all chaos detection features for a time series.

    Reference:
        [1-6] See module docstring for full citations

    Args:
        returns: Time series of returns

    Returns:
        Dictionary of chaos features
    """
    detector = ChaosDetector()
    return detector.compute_all_features(returns)


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    n = 2000

    # Generate chaotic time series (logistic map)
    def logistic_map(r, x, n):
        result = [x]
        for _ in range(n - 1):
            x = r * x * (1 - x)
            result.append(x)
        return np.array(result)

    chaotic = logistic_map(3.9, 0.5, n)  # Chaotic regime
    chaotic_returns = np.diff(chaotic)

    # Random walk
    random_returns = np.random.randn(n - 1) * 0.01

    print("=" * 60)
    print("CHAOS DETECTION - LOGISTIC MAP (CHAOTIC)")
    print("=" * 60)
    features_chaotic = create_chaos_features(chaotic_returns)
    for name, value in features_chaotic.items():
        print(f"{name:30} {value:.6f}")

    print("\n" + "=" * 60)
    print("CHAOS DETECTION - RANDOM WALK")
    print("=" * 60)
    features_random = create_chaos_features(random_returns)
    for name, value in features_random.items():
        print(f"{name:30} {value:.6f}")

    print("\n" + "=" * 60)
    print("PREDICTION HORIZON ANALYSIS")
    print("=" * 60)
    horizon = estimate_prediction_horizon(chaotic_returns)
    for name, value in horizon.items():
        if isinstance(value, float):
            print(f"{name:30} {value:.4f}")
        else:
            print(f"{name:30} {value}")
