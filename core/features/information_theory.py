"""
Information Theory Features for Financial Markets

Implements Shannon entropy, transfer entropy, and Kolmogorov complexity
approximations for feature engineering and predictability analysis.

Key Insight (Lo & MacKinlay 1988):
    Only ~12% of daily return variance is predictable using any information set.

Key Formulas:
    Shannon Entropy:     H(X) = -Σ p(x) log₂ p(x)
    Mutual Information:  I(X;Y) = H(Y) - H(Y|X)
    Transfer Entropy:    T_{X→Y} = Σ p(y_{t+1}, y_t, x_t) log[p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t)]
    Kolmogorov:          K(x) = min{|p| : U(p) = x} ≈ len(compress(x))

References:
    [1] Shannon, C.E. (1948). "A Mathematical Theory of Communication."
        Bell System Technical Journal, 27(3), 379-423.
    [2] Schreiber, T. (2000). "Measuring Information Transfer."
        Physical Review Letters, 85(2), 461.
    [3] Kolmogorov, A.N. (1965). "Three approaches to the quantitative
        definition of information." Problems of Information Transmission.
    [4] Lo, A.W. & MacKinlay, A.C. (1988). "Stock Market Prices Do Not
        Follow Random Walks." Review of Financial Studies.
    [5] Fama, E.F. (1970). "Efficient Capital Markets." Journal of Finance.
    [6] Cover, T.M. & Thomas, J.A. (2006). "Elements of Information Theory."

Author: Claude Code + Kevin
Created: 2026-01-22
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import zlib
from scipy import stats


class InformationTheoryFeatures:
    """
    Information theory feature calculator for financial time series.

    Implements:
        - Shannon entropy (market uncertainty)
        - Conditional entropy (remaining uncertainty after features)
        - Mutual information (predictive content)
        - Transfer entropy (information flow between assets)
        - Kolmogorov complexity (algorithmic randomness)

    Reference:
        [1] Shannon (1948): Entropy and information theory
        [2] Schreiber (2000): Transfer entropy
        [3] Kolmogorov (1965): Algorithmic complexity
    """

    def __init__(self, n_bins: int = 20, history_length: int = 5):
        """
        Initialize information theory feature calculator.

        Args:
            n_bins: Number of bins for discretization
            history_length: Lag length for transfer entropy
        """
        self.n_bins = n_bins
        self.history_length = history_length

    def compute_all_features(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray] = None,
        other_assets: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute all information theory features.

        Args:
            returns: Target return series
            features: Optional feature matrix for MI calculation
            other_assets: Dict of other asset returns for transfer entropy

        Returns:
            Dictionary of information theory features
        """
        result = {}

        # Basic entropy features
        result.update(self._entropy_features(returns))

        # Complexity features
        result.update(self._complexity_features(returns))

        # Mutual information if features provided
        if features is not None:
            result.update(self._mutual_information_features(returns, features))

        # Transfer entropy if other assets provided
        if other_assets is not None:
            result.update(self._transfer_entropy_features(returns, other_assets))

        return result

    def _entropy_features(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate entropy-based features.

        Shannon Entropy:
            H(X) = -Σ p(x) log₂ p(x)

        Maximum Entropy (uniform distribution):
            H_max = log₂(n_bins)

        Entropy Ratio (normalized):
            H_ratio = H(X) / H_max

        Reference:
            [1] Shannon (1948): "A Mathematical Theory of Communication"

        Returns:
            Dictionary with entropy features
        """
        # Discretize returns
        bins = np.percentile(returns, np.linspace(0, 100, self.n_bins + 1))
        bins[-1] += 1e-10
        digitized = np.digitize(returns, bins[:-1]) - 1
        digitized = np.clip(digitized, 0, self.n_bins - 1)

        # Probability distribution
        counts = np.bincount(digitized, minlength=self.n_bins)
        probs = counts / len(returns)

        # Shannon entropy
        entropy = self._shannon_entropy(probs)

        # Maximum entropy (uniform)
        max_entropy = np.log2(self.n_bins)

        # Entropy ratio (0 = deterministic, 1 = maximum uncertainty)
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

        # Direction entropy (binary up/down)
        direction = (returns > 0).astype(int)
        p_up = np.mean(direction)
        direction_probs = np.array([1 - p_up, p_up])
        direction_entropy = self._shannon_entropy(direction_probs)

        # Conditional entropy on lagged returns
        cond_entropy = self._conditional_entropy_lag(returns)

        return {
            'info_entropy': entropy,
            'info_entropy_ratio': entropy_ratio,
            'info_entropy_direction': direction_entropy,
            'info_conditional_entropy': cond_entropy,
            'info_predictability': max(0, direction_entropy - cond_entropy),
            'info_max_entropy': max_entropy
        }

    def _shannon_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Shannon entropy in bits.

        Formula:
            H(X) = -Σ p(x) log₂ p(x)

        Reference:
            [1] Shannon (1948)
        """
        p = probabilities[probabilities > 0]
        return -np.sum(p * np.log2(p))

    def _conditional_entropy_lag(
        self,
        returns: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Calculate conditional entropy H(X_t | X_{t-lag}).

        Formula:
            H(X|Y) = -Σ p(x,y) log₂ p(x|y)
                   = H(X,Y) - H(Y)

        This measures remaining uncertainty after knowing past.

        Reference:
            [1] Shannon (1948)
            [6] Cover & Thomas (2006)
        """
        if len(returns) <= lag:
            return 1.0

        # Create joint distribution
        x = returns[lag:]
        y = returns[:-lag]

        # Discretize
        bins = np.percentile(returns, np.linspace(0, 100, self.n_bins + 1))
        bins[-1] += 1e-10
        x_dig = np.clip(np.digitize(x, bins[:-1]) - 1, 0, self.n_bins - 1)
        y_dig = np.clip(np.digitize(y, bins[:-1]) - 1, 0, self.n_bins - 1)

        # Joint counts
        joint_counts = np.zeros((self.n_bins, self.n_bins))
        for i in range(len(x)):
            joint_counts[x_dig[i], y_dig[i]] += 1

        joint_probs = joint_counts / len(x)

        # Marginal for Y
        p_y = joint_probs.sum(axis=0)

        # Joint entropy H(X,Y)
        H_XY = self._shannon_entropy(joint_probs.flatten())

        # Marginal entropy H(Y)
        H_Y = self._shannon_entropy(p_y)

        # Conditional entropy H(X|Y) = H(X,Y) - H(Y)
        return max(0, H_XY - H_Y)

    def _complexity_features(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate algorithmic complexity features.

        Kolmogorov Complexity (approximated via compression):
            K(x) = min{|p| : U(p) = x}
            Approximation: K(x) ≈ len(compress(x))

        Normalized Compression Distance:
            NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))

        For financial time series after pattern removal:
            NCD ≈ 0.85-0.95 (near-random)

        Reference:
            [3] Kolmogorov (1965): "Three approaches to the quantitative
                definition of information"
        """
        # Convert to bytes for compression
        quantized = np.round(returns * 10000).astype(np.int16)
        data_bytes = quantized.tobytes()

        # Compressed length (approximates Kolmogorov complexity)
        compressed = zlib.compress(data_bytes, level=9)
        compressed_len = len(compressed)
        original_len = len(data_bytes)

        # Compression ratio (higher = more random/complex)
        compression_ratio = compressed_len / original_len

        # Normalized complexity (0 = highly compressible, 1 = incompressible)
        # Random data compresses to ~1.0 ratio
        normalized_complexity = min(1.0, compression_ratio)

        # Shuffled baseline (maximum randomness for this data)
        shuffled = np.random.permutation(quantized)
        shuffled_compressed = zlib.compress(shuffled.tobytes(), level=9)
        shuffled_ratio = len(shuffled_compressed) / original_len

        # Excess complexity (above shuffled baseline = structure)
        excess_structure = max(0, shuffled_ratio - compression_ratio)

        return {
            'info_kolmogorov_approx': compressed_len,
            'info_compression_ratio': compression_ratio,
            'info_normalized_complexity': normalized_complexity,
            'info_excess_structure': excess_structure,
            'info_randomness_score': compression_ratio / shuffled_ratio if shuffled_ratio > 0 else 1.0
        }

    def _mutual_information_features(
        self,
        returns: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate mutual information between features and returns.

        Mutual Information:
            I(X;Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
            I(X;Y) ≤ min(H(X), H(Y))

        Fano's Inequality (accuracy bound):
            P_e ≥ [H(Y|X) - 1] / log(|Y|)
            Max Accuracy ≈ 0.5 + I(X;Y) / (2 * H(Y))

        Reference:
            [1] Shannon (1948)
            [6] Cover & Thomas (2006): Elements of Information Theory
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Use PCA for dimensionality reduction
        if features.shape[1] > 5:
            features_centered = features - np.mean(features, axis=0)
            _, s, Vt = np.linalg.svd(features_centered, full_matrices=False)
            # Keep top 5 components
            features = features_centered @ Vt[:5].T

        # Direction as target
        direction = (returns > 0).astype(int)

        # Calculate MI for each feature dimension
        mi_values = []
        for i in range(features.shape[1]):
            mi = self._mutual_information_discrete(features[:, i], direction)
            mi_values.append(mi)

        # Aggregate MI (upper bound is min of individual entropies)
        total_mi = sum(mi_values)

        # Direction entropy
        p_up = np.mean(direction)
        H_Y = -p_up * np.log2(p_up + 1e-10) - (1 - p_up) * np.log2(1 - p_up + 1e-10)

        # Max accuracy from Fano's inequality
        if H_Y > 0:
            max_accuracy_fano = 0.5 + total_mi / (2 * H_Y)
        else:
            max_accuracy_fano = 0.5

        max_accuracy_fano = min(1.0, max_accuracy_fano)

        # Information efficiency (how much of max info is captured)
        info_efficiency = total_mi / H_Y if H_Y > 0 else 0

        return {
            'info_mutual_information': total_mi,
            'info_max_accuracy_fano': max_accuracy_fano,
            'info_efficiency': info_efficiency,
            'info_residual_entropy': max(0, H_Y - total_mi),
            'info_feature_redundancy': 1 - (total_mi / (sum(mi_values) + 1e-10))
        }

    def _mutual_information_discrete(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> float:
        """
        Calculate MI between continuous X and discrete Y.

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Discretize X
        bins = np.percentile(X, np.linspace(0, 100, self.n_bins + 1))
        bins[-1] += 1e-10
        X_dig = np.clip(np.digitize(X, bins[:-1]) - 1, 0, self.n_bins - 1)

        # Joint distribution
        n_y = len(np.unique(Y))
        joint_counts = np.zeros((self.n_bins, n_y))
        for i in range(len(X)):
            joint_counts[X_dig[i], Y[i]] += 1

        joint_probs = joint_counts / len(X)

        # Marginals
        p_x = joint_probs.sum(axis=1)
        p_y = joint_probs.sum(axis=0)

        # Entropies
        H_X = self._shannon_entropy(p_x)
        H_Y = self._shannon_entropy(p_y)
        H_XY = self._shannon_entropy(joint_probs.flatten())

        # MI
        return max(0, H_X + H_Y - H_XY)

    def _transfer_entropy_features(
        self,
        target: np.ndarray,
        other_assets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate transfer entropy from other assets to target.

        Transfer Entropy (Schreiber 2000):
            T_{X→Y} = Σ p(y_{t+1}, y_t^(k), x_t^(l)) log[p(y_{t+1}|y_t^(k), x_t^(l)) / p(y_{t+1}|y_t^(k))]

        Measures information flow: how much does X help predict Y
        beyond what Y's own past provides?

        Reference:
            [2] Schreiber, T. (2000). "Measuring Information Transfer."
                Physical Review Letters, 85(2), 461.
        """
        results = {}
        total_te_in = 0
        total_te_out = 0

        for name, other in other_assets.items():
            # Ensure same length
            min_len = min(len(target), len(other))
            y = target[:min_len]
            x = other[:min_len]

            # Transfer entropy X→Y
            te_in = self._transfer_entropy(x, y, self.history_length)

            # Transfer entropy Y→X
            te_out = self._transfer_entropy(y, x, self.history_length)

            # Net information flow
            net_flow = te_in - te_out

            results[f'info_te_from_{name}'] = te_in
            results[f'info_te_to_{name}'] = te_out
            results[f'info_te_net_{name}'] = net_flow

            total_te_in += te_in
            total_te_out += te_out

        results['info_te_total_inflow'] = total_te_in
        results['info_te_total_outflow'] = total_te_out
        results['info_te_net_total'] = total_te_in - total_te_out

        return results

    def _transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        k: int
    ) -> float:
        """
        Calculate transfer entropy from source to target.

        T_{X→Y} = H(Y_{t+1} | Y_t^k) - H(Y_{t+1} | Y_t^k, X_t^k)

        Where Y_t^k = (Y_t, Y_{t-1}, ..., Y_{t-k+1})

        Reference:
            [2] Schreiber (2000)
        """
        n = len(target)
        if n <= k + 1:
            return 0.0

        # Discretize
        bins_y = np.percentile(target, np.linspace(0, 100, self.n_bins + 1))
        bins_x = np.percentile(source, np.linspace(0, 100, self.n_bins + 1))
        bins_y[-1] += 1e-10
        bins_x[-1] += 1e-10

        y_dig = np.clip(np.digitize(target, bins_y[:-1]) - 1, 0, self.n_bins - 1)
        x_dig = np.clip(np.digitize(source, bins_x[:-1]) - 1, 0, self.n_bins - 1)

        # Create state vectors
        # y_{t+1}, y_t^k, x_t^k
        states = []
        for t in range(k, n - 1):
            y_next = y_dig[t + 1]
            y_past = tuple(y_dig[t - j] for j in range(k))
            x_past = tuple(x_dig[t - j] for j in range(k))
            states.append((y_next, y_past, x_past))

        if len(states) == 0:
            return 0.0

        # Count joint and marginal occurrences
        from collections import Counter
        joint_yyx = Counter(states)
        joint_yy = Counter((s[0], s[1]) for s in states)
        joint_yx = Counter((s[1], s[2]) for s in states)
        marginal_y = Counter(s[1] for s in states)

        # Calculate TE
        te = 0.0
        total = len(states)

        for state, count in joint_yyx.items():
            y_next, y_past, x_past = state

            p_yyx = count / total
            p_yy = joint_yy[(y_next, y_past)] / total
            p_yx = joint_yx[(y_past, x_past)] / total
            p_y = marginal_y[y_past] / total

            if p_yy > 0 and p_yx > 0 and p_y > 0:
                # p(y_{t+1}|y_t^k, x_t^k) / p(y_{t+1}|y_t^k)
                # = p(y_{t+1}, y_t^k, x_t^k) * p(y_t^k) / (p(y_{t+1}, y_t^k) * p(y_t^k, x_t^k))
                ratio = (p_yyx * p_y) / (p_yy * p_yx)
                if ratio > 0:
                    te += p_yyx * np.log2(ratio)

        return max(0, te)


def compute_market_efficiency_score(returns: np.ndarray) -> Dict[str, float]:
    """
    Compute market efficiency score based on information theory.

    EMH (Fama 1970):
        Strong form: H(r_t | Ω_{t-1}) ≈ H(r_t)
        If conditional entropy ≈ unconditional entropy, market is efficient.

    Efficiency Score:
        E = H(r_t | r_{t-1}^k) / H(r_t)
        E ≈ 1 → Efficient (unpredictable)
        E < 1 → Inefficient (predictable)

    Reference:
        [5] Fama (1970): EMH
        [4] Lo & MacKinlay (1988): Variance ratio tests

    Args:
        returns: Array of returns

    Returns:
        Dictionary with efficiency metrics
    """
    calc = InformationTheoryFeatures()

    # Unconditional entropy (direction)
    direction = (returns > 0).astype(int)
    p_up = np.mean(direction)
    H_unconditional = -p_up * np.log2(p_up + 1e-10) - (1 - p_up) * np.log2(1 - p_up + 1e-10)

    # Conditional entropy at different lags
    H_conditional = {}
    for lag in [1, 2, 5, 10]:
        if len(returns) > lag:
            H_conditional[lag] = calc._conditional_entropy_lag(returns, lag)
        else:
            H_conditional[lag] = H_unconditional

    # Efficiency scores
    efficiency_scores = {}
    for lag, H_cond in H_conditional.items():
        if H_unconditional > 0:
            efficiency_scores[lag] = H_cond / H_unconditional
        else:
            efficiency_scores[lag] = 1.0

    # Overall efficiency (average)
    overall_efficiency = np.mean(list(efficiency_scores.values()))

    # Predictability = 1 - efficiency
    predictability = 1 - overall_efficiency

    # Max accuracy from efficiency
    # Roughly: accuracy ≈ 0.5 + 0.5 * predictability
    max_accuracy = 0.5 + 0.5 * predictability

    return {
        'efficiency_score': overall_efficiency,
        'predictability_score': predictability,
        'efficiency_lag_1': efficiency_scores.get(1, 1.0),
        'efficiency_lag_5': efficiency_scores.get(5, 1.0),
        'efficiency_lag_10': efficiency_scores.get(10, 1.0),
        'unconditional_entropy': H_unconditional,
        'implied_max_accuracy': max_accuracy
    }


# Convenience function
def create_information_features(
    returns: np.ndarray,
    features: Optional[np.ndarray] = None,
    other_assets: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Create all information theory features for a time series.

    Reference:
        [1-6] See module docstring for full citations

    Args:
        returns: Target return series
        features: Optional feature matrix
        other_assets: Optional dict of other asset returns

    Returns:
        Dictionary of all information features
    """
    calc = InformationTheoryFeatures()
    return calc.compute_all_features(returns, features, other_assets)


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    n = 5000

    # Generate synthetic returns with some structure
    noise = np.random.randn(n) * 0.01
    ar_component = np.zeros(n)
    for i in range(1, n):
        ar_component[i] = 0.1 * ar_component[i-1] + noise[i]
    returns = ar_component

    print("=" * 60)
    print("INFORMATION THEORY FEATURES")
    print("=" * 60)

    features = create_information_features(returns)
    for name, value in features.items():
        print(f"{name:35} {value:.6f}")

    print("\n" + "=" * 60)
    print("MARKET EFFICIENCY ANALYSIS")
    print("=" * 60)

    efficiency = compute_market_efficiency_score(returns)
    for name, value in efficiency.items():
        print(f"{name:35} {value:.6f}")
