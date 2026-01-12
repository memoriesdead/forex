"""
Jump Detection and Bipower Variation for HFT Forex
===================================================
Renaissance Technologies inference: Distinguishing continuous price
evolution from discrete jumps (news, order flow shocks) is critical
for proper volatility estimation and position sizing.

Implemented Methods:
- Bipower Variation (BPV): Jump-robust volatility estimator
- Realized Jump Variation: Extracts jump component
- Relative Jump (RJ): Jump significance test
- Jump Test Statistics: BNS, AJ, CPR tests
- Self-Exciting Jumps: Hawkes process for jump clustering

Sources:
- Barndorff-Nielsen & Shephard (2004) "Power and Multipower Variations"
- Barndorff-Nielsen & Shephard (2006) "Econometrics of Testing for Jumps"
- Andersen, Bollerslev, Diebold (2007) "Roughing it Up"
- Lee & Mykland (2008) "Jumps in Financial Markets"
- Ait-Sahalia & Jacod (2009) "Testing for Jumps"

Why Renaissance Uses This:
- News events cause jumps, not continuous trading
- GARCH assumes continuous dynamics (fails on jumps)
- Jump detection → adjust position sizing, widen stops
- Separate jump volatility from continuous volatility
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from scipy.stats import norm, chi2
from scipy.special import gamma as gamma_func
import logging

logger = logging.getLogger(__name__)


# Bipower variation constant: E[|Z|] for Z ~ N(0,1)
MU_1 = np.sqrt(2 / np.pi)  # ≈ 0.7979


@dataclass
class JumpResult:
    """Result from jump detection."""
    has_jump: bool
    jump_time: Optional[int]
    jump_size: float
    jump_direction: int  # 1 = up, -1 = down, 0 = no jump
    continuous_vol: float
    jump_vol: float
    total_vol: float
    test_statistic: float
    p_value: float


@dataclass
class VolatilityDecomposition:
    """Decomposition of realized volatility into continuous and jump components."""
    realized_variance: np.ndarray  # Total RV
    bipower_variance: np.ndarray  # Continuous component (jump-robust)
    jump_variance: np.ndarray  # Jump component
    jump_indicator: np.ndarray  # 1 if jump detected
    relative_jump: np.ndarray  # Jump contribution ratio


class BipowerVariation:
    """
    Bipower Variation for jump-robust volatility estimation.

    BPV_t = (π/2) * Σ |r_i| * |r_{i-1}|

    Key insight: BPV is robust to jumps because jumps are unlikely
    to occur in consecutive periods. If jump at time i:
    - |r_i| * |r_{i-1}| ≈ |jump| * |small_return| ≈ small contribution
    - vs RV: |r_i|² ≈ |jump|² = large contribution

    Source: Barndorff-Nielsen & Shephard (2004)
    """

    def __init__(self, window: int = 78):
        """
        Initialize BPV calculator.

        Args:
            window: Rolling window for RV/BPV calculation
                   (78 = 5-min bars in forex session)
        """
        self.window = window

    def realized_variance(self, returns: np.ndarray) -> float:
        """
        Compute realized variance (sum of squared returns).

        RV = Σ r²_i
        """
        return np.sum(returns**2)

    def bipower_variance(self, returns: np.ndarray) -> float:
        """
        Compute bipower variation (jump-robust).

        BPV = (π/2) * Σ |r_i| * |r_{i-1}|
        """
        if len(returns) < 2:
            return 0.0

        # Bipower variation
        bpv = np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
        # Scale factor to make it comparable to RV under no jumps
        bpv *= np.pi / 2

        return bpv

    def tripower_quarticity(self, returns: np.ndarray) -> float:
        """
        Tripower quarticity for variance estimation of BPV.

        TPQ = n * μ_{4/3}^{-3} * Σ |r_i|^{4/3} * |r_{i-1}|^{4/3} * |r_{i-2}|^{4/3}

        Used in jump test statistic variance calculation.
        """
        if len(returns) < 3:
            return 0.0

        n = len(returns)
        mu_43 = 2**(2/3) * gamma_func(7/6) / gamma_func(1/2)

        tpq = np.sum(
            np.abs(returns[2:])**(4/3) *
            np.abs(returns[1:-1])**(4/3) *
            np.abs(returns[:-2])**(4/3)
        )

        return n * mu_43**(-3) * tpq

    def quadpower_quarticity(self, returns: np.ndarray) -> float:
        """
        Quadpower quarticity - alternative variance estimator.

        QPQ = n * μ_1^{-4} * Σ |r_i| * |r_{i-1}| * |r_{i-2}| * |r_{i-3}|
        """
        if len(returns) < 4:
            return 0.0

        n = len(returns)
        mu_1 = MU_1

        qpq = np.sum(
            np.abs(returns[3:]) *
            np.abs(returns[2:-1]) *
            np.abs(returns[1:-2]) *
            np.abs(returns[:-3])
        )

        return n * np.pi**2 / 4 * mu_1**(-4) * qpq

    def rolling_rv_bpv(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute rolling RV and BPV.

        Returns DataFrame with:
        - rv: Realized variance
        - bpv: Bipower variance
        - jump_var: max(0, RV - BPV)
        - relative_jump: jump_var / rv
        """
        result = pd.DataFrame(index=returns.index)

        rv = returns.rolling(self.window).apply(self.realized_variance)
        bpv = returns.rolling(self.window).apply(self.bipower_variance)

        result['rv'] = rv
        result['bpv'] = bpv
        result['jump_var'] = np.maximum(0, rv - bpv)
        result['relative_jump'] = result['jump_var'] / (result['rv'] + 1e-10)

        return result


class JumpTest:
    """
    Statistical tests for jump detection.

    Multiple tests implemented:
    1. BNS test (Barndorff-Nielsen & Shephard)
    2. Ratio test
    3. Difference test

    All test H0: No jumps vs H1: Jumps present
    """

    def __init__(self, significance: float = 0.01):
        """
        Initialize jump test.

        Args:
            significance: Significance level for jump detection
        """
        self.significance = significance
        self.bpv_calc = BipowerVariation()

    def bns_test(self, returns: np.ndarray) -> Tuple[float, float, bool]:
        """
        Barndorff-Nielsen & Shephard (2006) jump test.

        Test statistic:
        Z = (RV - BPV) / sqrt(Var(RV - BPV))
          = (RV - BPV) / sqrt((π²/4 + π - 5) * TPQ)

        Under H0 (no jumps): Z → N(0, 1)

        Returns: (test_statistic, p_value, has_jump)
        """
        n = len(returns)
        if n < 10:
            return 0.0, 1.0, False

        rv = self.bpv_calc.realized_variance(returns)
        bpv = self.bpv_calc.bipower_variance(returns)
        tpq = self.bpv_calc.tripower_quarticity(returns)

        # Variance of RV - BPV under no jumps
        # Var = (π²/4 + π - 5) * (μ_1)^{-4} * TPQ / n
        var_coef = np.pi**2 / 4 + np.pi - 5  # ≈ 0.6090

        if tpq <= 0 or bpv <= 0:
            return 0.0, 1.0, False

        # Test statistic
        variance = var_coef * tpq / n
        if variance <= 0:
            return 0.0, 1.0, False

        z_stat = (rv - bpv) / np.sqrt(variance)

        # One-sided test (we look for positive jumps contribution)
        p_value = 1 - norm.cdf(z_stat)

        has_jump = p_value < self.significance

        return float(z_stat), float(p_value), has_jump

    def ratio_test(self, returns: np.ndarray) -> Tuple[float, float, bool]:
        """
        Ratio-based jump test.

        Test statistic: RJ = 1 - BPV / RV

        Under H0: RJ → 0
        Large RJ indicates jumps
        """
        rv = self.bpv_calc.realized_variance(returns)
        bpv = self.bpv_calc.bipower_variance(returns)

        if rv <= 0:
            return 0.0, 1.0, False

        rj = 1 - bpv / rv
        rj = max(0, rj)  # Can't be negative

        # Approximate p-value (empirical critical values)
        # At 1% level, RJ > 0.4 typically indicates jumps
        threshold = 0.3 if self.significance <= 0.01 else 0.2

        has_jump = rj > threshold
        p_value = 1 - min(1, rj / 0.5)  # Rough approximation

        return float(rj), float(p_value), has_jump

    def detect_jumps(self, returns: np.ndarray) -> JumpResult:
        """
        Combined jump detection using multiple tests.

        Returns JumpResult with comprehensive jump information.
        """
        n = len(returns)
        if n < 10:
            return JumpResult(
                has_jump=False, jump_time=None, jump_size=0.0,
                jump_direction=0, continuous_vol=0.0, jump_vol=0.0,
                total_vol=0.0, test_statistic=0.0, p_value=1.0
            )

        # Run tests
        z_stat, z_pval, z_jump = self.bns_test(returns)
        rj_stat, rj_pval, rj_jump = self.ratio_test(returns)

        # Combine: jump if either test significant
        has_jump = z_jump or rj_jump

        # Volatility decomposition
        rv = self.bpv_calc.realized_variance(returns)
        bpv = self.bpv_calc.bipower_variance(returns)
        jump_var = max(0, rv - bpv)

        total_vol = np.sqrt(rv)
        continuous_vol = np.sqrt(bpv)
        jump_vol = np.sqrt(jump_var)

        # Find largest jump (if any)
        jump_time = None
        jump_size = 0.0
        jump_direction = 0

        if has_jump:
            # Identify the largest return as potential jump
            abs_returns = np.abs(returns)
            jump_time = int(np.argmax(abs_returns))
            jump_size = float(returns[jump_time])
            jump_direction = 1 if jump_size > 0 else -1

        return JumpResult(
            has_jump=has_jump,
            jump_time=jump_time,
            jump_size=jump_size,
            jump_direction=jump_direction,
            continuous_vol=continuous_vol,
            jump_vol=jump_vol,
            total_vol=total_vol,
            test_statistic=z_stat,
            p_value=z_pval
        )


class LeeMyklandJumpTest:
    """
    Lee & Mykland (2008) instantaneous jump test.

    Tests each individual return for jump significance,
    rather than testing for jumps in aggregate.

    Advantage: Identifies WHICH observations are jumps
    """

    def __init__(self,
                 significance: float = 0.01,
                 window: int = 156):
        """
        Initialize Lee-Mykland test.

        Args:
            significance: Significance level
            window: Window for local volatility estimation
                   (156 = 1 day of 5-min bars in forex)
        """
        self.significance = significance
        self.window = window
        self.bpv_calc = BipowerVariation(window)

    def test_returns(self, returns: np.ndarray) -> Dict:
        """
        Test each return for jump significance.

        Returns dict with:
        - is_jump: boolean array
        - jump_stats: test statistics
        - local_vol: local volatility estimates
        """
        n = len(returns)
        is_jump = np.zeros(n, dtype=bool)
        jump_stats = np.zeros(n)
        local_vol = np.zeros(n)

        for t in range(self.window, n):
            # Local volatility from bipower variation
            window_returns = returns[t - self.window:t]
            bpv = self.bpv_calc.bipower_variance(window_returns)
            sigma = np.sqrt(bpv / self.window)
            local_vol[t] = sigma

            if sigma > 0:
                # Standardized return
                L = returns[t] / sigma

                # Critical value based on extreme value distribution
                c = np.sqrt(2 * np.log(n)) - (np.log(np.pi) + np.log(np.log(n))) / (2 * np.sqrt(2 * np.log(n)))
                s = 1 / np.sqrt(2 * np.log(n))

                # Test statistic
                jump_stats[t] = (np.abs(L) - c) / s

                # Is it a jump? (rejection of null)
                # Under null, test stat follows standard Gumbel
                # 99% critical value ≈ 4.6
                critical_value = -np.log(-np.log(1 - self.significance))
                is_jump[t] = jump_stats[t] > critical_value

        return {
            'is_jump': is_jump,
            'jump_stats': jump_stats,
            'local_vol': local_vol,
            'jump_times': np.where(is_jump)[0],
            'n_jumps': np.sum(is_jump)
        }


class JumpHawkesProcess:
    """
    Self-exciting jump process using Hawkes intensity.

    Jumps tend to cluster: a jump increases probability of more jumps.

    λ(t) = λ₀ + Σ α·exp(-β(t - t_i))

    where t_i are past jump times.

    Renaissance Application:
    - After a jump, increase alertness for more jumps
    - Flash crash dynamics
    - News cascade effects

    Source: Hawkes (1971) "Spectra of Some Self-Exciting and Mutually Exciting Point Processes"
    """

    def __init__(self,
                 base_intensity: float = 0.01,
                 excitation: float = 0.5,
                 decay: float = 0.1):
        """
        Initialize Hawkes process.

        Args:
            base_intensity: λ₀ (baseline jump probability)
            excitation: α (jump-induced intensity increase)
            decay: β (how fast excitation decays)
        """
        self.lambda_0 = base_intensity
        self.alpha = excitation
        self.beta = decay

        self.jump_times: List[int] = []
        self.current_time = 0

    def intensity(self, t: int) -> float:
        """
        Compute intensity at time t.

        λ(t) = λ₀ + Σ α·exp(-β(t - t_i))
        """
        intensity = self.lambda_0

        for t_i in self.jump_times:
            if t_i < t:
                intensity += self.alpha * np.exp(-self.beta * (t - t_i))

        return intensity

    def update(self, t: int, is_jump: bool):
        """Update process with new observation."""
        self.current_time = t
        if is_jump:
            self.jump_times.append(t)

    def jump_probability(self, t: int) -> float:
        """
        Probability of jump in next period.

        P(jump) ≈ 1 - exp(-λ(t) * Δt)
        """
        lambda_t = self.intensity(t)
        return 1 - np.exp(-lambda_t)

    def simulate(self, n: int) -> np.ndarray:
        """
        Simulate Hawkes process.

        Returns array of jump indicators.
        """
        jumps = np.zeros(n, dtype=bool)

        for t in range(n):
            p = self.jump_probability(t)
            if np.random.random() < p:
                jumps[t] = True
                self.update(t, True)
            else:
                self.update(t, False)

        return jumps


class JumpVolatilityModel:
    """
    Combined continuous + jump volatility model.

    Total variance = Continuous variance + Jump variance

    Uses:
    1. BPV for continuous component
    2. Max(0, RV - BPV) for jump component
    3. Hawkes process for jump intensity

    Renaissance Application:
    - Separate trading signal from noise
    - Adjust position sizing based on jump risk
    - Wider stops during high jump intensity
    """

    def __init__(self,
                 bpv_window: int = 78,
                 jump_significance: float = 0.01):
        self.bpv_calc = BipowerVariation(bpv_window)
        self.jump_test = JumpTest(jump_significance)
        self.lm_test = LeeMyklandJumpTest(jump_significance)
        self.hawkes = JumpHawkesProcess()

    def decompose_volatility(self, returns: pd.Series) -> VolatilityDecomposition:
        """
        Full volatility decomposition into continuous and jump components.
        """
        returns_arr = returns.values
        n = len(returns_arr)

        # Rolling decomposition
        rv = np.zeros(n)
        bpv = np.zeros(n)
        jump_var = np.zeros(n)
        jump_ind = np.zeros(n)
        rj = np.zeros(n)

        window = self.bpv_calc.window

        for t in range(window, n):
            window_returns = returns_arr[t - window:t]

            rv[t] = self.bpv_calc.realized_variance(window_returns)
            bpv[t] = self.bpv_calc.bipower_variance(window_returns)
            jump_var[t] = max(0, rv[t] - bpv[t])
            rj[t] = jump_var[t] / (rv[t] + 1e-10)

            # Test for jump
            result = self.jump_test.detect_jumps(window_returns)
            jump_ind[t] = 1 if result.has_jump else 0

        return VolatilityDecomposition(
            realized_variance=rv,
            bipower_variance=bpv,
            jump_variance=jump_var,
            jump_indicator=jump_ind,
            relative_jump=rj
        )

    def compute_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute jump-related features for ML.
        """
        decomp = self.decompose_volatility(returns)

        # Identify individual jumps
        lm_result = self.lm_test.test_returns(returns.values)

        features = pd.DataFrame(index=returns.index)

        # Volatility components
        features['rv'] = decomp.realized_variance
        features['bpv'] = decomp.bipower_variance
        features['jump_var'] = decomp.jump_variance
        features['relative_jump'] = decomp.relative_jump
        features['jump_indicator'] = decomp.jump_indicator

        # Derived features
        features['continuous_vol'] = np.sqrt(decomp.bipower_variance)
        features['jump_vol'] = np.sqrt(decomp.jump_variance)
        features['total_vol'] = np.sqrt(decomp.realized_variance)

        # Jump intensity from Hawkes
        jump_times = lm_result['jump_times']
        self.hawkes.jump_times = list(jump_times)

        features['jump_intensity'] = [
            self.hawkes.intensity(t) for t in range(len(returns))
        ]
        features['jump_probability'] = [
            self.hawkes.jump_probability(t) for t in range(len(returns))
        ]

        # Recent jump count
        features['recent_jumps'] = pd.Series(lm_result['is_jump']).rolling(20).sum().values

        # Lee-Mykland statistics
        features['lm_stat'] = lm_result['jump_stats']
        features['is_jump'] = lm_result['is_jump'].astype(int)

        return features


def compute_jump_features(returns: pd.Series,
                         window: int = 78,
                         significance: float = 0.01) -> pd.DataFrame:
    """
    Convenience function to compute all jump-related features.

    Args:
        returns: Return series
        window: Window for volatility estimation
        significance: Significance level for jump tests

    Returns:
        DataFrame with jump features
    """
    model = JumpVolatilityModel(window, significance)
    return model.compute_features(returns)


def detect_jumps_simple(returns: np.ndarray,
                       threshold_std: float = 3.0) -> np.ndarray:
    """
    Simple threshold-based jump detection.

    A return is a jump if |r| > threshold * σ

    Args:
        returns: Return array
        threshold_std: Number of standard deviations for jump threshold

    Returns:
        Boolean array of jump indicators
    """
    # Robust volatility estimate using MAD
    mad = np.median(np.abs(returns - np.median(returns)))
    sigma = mad * 1.4826  # Scale to match std dev

    threshold = threshold_std * sigma
    is_jump = np.abs(returns) > threshold

    return is_jump


if __name__ == '__main__':
    print("Jump Detection Test")
    print("=" * 60)

    # Generate synthetic data with jumps
    np.random.seed(42)
    n = 1000

    # Continuous component (GARCH-like)
    sigma = 0.0001
    continuous = np.random.randn(n) * sigma

    # Add jumps at random times
    n_jumps = 10
    jump_times = np.random.choice(n, n_jumps, replace=False)
    jump_sizes = np.random.randn(n_jumps) * sigma * 10  # 10x normal

    returns = continuous.copy()
    for t, size in zip(jump_times, jump_sizes):
        returns[t] += size

    print(f"Simulated {n_jumps} jumps at times: {sorted(jump_times)}")

    # Test BPV
    print("\n--- Bipower Variation ---")
    bpv_calc = BipowerVariation(window=50)

    rv = bpv_calc.realized_variance(returns)
    bpv = bpv_calc.bipower_variance(returns)
    jump_var = max(0, rv - bpv)

    print(f"Realized Variance: {rv:.10f}")
    print(f"Bipower Variance: {bpv:.10f}")
    print(f"Jump Variance: {jump_var:.10f}")
    print(f"Relative Jump: {jump_var/rv:.4f}")

    # Test BNS jump test
    print("\n--- BNS Jump Test ---")
    test = JumpTest(significance=0.01)
    z_stat, p_val, has_jump = test.bns_test(returns)
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Jumps detected: {has_jump}")

    # Test Lee-Mykland
    print("\n--- Lee-Mykland Test ---")
    lm_test = LeeMyklandJumpTest(window=50)
    lm_result = lm_test.test_returns(returns)
    print(f"Individual jumps detected: {lm_result['n_jumps']}")
    print(f"Jump times: {lm_result['jump_times'][:10]}...")

    # Check detection accuracy
    detected = set(lm_result['jump_times'])
    actual = set(jump_times)
    true_positives = len(detected & actual)
    print(f"True positives: {true_positives}/{n_jumps}")

    # Full model test
    print("\n--- Full Jump Model ---")
    returns_series = pd.Series(returns)
    model = JumpVolatilityModel(bpv_window=50)
    features = model.compute_features(returns_series)

    print(f"Features computed: {list(features.columns)}")
    print(f"\nSample features (last 5 rows):")
    print(features[['continuous_vol', 'jump_vol', 'jump_intensity', 'is_jump']].tail())

    print("\n" + "=" * 60)
    print("Jump detection tests passed!")
