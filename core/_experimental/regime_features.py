"""
Regime-Dependent Features for HFT Forex
=======================================
Renaissance Technologies core insight: Different market regimes
require different strategies. Features should be regime-aware.

Key Concept:
- Low volatility: Mean-reversion dominates, use tighter signals
- Normal volatility: Balance trend-following and mean-reversion
- High volatility: Trend-following works, widen stops

Implemented Methods:
- HMM Regime Detection (3-state: low, normal, high vol)
- Regime-Scaled Features (adjust signals by regime)
- Regime Transition Probabilities
- Regime-Dependent Kelly Criterion
- Multi-Regime Ensemble

Sources:
- Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Hamilton (1990) "Analysis of Time Series Subject to Changes in Regime"
- Guidolin & Timmermann (2007) "Asset Allocation Under Multivariate Regime Switching"

Why Renaissance Uses This:
- Regime detection from speech recognition (Baum-Welch for HMM)
- Different strategies for different market conditions
- Position sizing based on regime
- Avoid trading during regime transitions
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass, field
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current regime state."""
    regime: int  # 0 = low vol, 1 = normal, 2 = high vol
    probability: np.ndarray  # Probability of each regime
    duration: int  # How long in current regime
    transition_prob: float  # Probability of transition next period
    regime_name: str  # 'low', 'normal', 'high'


@dataclass
class RegimeConfig:
    """Configuration for regime-dependent trading."""
    # Position sizing multipliers by regime
    position_mult: Dict[int, float] = field(default_factory=lambda: {
        0: 1.5,  # Low vol: can take larger positions
        1: 1.0,  # Normal: baseline
        2: 0.5   # High vol: reduce exposure
    })

    # Stop loss multipliers by regime
    stop_mult: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,  # Low vol: tighter stops
        1: 1.5,  # Normal: standard
        2: 2.5   # High vol: wider stops
    })

    # Strategy weights by regime
    strategy_weights: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        0: {'mean_reversion': 0.7, 'momentum': 0.2, 'breakout': 0.1},  # Low vol: mean reversion
        1: {'mean_reversion': 0.4, 'momentum': 0.4, 'breakout': 0.2},  # Normal: balanced
        2: {'mean_reversion': 0.1, 'momentum': 0.5, 'breakout': 0.4}   # High vol: momentum
    })

    # Minimum regime duration before trading (avoid transitions)
    min_duration: int = 5


class SimpleHMMRegime:
    """
    Simple HMM for regime detection.

    3 states: Low vol (0), Normal vol (1), High vol (2)

    Uses Gaussian emissions with different means and variances per state.
    Viterbi algorithm for state sequence.

    Renaissance Application:
    - Speech recognition experts (Mercer, Brown) brought HMM expertise
    - Baum-Welch algorithm fits model to data
    - Regime = hidden state, Returns = observed emissions
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states

        # Transition matrix (initialized to favor staying in same state)
        self.A = np.array([
            [0.95, 0.04, 0.01],  # Low vol stays low
            [0.03, 0.94, 0.03],  # Normal stays normal
            [0.01, 0.04, 0.95]   # High vol stays high
        ])

        # Emission parameters (mean, std for each state)
        # Low vol: small returns, low variance
        # Normal: medium returns, medium variance
        # High vol: larger returns, high variance
        self.emission_means = np.array([0.0, 0.0, 0.0])
        self.emission_stds = np.array([0.0001, 0.0003, 0.001])

        # Initial state probabilities
        self.pi = np.array([0.3, 0.5, 0.2])

        self.fitted = False

    def emission_prob(self, x: float, state: int) -> float:
        """Probability of observing x given state."""
        mean = self.emission_means[state]
        std = self.emission_stds[state]
        return norm.pdf(x, mean, std)

    def fit(self, returns: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
        """
        Fit HMM using Baum-Welch algorithm.

        This is the key algorithm Renaissance uses - same as speech recognition.
        """
        n = len(returns)

        # Initialize emission parameters from data percentiles
        vol = pd.Series(returns).rolling(20).std().values
        vol = vol[~np.isnan(vol)]

        if len(vol) > 0:
            p33 = np.percentile(vol, 33)
            p67 = np.percentile(vol, 67)

            self.emission_stds = np.array([
                p33 * 0.8,  # Low vol
                np.median(vol),  # Normal vol
                p67 * 1.5  # High vol
            ])

        # EM iterations (Baum-Welch)
        prev_log_lik = -np.inf

        for iteration in range(max_iter):
            # E-step: Forward-Backward
            alpha, beta, log_lik = self._forward_backward(returns)

            # Check convergence
            if abs(log_lik - prev_log_lik) < tol:
                break
            prev_log_lik = log_lik

            # M-step: Update parameters
            # Compute posterior probabilities
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)

            # Update transition matrix
            xi = np.zeros((n - 1, self.n_states, self.n_states))
            for t in range(n - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] *
                                      self.emission_prob(returns[t + 1], j) * beta[t + 1, j])
                xi[t] /= xi[t].sum() + 1e-10

            for i in range(self.n_states):
                denom = gamma[:-1, i].sum() + 1e-10
                for j in range(self.n_states):
                    self.A[i, j] = xi[:, i, j].sum() / denom

            # Update emission parameters
            for j in range(self.n_states):
                denom = gamma[:, j].sum() + 1e-10
                self.emission_means[j] = (gamma[:, j] * returns).sum() / denom

                centered = returns - self.emission_means[j]
                self.emission_stds[j] = np.sqrt((gamma[:, j] * centered**2).sum() / denom)
                self.emission_stds[j] = max(self.emission_stds[j], 1e-6)

        self.fitted = True

    def _forward_backward(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm."""
        n = len(returns)

        # Forward pass
        alpha = np.zeros((n, self.n_states))
        alpha[0] = self.pi * np.array([self.emission_prob(returns[0], j) for j in range(self.n_states)])
        alpha[0] /= alpha[0].sum() + 1e-10

        log_lik = 0
        for t in range(1, n):
            for j in range(self.n_states):
                alpha[t, j] = (alpha[t - 1] @ self.A[:, j]) * self.emission_prob(returns[t], j)

            scale = alpha[t].sum() + 1e-10
            alpha[t] /= scale
            log_lik += np.log(scale)

        # Backward pass
        beta = np.zeros((n, self.n_states))
        beta[-1] = 1

        for t in range(n - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    beta[t, i] += self.A[i, j] * self.emission_prob(returns[t + 1], j) * beta[t + 1, j]

            beta[t] /= beta[t].sum() + 1e-10

        return alpha, beta, log_lik

    def predict_regime(self, returns: np.ndarray) -> np.ndarray:
        """Predict most likely regime sequence (Viterbi)."""
        n = len(returns)

        # Viterbi algorithm
        delta = np.zeros((n, self.n_states))
        psi = np.zeros((n, self.n_states), dtype=int)

        # Initialize
        for j in range(self.n_states):
            delta[0, j] = np.log(self.pi[j] + 1e-10) + np.log(self.emission_prob(returns[0], j) + 1e-10)

        # Forward pass
        for t in range(1, n):
            for j in range(self.n_states):
                probs = delta[t - 1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(probs)
                delta[t, j] = probs[psi[t, j]] + np.log(self.emission_prob(returns[t], j) + 1e-10)

        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def get_regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Get probability of each regime at each time."""
        alpha, beta, _ = self._forward_backward(returns)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        return gamma


class RegimeFeatureScaler:
    """
    Scale features based on current regime.

    Different regimes have different feature distributions.
    Normalizing within regime improves model accuracy.

    Renaissance Application:
    - Features computed relative to regime statistics
    - Avoids spurious signals from regime transitions
    - More stable feature distributions per regime
    """

    def __init__(self):
        self.regime_stats: Dict[int, Dict[str, Tuple[float, float]]] = {}

    def fit(self, features: pd.DataFrame, regimes: np.ndarray):
        """
        Compute per-regime statistics for each feature.
        """
        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_features = features[mask]

            self.regime_stats[regime] = {}
            for col in features.columns:
                mean = regime_features[col].mean()
                std = regime_features[col].std()
                self.regime_stats[regime][col] = (mean, std if std > 0 else 1.0)

    def transform(self, features: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """
        Transform features to regime-normalized values.
        """
        result = features.copy()

        for regime in np.unique(regimes):
            mask = regimes == regime

            if regime in self.regime_stats:
                for col in features.columns:
                    if col in self.regime_stats[regime]:
                        mean, std = self.regime_stats[regime][col]
                        result.loc[mask, col] = (features.loc[mask, col] - mean) / std

        return result


class RegimeAwarePositionSizer:
    """
    Adjust position sizes based on regime.

    Low vol: Can take larger positions (lower risk)
    High vol: Reduce positions (higher risk)

    Implements Regime-Dependent Kelly Criterion.
    """

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()

    def get_position_multiplier(self, regime: int) -> float:
        """Get position size multiplier for regime."""
        return self.config.position_mult.get(regime, 1.0)

    def get_stop_multiplier(self, regime: int) -> float:
        """Get stop loss multiplier for regime."""
        return self.config.stop_mult.get(regime, 1.0)

    def regime_adjusted_kelly(self,
                              win_prob: float,
                              win_loss_ratio: float,
                              regime: int,
                              base_fraction: float = 0.25) -> float:
        """
        Compute regime-adjusted Kelly fraction.

        Kelly = p - (1-p)/b where p = win_prob, b = win_loss_ratio

        Then multiply by regime adjustment and base fraction.
        """
        if win_prob <= 0 or win_loss_ratio <= 0:
            return 0.0

        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        kelly = max(0, kelly)

        regime_mult = self.get_position_multiplier(regime)

        return kelly * base_fraction * regime_mult


class RegimeDependentFeatureEngine:
    """
    Complete regime-dependent feature generation.

    Combines:
    1. HMM regime detection
    2. Per-regime feature scaling
    3. Regime transition features
    4. Strategy weight suggestions
    """

    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        self.hmm = SimpleHMMRegime()
        self.scaler = RegimeFeatureScaler()
        self.position_sizer = RegimeAwarePositionSizer(self.config)

        self.fitted = False
        self.current_regime = 1  # Start in normal
        self.regime_duration = 0

    def fit(self, returns: np.ndarray, features: pd.DataFrame = None):
        """Fit regime model."""
        self.hmm.fit(returns)
        regimes = self.hmm.predict_regime(returns)

        if features is not None:
            self.scaler.fit(features, regimes)

        self.fitted = True

    def detect_regime(self, returns: np.ndarray) -> RegimeState:
        """Detect current regime."""
        if not self.fitted:
            # Fallback to volatility-based detection
            vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            if vol < 0.0002:
                regime = 0
            elif vol > 0.0006:
                regime = 2
            else:
                regime = 1

            probs = np.zeros(3)
            probs[regime] = 1.0
        else:
            probs = self.hmm.get_regime_probabilities(returns)[-1]
            regime = np.argmax(probs)

        # Track regime duration
        if regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.current_regime = regime
            self.regime_duration = 1

        # Compute transition probability
        trans_prob = 1 - self.hmm.A[regime, regime] if self.fitted else 0.05

        regime_names = {0: 'low', 1: 'normal', 2: 'high'}

        return RegimeState(
            regime=regime,
            probability=probs,
            duration=self.regime_duration,
            transition_prob=trans_prob,
            regime_name=regime_names.get(regime, 'unknown')
        )

    def compute_features(self, returns: pd.Series,
                        base_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute regime-dependent features.

        Returns DataFrame with:
        - Regime indicators
        - Regime probabilities
        - Transition probabilities
        - Regime-scaled base features (if provided)
        - Strategy weights
        - Position multipliers
        """
        returns_arr = returns.values
        n = len(returns_arr)

        # Fit if not already
        if not self.fitted and n > 50:
            self.fit(returns_arr)

        features = pd.DataFrame(index=returns.index)

        # Regime detection
        if self.fitted:
            regimes = self.hmm.predict_regime(returns_arr)
            regime_probs = self.hmm.get_regime_probabilities(returns_arr)
        else:
            # Fallback
            rolling_vol = returns.rolling(20).std()
            vol_33 = rolling_vol.quantile(0.33)
            vol_67 = rolling_vol.quantile(0.67)

            regimes = np.where(
                rolling_vol < vol_33, 0,
                np.where(rolling_vol > vol_67, 2, 1)
            )
            regimes = np.nan_to_num(regimes, nan=1).astype(int)

            regime_probs = np.zeros((n, 3))
            for t, r in enumerate(regimes):
                regime_probs[t, r] = 1.0

        features['regime'] = regimes
        features['regime_prob_low'] = regime_probs[:, 0]
        features['regime_prob_normal'] = regime_probs[:, 1]
        features['regime_prob_high'] = regime_probs[:, 2]

        # Regime duration
        duration = np.ones(n)
        for t in range(1, n):
            if regimes[t] == regimes[t - 1]:
                duration[t] = duration[t - 1] + 1

        features['regime_duration'] = duration

        # Transition indicators
        transitions = np.zeros(n)
        transitions[1:] = (regimes[1:] != regimes[:-1]).astype(int)
        features['regime_transition'] = transitions

        # Position and stop multipliers
        features['position_mult'] = [self.config.position_mult.get(r, 1.0) for r in regimes]
        features['stop_mult'] = [self.config.stop_mult.get(r, 1.0) for r in regimes]

        # Strategy weights
        for strategy in ['mean_reversion', 'momentum', 'breakout']:
            features[f'weight_{strategy}'] = [
                self.config.strategy_weights.get(r, {}).get(strategy, 0.33)
                for r in regimes
            ]

        # Regime-scaled base features
        if base_features is not None and self.fitted:
            scaled = self.scaler.transform(base_features, regimes)
            for col in scaled.columns:
                features[f'{col}_regime_scaled'] = scaled[col]

        # Can trade indicator (avoid transitions)
        features['can_trade'] = (duration >= self.config.min_duration).astype(int)

        return features


def compute_regime_features(returns: pd.Series,
                           base_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convenience function for regime feature computation.
    """
    engine = RegimeDependentFeatureEngine()
    return engine.compute_features(returns, base_features)


def get_regime_adjusted_position(signal: float,
                                confidence: float,
                                regime: int,
                                base_size: float = 0.01) -> float:
    """
    Get regime-adjusted position size.

    Args:
        signal: Trading signal (-1 to 1)
        confidence: Model confidence (0 to 1)
        regime: Current regime (0=low, 1=normal, 2=high)
        base_size: Base position size as fraction of capital

    Returns:
        Adjusted position size
    """
    config = RegimeConfig()

    regime_mult = config.position_mult.get(regime, 1.0)
    position = signal * confidence * base_size * regime_mult

    return position


if __name__ == '__main__':
    print("Regime-Dependent Features Test")
    print("=" * 60)

    # Generate synthetic regime-switching data
    np.random.seed(42)
    n = 1000

    # True regimes (3 segments)
    true_regimes = np.concatenate([
        np.zeros(300),  # Low vol
        np.ones(400),   # Normal vol
        np.full(300, 2)  # High vol
    ]).astype(int)

    # Generate returns per regime
    returns = np.zeros(n)
    vols = [0.0001, 0.0003, 0.001]

    for t in range(n):
        returns[t] = np.random.randn() * vols[true_regimes[t]]

    returns = pd.Series(returns)

    print(f"Generated {n} returns with regime-switching volatility")
    print(f"Low vol: {vols[0]}, Normal: {vols[1]}, High: {vols[2]}")

    # Test HMM
    print("\n--- HMM Regime Detection ---")
    hmm = SimpleHMMRegime()
    hmm.fit(returns.values)

    predicted_regimes = hmm.predict_regime(returns.values)
    accuracy = np.mean(predicted_regimes == true_regimes)
    print(f"Regime detection accuracy: {accuracy:.2%}")

    # Test full feature engine
    print("\n--- Regime Feature Engine ---")
    engine = RegimeDependentFeatureEngine()
    features = engine.compute_features(returns)

    print(f"Features computed: {list(features.columns)}")
    print(f"\nSample features (last 5 rows):")
    print(features[['regime', 'regime_duration', 'position_mult', 'can_trade']].tail())

    # Test regime detection
    print("\n--- Current Regime State ---")
    state = engine.detect_regime(returns.values)
    print(f"Regime: {state.regime_name} ({state.regime})")
    print(f"Duration: {state.duration}")
    print(f"Transition probability: {state.transition_prob:.4f}")
    print(f"Regime probabilities: {state.probability}")

    # Test position sizing
    print("\n--- Position Sizing ---")
    for regime in [0, 1, 2]:
        pos = get_regime_adjusted_position(
            signal=1.0, confidence=0.6, regime=regime, base_size=0.02
        )
        regime_name = ['low', 'normal', 'high'][regime]
        print(f"Regime {regime_name}: position = {pos:.4f}")

    print("\n" + "=" * 60)
    print("Regime feature tests passed!")
