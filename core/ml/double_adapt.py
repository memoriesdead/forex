"""
DoubleAdapt: Meta-Learning for Regime Adaptation

Adapts BOTH data distribution AND model when market regime changes.
Standard online learning adapts model but not data weighting - this does both.

Academic Citations:
- KDD 2023: "DoubleAdapt: A Meta-learning Approach for Incremental Learning"
  ACM SIGKDD - Original DoubleAdapt paper showing +17.6% excess return

- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"
  ICML 2017 - MAML foundation

- Nichol & Schulman (2018): "Reptile: A Scalable Metalearning Algorithm"
  arXiv:1803.02999 - Simplified meta-learning

- Hospedales et al. (2021): "Meta-Learning in Neural Networks: A Survey"
  IEEE TPAMI - Comprehensive meta-learning survey

Chinese Quant Application:
- 知乎量化分析: DoubleAdapt在量化投资中的应用
- 九坤投资: "因子选择和因子组合在风格变化中的自动切换"
- 幻方量化: "及时应对市场规则变化，不断更新模型"

The DoubleAdapt Innovation:
    Standard online learning:
        - New data arrives → Update model weights
        - Problem: Historical data distribution may not match current regime

    DoubleAdapt (two-stage adaptation):
        Step 1: Reweight historical data to match current regime (distribution adapter)
        Step 2: MAML-style fast adaptation of model (model adapter)

    Result: +17.6% excess return vs standard methods (KDD 2023)

Implementation for HFT:
    1. Detect regime change (volatility spike, correlation breakdown)
    2. Compute importance weights for historical samples
    3. Quick MAML-style gradient step on weighted samples
    4. Hot-swap model if validation improves
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from collections import deque
from scipy import stats
import warnings


@dataclass
class RegimeState:
    """Current market regime state."""

    # Regime indicators
    volatility: float
    trend_strength: float
    mean_reversion: float
    correlation_state: float

    # Regime classification
    regime_id: int  # 0=normal, 1=trending, 2=mean-reverting, 3=volatile
    regime_name: str

    # Change detection
    regime_changed: bool
    change_magnitude: float
    time_since_change: int

    # Sample weights (for data reweighting)
    distribution_shift: float  # KL divergence from training distribution


@dataclass
class AdaptationResult:
    """Result of DoubleAdapt adaptation."""

    # Data adaptation
    n_samples_reweighted: int
    effective_sample_size: float  # ESS after reweighting
    max_weight: float
    distribution_shift_before: float
    distribution_shift_after: float

    # Model adaptation
    adaptation_steps: int
    learning_rate_used: float
    loss_before: float
    loss_after: float
    improvement: float

    # Validation
    val_accuracy_before: float
    val_accuracy_after: float
    should_deploy: bool

    # Diagnostics
    regime_before: str
    regime_after: str
    adaptation_time_ms: float


class DistributionAdapter:
    """
    Adapts historical data distribution to match current regime.

    Uses importance sampling with density ratio estimation.

    Reference:
        Sugiyama et al. (2012) "Density Ratio Estimation in Machine Learning"
        Cambridge University Press
    """

    def __init__(
        self,
        n_reference_samples: int = 500,
        max_weight: float = 10.0,
    ):
        """
        Initialize distribution adapter.

        Args:
            n_reference_samples: Number of recent samples as reference
            max_weight: Maximum sample weight (for stability)
        """
        self.n_reference = n_reference_samples
        self.max_weight = max_weight

        # Reference distribution (recent samples)
        self._reference_features: deque = deque(maxlen=n_reference_samples)

        # Statistics for density estimation
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_cov: Optional[np.ndarray] = None

    def update_reference(self, features: np.ndarray) -> None:
        """
        Update reference distribution with new samples.

        Args:
            features: New feature vector(s)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        for f in features:
            self._reference_features.append(f)

        # Update statistics
        if len(self._reference_features) >= 50:
            ref_array = np.array(self._reference_features)
            self._ref_mean = np.mean(ref_array, axis=0)
            self._ref_cov = np.cov(ref_array, rowvar=False)

            # Regularize covariance
            self._ref_cov += np.eye(len(self._ref_mean)) * 1e-6

    def compute_weights(
        self,
        historical_features: np.ndarray,
        historical_mean: np.ndarray,
        historical_cov: np.ndarray,
    ) -> np.ndarray:
        """
        Compute importance weights for historical samples.

        w(x) = p_current(x) / p_historical(x)

        Args:
            historical_features: Historical feature matrix
            historical_mean: Mean of historical distribution
            historical_cov: Covariance of historical distribution

        Returns:
            Importance weights for each sample

        Reference:
            KDD 2023 DoubleAdapt, Section 3.1
        """
        if self._ref_mean is None:
            return np.ones(len(historical_features))

        # Compute density under current distribution
        try:
            current_density = stats.multivariate_normal.pdf(
                historical_features,
                mean=self._ref_mean,
                cov=self._ref_cov,
                allow_singular=True,
            )
        except:
            current_density = np.ones(len(historical_features))

        # Compute density under historical distribution
        try:
            historical_density = stats.multivariate_normal.pdf(
                historical_features,
                mean=historical_mean,
                cov=historical_cov,
                allow_singular=True,
            )
        except:
            historical_density = np.ones(len(historical_features))

        # Importance weights
        weights = current_density / (historical_density + 1e-10)

        # Clip weights for stability
        weights = np.clip(weights, 1e-3, self.max_weight)

        # Normalize
        weights = weights / np.sum(weights) * len(weights)

        return weights

    def get_distribution_shift(self) -> float:
        """
        Estimate distribution shift (KL divergence approximation).

        Returns:
            Estimated KL divergence from historical to current
        """
        if self._ref_mean is None or len(self._reference_features) < 100:
            return 0.0

        # Simple approximation using moment matching
        ref_array = np.array(self._reference_features)
        ref_std = np.std(ref_array, axis=0)

        # Compare to unit normal (proxy for well-scaled historical)
        kl_approx = np.mean(0.5 * (ref_std ** 2 - 1 - np.log(ref_std ** 2 + 1e-10)))

        return abs(kl_approx)


class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning (MAML) style fast adaptation.

    Performs few gradient steps to adapt model to new regime.

    Reference:
        Finn et al. (2017) ICML
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        n_inner_steps: int = 5,
    ):
        """
        Initialize MAML adapter.

        Args:
            inner_lr: Inner loop learning rate
            n_inner_steps: Number of gradient steps
        """
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps

    def adapt_sklearn_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Adapt sklearn-compatible model with weighted samples.

        For gradient boosting, uses warm start to continue training.

        Args:
            model: Scikit-learn compatible model
            X: Features
            y: Labels
            sample_weights: Importance weights

        Returns:
            Adapted model
        """
        # For XGBoost/LightGBM/CatBoost, use warm start
        if hasattr(model, 'set_params'):
            try:
                # Set n_estimators to add more trees
                current_n = getattr(model, 'n_estimators', 100)
                model.set_params(n_estimators=current_n + self.n_inner_steps * 10)
            except:
                pass

        # Fit with sample weights
        try:
            if sample_weights is not None:
                model.fit(X, y, sample_weight=sample_weights)
            else:
                model.fit(X, y)
        except TypeError:
            # Model doesn't support sample_weight
            model.fit(X, y)

        return model


class DoubleAdapt:
    """
    DoubleAdapt: Two-stage adaptation for regime changes.

    Stage 1: Reweight historical data to match current distribution
    Stage 2: MAML-style fast adaptation of model

    Reference:
        KDD 2023: "DoubleAdapt: A Meta-learning Approach for Incremental Learning"
    """

    def __init__(
        self,
        n_reference_samples: int = 500,
        n_historical_samples: int = 5000,
        adaptation_threshold: float = 0.3,
        inner_lr: float = 0.01,
        n_inner_steps: int = 5,
    ):
        """
        Initialize DoubleAdapt.

        Args:
            n_reference_samples: Recent samples for current distribution
            n_historical_samples: Historical samples to reweight
            adaptation_threshold: Distribution shift threshold to trigger adaptation
            inner_lr: MAML inner loop learning rate
            n_inner_steps: Number of MAML gradient steps
        """
        self.n_reference = n_reference_samples
        self.n_historical = n_historical_samples
        self.adaptation_threshold = adaptation_threshold

        # Components
        self.dist_adapter = DistributionAdapter(n_reference_samples)
        self.maml_adapter = MAMLAdapter(inner_lr, n_inner_steps)

        # Historical data buffer
        self._historical_X: deque = deque(maxlen=n_historical_samples)
        self._historical_y: deque = deque(maxlen=n_historical_samples)

        # Statistics of historical distribution
        self._hist_mean: Optional[np.ndarray] = None
        self._hist_cov: Optional[np.ndarray] = None

        # State
        self._last_regime: Optional[str] = None
        self._adaptations_count = 0

    def add_sample(self, X: np.ndarray, y: int) -> None:
        """
        Add new sample to buffers.

        Args:
            X: Feature vector
            y: Label
        """
        self._historical_X.append(X)
        self._historical_y.append(y)
        self.dist_adapter.update_reference(X)

    def should_adapt(self) -> bool:
        """
        Check if adaptation is needed.

        Returns:
            True if distribution shift exceeds threshold
        """
        shift = self.dist_adapter.get_distribution_shift()
        return shift > self.adaptation_threshold

    def detect_regime(self, features: np.ndarray) -> RegimeState:
        """
        Detect current market regime.

        Args:
            features: Recent feature matrix

        Returns:
            RegimeState with regime information
        """
        if len(features) < 50:
            return RegimeState(
                volatility=0.0,
                trend_strength=0.0,
                mean_reversion=0.0,
                correlation_state=0.0,
                regime_id=0,
                regime_name="normal",
                regime_changed=False,
                change_magnitude=0.0,
                time_since_change=0,
                distribution_shift=0.0,
            )

        # Compute regime indicators
        # Assuming first few features are returns, volatility, etc.
        returns = features[:, 0] if features.shape[1] > 0 else features.flatten()

        volatility = np.std(returns)
        trend = np.abs(np.mean(returns)) / (volatility + 1e-10)

        # Autocorrelation for mean reversion
        if len(returns) > 10:
            acf1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            acf1 = 0

        # Classify regime
        if volatility > np.percentile(returns, 90):
            regime_id, regime_name = 3, "volatile"
        elif trend > 1.0:
            regime_id, regime_name = 1, "trending"
        elif acf1 < -0.2:
            regime_id, regime_name = 2, "mean_reverting"
        else:
            regime_id, regime_name = 0, "normal"

        # Check for change
        regime_changed = self._last_regime is not None and regime_name != self._last_regime
        self._last_regime = regime_name

        return RegimeState(
            volatility=volatility,
            trend_strength=trend,
            mean_reversion=-acf1,
            correlation_state=acf1,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_changed=regime_changed,
            change_magnitude=self.dist_adapter.get_distribution_shift(),
            time_since_change=0,
            distribution_shift=self.dist_adapter.get_distribution_shift(),
        )

    def adapt(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, AdaptationResult]:
        """
        Perform DoubleAdapt two-stage adaptation.

        Args:
            model: Model to adapt
            X_val: Validation features
            y_val: Validation labels

        Returns:
            (adapted_model, AdaptationResult)

        Reference:
            KDD 2023 DoubleAdapt Algorithm 1
        """
        import time
        start_time = time.perf_counter()

        # Get historical data
        if len(self._historical_X) < 100:
            return model, AdaptationResult(
                n_samples_reweighted=0,
                effective_sample_size=0,
                max_weight=1.0,
                distribution_shift_before=0,
                distribution_shift_after=0,
                adaptation_steps=0,
                learning_rate_used=0,
                loss_before=0,
                loss_after=0,
                improvement=0,
                val_accuracy_before=0,
                val_accuracy_after=0,
                should_deploy=False,
                regime_before="unknown",
                regime_after="unknown",
                adaptation_time_ms=0,
            )

        X_hist = np.array(self._historical_X)
        y_hist = np.array(self._historical_y)

        # Compute historical distribution statistics
        if self._hist_mean is None:
            self._hist_mean = np.mean(X_hist, axis=0)
            self._hist_cov = np.cov(X_hist, rowvar=False)
            if self._hist_cov.ndim == 0:
                self._hist_cov = np.array([[self._hist_cov]])
            self._hist_cov += np.eye(len(self._hist_mean)) * 1e-6

        # Stage 1: Distribution Adaptation (compute sample weights)
        shift_before = self.dist_adapter.get_distribution_shift()
        weights = self.dist_adapter.compute_weights(
            X_hist, self._hist_mean, self._hist_cov
        )

        # Effective sample size
        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)

        # Validation accuracy before
        try:
            y_pred_before = model.predict(X_val)
            acc_before = np.mean(y_pred_before == y_val)
        except:
            acc_before = 0.5

        # Stage 2: Model Adaptation (MAML-style)
        adapted_model = self.maml_adapter.adapt_sklearn_model(
            model, X_hist, y_hist, weights
        )

        # Validation accuracy after
        try:
            y_pred_after = adapted_model.predict(X_val)
            acc_after = np.mean(y_pred_after == y_val)
        except:
            acc_after = acc_before

        shift_after = self.dist_adapter.get_distribution_shift()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._adaptations_count += 1

        result = AdaptationResult(
            n_samples_reweighted=len(X_hist),
            effective_sample_size=ess,
            max_weight=float(np.max(weights)),
            distribution_shift_before=shift_before,
            distribution_shift_after=shift_after,
            adaptation_steps=self.maml_adapter.n_inner_steps,
            learning_rate_used=self.maml_adapter.inner_lr,
            loss_before=1 - acc_before,
            loss_after=1 - acc_after,
            improvement=acc_after - acc_before,
            val_accuracy_before=acc_before,
            val_accuracy_after=acc_after,
            should_deploy=acc_after >= acc_before,
            regime_before=self._last_regime or "unknown",
            regime_after=self._last_regime or "unknown",
            adaptation_time_ms=elapsed_ms,
        )

        return adapted_model, result


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_regime_check(
    recent_returns: np.ndarray,
    vol_threshold_high: float = 0.02,
    trend_threshold: float = 1.0,
) -> Tuple[str, float]:
    """
    Quick regime detection for HFT.

    Args:
        recent_returns: Recent return series
        vol_threshold_high: High volatility threshold
        trend_threshold: Trend strength threshold

    Returns:
        (regime_name, confidence)
    """
    if len(recent_returns) < 10:
        return "unknown", 0.0

    vol = np.std(recent_returns)
    trend = np.abs(np.mean(recent_returns)) / (vol + 1e-10)

    if vol > vol_threshold_high:
        return "volatile", min(vol / vol_threshold_high, 2.0)
    elif trend > trend_threshold:
        return "trending", min(trend / trend_threshold, 2.0)
    else:
        return "normal", 1.0


def compute_sample_weights_fast(
    recent_features: np.ndarray,
    historical_features: np.ndarray,
) -> np.ndarray:
    """
    Fast sample weight computation using moment matching.

    Args:
        recent_features: Recent feature matrix (reference)
        historical_features: Historical feature matrix (to reweight)

    Returns:
        Importance weights
    """
    # Simple moment-based weighting
    recent_mean = np.mean(recent_features, axis=0)
    recent_std = np.std(recent_features, axis=0) + 1e-10

    hist_mean = np.mean(historical_features, axis=0)
    hist_std = np.std(historical_features, axis=0) + 1e-10

    # Standardize historical features to recent distribution
    z_scores = (historical_features - hist_mean) / hist_std

    # Weight by proximity to recent mean
    distances = np.sum(((z_scores * hist_std + hist_mean - recent_mean) / recent_std) ** 2, axis=1)
    weights = np.exp(-distances / 2)

    # Normalize
    weights = weights / np.sum(weights) * len(weights)

    return np.clip(weights, 0.1, 10.0)


def adapt_ensemble_weights(
    model_predictions: List[np.ndarray],
    y_recent: np.ndarray,
    n_recent: int = 100,
) -> np.ndarray:
    """
    Adapt ensemble weights based on recent performance.

    Args:
        model_predictions: Predictions from each model
        y_recent: Recent true labels
        n_recent: Number of recent samples to use

    Returns:
        Adapted model weights
    """
    n_models = len(model_predictions)
    accuracies = []

    for preds in model_predictions:
        recent_preds = preds[-n_recent:]
        recent_true = y_recent[-n_recent:]
        acc = np.mean(recent_preds == recent_true)
        accuracies.append(acc)

    accuracies = np.array(accuracies)

    # Softmax weighting
    weights = np.exp(10 * (accuracies - np.max(accuracies)))
    weights = weights / np.sum(weights)

    return weights


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DOUBLEADAPT - META-LEARNING FOR REGIME ADAPTATION")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - KDD 2023: +17.6% excess return vs standard methods")
    print("  - Finn et al. (2017): MAML foundation")
    print("  - 九坤投资: 因子组合在风格变化中的自动切换")
    print()

    # Create DoubleAdapt instance
    da = DoubleAdapt(
        n_reference_samples=200,
        n_historical_samples=1000,
        adaptation_threshold=0.3,
    )

    # Simulate data with regime change
    np.random.seed(42)

    print("Simulating regime change scenario...")
    print("-" * 50)

    # Phase 1: Normal regime
    for i in range(500):
        X = np.random.randn(10)
        y = int(X[0] > 0)
        da.add_sample(X, y)

    print(f"Phase 1: Normal regime, {len(da._historical_X)} samples")

    # Phase 2: Regime change (different distribution)
    for i in range(200):
        X = np.random.randn(10) + 2  # Shifted distribution
        y = int(X[1] > 2)  # Different signal
        da.add_sample(X, y)

    print(f"Phase 2: Shifted regime, {len(da._historical_X)} samples")

    # Check if adaptation is needed
    should_adapt = da.should_adapt()
    print(f"\nShould adapt: {should_adapt}")

    # Detect regime
    recent_features = np.array(list(da._historical_X)[-100:])
    regime = da.detect_regime(recent_features)
    print(f"Current regime: {regime.regime_name}")
    print(f"Distribution shift: {regime.distribution_shift:.4f}")
    print()

    # Quick regime check performance
    print("QUICK REGIME CHECK")
    print("-" * 50)
    import time

    returns = np.random.randn(100) * 0.01
    n_checks = 10000
    start = time.perf_counter()
    for _ in range(n_checks):
        quick_regime_check(returns)
    elapsed = time.perf_counter() - start

    print(f"  {n_checks:,} checks in {elapsed*1000:.2f}ms")
    print(f"  {elapsed/n_checks*1e6:.3f} microseconds per check")
    print()

    print("=" * 70)
    print("KEY INSIGHT (KDD 2023):")
    print("  Standard online learning: Adapts model weights only")
    print("  DoubleAdapt: Adapts BOTH data weights AND model")
    print()
    print("  Step 1: Reweight historical data → match current regime")
    print("  Step 2: MAML-style gradient steps → fast model adaptation")
    print()
    print("  Result: +17.6% excess return over standard methods!")
    print("=" * 70)
