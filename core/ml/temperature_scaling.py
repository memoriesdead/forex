"""
Temperature Scaling for Probability Calibration

Modern neural networks and ensemble models are often overconfident.
Temperature scaling is the simplest and most effective post-hoc
calibration method that learns a single scalar to soften predictions.

=============================================================================
CITATIONS (ACADEMIC - PEER REVIEWED)
=============================================================================

[1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks."
    International Conference on Machine Learning (ICML) 2017.
    URL: https://arxiv.org/abs/1706.04599
    PDF: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
    Key finding: Deep networks are miscalibrated; temperature scaling fixes it

[2] Platt, J. (1999).
    "Probabilistic Outputs for Support Vector Machines and Comparisons to
     Regularized Likelihood Methods."
    Advances in Large Margin Classifiers.
    URL: https://www.researchgate.net/publication/2594015
    Key finding: Platt scaling for SVMs (precursor to temperature scaling)

[3] Niculescu-Mizil, A., & Caruana, R. (2005).
    "Predicting Good Probabilities with Supervised Learning."
    International Conference on Machine Learning (ICML) 2005.
    URL: https://www.cs.cornell.edu/~caruana/niculescu.sziu.icml05.pdf
    Key finding: Comparison of calibration methods

[4] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning."
    AAAI Conference on Artificial Intelligence.
    URL: https://ojs.aaai.org/index.php/AAAI/article/view/9602
    Key finding: ECE metric for measuring calibration

=============================================================================

Formula:
    calibrated_prob = softmax(logits / T)

    Where T > 1 softens predictions (reduces overconfidence)

ECE (Expected Calibration Error):
    ECE = Σ (|B_m|/n) |acc(B_m) - conf(B_m)|

    Good calibration: ECE < 0.05 (5%)

Author: Claude Code
Date: 2026-01-25
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from dataclasses import dataclass
import warnings


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float  # Brier Score
    temperature: float  # Optimal temperature
    is_well_calibrated: bool  # ECE < 0.05

    def __repr__(self) -> str:
        return f"""
╔════════════════════════════════════════════════════════════════╗
║  CALIBRATION METRICS                                           ║
╠════════════════════════════════════════════════════════════════╣
║  Expected Calibration Error (ECE):   {self.ece:.4f}              ║
║  Maximum Calibration Error (MCE):    {self.mce:.4f}              ║
║  Brier Score:                        {self.brier:.4f}              ║
║  Temperature:                        {self.temperature:.4f}              ║
║  Well Calibrated (ECE < 0.05):       {"YES" if self.is_well_calibrated else "NO ":3}                 ║
╚════════════════════════════════════════════════════════════════╝
"""


class TemperatureScaler:
    """
    Temperature scaling for probability calibration.

    Citation [1]: Guo et al. (2017) ICML
        "Temperature scaling, a single-parameter variant of Platt scaling,
         is the most effective at calibrating modern neural networks."

    How it works:
        1. Train model normally
        2. Hold out validation set
        3. Learn optimal T by minimizing NLL on validation
        4. At inference: calibrated_p = softmax(logits / T)

    Why T > 1?
        - T > 1 makes distribution more uniform (less confident)
        - T < 1 makes distribution more peaked (more confident)
        - Overconfident models need T > 1 to soften predictions

    Example:
        >>> scaler = TemperatureScaler()
        >>> scaler.fit(logits_val, labels_val)
        >>> calibrated = scaler.calibrate(logits_test)
    """

    def __init__(self, initial_temperature: float = 1.5):
        """
        Initialize temperature scaler.

        Args:
            initial_temperature: Starting point for optimization
        """
        self.temperature = initial_temperature
        self._fitted = False
        self._validation_ece = None

    def _to_logits(self, probs: np.ndarray) -> np.ndarray:
        """
        Convert probabilities to logits.

        logit(p) = log(p / (1-p))

        Args:
            probs: Probabilities in [0, 1]

        Returns:
            Logits in (-inf, inf)
        """
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return np.log(probs / (1 - probs))

    def _to_probs(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert logits to probabilities using sigmoid.

        sigmoid(x) = 1 / (1 + exp(-x))

        Args:
            logits: Logits in (-inf, inf)

        Returns:
            Probabilities in [0, 1]
        """
        return 1 / (1 + np.exp(-logits))

    def fit(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        is_probs: bool = True
    ) -> 'TemperatureScaler':
        """
        Learn optimal temperature on validation set.

        Citation [1]: "We find that temperature scaling... learns a single
        scalar parameter T... by minimizing NLL on the validation set."

        Args:
            predictions: Model predictions (probabilities or logits)
            labels: True binary labels (0 or 1)
            is_probs: If True, predictions are probabilities; else logits

        Returns:
            self for chaining
        """
        if len(predictions) < 50:
            warnings.warn("Small validation set may give unreliable calibration")

        # Convert to logits if needed
        if is_probs:
            logits = self._to_logits(predictions)
        else:
            logits = predictions

        labels = np.asarray(labels)

        def negative_log_likelihood(T):
            """
            NLL loss for binary classification.

            NLL = -Σ[y*log(σ(z/T)) + (1-y)*log(1-σ(z/T))]
            """
            T_val = T[0]
            if T_val <= 0:
                return 1e10

            scaled_logits = logits / T_val
            probs = self._to_probs(scaled_logits)

            # Binary cross entropy
            eps = 1e-10
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(
                labels * np.log(probs) +
                (1 - labels) * np.log(1 - probs)
            )
            return nll

        # Optimize temperature
        result = minimize(
            negative_log_likelihood,
            x0=[self.temperature],
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)]  # T in [0.01, 10]
        )

        self.temperature = result.x[0]
        self._fitted = True

        # Calculate validation ECE
        calibrated = self.calibrate(predictions, is_probs=is_probs)
        self._validation_ece = expected_calibration_error(calibrated, labels)

        return self

    def calibrate(
        self,
        predictions: np.ndarray,
        is_probs: bool = True
    ) -> np.ndarray:
        """
        Calibrate predictions using learned temperature.

        calibrated = sigmoid(logits / T)

        Args:
            predictions: Model predictions (probabilities or logits)
            is_probs: If True, predictions are probabilities; else logits

        Returns:
            Calibrated probabilities
        """
        if is_probs:
            logits = self._to_logits(predictions)
        else:
            logits = predictions

        scaled_logits = logits / self.temperature
        return self._to_probs(scaled_logits)

    def get_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        is_probs: bool = True
    ) -> CalibrationMetrics:
        """
        Get full calibration metrics.

        Args:
            predictions: Model predictions
            labels: True labels
            is_probs: If True, predictions are probabilities

        Returns:
            CalibrationMetrics dataclass
        """
        calibrated = self.calibrate(predictions, is_probs=is_probs)

        ece = expected_calibration_error(calibrated, labels)
        mce = maximum_calibration_error(calibrated, labels)
        brier = brier_score(calibrated, labels)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier=brier,
            temperature=self.temperature,
            is_well_calibrated=(ece < 0.05)
        )


def expected_calibration_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).

    Citation [4]: Naeini et al. (2015) AAAI
        "ECE measures the difference between accuracy and confidence
         across different confidence bins."

    Formula:
        ECE = Σ_m (|B_m|/n) × |acc(B_m) - conf(B_m)|

    Where:
        B_m = samples in bin m
        acc(B_m) = accuracy in bin m
        conf(B_m) = average confidence in bin m

    Good calibration: ECE < 0.05 (5%)
    Perfect calibration: ECE = 0

    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins for calibration

    Returns:
        ECE value in [0, 1]
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    n = len(predictions)
    if n == 0:
        return 0.0

    ece = 0.0

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        # Find predictions in this bin
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            # Average confidence in bin
            avg_confidence = predictions[in_bin].mean()

            # Accuracy in bin
            avg_accuracy = labels[in_bin].mean()

            # Weighted absolute difference
            ece += (bin_size / n) * np.abs(avg_accuracy - avg_confidence)

    return ece


def maximum_calibration_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Maximum Calibration Error (MCE).

    MCE = max_m |acc(B_m) - conf(B_m)|

    Measures the worst-case calibration error across all bins.

    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins

    Returns:
        MCE value in [0, 1]
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    mce = 0.0

    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins

        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = predictions[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            mce = max(mce, np.abs(avg_accuracy - avg_confidence))

    return mce


def brier_score(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Brier Score for probabilistic predictions.

    Brier = (1/n) × Σ(p_i - y_i)²

    Properties:
        - Range: [0, 1]
        - Perfect: 0
        - Random guessing (p=0.5): 0.25
        - Decomposes into: reliability + resolution - uncertainty

    Args:
        predictions: Predicted probabilities
        labels: True binary labels

    Returns:
        Brier score in [0, 1]
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    return np.mean((predictions - labels) ** 2)


def reliability_diagram_data(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, List[float]]:
    """
    Generate data for reliability diagram.

    A reliability diagram plots:
        - X-axis: Mean predicted probability (confidence)
        - Y-axis: Fraction of positives (accuracy)

    Perfect calibration = diagonal line

    Citation [3]: Niculescu-Mizil & Caruana (2005)
        "Reliability diagrams are useful for visualizing
         how well calibrated predicted probabilities are."

    Args:
        predictions: Predicted probabilities
        labels: True binary labels
        n_bins: Number of bins

    Returns:
        Dictionary with 'mean_predicted', 'fraction_positive', 'bin_counts'
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    mean_predicted = []
    fraction_positive = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (predictions >= bin_boundaries[i]) & \
                 (predictions < bin_boundaries[i + 1])

        if np.sum(in_bin) > 0:
            mean_predicted.append(predictions[in_bin].mean())
            fraction_positive.append(labels[in_bin].mean())
            bin_counts.append(np.sum(in_bin))
        else:
            mean_predicted.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            fraction_positive.append(np.nan)
            bin_counts.append(0)

    return {
        'mean_predicted': mean_predicted,
        'fraction_positive': fraction_positive,
        'bin_counts': bin_counts,
        'bin_boundaries': bin_boundaries.tolist()
    }


class IsotonicCalibrator:
    """
    Isotonic regression calibration (alternative to temperature scaling).

    Fits a monotonic function to map predictions to calibrated probabilities.
    More flexible than temperature scaling but can overfit on small datasets.

    Citation [3]: Niculescu-Mizil & Caruana (2005)
        "Isotonic regression is consistently the best method for
         calibrating boosted trees and random forests."
    """

    def __init__(self):
        self._calibrator = None
        self._fitted = False

    def fit(self, predictions: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic regression calibrator."""
        from sklearn.isotonic import IsotonicRegression

        self._calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self._calibrator.fit(predictions, labels)
        self._fitted = True

        return self

    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Calibrate predictions using isotonic regression."""
        if not self._fitted:
            raise ValueError("Must call fit() before calibrate()")

        return self._calibrator.transform(predictions)


# Convenience functions
def calibrate_ensemble_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    test_predictions: Optional[np.ndarray] = None
) -> Tuple[TemperatureScaler, np.ndarray]:
    """
    Convenience function to calibrate ensemble predictions.

    Args:
        predictions: Validation predictions for fitting
        labels: Validation labels
        test_predictions: Optional test predictions to calibrate

    Returns:
        Tuple of (fitted scaler, calibrated predictions)
    """
    scaler = TemperatureScaler()
    scaler.fit(predictions, labels)

    if test_predictions is not None:
        calibrated = scaler.calibrate(test_predictions)
    else:
        calibrated = scaler.calibrate(predictions)

    return scaler, calibrated


def check_calibration(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.05
) -> Dict:
    """
    Quick calibration check for certainty validation.

    Used by 99.999% certainty system to verify predictions are trustworthy.

    Args:
        predictions: Model predictions
        labels: True labels
        threshold: ECE threshold for "well calibrated"

    Returns:
        Dictionary with calibration status
    """
    ece = expected_calibration_error(predictions, labels)
    mce = maximum_calibration_error(predictions, labels)
    brier = brier_score(predictions, labels)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'is_calibrated': ece < threshold,
        'certainty_check_passed': ece < threshold,
        'recommendation': 'CALIBRATED' if ece < threshold else 'NEEDS_TEMPERATURE_SCALING'
    }


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Simulate overconfident model predictions
    n = 1000
    true_probs = np.random.rand(n)
    labels = (np.random.rand(n) < true_probs).astype(int)

    # Overconfident predictions (pushed toward 0 or 1)
    overconfident = np.where(true_probs > 0.5,
                             0.7 + 0.25 * true_probs,
                             0.3 * true_probs)

    print("=" * 60)
    print("TEMPERATURE SCALING DEMO")
    print("=" * 60)

    # Before calibration
    ece_before = expected_calibration_error(overconfident, labels)
    print(f"\nBefore calibration:")
    print(f"  ECE = {ece_before:.4f}")
    print(f"  Well calibrated? {'YES' if ece_before < 0.05 else 'NO'}")

    # Fit temperature scaler
    scaler = TemperatureScaler()
    scaler.fit(overconfident, labels)

    print(f"\nLearned temperature: T = {scaler.temperature:.4f}")

    # After calibration
    calibrated = scaler.calibrate(overconfident)
    ece_after = expected_calibration_error(calibrated, labels)

    print(f"\nAfter calibration:")
    print(f"  ECE = {ece_after:.4f}")
    print(f"  Well calibrated? {'YES' if ece_after < 0.05 else 'NO'}")

    # Full metrics
    metrics = scaler.get_metrics(overconfident, labels)
    print(metrics)
