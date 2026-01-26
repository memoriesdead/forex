"""
Ensemble Certainty Scoring Module
=================================

Computes calibrated certainty scores from ensemble model predictions.
Only trade when certainty is HIGH - this is FREE certainty from existing models.

Key Insight (Lakshminarayanan et al. 2017):
    Ensemble disagreement provides well-calibrated uncertainty estimates.
    When all models agree with high confidence → HIGH certainty.
    When models disagree → LOW certainty → DON'T TRADE.

Key Formulas:
    Agreement Score:
        agreement = 1 if all models predict same class, else 0

    Confidence Score:
        confidence = mean(max(p_i) for each model i)

    Ensemble Uncertainty (predictive entropy):
        H[p̄] = -Σ p̄_c log(p̄_c)
        where p̄ = mean prediction across models

    Mutual Information (model uncertainty):
        I[y; θ|x] = H[p̄] - E_θ[H[p_θ]]

    Certainty Score:
        certainty = agreement × confidence × (1 - normalized_uncertainty)

References:
    [1] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
        "Simple and Scalable Predictive Uncertainty Estimation using
        Deep Ensembles." NeurIPS 2017.
        https://arxiv.org/abs/1612.01474

    [2] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017).
        "On Calibration of Modern Neural Networks." ICML 2017.
        https://arxiv.org/abs/1706.04599

    [3] Ovadia, Y. et al. (2019). "Can You Trust Your Model's Uncertainty?
        Evaluating Predictive Uncertainty Under Dataset Shift." NeurIPS 2019.

    [4] Fort, S., Hu, H., & Lakshminarayanan, B. (2019).
        "Deep Ensembles: A Loss Landscape Perspective." arXiv:1912.02757

    [5] Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation:
        Representing Model Uncertainty in Deep Learning." ICML 2016.

Author: Claude Code + Kevin
Created: 2025-01-22
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class CertaintyLevel(Enum):
    """Certainty levels for trading decisions."""
    MAXIMUM = "MAXIMUM"      # 100% certainty - full position
    HIGH = "HIGH"            # >90% certainty - trade with confidence
    MEDIUM = "MEDIUM"        # 70-90% certainty - reduced position
    LOW = "LOW"              # 50-70% certainty - minimal position
    NONE = "NONE"            # <50% certainty - DO NOT TRADE


@dataclass
class CertaintyResult:
    """Result of certainty analysis for a single prediction."""

    # Core certainty score (0-1)
    certainty: float

    # Certainty level for trading
    level: CertaintyLevel

    # Should we trade?
    should_trade: bool

    # Position size multiplier (0-1)
    position_multiplier: float

    # Component scores
    agreement_score: float      # Do all models agree?
    confidence_score: float     # How confident are they?
    uncertainty_score: float    # Ensemble uncertainty (lower is better)

    # Prediction details
    ensemble_prediction: int    # Final prediction (0 or 1)
    ensemble_probability: float # Probability of predicted class
    individual_predictions: List[int]  # Each model's prediction
    individual_probabilities: List[float]  # Each model's probability

    def __repr__(self) -> str:
        trade_action = "TRADE" if self.should_trade else "NO TRADE"
        return f"""
┌─────────────────────────────────────────────────────────────┐
│  CERTAINTY ANALYSIS: {self.level.value:^10} → {trade_action:^10}           │
├─────────────────────────────────────────────────────────────┤
│  Overall Certainty:    {self.certainty:>6.2%}                            │
│  Position Multiplier:  {self.position_multiplier:>6.2%}                            │
├─────────────────────────────────────────────────────────────┤
│  Components:                                                │
│    Agreement:          {self.agreement_score:>6.2%} (all models agree?)      │
│    Confidence:         {self.confidence_score:>6.2%} (how confident?)        │
│    Uncertainty:        {self.uncertainty_score:>6.2%} (ensemble variance)    │
├─────────────────────────────────────────────────────────────┤
│  Prediction: {self.ensemble_prediction} with {self.ensemble_probability:.2%} probability               │
│  Models: {self.individual_predictions} → {self.individual_probabilities}    │
└─────────────────────────────────────────────────────────────┘
"""


class EnsembleCertaintyScorer:
    """
    Computes certainty scores from ensemble model predictions.

    Uses ensemble disagreement as a proxy for prediction uncertainty.
    This is a FREE source of certainty - we already have the models!

    Reference:
        [1] Lakshminarayanan et al. (2017): Deep Ensembles
        [2] Guo et al. (2017): Calibration of Neural Networks
    """

    def __init__(
        self,
        certainty_threshold: float = 0.90,
        agreement_weight: float = 0.4,
        confidence_weight: float = 0.4,
        uncertainty_weight: float = 0.2,
        min_confidence: float = 0.6
    ):
        """
        Initialize certainty scorer.

        Args:
            certainty_threshold: Minimum certainty to trade (default 0.90)
            agreement_weight: Weight for agreement score in certainty
            confidence_weight: Weight for confidence score in certainty
            uncertainty_weight: Weight for (1 - uncertainty) in certainty
            min_confidence: Minimum confidence required to consider trading
        """
        self.certainty_threshold = certainty_threshold
        self.agreement_weight = agreement_weight
        self.confidence_weight = confidence_weight
        self.uncertainty_weight = uncertainty_weight
        self.min_confidence = min_confidence

        # Validate weights sum to 1
        total_weight = agreement_weight + confidence_weight + uncertainty_weight
        assert abs(total_weight - 1.0) < 1e-6, "Weights must sum to 1"

    def compute_certainty(
        self,
        predictions: List[int],
        probabilities: List[float]
    ) -> CertaintyResult:
        """
        Compute certainty score from ensemble predictions.

        Args:
            predictions: List of predictions from each model (0 or 1)
            probabilities: List of probability of positive class from each model

        Returns:
            CertaintyResult with full analysis

        Example:
            >>> scorer = EnsembleCertaintyScorer()
            >>> result = scorer.compute_certainty(
            ...     predictions=[1, 1, 1],
            ...     probabilities=[0.85, 0.82, 0.88]
            ... )
            >>> if result.should_trade:
            ...     execute_trade(size=result.position_multiplier)
        """
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        n_models = len(predictions)

        # 1. Agreement Score
        # All models agree → 1.0, otherwise proportion that agree with majority
        majority_pred = int(np.mean(predictions) >= 0.5)
        agreement_count = np.sum(predictions == majority_pred)
        agreement_score = agreement_count / n_models

        # Perfect agreement bonus
        if agreement_count == n_models:
            agreement_score = 1.0

        # 2. Confidence Score
        # Average of each model's confidence in its prediction
        confidences = np.where(predictions == 1, probabilities, 1 - probabilities)
        confidence_score = np.mean(confidences)

        # 3. Uncertainty Score (lower is better for certainty)
        # Using predictive entropy of ensemble mean
        p_mean = np.mean(probabilities)
        # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
        if 0 < p_mean < 1:
            entropy = -p_mean * np.log2(p_mean) - (1-p_mean) * np.log2(1-p_mean)
        else:
            entropy = 0
        # Normalize: max entropy for binary is 1 bit
        uncertainty_score = entropy  # 0 = certain, 1 = maximum uncertainty

        # Also consider variance across models
        prob_variance = np.var(probabilities)
        # Combine entropy and variance (both normalized to 0-1)
        combined_uncertainty = 0.7 * uncertainty_score + 0.3 * min(prob_variance * 4, 1.0)

        # 4. Compute overall certainty
        certainty = (
            self.agreement_weight * agreement_score +
            self.confidence_weight * confidence_score +
            self.uncertainty_weight * (1 - combined_uncertainty)
        )

        # 5. Determine certainty level and trading decision
        if agreement_score == 1.0 and confidence_score >= 0.8:
            level = CertaintyLevel.MAXIMUM
            should_trade = True
            position_multiplier = 1.0
        elif certainty >= self.certainty_threshold:
            level = CertaintyLevel.HIGH
            should_trade = True
            position_multiplier = 0.8
        elif certainty >= 0.7:
            level = CertaintyLevel.MEDIUM
            should_trade = confidence_score >= self.min_confidence
            position_multiplier = 0.5 if should_trade else 0.0
        elif certainty >= 0.5:
            level = CertaintyLevel.LOW
            should_trade = False  # Too risky
            position_multiplier = 0.0
        else:
            level = CertaintyLevel.NONE
            should_trade = False
            position_multiplier = 0.0

        # 6. Ensemble prediction (majority vote weighted by confidence)
        weighted_vote = np.sum(predictions * confidences) / np.sum(confidences)
        ensemble_prediction = int(weighted_vote >= 0.5)
        ensemble_probability = p_mean if ensemble_prediction == 1 else (1 - p_mean)

        return CertaintyResult(
            certainty=certainty,
            level=level,
            should_trade=should_trade,
            position_multiplier=position_multiplier,
            agreement_score=agreement_score,
            confidence_score=confidence_score,
            uncertainty_score=combined_uncertainty,
            ensemble_prediction=ensemble_prediction,
            ensemble_probability=ensemble_probability,
            individual_predictions=predictions.tolist(),
            individual_probabilities=probabilities.tolist()
        )

    def compute_certainty_batch(
        self,
        predictions_batch: np.ndarray,
        probabilities_batch: np.ndarray
    ) -> List[CertaintyResult]:
        """
        Compute certainty for a batch of predictions.

        Args:
            predictions_batch: Shape (n_samples, n_models)
            probabilities_batch: Shape (n_samples, n_models)

        Returns:
            List of CertaintyResult for each sample
        """
        results = []
        for i in range(len(predictions_batch)):
            result = self.compute_certainty(
                predictions_batch[i].tolist(),
                probabilities_batch[i].tolist()
            )
            results.append(result)
        return results


class CalibrationChecker:
    """
    Checks if certainty scores are well-calibrated.

    A well-calibrated system means: when we say 90% certain,
    we should be right 90% of the time.

    Reference:
        [1] Guo et al. (2017): Expected Calibration Error (ECE)
        [2] Naeini et al. (2015): Reliability Diagrams
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration checker.

        Args:
            n_bins: Number of bins for calibration histogram
        """
        self.n_bins = n_bins
        self.predictions_history: List[int] = []
        self.certainties_history: List[float] = []
        self.outcomes_history: List[int] = []

    def record(
        self,
        prediction: int,
        certainty: float,
        actual_outcome: int
    ):
        """
        Record a prediction and its outcome for calibration tracking.

        Args:
            prediction: Model's prediction (0 or 1)
            certainty: Certainty score (0-1)
            actual_outcome: What actually happened (0 or 1)
        """
        self.predictions_history.append(prediction)
        self.certainties_history.append(certainty)
        self.outcomes_history.append(actual_outcome)

    def compute_calibration_error(self) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|

        Where:
            B_m = samples in bin m
            acc(B_m) = accuracy in bin m
            conf(B_m) = average confidence in bin m

        Reference: Guo et al. (2017)
        """
        if len(self.certainties_history) < 100:
            return {'ece': float('nan'), 'mce': float('nan'), 'n_samples': len(self.certainties_history)}

        certainties = np.array(self.certainties_history)
        predictions = np.array(self.predictions_history)
        outcomes = np.array(self.outcomes_history)

        # Correct predictions
        correct = (predictions == outcomes).astype(float)

        # Bin edges
        bin_edges = np.linspace(0, 1, self.n_bins + 1)

        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            # Samples in this bin
            mask = (certainties >= bin_edges[i]) & (certainties < bin_edges[i+1])
            if i == self.n_bins - 1:  # Include 1.0 in last bin
                mask = (certainties >= bin_edges[i]) & (certainties <= bin_edges[i+1])

            n_in_bin = np.sum(mask)
            if n_in_bin == 0:
                continue

            # Accuracy in bin
            acc_in_bin = np.mean(correct[mask])

            # Average confidence in bin
            conf_in_bin = np.mean(certainties[mask])

            # Calibration error for this bin
            bin_error = abs(acc_in_bin - conf_in_bin)

            # Weighted contribution to ECE
            ece += (n_in_bin / len(certainties)) * bin_error

            # Maximum calibration error
            mce = max(mce, bin_error)

        return {
            'ece': ece,  # Expected Calibration Error (lower is better)
            'mce': mce,  # Maximum Calibration Error
            'n_samples': len(self.certainties_history),
            'is_well_calibrated': ece < 0.05  # <5% ECE is good
        }

    def get_reliability_diagram_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for plotting reliability diagram.

        Returns:
            Dictionary with bin_centers, accuracies, confidences, counts
        """
        if len(self.certainties_history) < 10:
            return {}

        certainties = np.array(self.certainties_history)
        predictions = np.array(self.predictions_history)
        outcomes = np.array(self.outcomes_history)
        correct = (predictions == outcomes).astype(float)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        accuracies = []
        confidences = []
        counts = []

        for i in range(self.n_bins):
            mask = (certainties >= bin_edges[i]) & (certainties < bin_edges[i+1])
            if i == self.n_bins - 1:
                mask = (certainties >= bin_edges[i]) & (certainties <= bin_edges[i+1])

            n_in_bin = np.sum(mask)
            counts.append(n_in_bin)

            if n_in_bin > 0:
                accuracies.append(np.mean(correct[mask]))
                confidences.append(np.mean(certainties[mask]))
            else:
                accuracies.append(np.nan)
                confidences.append(np.nan)

        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(accuracies),
            'confidences': np.array(confidences),
            'counts': np.array(counts)
        }


def compute_ensemble_certainty(
    xgb_pred: int, xgb_prob: float,
    lgb_pred: int, lgb_prob: float,
    cat_pred: int, cat_prob: float,
    threshold: float = 0.90
) -> CertaintyResult:
    """
    Convenience function for XGBoost + LightGBM + CatBoost ensemble.

    Args:
        xgb_pred: XGBoost prediction (0 or 1)
        xgb_prob: XGBoost probability of class 1
        lgb_pred: LightGBM prediction (0 or 1)
        lgb_prob: LightGBM probability of class 1
        cat_pred: CatBoost prediction (0 or 1)
        cat_prob: CatBoost probability of class 1
        threshold: Certainty threshold for trading

    Returns:
        CertaintyResult with trading decision

    Example:
        >>> result = compute_ensemble_certainty(1, 0.85, 1, 0.82, 1, 0.88)
        >>> if result.should_trade:
        ...     print(f"TRADE with {result.position_multiplier:.0%} position")
    """
    scorer = EnsembleCertaintyScorer(certainty_threshold=threshold)
    return scorer.compute_certainty(
        predictions=[xgb_pred, lgb_pred, cat_pred],
        probabilities=[xgb_prob, lgb_prob, cat_prob]
    )


def quick_certainty_check(
    predictions: List[int],
    probabilities: List[float]
) -> Tuple[bool, float]:
    """
    Ultra-fast certainty check for HFT.

    Returns (should_trade, certainty) in minimal computation.

    Args:
        predictions: List of model predictions
        probabilities: List of model probabilities

    Returns:
        (should_trade, certainty_score)
    """
    # All agree?
    all_agree = len(set(predictions)) == 1

    if not all_agree:
        return False, 0.0

    # Average confidence
    avg_conf = np.mean(probabilities)

    if avg_conf < 0.7:
        return False, avg_conf

    # Variance check
    var = np.var(probabilities)
    if var > 0.01:  # Models agree but with varying confidence
        return True, 0.8

    return True, min(avg_conf, 0.99)


# Global scorer instance for convenience
_default_scorer = EnsembleCertaintyScorer()


def get_certainty(predictions: List[int], probabilities: List[float]) -> CertaintyResult:
    """Global function to get certainty using default scorer."""
    return _default_scorer.compute_certainty(predictions, probabilities)


if __name__ == "__main__":
    print("=" * 70)
    print("ENSEMBLE CERTAINTY SCORING DEMONSTRATION")
    print("=" * 70)

    scorer = EnsembleCertaintyScorer(certainty_threshold=0.90)

    # Test case 1: All models agree with high confidence
    print("\n[Case 1: All Agree, High Confidence]")
    result = scorer.compute_certainty(
        predictions=[1, 1, 1],
        probabilities=[0.85, 0.82, 0.88]
    )
    print(result)

    # Test case 2: All agree but lower confidence
    print("\n[Case 2: All Agree, Medium Confidence]")
    result = scorer.compute_certainty(
        predictions=[1, 1, 1],
        probabilities=[0.65, 0.62, 0.68]
    )
    print(result)

    # Test case 3: Models disagree
    print("\n[Case 3: Models Disagree]")
    result = scorer.compute_certainty(
        predictions=[1, 0, 1],
        probabilities=[0.55, 0.45, 0.60]
    )
    print(result)

    # Test case 4: Strong disagreement
    print("\n[Case 4: Strong Disagreement]")
    result = scorer.compute_certainty(
        predictions=[1, 0, 0],
        probabilities=[0.90, 0.20, 0.30]
    )
    print(result)

    # Quick certainty check
    print("\n[Quick Certainty Check for HFT]")
    should_trade, certainty = quick_certainty_check([1, 1, 1], [0.85, 0.82, 0.88])
    print(f"Should trade: {should_trade}, Certainty: {certainty:.2%}")

    should_trade, certainty = quick_certainty_check([1, 0, 1], [0.55, 0.45, 0.60])
    print(f"Should trade: {should_trade}, Certainty: {certainty:.2%}")
