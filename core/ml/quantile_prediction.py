"""
Quantile Regression for Distribution Prediction

Predicts the FULL distribution of returns, not just direction.
"UP with 82%" is less useful than "95% chance between +2 and +15 pips".

Academic Citations:
- Koenker & Bassett (1978): "Regression Quantiles"
  Econometrica - Foundation of quantile regression

- Meinshausen (2006): "Quantile Regression Forests"
  JMLR - Quantile regression with random forests

- Gasthaus et al. (2019): "Probabilistic Forecasting with Spline Quantile Function RNNs"
  AISTATS - Deep learning quantile forecasting

- Lim et al. (2021): "Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting"
  International Journal of Forecasting - TFT with quantile outputs

- arXiv:2411.15674 (November 2024): "Quantile Regression for Financial Return Prediction"
  Latest research on distribution prediction

Chinese Quant Application:
- 华泰证券: "分位数回归在风险管理中的应用"
- 招商证券: Uses quantile predictions for VaR estimation
- 九坤投资: "分布预测比点预测更有价值" (Distribution > point prediction)

Key Insight:
    Point prediction: E[Y|X] = +5 pips (could be anywhere from -20 to +30)
    Quantile prediction: [q_0.05, q_0.50, q_0.95] = [-2, +5, +15] pips

    Now you KNOW the range with 90% confidence!

Trading Rules:
    - q_0.05 > 0: Even worst case is profitable → STRONG BUY
    - q_0.95 < 0: Even best case is losing → STRONG SELL
    - q_0.05 < 0 < q_0.95: Uncertain → REDUCE SIZE or SKIP
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from scipy import stats, optimize
import warnings


@dataclass
class QuantilePrediction:
    """Result of quantile prediction."""

    # Core quantiles (in return units, e.g., pips)
    q_05: float  # 5th percentile (worst case)
    q_25: float  # 25th percentile
    q_50: float  # Median
    q_75: float  # 75th percentile
    q_95: float  # 95th percentile (best case)

    # Derived metrics
    expected_return: float  # E[return] estimated from quantiles
    iqr: float  # Interquartile range (q_75 - q_25)
    range_90: float  # 90% range (q_95 - q_05)

    # Probability metrics
    prob_positive: float  # P(return > 0)
    prob_exceed_threshold: float  # P(return > threshold)

    # Trading signals
    direction: int  # 1=long, -1=short, 0=neutral
    signal_strength: str  # "strong", "moderate", "weak", "none"
    confidence_interval: Tuple[float, float]  # 90% CI

    # Risk metrics
    var_95: float  # Value at Risk (5th percentile of loss)
    cvar_95: float  # Conditional VaR (expected shortfall)

    # Position sizing
    suggested_size_multiplier: float  # 0 to 1


@dataclass
class QuantileModel:
    """Quantile regression model coefficients."""

    quantile: float
    coefficients: np.ndarray
    intercept: float


class QuantileRegressor:
    """
    Quantile Regression for financial returns.

    Estimates conditional quantiles Q_τ(Y|X) for multiple τ values.
    Uses pinball loss: L_τ(y, q) = (y - q) * (τ - I(y < q))

    Reference:
        Koenker & Bassett (1978), Equation (2.1)
    """

    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
        regularization: float = 0.01,
    ):
        """
        Initialize quantile regressor.

        Args:
            quantiles: Quantile levels to estimate
            regularization: L2 regularization strength
        """
        self.quantiles = sorted(quantiles)
        self.regularization = regularization
        self.models: Dict[float, QuantileModel] = {}
        self._fitted = False

    def _pinball_loss(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        tau: float,
    ) -> float:
        """
        Pinball (quantile) loss function.

        L_τ(y, q) = τ(y - q)⁺ + (1-τ)(q - y)⁺

        Reference:
            Koenker & Bassett (1978)
        """
        residual = y - y_pred
        return np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual))

    def _fit_single_quantile(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tau: float,
    ) -> QuantileModel:
        """Fit a single quantile using linear programming."""
        n_samples, n_features = X.shape

        # Add intercept
        X_aug = np.column_stack([np.ones(n_samples), X])
        n_params = n_features + 1

        def objective(params):
            y_pred = X_aug @ params
            loss = self._pinball_loss(y, y_pred, tau)
            reg = self.regularization * np.sum(params[1:] ** 2)
            return loss + reg

        # Initial guess (OLS)
        try:
            initial = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except:
            initial = np.zeros(n_params)

        # Optimize
        result = optimize.minimize(
            objective,
            initial,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )

        return QuantileModel(
            quantile=tau,
            coefficients=result.x[1:],
            intercept=result.x[0],
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Fit quantile regression models.

        Args:
            X: Features (n_samples, n_features)
            y: Target returns (n_samples,)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for tau in self.quantiles:
            self.models[tau] = self._fit_single_quantile(X, y, tau)

        self._fitted = True
        return self

    def predict_quantiles(
        self,
        X: np.ndarray,
        threshold: float = 0.0,
    ) -> QuantilePrediction:
        """
        Predict quantiles for new data.

        Args:
            X: Features (single sample or batch)
            threshold: Profit threshold for probability calculation

        Returns:
            QuantilePrediction with all quantiles and metrics
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")

        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict each quantile
        predictions = {}
        for tau, model in self.models.items():
            y_pred = model.intercept + X @ model.coefficients
            predictions[tau] = float(y_pred.mean())  # Average if batch

        # Extract standard quantiles
        q_05 = predictions.get(0.05, predictions[min(predictions.keys())])
        q_25 = predictions.get(0.25, q_05)
        q_50 = predictions.get(0.50, np.mean(list(predictions.values())))
        q_75 = predictions.get(0.75, q_50)
        q_95 = predictions.get(0.95, predictions[max(predictions.keys())])

        # Ensure monotonicity (quantile crossing fix)
        q_05, q_25, q_50, q_75, q_95 = sorted([q_05, q_25, q_50, q_75, q_95])

        # Expected return (mean of distribution)
        expected_return = (q_05 + q_25 + q_50 + q_75 + q_95) / 5

        # Ranges
        iqr = q_75 - q_25
        range_90 = q_95 - q_05

        # Probability of positive return (interpolate)
        if q_05 >= 0:
            prob_positive = 0.95
        elif q_95 <= 0:
            prob_positive = 0.05
        else:
            # Linear interpolation between quantiles
            # Find where 0 falls
            quantile_values = [q_05, q_25, q_50, q_75, q_95]
            quantile_probs = [0.05, 0.25, 0.50, 0.75, 0.95]
            prob_positive = np.interp(0, quantile_values, [0.95, 0.75, 0.5, 0.25, 0.05])
            prob_positive = 1 - prob_positive  # P(Y > 0)

        # Probability of exceeding threshold
        if q_05 >= threshold:
            prob_exceed = 0.95
        elif q_95 <= threshold:
            prob_exceed = 0.05
        else:
            prob_exceed = np.interp(
                threshold,
                [q_05, q_25, q_50, q_75, q_95],
                [0.95, 0.75, 0.5, 0.25, 0.05]
            )
            prob_exceed = 1 - prob_exceed

        # Direction and strength
        if q_05 > 0:
            direction = 1
            signal_strength = "strong"
        elif q_25 > 0:
            direction = 1
            signal_strength = "moderate"
        elif q_50 > 0:
            direction = 1
            signal_strength = "weak"
        elif q_95 < 0:
            direction = -1
            signal_strength = "strong"
        elif q_75 < 0:
            direction = -1
            signal_strength = "moderate"
        elif q_50 < 0:
            direction = -1
            signal_strength = "weak"
        else:
            direction = 0
            signal_strength = "none"

        # Risk metrics
        var_95 = -q_05 if direction == 1 else q_95  # Loss at 5%
        cvar_95 = -q_05  # Expected shortfall (simplified)

        # Position size multiplier
        # Strong signal = 1.0, weak = 0.5, none = 0
        size_map = {"strong": 1.0, "moderate": 0.75, "weak": 0.5, "none": 0.0}
        suggested_size = size_map[signal_strength]

        return QuantilePrediction(
            q_05=q_05,
            q_25=q_25,
            q_50=q_50,
            q_75=q_75,
            q_95=q_95,
            expected_return=expected_return,
            iqr=iqr,
            range_90=range_90,
            prob_positive=prob_positive,
            prob_exceed_threshold=prob_exceed,
            direction=direction,
            signal_strength=signal_strength,
            confidence_interval=(q_05, q_95),
            var_95=var_95,
            cvar_95=cvar_95,
            suggested_size_multiplier=suggested_size,
        )


class GradientBoostingQuantileRegressor:
    """
    Quantile regression using gradient boosting.

    Uses pinball loss with XGBoost/LightGBM/CatBoost.
    More powerful for non-linear relationships.

    Reference:
        LightGBM documentation: quantile regression objective
    """

    def __init__(
        self,
        quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        """
        Initialize GB quantile regressor.

        Args:
            quantiles: Quantile levels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models: Dict[float, Any] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingQuantileRegressor':
        """
        Fit gradient boosting quantile models.

        Args:
            X: Features
            y: Target

        Returns:
            self
        """
        try:
            import lightgbm as lgb
        except ImportError:
            warnings.warn("LightGBM not available, using linear quantile regression")
            self._fallback = QuantileRegressor(self.quantiles)
            self._fallback.fit(X, y)
            self._fitted = True
            return self

        for tau in self.quantiles:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=tau,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                verbose=-1,
            )
            model.fit(X, y)
            self.models[tau] = model

        self._fitted = True
        return self

    def predict_quantiles(
        self,
        X: np.ndarray,
        threshold: float = 0.0,
    ) -> QuantilePrediction:
        """Predict quantiles using gradient boosting models."""
        if hasattr(self, '_fallback'):
            return self._fallback.predict_quantiles(X, threshold)

        if not self._fitted:
            raise RuntimeError("Must call fit() first")

        X = np.atleast_2d(X)

        predictions = {}
        for tau, model in self.models.items():
            y_pred = model.predict(X)
            predictions[tau] = float(np.mean(y_pred))

        # Use same logic as linear quantile regressor
        q_05 = predictions.get(0.05, min(predictions.values()))
        q_25 = predictions.get(0.25, q_05)
        q_50 = predictions.get(0.50, np.mean(list(predictions.values())))
        q_75 = predictions.get(0.75, q_50)
        q_95 = predictions.get(0.95, max(predictions.values()))

        # Sort for monotonicity
        q_05, q_25, q_50, q_75, q_95 = sorted([q_05, q_25, q_50, q_75, q_95])

        # Same calculations as QuantileRegressor
        expected_return = (q_05 + q_25 + q_50 + q_75 + q_95) / 5
        iqr = q_75 - q_25
        range_90 = q_95 - q_05

        # Probability estimates
        if q_05 >= 0:
            prob_positive = 0.95
        elif q_95 <= 0:
            prob_positive = 0.05
        else:
            prob_positive = 0.5 + 0.45 * (q_50 / max(abs(q_05), abs(q_95), 0.001))

        prob_exceed = prob_positive if threshold == 0 else 0.5

        # Direction and strength
        if q_05 > 0:
            direction, signal_strength = 1, "strong"
        elif q_25 > 0:
            direction, signal_strength = 1, "moderate"
        elif q_50 > 0:
            direction, signal_strength = 1, "weak"
        elif q_95 < 0:
            direction, signal_strength = -1, "strong"
        elif q_75 < 0:
            direction, signal_strength = -1, "moderate"
        elif q_50 < 0:
            direction, signal_strength = -1, "weak"
        else:
            direction, signal_strength = 0, "none"

        size_map = {"strong": 1.0, "moderate": 0.75, "weak": 0.5, "none": 0.0}

        return QuantilePrediction(
            q_05=q_05,
            q_25=q_25,
            q_50=q_50,
            q_75=q_75,
            q_95=q_95,
            expected_return=expected_return,
            iqr=iqr,
            range_90=range_90,
            prob_positive=prob_positive,
            prob_exceed_threshold=prob_exceed,
            direction=direction,
            signal_strength=signal_strength,
            confidence_interval=(q_05, q_95),
            var_95=-q_05 if direction >= 0 else q_95,
            cvar_95=-q_05,
            suggested_size_multiplier=size_map[signal_strength],
        )


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_quantile_signal(
    q_05: float,
    q_50: float,
    q_95: float,
) -> Tuple[int, float, str]:
    """
    Quick trading signal from quantile predictions.

    Args:
        q_05: 5th percentile
        q_50: Median
        q_95: 95th percentile

    Returns:
        (direction, confidence, signal_strength)
    """
    if q_05 > 0:
        return 1, 0.95, "strong_long"
    elif q_95 < 0:
        return -1, 0.95, "strong_short"
    elif q_50 > 0 and q_05 > -abs(q_95) * 0.5:
        return 1, 0.6, "moderate_long"
    elif q_50 < 0 and q_95 < abs(q_05) * 0.5:
        return -1, 0.6, "moderate_short"
    else:
        return 0, 0.0, "uncertain"


def estimate_quantiles_from_ensemble(
    predictions: List[float],
    quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
) -> Dict[float, float]:
    """
    Estimate quantiles from ensemble of point predictions.

    Assumes predictions are samples from the predictive distribution.

    Args:
        predictions: List of predictions from ensemble
        quantiles: Quantile levels to estimate

    Returns:
        Dictionary mapping quantile levels to values
    """
    arr = np.array(predictions)
    return {q: float(np.percentile(arr, q * 100)) for q in quantiles}


def distribution_based_position_size(
    q_05: float,
    q_50: float,
    q_95: float,
    base_size: float = 1.0,
) -> float:
    """
    Calculate position size based on distribution shape.

    Args:
        q_05: 5th percentile
        q_50: Median
        q_95: 95th percentile
        base_size: Base position size

    Returns:
        Adjusted position size
    """
    # Risk/reward ratio from distribution
    upside = q_95 - q_50
    downside = q_50 - q_05

    if downside <= 0:
        downside = 0.001

    risk_reward = upside / downside

    # Probability of profit (estimated from quantiles)
    range_90 = q_95 - q_05
    if range_90 == 0:
        prob_profit = 0.5
    else:
        prob_profit = (q_95 - 0) / range_90 if q_05 < 0 < q_95 else (0.9 if q_05 > 0 else 0.1)

    # Kelly-style sizing
    if risk_reward <= 0 or prob_profit <= 0:
        return 0.0

    kelly = prob_profit - (1 - prob_profit) / risk_reward
    kelly = max(0, min(kelly, 0.5))  # Cap at 50%

    return base_size * kelly


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTILE REGRESSION - DISTRIBUTION PREDICTION")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Koenker & Bassett (1978): Econometrica - Foundation")
    print("  - arXiv:2411.15674 (2024): Financial return prediction")
    print("  - 华泰证券: Risk management applications")
    print()

    # Generate synthetic data
    np.random.seed(42)
    n = 500

    # Features
    X = np.random.randn(n, 5)

    # Target with heteroscedastic noise (volatility depends on features)
    signal = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    noise_scale = 0.5 + 0.3 * np.abs(X[:, 2])
    y = signal + noise_scale * np.random.randn(n)

    # Fit quantile regressor
    print("Fitting Quantile Regressor...")
    qr = QuantileRegressor()
    qr.fit(X, y)

    # Make predictions
    X_test = np.array([[0.5, 0.3, 0.1, 0.0, 0.0]])
    pred = qr.predict_quantiles(X_test)

    print()
    print("QUANTILE PREDICTION EXAMPLE")
    print("-" * 50)
    print(f"  5th percentile:   {pred.q_05:+.4f}")
    print(f"  25th percentile:  {pred.q_25:+.4f}")
    print(f"  Median (50th):    {pred.q_50:+.4f}")
    print(f"  75th percentile:  {pred.q_75:+.4f}")
    print(f"  95th percentile:  {pred.q_95:+.4f}")
    print()
    print(f"  Expected return:  {pred.expected_return:+.4f}")
    print(f"  90% Range:        [{pred.q_05:+.4f}, {pred.q_95:+.4f}]")
    print(f"  IQR:              {pred.iqr:.4f}")
    print()
    print(f"  P(return > 0):    {pred.prob_positive:.2%}")
    print(f"  Direction:        {pred.direction} ({pred.signal_strength})")
    print(f"  Size multiplier:  {pred.suggested_size_multiplier:.2f}")
    print()

    # Compare scenarios
    print("SCENARIO COMPARISON")
    print("-" * 50)

    scenarios = [
        ("Strong Long", np.array([[1.0, 1.0, 0.1, 0.0, 0.0]])),
        ("Uncertain", np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])),
        ("Strong Short", np.array([[-1.0, -1.0, 0.1, 0.0, 0.0]])),
    ]

    for name, X_scenario in scenarios:
        p = qr.predict_quantiles(X_scenario)
        print(f"  {name:12s}: q05={p.q_05:+.3f}, q50={p.q_50:+.3f}, q95={p.q_95:+.3f} → {p.signal_strength}")

    print()
    print("=" * 70)
    print("KEY INSIGHT:")
    print("  Point prediction: 'Expected return is +0.4'")
    print("  Quantile prediction: '90% of the time, return is between -0.2 and +1.0'")
    print()
    print("  Now you can make INFORMED position sizing decisions!")
    print("  Strong signal = entire 90% CI on one side of zero")
    print("=" * 70)
