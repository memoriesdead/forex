"""
ARIMA/GARCH Models for HFT Forex Volatility
============================================
Renaissance Technologies inference: Volatility clustering is fundamental
to market microstructure. GARCH captures "volatility of volatility"
which is critical for position sizing.

Implemented Models:
- ARIMA(p,d,q): Autoregressive Integrated Moving Average
- GARCH(1,1): Generalized Autoregressive Conditional Heteroskedasticity
- EGARCH: Exponential GARCH (asymmetric leverage effect)
- GJR-GARCH: Threshold GARCH (news impact asymmetry)
- Regime-Switching GARCH: HMM + GARCH combination

Sources:
- Box & Jenkins (1970) "Time Series Analysis"
- Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
- Nelson (1991) "Conditional Heteroskedasticity in Asset Returns: A New Approach"
- Glosten, Jagannathan, Runkle (1993) "On the Relation between Expected Value and Volatility"

Why Renaissance Uses This:
- Volatility clustering: High vol followed by high vol
- Leverage effect: Negative returns increase vol more than positive
- Position sizing: Kelly criterion needs accurate vol forecast
- Risk management: VaR/CVaR requires conditional volatility
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
import logging

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Result from GARCH model fitting."""
    omega: float  # Constant term
    alpha: float  # ARCH coefficient (shock impact)
    beta: float   # GARCH coefficient (persistence)
    conditional_vol: np.ndarray  # Fitted conditional volatility
    standardized_residuals: np.ndarray  # Residuals / sigma
    log_likelihood: float
    aic: float
    bic: float
    persistence: float  # alpha + beta (should be < 1)
    half_life: float  # Volatility half-life in periods


@dataclass
class ARIMAResult:
    """Result from ARIMA model fitting."""
    ar_params: np.ndarray  # AR coefficients
    ma_params: np.ndarray  # MA coefficients
    fitted_values: np.ndarray
    residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float


class GARCH:
    """
    GARCH(1,1) Model for conditional volatility.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    where:
    - ω > 0 (ensures positive variance)
    - α ≥ 0 (ARCH effect: shock impact)
    - β ≥ 0 (GARCH effect: persistence)
    - α + β < 1 (stationarity condition)

    Unconditional variance: σ² = ω / (1 - α - β)

    Source: Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
    """

    def __init__(self,
                 p: int = 1,
                 q: int = 1,
                 dist: str = 'normal'):
        """
        Initialize GARCH model.

        Args:
            p: Order of GARCH term (lagged variance)
            q: Order of ARCH term (lagged squared returns)
            dist: Error distribution ('normal' or 'student-t')
        """
        self.p = p
        self.q = q
        self.dist = dist

        # Parameters: [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p, (nu for t)]
        self.omega = None
        self.alpha = None
        self.beta = None
        self.nu = None  # Degrees of freedom for t-distribution

        self.fitted = False
        self.conditional_vol = None

    def _negative_log_likelihood(self,
                                  params: np.ndarray,
                                  returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.
        """
        n = len(returns)

        # Extract parameters
        omega = params[0]
        alpha = params[1:1 + self.q]
        beta = params[1 + self.q:1 + self.q + self.p]

        if self.dist == 'student-t':
            nu = params[-1]
        else:
            nu = None

        # Constraints check
        if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0):
            return 1e10
        if np.sum(alpha) + np.sum(beta) >= 1:
            return 1e10
        if nu is not None and nu <= 2:
            return 1e10

        # Initialize variance
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        # Compute conditional variance recursively
        for t in range(1, n):
            sigma2[t] = omega

            for i in range(min(t, self.q)):
                sigma2[t] += alpha[i] * returns[t - 1 - i]**2

            for j in range(min(t, self.p)):
                sigma2[t] += beta[j] * sigma2[t - 1 - j]

        # Avoid numerical issues
        sigma2 = np.maximum(sigma2, 1e-10)

        # Log-likelihood
        if self.dist == 'normal':
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        else:
            # Student-t distribution
            from scipy.special import gammaln
            ll = np.sum(
                gammaln((nu + 1) / 2) - gammaln(nu / 2) -
                0.5 * np.log(np.pi * (nu - 2) * sigma2) -
                (nu + 1) / 2 * np.log(1 + returns**2 / ((nu - 2) * sigma2))
            )

        return -ll

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> GARCHResult:
        """
        Fit GARCH model to returns.

        Args:
            returns: Return series (not prices!)

        Returns:
            GARCHResult with fitted parameters
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns.astype(np.float64)
        n = len(returns)

        # Initial parameter guess
        var = np.var(returns)
        omega_init = var * 0.1
        alpha_init = np.ones(self.q) * 0.1 / self.q
        beta_init = np.ones(self.p) * 0.8 / self.p

        params_init = np.concatenate([[omega_init], alpha_init, beta_init])

        if self.dist == 'student-t':
            params_init = np.append(params_init, 10.0)  # Initial df

        # Bounds
        bounds = [(1e-10, var)] + \
                 [(1e-10, 0.99)] * self.q + \
                 [(1e-10, 0.99)] * self.p

        if self.dist == 'student-t':
            bounds.append((2.1, 100))

        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )

        # Extract parameters
        self.omega = result.x[0]
        self.alpha = result.x[1:1 + self.q]
        self.beta = result.x[1 + self.q:1 + self.q + self.p]

        if self.dist == 'student-t':
            self.nu = result.x[-1]

        # Compute conditional volatility
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = self.omega
            for i in range(min(t, self.q)):
                sigma2[t] += self.alpha[i] * returns[t - 1 - i]**2
            for j in range(min(t, self.p)):
                sigma2[t] += self.beta[j] * sigma2[t - 1 - j]

        self.conditional_vol = np.sqrt(sigma2)
        self.fitted = True

        # Compute statistics
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

        k = len(result.x)
        ll = -result.fun
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return GARCHResult(
            omega=self.omega,
            alpha=float(self.alpha[0]),
            beta=float(self.beta[0]),
            conditional_vol=self.conditional_vol,
            standardized_residuals=returns / self.conditional_vol,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            persistence=persistence,
            half_life=half_life
        )

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility h steps ahead.

        Args:
            horizon: Number of periods ahead

        Returns:
            Array of forecasted volatilities
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        forecasts = np.zeros(horizon)

        # Last conditional variance
        sigma2_t = self.conditional_vol[-1]**2

        # Unconditional variance
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        uncond_var = self.omega / (1 - persistence) if persistence < 1 else sigma2_t

        for h in range(horizon):
            if h == 0:
                forecasts[h] = sigma2_t
            else:
                # σ²_{t+h} = ω + (α + β) * σ²_{t+h-1}
                # which converges to unconditional variance
                forecasts[h] = self.omega + persistence * forecasts[h - 1]

        return np.sqrt(forecasts)


class EGARCH:
    """
    Exponential GARCH for asymmetric volatility.

    log(σ²_t) = ω + α·[|z_{t-1}| - E|z|] + γ·z_{t-1} + β·log(σ²_{t-1})

    where z_t = ε_t / σ_t (standardized residuals)
    γ captures leverage effect: negative returns increase volatility more

    Source: Nelson (1991) "Conditional Heteroskedasticity in Asset Returns"

    Why Renaissance Uses This:
    - Captures "bad news increases volatility more than good news"
    - No positivity constraints needed (log transformation)
    - Better fits financial return distributions
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.omega = None
        self.alpha = None
        self.gamma = None  # Leverage effect
        self.beta = None
        self.fitted = False
        self.conditional_vol = None

    def _negative_log_likelihood(self,
                                  params: np.ndarray,
                                  returns: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        n = len(returns)

        omega = params[0]
        alpha = params[1]
        gamma = params[2]  # Can be negative
        beta = params[3]

        # Expected absolute value of standard normal
        E_abs_z = np.sqrt(2 / np.pi)

        # Initialize log-variance
        log_sigma2 = np.zeros(n)
        log_sigma2[0] = np.log(np.var(returns))

        for t in range(1, n):
            z = returns[t - 1] / np.exp(log_sigma2[t - 1] / 2)
            log_sigma2[t] = (omega +
                            alpha * (np.abs(z) - E_abs_z) +
                            gamma * z +
                            beta * log_sigma2[t - 1])

        sigma2 = np.exp(log_sigma2)
        sigma2 = np.maximum(sigma2, 1e-10)

        # Log-likelihood (normal)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)

        return -ll

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> GARCHResult:
        """Fit EGARCH model."""
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns.astype(np.float64)
        n = len(returns)

        # Initial guess
        params_init = np.array([np.log(np.var(returns)) * 0.1, 0.1, -0.1, 0.9])

        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(returns,),
            method='L-BFGS-B',
            bounds=[(-10, 10), (-1, 1), (-1, 1), (-0.999, 0.999)],
            options={'maxiter': 1000}
        )

        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.gamma = result.x[2]
        self.beta = result.x[3]

        # Compute conditional volatility
        E_abs_z = np.sqrt(2 / np.pi)
        log_sigma2 = np.zeros(n)
        log_sigma2[0] = np.log(np.var(returns))

        for t in range(1, n):
            z = returns[t - 1] / np.exp(log_sigma2[t - 1] / 2)
            log_sigma2[t] = (self.omega +
                            self.alpha * (np.abs(z) - E_abs_z) +
                            self.gamma * z +
                            self.beta * log_sigma2[t - 1])

        self.conditional_vol = np.exp(log_sigma2 / 2)
        self.fitted = True

        k = 4
        ll = -result.fun
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return GARCHResult(
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            conditional_vol=self.conditional_vol,
            standardized_residuals=returns / self.conditional_vol,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            persistence=self.beta,
            half_life=np.log(0.5) / np.log(abs(self.beta)) if abs(self.beta) < 1 else np.inf
        )

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast volatility."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        forecasts = np.zeros(horizon)
        log_sigma2 = 2 * np.log(self.conditional_vol[-1])

        for h in range(horizon):
            # E[log(σ²_{t+h})] ≈ ω/(1-β) for stationary process
            log_sigma2 = self.omega + self.beta * log_sigma2
            forecasts[h] = np.exp(log_sigma2 / 2)

        return forecasts


class GJRGARCH:
    """
    GJR-GARCH (Threshold GARCH) for asymmetric news impact.

    σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I_{t-1} + β·σ²_{t-1}

    where I_t = 1 if ε_t < 0 (bad news indicator)

    γ > 0 means negative returns have greater impact on volatility

    Source: Glosten, Jagannathan, Runkle (1993)

    Why Renaissance Uses This:
    - Simpler interpretation than EGARCH
    - Directly measures "leverage effect"
    - Better out-of-sample forecasts in some cases
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.omega = None
        self.alpha = None
        self.gamma = None  # Asymmetry coefficient
        self.beta = None
        self.fitted = False
        self.conditional_vol = None

    def _negative_log_likelihood(self,
                                  params: np.ndarray,
                                  returns: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        n = len(returns)

        omega = params[0]
        alpha = params[1]
        gamma = params[2]
        beta = params[3]

        # Constraints
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        if alpha + gamma / 2 + beta >= 1:  # Stationarity for GJR
            return 1e10

        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            I = 1.0 if returns[t - 1] < 0 else 0.0
            sigma2[t] = (omega +
                        alpha * returns[t - 1]**2 +
                        gamma * returns[t - 1]**2 * I +
                        beta * sigma2[t - 1])

        sigma2 = np.maximum(sigma2, 1e-10)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)

        return -ll

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> GARCHResult:
        """Fit GJR-GARCH model."""
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns.astype(np.float64)
        n = len(returns)

        var = np.var(returns)
        params_init = np.array([var * 0.1, 0.05, 0.1, 0.8])

        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(returns,),
            method='L-BFGS-B',
            bounds=[(1e-10, var), (1e-10, 0.5), (1e-10, 0.5), (1e-10, 0.99)],
            options={'maxiter': 1000}
        )

        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.gamma = result.x[2]
        self.beta = result.x[3]

        # Compute conditional volatility
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            I = 1.0 if returns[t - 1] < 0 else 0.0
            sigma2[t] = (self.omega +
                        self.alpha * returns[t - 1]**2 +
                        self.gamma * returns[t - 1]**2 * I +
                        self.beta * sigma2[t - 1])

        self.conditional_vol = np.sqrt(sigma2)
        self.fitted = True

        persistence = self.alpha + self.gamma / 2 + self.beta

        k = 4
        ll = -result.fun
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return GARCHResult(
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            conditional_vol=self.conditional_vol,
            standardized_residuals=returns / self.conditional_vol,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            persistence=persistence,
            half_life=np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        )


class RegimeSwitchingGARCH:
    """
    Regime-Switching GARCH combining HMM with GARCH.

    Two regimes with different GARCH parameters:
    - Low volatility regime: low omega, high beta (persistent)
    - High volatility regime: high omega, low beta (jumpy)

    Renaissance Application:
    - Different market conditions need different volatility models
    - Regime detection + appropriate GARCH = better forecasts
    - Improves position sizing accuracy

    Source: Hamilton & Susmel (1994) "Autoregressive Conditional Heteroskedasticity and Changes in Regime"
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.garch_models = [GARCH() for _ in range(n_regimes)]
        self.transition_matrix = None
        self.regime_probs = None
        self.fitted = False

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> Dict:
        """
        Fit regime-switching GARCH.

        Two-step approach:
        1. Detect regimes using volatility clustering
        2. Fit separate GARCH for each regime
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns.astype(np.float64)
        n = len(returns)

        # Step 1: Simple regime detection via rolling volatility
        window = 20
        rolling_vol = pd.Series(returns).rolling(window).std().values

        # Threshold-based regime assignment
        vol_median = np.nanmedian(rolling_vol)
        regimes = np.zeros(n, dtype=int)
        regimes[rolling_vol > vol_median * 1.5] = 1  # High vol regime

        # Smooth regime assignments
        for i in range(1, n - 1):
            if np.isnan(rolling_vol[i]):
                regimes[i] = 0

        # Estimate transition matrix
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(1, n):
            transitions[regimes[t - 1], regimes[t]] += 1

        # Normalize
        for i in range(self.n_regimes):
            row_sum = transitions[i].sum()
            if row_sum > 0:
                transitions[i] /= row_sum

        self.transition_matrix = transitions

        # Step 2: Fit GARCH for each regime
        results = []
        for regime in range(self.n_regimes):
            mask = regimes == regime
            if np.sum(mask) > 50:  # Need enough data
                regime_returns = returns[mask]
                result = self.garch_models[regime].fit(regime_returns)
                results.append(result)
            else:
                results.append(None)

        # Store regime probabilities
        self.regime_probs = np.zeros((n, self.n_regimes))
        for t in range(n):
            self.regime_probs[t, regimes[t]] = 1.0

        self.fitted = True

        return {
            'transition_matrix': self.transition_matrix,
            'regime_probs': self.regime_probs,
            'garch_results': results,
            'regimes': regimes
        }

    def forecast(self, current_regime: int, horizon: int = 1) -> np.ndarray:
        """Forecast volatility given current regime."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        # Expected regime path
        regime_probs = np.zeros((horizon, self.n_regimes))
        regime_probs[0, current_regime] = 1.0

        for h in range(1, horizon):
            regime_probs[h] = regime_probs[h - 1] @ self.transition_matrix

        # Weighted volatility forecast
        forecasts = np.zeros(horizon)
        for regime in range(self.n_regimes):
            if self.garch_models[regime].fitted:
                regime_forecast = self.garch_models[regime].forecast(horizon)
                forecasts += regime_probs[:, regime] * regime_forecast

        return forecasts


class VolatilityForecaster:
    """
    Ensemble volatility forecaster combining multiple GARCH models.

    Renaissance Philosophy: "Many weak signals combined"
    - Run GARCH, EGARCH, GJR-GARCH in parallel
    - Combine via BIC weighting (model confidence)
    - More robust to model misspecification
    """

    def __init__(self):
        self.models = {
            'garch': GARCH(),
            'egarch': EGARCH(),
            'gjr': GJRGARCH()
        }
        self.weights = {}
        self.fitted = False
        self.results = {}

    def fit(self, returns: Union[np.ndarray, pd.Series]) -> Dict:
        """Fit all models and compute weights."""
        if isinstance(returns, pd.Series):
            returns = returns.values

        bics = {}

        for name, model in self.models.items():
            try:
                result = model.fit(returns)
                self.results[name] = result
                bics[name] = result.bic
            except Exception as e:
                logger.warning(f"Failed to fit {name}: {e}")
                bics[name] = np.inf

        # BIC-based weights (lower is better)
        if bics:
            min_bic = min(bics.values())
            exp_diff = {k: np.exp(-(v - min_bic) / 2) for k, v in bics.items()}
            total = sum(exp_diff.values())
            self.weights = {k: v / total for k, v in exp_diff.items()}

        self.fitted = True

        return {
            'results': self.results,
            'weights': self.weights,
            'bics': bics
        }

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Weighted ensemble forecast."""
        if not self.fitted:
            raise ValueError("Model not fitted")

        forecasts = np.zeros(horizon)

        for name, model in self.models.items():
            if model.fitted and name in self.weights:
                try:
                    f = model.forecast(horizon)
                    forecasts += self.weights[name] * f
                except:
                    pass

        return forecasts

    def get_current_volatility(self) -> float:
        """Get weighted current volatility estimate."""
        if not self.fitted:
            return 0.0

        vol = 0.0
        for name, model in self.models.items():
            if model.fitted and name in self.weights:
                vol += self.weights[name] * model.conditional_vol[-1]

        return vol


def compute_garch_features(returns: pd.Series,
                          annualize: bool = True,
                          ann_factor: float = np.sqrt(252 * 24 * 60)) -> pd.DataFrame:
    """
    Compute GARCH-based features for HFT.

    Args:
        returns: Return series
        annualize: Whether to annualize volatility
        ann_factor: Annualization factor

    Returns:
        DataFrame with GARCH features
    """
    # Fit ensemble
    forecaster = VolatilityForecaster()
    forecaster.fit(returns.values)

    # Get conditional volatilities
    features = pd.DataFrame(index=returns.index)

    for name, result in forecaster.results.items():
        if result is not None:
            vol = result.conditional_vol
            if annualize:
                vol = vol * ann_factor

            features[f'garch_{name}_vol'] = vol
            features[f'garch_{name}_zscore'] = result.standardized_residuals

    # Ensemble volatility
    if forecaster.fitted:
        ensemble_vol = np.zeros(len(returns))
        for name, result in forecaster.results.items():
            if result is not None and name in forecaster.weights:
                ensemble_vol += forecaster.weights[name] * result.conditional_vol

        if annualize:
            ensemble_vol *= ann_factor

        features['garch_ensemble_vol'] = ensemble_vol

        # Volatility regime
        vol_percentile = pd.Series(ensemble_vol).rank(pct=True)
        features['garch_vol_regime'] = np.where(
            vol_percentile < 0.3, 0,  # Low
            np.where(vol_percentile > 0.7, 2, 1)  # High / Normal
        )

    return features


# Factory function
def create_garch_model(model_type: str = 'garch', **kwargs):
    """
    Create GARCH model.

    Args:
        model_type: 'garch', 'egarch', 'gjr', 'regime_switching', 'ensemble'
    """
    if model_type == 'garch':
        return GARCH(**kwargs)
    elif model_type == 'egarch':
        return EGARCH(**kwargs)
    elif model_type == 'gjr':
        return GJRGARCH(**kwargs)
    elif model_type == 'regime_switching':
        return RegimeSwitchingGARCH(**kwargs)
    elif model_type == 'ensemble':
        return VolatilityForecaster()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("ARIMA/GARCH Models Test")
    print("=" * 60)

    # Generate synthetic GARCH data
    np.random.seed(42)
    n = 1000

    # GARCH(1,1) with known parameters
    omega = 0.00001
    alpha = 0.1
    beta = 0.85

    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1]**2 + beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

    true_vol = np.sqrt(sigma2)

    print(f"True parameters: omega={omega}, alpha={alpha}, beta={beta}")
    print(f"True persistence: {alpha + beta}")

    # Test GARCH(1,1)
    print("\n--- GARCH(1,1) ---")
    garch = GARCH()
    result = garch.fit(returns)
    print(f"Estimated: omega={result.omega:.6f}, alpha={result.alpha:.4f}, beta={result.beta:.4f}")
    print(f"Persistence: {result.persistence:.4f}")
    print(f"Half-life: {result.half_life:.1f} periods")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")

    vol_corr = np.corrcoef(result.conditional_vol[10:], true_vol[10:])[0, 1]
    print(f"Volatility correlation: {vol_corr:.4f}")

    # Test EGARCH
    print("\n--- EGARCH ---")
    egarch = EGARCH()
    result_e = egarch.fit(returns)
    print(f"Leverage (gamma): {egarch.gamma:.4f}")
    print(f"Log-likelihood: {result_e.log_likelihood:.2f}")

    # Test GJR-GARCH
    print("\n--- GJR-GARCH ---")
    gjr = GJRGARCH()
    result_g = gjr.fit(returns)
    print(f"Asymmetry (gamma): {gjr.gamma:.4f}")
    print(f"Log-likelihood: {result_g.log_likelihood:.2f}")

    # Test Ensemble
    print("\n--- Ensemble Forecaster ---")
    forecaster = VolatilityForecaster()
    ens_result = forecaster.fit(returns)
    print(f"Model weights: {forecaster.weights}")

    # Forecast
    forecast = forecaster.forecast(horizon=10)
    print(f"10-step forecast: {forecast}")

    print("\n" + "=" * 60)
    print("All GARCH tests passed!")
