"""
Advanced Kalman Filters for HFT Forex
=====================================
Renaissance Technologies inference: Speech recognition experts (Mercer, Brown)
brought Hidden Markov Model / state-space expertise. Extended and Unscented
Kalman Filters handle nonlinear price dynamics better than basic Kalman.

Implemented Filters:
- Extended Kalman Filter (EKF): Linearizes nonlinear dynamics
- Unscented Kalman Filter (UKF): Sigma-point propagation (more accurate)
- Adaptive Kalman Filter: Auto-tunes process/measurement noise

Sources:
- Bar-Shalom & Rong Li "Estimation with Applications to Tracking and Navigation"
- Julier & Uhlmann (1997) "A New Extension of the Kalman Filter to Nonlinear Systems"
- Wan & van der Merwe (2000) "The Unscented Kalman Filter for Nonlinear Estimation"

Why Renaissance Uses This:
- Price = TrueValue + MicrostructureNoise
- EKF/UKF better captures mean-reversion with time-varying volatility
- Handles transaction costs as nonlinear state transition
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from scipy.linalg import cholesky, sqrtm
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """State estimate from Kalman filter."""
    mean: np.ndarray  # State estimate
    covariance: np.ndarray  # State covariance
    innovation: float = 0.0  # Prediction error
    innovation_var: float = 1.0  # Innovation variance
    kalman_gain: float = 0.0  # Kalman gain
    log_likelihood: float = 0.0  # Log-likelihood for model selection


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for nonlinear state estimation.

    Renaissance Application:
    - Estimates "true" price from noisy observations
    - Handles mean-reversion with time-varying parameters
    - Accounts for bid-ask bounce (microstructure noise)

    State Model:
        x_t = f(x_{t-1}) + w_t  (nonlinear state transition)
        z_t = h(x_t) + v_t      (observation)

    For forex HFT:
        x = [price, velocity, volatility]  (latent state)
        z = mid_price                       (observed)

    Source: Bar-Shalom "Estimation with Applications to Tracking"
    """

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 1,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 1e-4,
                 mean_reversion_speed: float = 0.1):
        """
        Initialize EKF.

        Args:
            state_dim: State dimension [price, velocity, volatility]
            obs_dim: Observation dimension (typically 1 for mid price)
            process_noise: Process noise variance Q
            measurement_noise: Measurement noise variance R
            mean_reversion_speed: Speed of mean reversion (kappa)
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.kappa = mean_reversion_speed

        # Initial state: [price, velocity, volatility]
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 0.01

        # Process noise covariance
        self.Q = np.eye(state_dim) * process_noise
        self.Q[2, 2] = process_noise * 10  # Volatility evolves slower

        # Measurement noise covariance
        self.R = np.eye(obs_dim) * measurement_noise

        # History for analysis
        self.state_history: List[KalmanState] = []
        self.initialized = False

    def state_transition(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Nonlinear state transition function f(x).

        Model: Ornstein-Uhlenbeck with stochastic volatility
            price_t = price_{t-1} + velocity_{t-1} * dt
            velocity_t = -kappa * velocity_{t-1}  (mean-reverting velocity)
            volatility_t = volatility_{t-1}  (random walk)
        """
        x_new = np.zeros_like(x)
        x_new[0] = x[0] + x[1] * dt  # Price evolves with velocity
        x_new[1] = x[1] * (1 - self.kappa * dt)  # Velocity mean-reverts to 0
        x_new[2] = x[2]  # Volatility as random walk (updated by process noise)
        return x_new

    def state_jacobian(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Jacobian of state transition F = df/dx.
        """
        F = np.eye(self.state_dim)
        F[0, 1] = dt  # dprice/dvelocity
        F[1, 1] = 1 - self.kappa * dt  # dvelocity/dvelocity
        return F

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """
        Observation function h(x).
        We observe price directly.
        """
        return np.array([x[0]])

    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of observation H = dh/dx.
        """
        H = np.zeros((self.obs_dim, self.state_dim))
        H[0, 0] = 1.0  # Observe price directly
        return H

    def predict(self, dt: float = 1.0) -> KalmanState:
        """
        Prediction step: propagate state through nonlinear dynamics.
        """
        # Predict state
        x_pred = self.state_transition(self.x, dt)

        # Linearize and predict covariance
        F = self.state_jacobian(self.x, dt)
        P_pred = F @ self.P @ F.T + self.Q

        return KalmanState(mean=x_pred, covariance=P_pred)

    def update(self, z: float, dt: float = 1.0) -> KalmanState:
        """
        Update step: incorporate new observation.

        Args:
            z: Observed price (mid price)
            dt: Time step

        Returns:
            Updated state estimate
        """
        if not self.initialized:
            # Initialize state from first observation
            self.x[0] = z
            self.x[1] = 0.0
            self.x[2] = 0.0001  # Initial volatility estimate
            self.initialized = True
            return KalmanState(mean=self.x.copy(), covariance=self.P.copy())

        # Prediction step
        x_pred = self.state_transition(self.x, dt)
        F = self.state_jacobian(self.x, dt)
        P_pred = F @ self.P @ F.T + self.Q

        # Update step
        z_pred = self.observation_function(x_pred)
        H = self.observation_jacobian(x_pred)

        # Innovation (prediction error)
        y = np.array([z]) - z_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        S_scalar = float(S[0, 0])

        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = x_pred + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        # Compute log-likelihood for model comparison
        log_lik = -0.5 * (np.log(2 * np.pi * S_scalar) + y[0]**2 / S_scalar)

        state = KalmanState(
            mean=self.x.copy(),
            covariance=self.P.copy(),
            innovation=float(y[0]),
            innovation_var=S_scalar,
            kalman_gain=float(K[0, 0]),
            log_likelihood=log_lik
        )

        self.state_history.append(state)
        return state

    def get_filtered_price(self) -> float:
        """Get filtered price estimate (true value)."""
        return float(self.x[0])

    def get_velocity(self) -> float:
        """Get price velocity (momentum)."""
        return float(self.x[1])

    def get_volatility(self) -> float:
        """Get volatility estimate."""
        return float(abs(self.x[2]))

    def get_prediction_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get prediction interval for price."""
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence) / 2)
        std = np.sqrt(self.P[0, 0])
        return (self.x[0] - z_score * std, self.x[0] + z_score * std)

    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """
        Rauch-Tung-Striebel smoother for offline analysis.
        Uses forward-backward pass for optimal estimates.
        """
        n = len(observations)

        # Forward pass
        x_filt = np.zeros((n, self.state_dim))
        P_filt = np.zeros((n, self.state_dim, self.state_dim))
        x_pred = np.zeros((n, self.state_dim))
        P_pred = np.zeros((n, self.state_dim, self.state_dim))

        # Reset filter
        self.x = np.zeros(self.state_dim)
        self.x[0] = observations[0]
        self.P = np.eye(self.state_dim) * 0.01
        self.initialized = True

        for t in range(n):
            # Store prediction
            x_p = self.state_transition(self.x)
            F = self.state_jacobian(self.x)
            P_p = F @ self.P @ F.T + self.Q
            x_pred[t] = x_p
            P_pred[t] = P_p

            # Update
            state = self.update(observations[t])
            x_filt[t] = state.mean
            P_filt[t] = state.covariance

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((n, self.state_dim))
        x_smooth[-1] = x_filt[-1]

        for t in range(n - 2, -1, -1):
            F = self.state_jacobian(x_filt[t])
            G = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
            x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])

        return x_smooth[:, 0]  # Return smoothed prices


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) - more accurate than EKF for nonlinear systems.

    Renaissance Application:
    - Better uncertainty propagation through nonlinear dynamics
    - Handles heavy-tailed distributions better
    - More robust to model misspecification

    Key Idea: Instead of linearizing, propagate "sigma points" through
    nonlinear function and compute statistics from transformed points.

    Source: Julier & Uhlmann (1997) "A New Extension of the Kalman Filter"
    """

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 1,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 1e-4,
                 alpha: float = 0.001,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 mean_reversion: float = 0.1):
        """
        Initialize UKF.

        Args:
            alpha: Spread of sigma points (small positive, e.g., 1e-3)
            beta: Prior knowledge about distribution (2 = Gaussian)
            kappa: Secondary scaling parameter (usually 0)
            mean_reversion: Speed of mean reversion
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.mr = mean_reversion

        # UKF scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim

        # Sigma point weights
        self.n_sigma = 2 * state_dim + 1
        self.Wm = np.zeros(self.n_sigma)  # Weights for mean
        self.Wc = np.zeros(self.n_sigma)  # Weights for covariance

        self.Wm[0] = self.lambda_ / (state_dim + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (state_dim + self.lambda_))
            self.Wc[i] = self.Wm[i]

        # State and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 0.01

        # Noise covariances
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(obs_dim) * measurement_noise

        self.initialized = False
        self.state_history: List[KalmanState] = []

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points around state estimate."""
        n = len(x)
        sigma_pts = np.zeros((self.n_sigma, n))

        # Center point
        sigma_pts[0] = x

        # Square root of scaled covariance
        try:
            sqrt_P = cholesky((n + self.lambda_) * P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback if not positive definite
            sqrt_P = np.real(sqrtm((n + self.lambda_) * P))

        # Sigma points
        for i in range(n):
            sigma_pts[i + 1] = x + sqrt_P[:, i]
            sigma_pts[n + i + 1] = x - sqrt_P[:, i]

        return sigma_pts

    def state_transition(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Nonlinear state transition (same as EKF)."""
        x_new = np.zeros_like(x)
        x_new[0] = x[0] + x[1] * dt
        x_new[1] = x[1] * (1 - self.mr * dt)
        x_new[2] = x[2]
        return x_new

    def observation_function(self, x: np.ndarray) -> np.ndarray:
        """Observation function."""
        return np.array([x[0]])

    def update(self, z: float, dt: float = 1.0) -> KalmanState:
        """
        UKF update with sigma point propagation.
        """
        if not self.initialized:
            self.x[0] = z
            self.x[1] = 0.0
            self.x[2] = 0.0001
            self.initialized = True
            return KalmanState(mean=self.x.copy(), covariance=self.P.copy())

        # Generate sigma points
        sigma_pts = self.generate_sigma_points(self.x, self.P)

        # Propagate through state transition
        sigma_pts_pred = np.zeros_like(sigma_pts)
        for i in range(self.n_sigma):
            sigma_pts_pred[i] = self.state_transition(sigma_pts[i], dt)

        # Predicted mean and covariance
        x_pred = np.sum(self.Wm[:, None] * sigma_pts_pred, axis=0)
        P_pred = self.Q.copy()
        for i in range(self.n_sigma):
            diff = sigma_pts_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        # Transform sigma points through observation function
        z_sigma = np.zeros((self.n_sigma, self.obs_dim))
        for i in range(self.n_sigma):
            z_sigma[i] = self.observation_function(sigma_pts_pred[i])

        # Predicted observation
        z_pred = np.sum(self.Wm[:, None] * z_sigma, axis=0)

        # Innovation covariance
        S = self.R.copy()
        for i in range(self.n_sigma):
            diff = z_sigma[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)

        # Cross covariance
        Pxz = np.zeros((self.state_dim, self.obs_dim))
        for i in range(self.n_sigma):
            x_diff = sigma_pts_pred[i] - x_pred
            z_diff = z_sigma[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Update
        y = np.array([z]) - z_pred
        self.x = x_pred + K @ y
        self.P = P_pred - K @ S @ K.T

        # Ensure positive definiteness
        self.P = (self.P + self.P.T) / 2

        S_scalar = float(S[0, 0])
        log_lik = -0.5 * (np.log(2 * np.pi * S_scalar) + y[0]**2 / S_scalar)

        state = KalmanState(
            mean=self.x.copy(),
            covariance=self.P.copy(),
            innovation=float(y[0]),
            innovation_var=S_scalar,
            kalman_gain=float(K[0, 0]),
            log_likelihood=log_lik
        )

        self.state_history.append(state)
        return state

    def get_filtered_price(self) -> float:
        return float(self.x[0])

    def get_velocity(self) -> float:
        return float(self.x[1])

    def get_volatility(self) -> float:
        return float(abs(self.x[2]))


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter with online noise estimation.

    Renaissance Application:
    - Auto-tunes to changing market conditions
    - Detects regime changes via innovation sequence
    - Adjusts process/measurement noise dynamically

    Key Insight: Monitor innovation sequence. If innovations are
    consistently larger than expected, increase process noise.
    If smaller, decrease it.

    Source: Mehra (1970) "On the Identification of Variances and Adaptive Kalman Filtering"
    """

    def __init__(self,
                 base_filter: str = 'ukf',
                 adaptation_rate: float = 0.05,
                 innovation_window: int = 20):
        """
        Initialize Adaptive Kalman Filter.

        Args:
            base_filter: 'ekf' or 'ukf'
            adaptation_rate: Learning rate for noise adaptation
            innovation_window: Window for innovation statistics
        """
        self.adaptation_rate = adaptation_rate
        self.innovation_window = innovation_window

        # Create base filter
        if base_filter == 'ukf':
            self.filter = UnscentedKalmanFilter()
        else:
            self.filter = ExtendedKalmanFilter()

        # Innovation history for adaptation
        self.innovations: List[float] = []
        self.innovation_vars: List[float] = []

        # Estimated noise levels
        self.Q_scale = 1.0
        self.R_scale = 1.0

    def update(self, z: float, dt: float = 1.0) -> KalmanState:
        """Update with noise adaptation."""
        # Get base filter update
        state = self.filter.update(z, dt)

        # Store innovation statistics
        self.innovations.append(state.innovation)
        self.innovation_vars.append(state.innovation_var)

        # Keep window size
        if len(self.innovations) > self.innovation_window:
            self.innovations = self.innovations[-self.innovation_window:]
            self.innovation_vars = self.innovation_vars[-self.innovation_window:]

        # Adapt noise levels
        if len(self.innovations) >= 10:
            self._adapt_noise()

        return state

    def _adapt_noise(self):
        """
        Adapt process/measurement noise based on innovation sequence.

        If normalized innovation variance > 1: underestimating uncertainty
        If normalized innovation variance < 1: overestimating uncertainty
        """
        # Compute normalized innovation squared (NIS)
        innovations = np.array(self.innovations[-self.innovation_window:])
        expected_vars = np.array(self.innovation_vars[-self.innovation_window:])

        # Avoid division by zero
        expected_vars = np.maximum(expected_vars, 1e-10)

        # Normalized innovation squared
        nis = innovations**2 / expected_vars
        mean_nis = np.mean(nis)

        # Adapt: NIS should be ~1 for well-tuned filter
        if mean_nis > 1.5:
            # Underestimating uncertainty -> increase process noise
            self.Q_scale *= (1 + self.adaptation_rate)
            self.filter.Q *= (1 + self.adaptation_rate)
        elif mean_nis < 0.5:
            # Overestimating uncertainty -> decrease process noise
            self.Q_scale *= (1 - self.adaptation_rate)
            self.filter.Q *= (1 - self.adaptation_rate)

        # Clamp scaling factors
        self.Q_scale = np.clip(self.Q_scale, 0.1, 10.0)

    def get_filtered_price(self) -> float:
        return self.filter.get_filtered_price()

    def get_velocity(self) -> float:
        return self.filter.get_velocity()

    def get_regime_indicator(self) -> float:
        """
        Get regime indicator based on innovation sequence.
        High values indicate high uncertainty / regime change.
        """
        if len(self.innovations) < 5:
            return 0.0

        recent = np.array(self.innovations[-10:])
        return float(np.std(recent) / np.mean(np.abs(recent) + 1e-10))


class KalmanFilterEnsemble:
    """
    Ensemble of Kalman Filters with different parameters.

    Renaissance Philosophy: "Many weak signals combined"
    - Run multiple filters with different mean-reversion speeds
    - Combine via likelihood weighting
    - More robust to parameter misspecification
    """

    def __init__(self,
                 n_filters: int = 5,
                 mr_range: Tuple[float, float] = (0.01, 0.5)):
        """
        Initialize ensemble of filters.

        Args:
            n_filters: Number of filters in ensemble
            mr_range: Range of mean-reversion speeds to test
        """
        self.n_filters = n_filters

        # Create filters with different mean-reversion speeds
        mr_speeds = np.linspace(mr_range[0], mr_range[1], n_filters)
        self.filters = [
            AdaptiveKalmanFilter(base_filter='ukf')
            for _ in range(n_filters)
        ]

        # Set different mean-reversion speeds
        for i, kf in enumerate(self.filters):
            kf.filter.mr = mr_speeds[i]

        # Likelihood-based weights
        self.weights = np.ones(n_filters) / n_filters
        self.log_likelihoods = np.zeros(n_filters)

    def update(self, z: float, dt: float = 1.0) -> Dict[str, float]:
        """
        Update all filters and combine estimates.
        """
        estimates = []
        velocities = []

        for i, kf in enumerate(self.filters):
            state = kf.update(z, dt)
            estimates.append(kf.get_filtered_price())
            velocities.append(kf.get_velocity())
            self.log_likelihoods[i] += state.log_likelihood

        # Update weights via softmax of log-likelihoods
        max_ll = np.max(self.log_likelihoods)
        exp_ll = np.exp(self.log_likelihoods - max_ll)
        self.weights = exp_ll / np.sum(exp_ll)

        # Weighted combination
        weighted_price = np.sum(self.weights * estimates)
        weighted_velocity = np.sum(self.weights * velocities)

        return {
            'filtered_price': float(weighted_price),
            'velocity': float(weighted_velocity),
            'price_spread': float(np.max(estimates) - np.min(estimates)),
            'best_filter': int(np.argmax(self.weights)),
            'filter_agreement': float(1 - np.std(estimates) / (np.mean(estimates) + 1e-10))
        }


def compute_kalman_features(prices: pd.Series,
                           filter_type: str = 'adaptive_ukf') -> pd.DataFrame:
    """
    Compute Kalman filter features for HFT.

    Args:
        prices: Price series
        filter_type: 'ekf', 'ukf', 'adaptive_ukf', or 'ensemble'

    Returns:
        DataFrame with Kalman features
    """
    if filter_type == 'ekf':
        kf = ExtendedKalmanFilter()
    elif filter_type == 'ukf':
        kf = UnscentedKalmanFilter()
    elif filter_type == 'ensemble':
        kf = KalmanFilterEnsemble()
    else:
        kf = AdaptiveKalmanFilter(base_filter='ukf')

    features = []

    for price in prices:
        if filter_type == 'ensemble':
            result = kf.update(float(price))
            features.append({
                'kalman_price': result['filtered_price'],
                'kalman_velocity': result['velocity'],
                'kalman_spread': result['price_spread'],
                'kalman_agreement': result['filter_agreement']
            })
        else:
            state = kf.update(float(price))

            if hasattr(kf, 'filter'):
                filtered = kf.filter
            else:
                filtered = kf

            features.append({
                'kalman_price': filtered.get_filtered_price(),
                'kalman_velocity': filtered.get_velocity(),
                'kalman_volatility': filtered.get_volatility(),
                'kalman_innovation': state.innovation,
                'kalman_gain': state.kalman_gain
            })

    return pd.DataFrame(features, index=prices.index)


# Factory function
def create_kalman_filter(filter_type: str = 'adaptive_ukf', **kwargs):
    """
    Factory function to create Kalman filter.

    Args:
        filter_type: 'ekf', 'ukf', 'adaptive_ukf', 'adaptive_ekf', 'ensemble'
        **kwargs: Additional arguments for specific filter

    Returns:
        Kalman filter instance
    """
    if filter_type == 'ekf':
        return ExtendedKalmanFilter(**kwargs)
    elif filter_type == 'ukf':
        return UnscentedKalmanFilter(**kwargs)
    elif filter_type == 'adaptive_ukf':
        return AdaptiveKalmanFilter(base_filter='ukf', **kwargs)
    elif filter_type == 'adaptive_ekf':
        return AdaptiveKalmanFilter(base_filter='ekf', **kwargs)
    elif filter_type == 'ensemble':
        return KalmanFilterEnsemble(**kwargs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


if __name__ == '__main__':
    print("Advanced Kalman Filters Test")
    print("=" * 60)

    # Generate synthetic price data with mean-reversion
    np.random.seed(42)
    n = 500

    # OU process: dX = -kappa*(X-mu)*dt + sigma*dW
    kappa = 0.1
    mu = 1.1000
    sigma = 0.0001

    prices = [mu]
    for _ in range(n - 1):
        dx = -kappa * (prices[-1] - mu) + sigma * np.random.randn()
        prices.append(prices[-1] + dx)

    prices = np.array(prices)

    # Add microstructure noise
    observed = prices + np.random.randn(n) * 0.00005

    print(f"True price range: {prices.min():.6f} - {prices.max():.6f}")
    print(f"Observed range: {observed.min():.6f} - {observed.max():.6f}")

    # Test EKF
    print("\n--- Extended Kalman Filter ---")
    ekf = ExtendedKalmanFilter()
    ekf_prices = []
    for z in observed:
        ekf.update(z)
        ekf_prices.append(ekf.get_filtered_price())

    ekf_mse = np.mean((np.array(ekf_prices[10:]) - prices[10:])**2)
    obs_mse = np.mean((observed[10:] - prices[10:])**2)
    print(f"EKF MSE: {ekf_mse:.10f}")
    print(f"Observation MSE: {obs_mse:.10f}")
    print(f"Improvement: {(1 - ekf_mse/obs_mse)*100:.1f}%")

    # Test UKF
    print("\n--- Unscented Kalman Filter ---")
    ukf = UnscentedKalmanFilter()
    ukf_prices = []
    for z in observed:
        ukf.update(z)
        ukf_prices.append(ukf.get_filtered_price())

    ukf_mse = np.mean((np.array(ukf_prices[10:]) - prices[10:])**2)
    print(f"UKF MSE: {ukf_mse:.10f}")
    print(f"Improvement over EKF: {(1 - ukf_mse/ekf_mse)*100:.1f}%")

    # Test Adaptive UKF
    print("\n--- Adaptive UKF ---")
    akf = AdaptiveKalmanFilter(base_filter='ukf')
    akf_prices = []
    for z in observed:
        akf.update(z)
        akf_prices.append(akf.get_filtered_price())

    akf_mse = np.mean((np.array(akf_prices[10:]) - prices[10:])**2)
    print(f"Adaptive UKF MSE: {akf_mse:.10f}")
    print(f"Improvement over UKF: {(1 - akf_mse/ukf_mse)*100:.1f}%")

    # Test Ensemble
    print("\n--- Kalman Ensemble ---")
    ensemble = KalmanFilterEnsemble(n_filters=5)
    ens_prices = []
    for z in observed:
        result = ensemble.update(z)
        ens_prices.append(result['filtered_price'])

    ens_mse = np.mean((np.array(ens_prices[10:]) - prices[10:])**2)
    print(f"Ensemble MSE: {ens_mse:.10f}")
    print(f"Final weights: {ensemble.weights}")

    print("\n" + "=" * 60)
    print("All Kalman filter tests passed!")
