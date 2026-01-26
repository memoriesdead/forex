"""
Pairs Trading with Kalman Filter for Forex
===========================================

ACADEMIC CITATIONS:
===================

Primary Papers:
    Rad et al. (2023)
    "Neural Augmented Kalman Filtering with Bollinger Bands for Pairs Trading"
    arXiv:2210.15448
    https://arxiv.org/abs/2210.15448

    Key Innovation: Kalman Filter for cointegration coefficient estimation
    with low complexity and latency. Achieves superior

 performance over
    static cointegration methods.

    Gonçalves & Stengos (2021)
    "Systematic risk in pairs trading and dynamic parameterization"
    Economics Letters, Vol. 202
    https://www.sciencedirect.com/science/article/abs/pii/S0165176521001191

    Key Innovation: Kalman Filter for intertemporal estimation of
    cointegration coefficients and ASR thresholds.

Related Work:
    - "A computing platform for pairs-trading via Kalman-HMM" (2017)
      Journal of Big Data, Springer
      https://link.springer.com/article/10.1186/s40537-017-0106-3

    - "Pairs trading strategy optimization using reinforcement learning" (2016)
      Soft Computing, Springer
      https://link.springer.com/article/10.1007/s00500-016-2298-4

METHODOLOGY:
============

Cointegration Model:
    Y_t = β * X_t + ε_t

    where:
    - Y_t: Price of asset 1 (e.g., EURUSD)
    - X_t: Price of asset 2 (e.g., GBPUSD)
    - β: Hedge ratio (cointegration coefficient)
    - ε_t: Spread (mean-reverting residual)

Kalman Filter State Space:
    State equation:  β_t = β_{t-1} + w_t
    Observation:     Y_t = β_t * X_t + v_t

    where:
    - w_t ~ N(0, Q): Process noise
    - v_t ~ N(0, R): Measurement noise

Trading Signals:
    Spread: s_t = Y_t - β_t * X_t
    Z-score: z_t = (s_t - μ) / σ

    Entry long: z_t < -threshold (spread below mean)
    Entry short: z_t > +threshold (spread above mean)
    Exit: z_t crosses zero

COINTEGRATED FOREX PAIRS:
==========================

High Correlation (Good for Pairs Trading):
    1. EURUSD / GBPUSD  (Euro zone correlation)
    2. AUDUSD / NZDUSD  (Oceania correlation)
    3. EURJPY / GBPJPY  (Yen cross correlation)
    4. USDCAD / USDCHF  (USD majors)
    5. EURAUD / EURGBP  (EUR crosses)

Testing Process:
    1. Johansen cointegration test
    2. Augmented Dickey-Fuller test on spread
    3. Half-life < 10 days (mean reversion speed)

CHINESE QUANT PERFORMANCE:
==========================

From "An innovative high-frequency statistical arbitrage in Chinese futures"
(Journal of Innovation & Knowledge, 2023):
    - 81% cumulative returns (in-sample)
    - 21% returns (out-of-sample) after transaction costs
    - Max drawdown: <1%
    - Framework: Cointegration + Kalman + Hurst index

Source: https://www.sciencedirect.com/science/article/pii/S2444569X23001257

USAGE:
======

    from core.trading.pairs_trading_kalman import KalmanPairsTrader

    trader = KalmanPairsTrader(
        symbol1='EURUSD',
        symbol2='GBPUSD',
        entry_threshold=2.0,  # Z-score threshold
        exit_threshold=0.5
    )

    # Feed price data
    signal = trader.update(price1=1.0850, price2=1.2650)

    if signal == 1:  # Long spread
        # Buy EURUSD, Sell GBPUSD
    elif signal == -1:  # Short spread
        # Sell EURUSD, Buy GBPUSD
    elif signal == 0:  # Exit
        # Close positions
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanPairsConfig:
    """
    Configuration for Kalman Filter Pairs Trading.

    Based on Rad et al. (2023) arXiv:2210.15448.
    """
    # Kalman Filter parameters
    process_variance: float = 1e-5   # Q: How much β can change per step
    measurement_variance: float = 1e-3  # R: Observation noise

    # Trading parameters
    entry_threshold: float = 2.0     # Z-score to enter trade
    exit_threshold: float = 0.5      # Z-score to exit trade
    stop_loss_threshold: float = 4.0  # Z-score to force exit

    # Spread statistics window
    lookback: int = 100  # Window for spread mean/std calculation

    # Position sizing
    max_position: float = 1.0  # Maximum position size


class KalmanFilter1D:
    """
    1D Kalman Filter for hedge ratio estimation.

    State: β (hedge ratio)
    Observation: Y = β*X + noise
    """

    def __init__(self, process_variance: float, measurement_variance: float):
        self.Q = process_variance      # Process noise covariance
        self.R = measurement_variance   # Measurement noise covariance

        # State
        self.beta = 0.0      # Hedge ratio estimate
        self.P = 1.0         # Estimate error covariance

    def update(self, y: float, x: float) -> float:
        """
        Kalman filter update step.

        Observation model: y = β * x + v

        Args:
            y: Observed value of asset 1
            x: Observed value of asset 2

        Returns:
            Updated β estimate
        """
        # Predict step
        beta_pred = self.beta  # β_t|t-1 = β_{t-1|t-1} (random walk)
        P_pred = self.P + self.Q  # P_t|t-1 = P_{t-1|t-1} + Q

        # Update step
        # Innovation: y_t - β_t|t-1 * x_t
        innovation = y - beta_pred * x

        # Innovation covariance: S = H * P * H^T + R
        # where H = x (observation matrix)
        S = x * P_pred * x + self.R

        # Kalman gain: K = P * H^T * S^{-1}
        K = P_pred * x / S

        # Update state: β_t|t = β_t|t-1 + K * innovation
        self.beta = beta_pred + K * innovation

        # Update covariance: P_t|t = (I - K*H) * P_t|t-1
        self.P = (1 - K * x) * P_pred

        return self.beta


class KalmanPairsTrader:
    """
    Pairs trading strategy using Kalman Filter for dynamic hedge ratio.

    Implementation based on:
        Rad et al. (2023) arXiv:2210.15448
        Gonçalves & Stengos (2021) Economics Letters
    """

    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        config: Optional[KalmanPairsConfig] = None
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.config = config or KalmanPairsConfig()

        # Kalman filter for β estimation
        self.kf = KalmanFilter1D(
            process_variance=self.config.process_variance,
            measurement_variance=self.config.measurement_variance
        )

        # Spread history (for mean/std calculation)
        self.spread_history = deque(maxlen=self.config.lookback)

        # Current position
        self.position = 0  # 1 = long spread, -1 = short spread, 0 = flat

        # Statistics
        self.num_trades = 0
        self.pnl = 0.0

        logger.info(
            f"Initialized Kalman Pairs Trader: {symbol1}/{symbol2}, "
            f"entry_z={self.config.entry_threshold}, "
            f"exit_z={self.config.exit_threshold}"
        )

    def update(
        self,
        price1: float,
        price2: float
    ) -> Tuple[int, Dict[str, float]]:
        """
        Update filter and generate trading signal.

        Args:
            price1: Current price of asset 1
            price2: Current price of asset 2

        Returns:
            signal: 1 (long spread), -1 (short spread), 0 (exit/hold)
            info: Dict with β, spread, z-score, etc.
        """
        # Update Kalman filter to get current β
        beta = self.kf.update(price1, price2)

        # Calculate spread
        spread = price1 - beta * price2

        # Add to history
        self.spread_history.append(spread)

        # Calculate z-score
        if len(self.spread_history) < 20:  # Need minimum history
            return 0, {
                'beta': beta,
                'spread': spread,
                'z_score': 0.0,
                'position': self.position
            }

        spread_mean = np.mean(self.spread_history)
        spread_std = np.std(self.spread_history)

        if spread_std < 1e-8:  # Avoid division by zero
            z_score = 0.0
        else:
            z_score = (spread - spread_mean) / spread_std

        # Generate trading signal
        signal = self._generate_signal(z_score)

        # Info for logging/monitoring
        info = {
            'beta': beta,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'z_score': z_score,
            'position': self.position,
            'signal': signal,
            'P': self.kf.P  # Kalman filter uncertainty
        }

        return signal, info

    def _generate_signal(self, z_score: float) -> int:
        """
        Generate trading signal based on z-score.

        Args:
            z_score: Current z-score of spread

        Returns:
            signal: 1 (long), -1 (short), 0 (exit/hold)
        """
        # Stop loss
        if abs(z_score) > self.config.stop_loss_threshold:
            if self.position != 0:
                logger.warning(f"Stop loss triggered at z={z_score:.2f}")
                self.position = 0
                return 0

        # Exit conditions
        if self.position == 1:  # Currently long spread
            if z_score > -self.config.exit_threshold:
                self.position = 0
                self.num_trades += 1
                return 0  # Exit long

        elif self.position == -1:  # Currently short spread
            if z_score < self.config.exit_threshold:
                self.position = 0
                self.num_trades += 1
                return 0  # Exit short

        # Entry conditions
        if self.position == 0:  # Currently flat
            if z_score < -self.config.entry_threshold:
                # Spread is below mean → Long spread
                self.position = 1
                return 1

            elif z_score > self.config.entry_threshold:
                # Spread is above mean → Short spread
                self.position = -1
                return -1

        # Hold current position
        return 0

    def get_position_sizes(self, capital: float) -> Tuple[float, float]:
        """
        Calculate position sizes for current signal.

        Args:
            capital: Total capital to allocate

        Returns:
            qty1: Quantity of asset 1
            qty2: Quantity of asset 2
        """
        if self.position == 0:
            return 0.0, 0.0

        # Allocate capital based on hedge ratio
        beta = self.kf.beta
        allocation = capital * self.config.max_position

        if self.position == 1:  # Long spread
            # Buy asset 1, sell asset 2
            qty1 = allocation / (1 + beta)
            qty2 = -beta * qty1

        else:  # Short spread
            # Sell asset 1, buy asset 2
            qty1 = -allocation / (1 + beta)
            qty2 = beta * qty1

        return qty1, qty2

    def get_stats(self) -> Dict[str, float]:
        """Get trading statistics."""
        return {
            'num_trades': self.num_trades,
            'current_position': self.position,
            'beta': self.kf.beta,
            'beta_uncertainty': self.kf.P
        }


# Pre-configured pairs for forex
FOREX_PAIRS = {
    'EUR_GBP': ('EURUSD', 'GBPUSD'),
    'AUD_NZD': ('AUDUSD', 'NZDUSD'),
    'EUR_JPY_GBP_JPY': ('EURJPY', 'GBPJPY'),
    'USD_CAD_CHF': ('USDCAD', 'USDCHF'),
    'EUR_AUD_GBP': ('EURAUD', 'EURGBP'),
}


def create_pairs_trader(
    pair_name: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5
) -> KalmanPairsTrader:
    """
    Create Kalman pairs trader for common forex pairs.

    Args:
        pair_name: One of FOREX_PAIRS keys
        entry_threshold: Z-score to enter
        exit_threshold: Z-score to exit

    Returns:
        KalmanPairsTrader instance
    """
    if pair_name not in FOREX_PAIRS:
        raise ValueError(f"Unknown pair: {pair_name}. Choose from {list(FOREX_PAIRS.keys())}")

    symbol1, symbol2 = FOREX_PAIRS[pair_name]

    config = KalmanPairsConfig(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )

    return KalmanPairsTrader(symbol1, symbol2, config)
