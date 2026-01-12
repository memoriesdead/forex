"""
Deep Learning Volatility Models
===============================
Hybrid GARCH + Deep Learning for volatility forecasting.

Sources:
- "A Hybrid GARCH and Deep Learning Method for Volatility Prediction" (Wiley 2024)
- "Multi-scale GARCH information" (ScienceDirect 2025)

Models:
1. GARCH-LSTM Hybrid: GARCH for regime, LSTM for dynamics
2. GRU-LSTM Hybrid: GRU features + LSTM temporal
3. Simple RNN for real-time (ultra-fast inference)

Why these matter for HFT:
- Better volatility forecasts = better Kelly sizing
- Better sizing = higher Sharpe, lower drawdowns
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, using numpy fallback")


@dataclass
class VolatilityPrediction:
    """Volatility prediction result."""
    current_vol: float  # Current volatility estimate
    forecast_1: float  # 1-step ahead forecast
    forecast_5: float  # 5-step ahead forecast
    forecast_10: float  # 10-step ahead forecast
    regime: int  # 0=low, 1=normal, 2=high
    confidence: float  # Prediction confidence (0-1)


class GARCHLSTMHybrid:
    """
    GARCH-LSTM Hybrid Volatility Model.

    Architecture:
    1. GARCH(1,1) for baseline volatility
    2. LSTM processes GARCH residuals to capture nonlinear dynamics
    3. Final prediction = GARCH baseline + LSTM adjustment

    This captures:
    - GARCH: Volatility clustering, mean reversion
    - LSTM: Regime shifts, nonlinear patterns
    """

    def __init__(self, hidden_size: int = 32, n_layers: int = 2,
                 lookback: int = 50, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lookback = lookback
        self.dropout = dropout

        # GARCH parameters (will be updated during fitting)
        self.omega = 0.00001
        self.alpha = 0.05
        self.beta = 0.90

        # LSTM model
        self.model = None
        self.is_fitted = False

        if HAS_TORCH:
            self._build_model()

    def _build_model(self):
        """Build LSTM model."""
        if not HAS_TORCH:
            return

        class LSTMVolModel(nn.Module):
            def __init__(self, input_size, hidden_size, n_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    dropout=dropout,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out

        # Input: [returns, garch_vol, residuals, lagged_vols]
        self.model = LSTMVolModel(4, self.hidden_size, self.n_layers, self.dropout)

    def _garch_filter(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply GARCH(1,1) filter.

        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

        Returns:
            (conditional_variance, standardized_residuals)
        """
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns[:min(20, n)])  # Initial variance

        for t in range(1, n):
            sigma2[t] = (self.omega +
                        self.alpha * returns[t-1]**2 +
                        self.beta * sigma2[t-1])

        # Standardized residuals
        residuals = returns / (np.sqrt(sigma2) + 1e-10)

        return sigma2, residuals

    def fit(self, returns: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """
        Fit GARCH-LSTM model.

        Args:
            returns: Return series (in decimal, not percentage)
            epochs: Training epochs
            lr: Learning rate
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, using simple GARCH")
            self._fit_garch_only(returns)
            return

        # Step 1: Fit GARCH parameters
        self._fit_garch_only(returns)

        # Step 2: Get GARCH outputs
        sigma2, residuals = self._garch_filter(returns)

        # Step 3: Prepare LSTM training data
        X, y = self._prepare_lstm_data(returns, sigma2, residuals)

        if len(X) < 100:
            logger.warning("Not enough data for LSTM training")
            self.is_fitted = True
            return

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        self.is_fitted = True
        logger.info("GARCH-LSTM model fitted")

    def _fit_garch_only(self, returns: np.ndarray):
        """Fit simple GARCH(1,1) using maximum likelihood approximation."""
        # Estimate initial parameters
        variance = np.var(returns)

        # Grid search for best parameters
        best_loss = float('inf')
        best_params = (self.omega, self.alpha, self.beta)

        for alpha in [0.01, 0.05, 0.1, 0.15]:
            for beta in [0.8, 0.85, 0.9, 0.95]:
                omega = variance * (1 - alpha - beta)
                if omega <= 0:
                    continue

                # Calculate log-likelihood proxy
                self.omega, self.alpha, self.beta = omega, alpha, beta
                sigma2, _ = self._garch_filter(returns)

                # Gaussian log-likelihood (simplified)
                ll = -0.5 * np.sum(np.log(sigma2 + 1e-10) + returns**2 / (sigma2 + 1e-10))
                loss = -ll

                if loss < best_loss:
                    best_loss = loss
                    best_params = (omega, alpha, beta)

        self.omega, self.alpha, self.beta = best_params
        logger.info(f"GARCH params: omega={self.omega:.6f}, alpha={self.alpha:.3f}, beta={self.beta:.3f}")

    def _prepare_lstm_data(self, returns: np.ndarray, sigma2: np.ndarray,
                          residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        n = len(returns)
        X, y = [], []

        for t in range(self.lookback, n - 1):
            # Features: [returns, garch_vol, residuals, lagged_realized_vol]
            features = np.column_stack([
                returns[t-self.lookback:t],
                np.sqrt(sigma2[t-self.lookback:t]),
                residuals[t-self.lookback:t],
                pd.Series(returns).rolling(10).std().values[t-self.lookback:t]
            ])
            X.append(features)

            # Target: next period realized vol
            y.append(np.abs(returns[t+1]))

        return np.array(X), np.array(y).reshape(-1, 1)

    def predict(self, returns: np.ndarray) -> VolatilityPrediction:
        """
        Predict volatility.

        Args:
            returns: Recent return series (at least lookback periods)

        Returns:
            VolatilityPrediction with forecasts
        """
        if len(returns) < self.lookback:
            # Not enough data
            vol = np.std(returns) if len(returns) > 1 else 0.01
            return VolatilityPrediction(vol, vol, vol, vol, 1, 0.5)

        # GARCH baseline
        sigma2, residuals = self._garch_filter(returns)
        current_vol = np.sqrt(sigma2[-1])

        # LSTM adjustment
        lstm_adj = 0.0
        if HAS_TORCH and self.model is not None and self.is_fitted:
            try:
                features = np.column_stack([
                    returns[-self.lookback:],
                    np.sqrt(sigma2[-self.lookback:]),
                    residuals[-self.lookback:],
                    pd.Series(returns).rolling(10).std().values[-self.lookback:]
                ])

                self.model.eval()
                with torch.no_grad():
                    X = torch.FloatTensor(features).unsqueeze(0)
                    lstm_adj = self.model(X).item()
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e}")

        # Combined prediction
        forecast_1 = max(0.0001, current_vol + lstm_adj)

        # Multi-step forecasts using GARCH mean reversion
        long_run_vol = np.sqrt(self.omega / (1 - self.alpha - self.beta + 1e-10))
        forecast_5 = current_vol * 0.8 + long_run_vol * 0.2
        forecast_10 = current_vol * 0.6 + long_run_vol * 0.4

        # Regime classification
        vol_percentile = (current_vol - long_run_vol) / (long_run_vol + 1e-10)
        if vol_percentile < -0.3:
            regime = 0  # Low vol
        elif vol_percentile > 0.5:
            regime = 2  # High vol
        else:
            regime = 1  # Normal

        # Confidence based on residual normality
        recent_residuals = residuals[-20:] if len(residuals) >= 20 else residuals
        normality = 1 - min(1, np.abs(np.mean(recent_residuals)))

        return VolatilityPrediction(
            current_vol=current_vol,
            forecast_1=forecast_1,
            forecast_5=forecast_5,
            forecast_10=forecast_10,
            regime=regime,
            confidence=normality
        )


class GRULSTMHybrid:
    """
    GRU-LSTM Hybrid for price direction prediction.

    Source: "GRU-LSTM Hybrid Network" (ScienceDirect 2020)
    Validated on EUR/USD, GBP/USD, USD/CAD, USD/CHF

    Architecture:
    1. GRU: Fast feature extraction (cheaper than LSTM)
    2. LSTM: Capture long-term dependencies
    3. Dense: Final prediction

    GRU is ~30% faster than LSTM with similar performance.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 64,
                 gru_layers: int = 1, lstm_layers: int = 1,
                 lookback: int = 50, dropout: float = 0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.model = None
        self.is_fitted = False
        self.feature_names = []

        if HAS_TORCH:
            self._build_model(gru_layers, lstm_layers, dropout)

    def _build_model(self, gru_layers, lstm_layers, dropout):
        """Build GRU-LSTM hybrid model."""
        if not HAS_TORCH:
            return

        class GRULSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, gru_layers, lstm_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=gru_layers,
                    dropout=dropout if gru_layers > 1 else 0,
                    batch_first=True
                )
                self.lstm = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=lstm_layers,
                    dropout=dropout if lstm_layers > 1 else 0,
                    batch_first=True
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                gru_out, _ = self.gru(x)
                lstm_out, _ = self.lstm(gru_out)
                out = self.fc(lstm_out[:, -1, :])
                return out

        self.model = GRULSTMModel(self.input_size, self.hidden_size,
                                  gru_layers, lstm_layers, dropout)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for GRU-LSTM.

        Standard features for forex:
        - Returns at multiple lags
        - Volatility
        - Momentum
        - Mean reversion signals
        """
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        features = pd.DataFrame(index=df.index)

        # Returns
        for lag in [1, 5, 10, 20]:
            features[f'ret_{lag}'] = pd.Series(mid).pct_change(lag).values

        # Volatility
        features['vol_10'] = pd.Series(mid).pct_change().rolling(10).std().values
        features['vol_20'] = pd.Series(mid).pct_change().rolling(20).std().values

        # Momentum
        features['mom_10'] = pd.Series(mid).pct_change(10).values
        features['rsi_14'] = self._compute_rsi(mid, 14)

        # Mean reversion
        features['zscore_20'] = ((mid - pd.Series(mid).rolling(20).mean()) /
                                 (pd.Series(mid).rolling(20).std() + 1e-10)).values

        # Time features
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            features['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)

        self.feature_names = list(features.columns)
        return features.fillna(0).values

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI."""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def fit(self, df: pd.DataFrame, target_col: str = 'target_direction_5',
           epochs: int = 50, lr: float = 0.001, batch_size: int = 64):
        """
        Fit GRU-LSTM model.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            epochs: Training epochs
            lr: Learning rate
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        # Prepare features
        X = self.prepare_features(df)
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            y_seq.append(y[i])

        X_tensor = torch.FloatTensor(np.array(X_seq))
        y_tensor = torch.FloatTensor(np.array(y_seq)).reshape(-1, 1)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        n_batches = len(X_tensor) // batch_size

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size

                optimizer.zero_grad()
                pred = self.model(X_tensor[start:end])
                loss = criterion(pred, y_tensor[start:end])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/n_batches:.4f}")

        self.is_fitted = True
        logger.info("GRU-LSTM model fitted")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict direction probabilities."""
        if not HAS_TORCH or not self.is_fitted:
            return np.full(len(df), 0.5)

        X = self.prepare_features(df)

        if len(X) < self.lookback:
            return np.full(len(df), 0.5)

        predictions = np.full(len(df), 0.5)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.lookback, len(X)):
                seq = torch.FloatTensor(X[i-self.lookback:i]).unsqueeze(0)
                pred = self.model(seq).item()
                predictions[i] = pred

        return predictions


class DeepVolatilityFeatures:
    """
    Generate deep learning volatility features for HFT.
    """

    def __init__(self):
        self.garch_lstm = GARCHLSTMHybrid()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit volatility models."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2
        returns = np.diff(np.log(mid))

        self.garch_lstm.fit(returns, epochs=50)
        self.is_fitted = True

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute deep volatility features."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2
        returns = np.diff(np.log(mid), prepend=0)

        features = pd.DataFrame(index=df.index)

        # Get predictions
        pred = self.garch_lstm.predict(returns)

        features['garch_lstm_vol'] = pred.current_vol
        features['garch_lstm_forecast_1'] = pred.forecast_1
        features['garch_lstm_forecast_5'] = pred.forecast_5
        features['garch_lstm_regime'] = pred.regime
        features['garch_lstm_confidence'] = pred.confidence

        # Vol surprise (realized vs forecast)
        realized_vol = pd.Series(returns).rolling(10).std().values
        features['vol_surprise'] = realized_vol - pred.forecast_1

        return features
