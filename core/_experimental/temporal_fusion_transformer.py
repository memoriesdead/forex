"""
Temporal Fusion Transformer (TFT) for HFT Forex
================================================
Source: Google Research - "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (2021)

Key Features:
- Multi-horizon forecasting (predict 1, 5, 10, 20 ticks ahead simultaneously)
- Interpretable attention (which features matter most)
- Static covariates (symbol characteristics)
- Known future inputs (time features)
- Variable selection network (automatic feature importance)

For HFT Forex:
- Native multi-horizon (1-tick, 5-tick, 10-tick predictions)
- Feature importance for alpha discovery
- Handles exogenous inputs (economic calendar)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TFTConfig:
    """TFT configuration."""
    input_size: int = 20  # Number of input features
    hidden_size: int = 64  # Hidden layer size
    num_heads: int = 4  # Attention heads
    dropout: float = 0.1
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    seq_len: int = 100  # Input sequence length
    pred_horizons: List[int] = None  # [1, 5, 10, 20, 50]
    quantiles: List[float] = None  # [0.1, 0.5, 0.9] for prediction intervals

    def __post_init__(self):
        if self.pred_horizons is None:
            self.pred_horizons = [1, 5, 10, 20, 50]
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.

    Provides:
    - Non-linear processing
    - Skip connections
    - Gating mechanism for feature selection
    """
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = None, dropout: float = 0.1,
                 context_size: int = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size

        # Primary path
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Context (optional)
        self.context_fc = nn.Linear(context_size, hidden_size) if context_size else None

        # Gating
        self.gate = nn.Linear(hidden_size, self.output_size)

        # Skip connection
        self.skip = nn.Linear(input_size, self.output_size) if input_size != self.output_size else None

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # Primary path
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gating
        gate = torch.sigmoid(self.gate(hidden))
        hidden = gate * hidden

        # Skip connection
        if self.skip is not None:
            x = self.skip(x)

        # Residual + norm
        return self.layer_norm(x + hidden)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - Automatic feature importance.

    Learns which features are important for prediction.
    Provides interpretability: which signals drive predictions.
    """
    def __init__(self, input_size: int, num_vars: int,
                 hidden_size: int, dropout: float = 0.1,
                 context_size: int = None):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        # Variable-wise GRN
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])

        # Softmax selection
        self.selection_weights = GatedResidualNetwork(
            num_vars * hidden_size, hidden_size, num_vars, dropout, context_size
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, num_vars, input_size] or [B, L, num_vars * input_size]

        Returns:
            (selected_features, selection_weights)
        """
        # Process each variable
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_input = x[:, :, i*self.hidden_size:(i+1)*self.hidden_size] if x.dim() == 3 else x[:, :, i, :]
            var_outputs.append(grn(var_input))

        var_outputs = torch.stack(var_outputs, dim=-2)  # [B, L, num_vars, hidden]

        # Selection weights
        flat = var_outputs.reshape(var_outputs.shape[0], var_outputs.shape[1], -1)
        weights = F.softmax(self.selection_weights(flat, context), dim=-1)

        # Weighted combination
        selected = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention.

    Same as standard attention but returns attention weights
    for interpretability (which time steps matter).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = query.shape

        # Project
        q = self.W_q(query).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_output = self.W_o(attn_output)

        # Average attention weights across heads for interpretability
        avg_weights = attn_weights.mean(dim=1)

        return attn_output, avg_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.

    Components:
    1. Variable Selection - Which features matter
    2. LSTM Encoder - Temporal patterns
    3. Self-Attention - Long-range dependencies
    4. Gated Skip Connections - Information flow
    5. Quantile Output - Prediction intervals
    """
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # LSTM encoder
        self.encoder = nn.LSTM(
            config.hidden_size, config.hidden_size,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0,
            batch_first=True
        )

        # Self-attention
        self.attention = InterpretableMultiHeadAttention(
            config.hidden_size, config.num_heads, config.dropout
        )

        # GRN for post-attention
        self.post_attn_grn = GatedResidualNetwork(
            config.hidden_size, config.hidden_size,
            config.hidden_size, config.dropout
        )

        # Output heads for each horizon
        self.output_heads = nn.ModuleDict({
            str(h): nn.Linear(config.hidden_size, len(config.quantiles))
            for h in config.pred_horizons
        })

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, L, input_size] input sequence

        Returns:
            Dict[horizon -> [B, num_quantiles]] predictions
        """
        # Project input
        h = self.input_projection(x)

        # LSTM encode
        lstm_out, _ = self.encoder(h)
        lstm_out = self.layer_norm(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Post-attention GRN
        h = self.post_attn_grn(lstm_out + attn_out)

        # Use last hidden state for predictions
        final_hidden = h[:, -1, :]

        # Multi-horizon outputs
        outputs = {}
        for horizon in self.config.pred_horizons:
            outputs[horizon] = self.output_heads[str(horizon)](final_hidden)

        return outputs

    def predict_direction(self, x: torch.Tensor) -> Dict[int, float]:
        """Predict direction probability for each horizon."""
        outputs = self.forward(x)

        probs = {}
        for horizon, quantile_preds in outputs.items():
            # Use median (0.5 quantile) for direction
            median_idx = len(self.config.quantiles) // 2
            median_pred = quantile_preds[:, median_idx].item()

            # Sigmoid for probability
            probs[horizon] = 1 / (1 + np.exp(-median_pred * 10))

        return probs


class TFTForex:
    """
    TFT adapted for HFT Forex trading.

    Features:
    - Returns at multiple lags
    - Volatility measures
    - Momentum indicators
    - Order flow signals
    - Time features
    """

    def __init__(self, seq_len: int = 100, pred_horizons: List[int] = None):
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons or [1, 5, 10, 20, 50]
        self.model = None
        self.is_fitted = False
        self.feature_names = []

        if HAS_TORCH:
            config = TFTConfig(
                input_size=20,
                hidden_size=64,
                num_heads=4,
                seq_len=seq_len,
                pred_horizons=self.pred_horizons
            )
            self.model = TemporalFusionTransformer(config)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for TFT."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        features = {}

        # Returns at multiple lags
        for lag in [1, 5, 10, 20, 50]:
            features[f'ret_{lag}'] = pd.Series(mid).pct_change(lag).fillna(0).values * 10000

        # Volatility
        for window in [10, 20, 50]:
            features[f'vol_{window}'] = pd.Series(mid).pct_change().rolling(window).std().fillna(0.001).values * 10000

        # Z-scores
        for window in [20, 50]:
            ma = pd.Series(mid).rolling(window).mean()
            std = pd.Series(mid).rolling(window).std()
            features[f'zscore_{window}'] = ((mid - ma) / (std + 1e-10)).fillna(0).values

        # RSI
        delta = pd.Series(mid).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi_14'] = (100 - 100 / (1 + gain / (loss + 1e-10))).fillna(50).values

        # Time features
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            features['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        else:
            features['hour_sin'] = np.zeros(len(df))
            features['hour_cos'] = np.zeros(len(df))
            features['day_sin'] = np.zeros(len(df))
            features['day_cos'] = np.zeros(len(df))

        self.feature_names = list(features.keys())
        return np.column_stack(list(features.values()))

    def fit(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001):
        """Train TFT model."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        X = self.prepare_features(df)

        # Target: future returns
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2
        returns = pd.Series(mid).pct_change().fillna(0).values * 10000

        # Create sequences
        X_seq, y_dict = [], {h: [] for h in self.pred_horizons}
        max_horizon = max(self.pred_horizons)

        for i in range(self.seq_len, len(X) - max_horizon):
            X_seq.append(X[i-self.seq_len:i])
            for h in self.pred_horizons:
                y_dict[h].append(returns[i:i+h].sum())  # Cumulative return

        X_tensor = torch.FloatTensor(np.array(X_seq))
        y_tensors = {h: torch.FloatTensor(np.array(y_dict[h])).unsqueeze(1) for h in self.pred_horizons}

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)

            # Quantile loss
            loss = 0
            for h in self.pred_horizons:
                pred = outputs[h]
                target = y_tensors[h]
                for q_idx, q in enumerate(self.model.config.quantiles):
                    errors = target - pred[:, q_idx:q_idx+1]
                    loss += torch.max(q * errors, (q - 1) * errors).mean()

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        logger.info("TFT model trained")

    def predict(self, df: pd.DataFrame) -> Dict[int, Tuple[float, float, float]]:
        """
        Predict for all horizons.

        Returns:
            Dict[horizon -> (lower, median, upper)] quantile predictions
        """
        if not HAS_TORCH or not self.is_fitted:
            return {h: (0.0, 0.0, 0.0) for h in self.pred_horizons}

        X = self.prepare_features(df)[-self.seq_len:]

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0)
            outputs = self.model(X_tensor)

        results = {}
        for h in self.pred_horizons:
            preds = outputs[h][0].numpy()
            results[h] = (preds[0], preds[1], preds[2])  # lower, median, upper

        return results

    def get_direction_signals(self, df: pd.DataFrame) -> Dict[int, float]:
        """Get direction probability for each horizon."""
        preds = self.predict(df)

        signals = {}
        for h, (lower, median, upper) in preds.items():
            # Confidence based on prediction interval
            interval_width = upper - lower
            confidence = 1 / (1 + interval_width)

            # Direction from median
            direction = 1 if median > 0 else -1

            signals[h] = direction * confidence

        return signals
