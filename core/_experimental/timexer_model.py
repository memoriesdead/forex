"""
TimeXer: Time Series with Exogenous Variables
==============================================
Source: NeurIPS 2024 - "TimeXer: Empowering Transformers for Time Series
Forecasting with Exogenous Variables"
Official: https://github.com/thuml/TimeXer

Key Innovation:
- Reconciles endogenous (price) and exogenous (economic) information
- No architectural modifications needed vs vanilla Transformer
- SOTA across 12 real-world benchmarks

For HFT Forex:
- Endogenous: Price ticks, returns, volatility
- Exogenous: Economic calendar, interest rates, DXY, VIX
- Avoid catastrophic losses during news events
- Exploit post-event reversions
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
    logger.warning("PyTorch not available")


@dataclass
class TimeXerConfig:
    """TimeXer model configuration."""
    seq_len: int = 96  # Input sequence length
    pred_len: int = 24  # Prediction horizon
    enc_in: int = 7  # Number of endogenous variables
    exo_in: int = 5  # Number of exogenous variables
    d_model: int = 64  # Model dimension
    n_heads: int = 4  # Attention heads
    e_layers: int = 2  # Encoder layers
    d_ff: int = 256  # Feed-forward dimension
    dropout: float = 0.1
    activation: str = 'gelu'


class TokenEmbedding(nn.Module):
    """Embed time series tokens."""
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        nn.init.kaiming_normal_(self.tokenConv.weight)

    def forward(self, x):
        # x: [B, L, C]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ExogenousEncoder(nn.Module):
    """
    Encode exogenous variables separately.

    Key insight from TimeXer:
    - Exogenous variables need different treatment
    - They inform but don't directly predict
    - Use cross-attention to inject into main stream
    """
    def __init__(self, exo_in: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.exo_embed = nn.Linear(exo_in, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, endo_hidden: torch.Tensor, exo_input: torch.Tensor) -> torch.Tensor:
        """
        Inject exogenous information into endogenous representation.

        Args:
            endo_hidden: [B, L, D] endogenous hidden states
            exo_input: [B, L, E] exogenous variables

        Returns:
            [B, L, D] enhanced hidden states
        """
        exo_embed = self.exo_embed(exo_input)  # [B, L, D]

        # Cross-attention: endo queries exo
        attn_out, _ = self.cross_attn(endo_hidden, exo_embed, exo_embed)

        # Residual + norm
        out = self.norm(endo_hidden + self.dropout(attn_out))
        return out


class TimeXerEncoderLayer(nn.Module):
    """Single TimeXer encoder layer with exogenous integration."""
    def __init__(self, config: TimeXerConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            dropout=config.dropout, batch_first=True
        )
        self.exo_encoder = ExogenousEncoder(
            config.exo_in, config.d_model,
            config.n_heads, config.dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Exogenous integration
        if exo is not None:
            x = self.exo_encoder(x, exo)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TimeXer(nn.Module):
    """
    TimeXer: Transformer for Time Series with Exogenous Variables.

    NeurIPS 2024 - SOTA for forecasting with external information.

    Architecture:
    1. Separate embeddings for endogenous and exogenous
    2. Cross-attention to inject exogenous info
    3. Standard Transformer encoder
    4. Linear projection for prediction
    """
    def __init__(self, config: TimeXerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.endo_embed = TokenEmbedding(config.enc_in, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.seq_len + config.pred_len)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TimeXerEncoderLayer(config) for _ in range(config.e_layers)
        ])

        # Prediction head
        self.projection = nn.Linear(config.d_model, config.enc_in)

    def forward(self, x_endo: torch.Tensor,
                x_exo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_endo: [B, L, C_endo] endogenous time series
            x_exo: [B, L, C_exo] exogenous variables (optional)

        Returns:
            [B, pred_len, C_endo] predictions
        """
        # Embed endogenous
        h = self.endo_embed(x_endo)  # [B, L, D]
        h = self.pos_embed(h)

        # Encode with exogenous integration
        for layer in self.encoder_layers:
            h = layer(h, x_exo)

        # Project to prediction
        out = self.projection(h[:, -self.config.pred_len:, :])

        return out


class TimeXerForex:
    """
    TimeXer adapted for HFT Forex trading.

    Endogenous variables:
    - Mid price returns
    - Bid-ask spread
    - Volume
    - Volatility
    - Momentum indicators

    Exogenous variables:
    - Economic calendar (binary events)
    - Interest rate differentials
    - DXY (dollar index)
    - VIX (volatility index)
    - Hour of day (cyclical)
    """

    def __init__(self, seq_len: int = 100, pred_len: int = 10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.is_fitted = False

        if HAS_TORCH:
            config = TimeXerConfig(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=7,  # price features
                exo_in=5,  # external features
                d_model=64,
                n_heads=4,
                e_layers=2
            )
            self.model = TimeXer(config)

    def prepare_endogenous(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare endogenous features."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        features = np.column_stack([
            pd.Series(mid).pct_change().fillna(0).values * 10000,  # Returns (bps)
            (df['ask'] - df['bid']).values / mid * 10000,  # Spread (bps)
            df['volume'].values / df['volume'].rolling(20).mean().values if 'volume' in df.columns else np.ones(len(df)),
            pd.Series(mid).pct_change().rolling(20).std().fillna(0.001).values * 10000,  # Volatility
            pd.Series(mid).pct_change(5).fillna(0).values * 10000,  # Momentum 5
            pd.Series(mid).pct_change(20).fillna(0).values * 10000,  # Momentum 20
            ((mid - pd.Series(mid).rolling(20).mean()) / (pd.Series(mid).rolling(20).std() + 1e-10)).fillna(0).values  # Z-score
        ])

        return features

    def prepare_exogenous(self, df: pd.DataFrame,
                         economic_events: Optional[pd.Series] = None,
                         dxy: Optional[pd.Series] = None,
                         vix: Optional[pd.Series] = None) -> np.ndarray:
        """Prepare exogenous features."""
        n = len(df)

        # Time features
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            hour_sin = np.sin(2 * np.pi * ts.dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * ts.dt.hour / 24)
        else:
            hour_sin = np.zeros(n)
            hour_cos = np.zeros(n)

        # Economic events (binary)
        events = economic_events.values if economic_events is not None else np.zeros(n)

        # External indices
        dxy_vals = dxy.pct_change().fillna(0).values * 10000 if dxy is not None else np.zeros(n)
        vix_vals = vix.pct_change().fillna(0).values * 10000 if vix is not None else np.zeros(n)

        return np.column_stack([hour_sin, hour_cos, events, dxy_vals, vix_vals])

    def fit(self, df: pd.DataFrame,
            exo_data: Optional[Dict] = None,
            epochs: int = 50, lr: float = 0.001, batch_size: int = 32):
        """
        Train TimeXer model.

        Args:
            df: DataFrame with OHLCV data
            exo_data: Dict with 'economic_events', 'dxy', 'vix'
            epochs: Training epochs
            lr: Learning rate
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        # Prepare data
        endo = self.prepare_endogenous(df)

        if exo_data:
            exo = self.prepare_exogenous(
                df,
                exo_data.get('economic_events'),
                exo_data.get('dxy'),
                exo_data.get('vix')
            )
        else:
            exo = self.prepare_exogenous(df)

        # Create sequences
        X_endo, X_exo, y = [], [], []
        for i in range(self.seq_len, len(endo) - self.pred_len):
            X_endo.append(endo[i-self.seq_len:i])
            X_exo.append(exo[i-self.seq_len:i])
            y.append(endo[i:i+self.pred_len])

        X_endo = torch.FloatTensor(np.array(X_endo))
        X_exo = torch.FloatTensor(np.array(X_exo))
        y = torch.FloatTensor(np.array(y))

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        n_batches = len(X_endo) // batch_size

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size

                optimizer.zero_grad()
                pred = self.model(X_endo[start:end], X_exo[start:end])
                loss = criterion(pred, y[start:end])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/n_batches:.6f}")

        self.is_fitted = True
        logger.info("TimeXer model trained")

    def predict(self, df: pd.DataFrame,
                exo_data: Optional[Dict] = None) -> np.ndarray:
        """Predict future returns."""
        if not HAS_TORCH or not self.is_fitted:
            return np.zeros(self.pred_len)

        endo = self.prepare_endogenous(df)[-self.seq_len:]

        if exo_data:
            exo = self.prepare_exogenous(
                df.iloc[-self.seq_len:],
                exo_data.get('economic_events'),
                exo_data.get('dxy'),
                exo_data.get('vix')
            )
        else:
            exo = self.prepare_exogenous(df.iloc[-self.seq_len:])

        self.model.eval()
        with torch.no_grad():
            X_endo = torch.FloatTensor(endo).unsqueeze(0)
            X_exo = torch.FloatTensor(exo).unsqueeze(0)
            pred = self.model(X_endo, X_exo)

        return pred[0, :, 0].numpy()  # Return predictions for first variable (returns)

    def get_direction_probability(self, df: pd.DataFrame,
                                  exo_data: Optional[Dict] = None) -> float:
        """Get probability of positive return over prediction horizon."""
        preds = self.predict(df, exo_data)
        cum_return = np.sum(preds)

        # Sigmoid to get probability
        prob = 1 / (1 + np.exp(-cum_return / 10))  # Scaled
        return prob
