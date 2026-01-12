"""
Attention Factors for Statistical Arbitrage
============================================
Source: arXiv:2510.11616 (2025)
"Attention Factors for Statistical Arbitrage"

Key Innovation:
- Jointly learns tradable arbitrage factors from characteristic embeddings
- Uses attention mechanism to combine weak signals
- Achieved Sharpe 4.0+ on U.S. equities (24-year backtest)
- Net of transaction costs: Sharpe 2.3

For HFT Forex:
- Learn factors from price patterns across pairs
- Attention weights reveal which patterns matter
- Statistical arbitrage on correlated pairs
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
class AttentionFactorConfig:
    """Attention Factor model configuration."""
    num_characteristics: int = 20  # Price-based characteristics
    embed_dim: int = 64  # Embedding dimension
    num_heads: int = 4  # Attention heads
    num_layers: int = 2
    num_factors: int = 5  # Learned factors
    dropout: float = 0.1
    seq_len: int = 100


class CharacteristicEmbedding(nn.Module):
    """
    Embed price characteristics into latent space.

    Characteristics (inspired by Fama-French for equities, adapted for forex):
    - Returns at multiple horizons
    - Volatility measures
    - Momentum scores
    - Mean reversion scores
    - Relative strength
    """
    def __init__(self, num_characteristics: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(num_characteristics, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] characteristics

        Returns:
            [B, T, D] embeddings
        """
        return self.norm(self.linear(x))


class FactorAttention(nn.Module):
    """
    Learn factor portfolios via attention.

    Factor loadings = attention weights over characteristics.
    Each factor captures a different pattern.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_factors: int, dropout: float):
        super().__init__()
        self.num_factors = num_factors

        # Factor queries (learnable)
        self.factor_queries = nn.Parameter(torch.randn(num_factors, embed_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Factor projection
        self.factor_proj = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] characteristic embeddings

        Returns:
            (factor_values, attention_weights)
            factor_values: [B, T, num_factors]
            attention_weights: [B, num_factors, T]
        """
        B, T, D = x.shape

        # Expand factor queries for batch
        queries = self.factor_queries.unsqueeze(0).expand(B, -1, -1)  # [B, F, D]

        # Cross-attention: factors attend to characteristics
        factor_embeds, attn_weights = self.attention(queries, x, x)

        # Project to factor values
        factor_values = self.factor_proj(factor_embeds).squeeze(-1)  # [B, F]

        return factor_values, attn_weights


class AttentionFactorModel(nn.Module):
    """
    Attention-based factor model for statistical arbitrage.

    Architecture:
    1. Embed characteristics
    2. Temporal attention for time-varying factors
    3. Factor attention to learn portfolio weights
    4. Output: expected return signal
    """
    def __init__(self, config: AttentionFactorConfig):
        super().__init__()
        self.config = config

        # Characteristic embedding
        self.char_embed = CharacteristicEmbedding(config.num_characteristics, config.embed_dim)

        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)

        # Factor attention
        self.factor_attention = FactorAttention(
            config.embed_dim, config.num_heads,
            config.num_factors, config.dropout
        )

        # Factor to return prediction
        self.return_predictor = nn.Sequential(
            nn.Linear(config.num_factors, config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] characteristics

        Returns:
            (predicted_return, factor_values, attention_weights)
        """
        # Embed
        h = self.char_embed(x)

        # Temporal encoding
        h = self.temporal_encoder(h)

        # Get last hidden state for factors
        h_last = h[:, -1:, :]  # [B, 1, D]

        # Factor extraction
        factor_values, attn_weights = self.factor_attention(h)

        # Predict return
        predicted_return = self.return_predictor(factor_values)

        return predicted_return, factor_values, attn_weights


class AttentionFactorsForex:
    """
    Attention Factors adapted for HFT Forex.

    Characteristics (20 total):
    - Returns: 1, 5, 10, 20, 50 ticks
    - Volatility: 10, 20, 50 windows
    - Momentum: 5, 10, 20, 50 windows
    - Mean reversion: z-scores at 20, 50
    - RSI: 14, 28
    - MACD histogram
    - Spread ratio
    - Volume ratio
    """

    def __init__(self, seq_len: int = 100, num_factors: int = 5):
        self.seq_len = seq_len
        self.num_factors = num_factors
        self.model = None
        self.is_fitted = False
        self.feature_names = []

        if HAS_TORCH:
            config = AttentionFactorConfig(
                num_characteristics=20,
                embed_dim=64,
                num_heads=4,
                num_factors=num_factors,
                seq_len=seq_len
            )
            self.model = AttentionFactorModel(config)

    def prepare_characteristics(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare 20 characteristics from price data."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        chars = {}

        # Returns at multiple horizons
        for lag in [1, 5, 10, 20, 50]:
            chars[f'ret_{lag}'] = pd.Series(mid).pct_change(lag).fillna(0).values * 10000

        # Volatility at multiple windows
        for window in [10, 20, 50]:
            chars[f'vol_{window}'] = pd.Series(mid).pct_change().rolling(window).std().fillna(0.001).values * 10000

        # Momentum (rolling mean of returns)
        for window in [5, 10, 20, 50]:
            chars[f'mom_{window}'] = pd.Series(mid).pct_change().rolling(window).mean().fillna(0).values * 10000

        # Z-scores (mean reversion)
        for window in [20, 50]:
            ma = pd.Series(mid).rolling(window).mean()
            std = pd.Series(mid).rolling(window).std()
            chars[f'zscore_{window}'] = ((mid - ma) / (std + 1e-10)).fillna(0).values

        # RSI
        for period in [14, 28]:
            delta = pd.Series(mid).diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            chars[f'rsi_{period}'] = ((100 - 100 / (1 + gain / (loss + 1e-10))).fillna(50).values - 50) / 50

        # MACD histogram
        ema12 = pd.Series(mid).ewm(span=12).mean()
        ema26 = pd.Series(mid).ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        chars['macd_hist'] = (macd - signal).fillna(0).values * 10000

        # Spread ratio
        if 'bid' in df.columns and 'ask' in df.columns:
            spread = (df['ask'] - df['bid']) / mid * 10000
            spread_ma = spread.rolling(20).mean()
            chars['spread_ratio'] = (spread / (spread_ma + 1e-10)).fillna(1).values - 1
        else:
            chars['spread_ratio'] = np.zeros(len(df))

        # Volume ratio
        if 'volume' in df.columns:
            vol = df['volume']
            vol_ma = vol.rolling(20).mean()
            chars['vol_ratio'] = (vol / (vol_ma + 1e-10)).fillna(1).values - 1
        else:
            chars['vol_ratio'] = np.zeros(len(df))

        self.feature_names = list(chars.keys())
        return np.column_stack(list(chars.values()))

    def fit(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001, batch_size: int = 32):
        """Train attention factor model."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        X = self.prepare_characteristics(df)

        # Target: future return
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2
        returns = pd.Series(mid).pct_change().shift(-1).fillna(0).values * 10000

        # Create sequences
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X) - 1):
            X_seq.append(X[i-self.seq_len:i])
            y_seq.append(returns[i])

        X_tensor = torch.FloatTensor(np.array(X_seq))
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(1)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        n_batches = len(X_tensor) // batch_size

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size

                optimizer.zero_grad()
                pred, factors, attn = self.model(X_tensor[start:end])
                loss = criterion(pred, y_tensor[start:end])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/n_batches:.6f}")

        self.is_fitted = True
        logger.info("Attention Factors model trained")

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict return and get factor exposures.

        Returns:
            Dict with predicted return, factors, and top characteristics
        """
        if not HAS_TORCH or not self.is_fitted:
            return {'predicted_return': 0.0, 'direction_prob': 0.5, 'factors': []}

        X = self.prepare_characteristics(df)[-self.seq_len:]

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0)
            pred, factors, attn_weights = self.model(X_tensor)

        predicted_return = pred.item()
        direction_prob = 1 / (1 + np.exp(-predicted_return))

        # Get factor values
        factor_values = factors[0].numpy().tolist()

        # Get top attention weights (which characteristics matter)
        attn = attn_weights[0].numpy()  # [num_factors, T]
        avg_attn = attn.mean(axis=0)  # Average across factors
        top_indices = np.argsort(avg_attn)[-5:][::-1]
        top_chars = [(self.feature_names[i], float(avg_attn[i])) for i in top_indices]

        return {
            'predicted_return': predicted_return,
            'direction_prob': direction_prob,
            'factors': factor_values,
            'top_characteristics': top_chars,
            'signal': 1 if direction_prob > 0.55 else (-1 if direction_prob < 0.45 else 0)
        }

    def get_factor_loadings(self) -> Dict[str, np.ndarray]:
        """Get factor loadings (for interpretability)."""
        if not self.is_fitted:
            return {}

        # Extract factor query weights
        queries = self.model.factor_attention.factor_queries.detach().numpy()

        return {
            f'factor_{i}': queries[i]
            for i in range(self.num_factors)
        }
