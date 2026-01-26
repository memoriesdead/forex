"""
Graph Neural Networks for Multi-Pair Forex Forecasting
=======================================================
Source: "A Survey on Graph Neural Networks for Time Series" (arXiv 2024)
MTGNN: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks"

Key Innovation:
- Model correlation structure between currency pairs as graph
- Learn information flow between pairs
- Capture lead-lag relationships
- Cross-pair arbitrage signals

For HFT Forex:
- EUR/USD influences GBP/USD (correlation)
- USD strength affects all USD pairs
- Risk-on/risk-off moves AUD, NZD vs JPY, CHF
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
class GNNConfig:
    """GNN configuration."""
    num_nodes: int = 7  # Number of currency pairs
    input_dim: int = 10  # Input features per node
    hidden_dim: int = 64
    output_dim: int = 1  # Direction prediction
    num_layers: int = 2
    seq_len: int = 100
    pred_len: int = 10
    dropout: float = 0.1


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer.

    Aggregates information from neighboring nodes (correlated pairs).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F] node features
            adj: [N, N] adjacency matrix

        Returns:
            [B, N, F'] transformed features
        """
        # Normalize adjacency
        rowsum = adj.sum(dim=1, keepdim=True) + 1e-6
        adj_norm = adj / rowsum

        # Graph convolution: H' = A * H * W
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution for time series on graph nodes.

    Captures temporal patterns for each node (currency pair).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, T, F] where N=nodes, T=time, F=features

        Returns:
            [B, N, T, F'] temporal features
        """
        B, N, T, F = x.shape

        # Reshape for conv: [B*N, F, T]
        x = x.reshape(B * N, T, F).transpose(1, 2)

        # Convolve
        x = self.conv(x)[:, :, :T]  # Causal: keep original length
        x = self.norm(x)
        x = self.activation(x)

        # Reshape back: [B, N, T, F']
        x = x.transpose(1, 2).reshape(B, N, T, -1)
        return x


class SpatioTemporalBlock(nn.Module):
    """
    Combined spatial (graph) and temporal processing.

    1. Temporal convolution per node
    2. Graph convolution across nodes
    3. Residual connection
    """
    def __init__(self, config: GNNConfig, in_dim: int, out_dim: int):
        super().__init__()

        # Temporal
        self.temporal = TemporalConvolution(in_dim, out_dim)

        # Spatial (graph)
        self.spatial = GraphConvolution(out_dim, out_dim)

        # Skip connection
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, T, F] spatiotemporal tensor
            adj: [N, N] adjacency matrix

        Returns:
            [B, N, T, F'] processed tensor
        """
        residual = x if self.skip is None else self.skip(x)

        # Temporal
        x = self.temporal(x)

        # Spatial for each time step
        B, N, T, F = x.shape
        spatial_out = []
        for t in range(T):
            h = self.spatial(x[:, :, t, :], adj)
            spatial_out.append(h)
        x = torch.stack(spatial_out, dim=2)

        # Residual
        x = self.norm(residual + self.dropout(x))
        return x


class AdaptiveGraphLearner(nn.Module):
    """
    Learn graph structure from data.

    Instead of fixed correlation matrix, learn optimal
    adjacency from node embeddings.
    """
    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self) -> torch.Tensor:
        """Return learned adjacency matrix."""
        # Similarity between node embeddings
        adj = torch.matmul(self.node_embeddings, self.node_embeddings.T)
        adj = F.softmax(adj, dim=1)
        return adj


class MTGNN(nn.Module):
    """
    Multivariate Time Series Graph Neural Network.

    For forex: each node = currency pair
    Edges = correlation/causality between pairs

    Architecture:
    1. Learn graph structure or use correlation matrix
    2. Spatiotemporal blocks: temporal conv + graph conv
    3. Output layer for each node
    """
    def __init__(self, config: GNNConfig, adj_matrix: np.ndarray = None):
        super().__init__()
        self.config = config

        # Graph structure
        if adj_matrix is not None:
            self.register_buffer('adj', torch.FloatTensor(adj_matrix))
            self.adaptive_graph = None
        else:
            self.adaptive_graph = AdaptiveGraphLearner(config.num_nodes)
            self.adj = None

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Spatiotemporal blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(config, config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])

        # Output
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim * config.seq_len, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

    def get_adjacency(self) -> torch.Tensor:
        """Get adjacency matrix (fixed or learned)."""
        if self.adaptive_graph is not None:
            return self.adaptive_graph()
        return self.adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, T, F] multi-pair time series

        Returns:
            [B, N, output_dim] predictions per pair
        """
        adj = self.get_adjacency()

        # Project input
        x = self.input_proj(x)

        # Spatiotemporal processing
        for block in self.blocks:
            x = block(x, adj)

        # Flatten temporal dimension
        B, N, T, F = x.shape
        x = x.reshape(B, N, T * F)

        # Output per node
        outputs = []
        for n in range(N):
            out = self.output(x[:, n, :])
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # [B, N, output_dim]


class GNNForex:
    """
    Graph Neural Network for multi-pair forex trading.

    Pairs as nodes:
    - EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD

    Edges based on:
    - Historical correlation
    - Shared USD exposure
    - Risk-on/risk-off grouping
    """

    PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']

    # Default correlation structure
    DEFAULT_ADJ = np.array([
        # EUR  GBP  JPY  CHF  AUD  NZD  CAD
        [1.0, 0.8, 0.3, 0.5, 0.6, 0.5, 0.4],  # EURUSD
        [0.8, 1.0, 0.4, 0.5, 0.7, 0.6, 0.5],  # GBPUSD
        [0.3, 0.4, 1.0, 0.7, 0.2, 0.2, 0.3],  # USDJPY
        [0.5, 0.5, 0.7, 1.0, 0.3, 0.3, 0.4],  # USDCHF
        [0.6, 0.7, 0.2, 0.3, 1.0, 0.9, 0.6],  # AUDUSD
        [0.5, 0.6, 0.2, 0.3, 0.9, 1.0, 0.5],  # NZDUSD
        [0.4, 0.5, 0.3, 0.4, 0.6, 0.5, 1.0],  # USDCAD
    ])

    def __init__(self, seq_len: int = 100, learn_graph: bool = True):
        self.seq_len = seq_len
        self.model = None
        self.is_fitted = False

        if HAS_TORCH:
            config = GNNConfig(
                num_nodes=len(self.PAIRS),
                input_dim=10,
                hidden_dim=64,
                seq_len=seq_len
            )
            adj = None if learn_graph else self.DEFAULT_ADJ
            self.model = MTGNN(config, adj)

    def prepare_multi_pair_data(self, pair_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Prepare data for all pairs.

        Args:
            pair_data: Dict[pair -> DataFrame with OHLCV]

        Returns:
            [T, N, F] tensor
        """
        all_features = []

        for pair in self.PAIRS:
            if pair in pair_data:
                df = pair_data[pair]
                features = self._prepare_single_pair(df)
            else:
                # Missing pair - use zeros
                features = np.zeros((len(list(pair_data.values())[0]), 10))
            all_features.append(features)

        # Stack: [N, T, F] -> transpose to [T, N, F]
        return np.stack(all_features, axis=0).transpose(1, 0, 2)

    def _prepare_single_pair(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for single pair."""
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        features = np.column_stack([
            pd.Series(mid).pct_change().fillna(0).values * 10000,
            pd.Series(mid).pct_change(5).fillna(0).values * 10000,
            pd.Series(mid).pct_change(20).fillna(0).values * 10000,
            pd.Series(mid).pct_change().rolling(20).std().fillna(0.001).values * 10000,
            ((mid - pd.Series(mid).rolling(20).mean()) / (pd.Series(mid).rolling(20).std() + 1e-10)).fillna(0).values,
            pd.Series(mid).pct_change().rolling(10).mean().fillna(0).values * 10000,
            (df['ask'] - df['bid']).values / mid * 10000 if 'bid' in df.columns else np.zeros(len(df)),
            df['volume'].values / df['volume'].rolling(20).mean().fillna(1).values if 'volume' in df.columns else np.ones(len(df)),
            np.zeros(len(df)),  # Placeholder
            np.zeros(len(df))   # Placeholder
        ])

        return features

    def fit(self, pair_data: Dict[str, pd.DataFrame],
           epochs: int = 50, lr: float = 0.001, batch_size: int = 32):
        """Train GNN on multi-pair data."""
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        data = self.prepare_multi_pair_data(pair_data)  # [T, N, F]
        T, N, F = data.shape

        # Create targets: next period return for each pair
        targets = []
        for pair in self.PAIRS:
            if pair in pair_data:
                mid = pair_data[pair]['mid'].values if 'mid' in pair_data[pair].columns else (pair_data[pair]['bid'] + pair_data[pair]['ask']).values / 2
                ret = pd.Series(mid).pct_change().shift(-1).fillna(0).values * 10000
            else:
                ret = np.zeros(T)
            targets.append(ret)
        targets = np.stack(targets, axis=1)  # [T, N]

        # Create sequences
        X, y = [], []
        for t in range(self.seq_len, T - 1):
            X.append(data[t-self.seq_len:t])  # [seq_len, N, F]
            y.append((targets[t] > 0).astype(float))  # Direction

        X = torch.FloatTensor(np.array(X)).transpose(1, 2)  # [B, N, T, F]
        y = torch.FloatTensor(np.array(y))  # [B, N]

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        n_batches = len(X) // batch_size

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size

                optimizer.zero_grad()
                pred = self.model(X[start:end])
                loss = criterion(pred.squeeze(-1), y[start:end])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/n_batches:.4f}")

        self.is_fitted = True
        logger.info("GNN model trained")

    def predict(self, pair_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Predict direction for all pairs.

        Returns:
            Dict[pair -> probability of up move]
        """
        if not HAS_TORCH or not self.is_fitted:
            return {pair: 0.5 for pair in self.PAIRS}

        data = self.prepare_multi_pair_data(pair_data)[-self.seq_len:]

        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(data).unsqueeze(0).transpose(1, 2)  # [1, N, T, F]
            logits = self.model(X)
            probs = torch.sigmoid(logits).squeeze().numpy()

        return {pair: float(probs[i]) for i, pair in enumerate(self.PAIRS)}

    def get_cross_pair_signals(self, pair_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Get cross-pair arbitrage signals.

        Identifies divergent pairs and mean-reversion opportunities.
        """
        probs = self.predict(pair_data)

        signals = {}
        for pair in self.PAIRS:
            # Find correlated pairs
            corr_pairs = [(p, self.DEFAULT_ADJ[self.PAIRS.index(pair), self.PAIRS.index(p)])
                         for p in self.PAIRS if p != pair and p in probs]
            corr_pairs.sort(key=lambda x: x[1], reverse=True)

            # Check for divergence with top correlated pair
            if corr_pairs:
                top_corr_pair, corr = corr_pairs[0]
                divergence = probs[pair] - probs[top_corr_pair]

                signals[pair] = {
                    'direction_prob': probs[pair],
                    'top_correlated': top_corr_pair,
                    'correlation': corr,
                    'divergence': divergence,
                    'arbitrage_signal': -divergence if abs(divergence) > 0.2 else 0  # Mean reversion
                }

        return signals
