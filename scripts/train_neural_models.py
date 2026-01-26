#!/usr/bin/env python3
"""
Train Neural Models - Time-Series-Library Integration
======================================================
Trains iTransformer, DLinear, and other neural architectures
from the Time-Series-Library for forex prediction.

Available Models:
- iTransformer: Best for multivariate OHLCV (ICLR 2024)
- DLinear/NLinear: Fast production models
- Autoformer: Decomposition-based
- Informer: Sparse attention
- FEDformer: Frequency domain

Usage:
    # Train all neural models for all pairs
    python scripts/train_neural_models.py --all

    # Train specific model for specific pairs
    python scripts/train_neural_models.py --pairs EURUSD,GBPUSD --model itransformer

    # Evaluate trained models
    python scripts/train_neural_models.py --evaluate

    # List available models
    python scripts/train_neural_models.py --list
"""

import argparse
import json
import sys
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# Available neural model architectures
AVAILABLE_MODELS = {
    'itransformer': 'iTransformer - Inverted Transformer (ICLR 2024)',
    'dlinear': 'DLinear - Decomposition Linear (fast)',
    'nlinear': 'NLinear - Normalized Linear (fast)',
    'autoformer': 'Autoformer - Auto-Correlation (NeurIPS 2021)',
    'informer': 'Informer - Sparse Attention (AAAI 2021)',
    'fedformer': 'FEDformer - Frequency Enhanced (ICML 2022)',
    'timesnet': 'TimesNet - Temporal 2D (ICLR 2023)',
    'patchtst': 'PatchTST - Patch Transformer (ICLR 2023)',
}


class ForexTimeSeriesDataset(Dataset):
    """Dataset for forex time series prediction."""

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 1,
        target_col: str = 'target_direction_10',
        feature_cols: Optional[List[str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: DataFrame with features and targets
            seq_len: Input sequence length (lookback)
            label_len: Label sequence length (for decoder)
            pred_len: Prediction length
            target_col: Target column name
            feature_cols: Feature columns to use (None = all numeric)
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_col = target_col

        # Get feature columns
        if feature_cols is None:
            # Use all numeric columns except targets
            target_cols = [c for c in data.columns if c.startswith('target_')]
            feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                           if c not in target_cols]

        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)

        # Extract data
        self.features = data[feature_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32) if target_col in data.columns else None

        # Normalize features
        self.mean = np.nanmean(self.features, axis=0)
        self.std = np.nanstd(self.features, axis=0) + 1e-8
        self.features = (self.features - self.mean) / self.std

        # Replace NaNs with 0
        self.features = np.nan_to_num(self.features, nan=0.0)

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.features[s_begin:s_end]
        seq_y = self.features[r_begin:r_end]

        # Target is direction at prediction point
        if self.targets is not None:
            target = self.targets[s_end + self.pred_len - 1]
        else:
            target = 0.0

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting.

    From: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)

    Key insight: Treat each feature as a token instead of each time step.
    This captures multivariate correlations better for OHLCV data.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        enc_in: int = 7,
        num_classes: int = 2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Inverted embedding: embed each feature across time
        self.embedding = nn.Linear(seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(enc_in * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes),
        )

    def forward(self, x, *args, **kwargs):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # Invert: (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # Embed each feature: (batch, features, d_model)
        x = self.embedding(x)

        # Transformer encoder
        x = self.encoder(x)

        # Flatten and classify
        x = x.reshape(batch_size, -1)
        logits = self.classifier(x)

        return logits


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear for Time Series.

    From: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

    Simple but effective: Decompose into trend + seasonal, apply linear layers.
    Very fast for production use.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 1,
        enc_in: int = 7,
        num_classes: int = 2,
        moving_avg: int = 25,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Moving average for decomposition
        self.moving_avg = nn.AvgPool1d(kernel_size=moving_avg, stride=1, padding=moving_avg // 2)

        # Linear layers for trend and seasonal
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

        # Combine features for classification
        self.classifier = nn.Sequential(
            nn.Linear(enc_in * pred_len * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, *args, **kwargs):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # Decompose: trend + seasonal
        # Permute for AvgPool1d: (batch, features, seq_len)
        x_perm = x.permute(0, 2, 1)

        # Get trend (moving average)
        trend = self.moving_avg(x_perm)
        # Adjust size to match original
        if trend.size(-1) != self.seq_len:
            trend = nn.functional.interpolate(trend, size=self.seq_len, mode='linear', align_corners=False)

        seasonal = x_perm - trend

        # Apply linear layers
        trend_out = self.linear_trend(trend)      # (batch, features, pred_len)
        seasonal_out = self.linear_seasonal(seasonal)

        # Combine and classify
        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # (batch, features, pred_len*2)
        combined = combined.reshape(batch_size, -1)
        logits = self.classifier(combined)

        return logits


class NLinear(nn.Module):
    """
    NLinear: Normalized Linear for Time Series.

    Subtracts last value before linear, adds back after.
    Good for non-stationary data like forex.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 1,
        enc_in: int = 7,
        num_classes: int = 2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        # Linear layer per feature
        self.linear = nn.Linear(seq_len, pred_len)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(enc_in * pred_len, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, *args, **kwargs):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # Get last value for normalization
        last = x[:, -1:, :]  # (batch, 1, features)

        # Subtract last value (normalize)
        x_norm = x - last

        # Permute and apply linear: (batch, features, seq_len) -> (batch, features, pred_len)
        x_perm = x_norm.permute(0, 2, 1)
        out = self.linear(x_perm)

        # Flatten and classify
        out = out.reshape(batch_size, -1)
        logits = self.classifier(out)

        return logits


class PatchTST(nn.Module):
    """
    PatchTST: Patch-based Transformer.

    From: "A Time Series is Worth 64 Words" (ICLR 2023)

    Patch time series like images, use channel-independent transformers.
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 1,
        enc_in: int = 7,
        num_classes: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.enc_in = enc_in

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding (per channel)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(enc_in * self.n_patches * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes),
        )

    def forward(self, x, *args, **kwargs):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # Create patches: (batch, n_patches, patch_len, features)
        patches = x.unfold(1, self.patch_len, self.stride)

        # Process each channel independently
        channel_outputs = []
        for i in range(self.enc_in):
            # (batch, n_patches, patch_len)
            channel_patches = patches[:, :, :, i] if i < patches.size(-1) else patches[:, :, :, -1]

            # Embed patches: (batch, n_patches, d_model)
            embedded = self.patch_embedding(channel_patches)
            embedded = embedded + self.pos_encoding

            # Transformer
            encoded = self.encoder(embedded)
            channel_outputs.append(encoded)

        # Concatenate channels and classify
        combined = torch.cat(channel_outputs, dim=-1)  # (batch, n_patches, enc_in * d_model)
        combined = combined.reshape(batch_size, -1)
        logits = self.classifier(combined)

        return logits


class NeuralModelTrainer:
    """Trainer for Time-Series-Library neural models."""

    def __init__(
        self,
        training_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        model_type: str = 'itransformer',
        seq_len: int = 96,
        pred_len: int = 1,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            training_dir: Directory with training data
            output_dir: Directory to save trained models
            model_type: Type of neural model
            seq_len: Input sequence length
            pred_len: Prediction length
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
        """
        self.training_dir = training_dir or PROJECT_ROOT / 'training_package'
        self.output_dir = output_dir or PROJECT_ROOT / 'models' / 'production' / 'neural'
        self.model_type = model_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def get_available_pairs(self) -> List[str]:
        """Get pairs with training data."""
        pairs = []
        for d in self.training_dir.iterdir():
            if d.is_dir() and (d / 'train.parquet').exists():
                if d.name not in ['features_cache', '__pycache__']:
                    pairs.append(d.name)
        return sorted(pairs)

    def create_model(self, n_features: int) -> nn.Module:
        """Create neural model based on type."""
        model_params = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': n_features,
            'num_classes': 2,
        }

        if self.model_type == 'itransformer':
            model = iTransformer(
                **model_params,
                d_model=256,
                n_heads=4,
                e_layers=2,
                d_ff=512,
                dropout=0.1,
            )
        elif self.model_type == 'dlinear':
            model = DLinear(**model_params, moving_avg=25)
        elif self.model_type == 'nlinear':
            model = NLinear(**model_params)
        elif self.model_type == 'patchtst':
            model = PatchTST(
                **model_params,
                patch_len=16,
                stride=8,
                d_model=128,
                n_heads=4,
                e_layers=2,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model.to(self.device)

    def train_pair(self, pair: str, target_col: str = 'target_direction_10') -> Dict:
        """
        Train neural model for a single pair.

        Args:
            pair: Currency pair symbol
            target_col: Target column name

        Returns:
            Training results dict
        """
        logger.info(f"Training {self.model_type} for {pair}...")

        pair_dir = self.training_dir / pair

        # Load data
        train_df = pd.read_parquet(pair_dir / 'train.parquet')
        val_df = pd.read_parquet(pair_dir / 'val.parquet')

        if target_col not in train_df.columns:
            logger.warning(f"  Target {target_col} not found, skipping")
            return {'success': False, 'error': f'Target {target_col} not found'}

        # Create datasets
        train_dataset = ForexTimeSeriesDataset(
            train_df, seq_len=self.seq_len, pred_len=self.pred_len, target_col=target_col
        )
        val_dataset = ForexTimeSeriesDataset(
            val_df, seq_len=self.seq_len, pred_len=self.pred_len, target_col=target_col
        )

        if len(train_dataset) < 100:
            logger.warning(f"  Too few samples for {pair}")
            return {'success': False, 'error': 'Too few samples'}

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Create model
        model = self.create_model(train_dataset.n_features)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training loop
        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for seq_x, seq_y, targets in train_loader:
                seq_x = seq_x.to(self.device)
                targets = targets.long().to(self.device)

                optimizer.zero_grad()
                logits = model(seq_x)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)

            scheduler.step()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            all_preds = []
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for seq_x, seq_y, targets in val_loader:
                    seq_x = seq_x.to(self.device)
                    targets = targets.long().to(self.device)

                    logits = model(seq_x)
                    probs = torch.softmax(logits, dim=1)

                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)

                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Calculate final metrics
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5

        # Save model
        model.load_state_dict(best_state)
        model_path = self.output_dir / f"{pair}_{self.model_type}.pt"
        torch.save({
            'model_state': best_state,
            'model_type': self.model_type,
            'n_features': train_dataset.n_features,
            'feature_cols': train_dataset.feature_cols,
            'mean': train_dataset.mean,
            'std': train_dataset.std,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
        }, model_path)

        results = {
            'success': True,
            'pair': pair,
            'model_type': self.model_type,
            'accuracy': best_val_acc,
            'auc': auc,
            'epochs': self.epochs,
            'saved_to': str(model_path),
        }

        logger.info(f"  Completed: accuracy={best_val_acc:.4f}, auc={auc:.4f}")

        return results

    def train_all(self, pairs: Optional[List[str]] = None, target_col: str = 'target_direction_10') -> Dict:
        """Train neural models for all pairs."""
        if pairs is None:
            pairs = self.get_available_pairs()

        results = {
            'success': [],
            'failed': [],
            'metrics': {},
        }

        for i, pair in enumerate(pairs):
            logger.info(f"\n[{i+1}/{len(pairs)}] {pair}")

            try:
                pair_results = self.train_pair(pair, target_col=target_col)

                if pair_results['success']:
                    results['success'].append(pair)
                    results['metrics'][pair] = pair_results
                else:
                    results['failed'].append(pair)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                results['failed'].append(pair)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save summary
        summary_path = self.output_dir / f'{self.model_type}_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nNeural Training Complete")
        logger.info(f"  Model: {self.model_type}")
        logger.info(f"  Success: {len(results['success'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"  Results saved to: {summary_path}")

        return results

    def evaluate(self, pairs: Optional[List[str]] = None) -> Dict:
        """Evaluate trained neural models."""
        if pairs is None:
            pairs = self.get_available_pairs()

        results = {}

        for pair in pairs:
            model_path = self.output_dir / f"{pair}_{self.model_type}.pt"
            if not model_path.exists():
                continue

            logger.info(f"Evaluating {pair}...")

            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            model = self.create_model(checkpoint['n_features'])
            model.load_state_dict(checkpoint['model_state'])
            model.eval()

            # Load test data
            pair_dir = self.training_dir / pair
            test_df = pd.read_parquet(pair_dir / 'test.parquet')

            # Create dataset with saved normalization
            test_dataset = ForexTimeSeriesDataset(
                test_df, seq_len=self.seq_len, pred_len=self.pred_len,
                feature_cols=checkpoint['feature_cols']
            )
            test_dataset.mean = checkpoint['mean']
            test_dataset.std = checkpoint['std']
            test_dataset.features = (test_df[checkpoint['feature_cols']].values - checkpoint['mean']) / checkpoint['std']
            test_dataset.features = np.nan_to_num(test_dataset.features, nan=0.0)

            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # Evaluate
            all_preds = []
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for seq_x, seq_y, targets in test_loader:
                    seq_x = seq_x.to(self.device)
                    logits = model(seq_x)
                    probs = torch.softmax(logits, dim=1)

                    _, predicted = torch.max(logits, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_targets.extend(targets.numpy())

            accuracy = accuracy_score(all_targets, all_preds)
            auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5

            results[pair] = {
                'accuracy': accuracy,
                'auc': auc,
                'samples': len(all_targets),
            }

            logger.info(f"  {pair}: accuracy={accuracy:.4f}, auc={auc:.4f}")

        return results


def list_models():
    """Print available neural models."""
    print("\nAvailable Neural Models:")
    print("=" * 60)
    for name, desc in AVAILABLE_MODELS.items():
        print(f"  {name:15} - {desc}")
    print("=" * 60)
    print("\nRecommendations:")
    print("  • itransformer: Best accuracy for multivariate OHLCV")
    print("  • dlinear/nlinear: Fastest for production")
    print("  • patchtst: Good balance of speed and accuracy")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Neural Models for Forex Prediction'
    )

    parser.add_argument('--all', action='store_true', help='Train for all pairs')
    parser.add_argument('--pairs', type=str, help='Comma-separated pairs')
    parser.add_argument('--model', type=str, default='itransformer',
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Neural model type')
    parser.add_argument('--target', type=str, default='target_direction_10',
                       help='Target column')
    parser.add_argument('--seq-len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    parser.add_argument('--list', action='store_true',
                       help='List available models')

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    trainer = NeuralModelTrainer(
        model_type=args.model,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if args.all:
        trainer.train_all(target_col=args.target)

    elif args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
        trainer.train_all(pairs=pairs, target_col=args.target)

    elif args.evaluate:
        trainer.evaluate()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
