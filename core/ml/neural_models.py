"""
Neural Models for Stacking Ensemble
====================================
Implements:
1. CNN1DClassifier - 1D Convolutional network for feature patterns
2. MLPClassifier - Multi-layer perceptron with batch normalization

Both models are designed to complement gradient boosting methods in stacking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for tabular data."""

    def __init__(
        self,
        input_dim: int,
        conv_layers: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
        fc_layers: List[int] = [256, 128],
    ):
        super().__init__()

        self.input_dim = input_dim

        # Reshape input to 1D sequence
        # Treat each feature as a timestep with 1 channel
        layers = []
        in_channels = 1

        for out_channels in conv_layers:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        fc = []
        fc_input = conv_layers[-1]
        for fc_dim in fc_layers:
            fc.extend([
                nn.Linear(fc_input, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            fc_input = fc_dim

        fc.append(nn.Linear(fc_input, 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        # Reshape to (batch, 1, features)
        x = x.unsqueeze(1)

        # Conv layers
        x = self.conv_layers(x)  # (batch, channels, features)

        # Global average pooling
        x = self.gap(x).squeeze(-1)  # (batch, channels)

        # FC layers
        x = self.fc_layers(x)  # (batch, 1)

        return x


class CNN1DClassifier:
    """
    1D-CNN Classifier wrapper with sklearn-like interface.

    Usage:
        model = CNN1DClassifier(input_dim=400)
        model.fit(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        conv_layers: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.3,
        fc_layers: List[int] = [256, 128],
        max_epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.conv_layers = conv_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.fc_layers = fc_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'CNN1DClassifier':
        """Fit the CNN model."""
        # Create model
        self.model = CNN1D(
            self.input_dim,
            self.conv_layers,
            self.kernel_size,
            self.dropout,
            self.fc_layers,
        ).to(self.device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"CNN early stopping at epoch {epoch}")
                        break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"CNN Epoch {epoch+1}: train_loss={train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            proba = torch.sigmoid(outputs).cpu().numpy().squeeze()

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'conv_layers': self.conv_layers,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'fc_layers': self.fc_layers,
            }
        }, path)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'CNN1DClassifier':
        """Load model from disk."""
        path = Path(path)
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        classifier = cls(
            input_dim=config['input_dim'],
            conv_layers=config['conv_layers'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            fc_layers=config['fc_layers'],
            device=device,
        )

        classifier.model = CNN1D(
            config['input_dim'],
            config['conv_layers'],
            config['kernel_size'],
            config['dropout'],
            config['fc_layers'],
        ).to(device)
        classifier.model.load_state_dict(checkpoint['model_state'])
        classifier.is_fitted = True

        return classifier


class MLP(nn.Module):
    """Multi-Layer Perceptron for tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [512, 256, 128, 64],
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPClassifier:
    """
    MLP Classifier wrapper with sklearn-like interface.

    Usage:
        model = MLPClassifier(input_dim=400)
        model.fit(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [512, 256, 128, 64],
        dropout: float = 0.3,
        batch_norm: bool = True,
        max_epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'MLPClassifier':
        """Fit the MLP model."""
        # Create model
        self.model = MLP(
            self.input_dim,
            self.hidden_layers,
            self.dropout,
            self.batch_norm,
        ).to(self.device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"MLP early stopping at epoch {epoch}")
                        break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"MLP Epoch {epoch+1}: train_loss={train_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            proba = torch.sigmoid(outputs).cpu().numpy().squeeze()

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'batch_norm': self.batch_norm,
            }
        }, path)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'MLPClassifier':
        """Load model from disk."""
        path = Path(path)
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        classifier = cls(
            input_dim=config['input_dim'],
            hidden_layers=config['hidden_layers'],
            dropout=config['dropout'],
            batch_norm=config['batch_norm'],
            device=device,
        )

        classifier.model = MLP(
            config['input_dim'],
            config['hidden_layers'],
            config['dropout'],
            config['batch_norm'],
        ).to(device)
        classifier.model.load_state_dict(checkpoint['model_state'])
        classifier.is_fitted = True

        return classifier


class EnsembleNeuralNet(nn.Module):
    """
    Combined CNN + MLP for richer feature extraction.

    Combines:
    - CNN path: Captures local patterns across features
    - MLP path: Captures global feature interactions
    - Fusion: Combines both paths
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: List[int] = [64, 128],
        mlp_hidden: List[int] = [256, 128],
        fusion_hidden: List[int] = [64],
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN path
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_gap = nn.AdaptiveAvgPool1d(1)

        # MLP path
        mlp_layers = []
        in_dim = input_dim
        for out_dim in mlp_hidden:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Fusion layer
        fusion_input = cnn_channels[-1] + mlp_hidden[-1]
        fusion_layers = []
        for out_dim in fusion_hidden:
            fusion_layers.extend([
                nn.Linear(fusion_input, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            fusion_input = out_dim
        fusion_layers.append(nn.Linear(fusion_input, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN path
        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_out = self.cnn_gap(cnn_out).squeeze(-1)

        # MLP path
        mlp_out = self.mlp(x)

        # Fusion
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fusion(fused)


class EnsembleNeuralClassifier:
    """
    Combined CNN + MLP classifier.

    Usage:
        model = EnsembleNeuralClassifier(input_dim=400)
        model.fit(X_train, y_train, X_val, y_val)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: List[int] = [64, 128],
        mlp_hidden: List[int] = [256, 128],
        fusion_hidden: List[int] = [64],
        dropout: float = 0.3,
        max_epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.mlp_hidden = mlp_hidden
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'EnsembleNeuralClassifier':
        """Fit the ensemble neural model."""
        self.model = EnsembleNeuralNet(
            self.input_dim,
            self.cnn_channels,
            self.mlp_hidden,
            self.fusion_hidden,
            self.dropout,
        ).to(self.device)

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            proba = torch.sigmoid(outputs).cpu().numpy().squeeze()

        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'cnn_channels': self.cnn_channels,
                'mlp_hidden': self.mlp_hidden,
                'fusion_hidden': self.fusion_hidden,
                'dropout': self.dropout,
            }
        }, path)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> 'EnsembleNeuralClassifier':
        """Load model from disk."""
        path = Path(path)
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        classifier = cls(
            input_dim=config['input_dim'],
            cnn_channels=config['cnn_channels'],
            mlp_hidden=config['mlp_hidden'],
            fusion_hidden=config['fusion_hidden'],
            dropout=config['dropout'],
            device=device,
        )

        classifier.model = EnsembleNeuralNet(
            config['input_dim'],
            config['cnn_channels'],
            config['mlp_hidden'],
            config['fusion_hidden'],
            config['dropout'],
        ).to(device)
        classifier.model.load_state_dict(checkpoint['model_state'])
        classifier.is_fitted = True

        return classifier


def create_neural_classifier(
    model_type: str = 'ensemble',
    n_features: int = 400,
    seq_len: int = 20,
    use_gpu: bool = True,
    **kwargs
):
    """
    Factory function to create neural classifiers.

    Args:
        model_type: 'cnn', 'mlp', or 'ensemble'
        n_features: Number of input features
        seq_len: Sequence length for CNN
        use_gpu: Whether to use GPU
        **kwargs: Additional model-specific arguments

    Returns:
        Neural classifier instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    if model_type == 'cnn':
        return CNN1DClassifier(
            n_features=n_features,
            seq_len=seq_len,
            device=device,
            **kwargs
        )
    elif model_type == 'mlp':
        return MLPClassifier(
            n_features=n_features,
            device=device,
            **kwargs
        )
    else:  # ensemble
        return EnsembleNeuralClassifier(
            n_features=n_features,
            seq_len=seq_len,
            device=device,
            **kwargs
        )


# Export
__all__ = [
    'CNN1DClassifier',
    'MLPClassifier',
    'EnsembleNeuralClassifier',
    'CNN1D',
    'MLP',
    'EnsembleNeuralNet',
    'create_neural_classifier',
]
