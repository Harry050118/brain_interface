"""Dynamic Graph Convolutional Neural Network (DGCNN) for EEG emotion recognition.

Based on: Song et al., "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks"
IEEE Transactions on Affective Computing, 2020.

Uses DE features as node inputs and learns dynamic adjacency matrices during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from base_model import BaseModel


class DynamicGraphConv(nn.Module):
    """Dynamic graph convolution layer.

    Learns the adjacency matrix from node features and applies graph convolution.
    A_ij = softmax(LeakyReLU(a * concat(h_i, h_j)))
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: node features, shape (batch, n_nodes, in_features)

        Returns:
            output: shape (batch, n_nodes, out_features)
        """
        batch, n_nodes, _ = x.shape

        # Transform features
        h = self.W(x)  # (batch, n_nodes, out_features)
        h = self.dropout(h)

        # Compute attention-based adjacency
        # Repeat for pairwise concatenation
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # (batch, n, n, d)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # (batch, n, n, d)
        concat = torch.cat([h_i, h_j], dim=-1)  # (batch, n, n, 2d)

        e = torch.matmul(concat, self.a)  # (batch, n, n)
        e = self.leaky_relu(e)
        A = F.softmax(e, dim=-1)  # row-normalized adjacency

        # Graph convolution: A @ h
        out = torch.matmul(A, h)  # (batch, n_nodes, out_features)

        return out


class DGCNN(nn.Module):
    """DGCNN model for EEG emotion recognition.

    Input: DE features per channel, reshaped as (n_nodes=n_channels, node_dim=n_bands)
    """

    def __init__(self, n_channels: int = 30, n_bands: int = 4,
                 hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands

        # Initial projection
        self.input_proj = nn.Linear(n_bands, hidden_dim)

        # Dynamic graph convolution layers
        self.gc_layers = nn.ModuleList([
            DynamicGraphConv(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * n_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: DE features, shape (batch, n_channels, n_bands)

        Returns:
            logits: shape (batch, 2)
        """
        # Project node features
        h = self.input_proj(x)  # (batch, n_channels, hidden_dim)
        h = F.relu(h)

        # Apply graph convolutions
        for gc in self.gc_layers:
            h = gc(h)
            h = F.relu(h)

        # Flatten and classify
        h = h.reshape(h.size(0), -1)  # (batch, n_channels * hidden_dim)
        return self.fc(h)


class DGCNNModel(BaseModel):
    """DGCNN model wrapper with BaseModel interface.

    Uses DE features reshaped as (n_channels, n_bands) per sample.
    """

    def __init__(self, n_channels: int = 30, n_bands: int = 4,
                 hidden_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.5, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 256,
                 weight_decay: float = 1e-4):
        super().__init__(name="DGCNN")
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _reshape_features(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat DE features to (n_samples, n_channels, n_bands)."""
        return X.reshape(-1, self.n_channels, self.n_bands)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Per-channel-per-band normalization."""
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> None:
        """Train DGCNN.

        Args:
            X: DE features, shape (n_samples, n_channels * n_bands)
            y: labels, shape (n_samples,)
            X_test: optional target domain data (unused for now, reserved for DA)
        """
        # Reshape and normalize
        X_reshaped = self._reshape_features(X)
        X_norm = self._normalize(X_reshaped, fit=True)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_norm)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Model
        self.model = DGCNN(
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += len(batch_y)

            scheduler.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X_reshaped = self._reshape_features(X)
        X_norm = self._normalize(X_reshaped, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_reshaped = self._reshape_features(X)
        X_norm = self._normalize(X_reshaped, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def save(self, path: str) -> None:
        """Save DGCNN model weights and scaler stats."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "n_channels": self.n_channels,
            "n_bands": self.n_bands,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "name": self.name,
        }, path)
        print(f"  Saved DGCNN to: {path}")

    @classmethod
    def load(cls, path: str) -> "DGCNNModel":
        """Load DGCNN model weights and scaler stats."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_channels=checkpoint["n_channels"],
            n_bands=checkpoint["n_bands"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )
        model.scaler_mean = checkpoint["scaler_mean"]
        model.scaler_std = checkpoint["scaler_std"]
        model.model = DGCNN(
            n_channels=model.n_channels,
            n_bands=model.n_bands,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout
        ).to(model.device)
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.eval()
        model.is_fitted = True
        return model
