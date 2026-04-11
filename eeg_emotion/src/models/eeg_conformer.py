"""EEG Conformer for emotion recognition.

Combines CNN (spatial feature extraction) with Transformer (temporal modeling).
Works directly on raw EEG windows: (n_channels, window_size).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from base_model import BaseModel


class SpatialConv(nn.Module):
    """Spatial convolution for EEG channels.

    Uses (n_channels, 1) kernel to learn spatial patterns across channels,
    followed by temporal pooling.
    """

    def __init__(self, n_channels: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, (n_channels, 1), bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, time)
        return self.conv(x)  # (batch, hidden_dim, 1, time)


class ConformerBlock(nn.Module):
    """Single Conformer block: MHSA + FFN with residual connections."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Feed-forward
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class EEGConformer(nn.Module):
    """EEG Conformer model.

    Architecture:
    1. Spatial conv to learn channel-level patterns
    2. Time pooling to reduce sequence length
    3. Transformer blocks for temporal modeling
    4. Global average pooling + classification
    """

    def __init__(self, n_channels: int = 30, window_size: int = 2500,
                 hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 4, dropout: float = 0.3,
                 time_pool: int = 10):
        super().__init__()
        self.time_pool = time_pool

        # Spatial convolution: reduces 30 channels to hidden_dim
        self.spatial_conv = SpatialConv(n_channels, hidden_dim)

        # After conv: (batch, hidden_dim, 1, window_size)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

        self.pooled_time = window_size // time_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: raw EEG, shape (batch, n_channels, window_size)

        Returns:
            logits: shape (batch, 2)
        """
        # Add channel dim for Conv2d: (batch, 1, channels, time)
        x = x.unsqueeze(1)

        # Spatial convolution: (batch, hidden_dim, 1, window_size)
        x = self.spatial_conv(x)

        # Squeeze and transpose: (batch, window_size, hidden_dim)
        x = x.squeeze(2).transpose(1, 2)

        # Time pooling via mean: (batch, window_size//pool, hidden_dim)
        bs, seq_len, feat = x.shape
        pool_size = self.time_pool
        n_steps = seq_len // pool_size
        x = x[:, :n_steps * pool_size, :]  # trim if needed
        x = x.reshape(bs, n_steps, pool_size, feat).mean(dim=2)

        # Project
        x = self.input_proj(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, hidden_dim)

        return self.classifier(x)


class EEGConformerModel(BaseModel):
    """EEG Conformer wrapper with BaseModel interface.

    Works on raw EEG windows: (n_samples, n_channels, window_size).
    """

    def __init__(self, n_channels: int = 30, window_size: int = 2500,
                 hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 4, dropout: float = 0.3,
                 learning_rate: float = 0.0005, epochs: int = 100,
                 batch_size: int = 32, weight_decay: float = 1e-4,
                 time_pool: int = 10):
        super().__init__(name="EEG-Conformer")
        self.use_raw = True  # Uses raw EEG, not DE features
        self.n_channels = n_channels
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.time_pool = time_pool

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Global normalization."""
        if fit:
            self.scaler_mean = X.mean(axis=(0, 1), keepdims=True)
            self.scaler_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> None:
        """Train EEG Conformer.

        Args:
            X: raw EEG windows, shape (n_samples, n_channels, window_size)
            y: labels, shape (n_samples,)
        """
        X_norm = self._normalize(X, fit=True)

        X_tensor = torch.FloatTensor(X_norm)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = EEGConformer(
            n_channels=self.n_channels,
            window_size=self.window_size,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            time_pool=self.time_pool
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr,
            total_steps=len(loader) * self.epochs,
            pct_start=0.1
        )
        criterion = nn.CrossEntropyLoss()

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * len(batch_y)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += len(batch_y)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X_norm = self._normalize(X, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_norm = self._normalize(X, fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def save(self, path: str) -> None:
        """Save Conformer model weights and scaler stats."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "n_channels": self.n_channels,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "time_pool": self.time_pool,
            "name": self.name,
        }, path)
        print(f"  Saved EEG-Conformer to: {path}")

    @classmethod
    def load(cls, path: str) -> "EEGConformerModel":
        """Load Conformer model weights and scaler stats."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_channels=checkpoint["n_channels"],
            window_size=checkpoint["window_size"],
            hidden_dim=checkpoint["hidden_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            time_pool=checkpoint["time_pool"],
        )
        model.scaler_mean = checkpoint["scaler_mean"]
        model.scaler_std = checkpoint["scaler_std"]
        model.model = EEGConformer(
            n_channels=model.n_channels,
            window_size=model.window_size,
            hidden_dim=model.hidden_dim,
            num_heads=model.num_heads,
            num_layers=model.num_layers,
            dropout=model.dropout,
            time_pool=model.time_pool,
        ).to(model.device)
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.eval()
        model.is_fitted = True
        return model
