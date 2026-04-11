"""Lightweight Dynamic Graph CNN for EEG emotion recognition.

The first version was intentionally close to a simple graph-attention layer, but
it was too large for this small cross-subject dataset. This version uses residual
connections, LayerNorm, graph-level pooling, and smaller default capacity.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from base_model import BaseModel


class DynamicGraphConv(nn.Module):
    """Attention-based dynamic graph convolution with normalization."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.35):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.empty(2 * out_features))
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        n_nodes = h.size(1)

        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        e = torch.matmul(torch.cat([h_i, h_j], dim=-1), self.attn)
        adjacency = F.softmax(self.leaky_relu(e), dim=-1)

        out = torch.matmul(adjacency, h)
        out = self.dropout(out)
        return self.norm(out)


class ResidualGraphBlock(nn.Module):
    """Dynamic graph layer plus residual connection."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.graph = DynamicGraphConv(hidden_dim, hidden_dim, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.graph(x)
        x = x + self.ffn(x)
        return x


class DGCNN(nn.Module):
    """Small residual DGCNN over channel-wise band features."""

    def __init__(self, n_channels: int = 30, n_bands: int = 4,
                 hidden_dim: int = 32, num_layers: int = 2,
                 dropout: float = 0.35):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.input_proj = nn.Sequential(
            nn.Linear(n_bands, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualGraphBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return self.classifier(torch.cat([mean_pool, max_pool], dim=-1))


class DGCNNModel(BaseModel):
    """DGCNN wrapper with BaseModel interface."""

    def __init__(self, n_channels: int = 30, n_bands: int = 4,
                 hidden_dim: int = 32, num_layers: int = 2,
                 dropout: float = 0.35, learning_rate: float = 0.0005,
                 epochs: int = 60, batch_size: int = 256,
                 weight_decay: float = 1e-3):
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
        return X.reshape(-1, self.n_channels, self.n_bands)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: Optional[np.ndarray] = None) -> None:
        X_norm = self._normalize(self._reshape_features(X), fit=True)
        X_tensor = torch.FloatTensor(X_norm)
        y_tensor = torch.LongTensor(y)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model = DGCNN(
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.epochs)
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
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

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(self._reshape_features(X), fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(self._reshape_features(X), fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return F.softmax(self.model(X_tensor), dim=1).cpu().numpy()

    def save(self, path: str) -> None:
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
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "name": self.name,
        }, path)
        print(f"  Saved DGCNN to: {path}")

    @classmethod
    def load(cls, path: str) -> "DGCNNModel":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_channels=checkpoint["n_channels"],
            n_bands=checkpoint["n_bands"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            learning_rate=checkpoint.get("learning_rate", 0.0005),
            epochs=checkpoint.get("epochs", 60),
            batch_size=checkpoint.get("batch_size", 256),
            weight_decay=checkpoint.get("weight_decay", 1e-3),
        )
        model.scaler_mean = checkpoint["scaler_mean"]
        model.scaler_std = checkpoint["scaler_std"]
        model.model = DGCNN(
            n_channels=model.n_channels,
            n_bands=model.n_bands,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout,
        ).to(model.device)
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.eval()
        model.is_fitted = True
        return model
