"""Domain-adversarial graph model for cross-subject EEG emotion recognition.

This module defines the model pieces only. Training code is added separately so
the old DGCNN/SVM paths remain unchanged.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel
from dgcnn import ResidualGraphBlock


class GradientReverseFunction(torch.autograd.Function):
    """Identity in the forward pass, negative scaled gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReverseLayer(nn.Module):
    """Gradient reversal layer used by DANN-style domain adversarial training."""

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return GradientReverseFunction.apply(x, lambda_)


class DomainAdversarialDGCNN(nn.Module):
    """Residual graph encoder with emotion and subject-domain heads."""

    def __init__(
        self,
        n_channels: int = 30,
        n_bands: int = 4,
        n_domains: int = 60,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.35,
        domain_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_domains = n_domains

        self.input_proj = nn.Sequential(
            nn.Linear(n_bands, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualGraphBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        pooled_dim = hidden_dim * 2

        self.emotion_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self.grl = GradientReverseLayer()
        self.domain_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, domain_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(domain_hidden_dim, n_domains),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0):
        features = self.encode(x)
        emotion_logits = self.emotion_head(features)
        domain_logits = self.domain_head(self.grl(features, grl_lambda))
        return emotion_logits, domain_logits

    def predict_emotion(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.emotion_head(features)


class DomainAdversarialDGCNNModel(BaseModel):
    """BaseModel-compatible wrapper for inference and persistence.

    The `fit` method is intentionally not implemented here because domain
    adversarial training needs subject IDs. Stage 2 adds the training loop.
    """

    def __init__(
        self,
        n_channels: int = 30,
        n_bands: int = 4,
        n_domains: int = 60,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.35,
        domain_hidden_dim: int = 64,
    ):
        super().__init__(name="DANN-DGCNN")
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_domains = n_domains
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.domain_hidden_dim = domain_hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DomainAdversarialDGCNN] = None
        self.scaler_mean = None
        self.scaler_std = None

    def _reshape_features(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, self.n_channels, self.n_bands)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = X.mean(axis=0)
            self.scaler_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def build_model(self) -> DomainAdversarialDGCNN:
        self.model = DomainAdversarialDGCNN(
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            n_domains=self.n_domains,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            domain_hidden_dim=self.domain_hidden_dim,
        ).to(self.device)
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, X_test: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError("Use the domain-adversarial training loop with subject IDs.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(self._reshape_features(X), fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model.predict_emotion(X_tensor).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize(self._reshape_features(X), fit=False)
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict_emotion(X_tensor)
            return F.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "n_channels": self.n_channels,
            "n_bands": self.n_bands,
            "n_domains": self.n_domains,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "domain_hidden_dim": self.domain_hidden_dim,
            "name": self.name,
        }, path)

    @classmethod
    def load(cls, path: str) -> "DomainAdversarialDGCNNModel":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            n_channels=checkpoint["n_channels"],
            n_bands=checkpoint["n_bands"],
            n_domains=checkpoint["n_domains"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            domain_hidden_dim=checkpoint.get("domain_hidden_dim", 64),
        )
        model.scaler_mean = checkpoint["scaler_mean"]
        model.scaler_std = checkpoint["scaler_std"]
        model.build_model()
        model.model.load_state_dict(checkpoint["model_state"])
        model.model.eval()
        model.is_fitted = True
        return model
