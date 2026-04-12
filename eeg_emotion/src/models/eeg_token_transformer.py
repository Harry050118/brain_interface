"""Channel-token Transformer for raw EEG windows."""

import math

import torch
import torch.nn as nn


class EEGTokenTransformer(nn.Module):
    """Patch raw EEG per channel and classify with a Transformer encoder.

    The token layout is close to biosignal Transformer models: each channel is
    split into temporal patches, then channel and time embeddings are added
    before global self-attention.
    """

    def __init__(
        self,
        n_channels: int = 30,
        window_size: int = 2500,
        patch_size: int = 125,
        patch_stride: int = 125,
        embed_dim: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim

        n_patches = 1 + max(0, (window_size - patch_size) // patch_stride)
        self.n_patches = n_patches

        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            bias=False,
        )
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, n_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.kaiming_uniform_(self.patch_embed.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: raw EEG, shape (batch, channels, time).
        """
        batch_size, n_channels, n_times = x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}")
        if n_times != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {n_times}")

        x = x.reshape(batch_size * n_channels, 1, n_times)
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens.reshape(batch_size, n_channels, self.n_patches, self.embed_dim)
        tokens = tokens + self.channel_embed + self.time_embed
        tokens = tokens.reshape(batch_size, n_channels * self.n_patches, self.embed_dim)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder(tokens)
        cls_out = self.norm(tokens[:, 0])
        return self.head(cls_out)


class EEGFactorizedTransformer(nn.Module):
    """Factorized temporal-then-channel Transformer for raw EEG."""

    def __init__(
        self,
        n_channels: int = 30,
        window_size: int = 2500,
        patch_size: int = 250,
        patch_stride: int = 250,
        embed_dim: int = 48,
        num_heads: int = 4,
        temporal_layers: int = 1,
        channel_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.n_patches = 1 + max(0, (window_size - patch_size) // patch_stride)

        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            bias=False,
        )
        self.time_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, n_channels, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        channel_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.channel_encoder = nn.TransformerEncoder(channel_layer, num_layers=channel_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.kaiming_uniform_(self.patch_embed.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: raw EEG, shape (batch, channels, time).
        """
        batch_size, n_channels, n_times = x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_channels}")
        if n_times != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {n_times}")

        x = x.reshape(batch_size * n_channels, 1, n_times)
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens + self.time_embed
        tokens = self.temporal_encoder(tokens).mean(dim=1)
        tokens = tokens.reshape(batch_size, n_channels, self.embed_dim)
        tokens = tokens + self.channel_embed

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.channel_encoder(tokens)
        cls_out = self.norm(tokens[:, 0])
        return self.head(cls_out)
