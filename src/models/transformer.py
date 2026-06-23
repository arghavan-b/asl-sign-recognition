"""Transformer-encoder sequence classifier over landmark sequences."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, F); mask: (B, T) True=padding. Returns logits (B, C)."""
        b = x.size(0)
        h = self.input_proj(x)
        cls = self.cls.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos(h)
        if mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
            key_padding = torch.cat([cls_mask, mask], dim=1)  # (B, T+1)
        else:
            key_padding = None
        h = self.encoder(h, src_key_padding_mask=key_padding)
        return self.head(h[:, 0])  # CLS token
