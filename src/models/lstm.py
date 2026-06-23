"""BiLSTM sequence classifier over landmark sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, F); mask: (B, T) True=padding. Returns logits (B, C)."""
        h = self.input_proj(x)
        if mask is not None:
            lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # Masked mean pooling over valid timesteps.
            valid = (~mask).unsqueeze(-1).float()
            pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            out, _ = self.lstm(h)
            pooled = out.mean(dim=1)
        return self.head(pooled)
