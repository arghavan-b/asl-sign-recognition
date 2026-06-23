"""Model factory."""

from __future__ import annotations

from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier


def build_model(cfg: dict, feature_dim: int, num_classes: int):
    """Construct a model from the `model` block of the config."""
    m = cfg["model"]
    mtype = m.get("type", "gru").lower()
    if mtype in ("lstm", "gru"):
        cls = LSTMClassifier if mtype == "lstm" else GRUClassifier
        return cls(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=m.get("hidden_dim", 256),
            num_layers=m.get("num_layers", 2),
            dropout=m.get("dropout", 0.3),
            bidirectional=m.get("bidirectional", True),
        )
    if mtype == "transformer":
        return TransformerClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=m.get("hidden_dim", 256),
            num_layers=m.get("num_layers", 4),
            num_heads=m.get("num_heads", 8),
            dropout=m.get("dropout", 0.3),
        )
    raise ValueError(
        f"Unknown model type: {mtype!r} (expected 'gru', 'lstm', or 'transformer')"
    )


__all__ = ["build_model", "GRUClassifier", "LSTMClassifier", "TransformerClassifier"]
