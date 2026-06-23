"""Smoke tests that run without mediapipe/opencv or real data.

Validates the landmark spec math, dataset collation, and that both models
do a forward pass with a padding mask and produce correct logit shapes.
"""

import numpy as np
import torch

from src.dataset import collate
from src.landmarks import LandmarkSpec, normalize_sequence
from src.models import build_model


def test_feature_dim_default():
    spec = LandmarkSpec()
    # pose(33*3) + face(468*3) + lh(21*3) + rh(21*3) = 99 + 1404 + 63 + 63
    assert spec.feature_dim == 1629


def test_feature_dim_hands_only():
    spec = LandmarkSpec(use_pose=False, use_face=False)
    assert spec.feature_dim == 126


def test_normalize_runs():
    spec = LandmarkSpec()
    seq = np.random.rand(10, spec.feature_dim).astype(np.float32)
    out = normalize_sequence(seq, spec)
    assert out.shape == seq.shape


def test_collate_pads():
    a = (torch.randn(5, 8), 5, 0)
    b = (torch.randn(3, 8), 3, 1)
    x, lengths, mask, y = collate([a, b])
    assert x.shape == (2, 5, 8)
    assert mask[1, 3:].all() and not mask[0].any()
    assert y.tolist() == [0, 1]


def _cfg(mtype):
    return {
        "model": {
            "type": mtype, "hidden_dim": 32, "num_layers": 2,
            "num_heads": 4, "dropout": 0.1, "bidirectional": True,
        }
    }


def test_lstm_forward():
    model = build_model(_cfg("lstm"), feature_dim=1629, num_classes=10)
    x = torch.randn(4, 16, 1629)
    mask = torch.zeros(4, 16, dtype=torch.bool)
    mask[0, 10:] = True
    assert model(x, mask).shape == (4, 10)


def test_gru_forward():
    model = build_model(_cfg("gru"), feature_dim=1629, num_classes=10)
    x = torch.randn(4, 16, 1629)
    mask = torch.zeros(4, 16, dtype=torch.bool)
    mask[0, 10:] = True
    assert model(x, mask).shape == (4, 10)


def test_transformer_forward():
    model = build_model(_cfg("transformer"), feature_dim=1629, num_classes=10)
    x = torch.randn(4, 16, 1629)
    mask = torch.zeros(4, 16, dtype=torch.bool)
    mask[0, 10:] = True
    assert model(x, mask).shape == (4, 10)
