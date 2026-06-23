"""PyTorch dataset over cached landmark arrays.

Scans <landmark_dir>/<gloss>/<clip>.npy, builds a label vocabulary from the
gloss folder names, and yields (sequence, length, label) samples. A collate
function pads variable-length clips into a batch with a padding mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .landmarks import LandmarkSpec, augment_sequence, normalize_sequence, select_groups


def _fit_length(seq: np.ndarray, max_frames: int) -> np.ndarray:
    """Truncate (uniformly subsample) or return as-is to <= max_frames."""
    t = len(seq)
    if t <= max_frames:
        return seq
    idx = np.linspace(0, t - 1, max_frames).round().astype(int)
    return seq[idx]


class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        label_names: list[str],
        spec: LandmarkSpec,
        max_frames: int = 64,
        min_frames: int = 8,
        normalize: bool = True,
        augment: bool = False,
        aug_cfg: dict | None = None,
        cache: bool = False,
        cache_store: dict | None = None,
    ) -> None:
        self.samples = samples
        self.label_names = label_names
        self.spec = spec
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.normalize = normalize
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.cache = cache
        # Shared across the parent dataset and its subsets, so each clip's
        # deterministic features are loaded/normalized/sliced exactly once.
        self._cache: dict | None = cache_store if cache_store is not None else (
            {} if cache else None
        )
        self._rng = np.random.default_rng()

    def _features(self, path: Path) -> np.ndarray:
        """Deterministic features for one clip: load → normalize → select groups.

        Memoized in the shared RAM cache when caching is enabled. Augmentation
        (random) is applied on top in __getitem__, never cached.
        """
        if self._cache is not None:
            hit = self._cache.get(path)
            if hit is not None:
                return hit
        seq = np.load(path).astype(np.float32)
        if self.normalize:
            seq = normalize_sequence(seq, self.spec)
        seq = select_groups(seq, self.spec)
        if self._cache is not None:
            self._cache[path] = seq
        return seq

    def preload(self) -> int:
        """Eagerly fill the cache; returns total bytes held. No-op if uncached."""
        if self._cache is None:
            return 0
        for path, _ in self.samples:
            self._features(path)
        return sum(a.nbytes for a in self._cache.values())

    def subset(self, indices, augment: bool = False) -> "LandmarkDataset":
        """A view over `indices` with its own augment flag (train vs. val),
        sharing the parent's RAM cache."""
        return LandmarkDataset(
            [self.samples[i] for i in indices],
            self.label_names,
            self.spec,
            max_frames=self.max_frames,
            min_frames=self.min_frames,
            normalize=self.normalize,
            augment=augment,
            aug_cfg=self.aug_cfg,
            cache=self.cache,
            cache_store=self._cache,
        )

    @classmethod
    def discover(
        cls,
        landmark_dir: str | Path,
        spec: LandmarkSpec,
        label_names: list[str] | None = None,
        **kwargs,
    ) -> "LandmarkDataset":
        landmark_dir = Path(landmark_dir)
        files = sorted(landmark_dir.rglob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy landmark files under {landmark_dir}")
        if label_names is None:
            label_names = sorted({f.parent.name for f in files})
        label_to_idx = {name: i for i, name in enumerate(label_names)}
        samples = [
            (f, label_to_idx[f.parent.name])
            for f in files
            if f.parent.name in label_to_idx
        ]
        return cls(samples, label_names, spec, **kwargs)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def save_labels(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.label_names, indent=2))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        # Deterministic part (load → normalize → select groups) is cached;
        # augmentation is applied fresh on top, on the selected layout.
        seq = self._features(path)
        if self.augment and self.aug_cfg.get("enabled", True):
            seq = augment_sequence(seq, self.aug_cfg, self._rng, self.spec)
        seq = _fit_length(seq, self.max_frames)
        if len(seq) < self.min_frames:
            # Pad short clips up to min_frames by repeating the last frame.
            pad = np.repeat(seq[-1:], self.min_frames - len(seq), axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        return torch.from_numpy(np.ascontiguousarray(seq)), len(seq), label


def collate(batch):
    """Pad sequences to the longest in the batch; return (x, lengths, mask, y)."""
    seqs, lengths, labels = zip(*batch)
    max_t = max(lengths)
    feat = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_t, feat, dtype=torch.float32)
    mask = torch.zeros(len(seqs), max_t, dtype=torch.bool)  # True = padding
    for i, (s, t) in enumerate(zip(seqs, lengths)):
        x[i, :t] = s
        mask[i, t:] = True
    return x, torch.tensor(lengths), mask, torch.tensor(labels, dtype=torch.long)
