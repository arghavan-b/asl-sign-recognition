"""Training loop for the landmark sign classifier.

Usage:
    python -m src.train --config configs/default.yaml

Splits: if data/splits/{train,val}.txt exist (one gloss-folder-relative .npy
path per line), they are used — build these from DISJOINT signers so val/test
measures generalization to people the model never saw. Otherwise a random
split is created from config fractions (a weaker signal; see README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import LandmarkDataset, collate
from .extract import build_spec
from .models import build_model
from .utils import get_logger, load_config, resolve_device, set_seed

log = get_logger("train")


def _split_indices(n: int, val_frac: float, seed: int):
    """Random two-way train/val split. No test slice is reserved — leaderboard
    or the signer-disjoint split files serve as the held-out test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


def _load_split_files(full, landmark_dir, splits_dir):
    """Return (train_idx, val_idx) from data/splits/{train,val}.txt, or None.

    Split files list landmark paths relative to landmark_dir (e.g. HELLO/123.npy),
    typically signer-disjoint (see scripts/make_participant_split.py).
    """
    splits_dir = Path(splits_dir)
    train_f, val_f = splits_dir / "train.txt", splits_dir / "val.txt"
    if not (train_f.exists() and val_f.exists()):
        return None

    landmark_dir = Path(landmark_dir)
    rel_to_idx = {
        Path(p).relative_to(landmark_dir).as_posix(): i
        for i, (p, _) in enumerate(full.samples)
    }

    def read(path):
        idx, missing = [], 0
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line in rel_to_idx:
                idx.append(rel_to_idx[line])
            else:
                missing += 1
        return idx, missing

    train_idx, m_train = read(train_f)
    val_idx, m_val = read(val_f)
    if not train_idx or not val_idx:
        return None
    return train_idx, val_idx, m_train, m_val


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, _, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss_sum += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    spec = build_spec(cfg)
    device = resolve_device(cfg["train"].get("device", "auto"))
    log.info("Device: %s | feature_dim: %d", device, spec.feature_dim)

    dcfg = cfg["data"]
    cache_ram = bool(dcfg.get("cache_in_ram", False))
    full = LandmarkDataset.discover(
        dcfg["landmark_dir"],
        spec,
        max_frames=dcfg["max_frames"],
        min_frames=dcfg["min_frames"],
        normalize=cfg["landmarks"].get("normalize", True),
        aug_cfg=cfg.get("augment", {}),
        cache=cache_ram,
    )
    log.info("Dataset: %d clips, %d classes", len(full), full.num_classes)
    if cache_ram:
        log.info("Preloading features into RAM (one-time)…")
        nbytes = full.preload()
        log.info("Cache: %d clips, %.2f GB in RAM", len(full), nbytes / 1e9)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    full.save_labels(ckpt_dir / "labels.json")

    split = _load_split_files(full, dcfg["landmark_dir"], dcfg.get("splits_dir", "data/splits"))
    if split is not None:
        train_idx, val_idx, m_train, m_val = split
        log.info(
            "Signer-disjoint split: %d train / %d val clips (%d+%d lines unmatched)",
            len(train_idx), len(val_idx), m_train, m_val,
        )
    else:
        train_idx, val_idx = _split_indices(len(full), dcfg["val_fraction"], cfg["seed"])
        log.warning(
            "No split files in %s — using RANDOM split (not signer-disjoint; "
            "accuracy will be optimistic). Run scripts/make_participant_split.py.",
            dcfg.get("splits_dir", "data/splits"),
        )
    aug_on = bool(cfg.get("augment", {}).get("enabled", True))
    train_ds = full.subset(list(train_idx), augment=aug_on)
    val_ds = full.subset(list(val_idx), augment=False)
    log.info("Augmentation: %s (train only)", "on" if aug_on else "off")

    tcfg = cfg["train"]
    # With an in-RAM cache, serve from the main process: macOS spawns workers
    # that wouldn't share the preloaded cache (and would re-load it per worker).
    workers = 0 if cache_ram else tcfg["num_workers"]
    if cache_ram and tcfg["num_workers"]:
        log.info("num_workers forced to 0 (serving from RAM cache)")
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        collate_fn=collate, num_workers=workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=workers,
    )

    # Clean-train probe: a fixed sample of TRAIN clips with augmentation OFF, so
    # we can read train accuracy on clean inputs and compare it to val accuracy.
    # clean_train_acc >> val_acc => overfitting; both low and close => underfit /
    # needs more epochs or capacity.
    rng = np.random.default_rng(cfg["seed"])
    probe_idx = list(train_idx)
    if len(probe_idx) > 2000:
        probe_idx = list(rng.choice(np.asarray(probe_idx), size=2000, replace=False))
    probe_ds = full.subset(probe_idx, augment=False)
    probe_loader = DataLoader(
        probe_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=workers,
    )

    model = build_model(cfg, spec.feature_dim, full.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s, %.2fM params", cfg["model"]["type"], n_params / 1e6)

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    # Linear warmup (config: warmup_epochs) then cosine anneal — lets us use a
    # higher base LR without early instability.
    epochs = tcfg["epochs"]
    warmup = min(int(tcfg.get("warmup_epochs", 0)), max(epochs - 1, 0))
    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=0.1, total_iters=warmup
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs - warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, [warm, cos], milestones=[warmup]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    accum = max(int(tcfg.get("accum_steps", 1)), 1)
    if accum > 1:
        log.info("Gradient accumulation: %d steps (effective batch %d)",
                 accum, tcfg["batch_size"] * accum)

    best_acc, patience = 0.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        optim.zero_grad()
        n_batches = len(train_loader)
        for i, (x, _, mask, y) in enumerate(train_loader):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            loss = criterion(model(x, mask), y)
            running += loss.item() * len(y)
            (loss / accum).backward()  # scale so accumulated grad ~ mean over the
            # effective batch; step only every `accum` micro-batches.
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optim.step()
                optim.zero_grad()
        sched.step()
        train_loss = running / max(len(train_ds), 1)
        _, train_acc = evaluate(model, probe_loader, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        log.info(
            "epoch %3d | train_loss %.4f | clean_train_acc %.3f | "
            "val_loss %.4f | val_acc %.3f",
            epoch, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save(
                {"model": model.state_dict(), "config": cfg,
                 "labels": full.label_names, "val_acc": val_acc},
                ckpt_dir / "best.pt",
            )
            log.info("  ↳ saved best.pt (val_acc %.3f)", val_acc)
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                log.info("Early stopping at epoch %d (best val_acc %.3f)", epoch, best_acc)
                break

    log.info("Training complete. Best val_acc: %.3f", best_acc)


if __name__ == "__main__":
    main()
