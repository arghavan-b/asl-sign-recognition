"""Run an ablation matrix and print a val-accuracy comparison table.

Trains several variants of the same setup — augmentation on/off crossed with
face on/off — each into its own checkpoint dir, then reports the best val
accuracy of each so you can see what actually helps in one shot, instead of
hand-editing configs between runs.

All variants share the same data, split, and seed, so the only thing changing
is the knob under test.

Usage
-----
# Full 2x2 matrix, 25 epochs each (good for a quick read):
python scripts/compare_runs.py --epochs 25

# Just two variants:
python scripts/compare_runs.py --only aug,aug_noface --epochs 30

# See what would run without training anything:
python scripts/compare_runs.py --dry-run

Results print as a table and are also written to <out-dir>/summary.md.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# name -> dotted-key overrides applied on top of the base config.
VARIANTS: dict[str, dict] = {
    "baseline":     {"augment.enabled": False, "landmarks.use_face": True},
    "aug":          {"augment.enabled": True,  "landmarks.use_face": True},
    "aug_noface":   {"augment.enabled": True,  "landmarks.use_face": False},
    "noaug_noface": {"augment.enabled": False, "landmarks.use_face": False},
}


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def read_val_acc(ckpt_path: Path) -> float | None:
    """Read best val_acc from a saved checkpoint (lazy torch import)."""
    if not ckpt_path.exists():
        return None
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return float(ckpt.get("val_acc")) if "val_acc" in ckpt else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/default.yaml", type=Path)
    ap.add_argument("--out-dir", default=Path("runs/compare"), type=Path)
    ap.add_argument("--epochs", type=int, default=0,
                    help="Override epochs for every run (0 = use config value)")
    ap.add_argument("--only", default="", help="Comma-separated subset of variant names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        ap.error(f"Unknown variant(s): {unknown}. Choose from {list(VARIANTS)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Planned runs:")
    for n in names:
        ov = dict(VARIANTS[n])
        if args.epochs:
            ov["train.epochs"] = args.epochs
        print(f"  {n:<14} {ov}")
    if args.dry_run:
        print("\n(dry run — nothing trained)")
        return

    results: list[tuple[str, float | None]] = []
    for n in names:
        cfg = copy.deepcopy(base)
        for k, v in VARIANTS[n].items():
            set_dotted(cfg, k, v)
        if args.epochs:
            set_dotted(cfg, "train.epochs", args.epochs)
        ckpt_dir = args.out_dir / n
        set_dotted(cfg, "train.checkpoint_dir", str(ckpt_dir))

        with tempfile.NamedTemporaryFile(
            "w", suffix=f"_{n}.yaml", delete=False, dir=args.out_dir
        ) as f:
            yaml.safe_dump(cfg, f)
            cfg_path = f.name

        print(f"\n=== Training variant: {n} ===")
        proc = subprocess.run([sys.executable, "-m", "src.train", "--config", cfg_path])
        if proc.returncode != 0:
            print(f"  variant {n} FAILED (exit {proc.returncode})")
            results.append((n, None))
            continue
        results.append((n, read_val_acc(ckpt_dir / "best.pt")))

    # --- report ---
    lines = ["", "=" * 40, "Comparison — best val accuracy", "=" * 40,
             f"{'variant':<16}{'val_acc':>10}"]
    for n, acc in results:
        lines.append(f"{n:<16}{(f'{acc:.4f}' if acc is not None else 'n/a'):>10}")
    best = max((r for r in results if r[1] is not None), key=lambda r: r[1], default=None)
    if best:
        lines += ["", f"Best: {best[0]} ({best[1]:.4f})"]
    report = "\n".join(lines)
    print(report)
    (args.out_dir / "summary.md").write_text("```\n" + report + "\n```\n")
    print(f"\nSaved {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()
