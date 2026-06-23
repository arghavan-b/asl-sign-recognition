"""Build a participant-disjoint (signer-disjoint) train/val split.

The Google/PopSign dataset labels each sample with a `participant_id`. Splitting
on *participants* — not individual clips — is the only honest measure of whether
the model generalizes to people it never trained on (see DESIGN.md §4/§8). A
random clip split leaks the same signer into train and val and inflates accuracy.

This reads `participant_id` from the Kaggle `train.csv`, maps it onto the
landmark `.npy` files that actually exist on disk (so it works no matter how you
ran the adapter — full set, --vocab filtered, or --max-per-sign capped), assigns
whole participants to train or val, and writes:

    data/splits/train.txt   (lines: <SIGN>/<sequence_id>.npy)
    data/splits/val.txt

`src.train` picks these up automatically when present.

Usage
-----
python scripts/make_participant_split.py \
       --data-dir data/asl-signs-kaggle \
       --landmark-dir data/landmarks \
       --splits-out data/splits --val-frac 0.2

# Preview the split (participants/clip counts) without writing:
python scripts/make_participant_split.py --data-dir data/asl-signs-kaggle \
       --landmark-dir data/landmarks --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def seq_to_participant(data_dir: Path) -> dict[str, str]:
    """Map sequence_id -> participant_id from train.csv."""
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found in {data_dir}")
    out: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "participant_id" not in (reader.fieldnames or []):
            raise ValueError("train.csv has no participant_id column")
        for r in reader:
            seq = str(r.get("sequence_id") or Path(r.get("path", "")).stem)
            out[seq] = str(r["participant_id"])
    return out


def index_landmarks(landmark_dir: Path) -> list[tuple[str, str]]:
    """Return (relpath_posix, sequence_id) for every .npy under landmark_dir."""
    files = sorted(landmark_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files under {landmark_dir}")
    return [(p.relative_to(landmark_dir).as_posix(), p.stem) for p in files]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Kaggle asl-signs dir (contains train.csv)")
    ap.add_argument("--landmark-dir", default=Path("data/landmarks"), type=Path)
    ap.add_argument("--splits-out", default=Path("data/splits"), type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of PARTICIPANTS held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seq2part = seq_to_participant(args.data_dir)
    files = index_landmarks(args.landmark_dir)

    # Attach a participant to each existing landmark file.
    tagged: list[tuple[str, str]] = []  # (relpath, participant)
    unmatched = 0
    for rel, seq in files:
        part = seq2part.get(seq)
        if part is None:
            unmatched += 1
            continue
        tagged.append((rel, part))

    participants = sorted({p for _, p in tagged})
    rng = random.Random(args.seed)
    rng.shuffle(participants)
    n_val = max(1, round(len(participants) * args.val_frac))
    val_parts = set(participants[:n_val])
    train_parts = set(participants[n_val:])

    train_lines = [rel for rel, p in tagged if p in train_parts]
    val_lines = [rel for rel, p in tagged if p in val_parts]

    print(f"Landmark files:        {len(files)}")
    if unmatched:
        print(f"Unmatched (no train.csv participant): {unmatched}")
    print(f"Participants:          {len(participants)} "
          f"({len(train_parts)} train / {len(val_parts)} val)")
    print(f"Clips:                 {len(train_lines)} train / {len(val_lines)} val")
    print(f"Val participant IDs:   {', '.join(sorted(val_parts))}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    args.splits_out.mkdir(parents=True, exist_ok=True)
    (args.splits_out / "train.txt").write_text("\n".join(sorted(train_lines)) + "\n")
    (args.splits_out / "val.txt").write_text("\n".join(sorted(val_lines)) + "\n")
    print(f"\nWrote {args.splits_out}/train.txt and val.txt")
    print("Run training as usual — src.train will use these automatically.")


if __name__ == "__main__":
    main()
