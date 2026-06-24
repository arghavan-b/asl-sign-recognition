"""Adapt the Google/PopSign Isolated Sign Language Recognition data to our format.

The Kaggle "Google - Isolated Sign Language Recognition" dataset ships MediaPipe
Holistic landmarks already extracted (no video), as one parquet per sign sample.
This converts them directly into our cached-landmark layout, skipping the
`src.extract` step entirely:

    data/landmarks/<SIGN>/<sequence_id>.npy     shape (T, 1629) float32

so you can go straight to `python -m src.train`.

------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------
1. Download the dataset (requires a Kaggle account + competition rules accepted):
       https://www.kaggle.com/competitions/asl-signs/data
   You need `train.csv` and the `train_landmark_files/` directory.
2. pip install pandas pyarrow      # to read the parquet landmark files

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------

uv run python scripts/adapt_google_islr.py --data-dir data/asl-signs-kaggle/\
       --vocab configs/daily_glosses.txt --dry-run

# Convert (optionally filtered to your vocab, capped per sign):
uv run python scripts/adapt_google_islr.py --data-dir /data/asl-signs-kaggle/ \
       --vocab configs/daily_glosses.txt --out data/landmarks --max-per-sign 200

Notes
-----
* The dataset's MediaPipe layout is pose(33) + face(468) + hands(21 each) =
  1629 dims with (x, y, z) per point — identical to our LandmarkSpec default,
  so no re-extraction or re-ordering of *groups* is needed.
* Missing landmarks are NaN in the source; we zero-fill them to match our
  "absent group = zeros" convention.
* Omit --vocab to convert all 250 signs.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# Canonical group order + sizes — MUST match src/landmarks.py flatten order.
GROUP_ORDER = [("pose", 33), ("face", 468), ("left_hand", 21), ("right_hand", 21)]
TYPE_BASE: dict[str, int] = {}
_off = 0
for _name, _n in GROUP_ORDER:
    TYPE_BASE[_name] = _off
    _off += _n
N_POINTS = _off            # 543
FEATURE_DIM = N_POINTS * 3  # 1629


def normalize(gloss: str) -> str:
    g = re.sub(r"[-_]+", " ", gloss.strip().lower())
    return re.sub(r"\s+", " ", g)


def load_vocab(path: Path) -> dict[str, str]:
    vocab: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            vocab[normalize(line)] = line
    return vocab


def read_index(data_dir: Path) -> list[dict]:
    """Read train.csv rows: {path, sequence_id, sign}."""
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv not found in {data_dir}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def parquet_to_array(parquet_path: Path) -> np.ndarray:
    """Convert one sample parquet to (T, 1629) float32 in our group order."""
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(TYPE_BASE)]
    frames = np.sort(df["frame"].unique())
    frame_to_i = {f: i for i, f in enumerate(frames)}

    arr = np.zeros((len(frames), N_POINTS, 3), dtype=np.float32)
    fi = df["frame"].map(frame_to_i).to_numpy()
    pos = df["type"].map(TYPE_BASE).to_numpy() + df["landmark_index"].to_numpy()
    arr[fi, pos, 0] = df["x"].to_numpy()
    arr[fi, pos, 1] = df["y"].to_numpy()
    arr[fi, pos, 2] = df["z"].to_numpy()
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.reshape(len(frames), FEATURE_DIM)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Kaggle asl-signs dir (contains train.csv, train_landmark_files/)")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="Optional gloss list to filter to (else all 250 signs)")
    ap.add_argument("--out", type=Path, default=Path("data/landmarks"))
    ap.add_argument("--max-per-sign", type=int, default=0, help="0 = no cap")
    ap.add_argument("--no-face", action="store_true",
                    help="Write hands+pose only (225-d) — ~7x smaller on disk. "
                         "Use when training with landmarks.use_face=false (e.g. on "
                         "Kaggle's 20GB working limit).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Columns to drop when --no-face (face block, in value units).
    face_cols = np.s_[TYPE_BASE["face"] * 3: TYPE_BASE["left_hand"] * 3]
    out_dim = FEATURE_DIM - (TYPE_BASE["left_hand"] - TYPE_BASE["face"]) * 3
    if args.no_face:
        print(f"--no-face: writing {out_dim}-d (pose+hands) arrays")

    rows = read_index(args.data_dir)
    all_signs = sorted({r["sign"] for r in rows})

    vocab = load_vocab(args.vocab) if args.vocab else None
    if vocab is not None:
        keep = {s: vocab[normalize(s)] for s in all_signs if normalize(s) in vocab}
        missing = sorted(set(vocab.values()) - set(keep.values()))
    else:
        keep = {s: s for s in all_signs}
        missing = []

    print(f"Dataset signs: {len(all_signs)} | converting: {len(keep)}")
    if vocab is not None:
        print(f"Vocab signs not in this dataset ({len(missing)}): {', '.join(missing) or 'none'}")

    counts: dict[str, int] = {}
    for r in rows:
        if r["sign"] in keep:
            counts[keep[r["sign"]]] = counts.get(keep[r["sign"]], 0) + 1
    for label in sorted(counts):
        print(f"  {label:<16} {counts[label]:>4} samples")
    print(f"Total samples to convert: {sum(counts.values())}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    written, per_sign = 0, {}
    for r in rows:
        if r["sign"] not in keep:
            continue
        label = keep[r["sign"]]
        if args.max_per_sign and per_sign.get(label, 0) >= args.max_per_sign:
            continue
        seq_id = r.get("sequence_id") or Path(r["path"]).stem
        dest = args.out / label / f"{seq_id}.npy"
        if dest.exists():
            written += 1
            per_sign[label] = per_sign.get(label, 0) + 1
            continue
        arr = parquet_to_array(args.data_dir / r["path"])
        if arr.shape[0] == 0:
            continue
        if args.no_face:
            arr = np.delete(arr, face_cols, axis=1)  # -> (T, 225) pose+hands
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, arr)
        written += 1
        per_sign[label] = per_sign.get(label, 0) + 1
        if written % 250 == 0:
            print(f"  …{written} written")

    print(f"\nDone. Wrote {written} landmark files to {args.out}")


if __name__ == "__main__":
    main()
