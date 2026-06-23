"""Organize the ASL Citizen dataset into our data/raw/ layout.

ASL Citizen (Microsoft Research) ships as a download with a `videos/` folder
and split CSVs (train/val/test). This script files each clip under its gloss:

    data/raw/<GLOSS>/<video>.mp4

and — because ASL Citizen's train/val/test split is **signer-disjoint** — also
writes split lists our trainer can use for honest cross-signer evaluation:

    data/splits/{train,val}.txt    (lines: <GLOSS>/<stem>.npy)

------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------
Download + accept the license at:
    https://www.microsoft.com/en-us/research/project/asl-citizen/
Unzip it; point --src at the folder that contains `videos/` and the CSV splits.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
# Preview gloss overlap with your vocab (no files touched):
python scripts/organize_asl_citizen.py --src /path/to/ASL_Citizen \
       --vocab configs/daily_glosses.txt --dry-run

# Symlink matching clips into data/raw/ and write split files:
python scripts/organize_asl_citizen.py --src /path/to/ASL_Citizen \
       --vocab configs/daily_glosses.txt --out data/raw --splits-out data/splits

Notes
-----
* Default is to SYMLINK (no extra disk); pass --copy to duplicate files instead.
* Omit --vocab to organize every gloss in the dataset.
* Column names are detected case-insensitively (Gloss / Video file).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
SPLIT_FILES = {"train": ["train.csv"], "val": ["val.csv", "valid.csv"], "test": ["test.csv"]}


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


def _find_col(fieldnames: list[str], *candidates: str) -> str | None:
    lut = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c in lut:
            return lut[c]
    # Fuzzy: first field whose lowercased name contains a candidate token.
    for c in candidates:
        for low, orig in lut.items():
            if c in low:
                return orig
    return None


def read_split(src: Path, names: list[str]) -> list[dict] | None:
    for name in names:
        p = src / name
        if p.exists():
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                gcol = _find_col(reader.fieldnames or [], "gloss", "sign")
                vcol = _find_col(reader.fieldnames or [], "video file", "video", "filename", "file")
                return [{"gloss": r.get(gcol, ""), "video": r.get(vcol, "")} for r in rows]
    return None


def resolve_video(src: Path, name: str) -> Path | None:
    """Locate a video file by name, searching videos/ then the tree."""
    if not name:
        return None
    cand = src / "videos" / name
    if cand.exists():
        return cand
    cand = src / name
    if cand.exists():
        return cand
    # Last resort: search by stem (handles a different extension).
    stem = Path(name).stem
    for p in src.rglob(stem + ".*"):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Extracted ASL Citizen dir")
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--splits-out", type=Path, default=Path("data/splits"))
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab) if args.vocab else None
    splits = {k: read_split(args.src, names) for k, names in SPLIT_FILES.items()}
    found_splits = {k: v for k, v in splits.items() if v}
    if not found_splits:
        raise FileNotFoundError(
            f"No split CSVs (train/val/test) found in {args.src}. "
            "Point --src at the folder containing the dataset CSVs."
        )

    # Report overlap.
    all_gloss = {normalize(r["gloss"]) for rows in found_splits.values() for r in rows}
    if vocab is not None:
        present = {vocab[g] for g in all_gloss if g in vocab}
        missing = sorted(set(vocab.values()) - present)
        print(f"Vocab signs: {len(vocab)} | present in ASL Citizen: {len(present)}")
        print(f"Missing ({len(missing)}): {', '.join(missing) or 'none'}")
    print(f"Splits found: {', '.join(found_splits)}")
    for k, rows in found_splits.items():
        kept = sum(1 for r in rows if vocab is None or normalize(r["gloss"]) in vocab)
        print(f"  {k:<5} {kept} matching clips")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return

    args.splits_out.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    for split_name, rows in found_splits.items():
        split_lines: list[str] = []
        for r in rows:
            key = normalize(r["gloss"])
            if vocab is not None and key not in vocab:
                continue
            label = vocab[key] if vocab is not None else r["gloss"].strip()
            video = resolve_video(args.src, r["video"])
            if video is None:
                continue
            dest = args.out / label / video.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                if args.copy:
                    shutil.copy2(video, dest)
                else:
                    dest.symlink_to(video.resolve())
            split_lines.append(f"{label}/{Path(video.name).stem}.npy")
            totals[split_name] = totals.get(split_name, 0) + 1
        # ASL Citizen test split has no labels for our purposes; keep train/val.
        if split_name in ("train", "val") and split_lines:
            (args.splits_out / f"{split_name}.txt").write_text("\n".join(split_lines) + "\n")

    print("\nOrganized clips:")
    for k, n in totals.items():
        print(f"  {k:<5} {n}")
    print(f"\nVideos in {args.out}; signer-disjoint split lists in {args.splits_out}.")
    print("Next: python -m src.extract --input data/raw --output data/landmarks")


if __name__ == "__main__":
    main()
