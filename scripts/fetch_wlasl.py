"""Fetch WLASL clips for a chosen vocabulary into the data/raw/ layout.

Given the WLASL metadata JSON and a gloss list (e.g. configs/daily_glosses.txt),
this filters WLASL to your vocabulary, downloads the matching video instances,
and trims each clip to its annotated frame range — producing:

    data/raw/<GLOSS>/<video_id>.mp4

ready for `python -m src.extract`.

------------------------------------------------------------------------------
Setup (one time)
------------------------------------------------------------------------------
1. Get the WLASL metadata JSON. It ships in the WLASL repo:
       https://github.com/dxli94/WLASL  ->  start_kit/WLASL_v0.3.json
   Clone the repo or download that single file. (The JSON is metadata only;
   raw videos are fetched from their original hosts by this script.)
2. Install the data-prep tools:
       pip install yt-dlp opencv-python
   (yt-dlp pulls YouTube/host-hosted clips; OpenCV trims to the frame range.)

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
# See what matches your vocab WITHOUT downloading anything:
python scripts/fetch_wlasl.py --metadata WLASL_v0.3.json \
       --vocab configs/daily_glosses.txt --dry-run

# Download up to 20 clips per sign into data/raw/:
python scripts/fetch_wlasl.py --metadata WLASL_v0.3.json \
       --vocab configs/daily_glosses.txt --out data/raw --max-per-gloss 20

Notes
-----
* Many WLASL source URLs are dead links (the dataset is years old); the script
  skips failures and keeps going, then reports how many clips it actually got.
* Use --split train to download only the WLASL train partition, etc.
* Licensing: WLASL videos are research-use; do not redistribute the raw clips.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ----------------------------------------------------------------------------
# Vocabulary / gloss matching
# ----------------------------------------------------------------------------

def normalize(gloss: str) -> str:
    """Lowercase, collapse separators/whitespace — for matching only."""
    g = gloss.strip().lower()
    g = re.sub(r"[-_]+", " ", g)
    g = re.sub(r"\s+", " ", g)
    return g


def load_vocab(path: Path) -> dict[str, str]:
    """Return {normalized_gloss: original_label}. Skips blank/comment lines.

    The original label (e.g. 'THANK-YOU') becomes the data/raw/ folder name,
    so it must match the class label you want to train on.
    """
    vocab: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vocab[normalize(line)] = line
    return vocab


def load_metadata(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Unexpected WLASL JSON shape; expected a top-level list.")
    return data


def match_vocab(metadata: list[dict], vocab: dict[str, str]) -> dict[str, list[dict]]:
    """Map original label -> list of WLASL instance dicts."""
    matched: dict[str, list[dict]] = {}
    for entry in metadata:
        key = normalize(entry.get("gloss", ""))
        if key in vocab:
            matched.setdefault(vocab[key], []).extend(entry.get("instances", []))
    return matched


# ----------------------------------------------------------------------------
# Download + trim
# ----------------------------------------------------------------------------

def download_raw(url: str, dest: Path) -> bool:
    """Download a single video to `dest`. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # yt-dlp handles YouTube and most direct hosts; prefer an mp4 stream.
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "mp4/best", "-o", str(dest), url,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode == 0 and dest.exists():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: direct HTTP download for plain file URLs.
    if re.search(r"\.(mp4|mov|avi|mkv|webm)(\?|$)", url, re.I):
        try:
            import urllib.request

            urllib.request.urlretrieve(url, dest)  # noqa: S310 (research data)
            return dest.exists()
        except Exception:
            return False
    return False


def trim_clip(path: Path, frame_start: int, frame_end: int) -> None:
    """Trim `path` in place to [frame_start, frame_end] (1-indexed, inclusive).

    WLASL frame_start is 1-indexed; frame_end == -1 means 'to end'. No-op if
    the range covers the whole clip or OpenCV is unavailable.
    """
    if frame_start <= 1 and frame_end == -1:
        return
    try:
        import cv2
    except ImportError:
        return

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(frame_start - 1, 0)
    end = total if frame_end == -1 else min(frame_end, total)
    if start == 0 and end >= total:
        cap.release()
        return

    tmp = path.with_suffix(".trim.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= end:
            break
        if idx >= start:
            out.write(frame)
        idx += 1
    cap.release()
    out.release()
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path,
                    help="Path to WLASL_v0.3.json")
    ap.add_argument("--vocab", default=Path("configs/daily_glosses.txt"), type=Path)
    ap.add_argument("--out", default=Path("data/raw"), type=Path)
    ap.add_argument("--max-per-gloss", type=int, default=0,
                    help="Cap clips per sign (0 = no cap)")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="Only fetch this WLASL partition")
    ap.add_argument("--no-trim", action="store_true",
                    help="Keep full source videos; skip frame-range trimming")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report matches and clip counts; download nothing")
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)
    metadata = load_metadata(args.metadata)
    matched = match_vocab(metadata, vocab)

    found = sorted(matched)
    missing = sorted(set(vocab.values()) - set(matched))

    print(f"Vocabulary: {len(vocab)} signs")
    print(f"Found in WLASL: {len(found)}")
    print(f"Missing from WLASL: {len(missing)}")
    if missing:
        print("  (no WLASL gloss matched — edit these in your vocab file:)")
        print("   " + ", ".join(missing))
    print()

    total_available = 0
    for label in found:
        insts = matched[label]
        if args.split:
            insts = [i for i in insts if i.get("split") == args.split]
        total_available += len(insts)
        print(f"  {label:<16} {len(insts):>3} clips")

    print(f"\nTotal clips available: {total_available}")
    if args.dry_run:
        print("\n(dry run — nothing downloaded)")
        return

    got, tried = 0, 0
    for label in found:
        insts = matched[label]
        if args.split:
            insts = [i for i in insts if i.get("split") == args.split]
        if args.max_per_gloss:
            insts = insts[: args.max_per_gloss]
        for inst in insts:
            url = inst.get("url")
            vid = str(inst.get("video_id", "")) or f"{label}_{tried}"
            if not url:
                continue
            dest = args.out / label / f"{vid}.mp4"
            if dest.exists():
                got += 1
                continue
            tried += 1
            print(f"  ↓ {label}/{vid} ", end="", flush=True)
            if download_raw(url, dest):
                if not args.no_trim:
                    trim_clip(dest, int(inst.get("frame_start", 1)),
                              int(inst.get("frame_end", -1)))
                got += 1
                print("ok")
            else:
                print("failed (dead link / unsupported host)")

    print(f"\nDone. Downloaded {got} clips into {args.out}")
    if got == 0 and tried > 0:
        print("All attempts failed — check yt-dlp is installed and links are live.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
