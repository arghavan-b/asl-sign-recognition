"""Download the YouTube-ASL corpus from its released video-ID list.

YouTube-ASL (Google Research, ~984 h, ~11k videos) ships only a list of YouTube
video IDs — you fetch the videos + captions yourself. This is a resumable
`yt-dlp` loop over that ID list:

    raw-video/<video_id>.mp4
    raw-video/<video_id>.en.vtt      (human captions, when available)

YouTube-ASL is sentence-level continuous signing — i.e. PRETRAINING material
toward Phase 3 (continuous), not isolated Phase-1 MVP clips. Run landmark
extraction afterward and keep only the .npy arrays (smaller + matches the
project's privacy posture).

------------------------------------------------------------------------------
Setup
------------------------------------------------------------------------------
1. Get the ID list from the YouTube-ASL README:
       https://github.com/google-research/google-research/tree/master/youtube_asl
   Save the IDs as a text file (one video_id per line) — e.g. youtube_asl_ids.txt
2. pip install yt-dlp

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
# Preview how many IDs / how many already downloaded:
python scripts/fetch_youtube_asl.py --ids youtube_asl_ids.txt \
       --dest data/youtube_asl --dry-run

# Download (resumable — re-run any time; it skips finished IDs):
python scripts/fetch_youtube_asl.py --ids youtube_asl_ids.txt \
       --dest data/youtube_asl

# Subset for a quick test:
python scripts/fetch_youtube_asl.py --ids youtube_asl_ids.txt \
       --dest data/youtube_asl --limit 25

Notes
-----
* Many IDs are gone (users delete/privatize videos); failures are logged and
  skipped, and written to <dest>/_failed.txt so you don't retry them forever.
* Grabs HUMAN captions (--write-subs), not auto-generated ones — YouTube-ASL's
  text relies on the real captions. Pass --no-captions to skip.
* Licensing: the ID list is CC BY 4.0; the videos are under YouTube ToS, so
  assess commercial-training rights yourself. See DESIGN.md §14.
* Expect days of rate-limited downloading and ~1 TB for the full set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def load_ids(path: Path) -> list[str]:
    """One video_id per line; tolerates full URLs and blank/comment lines."""
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept a bare ID or a watch URL.
        if "watch?v=" in line:
            line = line.split("watch?v=", 1)[1].split("&", 1)[0]
        elif "youtu.be/" in line:
            line = line.split("youtu.be/", 1)[1].split("?", 1)[0]
        ids.append(line)
    return ids


def already_done(dest: Path, vid: str) -> bool:
    """True if a video file for this ID already exists."""
    return any(dest.glob(f"{vid}.*")) and not (dest / f"{vid}.part").exists()


def load_failed(dest: Path) -> set[str]:
    f = dest / "_failed.txt"
    if f.exists():
        return {x.strip() for x in f.read_text().splitlines() if x.strip()}
    return set()


def record_failed(dest: Path, vid: str) -> None:
    with open(dest / "_failed.txt", "a") as f:
        f.write(vid + "\n")


def download_one(vid: str, dest: Path, captions: bool) -> bool:
    url = f"https://www.youtube.com/watch?v={vid}"
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "mp4/best",
        "-o", str(dest / "%(id)s.%(ext)s"),
    ]
    if captions:
        cmd += [
            "--write-subs", "--no-write-auto-subs",
            "--sub-langs", "en.*", "--sub-format", "vtt",
        ]
    cmd.append(url)
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=900)
        return res.returncode == 0 and already_done(dest, vid)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. `pip install yt-dlp`.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ids", required=True, type=Path,
                    help="Text file of YouTube-ASL video IDs (one per line)")
    ap.add_argument("--dest", default=Path("data/youtube_asl"), type=Path)
    ap.add_argument("--limit", type=int, default=0, help="Only process first N IDs (0 = all)")
    ap.add_argument("--no-captions", action="store_true", help="Skip caption download")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ids = load_ids(args.ids)
    if args.limit:
        ids = ids[: args.limit]
    args.dest.mkdir(parents=True, exist_ok=True)

    failed = load_failed(args.dest)
    done = [v for v in ids if already_done(args.dest, v)]
    skip = failed - set(done)
    todo = [v for v in ids if v not in done and v not in failed]

    print(f"IDs in list:        {len(ids)}")
    print(f"Already downloaded: {len(done)}")
    print(f"Known-failed (skip):{len(skip)}")
    print(f"To download:        {len(todo)}")
    if args.dry_run:
        print("\n(dry run — nothing downloaded)")
        return

    got = 0
    for i, vid in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {vid} ", end="", flush=True)
        if download_one(vid, args.dest, captions=not args.no_captions):
            got += 1
            print("ok")
        else:
            record_failed(args.dest, vid)
            print("failed (gone/private/blocked) — logged")

    print(f"\nDone this run: {got} new videos. Total in {args.dest}: {len(done) + got}")
    print("Next: python -m src.extract --input <clips> --output data/landmarks")


if __name__ == "__main__":
    main()
