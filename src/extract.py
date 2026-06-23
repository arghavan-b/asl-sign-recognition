"""Extract MediaPipe Holistic landmarks from videos and cache them as .npy.

Run once over your raw clips; training then reads the cached arrays, which is
dramatically faster than re-running MediaPipe every epoch.

Input layout :  <input>/<gloss>/<clip>.mp4
Output layout:  <output>/<gloss>/<clip>.npy   shape (T, feature_dim) float32

Usage:
    python -m src.extract --input data/raw --output data/landmarks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .landmarks import LandmarkSpec, flatten_results
from .utils import get_logger

log = get_logger("extract")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract_video(path: Path, holistic, spec: LandmarkSpec) -> np.ndarray:
    """Return (T, feature_dim) landmark array for one video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            frames.append(flatten_results(results, spec))
    finally:
        cap.release()
    if not frames:
        return np.zeros((0, spec.feature_dim), dtype=np.float32)
    return np.stack(frames, axis=0)


def build_spec(cfg: dict) -> LandmarkSpec:
    lm = cfg.get("landmarks", {})
    return LandmarkSpec(
        use_pose=lm.get("use_pose", True),
        use_face=lm.get("use_face", True),
        use_left_hand=lm.get("use_left_hand", True),
        use_right_hand=lm.get("use_right_hand", True),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MediaPipe landmarks from videos.")
    ap.add_argument("--input", default="data/raw", type=Path)
    ap.add_argument("--output", default="data/landmarks", type=Path)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overwrite", action="store_true", help="Re-extract existing .npy")
    args = ap.parse_args()

    # Imported here so the rest of the module is testable without mediapipe.
    import mediapipe as mp

    from .utils import load_config

    cfg = load_config(args.config)
    spec = build_spec(cfg)
    log.info("Feature dim: %d", spec.feature_dim)

    videos = [p for p in args.input.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        log.warning("No videos found under %s", args.input)
        return
    log.info("Found %d videos", len(videos))

    mp_holistic = mp.solutions.holistic
    n_done = 0
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,  # 468 face points (matches LandmarkSpec)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for vid in videos:
            rel = vid.relative_to(args.input).with_suffix(".npy")
            out_path = args.output / rel
            if out_path.exists() and not args.overwrite:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            arr = extract_video(vid, holistic, spec)
            if arr.shape[0] == 0:
                log.warning("No frames decoded: %s", vid)
                continue
            np.save(out_path, arr)
            n_done += 1
            if n_done % 25 == 0:
                log.info("Extracted %d / %d", n_done, len(videos))

    log.info("Done. Wrote %d landmark files to %s", n_done, args.output)


if __name__ == "__main__":
    main()
