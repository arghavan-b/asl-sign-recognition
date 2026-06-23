"""Inference on a video clip or live webcam, with confidence gating.

Emits the top prediction and its softmax confidence. When confidence falls
below the configured threshold, surfaces the top-k candidates instead of a
single answer — the hook for the Phase 2 clarification / interpreter loop.

Usage:
    python -m src.infer --checkpoint checkpoints/best.pt --source webcam
    python -m src.infer --checkpoint checkpoints/best.pt --source clip.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .extract import build_spec, extract_video
from .landmarks import normalize_sequence
from .models import build_model
from .utils import get_logger, resolve_device

log = get_logger("infer")


def _prep(seq: np.ndarray, cfg: dict, spec) -> torch.Tensor:
    if cfg["landmarks"].get("normalize", True):
        seq = normalize_sequence(seq, spec)
    max_f = cfg["data"]["max_frames"]
    if len(seq) > max_f:
        idx = np.linspace(0, len(seq) - 1, max_f).round().astype(int)
        seq = seq[idx]
    return torch.from_numpy(seq).unsqueeze(0).float()  # (1, T, F)


def predict(model, x: torch.Tensor, labels: list[str], cfg: dict, device: str):
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x.to(device)), dim=1).squeeze(0).cpu()
    k = min(cfg["infer"]["top_k"], len(labels))
    conf, idx = probs.topk(k)
    top = [(labels[i], float(c)) for c, i in zip(conf, idx)]
    gated = top[0][1] < cfg["infer"]["confidence_threshold"]
    return top, gated


def run_clip(path: Path, model, labels, cfg, spec, device) -> None:
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False
    ) as holistic:
        seq = extract_video(path, holistic, spec)
    if seq.shape[0] == 0:
        log.error("No frames decoded from %s", path)
        return
    top, gated = predict(model, _prep(seq, cfg, spec), labels, cfg, device)
    _report(top, gated)


def run_webcam(model, labels, cfg, spec, device) -> None:
    import cv2
    import mediapipe as mp

    from .landmarks import flatten_results

    mp_holistic = mp.solutions.holistic
    buf: list[np.ndarray] = []
    win = cfg["data"]["max_frames"]
    cap = cv2.VideoCapture(0)
    log.info("Webcam started. Press 'q' to quit.")
    with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=False) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf.append(flatten_results(holistic.process(rgb), spec))
            buf = buf[-win:]
            if len(buf) >= cfg["data"]["min_frames"]:
                seq = np.stack(buf, axis=0)
                top, gated = predict(model, _prep(seq, cfg, spec), labels, cfg, device)
                label = f"{top[0][0]} ({top[0][1]:.2f})"
                color = (0, 165, 255) if gated else (0, 200, 0)
                cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("ASL sign recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def _report(top, gated) -> None:
    if gated:
        print("Low confidence — candidates (needs clarification / interpreter):")
        for name, c in top:
            print(f"  {name:<20} {c:.3f}")
    else:
        name, c = top[0]
        print(f"Prediction: {name}  (confidence {c:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--source", required=True, help="'webcam' or a video file path")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, labels = ckpt["config"], ckpt["labels"]
    spec = build_spec(cfg)
    device = resolve_device(cfg["train"].get("device", "auto"))

    model = build_model(cfg, spec.feature_dim, len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    log.info("Loaded %s (%d classes, val_acc %.3f)",
             args.checkpoint, len(labels), ckpt.get("val_acc", float("nan")))

    if args.source == "webcam":
        run_webcam(model, labels, cfg, spec, device)
    else:
        run_clip(Path(args.source), model, labels, cfg, spec, device)


if __name__ == "__main__":
    main()
