"""Landmark specification, flattening, and normalization.

MediaPipe Holistic returns four landmark groups per frame:
    pose        33 points  x (x, y, z, visibility)  -> 132 values
    face       468 points  x (x, y, z)              -> 1404 values
    left_hand   21 points  x (x, y, z)              -> 63 values
    right_hand  21 points  x (x, y, z)              -> 63 values
Total per frame: 1662 values (full) -> we keep a 1629-d default vector
(pose without visibility = 99) — see FEATURE_DIM below.

Missing groups (e.g. a hand out of frame) are encoded as zeros so the
sequence length is constant and the model can learn the "absent" pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Point counts per group (MediaPipe Holistic).
N_POSE = 33
N_FACE = 468
N_HAND = 21

# Values per point.
POSE_DIM = 3   # x, y, z  (visibility dropped — noisy, weakly informative)
FACE_DIM = 3
HAND_DIM = 3

# Canonical full layout (all groups), in the order flatten_results emits them.
# Used to slice a full cached vector down to a spec, and to address point groups
# for augmentation. Cached arrays (extract.py full, adapt_google_islr.py) follow
# this order: pose, face, left_hand, right_hand.
_FULL_GROUPS = [
    ("pose", N_POSE, POSE_DIM),
    ("face", N_FACE, FACE_DIM),
    ("left_hand", N_HAND, HAND_DIM),
    ("right_hand", N_HAND, HAND_DIM),
]
FULL_FEATURE_DIM = sum(n * d for _, n, d in _FULL_GROUPS)  # 1629
N_POINTS_FULL = sum(n for _, n, _ in _FULL_GROUPS)         # 543

# Point-index ranges per group in the full (N_POINTS_FULL, 3) view.
_POINT_RANGES: dict[str, tuple[int, int]] = {}
_p = 0
for _name, _n, _ in _FULL_GROUPS:
    _POINT_RANGES[_name] = (_p, _p + _n)
    _p += _n


@dataclass(frozen=True)
class LandmarkSpec:
    """Which groups are included; defines the flattened feature vector."""

    use_pose: bool = True
    use_face: bool = True
    use_left_hand: bool = True
    use_right_hand: bool = True

    def group_sizes(self) -> dict[str, int]:
        return {
            "pose": N_POSE * POSE_DIM if self.use_pose else 0,
            "face": N_FACE * FACE_DIM if self.use_face else 0,
            "left_hand": N_HAND * HAND_DIM if self.use_left_hand else 0,
            "right_hand": N_HAND * HAND_DIM if self.use_right_hand else 0,
        }

    @property
    def feature_dim(self) -> int:
        return sum(self.group_sizes().values())


def _group_array(landmark_list, n_points: int, dim: int) -> np.ndarray:
    """Convert a MediaPipe landmark list to a flat (n_points*dim,) array.

    Returns zeros when the group is absent (landmark_list is None).
    """
    if landmark_list is None:
        return np.zeros(n_points * dim, dtype=np.float32)
    out = np.empty((n_points, dim), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        if dim > 2:
            out[i, 2] = lm.z
    return out.reshape(-1)


def flatten_results(results, spec: LandmarkSpec) -> np.ndarray:
    """Flatten one MediaPipe Holistic `results` object into a feature vector."""
    parts: list[np.ndarray] = []
    if spec.use_pose:
        parts.append(_group_array(results.pose_landmarks, N_POSE, POSE_DIM))
    if spec.use_face:
        parts.append(_group_array(results.face_landmarks, N_FACE, FACE_DIM))
    if spec.use_left_hand:
        parts.append(_group_array(results.left_hand_landmarks, N_HAND, HAND_DIM))
    if spec.use_right_hand:
        parts.append(_group_array(results.right_hand_landmarks, N_HAND, HAND_DIM))
    return np.concatenate(parts, axis=0).astype(np.float32)


def normalize_sequence(seq: np.ndarray, spec: LandmarkSpec) -> np.ndarray:
    """Translation/scale normalize a full-layout (T, 1629) sequence.

    Centers each frame on the pose mid-hip and scales by shoulder width so the
    representation is invariant to the signer's position and distance from the
    camera. Assumes the canonical full layout (pose first); call before
    select_groups. No-op when pose is excluded by the spec.

    Robust to missing pose: on frames where the shoulders are absent (zero-
    filled) the per-frame shoulder width collapses to ~0, which previously got
    clamped to 1e-3 and blew the coordinates up ~1000x. We instead fall back to
    the median shoulder width across valid frames (or 1.0 if none), so degenerate
    frames stay on the same scale as the rest.
    """
    if not spec.use_pose:
        return seq
    seq = seq.copy()
    pose_size = N_POSE * POSE_DIM
    pose = seq[:, :pose_size].reshape(len(seq), N_POSE, POSE_DIM)

    mid_hip = pose[:, [23, 24], :].mean(axis=1, keepdims=True)  # (T,1,3)
    shoulder = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)  # (T,)

    valid = shoulder > 1e-3
    fallback = float(np.median(shoulder[valid])) if valid.any() else 1.0
    scale = np.where(valid, shoulder, fallback)[:, None, None]  # (T,1,1)

    full = seq.reshape(len(seq), -1, 3)
    full = (full - mid_hip) / scale
    return full.reshape(len(seq), -1).astype(np.float32)


def group_point_ranges(spec: LandmarkSpec) -> dict[str, tuple[int, int]]:
    """Point-index ranges per enabled group in the SELECTED layout.

    Mirrors select_groups ordering (pose, face, left_hand, right_hand, enabled
    only). Lets augmentation locate the hands regardless of whether face is in.
    """
    sizes = {"pose": N_POSE, "face": N_FACE, "left_hand": N_HAND, "right_hand": N_HAND}
    keep = {
        "pose": spec.use_pose, "face": spec.use_face,
        "left_hand": spec.use_left_hand, "right_hand": spec.use_right_hand,
    }
    ranges: dict[str, tuple[int, int]] = {}
    off = 0
    for name in ("pose", "face", "left_hand", "right_hand"):
        if keep[name]:
            ranges[name] = (off, off + sizes[name])
            off += sizes[name]
    return ranges


def select_groups(seq: np.ndarray, spec: LandmarkSpec) -> np.ndarray:
    """Slice a full-layout (T, 1629) sequence down to the spec's enabled groups.

    Cached arrays are always stored full (all groups); the spec controls what the
    model actually consumes. If `seq` is already narrower than the full layout
    (e.g. extract.py wrote a reduced spec), it's returned unchanged.
    """
    if seq.shape[1] != FULL_FEATURE_DIM or spec.feature_dim == FULL_FEATURE_DIM:
        return seq
    pts = seq.reshape(len(seq), N_POINTS_FULL, 3)
    keep = {
        "pose": spec.use_pose,
        "face": spec.use_face,
        "left_hand": spec.use_left_hand,
        "right_hand": spec.use_right_hand,
    }
    parts = [pts[:, a:b, :] for name, (a, b) in _POINT_RANGES.items() if keep[name]]
    out = np.concatenate(parts, axis=1) if parts else pts[:, :0, :]
    return out.reshape(len(seq), -1).astype(np.float32)


def augment_sequence(
    seq: np.ndarray, cfg: dict, rng: np.random.Generator, spec: LandmarkSpec | None = None
) -> np.ndarray:
    """Apply landmark-space augmentation to a (T, n_points*3) sequence.

    Works on any selected layout (full 1629 or e.g. 225 hands+pose). `spec`
    describes the layout so mirror can swap the correct hand blocks; if omitted,
    the full canonical layout is assumed. All transforms are landmark-space and
    TRAIN ONLY (DESIGN.md / CONTINUOUS_DESIGN S13). Order: temporal → geometric →
    noise/dropout. Absent (zero-filled) points stay absent after additive ops.

    cfg keys (all optional): temporal_speed [lo,hi], rotate_deg, scale, shift,
    jitter_std, mirror_prob, point_dropout, frame_dropout, hand_dropout_prob.
    """
    n_points = seq.shape[1] // 3
    ranges = group_point_ranges(spec) if spec is not None else dict(_POINT_RANGES)
    pts = seq.reshape(len(seq), n_points, 3).astype(np.float32).copy()

    # --- temporal: resample to a random speed ---
    lo, hi = cfg.get("temporal_speed", [0.7, 1.4])
    speed = rng.uniform(lo, hi)
    new_t = max(4, int(round(len(pts) / speed)))
    if new_t != len(pts):
        idx = np.linspace(0, len(pts) - 1, new_t)
        lo_i = np.floor(idx).astype(int)
        hi_i = np.minimum(lo_i + 1, len(pts) - 1)
        frac = (idx - lo_i)[:, None, None]
        pts = pts[lo_i] * (1 - frac) + pts[hi_i] * frac  # linear interp in time

    # --- frame dropout ---
    fd = cfg.get("frame_dropout", 0.1)
    if fd > 0 and len(pts) > 4:
        keep = rng.random(len(pts)) >= fd
        if keep.sum() >= 4:
            pts = pts[keep]

    # --- mirror (left/right flip): negate x, swap hands if both present ---
    if rng.random() < cfg.get("mirror_prob", 0.5):
        pts[..., 0] = -pts[..., 0]
        if "left_hand" in ranges and "right_hand" in ranges:
            (la, lb), (ra, rb) = ranges["left_hand"], ranges["right_hand"]
            tmp = pts[:, la:lb, :].copy()
            pts[:, la:lb, :] = pts[:, ra:rb, :]
            pts[:, ra:rb, :] = tmp

    # --- rotation about origin in the x-y plane (zeros preserved) ---
    rot = np.deg2rad(rng.uniform(-1, 1) * cfg.get("rotate_deg", 13.0))
    c, s = np.cos(rot), np.sin(rot)
    x, y = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = c * x - s * y
    pts[..., 1] = s * x + c * y

    # --- isotropic scale (zeros preserved) ---
    sc = 1.0 + rng.uniform(-1, 1) * cfg.get("scale", 0.15)
    pts[..., :2] *= sc

    # --- shift + jitter (additive: re-zero absent points after) ---
    absent = (np.abs(pts).sum(axis=-1, keepdims=True) == 0)
    shift_mag = cfg.get("shift", 0.05)
    if shift_mag > 0:
        pts[..., :2] += rng.uniform(-shift_mag, shift_mag, size=(1, 1, 2))
    jit = cfg.get("jitter_std", 0.01)
    if jit > 0:
        pts += rng.normal(0.0, jit, size=pts.shape).astype(np.float32)
    pts = np.where(absent, 0.0, pts)

    # --- spatial point dropout (occlusion sim) ---
    pd = cfg.get("point_dropout", 0.1)
    if pd > 0:
        drop = rng.random((1, n_points, 1)) < pd
        pts = np.where(drop, 0.0, pts)

    # --- whole-hand dropout for the clip (reduced/one-handed signing) ---
    hands = [r for r in ("left_hand", "right_hand") if r in ranges]
    if hands and rng.random() < cfg.get("hand_dropout_prob", 0.2):
        a, b = ranges[hands[rng.integers(len(hands))]]
        pts[:, a:b, :] = 0.0

    return np.ascontiguousarray(pts.reshape(len(pts), -1), dtype=np.float32)
