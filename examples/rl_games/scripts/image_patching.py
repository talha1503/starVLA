"""Ghost-trail image patching for Flappy Bird frames.

Builds a single composite image where the most recent frame is the
authoritative background (current pipes/ground/bird, always correct), and up
to N earlier frames contribute fading "ghost" cutouts of the bird at its past
on-screen positions.

Flappy Bird's world scrolls (pipes move left over time) while the bird stays
at a roughly fixed screen x -- so unlike a static-camera scene, a ghost's old
(x, y) position can be occluded by a pipe that has since scrolled into that
exact spot. Ghosts are therefore clipped to the *current* frame's pipe/ground
mask before compositing: wherever the current frame already has a solid
pipe/ground pixel, no ghost is drawn there, so a ghost never appears to float
through an obstacle that wasn't there when it was captured. This mask is
recomputed fresh from the current frame every time, so it stays correct for
every timestep in an episode, not just frames where the background happens to
be unchanged.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def _natural_key(p: str):
    # Sort frame2.png before frame10.png
    base = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def load_frames_from_paths(image_paths: Sequence[str]) -> List[np.ndarray]:
    """Load a sequence of image file paths into naturally-sorted RGB arrays."""
    paths = sorted(image_paths, key=_natural_key)
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for ghost-trail patching (pip install opencv-python)")


def _make_pipe_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Mask of pipe (and bush) colored pixels.

    Cast to int16 before +/- comparisons to avoid uint8 wraparound.
    """
    _require_cv2()
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Pipes/bushes are yellow-green-ish; sky is ~H=90+, excluded by the upper bound.
    pipe_hsv = cv2.inRange(
        hsv,
        np.array([30, 30, 40], np.uint8),
        np.array([75, 255, 255], np.uint8),
    ).astype(bool)

    Ri = frame_rgb[..., 0].astype(np.int16)
    Gi = frame_rgb[..., 1].astype(np.int16)
    Bi = frame_rgb[..., 2].astype(np.int16)
    pipe_rgb = (Gi > 70) & ((Gi - Ri) > 12) & ((Gi - Bi) > 12)

    pipe = pipe_hsv | pipe_rgb

    # Dilate so pipe edges/caps don't leak into the bird mask.
    pipe_u8 = pipe.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    pipe_u8 = cv2.dilate(pipe_u8, k, iterations=1)
    return pipe_u8.astype(bool)


def _make_ground_mask(frame_rgb: np.ndarray, ground_fraction: float = 0.22) -> np.ndarray:
    """Mask of the ground band (bushes/stripe/dirt) at the bottom of the frame.

    This renderer always places the ground at the same fixed bottom slice of
    the frame, so a fixed-row cutoff is more robust here than a color
    heuristic: it can't false-positive on sky/clouds, and it covers the dirt
    strip that isn't green enough for the pipe-color heuristic to catch.
    Measured empirically: the bush/stripe/dirt band starts at ~78% of frame
    height, so a 22% bottom slice covers it with a small safety margin.
    """
    H, W = frame_rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    ground_start = int(H * (1.0 - ground_fraction))
    mask[ground_start:, :] = True
    return mask


def _make_occlusion_mask(frame_rgb: np.ndarray, ground_fraction: float = 0.22) -> np.ndarray:
    """Everything in the *current* frame that should occlude a ghost.

    Recomputed fresh per-frame from the current frame alone, so it stays
    correct regardless of how far pipes have scrolled since a ghost's source
    frame was captured.
    """
    return _make_pipe_mask(frame_rgb) | _make_ground_mask(frame_rgb, ground_fraction)


def _segment_bird(
    frame_rgb: np.ndarray,
    occlusion_mask: Optional[np.ndarray] = None,
    sky_patch: int = 40,
    sky_tol: int = 6,
) -> Optional[np.ndarray]:
    """Segment the bird sprite via color, without using motion.

    Builds a core (orange body + white belly), expands it to include the dark
    outline, excludes pipes/ground, then picks the best connected component
    near the bird's typical x. Finally strips any pixels matching the frame's
    dominant sky color, which removes stray blue/cloud leakage from dilation.
    """
    _require_cv2()
    H, W = frame_rgb.shape[:2]
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    orange = cv2.inRange(
        hsv,
        np.array([5, 70, 70], np.uint8),
        np.array([40, 255, 255], np.uint8),
    ).astype(bool)
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 210], np.uint8),
        np.array([180, 70, 255], np.uint8),
    ).astype(bool)

    occlusion = occlusion_mask if occlusion_mask is not None else _make_occlusion_mask(frame_rgb)
    core = (orange | white) & (~occlusion)

    core_u8 = core.astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_u8 = cv2.dilate(core_u8, k5, iterations=2)

    dark = hsv[..., 2] < 120  # outline pixels
    cand = core_u8.astype(bool) & (~occlusion) & (dark | orange | white)

    cand_u8 = cand.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, k3, iterations=1)
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE, k3, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    if num <= 1:
        return None

    target_x = int(W * 0.35)  # bird is usually around here
    best = None
    best_score = None
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if not (80 <= area <= 9000):
            continue
        comp = labels == lab
        core_cnt = int((core & comp).sum())
        if core_cnt < 30:
            continue
        cx, _ = cents[lab]
        dist = abs(cx - target_x)
        # Prefer lots of core pixels, then closer x, then larger area.
        score = (-core_cnt, dist, -area)
        if best is None or score < best_score:
            best, best_score = lab, score

    if best is None:
        return None

    mask = labels == best

    # Remove pixels matching the frame's dominant sky color.
    p = min(sky_patch, H, W)
    patch = frame_rgb[:p, :p].reshape(-1, 3)
    colors, counts = np.unique(patch, axis=0, return_counts=True)
    sky_rgb = colors[int(np.argmax(counts))].astype(np.int16)
    Ri = frame_rgb[..., 0].astype(np.int16)
    Gi = frame_rgb[..., 1].astype(np.int16)
    Bi = frame_rgb[..., 2].astype(np.int16)
    d = np.abs(Ri - sky_rgb[0]) + np.abs(Gi - sky_rgb[1]) + np.abs(Bi - sky_rgb[2])
    mask = mask & ~(d <= sky_tol)

    # Keep only the largest remaining component (removes tiny stray dots).
    mask_u8 = mask.astype(np.uint8) * 255
    num2, lab2, stats2, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num2 > 1:
        best2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        mask = lab2 == best2

    return mask


def build_ghost_trail_image(
    frames: Sequence[np.ndarray],
    *,
    gamma: float = 0.7,
    min_alpha: int = 35,
    ground_fraction: float = 0.22,
) -> np.ndarray:
    """Composite a ghost trail onto the LAST frame in `frames`.

    Args:
        frames: chronologically ordered RGB frames. ``frames[-1]`` is the
            current frame, used untouched as the background. ``frames[:-1]``
            are the trail, oldest first, each rendered as a fading bird-only
            cutout clipped to the current frame's pipe/ground mask.
        gamma: <1 makes older ghosts more visible (less aggressive fade).
        min_alpha: opacity floor (0-255) for the oldest ghost.
        ground_fraction: bottom fraction of the frame treated as ground.

    Returns:
        An RGB array the same shape as ``frames[-1]``.
    """
    if not frames:
        raise ValueError("frames must contain at least the current frame")

    base = frames[-1]
    H, W = base.shape[:2]
    composite = Image.fromarray(base).convert("RGBA")

    ghost_frames = frames[:-1]
    if not ghost_frames:
        return np.array(composite.convert("RGB"))

    occlusion = _make_occlusion_mask(base, ground_fraction)

    masks: List[Optional[np.ndarray]] = []
    for fr in ghost_frames:
        mask = _segment_bird(fr)
        if mask is None or not mask.any():
            masks.append(None)
            continue
        # Clip the ghost to areas the current frame doesn't already occlude,
        # so it never appears to float through a pipe/ground that has since
        # scrolled into its old position.
        clipped = mask & (~occlusion)
        masks.append(clipped if clipped.any() else None)

    L = len(masks)
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        x = (i + 1) / L  # (0, 1], oldest -> smallest, newest ghost -> 1.0
        alpha_value = int(min_alpha + (255 - min_alpha) * (x ** gamma))

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = ghost_frames[i]
        rgba[mask, 3] = alpha_value
        composite = Image.alpha_composite(composite, Image.fromarray(rgba))

    return np.array(composite.convert("RGB"))


def isolate_and_overlay_ghost_trail(image_paths: Sequence[str], out_path: str, **kwargs) -> str:
    """Convenience wrapper: load files in natural order, build the ghost trail
    for the last frame, and save it to `out_path`.

    Extra kwargs are forwarded to :func:`build_ghost_trail_image`.
    """
    frames = load_frames_from_paths(image_paths)
    result = build_ghost_trail_image(frames, **kwargs)
    Image.fromarray(result).save(out_path)
    return out_path


# Example:
# out = isolate_and_overlay_ghost_trail(
#     image_paths=sorted(glob("frames/*.png")),
#     out_path="ghost.png",
# )
# print("saved:", out)
