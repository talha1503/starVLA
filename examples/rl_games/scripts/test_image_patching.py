#!/usr/bin/env python
"""Build 5 sanity-check ghost-trail composites for image_patching.py.

For `--num-sets` random positions in a single episode of
latency-sensitive-bench/flappy_200ep (latency 0), this takes the 8 frames
ending at that position (the last of which is the "current" frame, the
preceding 7 are the ghost trail), runs build_ghost_trail_image on them, and
saves the inputs + composite to disk for visual inspection.

Usage:
    python examples/rl_games/scripts/test_image_patching.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.image_patching import build_ghost_trail_image

DATASET_NAME = "latency-sensitive-bench/flappy_200ep"
DATASET_SUBDIR = "flappy_fix_latency_0_200ep"
TRAIL_LEN = 7  # ghosts
WINDOW_LEN = TRAIL_LEN + 1  # ghosts + current frame


def load_episode_rows(episode_idx: int, cache_dir: str | None):
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_NAME,
        split="train",
        data_dir=DATASET_SUBDIR,
        verification_mode="no_checks",
        cache_dir=cache_dir,
        columns=["episode_idx", "decision_step", "image"],
    )
    ds = ds.filter(lambda ex: ex["episode_idx"] == episode_idx)
    ds = ds.sort("decision_step")
    return ds


def make_contact_sheet(frames: list[np.ndarray], out_path: Path, cols: int = 4) -> None:
    H, W = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * H, cols * W, 3), 255, dtype=np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = fr
    Image.fromarray(sheet).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--num-sets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent / "ghost_trail_test_outputs"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episode {args.episode_idx} from {DATASET_NAME} ({DATASET_SUBDIR})...")
    ds = load_episode_rows(args.episode_idx, args.cache_dir)
    n = len(ds)
    print(f"Episode {args.episode_idx} has {n} rows.")
    if n <= WINDOW_LEN:
        raise ValueError(f"Episode {args.episode_idx} only has {n} rows; need > {WINDOW_LEN}.")

    rng = random.Random(args.seed)
    positions = rng.sample(range(WINDOW_LEN - 1, n), args.num_sets)

    for set_idx, pos in enumerate(positions):
        window_idxs = list(range(pos - (WINDOW_LEN - 1), pos + 1))
        rows = ds.select(window_idxs)
        steps = list(rows["decision_step"])
        frames = [np.array(img.convert("RGB")) for img in rows["image"]]

        set_dir = out_dir / f"set_{set_idx}_pos_{pos}"
        set_dir.mkdir(parents=True, exist_ok=True)

        for j, fr in enumerate(frames):
            tag = "current" if j == len(frames) - 1 else f"ghost_t-{len(frames) - 1 - j}"
            Image.fromarray(fr).save(set_dir / f"frame_{j:02d}_{tag}_step{steps[j]}.png")

        make_contact_sheet(frames, set_dir / "contact_sheet_inputs.png")

        composite = build_ghost_trail_image(frames)
        Image.fromarray(composite).save(set_dir / "ghost_trail_composite.png")

        print(f"set {set_idx}: pos={pos} decision_steps={steps} -> {set_dir}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
