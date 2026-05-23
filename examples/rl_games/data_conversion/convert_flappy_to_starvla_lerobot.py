#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.data_conversion.lerobot_writer import LeRobotDatasetSpec, convert_lerobot_dataset


ACTION_LABELS = ["NOOP", "FLAP"]
ACTION_DIM = len(ACTION_LABELS)
FPS = 30


def _one_hot(action_id: int) -> list[float]:
    if action_id < 0 or action_id >= ACTION_DIM:
        raise ValueError(f"action_id={action_id} is outside Flappy action range [0, {ACTION_DIM - 1}]")
    values = [0.0] * ACTION_DIM
    values[action_id] = 1.0
    return values


def _action(row: dict[str, Any]) -> list[float]:
    return _one_hot(int(row["action_id"]))


def _row_extra(row: dict[str, Any]) -> dict[str, int]:
    return {"action_id": int(row["action_id"])}


def _spec() -> LeRobotDatasetSpec:
    return LeRobotDatasetSpec(
        display_name="Flappy",
        action_labels=ACTION_LABELS,
        fps=FPS,
        meta_columns=("episode_idx", "t", "action_id", "done", "reward", "prompt"),
        action=_action,
        row_extra=_row_extra,
        include_action_id=True,
    )


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
) -> dict[str, Any]:
    return convert_lerobot_dataset(
        dataset_name,
        output_dir,
        spec=_spec(),
        cache_dir=cache_dir,
        max_episodes=max_episodes,
        force=force,
        require_latency_prompt_map=require_latency_prompt_map,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=False,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
