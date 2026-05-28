#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.data_conversion.lerobot_writer import LeRobotDatasetSpec, convert_lerobot_dataset


ACTION_LABELS = [
    "TURN_NOOP",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_NOOP",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "STRAFE_NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK_NOOP",
    "ATTACK",
]
ACTION_DIM = len(ACTION_LABELS)
ACTION_LAYOUT_FACTORIZED_11 = "factorized_11"
ACTION_LAYOUT_JOINT_54 = "joint_54"
JOINT_ACTION_DIM = 54
FPS = 35


def _row_get(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row.get(name) is not None:
            return row[name]
    return default


def _keep_latency(row: dict[str, Any], latency_raw_frame_filter: list[int] | None) -> bool:
    if not latency_raw_frame_filter:
        return True
    allowed = {int(value) for value in latency_raw_frame_filter}
    return int(row["latency_raw_frames"]) in allowed


def _action_from_text(text: str) -> list[float]:
    labels = {item.strip().upper() for item in str(text).split("+")}
    return [1.0 if label in labels else 0.0 for label in ACTION_LABELS]


def _factorized_action_vector(row: dict[str, Any]) -> list[float]:
    raw_action = _row_get(row, ("action", "actions"))
    if raw_action is not None:
        values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
        if len(values) != ACTION_DIM:
            raise ValueError(f"Deadly Corridor action must have {ACTION_DIM} values, got {len(values)}")
        return [1.0 if float(value) >= 0.5 else 0.0 for value in values]
    if "action_text" in row and row.get("action_text") is not None:
        return _action_from_text(str(row["action_text"]))
    raise ValueError("Deadly Corridor dataset rows must contain `action` or `action_text`")


def _factorized_tuple(row: dict[str, Any]) -> list[int]:
    values = _factorized_action_vector(row)
    return [
        int(np.argmax(values[0:3])),
        int(np.argmax(values[3:6])),
        int(np.argmax(values[6:9])),
        int(np.argmax(values[9:11])),
    ]


def deadly_corridor_tuple_to_joint_id(action_tuple: list[int]) -> int:
    turn, move, strafe, attack = action_tuple
    return (((turn * 3) + move) * 3 + strafe) * 2 + attack


def joint_id_to_deadly_corridor_tuple(joint_id: int) -> list[int]:
    attack = joint_id % 2
    strafe = (joint_id // 2) % 3
    move = (joint_id // 6) % 3
    turn = (joint_id // 18) % 3
    return [turn, move, strafe, attack]


def _joint_action_vector(row: dict[str, Any]) -> list[float]:
    joint_id = deadly_corridor_tuple_to_joint_id(_factorized_tuple(row))
    return [1.0 if index == joint_id else 0.0 for index in range(JOINT_ACTION_DIM)]


def _action_vector(row: dict[str, Any], action_layout: str = ACTION_LAYOUT_FACTORIZED_11) -> list[float]:
    return {
        ACTION_LAYOUT_FACTORIZED_11: _factorized_action_vector,
        ACTION_LAYOUT_JOINT_54: _joint_action_vector,
    }[action_layout](row)


def _row_index(row: dict[str, Any], row_idx: int) -> tuple[int, int]:
    episode_idx = int(_row_get(row, ("episode_idx", "episode_index", "episode")))
    timestep = int(_row_get(row, ("t", "frame_index", "frame_idx", "step"), row_idx))
    return episode_idx, timestep


def _done(row: dict[str, Any]) -> bool:
    return bool(_row_get(row, ("done", "terminal", "terminated"), False))


def _reward(row: dict[str, Any]) -> float:
    return float(_row_get(row, ("reward", "rewards"), 0.0))


def _joint_action_labels() -> list[str]:
    labels = []
    for joint_id in range(JOINT_ACTION_DIM):
        turn, move, strafe, attack = joint_id_to_deadly_corridor_tuple(joint_id)
        labels.append(
            "+".join(
                [
                    ACTION_LABELS[turn],
                    ACTION_LABELS[3 + move],
                    ACTION_LABELS[6 + strafe],
                    ACTION_LABELS[9 + attack],
                ]
            )
        )
    return labels


def _spec(latency_raw_frame_filter: list[int] | None, action_layout: str) -> LeRobotDatasetSpec:
    action_labels = {
        ACTION_LAYOUT_FACTORIZED_11: ACTION_LABELS,
        ACTION_LAYOUT_JOINT_54: _joint_action_labels(),
    }[action_layout]
    return LeRobotDatasetSpec(
        display_name="Deadly Corridor",
        action_labels=action_labels,
        fps=FPS,
        meta_columns=("episode_idx", "t", "prompt", "latency_raw_frames", "latency_ms"),
        action=lambda row: _action_vector(row, action_layout=action_layout),
        row_index=_row_index,
        done=_done,
        reward=_reward,
        row_filter=lambda row: _keep_latency(row, latency_raw_frame_filter),
        load_split_retry_without_columns=True,
        empty_split_suffix=lambda: (
            f" after latency_raw_frame_filter={latency_raw_frame_filter}"
            if latency_raw_frame_filter
            else ""
        ),
        manifest_extra=lambda: {
            "latency_raw_frame_filter": latency_raw_frame_filter,
            "action_layout": action_layout,
        },
    )


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
    latency_raw_frame_filter: list[int] | None = None,
    action_layout: str = ACTION_LAYOUT_FACTORIZED_11,
) -> dict[str, Any]:
    return convert_lerobot_dataset(
        dataset_name,
        output_dir,
        spec=_spec(latency_raw_frame_filter, action_layout),
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
    parser.add_argument("--require-latency-prompt-map", "--require_latency_prompt_map", action="store_true")
    parser.add_argument("--latency-raw-frame-filter", "--latency_raw_frame_filter", default=None)
    parser.add_argument(
        "--action-layout",
        "--action_layout",
        choices=[ACTION_LAYOUT_FACTORIZED_11, ACTION_LAYOUT_JOINT_54],
        default=ACTION_LAYOUT_FACTORIZED_11,
    )
    args = parser.parse_args()

    latency_raw_frame_filter = None
    if args.latency_raw_frame_filter:
        latency_raw_frame_filter = [int(item) for item in args.latency_raw_frame_filter.split(",") if item.strip()]

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=args.require_latency_prompt_map,
        latency_raw_frame_filter=latency_raw_frame_filter,
        action_layout=args.action_layout,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
