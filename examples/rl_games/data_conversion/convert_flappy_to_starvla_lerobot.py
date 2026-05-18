#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.data_conversion.verify_flappy_dataset import build_latency_prompt_map


ACTION_LABELS = ["NOOP", "FLAP"]
ACTION_DIM = len(ACTION_LABELS)
STATE_DIM = 1
FPS = 30


def _load_split(dataset_name: str, split: str, cache_dir: str | None = None, columns: list[str] | None = None):
    if split == "train":
        try:
            return load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=columns)
        except (ValueError, KeyError):
            pass
    else:
        for candidate in ("validation", "val", "test"):
            try:
                ds = load_dataset(dataset_name, split=candidate, cache_dir=cache_dir, columns=columns)
                if len(ds) > 0:
                    return ds
            except (ValueError, KeyError):
                continue

    split_values = {"train"} if split == "train" else {"validation", "val", "test"}
    load_columns = list(columns or [])
    if "split" not in load_columns:
        load_columns.append("split")
    ds_all = load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=load_columns or None)
    return ds_all.filter(lambda row: str(row["split"]).lower() in split_values)


def _png_bytes(image: Any) -> bytes:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _one_hot(action_id: int) -> list[float]:
    if action_id < 0 or action_id >= ACTION_DIM:
        raise ValueError(f"action_id={action_id} is outside Flappy action range [0, {ACTION_DIM - 1}]")
    values = [0.0] * ACTION_DIM
    values[action_id] = 1.0
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metadata(dataset_dir: Path, *, episode_lengths: list[int], task_prompts: list[str]) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modality = {
        "state": {
            "game_state": {
                "start": 0,
                "end": STATE_DIM,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "button": {
                "start": 0,
                "end": ACTION_DIM,
                "dtype": "float32",
                "absolute": True,
                "original_key": "action",
            }
        },
        "video": {
            "image": {
                "original_key": "observation.image",
            }
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    info = {
        "codebase_version": "v2.0",
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.image": {
                "dtype": "image",
                "shape": [84, 84, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": FPS},
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ACTION_LABELS,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    episodes = [
        {"episode_index": idx, "length": int(length)}
        for idx, length in enumerate(episode_lengths)
    ]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": idx, "task": prompt} for idx, prompt in enumerate(task_prompts)],
    )


def _write_episode(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.image": pa.array(
                [{"bytes": row["image_bytes"], "path": None} for row in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "observation.state": pa.array(
                [[0.0] * STATE_DIM for _ in rows],
                type=pa.list_(pa.float32(), STATE_DIM),
            ),
            "action": pa.array(
                [row["action"] for row in rows],
                type=pa.list_(pa.float32(), ACTION_DIM),
            ),
            "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
            "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
            "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
            "action_id": pa.array([row["action_id"] for row in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
) -> dict[str, Any]:
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)

    def _convert_split(split: str, split_output_dir: Path) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        ds_meta = _load_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            columns=["episode_idx", "t", "action_id", "done", "reward", "prompt"],
        )
        if len(ds_meta) == 0:
            raise ValueError(f"{dataset_name} has no {split} rows")

        episode_indices: dict[int, list[tuple[int, int]]] = {}
        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(tqdm(ds_meta, desc=f"Indexing Flappy {split} rows")):
            episode_idx = int(row["episode_idx"])
            episode_indices.setdefault(episode_idx, []).append((int(row["t"]), row_idx))
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)

        original_episode_ids = sorted(episode_indices)
        if max_episodes is not None:
            original_episode_ids = original_episode_ids[:max_episodes]
        for episode_id in original_episode_ids:
            episode_indices[episode_id].sort(key=lambda item: item[0])

        ds_full = _load_split(dataset_name, split, cache_dir=cache_dir)
        episode_lengths: list[int] = []

        for new_episode_idx, original_episode_idx in enumerate(tqdm(original_episode_ids, desc=f"Writing Flappy {split} LeRobot episodes")):
            row_indices = [row_idx for _, row_idx in episode_indices[original_episode_idx]]
            episode = ds_full.select(row_indices)
            out_rows = []
            for frame_idx, row in enumerate(episode):
                prompt = str(row["prompt"])
                if "latency" in ds_full.column_names and row.get("latency") is not None:
                    latency_rows.append({
                        "latency": row["latency"],
                        "latency_ms": row.get("latency_ms"),
                        "prompt": prompt,
                    })
                out_rows.append({
                    "image_bytes": _png_bytes(row["image"]),
                    "action": _one_hot(int(row["action_id"])),
                    "timestamp": float(frame_idx) / FPS,
                    "episode_index": new_episode_idx,
                    "frame_index": frame_idx,
                    "task_index": prompt_to_task_index[prompt],
                    "done": bool(row["done"]),
                    "reward": float(row["reward"]),
                    "action_id": int(row["action_id"]),
                })
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            _write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
            )

        _write_metadata(split_output_dir, episode_lengths=episode_lengths, task_prompts=task_prompts)

        if latency_rows:
            try:
                latency_prompt_map = build_latency_prompt_map(latency_rows)
                (split_output_dir / "latency_prompt_map.json").write_text(
                    json.dumps(latency_prompt_map, indent=2),
                    encoding="utf-8",
                )
            except ValueError:
                if require_latency_prompt_map:
                    raise
                # Non-latency or malformed latency columns should not block single-latency training.
                pass
        elif require_latency_prompt_map:
            raise ValueError(f"{dataset_name} {split} split has no latency rows; cannot build latency_prompt_map.json")

        manifest = {
            "dataset_name": split_output_dir.name,
            "split": split,
            "source": dataset_name,
            "format": "starvla_lerobot_v2_image_parquet",
            "action_labels": ACTION_LABELS,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "episodes": len(episode_lengths),
            "frames": int(sum(episode_lengths)),
            "task_prompts": task_prompts,
        }
        (split_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    train_manifest = _convert_split("train", output_dir)
    val_manifest = _convert_split("validation", val_output_dir)
    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    (output_dir / "manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    return train_manifest


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
