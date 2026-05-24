from __future__ import annotations

import io
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as DatasetImage
from datasets import Sequence as DatasetSequence
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from examples.rl_games.data_conversion.image_stack import (
    IMAGE_STACK_ORDER,
    IMAGE_STACK_SOURCE,
    image_array,
    image_feature,
    image_stack_features,
    image_stack_modality,
    image_stack_table_columns,
)
from examples.rl_games.data_conversion.latency_prompt_map import build_latency_prompt_map


STATE_DIM = 1
LOCAL_PARQUET_BATCH_SIZE = 2048


def _default_row_index(row: Mapping[str, Any], row_idx: int) -> tuple[int, int]:
    return int(row["episode_idx"]), int(row["t"])


def _default_done(row: Mapping[str, Any]) -> bool:
    return bool(row["done"])


def _default_reward(row: Mapping[str, Any]) -> float:
    return float(row["reward"])


def _empty_split_suffix() -> str:
    return ""


def _empty_manifest_extra() -> Mapping[str, Any]:
    return {}


def _empty_row_extra(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class LeRobotDatasetSpec:
    display_name: str
    action_labels: Sequence[str]
    fps: int
    meta_columns: Sequence[str]
    action: Callable[[Mapping[str, Any]], list[float]]
    row_index: Callable[[Mapping[str, Any], int], tuple[int, int]] = _default_row_index
    done: Callable[[Mapping[str, Any]], bool] = _default_done
    reward: Callable[[Mapping[str, Any]], float] = _default_reward
    row_extra: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _empty_row_extra
    include_action_id: bool = False
    row_filter: Callable[[Mapping[str, Any]], bool] | None = None
    load_split_retry_without_columns: bool = False
    empty_split_suffix: Callable[[], str] = _empty_split_suffix
    manifest_extra: Callable[[], Mapping[str, Any]] = _empty_manifest_extra

    @property
    def action_dim(self) -> int:
        return len(self.action_labels)


def load_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    columns: list[str] | None = None,
    *,
    retry_without_columns: bool = False,
):
    split_values = {"train"} if split == "train" else {"validation", "val", "test"}
    physical_split_files = (Path(dataset_name) / "train.parquet").is_file() and (
        Path(dataset_name) / "val.parquet"
    ).is_file()

    def _maybe_filter_split(ds):
        if physical_split_files:
            return ds
        if "split" in ds.column_names:
            return ds.filter(lambda row: str(row["split"]).lower() in split_values)
        return ds

    if split == "train":
        try:
            ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=columns)
            return _maybe_filter_split(ds)
        except (ValueError, KeyError):
            if retry_without_columns and columns is not None:
                return _maybe_filter_split(load_dataset(dataset_name, split="train", cache_dir=cache_dir))
            pass
    else:
        for candidate in ("validation", "val", "test"):
            try:
                ds = load_dataset(dataset_name, split=candidate, cache_dir=cache_dir, columns=columns)
                if len(ds) > 0:
                    return _maybe_filter_split(ds)
            except (ValueError, KeyError):
                if retry_without_columns and columns is not None:
                    try:
                        ds = load_dataset(dataset_name, split=candidate, cache_dir=cache_dir)
                        if len(ds) > 0:
                            return _maybe_filter_split(ds)
                    except (ValueError, KeyError):
                        continue
                else:
                    continue

    load_columns = list(columns or [])
    if "split" not in load_columns:
        load_columns.append("split")
    try:
        ds_all = load_dataset(dataset_name, split="train", cache_dir=cache_dir, columns=load_columns or None)
    except (ValueError, KeyError):
        if retry_without_columns:
            ds_all = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        else:
            raise
    return _maybe_filter_split(ds_all)


def png_bytes(image: Any) -> bytes:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_entry_bytes(image: Any) -> bytes:
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if image_bytes is not None:
            return image_bytes
        image_path = image.get("path")
        if image_path is not None:
            return Path(image_path).read_bytes()
    return png_bytes(image)


def _decode_image_stack_as_bytes(ds):
    return ds.cast_column("image_stack", DatasetSequence(DatasetImage(decode=False)))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    task_prompts: list[str],
    image_stack_size: int,
    spec: LeRobotDatasetSpec,
) -> None:
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
                "end": spec.action_dim,
                "dtype": "float32",
                "absolute": True,
                "original_key": "action",
            }
        },
        "video": image_stack_modality(image_stack_size),
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    info = {
        "codebase_version": "v2.0",
        "fps": spec.fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.image": image_feature(spec.fps),
            **image_stack_features(image_stack_size, spec.fps),
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": [spec.action_dim],
                "names": list(spec.action_labels),
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    episodes = [{"episode_index": idx, "length": int(length)} for idx, length in enumerate(episode_lengths)]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": idx, "task": prompt} for idx, prompt in enumerate(task_prompts)])


def write_episode(path: Path, rows: list[dict[str, Any]], *, image_stack_size: int, spec: LeRobotDatasetSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "observation.image": image_array([row["image_bytes"] for row in rows]),
        **image_stack_table_columns(rows, image_stack_size),
        "observation.state": pa.array(
            [[0.0] * STATE_DIM for _ in rows],
            type=pa.list_(pa.float32(), STATE_DIM),
        ),
        "action": pa.array(
            [row["action"] for row in rows],
            type=pa.list_(pa.float32(), spec.action_dim),
        ),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
        "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
    }
    if spec.include_action_id:
        columns["action_id"] = pa.array([row["action_id"] for row in rows], type=pa.int64())
    pq.write_table(pa.table(columns), path)


def _sort_episode_indices(episode_indices: dict[int, list[tuple[int, int]]]) -> list[int]:
    original_episode_ids = sorted(episode_indices)
    for episode_id in original_episode_ids:
        episode_indices[episode_id].sort(key=lambda item: item[0])
    return original_episode_ids


def _apply_dataset_filter(ds, spec: LeRobotDatasetSpec):
    if spec.row_filter is None:
        return ds
    return ds.filter(spec.row_filter)


def _has_local_raw_splits(dataset_name: str) -> bool:
    dataset_dir = Path(dataset_name)
    return (dataset_dir / "train.parquet").is_file() and (dataset_dir / "val.parquet").is_file()


def _local_raw_split_path(dataset_name: str, split: str) -> Path:
    split_name = "train" if split == "train" else "val"
    return Path(dataset_name) / f"{split_name}.parquet"


def convert_lerobot_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    spec: LeRobotDatasetSpec,
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

    def _write_split_outputs(
        *,
        split: str,
        split_output_dir: Path,
        episode_lengths: list[int],
        task_prompts: list[str],
        image_stack_size: int,
        latency_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        write_metadata(
            split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            image_stack_size=image_stack_size,
            spec=spec,
        )

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
        elif require_latency_prompt_map:
            raise ValueError(f"{dataset_name} {split} split has no latency rows; cannot build latency_prompt_map.json")

        manifest = {
            "dataset_name": split_output_dir.name,
            "split": split,
            "source": dataset_name,
            "format": "starvla_lerobot_v2_image_parquet",
            "action_labels": list(spec.action_labels),
            "action_dim": spec.action_dim,
            "state_dim": STATE_DIM,
            "image_stack_size": image_stack_size,
            "image_stack_order": IMAGE_STACK_ORDER,
            "image_stack_source": IMAGE_STACK_SOURCE,
            **spec.manifest_extra(),
            "episodes": len(episode_lengths),
            "frames": int(sum(episode_lengths)),
            "task_prompts": task_prompts,
        }
        (split_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def _convert_local_raw_split(split: str, split_output_dir: Path) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = pq.ParquetFile(_local_raw_split_path(dataset_name, split))
        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []
        episode_lengths: list[int] = []
        image_stack_size = 0
        raw_row_idx = 0
        current_episode_idx: int | None = None
        current_rows: list[tuple[int, dict[str, Any]]] = []

        def _write_current_episode() -> None:
            nonlocal image_stack_size, current_rows
            new_episode_idx = len(episode_lengths)
            current_rows.sort(key=lambda item: item[0])
            out_rows = []
            for frame_idx, (_, row) in enumerate(current_rows):
                prompt = str(row["prompt"])
                if prompt not in prompt_to_task_index:
                    prompt_to_task_index[prompt] = len(task_prompts)
                    task_prompts.append(prompt)
                if "latency_raw_frames" in row and row.get("latency_raw_frames") is not None:
                    latency_rows.append(
                        {
                            "latency_raw_frames": row["latency_raw_frames"],
                            "latency_ms": row.get("latency_ms"),
                            "prompt": prompt,
                        }
                    )
                image_stack_bytes = [image_entry_bytes(image) for image in row["image_stack"]]
                image_stack_size = len(image_stack_bytes)
                out_rows.append(
                    {
                        "image_bytes": image_stack_bytes[-1],
                        "image_stack_bytes": image_stack_bytes,
                        "action": spec.action(row),
                        "timestamp": float(frame_idx) / spec.fps,
                        "episode_index": new_episode_idx,
                        "frame_index": frame_idx,
                        "task_index": prompt_to_task_index[prompt],
                        "done": spec.done(row),
                        "reward": spec.reward(row),
                        **spec.row_extra(row),
                    }
                )
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
                image_stack_size=image_stack_size,
                spec=spec,
            )
            current_rows = []

        with tqdm(
            total=parquet_file.metadata.num_rows,
            desc=f"Converting {spec.display_name} {split} rows",
        ) as progress:
            for batch in parquet_file.iter_batches(batch_size=LOCAL_PARQUET_BATCH_SIZE):
                for row in pa.Table.from_batches([batch]).to_pylist():
                    if spec.row_filter is None or spec.row_filter(row):
                        episode_idx, timestep = spec.row_index(row, raw_row_idx)
                        if current_episode_idx is None:
                            current_episode_idx = episode_idx
                        if episode_idx != current_episode_idx:
                            _write_current_episode()
                            if max_episodes is not None and len(episode_lengths) >= max_episodes:
                                progress.update(batch.num_rows)
                                return _write_split_outputs(
                                    split=split,
                                    split_output_dir=split_output_dir,
                                    episode_lengths=episode_lengths,
                                    task_prompts=task_prompts,
                                    image_stack_size=image_stack_size,
                                    latency_rows=latency_rows,
                                )
                            current_episode_idx = episode_idx
                        current_rows.append((timestep, row))
                    raw_row_idx += 1
                progress.update(batch.num_rows)

        if current_rows and (max_episodes is None or len(episode_lengths) < max_episodes):
            _write_current_episode()
        if not episode_lengths:
            raise ValueError(f"{dataset_name} has no {split} rows{spec.empty_split_suffix()}")
        return _write_split_outputs(
            split=split,
            split_output_dir=split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            image_stack_size=image_stack_size,
            latency_rows=latency_rows,
        )

    def _convert_split(split: str, split_output_dir: Path) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        ds_meta = load_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            columns=list(spec.meta_columns),
            retry_without_columns=spec.load_split_retry_without_columns,
        )
        ds_meta = _apply_dataset_filter(ds_meta, spec)
        if len(ds_meta) == 0:
            raise ValueError(f"{dataset_name} has no {split} rows{spec.empty_split_suffix()}")

        episode_indices: dict[int, list[tuple[int, int]]] = {}
        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(tqdm(ds_meta, desc=f"Indexing {spec.display_name} {split} rows")):
            episode_idx, timestep = spec.row_index(row, row_idx)
            episode_indices.setdefault(episode_idx, []).append((timestep, row_idx))
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)

        original_episode_ids = _sort_episode_indices(episode_indices)
        if max_episodes is not None:
            original_episode_ids = original_episode_ids[:max_episodes]

        ds_full = _apply_dataset_filter(
            load_split(
                dataset_name,
                split,
                cache_dir=cache_dir,
                retry_without_columns=spec.load_split_retry_without_columns,
            ),
            spec,
        )
        ds_full = _decode_image_stack_as_bytes(ds_full)
        image_stack_size = len(ds_full[0]["image_stack"])
        episode_lengths: list[int] = []

        for new_episode_idx, original_episode_idx in enumerate(
            tqdm(original_episode_ids, desc=f"Writing {spec.display_name} {split} LeRobot episodes")
        ):
            row_indices = [row_idx for _, row_idx in episode_indices[original_episode_idx]]
            episode = ds_full.select(row_indices)
            out_rows = []
            for frame_idx, row in enumerate(episode):
                prompt = str(row["prompt"])
                if "latency_raw_frames" in ds_full.column_names and row.get("latency_raw_frames") is not None:
                    latency_rows.append(
                        {
                            "latency_raw_frames": row["latency_raw_frames"],
                            "latency_ms": row.get("latency_ms"),
                            "prompt": prompt,
                        }
                    )
                image_stack_bytes = [image_entry_bytes(image) for image in row["image_stack"]]
                out_rows.append(
                    {
                        "image_bytes": image_stack_bytes[-1],
                        "image_stack_bytes": image_stack_bytes,
                        "action": spec.action(row),
                        "timestamp": float(frame_idx) / spec.fps,
                        "episode_index": new_episode_idx,
                        "frame_index": frame_idx,
                        "task_index": prompt_to_task_index[prompt],
                        "done": spec.done(row),
                        "reward": spec.reward(row),
                        **spec.row_extra(row),
                    }
                )
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
                image_stack_size=image_stack_size,
                spec=spec,
            )

        return _write_split_outputs(
            split=split,
            split_output_dir=split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            image_stack_size=image_stack_size,
            latency_rows=latency_rows,
        )

    split_converter = _convert_local_raw_split if _has_local_raw_splits(dataset_name) else _convert_split
    train_manifest = split_converter("train", output_dir)
    val_manifest = split_converter("validation", val_output_dir)
    train_manifest["validation_dataset_name"] = val_output_dir.name
    train_manifest["validation_episodes"] = val_manifest["episodes"]
    train_manifest["validation_frames"] = val_manifest["frames"]
    (output_dir / "manifest.json").write_text(json.dumps(train_manifest, indent=2), encoding="utf-8")
    return train_manifest
