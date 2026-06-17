#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.data_conversion.verify_flappy_dataset import build_latency_prompt_map, latency_id_from_row


ACTION_LABELS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
]
ACTION_DIM = len(ACTION_LABELS)
FACTORIZED_11_ACTION_LABELS = [
    "TURN_NONE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_NONE",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "STRAFE_NONE",
    "STRAFE_LEFT",
    "STRAFE_RIGHT",
    "ATTACK_OFF",
    "ATTACK_ON",
]
FACTORIZED_11_ACTION_DIM = len(FACTORIZED_11_ACTION_LABELS)
ACTION_LAYOUT_MULTIBINARY_7 = "multibinary_7"
ACTION_LAYOUT_FACTORIZED_11 = "factorized_11"
BRIDGE_ACTION_DIM = 7
STATE_DIM = 1
BRIDGE_STATE_DIM = 7
FPS = 35
LATENCY_FRAMESKIP = 4
EpisodeKey = int | tuple[int, int]


class DeadlyCorridorColumns(NamedTuple):
    frame: str
    reward: str
    done: str | None
    latency: str | None
    latency_ms: str | None


def _local_parquet_files(dataset_name: str, split: str, dataset_source_subdir: str | None = None) -> list[str] | None:
    dataset_path = Path(dataset_name).expanduser()
    if not dataset_path.exists():
        return None
    if dataset_source_subdir not in (None, ""):
        dataset_path = dataset_path / str(dataset_source_subdir)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset_source_subdir={dataset_source_subdir!r} does not exist under {dataset_name!r}"
            )
    dataset_path = dataset_path.resolve(strict=True)
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"dataset_name={dataset_name!r} exists but is not a parquet file")
        return [str(dataset_path)]

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"dataset_name={dataset_name!r} exists but contains no parquet files")

    split_markers = {"train"} if split == "train" else {"validation", "val", "test"}
    split_files = [
        parquet_file
        for parquet_file in parquet_files
        if any(marker in part.lower() for marker in split_markers for part in parquet_file.relative_to(dataset_path).parts)
    ]
    return [str(parquet_file) for parquet_file in (split_files or parquet_files)]


def _local_parquet_columns(dataset_name: str, split: str, dataset_source_subdir: str | None = None) -> set[str] | None:
    local_files = _local_parquet_files(dataset_name, split, dataset_source_subdir)
    if local_files is None:
        return None
    columns: set[str] = set()
    for parquet_file in local_files:
        columns.update(pq.read_schema(parquet_file).names)
    return columns


def _resolve_required_column(available: set[str] | None, names: tuple[str, ...], label: str) -> str:
    if available is None:
        return names[0]
    for name in names:
        if name in available:
            return name
    raise ValueError(
        f"Deadly Corridor dataset is missing required {label} column. "
        f"Expected one of {names}; available={sorted(available)}"
    )


def _resolve_optional_column(available: set[str] | None, names: tuple[str, ...]) -> str | None:
    if available is None:
        return names[0]
    for name in names:
        if name in available:
            return name
    return None


def _resolve_deadly_corridor_columns(
    dataset_name: str,
    split: str,
    want_latency: bool,
    dataset_source_subdir: str | None = None,
) -> DeadlyCorridorColumns:
    available = _local_parquet_columns(dataset_name, split, dataset_source_subdir)
    return DeadlyCorridorColumns(
        frame=_resolve_required_column(available, ("t", "decision_step", "frame_index", "frame_idx", "step"), "frame index"),
        reward=_resolve_required_column(available, ("reward", "raw_reward", "rewards"), "reward"),
        done=_resolve_optional_column(available, ("done", "terminal", "terminated")),
        latency=_resolve_optional_column(available, ("latency", "latency_raw_frames")) if want_latency else None,
        latency_ms=_resolve_optional_column(available, ("latency_ms",)) if want_latency else None,
    )


def _deadly_corridor_column_candidates(
    dataset_name: str,
    split: str,
    want_latency: bool,
    dataset_source_subdir: str | None = None,
) -> list[DeadlyCorridorColumns]:
    available = _local_parquet_columns(dataset_name, split, dataset_source_subdir)
    if available is not None:
        return [_resolve_deadly_corridor_columns(dataset_name, split, want_latency, dataset_source_subdir)]

    base_candidates = (
        DeadlyCorridorColumns(frame="t", reward="reward", done="done", latency="latency", latency_ms="latency_ms"),
        DeadlyCorridorColumns(frame="decision_step", reward="raw_reward", done=None, latency="latency_raw_frames", latency_ms="latency_ms"),
        DeadlyCorridorColumns(frame="decision_step", reward="raw_reward", done=None, latency="latency", latency_ms="latency_ms"),
        DeadlyCorridorColumns(frame="t", reward="reward", done="done", latency="latency_raw_frames", latency_ms="latency_ms"),
        DeadlyCorridorColumns(frame="frame_index", reward="rewards", done="done", latency="latency", latency_ms="latency_ms"),
        DeadlyCorridorColumns(frame="step", reward="reward", done="terminated", latency="latency", latency_ms="latency_ms"),
    )
    if want_latency:
        return list(base_candidates)
    return [
        DeadlyCorridorColumns(frame=columns.frame, reward=columns.reward, done=columns.done, latency=None, latency_ms=None)
        for columns in base_candidates
    ]


def _load_hf_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_source_subdir: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    columns: list[str] | None = None,
):
    load_kwargs = {"split": split, "cache_dir": cache_dir, "columns": columns}
    if dataset_source_subdir not in (None, ""):
        load_kwargs["data_dir"] = str(dataset_source_subdir)
    if dataset_config_name not in (None, ""):
        return load_dataset(dataset_name, dataset_config_name, **load_kwargs)
    return load_dataset(dataset_name, **load_kwargs)


def _load_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    columns: list[str] | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
):
    split_values = {"train"} if split == "train" else {"validation", "val", "test"}

    def _filter_internal_split(ds):
        if "split" in ds.column_names:
            return ds.filter(lambda row: str(row["split"]).lower() in split_values)
        return ds

    local_files = _local_parquet_files(dataset_name, split, dataset_source_subdir)
    if local_files is not None:
        load_columns = list(columns) if columns is not None else None
        if load_columns is not None and "split" not in load_columns:
            load_columns.append("split")
        try:
            ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=load_columns)
        except (ValueError, KeyError):
            if columns is None:
                raise
            ds = load_dataset("parquet", data_files=local_files, split="train", cache_dir=cache_dir, columns=columns)
        return _filter_internal_split(ds)

    if split == "train":
        try:
            ds = _load_hf_dataset(
                dataset_name, dataset_config_name, dataset_source_subdir,
                split="train", cache_dir=cache_dir, columns=columns,
            )
            return _filter_internal_split(ds)
        except (ValueError, KeyError):
            if columns is not None:
                return _filter_internal_split(
                    _load_hf_dataset(
                        dataset_name, dataset_config_name, dataset_source_subdir,
                        split="train", cache_dir=cache_dir,
                    )
                )
            pass
    else:
        for candidate in ("validation", "val", "test"):
            try:
                ds = _load_hf_dataset(
                    dataset_name, dataset_config_name, dataset_source_subdir,
                    split=candidate, cache_dir=cache_dir, columns=columns,
                )
                if len(ds) > 0:
                    return ds
            except (ValueError, KeyError):
                if columns is not None:
                    try:
                        ds = _load_hf_dataset(
                            dataset_name, dataset_config_name, dataset_source_subdir,
                            split=candidate, cache_dir=cache_dir,
                        )
                        if len(ds) > 0:
                            return _filter_internal_split(ds)
                    except (ValueError, KeyError):
                        continue
                else:
                    continue

    load_columns = list(columns or [])
    if "split" not in load_columns:
        load_columns.append("split")
    try:
        ds_all = _load_hf_dataset(
            dataset_name, dataset_config_name, dataset_source_subdir,
            split="train", cache_dir=cache_dir, columns=load_columns or None,
        )
    except (ValueError, KeyError):
        ds_all = _load_hf_dataset(
            dataset_name, dataset_config_name, dataset_source_subdir,
            split="train", cache_dir=cache_dir,
        )
    return ds_all.filter(lambda row: str(row["split"]).lower() in split_values)


def _row_get(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row.get(name) is not None:
            return row[name]
    return default


def _row_latency(row: dict[str, Any], latency_column: str | None) -> int | None:
    return latency_id_from_row(
        row,
        frameskip=LATENCY_FRAMESKIP,
        latency_column=latency_column,
        default_latency=None,
    )


def _filter_latency(ds, latency_filter: list[int] | None, *, latency_column: str | None):
    if not latency_filter:
        return ds
    if latency_column is None or latency_column not in ds.column_names:
        raise ValueError("latency_filter was requested, but the dataset has no latency column")
    allowed = {int(value) for value in latency_filter}
    return ds.filter(lambda row: _row_latency(row, latency_column) in allowed)


def _episode_key(episode_idx: int, latency: int | None) -> EpisodeKey:
    return (episode_idx, int(latency)) if latency is not None else episode_idx


def _episode_sort_key(episode_key: EpisodeKey) -> tuple[int, int]:
    if isinstance(episode_key, tuple):
        return episode_key
    return (episode_key, -1)


def _load_index_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None,
    *,
    want_latency: bool,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
):
    candidate_columns = _deadly_corridor_column_candidates(
        dataset_name,
        split,
        want_latency=want_latency,
        dataset_source_subdir=dataset_source_subdir,
    )
    last_error: Exception | None = None
    for deadly_corridor_columns in candidate_columns:
        columns = ["episode_idx", deadly_corridor_columns.frame, "prompt"]
        if deadly_corridor_columns.latency is not None:
            columns.append(deadly_corridor_columns.latency)
        if deadly_corridor_columns.latency_ms is not None:
            columns.append(deadly_corridor_columns.latency_ms)
        try:
            return (
                _load_split(
                    dataset_name,
                    split,
                    cache_dir=cache_dir,
                    columns=columns,
                    dataset_config_name=dataset_config_name,
                    dataset_source_subdir=dataset_source_subdir,
                ),
                deadly_corridor_columns,
            )
        except Exception as exc:
            last_error = exc
            if len(candidate_columns) == 1:
                raise

    try:
        ds = _load_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
        )
    except Exception:
        if last_error is not None:
            raise last_error
        raise
    available = set(ds.column_names)
    return (
        ds,
        DeadlyCorridorColumns(
            frame=_resolve_required_column(available, ("t", "decision_step", "frame_index", "frame_idx", "step"), "frame index"),
            reward=_resolve_required_column(available, ("reward", "raw_reward", "rewards"), "reward"),
            done=_resolve_optional_column(available, ("done", "terminal", "terminated")),
            latency=_resolve_optional_column(available, ("latency", "latency_raw_frames")) if want_latency else None,
            latency_ms=_resolve_optional_column(available, ("latency_ms",)) if want_latency else None,
        ),
    )


def _row_done(row: dict[str, Any], done_column: str | None, *, frame_idx: int, episode_length: int) -> bool:
    if done_column is not None and done_column in row:
        return bool(row[done_column])
    return frame_idx == episode_length - 1


def _png_bytes(image: Any) -> bytes:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_action_carrier(action_carrier: str) -> str:
    carrier = str(action_carrier or "native").lower()
    if carrier in {"native", "bridge"}:
        return carrier
    raise ValueError(f"Unsupported action_carrier={action_carrier!r}; expected native or bridge")


def _normalize_action_layout(action_layout: str) -> str:
    layout = str(action_layout or ACTION_LAYOUT_MULTIBINARY_7).lower()
    if layout in {ACTION_LAYOUT_MULTIBINARY_7, ACTION_LAYOUT_FACTORIZED_11}:
        return layout
    raise ValueError(f"Unsupported action_layout={action_layout!r}; expected multibinary_7 or factorized_11")


def _active_action_dim(action_layout: str) -> int:
    return FACTORIZED_11_ACTION_DIM if _normalize_action_layout(action_layout) == ACTION_LAYOUT_FACTORIZED_11 else ACTION_DIM


def _action_dim(action_carrier: str, action_layout: str = ACTION_LAYOUT_MULTIBINARY_7) -> int:
    if _normalize_action_carrier(action_carrier) == "bridge":
        return BRIDGE_ACTION_DIM
    return _active_action_dim(action_layout)


def _action_labels(action_carrier: str, action_layout: str = ACTION_LAYOUT_MULTIBINARY_7) -> list[str]:
    # Deadly Corridor already uses the 7D semantic bridge carrier natively.
    if _normalize_action_carrier(action_carrier) == "bridge":
        return list(ACTION_LABELS)
    if _normalize_action_layout(action_layout) == ACTION_LAYOUT_FACTORIZED_11:
        return list(FACTORIZED_11_ACTION_LABELS)
    return list(ACTION_LABELS)


def _state_dim(action_carrier: str) -> int:
    return BRIDGE_STATE_DIM if _normalize_action_carrier(action_carrier) == "bridge" else STATE_DIM


def _state_labels(action_carrier: str) -> list[str]:
    if _normalize_action_carrier(action_carrier) == "native":
        return ["state"]
    return [f"BRIDGE_STATE_{idx}" for idx in range(BRIDGE_STATE_DIM)]


def _action_from_text(text: str) -> list[float]:
    normalized = str(text).upper()
    return [1.0 if label in normalized else 0.0 for label in ACTION_LABELS]


def _factorized_one_hot(action_tuple: Any) -> list[float]:
    turn, move, strafe, attack = [int(value) for value in action_tuple]
    values = [0.0] * FACTORIZED_11_ACTION_DIM
    values[turn] = 1.0
    values[3 + move] = 1.0
    values[6 + strafe] = 1.0
    values[9 + attack] = 1.0
    return values


def _action_vector(row: dict[str, Any], action_layout: str = ACTION_LAYOUT_MULTIBINARY_7) -> list[float]:
    if _normalize_action_layout(action_layout) == ACTION_LAYOUT_FACTORIZED_11:
        if "action_tuple" in row and row["action_tuple"] is not None:
            return _factorized_one_hot(row["action_tuple"])
        raw_action = _row_get(row, ("action", "actions"))
        if raw_action is not None:
            values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
            if len(values) != FACTORIZED_11_ACTION_DIM:
                raise ValueError(
                    f"Deadly Corridor factorized action must have {FACTORIZED_11_ACTION_DIM} values, got {len(values)}"
                )
            return [float(value) for value in values]
        raise ValueError("Deadly Corridor factorized rows must contain `action_tuple` or 11D `action`")

    raw_action = _row_get(row, ("action", "actions"))
    if raw_action is not None:
        values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
        if len(values) != ACTION_DIM:
            raise ValueError(f"Deadly Corridor action must have {ACTION_DIM} values, got {len(values)}")
        return [1.0 if float(value) >= 0.5 else 0.0 for value in values]
    if "action_text" in row and row.get("action_text") is not None:
        return _action_from_text(str(row["action_text"]))
    raise ValueError("Deadly Corridor dataset rows must contain `action` or `action_text`")


def _select_episode_ids(
    episode_ids: list[EpisodeKey],
    episode_latencies: dict[EpisodeKey, int],
    *,
    max_episodes: int | None,
    require_latency_prompt_map: bool,
    episodes_per_latency: int | None = None,
) -> list[EpisodeKey]:
    if episodes_per_latency is not None:
        if not episode_latencies:
            raise ValueError("episodes_per_latency was requested, but no episode latency metadata is available")
        selected: list[EpisodeKey] = []
        for latency in sorted(set(episode_latencies.values())):
            latency_episode_ids = [episode_id for episode_id in episode_ids if episode_latencies.get(episode_id) == latency]
            selected.extend(latency_episode_ids[: int(episodes_per_latency)])
        return selected

    if max_episodes is None:
        return episode_ids
    if not require_latency_prompt_map:
        return episode_ids[:max_episodes]

    selected: list[EpisodeKey] = []
    selected_set: set[EpisodeKey] = set()
    for latency in sorted(set(episode_latencies.values())):
        for episode_id in episode_ids:
            if episode_id in selected_set:
                continue
            if episode_latencies.get(episode_id) == latency:
                selected.append(episode_id)
                selected_set.add(episode_id)
                break

    target_count = max(max_episodes, len(selected))
    for episode_id in episode_ids:
        if len(selected) >= target_count:
            break
        if episode_id not in selected_set:
            selected.append(episode_id)
            selected_set.add(episode_id)
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metadata(
    dataset_dir: Path,
    *,
    episode_lengths: list[int],
    task_prompts: list[str],
    action_dim: int,
    action_labels: list[str],
    state_dim: int,
    state_labels: list[str],
) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modality = {
        "state": {
            "game_state": {
                "start": 0,
                "end": state_dim,
                "dtype": "float32",
                "absolute": True,
                "original_key": "observation.state",
            }
        },
        "action": {
            "button": {
                "start": 0,
                "end": action_dim,
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
                "shape": [state_dim],
                "names": state_labels,
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": action_labels,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "latency": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    episodes = [{"episode_index": idx, "length": int(length)} for idx, length in enumerate(episode_lengths)]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": idx, "task": prompt} for idx, prompt in enumerate(task_prompts)])


def _write_episode(path: Path, rows: list[dict[str, Any]], *, action_dim: int, state_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "observation.image": pa.array(
                [{"bytes": row["image_bytes"], "path": None} for row in rows],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "observation.state": pa.array(
                [[0.0] * state_dim for _ in rows],
                type=pa.list_(pa.float32(), state_dim),
            ),
            "action": pa.array(
                [row["action"] for row in rows],
                type=pa.list_(pa.float32(), action_dim),
            ),
            "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float64()),
            "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
            "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
            "latency": pa.array([row["latency"] for row in rows], type=pa.int64()),
            "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
            "reward": pa.array([row["reward"] for row in rows], type=pa.float32()),
        }
    )
    pq.write_table(table, path)


def convert_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    max_episodes: int | None = None,
    force: bool = False,
    require_latency_prompt_map: bool = False,
    latency_filter: list[int] | None = None,
    episodes_per_latency: int | None = None,
    action_carrier: str = "native",
    action_layout: str = ACTION_LAYOUT_MULTIBINARY_7,
) -> dict[str, Any]:
    action_carrier = _normalize_action_carrier(action_carrier)
    action_layout = _normalize_action_layout(action_layout)
    action_dim = _action_dim(action_carrier, action_layout)
    action_labels = _action_labels(action_carrier, action_layout)
    active_action_dim = _active_action_dim(action_layout)
    state_dim = _state_dim(action_carrier)
    state_labels = _state_labels(action_carrier)
    val_output_dir = output_dir.with_name(f"{output_dir.name}__val")
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if val_output_dir.exists() and force:
        shutil.rmtree(val_output_dir)

    def _convert_split(split: str, split_output_dir: Path) -> dict[str, Any]:
        split_output_dir.mkdir(parents=True, exist_ok=True)
        want_latency = bool(require_latency_prompt_map or latency_filter)
        ds_meta, deadly_corridor_columns = _load_index_split(
            dataset_name,
            split,
            cache_dir=cache_dir,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
            want_latency=want_latency,
        )
        ds_meta = _filter_latency(
            ds_meta,
            latency_filter,
            latency_column=deadly_corridor_columns.latency,
        )
        if len(ds_meta) == 0:
            suffix = f" after latency_filter={latency_filter}" if latency_filter else ""
            raise ValueError(f"{dataset_name} has no {split} rows{suffix}")

        episode_indices: dict[EpisodeKey, list[tuple[int, int]]] = {}
        episode_latencies: dict[EpisodeKey, int] = {}
        prompt_to_task_index: dict[str, int] = {}
        task_prompts: list[str] = []
        latency_rows: list[dict[str, Any]] = []

        for row_idx, row in enumerate(tqdm(ds_meta, desc=f"Indexing Deadly Corridor {split} rows")):
            episode_idx = int(_row_get(row, ("episode_idx", "episode_index", "episode")))
            timestep = int(row[deadly_corridor_columns.frame])
            latency = _row_latency(row, deadly_corridor_columns.latency)
            episode_key = _episode_key(episode_idx, latency)
            episode_indices.setdefault(episode_key, []).append((timestep, row_idx))
            if require_latency_prompt_map and latency is not None:
                episode_latencies.setdefault(episode_key, latency)
            prompt = str(row["prompt"])
            if prompt not in prompt_to_task_index:
                prompt_to_task_index[prompt] = len(task_prompts)
                task_prompts.append(prompt)

        original_episode_ids = sorted(episode_indices, key=_episode_sort_key)
        original_episode_ids = _select_episode_ids(
            original_episode_ids,
            episode_latencies,
            max_episodes=max_episodes,
            require_latency_prompt_map=require_latency_prompt_map,
            episodes_per_latency=episodes_per_latency,
        )
        for episode_id in original_episode_ids:
            episode_indices[episode_id].sort(key=lambda item: item[0])

        ds_full = _filter_latency(
            _load_split(
                dataset_name,
                split,
                cache_dir=cache_dir,
                dataset_config_name=dataset_config_name,
                dataset_source_subdir=dataset_source_subdir,
            ),
            latency_filter,
            latency_column=deadly_corridor_columns.latency,
        )
        episode_lengths: list[int] = []

        for new_episode_idx, original_episode_idx in enumerate(
            tqdm(original_episode_ids, desc=f"Writing Deadly Corridor {split} LeRobot episodes")
        ):
            row_indices = [row_idx for _, row_idx in episode_indices[original_episode_idx]]
            episode = ds_full.select(row_indices)
            out_rows = []
            for frame_idx, row in enumerate(episode):
                prompt = str(row["prompt"])
                latency = _row_latency(row, deadly_corridor_columns.latency)
                if latency is not None:
                    latency_ms = row.get(deadly_corridor_columns.latency_ms) if deadly_corridor_columns.latency_ms is not None else None
                    latency_rows.append(
                        {
                            "latency": latency,
                            "latency_ms": latency_ms,
                            "prompt": prompt,
                        }
                    )
                out_rows.append(
                    {
                        "image_bytes": _png_bytes(_row_get(row, ("image", "observation.image", "obs"))),
                        "action": _action_vector(row, action_layout),
                        "timestamp": float(frame_idx) / FPS,
                        "episode_index": new_episode_idx,
                        "frame_index": frame_idx,
                        "task_index": prompt_to_task_index[prompt],
                        "done": _row_done(
                            row,
                            deadly_corridor_columns.done,
                            frame_idx=frame_idx,
                            episode_length=len(episode),
                        ),
                        "reward": float(row[deadly_corridor_columns.reward]),
                    }
                )
            episode_lengths.append(len(out_rows))
            episode_chunk = new_episode_idx // 1000
            _write_episode(
                split_output_dir / f"data/chunk-{episode_chunk:03d}/episode_{new_episode_idx:06d}.parquet",
                out_rows,
                action_dim=action_dim,
                state_dim=state_dim,
            )

        _write_metadata(
            split_output_dir,
            episode_lengths=episode_lengths,
            task_prompts=task_prompts,
            action_dim=action_dim,
            action_labels=action_labels,
            state_dim=state_dim,
            state_labels=state_labels,
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
            "source_config": dataset_config_name,
            "source_subdir": dataset_source_subdir,
            "format": "starvla_lerobot_v2_image_parquet",
            "action_labels": action_labels,
            "action_dim": action_dim,
            "active_action_dim": active_action_dim,
            "action_carrier": action_carrier,
            "action_layout": action_layout,
            "bridge_action_dim": BRIDGE_ACTION_DIM if action_carrier == "bridge" else None,
            "state_dim": state_dim,
            "active_state_dim": STATE_DIM,
            "state_carrier": action_carrier,
            "latency_metadata": True,
            "latency_filter": latency_filter,
            "episodes_per_latency": int(episodes_per_latency) if episodes_per_latency is not None else None,
            "max_episodes": int(max_episodes) if max_episodes is not None else None,
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
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-latency-prompt-map", "--require_latency_prompt_map", action="store_true")
    parser.add_argument("--latency-filter", "--latency_filter", default=None)
    parser.add_argument("--episodes-per-latency", "--episodes_per_latency", type=int, default=None)
    parser.add_argument("--action-carrier", "--action_carrier", choices=["native", "bridge"], default="native")
    parser.add_argument(
        "--action-layout",
        "--action_layout",
        choices=[ACTION_LAYOUT_MULTIBINARY_7, ACTION_LAYOUT_FACTORIZED_11],
        default=ACTION_LAYOUT_MULTIBINARY_7,
    )
    args = parser.parse_args()

    latency_filter = None
    if args.latency_filter:
        latency_filter = [int(item) for item in args.latency_filter.split(",") if item.strip()]

    manifest = convert_dataset(
        args.dataset_name,
        Path(args.output_dir),
        cache_dir=args.cache_dir,
        dataset_config_name=args.dataset_config_name,
        dataset_source_subdir=args.dataset_source_subdir,
        max_episodes=args.max_episodes,
        force=args.force,
        require_latency_prompt_map=args.require_latency_prompt_map,
        latency_filter=latency_filter,
        episodes_per_latency=args.episodes_per_latency,
        action_carrier=args.action_carrier,
        action_layout=args.action_layout,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
