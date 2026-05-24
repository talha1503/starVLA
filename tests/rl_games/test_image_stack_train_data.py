import io
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from PIL import Image

STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import (  # noqa: E402
    ACTION_LABELS,
    FPS,
    _one_hot,
)
from examples.rl_games.data_conversion.convert_deadly_corridor_to_starvla_lerobot import (  # noqa: E402
    _spec as deadly_corridor_spec,
)
from examples.rl_games.data_conversion.image_stack import latest_image_from_stack  # noqa: E402
from examples.rl_games.data_conversion import lerobot_writer  # noqa: E402
from examples.rl_games.data_conversion.lerobot_writer import (  # noqa: E402
    LeRobotDatasetSpec,
    convert_lerobot_dataset,
    png_bytes,
    write_episode,
    write_metadata,
)


FLAPPY_SPEC = LeRobotDatasetSpec(
    display_name="Flappy",
    action_labels=ACTION_LABELS,
    fps=FPS,
    meta_columns=("episode_idx", "t", "action_id", "done", "reward", "prompt"),
    action=lambda row: _one_hot(int(row["action_id"])),
    row_extra=lambda row: {"action_id": int(row["action_id"])},
    include_action_id=True,
)


def _load_helper_module():
    helper_path = STARVLA_ROOT / "starVLA/dataloader/rl_games_image_stack.py"
    spec = importlib.util.spec_from_file_location("rl_games_image_stack", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (4, 4), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _write_raw_export_parquet(path: Path, rows: list[dict]) -> None:
    image_stack_type = pa.list_(
        pa.struct(
            [
                ("bytes", pa.binary()),
                ("path", pa.string()),
            ]
        )
    )
    pq.write_table(
        pa.table(
            {
                "image_stack": pa.array([row["image_stack"] for row in rows], type=image_stack_type),
                "split": pa.array([row["split"] for row in rows], type=pa.string()),
                "prompt": pa.array([row["prompt"] for row in rows], type=pa.string()),
                "action_id": pa.array([row["action_id"] for row in rows], type=pa.int64()),
                "action_text": pa.array([row["action_text"] for row in rows], type=pa.string()),
                "latency_raw_frames": pa.array([row["latency_raw_frames"] for row in rows], type=pa.int64()),
                "latency_ms": pa.array([row["latency_ms"] for row in rows], type=pa.float64()),
                "reward": pa.array([row["reward"] for row in rows], type=pa.float64()),
                "done": pa.array([row["done"] for row in rows], type=pa.bool_()),
                "episode_idx": pa.array([row["episode_idx"] for row in rows], type=pa.int64()),
                "t": pa.array([row["t"] for row in rows], type=pa.int64()),
                "seed": pa.array([row["seed"] for row in rows], type=pa.int64()),
                "deterministic": pa.array([row["deterministic"] for row in rows], type=pa.bool_()),
            }
        ),
        path,
    )


def test_rl_games_converter_writes_image_stack_video_columns(tmp_path: Path):
    newest = _png_bytes((0, 0, 255))
    rows = [
        {
            "image_bytes": newest,
            "image_stack_bytes": [
                _png_bytes((255, 0, 0)),
                _png_bytes((0, 255, 0)),
                newest,
            ],
            "action": _one_hot(1),
            "timestamp": 0.0,
            "episode_index": 0,
            "frame_index": 0,
            "task_index": 0,
            "done": False,
            "reward": 1.0,
            "action_id": 1,
        }
    ]

    write_episode(tmp_path / "episode.parquet", rows, image_stack_size=3, spec=FLAPPY_SPEC)
    write_metadata(tmp_path, episode_lengths=[1], task_prompts=["play"], image_stack_size=3, spec=FLAPPY_SPEC)

    table = pq.read_table(tmp_path / "episode.parquet")
    assert "observation.image" in table.column_names
    assert "observation.image_stack_00" in table.column_names
    assert "observation.image_stack_01" in table.column_names
    assert "observation.image_stack_02" in table.column_names
    assert table["observation.image"][0].as_py()["bytes"] == newest
    assert table["observation.image_stack_02"][0].as_py()["bytes"] == newest

    modality = json.loads((tmp_path / "meta/modality.json").read_text(encoding="utf-8"))
    assert list(modality["video"]) == ["image_stack_00", "image_stack_01", "image_stack_02"]
    assert modality["video"]["image_stack_00"]["original_key"] == "observation.image_stack_00"

    info = json.loads((tmp_path / "meta/info.json").read_text(encoding="utf-8"))
    assert "observation.image" in info["features"]
    assert "observation.image_stack_02" in info["features"]


def test_rl_games_converter_derives_single_image_from_latest_stack_frame():
    oldest = Image.new("RGB", (4, 4), (255, 0, 0))
    newest = Image.new("RGB", (4, 4), (0, 0, 255))
    row = {"image_stack": [oldest, newest]}

    image_bytes = png_bytes(latest_image_from_stack(row))

    assert image_bytes == _png_bytes((0, 0, 255))


def test_lerobot_writer_converts_stack_frames_and_manifest(monkeypatch, tmp_path: Path):
    rows = [
        {
            "episode_idx": 0,
            "t": 0,
            "action_id": 1,
            "done": False,
            "reward": 1.0,
            "prompt": "play",
            "image_stack": [
                Image.new("RGB", (4, 4), (255, 0, 0)),
                Image.new("RGB", (4, 4), (0, 0, 255)),
            ],
            "split": "train",
        },
        {
            "episode_idx": 0,
            "t": 0,
            "action_id": 0,
            "done": True,
            "reward": 0.0,
            "prompt": "play",
            "image_stack": [
                Image.new("RGB", (4, 4), (0, 255, 0)),
                Image.new("RGB", (4, 4), (255, 255, 0)),
            ],
            "split": "validation",
        },
    ]
    dataset = Dataset.from_list(rows)

    def fake_load_dataset(dataset_name, split, cache_dir=None, columns=None):
        del dataset_name, cache_dir
        selected = dataset.filter(lambda row: row["split"] == ("train" if split == "train" else "validation"))
        if columns is None:
            return selected
        return selected.select_columns(list(columns))

    monkeypatch.setattr(lerobot_writer, "load_dataset", fake_load_dataset)

    manifest = convert_lerobot_dataset(
        "fake/flappy",
        tmp_path / "flappy",
        spec=FLAPPY_SPEC,
        max_episodes=1,
    )

    table = pq.read_table(tmp_path / "flappy/data/chunk-000/episode_000000.parquet")
    assert table["observation.image"][0].as_py()["bytes"] == table["observation.image_stack_01"][0].as_py()["bytes"]
    assert manifest["image_stack_order"] == "oldest_to_newest"
    assert manifest["image_stack_source"] == "policy_observation_frame_stack"
    assert manifest["image_stack_size"] == 2


def test_lerobot_writer_converts_local_raw_parquet_without_datasets_cache(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    oldest = _png_bytes((255, 0, 0))
    newest = _png_bytes((0, 0, 255))
    val_image = _png_bytes((0, 255, 0))

    _write_raw_export_parquet(
        raw_dir / "train.parquet",
        [
            {
                "image_stack": [{"bytes": oldest, "path": None}, {"bytes": newest, "path": None}],
                "split": "train",
                "prompt": "play",
                "action_id": 1,
                "action_text": "flap",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "reward": 1.0,
                "done": True,
                "episode_idx": 0,
                "t": 0,
                "seed": 11,
                "deterministic": True,
            }
        ],
    )
    _write_raw_export_parquet(
        raw_dir / "val.parquet",
        [
            {
                "image_stack": [{"bytes": val_image, "path": None}, {"bytes": newest, "path": None}],
                "split": "val",
                "prompt": "play",
                "action_id": 0,
                "action_text": "noop",
                "latency_raw_frames": 0,
                "latency_ms": 0.0,
                "reward": 0.0,
                "done": True,
                "episode_idx": 1,
                "t": 0,
                "seed": 12,
                "deterministic": True,
            }
        ],
    )

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("local raw parquet conversion must not call datasets.load_dataset")

    monkeypatch.setattr(lerobot_writer, "load_dataset", fail_load_dataset)

    manifest = convert_lerobot_dataset(
        str(raw_dir),
        tmp_path / "flappy",
        spec=FLAPPY_SPEC,
        max_episodes=1,
    )

    table = pq.read_table(tmp_path / "flappy/data/chunk-000/episode_000000.parquet")
    assert table["observation.image"][0].as_py()["bytes"] == newest
    assert table["observation.image_stack_00"][0].as_py()["bytes"] == oldest
    assert table["observation.image_stack_01"][0].as_py()["bytes"] == newest
    assert manifest["frames"] == 1
    assert manifest["validation_frames"] == 1


def test_lerobot_writer_filters_local_raw_parquet_rows(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    image_bytes = _png_bytes((255, 255, 255))
    base_row = {
        "image_stack": [{"bytes": image_bytes, "path": None}],
        "split": "train",
        "prompt": "move",
        "action_id": 0,
        "action_text": "ATTACK",
        "latency_ms": 0.0,
        "reward": 1.0,
        "done": True,
        "t": 0,
        "seed": 11,
        "deterministic": True,
    }
    _write_raw_export_parquet(
        raw_dir / "train.parquet",
        [
            {**base_row, "episode_idx": 0, "latency_raw_frames": 1},
            {**base_row, "episode_idx": 1, "latency_raw_frames": 0},
        ],
    )
    _write_raw_export_parquet(
        raw_dir / "val.parquet",
        [
            {**base_row, "split": "val", "episode_idx": 2, "latency_raw_frames": 0},
        ],
    )

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("local raw parquet conversion must not call datasets.load_dataset")

    monkeypatch.setattr(lerobot_writer, "load_dataset", fail_load_dataset)

    manifest = convert_lerobot_dataset(
        str(raw_dir),
        tmp_path / "deadly_corridor",
        spec=deadly_corridor_spec([0]),
    )

    table = pq.read_table(tmp_path / "deadly_corridor/data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 1
    assert table["action"][0].as_py()[-1] == 1.0
    assert manifest["frames"] == 1
    assert manifest["latency_raw_frame_filter"] == [0]


def test_rl_games_train_uses_image_stack_video_keys(tmp_path: Path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "modality.json").write_text(
        json.dumps(
            {
                "video": {
                    "image": {"original_key": "observation.image"},
                    "image_stack_01": {"original_key": "observation.image_stack_01"},
                    "image_stack_00": {"original_key": "observation.image_stack_00"},
                }
            }
        ),
        encoding="utf-8",
    )
    helper = _load_helper_module()
    modality_config = {
        "video": SimpleNamespace(delta_indices=[0], modality_keys=["video.image"]),
        "state": SimpleNamespace(delta_indices=[0], modality_keys=["state.game_state"]),
    }

    updated = helper.apply_rl_games_image_stack_video_keys(
        tmp_path,
        "rl_games_flappy",
        modality_config,
    )

    assert updated["video"].modality_keys == ["video.image_stack_00", "video.image_stack_01"]
    assert updated["video"].delta_indices == [0]
    assert updated["state"].modality_keys == ["state.game_state"]
