import io
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
from PIL import Image

STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import (  # noqa: E402
    _one_hot,
    _write_episode,
    _write_metadata,
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

    _write_episode(tmp_path / "episode.parquet", rows, image_stack_size=3)
    _write_metadata(tmp_path, episode_lengths=[1], task_prompts=["play"], image_stack_size=3)

    table = pq.read_table(tmp_path / "episode.parquet")
    assert "observation.image" in table.column_names
    assert "observation.image_stack_00" in table.column_names
    assert "observation.image_stack_01" in table.column_names
    assert "observation.image_stack_02" in table.column_names
    assert table["observation.image_stack_02"][0].as_py()["bytes"] == newest

    modality = json.loads((tmp_path / "meta/modality.json").read_text(encoding="utf-8"))
    assert list(modality["video"]) == ["image_stack_00", "image_stack_01", "image_stack_02"]
    assert modality["video"]["image_stack_00"]["original_key"] == "observation.image_stack_00"

    info = json.loads((tmp_path / "meta/info.json").read_text(encoding="utf-8"))
    assert "observation.image" in info["features"]
    assert "observation.image_stack_02" in info["features"]


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
