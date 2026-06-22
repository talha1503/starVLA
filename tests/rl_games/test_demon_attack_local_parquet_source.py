from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _optional_dependency_stubs() -> dict[str, ModuleType]:
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None

    pyarrow = ModuleType("pyarrow")
    pyarrow_parquet = ModuleType("pyarrow.parquet")
    return {
        "datasets": datasets,
        "pyarrow": pyarrow,
        "pyarrow.parquet": pyarrow_parquet,
    }


@pytest.fixture()
def convert_demon_attack(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "examples.rl_games.bash_scripts.gr00t.data_conversion.convert_demon_attack_to_starvla_lerobot"
    sys.modules.pop(module_name, None)
    for dependency_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, dependency_name, module)

    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_convert_demon_attack_resolves_local_parquet_directory(
    tmp_path: Path,
    convert_demon_attack: ModuleType,
) -> None:
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "part-000.parquet").touch()
    (validation_dir / "part-000.parquet").touch()

    train_files = convert_demon_attack._local_parquet_files(str(tmp_path), "train")
    validation_files = convert_demon_attack._local_parquet_files(str(tmp_path), "validation")

    assert train_files == [str(train_dir / "part-000.parquet")]
    assert validation_files == [str(validation_dir / "part-000.parquet")]


def test_convert_demon_attack_returns_absolute_local_parquet_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    convert_demon_attack: ModuleType,
) -> None:
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)
    parquet_file = train_dir / "part-000.parquet"
    parquet_file.touch()
    monkeypatch.chdir(tmp_path)

    train_files = convert_demon_attack._local_parquet_files("dataset", "train")

    assert train_files == [str(parquet_file)]


def test_convert_demon_attack_resolves_clean_v1_column_aliases(
    monkeypatch: pytest.MonkeyPatch,
    convert_demon_attack: ModuleType,
) -> None:
    monkeypatch.setattr(
        convert_demon_attack,
        "_local_parquet_columns",
        lambda dataset_name, split, dataset_source_subdir=None: {
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        },
    )

    columns = convert_demon_attack._resolve_demon_attack_columns("demon_attack_clean_v1", "train", want_latency=True)

    assert columns.frame == "decision_step"
    assert columns.reward == "raw_reward"
    assert columns.done is None
    assert columns.latency == "latency_raw_frames"
    assert columns.latency_ms == "latency_ms"


def test_convert_demon_attack_marks_last_frame_done_when_done_column_is_absent(
    convert_demon_attack: ModuleType,
) -> None:
    assert convert_demon_attack._row_done({"done": True}, "done", frame_idx=0, episode_length=3) is True
    assert convert_demon_attack._row_done({}, None, frame_idx=0, episode_length=3) is False
    assert convert_demon_attack._row_done({}, None, frame_idx=2, episode_length=3) is True
