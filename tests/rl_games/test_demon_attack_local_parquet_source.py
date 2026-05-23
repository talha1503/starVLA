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
    module_name = "examples.rl_games.data_conversion.convert_demon_attack_to_starvla_lerobot"
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
