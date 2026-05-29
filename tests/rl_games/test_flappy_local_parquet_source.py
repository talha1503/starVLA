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
def flappy_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    module_names = (
        "examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot",
        "examples.rl_games.data_conversion.verify_flappy_dataset",
    )
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, module_name, module)

    convert_flappy = importlib.import_module("examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot")
    verify_flappy = importlib.import_module("examples.rl_games.data_conversion.verify_flappy_dataset")
    yield convert_flappy, verify_flappy
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def test_convert_flappy_resolves_local_parquet_directory(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    convert_flappy, _ = flappy_modules
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "part-000.parquet").touch()
    (validation_dir / "part-000.parquet").touch()

    train_files = convert_flappy._local_parquet_files(str(tmp_path), "train")
    validation_files = convert_flappy._local_parquet_files(str(tmp_path), "validation")

    assert train_files == [str(train_dir / "part-000.parquet")]
    assert validation_files == [str(validation_dir / "part-000.parquet")]


def test_verify_flappy_resolves_local_parquet_directory(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, verify_flappy = flappy_modules
    parquet_file = tmp_path / "part-000.parquet"
    parquet_file.touch()

    files = verify_flappy._local_parquet_files(str(tmp_path))

    assert files == [str(parquet_file)]


def test_verify_flappy_prefers_train_parquet_files(
    tmp_path: Path,
    flappy_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, verify_flappy = flappy_modules
    train_file = tmp_path / "train-00000.parquet"
    validation_file = tmp_path / "validation-00000.parquet"
    train_file.touch()
    validation_file.touch()

    files = verify_flappy._local_parquet_files(str(tmp_path))

    assert files == [str(train_file)]
