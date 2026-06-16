from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _optional_dependency_stubs() -> dict[str, ModuleType]:
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None

    numpy = ModuleType("numpy")
    pyarrow = ModuleType("pyarrow")
    pyarrow_parquet = ModuleType("pyarrow.parquet")

    pil = ModuleType("PIL")
    pil_image = ModuleType("PIL.Image")
    pil.Image = pil_image

    return {
        "datasets": datasets,
        "numpy": numpy,
        "pyarrow": pyarrow,
        "pyarrow.parquet": pyarrow_parquet,
        "PIL": pil,
        "PIL.Image": pil_image,
    }


@pytest.fixture()
def converters(monkeypatch: pytest.MonkeyPatch) -> list[ModuleType]:
    module_names = [
        "examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot",
        "examples.rl_games.data_conversion.convert_demon_attack_to_starvla_lerobot",
        "examples.rl_games.data_conversion.convert_deadly_corridor_to_starvla_lerobot",
        "examples.rl_games.data_conversion.verify_flappy_dataset",
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    for module_name, module in _optional_dependency_stubs().items():
        monkeypatch.setitem(sys.modules, module_name, module)

    loaded = [importlib.import_module(module_name) for module_name in module_names[:3]]
    yield loaded
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def test_mixed_latency_variants_are_distinct_episode_keys(converters: list[ModuleType]) -> None:
    for converter in converters:
        first = converter._episode_key(0, 0)
        second = converter._episode_key(0, 2)

        assert first != second
        assert sorted([second, first], key=converter._episode_sort_key) == [first, second]
        assert converter._select_episode_ids(
            [first, second],
            {first: 0, second: 2},
            max_episodes=None,
            require_latency_prompt_map=True,
            episodes_per_latency=1,
        ) == [first, second]


def test_demon_attack_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    demon = converters[1]
    calls = []

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "action_id",
            "raw_reward",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        ]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(demon, "load_dataset", fake_load_dataset)

    ds = demon._load_index_split(
        "talha1503/demon_attack_mixed_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=True,
    )

    assert isinstance(ds, FakeDataset)
    assert calls[0][1]["columns"] == [
        "episode_idx",
        "t",
        "action_id",
        "done",
        "reward",
        "prompt",
        "latency",
        "latency_ms",
    ]
    assert calls[1][1]["columns"] == [
        "episode_idx",
        "t",
        "action_id",
        "done",
        "reward",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]
    assert calls[2][1]["columns"] == [
        "episode_idx",
        "t",
        "action_id",
        "done",
        "reward",
        "prompt",
        "latency",
    ]
    assert calls[3][1]["columns"] == [
        "episode_idx",
        "t",
        "action_id",
        "done",
        "reward",
        "prompt",
        "latency_raw_frames",
    ]
    assert calls[-1][1]["columns"] == [
        "episode_idx",
        "decision_step",
        "action_id",
        "raw_reward",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]


def test_demon_attack_zero_latency_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    demon = converters[1]
    calls = []

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt"]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(demon, "load_dataset", fake_load_dataset)

    ds = demon._load_index_split(
        "talha1503/demon_attack_zero_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=False,
    )

    assert isinstance(ds, FakeDataset)
    assert calls[-1][1]["columns"] == ["episode_idx", "decision_step", "action_id", "raw_reward", "prompt"]


def test_deadly_corridor_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    deadly = converters[2]
    calls = []

    class FakeDataset:
        column_names = [
            "episode_idx",
            "decision_step",
            "prompt",
            "latency_raw_frames",
            "latency_ms",
        ]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(deadly, "load_dataset", fake_load_dataset)

    ds = deadly._load_index_split(
        "talha1503/deadly_corridor_mixed_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=True,
    )

    assert isinstance(ds, FakeDataset)
    assert calls[-1][1]["columns"] == [
        "episode_idx",
        "decision_step",
        "prompt",
        "latency_raw_frames",
        "latency_ms",
    ]


def test_deadly_corridor_zero_latency_index_split_retries_canonical_hf_columns(
    monkeypatch: pytest.MonkeyPatch,
    converters: list[ModuleType],
) -> None:
    deadly = converters[2]
    calls = []

    class FakeDataset:
        column_names = ["episode_idx", "decision_step", "prompt"]

        def filter(self, fn):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        columns = kwargs.get("columns") or []
        missing = [column for column in columns if column not in FakeDataset.column_names]
        if missing:
            raise RuntimeError(f"No match for FieldRef.Name({missing[0]})")
        return FakeDataset()

    monkeypatch.setattr(deadly, "load_dataset", fake_load_dataset)

    ds = deadly._load_index_split(
        "talha1503/deadly_corridor_zero_latency_parquet",
        "train",
        cache_dir="/tmp/cache",
        want_latency=False,
    )

    assert isinstance(ds, FakeDataset)
    assert calls[-1][1]["columns"] == ["episode_idx", "decision_step", "prompt"]
