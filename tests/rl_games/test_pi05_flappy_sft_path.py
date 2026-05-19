from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import yaml

from examples.rl_games.scripts import setup_training_assets
from starVLA.training.rl_games.action_spec import apply_action_spec
from starVLA.training.rl_games.alias import MODEL_ALIAS_TO_FRAMEWORK, apply_model_alias


REPO_ROOT = Path(__file__).resolve().parents[2]


class ExpectedExperimentConfig(TypedDict):
    name: str
    mode: str
    run_id: str
    source_hf: str
    converted_name: str
    latencies: list[int]


EXPECTED_PI05_FLAPPY_EXPERIMENTS: dict[str, ExpectedExperimentConfig] = {
    "single": {
        "name": "pi05_flappy_single.yaml",
        "mode": "single",
        "run_id": "pi05_flappy_single",
        "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
        "converted_name": "flappy_train",
        "latencies": [0],
    },
    "mixed_latency": {
        "name": "pi05_flappy_mixed_latency.yaml",
        "mode": "mixed_latency",
        "run_id": "pi05_flappy_mixed_latency",
        "source_hf": "talha1503/flappy_bird_mixed_latency_parquet",
        "converted_name": "flappy_mixed_latency_train",
        "latencies": [0, 1, 2, 3, 4, 5],
    },
}


def _namespace(mapping: dict[str, Any]) -> SimpleNamespace:
    values: dict[str, Any] = {}
    for key, value in mapping.items():
        values[key] = _namespace(value) if isinstance(value, dict) else value
    return SimpleNamespace(**values)


def _load_experiment_config(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "examples" / "rl_games" / "experiments" / name
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"Experiment config did not load as a mapping: {path}")
    return loaded


def _assert_pi05_flappy_experiment(cfg: dict[str, Any], expected: ExpectedExperimentConfig) -> None:
    assert cfg["model"] == "pi05"
    assert cfg["env"] == "flappy"
    assert cfg["mode"] == expected["mode"]
    assert cfg["run_id"] == expected["run_id"]
    assert cfg["conda"]["env_name"] == "starvla_rl_games_pi05"
    assert cfg["dataset"]["source_hf"] == expected["source_hf"]
    assert cfg["dataset"]["converted_name"] == expected["converted_name"]
    assert cfg["framework"]["name"] == "QwenPI_v3"
    assert cfg["framework"]["action_model"]["action_dim"] == 2
    assert cfg["framework"]["action_model"]["action_env_dim"] == 2
    assert cfg["framework"]["action_model"]["state_dim"] == 1
    assert cfg["framework"]["action_model"]["action_horizon"] == 1
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["action_dit_hidden_dim"] == 1024
    assert cfg["rl_games"]["model_alias"] == "pi-0.5"
    assert cfg["rl_games"]["latencies"] == expected["latencies"]


def test_pi05_alias_resolves_to_qwenpi_v3() -> None:
    cfg = _namespace({
        "rl_games": {"model_alias": "pi-0.5"},
        "framework": {"name": "QwenPI"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["pi-0.5"] == "QwenPI_v3"
    assert cfg.framework.name == "QwenPI_v3"


def test_pi05_action_spec_preserves_model_dim_and_sets_env_dim() -> None:
    cfg = _namespace({
        "rl_games": {
            "task": "flappy",
            "model_alias": "pi-0.5",
            "env_eval": {"deadly": {"action_layout": "multibinary_7"}},
        },
        "framework": {
            "action_model": {
                "action_dim": 2,
                "action_env_dim": 0,
            }
        },
    })

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 2
    assert cfg.framework.action_model.action_env_dim == 2


def test_pi05_uses_managed_flappy_dataset_setup_only_for_flappy() -> None:
    assert setup_training_assets._uses_managed_flappy_dataset("pi05") is True
    assert setup_training_assets._uses_managed_flappy_dataset("pi0") is True
    assert setup_training_assets._uses_managed_flappy_dataset("openvla") is True
    assert setup_training_assets._uses_managed_flappy_dataset("gr00t") is False


def test_pi05_flappy_single_experiment_uses_qwenpi_v3() -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["single"]
    cfg = _load_experiment_config(expected["name"])

    _assert_pi05_flappy_experiment(cfg, expected)


def test_pi05_flappy_mixed_experiment_uses_qwenpi_v3() -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["mixed_latency"]
    cfg = _load_experiment_config(expected["name"])

    _assert_pi05_flappy_experiment(cfg, expected)
