from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
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


def _setup_args(tmp_path: Path, model: str, env: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        env=env,
        mode="single",
        dataset_local_dir=str(tmp_path / f"{model}_{env}_datasets"),
        base_model_dir=str(tmp_path / f"{model}_{env}_base_model"),
        base_model_repo_id="test/base-model",
        checkpoint_local_dir=str(tmp_path / f"{model}_{env}_checkpoints"),
        checkpoint_load="none",
        checkpoint_hf_repo_id="",
        hf_repo_id="",
        checkpoint_sync_repo_id="",
        checkpoint_sync_enabled="false",
    )


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
                "action_dim": 32,
                "action_env_dim": 0,
            }
        },
    })

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 32
    assert cfg.framework.action_model.action_env_dim == 2


def test_pi05_setup_assets_uses_public_flappy_route_only_for_flappy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_flappy_dataset(args: SimpleNamespace) -> dict[str, Any]:
        calls.append(f"{args.model}:{args.env}:flappy")
        return {
            "dataset_ready": True,
            "dataset_local_dir": args.dataset_local_dir,
            "data_mix": f"{args.model}_flappy_train",
            "eval_data_mix": f"{args.model}_flappy_eval",
            "latency_prompt_map_path": None,
        }

    def fake_demon_attack_dataset(args: SimpleNamespace) -> dict[str, Any]:
        calls.append(f"{args.model}:{args.env}:demon_attack")
        return {
            "dataset_ready": True,
            "dataset_local_dir": args.dataset_local_dir,
            "data_mix": f"{args.model}_demon_attack_train",
            "eval_data_mix": f"{args.model}_demon_attack_eval",
            "latency_prompt_map_path": None,
        }

    def fake_deadly_corridor_dataset(args: SimpleNamespace) -> dict[str, Any]:
        calls.append(f"{args.model}:{args.env}:deadly_corridor")
        return {
            "dataset_ready": True,
            "dataset_local_dir": args.dataset_local_dir,
            "data_mix": f"{args.model}_deadly_corridor_train",
            "eval_data_mix": f"{args.model}_deadly_corridor_eval",
            "latency_prompt_map_path": None,
        }

    def fake_base_model(model: str, base_model_dir: Path, base_model_repo_id: str | None) -> dict[str, Any]:
        return {
            "base_model_dir": str(base_model_dir),
            "base_model_repo_id": base_model_repo_id,
            "base_model_downloaded": False,
        }

    monkeypatch.setattr(setup_training_assets, "_ensure_flappy_dataset", fake_flappy_dataset)
    monkeypatch.setattr(setup_training_assets, "_ensure_demon_attack_dataset", fake_demon_attack_dataset)
    monkeypatch.setattr(setup_training_assets, "_ensure_deadly_corridor_dataset", fake_deadly_corridor_dataset)
    monkeypatch.setattr(setup_training_assets, "_ensure_base_model", fake_base_model)

    pi0_flappy = setup_training_assets.setup_assets(_setup_args(tmp_path, "pi0", "flappy"))
    openvla_flappy = setup_training_assets.setup_assets(_setup_args(tmp_path, "openvla", "flappy"))
    pi05_demon_attack = setup_training_assets.setup_assets(_setup_args(tmp_path, "pi05", "demon_attack"))
    pi05_flappy = setup_training_assets.setup_assets(_setup_args(tmp_path, "pi05", "flappy"))

    assert pi0_flappy["data_mix"] == "pi0_flappy_train"
    assert openvla_flappy["data_mix"] == "openvla_flappy_train"
    assert pi05_demon_attack["data_mix"] is None
    assert "pi05:demon_attack:demon_attack" not in calls
    assert pi05_flappy["data_mix"] == "pi05_flappy_train"
    assert "pi05:flappy:flappy" in calls


def test_pi05_flappy_single_experiment_uses_qwenpi_v3() -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["single"]
    cfg = _load_experiment_config(expected["name"])

    _assert_pi05_flappy_experiment(cfg, expected)


def test_pi05_flappy_mixed_experiment_uses_qwenpi_v3() -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["mixed_latency"]
    cfg = _load_experiment_config(expected["name"])

    _assert_pi05_flappy_experiment(cfg, expected)
