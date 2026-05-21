from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
import yaml

from examples.rl_games.scripts import run_experiment, setup_training_assets
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
        "name": "pi05/bridge/single/flappy.yaml",
        "mode": "single",
        "run_id": "pi05_flappy_single",
        "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
        "converted_name": "flappy_train",
        "latencies": [0],
    },
    "mixed_latency": {
        "name": "pi05/bridge/mixed_latency/flappy.yaml",
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


def _load_model_config(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "examples" / "rl_games" / "config" / "model" / name
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"Model config did not load as a mapping: {path}")
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
    assert cfg["paths"]["base_model_dir"] == "playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    assert cfg["base_model"]["repo_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert cfg["initialization"]["checkpoint_local_dir"] == "playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1"
    assert cfg["initialization"]["checkpoint_hf_repo_id"] == "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
    assert cfg["initialization"]["checkpoint_filename"] == "checkpoints/steps_50000_pytorch_model.pt"
    assert cfg["dataset"]["source_hf"] == expected["source_hf"]
    assert cfg["dataset"]["converted_name"] == expected["converted_name"]
    assert cfg["framework"]["name"] == "QwenPI_v3"
    assert cfg["framework"]["action_model"]["action_dim"] == 7
    assert cfg["framework"]["action_model"]["action_env_dim"] == 2
    assert cfg["framework"]["action_model"]["state_dim"] == 7
    assert cfg["framework"]["action_model"]["action_horizon"] == 1
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["action_dit_hidden_dim"] == 1024
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["output_dim"] == 1024
    assert cfg["rl_games"]["model_alias"] == "pi-0.5"
    assert cfg["rl_games"]["initialization_mode"] == "bridge"
    assert cfg["rl_games"]["action_carrier"] == "bridge"
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


def test_pi05_model_config_uses_qwen3_base_backbone() -> None:
    cfg = _load_model_config("pi05.yaml")

    assert cfg["framework"]["qwenvl"]["base_vlm"] == (
        "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    )


def test_run_train_pi05_bridge_initializer_matches_official_pi_v3_checkpoint() -> None:
    script = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(encoding="utf-8")

    assert 'pi0|pi05) INITIALIZATION_LOCAL_DIR="playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1" ;;' in script
    assert 'pi05) INITIALIZATION_HF_REPO_ID="StarVLA/Qwen3VL-PI_v3-Bridge-RT_1" ;;' in script
    assert 'pi05) INITIALIZATION_CHECKPOINT_FILENAME="checkpoints/steps_50000_pytorch_model.pt" ;;' in script


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


def test_pi05_flappy_single_experiment_forwards_qwenpi_v3_diffusion_width(tmp_path: Path) -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["single"]
    cfg = _load_experiment_config(expected["name"])
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = run_experiment._trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "framework.action_model.diffusion_model_cfg.action_dit_hidden_dim=1024" in cmd
    assert "framework.action_model.diffusion_model_cfg.output_dim=1024" in cmd


def test_pi05_flappy_single_setup_resolves_local_initialization_dir(tmp_path: Path) -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["single"]
    cfg = _load_experiment_config(expected["name"])

    setup_args = run_experiment._setup_namespace(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.initialization_local_dir == str(
        tmp_path / "playground" / "Pretrained_models" / "Qwen3VL-PI_v3-Bridge-RT_1"
    )


def test_pi05_setup_assets_prefers_local_bridge_initialization_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_repo = tmp_path / "Qwen3VL-PI_v3-Bridge-RT_1"
    checkpoint_file = local_repo / "checkpoints" / "steps_50000_pytorch_model.pt"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"checkpoint")
    args = _setup_args(tmp_path, "pi05", "flappy")
    args.initialization_mode = "bridge"
    args.initialization_local_dir = str(local_repo)
    args.initialization_hf_repo_id = "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
    args.initialization_checkpoint_filename = "checkpoints/steps_50000_pytorch_model.pt"

    def fake_flappy_dataset(args: SimpleNamespace) -> dict[str, Any]:
        return {
            "dataset_ready": True,
            "dataset_local_dir": args.dataset_local_dir,
            "data_mix": "flappy_train__bridge",
            "eval_data_mix": "flappy_train__bridge__val",
            "latency_prompt_map_path": None,
        }

    def fake_base_model(model: str, base_model_dir: Path, base_model_repo_id: str | None) -> dict[str, Any]:
        return {
            "base_model_dir": str(base_model_dir),
            "base_model_repo_id": base_model_repo_id,
            "base_model_downloaded": False,
        }

    def fail_hf_download(repo_id: str, filename: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
        raise AssertionError("HF checkpoint download should not run when local initialization exists")

    monkeypatch.setattr(setup_training_assets, "_ensure_flappy_dataset", fake_flappy_dataset)
    monkeypatch.setattr(setup_training_assets, "_ensure_base_model", fake_base_model)
    monkeypatch.setattr(setup_training_assets, "_download_hf_checkpoint_file", fail_hf_download)

    setup = setup_training_assets.setup_assets(args)

    assert setup["pretrained_checkpoint"] == str(checkpoint_file.resolve())
    assert setup["initialization_source"] == "local"
    assert setup["initialization_local_dir"] == str(local_repo.resolve())
    assert setup["initialization_step"] == 50000


def test_pi05_setup_assets_falls_back_to_hf_when_local_initialization_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_local_repo = tmp_path / "missing_Qwen3VL-PI_v3-Bridge-RT_1"
    downloaded_checkpoint = tmp_path / "downloaded" / "checkpoints" / "steps_50000_pytorch_model.pt"
    downloaded_checkpoint.parent.mkdir(parents=True)
    downloaded_checkpoint.write_bytes(b"checkpoint")
    args = _setup_args(tmp_path, "pi05", "flappy")
    args.initialization_mode = "bridge"
    args.initialization_local_dir = str(missing_local_repo)
    args.initialization_hf_repo_id = "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
    args.initialization_checkpoint_filename = "checkpoints/steps_50000_pytorch_model.pt"

    def fake_flappy_dataset(args: SimpleNamespace) -> dict[str, Any]:
        return {
            "dataset_ready": True,
            "dataset_local_dir": args.dataset_local_dir,
            "data_mix": "flappy_train__bridge",
            "eval_data_mix": "flappy_train__bridge__val",
            "latency_prompt_map_path": None,
        }

    def fake_base_model(model: str, base_model_dir: Path, base_model_repo_id: str | None) -> dict[str, Any]:
        return {
            "base_model_dir": str(base_model_dir),
            "base_model_repo_id": base_model_repo_id,
            "base_model_downloaded": False,
        }

    def fake_hf_download(repo_id: str, filename: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
        assert repo_id == "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
        assert filename == "checkpoints/steps_50000_pytorch_model.pt"
        return downloaded_checkpoint, 50000, None

    monkeypatch.setattr(setup_training_assets, "_ensure_flappy_dataset", fake_flappy_dataset)
    monkeypatch.setattr(setup_training_assets, "_ensure_base_model", fake_base_model)
    monkeypatch.setattr(setup_training_assets, "_download_hf_checkpoint_file", fake_hf_download)

    setup = setup_training_assets.setup_assets(args)

    assert setup["pretrained_checkpoint"] == str(downloaded_checkpoint)
    assert setup["initialization_source"] == "hf"
    assert setup["initialization_local_dir"] == str(missing_local_repo)
    assert setup["initialization_step"] == 50000


def test_pi05_flappy_mixed_experiment_uses_qwenpi_v3() -> None:
    expected = EXPECTED_PI05_FLAPPY_EXPERIMENTS["mixed_latency"]
    cfg = _load_experiment_config(expected["name"])

    _assert_pi05_flappy_experiment(cfg, expected)
