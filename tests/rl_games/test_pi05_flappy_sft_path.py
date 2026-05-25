from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from examples.rl_games.scripts import launch_train, setup_training_assets
from starVLA.training.rl_games.action_spec import apply_action_spec
from starVLA.training.rl_games.alias import MODEL_ALIAS_TO_FRAMEWORK, apply_model_alias


REPO_ROOT = Path(__file__).resolve().parents[2]


def _namespace(mapping: dict[str, Any]) -> SimpleNamespace:
    values: dict[str, Any] = {}
    for key, value in mapping.items():
        values[key] = _namespace(value) if isinstance(value, dict) else value
    return SimpleNamespace(**values)


def _load_model_config(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "examples" / "rl_games" / "config" / "model" / name
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"Model config did not load as a mapping: {path}")
    return loaded


def _load_command(name: str) -> str:
    return (REPO_ROOT / "commands" / name).read_text(encoding="utf-8")


def _compose_train_cfg(*, model: str, env: str, init: str, mode: str) -> Any:
    return launch_train.compose_training_config(
        config_name="train",
        model=model,
        env=env,
        init=init,
        mode=mode,
        overrides=[],
    )


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


def test_pi05_alias_resolves_to_qwenpi_v3() -> None:
    cfg = _namespace({
        "rl_games": {"model_alias": "pi-0.5"},
        "framework": {"name": "QwenPI"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["pi-0.5"] == "QwenPI_v3"
    assert cfg.framework.name == "QwenPI_v3"


def test_pi0_bridge_alias_resolves_to_qwenpi() -> None:
    cfg = _namespace({
        "rl_games": {
            "model_alias": "pi-0",
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
        },
        "framework": {"name": "QwenPI_v3"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["pi-0"] == "QwenPI"
    assert cfg.framework.name == "QwenPI"


def test_pi0_model_config_uses_released_qwen_pi_design() -> None:
    cfg = _load_model_config("pi0.yaml")

    assert cfg["framework"]["name"] == "QwenPI"
    assert cfg["framework"]["qwenvl"]["base_vlm"] == (
        "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action"
    )
    assert cfg["framework"]["action_model"]["action_model_type"] == "DiT-Qwen"
    assert cfg["framework"]["action_model"]["hidden_size"] == 1024
    assert cfg["framework"]["action_model"]["action_dim"] == 7
    assert cfg["framework"]["action_model"]["repeated_diffusion_steps"] == 8
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["cross_attention_dim"] == 2048
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["num_layers"] == 16
    assert cfg["framework"]["action_model"]["diffusion_model_cfg"]["output_dim"] == 1024


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

    assert 'pi05) INITIALIZATION_LOCAL_DIR="playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1" ;;' in script
    assert 'pi05) INITIALIZATION_HF_REPO_ID="StarVLA/Qwen3VL-PI_v3-Bridge-RT_1" ;;' in script
    assert 'pi05) INITIALIZATION_CHECKPOINT_FILENAME="checkpoints/steps_50000_pytorch_model.pt" ;;' in script


def test_run_train_pi0_bridge_initializer_matches_released_qwen_pi_checkpoint() -> None:
    script = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(encoding="utf-8")

    assert 'pi0) INITIALIZATION_LOCAL_DIR="playground/Pretrained_models/Qwen-PI-Bridge-RT-1" ;;' in script
    assert 'pi0) INITIALIZATION_HF_REPO_ID="StarVLA/Qwen-PI-Bridge-RT-1" ;;' in script
    assert 'pi0) INITIALIZATION_CHECKPOINT_FILENAME="checkpoints/steps_30000_pytorch_model.pt" ;;' in script


@pytest.mark.parametrize(
    "name",
    [
        "train_flappy_pi0.sh",
        "train_demon_attack_pi0.sh",
        "train_deadly_corridor_pi0.sh",
    ],
)
def test_pi0_commands_use_released_qwen_pi_bridge_initializer(name: str) -> None:
    command = _load_command(name)

    assert "model=pi0" in command
    assert "init=bridge" in command
    assert "checkpoint.load=none" in command
    assert "checkpoint.local.keep_last_n=2" in command
    assert "Qwen3VL-PI_v3-Bridge-RT_1" not in command


def test_rl_games_yaml_eval_max_steps_are_3600() -> None:
    paths = sorted((REPO_ROOT / "examples" / "rl_games").rglob("*.yaml"))
    mismatches: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "max_episode_steps:" in line:
                mismatches.append(f"{path.relative_to(REPO_ROOT)}:{line_number}:{line.strip()}")
            if "max_steps_per_episode:" in line:
                value = line.split(":", 1)[1].strip()
                if value != "3600":
                    mismatches.append(f"{path.relative_to(REPO_ROOT)}:{line_number}:{line.strip()}")

    assert mismatches == []


def test_pi05_setup_assets_routes_all_rl_games_environments(
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
    pi05_deadly_corridor = setup_training_assets.setup_assets(_setup_args(tmp_path, "pi05", "deadly_corridor"))
    pi05_flappy = setup_training_assets.setup_assets(_setup_args(tmp_path, "pi05", "flappy"))

    assert pi0_flappy["data_mix"] == "pi0_flappy_train"
    assert openvla_flappy["data_mix"] == "openvla_flappy_train"
    assert pi05_demon_attack["data_mix"] == "pi05_demon_attack_train"
    assert pi05_demon_attack["eval_data_mix"] == "pi05_demon_attack_eval"
    assert pi05_deadly_corridor["data_mix"] == "pi05_deadly_corridor_train"
    assert pi05_deadly_corridor["eval_data_mix"] == "pi05_deadly_corridor_eval"
    assert "pi05:demon_attack:demon_attack" in calls
    assert "pi05:deadly_corridor:deadly_corridor" in calls
    assert pi05_flappy["data_mix"] == "pi05_flappy_train"
    assert "pi05:flappy:flappy" in calls


def test_ready_local_dataset_ignores_manifest_source_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root_dir = tmp_path / "datasets"
    train_dataset_dir = data_root_dir / "flappy_train__bridge"
    eval_dataset_dir = data_root_dir / "flappy_train__bridge__val"
    for dataset_dir in (train_dataset_dir, eval_dataset_dir):
        (dataset_dir / "meta").mkdir(parents=True)
        (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
        for metadata_name in ("modality.json", "info.json"):
            (dataset_dir / "meta" / metadata_name).write_text("{}", encoding="utf-8")
        for metadata_name in ("episodes.jsonl", "tasks.jsonl"):
            (dataset_dir / "meta" / metadata_name).write_text("{}\n", encoding="utf-8")
        (dataset_dir / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"PAR1")
        (dataset_dir / "manifest.json").write_text(
            json.dumps({
                "source": "previous/raw-source",
                "action_carrier": "bridge",
                "latency_filter": None,
            }),
            encoding="utf-8",
        )

    def reject_verify_dataset(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("ready local dataset should not verify raw source")

    def reject_convert_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("ready local dataset should not be rebuilt")

    def fake_validate_starvla_dataset(data_root_dir: Path, data_mix: str) -> dict[str, Any]:
        return {
            "dataset_stats_path": str(data_root_dir / data_mix / "dataset_statistics.json"),
            "dataset_num_steps": 1,
            "dataset_num_trajectories": 1,
            "dataset_robot_type": "flappy",
            "dataset_embodiment_tag": "flappy",
        }

    monkeypatch.setattr(setup_training_assets, "_validate_starvla_dataset", fake_validate_starvla_dataset)
    args = SimpleNamespace(
        dataset_local_dir=str(data_root_dir),
        initialization_mode="bridge",
        action_carrier="bridge",
        converted_dataset_name="flappy_train",
        source_dataset_hf="data/flappy_fix_latency_0_parquet",
        setup_force="false",
        dataset_force_download="false",
        mode="single",
        latency_mode="single",
        verify_rows=200,
        dataset_cache_dir=None,
        max_episodes=None,
        latency_filter=None,
    )

    result = setup_training_assets._ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=reject_convert_dataset,
        verify_dataset=reject_verify_dataset,
    )

    assert result["dataset_converted"] is False
    assert result["data_mix"] == "flappy_train__bridge"
    assert result["eval_data_mix"] == "flappy_train__bridge__val"


@pytest.mark.parametrize(
    ("model", "env", "mode", "action_env_dim"),
    [
        ("pi05", "flappy", "single", 2),
        ("pi05", "flappy", "mixed_latency", 2),
        ("pi05", "demon_attack", "single", 6),
        ("pi05", "demon_attack", "mixed_latency", 6),
        ("pi05", "deadly_corridor", "single", 7),
        ("pi05", "deadly_corridor", "mixed_latency", 7),
    ],
)
def test_pi05_bridge_composed_config_uses_qwenpi_v3(
    model: str,
    env: str,
    mode: str,
    action_env_dim: int,
) -> None:
    cfg = _compose_train_cfg(model=model, env=env, init="bridge", mode=mode)

    assert cfg.framework.name == "QwenPI_v3"
    assert cfg.rl_games.model_alias == "pi-0.5"
    assert cfg.rl_games.initialization_mode == "bridge"
    assert cfg.rl_games.action_carrier == "bridge"
    assert cfg.initialization.checkpoint_hf_repo_id == "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
    assert cfg.initialization.checkpoint_filename == "checkpoints/steps_50000_pytorch_model.pt"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == action_env_dim


@pytest.mark.parametrize(
    ("env", "mode", "action_env_dim"),
    [
        ("flappy", "single", 2),
        ("flappy", "mixed_latency", 2),
        ("demon_attack", "single", 6),
        ("demon_attack", "mixed_latency", 6),
        ("deadly_corridor", "single", 7),
        ("deadly_corridor", "mixed_latency", 7),
    ],
)
def test_pi05_bridge_composed_config_forwards_qwenpi_v3_command_overrides(
    env: str,
    mode: str,
    action_env_dim: int,
    tmp_path: Path,
) -> None:
    cfg = _compose_train_cfg(model="pi05", env=env, init="bridge", mode=mode)
    expected_latency_values = "[0]" if mode == "single" else "[0,1,2,3,4,5]"
    expected_post_train_latencies = "[0,1,2,3,4]"
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "framework.action_model.diffusion_model_cfg.action_dit_hidden_dim=1024" in cmd
    assert "framework.action_model.diffusion_model_cfg.output_dim=1024" in cmd
    assert f"framework.action_model.action_env_dim={action_env_dim}" in cmd
    assert "rl_games.env_eval.mid_train.interval_steps=100" in cmd
    assert f"rl_games.env_eval.mid_train.latencies={expected_latency_values}" in cmd
    assert "rl_games.env_eval.image_size=224" in cmd
    assert f"rl_games.env_eval.post_train.latencies={expected_post_train_latencies}" in cmd
    if env == "deadly_corridor":
        assert "rl_games.env_eval.deadly.action_layout=multibinary_7" in cmd


def test_launch_train_setup_namespace_uses_composed_hydra_config(tmp_path: Path) -> None:
    cfg = _compose_train_cfg(model="pi05", env="flappy", init="bridge", mode="single")

    setup_args = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.model == "pi05"
    assert setup_args.env == "flappy"
    assert setup_args.mode == "single"
    assert setup_args.initialization_mode == "bridge"
    assert setup_args.action_carrier == "bridge"
    assert setup_args.dataset_local_dir == str(tmp_path / "playground" / "Datasets" / "rl_games")
    assert setup_args.converted_dataset_name == "flappy_train"
    assert setup_args.initialization_checkpoint_filename == "checkpoints/steps_50000_pytorch_model.pt"


def test_launch_train_setup_namespace_forwards_explicit_dataset_source_hf(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="pi05",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["dataset.source_hf=owner/flappy_source"],
    )

    setup_args = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.source_dataset_hf == "owner/flappy_source"


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
