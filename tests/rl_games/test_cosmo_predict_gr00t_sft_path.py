from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from examples.rl_games.scripts import launch_train
from starVLA.training.rl_games.alias import MODEL_ALIAS_TO_FRAMEWORK, apply_model_alias


REPO_ROOT = Path(__file__).resolve().parents[2]


def _namespace(mapping: dict) -> SimpleNamespace:
    values = {}
    for key, value in mapping.items():
        values[key] = _namespace(value) if isinstance(value, dict) else value
    return SimpleNamespace(**values)


def test_cosmo_predict_gr00t_alias_resolves_to_world_model_gr00t_framework() -> None:
    cfg = _namespace({
        "rl_games": {"model_alias": "cosmo_predict_gr00t"},
        "framework": {"name": "QwenGR00T"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["cosmo_predict_gr00t"] == "CosmoPredict2GR00T"
    assert cfg.framework.name == "CosmoPredict2GR00T"


def test_cosmo_predict_gr00t_flappy_config_matches_released_checkpoint_shape() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="cosmo_predict_gr00t",
        env="flappy",
        init="cosmo_predict_gr00t_libero",
        mode="single",
        overrides=[],
    )

    assert cfg.model == "cosmo_predict_gr00t"
    assert cfg.rl_games.model_alias == "cosmo_predict_gr00t"
    assert cfg.framework.name == "CosmoPredict2GR00T"
    assert cfg.rl_games.initialization_mode == "bridge"
    assert cfg.rl_games.action_carrier == "bridge"
    assert cfg.framework.action_model.action_model_type == "DiT-B"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 2
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.future_action_window_size == 7
    assert cfg.framework.action_model.past_action_window_size == 0
    assert cfg.framework.action_model.repeated_diffusion_steps == 8
    assert cfg.framework.action_model.num_inference_timesteps == 4
    assert cfg.framework.action_model.num_target_vision_tokens == 32
    assert cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim == 2048
    assert cfg.dataset.single_converted_name == "flappy_train__bridge"
    assert cfg.dataset.converted_name == "flappy_train__bridge"
    assert cfg.datasets.vla_data.data_mix == "flappy_train__bridge"
    assert cfg.datasets.vla_data.include_state is True
    assert list(cfg.datasets.vla_data.observation_indices) == [-3, -2, -1, 0]
    assert list(cfg.datasets.vla_data.state_indices) == [0]
    assert list(cfg.datasets.vla_data.action_indices) == list(range(8))
    assert cfg.datasets.vla_data.pack_image_sequence is True
    assert cfg.datasets.vla_data.image_sequence_length == 4
    assert cfg.datasets.vla_data.context_images_column == "observation.context_images"
    assert cfg.base_model.repo_id == "nvidia/Cosmos-Predict2-2B-Video2World"
    assert cfg.initialization.checkpoint_hf_repo_id == "StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1"
    assert cfg.initialization.checkpoint_filename == "checkpoints/steps_50000_pytorch_model.pt"


def test_launch_train_forwards_world_model_path_for_cosmo_predict_gr00t(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="cosmo_predict_gr00t",
        env="flappy",
        init="cosmo_predict_gr00t_libero",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "cosmo_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert f"framework.world_model.base_wm={tmp_path / 'cosmo_base'}" in cmd
    assert f"framework.qwenvl.base_vlm={tmp_path / 'cosmo_base'}" in cmd


def test_run_train_shell_knows_cosmo_predict_gr00t_defaults() -> None:
    run_train_source = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(
        encoding="utf-8"
    )
    setup_source = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "setup_training_assets.py").read_text(
        encoding="utf-8"
    )

    assert "cosmo_predict_gr00t" in run_train_source
    assert "StarVLA/WM4A-CosmoPredict-GR00T-LIBERO-4in1" in run_train_source
    assert "nvidia/Cosmos-Predict2-2B-Video2World" in run_train_source
    assert "framework.world_model.base_wm=$RESOLVED_BASE_MODEL" in run_train_source
    assert '"cosmo_predict_gr00t"' in setup_source


def test_cosmo_predict_gr00t_flappy_command_uses_context_bridge_dataset() -> None:
    command_source = (REPO_ROOT / "commands" / "train_flappy_cosmo_predict_gr00t.sh").read_text(
        encoding="utf-8"
    )

    assert "model=cosmo_predict_gr00t" in command_source
    assert "init=cosmo_predict_gr00t_libero" in command_source
    assert "paths.dataset_local_dir=data/flappy_fix_latency_0_200ep_context4" in command_source
    assert "datasets.vla_data.data_mix=flappy_train__bridge" in command_source
    assert "datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in command_source
