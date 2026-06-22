from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.rl_games.scripts import launch_train
from starVLA.training.rl_games.alias import MODEL_ALIAS_TO_FRAMEWORK, apply_model_alias


REPO_ROOT = Path(__file__).resolve().parents[2]


def _namespace(mapping: dict) -> SimpleNamespace:
    values = {}
    for key, value in mapping.items():
        values[key] = _namespace(value) if isinstance(value, dict) else value
    return SimpleNamespace(**values)


def test_wan_oft_alias_resolves_to_wan_oft_framework() -> None:
    cfg = _namespace({
        "rl_games": {"model_alias": "wan_oft"},
        "framework": {"name": "QwenOFT"},
    })

    apply_model_alias(cfg)

    assert MODEL_ALIAS_TO_FRAMEWORK["wan_oft"] == "WanOFT"
    assert cfg.framework.name == "WanOFT"


def test_wan_oft_flappy_config_preserves_released_checkpoint_shapes() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )

    assert cfg.framework.name == "WanOFT"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == 2
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 8
    assert cfg.framework.action_model.future_action_window_size == 7
    assert cfg.framework.action_model.past_action_window_size == 0
    assert cfg.datasets.vla_data.include_state is True
    assert list(cfg.datasets.vla_data.observation_indices) == [-3, -2, -1, 0]
    assert list(cfg.datasets.vla_data.state_indices) == [0]
    assert list(cfg.datasets.vla_data.action_indices) == list(range(8))
    assert cfg.datasets.vla_data.pack_image_sequence is True
    assert cfg.datasets.vla_data.image_sequence_length == 4
    assert cfg.datasets.vla_data.context_images_column == "observation.context_images"
    assert cfg.base_model.repo_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert cfg.initialization.checkpoint_hf_repo_id == "StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1"
    assert cfg.initialization.checkpoint_filename == "checkpoints/steps_60000_pytorch_model.pt"


def test_launch_train_forwards_world_model_path_only_for_wan_oft(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert f"framework.world_model.base_wm={tmp_path / 'wan_base'}" in cmd
    assert f"framework.qwenvl.base_vlm={tmp_path / 'wan_base'}" in cmd

    openvla_cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )
    openvla_cmd = launch_train.build_trainer_command(openvla_cfg, setup, tmp_path, "results/Checkpoints")

    assert not any(item.startswith("framework.world_model.base_wm=") for item in openvla_cmd)


def test_launch_train_forwards_explicit_wan_oft_bridge_data_mix(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="flappy",
        init="wan_oft_libero",
        mode="single",
        overrides=[
            "datasets.vla_data.data_mix=flappy_train__bridge",
            "datasets.vla_data.eval_data_mix=flappy_train__bridge__val",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "datasets.vla_data.data_mix=flappy_train__bridge" in cmd
    assert "datasets.vla_data.eval_data_mix=flappy_train__bridge__val" in cmd


def test_train_starvla_uses_device_aware_distributed_barrier() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "def _distributed_barrier()" in trainer_text
    assert "dist.barrier(device_ids=[device_idx])" in trainer_text
    assert trainer_text.count("dist.barrier()") == 1
    assert trainer_text.count("_distributed_barrier()") == 5


def test_rl_games_modality_indices_can_be_overridden_for_wan_oft_clip() -> None:
    temporal_clip = importlib.import_module("starVLA.training.rl_games.temporal_clip")

    resolved = temporal_clip.resolve_modality_indices(
        default_observation_indices=[0],
        default_state_indices=[0],
        default_action_indices=[0],
        data_cfg={
            "observation_indices": [-3, -2, -1, 0],
            "state_indices": [0],
            "action_indices": list(range(8)),
        },
    )

    assert resolved.observation_indices == [-3, -2, -1, 0]
    assert resolved.language_indices == [0]
    assert resolved.state_indices == [0]
    assert resolved.action_indices == list(range(8))


def test_pack_sample_preserves_configured_temporal_clip() -> None:
    temporal_clip = importlib.import_module("starVLA.training.rl_games.temporal_clip")

    frames = np.stack([
        np.full((4, 4, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ])

    images = temporal_clip.pack_image_sequence(
        frames=frames,
        pack_image_sequence=True,
        image_sequence_length=4,
        resize_size=(224, 224),
    )

    assert len(images) == 4
    assert [int(np.asarray(image.resize((4, 4)))[0, 0, 0]) for image in images] == [10, 20, 30, 40]
