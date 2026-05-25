from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from starVLA.training.rl_games.config_validation import validate_rl_games_config


def _build_cfg(data: dict) -> DictConfig:
    return OmegaConf.create(data)


def test_rejects_pi05_scratch() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "pi-0.5",
                "task": "flappy",
                "initialization_mode": "scratch",
                "action_carrier": "native",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": None,
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": None,
            },
        }
    )

    with pytest.raises(ValueError, match="pi-0.5 scratch is not supported"):
        validate_rl_games_config(cfg)


def test_rejects_bridge_without_checkpoint_metadata() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": None,
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": None,
            },
        }
    )

    with pytest.raises(ValueError, match="bridge initialization requires"):
        validate_rl_games_config(cfg)


def test_rejects_bridge_with_blank_checkpoint_sources() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": "",
                "checkpoint_hf_repo_id": "",
                "checkpoint_filename": "checkpoints/steps_1000_pytorch_model.pt",
            },
        }
    )

    with pytest.raises(ValueError, match="bridge initialization requires"):
        validate_rl_games_config(cfg)


def test_accepts_bridge_with_checkpoint_metadata() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": "playground/checkpoints",
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": "checkpoints/steps_1000_pytorch_model.pt",
            },
        }
    )

    validate_rl_games_config(cfg)
