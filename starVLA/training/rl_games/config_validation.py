from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

BRIDGE_INITIALIZATION_MODES: set[str] = {"bridge", "pre-trained", "pretrained"}


def _select_value(cfg: Any, key: str) -> Any:
    return OmegaConf.select(cfg, key)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _require_non_empty_value(cfg: Any, key: str) -> Any:
    value = _select_value(cfg=cfg, key=key)
    if _is_missing_value(value=value):
        raise ValueError(f"Missing required RL-games config field: {key}")
    return value


def _normalize_mode(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_bridge_initialization(cfg: Any) -> bool:
    initialization_mode = _normalize_mode(_select_value(cfg=cfg, key="rl_games.initialization_mode"))
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    return initialization_mode in BRIDGE_INITIALIZATION_MODES or action_carrier == "bridge"


def _has_checkpoint_source(cfg: Any) -> bool:
    checkpoint_local_dir = _select_value(cfg=cfg, key="initialization.checkpoint_local_dir")
    checkpoint_hf_repo_id = _select_value(cfg=cfg, key="initialization.checkpoint_hf_repo_id")
    return not _is_missing_value(value=checkpoint_local_dir) or not _is_missing_value(value=checkpoint_hf_repo_id)


def _validate_bridge_initialization(cfg: Any) -> None:
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    if action_carrier != "bridge":
        raise ValueError("bridge initialization requires rl_games.action_carrier=bridge")

    if not _has_checkpoint_source(cfg=cfg):
        raise ValueError(
            "bridge initialization requires initialization.checkpoint_local_dir or "
            "initialization.checkpoint_hf_repo_id"
        )

    checkpoint_filename = _select_value(cfg=cfg, key="initialization.checkpoint_filename")
    if _is_missing_value(value=checkpoint_filename):
        raise ValueError("bridge initialization requires initialization.checkpoint_filename")


def _validate_latency_values(cfg: Any) -> None:
    latency_values = _require_non_empty_value(cfg=cfg, key="rl_games.env_eval.latency.values")
    if len(latency_values) == 0:
        raise ValueError("Missing required RL-games config field: rl_games.env_eval.latency.values")


def validate_rl_games_config(cfg: Any) -> None:
    model_alias = str(_require_non_empty_value(cfg=cfg, key="rl_games.model_alias"))
    _require_non_empty_value(cfg=cfg, key="rl_games.task")
    _require_non_empty_value(cfg=cfg, key="dataset.converted_name")
    _require_non_empty_value(cfg=cfg, key="base_model.repo_id")
    _validate_latency_values(cfg=cfg)

    initialization_mode = _normalize_mode(_select_value(cfg=cfg, key="rl_games.initialization_mode"))
    if model_alias == "pi-0.5" and initialization_mode == "scratch":
        raise ValueError("pi-0.5 scratch is not supported")

    if _is_bridge_initialization(cfg=cfg):
        _validate_bridge_initialization(cfg=cfg)
