from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class _ConfigKeyAccessor:
    def __init__(self, key: str, original_descriptor: Any) -> None:
        self._key = key
        self._original_descriptor = original_descriptor

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        if self._key in instance:
            return instance[self._key]
        return self._original_descriptor.__get__(instance, owner)


def _patch_dictconfig_values_accessor() -> None:
    current_descriptor = getattr(DictConfig, "values", None)
    if isinstance(current_descriptor, _ConfigKeyAccessor):
        return
    if current_descriptor is None:
        raise RuntimeError("OmegaConf DictConfig.values descriptor is missing")
    DictConfig.values = _ConfigKeyAccessor(key="values", original_descriptor=current_descriptor)


_patch_dictconfig_values_accessor()


def _select_value(cfg: Any, key: str) -> Any:
    return OmegaConf.select(cfg, key)


def _require_non_empty_value(cfg: Any, key: str) -> Any:
    value = _select_value(cfg=cfg, key=key)
    if value is None:
        raise ValueError(f"Missing required RL-games config field: {key}")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Missing required RL-games config field: {key}")
    return value


def _normalize_mode(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_bridge_initialization(cfg: Any) -> bool:
    initialization_mode = _normalize_mode(_select_value(cfg=cfg, key="rl_games.initialization_mode"))
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    return initialization_mode == "bridge" or action_carrier == "bridge"


def _validate_bridge_initialization(cfg: Any) -> None:
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    if action_carrier != "bridge":
        raise ValueError("bridge initialization requires rl_games.action_carrier=bridge")

    checkpoint_local_dir = _select_value(cfg=cfg, key="initialization.checkpoint_local_dir")
    checkpoint_hf_repo_id = _select_value(cfg=cfg, key="initialization.checkpoint_hf_repo_id")
    if checkpoint_local_dir is None and checkpoint_hf_repo_id is None:
        raise ValueError(
            "bridge initialization requires initialization.checkpoint_local_dir or "
            "initialization.checkpoint_hf_repo_id"
        )

    checkpoint_filename = _select_value(cfg=cfg, key="initialization.checkpoint_filename")
    if checkpoint_filename is None or (isinstance(checkpoint_filename, str) and not checkpoint_filename.strip()):
        raise ValueError("bridge initialization requires initialization.checkpoint_filename")


def _validate_latency_values(cfg: Any) -> None:
    values = _require_non_empty_value(cfg=cfg, key="rl_games.env_eval.latency.values")
    latency_values = list(values)
    if not latency_values:
        raise ValueError("Missing required RL-games config field: rl_games.env_eval.latency.values")


def validate_rl_games_config(cfg: Any) -> None:
    model_alias = str(_require_non_empty_value(cfg=cfg, key="rl_games.model_alias"))
    _require_non_empty_value(cfg=cfg, key="rl_games.task")
    _require_non_empty_value(cfg=cfg, key="datasets.vla_data.data_mix")
    _require_non_empty_value(cfg=cfg, key="base_model.repo_id")
    _validate_latency_values(cfg=cfg)

    initialization_mode = _normalize_mode(_select_value(cfg=cfg, key="rl_games.initialization_mode"))
    if model_alias == "pi-0.5" and initialization_mode == "scratch":
        raise ValueError("pi-0.5 scratch is not supported")

    if _is_bridge_initialization(cfg=cfg):
        _validate_bridge_initialization(cfg=cfg)
