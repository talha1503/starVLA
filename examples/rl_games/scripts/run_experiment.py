#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.setup_training_assets import setup_assets


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read experiment configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _parse_scalar(value: str) -> Any:
    try:
        import yaml
        return yaml.safe_load(value)
    except Exception:
        return value


def _apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override}")
    key, raw_value = override.split("=", 1)
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override key: {key}")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {key}: {part} is not a mapping")
        node = child
    node[parts[-1]] = _parse_scalar(raw_value)


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expanduser(str(value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str(workspace_dir / path)


def _repo_or_workspace_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expanduser(str(value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return str(workspace_dir / path)


def _latencies_expr(values: Any) -> str | None:
    if values in (None, ""):
        return None
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    return "[" + ",".join(str(int(value)) for value in values) + "]"


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def _append_override(
    cmd: list[str],
    cfg: dict[str, Any],
    config_path: str,
    hydra_path: str | None = None,
    default: Any = None,
) -> None:
    value = _get(cfg, config_path, default)
    if value in (None, ""):
        return
    cmd.append(f"{hydra_path or config_path}={_hydra_value(value)}")


def _first_config_value(cfg: dict[str, Any], paths: list[str], default: Any = None) -> Any:
    for path in paths:
        value = _get(cfg, path)
        if value not in (None, ""):
            return value
    return default


def _append_eval_stage_overrides(cmd: list[str], cfg: dict[str, Any], stage_name: str, hydra_name: str) -> None:
    prefix = f"rl_games.{stage_name}"
    hydra_prefix = f"rl_games.env_eval.{hydra_name}"

    mappings = [
        ("enabled", "enabled"),
        ("interval_steps", "interval_steps"),
        ("num_episodes", "num_episodes"),
        ("max_steps_per_episode", "max_steps_per_episode"),
    ]
    for config_key, hydra_key in mappings:
        value = _get(cfg, f"{prefix}.{config_key}")
        if value not in (None, ""):
            cmd.append(f"{hydra_prefix}.{hydra_key}={_hydra_value(value)}")

    latencies = _get(cfg, f"{prefix}.latencies")
    if latencies not in (None, ""):
        cmd.append(f"{hydra_prefix}.latencies={_latencies_expr(latencies)}")


def _setup_namespace(cfg: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_get(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    return SimpleNamespace(
        model=str(_get(cfg, "model")),
        env=str(_get(cfg, "env")),
        mode=str(_get(cfg, "mode")),
        latency_mode=str(_get(cfg, "rl_games.latency_mode", "") or ""),
        source_dataset_hf=str(_get(cfg, "dataset.source_hf", "") or ""),
        dataset_local_dir=_resolve_path(_get(cfg, "paths.dataset_local_dir"), workspace_dir),
        converted_dataset_name=str(_get(cfg, "dataset.converted_name", "flappy_train")),
        dataset_cache_dir=(
            _resolve_path(_get(cfg, "paths.dataset_cache_dir"), workspace_dir)
            if _get(cfg, "paths.dataset_cache_dir") not in (None, "")
            else None
        ),
        dataset_force_download=str(_as_bool(_get(cfg, "dataset.force_download", False))).lower(),
        setup_force=str(_as_bool(_get(cfg, "dataset.setup_force", False))).lower(),
        verify_rows=int(_get(cfg, "dataset.verify_rows", 200)),
        max_episodes=(
            None
            if _get(cfg, "dataset.max_episodes") in (None, "")
            else int(_get(cfg, "dataset.max_episodes"))
        ),
        base_model_dir=_resolve_path(_get(cfg, "paths.base_model_dir"), workspace_dir),
        base_model_repo_id=_get(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint_load=str(_get(cfg, "checkpoint.load", "auto")),
        checkpoint_hf_repo_id=str(_get(cfg, "checkpoint.hf_repo_id", "") or ""),
        hf_repo_id=str(_get(cfg, "checkpoint.sync_repo_id", "") or ""),
    )


def _trainer_command(cfg: dict[str, Any], setup: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> list[str]:
    cmd = [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        str(_get(cfg, "config_name", "train")),
        f"model={_get(cfg, 'model')}",
        f"env={_get(cfg, 'env')}",
        f"mode={_get(cfg, 'mode')}",
        f"run_id={_get(cfg, 'run_id')}",
        f"run_root_dir={run_root_dir}",
        f"seed={_get(cfg, 'seed', 42)}",
        f"wandb_entity={_get(cfg, 'wandb.entity', 'your_wandb_entity')}",
        f"wandb_project={_get(cfg, 'wandb.project', 'starVLA_rl_games')}",
        f"rl_games.env_eval.enabled={str(_as_bool(_first_config_value(cfg, ['rl_games.env_eval_enabled'], True))).lower()}",
        f"checkpoint.sync.enabled={str(_as_bool(_get(cfg, 'checkpoint.sync_enabled', False))).lower()}",
        f"checkpoint.sync.keep_last_n={_get(cfg, 'checkpoint.hf_keep_last_n', 0)}",
        f"checkpoint.local.keep_last_n={_get(cfg, 'checkpoint.local_keep_last_n', 3)}",
        f"trainer.is_resume={str(bool(setup.get('resume_found'))).lower()}",
    ]

    trainer_overrides = [
        "trainer.max_train_steps",
        "trainer.num_warmup_steps",
        "trainer.save_interval",
        "trainer.eval_interval",
        "trainer.logging_frequency",
        "trainer.gradient_accumulation_steps",
        ("trainer.batch_size", "datasets.vla_data.per_device_batch_size"),
        "trainer.learning_rate.base",
        "trainer.learning_rate.qwen_vl_interface",
        "trainer.learning_rate.action_model",
        "trainer.lr_scheduler_type",
        "trainer.scheduler_specific_kwargs.min_lr",
        "trainer.freeze_modules",
        "trainer.loss_scale.vla",
        "trainer.loss_scale.vlm",
        "trainer.max_grad_norm",
        "trainer.weight_decay",
        "trainer.gradient_clipping",
        "trainer.optimizer.name",
        "trainer.optimizer.betas",
        "trainer.optimizer.eps",
        "trainer.optimizer.weight_decay",
        "trainer.save_format",
    ]
    for override in trainer_overrides:
        if isinstance(override, tuple):
            _append_override(cmd, cfg, override[0], override[1])
        else:
            _append_override(cmd, cfg, override)

    data_overrides = [
        ("train_data.action_type", "datasets.vla_data.action_type"),
        ("train_data.sequential_step_sampling", "datasets.vla_data.sequential_step_sampling"),
        ("train_data.load_all_data_for_training", "datasets.vla_data.load_all_data_for_training"),
        ("train_data.obs_image_size", "datasets.vla_data.obs_image_size"),
        ("train_data.video_backend", "datasets.vla_data.video_backend"),
    ]
    for config_path, hydra_path in data_overrides:
        _append_override(cmd, cfg, config_path, hydra_path)

    sync_repo = _get(cfg, "checkpoint.sync_repo_id") or _get(cfg, "checkpoint.hf_repo_id")
    if sync_repo:
        cmd.append(f"checkpoint.sync.repo_id={sync_repo}")

    data_root = setup.get("dataset_local_dir") or _resolve_path(_get(cfg, "paths.dataset_local_dir"), workspace_dir)
    cmd.append(f"datasets.vla_data.data_root_dir={data_root}")
    if setup.get("data_mix"):
        cmd.append(f"datasets.vla_data.data_mix={setup['data_mix']}")
    if setup.get("base_model_dir"):
        cmd.append(f"framework.qwenvl.base_vlm={setup['base_model_dir']}")

    optional = {
        "rl_games.task": _get(cfg, "rl_games.task"),
        "rl_games.model_alias": _get(cfg, "rl_games.model_alias"),
        "rl_games.env_eval.latency.mode": _get(cfg, "rl_games.latency_mode"),
        "rl_games.env_eval.frameskip": _get(cfg, "rl_games.frameskip"),
        "rl_games.env_eval.image_size": _get(cfg, "rl_games.image_size"),
        "rl_games.env_eval.task_description": _get(cfg, "rl_games.task_description"),
    }
    for key, value in optional.items():
        if value not in (None, ""):
            cmd.append(f"{key}={value}")

    latencies = _latencies_expr(_get(cfg, "rl_games.latencies"))
    if latencies:
        cmd.append(f"rl_games.env_eval.latency.values={latencies}")

    prompt_map = _get(cfg, "rl_games.latency_prompt_map_path") or setup.get("latency_prompt_map_path")
    if prompt_map:
        cmd.append(f"rl_games.env_eval.latency.prompt_map_path={prompt_map}")

    _append_eval_stage_overrides(cmd, cfg, "mid_train_eval", "mid_train")
    _append_eval_stage_overrides(cmd, cfg, "post_train_eval", "post_train")

    if str(_get(cfg, "env")) == "deadly_corridor" or str(_get(cfg, "rl_games.task", "")) == "deadly_corridor":
        cmd.append(f"rl_games.env_eval.deadly.action_layout={_get(cfg, 'rl_games.deadly_action_layout', 'multibinary_7')}")

    return cmd


def _launch_command(cfg: dict[str, Any], trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    if _as_bool(_get(cfg, "launch.use_accelerate", True)):
        accelerate_config = _repo_or_workspace_path(_get(cfg, "paths.accelerate_config"), workspace_dir)
        return [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(_get(cfg, "launch.num_processes", 1)),
            *trainer_cmd,
        ]
    return [sys.executable, *trainer_cmd]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    cfg = _load_yaml(config_path)
    for override in args.overrides:
        _apply_override(cfg, override)

    workspace_dir = Path(_resolve_path(_get(cfg, "workspace_dir", "."), REPO_ROOT)).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_get(cfg, "paths.run_root_dir", "results/Checkpoints"), workspace_dir)

    gpus = _get(cfg, "launch.gpus")
    if gpus not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    with contextlib.redirect_stdout(sys.stderr):
        setup = setup_assets(_setup_namespace(cfg, workspace_dir, run_root_dir))

    preprocess_cmd = _get(cfg, "preprocess_cmd")
    if preprocess_cmd not in (None, ""):
        subprocess.run(str(preprocess_cmd), shell=True, check=True, cwd=str(REPO_ROOT))

    trainer_cmd = _trainer_command(cfg, setup, workspace_dir, run_root_dir)
    launch_cmd = _launch_command(cfg, trainer_cmd, workspace_dir)

    print("Setup summary:")
    for key in sorted(setup):
        print(f"  {key}: {setup[key]}")
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))

    if _as_bool(_get(cfg, "launch.dry_run", False)):
        return 0
    subprocess.run(launch_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
