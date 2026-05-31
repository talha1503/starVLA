#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"
DEFAULT_CONFIG_NAME = "train"
DEFAULT_MODEL = "openvla"
DEFAULT_ENV = "flappy"
DEFAULT_INIT = "scratch"
DEFAULT_MODE = "single"
DEFAULT_RUN_ROOT_DIR = "results/Checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.setup_training_assets import setup_assets
from starVLA.training.rl_games import apply_action_spec, apply_model_alias, validate_rl_games_config
from starVLA.training.rl_games.auth import login_training_services


def _cfg_get(cfg: Any, path: str) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, path)
    node = cfg
    for part in path.split("."):
        if node is None:
            return None
        if isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
            continue
        if not hasattr(node, part):
            return None
        node = getattr(node, part)
    return node


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((workspace_dir / path).resolve())


def _repo_or_workspace_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return str(repo_path)
    return str((workspace_dir / path).resolve())


def _workspace_dir(cfg: Any) -> Path:
    configured = _cfg_get(cfg, "workspace_dir")
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    return REPO_ROOT


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
    if isinstance(value, str) and any(ch.isspace() or ch in {",", ":", "{", "}", "[", "]"} for ch in value):
        return shlex.quote(value)
    return str(value)


def _append_override(cmd: list[str], cfg: Any, cfg_path: str, hydra_path: str) -> None:
    value = _cfg_get(cfg, cfg_path)
    if value in (None, ""):
        return
    cmd.append(f"{hydra_path}={_hydra_value(value)}")


def _append_eval_stage_overrides(cmd: list[str], cfg: Any, stage_name: str, hydra_name: str) -> None:
    prefix = f"rl_games.{stage_name}"
    hydra_prefix = f"rl_games.env_eval.{hydra_name}"

    for cfg_key, hydra_key in (
        ("enabled", "enabled"),
        ("interval_steps", "interval_steps"),
        ("num_episodes", "num_episodes"),
        ("max_steps_per_episode", "max_steps_per_episode"),
    ):
        value = _cfg_get(cfg, f"{prefix}.{cfg_key}")
        if value not in (None, ""):
            cmd.append(f"{hydra_prefix}.{hydra_key}={_hydra_value(value)}")

    latencies = _cfg_get(cfg, f"{prefix}.latencies")
    if latencies not in (None, ""):
        cmd.append(f"{hydra_prefix}.latencies={_latencies_expr(latencies)}")


def _safe_suffix(value: Any) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "debug")).strip("_")
    return suffix or "debug"


def _dataset_setup_values(cfg: Any) -> tuple[str, int | None]:
    converted_name = str(_cfg_get(cfg, "dataset.converted_name") or "flappy_train")
    max_episodes_value = _cfg_get(cfg, "dataset.max_episodes")
    max_episodes = None if max_episodes_value in (None, "") else int(max_episodes_value)

    if _as_bool(_cfg_get(cfg, "dataset.debug_subset.enabled")):
        debug_max = _cfg_get(cfg, "dataset.debug_subset.max_episodes")
        if debug_max in (None, ""):
            debug_max = max_episodes if max_episodes is not None else 5
        max_episodes = int(debug_max)
        suffix = _safe_suffix(_cfg_get(cfg, "dataset.debug_subset.suffix"))
        debug_suffix = "debug" if suffix == "debug" else f"debug_{suffix}"
        converted_name = f"{converted_name}__{debug_suffix}_{max_episodes}ep"

    return converted_name, max_episodes


def _optional_int_list(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def compose_training_config(
    config_name: str,
    model: str,
    env: str,
    init: str,
    mode: str,
    overrides: Sequence[str],
) -> DictConfig:
    compose_overrides = [
        f"model={model}",
        f"env={env}",
        f"init={init}",
        f"mode={mode}",
        *overrides,
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=compose_overrides)
    validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    return cfg


def setup_namespace_from_cfg(cfg: Any, workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_cfg_get(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    converted_dataset_name, max_episodes = _dataset_setup_values(cfg)

    dataset_cache_dir = _cfg_get(cfg, "paths.dataset_cache_dir")
    initialization_local_dir = _cfg_get(cfg, "initialization.checkpoint_local_dir")
    cross_task_cfg = _cfg_get(cfg, "rl_games.cross_task")
    cross_task = (
        OmegaConf.to_container(cross_task_cfg, resolve=True)
        if OmegaConf.is_config(cross_task_cfg)
        else (cross_task_cfg or {})
    )

    return SimpleNamespace(
        model=str(_cfg_get(cfg, "model")),
        env=str(_cfg_get(cfg, "env")),
        mode=str(_cfg_get(cfg, "mode")),
        initialization_mode=str(_cfg_get(cfg, "rl_games.initialization_mode") or ""),
        action_carrier=str(_cfg_get(cfg, "rl_games.action_carrier") or ""),
        latency_mode=str(_cfg_get(cfg, "rl_games.env_eval.latency.mode") or ""),
        source_dataset_hf=str(_cfg_get(cfg, "dataset.source_hf") or ""),
        dataset_local_dir=_resolve_path(_cfg_get(cfg, "paths.dataset_local_dir"), workspace_dir),
        converted_dataset_name=converted_dataset_name,
        dataset_cache_dir=(
            _resolve_path(dataset_cache_dir, workspace_dir)
            if dataset_cache_dir not in (None, "")
            else None
        ),
        dataset_force_download=str(_as_bool(_cfg_get(cfg, "dataset.force_download"))).lower(),
        setup_force=str(_as_bool(_cfg_get(cfg, "dataset.setup_force"))).lower(),
        verify_rows=int(_cfg_get(cfg, "dataset.verify_rows") or 200),
        max_episodes=max_episodes,
        latency_filter=_optional_int_list(_cfg_get(cfg, "dataset.latency_filter")),
        base_model_dir=_resolve_path(_cfg_get(cfg, "paths.base_model_dir"), workspace_dir),
        base_model_repo_id=_cfg_get(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint_load=str(_cfg_get(cfg, "checkpoint.load") or "auto"),
        checkpoint_hf_repo_id=str(_cfg_get(cfg, "checkpoint.hf_repo_id") or ""),
        initialization_local_dir=(
            _resolve_path(initialization_local_dir, workspace_dir)
            if initialization_local_dir not in (None, "")
            else ""
        ),
        initialization_hf_repo_id=str(_cfg_get(cfg, "initialization.checkpoint_hf_repo_id") or ""),
        initialization_checkpoint_filename=str(_cfg_get(cfg, "initialization.checkpoint_filename") or ""),
        cross_task=cross_task,
        checkpoint_sync_enabled=str(_as_bool(_cfg_get(cfg, "checkpoint.sync.enabled"))).lower(),
        checkpoint_sync_repo_id=str(_cfg_get(cfg, "checkpoint.sync.repo_id") or ""),
        hf_repo_id="",
    )


def build_trainer_command(cfg: Any, setup: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> list[str]:
    cmd = [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        str(_cfg_get(cfg, "config_name") or DEFAULT_CONFIG_NAME),
        f"model={_cfg_get(cfg, 'model')}",
        f"env={_cfg_get(cfg, 'env')}",
        f"init={_cfg_get(cfg, 'init')}",
        f"mode={_cfg_get(cfg, 'mode')}",
        f"run_id={_cfg_get(cfg, 'run_id')}",
        f"run_root_dir={run_root_dir}",
        f"seed={_cfg_get(cfg, 'seed') or 42}",
        f"wandb_entity={_cfg_get(cfg, 'wandb_entity') or 'your_wandb_entity'}",
        f"wandb_project={_cfg_get(cfg, 'wandb_project') or 'starVLA_rl_games'}",
        f"rl_games.env_eval.enabled={str(_as_bool(_cfg_get(cfg, 'rl_games.env_eval.enabled'))).lower()}",
        f"checkpoint.sync.enabled={str(_as_bool(_cfg_get(cfg, 'checkpoint.sync.enabled'))).lower()}",
        f"checkpoint.sync.keep_last_n={_cfg_get(cfg, 'checkpoint.sync.keep_last_n') or 0}",
        f"checkpoint.local.keep_last_n={_cfg_get(cfg, 'checkpoint.local.keep_last_n') or 3}",
        f"trainer.is_resume={str(bool(setup.get('resume_found'))).lower()}",
    ]

    if setup.get("resume_checkpoint"):
        cmd.append(f"trainer.pretrained_checkpoint={setup['resume_checkpoint']}")
        cmd.append(f"trainer.resume_step={int(setup.get('resume_step') or 0)}")
    elif setup.get("pretrained_checkpoint"):
        cmd.append(f"trainer.pretrained_checkpoint={setup['pretrained_checkpoint']}")
        cmd.append("trainer.resume_step=0")

    for cfg_path, hydra_path in (
        ("trainer.max_train_steps", "trainer.max_train_steps"),
        ("trainer.num_warmup_steps", "trainer.num_warmup_steps"),
        ("trainer.save_interval", "trainer.save_interval"),
        ("trainer.eval_interval", "trainer.eval_interval"),
        ("trainer.eval_num_batches", "trainer.eval_num_batches"),
        ("trainer.logging_frequency", "trainer.logging_frequency"),
        ("trainer.gradient_accumulation_steps", "trainer.gradient_accumulation_steps"),
        ("trainer.distributed_backend", "trainer.distributed_backend"),
        ("trainer.learning_rate.base", "trainer.learning_rate.base"),
        ("trainer.learning_rate.qwen_vl_interface", "trainer.learning_rate.qwen_vl_interface"),
        ("trainer.learning_rate.action_model", "trainer.learning_rate.action_model"),
        ("trainer.lr_scheduler_type", "trainer.lr_scheduler_type"),
        ("trainer.scheduler_specific_kwargs.min_lr", "trainer.scheduler_specific_kwargs.min_lr"),
        ("trainer.freeze_modules", "trainer.freeze_modules"),
        ("trainer.loss_scale.vla", "trainer.loss_scale.vla"),
        ("trainer.loss_scale.vlm", "trainer.loss_scale.vlm"),
        ("trainer.max_grad_norm", "trainer.max_grad_norm"),
        ("trainer.weight_decay", "trainer.weight_decay"),
        ("trainer.gradient_clipping", "trainer.gradient_clipping"),
        ("trainer.optimizer.name", "trainer.optimizer.name"),
        ("trainer.optimizer.betas", "trainer.optimizer.betas"),
        ("trainer.optimizer.eps", "trainer.optimizer.eps"),
        ("trainer.optimizer.weight_decay", "trainer.optimizer.weight_decay"),
        ("trainer.optimizer.fused", "trainer.optimizer.fused"),
        ("trainer.save_format", "trainer.save_format"),
        ("framework.name", "framework.name"),
        ("framework.qwenvl.attn_implementation", "framework.qwenvl.attn_implementation"),
        ("framework.qwenvl.enable_gradient_checkpointing", "framework.qwenvl.enable_gradient_checkpointing"),
        ("framework.action_model.action_model_type", "framework.action_model.action_model_type"),
        ("framework.action_model.action_dim", "framework.action_model.action_dim"),
        ("framework.action_model.action_env_dim", "framework.action_model.action_env_dim"),
        ("framework.action_model.state_dim", "framework.action_model.state_dim"),
        ("framework.action_model.loss_type", "framework.action_model.loss_type"),
        ("framework.action_model.action_horizon", "framework.action_model.action_horizon"),
        ("framework.action_model.future_action_window_size", "framework.action_model.future_action_window_size"),
        ("framework.action_model.past_action_window_size", "framework.action_model.past_action_window_size"),
        ("framework.action_model.repeated_diffusion_steps", "framework.action_model.repeated_diffusion_steps"),
        ("framework.action_model.num_inference_timesteps", "framework.action_model.num_inference_timesteps"),
        ("framework.action_model.num_target_vision_tokens", "framework.action_model.num_target_vision_tokens"),
        ("framework.action_model.add_pos_embed", "framework.action_model.add_pos_embed"),
        ("framework.action_model.max_seq_len", "framework.action_model.max_seq_len"),
        ("framework.action_model.hidden_size", "framework.action_model.hidden_size"),
        ("framework.action_model.action_hidden_dim", "framework.action_model.action_hidden_dim"),
        ("framework.action_model.noise_beta_alpha", "framework.action_model.noise_beta_alpha"),
        ("framework.action_model.noise_beta_beta", "framework.action_model.noise_beta_beta"),
        ("framework.action_model.noise_s", "framework.action_model.noise_s"),
        ("framework.action_model.num_timestep_buckets", "framework.action_model.num_timestep_buckets"),
        (
            "framework.action_model.diffusion_model_cfg.action_dit_hidden_dim",
            "framework.action_model.diffusion_model_cfg.action_dit_hidden_dim",
        ),
        (
            "framework.action_model.diffusion_model_cfg.cross_attention_dim",
            "framework.action_model.diffusion_model_cfg.cross_attention_dim",
        ),
        ("framework.action_model.diffusion_model_cfg.dropout", "framework.action_model.diffusion_model_cfg.dropout"),
        (
            "framework.action_model.diffusion_model_cfg.final_dropout",
            "framework.action_model.diffusion_model_cfg.final_dropout",
        ),
        (
            "framework.action_model.diffusion_model_cfg.interleave_self_attention",
            "framework.action_model.diffusion_model_cfg.interleave_self_attention",
        ),
        ("framework.action_model.diffusion_model_cfg.norm_type", "framework.action_model.diffusion_model_cfg.norm_type"),
        (
            "framework.action_model.diffusion_model_cfg.num_layers",
            "framework.action_model.diffusion_model_cfg.num_layers",
        ),
        ("framework.action_model.diffusion_model_cfg.output_dim", "framework.action_model.diffusion_model_cfg.output_dim"),
        (
            "framework.action_model.diffusion_model_cfg.positional_embeddings",
            "framework.action_model.diffusion_model_cfg.positional_embeddings",
        ),
        (
            "framework.action_model.diffusion_model_cfg.attention_head_dim",
            "framework.action_model.diffusion_model_cfg.attention_head_dim",
        ),
        ("datasets.vla_data.include_state", "datasets.vla_data.include_state"),
        ("datasets.vla_data.action_type", "datasets.vla_data.action_type"),
        ("datasets.vla_data.sequential_step_sampling", "datasets.vla_data.sequential_step_sampling"),
        ("datasets.vla_data.per_device_batch_size", "datasets.vla_data.per_device_batch_size"),
        ("datasets.vla_data.load_all_data_for_training", "datasets.vla_data.load_all_data_for_training"),
        ("datasets.vla_data.obs_image_size", "datasets.vla_data.obs_image_size"),
        ("datasets.vla_data.video_backend", "datasets.vla_data.video_backend"),
    ):
        _append_override(cmd, cfg, cfg_path, hydra_path)

    data_root = setup.get("dataset_local_dir") or _resolve_path(_cfg_get(cfg, "paths.dataset_local_dir"), workspace_dir)
    cmd.append(f"datasets.vla_data.data_root_dir={data_root}")

    if setup.get("data_mix"):
        cmd.append(f"datasets.vla_data.data_mix={setup['data_mix']}")
    if setup.get("eval_data_mix"):
        cmd.append(f"datasets.vla_data.eval_data_mix={setup['eval_data_mix']}")
    if setup.get("base_model_dir"):
        cmd.append(f"framework.qwenvl.base_vlm={setup['base_model_dir']}")

    for hydra_key, cfg_path in (
        ("rl_games.task", "rl_games.task"),
        ("rl_games.model_alias", "rl_games.model_alias"),
        ("rl_games.initialization_mode", "rl_games.initialization_mode"),
        ("rl_games.action_carrier", "rl_games.action_carrier"),
        ("rl_games.env_eval.latency.mode", "rl_games.env_eval.latency.mode"),
        ("rl_games.env_eval.frameskip", "rl_games.env_eval.frameskip"),
        ("rl_games.env_eval.image_size", "rl_games.env_eval.image_size"),
        ("rl_games.env_eval.seed", "rl_games.env_eval.seed"),
        ("rl_games.env_eval.fixed_episode_seeds", "rl_games.env_eval.fixed_episode_seeds"),
        ("rl_games.env_eval.latency_seed_stride", "rl_games.env_eval.latency_seed_stride"),
        ("rl_games.env_eval.task_seed_stride", "rl_games.env_eval.task_seed_stride"),
        ("rl_games.env_eval.task_description", "rl_games.env_eval.task_description"),
    ):
        value = _cfg_get(cfg, cfg_path)
        if value not in (None, ""):
            cmd.append(f"{hydra_key}={_hydra_value(value)}")

    latencies = _latencies_expr(_cfg_get(cfg, "rl_games.env_eval.latency.values"))
    if latencies:
        cmd.append(f"rl_games.env_eval.latency.values={latencies}")

    prompt_map = setup.get("latency_prompt_map_path") or _cfg_get(cfg, "rl_games.env_eval.latency.prompt_map_path")
    if prompt_map not in (None, ""):
        cmd.append(f"rl_games.env_eval.latency.prompt_map_path={prompt_map}")

    _append_eval_stage_overrides(cmd, cfg, "env_eval.mid_train", "mid_train")
    _append_eval_stage_overrides(cmd, cfg, "env_eval.post_train", "post_train")

    if str(_cfg_get(cfg, "env")) == "deadly_corridor" or str(_cfg_get(cfg, "rl_games.task") or "") == "deadly_corridor":
        action_layout = _cfg_get(cfg, "rl_games.env_eval.deadly.action_layout") or "multibinary_7"
        cmd.append(f"rl_games.env_eval.deadly.action_layout={action_layout}")

    sync_repo_id = _cfg_get(cfg, "checkpoint.sync.repo_id")
    if sync_repo_id not in (None, ""):
        cmd.append(f"checkpoint.sync.repo_id={sync_repo_id}")

    return cmd


def build_launch_command(cfg: Any, trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    distributed_backend = str(_cfg_get(cfg, "trainer.distributed_backend") or "deepspeed").lower()
    if distributed_backend == "none":
        return [sys.executable, *trainer_cmd]

    if _as_bool(_cfg_get(cfg, "launch.use_accelerate")):
        accelerate_config = _repo_or_workspace_path(_cfg_get(cfg, "paths.accelerate_config"), workspace_dir)
        return [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "--num_processes",
            str(_cfg_get(cfg, "launch.num_processes") or 1),
            *trainer_cmd,
        ]

    return [sys.executable, *trainer_cmd]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--run-root-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = _parse_args(sys.argv[1:] if argv is None else argv)

    hydra_overrides = list(cli_args.overrides)
    if cli_args.workspace_dir not in (None, ""):
        hydra_overrides.append(f"workspace_dir={cli_args.workspace_dir}")
    if cli_args.run_root_dir not in (None, ""):
        hydra_overrides.append(f"paths.run_root_dir={cli_args.run_root_dir}")
        hydra_overrides.append(f"run_root_dir={cli_args.run_root_dir}")
    if cli_args.dry_run:
        hydra_overrides.append("launch.dry_run=true")

    cfg = compose_training_config(
        config_name=str(cli_args.config_name),
        model=str(cli_args.model),
        env=str(cli_args.env),
        init=str(cli_args.init),
        mode=str(cli_args.mode),
        overrides=hydra_overrides,
    )

    workspace_dir = _workspace_dir(cfg)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_cfg_get(cfg, "paths.run_root_dir") or DEFAULT_RUN_ROOT_DIR, workspace_dir)

    login_training_services(cfg, workspace_dir=workspace_dir, repo_root=REPO_ROOT)

    gpus = _cfg_get(cfg, "launch.gpus")
    if gpus not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    with contextlib.redirect_stdout(sys.stderr):
        setup = setup_assets(setup_namespace_from_cfg(cfg, workspace_dir, run_root_dir))

    preprocess_cmd = _cfg_get(cfg, "preprocess_cmd")
    if preprocess_cmd not in (None, ""):
        subprocess.run(str(preprocess_cmd), shell=True, check=True, cwd=str(REPO_ROOT))

    trainer_cmd = build_trainer_command(cfg, setup, workspace_dir, run_root_dir)
    launch_cmd = build_launch_command(cfg, trainer_cmd, workspace_dir)

    print("Setup summary:")
    for key in sorted(setup):
        print(f"  {key}: {setup[key]}")
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))

    if _as_bool(_cfg_get(cfg, "launch.dry_run")):
        return 0

    subprocess.run(launch_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
