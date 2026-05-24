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

CONFIG_DIR = REPO_ROOT / "examples/rl_games/config"
SETUP_CONTROLLED_HYDRA_PATHS = {
    "run_root_dir",
    "datasets.vla_data.data_root_dir",
    "datasets.vla_data.data_mix",
    "datasets.vla_data.eval_data_mix",
    "framework.qwenvl.base_vlm",
    "rl_games.env_eval.latency.prompt_map_path",
    "trainer.is_resume",
    "trainer.pretrained_checkpoint",
    "trainer.resume_step",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read experiment configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _iter_leaf_paths(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_paths(child, path)
    elif prefix:
        yield prefix


def _hydra_config_leaf_paths() -> list[str]:
    paths: set[str] = set()
    for path in CONFIG_DIR.rglob("*.yaml"):
        for leaf_path in _iter_leaf_paths(_load_yaml(path)):
            if leaf_path != "defaults" and not leaf_path.startswith("hydra."):
                paths.add(leaf_path)
    return sorted(paths)


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


def _has(cfg: dict[str, Any], path: str) -> bool:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


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
    return str(workspace_dir / path)


def _repo_or_workspace_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return str(workspace_dir / path)


def _workspace_dir(cfg: dict[str, Any]) -> Path:
    configured = _get(cfg, "workspace_dir")
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    default_workspace = Path("/workspace")
    if default_workspace.exists():
        return default_workspace.resolve()
    return REPO_ROOT


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Auth env file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ[key] = _strip_env_quotes(value)


def _load_auth_env(cfg: dict[str, Any], workspace_dir: Path) -> None:
    env_file = _get(cfg, "auth.env_file")
    if env_file not in (None, ""):
        _load_env_file(Path(_repo_or_workspace_path(env_file, workspace_dir)))

    hf_token_env = str(_get(cfg, "auth.hf_token_env", "HF_TOKEN") or "HF_TOKEN")
    hf_token = os.environ.get(hf_token_env)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    wandb_key_env = str(_get(cfg, "auth.wandb_api_key_env", "WANDB_API_KEY") or "WANDB_API_KEY")
    wandb_key = os.environ.get(wandb_key_env)
    if wandb_key:
        os.environ.setdefault("WANDB_API_KEY", wandb_key)


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if isinstance(value, str) and any(ch.isspace() or ch in {",", ":", "{", "}", "[", "]"} for ch in value):
        return shlex.quote(value)
    return str(value)


def _append_hydra_leaf_overrides(cmd: list[str], cfg: dict[str, Any]) -> None:
    for path in _hydra_config_leaf_paths():
        if path in SETUP_CONTROLLED_HYDRA_PATHS:
            continue
        if not _has(cfg, path):
            continue
        value = _get(cfg, path)
        if value == "":
            continue
        cmd.append(f"{path}={_hydra_value(value)}")


def _setup_namespace(cfg: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_get(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    return SimpleNamespace(
        model=str(_get(cfg, "model")),
        env=str(_get(cfg, "env")),
        mode=str(_get(cfg, "mode")),
        latency_mode=str(_get(cfg, "rl_games.env_eval.latency.mode", "") or ""),
        converted_dataset_hf=str(_get(cfg, "dataset.converted_hf", "") or ""),
        dataset_local_dir=_resolve_path(_get(cfg, "datasets.vla_data.data_root_dir"), workspace_dir),
        converted_dataset_name=str(_get(cfg, "datasets.vla_data.data_mix", "flappy_train")),
        dataset_force_download=str(_as_bool(_get(cfg, "dataset.force_download", False))).lower(),
        base_model_dir=_resolve_path(_get(cfg, "framework.qwenvl.base_vlm"), workspace_dir),
        base_model_repo_id=_get(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint_load=str(_get(cfg, "checkpoint.load", "auto")),
        checkpoint_hf_repo_id=str(_get(cfg, "checkpoint.hf_repo_id", "") or ""),
        checkpoint_sync_enabled=str(_as_bool(_get(cfg, "checkpoint.sync.enabled", False))).lower(),
        checkpoint_sync_repo_id=str(_get(cfg, "checkpoint.sync.repo_id", "") or ""),
        hf_repo_id="",
    )


def _trainer_command(cfg: dict[str, Any], setup: dict[str, Any], workspace_dir: Path, run_root_dir: str) -> list[str]:
    cmd = [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        str(_get(cfg, "config_name", "train")),
        f"model={_get(cfg, 'model')}",
        f"env={_get(cfg, 'env')}",
        f"mode={_get(cfg, 'mode')}",
        f"run_root_dir={run_root_dir}",
    ]

    _append_hydra_leaf_overrides(cmd, cfg)

    data_root = setup.get("dataset_local_dir") or _resolve_path(_get(cfg, "datasets.vla_data.data_root_dir"), workspace_dir)
    cmd.append(f"datasets.vla_data.data_root_dir={data_root}")
    if setup.get("data_mix"):
        cmd.append(f"datasets.vla_data.data_mix={setup['data_mix']}")
    if setup.get("eval_data_mix"):
        cmd.append(f"datasets.vla_data.eval_data_mix={setup['eval_data_mix']}")
    if setup.get("base_model_dir"):
        cmd.append(f"framework.qwenvl.base_vlm={setup['base_model_dir']}")

    prompt_map = _get(cfg, "rl_games.env_eval.latency.prompt_map_path") or setup.get("latency_prompt_map_path")
    if prompt_map:
        cmd.append(f"rl_games.env_eval.latency.prompt_map_path={prompt_map}")

    cmd.append(f"trainer.is_resume={str(bool(setup.get('resume_found'))).lower()}")
    if setup.get("resume_checkpoint"):
        cmd.append(f"trainer.pretrained_checkpoint={setup['resume_checkpoint']}")
        cmd.append(f"trainer.resume_step={int(setup.get('resume_step') or 0)}")

    return cmd


def _launch_command(cfg: dict[str, Any], trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    if str(_get(cfg, "trainer.distributed_backend", "deepspeed")).lower() == "none":
        return [sys.executable, *trainer_cmd]
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

    workspace_dir = _workspace_dir(cfg)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_get(cfg, "run_root_dir", "results/Checkpoints"), workspace_dir)
    _load_auth_env(cfg, workspace_dir)

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
