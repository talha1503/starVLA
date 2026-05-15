#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.rl_games import RlGamesEvalRunner, apply_action_spec, apply_model_alias


def _str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _flatten_eval_for_wandb(result, stage: str) -> dict:
    payload = {}
    aggregate = result.aggregate
    prefix = f"rl_games_eval/{stage}"
    payload[f"{prefix}/total_episodes"] = float(aggregate.get("total_episodes", 0))
    payload[f"{prefix}/mean_reward"] = float(aggregate.get("mean_reward", 0.0))
    payload[f"{prefix}/mean_length"] = float(aggregate.get("mean_length", 0.0))
    payload[f"{prefix}/task_count"] = float(aggregate.get("task_count", 0))

    for key, metrics in result.per_latency.items():
        key_slug = key.replace("/", "__")
        payload[f"{prefix}/{key_slug}/mean_reward"] = float(metrics.get("mean_reward", 0.0))
        payload[f"{prefix}/{key_slug}/mean_length"] = float(metrics.get("mean_length", 0.0))
        payload[f"{prefix}/{key_slug}/num_episodes"] = float(metrics.get("num_episodes", 0))
    return payload


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None, step: int | None) -> tuple[Path, int]:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path, _step_from_name(path.name)

    if step is not None:
        pt = checkpoint_dir / f"steps_{step}_pytorch_model.pt"
        st = checkpoint_dir / f"steps_{step}_model.safetensors"
        if pt.exists():
            return pt, step
        if st.exists():
            return st, step
        raise FileNotFoundError(f"No checkpoint found for step={step} in {checkpoint_dir}")

    candidates = []
    for file_path in checkpoint_dir.glob("steps_*_pytorch_model.pt"):
        s = _step_from_name(file_path.name)
        candidates.append((s, file_path))
    for file_path in checkpoint_dir.glob("steps_*_model.safetensors"):
        s = _step_from_name(file_path.name)
        candidates.append((s, file_path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1], candidates[-1][0]


def _step_from_name(name: str) -> int:
    try:
        return int(name.split("steps_")[1].split("_")[0])
    except Exception:
        return 0


def _load_model(cfg, checkpoint_path: Path):
    model = build_framework(cfg)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--step", default=None, type=int)
    parser.add_argument("--stage", default="post_train", type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--wandb-enabled", default="true", type=str)
    parser.add_argument("--wandb-project", default=None, type=str)
    parser.add_argument("--wandb-entity", default=None, type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if args.config else run_dir / "config.full.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    cfg = apply_config_compat(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    cfg.output_dir = str(run_dir)

    checkpoint_path, step = _resolve_checkpoint(run_dir=run_dir, checkpoint=args.checkpoint, step=args.step)
    print(f"Using checkpoint: {checkpoint_path}")

    model = _load_model(cfg=cfg, checkpoint_path=checkpoint_path)
    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(run_dir))
    result = runner.run(model=model, step=step, stage=args.stage)
    print(result.aggregate)

    if _str2bool(args.wandb_enabled):
        project = args.wandb_project or str(getattr(cfg, "wandb_project", "starVLA_rl_games"))
        entity = args.wandb_entity if args.wandb_entity is not None else getattr(cfg, "wandb_entity", None)
        run_name = args.wandb_run_name or f"{run_dir.name}__{args.stage}__step_{step}"
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=os.path.join(str(run_dir), "wandb"),
            group="rl-games-eval",
        )
        wandb.log(_flatten_eval_for_wandb(result=result, stage=args.stage), step=step)
        wandb.finish()


if __name__ == "__main__":
    main()
