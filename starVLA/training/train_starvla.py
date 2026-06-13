# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.rl_games import CheckpointSyncManager, RlGamesEvalRunner, apply_action_spec, apply_model_alias, validate_rl_games_config
from starVLA.training.rl_games.auth import login_training_services
from starVLA.training.rl_games.eval_core import EvalResult
from starVLA.training.rl_games import action_cc_f1
from starVLA.training.train_step_events import should_run_step_interval_event
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = logging.getLogger(__name__)


def _as_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> tuple[DataLoader, DataLoader | None]:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vla_eval_dataloader = None
    eval_data_mix = getattr(cfg.datasets.vla_data, "eval_data_mix", None)
    if eval_data_mix:
        logger.info(f"Creating VLA Eval Dataset with Mixture `{eval_data_mix}`")
        vla_eval_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            data_mix=str(eval_data_mix),
            mode="eval",
            save_statistics_filename="dataset_statistics_eval.json",
        )

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader, vla_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    fused = bool(getattr(cfg.trainer.optimizer, "fused", False))
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def _build_accelerator(cfg) -> Accelerator:
    grad_accum_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    distributed_backend = str(getattr(cfg.trainer, "distributed_backend", "deepspeed")).lower()
    accelerator_kwargs = {"gradient_accumulation_steps": grad_accum_steps}
    if distributed_backend == "deepspeed":
        accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin()
    local_accelerator = Accelerator(**accelerator_kwargs)
    local_accelerator.print(local_accelerator.state)
    return local_accelerator


def _pin_cuda_device_from_local_rank() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank in (None, ""):
        return
    try:
        device_idx = int(local_rank)
    except ValueError:
        logger.warning("Ignoring invalid LOCAL_RANK=%r for CUDA device pinning", local_rank)
        return

    visible_devices = torch.cuda.device_count()
    if device_idx < 0 or device_idx >= visible_devices:
        logger.warning(
            "LOCAL_RANK=%s cannot be mapped to a CUDA device; visible CUDA device count is %s",
            local_rank,
            visible_devices,
        )
        return
    torch.cuda.set_device(device_idx)


def _preload_model_checkpoint_before_accelerator(cfg, model):
    """Load model-only checkpoints before ZeRO-3 can replace params with shards."""
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return model

    pretrained_checkpoint = getattr(trainer_cfg, "pretrained_checkpoint", None)
    is_resume = bool(getattr(trainer_cfg, "is_resume", False))
    if not pretrained_checkpoint or is_resume:
        return model

    reload_modules = getattr(trainer_cfg, "reload_modules", None)
    model = TrainerUtils.load_pretrained_backbones(
        model,
        pretrained_checkpoint,
        reload_modules=reload_modules,
    )
    trainer_cfg.pretrained_checkpoint = None
    logger.info(
        "Preloaded model checkpoint before Accelerator/DeepSpeed initialization: %s",
        pretrained_checkpoint,
    )
    return model


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vla_eval_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_eval_dataloader = vla_eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._checkpoint_sync_manager = CheckpointSyncManager(cfg=self.config)
        self._rl_games_eval_runner = None
        self._pending_full_state_resume = None
        self._save_best_model_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_best_model", None),
            default=True,
        )
        self._save_pt_file_enabled = _as_bool(
            getattr(getattr(self.config, "checkpoint", {}), "save_pt_file", None),
            default=False,
        )
        self._best_score = float("-inf")
        self._best_step = 0
        self._best_state_path = None
        self._best_metadata_path = None
        if hasattr(self.config, "rl_games") and hasattr(self.config.rl_games, "env_eval"):
            enabled = bool(getattr(self.config.rl_games.env_eval, "enabled", False))
            if enabled:
                self._rl_games_eval_runner = RlGamesEvalRunner(cfg=self.config, output_dir=self.config.output_dir)

    def _save_periodic_checkpoints_enabled(self) -> bool:
        return not self._save_best_model_enabled

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        if self.vla_eval_dataloader is not None:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_eval_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vla_eval_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        if self._pending_full_state_resume:
            self._load_checkpoint(self._pending_full_state_resume)
        else:
            self._adjust_lr_scheduler_for_resume()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._best_state_path = os.path.join(self.checkpoint_dir, "best_state")
        self._best_metadata_path = os.path.join(self.checkpoint_dir, "best_model_metadata.json")
        self._load_best_checkpoint_metadata()

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            if pretrained_checkpoint:
                resume_from_checkpoint = str(pretrained_checkpoint)
                self.completed_steps = int(
                    getattr(self.config.trainer, "resume_step", None) or self._step_from_checkpoint_path(resume_from_checkpoint)
                )
            else:
                resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(self.resume_from_checkpoint):
                    self._pending_full_state_resume = self.resume_from_checkpoint
                    logger.info(
                        f"Will resume full training state from checkpoint: "
                        f"{self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                else:
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                    logger.info(
                        f"Resuming model weights from checkpoint: "
                        f"{self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    @staticmethod
    def _step_from_checkpoint_path(checkpoint_path: str) -> int:
        match = re.search(r"steps_(\d+)", os.path.basename(str(checkpoint_path)))
        return int(match.group(1)) if match else 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _load_best_checkpoint_metadata(self):
        if not self._best_metadata_path or not os.path.isfile(self._best_metadata_path):
            return
        try:
            with open(self._best_metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self._best_score = float(metadata.get("best_score", self._best_score))
            self._best_step = int(metadata.get("best_step", self._best_step))
            logger.info(
                "Loaded best checkpoint metadata: step=%s score=%s",
                self._best_step,
                self._best_score,
            )
        except Exception as exc:
            logger.warning("Could not load best checkpoint metadata from %s: %s", self._best_metadata_path, exc)

    @staticmethod
    def _rl_games_best_score(eval_result) -> float:
        if "macro_mean_reward" in eval_result.aggregate:
            return float(eval_result.aggregate["macro_mean_reward"])
        bucket_scores = [
            float(metrics["mean_reward"])
            for metrics in eval_result.per_latency.values()
            if metrics is not None and "mean_reward" in metrics
        ]
        if bucket_scores:
            return float(np.mean(bucket_scores))
        return float(eval_result.aggregate.get("mean_reward", 0.0))

    @staticmethod
    def _eval_result_payload(eval_result) -> Dict[str, Any]:
        return {
            "per_latency": eval_result.per_latency,
            "aggregate": eval_result.aggregate,
        }

    def _distributed_rl_games_eval_enabled(self) -> bool:
        if self._rl_games_eval_runner is None:
            return False
        env_eval = getattr(self.config.rl_games, "env_eval", None)
        mode = str(getattr(env_eval, "distributed_mode", "none") or "none").strip().lower()
        return mode in {"rank_sharded", "distributed", "sharded"} and int(self.accelerator.num_processes) > 1

    def _run_distributed_rl_games_eval(self, stage: str):
        if self._rl_games_eval_runner is None:
            return None

        local_result = self._rl_games_eval_runner.run(
            model=self.accelerator.unwrap_model(self.model),
            step=self.completed_steps,
            stage=stage,
            shard_rank=int(self.accelerator.process_index),
            shard_count=int(self.accelerator.num_processes),
            save=False,
        )
        local_payload = self._eval_result_payload(local_result)

        if dist.is_available() and dist.is_initialized():
            gathered_payloads = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_payloads, local_payload)
        else:
            gathered_payloads = [local_payload]

        partial_results = [
            EvalResult(per_latency=payload["per_latency"], aggregate=payload["aggregate"])
            for payload in gathered_payloads
            if payload is not None
        ]
        merged_result = self._rl_games_eval_runner.merge_results(
            partial_results,
            step=self.completed_steps,
            stage=stage,
        )
        if self.accelerator.is_main_process:
            self._rl_games_eval_runner.save(result=merged_result, step=self.completed_steps, stage=stage)
        self.accelerator.wait_for_everyone()
        return merged_result

    def _best_config_paths(self) -> list[str]:
        output_dir = Path(self.config.output_dir)
        return [
            str(output_dir / "config.full.yaml"),
            str(output_dir / "config.yaml"),
        ]

    def _save_best_checkpoint(self, eval_result, score: float, stage: str) -> bool:
        if not self._save_best_model_enabled:
            return False
        if score <= self._best_score:
            return False
        if not self._best_state_path or not self._best_metadata_path:
            raise RuntimeError("Best checkpoint paths were not initialized")

        if self.accelerator.is_main_process and os.path.exists(self._best_state_path):
            shutil.rmtree(self._best_state_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self._best_state_path, safe_serialization=True)
        self.accelerator.wait_for_everyone()

        self._best_score = float(score)
        self._best_step = int(self.completed_steps)

        if self.accelerator.is_main_process:
            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            metadata = {
                "best_step": self._best_step,
                "best_score": self._best_score,
                "score_name": "mid_train_macro_mean_reward",
                "stage": stage,
                "eval_result_path": getattr(eval_result, "path", None),
                "aggregate": eval_result.aggregate,
                "per_latency_scores": {
                    key: float(metrics.get("mean_reward", 0.0))
                    for key, metrics in eval_result.per_latency.items()
                },
            }
            with open(self._best_metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            self.accelerator.print(
                f"✅ New best checkpoint saved at {self._best_state_path} "
                f"(step={self._best_step}, score={self._best_score:.6f})"
            )
            self._checkpoint_sync_manager.sync_best_checkpoint(
                state_path=self._best_state_path,
                metadata_path=self._best_metadata_path,
                config_paths=self._best_config_paths(),
            )
        self.accelerator.wait_for_everyone()
        return True

    def _save_checkpoint(self):
        """Save current training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        state_checkpoint_path = checkpoint_path + "_state"

        if self.accelerator.is_main_process and os.path.exists(state_checkpoint_path):
            shutil.rmtree(state_checkpoint_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(state_checkpoint_path, safe_serialization=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.print(f"✅ Full training state saved at {state_checkpoint_path}")

        if self.accelerator.is_main_process:
            model_checkpoint_path = None
            if self._save_pt_file_enabled:
                state_dict = self.accelerator.get_state_dict(self.model)
                model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"
                torch.save(state_dict, model_checkpoint_path)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint state saved at {state_checkpoint_path}")
            if model_checkpoint_path is not None:
                self.accelerator.print(f"✅ Model checkpoint saved at {model_checkpoint_path}")
            self._checkpoint_sync_manager.register_local_checkpoint(
                step=self.completed_steps,
                state_path=state_checkpoint_path,
                model_path=model_checkpoint_path,
            )

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    @staticmethod
    def _total_grad_norm(parameters) -> float:
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if not grads:
            return 0.0
        device = grads[0].device
        norms = torch.stack([torch.linalg.vector_norm(g.to(device), 2) for g in grads])
        return float(torch.linalg.vector_norm(norms, 2).item())

    @staticmethod
    def _append_rl_games_eval_metrics(step_metrics: Dict[str, float], eval_result, stage: str) -> Dict[str, float]:
        aggregate = eval_result.aggregate
        prefix = f"rl_games_eval/{stage}"
        step_metrics[f"{prefix}/total_episodes"] = float(aggregate.get("total_episodes", 0))
        step_metrics[f"{prefix}/mean_reward"] = float(aggregate.get("mean_reward", 0.0))
        step_metrics[f"{prefix}/mean_length"] = float(aggregate.get("mean_length", 0.0))
        step_metrics[f"{prefix}/std_reward"] = float(aggregate.get("std_reward", 0.0))
        step_metrics[f"{prefix}/std_length"] = float(aggregate.get("std_length", 0.0))
        step_metrics[f"{prefix}/macro_mean_reward"] = float(
            aggregate.get("macro_mean_reward", aggregate.get("mean_reward", 0.0))
        )
        step_metrics[f"{prefix}/macro_mean_length"] = float(
            aggregate.get("macro_mean_length", aggregate.get("mean_length", 0.0))
        )
        step_metrics[f"{prefix}/task_count"] = float(aggregate.get("task_count", 0))

        for key, metrics in eval_result.per_latency.items():
            key_slug = key.replace("/", "__")
            step_metrics[f"{prefix}/{key_slug}/mean_reward"] = float(metrics.get("mean_reward", 0.0))
            step_metrics[f"{prefix}/{key_slug}/mean_length"] = float(metrics.get("mean_length", 0.0))
            step_metrics[f"{prefix}/{key_slug}/std_reward"] = float(metrics.get("std_reward", 0.0))
            step_metrics[f"{prefix}/{key_slug}/std_length"] = float(metrics.get("std_length", 0.0))
            step_metrics[f"{prefix}/{key_slug}/num_episodes"] = float(metrics.get("num_episodes", 0))
        return step_metrics

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            gradients_synced = bool(self.accelerator.sync_gradients)
            if gradients_synced:
                if should_run_step_interval_event(
                    completed_steps=self.completed_steps,
                    interval=self.config.trainer.eval_interval,
                    gradients_synced=gradients_synced,
                ):
                    step_metrics = self.eval_action_loss(step_metrics)
                    step_metrics = self.eval_action_cc_f1(step_metrics)
                if self._rl_games_eval_runner is not None:
                    eval_every = self._rl_games_eval_runner.interval_steps()
                    if eval_every > 0 and should_run_step_interval_event(
                        completed_steps=self.completed_steps,
                        interval=eval_every,
                        gradients_synced=gradients_synced,
                    ):
                        if self._rl_games_eval_runner.is_enabled(stage="mid_train"):
                            if self._distributed_rl_games_eval_enabled():
                                eval_result = self._run_distributed_rl_games_eval(stage="mid_train")
                            else:
                                eval_result = self._rl_games_eval_runner.run(
                                    model=self.accelerator.unwrap_model(self.model),
                                    step=self.completed_steps,
                                    stage="mid_train",
                                )
                            step_metrics = self._append_rl_games_eval_metrics(
                                step_metrics=step_metrics,
                                eval_result=eval_result,
                                stage="mid_train",
                            )
                            eval_score = self._rl_games_best_score(eval_result)
                            is_new_best = self._save_best_checkpoint(
                                eval_result=eval_result,
                                score=eval_score,
                                stage="mid_train",
                            )
                            step_metrics["checkpoint/current_eval_score"] = float(eval_score)
                            step_metrics["checkpoint/best_score"] = float(self._best_score)
                            step_metrics["checkpoint/best_step"] = float(self._best_step)
                            step_metrics["checkpoint/is_new_best"] = float(is_new_best)
                            self._checkpoint_sync_manager.sync_eval_result(
                                eval_path=eval_result.path,
                                stage="mid_train",
                                step=self.completed_steps,
                            )

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if should_run_step_interval_event(
                completed_steps=self.completed_steps,
                interval=self.config.trainer.logging_frequency,
                gradients_synced=gradients_synced,
            ):
                self._log_metrics(step_metrics)

            if self._save_periodic_checkpoints_enabled() and should_run_step_interval_event(
                completed_steps=self.completed_steps,
                interval=self.config.trainer.save_interval,
                gradients_synced=gradients_synced,
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_loss(self, step_metrics: dict = None) -> dict:
        """Compute held-out action loss over the validation LeRobot dataset."""
        step_metrics = step_metrics or {}
        if self.vla_eval_dataloader is None:
            return step_metrics

        was_training = self.model.training
        self.model.eval()
        losses = []
        num_batches = int(getattr(self.config.trainer, "eval_num_batches", 20))
        with torch.no_grad():
            for batch_idx, batch_vla in enumerate(self.vla_eval_dataloader):
                if batch_idx >= num_batches:
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                losses.append(self.accelerator.gather(action_loss.detach().reshape(1)).mean())

        if was_training:
            self.model.train()

        if losses and self.accelerator.is_main_process:
            eval_loss = torch.stack(losses).mean().item()
            step_metrics["eval/loss"] = eval_loss

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Backward-compatible alias for the held-out action loss evaluator."""
        return self.eval_action_loss(step_metrics)

    def eval_action_cc_f1(self, step_metrics: dict = None) -> dict:
        """Component-based Control-Critical F1 over the held-out teacher set.

        Unlike ``eval_action_loss`` (which samples frames randomly via the
        shuffled eval dataloader), CC-F1 needs *contiguous, fully-covered*
        episodes so the ``±K`` temporal-tolerance matching is meaningful. So this
        runs a dedicated sequential pass: it walks each underlying LeRobot
        dataset's ``all_steps`` in order (bypassing the random mixture sampler),
        shards whole episodes across ranks, runs ``predict_action`` (the same
        decode path as rollout), decodes per-frame action *components* and matches
        teacher vs model events per episode. Counts are all-reduced and reduced to
        CC-F1 on the main process. See ``action_cc_f1`` for the metric definition.
        """
        step_metrics = step_metrics or {}
        if self.vla_eval_dataloader is None:
            return step_metrics
        task = str(getattr(getattr(self.config, "rl_games", None), "task", "") or "").lower()
        if task not in action_cc_f1.SUPPORTED_TASKS:
            return step_metrics
        if not bool(getattr(self.config.trainer, "eval_action_classification", True)):
            return step_metrics

        # Resolve spec (deadly_corridor depends on its action layout) and tolerance.
        # Read the layout exactly like action_spec._deadly_action_dim (same default)
        # so the decode matches the action dim the model was actually trained with.
        deadly_layout = action_cc_f1.DEADLY_MULTIBINARY_7
        if task == "deadly_corridor":
            deadly_cfg = getattr(getattr(getattr(self.config, "rl_games", None), "env_eval", None), "deadly", None)
            deadly_layout = str(getattr(deadly_cfg, "action_layout", action_cc_f1.DEADLY_MULTIBINARY_7))
        spec = action_cc_f1.get_spec(task, deadly_layout)
        k = spec.default_k
        if task != "flappy":  # flappy stays per-frame (K=0) to match the shipped flap-F1
            override = getattr(self.config.trainer, "cc_f1_tolerance", None)
            if override is not None:
                k = int(override)

        dataset = getattr(self.vla_eval_dataloader, "dataset", None)
        if dataset is None:
            return step_metrics
        single_datasets = list(getattr(dataset, "datasets", None) or [dataset])

        # Enumerate episodes as contiguous same-trajectory runs of all_steps.
        # episodes: list of (single_ds, episode_key, [(flat_idx, base_index), ...]).
        episodes = []
        for ds in single_datasets:
            all_steps = getattr(ds, "all_steps", None)
            if not all_steps:
                continue
            tag = getattr(ds, "tag", "ds")
            cur_traj = None
            cur = []
            for flat_idx, step in enumerate(all_steps):
                traj_id, base_idx = step
                if cur and traj_id != cur_traj:
                    episodes.append((ds, f"{tag}:{cur_traj}", cur))
                    cur = []
                cur_traj = traj_id
                cur.append((int(flat_idx), int(base_idx)))
            if cur:
                episodes.append((ds, f"{tag}:{cur_traj}", cur))
        if not episodes:
            return step_metrics

        # Shard whole episodes across ranks, bounded by a per-rank frame budget.
        num_procs = int(self.accelerator.num_processes)
        rank = int(self.accelerator.process_index)
        bs = int(self.config.datasets.vla_data.per_device_batch_size)
        frame_budget = max(1, int(getattr(self.config.trainer, "eval_num_batches", 20)) * bs)
        assigned = []
        frame_count = 0
        for ep_idx, ep in enumerate(episodes):
            if ep_idx % num_procs != rank:
                continue
            assigned.append(ep)
            frame_count += len(ep[2])
            if frame_count >= frame_budget:
                break

        was_training = self.model.training
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        counts = action_cc_f1.new_counts(spec)

        # Materialize assigned frames (tagged with their episode), batch for inference.
        tagged = []  # (episode_key, base_index, packed_sample)
        for ds, episode_key, frames in assigned:
            for flat_idx, base_idx in frames:
                tagged.append((episode_key, base_idx, ds[flat_idx]))

        per_ep = {}  # episode_key -> list of (base_index, teacher_components, model_components)
        with torch.no_grad():
            for start in range(0, len(tagged), bs):
                chunk = tagged[start : start + bs]
                normalized = unwrapped.predict_action(examples=[c[2] for c in chunk])["normalized_actions"]
                for i, (episode_key, base_idx, sample) in enumerate(chunk):
                    teacher_vec = np.asarray(sample["action"])[0, :]
                    model_vec = normalized[i, 0, :]
                    per_ep.setdefault(episode_key, []).append(
                        (base_idx, spec.comp_fn(teacher_vec), spec.comp_fn(model_vec))
                    )

        if was_training:
            self.model.train()

        for items in per_ep.values():
            items.sort(key=lambda r: r[0])
            action_cc_f1.accumulate_episode(
                spec,
                frames=[r[0] for r in items],
                teacher_comps=[r[1] for r in items],
                model_comps=[r[2] for r in items],
                k=k,
                counts=counts,
            )

        # Counts are small integers (well within float32's exact range); float32
        # keeps NCCL all_reduce broadly compatible across torch builds.
        counts_t = torch.tensor(counts, device=self.accelerator.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)
        if self.accelerator.is_main_process:
            step_metrics.update(action_cc_f1.reduce_metrics(spec, counts_t.detach().cpu().numpy()))

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            elif self.accelerator.sync_gradients:
                grad_norm = self._total_grad_norm(self.model.parameters())

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        action_loss_value = action_loss.item()
        metrics = {
            "train/loss": action_loss_value,
        }
        if grad_norm is not None:
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.detach().float().item()
            metrics["train/grad_norm_pre_clip"] = float(grad_norm)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        save_interval = int(getattr(self.config.trainer, "save_interval", 0) or 0)
        if (
            self._save_periodic_checkpoints_enabled()
            and self.completed_steps > 0
            and (save_interval <= 0 or self.completed_steps % save_interval != 0)
        ):
            self._save_checkpoint()

        if self.accelerator.is_main_process:
            if self._save_best_model_enabled:
                logger.info(
                    "Training complete. Best checkpoint step=%s score=%s path=%s",
                    self._best_step,
                    self._best_score,
                    self._best_state_path,
                )
            else:
                logger.info(f"Training complete. Final training state is saved under {self.checkpoint_dir}")

        if (
            self._save_best_model_enabled
            and self._best_state_path
            and os.path.isdir(self._best_state_path)
        ):
            self.accelerator.print(
                f"Loading best checkpoint for post-train eval: "
                f"{self._best_state_path} (step={self._best_step}, score={self._best_score:.6f})"
            )
            self._load_checkpoint(self._best_state_path)
        elif self._save_best_model_enabled:
            logger.warning(
                "checkpoint.save_best_model=true but no best_state checkpoint was found; "
                "post-train eval will use the current in-memory model."
            )
        self.accelerator.wait_for_everyone()

        if self._rl_games_eval_runner is not None and self._rl_games_eval_runner.is_enabled(stage="post_train"):
            if self._distributed_rl_games_eval_enabled():
                eval_result = self._run_distributed_rl_games_eval(stage="post_train")
            elif self.accelerator.is_main_process:
                eval_result = self._rl_games_eval_runner.run(
                    model=self.accelerator.unwrap_model(self.model),
                    step=self.completed_steps,
                    stage="post_train",
                )
            else:
                eval_result = None

            if self.accelerator.is_main_process and eval_result is not None:
                final_metrics = self._append_rl_games_eval_metrics(
                    step_metrics={},
                    eval_result=eval_result,
                    stage="post_train",
                )
                if self._save_best_model_enabled:
                    final_metrics["checkpoint/best_score"] = float(self._best_score)
                    final_metrics["checkpoint/best_step"] = float(self._best_step)
                self._checkpoint_sync_manager.sync_eval_result(
                    eval_path=eval_result.path,
                    stage="post_train",
                    step=self.completed_steps,
                )
                wandb.log(final_metrics, step=self.completed_steps)

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    _pin_cuda_device_from_local_rank()
    logger.info("VLA Training :: Warming Up")
    if hasattr(cfg, "rl_games"):
        login_training_services(cfg, workspace_dir=getattr(cfg, "workspace_dir", None))
        validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla = _preload_model_checkpoint_before_accelerator(cfg=cfg, model=vla)

    accelerator = _build_accelerator(cfg)

    vla_train_dataloader, vla_eval_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_eval_dataloader=vla_eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
