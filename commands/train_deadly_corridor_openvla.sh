#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=deadly_corridor \
  init=bridge \
  run_id=openvla_deadly_corridor_fix_latency_0 \
  paths.dataset_local_dir=data/deadly_corridor_fix_latency_0 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  datasets.vla_data.per_device_batch_size=16 \
  checkpoint.load=none \
  "$@"
