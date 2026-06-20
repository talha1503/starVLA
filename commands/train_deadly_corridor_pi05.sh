#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=bridge \
  run_id=pi05_deadly_corridor_fix_latency_0_1000ep \
  paths.dataset_local_dir=data/deadly_corridor_fix_latency_0_1000ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=8 \
  datasets.vla_data.per_device_batch_size=16 \
  trainer.max_train_steps=500 \
  trainer.save_interval=100 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
