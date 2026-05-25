#!/usr/bin/env bash

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=openvla_flappy_fix_latency_0 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_parquet \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  trainer.max_train_steps=5000 \
  checkpoint.load=local \
  rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
