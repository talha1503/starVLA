#!/usr/bin/env bash

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=demon_attack \
  init=bridge \
  run_id=openvla_demon_attack_fix_latency_1 \
  paths.dataset_local_dir=data/demon_attack_fix_latency_1 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  checkpoint.load=none \
  rl_games.env_eval.latency.values=[1] \
  rl_games.env_eval.mid_train.latencies=[1] \
