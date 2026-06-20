#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  mode=curriculum_exclusive \
  run_id=flappy_curriculum_exclusive_40ep_per_latency \
  paths.dataset_local_dir=data/flappy_mixed_latency_40ep_per_lat \
  'dataset.latency_filter=[0,1,2,3,4]' \
  dataset.episodes_per_latency=40 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=4 \
  datasets.vla_data.per_device_batch_size=32 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none