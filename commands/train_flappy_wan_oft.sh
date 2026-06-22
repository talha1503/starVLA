#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=wan_oft \
  env=flappy \
  init=wan_oft_libero \
  run_id=wan_oft_flappy_fix_latency_0_context4 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep_context4 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  datasets.vla_data.per_device_batch_size=1 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
