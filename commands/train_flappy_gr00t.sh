#!/usr/bin/env bash

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

: "${WANDB_ENTITY:?Set WANDB_ENTITY to your W&B entity before running this training command}"

python examples/rl_games/scripts/launch_train.py \
  model=gr00t \
  env=flappy \
  init=bridge \
  mode=single \
  conda.env_name=starvla_gr00t \
  workspace_dir=/inspire/hdd/project/spatialintelligence/public/lzj/starVLA \
  run_id=gr00t_flappy_fix_latency_0 \
  wandb_entity="$WANDB_ENTITY" \
  paths.dataset_local_dir=data/flappy_fix_latency_0_parquet \
  dataset.source_hf=data/raw/flappy_bird_zero_latency_parquet \
  dataset.converted_name=flappy_train \
  dataset.setup_force=false \
  dataset.force_download=false \
  paths.base_model_dir=playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  initialization.checkpoint_local_dir=playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1 \
  initialization.checkpoint_filename=checkpoints/steps_20000_pytorch_model.pt \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  trainer.max_train_steps=2000 \
  trainer.save_interval=100 \
  trainer.eval_interval=100 \
  checkpoint.load=none \
  checkpoint.local.keep_last_n=2 \
  checkpoint.sync.enabled=false \
  rl_games.env_eval.latency.values=[0] \
  rl_games.env_eval.mid_train.interval_steps=100 \
  rl_games.env_eval.mid_train.latencies=[0] \
  rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
