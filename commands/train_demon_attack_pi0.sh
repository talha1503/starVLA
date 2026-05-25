#!/usr/bin/env bash

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

: "${WANDB_ENTITY:?Set WANDB_ENTITY to your W&B entity before running this training command}"

python examples/rl_games/scripts/launch_train.py \
  model=pi0 \
  env=demon_attack \
  init=bridge \
  mode=single \
  conda.env_name=starvla_pi0 \
  workspace_dir=/inspire/hdd/project/spatialintelligence/public/lzj/starVLA \
  run_id=pi0_demon_attack_fix_latency_1 \
  wandb_entity="$WANDB_ENTITY" \
  paths.dataset_local_dir=data/demon_attack_fix_latency_1 \
  dataset.converted_name=demon_attack_train \
  dataset.setup_force=false \
  dataset.force_download=false \
  paths.base_model_dir=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  base_model.repo_id=StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  initialization.checkpoint_local_dir=playground/Pretrained_models/Qwen-PI-Bridge-RT-1 \
  initialization.checkpoint_hf_repo_id=StarVLA/Qwen-PI-Bridge-RT-1 \
  initialization.checkpoint_filename=checkpoints/steps_30000_pytorch_model.pt \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  trainer.max_train_steps=2000 \
  trainer.save_interval=100 \
  trainer.eval_interval=100 \
  checkpoint.load=none \
  checkpoint.local.keep_last_n=2 \
  checkpoint.sync.enabled=false \
  rl_games.env_eval.latency.values=[1] \
  rl_games.env_eval.mid_train.interval_steps=100 \
  rl_games.env_eval.mid_train.latencies=[1] \
  rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
