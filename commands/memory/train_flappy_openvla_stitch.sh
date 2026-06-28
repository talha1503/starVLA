#!/usr/bin/env bash
# QwenOFT on flappy with STITCHED multi-image input (token count of a single image).
# num_obs_frames=4 + stitch_grid=[2,2] -> the last 4 frames are tiled into one 224x224
# canvas, so the model sees temporal context at the cost of per-frame resolution but no
# extra tokens. Set the eval env's frame_stack to 4 to match.

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_qwenoft_stitch2x2 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep \
  datasets.vla_data.num_obs_frames=4 \
  datasets.vla_data.image_mode=stitch \
  'datasets.vla_data.stitch_grid=[2,2]' \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=1 \
  datasets.vla_data.per_device_batch_size=32 \
  trainer.max_train_steps=4000 \
  trainer.save_interval=500 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_safetensors_file=true \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
