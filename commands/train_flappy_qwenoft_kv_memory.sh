#!/usr/bin/env bash
# QwenOFT on flappy with fixed-size streaming KV MEMORY (req3), trained WITH the memory
# so train and inference match. The dataloader emits R=rollout_len frames per sample
# (image_mode must be multiframe, not stitch); _forward_memory replays them through the
# memory with truncated BPTT. rollout_len=8 > window=4 so training exercises eviction.
#
# At eval the memory is read from the checkpoint config automatically: only the newest
# frame is encoded each step (past frames come from the per-slot cache), so the eval env
# can keep frame_stack=1. The per-slot memory is reset at episode boundaries.

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_qwenoft_kv_memory_w4_r8 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep \
  datasets.vla_data.num_obs_frames=8 \
  datasets.vla_data.image_mode=multiframe \
  framework.kv_memory.enabled=true \
  framework.kv_memory.window=4 \
  framework.kv_memory.rollout_len=8 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=8 \
  datasets.vla_data.per_device_batch_size=4 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
