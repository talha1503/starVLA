#!/usr/bin/env bash
set -euo pipefail

# launch_train.py 用相对路径调用，必须在 starVLA/ 目录下运行。
# 这里自己 cd 到 starVLA/（脚本在 starVLA/commands/memory/ 下，../.. = starVLA/），
# 这样无论从哪个 cwd `bash` 本脚本都不会 file-not-found。
cd "$(dirname "$0")/../.."

# v2: freeze ViT/connector, tied embedding/lm_head, and LLM bottom layers.
# OpenVLA action head uses hidden states, not lm_head logits.
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_last_8_layers_corrected \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=2 \
  trainer.freeze_vit=true \
  trainer.freeze_tied_embedding=true \
  trainer.freeze_llm_layers=[0,27] \
  datasets.vla_data.per_device_batch_size=64 \
  datasets.vla_data.image_mode=single \
  datasets.vla_data.num_obs_frames=1 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=500 \
  trainer.profile_timing.enabled=true \
  trainer.profile_timing.log_interval=10 \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[2] \
  rl_games.env_eval.mid_train.interval_steps=250 \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
