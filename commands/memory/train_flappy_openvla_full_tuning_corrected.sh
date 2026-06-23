#!/usr/bin/env bash
set -euo pipefail

# launch_train.py 用相对路径调用，必须在 starVLA/ 目录下运行。
# 这里自己 cd 到 starVLA/（脚本在 starVLA/commands/memory/ 下，../.. = starVLA/），
# 这样无论从哪个 cwd `bash` 本脚本都不会 file-not-found。
cd "$(dirname "$0")/../.."

# full tuning：不设置 freeze_llm_bottom_ratio，保持 LLM 全量参与训练。
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_full_tuning_corrected \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=4 \
  datasets.vla_data.per_device_batch_size=32 \
  datasets.vla_data.image_mode=single \
  datasets.vla_data.num_obs_frames=1 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=500 \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[2] \
  rl_games.env_eval.mid_train.interval_steps=250 \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
