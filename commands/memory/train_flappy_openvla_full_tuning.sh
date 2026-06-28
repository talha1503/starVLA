#!/usr/bin/env bash
set -euo pipefail

# launch_train.py 用相对路径调用，必须在 starVLA/ 目录下运行。
# 这里自己 cd 到 starVLA/（脚本在 starVLA/commands/memory/ 下，../.. = starVLA/），
# 这样无论从哪个 cwd `bash` 本脚本都不会 file-not-found。
cd "$(dirname "$0")/../.."

# freeze 底部 N 层：n = round(ratio * total)，冻结的是 language_model.layers 前 n 层（靠近输入的底部）。
# Qwen3-VL-4B 文本解码层 total=36（config.json text_config.num_hidden_layers）。
# 解冻最后 8 层 => 冻结底部 28 层 => ratio = 28/36 = 0.7778（round(0.7778*36)=28）。
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_full_tuning \
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
  checkpoint.save_safetensors_file=true \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
