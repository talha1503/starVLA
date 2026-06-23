#!/usr/bin/env bash
set -euo pipefail

# launch_train.py uses relative paths; run from starVLA/.
cd "$(dirname "$0")/../.."

# Corrected: freeze ViT/visual connector and freeze LLM layers [0,27], leaving LLM top 8 layers + action head trainable.
# Qwen3-VL-4B text decoder total=36.
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_last_8_layers_corrected \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=2 \
  trainer.freeze_vit=false \
  trainer.freeze_llm_layers=[0,27] \
  datasets.vla_data.per_device_batch_size=64 \
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
