#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# full tuning: no freeze_vit/freeze_llm_layers overrides.
# bs32: only use per_device_batch_size=32, gradient_accumulation_steps=1, effective_batch_size=32
# compared with corrected, see if bs=32 does not harm result
# 3k steps: compared with bs32 whose total training steps to be 5k
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_full_tuning_corrected_bs32_3k_steps \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=1 \
  trainer.eval_interval=250 \
  trainer.eval_num_batches=50 \
  datasets.vla_data.per_device_batch_size=32 \
  datasets.vla_data.image_mode=single \
  datasets.vla_data.num_obs_frames=1 \
  trainer.max_train_steps=3000 \
  trainer.save_interval=500 \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[2] \
  rl_games.env_eval.mid_train.interval_steps=250 \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
