#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=wan_oft \
  env=flappy \
  init=wan_oft_libero \
  run_id=wan_oft_flappy_fix_latency_0_context4_horizon1 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep_context4 \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.reload_modules=[backbone,action_model] \
  datasets.vla_data.per_device_batch_size=8 \
  datasets.vla_data.data_mix=flappy_train__bridge \
  datasets.vla_data.eval_data_mix=flappy_train__bridge__val \
  datasets.vla_data.action_indices=[0] \
  framework.action_model.action_horizon=1 \
  framework.action_model.future_action_window_size=0 \
  framework.action_model.past_action_window_size=0 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  rl_games.env_eval.enabled=true \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.interval_steps=100 \
  rl_games.env_eval.mid_train.latencies=[0] \
  rl_games.env_eval.mid_train.num_episodes=5 \
  rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
