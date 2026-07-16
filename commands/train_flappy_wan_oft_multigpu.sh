#!/usr/bin/env bash

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

python examples/rl_games/scripts/launch_train.py \
  model=wan_oft \
  env=flappy \
  init=wan_oft_libero \
  run_id=wan_oft_flappy_fix_latency_0_context4_multigpu \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep_context4 \
  trainer.distributed_backend=deepspeed \
  launch.use_accelerate=true \
  launch.gpus="'${WAN_OFT_GPUS:-0,1}'" \
  launch.num_processes=${WAN_OFT_NUM_PROCESSES:-2} \
  trainer.gradient_accumulation_steps=8 \
  datasets.vla_data.per_device_batch_size=8 \
  datasets.vla_data.data_mix=flappy_train__bridge \
  datasets.vla_data.eval_data_mix=flappy_train__bridge__val \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  rl_games.env_eval.enabled=true \
  rl_games.env_eval.eval_backend=eval_core \
  rl_games.env_eval.distributed_mode=rank_sharded \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[0] \
  rl_games.env_eval.post_train.enabled=true \
  rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
