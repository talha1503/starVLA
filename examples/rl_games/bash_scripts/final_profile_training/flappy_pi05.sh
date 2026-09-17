#!/usr/bin/env bash

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

conda activate starvla_rl_games_pi05

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=pi05 \
    env=flappy \
    init=bridge \
    mode=mixed_latency \
    run_id="pi05_bridge_flappy_rtx3090_profile_1000ep_7k2steps_final" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/pi05_bridge_flappy_rtx3090_profile_1000ep_7k2steps_final" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="latency-sensitive-bench/pi05_bridge_flappy_rtx3090_profile_1000ep_7k2steps_final" \
    dataset.source_hf=latency-sensitive-bench/memory-rollouts \
    dataset.source_subdir=flappy_pi05_rtx3090_profile_1000ep_7k2steps \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=False \
    trainer.max_train_steps=5000 \
    trainer.save_interval=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=5000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
