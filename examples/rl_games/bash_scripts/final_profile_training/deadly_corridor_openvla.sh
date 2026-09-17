#!/usr/bin/env bash

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=deadly_corridor \
    init=bridge \
    mode=mixed_latency \
    run_id="openvla_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/openvla_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="latency-sensitive-bench/openvla_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    dataset.source_hf=latency-sensitive-bench/memory-rollouts \
    dataset.source_subdir=deadly_corridor_openvla_rtx3090_profile_1000ep_7k2steps \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=False \
    trainer.max_train_steps=500 \
    trainer.save_interval=500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=500 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    framework.action_model.loss_type=multibinary_bce \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.mid_train.interval_steps=50 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
