#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/talha}"
DATA_WORKSPACE_DIR="${DATA_WORKSPACE_DIR:-/mnt/local/talha}"
export CODE_ROOT DATA_WORKSPACE_DIR

if [[ ! -d "${CODE_ROOT}/starVLA" ]]; then
  echo "[error] missing StarVLA repo: ${CODE_ROOT}/starVLA" >&2
  exit 2
fi

if [[ -d "${CODE_ROOT}/latency-sensitive-bench/.git" ]]; then
  git -C "${CODE_ROOT}/latency-sensitive-bench" submodule update --init --recursive
else
  WORKSPACE_DIR="${CODE_ROOT}" bash "${CODE_ROOT}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"
fi

cd "${CODE_ROOT}/starVLA"

conda activate starvla_rl_games_gr00t

WORKSPACE_DIR="${CODE_ROOT}" bash "${CODE_ROOT}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

mkdir -p "${DATA_WORKSPACE_DIR}"
export PYTHONPATH="${CODE_ROOT}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=deadly_corridor \
    init=bridge \
    mode=mixed_latency \
    run_id="gr00t_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final_action_1e-4_backbone_1e-5" \
    trainer.distributed_backend=none \
    workspace_dir="$DATA_WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/gr00t_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final_action_1e-4_backbone_1e-5" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="latency-sensitive-bench/gr00t_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final_action_1e-4_backbone_1e-5" \
    dataset.source_hf=latency-sensitive-bench/memory-rollouts \
    dataset.source_subdir=deadly_corridor_gr00t_rtx3090_profile_1000ep_7k2steps \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=False \
    trainer.max_train_steps=500 \
    trainer.save_interval=500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    trainer.learning_rate.base=1e-05 \
    trainer.learning_rate.qwen_vl_interface=1e-05 \
    trainer.learning_rate.action_model=1e-04 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.mid_train.interval_steps=50 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.post_train.latencies=[0,2,4,6,8] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
