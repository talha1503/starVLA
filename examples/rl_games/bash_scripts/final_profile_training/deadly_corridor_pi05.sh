#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/talha}"
DATA_WORKSPACE_DIR="${DATA_WORKSPACE_DIR:-/mnt/local/talha}"
export CODE_ROOT DATA_WORKSPACE_DIR

if [[ ! -d "${CODE_ROOT}/starVLA" ]]; then
  echo "[error] missing StarVLA repo: ${CODE_ROOT}/starVLA" >&2
  exit 2
fi
if [[ ! -d "${CODE_ROOT}/latency-sensitive-bench/.git" ]]; then
  echo "[error] missing latency-sensitive-bench repo: ${CODE_ROOT}/latency-sensitive-bench" >&2
  exit 2
fi

cd "${CODE_ROOT}/latency-sensitive-bench"
git config --global url."https://github.com/".insteadOf git@github.com:
git config --global url."https://github.com/".insteadOf ssh://git@github.com/
git config -f .gitmodules submodule.flappy-bird-gymnasium.url https://github.com/mindorigin150/flappy-bird-gymnasium.git
git submodule sync -- third_party/starVLA third_party/flappy-bird-gymnasium third_party/sample-factory
git submodule update --init --recursive third_party/starVLA
git submodule update --init --recursive third_party/sample-factory

if [[ ! -f third_party/flappy-bird-gymnasium/pyproject.toml && ! -f third_party/flappy-bird-gymnasium/setup.py ]]; then
  git submodule deinit -f -- third_party/flappy-bird-gymnasium || true
  rm -rf .git/modules/third_party/flappy-bird-gymnasium
  rm -rf third_party/flappy-bird-gymnasium
fi
git submodule update --init --recursive third_party/flappy-bird-gymnasium

cd "${CODE_ROOT}/starVLA"

if command -v apt-get >/dev/null 2>&1 && [[ "${SKIP_APT_INSTALL:-0}" != "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y \
    pkg-config \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
else
  echo "[error] conda not found; install/source conda before running this script" >&2
  exit 2
fi

conda activate starvla_rl_games_pi05

WORKSPACE_DIR="${CODE_ROOT}" bash "${CODE_ROOT}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

mkdir -p "${DATA_WORKSPACE_DIR}"
export PYTHONPATH="${CODE_ROOT}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=pi05 \
    env=deadly_corridor \
    init=bridge \
    mode=mixed_latency \
    run_id="pi05_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    trainer.distributed_backend=none \
    workspace_dir="$DATA_WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/pi05_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="latency-sensitive-bench/pi05_bridge_deadly_corridor_rtx3090_profile_1000ep_7k2steps_final" \
    dataset.source_hf=latency-sensitive-bench/memory-rollouts \
    dataset.source_subdir=deadly_corridor_pi05_rtx3090_profile_1000ep_7k2steps \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=False \
    trainer.max_train_steps=500 \
    trainer.save_interval=500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=500 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
