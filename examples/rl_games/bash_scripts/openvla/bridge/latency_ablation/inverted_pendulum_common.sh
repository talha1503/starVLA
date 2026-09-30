#!/usr/bin/env bash
set -euo pipefail

IP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP_REPO_ROOT="$(cd "${IP_SCRIPT_DIR}/../../../../../.." && pwd)"
IP_WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${IP_REPO_ROOT}/.." && pwd)}"
IP_BENCHMARK_ROOT="${BENCHMARK_ROOT:-${IP_WORKSPACE_DIR}/latency-sensitive-bench}"

export WORKSPACE_DIR="${IP_WORKSPACE_DIR}"
export PYTHONPATH="${IP_BENCHMARK_ROOT}:${IP_REPO_ROOT}:${PYTHONPATH:-}"

IP_DATASET_REPO="${IP_DATASET_REPO:-latency-sensitive-bench/inverted_pendulum_200ep}"
IP_SOURCE_ROOT="${IP_SOURCE_ROOT:-${IP_REPO_ROOT}/playground/Datasets/rl_games/_hf_sources/inverted_pendulum_latency024_7k2steps}"
IP_DATASET_LOCAL_DIR="${IP_DATASET_LOCAL_DIR:-${IP_REPO_ROOT}/playground/Datasets/rl_games}"
IP_CONVERTED_NAME="${IP_CONVERTED_NAME:-inverted_pendulum_mixed_latency_024_7k2steps_bridge}"
IP_SOURCE_TEMPLATE="${IP_SOURCE_TEMPLATE:-inverted_pendulum_fixed_latency_0_7k2steps}"
IP_RUN_ROOT_DIR="${IP_RUN_ROOT_DIR:-${IP_WORKSPACE_DIR}/outputs/openvla_runs}"
IP_HF_NAMESPACE="${IP_HF_NAMESPACE:-talha15032}"
IP_WANDB_ENTITY="${IP_WANDB_ENTITY:-talha1503}"
IP_MAX_TRAIN_STEPS="${IP_MAX_TRAIN_STEPS:-5000}"
IP_SAVE_INTERVAL="${IP_SAVE_INTERVAL:-5000}"
IP_EVAL_INTERVAL="${IP_EVAL_INTERVAL:-5000}"
IP_BATCH_SIZE="${IP_BATCH_SIZE:-8}"
IP_GRAD_ACCUM="${IP_GRAD_ACCUM:-2}"
IP_EVAL_NUM_BATCHES="${IP_EVAL_NUM_BATCHES:-200}"

IP_BASE_PROMPT="Keep the pole upright by applying horizontal cart forces. Choose exactly one action from: push_left_hard, push_left_soft, zero_force, push_right_soft, push_right_hard."

ip_activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "/opt/conda/etc/profile.d/conda.sh"
  fi
  conda activate starvla_rl_games_openvla
}

ip_prepare_source_dataset() {
  mkdir -p "${IP_SOURCE_ROOT}"
  IP_DATASET_REPO="${IP_DATASET_REPO}" IP_SOURCE_ROOT="${IP_SOURCE_ROOT}" python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["IP_DATASET_REPO"]
source_root = Path(os.environ["IP_SOURCE_ROOT"]).expanduser().resolve()

configs = {
    0: "inverted_pendulum_fixed_latency_0_200ep_7k2steps",
    2: "inverted_pendulum_fixed_latency_2_1000ep_7k2steps",
    4: "inverted_pendulum_fixed_latency_4_1000ep_7k2steps",
}

snapshot = Path(
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"{name}/*.parquet" for name in configs.values()],
    )
)

for latency, remote_name in configs.items():
    src = snapshot / remote_name
    dst = source_root / f"inverted_pendulum_fixed_latency_{latency}_7k2steps"
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        raise FileNotFoundError(f"Missing downloaded HF config directory: {src}")
    for parquet in src.glob("*.parquet"):
        target = dst / parquet.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(parquet)
        except OSError:
            shutil.copy2(parquet, target)

print(source_root)
PY
}

ip_install_stack() {
  cd "${IP_REPO_ROOT}"
  if [[ -f examples/rl_games/bash_scripts/install/pre_launch.sh ]]; then
    bash examples/rl_games/bash_scripts/install/pre_launch.sh
  fi
  bash examples/rl_games/install/install_stack.sh openvla gymnasium
  ip_activate_conda
  if [[ -f examples/rl_games/install/flash_attn.sh ]]; then
    bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh
  fi
  if [[ -f examples/rl_games/bash_scripts/install/latency_deps.sh ]]; then
    bash examples/rl_games/bash_scripts/install/latency_deps.sh
  fi
}

ip_run_openvla_training() {
  local run_id="$1"
  local prompt_mode="$2"
  local hf_repo_id="${IP_HF_NAMESPACE}/${run_id}"

  ip_install_stack
  ip_prepare_source_dataset

  cd "${IP_REPO_ROOT}"
  python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=gymnasium \
    init=bridge \
    mode=mixed_latency \
    run_id="${run_id}" \
    trainer.distributed_backend=none \
    workspace_dir="${IP_WORKSPACE_DIR}" \
    paths.run_root_dir="${IP_RUN_ROOT_DIR}" \
    run_root_dir="${IP_RUN_ROOT_DIR}" \
    wandb_entity="${IP_WANDB_ENTITY}" \
    checkpoint.hf_repo_id="${hf_repo_id}" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="${hf_repo_id}" \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    dataset.source_hf="${IP_SOURCE_ROOT}" \
    dataset.source_subdir="${IP_SOURCE_TEMPLATE}" \
    dataset.converted_name="${IP_CONVERTED_NAME}" \
    dataset.latency_filter=[0,2,4] \
    dataset.target_latency_unit=raw_frames \
    dataset.force_download=false \
    dataset.setup_force=false \
    paths.dataset_local_dir="${IP_DATASET_LOCAL_DIR}" \
    ++rl_games.gymnasium.task_contract.task_name=inverted_pendulum \
    ++rl_games.gymnasium.task_contract.env_id=LatencyBench/InvertedPendulumDiscrete-v0 \
    ++rl_games.gymnasium.task_contract.registration_imports=[latency_bench.envs.gymnasium_mujoco] \
    ++rl_games.gymnasium.task_contract.env_fps=25.0 \
    ++rl_games.gymnasium.task_contract.obs_fps=25.0 \
    ++rl_games.gymnasium.task_contract.frame_stack=1 \
    "++rl_games.gymnasium.task_contract.base_prompt='${IP_BASE_PROMPT}'" \
    ++rl_games.gymnasium.task_contract.action_labels=[push_left_hard,push_left_soft,zero_force,push_right_soft,push_right_hard] \
    ++rl_games.gymnasium.task_contract.action_values=[0,1,2,3,4] \
    ++rl_games.gymnasium.task_contract.noop_action_id=2 \
    ++rl_games.gymnasium.task_contract.make_kwargs.base_env_id=InvertedPendulum-v4 \
    ++rl_games.gymnasium.task_contract.make_kwargs.render_mode=rgb_array \
    ++rl_games.gymnasium.task_contract.make_kwargs.force_values=[-3.0,-1.5,0.0,1.5,3.0] \
    ++rl_games.gymnasium.task_contract.make_kwargs.base_make_kwargs={} \
    ++rl_games.gymnasium.task_contract.state_space.labels=[cart_position,pole_angle,cart_velocity,pole_angular_velocity] \
    ++datasets.vla_data.gymnasium_task_contract.task_name=inverted_pendulum \
    ++datasets.vla_data.gymnasium_task_contract.env_id=LatencyBench/InvertedPendulumDiscrete-v0 \
    ++datasets.vla_data.gymnasium_task_contract.registration_imports=[latency_bench.envs.gymnasium_mujoco] \
    ++datasets.vla_data.gymnasium_task_contract.env_fps=25.0 \
    ++datasets.vla_data.gymnasium_task_contract.obs_fps=25.0 \
    ++datasets.vla_data.gymnasium_task_contract.frame_stack=1 \
    "++datasets.vla_data.gymnasium_task_contract.base_prompt='${IP_BASE_PROMPT}'" \
    ++datasets.vla_data.gymnasium_task_contract.action_labels=[push_left_hard,push_left_soft,zero_force,push_right_soft,push_right_hard] \
    ++datasets.vla_data.gymnasium_task_contract.action_values=[0,1,2,3,4] \
    ++datasets.vla_data.gymnasium_task_contract.noop_action_id=2 \
    ++datasets.vla_data.gymnasium_task_contract.make_kwargs.base_env_id=InvertedPendulum-v4 \
    ++datasets.vla_data.gymnasium_task_contract.make_kwargs.render_mode=rgb_array \
    ++datasets.vla_data.gymnasium_task_contract.make_kwargs.force_values=[-3.0,-1.5,0.0,1.5,3.0] \
    ++datasets.vla_data.gymnasium_task_contract.make_kwargs.base_make_kwargs={} \
    ++datasets.vla_data.gymnasium_task_contract.state_space.labels=[cart_position,pole_angle,cart_velocity,pole_angular_velocity] \
    datasets.vla_data.active_action_dim=5 \
    datasets.vla_data.include_state=true \
    datasets.vla_data.action_type=discrete \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    datasets.vla_data.prompt_mode="${prompt_mode}" \
    datasets.vla_data.per_device_batch_size="${IP_BATCH_SIZE}" \
    framework.action_model.action_dim=5 \
    framework.action_model.action_env_dim=5 \
    framework.action_model.state_dim=4 \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_horizon=1 \
    framework.action_model.future_action_window_size=0 \
    framework.action_model.past_action_window_size=0 \
    trainer.max_train_steps="${IP_MAX_TRAIN_STEPS}" \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval="${IP_EVAL_INTERVAL}" \
    trainer.save_interval="${IP_SAVE_INTERVAL}" \
    trainer.eval_num_batches="${IP_EVAL_NUM_BATCHES}" \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps="${IP_GRAD_ACCUM}" \
    trainer.eval_action_classification=true \
    rl_games.env_eval.enabled=false \
    rl_games.env_eval.prompt_mode="${prompt_mode}" \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=false
}
