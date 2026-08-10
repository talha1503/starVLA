#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash commands/openvla/train_gymnasium_openvla.sh \
#     playground/Datasets/rl_games \
#     air_raid_fixed_l5 \
#     playground/Datasets/rl_games/_generated_mixtures/air_raid_fixed_l5.json \
#     trainer.max_train_steps=2
DATASET_LOCAL_DIR="$1"
DATA_MIX="$2"
CUSTOM_MIXTURES_PATH="$3"
shift 3

MANIFEST_PATH="${DATASET_LOCAL_DIR}/${DATA_MIX}/manifest.json"
TASK_CONTRACT="$(
  python -c '
import json
import sys

from examples.rl_games.scripts.launch_train import _hydra_value

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
contract = manifest["gymnasium_task"]
keys = (
    "task_name",
    "env_id",
    "make_kwargs",
    "registration_imports",
    "action_labels",
    "action_values",
    "noop_action_id",
    "base_prompt",
    "env_fps",
    "obs_fps",
    "frame_stack",
)
print(_hydra_value({key: contract[key] for key in keys}))
' "${MANIFEST_PATH}"
)"
ACTION_ENV_DIM="$(
  python -c '
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["active_action_dim"])
' "${MANIFEST_PATH}"
)"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
RUN_ID="${RUN_ID:-${DATA_MIX}_openvla_sft}"

python examples/rl_games/scripts/launch_train.py \
  --model openvla \
  --env gymnasium \
  --init scratch \
  --mode single \
  run_id="${RUN_ID}" \
  paths.dataset_local_dir="${DATASET_LOCAL_DIR}" \
  dataset.single_converted_name="${DATA_MIX}" \
  datasets.vla_data.data_mix="${DATA_MIX}" \
  datasets.vla_data.custom_mixtures_path="${CUSTOM_MIXTURES_PATH}" \
  rl_games.gymnasium.task_contract="${TASK_CONTRACT}" \
  framework.action_model.action_env_dim="${ACTION_ENV_DIM}" \
  datasets.vla_data.num_obs_frames=1 \
  datasets.vla_data.image_mode=single \
  framework.kv_memory.enabled=false \
  rl_games.env_eval.enabled=false \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
  trainer.save_interval="${SAVE_INTERVAL}" \
  trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  checkpoint.load=none \
  "$@"
