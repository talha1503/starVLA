#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_flappy_wan_oft_curriculum_pipeline.sh --mode <cumulative|exclusive> [options]

Runs the Flappy Bird context-window WanOFT curriculum pipeline:
  1. install/update the starvla_wanoft env
  2. download WanOFT checkpoints
  3. download raw context-image rollout data for all selected latencies
  4. convert raw data into StarVLA LeRobot mix-latency format
  5. train WanOFT with the selected curriculum mode
  6. upload the run directory to Hugging Face under {RUN_ID}

Options:
  --mode <cumulative|exclusive>  Required curriculum mode
  --conda-env <name>             Conda env name (default: starvla_wanoft)
  --python-version <ver>         Python version for bootstrap (default: 3.10)
  --context-window <N>           Context window size (default: 5)
  --max-episodes <N>             Dataset episode count used in path names (default: 200)
  --episodes-per-latency <N>     Episode budget per latency passed to converter (default: max-episodes)
  --max-train-steps <N>          Training steps (default: 2000)
  --latencies <csv>              Curriculum/data/eval latency ids (default: 0,1,2,3,4)
  --upload-repo <repo>           HF model repo for run upload (default: latency-sensitive-bench/wanoft_flappy_200ep)
  --upload-path <path>           Path inside the HF repo (default: <run_id>)
  --raw-dataset-repo <repo>      HF dataset repo for raw data (default: latency-sensitive-bench/flappy_200ep_context<context-window>)
  --run-id <id>                  Override run id
  --skip-env-setup               Do not run examples/rl_games/install/bootstrap.sh
  --skip-checkpoints             Do not download Wan base/init checkpoints
  --skip-data-download           Do not download raw rollout data
  --skip-convert                 Do not convert raw data
  --skip-train                   Do not run training
  --skip-upload                  Do not upload the run directory
  -h, --help                     Show this help
EOF
}

MODE=""
CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla_wanoft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"
MAX_EPISODES="${MAX_EPISODES:-200}"
EPISODES_PER_LATENCY="${EPISODES_PER_LATENCY:-${MAX_EPISODES}}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-50}"
MID_TRAIN_INTERVAL="${MID_TRAIN_INTERVAL:-${EVAL_INTERVAL}}"
MID_TRAIN_EPISODES="${MID_TRAIN_EPISODES:-5}"
POST_TRAIN_EPISODES="${POST_TRAIN_EPISODES:-20}"
PHASE_STEPS="${PHASE_STEPS:-null}"
EVAL_AT_PHASE_END="${EVAL_AT_PHASE_END:-false}"
SAVE_AT_PHASE_END="${SAVE_AT_PHASE_END:-false}"
LATENCY_FILTER_CSV="${LATENCY_FILTER_CSV:-0,1,2,3,4}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-dongqianyu99-zhejiang-university}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-starVLA_rl_games}"
RAW_DATASET_REPO="${RAW_DATASET_REPO:-}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-data/raw}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
RUN_ID="${RUN_ID:-}"
UPLOAD_REPO="${UPLOAD_REPO:-latency-sensitive-bench/wanoft_flappy_200ep}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
INIT_CHECKPOINT_REPO="${INIT_CHECKPOINT_REPO:-StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-playground/Pretrained_models/WM4A-Wan2d2-OFT-LIBERO-4in1}"
SKIP_ENV_SETUP="false"
SKIP_CHECKPOINTS="false"
SKIP_DATA_DOWNLOAD="false"
SKIP_CONVERT="false"
SKIP_TRAIN="false"
SKIP_UPLOAD="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --context-window)
      CONTEXT_WINDOW="$2"
      shift 2
      ;;
    --max-episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --episodes-per-latency)
      EPISODES_PER_LATENCY="$2"
      shift 2
      ;;
    --max-train-steps)
      MAX_TRAIN_STEPS="$2"
      shift 2
      ;;
    --latencies)
      LATENCY_FILTER_CSV="$2"
      shift 2
      ;;
    --upload-repo)
      UPLOAD_REPO="$2"
      shift 2
      ;;
    --upload-path)
      UPLOAD_PATH_IN_REPO="$2"
      shift 2
      ;;
    --raw-dataset-repo)
      RAW_DATASET_REPO="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --skip-env-setup)
      SKIP_ENV_SETUP="true"
      shift
      ;;
    --skip-checkpoints)
      SKIP_CHECKPOINTS="true"
      shift
      ;;
    --skip-data-download)
      SKIP_DATA_DOWNLOAD="true"
      shift
      ;;
    --skip-convert)
      SKIP_CONVERT="true"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN="true"
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[flappy-wanoft-curriculum] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  cumulative)
    TRAIN_MODE="curriculum_cumulative"
    ;;
  exclusive)
    TRAIN_MODE="curriculum_exclusive"
    ;;
  "")
    echo "[flappy-wanoft-curriculum] --mode is required." >&2
    usage >&2
    exit 2
    ;;
  *)
    echo "[flappy-wanoft-curriculum] --mode must be cumulative or exclusive, got: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

if ! [[ "${CONTEXT_WINDOW}" =~ ^[0-9]+$ ]] || [[ "${CONTEXT_WINDOW}" -lt 2 ]]; then
  echo "[flappy-wanoft-curriculum] --context-window must be an integer >= 2, got: ${CONTEXT_WINDOW}" >&2
  exit 2
fi
if ! [[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]] || [[ "${MAX_EPISODES}" -lt 1 ]]; then
  echo "[flappy-wanoft-curriculum] --max-episodes must be a positive integer, got: ${MAX_EPISODES}" >&2
  exit 2
fi
if ! [[ "${EPISODES_PER_LATENCY}" =~ ^[0-9]+$ ]] || [[ "${EPISODES_PER_LATENCY}" -lt 1 ]]; then
  echo "[flappy-wanoft-curriculum] --episodes-per-latency must be a positive integer, got: ${EPISODES_PER_LATENCY}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LATENCY_FILTER_LIST="[${LATENCY_FILTER_CSV}]"
OBSERVATION_INDICES="$(python - "${CONTEXT_WINDOW}" <<'PY'
import sys

context_window = int(sys.argv[1])
print("[" + ",".join(str(index) for index in range(-(context_window - 1), 1)) + "]")
PY
)"
RAW_DATASET_REPO="${RAW_DATASET_REPO:-latency-sensitive-bench/flappy_200ep_context${CONTEXT_WINDOW}}"
RAW_TEMPLATE_SUBDIR="flappy_fix_latency_0_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"
CONVERTED_DATA_ROOT="data/flappy_mixed_latency_${MAX_EPISODES}ep_per_lat_context${CONTEXT_WINDOW}"
CONVERTED_DATA_DIR="${CONVERTED_DATA_ROOT}/flappy_mixed_latency_train__bridge"
PROMPT_MAP_PATH="${CONVERTED_DATA_DIR}/latency_prompt_map.json"
RUN_ID="${RUN_ID:-wan_oft_flappy_mix_latency_context${CONTEXT_WINDOW}_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_curriculum_${MODE}}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-${RUN_ID}}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "[flappy-wanoft-curriculum] Hugging Face CLI command 'hf' is not available in PATH." >&2
    echo "[flappy-wanoft-curriculum] Re-run without --skip-env-setup, or install huggingface-hub in ${CONDA_ENV_NAME}." >&2
    exit 1
  fi
}

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[flappy-wanoft-curriculum] conda is required but was not found in PATH." >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
}

if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
  echo "[flappy-wanoft-curriculum] Installing/updating env: ${CONDA_ENV_NAME}"
  bash examples/rl_games/install/bootstrap.sh \
    --conda-env "${CONDA_ENV_NAME}" \
    --python-version "${PYTHON_VERSION}" \
    --model wan_oft \
    --env flappy
fi

activate_conda_env
ensure_hf_cli

if [[ "${SKIP_CHECKPOINTS}" != "true" ]]; then
  echo "[flappy-wanoft-curriculum] Downloading Wan base model checkpoint"
  hf download "${BASE_MODEL_REPO}" \
    --local-dir "${BASE_MODEL_DIR}"

  echo "[flappy-wanoft-curriculum] Downloading WanOFT initialization checkpoint"
  hf download "${INIT_CHECKPOINT_REPO}" \
    --local-dir "${INIT_CHECKPOINT_DIR}"
fi

if [[ "${SKIP_DATA_DOWNLOAD}" != "true" ]]; then
  echo "[flappy-wanoft-curriculum] Downloading raw data from ${RAW_DATASET_REPO}"
  hf download "${RAW_DATASET_REPO}" \
    --repo-type dataset \
    --include "flappy_fix_latency_*_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}/**" \
    --local-dir "${RAW_DATA_ROOT}"
fi

if [[ "${SKIP_CONVERT}" != "true" ]]; then
  echo "[flappy-wanoft-curriculum] Converting raw data into ${CONVERTED_DATA_DIR}"
  python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py \
    --dataset-name "${RAW_DATA_ROOT}" \
    --dataset-source-subdir "${RAW_TEMPLATE_SUBDIR}" \
    --source-metadata "${RAW_DATA_ROOT}/${RAW_TEMPLATE_SUBDIR}/metadata.json" \
    --source-latency-column latency_raw_frames \
    --target-latency-unit observation_steps \
    --output-dir "${CONVERTED_DATA_DIR}" \
    --action-carrier bridge \
    --latency-filter "${LATENCY_FILTER_CSV}" \
    --episodes-per-latency "${EPISODES_PER_LATENCY}" \
    --context-images-column context_images \
    --image-sequence-length "${CONTEXT_WINDOW}" \
    --force
fi

if [[ "${SKIP_TRAIN}" != "true" ]]; then
  echo "[flappy-wanoft-curriculum] Training mode=${MODE} run_id=${RUN_ID}"
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
  export WANDB_PROJECT="${WANDB_PROJECT_VALUE}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

  python examples/rl_games/scripts/launch_train.py \
    model=wan_oft \
    env=flappy \
    init=wan_oft_libero \
    mode="${TRAIN_MODE}" \
    run_id="${RUN_ID}" \
    paths.run_root_dir="${RUN_ROOT_DIR}" \
    paths.base_model_dir="${BASE_MODEL_DIR}" \
    paths.dataset_local_dir="${CONVERTED_DATA_ROOT}" \
    initialization.checkpoint_local_dir="${INIT_CHECKPOINT_DIR}" \
    dataset.source_hf="${RAW_DATA_ROOT}" \
    dataset.source_subdir="${RAW_TEMPLATE_SUBDIR}" \
    dataset.converted_name=flappy_mixed_latency_train__bridge \
    "dataset.latency_filter=${LATENCY_FILTER_LIST}" \
    dataset.episodes_per_latency="${EPISODES_PER_LATENCY}" \
    trainer.distributed_backend=none \
    launch.use_accelerate=false \
    launch.num_processes=1 \
    trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
    datasets.vla_data.data_mix=flappy_mixed_latency_train__bridge \
    datasets.vla_data.eval_data_mix=flappy_mixed_latency_train__bridge__val \
    "datasets.vla_data.obs_image_size=[224,224]" \
    datasets.vla_data.image_sequence_length="${CONTEXT_WINDOW}" \
    "datasets.vla_data.observation_indices=${OBSERVATION_INDICES}" \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.eval_sequential_step_sampling=false \
    datasets.vla_data.action_balance.enabled=false \
    datasets.vla_data.latency_curriculum.enabled=true \
    datasets.vla_data.latency_curriculum.strategy="${MODE}" \
    "datasets.vla_data.latency_curriculum.latencies=${LATENCY_FILTER_LIST}" \
    "datasets.vla_data.latency_curriculum.phase_steps=${PHASE_STEPS}" \
    datasets.vla_data.latency_curriculum.eval_at_phase_end="${EVAL_AT_PHASE_END}" \
    datasets.vla_data.latency_curriculum.save_at_phase_end="${SAVE_AT_PHASE_END}" \
    framework.world_model.num_frames="${CONTEXT_WINDOW}" \
    framework.action_model.loss_type=current_discrete_ce \
    "+trainer.learning_rate.action_query_proj=1.0e-4" \
    trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
    trainer.num_warmup_steps=0 \
    trainer.lr_scheduler_type=cosine_with_min_lr \
    trainer.scheduler_specific_kwargs.min_lr=1.0e-6 \
    trainer.eval_interval="${EVAL_INTERVAL}" \
    trainer.eval_num_batches="${EVAL_NUM_BATCHES}" \
    trainer.eval_action_classification=false \
    trainer.save_interval=0 \
    rl_games.env_eval.enabled=true \
    rl_games.env_eval.image_size=224 \
    rl_games.env_eval.vectorized.enabled=false \
    rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}" \
    rl_games.env_eval.mid_train.enabled=true \
    rl_games.env_eval.mid_train.interval_steps="${MID_TRAIN_INTERVAL}" \
    "rl_games.env_eval.mid_train.latencies=${LATENCY_FILTER_LIST}" \
    rl_games.env_eval.mid_train.num_episodes="${MID_TRAIN_EPISODES}" \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.post_train.enabled=true \
    "rl_games.env_eval.post_train.latencies=${LATENCY_FILTER_LIST}" \
    rl_games.env_eval.post_train.num_episodes="${POST_TRAIN_EPISODES}" \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    checkpoint.save_pt_file=true \
    checkpoint.save_training_state=false \
    checkpoint.save_best_model=false \
    checkpoint.save_final_model=true \
    checkpoint.local.keep_last_n=1 \
    checkpoint.sync.enabled=false \
    checkpoint.load=none
fi

if [[ "${SKIP_UPLOAD}" != "true" ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[flappy-wanoft-curriculum] Training output directory does not exist: ${RUN_DIR}" >&2
    exit 1
  fi
  echo "[flappy-wanoft-curriculum] Uploading ${RUN_DIR} to ${UPLOAD_REPO}:${UPLOAD_PATH_IN_REPO}"
  hf upload "${UPLOAD_REPO}" "${RUN_DIR}" "${UPLOAD_PATH_IN_REPO}" \
    --exclude "wandb/**" \
    --repo-type model
fi

echo "[flappy-wanoft-curriculum] Complete: mode=${MODE} run_id=${RUN_ID}"
