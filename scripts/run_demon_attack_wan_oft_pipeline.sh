#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_demon_attack_wan_oft_pipeline.sh --latency <N> [options]

Runs the Demon Attack context-window WanOFT pipeline:
  1. install/update the starvla_wanoft env
  2. download WanOFT checkpoints
  3. download raw context-image rollout data for the selected latency
  4. convert raw data into StarVLA LeRobot format
  5. train WanOFT
  6. upload the checkpoint directory to Hugging Face

Options:
  --latency <N>             Required fixed latency id, e.g. 0, 2, 6
  --conda-env <name>        Conda env name (default: starvla_wanoft)
  --python-version <ver>    Python version for bootstrap (default: 3.10)
  --context-window <N>      Context window size (default: 5)
  --max-episodes <N>        Dataset episode count used in path names (default: 200)
  --max-train-steps <N>     Training steps forwarded to train command (default: 2000)
  --upload-repo <repo>      HF model repo for checkpoint upload (default: latency-sensitive-bench/demon_attack_200ep)
  --upload-path <path>      Path inside the HF repo (default: ./<run_id>)
  --raw-dataset-repo <repo> HF dataset repo for raw data (default: latency-sensitive-bench/demon_attack_200ep_context<context-window>)
  --skip-env-setup          Do not run examples/rl_games/install/bootstrap.sh
  --skip-checkpoints        Do not download Wan base/init checkpoints
  --skip-data-download      Do not download raw rollout data
  --skip-convert            Do not convert raw data
  --skip-train              Do not run training
  --skip-upload             Do not upload checkpoint directory
  -h, --help                Show this help
EOF
}

LATENCY=""
CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla_wanoft}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"
MAX_EPISODES="${MAX_EPISODES:-200}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-dongqianyu99-zhejiang-university}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-starVLA_rl_games}"
RAW_DATASET_REPO="${RAW_DATASET_REPO:-}"
UPLOAD_REPO="${UPLOAD_REPO:-latency-sensitive-bench/demon_attack_200ep}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-data/raw}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
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
    --latency)
      LATENCY="$2"
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
    --max-train-steps)
      MAX_TRAIN_STEPS="$2"
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
      echo "[demon-wanoft] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${LATENCY}" ]]; then
  echo "[demon-wanoft] --latency is required." >&2
  usage >&2
  exit 2
fi
if ! [[ "${LATENCY}" =~ ^[0-9]+$ ]]; then
  echo "[demon-wanoft] --latency must be a non-negative integer, got: ${LATENCY}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAW_DATASET_REPO="${RAW_DATASET_REPO:-latency-sensitive-bench/demon_attack_200ep_context${CONTEXT_WINDOW}}"
RAW_SUBDIR="demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"
RAW_DATA_DIR="${RAW_DATA_ROOT}/${RAW_SUBDIR}"
CONVERTED_DATA_ROOT="data/demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"
CONVERTED_DATA_DIR="${CONVERTED_DATA_ROOT}/demon_attack_train__bridge"
RUN_ID="wan_oft_demon_attack_fix_latency_${LATENCY}_context${CONTEXT_WINDOW}_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentce"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-./${RUN_ID}}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "[demon-wanoft] Hugging Face CLI command 'hf' is not available in PATH." >&2
    echo "[demon-wanoft] Re-run without --skip-env-setup, or install huggingface-hub in ${CONDA_ENV_NAME}." >&2
    exit 1
  fi
}

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[demon-wanoft] conda is required but was not found in PATH." >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
}

if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
  echo "[demon-wanoft] Installing/updating env: ${CONDA_ENV_NAME}"
  bash examples/rl_games/install/bootstrap.sh \
    --conda-env "${CONDA_ENV_NAME}" \
    --python-version "${PYTHON_VERSION}" \
    --model wan_oft \
    --env demon_attack
fi

activate_conda_env
ensure_hf_cli

if [[ "${SKIP_CHECKPOINTS}" != "true" ]]; then
  echo "[demon-wanoft] Downloading Wan base model checkpoint"
  hf download "${BASE_MODEL_REPO}" \
    --local-dir "${BASE_MODEL_DIR}"

  echo "[demon-wanoft] Downloading WanOFT initialization checkpoint"
  hf download "${INIT_CHECKPOINT_REPO}" \
    --local-dir "${INIT_CHECKPOINT_DIR}"
fi

if [[ "${SKIP_DATA_DOWNLOAD}" != "true" ]]; then
  echo "[demon-wanoft] Downloading raw data ${RAW_SUBDIR} from ${RAW_DATASET_REPO}"
  hf download "${RAW_DATASET_REPO}" \
    --repo-type dataset \
    --include "${RAW_SUBDIR}/**" \
    --local-dir "${RAW_DATA_ROOT}"
fi

if [[ "${SKIP_CONVERT}" != "true" ]]; then
  echo "[demon-wanoft] Converting raw data into ${CONVERTED_DATA_DIR}"
  python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_demon_attack_to_starvla_lerobot.py \
    --dataset-name "${RAW_DATA_DIR}" \
    --output-dir "${CONVERTED_DATA_DIR}" \
    --action-carrier bridge \
    --context-images-column context_images \
    --image-sequence-length "${CONTEXT_WINDOW}" \
    --force
fi

if [[ "${SKIP_TRAIN}" != "true" ]]; then
  echo "[demon-wanoft] Training run_id=${RUN_ID}"
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
  export WANDB_PROJECT="${WANDB_PROJECT_VALUE}"
  CONTEXT_WINDOW="${CONTEXT_WINDOW}" \
  MAX_EPISODES="${MAX_EPISODES}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  DATASET_LOCAL_DIR="${CONVERTED_DATA_ROOT}" \
  RUN_ID="${RUN_ID}" \
    bash commands/train_demon_attack_wan_oft.sh "${LATENCY}"
fi

if [[ "${SKIP_UPLOAD}" != "true" ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[demon-wanoft] Training output directory does not exist: ${RUN_DIR}" >&2
    exit 1
  fi
  echo "[demon-wanoft] Uploading ${RUN_DIR} to ${UPLOAD_REPO}:${UPLOAD_PATH_IN_REPO}"
  hf upload "${UPLOAD_REPO}" "${RUN_DIR}" "${UPLOAD_PATH_IN_REPO}" \
    --exclude "wandb/**" \
    --repo-type model
fi

echo "[demon-wanoft] Complete: latency=${LATENCY} run_id=${RUN_ID}"
