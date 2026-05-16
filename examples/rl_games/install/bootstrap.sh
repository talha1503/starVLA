#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla_rl_games}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MODEL_TARGET="${MODEL_TARGET:-all}"
ENV_TARGET="${ENV_TARGET:-all}"
RUN_VALIDATE="${RUN_VALIDATE:-true}"

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/bootstrap.sh [options]

Options:
  --conda-env <name>        Conda env name (default: ${CONDA_ENV_NAME})
  --python-version <ver>    Python version for new env (default: ${PYTHON_VERSION})
  --model <name|all>        openvla|pi0|gr00t|all (default: ${MODEL_TARGET})
  --env <name|all>          flappy|demon_attack|deadly_corridor|all (default: ${ENV_TARGET})
  --skip-validate           Skip final validation step
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --model)
      MODEL_TARGET="$2"
      shift 2
      ;;
    --env)
      ENV_TARGET="$2"
      shift 2
      ;;
    --skip-validate)
      RUN_VALIDATE="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bootstrap] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] conda is required but not found in PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "[bootstrap] Using existing conda env: ${CONDA_ENV_NAME}"
else
  echo "[bootstrap] Creating conda env ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${CONDA_ENV_NAME}"

MODELS=(openvla pi0 gr00t)
ENVS=(flappy demon_attack deadly_corridor)

if [[ "${MODEL_TARGET}" == "all" ]]; then
  MODELS_TO_INSTALL=("${MODELS[@]}")
else
  MODELS_TO_INSTALL=("${MODEL_TARGET}")
fi

if [[ "${ENV_TARGET}" == "all" ]]; then
  ENVS_TO_INSTALL=("${ENVS[@]}")
else
  ENVS_TO_INSTALL=("${ENV_TARGET}")
fi

for model in "${MODELS_TO_INSTALL[@]}"; do
  case "${model}" in
    openvla|pi0|gr00t) ;;
    *)
      echo "[bootstrap] Invalid --model '${model}'. Expected openvla|pi0|gr00t|all." >&2
      exit 1
      ;;
  esac
done

for env_name in "${ENVS_TO_INSTALL[@]}"; do
  case "${env_name}" in
    flappy|demon_attack|deadly_corridor) ;;
    *)
      echo "[bootstrap] Invalid --env '${env_name}'. Expected flappy|demon_attack|deadly_corridor|all." >&2
      exit 1
      ;;
  esac
done

echo "[bootstrap] Installing common dependencies"
PYTHON_BIN=python "${SCRIPT_DIR}/common.sh"

for model in "${MODELS_TO_INSTALL[@]}"; do
  echo "[bootstrap] Installing model dependencies: ${model}"
  PYTHON_BIN=python "${SCRIPT_DIR}/model/${model}.sh"
done

for env_name in "${ENVS_TO_INSTALL[@]}"; do
  echo "[bootstrap] Installing environment dependencies: ${env_name}"
  PYTHON_BIN=python "${SCRIPT_DIR}/env/${env_name}.sh"
done

if [[ "${RUN_VALIDATE}" == "true" ]]; then
  echo "[bootstrap] Running validation"
  PYTHON_BIN=python "${SCRIPT_DIR}/validate/common.sh"
fi

echo "[bootstrap] Complete."
echo "[bootstrap] Activate later with: conda activate ${CONDA_ENV_NAME}"
echo "[bootstrap] Repo root: ${REPO_ROOT}"
