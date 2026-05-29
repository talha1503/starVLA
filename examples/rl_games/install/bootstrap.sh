#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla_rl_games}"
PYTHON_VERSION="3.10"
MODEL_TARGET="${MODEL_TARGET:-all}"
ENV_TARGET="${ENV_TARGET:-all}"
RUN_VALIDATE="${RUN_VALIDATE:-true}"
SPLIT_ENVS="${SPLIT_ENVS:-false}"

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/bootstrap.sh [options]

Options:
  --conda-env <name>        Conda env name (default: ${CONDA_ENV_NAME})
  --python-version <ver>    Python version for new env (default: ${PYTHON_VERSION})
  --model <name|all>        openvla|pi0|gr00t|all (default: ${MODEL_TARGET})
  --env <name|all>          flappy|demon_attack|deadly_corridor|all (default: ${ENV_TARGET})
  --split-envs              Create/use one env per model: <conda-env>_<model>
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
    --split-envs)
      SPLIT_ENVS="true"
      shift
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

ensure_conda_env() {
  local env_name="$1"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[bootstrap] Using existing conda env: ${env_name}"
  else
    echo "[bootstrap] Creating conda env ${env_name} (python=${PYTHON_VERSION})"
    conda create -n "${env_name}" "python=${PYTHON_VERSION}" -y
  fi
}

validate_active_python_version() {
  local env_name="$1"
  local active_py_mm requested_py_mm
  active_py_mm="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_py_mm="$(echo "${PYTHON_VERSION}" | awk -F. '{if (NF>=2) print $1"."$2; else print $1}')"
  if [[ "${active_py_mm}" != "${requested_py_mm}" ]]; then
    echo "[bootstrap] Conda env '${env_name}' has Python ${active_py_mm}, expected ${requested_py_mm}." >&2
    echo "[bootstrap] Use a different --conda-env name, or remove and recreate this env." >&2
    exit 1
  fi
}

install_in_active_env() {
  local model="$1"
  shift
  local envs=("$@")

  echo "[bootstrap] Installing common dependencies"
  PYTHON_BIN=python "${SCRIPT_DIR}/common.sh"

  echo "[bootstrap] Installing model dependencies: ${model}"
  PYTHON_BIN=python "${SCRIPT_DIR}/model/${model}.sh"

  for env_name in "${envs[@]}"; do
    echo "[bootstrap] Installing environment dependencies: ${env_name}"
    PYTHON_BIN=python "${SCRIPT_DIR}/env/${env_name}.sh"
  done
}

install_eval_extra_in_active_env() {
  echo "[bootstrap] Installing latency-bench eval extra dependencies"
  PYTHON_BIN=python "${SCRIPT_DIR}/eval_extra.sh"
}

validate_in_active_env() {
  local run_validate="${1:-true}"
  if [[ "${run_validate}" == "true" ]]; then
    echo "[bootstrap] Running validation"
    PYTHON_BIN=python "${SCRIPT_DIR}/validate/common.sh"
  fi
}

if [[ "${SPLIT_ENVS}" == "true" ]]; then
  for model in "${MODELS_TO_INSTALL[@]}"; do
    TARGET_ENV_NAME="${CONDA_ENV_NAME}_${model}"
    ensure_conda_env "${TARGET_ENV_NAME}"
    conda activate "${TARGET_ENV_NAME}"
    validate_active_python_version "${TARGET_ENV_NAME}"

    MODEL_ENVS=("${ENVS_TO_INSTALL[@]}")

    echo "[bootstrap] Installing model=${model} in env=${TARGET_ENV_NAME} with env targets: ${MODEL_ENVS[*]}"
    install_in_active_env "${model}" "${MODEL_ENVS[@]}"
    install_eval_extra_in_active_env
    validate_in_active_env "${RUN_VALIDATE}"
  done

  echo "[bootstrap] Complete."
  echo "[bootstrap] Split env mode used. Activate with: conda activate ${CONDA_ENV_NAME}_<model>"
  echo "[bootstrap] Repo root: ${REPO_ROOT}"
  exit 0
fi

ensure_conda_env "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
validate_active_python_version "${CONDA_ENV_NAME}"
install_in_active_env "${MODELS_TO_INSTALL[0]}" "${ENVS_TO_INSTALL[@]}"

if [[ ${#MODELS_TO_INSTALL[@]} -gt 1 ]]; then
  for ((i=1; i<${#MODELS_TO_INSTALL[@]}; i++)); do
    echo "[bootstrap] Installing additional model dependencies in shared env: ${MODELS_TO_INSTALL[$i]}"
    PYTHON_BIN=python "${SCRIPT_DIR}/model/${MODELS_TO_INSTALL[$i]}.sh"
  done
fi

install_eval_extra_in_active_env
validate_in_active_env "${RUN_VALIDATE}"

echo "[bootstrap] Complete."
echo "[bootstrap] Activate later with: conda activate ${CONDA_ENV_NAME}"
echo "[bootstrap] Repo root: ${REPO_ROOT}"
