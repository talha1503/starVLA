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
  --model <name|all>        openvla|pi0|pi05|gr00t|wan_oft|all (default: ${MODEL_TARGET})
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

# Keep uv's cache on the same filesystem as the conda envs so packages are
# hardlinked into each env (download/disk dedup) instead of silently copied.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CONDA_BASE}/.uv_cache}"
# Some cloud images (e.g. vast.ai) export UV_NO_CACHE=1 (disables uv's wheel cache, so
# every per-model env re-downloads the identical multi-GB torch/CUDA stack) and
# UV_LINK_MODE=copy (defeats the cache->env hardlink dedup). Clear both so the first
# build warms the shared cache and the rest hardlink from it.
unset UV_NO_CACHE UV_LINK_MODE

# Route torch + pypi through the fastest mirror. When launched via the repo install.sh
# this is already chosen and exported; for a standalone bootstrap run, speed-test here
# (the helper lives in the parent repo: starVLA is a submodule under it).
if [[ -z "${STARVLA_TORCH_INDEX_BASE:-}" ]]; then
  MIRRORS_HELPER="${REPO_ROOT}/../scripts/bash_scripts/_mirrors.sh"
  if [[ -f "${MIRRORS_HELPER}" ]]; then
    # shellcheck source=/dev/null
    source "${MIRRORS_HELPER}"
    export_mirror_env "${STARVLA_PROBE_PROFILE:-cu128}"
  fi
fi

MODELS=(openvla pi0 pi05 gr00t wan_oft)
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
    openvla|pi0|pi05|gr00t|wan_oft) ;;
    *)
      echo "[bootstrap] Invalid --model '${model}'. Expected openvla|pi0|pi05|gr00t|wan_oft|all." >&2
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

target_validator_name() {
  local model="$1"
  local env_name="$2"
  case "${model}:${env_name}" in
    pi0:demon_attack) echo "pi0_demon.sh" ;;
    gr00t:deadly_corridor) echo "gr00t_deadly.sh" ;;
    *) echo "${model}_${env_name}.sh" ;;
  esac
}

run_common_validation() {
  PYTHON_BIN=python "${SCRIPT_DIR}/validate/common.sh"
}

run_target_validators_for_model() {
  local model="$1"
  shift
  local envs=("$@")
  local env_name validator_name target_validator

  for env_name in "${envs[@]}"; do
    validator_name="$(target_validator_name "${model}" "${env_name}")"
    target_validator="${SCRIPT_DIR}/validate/${validator_name}"
    if [[ -x "${target_validator}" ]]; then
      PYTHON_BIN=python "${target_validator}"
    elif [[ -f "${target_validator}" ]]; then
      PYTHON_BIN=python bash "${target_validator}"
    else
      echo "[bootstrap] No target-specific validator found for ${model}/${env_name}; common validation passed."
    fi
  done
}

install_in_active_env() {
  local model="$1"
  local run_validate="${2:-true}"
  shift 2
  local envs=("$@")

  echo "[bootstrap] Installing common dependencies"
  PYTHON_BIN=python "${SCRIPT_DIR}/common.sh"

  echo "[bootstrap] Installing model dependencies: ${model}"
  PYTHON_BIN=python "${SCRIPT_DIR}/model/${model}.sh"

  for env_name in "${envs[@]}"; do
    echo "[bootstrap] Installing environment dependencies: ${env_name}"
    PYTHON_BIN=python "${SCRIPT_DIR}/env/${env_name}.sh"
  done

  if [[ "${run_validate}" == "true" ]]; then
    echo "[bootstrap] Running validation"
    run_common_validation
    run_target_validators_for_model "${model}" "${envs[@]}"
  fi
}

# Build one model env from scratch in the currently active env: common stack +
# model deps + per-task env deps (validation runs separately). uv dedups the
# heavy stack across envs via its shared cache + hardlinks, so there is no need
# to clone a base env (conda --clone cannot reproduce a pip-dominated env).
build_split_env() {
  local model="$1"
  local target="$2"
  conda activate "${target}"
  validate_active_python_version "${target}"
  install_in_active_env "${model}" "false" "${ENVS_TO_INSTALL[@]}"
}

if [[ "${SPLIT_ENVS}" == "true" ]]; then
  # Create the (empty) per-model conda envs serially (conda metadata is not
  # concurrency-safe), collecting their names.
  TARGET_ENVS=()
  for model in "${MODELS_TO_INSTALL[@]}"; do
    target="${CONDA_ENV_NAME}_${model}"
    TARGET_ENVS+=("${target}")
    ensure_conda_env "${target}"
  done

  # Build the first env serially: fail-fast smoke test + warms the uv cache
  # (torch and the common stack) so the parallel batch only hardlinks.
  echo "[bootstrap] Building ${TARGET_ENVS[0]} (model=${MODELS_TO_INSTALL[0]}) — warms uv cache"
  build_split_env "${MODELS_TO_INSTALL[0]}" "${TARGET_ENVS[0]}"
  conda deactivate

  # Build the remaining envs in parallel against the warm uv cache. uv's cache is
  # concurrency-safe (per-package locks), so there is no duplicate download.
  pids=()
  for ((i=1; i<${#MODELS_TO_INSTALL[@]}; i++)); do
    model="${MODELS_TO_INSTALL[$i]}"
    target="${TARGET_ENVS[$i]}"
    echo "[bootstrap] Building ${target} (model=${model}) in background — log: /tmp/bootstrap_${model}.log"
    (
      build_split_env "${model}" "${target}"
    ) > "/tmp/bootstrap_${model}.log" 2>&1 &
    pids+=("$!")
  done

  fail=0
  for ((j=0; j<${#pids[@]}; j++)); do
    model="${MODELS_TO_INSTALL[$((j+1))]}"
    if wait "${pids[$j]}"; then
      echo "[bootstrap] Install OK: ${model}"
    else
      echo "[bootstrap] Install FAILED: ${model} (see /tmp/bootstrap_${model}.log)" >&2
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    exit 1
  fi

  # Validation runs serially: model load + GPU work should not contend.
  if [[ "${RUN_VALIDATE}" == "true" ]]; then
    for i in "${!MODELS_TO_INSTALL[@]}"; do
      model="${MODELS_TO_INSTALL[$i]}"
      target="${TARGET_ENVS[$i]}"
      conda activate "${target}"
      echo "[bootstrap] Validating ${target}"
      run_common_validation
      run_target_validators_for_model "${model}" "${ENVS_TO_INSTALL[@]}"
      conda deactivate
    done
  fi

  echo "[bootstrap] Complete."
  echo "[bootstrap] Split env mode used. Activate with: conda activate ${CONDA_ENV_NAME}_<model>"
  echo "[bootstrap] Repo root: ${REPO_ROOT}"
  exit 0
fi

ensure_conda_env "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
validate_active_python_version "${CONDA_ENV_NAME}"
install_in_active_env "${MODELS_TO_INSTALL[0]}" "false" "${ENVS_TO_INSTALL[@]}"

if [[ ${#MODELS_TO_INSTALL[@]} -gt 1 ]]; then
  for ((i=1; i<${#MODELS_TO_INSTALL[@]}; i++)); do
    echo "[bootstrap] Installing additional model dependencies in shared env: ${MODELS_TO_INSTALL[$i]}"
    PYTHON_BIN=python "${SCRIPT_DIR}/model/${MODELS_TO_INSTALL[$i]}.sh"
  done
fi

if [[ "${RUN_VALIDATE}" == "true" ]]; then
  echo "[bootstrap] Running validation"
  run_common_validation
  for model in "${MODELS_TO_INSTALL[@]}"; do
    run_target_validators_for_model "${model}" "${ENVS_TO_INSTALL[@]}"
  done
fi

echo "[bootstrap] Complete."
echo "[bootstrap] Activate later with: conda activate ${CONDA_ENV_NAME}"
echo "[bootstrap] Repo root: ${REPO_ROOT}"
