#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
ENV_NAME=""
CONDA_ENV_NAME=""
PYTHON_VERSION="3.10"
USE_CONDA="true"

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/install_stack.sh [options] <model> <env>

Examples:
  bash examples/rl_games/install/install_stack.sh openvla flappy
  bash examples/rl_games/install/install_stack.sh --conda-env my_openvla openvla flappy

Arguments:
  <model>                  openvla|pi0|gr00t
  <env>                    flappy|demon_attack|deadly_corridor

Options:
  --conda-env <name>       Conda env name (default: starvla_rl_games_<model>)
  --python-version <ver>   Python version for new conda env (default: ${PYTHON_VERSION})
  --no-conda               Use the current python instead of creating/activating conda
  -h, --help               Show this help
EOF
}

parse_args() {
  local args=()

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
      --no-conda)
        USE_CONDA="false"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      -*)
        echo "[install_stack] Unknown option: $1" >&2
        usage
        exit 1
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#args[@]} -ne 2 ]]; then
    usage
    exit 1
  fi

  MODEL="${args[0]}"
  ENV_NAME="${args[1]}"
}

validate_targets() {
  case "${MODEL}" in
    openvla|pi0|gr00t) ;;
    *)
      echo "[install_stack] Invalid model '${MODEL}'. Expected openvla|pi0|gr00t." >&2
      exit 1
      ;;
  esac

  case "${ENV_NAME}" in
    flappy|demon_attack|deadly_corridor) ;;
    *)
      echo "[install_stack] Invalid env '${ENV_NAME}'. Expected flappy|demon_attack|deadly_corridor." >&2
      exit 1
      ;;
  esac
}

setup_conda_env() {
  if [[ "${USE_CONDA}" != "true" ]]; then
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[install_stack] conda is required. Use --no-conda only if your current python env is already correct." >&2
    exit 1
  fi

  if [[ -z "${CONDA_ENV_NAME}" ]]; then
    CONDA_ENV_NAME="starvla_rl_games_${MODEL}"
  fi

  local conda_base
  conda_base="$(conda info --base)"
  source "${conda_base}/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "[install_stack] Using existing conda env: ${CONDA_ENV_NAME}"
  else
    echo "[install_stack] Creating conda env ${CONDA_ENV_NAME} with python=${PYTHON_VERSION}"
    conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  conda activate "${CONDA_ENV_NAME}"
  assert_python_version "${CONDA_ENV_NAME}"
}

assert_python_version() {
  local env_label="$1"
  local active_py requested_py

  active_py="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_py="$(echo "${PYTHON_VERSION}" | awk -F. '{if (NF>=2) print $1"."$2; else print $1}')"

  if [[ "${active_py}" != "${requested_py}" ]]; then
    echo "[install_stack] Python mismatch in '${env_label}': got ${active_py}, expected ${requested_py}." >&2
    echo "[install_stack] Use a new --conda-env name, or remove and recreate the existing env." >&2
    exit 1
  fi
}

install_stack() {
  echo "[install_stack] Installing starVLA base dependencies"
  "${BASE_DIR}/common.sh"

  echo "[install_stack] Installing model dependencies: ${MODEL}"
  "${BASE_DIR}/model/${MODEL}.sh"

  echo "[install_stack] Installing environment dependencies: ${ENV_NAME}"
  "${BASE_DIR}/env/${ENV_NAME}.sh"

  echo "[install_stack] Running validation"
  "${BASE_DIR}/validate/common.sh"
  local target_validator="${BASE_DIR}/validate/${MODEL}_${ENV_NAME}.sh"
  if [[ -x "${target_validator}" ]]; then
    "${target_validator}"
  elif [[ -f "${target_validator}" ]]; then
    bash "${target_validator}"
  else
    echo "[install_stack] No target-specific validator found for ${MODEL}/${ENV_NAME}; common validation passed."
  fi
}

parse_args "$@"
validate_targets
setup_conda_env
install_stack

echo "[install_stack] Complete: model=${MODEL} env=${ENV_NAME}"
if [[ "${USE_CONDA}" == "true" ]]; then
  echo "[install_stack] Activate later with: conda activate ${CONDA_ENV_NAME}"
fi
