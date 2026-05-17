#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
USE_CONDA="true"

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/install_stack.sh [options] <model> <env>

Arguments:
  <model>                  openvla|pi0|gr00t
  <env>                    flappy|demon_attack|deadly_corridor

Options:
  --conda-env <name>       Conda env name (default: starvla_rl_games_<model>)
  --python-version <ver>   Python version for new env (default: ${PYTHON_VERSION})
  --no-conda               Skip conda create/activate and use current python
  -h, --help               Show this help
EOF
}

ARGS=()
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
      echo "[install_stack] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -ne 2 ]]; then
  usage
  exit 1
fi

MODEL="${ARGS[0]}"
ENV_NAME="${ARGS[1]}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

if [[ -z "${CONDA_ENV_NAME}" ]]; then
  CONDA_ENV_NAME="starvla_rl_games_${MODEL}"
fi

if [[ "${USE_CONDA}" == "true" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[install_stack] conda is required but not found in PATH. Use --no-conda to skip conda setup." >&2
    exit 1
  fi

  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "[install_stack] Using existing conda env: ${CONDA_ENV_NAME}"
  else
    echo "[install_stack] Creating conda env ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  conda activate "${CONDA_ENV_NAME}"

  ACTIVE_PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  REQUESTED_PY_MM="$(echo "${PYTHON_VERSION}" | awk -F. '{if (NF>=2) print $1"."$2; else print $1}')"
  if [[ "${ACTIVE_PY_MM}" != "${REQUESTED_PY_MM}" ]]; then
    echo "[install_stack] Conda env '${CONDA_ENV_NAME}' has Python ${ACTIVE_PY_MM}, expected ${REQUESTED_PY_MM}." >&2
    echo "[install_stack] Use a different --conda-env name, or remove and recreate this env." >&2
    exit 1
  fi
fi

"$BASE_DIR/common.sh"
"$BASE_DIR/model/${MODEL}.sh"
"$BASE_DIR/env/${ENV_NAME}.sh"
"$BASE_DIR/validate/common.sh"

echo "[install_stack] installed model=${MODEL} env=${ENV_NAME}"
