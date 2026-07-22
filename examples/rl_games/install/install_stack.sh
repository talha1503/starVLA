#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME=""
PYTHON_VERSION="3.10"
TORCH_PROFILE="${STARVLA_TORCH_PROFILE:-auto}"
USE_CONDA="true"
ACCEPT_ROM_LICENSE="false"
POSITIONAL=()

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/install_stack.sh [options] <model> <env>

Compatibility entrypoint for a full development/training environment.

Arguments:
  <model>                  openvla|pi0|pi05|gr00t|wan_oft
  <env>                    flappy|demon_attack|deadly_corridor|cross_task

Options:
  --conda-env <name>       Conda env name (default: starvla_rl_games_<model>)
  --python-version <ver>   Python version for a new env (default: ${PYTHON_VERSION})
  --torch-profile <name>   auto|cpu|cu126|cu128|cu130
  --no-conda               Install into the active Python environment
  --accept-rom-license     Permit AutoROM to accept and download Atari ROMs
  -h, --help               Show this help
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
    --torch-profile)
      TORCH_PROFILE="$2"
      shift 2
      ;;
    --no-conda)
      USE_CONDA="false"
      shift
      ;;
    --accept-rom-license)
      ACCEPT_ROM_LICENSE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  usage
  exit 1
fi

MODEL="${POSITIONAL[0]}"
ENV_NAME="${POSITIONAL[1]}"
ARGS=(
  --tier dev
  --model "${MODEL}"
  --python-version "${PYTHON_VERSION}"
  --torch-profile "${TORCH_PROFILE}"
)

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  ARGS+=(--conda-env "${CONDA_ENV_NAME}")
fi
if [[ "${USE_CONDA}" == "false" ]]; then
  ARGS+=(--current-env)
fi
if [[ "${ACCEPT_ROM_LICENSE}" == "true" ]]; then
  ARGS+=(--accept-rom-license)
fi

if [[ "${ENV_NAME}" == "cross_task" ]]; then
  ARGS+=(--env flappy --env demon_attack)
else
  ARGS+=(--env "${ENV_NAME}")
fi

exec "${INSTALL_DIR}/bootstrap.sh" "${ARGS[@]}"
