#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=_host.sh
source "${SCRIPT_DIR}/_host.sh"
# shellcheck source=_torch_profile.sh
source "${SCRIPT_DIR}/_torch_profile.sh"

INSTALL_TIER="use"
MODEL_TARGET="openvla"
PYTHON_VERSION="3.10"
CONDA_ENV_NAME=""
USE_CONDA="true"
RUN_VALIDATE="true"
ACCEPT_ROM_LICENSE="false"
TORCH_PROFILE="${STARVLA_TORCH_PROFILE:-auto}"
REQUESTED_ENVS=()

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/bootstrap.sh [options]

Options:
  --tier <use|dev>         Dependency tier (default: ${INSTALL_TIER})
  --model <name>           openvla|pi0|pi05|gr00t|wan_oft (default: ${MODEL_TARGET})
  --env <name|all>         Repeatable: flappy|demon_attack|deadly_corridor|all (default: all)
  --torch-profile <name>   auto|cpu|cu126|cu128|cu130 (default: ${TORCH_PROFILE})
  --conda-env <name>       Conda env name (default: starvla_rl_games_<model>)
  --python-version <ver>   Python version for a new env (default: ${PYTHON_VERSION})
  --current-env            Install into the active Python environment
  --accept-rom-license     Permit AutoROM to accept and download Atari ROMs
  --skip-validate          Skip final validation
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      INSTALL_TIER="$2"
      shift 2
      ;;
    --model)
      MODEL_TARGET="$2"
      shift 2
      ;;
    --env)
      REQUESTED_ENVS+=("$2")
      shift 2
      ;;
    --torch-profile)
      TORCH_PROFILE="$2"
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
    --current-env|--no-conda)
      USE_CONDA="false"
      shift
      ;;
    --accept-rom-license)
      ACCEPT_ROM_LICENSE="true"
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
      echo "[bootstrap] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${INSTALL_TIER}" in
  use|dev) ;;
  *)
    echo "[bootstrap] invalid tier '${INSTALL_TIER}'; expected use|dev" >&2
    exit 1
    ;;
esac

case "${MODEL_TARGET}" in
  openvla|pi0|pi05|gr00t|wan_oft) ;;
  *)
    echo "[bootstrap] invalid model '${MODEL_TARGET}'; expected openvla|pi0|pi05|gr00t|wan_oft" >&2
    exit 1
    ;;
esac

ENVS_TO_INSTALL=()
if [[ ${#REQUESTED_ENVS[@]} -eq 0 ]]; then
  ENVS_TO_INSTALL=(flappy demon_attack deadly_corridor)
else
  if [[ ${#REQUESTED_ENVS[@]} -gt 1 ]]; then
    for env_name in "${REQUESTED_ENVS[@]}"; do
      if [[ "${env_name}" == "all" ]]; then
        echo "[bootstrap] --env all cannot be combined with another --env" >&2
        exit 1
      fi
    done
  fi
  for env_name in "${REQUESTED_ENVS[@]}"; do
    case "${env_name}" in
      all)
        ENVS_TO_INSTALL=(flappy demon_attack deadly_corridor)
        ;;
      flappy|demon_attack|deadly_corridor)
        ENVS_TO_INSTALL+=("${env_name}")
        ;;
      *)
        echo "[bootstrap] invalid env '${env_name}'; expected flappy|demon_attack|deadly_corridor|all" >&2
        exit 1
        ;;
    esac
  done
fi

SELECTED_TORCH_PROFILE="$(resolve_torch_profile "${TORCH_PROFILE}")"
export STARVLA_INSTALL_TIER="${INSTALL_TIER}"
export STARVLA_TORCH_PROFILE="${SELECTED_TORCH_PROFILE}"
export ACCEPT_ROM_LICENSE

ensure_conda_env() {
  local env_name="$1"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[bootstrap] using existing conda env: ${env_name}"
  else
    echo "[bootstrap] creating conda env ${env_name} (python=${PYTHON_VERSION})"
    conda create -n "${env_name}" -c conda-forge --override-channels "python=${PYTHON_VERSION}" -y
  fi
}

validate_active_python_version() {
  local env_name="$1"
  local active_py requested_py
  active_py="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  requested_py="$(echo "${PYTHON_VERSION}" | awk -F. '{if (NF>=2) print $1"."$2; else print $1}')"
  if [[ "${active_py}" != "${requested_py}" ]]; then
    echo "[bootstrap] env '${env_name}' uses Python ${active_py}; expected ${requested_py}" >&2
    exit 1
  fi
}

target_validator_name() {
  local model="$1"
  local env_name="$2"
  case "${model}:${env_name}" in
    pi0:demon_attack) echo "pi0_demon.sh" ;;
    *) echo "${model}_${env_name}.sh" ;;
  esac
}

run_validation() {
  PYTHON_BIN=python "${SCRIPT_DIR}/validate/common.sh"
  local env_name validator
  for env_name in "${ENVS_TO_INSTALL[@]}"; do
    validator="${SCRIPT_DIR}/validate/$(target_validator_name "${MODEL_TARGET}" "${env_name}")"
    if [[ -f "${validator}" ]]; then
      PYTHON_BIN=python bash "${validator}"
    else
      echo "[bootstrap] no validator registered for ${MODEL_TARGET}/${env_name}; skipped"
    fi
  done
}

if [[ "${USE_CONDA}" == "true" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[bootstrap] conda is required unless --current-env is used" >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-${CONDA_BASE}/.uv_cache}"
  unset UV_NO_CACHE UV_LINK_MODE

  CONDA_ENV_NAME="${CONDA_ENV_NAME:-starvla_rl_games_${MODEL_TARGET}}"
  ensure_conda_env "${CONDA_ENV_NAME}"
  conda activate "${CONDA_ENV_NAME}"
  validate_active_python_version "${CONDA_ENV_NAME}"
fi

echo "[bootstrap] tier=${INSTALL_TIER} model=${MODEL_TARGET} torch=${SELECTED_TORCH_PROFILE} envs=${ENVS_TO_INSTALL[*]}"
PYTHON_BIN=python "${SCRIPT_DIR}/common.sh"
PYTHON_BIN=python "${SCRIPT_DIR}/model/${MODEL_TARGET}.sh"

for env_name in "${ENVS_TO_INSTALL[@]}"; do
  PYTHON_BIN=python "${SCRIPT_DIR}/env/${env_name}.sh"
done

if [[ "${RUN_VALIDATE}" == "true" ]]; then
  run_validation
fi

echo "[bootstrap] complete"
if [[ "${USE_CONDA}" == "true" ]]; then
  echo "[bootstrap] activate with: conda activate ${CONDA_ENV_NAME}"
fi
echo "[bootstrap] repo root: ${REPO_ROOT}"
