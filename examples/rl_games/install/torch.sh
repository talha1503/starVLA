#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_PROFILE="${STARVLA_TORCH_PROFILE:-auto}"
# Wheel index base. Defaults to the official PyTorch CDN; install.sh / bootstrap.sh
# speed-test mirrors and export STARVLA_TORCH_INDEX_BASE (e.g. the Aliyun mirror) to
# route these multi-GB downloads through the fastest provider.
TORCH_INDEX_BASE="${STARVLA_TORCH_INDEX_BASE:-https://download.pytorch.org/whl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"
# shellcheck source=_torch_profile.sh
source "${SCRIPT_DIR}/_torch_profile.sh"

install_torch() {
  local profile="$1"

  case "${profile}" in
    cu130)
      echo "[install/torch] Installing PyTorch 2.12.1 CUDA 13.0 stack for Blackwell"
      pip_install torch==2.12.1+cu130 torchvision==0.27.1+cu130 \
        --index-url "${TORCH_INDEX_BASE}/cu130"
      ;;
    cu128)
      echo "[install/torch] Installing PyTorch 2.10.0 CUDA 12.8 stack"
      pip_install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
        --index-url "${TORCH_INDEX_BASE}/cu128"
      ;;
    cu126)
      echo "[install/torch] Installing PyTorch 2.10.0 CUDA 12.6 stack"
      pip_install torch==2.10.0+cu126 torchvision==0.25.0+cu126 \
        --index-url "${TORCH_INDEX_BASE}/cu126"
      ;;
    cpu)
      echo "[install/torch] Installing PyTorch 2.10.0 CPU stack"
      pip_install torch==2.10.0 torchvision==0.25.0 \
        --index-url "${TORCH_INDEX_BASE}/cpu"
      ;;
    *)
      echo "[install/torch] unknown resolved profile '${profile}'" >&2
      exit 1
      ;;
  esac
}

TORCH_PROFILE="$(resolve_torch_profile "${TORCH_PROFILE}")"
export STARVLA_TORCH_PROFILE="${TORCH_PROFILE}"

install_torch "${TORCH_PROFILE}"

"$PYTHON_BIN" - <<'PY'
import torch

print(f"[install/torch] torch={torch.__version__} cuda={torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[install/torch] gpu={torch.cuda.get_device_name(0)} capability={cap[0]}.{cap[1]}")
PY
