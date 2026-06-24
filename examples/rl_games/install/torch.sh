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

detect_torch_profile() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  local compute_caps
  compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  # Consumer Blackwell (sm_120) keeps the tested CUDA 12.8 stack.
  if echo "${compute_caps}" | grep -qx "12.0"; then
    echo "cu128"
  # Datacenter Blackwell (sm_100, compute cap 10.x, e.g. B200) needs CUDA 13.
  elif echo "${compute_caps}" | awk -F. '$1 >= 10 {found=1} END {exit found ? 0 : 1}'; then
    echo "cu130"
  else
    echo "cu124"
  fi
}

install_torch() {
  local profile="$1"

  case "${profile}" in
    cu130)
      echo "[install/torch] Installing PyTorch 2.10.0 CUDA 13.0 stack for datacenter Blackwell/sm_100"
      pip_install torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 \
        --index-url "${TORCH_INDEX_BASE}/cu130"
      ;;
    cu128)
      echo "[install/torch] Installing PyTorch 2.7.1 CUDA 12.8 stack for Blackwell/sm_120"
      pip_install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
        --index-url "${TORCH_INDEX_BASE}/cu128"
      ;;
    cu126)
      echo "[install/torch] Installing PyTorch 2.6.0 CUDA 12.6 stack"
      pip_install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
        --index-url "${TORCH_INDEX_BASE}/cu126"
      ;;
    cu124)
      echo "[install/torch] Installing PyTorch 2.6.0 CUDA 12.4 stack"
      pip_install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
        --index-url "${TORCH_INDEX_BASE}/cu124"
      ;;
    cpu)
      echo "[install/torch] Installing PyTorch 2.6.0 CPU stack"
      pip_install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url "${TORCH_INDEX_BASE}/cpu"
      ;;
    *)
      echo "[install/torch] Unknown STARVLA_TORCH_PROFILE='${profile}'. Expected auto|cu124|cu126|cu128|cu130|cpu." >&2
      exit 1
      ;;
  esac
}

if [[ "${TORCH_PROFILE}" == "auto" ]]; then
  TORCH_PROFILE="$(detect_torch_profile)"
fi

install_torch "${TORCH_PROFILE}"

"$PYTHON_BIN" - <<'PY'
import torch

print(f"[install/torch] torch={torch.__version__} cuda={torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[install/torch] gpu={torch.cuda.get_device_name(0)} capability={cap[0]}.{cap[1]}")
PY
