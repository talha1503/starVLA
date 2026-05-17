#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_PROFILE="${STARVLA_TORCH_PROFILE:-auto}"

detect_torch_profile() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  local compute_caps
  compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  if echo "${compute_caps}" | grep -qx "12.0"; then
    echo "cu128"
  else
    echo "cu124"
  fi
}

install_torch() {
  local profile="$1"

  case "${profile}" in
    cu128)
      echo "[install/torch] Installing PyTorch 2.7.1 CUDA 12.8 stack for Blackwell/sm_120"
      "$PYTHON_BIN" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128
      ;;
    cu126)
      echo "[install/torch] Installing PyTorch 2.6.0 CUDA 12.6 stack"
      "$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu126
      ;;
    cu124)
      echo "[install/torch] Installing PyTorch 2.6.0 CUDA 12.4 stack"
      "$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
      ;;
    cpu)
      echo "[install/torch] Installing PyTorch 2.6.0 CPU stack"
      "$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cpu
      ;;
    *)
      echo "[install/torch] Unknown STARVLA_TORCH_PROFILE='${profile}'. Expected auto|cu124|cu126|cu128|cpu." >&2
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
