#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_VERSION="${STARVLA_FLASH_ATTN_VERSION:-2.8.3}"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/flash_attn.sh [--print-url]

Environment:
  PYTHON_BIN                         Python executable to inspect and install into.
  STARVLA_FLASH_ATTN_VERSION         FlashAttention release version (default: 2.8.3).
  STARVLA_FLASH_ATTN_WHEEL_URL       Exact wheel URL override.
EOF
}

flash_attn_wheel_url() {
  if [[ -n "${STARVLA_FLASH_ATTN_WHEEL_URL:-}" ]]; then
    printf '%s\n' "${STARVLA_FLASH_ATTN_WHEEL_URL}"
    return
  fi

  local runtime_info py_tag torch_mm cuda_tag abi
  runtime_info="$(
    "$PYTHON_BIN" - <<'PY'
import sys

import torch

python_tag = sys.implementation.cache_tag
torch_version = torch.__version__.split("+", 1)[0]
torch_parts = torch_version.split(".")
torch_mm = ".".join(torch_parts[:2])
torch_cuda = torch.version.cuda
if torch_cuda is None:
    raise SystemExit("[install/flash_attn] CUDA-enabled PyTorch is required for flash-attn wheels")
cuda_major = torch_cuda.split(".", 1)[0]
cuda_tag = f"cu{cuda_major}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(python_tag)
print(torch_mm)
print(cuda_tag)
print(abi)
PY
  )"

  mapfile -t fields <<< "${runtime_info}"
  py_tag="${fields[0]}"
  torch_mm="${fields[1]}"
  cuda_tag="${fields[2]}"
  abi="${fields[3]}"
  case "${torch_mm}/${cuda_tag}" in
    2.6/cu12|2.7/cu12) ;;
    *)
      echo "[install/flash_attn] unsupported torch/CUDA runtime: torch=${torch_mm} cuda=${cuda_tag}" >&2
      exit 1
      ;;
  esac
  printf 'https://github.com/Dao-AILab/flash-attention/releases/download/v%s/flash_attn-%s+%storch%scxx11abi%s-%s-%s-linux_x86_64.whl\n' \
    "${FLASH_ATTN_VERSION}" \
    "${FLASH_ATTN_VERSION}" \
    "${cuda_tag}" \
    "${torch_mm}" \
    "${abi}" \
    "${py_tag}" \
    "${py_tag}"
}

PRINT_URL="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-url)
      PRINT_URL="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install/flash_attn] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

FLASH_ATTN_WHEEL_URL="$(flash_attn_wheel_url)"
if [[ "${PRINT_URL}" == "true" ]]; then
  printf '%s\n' "${FLASH_ATTN_WHEEL_URL}"
  exit 0
fi

echo "[install/flash_attn] Installing ${FLASH_ATTN_WHEEL_URL}"
"$PYTHON_BIN" -m pip install --no-deps "${FLASH_ATTN_WHEEL_URL}"
"$PYTHON_BIN" -c "import flash_attn, flash_attn_2_cuda; print(f'ok-flash-attn-{flash_attn.__version__}')"

echo "[install/flash_attn] done"
