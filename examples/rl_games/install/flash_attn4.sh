#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN4_VERSION="${STARVLA_FLASH_ATTN4_VERSION:-4.0.0b19}"
FLASH_ATTN4_RELEASE_TAG="${STARVLA_FLASH_ATTN4_RELEASE_TAG:-fa4-v4.0.0.beta19}"
FLASH_ATTN4_WHEEL_URL="${STARVLA_FLASH_ATTN4_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/${FLASH_ATTN4_RELEASE_TAG}/flash_attn_4-${FLASH_ATTN4_VERSION}-py3-none-any.whl}"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/flash_attn4.sh [--print-url|--check]

Installs FlashAttention-4 for PyTorch FlexAttention BACKEND=FLASH. This is the
CuTeDSL package that provides flash_attn.cute; it is separate from legacy
flash-attn / flash_attn_2_cuda used by Hugging Face flash_attention_2.

Options:
  --print-url   Print the resolved FA4 wheel URL and exit.
  --check       Report FA4 status in the active env and exit without installing.

Environment:
  PYTHON_BIN                         Python executable to inspect and install into.
  STARVLA_FLASH_ATTN4_VERSION        FA4 release version (default: 4.0.0b19).
  STARVLA_FLASH_ATTN4_RELEASE_TAG    GitHub release tag (default: fa4-v4.0.0.beta19).
  STARVLA_FLASH_ATTN4_WHEEL_URL      Exact wheel URL override.
EOF
}

run_check() {
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import flash_attn.cute  # noqa: F401
    import flash_attn.cute.interface  # noqa: F401
    print("[flash_attn4/check] OK")
    sys.exit(0)
except ImportError as exc:
    print(f"[flash_attn4/check] MISSING: {exc}")
    sys.exit(1)
except Exception as exc:  # pragma: no cover - unexpected import error
    print(f"[flash_attn4/check] ERROR: {exc}")
    sys.exit(1)
PY
}

install_flash_attn4() {
  echo "[install/flash_attn4] Installing FlashAttention-4: ${FLASH_ATTN4_WHEEL_URL}"
  "$PYTHON_BIN" -m pip install "flash-attn-4[cu13] @ ${FLASH_ATTN4_WHEEL_URL}"
}

MODE="install"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-url)
      MODE="print-url"
      shift
      ;;
    --check)
      MODE="check"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install/flash_attn4] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  print-url)
    printf '%s\n' "${FLASH_ATTN4_WHEEL_URL}"
    exit 0
    ;;
  check)
    if run_check; then exit 0; fi
    exit 1
    ;;
esac

install_flash_attn4
run_check
echo "[install/flash_attn4] done"
