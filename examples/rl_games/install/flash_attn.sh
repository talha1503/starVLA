#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_VERSION="${STARVLA_FLASH_ATTN_VERSION:-2.8.3.post1}"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/flash_attn.sh [--print-url|--check]

Installs legacy flash-attn into the active env, matching a prebuilt wheel to the
active torch/CUDA/Python/ABI. Best-effort by default: if no wheel matches it warns
and exits 0 (models fall back to sdpa). Set STARVLA_FLASH_ATTN_BUILD_FROM_SOURCE=1
to allow a source build, or STARVLA_FLASH_ATTN_REQUIRED=1 to make failure fatal.

Options:
  --print-url   Print the resolved prebuilt wheel URL and exit (fails on an
                unsupported torch/CUDA runtime; does not install).
  --check       Report flash-attn status in the active env (OK / MISSING /
                ABI-MISMATCH) and exit 0 without installing.

Environment:
  PYTHON_BIN                         Python executable to inspect and install into.
  STARVLA_FLASH_ATTN_VERSION         FlashAttention release version (default: 2.8.3.post1).
  STARVLA_FLASH_ATTN_WHEEL_URL       Exact wheel URL override.
  STARVLA_FLASH_ATTN_BUILD_FROM_SOURCE
                                     Set to 1 to build when no wheel matches.
  STARVLA_FLASH_ATTN_REQUIRED        Set to 1 to fail the install when flash-attn
                                     cannot be installed (default: best-effort).
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

python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
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
  # Prebuilt-wheel matrix: combos Dao-AILab ships legacy flash-attn wheels for.
  # torch 2.12 / CUDA 13 uses flash-attn-4 for FlexAttention BACKEND=FLASH and
  # should not silently source-build this legacy package.
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

verify_import() {
  "$PYTHON_BIN" -c "import flash_attn, flash_attn_2_cuda" >/dev/null 2>&1
}

install_from_wheel() {
  local url="$1"
  echo "[install/flash_attn] Installing prebuilt wheel: ${url}"
  "$PYTHON_BIN" -m pip install --no-deps "${url}"
}

install_from_source() {
  echo "[install/flash_attn] Building flash-attn==${FLASH_ATTN_VERSION} from source (--no-build-isolation)"
  "$PYTHON_BIN" -m pip install ninja packaging wheel
  "$PYTHON_BIN" -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
}

# Best-effort install: prebuilt wheel by default. Source builds are explicit only
# because missing wheels on torch/CUDA 13 otherwise block fresh installs for a
# package that FlexAttention BACKEND=FLASH does not use.
do_install() {
  local url
  if url="$(flash_attn_wheel_url 2>/dev/null)"; then
    if install_from_wheel "${url}" && verify_import; then
      echo "[install/flash_attn] installed from prebuilt wheel"
      return 0
    fi
    echo "[install/flash_attn] prebuilt wheel path failed" >&2
  else
    echo "[install/flash_attn] no matching prebuilt wheel for this torch/CUDA" >&2
  fi

  if [[ "${STARVLA_FLASH_ATTN_BUILD_FROM_SOURCE:-0}" != "1" ]]; then
    echo "[install/flash_attn] source build disabled; set STARVLA_FLASH_ATTN_BUILD_FROM_SOURCE=1 to build" >&2
    return 1
  fi

  if install_from_source && verify_import; then
    echo "[install/flash_attn] installed from source build"
    return 0
  fi
  return 1
}

# Reports flash-attn status and exits 0 only when it imports cleanly. A non-zero
# exit lets callers gate a reinstall (`flash_attn.sh --check || flash_attn.sh`);
# the validator calls it with `|| true` to keep the report non-fatal.
run_check() {
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import flash_attn
    import flash_attn_2_cuda  # noqa: F401
    print(f"[flash_attn/check] OK {flash_attn.__version__}")
    sys.exit(0)
except ImportError as exc:
    msg = str(exc)
    if "undefined symbol" in msg or "cannot import name" in msg:
        print(f"[flash_attn/check] ABI-MISMATCH (rebuild needed): {msg}")
    else:
        print("[flash_attn/check] MISSING (models will fall back to sdpa)")
    sys.exit(1)
except Exception as exc:  # pragma: no cover - unexpected import error
    print(f"[flash_attn/check] ERROR: {exc}")
    sys.exit(1)
PY
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
      echo "[install/flash_attn] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  print-url)
    # Strict: exits non-zero on an unsupported runtime (no source-build fallback).
    FLASH_ATTN_WHEEL_URL="$(flash_attn_wheel_url)"
    printf '%s\n' "${FLASH_ATTN_WHEEL_URL}"
    exit 0
    ;;
  check)
    if run_check; then exit 0; fi
    exit 1
    ;;
esac

if do_install; then
  echo "[install/flash_attn] done"
  exit 0
fi

echo "[install/flash_attn] [WARNING] flash-attn unavailable; models will fall back to sdpa" >&2
if [[ "${STARVLA_FLASH_ATTN_REQUIRED:-0}" == "1" ]]; then
  echo "[install/flash_attn] STARVLA_FLASH_ATTN_REQUIRED=1 set; failing." >&2
  exit 1
fi
exit 0
