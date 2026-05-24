#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_WHEEL_URL="${STARVLA_FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

"$PYTHON_BIN" -m pip install peft "imageio[ffmpeg]" draccus "datasets>=3.0"
"$PYTHON_BIN" -m pip install --no-deps "$FLASH_ATTN_WHEEL_URL"
"$PYTHON_BIN" -c "import flash_attn, flash_attn_2_cuda; print(f'ok-flash-attn-{flash_attn.__version__}')"

echo "[install/model/openvla] done"
