#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLASH_ATTN_WHEEL_URL="${STARVLA_FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"

"$PYTHON_BIN" -m pip install "datasets>=3.0"
"$PYTHON_BIN" -m pip install jax flax augmax beartype gcsfs openpi-client
"$PYTHON_BIN" -m pip install jaxtyping==0.2.36 tyro==1.0.12 ml-collections==1.0.0 sentencepiece==0.2.1 chex==0.1.90 numpydantic==1.8.0
"$PYTHON_BIN" -m pip install --no-deps "$FLASH_ATTN_WHEEL_URL"
"$PYTHON_BIN" -c "import flash_attn, flash_attn_2_cuda; print(f'ok-flash-attn-{flash_attn.__version__}')"

echo "[install/model/pi0] done"
