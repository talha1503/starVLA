#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON_BIN" -m pip install "datasets>=3.0"
"$PYTHON_BIN" -m pip install jax flax augmax beartype gcsfs openpi-client
"$PYTHON_BIN" -m pip install jaxtyping==0.2.36 tyro==1.0.12 ml-collections==1.0.0 sentencepiece==0.2.1 chex==0.1.90 numpydantic==1.8.0
"${SCRIPT_DIR}/../flash_attn.sh"

echo "[install/model/pi0] done"
