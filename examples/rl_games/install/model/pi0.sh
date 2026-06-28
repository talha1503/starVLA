#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "datasets>=3.0"
pip_install jax flax augmax beartype gcsfs openpi-client
pip_install jaxtyping==0.2.36 tyro==1.0.12 ml-collections==1.0.0 sentencepiece==0.2.1 chex==0.1.90 numpydantic==1.8.0

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"
if [[ "${STARVLA_TORCH_PROFILE:-}" == "cu130" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn4.sh"
fi

echo "[install/model/pi0] done"
