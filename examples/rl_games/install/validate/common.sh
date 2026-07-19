#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

"$PYTHON_BIN" -c "import omegaconf, torch; print('ok-common-use')"
"$PYTHON_BIN" -c "import starVLA; print('ok-starVLA')"
"$PYTHON_BIN" -m latency_bench.run --help >/dev/null
"$PYTHON_BIN" -m compileall "$REPO_ROOT/starVLA/training/rl_games" >/dev/null

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  "$PYTHON_BIN" -c "import hydra; print('ok-common-dev')"
  if [[ "${STARVLA_TORCH_PROFILE}" != "cpu" ]]; then
    INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PYTHON_BIN="$PYTHON_BIN" "${INSTALL_DIR}/flash_attn.sh" --check
  fi
fi

echo "[validate/common] tier=${INSTALL_TIER} done"
