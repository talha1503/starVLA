#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

"$PYTHON_BIN" -c "import omegaconf, hydra, torch; print('ok-common')"
"$PYTHON_BIN" -c "import starVLA; print('ok-starVLA')"
"$PYTHON_BIN" -m compileall "$REPO_ROOT/starVLA/training/rl_games" >/dev/null

# Report flash-attn status for this env (non-fatal): OK / MISSING / ABI-MISMATCH.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PYTHON_BIN" "${INSTALL_DIR}/flash_attn.sh" --check || true

echo "[validate/common] done"
