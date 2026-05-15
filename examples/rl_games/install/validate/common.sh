#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

"$PYTHON_BIN" -c "import omegaconf, hydra, torch; print('ok-common')"
"$PYTHON_BIN" -c "import starVLA; print('ok-starVLA')"
"$PYTHON_BIN" -m compileall "$REPO_ROOT/starVLA/training/rl_games" >/dev/null

echo "[validate/common] done"
