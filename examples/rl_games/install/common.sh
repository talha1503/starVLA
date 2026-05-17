#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

"$PYTHON_BIN" - <<'PY'
import sys
v = sys.version_info
if not ((v.major, v.minor) >= (3, 10) and (v.major, v.minor) < (3, 13)):
    raise SystemExit(
        f"[install/common] Unsupported Python {v.major}.{v.minor}. "
        "Use Python 3.10-3.12 (recommended: 3.10)."
    )
PY

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install -e "$REPO_ROOT"
"$PYTHON_BIN" -m pip install omegaconf hydra-core tqdm wandb huggingface_hub

echo "[install/common] done"
