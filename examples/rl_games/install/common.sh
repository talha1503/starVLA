#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"

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
ensure_uv
"${SCRIPT_DIR}/torch.sh"

FILTERED_REQUIREMENTS="$(mktemp)"
grep -Ev '^(torch|torchvision|torchaudio)([<=>[:space:]]|$)' "$REPO_ROOT/requirements.txt" > "$FILTERED_REQUIREMENTS"
pip_install -r "$FILTERED_REQUIREMENTS"
rm -f "$FILTERED_REQUIREMENTS"

pip_install -e "$REPO_ROOT"
pip_install omegaconf hydra-core tqdm wandb huggingface_hub

echo "[install/common] done"
