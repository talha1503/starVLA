#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"

# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"
# shellcheck source=_host.sh
source "${SCRIPT_DIR}/_host.sh"

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

pip_install -r "$REPO_ROOT/requirements.txt"
pip_install -e "$REPO_ROOT"
pip_install -e "$LATENCY_BENCH_ROOT"

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install -r "$REPO_ROOT/requirements-dev.txt"
  pip_install -e "$REPO_ROOT[dev]"
fi

echo "[install/common] tier=${INSTALL_TIER} done"
