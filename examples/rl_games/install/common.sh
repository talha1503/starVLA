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

# Make the parent repo's latency_bench package importable from the training env.
# starVLA is nested under the parent repo (latency-sensitive-bench), which contains
# latency_bench/ but is not itself a pip package. Register the parent root via a .pth
# file — the same mechanism editable installs use — so `import latency_bench` works
# natively. This replaces the old in-code sys.path.insert(parents[3]) hack in
# train_starvla.py.
PARENT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
if [ -d "${PARENT_REPO_ROOT}/latency_bench" ]; then
  SITE_DIR="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  echo "${PARENT_REPO_ROOT}" > "${SITE_DIR}/latency_bench_repo.pth"
  echo "[install/common] registered latency_bench root: ${PARENT_REPO_ROOT} -> ${SITE_DIR}/latency_bench_repo.pth"
else
  echo "[install/common] WARN: latency_bench not found at ${PARENT_REPO_ROOT}; skipping .pth (standalone starVLA checkout?)"
fi

echo "[install/common] done"
