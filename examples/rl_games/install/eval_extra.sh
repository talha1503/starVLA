#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_REPO_ROOT="$(cd "${STARVLA_ROOT}/.." && pwd)"
# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"

(
  cd "${BENCH_REPO_ROOT}"
  pip_install -r requirements_starvla_eval_extra.txt
)

echo "[install/eval_extra] done"
