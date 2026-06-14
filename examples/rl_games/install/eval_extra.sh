#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_REPO_ROOT="$(cd "${STARVLA_ROOT}/.." && pwd)"

(
  cd "${BENCH_REPO_ROOT}"
  "$PYTHON_BIN" -m pip install -r requirements_starvla_eval_extra.txt
)

echo "[install/eval_extra] done"
