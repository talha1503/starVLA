#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install vizdoom gymnasium

echo "[install/env/deadly_corridor] done"
