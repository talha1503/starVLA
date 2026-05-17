#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install flappy-bird-gymnasium

echo "[install/env/flappy] done"
