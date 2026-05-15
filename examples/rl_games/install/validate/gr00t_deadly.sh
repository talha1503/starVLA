#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c "import vizdoom, gymnasium, ray; print('ok-gr00t-deadly')"
