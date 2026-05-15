#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c "import transformers, gymnasium, ale_py; print('ok-pi0-demon')"
