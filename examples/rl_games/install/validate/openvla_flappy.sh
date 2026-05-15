#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c "import transformers; import flappy_bird_gymnasium; print('ok-openvla-flappy')"
