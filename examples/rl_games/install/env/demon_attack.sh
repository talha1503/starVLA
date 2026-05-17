#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install ale-py "gymnasium[atari]" autorom
AutoROM --accept-license || true

echo "[install/env/demon_attack] done"
