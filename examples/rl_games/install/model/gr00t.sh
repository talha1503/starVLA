#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON_BIN" -m pip install "datasets>=3.0" decord
"$PYTHON_BIN" -m pip install "ray[default]==2.47.0"
"$PYTHON_BIN" -m pip install peft "imageio[ffmpeg]" draccus opencv-python-headless
"${SCRIPT_DIR}/../flash_attn.sh"

echo "[install/model/gr00t] done"
