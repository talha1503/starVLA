#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install "transformers>=4.53.0" "datasets>=3.0" decord
"$PYTHON_BIN" -m pip install "ray[default]==2.47.0"
"$PYTHON_BIN" -m pip install peft timm "imageio[ffmpeg]" draccus rich opencv-python-headless

echo "[install/model/gr00t] done"
