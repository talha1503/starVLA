#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install peft "imageio[ffmpeg]" draccus "datasets>=3.0"

echo "[install/model/openvla] done"
