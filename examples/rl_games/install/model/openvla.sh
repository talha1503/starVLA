#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install "transformers==4.40.1" "tokenizers==0.19.1"
"$PYTHON_BIN" -m pip install peft timm "imageio[ffmpeg]" draccus rich "datasets>=3.0"

echo "[install/model/openvla] done"
