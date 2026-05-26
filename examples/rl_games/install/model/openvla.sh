#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON_BIN" -m pip install peft "imageio[ffmpeg]" draccus "datasets>=3.0"
"${SCRIPT_DIR}/../flash_attn.sh"

echo "[install/model/openvla] done"
