#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "datasets>=3.0" peft safetensors "imageio[ffmpeg]" draccus opencv-python-headless
pip_install diffusers qwen-vl-utils decord eva-decord av timm tyro

echo "[install/model/wan_oft] done"
