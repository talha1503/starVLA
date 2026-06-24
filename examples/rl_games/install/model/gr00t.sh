#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "datasets>=3.0" decord
pip_install "ray[default]==2.47.0"
pip_install peft "imageio[ffmpeg]" draccus opencv-python-headless

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"

echo "[install/model/gr00t] done"
