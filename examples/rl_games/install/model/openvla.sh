#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install peft "imageio[ffmpeg]" draccus "datasets>=3.0"

echo "[install/model/openvla] done"
