#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install peft "imageio[ffmpeg]" draccus

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install "datasets>=3.0"
  if [[ "${STARVLA_TORCH_PROFILE}" != "cpu" ]]; then
    INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"
    if [[ "${STARVLA_TORCH_PROFILE}" == "cu130" ]]; then
      PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn4.sh"
    fi
  fi
fi

echo "[install/model/openvla] tier=${INSTALL_TIER} done"
