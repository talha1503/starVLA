#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: install_stack.sh <model: openvla|pi0|gr00t> <env: flappy|demon_attack|deadly_corridor>"
  exit 1
fi

MODEL="$1"
ENV_NAME="$2"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$BASE_DIR/common.sh"
"$BASE_DIR/model/${MODEL}.sh"
"$BASE_DIR/env/${ENV_NAME}.sh"
"$BASE_DIR/validate/common.sh"

echo "[install_stack] installed model=${MODEL} env=${ENV_NAME}"
