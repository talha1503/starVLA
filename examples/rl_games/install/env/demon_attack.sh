#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install ale-py "gymnasium[atari]" autorom
AutoROM --accept-license
"$PYTHON_BIN" - <<'PY'
import gymnasium as gym
import ale_py  # noqa: F401

env = gym.make("ALE/DemonAttack-v5")
env.close()
print("ok-demon-attack-env")
PY

echo "[install/env/demon_attack] done"
