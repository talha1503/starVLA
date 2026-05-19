#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import ray  # noqa: F401
import transformers  # noqa: F401

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T  # noqa: F401

env = gym.make("FlappyBird-v0")
env.reset()
env.close()
print("ok-gr00t-flappy")
PY
