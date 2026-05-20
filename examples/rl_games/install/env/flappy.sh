#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install flappy-bird-gymnasium
"$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall --no-deps "pygame==2.6.1" pillow flappy-bird-gymnasium

"$PYTHON_BIN" - <<'PY'
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym

env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env.reset()
frame = env.render()
env.close()

if frame is None:
    raise SystemExit("[install/env/flappy] Flappy render returned None")
print("[install/env/flappy] env render ok")
PY

echo "[install/env/flappy] done"
