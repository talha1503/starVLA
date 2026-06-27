#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

# Install the repo's custom flappy fork (provides GPU rendering / render_state,
# used by latency_bench's in-process flappy eval). Stock PyPI flappy-bird-gymnasium
# lacks flappy_bird_gymnasium.envs.render_state.BatchedFlappyRenderState. The fork
# lives at the parent repo root (this submodule is nested under it).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
FLAPPY_FORK="${REPO_ROOT}/flappy-bird-gymnasium"
if [ -d "${FLAPPY_FORK}" ]; then
  pip_install -e "${FLAPPY_FORK}"
else
  echo "[install/env/flappy] WARN: fork not found at ${FLAPPY_FORK}; falling back to PyPI (no GPU rendering)"
  pip_install flappy-bird-gymnasium
fi
# force-reinstall + no-deps is load-bearing: regular pygame (not pygame-ce), not
# pygame-ce that another dep may have pulled. pillow is needed by the fork's
# _load_sprite PIL fallback (see below). Drop --no-cache-dir so the wheel comes from cache.
pip_install --force-reinstall --no-deps "pygame==2.6.1" pillow

# The libpng symbol-interposition fix lives IN the fork source now (the fork is our
# own submodule): flappy_bird_gymnasium.envs.utils._load_sprite falls back to Pillow
# when SDL2_image's bundled libpng — interposed by PyAV/OpenCV/Pillow/torchvision in
# the heavy trainer process — cannot decode a sprite. So this script no longer patches
# any installed file; it only smoke-tests that rendering works.
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
