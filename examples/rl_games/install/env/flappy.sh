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
# force-reinstall + no-deps is load-bearing: regular pygame (not pygame-ce). The PIL
# fallback patch below is the real safety net: inside the heavy trainer process,
# SDL2_image's bundled libpng gets symbol-interposed by PyAV/OpenCV/Pillow/torchvision
# libpng and fails to decode the sprites; the fallback decodes via Pillow instead, so
# no LD_PRELOAD is needed at training time. Drop --no-cache-dir so the wheel comes from cache.
pip_install --force-reinstall --no-deps "pygame==2.6.1" pillow

"$PYTHON_BIN" - <<'PY'
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import flappy_bird_gymnasium  # noqa: F401
import flappy_bird_gymnasium.envs.utils as flappy_utils
import gymnasium as gym
from pathlib import Path

utils_path = Path(flappy_utils.__file__)
source = utils_path.read_text(encoding="utf-8")
marker = "[StarVLA PATCH] PIL fallback"
if marker not in source:
    patch = '''

# [StarVLA PATCH] PIL fallback for sprites pygame's libpng cannot decode.
import pygame as _starvla_pygame
from PIL import Image as _starvla_PILImage
_starvla_orig_load_sprite = _load_sprite
def _load_sprite(filename, convert, alpha=True):  # noqa: F811
    try:
        return _starvla_orig_load_sprite(filename, convert, alpha)
    except _starvla_pygame.error:
        path = f"{SPRITES_PATH}/{filename}"
        mode = "RGBA" if alpha else "RGB"
        pil = _starvla_PILImage.open(path).convert(mode)
        surface = _starvla_pygame.image.fromstring(pil.tobytes(), pil.size, mode)
        if convert:
            try:
                surface = surface.convert_alpha() if alpha else surface.convert()
            except _starvla_pygame.error:
                pass
        return surface
'''
    utils_path.write_text(source + patch, encoding="utf-8")
    print(f"[install/env/flappy] patched sprite loader: {utils_path}")
else:
    print(f"[install/env/flappy] sprite loader already patched: {utils_path}")

# Reload the module so validation below uses the patched loader in this process.
import importlib
importlib.reload(flappy_utils)

env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env.reset()
frame = env.render()
env.close()

if frame is None:
    raise SystemExit("[install/env/flappy] Flappy render returned None")
print("[install/env/flappy] env render ok")
PY

echo "[install/env/flappy] done"
