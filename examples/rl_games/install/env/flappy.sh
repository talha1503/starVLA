#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m pip install flappy-bird-gymnasium
"$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall --no-deps "pygame==2.6.1" pillow flappy-bird-gymnasium

"$PYTHON_BIN" - <<'PY'
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path

import flappy_bird_gymnasium  # noqa: F401
import flappy_bird_gymnasium.envs.utils as flappy_utils
import gymnasium as gym
import pygame
from PIL import Image

repaired = []
for sprite_path in sorted(Path(flappy_utils.SPRITES_PATH).glob("*.png")):
    try:
        pygame.image.load(str(sprite_path))
        continue
    except pygame.error:
        pass

    image = Image.open(sprite_path)
    mode = "RGBA" if image.mode in {"RGBA", "LA", "P"} else "RGB"
    image.convert(mode).save(sprite_path, format="PNG", optimize=False)
    pygame.image.load(str(sprite_path))
    repaired.append(sprite_path.name)

if repaired:
    print(f"[install/env/flappy] repaired sprite PNGs: {', '.join(repaired)}")

env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
env.reset()
frame = env.render()
env.close()

if frame is None:
    raise SystemExit("[install/env/flappy] Flappy render returned None")
print("[install/env/flappy] env render ok")
PY

echo "[install/env/flappy] done"
