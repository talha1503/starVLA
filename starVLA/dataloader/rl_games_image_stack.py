from __future__ import annotations

import json
from pathlib import Path


def apply_rl_games_image_stack_video_keys(dataset_path: Path, robot_type: str, modality_config: dict):
    if not robot_type.startswith("rl_games_"):
        return modality_config

    modality_path = Path(dataset_path) / "meta/modality.json"
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    stack_video_keys = [
        f"video.{key}"
        for key in sorted(modality["video"])
        if key.startswith("image_stack_")
    ]
    if not stack_video_keys:
        return modality_config

    updated = dict(modality_config)
    video_config = modality_config["video"]
    updated["video"] = type(video_config)(
        delta_indices=video_config.delta_indices,
        modality_keys=stack_video_keys,
    )
    return updated
