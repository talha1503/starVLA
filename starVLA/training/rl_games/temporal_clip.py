from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModalityWindowIndices:
    observation_indices: list[int]
    language_indices: list[int]
    state_indices: list[int]
    action_indices: list[int]


def _cfg_get(data_cfg: Mapping[str, Any] | Any | None, key: str) -> Any:
    if data_cfg is None:
        return None
    getter = getattr(data_cfg, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(data_cfg, key, None)


def _int_list(value: Sequence[int] | None, label: str) -> list[int]:
    if value is None:
        raise ValueError(f"Missing {label}")
    out = [int(item) for item in value]
    if not out:
        raise ValueError(f"{label} must not be empty")
    return out


def resolve_modality_indices(
    default_observation_indices: Sequence[int],
    default_state_indices: Sequence[int],
    default_action_indices: Sequence[int],
    data_cfg: Mapping[str, Any] | Any | None,
) -> ModalityWindowIndices:
    observation_override = _cfg_get(data_cfg, "observation_indices")
    language_override = _cfg_get(data_cfg, "language_indices")
    state_override = _cfg_get(data_cfg, "state_indices")
    action_override = _cfg_get(data_cfg, "action_indices")

    observation_indices = _int_list(
        observation_override if observation_override is not None else default_observation_indices,
        "observation_indices",
    )
    language_indices = _int_list(
        language_override if language_override is not None else ([0] if observation_override is not None else observation_indices),
        "language_indices",
    )
    state_indices = _int_list(
        state_override if state_override is not None else ([0] if observation_override is not None else default_state_indices),
        "state_indices",
    )
    action_indices = _int_list(
        action_override if action_override is not None else default_action_indices,
        "action_indices",
    )

    return ModalityWindowIndices(
        observation_indices=observation_indices,
        language_indices=language_indices,
        state_indices=state_indices,
        action_indices=action_indices,
    )


def pack_image_sequence(
    frames: np.ndarray,
    pack_image_sequence: bool,
    image_sequence_length: int,
    resize_size: tuple[int, int],
) -> list[Image.Image]:
    if frames.ndim != 4:
        raise ValueError(f"Expected image frames shaped [T,H,W,C], got shape={frames.shape}")
    if image_sequence_length <= 0:
        raise ValueError(f"image_sequence_length must be positive, got {image_sequence_length}")

    selected_frames = frames if pack_image_sequence else frames[:1]
    expected_length = image_sequence_length if pack_image_sequence else 1
    if selected_frames.shape[0] != expected_length:
        raise ValueError(
            f"Expected {expected_length} image frame(s), got {selected_frames.shape[0]} "
            f"with pack_image_sequence={pack_image_sequence}"
        )

    return [Image.fromarray(frame).resize(resize_size) for frame in selected_frames]
