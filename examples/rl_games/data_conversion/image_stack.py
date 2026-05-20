from __future__ import annotations

from typing import Any

import pyarrow as pa


IMAGE_STRUCT_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
IMAGE_STACK_ORDER = "oldest_to_newest"
IMAGE_STACK_SOURCE = "policy_observation_frame_stack"


def image_column(index: int) -> str:
    return f"observation.image_stack_{index:02d}"


def video_key(index: int) -> str:
    return f"image_stack_{index:02d}"


def image_feature(fps: int) -> dict[str, Any]:
    return {
        "dtype": "image",
        "shape": [84, 84, 3],
        "names": ["height", "width", "channel"],
        "video_info": {"video.fps": fps},
    }


def image_array(image_bytes: list[bytes]) -> pa.Array:
    return pa.array(
        [{"bytes": value, "path": None} for value in image_bytes],
        type=IMAGE_STRUCT_TYPE,
    )


def image_stack_table_columns(rows: list[dict[str, Any]], image_stack_size: int) -> dict[str, pa.Array]:
    return {
        image_column(index): image_array([row["image_stack_bytes"][index] for row in rows])
        for index in range(image_stack_size)
    }


def image_stack_modality(image_stack_size: int) -> dict[str, dict[str, str]]:
    return {
        video_key(index): {"original_key": image_column(index)}
        for index in range(image_stack_size)
    }


def image_stack_features(image_stack_size: int, fps: int) -> dict[str, dict[str, Any]]:
    return {
        image_column(index): image_feature(fps)
        for index in range(image_stack_size)
    }
