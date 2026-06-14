from __future__ import annotations

from typing import Any, Callable

import numpy as np


def decode_rl_games_actions(*, normalized_actions: np.ndarray, env_name: str) -> dict[str, Any]:
    raw_scores = np.asarray(normalized_actions)
    decoder = {
        "flappy": lambda: _decode_discrete(raw_scores, 2),
        "demon_attack": lambda: _decode_discrete(raw_scores, 6),
        "deadly_corridor": lambda: _decode_deadly_corridor(raw_scores),
    }[str(env_name)]()
    actions, action_output_type = decoder()
    return {
        "actions": actions,
        "raw_action_scores": raw_scores,
        "action_output_type": action_output_type,
    }


def _decode_discrete(raw_scores: np.ndarray, n_actions: int) -> Callable[[], tuple[np.ndarray, str]]:
    def decode() -> tuple[np.ndarray, str]:
        actions = np.argmax(raw_scores[..., :n_actions], axis=-1)[..., None]
        return actions.astype(np.int64), "rl_games_discrete_id"

    return decode


def _decode_deadly_corridor(raw_scores: np.ndarray) -> Callable[[], tuple[np.ndarray, str]]:
    return {
        7: lambda: (_decode_deadly_semantic_7(raw_scores), "rl_games_deadly_corridor_tuple"),
        11: lambda: (_decode_deadly_factorized_11(raw_scores), "rl_games_deadly_corridor_tuple"),
        54: lambda: (_decode_deadly_joint_54(raw_scores), "rl_games_deadly_corridor_tuple"),
    }[raw_scores.shape[-1]]


def _decode_deadly_factorized_11(raw_scores: np.ndarray) -> np.ndarray:
    turn = np.argmax(raw_scores[..., 0:3], axis=-1)
    move = np.argmax(raw_scores[..., 3:6], axis=-1)
    strafe = np.argmax(raw_scores[..., 6:9], axis=-1)
    attack = np.argmax(raw_scores[..., 9:11], axis=-1)
    return np.stack((turn, move, strafe, attack), axis=-1).astype(np.int64)


def _decode_deadly_joint_54(raw_scores: np.ndarray) -> np.ndarray:
    joint_id = np.argmax(raw_scores, axis=-1)
    attack = joint_id % 2
    strafe = (joint_id // 2) % 3
    move = (joint_id // 6) % 3
    turn = (joint_id // 18) % 3
    return np.stack((turn, move, strafe, attack), axis=-1).astype(np.int64)


def _decode_deadly_semantic_7(raw_scores: np.ndarray) -> np.ndarray:
    action_id = np.argmax(raw_scores, axis=-1)
    semantic_actions = np.asarray(
        (
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 2, 0),
            (1, 0, 0, 0),
            (2, 0, 0, 0),
            (0, 0, 0, 1),
        ),
        dtype=np.int64,
    )
    return semantic_actions[action_id]
