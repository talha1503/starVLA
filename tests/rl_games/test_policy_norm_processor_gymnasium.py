import numpy as np
import pytest

from deployment.model_server.policy_norm_processor import (
    PolicyNormProcessor,
    _resolve_robot_type,
)


@pytest.mark.parametrize(
    ("task_name", "data_mix", "robot_type"),
    [
        ("flappy", "flappy_train", "rl_games_flappy"),
        ("demon_attack", "demon_attack_train", "rl_games_demon_attack"),
        ("deadly_corridor", "deadly_corridor_train", "rl_games_deadly_corridor"),
    ],
)
def test_legacy_rl_games_resolves_from_static_mixture(
    task_name: str,
    data_mix: str,
    robot_type: str,
) -> None:
    config = {
        "rl_games": {"task": task_name},
        "datasets": {"vla_data": {"data_mix": data_mix}},
    }

    assert _resolve_robot_type(config) == robot_type


def test_unknown_static_mixture_still_fails() -> None:
    config = {
        "rl_games": {"task": "flappy"},
        "datasets": {"vla_data": {"data_mix": "unregistered_custom_mix"}},
    }

    with pytest.raises(KeyError, match="unregistered_custom_mix"):
        _resolve_robot_type(config)


def test_live_native_processor_applies_training_q99_transform() -> None:
    processor = PolicyNormProcessor.from_config(
        model_cfg={},
        norm_stats={
            "gymnasium": {
                "state": {"q01": [0.0, 10.0], "q99": [2.0, 14.0]},
                "action": {"q01": [-2.0, 0.0], "q99": [2.0, 4.0]},
            }
        },
        unnorm_key="gymnasium",
        robot_type="rl_games_gymnasium_native",
    )

    np.testing.assert_allclose(
        processor.apply_state(np.asarray([[1.0, 12.0]], dtype=np.float32)),
        [[0.0, 0.0]],
    )
    np.testing.assert_allclose(
        processor.unapply_actions(np.asarray([[0.0, 0.5]], dtype=np.float32)),
        [[0.0, 3.0]],
    )
