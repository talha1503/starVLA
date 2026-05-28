import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from starVLA.training.rl_games.eval_core import (  # noqa: E402
    ActionLatencyQueue,
    _TaskEvaluator,
    decode_deadly_factorized_11,
    decode_deadly_joint_54,
    decode_deadly_multibinary_7,
    decode_discrete_argmax,
)


def test_decode_discrete_argmax():
    values = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    assert decode_discrete_argmax(values, 3) == 1


def test_decode_deadly_multibinary_7():
    values = np.array([0.2, 0.6, 0.51, 0.49, 0.8, 0.1, 0.9], dtype=np.float32)
    assert decode_deadly_multibinary_7(values) == [0, 1, 1, 0, 1, 0, 1]


def test_decode_deadly_factorized_11():
    values = np.array([
        0.1, 0.8, 0.1,
        0.2, 0.1, 0.7,
        0.3, 0.6, 0.1,
        0.4, 0.6,
    ], dtype=np.float32)
    assert decode_deadly_factorized_11(values) == [1, 2, 1, 1]


def test_decode_deadly_joint_54():
    values = np.zeros(54, dtype=np.float32)
    values[11] = 1.0
    assert decode_deadly_joint_54(values) == [0, 1, 2, 1]


def test_action_latency_queue():
    queue = ActionLatencyQueue(latency=2, default_action=0)
    queue.reset()
    outputs = [queue.schedule_and_get(a) for a in [1, 2, 3, 4]]
    assert outputs == [0, 0, 1, 2]


def _evaluator(task: str, action_layout: str):
    evaluator = _TaskEvaluator.__new__(_TaskEvaluator)
    evaluator.task = task
    evaluator.cfg = SimpleNamespace(
        framework=SimpleNamespace(action_model=SimpleNamespace(action_layout=action_layout))
    )
    return evaluator


def test_demon_attack_eval_decodes_six_way_categorical_action():
    evaluator = _evaluator("demon_attack", "demon_attack_categorical_6")

    assert evaluator._decode_action(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 9.0])) == 5


def test_deadly_corridor_eval_decodes_factorized_layout_from_action_model_config():
    evaluator = _evaluator("deadly_corridor", "deadly_corridor_factorized_11")

    decoded = evaluator._decode_action(
        np.asarray([0.0, 9.0, 1.0, 0.0, 0.0, 8.0, 0.0, 7.0, 0.0, 6.0, 0.0])
    )

    assert decoded == [0, 1, 1, 0, 1, 0, 0]


def test_deadly_corridor_eval_decodes_joint_fifty_four_layout_from_action_model_config():
    evaluator = _evaluator("deadly_corridor", "deadly_corridor_joint_54")
    vector = np.zeros(54, dtype=np.float32)
    vector[11] = 1.0

    decoded = evaluator._decode_action(vector)

    assert decoded == [1, 0, 0, 1, 0, 0, 1]


def test_task_evaluator_omits_state_when_training_config_excludes_state():
    cfg = SimpleNamespace(
        rl_games=SimpleNamespace(env_eval=SimpleNamespace(image_size=4, frameskip=1)),
        framework=SimpleNamespace(action_model=SimpleNamespace()),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(include_state=False)),
    )
    evaluator = _TaskEvaluator("flappy", cfg)

    example = evaluator._build_example(np.zeros((4, 4, 3), dtype=np.uint8), "flap now")

    assert example["lang"] == "flap now"
    assert "state" not in example


def test_task_evaluator_includes_state_when_training_config_includes_state():
    cfg = SimpleNamespace(
        rl_games=SimpleNamespace(env_eval=SimpleNamespace(image_size=4, frameskip=1)),
        framework=SimpleNamespace(action_model=SimpleNamespace(state_dim=1)),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(include_state=True)),
    )
    evaluator = _TaskEvaluator("flappy", cfg)

    example = evaluator._build_example(np.zeros((4, 4, 3), dtype=np.uint8), "flap now")

    assert example["state"].shape == (1, 1)
    assert np.all(example["state"] == 0.0)
