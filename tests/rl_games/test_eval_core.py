import numpy as np
from types import SimpleNamespace

from starVLA.training.rl_games.eval_core import (
    ActionLatencyQueue,
    _TaskEvaluator,
    decode_deadly_factorized_11,
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


def test_action_latency_queue():
    queue = ActionLatencyQueue(latency=2, default_action=0)
    queue.reset()
    outputs = [queue.schedule_and_get(a) for a in [1, 2, 3, 4]]
    assert outputs == [0, 0, 1, 2]


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
