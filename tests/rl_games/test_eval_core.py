import numpy as np

from starVLA.training.rl_games.eval_core import (
    ActionLatencyQueue,
    decode_deadly_factorized_11,
    decode_deadly_multibinary_7,
    decode_discrete_argmax,
)


def test_decode_discrete_argmax():
    values = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    assert decode_discrete_argmax(values, 3) == 1


def test_decode_discrete_argmax_ignores_bridge_padding():
    values = np.array([0.1, 0.2, 99.0, 99.0, 99.0, 99.0, 99.0], dtype=np.float32)
    assert decode_discrete_argmax(values, 2) == 1


def test_decode_deadly_multibinary_7():
    values = np.array([0.2, 0.6, 0.51, 0.49, 0.8, 0.1, 0.9], dtype=np.float32)
    assert decode_deadly_multibinary_7(values) == [0, 1, 1, 0, 1, 0, 1]


def test_decode_deadly_multibinary_7_ignores_extra_padding():
    values = np.array([0.2, 0.6, 0.51, 0.49, 0.8, 0.1, 0.9, 99.0], dtype=np.float32)
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
