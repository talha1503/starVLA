import numpy as np
from omegaconf import OmegaConf

from starVLA.training.rl_games.eval_core import (
    ActionLatencyQueue,
    RlGamesEvalRunner,
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


def test_eval_runner_infers_latencies_from_prompt_map_when_values_empty(tmp_path):
    prompt_map = tmp_path / "latency_prompt_map.json"
    prompt_map.write_text(
        '{"0": {"latency": 0, "latency_ms": 0.0, "prompt": "zero"},'
        ' "2": {"latency": 2, "latency_ms": 66.6, "prompt": "two"}}',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "task": "demon_attack",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "latency": {
                        "values": [],
                        "prompt_map_path": str(prompt_map),
                    },
                },
            }
        }
    )

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))

    assert runner._get_latency_values(stage="mid_train") == [0, 2]
