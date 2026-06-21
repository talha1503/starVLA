import numpy as np
import sys
import types
from omegaconf import OmegaConf

from starVLA.training.rl_games.eval_core import (
    ActionLatencyQueue,
    RlGamesEvalRunner,
    _TaskEvaluator,
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


def test_task_evaluator_reuses_episode_seeds_across_tasks_and_latencies_by_default():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    for task in ("flappy", "demon_attack", "deadly_corridor"):
        evaluator = _TaskEvaluator(task=task, cfg=cfg)
        assert [evaluator._episode_seed(latency=0, episode=episode) for episode in range(3)] == [42, 43, 44]
        assert [evaluator._episode_seed(latency=8, episode=episode) for episode in range(3)] == [42, 43, 44]


def test_task_evaluator_respects_explicit_latency_seed_stride():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "latency_seed_stride": 1000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)

    assert [evaluator._episode_seed(latency=8, episode=episode) for episode in range(3)] == [8042, 8043, 8044]


def test_task_evaluator_respects_explicit_task_seed_stride():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "task_seed_stride": 100000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="demon_attack", cfg=cfg)

    assert [evaluator._episode_seed(latency=0, episode=episode) for episode in range(3)] == [100042, 100043, 100044]


def test_task_evaluator_builds_temporal_image_examples_for_wan_oft():
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "frameskip": 1,
                    "image_size": 4,
                },
            },
            "datasets": {
                "vla_data": {
                    "pack_image_sequence": True,
                    "image_sequence_length": 4,
                    "include_state": True,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 7,
                },
            },
        }
    )
    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)
    frames = [
        np.full((4, 4, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ]

    history = evaluator._initial_model_history(frames[0])
    for frame in frames[1:]:
        history = evaluator._advance_model_history(history, frame)
    example = evaluator._make_model_example(history, "play flappy")

    assert len(example["image"]) == 4
    assert [int(np.asarray(image)[0, 0, 0]) for image in example["image"]] == [10, 20, 30, 40]
    assert example["state"].shape == (1, 7)


def test_task_evaluator_saved_seed_overrides_take_precedence():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "latency_seed_stride": 1000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)

    assert evaluator._episode_seed_for_run(latency=8, episode=1, seed_overrides={1: 123456}) == 123456
    assert evaluator._episode_seed_for_run(latency=8, episode=2, seed_overrides={1: 123456}) == 8044
    assert evaluator._episode_seed_for_run(latency=8, episode=3, seed_overrides={3: None}) is None


class _FixedRng:
    def __init__(self, value: int):
        self.value = value

    def integers(self, low: int, high: int) -> int:
        assert low == 1
        assert high == 31
        return self.value


class _FakeDemonAttackEnv:
    def __init__(self):
        self.np_random = _FixedRng(value=30)
        self.reset_calls = []
        self.step_actions = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return "reset_obs", {"reset": True}

    def step(self, action):
        self.step_actions.append(action)
        return "noop_obs", 0.0, False, False, {"noop": True}


def test_demon_attack_env_uses_noop_reset_max_30_by_default(monkeypatch):
    fake_env = _FakeDemonAttackEnv()
    fake_gym = types.SimpleNamespace(make=lambda *args, **kwargs: fake_env)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "ale_py", types.SimpleNamespace())
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "frameskip": 4,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="demon_attack", cfg=cfg)
    env = evaluator._make_env()
    obs, info = env.reset(seed=42)

    assert obs == "noop_obs"
    assert info == {"noop": True}
    assert fake_env.reset_calls == [{"seed": 42}]
    assert fake_env.step_actions == [0] * 30


def test_eval_runner_saves_result_after_each_episode(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "demon_attack",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "seed": 42,
                    "post_train": {
                        "enabled": True,
                        "latencies": [0],
                        "num_episodes": 2,
                        "max_steps_per_episode": 1,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )
    save_totals = []
    original_save = RlGamesEvalRunner._save

    def spy_save(self, result, step, stage):
        save_totals.append(int(result.aggregate["total_episodes"]))
        return original_save(self, result, step, stage)

    def fake_run_latency(self, **kwargs):
        first_episode_metrics = {
            "latency": 0,
            "num_episodes": 1,
            "mean_reward": 1.0,
            "mean_length": 1.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "episode_rewards": [1.0],
            "episode_lengths": [1],
            "decoded_action_hist": {"0": 1},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [42],
        }
        second_episode_metrics = {
            "latency": 0,
            "num_episodes": 2,
            "mean_reward": 1.5,
            "mean_length": 1.0,
            "std_reward": 0.5,
            "std_length": 0.0,
            "episode_rewards": [1.0, 2.0],
            "episode_lengths": [1, 1],
            "decoded_action_hist": {"0": 2},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [42, 43],
        }
        kwargs["on_episode_complete"](first_episode_metrics)
        kwargs["on_episode_complete"](second_episode_metrics)
        return second_episode_metrics

    monkeypatch.setattr(RlGamesEvalRunner, "_save", spy_save)
    monkeypatch.setattr(_TaskEvaluator, "run_latency", fake_run_latency)

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))
    result = runner.run(model=object(), step=2500, stage="post_train")

    assert save_totals == [1, 2, 2]
    assert result.path == str(tmp_path / "eval" / "post_train" / "step_2500.json")


def test_eval_runner_passes_saved_seed_overrides_by_latency(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "flappy",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "mid_train": {
                        "enabled": True,
                        "latencies": [0],
                        "num_episodes": 2,
                        "max_steps_per_episode": 1,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )
    seen_seed_overrides = []
    seen_episode_indices = []

    def fake_run_latency(self, **kwargs):
        seen_seed_overrides.append(kwargs["seed_overrides"])
        seen_episode_indices.append(kwargs["episode_indices"])
        return {
            "latency": 0,
            "num_episodes": 2,
            "mean_reward": 0.0,
            "mean_length": 1.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "episode_rewards": [0.0, 0.0],
            "episode_lengths": [1, 1],
            "decoded_action_hist": {"0": 2},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [9001, 9002],
            "episode_indices": [5, 7],
        }

    monkeypatch.setattr(_TaskEvaluator, "run_latency", fake_run_latency)

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))
    result = runner.run(
        model=object(),
        step=1000,
        stage="mid_train",
        episode_seed_overrides={"flappy/latency_0": {5: 9001, 7: 9002}},
    )

    assert seen_seed_overrides == [{5: 9001, 7: 9002}]
    assert seen_episode_indices == [[5, 7]]
    assert result.per_latency["flappy/latency_0"]["episode_seeds"] == [9001, 9002]
