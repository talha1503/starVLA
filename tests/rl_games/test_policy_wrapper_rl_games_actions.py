from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def _load_policy_wrapper_module():
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = object()
    base_framework_module = ModuleType("starVLA.model.framework.base_framework")
    base_framework_module.baseframework = SimpleNamespace()
    share_tools_module = ModuleType("starVLA.model.framework.share_tools")
    share_tools_module.read_mode_config = lambda ckpt_path: ({}, {})
    norm_processor_module = ModuleType("deployment.model_server.policy_norm_processor")
    norm_processor_module.PolicyNormProcessor = object
    sys.modules["starVLA.model.framework.base_framework"] = base_framework_module
    sys.modules["starVLA.model.framework.share_tools"] = share_tools_module
    sys.modules["deployment.model_server.policy_norm_processor"] = norm_processor_module
    sys.modules["torch"] = torch_module

    module_path = STARVLA_ROOT / "deployment/model_server/policy_wrapper.py"
    spec = importlib.util.spec_from_file_location("policy_wrapper_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.unnorm_key = "new_embodiment"
        self.available_unnorm_keys = ["new_embodiment"]
        self.action_keys = ["action"]
        self.state_keys = []

    def unapply_actions(self, values):
        self.calls.append(np.asarray(values).copy())
        return np.asarray(values) + 10.0


class FakeFramework:
    def __init__(self, normalized_actions):
        self.normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        self.calls = []

    def predict_action(self, examples, **kwargs):
        self.calls.append({"examples": examples, "kwargs": kwargs})
        return {"normalized_actions": self.normalized_actions}


class FakeProfiler:
    def __init__(self):
        self.timings = {}

    def time(self, stage):
        @contextmanager
        def timer():
            yield
            self.timings[stage] = 1.0

        return timer()


def test_decode_flappy_logits_to_discrete_action_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32),
        env_name="flappy",
    )

    assert decoded["action_output_type"] == "rl_games_discrete_id"
    assert decoded["actions"].tolist() == [[[1], [0]]]
    assert np.allclose(decoded["raw_action_scores"], [[[0.1, 0.9], [0.8, 0.2]]])


def test_decode_demon_attack_logits_to_discrete_action_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray([[[0.0, 0.1, 3.2, -0.4, 0.2, 0.0]]], dtype=np.float32),
        env_name="demon_attack",
    )

    assert decoded["action_output_type"] == "rl_games_discrete_id"
    assert decoded["actions"].tolist() == [[[2]]]


def test_decode_deadly_corridor_factorized_logits_to_tuple_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray(
            [[[0.0, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1, 0.7, 0.0, 0.3, 0.9]]],
            dtype=np.float32,
        ),
        env_name="deadly_corridor",
    )

    assert decoded["action_output_type"] == "rl_games_deadly_corridor_tuple"
    assert decoded["actions"].tolist() == [[[2, 0, 1, 1]]]


def test_decode_deadly_corridor_joint_logits_to_tuple_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    raw = np.zeros((1, 1, 54), dtype=np.float32)
    raw[0, 0, 11] = 1.0
    decoded = decode_rl_games_actions(normalized_actions=raw, env_name="deadly_corridor")

    assert decoded["action_output_type"] == "rl_games_deadly_corridor_tuple"
    assert decoded["actions"].tolist() == [[[0, 1, 2, 1]]]


def test_policy_wrapper_default_mode_keeps_unnormalized_actions():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[1.0, 2.0]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "deployment"
    wrapper._rl_games_env_name = None
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="new_embodiment")

    assert prediction["actions"].tolist() == [[[11.0, 12.0]]]
    assert len(processor.calls) == 1


def test_policy_wrapper_rl_games_mode_returns_decoded_actions_without_unapply():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[0.1, 0.9]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "flappy"
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="new_embodiment")

    assert prediction["actions"].tolist() == [[[1]]]
    assert np.allclose(prediction["raw_action_scores"], [[[0.1, 0.9]]])
    assert prediction["action_output_type"] == "rl_games_discrete_id"
    assert processor.calls == []


def test_policy_wrapper_default_mode_records_unnormalize_timing():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[1.0, 2.0]]])
    profiler = FakeProfiler()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "deployment"
    wrapper._rl_games_env_name = None
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
        profiler=profiler,
    )

    assert prediction["actions"].tolist() == [[[11.0, 12.0]]]
    assert framework.calls[0]["kwargs"]["profiler"] is profiler
    assert profiler.timings["starvla_wrapper_unnormalize_ms"] == 1.0


def test_policy_wrapper_rl_games_mode_records_decode_timing():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[0.1, 0.9]]])
    profiler = FakeProfiler()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "flappy"
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
        profiler=profiler,
    )

    assert prediction["actions"].tolist() == [[[1]]]
    assert framework.calls[0]["kwargs"]["profiler"] is profiler
    assert profiler.timings["starvla_wrapper_rl_games_decode_ms"] == 1.0
    assert processor.calls == []
