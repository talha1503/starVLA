import sys
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def _load_action_postprocess_module():
    module_path = STARVLA_ROOT / "deployment/model_server/action_postprocess.py"
    spec = importlib.util.spec_from_file_location("policy_action_postprocess_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


postprocess_actions = _load_action_postprocess_module().postprocess_actions


def _load_policy_wrapper_module():
    base_framework_module = ModuleType("starVLA.model.framework.base_framework")
    base_framework_module.baseframework = SimpleNamespace()
    share_tools_module = ModuleType("starVLA.model.framework.share_tools")
    share_tools_module.read_mode_config = lambda ckpt_path: ({}, {})
    norm_processor_module = ModuleType("deployment.model_server.policy_norm_processor")
    norm_processor_module.PolicyNormProcessor = object
    sys.modules["starVLA.model.framework.base_framework"] = base_framework_module
    sys.modules["starVLA.model.framework.share_tools"] = share_tools_module
    sys.modules["deployment.model_server.policy_norm_processor"] = norm_processor_module

    module_path = STARVLA_ROOT / "deployment/model_server/policy_wrapper.py"
    spec = importlib.util.spec_from_file_location("policy_wrapper_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def unapply_actions(self, values):
        self.calls.append(np.asarray(values).copy())
        return np.asarray(values) + 10.0


def test_l1_output_unapplies_actions_per_batch_item():
    processor = FakeProcessor()
    normalized = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

    actions = postprocess_actions(normalized, processor, "l1")

    assert np.allclose(actions, normalized + 10.0)
    assert len(processor.calls) == 2
    assert np.allclose(processor.calls[0], normalized[0])
    assert np.allclose(processor.calls[1], normalized[1])


def test_ce_output_returns_logits_without_unapply():
    processor = FakeProcessor()
    logits = np.asarray([[[0.1, 0.9]]], dtype=np.float32)

    actions = postprocess_actions(logits, processor, "ce")

    assert np.allclose(actions, logits)
    assert processor.calls == []


def test_factorized_ce_output_returns_logits_without_unapply():
    processor = FakeProcessor()
    logits = np.asarray([[[0.0, 1.0, 0.2, 0.4, -0.5, 0.8, 0.1, 0.3, 0.6, -0.1, 1.2]]], dtype=np.float32)

    actions = postprocess_actions(logits, processor, "factorized_ce")

    assert np.allclose(actions, logits)
    assert processor.calls == []


def test_policy_wrapper_returns_normalized_and_postprocessed_actions(monkeypatch):
    policy_wrapper_module = _load_policy_wrapper_module()

    class FakeFramework:
        def predict_action(self, examples, **kwargs):
            return {"normalized_actions": np.asarray([[[1.0, 2.0]]], dtype=np.float32)}

    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework()
    wrapper._default_unnorm_key = "demon_attack_train__bridge"
    wrapper._available_unnorm_keys = ["demon_attack_train__bridge"]
    wrapper._action_loss_type = "l1"
    monkeypatch.setattr(wrapper, "_get_processor", lambda unnorm_key: FakeProcessor())

    prediction = wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="demon_attack_train__bridge")

    assert np.allclose(prediction["normalized_actions"], [[[1.0, 2.0]]])
    assert np.allclose(prediction["actions"], [[[11.0, 12.0]]])
    assert prediction["action_output_type"] == policy_wrapper_module.ACTION_OUTPUT_TYPES["l1"]


def test_policy_wrapper_requires_normalized_actions(monkeypatch):
    policy_wrapper_module = _load_policy_wrapper_module()

    class FakeFramework:
        def predict_action(self, examples, **kwargs):
            return {"actions": np.asarray([[[1.0, 2.0]]], dtype=np.float32)}

    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework()
    wrapper._default_unnorm_key = "demon_attack_train__bridge"
    wrapper._available_unnorm_keys = ["demon_attack_train__bridge"]
    wrapper._action_loss_type = "l1"
    monkeypatch.setattr(wrapper, "_get_processor", lambda unnorm_key: FakeProcessor())

    with pytest.raises(KeyError, match="normalized_actions"):
        wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="demon_attack_train__bridge")
