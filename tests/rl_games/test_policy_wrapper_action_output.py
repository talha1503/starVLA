import sys
import importlib.util
from pathlib import Path

import numpy as np


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
