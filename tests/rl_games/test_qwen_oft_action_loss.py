import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def _load_action_loss_module():
    module_path = STARVLA_ROOT / "starVLA/model/framework/VLM4A/action_loss.py"
    spec = importlib.util.spec_from_file_location("qwen_oft_action_loss_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qwen_oft_action_loss = _load_action_loss_module().qwen_oft_action_loss


def test_l1_action_loss_matches_torch_l1():
    pred_actions = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]])
    actions_target = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])
    l1_loss = nn.L1Loss()

    loss = qwen_oft_action_loss(pred_actions, actions_target, "l1", l1_loss)

    assert torch.allclose(loss, l1_loss(pred_actions, actions_target))


def test_ce_action_loss_matches_cross_entropy_and_backpropagates():
    pred_actions = torch.tensor(
        [[[0.0, 1.0, -0.5, 0.2, 0.3, -0.1], [1.1, -0.4, 0.5, 0.0, -0.3, 0.8]]],
        requires_grad=True,
    )
    actions_target = torch.tensor(
        [[[-1.0, 1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0]]]
    )

    loss = qwen_oft_action_loss(pred_actions, actions_target, "ce", nn.L1Loss())
    expected = F.cross_entropy(
        pred_actions.reshape(-1, pred_actions.shape[-1]),
        torch.tensor([1, 5]),
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert pred_actions.grad is not None


def test_factorized_ce_action_loss_sums_deadly_corridor_factor_losses():
    pred_actions = torch.tensor(
        [
            [
                [0.0, 2.0, -1.0, 0.5, 1.0, -0.5, 0.1, -0.2, 1.5, -0.3, 0.7],
                [1.2, -0.4, 0.2, -1.0, 0.1, 2.0, 1.1, 0.3, -0.6, 0.8, -0.2],
            ]
        ],
        requires_grad=True,
    )
    actions_target = torch.tensor(
        [
            [
                [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
            ]
        ]
    )

    loss = qwen_oft_action_loss(pred_actions, actions_target, "factorized_ce", nn.L1Loss())
    expected = (
        F.cross_entropy(pred_actions[..., 0:3].reshape(-1, 3), torch.tensor([1, 0]))
        + F.cross_entropy(pred_actions[..., 3:6].reshape(-1, 3), torch.tensor([1, 2]))
        + F.cross_entropy(pred_actions[..., 6:9].reshape(-1, 3), torch.tensor([2, 0]))
        + F.cross_entropy(pred_actions[..., 9:11].reshape(-1, 2), torch.tensor([1, 0]))
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert pred_actions.grad is not None
