from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT


@pytest.mark.parametrize("loss_type", ["mse", "l2"])
def test_qwen_oft_mse_loss_uses_only_active_action_dimensions(loss_type: str) -> None:
    model = Qwenvl_OFT.__new__(Qwenvl_OFT)
    torch.nn.Module.__init__(model)
    model.action_env_dim = 3
    prediction = torch.tensor([[[0.5, -0.5, 0.25, 99.0]]])
    target = torch.tensor([[[1.0, -1.0, 0.0, -99.0]]])

    loss = model._compute_action_loss_for_type(
        prediction,
        target,
        loss_type=loss_type,
    )

    assert loss == pytest.approx(F.mse_loss(prediction[..., :3], target[..., :3]).item())
