import torch
import torch.nn.functional as F


def ce_action_loss(
    pred_actions: torch.Tensor,
    actions_target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    class_ids = torch.argmax(actions_target, dim=-1)
    return F.cross_entropy(
        pred_actions.reshape(-1, pred_actions.shape[-1]).float(),
        class_ids.reshape(-1),
        label_smoothing=label_smoothing,
    )


def deadly_factorized_ce_action_loss(
    pred_actions: torch.Tensor,
    actions_target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    turn_loss = ce_action_loss(
        pred_actions[..., 0:3],
        actions_target[..., 0:3],
        label_smoothing=label_smoothing,
    )
    move_loss = ce_action_loss(
        pred_actions[..., 3:6],
        actions_target[..., 3:6],
        label_smoothing=label_smoothing,
    )
    strafe_loss = ce_action_loss(
        pred_actions[..., 6:9],
        actions_target[..., 6:9],
        label_smoothing=label_smoothing,
    )
    attack_loss = ce_action_loss(
        pred_actions[..., 9:11],
        actions_target[..., 9:11],
        label_smoothing=label_smoothing,
    )
    return turn_loss + move_loss + strafe_loss + attack_loss


def qwen_oft_action_loss(
    pred_actions: torch.Tensor,
    actions_target: torch.Tensor,
    loss_type: str,
    l1_loss,
    ce_label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss_fns = {
        "l1": lambda: l1_loss(pred_actions, actions_target),
        "ce": lambda: ce_action_loss(pred_actions, actions_target, label_smoothing=ce_label_smoothing),
        "factorized_ce": lambda: deadly_factorized_ce_action_loss(
            pred_actions,
            actions_target,
            label_smoothing=ce_label_smoothing,
        ),
    }
    return loss_fns[loss_type]()
