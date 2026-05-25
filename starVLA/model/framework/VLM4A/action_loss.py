import torch
import torch.nn.functional as F


def ce_action_loss(pred_actions: torch.Tensor, actions_target: torch.Tensor) -> torch.Tensor:
    class_ids = torch.argmax(actions_target, dim=-1)
    return F.cross_entropy(
        pred_actions.reshape(-1, pred_actions.shape[-1]).float(),
        class_ids.reshape(-1),
    )


def deadly_factorized_ce_action_loss(pred_actions: torch.Tensor, actions_target: torch.Tensor) -> torch.Tensor:
    turn_loss = ce_action_loss(pred_actions[..., 0:3], actions_target[..., 0:3])
    move_loss = ce_action_loss(pred_actions[..., 3:6], actions_target[..., 3:6])
    strafe_loss = ce_action_loss(pred_actions[..., 6:9], actions_target[..., 6:9])
    attack_loss = ce_action_loss(pred_actions[..., 9:11], actions_target[..., 9:11])
    return turn_loss + move_loss + strafe_loss + attack_loss


def qwen_oft_action_loss(
    pred_actions: torch.Tensor,
    actions_target: torch.Tensor,
    loss_type: str,
    l1_loss,
) -> torch.Tensor:
    loss_fns = {
        "l1": lambda: l1_loss(pred_actions, actions_target),
        "ce": lambda: ce_action_loss(pred_actions, actions_target),
        "factorized_ce": lambda: deadly_factorized_ce_action_loss(pred_actions, actions_target),
    }
    return loss_fns[loss_type]()
