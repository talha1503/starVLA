from __future__ import annotations


# RL-games discrete action heads use action_layout as the human-facing semantic
# contract and derive action_dim from it. The layout names encode the reason for
# each dimension:
# - flappy_categorical_2: one categorical action over [NOOP, FLAP].
# - demon_attack_categorical_6: one categorical action over the flattened
#   movement/fire product [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
#   rather than a factorized 3+2 head.
# - deadly_corridor_multibinary_7: the legacy direct Doom button vector.
# - deadly_corridor_factorized_11: four grouped categorical heads represented
#   in one vector: turn(3) + move(3) + strafe(3) + attack(2) = 11.
# - deadly_corridor_joint_54: one categorical action over the full product
#   turn(3) * move(3) * strafe(3) * attack(2) = 54.
ACTION_LAYOUT_DIMS = {
    "flappy_categorical_2": 2,
    "demon_attack_categorical_6": 6,
    "deadly_corridor_multibinary_7": 7,
    "deadly_corridor_factorized_11": 11,
    "deadly_corridor_joint_54": 54,
}


def _env_action_dim(cfg):
    layout = cfg.framework.action_model.action_layout
    return ACTION_LAYOUT_DIMS[layout]


def _uses_action_layout(cfg) -> bool:
    return cfg.rl_games.task != "cross_task"


def apply_action_spec(cfg) -> None:
    """Apply env-specific action dimensionality policy for RL-games setups."""
    rl_games = getattr(cfg, "rl_games", None)
    framework = getattr(cfg, "framework", None)
    if rl_games is None or framework is None or not hasattr(framework, "action_model"):
        return

    if not _uses_action_layout(cfg):
        return

    env_dim = _env_action_dim(cfg)
    action_cfg = framework.action_model
    model_alias = rl_games.model_alias

    if model_alias in {"pi-0", "gr00t"}:
        model_action_dim = action_cfg.action_dim
        if model_action_dim < env_dim:
            raise ValueError(
                f"Model action_dim={model_action_dim} is smaller than env action dim={env_dim} "
                f"for model_alias={model_alias}, task={rl_games.task}"
            )
        action_cfg.action_env_dim = env_dim
        return

    # openvla and other dense/discrete heads should emit env action space directly.
    action_cfg.action_dim = env_dim
    action_cfg.action_env_dim = env_dim
