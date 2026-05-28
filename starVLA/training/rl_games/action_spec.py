from __future__ import annotations


def _deadly_action_dim(cfg) -> int:
    deadly_cfg = getattr(getattr(cfg.rl_games, "env_eval", None), "deadly", None)
    layout = str(getattr(deadly_cfg, "action_layout", "multibinary_7"))
    if layout == "multibinary_7":
        return 7
    if layout == "factorized_11":
        return 11
    if layout == "joint_54":
        return 54
    raise ValueError(f"Unsupported deadly action layout: {layout}")


def _env_action_dim(cfg):
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return None
    task = str(getattr(rl_games, "task", "flappy"))
    if task == "flappy":
        return 2
    if task == "demon_attack":
        return 6
    if task == "deadly_corridor":
        return _deadly_action_dim(cfg)
    if task == "cross_task":
        return None
    return None


def apply_action_spec(cfg) -> None:
    """Apply env-specific action dimensionality policy for RL-games setups."""
    rl_games = getattr(cfg, "rl_games", None)
    framework = getattr(cfg, "framework", None)
    if rl_games is None or framework is None or not hasattr(framework, "action_model"):
        return

    env_dim = _env_action_dim(cfg)
    if env_dim is None:
        return

    action_cfg = framework.action_model
    model_alias = str(getattr(rl_games, "model_alias", "openvla"))

    if model_alias in {"pi-0", "gr00t"}:
        model_action_dim = int(getattr(action_cfg, "action_dim", 0))
        if model_action_dim < env_dim:
            raise ValueError(
                f"Model action_dim={model_action_dim} is smaller than env action dim={env_dim} "
                f"for model_alias={model_alias}, task={getattr(rl_games, 'task', None)}"
            )
        action_cfg.action_env_dim = int(env_dim)
        return

    # openvla and other dense/discrete heads should emit env action space directly.
    action_cfg.action_dim = int(env_dim)
    action_cfg.action_env_dim = int(env_dim)
