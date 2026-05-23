from __future__ import annotations

from typing import Dict


MODEL_ALIAS_TO_FRAMEWORK: Dict[str, str] = {
    "openvla": "QwenOFT",
    "pi-0": "QwenPI",
    "gr00t": "QwenGR00T",
}

BRIDGE_INIT_MODES = {"bridge", "pre-trained", "pretrained"}


def apply_model_alias(cfg) -> None:
    """Resolve rl_games.model_alias -> framework.name in-place if provided."""
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return

    model_alias = getattr(rl_games, "model_alias", None)
    if not model_alias:
        return

    if model_alias not in MODEL_ALIAS_TO_FRAMEWORK:
        valid_aliases = ", ".join(sorted(MODEL_ALIAS_TO_FRAMEWORK))
        raise ValueError(f"Unknown rl_games.model_alias={model_alias!r}. Valid aliases: {valid_aliases}")

    framework_name = MODEL_ALIAS_TO_FRAMEWORK[model_alias]
    init_mode = str(getattr(rl_games, "initialization_mode", "") or "").lower()
    action_carrier = str(getattr(rl_games, "action_carrier", "") or "").lower()
    if model_alias == "pi-0" and (init_mode in BRIDGE_INIT_MODES or action_carrier == "bridge"):
        framework_name = "QwenPI_v3"

    cfg.framework.name = framework_name
