import sys
from pathlib import Path
from types import SimpleNamespace


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def _cfg(*, task: str, action_layout: str, model_alias: str = "openvla"):
    return SimpleNamespace(
        rl_games=SimpleNamespace(task=task, model_alias=model_alias),
        framework=SimpleNamespace(
            action_model=SimpleNamespace(action_layout=action_layout, action_dim=2, action_env_dim=2)
        ),
    )


def test_action_spec_maps_flappy_categorical_layout_to_two_dimensional_head():
    from starVLA.training.rl_games.action_spec import apply_action_spec

    cfg = _cfg(task="flappy", action_layout="flappy_categorical_2")

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 2
    assert cfg.framework.action_model.action_env_dim == 2


def test_action_spec_maps_demon_attack_categorical_layout_to_six_dimensional_head():
    from starVLA.training.rl_games.action_spec import apply_action_spec

    cfg = _cfg(task="demon_attack", action_layout="demon_attack_categorical_6")

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 6
    assert cfg.framework.action_model.action_env_dim == 6


def test_action_spec_maps_deadly_corridor_factorized_layout_to_eleven_dimensional_head():
    from starVLA.training.rl_games.action_spec import apply_action_spec

    cfg = _cfg(task="deadly_corridor", action_layout="deadly_corridor_factorized_11")

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 11
    assert cfg.framework.action_model.action_env_dim == 11


def test_action_spec_maps_deadly_corridor_joint_layout_to_fifty_four_dimensional_head():
    from starVLA.training.rl_games.action_spec import apply_action_spec

    cfg = _cfg(task="deadly_corridor", action_layout="deadly_corridor_joint_54")

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 54
    assert cfg.framework.action_model.action_env_dim == 54
