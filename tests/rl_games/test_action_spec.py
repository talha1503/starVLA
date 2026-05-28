import sys
from pathlib import Path
from types import SimpleNamespace


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def test_deadly_corridor_joint_fifty_four_action_spec_sets_openvla_action_dim():
    from starVLA.training.rl_games.action_spec import apply_action_spec

    cfg = SimpleNamespace(
        rl_games=SimpleNamespace(
            task="deadly_corridor",
            model_alias="openvla",
            env_eval=SimpleNamespace(deadly=SimpleNamespace(action_layout="joint_54")),
        ),
        framework=SimpleNamespace(action_model=SimpleNamespace(action_dim=2, action_env_dim=2)),
    )

    apply_action_spec(cfg)

    assert cfg.framework.action_model.action_dim == 54
    assert cfg.framework.action_model.action_env_dim == 54
