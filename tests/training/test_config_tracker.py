from omegaconf import OmegaConf

from starVLA.training.trainer_utils.checkpoint_config import track_vla_checkpoint_config_fields
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig


def test_vla_checkpoint_fields_are_tracked_by_explicit_access():
    cfg = AccessTrackedConfig(
        OmegaConf.create(
            {
                "framework": {
                    "action_model": {
                        "action_dim": 6,
                        "state_dim": 1,
                    },
                },
                "datasets": {
                    "vla_data": {
                        "include_state": False,
                    },
                },
            }
        )
    )
    assert cfg.framework.action_model.action_dim == 6

    track_vla_checkpoint_config_fields(cfg)

    exported = cfg.export_accessed_config(use_original_values=False)

    assert exported["framework"]["action_model"]["action_dim"] == 6
    assert exported["framework"]["action_model"]["state_dim"] == 1
    assert exported["datasets"]["vla_data"]["include_state"] is False
