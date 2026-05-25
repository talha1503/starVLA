from __future__ import annotations


def track_vla_checkpoint_config_fields(cfg) -> None:
    """Mark VLA checkpoint inference fields as part of accessed config."""
    cfg.framework.action_model.state_dim
    cfg.datasets.vla_data.include_state
