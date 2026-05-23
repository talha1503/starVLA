__all__ = ["apply_model_alias", "apply_action_spec", "CheckpointSyncManager", "RlGamesEvalRunner"]


def __getattr__(name):
    if name == "apply_model_alias":
        from .alias import apply_model_alias

        return apply_model_alias
    if name == "apply_action_spec":
        from .action_spec import apply_action_spec

        return apply_action_spec
    if name == "CheckpointSyncManager":
        from .checkpoint_sync import CheckpointSyncManager

        return CheckpointSyncManager
    if name == "RlGamesEvalRunner":
        from .eval_core import RlGamesEvalRunner

        return RlGamesEvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
