def should_run_step_interval_event(completed_steps: int, interval: int, gradients_synced: bool) -> bool:
    if interval <= 0:
        raise ValueError(f"Step interval must be positive, got {interval}.")
    return gradients_synced and completed_steps > 0 and completed_steps % interval == 0
