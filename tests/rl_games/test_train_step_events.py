from starVLA.training.train_step_events import should_run_step_interval_event


def test_step_interval_event_waits_for_gradient_sync() -> None:
    assert not should_run_step_interval_event(completed_steps=100, interval=100, gradients_synced=False)


def test_step_interval_event_runs_on_synced_interval_step() -> None:
    assert should_run_step_interval_event(completed_steps=100, interval=100, gradients_synced=True)


def test_step_interval_event_skips_zero_and_non_interval_steps() -> None:
    assert not should_run_step_interval_event(completed_steps=0, interval=100, gradients_synced=True)
    assert not should_run_step_interval_event(completed_steps=99, interval=100, gradients_synced=True)
