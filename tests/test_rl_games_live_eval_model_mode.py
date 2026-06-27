from types import SimpleNamespace


class _ModeTrackingModel:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class _Accelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def __init__(self):
        self.wait_calls = 0

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        self.wait_calls += 1


class _LiveEvalRunner:
    def __init__(self, *, raises=False):
        self.raises = raises
        self.model_modes = []

    def run(self, *, model, step, stage, **kwargs):
        self.model_modes.append(model.training)
        if self.raises:
            raise RuntimeError("eval failed")
        from starVLA.training.rl_games.eval_core import EvalResult

        return EvalResult(per_latency={}, aggregate={"mean_reward": 3.0}, path=f"/tmp/{stage}-{step}.json")

    def merge_results(self, partial_results, *, step, stage):
        from starVLA.training.rl_games.eval_core import EvalResult

        return EvalResult(per_latency={}, aggregate={"mean_reward": 4.0}, path=f"/tmp/merged-{stage}-{step}.json")

    def save(self, *, result, step, stage):
        self.saved = (result, step, stage)


def _trainer(runner):
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = _ModeTrackingModel()
    trainer.accelerator = _Accelerator()
    trainer.completed_steps = 250
    trainer._rl_games_eval_runner = runner
    trainer.config = SimpleNamespace(
        rl_games=SimpleNamespace(
            env_eval=SimpleNamespace(distributed_mode="none"),
        ),
    )
    return trainer


def test_live_rl_games_eval_runs_with_model_in_eval_mode_and_restores_train_mode():
    runner = _LiveEvalRunner()
    trainer = _trainer(runner)

    result = trainer._run_rl_games_eval_with_model_mode(stage="mid_train")

    assert runner.model_modes == [False]
    assert result.aggregate["mean_reward"] == 3.0
    assert trainer.model.training is True


def test_live_rl_games_eval_restores_train_mode_when_runner_raises():
    runner = _LiveEvalRunner(raises=True)
    trainer = _trainer(runner)

    try:
        trainer._run_rl_games_eval_with_model_mode(stage="mid_train")
    except RuntimeError as exc:
        assert str(exc) == "eval failed"

    assert runner.model_modes == [False]
    assert trainer.model.training is True

