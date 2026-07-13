from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from starVLA.training.train_starvla import VLATrainer


class _FakeAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0

    def unwrap_model(self, model: "_FakeModel") -> "_FakeModel":
        return model


class _FakeModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class _FakeRunner:
    def __init__(self) -> None:
        self.model_training_during_run: bool | None = None

    def run(
        self,
        model: _FakeModel,
        step: int,
        stage: str,
    ) -> SimpleNamespace:
        self.model_training_during_run = bool(model.training)
        return SimpleNamespace(
            per_latency={},
            aggregate={
                "stage": stage,
                "step": step,
                "mean_reward": 0.0,
            },
            path=None,
        )


def test_rl_games_eval_runs_in_eval_mode_and_restores_training_mode() -> None:
    trainer = object.__new__(VLATrainer)
    model = _FakeModel()
    runner = _FakeRunner()
    trainer.model = model
    trainer.accelerator = _FakeAccelerator()
    trainer._rl_games_eval_runner = runner
    trainer.completed_steps = 7

    result = trainer._run_rl_games_eval_with_model_mode(stage="mid_train")

    assert result.aggregate["stage"] == "mid_train"
    assert runner.model_training_during_run is False
    assert model.training is True
