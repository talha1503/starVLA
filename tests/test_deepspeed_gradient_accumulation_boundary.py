from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _ForbiddenAccelerator:
    is_local_main_process = False
    sync_gradients = False

    def __init__(self):
        from accelerate.utils import DistributedType

        self.distributed_type = DistributedType.DEEPSPEED

    def accumulate(self, model):
        raise AssertionError("DeepSpeed path must not use accelerator.accumulate")

    def backward(self, loss):
        raise AssertionError("DeepSpeed path must call model.backward directly")


class _Scheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _DeepSpeedLikeModel:
    def __init__(self, boundary_every):
        self.boundary_every = boundary_every
        self.forward_calls = 0
        self.backward_calls = 0
        self.step_calls = 0

    def forward(self, batch):
        import torch

        self.forward_calls += 1
        return {"action_loss": torch.tensor(float(self.forward_calls))}

    def backward(self, loss):
        self.backward_calls += 1

    def is_gradient_accumulation_boundary(self):
        return self.backward_calls % self.boundary_every == 0

    def step(self):
        self.step_calls += 1


def _trainer(boundary_every=4):
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = _DeepSpeedLikeModel(boundary_every)
    trainer.accelerator = _ForbiddenAccelerator()
    trainer.lr_scheduler = _Scheduler()
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(
        framework=SimpleNamespace(kv_memory=SimpleNamespace(enabled=False, train_rebatch=False)),
        trainer=SimpleNamespace(gradient_clipping=None),
    )
    trainer._profile_timing_should_log = lambda *args, **kwargs: False
    return trainer


def test_deepspeed_engine_owns_accumulation_boundary():
    from starVLA.training import train_starvla

    original_autocast = train_starvla.torch.autocast
    train_starvla.torch.autocast = lambda *args, **kwargs: nullcontext()
    try:
        trainer = _trainer(boundary_every=4)
        metrics = [trainer._train_step({"batch": idx}) for idx in range(4)]
    finally:
        train_starvla.torch.autocast = original_autocast

    assert [item["_optimizer_step"] for item in metrics] == [False, False, False, True]
    assert "train/loss" not in metrics[0]
    assert metrics[3]["train/loss"] == 4.0
    assert trainer.model.forward_calls == 4
    assert trainer.model.backward_calls == 4
    assert trainer.model.step_calls == 4
    assert trainer.lr_scheduler.steps == 1


if __name__ == "__main__":
    test_deepspeed_engine_owns_accumulation_boundary()
