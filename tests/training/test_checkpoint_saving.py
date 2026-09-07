from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

from starVLA.training import train_starvla
from starVLA.training.train_starvla import VLATrainer


class _FakeAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 1
        self.gradient_accumulation_steps = 1
        self.save_state_calls: list[str] = []
        self.messages: list[str] = []

    def save_state(self, output_dir: str, safe_serialization: bool) -> None:
        self.save_state_calls.append(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def wait_for_everyone(self) -> None:
        return None

    def print(self, message: str) -> None:
        self.messages.append(message)

    def get_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return model.state_dict()


@pytest.mark.parametrize(
    ("wandb_name", "expected"),
    [("mikasa-bootstrap-qwenoft", "mikasa-bootstrap-qwenoft"), (None, "bootstrap")],
)
def test_init_wandb_uses_explicit_name_or_run_id(tmp_path, monkeypatch, wandb_name, expected):
    config = {
        "output_dir": str(tmp_path),
        "run_id": "bootstrap",
        "wandb_project": "starvla_tasks",
        "wandb_entity": None,
    }
    if wandb_name is not None:
        config["wandb_name"] = wandb_name
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create(config)
    trainer.accelerator = _FakeAccelerator()
    init_calls = []
    monkeypatch.setattr(train_starvla.wandb, "init", lambda **kwargs: init_calls.append(kwargs))

    trainer._init_wandb()

    assert init_calls[0]["name"] == expected


def test_model_only_checkpoint_does_not_save_full_training_state(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "datasets": {"vla_data": {"per_device_batch_size": 1}},
            "checkpoint": {
                "save_best_model": False,
                "save_final_model": True,
                "save_pt_file": True,
                "save_training_state": False,
                "local": {"keep_last_n": 0},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            },
        }
    )
    model = torch.nn.Linear(2, 2, bias=False)
    accelerator = _FakeAccelerator()
    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=None,
        vla_eval_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        accelerator=accelerator,
    )
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(trainer.checkpoint_dir).mkdir()
    trainer.completed_steps = 400

    trainer._save_checkpoint()

    model_path = Path(trainer.checkpoint_dir) / "steps_400_pytorch_model.pt"
    state_path = Path(trainer.checkpoint_dir) / "steps_400_state"
    summary_path = tmp_path / "summary.jsonl"

    assert accelerator.save_state_calls == []
    assert model_path.exists()
    assert not state_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8").strip()) == {"steps": 400}


def test_round_resume_restores_optimizer_and_lr_schedule(tmp_path, monkeypatch):
    from accelerate import Accelerator

    monkeypatch.setattr(VLATrainer, '_init_wandb', lambda self: None)

    def make_trainer(output, checkpoint=None):
        output.mkdir()
        cfg = OmegaConf.create({
            'output_dir': str(output), 'seed': 0,
            'datasets': {'vla_data': {'per_device_batch_size': 1}},
            'trainer': {'stop_after_steps': 2 if checkpoint is None else 4,
                        'max_train_steps': 4, 'is_resume': checkpoint is not None,
                        'pretrained_checkpoint': None if checkpoint is None else str(checkpoint),
                        'freeze_modules': '', 'freeze_vit': False,
                        'freeze_tied_embedding': False, 'freeze_llm_layers': []},
            'checkpoint': {'save_best_model': False, 'save_final_model': True,
                           'save_pt_file': True, 'save_training_state': True,
                           'local': {'keep_last_n': 0},
                           'sync': {'enabled': False, 'repo_id': None, 'keep_last_n': 0}},
        })
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 5)
        trainer = VLATrainer(cfg, model, None, None, optimizer, scheduler, Accelerator(cpu=True))
        trainer.prepare_training()
        return trainer

    def update(trainer):
        trainer.optimizer.zero_grad()
        trainer.accelerator.backward(trainer.model(torch.ones(1, 2)).square().sum())
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.completed_steps += 1

    original = make_trainer(tmp_path / 'original')
    update(original)
    update(original)
    original._save_checkpoint()
    resumed = make_trainer(tmp_path / 'resumed',
                           Path(original.checkpoint_dir) / 'steps_2_state')
    assert resumed.completed_steps == 2
    assert resumed.lr_scheduler.last_epoch == original.lr_scheduler.last_epoch
    assert resumed.optimizer.param_groups[0]['lr'] == original.optimizer.param_groups[0]['lr']
    update(original)
    update(resumed)
    for expected, actual in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert resumed.lr_scheduler.get_last_lr() == original.lr_scheduler.get_last_lr()
