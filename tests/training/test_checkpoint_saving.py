from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")

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
