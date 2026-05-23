from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from accelerate.utils import DistributedType
import pytest
import torch
from torch import nn

from starVLA.model.framework.peft_checkpoint import ACTION_MODEL_PT, load_lora_adapter_checkpoint, save_lora_adapter_checkpoint
from starVLA.training.rl_games.checkpoint_sync import CheckpointSyncManager
from starVLA.training.train_starvla import VLATrainer


def test_local_keep_last_n_prunes_old_files(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "local": {"keep_last_n": 2},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            }
        }
    )
    manager = CheckpointSyncManager(cfg=cfg)

    files = []
    for step in [1, 2, 3]:
        path = tmp_path / f"steps_{step}_pytorch_model.pt"
        path.write_text("x", encoding="utf-8")
        files.append(path)
        manager.register_local_checkpoint(step=step, model_path=str(path))

    assert not files[0].exists()
    assert files[1].exists()
    assert files[2].exists()


def test_local_keep_last_n_prunes_old_lora_adapter_dirs(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "local": {"keep_last_n": 1},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            }
        }
    )
    manager = CheckpointSyncManager(cfg=cfg)

    old_dir = tmp_path / "steps_1_lora_adapter"
    old_dir.mkdir()
    (old_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    new_dir = tmp_path / "steps_2_lora_adapter"
    new_dir.mkdir()
    (new_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    manager.register_local_checkpoint(step=1, model_path=str(old_dir))
    manager.register_local_checkpoint(step=2, model_path=str(new_dir))

    assert not old_dir.exists()
    assert new_dir.exists()


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)


class _TinyLoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        peft = pytest.importorskip("peft")
        self.qwen_vl_interface = SimpleNamespace(
            model=peft.get_peft_model(
                _TinyBackbone(),
                peft.LoraConfig(r=1, lora_alpha=1, target_modules=["linear"], bias="none"),
            )
        )
        self.action_model = nn.Linear(2, 1)


def _prefixed_state_dict(model):
    return {f"qwen_vl_interface.model.{name}": value.detach().clone() for name, value in model.qwen_vl_interface.model.state_dict().items()} | {
        f"action_model.{name}": value.detach().clone() for name, value in model.action_model.state_dict().items()
    }


def test_lora_adapter_checkpoint_uses_supplied_full_state_dict(tmp_path: Path):
    source = _TinyLoraModel()
    for param in source.parameters():
        param.data.fill_(0.25)
    gathered_state = _prefixed_state_dict(source)

    for param in source.parameters():
        param.data.fill_(0.75)

    save_lora_adapter_checkpoint(source, tmp_path, "pt", model_state_dict=gathered_state)

    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / ACTION_MODEL_PT).exists()

    loaded = _TinyLoraModel()
    load_lora_adapter_checkpoint(loaded, tmp_path)

    assert all(
        torch.equal(actual, gathered_state[f"action_model.{name}"])
        for name, actual in loaded.action_model.state_dict().items()
    )
    loaded_lora = loaded.qwen_vl_interface.model.state_dict()
    assert all(
        torch.equal(loaded_lora[name], value)
        for name, value in {
            name: gathered_state[f"qwen_vl_interface.model.{name}"]
            for name in loaded_lora
            if "lora_" in name
        }.items()
        if "lora_" in name
    )


class _FakeAccelerator:
    def __init__(self, *, is_main_process: bool, stage: int):
        self.is_main_process = is_main_process
        self.distributed_type = DistributedType.DEEPSPEED
        self.deepspeed_config = {"zero_optimization": {"stage": stage}}
        self.calls = 0

    def get_state_dict(self, model):
        self.calls += 1
        return {"value": torch.tensor([float(self.calls)])} if self.is_main_process else None


def test_checkpoint_state_dict_calls_accelerator_on_all_zero3_ranks():
    trainer = object.__new__(VLATrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.accelerator = _FakeAccelerator(is_main_process=False, stage=3)

    assert trainer._model_state_dict_for_checkpoint() is None
    assert trainer.accelerator.calls == 1


def test_checkpoint_state_dict_skips_non_main_non_zero3_ranks():
    trainer = object.__new__(VLATrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.accelerator = _FakeAccelerator(is_main_process=False, stage=2)

    assert trainer._model_state_dict_for_checkpoint() is None
    assert trainer.accelerator.calls == 0
