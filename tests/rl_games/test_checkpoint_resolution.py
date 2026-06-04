from pathlib import Path

import pytest
import torch
from torch import nn

from starVLA.model.framework import base_framework
from starVLA.model.framework.share_tools import resolve_checkpoint_path


def test_resolve_checkpoint_path_accepts_full_model_files(tmp_path: Path):
    for filename in ("pytorch_model.pt", "pytorch_model.pth", "model.safetensors"):
        checkpoint_path = tmp_path / filename
        checkpoint_path.write_text("weights", encoding="utf-8")

        assert resolve_checkpoint_path(checkpoint_path) == checkpoint_path


def test_resolve_checkpoint_path_keeps_lora_adapter_dir(tmp_path: Path):
    final_model = tmp_path / "run" / "final_model"
    final_model.mkdir(parents=True)
    (final_model / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert resolve_checkpoint_path(final_model) == final_model


def test_resolve_checkpoint_path_finds_full_final_model_file(tmp_path: Path):
    final_model = tmp_path / "run" / "final_model"
    final_model.mkdir(parents=True)
    weights_path = final_model / "model.safetensors"
    weights_path.write_text("weights", encoding="utf-8")

    assert resolve_checkpoint_path(final_model) == weights_path


def test_resolve_checkpoint_path_rejects_ambiguous_full_final_model(tmp_path: Path):
    final_model = tmp_path / "run" / "final_model"
    final_model.mkdir(parents=True)
    (final_model / "pytorch_model.pt").write_text("weights", encoding="utf-8")
    (final_model / "model.safetensors").write_text("weights", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Ambiguous full-model checkpoint files"):
        resolve_checkpoint_path(final_model)


def test_from_pretrained_loads_resolved_final_model_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    final_model = tmp_path / "run" / "final_model"
    final_model.mkdir(parents=True)
    weights_path = final_model / "pytorch_model.pth"
    weights_path.write_bytes(b"weights")

    model = nn.Linear(1, 1)
    expected_state = {name: torch.full_like(value, 0.5) for name, value in model.state_dict().items()}
    loaded_paths = []

    monkeypatch.setattr(
        base_framework,
        "read_mode_config",
        lambda checkpoint_path: ({"trainer": {}, "framework": {"name": "fake"}}, {}),
    )
    monkeypatch.setattr(base_framework, "build_framework", lambda cfg: model)

    def fake_torch_load(checkpoint_path, map_location):
        loaded_paths.append(Path(checkpoint_path))
        return expected_state

    monkeypatch.setattr(base_framework.torch, "load", fake_torch_load)

    loaded_model = base_framework.baseframework.from_pretrained(final_model)

    assert loaded_model is model
    assert loaded_paths == [weights_path]
    assert all(torch.equal(value, expected_state[name]) for name, value in model.state_dict().items())
