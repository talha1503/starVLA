from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from examples.rl_games.scripts import launch_train


REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ("openvla", "pi0", "pi05", "gr00t")
ENVS = ("flappy", "demon_attack", "deadly_corridor")


def _command_path(model: str, env: str) -> Path:
    return REPO_ROOT / "commands" / f"train_{env}_{model}.sh"


def test_training_command_matrix_targets_hydra_launcher() -> None:
    for model in MODELS:
        for env in ENVS:
            command_path = _command_path(model, env)

            assert command_path.exists(), f"Missing command wrapper: {command_path}"

            command_text = command_path.read_text(encoding="utf-8")
            assert "python examples/rl_games/scripts/launch_train.py" in command_text
            assert "examples/rl_games/experiments/" not in command_text
            assert f"model={model}" in command_text
            assert f"env={env}" in command_text
            assert "init=bridge" in command_text
            assert "mode=single" not in command_text
            assert "WANDB_ENTITY" not in command_text
            assert "wandb_entity=" not in command_text
            assert "rl_games.env_eval.post_train.latencies=" not in command_text
            assert "trainer.batch_size=" not in command_text
            assert "datasets.vla_data.per_device_batch_size=16" in command_text
            assert "dataset.source_hf=data/" not in command_text


def test_training_commands_are_valid_bash() -> None:
    command_paths = [str(_command_path(model, env)) for model in MODELS for env in ENVS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)


def test_launcher_does_not_translate_trainer_batch_size_alias() -> None:
    launcher_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "launch_train.py").read_text(
        encoding="utf-8"
    )

    assert '"trainer.batch_size"' not in launcher_text


def test_launcher_forwards_canonical_per_device_batch_size_override(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["datasets.vla_data.per_device_batch_size=16"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "datasets.vla_data.per_device_batch_size=16" in cmd


def test_launcher_defaults_to_one_last_checkpoint_and_no_pt_file(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "checkpoint.local.keep_last_n=1" in cmd
    assert "checkpoint.save_best_model=true" in cmd
    assert "checkpoint.save_pt_file=false" in cmd


def test_launcher_defaults_workspace_to_repo_root_when_env_is_unset(monkeypatch) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/workspace":
            return True
        return original_exists(path)

    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    monkeypatch.setattr(Path, "exists", fake_exists)

    cfg = OmegaConf.create({"workspace_dir": "WORKSPACE_DIR"})

    assert launch_train._workspace_dir(cfg) == launch_train.REPO_ROOT


def test_vla_trainer_saves_last_checkpoints_independently_from_best_model() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert ") and not self._save_best_model_enabled:" not in trainer_text


def test_vla_trainer_pt_checkpoint_file_is_optional() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "self._save_pt_file_enabled" in trainer_text
    assert "safe_serialization=True" in trainer_text
    assert "self.accelerator.get_state_dict(self.model)" in trainer_text
    assert 'model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"' in trainer_text
    assert "torch.save(state_dict, model_checkpoint_path)" in trainer_text
    assert "model_path=model_checkpoint_path" in trainer_text
