from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ("openvla", "pi0", "pi05", "gr00t")
ENVS = ("flappy", "demon_attack", "deadly_corridor")


def _command_path(model: str, env: str) -> Path:
    return REPO_ROOT / "commands" / f"train_{env}_{model}.sh"


def _experiment_path(model: str, env: str) -> str:
    return f"examples/rl_games/experiments/{model}/bridge/single/{env}.yaml"


def test_training_command_matrix_targets_existing_bridge_single_configs() -> None:
    for model in MODELS:
        for env in ENVS:
            command_path = _command_path(model, env)
            experiment_path = _experiment_path(model, env)

            assert command_path.exists(), f"Missing command wrapper: {command_path}"
            assert (REPO_ROOT / experiment_path).exists(), f"Missing experiment config: {experiment_path}"

            command_text = command_path.read_text(encoding="utf-8")
            assert experiment_path in command_text
            assert f"conda.env_name=starvla_{model}" in command_text
            assert 'wandb.entity="$WANDB_ENTITY"' in command_text
            assert "rl_games.post_train_eval.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]" in command_text


def test_training_commands_are_valid_bash() -> None:
    command_paths = [str(_command_path(model, env)) for model in MODELS for env in ENVS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)
