import os
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bootstrap_does_not_install_latency_bench_eval_extra_dependencies() -> None:
    repo_root: Path = _repo_root()
    bootstrap_script = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    bootstrap_source: str = bootstrap_script.read_text()

    assert "Installing latency-bench eval extra dependencies" not in bootstrap_source
    assert "eval_extra.sh" not in bootstrap_source


def _help_output(script: Path) -> str:
    return subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_bootstrap_exposes_tiered_single_model_install() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    bootstrap_help: str = _help_output(install_dir / "bootstrap.sh")

    assert "--tier <use|dev>" in bootstrap_help
    assert "openvla|pi0|pi05|gr00t|wan_oft" in bootstrap_help
    assert "default: openvla" in bootstrap_help
    assert "--split-envs" not in bootstrap_help

    rejected_all = subprocess.run(
        [
            "bash",
            str(install_dir / "bootstrap.sh"),
            "--model",
            "all",
            "--current-env",
            "--torch-profile",
            "cpu",
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected_all.returncode != 0
    assert "invalid model 'all'" in rejected_all.stderr


def test_install_stack_remains_a_training_compatibility_entrypoint(tmp_path: Path) -> None:
    repo_root: Path = _repo_root()
    source_install_stack: Path = repo_root / "examples" / "rl_games" / "install" / "install_stack.sh"
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    install_stack = install_dir / "install_stack.sh"
    shutil.copy2(source_install_stack, install_stack)
    fake_bootstrap = install_dir / "bootstrap.sh"
    fake_bootstrap.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_bootstrap.chmod(0o755)

    install_stack_help: str = _help_output(install_stack)
    result = subprocess.run(
        [
            "bash",
            str(install_stack),
            "--no-conda",
            "--accept-rom-license",
            "--torch-profile",
            "cpu",
            "gr00t",
            "cross_task",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Compatibility entrypoint for a full development/training environment" in install_stack_help
    assert "openvla|pi0|pi05|gr00t|wan_oft" in install_stack_help
    assert result.stdout.splitlines() == [
        "--tier",
        "dev",
        "--model",
        "gr00t",
        "--python-version",
        "3.10",
        "--torch-profile",
        "cpu",
        "--current-env",
        "--accept-rom-license",
        "--env",
        "flappy",
        "--env",
        "demon_attack",
    ]


def test_env_all_cannot_hide_an_invalid_repeated_selector() -> None:
    repo_root: Path = _repo_root()
    bootstrap: Path = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    result = subprocess.run(
        [
            "bash",
            str(bootstrap),
            "--env",
            "all",
            "--env",
            "typo",
            "--current-env",
            "--torch-profile",
            "cpu",
            "--skip-validate",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--env all cannot be combined with another --env" in result.stderr


def test_training_dependencies_are_not_in_the_use_manifest() -> None:
    repo_root: Path = _repo_root()
    use_requirements: str = (repo_root / "requirements.txt").read_text()
    dev_requirements: str = (repo_root / "requirements-dev.txt").read_text()

    for dependency in ("datasets", "deepspeed", "hydra-core", "wandb"):
        assert dependency not in use_requirements
        assert dependency in dev_requirements


def test_torch_auto_profile_uses_cpu_without_nvidia_smi() -> None:
    repo_root: Path = _repo_root()
    resolver: Path = repo_root / "examples" / "rl_games" / "install" / "_torch_profile.sh"
    result = subprocess.run(
        ["/bin/bash", "-c", f'source "{resolver}"; resolve_torch_profile auto'],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/nonexistent"},
    )

    assert result.stdout.strip() == "cpu"


def test_torch_auto_profile_does_not_guess_after_query_failure(tmp_path: Path) -> None:
    repo_root: Path = _repo_root()
    resolver: Path = repo_root / "examples" / "rl_games" / "install" / "_torch_profile.sh"
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/bin/sh\nexit 1\n")
    fake_nvidia_smi.chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", "-c", f'source "{resolver}"; resolve_torch_profile auto'],
        capture_output=True,
        text=True,
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
    )

    assert result.returncode != 0
    assert "unable to query GPU compute capability" in result.stderr


def test_wan_oft_install_assets_are_registered() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    model_installer: Path = install_dir / "model" / "wan_oft.sh"
    model_installer_source: str = model_installer.read_text()
    flappy_validator: Path = install_dir / "validate" / "wan_oft_flappy.sh"
    flappy_validator_source: str = flappy_validator.read_text()

    assert model_installer.is_file()
    assert os.access(model_installer, os.X_OK)
    assert "pyarrow" in model_installer_source
    assert "huggingface-hub" in model_installer_source
    assert flappy_validator.is_file()
    assert os.access(flappy_validator, os.X_OK)
    assert "from diffusers import AutoencoderKLWan, WanTransformer3DModel" in flappy_validator_source
    assert "from transformers import UMT5EncoderModel" in flappy_validator_source
