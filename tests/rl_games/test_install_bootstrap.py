import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bootstrap_does_not_install_latency_bench_eval_extra_dependencies() -> None:
    repo_root: Path = _repo_root()
    bootstrap_script = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    bootstrap_source: str = bootstrap_script.read_text()

    assert "Installing latency-bench eval extra dependencies" not in bootstrap_source
    assert "eval_extra.sh" not in bootstrap_source


def test_bootstrap_and_install_stack_accept_wan_oft_target() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    bootstrap_source: str = (install_dir / "bootstrap.sh").read_text()
    install_stack_source: str = (install_dir / "install_stack.sh").read_text()

    assert "openvla|pi0|pi05|gr00t|wan_oft|all" in bootstrap_source
    assert "MODELS=(openvla pi0 pi05 gr00t wan_oft)" in bootstrap_source
    assert "openvla|pi0|pi05|gr00t|wan_oft" in install_stack_source


def test_wan_oft_install_assets_are_registered() -> None:
    repo_root: Path = _repo_root()
    install_dir: Path = repo_root / "examples" / "rl_games" / "install"
    model_installer: Path = install_dir / "model" / "wan_oft.sh"
    flappy_validator: Path = install_dir / "validate" / "wan_oft_flappy.sh"
    flappy_validator_source: str = flappy_validator.read_text()

    assert model_installer.is_file()
    assert os.access(model_installer, os.X_OK)
    assert flappy_validator.is_file()
    assert os.access(flappy_validator, os.X_OK)
    assert "from diffusers import AutoencoderKLWan, WanTransformer3DModel" in flappy_validator_source
    assert "from transformers import UMT5EncoderModel" in flappy_validator_source
