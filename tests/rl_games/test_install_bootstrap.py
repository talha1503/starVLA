from pathlib import Path


def test_bootstrap_does_not_install_latency_bench_eval_extra_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bootstrap_script = repo_root / "examples" / "rl_games" / "install" / "bootstrap.sh"
    bootstrap_source = bootstrap_script.read_text()

    assert "Installing latency-bench eval extra dependencies" not in bootstrap_source
    assert "eval_extra.sh" not in bootstrap_source
