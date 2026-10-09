"""Launch mixed-latency MIKASA H1 training using editable dot-list overrides."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.MIKASA.scripts.prepare_mixed_latency_h1 import main as prepare  # noqa: E402


def main(argv=None) -> None:
    base = OmegaConf.load(ROOT / "examples/MIKASA/train_files/mikasa_mixed_01234_h1.yaml")
    runtime = {
        "model": "openvla",
        "env": "mikasa",
        "init": "scratch",
        "mode": "mixed",
        "workspace_dir": str(ROOT.parent),
        "dataset": {
            "source_hf": "latency-sensitive-bench/benchmark-datasets",
            "revision": "main",
            "local_source_root": None,
            "converted_name": "${run_id}",
            "latency_filter": [0, 1, 2, 3, 4],
            "episodes_per_latency": 250,
            "validation_episodes_per_latency": 25,
        },
        "paths": {"base_model_dir": None, "dataset_local_dir": str(ROOT / "playground/Datasets")},
        "launch": {"num_processes": 1, "dry_run": False},
        "mikasa": {
            "benchmark_root": "${workspace_dir}/latency-aware-agents",
            "task_python": "${mikasa.benchmark_root}/third_party/MIKASA-Robo/.venv/bin/python",
        },
        "rl_games": {
            "env_eval": {
                "eval_backend": "latency_bench",
                "mid_train": {"enabled": False},
                "post_train": {
                    "enabled": True,
                    "latencies": [0, 1, 2, 3, 4],
                    "num_episodes": 200,
                    "max_steps_per_episode": 60,
                },
            }
        },
        "checkpoint": {
            "hf_repo_id": "talha15032/${run_id}",
            "load": "none",
            "sync": {"repo_id": "${checkpoint.hf_repo_id}"},
        },
    }
    cfg = OmegaConf.merge(base, runtime, OmegaConf.from_dotlist(sys.argv[1:] if argv is None else argv))
    if (cfg.model, cfg.env, cfg.init, cfg.mode) != ("openvla", "mikasa", "scratch", "mixed"):
        raise ValueError("This launcher uses the paper's OpenVLA/QwenOFT MIKASA H1 scratch-initialized mixed recipe")
    if cfg.trainer.distributed_backend != "none" or cfg.launch.num_processes != 1:
        raise ValueError("This launcher runs directly on one GPU with trainer.distributed_backend=none")
    global_batch = (
        cfg.launch.num_processes * cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps
    )
    if global_batch != 128:
        raise ValueError("The paper's recipe requires GPU count x microbatch x accumulation = 128")
    os.chdir(ROOT)
    benchmark = Path(cfg.mikasa.benchmark_root).expanduser().resolve()
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(benchmark), str(benchmark / "third_party/MIKASA-Robo")])
    os.environ["WANDB_ENTITY"] = cfg.wandb_entity
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    assets = Path(cfg.paths.dataset_local_dir).expanduser().resolve() / cfg.dataset.converted_name
    run_root = Path(cfg.run_root_dir).expanduser().resolve()
    post = cfg.rl_games.env_eval.post_train
    command = [
        "--output",
        str(assets),
        "--run-id",
        cfg.run_id,
        "--run-root",
        str(run_root),
        "--dataset-hf-repo",
        cfg.dataset.source_hf,
        "--dataset-revision",
        cfg.dataset.revision,
        "--latencies",
        *[str(latency) for latency in cfg.dataset.latency_filter],
        "--train-episodes",
        str(cfg.dataset.episodes_per_latency),
        "--val-episodes",
        str(cfg.dataset.validation_episodes_per_latency),
        "--num-gpus",
        str(cfg.launch.num_processes),
        "--micro-batch",
        str(cfg.datasets.vla_data.per_device_batch_size),
        "--prompt-mode",
        cfg.datasets.vla_data.prompt_mode,
        "--model-hf-repo",
        cfg.checkpoint.sync.repo_id,
        "--wandb-entity",
        cfg.wandb_entity,
        "--wandb-project",
        cfg.wandb_project,
        "--vla-python",
        sys.executable,
        "--eval-episodes",
        str(post.num_episodes),
    ]
    if cfg.dataset.local_source_root is not None:
        command.extend(["--source-root", cfg.dataset.local_source_root])
    if cfg.paths.base_model_dir is not None:
        command.extend(["--backbone-path", cfg.paths.base_model_dir])
    if cfg.framework.qwenvl.enable_gradient_checkpointing:
        command.append("--gradient-checkpointing")
    prepare(command)

    generated = yaml.safe_load((assets / "training.yaml").read_text())
    # Native evaluation is owned by this launcher; keep RL-games hooks out of the trainer.
    train = OmegaConf.to_container(cfg, resolve=True)
    for key in ("model", "env", "init", "mode", "paths", "launch", "dataset", "mikasa", "rl_games"):
        del train[key]
    train["framework"]["qwenvl"]["base_vlm"] = generated["framework"]["qwenvl"]["base_vlm"]
    for key in ("task_contract_path", "custom_mixtures_path"):
        train["datasets"]["vla_data"][key] = generated["datasets"]["vla_data"][key]
    train["run_root_dir"] = str(run_root)
    config_path = assets / "training.yaml"
    config_path.write_text(yaml.safe_dump(train, sort_keys=False))
    for latency in post.latencies:
        path = assets / f"eval_l{latency}.yaml"
        evaluation = yaml.safe_load(path.read_text())
        evaluation["policy"]["checkpoint_path"] = str(
            run_root / cfg.run_id / f"checkpoints/steps_{cfg.trainer.stop_after_steps}_pytorch_model.pt"
        )
        evaluation["evaluation"]["eval_max_steps"] = post.max_steps_per_episode
        path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
    provenance_path = assets / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["launch_config"] = OmegaConf.to_container(cfg, resolve=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Training config: {config_path}")
    if cfg.launch.dry_run:
        return
    subprocess.run(
        [
            sys.executable,
            "starVLA/training/train_starvla.py",
            "--config_yaml",
            str(config_path),
        ],
        cwd=ROOT,
        check=True,
    )
    trained = run_root / cfg.run_id
    if post.enabled:
        for latency in post.latencies:
            subprocess.run(
                [
                    cfg.mikasa.task_python,
                    str(benchmark / "scripts/mikasa/evaluate.py"),
                    "latency-eval",
                    "--eval-config",
                    str(assets / f"eval_l{latency}.yaml"),
                ],
                check=True,
            )
    if cfg.checkpoint.sync.enabled:
        api = HfApi()
        api.create_repo(repo_id=cfg.checkpoint.sync.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=cfg.checkpoint.sync.repo_id,
            repo_type="model",
            folder_path=str(trained),
            allow_patterns=["*.yaml", "dataset_statistics.json", "post_train_eval/**"],
        )
        api.upload_file(
            repo_id=cfg.checkpoint.sync.repo_id,
            repo_type="model",
            path_or_fileobj=str(provenance_path),
            path_in_repo="provenance.json",
        )


if __name__ == "__main__":
    main()
