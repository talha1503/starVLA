"""Prepare the five successful H1 datasets and the Figure 2 QwenOFT training recipe."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, snapshot_download

STARVLA_ROOT = Path(__file__).resolve().parents[3]
LATENCIES = (0, 1, 2, 3, 4)
BASE_PROMPT = "Intercept the rolling ball and grasp it to stop it."
BACKBONE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BACKBONE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASELINE_REVISION = "56b2c095d76ea5b10dd304e6e03c2ca81801630b"
BASELINE_PATH = (
    "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/"
    "mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z"
)


def latency_prompt(latency: int) -> str:
    return (
        f"{BASE_PROMPT} Current action latency is {latency} raw frames ({latency * 50.0:.2f} ms). "
        "The environment runs at 20 FPS and observations are emitted at 20 FPS. Choose the best next action."
    )


def _statistics(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def select_episode_subset(target: Path, count: int) -> list[int]:
    """Keep the first complete LeRobot v3 episodes and their original video offsets."""
    episode_paths = sorted((target / "meta/episodes").glob("*/*.parquet"))
    episodes = pd.concat([pd.read_parquet(path) for path in episode_paths]).sort_values("episode_index")
    selected = episodes.iloc[:count]
    episode_ids = selected["episode_index"].astype(int).tolist()
    for paths in (episode_paths, sorted((target / "data").glob("*/*.parquet"))):
        for path in paths:
            table = pq.read_table(path)
            selected_rows = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episode_ids)))
            if not selected_rows.num_rows:
                path.unlink()
            else:
                pq.write_table(selected_rows, path)
    info_path = target / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = count
    info["total_frames"] = int(selected["length"].sum())
    info["splits"] = {"train": f"0:{count}"}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return episode_ids


def prepare_datasets(
    sources: dict[int, Path], output: Path, *, latencies=LATENCIES, train_episodes=250, val_episodes=25
) -> dict:
    """Select packaged episodes; freeze normalization using selected training frames only."""
    train_arrays = {"observation.state": [], "action": []}
    mixture = {"train": [], "validation": []}
    splits = []
    selections = {}
    for latency in latencies:
        for split, directory, episodes in (
            ("train", "lerobot", train_episodes),
            ("validation", "lerobot__val", val_episodes),
        ):
            source = sources[latency] / directory
            info = json.loads((source / "meta/info.json").read_text())
            if not 0 < episodes <= info["total_episodes"]:
                raise ValueError(f"L{latency} {split} requests {episodes} episodes; source has {info['total_episodes']}")
            contract = json.loads((source / "task_contract.json").read_text())
            if (contract["action_horizon"], contract["state_dim"], contract["action_dim"]) != (1, 7, 7):
                raise ValueError(f"L{latency} {split} must use the 7D MIKASA H1 contract")
            target = output / f"fixed_l{latency}" / directory
            if episodes < info["total_episodes"] and target.exists():
                raise ValueError("Episode subsets require a fresh dataset.converted_name to preserve existing runs")
            shutil.copytree(source, target, dirs_exist_ok=True)
            if episodes < info["total_episodes"]:
                episode_ids = select_episode_subset(target, episodes)
            else:
                episode_ids = sorted(
                    int(index)
                    for path in (target / "meta/episodes").glob("*/*.parquet")
                    for index in pd.read_parquet(path)["episode_index"]
                )
            selections.setdefault(str(latency), {})[split] = episode_ids
            tasks_path = target / "meta/tasks.parquet"
            tasks = pd.read_parquet(tasks_path)
            # LeRobot v3 stores instruction strings in the dataframe index.
            tasks.index = pd.Index([latency_prompt(latency) for _ in tasks.index], name=tasks.index.name)
            tasks.to_parquet(tasks_path)
            contract["prompt"] = BASE_PROMPT
            splits.append((target, contract))
            mixture[split].append([str(target.resolve()), 1.0, contract["robot_type"]])
            if split == "train":
                for parquet in sorted((target / "data").glob("*/*.parquet")):
                    table = pq.read_table(parquet, columns=list(train_arrays))
                    for column in train_arrays:
                        train_arrays[column].extend(table.column(column).to_pylist())

    statistics = {key: _statistics(np.asarray(values, dtype=np.float32)) for key, values in train_arrays.items()}
    normalization = {"state": statistics["observation.state"], "action": statistics["action"]}
    stats_payload = {"__format_version": 2, "__cache_config": {"mode": "abs"}, "statistics": statistics}
    for target, contract in splits:
        contract["normalization"] = normalization
        (target / "task_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        (target / "meta/stats_gr00t.json").write_text(json.dumps(stats_payload, indent=2) + "\n")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mixture.json").write_text(json.dumps(mixture, indent=2) + "\n")
    prompt_map = {
        str(latency): {"latency_raw_frames": latency, "latency_ms": latency * 50.0, "prompt": latency_prompt(latency)}
        for latency in LATENCIES
    }
    (output / "latency_prompt_map.json").write_text(json.dumps(prompt_map, indent=2) + "\n")
    return {
        "train_episodes": len(latencies) * train_episodes,
        "validation_episodes": len(latencies) * val_episodes,
        "train_frames": len(train_arrays["action"]),
        "normalization": normalization,
        "selected_episode_indices": selections,
    }


def training_config(args, backbone: Path) -> dict:
    config = yaml.safe_load((STARVLA_ROOT / "examples/MIKASA/train_files/mikasa_mixed_01234_h1.yaml").read_text())
    batch = args.num_gpus * args.micro_batch
    if 128 % batch:
        raise ValueError("GPU count x microbatch must divide the baseline global batch of 128")
    config.update(
        run_id=args.run_id,
        run_root_dir=str(args.run_root.resolve()),
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    config["framework"]["qwenvl"].update(
        base_vlm=str(backbone),
        enable_gradient_checkpointing=args.gradient_checkpointing,
    )
    data = config["datasets"]["vla_data"]
    data.update(
        task_contract_path=str(args.output / f"fixed_l{args.latencies[0]}/lerobot/task_contract.json"),
        custom_mixtures_path=str(args.output / "mixture.json"),
        per_device_batch_size=args.micro_batch,
        prompt_mode=args.prompt_mode,
    )
    config["trainer"]["gradient_accumulation_steps"] = 128 // batch
    config["checkpoint"]["sync"]["repo_id"] = args.model_hf_repo
    return config


def eval_config(args, backbone: Path, latency: int) -> dict:
    trained = args.run_root.resolve() / args.run_id
    return {
        "experiment": {"name": f"{args.run_id}_eval_l{latency}", "seed": 4242424242},
        "executor": {
            "mode": "simulated",
            "simulated_inference_pool": True,
            "simulated_worker_capacity": 1,
            "inference_devices": ["cuda:0"],
            "inference_batch_size": 8,
        },
        "env": {
            "name": "mikasa_intercept_grab_fast",
            "env_fps": 20,
            "obs_fps": 20,
            "frame_stack": 1,
            "simulator_device": "cuda:0",
            "noop_action": [0.0] * 7,
            "base_prompt": BASE_PROMPT,
        },
        "latency": {"method": "fixed", "fixed_latency_ms": latency * 50.0, "seed": 0, "sync_cuda": False},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready", "hold_last_chunk_action": True},
        "policy": {
            "type": "starvla",
            "action_prefix": {"mode": "none"},
            "checkpoint_path": str(trained / "checkpoints/steps_5000_pytorch_model.pt"),
            "model_config_path": str(trained / "config.full.yaml"),
            "task_contract_path": str(args.output / f"fixed_l{args.latencies[0]}/lerobot/task_contract.json"),
            "backbone_path": str(backbone),
            "worker_python_executable": args.vla_python,
            "device": "cuda:0",
            "state_info_key": "mikasa_proprio",
            "image_views_info_key": "mikasa_image_views",
            "latency_prompt_map_path": str(args.output / "latency_prompt_map.json"),
            "prompt_mode": args.prompt_mode,
        },
        "evaluation": {
            "eval_episodes": args.eval_episodes,
            "eval_parallel_envs": 10,
            "eval_max_steps": 60,
            "eval_deterministic": True,
        },
        "logging": {
            "output_dir": str(trained / f"post_train_eval/fixed_{latency}"),
            "video": {"enabled": False, "num_bins": 10},
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--dataset-hf-repo", default="latency-sensitive-bench/benchmark-datasets")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--latencies", type=int, nargs="+", default=list(LATENCIES))
    parser.add_argument("--train-episodes", type=int, default=250)
    parser.add_argument("--val-episodes", type=int, default=25)
    parser.add_argument("--source-train-episodes", type=int, default=250)
    parser.add_argument("--source-val-episodes", type=int, default=25)
    parser.add_argument("--backbone-path", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--prompt-mode", choices=("raw", "latency_neutral"), default="raw")
    parser.add_argument("--model-hf-repo", required=True)
    parser.add_argument("--wandb-entity", default="talha1503")
    parser.add_argument("--wandb-project", default="starvla_tasks")
    parser.add_argument("--vla-python", required=True)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.source_root is not None:
        sources = {
            latency: args.source_root.expanduser().resolve() / f"fixed_l{latency}/dataset" for latency in args.latencies
        }
        source_reference = {"local_root": str(args.source_root.expanduser().resolve())}
    else:
        revision = HfApi().dataset_info(args.dataset_hf_repo, revision=args.dataset_revision).sha
        prefixes = {
            latency: (
                f"fixed-latency-l{latency}/mikasa-intercept-grab-fast/teacher-rollouts-h1-success-"
                f"{args.source_train_episodes}train-{args.source_val_episodes}val"
            )
            for latency in args.latencies
        }
        snapshot = Path(
            snapshot_download(
                repo_id=args.dataset_hf_repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    f"{prefix}/{split}/**" for prefix in prefixes.values() for split in ("lerobot", "lerobot__val")
                ],
            )
        )
        sources = {latency: snapshot / prefix for latency, prefix in prefixes.items()}
        source_reference = {"repo_id": args.dataset_hf_repo, "revision": revision, "paths": prefixes}
    counts = prepare_datasets(
        sources,
        args.output,
        latencies=args.latencies,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
    )
    if args.backbone_path is not None:
        backbone = args.backbone_path.expanduser().resolve()
    else:
        backbone = Path(
            snapshot_download(
                repo_id=BACKBONE_REPO,
                revision=BACKBONE_REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
            )
        )
    config = training_config(args, backbone)
    (args.output / "training.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for latency in LATENCIES:
        (args.output / f"eval_l{latency}.yaml").write_text(
            yaml.safe_dump(eval_config(args, backbone, latency), sort_keys=False)
        )
    provenance = {
        "baseline": {
            "repo_id": "latency-sensitive-bench/benchmark-models",
            "revision": BASELINE_REVISION,
            "path": BASELINE_PATH,
            "benchmark_code_revision": "2f18d4620afd1861059d8cecfc2b1410a4a4777d",
            "wandb_run": "zihanwang-ai-northwestern-university/starvla_tasks/orf7ua9j",
        },
        "dataset": source_reference,
        "backbone": {"repo_id": BACKBONE_REPO, "revision": BACKBONE_REVISION},
        "counts": counts,
        "prompt_mode": args.prompt_mode,
        "global_batch": 128,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Prepared {counts['train_episodes']} train / {counts['validation_episodes']} validation episodes; "
        "H1, global batch 128."
    )


if __name__ == "__main__":
    main()
