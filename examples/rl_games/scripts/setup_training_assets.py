#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP_FILE_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _dataset_ready(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "meta/modality.json",
        dataset_dir / "meta/info.json",
        dataset_dir / "meta/episodes.jsonl",
        dataset_dir / "meta/tasks.jsonl",
    ]
    return all(path.exists() for path in required) and any(dataset_dir.glob("data/*/*.parquet"))


def _find_latest_local_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    if not checkpoint_dir.exists():
        return None, 0
    candidates: list[tuple[int, Path]] = []
    for item in checkpoint_dir.iterdir():
        if not item.is_file():
            continue
        match = STEP_FILE_RE.match(item.name)
        if match:
            candidates.append((int(match.group(1)), item))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda item: item[0])
    step, path = candidates[-1]
    return path, step


def _download_latest_hf_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        return None, 0, f"huggingface_hub import failed: {exc}"

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return None, 0, f"could not list HF checkpoint repo {repo_id}: {exc}"

    candidates: list[tuple[int, str]] = []
    for file_path in files:
        match = STEP_FILE_RE.match(os.path.basename(file_path))
        if match:
            candidates.append((int(match.group(1)), file_path))
    if not candidates:
        return None, 0, None

    candidates.sort(key=lambda item: item[0])
    step, chosen = candidates[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=chosen,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        return None, 0, f"could not download HF checkpoint {chosen}: {exc}"
    return Path(local_path), step, None


def _ensure_base_model(model: str, base_model_dir: Path, base_model_repo_id: str | None) -> dict[str, Any]:
    info = {
        "base_model_dir": str(base_model_dir),
        "base_model_repo_id": base_model_repo_id,
        "base_model_downloaded": False,
    }
    if _has_files(base_model_dir):
        return info
    if not base_model_repo_id:
        raise ValueError(f"base model directory is missing and no repo id was provided: {base_model_dir}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download base model weights") from exc

    base_model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=base_model_repo_id,
        repo_type="model",
        local_dir=str(base_model_dir),
    )
    info["base_model_downloaded"] = True
    return info


def _validate_starvla_dataset(data_root_dir: Path, data_mix: str) -> dict[str, Any]:
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.dataloader.gr00t_lerobot.registry import (
        DATASET_NAMED_MIXTURES,
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
    )

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    dataset_name, _, robot_type = mixture[0]
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    dataset = make_LeRobotSingleDataset(
        data_root_dir=data_root_dir,
        data_name=dataset_name,
        robot_type=robot_type,
        data_cfg={
            "include_state": False,
            "video_backend": "torchvision_av",
            "lerobot_version": "v2.0",
        },
    )
    stats_path = data_root_dir / dataset_name / "dataset_statistics.json"
    dataset._save_dataset_statistics_(stats_path)
    return {
        "dataset_stats_path": str(stats_path),
        "dataset_num_steps": len(dataset),
        "dataset_num_trajectories": len(dataset.trajectory_ids),
        "dataset_robot_type": robot_type,
        "dataset_embodiment_tag": str(ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type)),
    }


def _ensure_flappy_dataset(args) -> dict[str, Any]:
    from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import convert_dataset
    from examples.rl_games.data_conversion.verify_flappy_dataset import verify_dataset

    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    data_mix = args.converted_dataset_name
    dataset_dir = data_root_dir / data_mix
    force = _str2bool(args.setup_force) or _str2bool(args.dataset_force_download)
    mixed_latency = args.mode == "mixed_latency" or str(args.latency_mode or "").lower() == "mixed"
    prompt_map = dataset_dir / "latency_prompt_map.json"

    rebuild = force or not _dataset_ready(dataset_dir) or (mixed_latency and not prompt_map.exists())
    converted = False
    if rebuild:
        if not args.source_dataset_hf:
            raise ValueError(
                f"{dataset_dir} is not ready; pass --source-dataset-hf so setup can verify and convert it"
            )
        verify_dataset(
            args.source_dataset_hf,
            rows=args.verify_rows,
            cache_dir=args.dataset_cache_dir,
            strict=True,
            allow_mixed_latency_prompts=mixed_latency,
        )
        convert_dataset(
            args.source_dataset_hf,
            dataset_dir,
            cache_dir=args.dataset_cache_dir,
            max_episodes=args.max_episodes,
            force=rebuild,
        )
        converted = True

    validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=data_mix)
    return {
        "dataset_ready": True,
        "dataset_converted": converted,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(dataset_dir),
        "data_mix": data_mix,
        "latency_prompt_map_path": str(prompt_map) if prompt_map.exists() else None,
        **validation,
    }


def setup_assets(args) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": args.model,
        "env": args.env,
        "mode": args.mode,
    }

    if args.model == "openvla" and args.env == "flappy":
        result.update(_ensure_flappy_dataset(args))
    else:
        data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
        result.update({
            "dataset_ready": _has_files(data_root_dir),
            "dataset_local_dir": str(data_root_dir),
            "data_mix": None,
            "latency_prompt_map_path": None,
        })

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    result.update(_ensure_base_model(args.model, base_model_dir, args.base_model_repo_id))

    checkpoint_dir = Path(args.checkpoint_local_dir).expanduser().resolve()
    if args.checkpoint_load != "none":
        local_ckpt, local_step = _find_latest_local_checkpoint(checkpoint_dir)
        if local_ckpt is not None:
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result

    hf_repo_id = args.checkpoint_hf_repo_id or args.hf_repo_id
    if args.checkpoint_load in {"auto", "hf"} and hf_repo_id:
        hf_ckpt, hf_step, hf_error = _download_latest_hf_checkpoint(hf_repo_id, checkpoint_dir)
        if hf_ckpt is not None:
            result.update({
                "resume_found": True,
                "resume_source": "hf",
                "resume_checkpoint": str(hf_ckpt),
                "resume_step": hf_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result
        if hf_error:
            result["checkpoint_hf_warning"] = hf_error

    result.update({
        "resume_found": False,
        "resume_source": None,
        "resume_checkpoint": None,
        "resume_step": 0,
        "checkpoint_local_dir": str(checkpoint_dir),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--latency-mode", default="")
    parser.add_argument("--source-dataset-hf", default="")
    parser.add_argument("--dataset-local-dir", required=True)
    parser.add_argument("--converted-dataset-name", default="flappy_train")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-force-download", default="false")
    parser.add_argument("--setup-force", default="false")
    parser.add_argument("--verify-rows", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-model-repo-id", default=None)
    parser.add_argument("--checkpoint-local-dir", required=True)
    parser.add_argument("--checkpoint-load", choices=["auto", "none", "local", "hf"], default="auto")
    parser.add_argument("--checkpoint-hf-repo-id", default="")
    parser.add_argument("--hf-repo-id", default="")
    args = parser.parse_args()
    if args.dataset_cache_dir == "":
        args.dataset_cache_dir = None
    if args.base_model_repo_id == "":
        args.base_model_repo_id = None

    with contextlib.redirect_stdout(sys.stderr):
        result = setup_assets(args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
