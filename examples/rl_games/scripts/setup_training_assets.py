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

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STEP_FILE_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
STEP_STATE_RE = re.compile(r"steps_(\d+)_state$")
STEP_LORA_ADAPTER_RE = re.compile(r"steps_(\d+)_lora_adapter$")


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


def _find_latest_local_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    if not checkpoint_dir.exists():
        return None, 0, None
    candidates: list[tuple[int, int, Path, str]] = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            match = STEP_STATE_RE.match(item.name)
            if match:
                candidates.append((int(match.group(1)), 1, item, "state"))
                continue
            match = STEP_LORA_ADAPTER_RE.match(item.name)
            if match:
                candidates.append((int(match.group(1)), 0, item, "model"))
                continue
        elif item.is_file():
            match = STEP_FILE_RE.match(item.name)
            if match:
                candidates.append((int(match.group(1)), 0, item, "model"))
    if not candidates:
        return None, 0, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _, path, kind = candidates[-1]
    return path, step, kind


def _download_latest_hf_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:
        return None, 0, None, f"huggingface_hub import failed: {exc}"

    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return None, 0, None, f"could not list HF checkpoint repo {repo_id}: {exc}"

    candidates: list[tuple[int, int, str, str]] = []
    seen_state_dirs = set()
    for file_path in files:
        first_part = file_path.split("/", 1)[0]
        state_match = STEP_STATE_RE.match(first_part)
        if state_match:
            if first_part not in seen_state_dirs:
                candidates.append((int(state_match.group(1)), 1, first_part, "state"))
                seen_state_dirs.add(first_part)
            continue
        lora_match = STEP_LORA_ADAPTER_RE.match(first_part)
        if lora_match:
            if first_part not in seen_state_dirs:
                candidates.append((int(lora_match.group(1)), 0, first_part, "model"))
                seen_state_dirs.add(first_part)
            continue
        file_match = STEP_FILE_RE.match(os.path.basename(file_path))
        if file_match:
            candidates.append((int(file_match.group(1)), 0, file_path, "model"))
    if not candidates:
        return None, 0, None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _, chosen, kind = candidates[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "state" or STEP_LORA_ADAPTER_RE.match(chosen):
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"{chosen}/**"],
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
            )
            local_path = checkpoint_dir / chosen
        else:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=chosen,
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
            )
    except Exception as exc:
        return None, 0, None, f"could not download HF checkpoint {chosen}: {exc}"
    return Path(local_path), step, kind, None


def _is_missing_hf_repo_error(error: str | None) -> bool:
    if not error:
        return False
    return "Repository Not Found" in error or "404 Client Error" in error


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
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        get_dataset_named_mixture,
    )

    mixture = get_dataset_named_mixture(data_mix)
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


def _ensure_rl_games_lerobot_dataset(args) -> dict[str, Any]:
    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    data_mix = args.converted_dataset_name
    dataset_dir = data_root_dir / data_mix
    eval_data_mix = f"{data_mix}__val"
    eval_dataset_dir = data_root_dir / eval_data_mix
    force = _str2bool(args.dataset_force_download)
    mixed_latency = args.mode == "mixed_latency" or str(args.latency_mode or "").lower() == "mixed"
    prompt_map = dataset_dir / "latency_prompt_map.json"
    converted_dataset_hf = str(args.converted_dataset_hf or "")

    def _mixed_prompt_map_ready() -> bool:
        if not mixed_latency:
            return True
        if not prompt_map.exists():
            return False
        mapping = json.loads(prompt_map.read_text(encoding="utf-8"))
        return len(mapping) > 1

    downloaded = False
    ready = _dataset_ready(dataset_dir) and _dataset_ready(eval_dataset_dir) and _mixed_prompt_map_ready()
    if force or not ready:
        if converted_dataset_hf:
            data_root_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=converted_dataset_hf,
                repo_type="dataset",
                local_dir=str(data_root_dir),
            )
            downloaded = True
            ready = _dataset_ready(dataset_dir) and _dataset_ready(eval_dataset_dir) and _mixed_prompt_map_ready()
        if not ready:
            raise FileNotFoundError(
                f"converted LeRobot dataset is not ready under {data_root_dir}: "
                f"expected {dataset_dir} and {eval_dataset_dir}. "
                "Prepare it with scripts/rollout_data/prepare_starvla_lerobot_dataset.py "
                "or set dataset.converted_hf to a converted dataset repo."
            )

    validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=data_mix)
    eval_validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=eval_data_mix)
    return {
        "dataset_ready": True,
        "dataset_downloaded": downloaded,
        "converted_dataset_hf": converted_dataset_hf or None,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(dataset_dir),
        "data_mix": data_mix,
        "eval_data_mix": eval_data_mix,
        "eval_dataset_dir": str(eval_dataset_dir),
        "latency_prompt_map_path": str(prompt_map) if prompt_map.exists() else None,
        **validation,
        "eval_dataset_stats_path": eval_validation["dataset_stats_path"],
        "eval_dataset_num_steps": eval_validation["dataset_num_steps"],
        "eval_dataset_num_trajectories": eval_validation["dataset_num_trajectories"],
    }


def setup_assets(args) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": args.model,
        "env": args.env,
        "mode": args.mode,
    }

    if args.model in {"openvla", "pi0"} and args.env in {"flappy", "demon_attack", "deadly_corridor"}:
        result.update(_ensure_rl_games_lerobot_dataset(args))
    else:
        data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
        result.update({
            "dataset_ready": _has_files(data_root_dir),
            "dataset_local_dir": str(data_root_dir),
            "data_mix": None,
            "eval_data_mix": None,
            "latency_prompt_map_path": None,
        })

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    result.update(_ensure_base_model(args.model, base_model_dir, args.base_model_repo_id))

    checkpoint_dir = Path(args.checkpoint_local_dir).expanduser().resolve()
    local_ckpt, local_step, local_kind = (None, 0, None)
    if args.checkpoint_load in {"auto", "local"}:
        local_ckpt, local_step, local_kind = _find_latest_local_checkpoint(checkpoint_dir)
        if local_ckpt is not None and args.checkpoint_load == "local":
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_kind": local_kind,
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result

    hf_repo_id = args.checkpoint_hf_repo_id or args.hf_repo_id
    if args.checkpoint_load in {"auto", "hf"} and hf_repo_id:
        hf_ckpt, hf_step, hf_kind, hf_error = _download_latest_hf_checkpoint(hf_repo_id, checkpoint_dir)
        hf_is_better = (
            args.checkpoint_load == "hf"
            or local_ckpt is None
            or hf_step > local_step
            or (hf_step == local_step and hf_kind == "state" and local_kind != "state")
        )
        if hf_ckpt is not None and hf_is_better:
            result.update({
                "resume_found": True,
                "resume_source": "hf",
                "resume_kind": hf_kind,
                "resume_checkpoint": str(hf_ckpt),
                "resume_step": hf_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result
        if local_ckpt is not None:
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_kind": local_kind,
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            if hf_ckpt is not None:
                result["checkpoint_hf_status"] = (
                    f"local checkpoint step {local_step} is newer than or equal to HF step {hf_step}; using local"
                )
            return result
        if hf_error:
            sync_repo_id = str(getattr(args, "checkpoint_sync_repo_id", "") or "")
            sync_enabled = _str2bool(getattr(args, "checkpoint_sync_enabled", False))
            if (
                args.checkpoint_load == "auto"
                and sync_enabled
                and sync_repo_id == hf_repo_id
            ):
                result["checkpoint_hf_status"] = (
                    f"HF resume repo {hf_repo_id} was not available during auto-resume; "
                    "starting from local/base model and using it as the checkpoint sync destination"
                )
            else:
                result["checkpoint_hf_warning"] = hf_error

    if local_ckpt is not None:
        result.update({
            "resume_found": True,
            "resume_source": "local",
            "resume_kind": local_kind,
            "resume_checkpoint": str(local_ckpt),
            "resume_step": local_step,
            "checkpoint_local_dir": str(checkpoint_dir),
        })
        return result

    result.update({
        "resume_found": False,
        "resume_source": None,
        "resume_kind": None,
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
    parser.add_argument("--converted-dataset-hf", default="")
    parser.add_argument("--dataset-local-dir", required=True)
    parser.add_argument("--converted-dataset-name", default="flappy_train")
    parser.add_argument("--dataset-force-download", default="false")
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-model-repo-id", default=None)
    parser.add_argument("--checkpoint-local-dir", required=True)
    parser.add_argument("--checkpoint-load", choices=["auto", "none", "local", "hf"], default="auto")
    parser.add_argument("--checkpoint-hf-repo-id", default="")
    parser.add_argument("--checkpoint-sync-enabled", default="false")
    parser.add_argument("--checkpoint-sync-repo-id", default="")
    parser.add_argument("--hf-repo-id", default="")
    args = parser.parse_args()
    if args.base_model_repo_id == "":
        args.base_model_repo_id = None

    with contextlib.redirect_stdout(sys.stderr):
        result = setup_assets(args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
