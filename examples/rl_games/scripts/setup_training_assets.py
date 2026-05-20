#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import inspect
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
STEP_STATE_RE = re.compile(r"steps_(\d+)_state$")
DEBUG_DATASET_RE = re.compile(r"^(?P<base>.+)(?P<debug>__debug(?:_[A-Za-z0-9_-]+)?_\d+ep)$")


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "checkpoint"


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
        file_match = STEP_FILE_RE.match(os.path.basename(file_path))
        if file_match:
            candidates.append((int(file_match.group(1)), 0, file_path, "model"))
    if not candidates:
        return None, 0, None, None

    candidates.sort(key=lambda item: (item[0], item[1]))
    step, _, chosen, kind = candidates[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "state":
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


def _download_hf_checkpoint_file(repo_id: str, filename: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        return None, 0, f"huggingface_hub import failed: {exc}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        return None, 0, f"could not download HF checkpoint {filename} from {repo_id}: {exc}"

    file_match = STEP_FILE_RE.match(os.path.basename(filename))
    step = int(file_match.group(1)) if file_match else 0
    return Path(local_path), step, None


def _resolve_local_initialization_checkpoint(local_dir: str, filename: str) -> tuple[Path | None, int]:
    if not local_dir or not filename:
        return None, 0
    checkpoint_path = (Path(local_dir).expanduser() / filename).resolve()
    if not checkpoint_path.exists():
        return None, 0
    file_match = STEP_FILE_RE.match(checkpoint_path.name)
    step = int(file_match.group(1)) if file_match else 0
    return checkpoint_path, step


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


def _initialization_mode(args) -> str:
    return str(getattr(args, "initialization_mode", None) or "scratch").lower()


def _action_carrier(args) -> str:
    configured = str(getattr(args, "action_carrier", "") or "").lower()
    if configured in {"native", "bridge"}:
        return configured
    if _initialization_mode(args) in {"pre-trained", "pretrained", "bridge"}:
        return "bridge"
    return "native"


def _carrier_dataset_name(data_mix: str, action_carrier: str) -> str:
    if action_carrier != "bridge" or "__bridge" in data_mix:
        return data_mix
    match = DEBUG_DATASET_RE.match(data_mix)
    if match:
        return f"{match.group('base')}__bridge{match.group('debug')}"
    return f"{data_mix}__bridge"


def _ensure_rl_games_lerobot_dataset(args, *, convert_dataset, verify_dataset) -> dict[str, Any]:
    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    action_carrier = _action_carrier(args)
    data_mix = _carrier_dataset_name(args.converted_dataset_name, action_carrier)
    dataset_dir = data_root_dir / data_mix
    eval_data_mix = f"{data_mix}__val"
    eval_dataset_dir = data_root_dir / eval_data_mix
    force = _str2bool(args.setup_force) or _str2bool(args.dataset_force_download)
    mixed_latency = args.mode == "mixed_latency" or str(args.latency_mode or "").lower() == "mixed"
    prompt_map = dataset_dir / "latency_prompt_map.json"

    def _manifest_matches(dataset_path: Path) -> bool:
        manifest_path = dataset_path / "manifest.json"
        if not manifest_path.exists():
            return True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if args.source_dataset_hf and str(manifest.get("source", "")) != str(args.source_dataset_hf):
            return False
        if str(manifest.get("action_carrier", "native")) != action_carrier:
            return False
        expected_latency_filter = getattr(args, "latency_filter", None)
        if expected_latency_filter is not None:
            manifest_latency_filter = manifest.get("latency_filter")
            if manifest_latency_filter != [int(value) for value in expected_latency_filter]:
                return False
        return True

    def _mixed_prompt_map_ready() -> bool:
        if not mixed_latency:
            return True
        if not prompt_map.exists():
            return False
        try:
            mapping = json.loads(prompt_map.read_text(encoding="utf-8"))
        except Exception:
            return False
        return len(mapping) > 1

    rebuild = (
        force
        or not _dataset_ready(dataset_dir)
        or not _dataset_ready(eval_dataset_dir)
        or not _manifest_matches(dataset_dir)
        or not _manifest_matches(eval_dataset_dir)
        or not _mixed_prompt_map_ready()
    )
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
        convert_kwargs = {
            "cache_dir": args.dataset_cache_dir,
            "max_episodes": args.max_episodes,
            "force": rebuild,
            "require_latency_prompt_map": mixed_latency,
        }
        if "latency_filter" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["latency_filter"] = getattr(args, "latency_filter", None)
        if "action_carrier" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["action_carrier"] = action_carrier
        convert_dataset(args.source_dataset_hf, dataset_dir, **convert_kwargs)
        converted = True
        if mixed_latency and not _mixed_prompt_map_ready():
            raise ValueError(
                f"mixed-latency dataset conversion did not create a usable prompt map: {prompt_map}. "
                "Check that the selected training episodes contain latency/prompt columns for more than one latency."
            )

    validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=data_mix)
    eval_validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=eval_data_mix)
    return {
        "dataset_ready": True,
        "dataset_converted": converted,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(dataset_dir),
        "data_mix": data_mix,
        "eval_data_mix": eval_data_mix,
        "eval_dataset_dir": str(eval_dataset_dir),
        "action_carrier": action_carrier,
        "latency_prompt_map_path": str(prompt_map) if prompt_map.exists() else None,
        **validation,
        "eval_dataset_stats_path": eval_validation["dataset_stats_path"],
        "eval_dataset_num_steps": eval_validation["dataset_num_steps"],
        "eval_dataset_num_trajectories": eval_validation["dataset_num_trajectories"],
    }


def _ensure_flappy_dataset(args) -> dict[str, Any]:
    from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import convert_dataset
    from examples.rl_games.data_conversion.verify_flappy_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
    )


def _ensure_demon_attack_dataset(args) -> dict[str, Any]:
    from examples.rl_games.data_conversion.convert_demon_attack_to_starvla_lerobot import convert_dataset
    from examples.rl_games.data_conversion.verify_demon_attack_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
    )


def _ensure_deadly_corridor_dataset(args) -> dict[str, Any]:
    from examples.rl_games.data_conversion.convert_deadly_corridor_to_starvla_lerobot import convert_dataset
    from examples.rl_games.data_conversion.verify_deadly_corridor_dataset import verify_dataset

    return _ensure_rl_games_lerobot_dataset(
        args,
        convert_dataset=convert_dataset,
        verify_dataset=verify_dataset,
    )


def setup_assets(args) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": args.model,
        "env": args.env,
        "mode": args.mode,
        "initialization_mode": _initialization_mode(args),
        "action_carrier": _action_carrier(args),
    }

    if args.model in {"openvla", "pi0", "pi05", "gr00t"} and args.env == "flappy":
        result.update(_ensure_flappy_dataset(args))
    elif args.model in {"openvla", "pi0", "gr00t"} and args.env == "demon_attack":
        result.update(_ensure_demon_attack_dataset(args))
    elif args.model in {"openvla", "pi0", "gr00t"} and args.env == "deadly_corridor":
        result.update(_ensure_deadly_corridor_dataset(args))
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

    initialization_hf_repo_id = str(getattr(args, "initialization_hf_repo_id", "") or "")
    initialization_local_dir = str(getattr(args, "initialization_local_dir", "") or "")
    initialization_checkpoint_filename = str(getattr(args, "initialization_checkpoint_filename", "") or "")
    has_initialization_source = bool(initialization_local_dir or initialization_hf_repo_id)
    if has_initialization_source and _initialization_mode(args) in {"bridge", "pre-trained", "pretrained"}:
        if local_ckpt is not None and args.checkpoint_load == "auto":
            result.update({
                "resume_found": True,
                "resume_source": "local",
                "resume_kind": local_kind,
                "resume_checkpoint": str(local_ckpt),
                "resume_step": local_step,
                "checkpoint_local_dir": str(checkpoint_dir),
            })
            return result

        local_init_ckpt, local_init_step = _resolve_local_initialization_checkpoint(
            initialization_local_dir,
            initialization_checkpoint_filename,
        )
        if local_init_ckpt is not None:
            result.update({
                "resume_found": False,
                "resume_source": None,
                "resume_kind": None,
                "resume_checkpoint": None,
                "resume_step": 0,
                "checkpoint_local_dir": str(checkpoint_dir),
                "pretrained_checkpoint": str(local_init_ckpt),
                "initialization_source": "local",
                "initialization_local_dir": str(Path(initialization_local_dir).expanduser().resolve()),
                "initialization_hf_repo_id": initialization_hf_repo_id or None,
                "initialization_checkpoint_filename": initialization_checkpoint_filename or None,
                "initialization_step": local_init_step,
            })
            return result

        if not initialization_hf_repo_id:
            raise FileNotFoundError(
                f"Bridge initialization checkpoint not found under initialization_local_dir="
                f"{initialization_local_dir!r} with filename={initialization_checkpoint_filename!r}, "
                "and no initialization_hf_repo_id was provided for download."
            )

        init_dir = checkpoint_dir / "_initialization" / _safe_path_name(initialization_hf_repo_id)
        if initialization_checkpoint_filename:
            init_ckpt, init_step, init_error = _download_hf_checkpoint_file(
                initialization_hf_repo_id,
                initialization_checkpoint_filename,
                init_dir,
            )
            init_kind = "model" if init_ckpt is not None else None
        else:
            init_ckpt, init_step, init_kind, init_error = _download_latest_hf_checkpoint(initialization_hf_repo_id, init_dir)
        if init_ckpt is None:
            raise RuntimeError(
                f"Could not download bridge initialization checkpoint from {initialization_hf_repo_id}: "
                f"{init_error or 'no checkpoint files found'}"
            )
        if init_kind == "state":
            raise ValueError(
                f"Bridge initialization expected model weights, but {initialization_hf_repo_id} resolved "
                f"to a full training-state directory: {init_ckpt}"
            )
        result.update({
            "resume_found": False,
            "resume_source": None,
            "resume_kind": None,
            "resume_checkpoint": None,
            "resume_step": 0,
            "checkpoint_local_dir": str(checkpoint_dir),
            "pretrained_checkpoint": str(init_ckpt),
            "initialization_source": "hf",
            "initialization_local_dir": initialization_local_dir or None,
            "initialization_hf_repo_id": initialization_hf_repo_id,
            "initialization_checkpoint_filename": initialization_checkpoint_filename or None,
            "initialization_step": init_step,
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
    parser.add_argument("--initialization-mode", default="")
    parser.add_argument("--action-carrier", choices=["", "native", "bridge"], default="")
    parser.add_argument("--latency-mode", default="")
    parser.add_argument("--source-dataset-hf", default="")
    parser.add_argument("--dataset-local-dir", required=True)
    parser.add_argument("--converted-dataset-name", default="flappy_train")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-force-download", default="false")
    parser.add_argument("--setup-force", default="false")
    parser.add_argument("--verify-rows", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--latency-filter", default=None)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-model-repo-id", default=None)
    parser.add_argument("--checkpoint-local-dir", required=True)
    parser.add_argument("--checkpoint-load", choices=["auto", "none", "local", "hf"], default="auto")
    parser.add_argument("--checkpoint-hf-repo-id", default="")
    parser.add_argument("--initialization-local-dir", default="")
    parser.add_argument("--initialization-hf-repo-id", default="")
    parser.add_argument("--initialization-checkpoint-filename", default="")
    parser.add_argument("--checkpoint-sync-enabled", default="false")
    parser.add_argument("--checkpoint-sync-repo-id", default="")
    parser.add_argument("--hf-repo-id", default="")
    args = parser.parse_args()
    if args.dataset_cache_dir == "":
        args.dataset_cache_dir = None
    if args.base_model_repo_id == "":
        args.base_model_repo_id = None
    if isinstance(args.latency_filter, str) and args.latency_filter:
        args.latency_filter = [int(item) for item in args.latency_filter.split(",") if item.strip()]
    elif args.latency_filter == "":
        args.latency_filter = None

    with contextlib.redirect_stdout(sys.stderr):
        result = setup_assets(args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
