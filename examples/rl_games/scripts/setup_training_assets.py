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
INITIALIZATION_SOURCE_MODES = {"bridge", "pre-trained", "pretrained", "backbone_bridge_factorized11"}


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _safe_path_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "checkpoint"


def _safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "value"


def _latency_suffix(latencies: list[int] | None) -> str:
    if not latencies:
        return "all"
    return "lat" + "_".join(str(int(value)) for value in latencies)


def _derived_dataset_name(base_name: str, *, latency_filter: list[int] | None, episodes_per_latency: int | None, max_episodes: int | None) -> str:
    pieces = [str(base_name), _latency_suffix(latency_filter)]
    if episodes_per_latency is not None:
        pieces.append(f"{int(episodes_per_latency)}ep_per_lat")
    elif max_episodes is not None:
        pieces.append(f"{int(max_episodes)}ep")
    return "__".join(_safe_token(piece) for piece in pieces)


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


def _load_source_latency_prompt_map(
    dataset_name: str,
    *,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    frameskip: int = 1,
) -> dict[str, dict[str, Any]]:
    from datasets import load_dataset
    from examples.rl_games.data_conversion.verify_flappy_dataset import build_latency_prompt_map

    def _load(columns: list[str] | None = None):
        load_kwargs = {"split": "train", "cache_dir": cache_dir, "columns": columns}
        if dataset_source_subdir not in (None, ""):
            load_kwargs["data_dir"] = dataset_source_subdir
        if dataset_config_name not in (None, ""):
            return load_dataset(dataset_name, dataset_config_name, **load_kwargs)
        return load_dataset(dataset_name, **load_kwargs)

    try:
        ds = _load(columns=["prompt", "latency", "latency_ms", "split"])
    except Exception:
        try:
            ds = _load(columns=["prompt", "latency_raw_frames", "latency_ms", "split"])
        except Exception:
            try:
                ds = _load(columns=["prompt", "latency", "latency_ms"])
            except Exception:
                try:
                    ds = _load(columns=["prompt", "latency_raw_frames", "latency_ms"])
                except Exception:
                    ds = _load()
                    if "prompt" not in ds.column_names or not ({"latency", "latency_raw_frames"} & set(ds.column_names)):
                        raise ValueError(
                            f"prompt source dataset {dataset_name} is missing columns required for a latency prompt map; "
                            f"available columns: {ds.column_names}"
                        )
    return build_latency_prompt_map(ds, frameskip=frameskip)


def _task_latency_frameskip(task_name: str) -> int:
    return 1 if str(task_name) == "flappy" else 4


def _write_prompt_map(path: Path, prompt_map: dict[str, dict[str, Any]], *, latency_filter: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if latency_filter:
        allowed = {int(value) for value in latency_filter}
        prompt_map = {str(k): v for k, v in prompt_map.items() if int(k) in allowed}
    path.write_text(json.dumps(prompt_map, indent=2), encoding="utf-8")
    return path


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


def _read_best_checkpoint_step(checkpoint_dir: Path) -> int:
    metadata_path = checkpoint_dir / "best_model_metadata.json"
    if not metadata_path.exists():
        return 0
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    for key in ("best_step", "step"):
        if metadata.get(key) is not None:
            try:
                return int(metadata[key])
            except Exception:
                return 0
    return 0


def _find_local_best_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    best_state = checkpoint_dir / "best_state"
    if not best_state.is_dir():
        return None, 0, None
    return best_state, _read_best_checkpoint_step(checkpoint_dir), "state"


def _candidate_resume_paths(checkpoint: str, checkpoint_dir: Path) -> list[Path]:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute():
        return [checkpoint_path]

    run_dir = checkpoint_dir.parent
    return [
        run_dir / checkpoint_path,
        checkpoint_dir / checkpoint_path.name,
        checkpoint_dir / checkpoint_path,
    ]


def _resolve_existing_resume_path(checkpoint: str, checkpoint_dir: Path) -> Path:
    candidates = _candidate_resume_paths(checkpoint, checkpoint_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"explicit resume checkpoint does not exist: {checkpoint}. Checked: {checked}")


def _resolve_explicit_resume_checkpoint(checkpoint: str, checkpoint_dir: Path) -> tuple[Path, int, str]:
    checkpoint_path = _resolve_existing_resume_path(checkpoint, checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"explicit resume checkpoint does not exist: {checkpoint_path}")
    if checkpoint_path.is_dir():
        state_match = STEP_STATE_RE.match(checkpoint_path.name)
        if state_match:
            return checkpoint_path, int(state_match.group(1)), "state"
        if checkpoint_path.name == "best_state":
            return checkpoint_path, _read_best_checkpoint_step(checkpoint_path.parent), "state"
        raise ValueError(
            f"unsupported explicit resume checkpoint directory: {checkpoint_path}. "
            "Expected steps_<N>_state/ or best_state/."
        )
    if checkpoint_path.is_file():
        file_match = STEP_FILE_RE.match(checkpoint_path.name)
        if file_match:
            return checkpoint_path, int(file_match.group(1)), "model"
        if checkpoint_path.parent.name.endswith("_state"):
            state_match = STEP_STATE_RE.match(checkpoint_path.parent.name)
            if state_match and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}:
                return checkpoint_path.parent, int(state_match.group(1)), "state"
        if checkpoint_path.parent.name == "best_state" and checkpoint_path.name in {"model.safetensors", "pytorch_model.bin"}:
            return checkpoint_path.parent, _read_best_checkpoint_step(checkpoint_path.parent.parent), "state"
    raise ValueError(
        f"unsupported explicit resume checkpoint: {checkpoint_path}. "
        "Expected steps_<N>_state/, steps_<N>_pytorch_model.pt, steps_<N>_model.safetensors, "
        "best_state/, or a model file inside a state directory."
    )


def _download_latest_hf_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:
        return None, 0, None, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        files = HfApi(token=hf_token).list_repo_files(repo_id=repo_id, repo_type="model")
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
                token=hf_token,
            )
            local_path = checkpoint_dir / chosen
        else:
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=chosen,
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
                token=hf_token,
            )
    except Exception as exc:
        return None, 0, None, f"could not download HF checkpoint {chosen}: {exc}"
    return Path(local_path), step, kind, None


def _download_hf_best_checkpoint(repo_id: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None, str | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        return None, 0, None, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        files = HfApi(token=hf_token).list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        return None, 0, None, f"could not list HF checkpoint repo {repo_id}: {exc}"

    if not any(file_path.startswith("best_state/") for file_path in files):
        return None, 0, None, None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=["best_state/**", "best_model_metadata.json"],
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except Exception as exc:
        return None, 0, None, f"could not download HF best checkpoint from {repo_id}: {exc}"

    best_state = checkpoint_dir / "best_state"
    if not best_state.is_dir():
        return None, 0, None, f"HF best checkpoint from {repo_id} did not contain best_state/"
    return best_state, _read_best_checkpoint_step(checkpoint_dir), "state", None


def _download_hf_checkpoint_file(repo_id: str, filename: str, checkpoint_dir: Path) -> tuple[Path | None, int, str | None]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        return None, 0, f"huggingface_hub import failed: {exc}"

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
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
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    snapshot_download(
        repo_id=base_model_repo_id,
        repo_type="model",
        local_dir=str(base_model_dir),
        token=hf_token,
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
    total_steps = 0
    total_trajectories = 0
    first_stats: dict[str, Any] | None = None
    for dataset_name, _, robot_type in mixture:
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
        total_steps += len(dataset)
        total_trajectories += len(dataset.trajectory_ids)
        if first_stats is None:
            first_stats = {
                "dataset_stats_path": str(stats_path),
                "dataset_robot_type": robot_type,
                "dataset_embodiment_tag": str(ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type)),
            }
    assert first_stats is not None
    return {
        **first_stats,
        "dataset_num_steps": total_steps,
        "dataset_num_trajectories": total_trajectories,
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
    action_layout = str(getattr(args, "deadly_action_layout", "") or "")
    source_config_name = getattr(args, "source_dataset_config_name", None)
    source_config_name = None if source_config_name in (None, "") else str(source_config_name)
    source_subdir = getattr(args, "source_dataset_subdir", None)
    source_subdir = None if source_subdir in (None, "") else str(source_subdir)
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
        if str(manifest.get("action_carrier", "native")) != action_carrier:
            return False
        if (manifest.get("source_config") or None) != source_config_name:
            return False
        if (manifest.get("source_subdir") or None) != source_subdir:
            return False
        if action_layout and ("action_layout" not in manifest or str(manifest["action_layout"]) != action_layout):
            return False
        if mixed_latency and manifest.get("latency_metadata") is not True:
            return False
        expected_latency_filter = getattr(args, "latency_filter", None)
        if expected_latency_filter is not None:
            manifest_latency_filter = manifest.get("latency_filter")
            if manifest_latency_filter != [int(value) for value in expected_latency_filter]:
                return False
        expected_episodes_per_latency = getattr(args, "episodes_per_latency", None)
        if expected_episodes_per_latency is not None:
            if manifest.get("episodes_per_latency") != int(expected_episodes_per_latency):
                return False
        expected_max_episodes = getattr(args, "max_episodes", None)
        if expected_max_episodes is not None:
            if manifest.get("max_episodes") != int(expected_max_episodes):
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
        if not _str2bool(getattr(args, "skip_verification", "false")):
            verify_kwargs = {
                "rows": args.verify_rows,
                "cache_dir": args.dataset_cache_dir,
                "strict": True,
                "allow_mixed_latency_prompts": mixed_latency,
            }
            if "dataset_config_name" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["dataset_config_name"] = source_config_name
            if "dataset_source_subdir" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["dataset_source_subdir"] = source_subdir
            if action_layout and "action_layout" in inspect.signature(verify_dataset).parameters:
                verify_kwargs["action_layout"] = action_layout
            verify_dataset(args.source_dataset_hf, **verify_kwargs)
        convert_kwargs = {
            "cache_dir": args.dataset_cache_dir,
            "max_episodes": args.max_episodes,
            "force": rebuild,
            "require_latency_prompt_map": mixed_latency,
        }
        if "dataset_config_name" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["dataset_config_name"] = source_config_name
        if "dataset_source_subdir" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["dataset_source_subdir"] = source_subdir
        if "latency_filter" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["latency_filter"] = getattr(args, "latency_filter", None)
        if "episodes_per_latency" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["episodes_per_latency"] = getattr(args, "episodes_per_latency", None)
        if "action_carrier" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["action_carrier"] = action_carrier
        if action_layout and "action_layout" in inspect.signature(convert_dataset).parameters:
            convert_kwargs["action_layout"] = action_layout
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


def _task_converter_and_verifier(task: str):
    if task == "flappy":
        from examples.rl_games.data_conversion.convert_flappy_to_starvla_lerobot import convert_dataset
        from examples.rl_games.data_conversion.verify_flappy_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_flappy"
    if task == "demon_attack":
        from examples.rl_games.data_conversion.convert_demon_attack_to_starvla_lerobot import convert_dataset
        from examples.rl_games.data_conversion.verify_demon_attack_dataset import verify_dataset
        return convert_dataset, verify_dataset, "rl_games_demon_attack"
    raise ValueError(f"cross-task rl_games currently supports only flappy and demon_attack, got {task!r}")


def _get_task_value(task_cfg: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in task_cfg and task_cfg[name] not in (None, ""):
            return task_cfg[name]
    return default


def _ensure_cross_task_datasets(args) -> dict[str, Any]:
    from starVLA.dataloader.gr00t_lerobot.registry import load_custom_mixtures

    cross_cfg = getattr(args, "cross_task", None) or {}
    train_tasks = cross_cfg.get("train_tasks") if isinstance(cross_cfg, dict) else None
    if not train_tasks:
        raise ValueError("rl_games.cross_task.train_tasks must contain at least flappy and demon_attack entries")

    data_root_dir = Path(args.dataset_local_dir).expanduser().resolve()
    action_carrier = _action_carrier(args)
    if action_carrier != "bridge":
        raise ValueError("cross-task OpenVLA RL-games currently requires action_carrier=bridge")

    mixture_entries: list[list[Any]] = []
    val_mixture_entries: list[list[Any]] = []
    prompt_maps: dict[str, str] = {}
    converted_names: dict[str, str] = {}
    converted = False
    force = _str2bool(args.setup_force) or _str2bool(args.dataset_force_download)

    for raw_task_cfg in train_tasks:
        task_cfg = dict(raw_task_cfg)
        task_name = str(_get_task_value(task_cfg, "name", "task"))
        convert_dataset, verify_dataset, robot_type = _task_converter_and_verifier(task_name)

        train_source_value = _get_task_value(task_cfg, "train_source_hf", "source_hf")
        prompt_source_value = _get_task_value(task_cfg, "prompt_source_hf", default=train_source_value)
        train_config_value = _get_task_value(task_cfg, "train_config_name", "source_config_name", "config_name", default=None)
        prompt_config_value = _get_task_value(task_cfg, "prompt_config_name", default=train_config_value)
        train_subdir_value = _get_task_value(task_cfg, "train_source_subdir", "source_subdir", default=None)
        prompt_subdir_value = _get_task_value(task_cfg, "prompt_source_subdir", default=train_subdir_value)
        train_source = str(train_source_value or "")
        prompt_source = str(prompt_source_value or "")
        train_config_name = None if train_config_value in (None, "") else str(train_config_value)
        prompt_config_name = None if prompt_config_value in (None, "") else str(prompt_config_value)
        train_subdir = None if train_subdir_value in (None, "") else str(train_subdir_value)
        prompt_subdir = None if prompt_subdir_value in (None, "") else str(prompt_subdir_value)
        if not train_source:
            raise ValueError(f"cross-task train task {task_name} is missing train_source_hf/source_hf")
        if not prompt_source:
            raise ValueError(f"cross-task train task {task_name} is missing prompt_source_hf")

        latency_filter = _get_task_value(task_cfg, "train_latency_filter", "latency_filter", default=None)
        latency_filter = [int(value) for value in latency_filter] if latency_filter not in (None, "") else None
        episodes_per_latency = _get_task_value(task_cfg, "episodes_per_latency", default=None)
        episodes_per_latency = int(episodes_per_latency) if episodes_per_latency not in (None, "") else None
        max_episodes = _get_task_value(task_cfg, "max_episodes", default=None)
        max_episodes = int(max_episodes) if max_episodes not in (None, "") else None
        weight = float(_get_task_value(task_cfg, "weight", default=1.0))

        base_converted_name = str(_get_task_value(task_cfg, "converted_name", default=f"{task_name}_cross_task_train"))
        converted_base = _derived_dataset_name(
            base_converted_name,
            latency_filter=latency_filter,
            episodes_per_latency=episodes_per_latency,
            max_episodes=max_episodes,
        )
        data_mix = _carrier_dataset_name(converted_base, action_carrier)
        dataset_dir = data_root_dir / data_mix
        eval_data_mix = f"{data_mix}__val"
        eval_dataset_dir = data_root_dir / eval_data_mix

        prompt_map = _load_source_latency_prompt_map(
            prompt_source,
            cache_dir=args.dataset_cache_dir,
            dataset_config_name=prompt_config_name,
            dataset_source_subdir=prompt_subdir,
            frameskip=_task_latency_frameskip(task_name),
        )
        prompt_dir = data_root_dir / "_prompt_maps" / data_mix
        eval_prompt_map_path = _write_prompt_map(prompt_dir / "eval_latency_prompt_map.json", prompt_map)
        train_prompt_map_path = _write_prompt_map(
            prompt_dir / "train_latency_prompt_map.json",
            prompt_map,
            latency_filter=latency_filter,
        )

        def _manifest_matches(dataset_path: Path) -> bool:
            manifest_path = dataset_path / "manifest.json"
            if not manifest_path.exists():
                return False
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return False
            return (
                str(manifest.get("source", "")) == train_source
                and str(manifest.get("prompt_source", "")) == prompt_source
                and (manifest.get("source_config") or None) == train_config_name
                and (manifest.get("prompt_source_config") or None) == prompt_config_name
                and (manifest.get("source_subdir") or None) == train_subdir
                and (manifest.get("prompt_source_subdir") or None) == prompt_subdir
                and str(manifest.get("action_carrier", "")) == action_carrier
                and manifest.get("latency_filter") == latency_filter
                and manifest.get("episodes_per_latency") == episodes_per_latency
                and manifest.get("max_episodes") == max_episodes
            )

        rebuild = (
            force
            or not _dataset_ready(dataset_dir)
            or not _dataset_ready(eval_dataset_dir)
            or not _manifest_matches(dataset_dir)
            or not _manifest_matches(eval_dataset_dir)
        )
        if rebuild:
            allow_mixed = bool(latency_filter) and train_source == prompt_source
            verify_dataset(
                train_source,
                rows=args.verify_rows,
                cache_dir=args.dataset_cache_dir,
                dataset_config_name=train_config_name,
                dataset_source_subdir=train_subdir,
                strict=True,
                allow_mixed_latency_prompts=allow_mixed,
            )
            convert_dataset(
                train_source,
                dataset_dir,
                cache_dir=args.dataset_cache_dir,
                dataset_config_name=train_config_name,
                dataset_source_subdir=train_subdir,
                max_episodes=max_episodes,
                force=rebuild,
                require_latency_prompt_map=bool(latency_filter),
                latency_filter=latency_filter,
                episodes_per_latency=episodes_per_latency,
                prompt_map_override=prompt_map,
                default_latency=0,
                action_carrier=action_carrier,
            )
            for manifest_path in (dataset_dir / "manifest.json", eval_dataset_dir / "manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["prompt_source"] = prompt_source
                manifest["prompt_source_config"] = prompt_config_name
                manifest["prompt_source_subdir"] = prompt_subdir
                manifest["eval_prompt_map_path"] = str(eval_prompt_map_path)
                manifest["train_prompt_map_path"] = str(train_prompt_map_path)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            converted = True

        mixture_entries.append([data_mix, weight, robot_type])
        val_mixture_entries.append([eval_data_mix, weight, robot_type])
        prompt_maps[task_name] = str(eval_prompt_map_path)
        converted_names[task_name] = data_mix

    mixture_name = "cross__" + "__".join(converted_names[task] for task in sorted(converted_names))
    mixture_name = _safe_token(mixture_name)
    eval_mixture_name = f"{mixture_name}__val"
    custom_mixtures_path = data_root_dir / "_generated_mixtures" / f"{mixture_name}.json"
    custom_mixtures_path.parent.mkdir(parents=True, exist_ok=True)
    custom_mixtures_path.write_text(
        json.dumps({mixture_name: mixture_entries, eval_mixture_name: val_mixture_entries}, indent=2),
        encoding="utf-8",
    )
    load_custom_mixtures(custom_mixtures_path)
    validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=mixture_name)
    eval_validation = _validate_starvla_dataset(data_root_dir=data_root_dir, data_mix=eval_mixture_name)
    return {
        "dataset_ready": True,
        "dataset_converted": converted,
        "dataset_local_dir": str(data_root_dir),
        "dataset_dir": str(data_root_dir),
        "data_mix": mixture_name,
        "eval_data_mix": eval_mixture_name,
        "custom_mixtures_path": str(custom_mixtures_path),
        "cross_task_prompt_maps": prompt_maps,
        "cross_task_datasets": converted_names,
        "action_carrier": action_carrier,
        **validation,
        "eval_dataset_stats_path": eval_validation["dataset_stats_path"],
        "eval_dataset_num_steps": eval_validation["dataset_num_steps"],
        "eval_dataset_num_trajectories": eval_validation["dataset_num_trajectories"],
    }


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

    supported_models = {"openvla", "pi0", "pi05", "gr00t"}
    if args.model in supported_models and str(getattr(args, "env", "")) == "cross_task":
        result.update(_ensure_cross_task_datasets(args))
    elif args.model in supported_models and args.env == "flappy":
        result.update(_ensure_flappy_dataset(args))
    elif args.model in supported_models and args.env == "demon_attack":
        result.update(_ensure_demon_attack_dataset(args))
    elif args.model in supported_models and args.env == "deadly_corridor":
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
    explicit_checkpoint = str(getattr(args, "checkpoint", "") or "")
    if explicit_checkpoint:
        resume_checkpoint, resume_step, resume_kind = _resolve_explicit_resume_checkpoint(explicit_checkpoint, checkpoint_dir)
        result.update({
            "resume_found": True,
            "resume_source": "explicit",
            "resume_kind": resume_kind,
            "resume_checkpoint": str(resume_checkpoint),
            "resume_step": resume_step,
            "checkpoint_local_dir": str(checkpoint_dir),
        })
        return result

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

    initialization_hf_repo_id = str(getattr(args, "initialization_hf_repo_id", "") or "")
    initialization_local_dir = str(getattr(args, "initialization_local_dir", "") or "")
    initialization_checkpoint_filename = str(getattr(args, "initialization_checkpoint_filename", "") or "")
    has_initialization_source = bool(initialization_local_dir or initialization_hf_repo_id)
    if has_initialization_source and _initialization_mode(args) in INITIALIZATION_SOURCE_MODES:
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
    parser.add_argument("--deadly-action-layout", default="")
    parser.add_argument("--latency-mode", default="")
    parser.add_argument("--source-dataset-hf", default="")
    parser.add_argument("--source-dataset-config-name", default=None)
    parser.add_argument("--dataset-local-dir", required=True)
    parser.add_argument("--converted-dataset-name", default="flappy_train")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--dataset-force-download", default="false")
    parser.add_argument("--setup-force", default="false")
    parser.add_argument("--skip-verification", default="false")
    parser.add_argument("--verify-rows", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episodes-per-latency", type=int, default=None)
    parser.add_argument("--latency-filter", default=None)
    parser.add_argument("--base-model-dir", required=True)
    parser.add_argument("--base-model-repo-id", default=None)
    parser.add_argument("--checkpoint-local-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-load", choices=["auto", "none", "local", "hf"], default="auto")
    parser.add_argument("--checkpoint-hf-repo-id", default="")
    parser.add_argument("--checkpoint-save-best-model", default="true")
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
