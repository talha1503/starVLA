from pathlib import Path
from types import SimpleNamespace
import inspect

import pytest


def _args(tmp_path: Path, **overrides):
    values = {
        "model": "openvla",
        "env": "demon_attack",
        "mode": "single",
        "dataset_local_dir": str(tmp_path / "datasets"),
        "converted_dataset_name": "demon_attack_train",
        "dataset_force_download": "false",
        "latency_mode": "",
        "converted_dataset_hf": "",
        "base_model_dir": str(tmp_path / "base_model"),
        "base_model_repo_id": None,
        "checkpoint_local_dir": str(tmp_path / "checkpoints"),
        "checkpoint_load": "none",
        "checkpoint_hf_repo_id": "",
        "checkpoint_sync_enabled": "false",
        "checkpoint_sync_repo_id": "",
        "hf_repo_id": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_setup_requires_converted_dataset_without_raw_conversion(monkeypatch, tmp_path: Path):
    from examples.rl_games.scripts import setup_training_assets as setup

    with pytest.raises(FileNotFoundError, match="converted LeRobot dataset is not ready"):
        setup._ensure_rl_games_lerobot_dataset(_args(tmp_path))


def test_setup_downloads_converted_dataset_when_missing(monkeypatch, tmp_path: Path):
    from examples.rl_games.scripts import setup_training_assets as setup

    downloads = []
    validations = []

    def fake_snapshot_download(**kwargs):
        downloads.append(kwargs)
        root = Path(kwargs["local_dir"])
        for name in ("demon_attack_train", "demon_attack_train__val"):
            dataset_dir = root / name
            (dataset_dir / "meta").mkdir(parents=True)
            (dataset_dir / "meta" / "modality.json").write_text("{}", encoding="utf-8")
            (dataset_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
            (dataset_dir / "meta" / "episodes.jsonl").write_text("", encoding="utf-8")
            (dataset_dir / "meta" / "tasks.jsonl").write_text("", encoding="utf-8")
            (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
            (dataset_dir / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"data")

    def fake_validate(data_root_dir, data_mix):
        validations.append((Path(data_root_dir), data_mix))
        return {
            "dataset_stats_path": str(Path(data_root_dir) / data_mix / "dataset_statistics.json"),
            "dataset_num_steps": 1,
            "dataset_num_trajectories": 1,
            "dataset_robot_type": "rl_games_demon_attack",
            "dataset_embodiment_tag": "new_embodiment",
        }

    monkeypatch.setattr(setup, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(setup, "_validate_starvla_dataset", fake_validate)

    result = setup._ensure_rl_games_lerobot_dataset(
        _args(tmp_path, converted_dataset_hf="user/demon_attack_train_lerobot"),
    )

    assert downloads == [
        {
            "repo_id": "user/demon_attack_train_lerobot",
            "repo_type": "dataset",
            "local_dir": str((tmp_path / "datasets").resolve()),
        }
    ]
    assert validations == [
        ((tmp_path / "datasets").resolve(), "demon_attack_train"),
        ((tmp_path / "datasets").resolve(), "demon_attack_train__val"),
    ]
    assert result["dataset_ready"] is True
    assert result["dataset_downloaded"] is True


def test_setup_training_assets_has_clean_converted_dataset_interface():
    from examples.rl_games.scripts import setup_training_assets as setup

    assert list(inspect.signature(setup._ensure_rl_games_lerobot_dataset).parameters) == ["args"]
    source = Path(setup.__file__).read_text(encoding="utf-8")
    bad_fragments = [
        "source" + "-dataset-hf",
        "source" + "_dataset_hf",
        "del " + "convert_dataset",
        "snapshot_download" + " = None",
        "dataset" + "_converted",
        "getattr(args, " + '"converted_dataset_hf"',
        "_ensure_" + "flappy_dataset",
        "_ensure_" + "demon_attack_dataset",
        "_ensure_" + "deadly_corridor_dataset",
        "dataset" + "_cache_dir",
        "verify" + "_rows",
        "max" + "_episodes",
        "latency" + "_raw_frame_filter",
        "setup" + "_force",
    ]
    for fragment in bad_fragments:
        assert fragment not in source


def test_setup_assets_uses_one_lerobot_dataset_path_for_rl_games_envs(monkeypatch, tmp_path: Path):
    from examples.rl_games.scripts import setup_training_assets as setup

    calls = []

    def fake_ensure_dataset(args):
        calls.append((args.model, args.env))
        return {"dataset_ready": True, "data_mix": args.converted_dataset_name}

    monkeypatch.setattr(setup, "_ensure_rl_games_lerobot_dataset", fake_ensure_dataset)
    monkeypatch.setattr(setup, "_ensure_base_model", lambda *args: {"base_model_downloaded": False})

    for env_name in ("flappy", "demon_attack", "deadly_corridor"):
        result = setup.setup_assets(_args(tmp_path, env=env_name))
        assert result["dataset_ready"] is True

    assert calls == [
        ("openvla", "flappy"),
        ("openvla", "demon_attack"),
        ("openvla", "deadly_corridor"),
    ]


def test_run_experiment_no_longer_passes_conversion_controls_to_setup():
    from examples.rl_games.scripts import run_experiment

    source = Path(run_experiment.__file__).read_text(encoding="utf-8")
    bad_fragments = [
        "_dataset" + "_setup_values",
        "_optional" + "_int_list",
        "dataset.max" + "_episodes",
        "dataset.debug" + "_subset",
        "paths.dataset" + "_cache_dir",
        "dataset.verify" + "_rows",
        "dataset.latency" + "_raw_frame_filter",
        "dataset.setup" + "_force",
        "max" + "_episodes",
        "dataset" + "_cache_dir",
        "verify" + "_rows",
        "latency" + "_raw_frame_filter",
        "setup" + "_force",
    ]
    for fragment in bad_fragments:
        assert fragment not in source


def test_run_experiment_explicit_null_leaf_overrides_hydra_default():
    from examples.rl_games.scripts import run_experiment

    cmd = []
    cfg = {
        "datasets": {
            "vla_data": {
                "obs_image_size": None,
            },
        },
    }

    run_experiment._append_hydra_leaf_overrides(cmd, cfg)

    assert "datasets.vla_data.obs_image_size=null" in cmd


def test_run_experiment_omitted_leaf_does_not_override_hydra_default():
    from examples.rl_games.scripts import run_experiment

    cmd = []
    cfg = {
        "datasets": {
            "vla_data": {
                "per_device_batch_size": 16,
            },
        },
    }

    run_experiment._append_hydra_leaf_overrides(cmd, cfg)

    assert "datasets.vla_data.obs_image_size=null" not in cmd
    assert not any(item.startswith("datasets.vla_data.obs_image_size=") for item in cmd)


def test_run_experiment_list_leaf_override_still_uses_hydra_list_syntax():
    from examples.rl_games.scripts import run_experiment

    cmd = []
    cfg = {
        "datasets": {
            "vla_data": {
                "obs_image_size": [84, 84],
            },
        },
    }

    run_experiment._append_hydra_leaf_overrides(cmd, cfg)

    assert "datasets.vla_data.obs_image_size=[84,84]" in cmd


def test_run_experiment_empty_string_leaf_still_does_not_override_hydra_default():
    from examples.rl_games.scripts import run_experiment

    cmd = []
    cfg = {
        "datasets": {
            "vla_data": {
                "obs_image_size": "",
            },
        },
    }

    run_experiment._append_hydra_leaf_overrides(cmd, cfg)

    assert not any(item.startswith("datasets.vla_data.obs_image_size=") for item in cmd)
