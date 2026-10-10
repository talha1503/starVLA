import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from examples.MIKASA.scripts import prepare_mixed_latency_h1 as prepare


def write_source(root, count, offset):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/shared.mp4").write_bytes(b"unchanged encoded camera frames")
    info = {"total_episodes": count, "total_frames": count * 2, "splits": {"train": f"0:{count}"}}
    (root / "meta/info.json").write_text(json.dumps(info))
    contract = {"action_horizon": 1, "state_dim": 7, "action_dim": 7, "robot_type": "latency_mikasa_h1"}
    (root / "task_contract.json").write_text(json.dumps(contract))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["original instruction"], name="task")).to_parquet(
        root / "meta/tasks.parquet"
    )
    for file_index, first in enumerate(range(0, count, 125)):
        ids = list(range(first, min(first + 125, count)))
        pd.DataFrame(
            {
                "episode_index": ids,
                "length": [2] * len(ids),
                "data/chunk_index": [0] * len(ids),
                "data/file_index": [file_index] * len(ids),
                "videos/observation.images.top/from_timestamp": [index / 10 for index in ids],
            }
        ).to_parquet(root / f"meta/episodes/chunk-000/file-{file_index:03d}.parquet", index=False)
        rows = [
            {
                "episode_index": index,
                "index": index * 2 + step,
                "observation.state": [offset + index] * 7,
                "action": [index / 250] * 7,
            }
            for index in ids
            for step in range(2)
        ]
        pd.DataFrame(rows).to_parquet(root / f"data/chunk-000/file-{file_index:03d}.parquet", index=False)


def test_main_subsets_existing_hub_source_without_validation_leakage(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    for latency in range(5):
        source = snapshot / (
            f"fixed-latency-l{latency}/mikasa-intercept-grab-fast/" "teacher-rollouts-h1-success-250train-25val"
        )
        write_source(source / "lerobot", 250, latency * 1000)
        write_source(source / "lerobot__val", 25, 100000)
    original_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshot.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(
        prepare, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha="pinned"))
    )

    def download(**kwargs):
        assert kwargs["revision"] == "pinned"
        assert all("250train-25val" in pattern for pattern in kwargs["allow_patterns"])
        return str(snapshot)

    monkeypatch.setattr(prepare, "snapshot_download", download)
    output = tmp_path / "selected"
    prepare.main(
        [
            "--output",
            str(output),
            "--run-id",
            "subset50",
            "--run-root",
            str(tmp_path / "runs"),
            "--num-gpus",
            "1",
            "--train-episodes",
            "50",
            "--val-episodes",
            "25",
            "--backbone-path",
            str(tmp_path / "backbone"),
            "--model-hf-repo",
            "test/subset50",
            "--vla-python",
            "python",
        ]
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["counts"]["train_episodes"] == 250
    assert provenance["counts"]["validation_episodes"] == 125
    assert provenance["counts"]["train_frames"] == 500
    normalization = provenance["counts"]["normalization"]
    assert normalization["state"]["max"] == [4049.0] * 7
    assert normalization["state"]["mean"] == [2024.5] * 7
    for latency in range(5):
        train = output / f"fixed_l{latency}/lerobot"
        info = json.loads((train / "meta/info.json").read_text())
        assert (info["total_episodes"], info["total_frames"], info["splits"]) == (50, 100, {"train": "0:50"})
        episodes = pd.concat([pd.read_parquet(path) for path in (train / "meta/episodes").glob("*/*.parquet")])
        assert episodes["episode_index"].tolist() == list(range(50))
        assert episodes["videos/observation.images.top/from_timestamp"].tolist() == [index / 10 for index in range(50)]
        rows = pq.read_table(train / "data/chunk-000/file-000.parquet")
        assert rows.num_rows == 100
        assert set(rows["episode_index"].to_pylist()) == set(range(50))
        assert not (train / "data/chunk-000/file-001.parquet").exists()
        assert (train / "videos/shared.mp4").read_bytes() == b"unchanged encoded camera frames"
        assert pd.read_parquet(train / "meta/tasks.parquet").index[0] == prepare.latency_prompt(latency)
        val = output / f"fixed_l{latency}/lerobot__val"
        assert json.loads((val / "meta/info.json").read_text())["total_episodes"] == 25
        assert json.loads((val / "task_contract.json").read_text())["normalization"] == normalization
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in original_hashes.items())


def test_full_source_budget_retains_all_episodes(tmp_path):
    source = tmp_path / "source"
    write_source(source / "lerobot", 250, 0)
    write_source(source / "lerobot__val", 25, 100000)
    output = tmp_path / "output"
    counts = prepare.prepare_datasets({0: source}, output, latencies=[0])
    assert counts["train_episodes"] == 250
    assert counts["validation_episodes"] == 25
    assert counts["train_frames"] == 500
    assert counts["normalization"]["state"]["max"] == [249.0] * 7
    assert counts["selected_episode_indices"]["0"]["train"] == list(range(250))
    assert (output / "fixed_l0/lerobot/data/chunk-000/file-001.parquet").is_file()


def test_rejects_unavailable_count_and_existing_subset_destination(tmp_path):
    source = tmp_path / "source"
    write_source(source / "lerobot", 250, 0)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source has 250"):
        prepare.prepare_datasets({0: source}, output, latencies=[0], train_episodes=251)
    (output / "fixed_l0/lerobot").mkdir(parents=True)
    sentinel = output / "fixed_l0/lerobot/sentinel"
    sentinel.write_text("existing experiment")
    with pytest.raises(ValueError, match=r"fresh dataset\.converted_name"):
        prepare.prepare_datasets({0: source}, output, latencies=[0], train_episodes=50)
    assert sentinel.read_text() == "existing experiment"
