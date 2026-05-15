from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.rl_games.checkpoint_sync import CheckpointSyncManager


def test_local_keep_last_n_prunes_old_files(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "local": {"keep_last_n": 2},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            }
        }
    )
    manager = CheckpointSyncManager(cfg=cfg)

    files = []
    for step in [1, 2, 3]:
        path = tmp_path / f"steps_{step}_pytorch_model.pt"
        path.write_text("x", encoding="utf-8")
        files.append(path)
        manager.register_local_checkpoint(step=step, path=str(path))

    assert not files[0].exists()
    assert files[1].exists()
    assert files[2].exists()
