import os
from pathlib import Path
import sys

from torch.utils.data import DataLoader, Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.dataloader.worker_init import _DISTRIBUTED_ENV_KEYS, cpu_only_worker_init


class _WorkerEnvDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            **{key: os.environ.get(key, "<unset>") for key in _DISTRIBUTED_ENV_KEYS},
        }


def test_cpu_only_worker_init_hides_gpus_and_drops_distributed_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        monkeypatch.setenv(key, f"value-{index}")

    cpu_only_worker_init(0)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    for key in _DISTRIBUTED_ENV_KEYS:
        assert key not in os.environ


def test_worker_is_cpu_only_and_parent_env_is_untouched(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        monkeypatch.setenv(key, f"value-{index}")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
        worker_init_fn=cpu_only_worker_init,
    )
    worker_env = next(iter(loader))

    # The worker_init_fn ran inside the worker before __getitem__.
    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    for key in _DISTRIBUTED_ENV_KEYS:
        assert worker_env[key] == ["<unset>"]

    # The parent process env is never touched.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        assert os.environ[key] == f"value-{index}"
