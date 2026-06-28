import os
from pathlib import Path
import sys

from torch.utils.data import DataLoader, Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.dataloader.worker_context import CPU_ONLY_WORKER_CONTEXT, _DISTRIBUTED_ENV_KEYS


class _WorkerEnvDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            **{key: os.environ.get(key, "<unset>") for key in _DISTRIBUTED_ENV_KEYS},
        }


def test_cpu_only_worker_context_cleans_spawn_environment_and_restores_parent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        monkeypatch.setenv(key, f"value-{index}")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        multiprocessing_context=CPU_ONLY_WORKER_CONTEXT,
    )
    worker_env = next(iter(loader))

    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    for key in _DISTRIBUTED_ENV_KEYS:
        assert worker_env[key] == ["<unset>"]

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        assert os.environ[key] == f"value-{index}"


def test_cpu_only_worker_context_survives_dataloader_reconstruction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        multiprocessing_context=CPU_ONLY_WORKER_CONTEXT,
    )
    reconstructed = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        multiprocessing_context=loader.multiprocessing_context,
    )

    worker_env = next(iter(reconstructed))

    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
