import os
from pathlib import Path
import sys

from torch.utils.data import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _CudaVisibleDevicesDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return os.environ["CUDA_VISIBLE_DEVICES"]


def test_worker_inherits_empty_cuda_visible_devices_and_parent_is_restored(monkeypatch):
    from starVLA.dataloader.cpu_only_worker_dataloader import CpuOnlyWorkerDataLoader

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    loader = CpuOnlyWorkerDataLoader(
        _CudaVisibleDevicesDataset(),
        batch_size=1,
        num_workers=1,
    )

    assert next(iter(loader)) == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_parent_without_cuda_visible_devices_stays_unset(monkeypatch):
    from starVLA.dataloader.cpu_only_worker_dataloader import CpuOnlyWorkerDataLoader

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    loader = CpuOnlyWorkerDataLoader(
        _CudaVisibleDevicesDataset(),
        batch_size=1,
        num_workers=1,
    )

    assert next(iter(loader)) == [""]
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_cpu_only_worker_context_survives_dataloader_reconstruction(monkeypatch):
    from torch.utils.data import DataLoader

    from starVLA.dataloader.cpu_only_worker_dataloader import CpuOnlyWorkerDataLoader

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    loader = CpuOnlyWorkerDataLoader(
        _CudaVisibleDevicesDataset(),
        batch_size=1,
        num_workers=1,
    )
    reconstructed = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        multiprocessing_context=loader.multiprocessing_context,
    )

    assert next(iter(reconstructed)) == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
