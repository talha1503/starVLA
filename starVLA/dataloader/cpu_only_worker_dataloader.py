import os
import multiprocessing.context

from torch.utils.data import DataLoader


class _CpuOnlyWorkerSpawnProcess(multiprocessing.context.SpawnProcess):
    def start(self):
        had_cuda_visible_devices = "CUDA_VISIBLE_DEVICES" in os.environ
        if had_cuda_visible_devices:
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            return super().start()
        finally:
            if had_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]


class CpuOnlyWorkerSpawnContext(multiprocessing.context.SpawnContext):
    Process = _CpuOnlyWorkerSpawnProcess


CPU_ONLY_WORKER_SPAWN_CONTEXT = CpuOnlyWorkerSpawnContext()


class CpuOnlyWorkerDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["multiprocessing_context"] = CPU_ONLY_WORKER_SPAWN_CONTEXT
        super().__init__(*args, **kwargs)
