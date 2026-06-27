import os
import multiprocessing.context

from torch.utils.data import DataLoader


_DISTRIBUTED_ENV_KEYS = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "LOCAL_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


class _CpuOnlyWorkerSpawnProcess(multiprocessing.context.SpawnProcess):
    def start(self):
        had_cuda_visible_devices = "CUDA_VISIBLE_DEVICES" in os.environ
        if had_cuda_visible_devices:
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        distributed_env = {key: os.environ[key] for key in _DISTRIBUTED_ENV_KEYS if key in os.environ}

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for key in distributed_env:
            del os.environ[key]
        try:
            return super().start()
        finally:
            if had_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            for key, value in distributed_env.items():
                os.environ[key] = value


class CpuOnlyWorkerSpawnContext(multiprocessing.context.SpawnContext):
    Process = _CpuOnlyWorkerSpawnProcess


CPU_ONLY_WORKER_SPAWN_CONTEXT = CpuOnlyWorkerSpawnContext()


class CpuOnlyWorkerDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["multiprocessing_context"] = CPU_ONLY_WORKER_SPAWN_CONTEXT
        super().__init__(*args, **kwargs)
