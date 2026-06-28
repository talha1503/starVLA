import os

# Distributed coordinator env vars. A DataLoader worker has no business joining the
# process group, so we drop its inherited copy before any data is fetched. This only
# affects the worker process -- the parent (and its DeepSpeed init) is never touched.
_DISTRIBUTED_ENV_KEYS = (
    "LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    "GROUP_RANK", "LOCAL_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
)


def cpu_only_worker_init(worker_id):
    """DataLoader worker_init_fn that keeps spawn workers CPU-only.

    Runs inside each worker before the first batch is fetched. Hides all GPUs (so no
    CUDA context is ever created) and removes the distributed env (so the worker never
    joins the process group). Safe because the worker's data path is pure CPU -- nothing
    touches CUDA before this runs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in _DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)
