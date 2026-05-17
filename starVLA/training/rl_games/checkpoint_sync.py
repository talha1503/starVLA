from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class CheckpointRecord:
    step: int
    path: str


class CheckpointSyncManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._saved: List[CheckpointRecord] = []
        self._sync_enabled = bool(
            getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "enabled", False)
        )
        self._hf_repo_id = getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "repo_id", None)
        self._hf_keep_last_n = int(getattr(getattr(getattr(cfg, "checkpoint", {}), "sync", {}), "keep_last_n", 0))
        self._local_keep_last_n = int(getattr(getattr(getattr(cfg, "checkpoint", {}), "local", {}), "keep_last_n", 0))

    def register_local_checkpoint(self, step: int, path: str) -> None:
        self._saved.append(CheckpointRecord(step=step, path=path))
        self._saved.sort(key=lambda record: record.step)
        self._prune_local_checkpoints()
        if self._sync_enabled and self._hf_repo_id:
            self._sync_to_hf(path=path)
            self._prune_hf_checkpoints()

    def _prune_local_checkpoints(self) -> None:
        if self._local_keep_last_n <= 0:
            return
        while len(self._saved) > self._local_keep_last_n:
            old = self._saved.pop(0)
            if os.path.exists(old.path):
                try:
                    os.remove(old.path)
                except OSError:
                    # Non-fatal: keep training even if cleanup fails.
                    pass

    def _sync_to_hf(self, path: str) -> None:
        # Lazy-import and soft-fail to avoid hard dependency for local-only users.
        try:
            from huggingface_hub import HfApi, upload_file
        except Exception:
            return
        if not os.path.isfile(path):
            return
        try:
            api = HfApi()
            api.create_repo(repo_id=self._hf_repo_id, repo_type="model", exist_ok=True)
            upload_file(
                path_or_fileobj=path,
                path_in_repo=os.path.basename(path),
                repo_id=self._hf_repo_id,
                repo_type="model",
            )
        except Exception:
            # Non-fatal by design.
            return

    def _prune_hf_checkpoints(self) -> None:
        if self._hf_keep_last_n <= 0:
            return
        try:
            from huggingface_hub import HfApi
        except Exception:
            return
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id=self._hf_repo_id, repo_type="model")
            # Keep only files following local checkpoint naming convention.
            step_files = []
            for file_path in files:
                if not file_path.startswith("steps_"):
                    continue
                try:
                    step = int(file_path.split("steps_")[1].split("_")[0])
                except Exception:
                    continue
                step_files.append((step, file_path))
            step_files.sort(key=lambda item: item[0])
            if len(step_files) <= self._hf_keep_last_n:
                return
            to_delete = step_files[: len(step_files) - self._hf_keep_last_n]
            for _, path_in_repo in to_delete:
                api.delete_file(
                    path_in_repo=path_in_repo,
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                )
        except Exception:
            return
