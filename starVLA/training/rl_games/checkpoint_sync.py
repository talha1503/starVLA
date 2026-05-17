from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckpointRecord:
    step: int
    state_path: Optional[str] = None
    model_path: Optional[str] = None


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

    def register_local_checkpoint(self, step: int, state_path: str | None = None, model_path: str | None = None) -> None:
        self._saved.append(CheckpointRecord(step=step, state_path=state_path, model_path=model_path))
        self._saved.sort(key=lambda record: record.step)
        self._prune_local_checkpoints()
        if self._sync_enabled and self._hf_repo_id:
            self._sync_to_hf(state_path=state_path, model_path=model_path)
            self._prune_hf_checkpoints()

    def _prune_local_checkpoints(self) -> None:
        if self._local_keep_last_n <= 0:
            return
        while len(self._saved) > self._local_keep_last_n:
            old = self._saved.pop(0)
            for path in (old.state_path, old.model_path):
                if not path or not os.path.exists(path):
                    continue
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except OSError:
                    # Non-fatal: keep training even if cleanup fails.
                    pass

    def _sync_to_hf(self, state_path: str | None = None, model_path: str | None = None) -> None:
        # Lazy-import and soft-fail to avoid hard dependency for local-only users.
        try:
            from huggingface_hub import HfApi, upload_file, upload_folder
        except Exception:
            return
        try:
            api = HfApi()
            api.create_repo(repo_id=self._hf_repo_id, repo_type="model", exist_ok=True)
            if state_path and os.path.isdir(state_path):
                upload_folder(
                    folder_path=state_path,
                    path_in_repo=os.path.basename(state_path),
                    repo_id=self._hf_repo_id,
                    repo_type="model",
                )
            if model_path and os.path.isfile(model_path):
                upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=os.path.basename(model_path),
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
            paths_by_step = {}
            for file_path in files:
                first_part = file_path.split("/", 1)[0]
                if first_part.startswith("steps_") and first_part.endswith("_state"):
                    try:
                        step = int(first_part.split("steps_")[1].split("_state")[0])
                    except Exception:
                        continue
                    paths_by_step.setdefault(step, set()).add(file_path)
                    continue
                if os.path.basename(file_path).startswith("steps_"):
                    try:
                        step = int(os.path.basename(file_path).split("steps_")[1].split("_")[0])
                    except Exception:
                        continue
                    paths_by_step.setdefault(step, set()).add(file_path)
            steps = sorted(paths_by_step)
            if len(steps) <= self._hf_keep_last_n:
                return
            for step in steps[: len(steps) - self._hf_keep_last_n]:
                for delete_path in sorted(paths_by_step[step]):
                    api.delete_file(
                        path_in_repo=delete_path,
                        repo_id=self._hf_repo_id,
                        repo_type="model",
                    )
        except Exception:
            return
