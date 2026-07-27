from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def _iter_numeric_tensors(value: Any, path: str):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            yield path, value
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            yield path, value
        return
    if isinstance(value, (float, np.floating)):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_numeric_tensors(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_numeric_tensors(item, f"{path}[{index}]")


def _nonfinite_summary(
    value: torch.Tensor | np.ndarray | float | np.floating,
) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        finite = torch.isfinite(detached)
        nonfinite_count = int((~finite).sum().item())
        if nonfinite_count == 0:
            return None
        finite_values = detached[finite]
        summary = {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": detached.numel(),
            "nonfinite_count": nonfinite_count,
        }
        if finite_values.numel():
            summary["finite_min"] = float(finite_values.min().float().item())
            summary["finite_max"] = float(finite_values.max().float().item())
        return summary

    if isinstance(value, (float, np.floating)):
        if math.isfinite(value):
            return None
        return {
            "shape": [],
            "dtype": type(value).__name__,
            "device": "cpu",
            "numel": 1,
            "nonfinite_count": 1,
        }

    finite = np.isfinite(value)
    nonfinite_count = int((~finite).sum())
    if nonfinite_count == 0:
        return None
    finite_values = value[finite]
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": "cpu",
        "numel": int(value.size),
        "nonfinite_count": nonfinite_count,
    }
    if finite_values.size:
        summary["finite_min"] = float(finite_values.min())
        summary["finite_max"] = float(finite_values.max())
    return summary


def _cpu_clone(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


class NanDebugSession:
    def __init__(
        self,
        *,
        output_dir: Path,
        output_subdir: str,
        rank: int,
        capture_raw_batch: bool,
        save_state_on_failure: bool,
        trace_modules: bool,
        deepspeed_zero2: bool,
        config_path: Path,
        model: torch.nn.Module,
        optimizer,
    ):
        self.output_dir = output_dir / output_subdir / f"rank_{rank}"
        self.rank = rank
        self.capture_raw_batch = capture_raw_batch
        self.save_state_on_failure = save_state_on_failure
        self.deepspeed_zero2 = deepspeed_zero2
        self.model = model
        self.optimizer = optimizer
        self.config_path = config_path
        self.microstep = 0
        self.optimizer_step = 0
        self.batch = None
        self.raw_batch = None
        self.rng_state_before_step = None
        self.train_state_before_step = None
        self._pending_incident = None
        self._hook_handles = []
        self._attach_gradient_hooks()
        if trace_modules:
            self._attach_module_hooks()

    def _attach_gradient_hooks(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self._hook_handles.append(parameter.register_hook(self._gradient_hook(name)))

    def _gradient_hook(self, name: str):
        def hook(gradient):
            summary = _nonfinite_summary(gradient)
            if summary is not None:
                self._record_incident(
                    location="backward.gradients",
                    tensor_path=f"backward.gradients.{name}",
                    summary=summary,
                )
            return gradient

        return hook

    def _attach_module_hooks(self) -> None:
        backbone = self.model.qwen_vl_interface.model.model
        for index, layer in enumerate(backbone.language_model.layers):
            self._hook_handles.append(
                layer.register_forward_hook(self._module_hook(f"qwen.language_model.layers.{index}"))
            )
        self._hook_handles.append(backbone.register_forward_hook(self._module_hook("qwen.backbone")))
        self._hook_handles.append(
            self.model.action_model.register_forward_hook(self._module_hook("action_model"))
        )

    def _module_hook(self, location: str):
        def hook(_module, _inputs, output):
            self.inspect(location, output)

        return hook

    def begin_step(self, *, optimizer_step: int, batch) -> None:
        self.microstep += 1
        self.optimizer_step = optimizer_step
        self._pending_incident = None
        self.rng_state_before_step = _rng_state()
        if self.save_state_on_failure:
            self.train_state_before_step = {
                "trainable_model": {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                },
                "gradients": {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                },
                "optimizer": _cpu_clone(self.optimizer.state_dict()),
            }
            if self.deepspeed_zero2:
                self.train_state_before_step["deepspeed_zero2"] = {
                    "micro_step_id": self.optimizer.micro_step_id,
                    "averaged_gradients": _cpu_clone(self.optimizer.averaged_gradients),
                    "fp32_partition_gradients": [
                        _cpu_clone(parameter.grad)
                        for parameter in self.optimizer.single_partition_of_fp32_groups
                    ],
                }
        raw_batch = []
        if self.capture_raw_batch:
            for sample in batch:
                raw_batch.append(sample.pop("_nan_debug"))
        self.batch = batch
        self.raw_batch = raw_batch
        if self.capture_raw_batch:
            self.inspect("batch.raw", raw_batch)
        self.inspect("batch.transformed", batch)

    def finish_step(self) -> None:
        self.batch = None
        self.raw_batch = None
        self.rng_state_before_step = None
        self.train_state_before_step = None

    def inspect(self, location: str, value: Any) -> None:
        for tensor_path, tensor in _iter_numeric_tensors(value, location):
            summary = _nonfinite_summary(tensor)
            if summary is not None:
                self._record_incident(
                    location=location,
                    tensor_path=tensor_path,
                    summary=summary,
                )
                return

    def _record_incident(
        self,
        *,
        location: str,
        tensor_path: str,
        summary: dict[str, Any],
    ) -> None:
        if self._pending_incident is None:
            self._pending_incident = {
                "location": location,
                "tensor_path": tensor_path,
                "summary": summary,
            }

    def raise_if_nonfinite(self, checkpoint: str) -> None:
        local_nonfinite = self._pending_incident is not None
        sync_flag = torch.tensor(
            int(local_nonfinite),
            dtype=torch.int32,
            device=next(self.model.parameters()).device,
        )
        if dist.is_initialized():
            dist.all_reduce(sync_flag, op=dist.ReduceOp.MAX)
        if sync_flag.item() == 0:
            return

        incident_dir = None
        write_error = None
        try:
            if self._pending_incident is not None:
                incident_dir = self._write_incident(**self._pending_incident)
        except Exception as error:
            write_error = error
        finally:
            # Peers wait until the reporting rank either finishes persistence or
            # records its write error, then all workers terminate together.
            if dist.is_initialized():
                dist.all_reduce(sync_flag, op=dist.ReduceOp.MAX)

        if write_error is not None:
            raise write_error
        if incident_dir is not None:
            raise FloatingPointError(
                f"Non-finite tensor at {self._pending_incident['tensor_path']}; "
                f"incident saved to {incident_dir}"
            )
        raise FloatingPointError(
            f"Non-finite tensor detected on another rank at {checkpoint}"
        )

    def inspect_gradients(self, location: str) -> None:
        gradients = {
            name: parameter.grad
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        }
        self.inspect(location, gradients)

    def inspect_parameters(self, location: str) -> None:
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.inspect(location, parameters)

    def inspect_optimizer(self, location: str) -> None:
        self.inspect(location, self.optimizer.state_dict())

    def _write_incident(
        self,
        *,
        location: str,
        tensor_path: str,
        summary: dict[str, Any],
    ) -> Path:
        incident_dir = self.output_dir / (
            f"optimizer_step_{self.optimizer_step:08d}_"
            f"microstep_{self.microstep:08d}_incident_{time.time_ns()}"
        )
        incident_dir.mkdir(parents=True, exist_ok=False)
        incident = {
            "rank": self.rank,
            "optimizer_step": self.optimizer_step,
            "microstep": self.microstep,
            "location": location,
            "tensor_path": tensor_path,
            "tensor": summary,
        }
        (incident_dir / "incident.json").write_text(
            json.dumps(incident, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        shutil.copy2(self.config_path, incident_dir / "config.full.yaml")
        shutil.copy2(
            self.config_path.parent / "dataset_statistics.json",
            incident_dir / "dataset_statistics.json",
        )
        torch.save(self.rng_state_before_step, incident_dir / "rng_state_before_step.pt")
        torch.save(_rng_state(), incident_dir / "rng_state_at_failure.pt")
        torch.save(
            {
                "transformed": self.batch,
                "raw": self.raw_batch,
            },
            incident_dir / "batch.pt",
        )
        if self.save_state_on_failure:
            torch.save(self.train_state_before_step, incident_dir / "train_state_before_step.pt")
        return incident_dir

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
