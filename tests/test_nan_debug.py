from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from starVLA.training.nan_debug import NanDebugSession
from starVLA.training import nan_debug as nan_debug_module


class _TraceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torch.nn.Module()
        backbone.language_model = torch.nn.Module()
        backbone.language_model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.qwen_vl_interface = SimpleNamespace(
            model=SimpleNamespace(model=backbone),
        )
        self.action_model = torch.nn.Identity()
        self.weight = torch.nn.Parameter(torch.ones(1))


def _session(
    tmp_path: Path,
    *,
    capture_raw_batch: bool = True,
    trace_modules: bool = False,
    save_state_on_failure: bool = False,
    deepspeed_zero2: bool = False,
    optimizer=None,
) -> NanDebugSession:
    config_path = tmp_path / "config.full.yaml"
    config_path.write_text("debug: true\n", encoding="utf-8")
    (tmp_path / "dataset_statistics.json").write_text("{}\n", encoding="utf-8")
    model = _TraceModel()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return NanDebugSession(
        output_dir=tmp_path,
        output_subdir="nan_debug",
        rank=0,
        capture_raw_batch=capture_raw_batch,
        save_state_on_failure=save_state_on_failure,
        trace_modules=trace_modules,
        deepspeed_zero2=deepspeed_zero2,
        config_path=config_path,
        model=model,
        optimizer=optimizer,
    )


def test_nan_debug_records_first_nonfinite_tensor(tmp_path):
    session = _session(tmp_path)
    batch = [
        {
            "action": torch.zeros(1),
            "_nan_debug": {
                "dataset_name": "flappy",
                "dataset_index": 3,
                "trajectory_id": 1,
                "base_index": 9,
                "raw_data": {"action": torch.zeros(1)},
            },
        }
    ]
    session.begin_step(optimizer_step=7, batch=batch)

    session.inspect("forward.output", {"loss": torch.tensor(float("nan"))})
    with pytest.raises(FloatingPointError, match="forward.output.loss"):
        session.raise_if_nonfinite("forward")

    incident_dir = next(
        (tmp_path / "nan_debug" / "rank_0").glob(
            "optimizer_step_00000007_microstep_00000001_incident_*"
        )
    )
    assert (incident_dir / "incident.json").is_file()
    assert (incident_dir / "batch.pt").is_file()
    assert (incident_dir / "rng_state_before_step.pt").is_file()
    assert (incident_dir / "rng_state_at_failure.pt").is_file()
    assert "_nan_debug" not in batch[0]


def test_nan_debug_leaves_finite_step_without_artifacts(tmp_path):
    session = _session(tmp_path, capture_raw_batch=False)
    session.begin_step(optimizer_step=1, batch=[{"action": torch.zeros(1)}])
    session.inspect("forward.output", {"loss": torch.tensor(1.0)})
    session.raise_if_nonfinite("forward")
    session.inspect_gradients("backward.gradients")
    session.raise_if_nonfinite("backward")
    session.inspect_parameters("optimizer.parameters")
    session.raise_if_nonfinite("optimizer")
    session.finish_step()

    assert not (tmp_path / "nan_debug").exists()


def test_nan_debug_gradient_hook_survives_cleared_parameter_grad(tmp_path):
    session = _session(tmp_path, capture_raw_batch=False)
    session.begin_step(optimizer_step=2, batch=[{"action": torch.zeros(1)}])

    (session.model.weight * torch.tensor(float("nan"))).sum().backward()
    session.model.weight.grad = None

    with pytest.raises(FloatingPointError, match="backward.gradients.weight"):
        session.raise_if_nonfinite("backward")


def test_nan_debug_module_hook_records_first_nonfinite_output(tmp_path):
    session = _session(tmp_path, capture_raw_batch=False, trace_modules=True)
    session.begin_step(optimizer_step=3, batch=[{"action": torch.zeros(1)}])

    session.model.action_model(torch.tensor(float("nan")))

    with pytest.raises(FloatingPointError, match="action_model"):
        session.raise_if_nonfinite("forward")


def test_nan_debug_captures_zero2_accumulation_state(tmp_path):
    partition = SimpleNamespace(grad=torch.tensor([5.0]))
    optimizer = SimpleNamespace(
        micro_step_id=4,
        averaged_gradients={0: [torch.tensor([3.0])]},
        single_partition_of_fp32_groups=[partition],
        state_dict=lambda: {"state": {"step": 2}},
    )
    session = _session(
        tmp_path,
        capture_raw_batch=False,
        save_state_on_failure=True,
        deepspeed_zero2=True,
        optimizer=optimizer,
    )

    session.begin_step(optimizer_step=5, batch=[{"action": torch.zeros(1)}])

    zero2_state = session.train_state_before_step["deepspeed_zero2"]
    assert zero2_state["micro_step_id"] == 4
    assert torch.equal(zero2_state["averaged_gradients"][0][0], torch.tensor([3.0]))
    assert torch.equal(zero2_state["fp32_partition_gradients"][0], torch.tensor([5.0]))


def test_nan_debug_remote_rank_waits_for_incident_persistence(tmp_path, monkeypatch):
    session = _session(tmp_path, capture_raw_batch=False)
    session.begin_step(optimizer_step=6, batch=[{"action": torch.zeros(1)}])
    calls = []

    monkeypatch.setattr(nan_debug_module.dist, "is_initialized", lambda: True)

    def remote_nonfinite(sync_flag, op):
        calls.append(op)
        sync_flag.fill_(1)

    monkeypatch.setattr(nan_debug_module.dist, "all_reduce", remote_nonfinite)

    with pytest.raises(FloatingPointError, match="another rank"):
        session.raise_if_nonfinite("forward")

    assert calls == [
        nan_debug_module.dist.ReduceOp.MAX,
        nan_debug_module.dist.ReduceOp.MAX,
    ]


def test_nan_debug_artifact_write_failure_synchronizes_before_reraising(
    tmp_path,
    monkeypatch,
):
    session = _session(tmp_path, capture_raw_batch=False)
    session.begin_step(optimizer_step=7, batch=[{"action": torch.zeros(1)}])
    session.inspect("forward.output", {"loss": torch.tensor(float("nan"))})
    calls = []

    monkeypatch.setattr(nan_debug_module.dist, "is_initialized", lambda: True)

    def local_nonfinite(_sync_flag, op):
        calls.append(op)

    monkeypatch.setattr(nan_debug_module.dist, "all_reduce", local_nonfinite)

    def fail_write(**_incident):
        raise OSError("disk full")

    monkeypatch.setattr(session, "_write_incident", fail_write)

    with pytest.raises(OSError, match="disk full"):
        session.raise_if_nonfinite("forward")

    assert calls == [
        nan_debug_module.dist.ReduceOp.MAX,
        nan_debug_module.dist.ReduceOp.MAX,
    ]
