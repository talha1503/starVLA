# MIKASA InterceptGrabFast mixed-latency H1

These two scripts train the paper's OpenVLA implementation (QwenOFT) on the
successful H1 datasets collected at 0, 1, 2, 3, and 4 frames of latency.
At the native 20 Hz cadence these delays are 0, 50, 100, 150, and 200 ms.

- `mikasa_mixed_01234.sh`: include the known latency in the instruction.
- `mikasa_mixed_01234_no_latency_prompt.sh`: remove only the latency instruction
  suffix during training and evaluation.

Each latency supplies 250 training episodes and 25 validation episodes, for
1,250/125 episodes total. All five datasets must be packaged before launching.
Both scripts follow the Demon Attack layout: install the OpenVLA environment,
activate it, and run one Python launcher with explicit `key=value` settings.
MIKASA uses `examples/MIKASA/scripts/launch_train.py` for packaged LeRobot data
and its native simulator evaluation.

```bash
cd /home/ubuntu/talha/starVLA
export HF_TOKEN=your_token
export WANDB_API_KEY=your_key

CUDA_VISIBLE_DEVICES=0 bash examples/rl_games/bash_scripts/openvla/bridge/latency_ablation/mikasa_mixed_01234.sh
CUDA_VISIBLE_DEVICES=0 bash examples/rl_games/bash_scripts/openvla/bridge/latency_ablation/mikasa_mixed_01234_no_latency_prompt.sh
```

The two files contain the full training command and have separate run IDs,
dataset preparation folders, and model repositories. Edit the visible settings
or append dot-list overrides. Training runs directly with Python on the selected
GPU using `trainer.distributed_backend=none`. Keep GPU count at 1 and retain
global batch 128 by changing microbatch and accumulation together, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/rl_games/bash_scripts/openvla/bridge/latency_ablation/mikasa_mixed_01234.sh \
    datasets.vla_data.per_device_batch_size=4 trainer.gradient_accumulation_steps=32
```

Useful overrides:

| Setting | Default / purpose |
| --- | --- |
| `workspace_dir` | Parent of the StarVLA checkout |
| `mikasa.benchmark_root` | `${workspace_dir}/latency-aware-agents` |
| `mikasa.task_python` | Benchmark's MIKASA simulator interpreter |
| `dataset.local_source_root` | Use local `outputs/mikasa_successful_h1` instead of HF |
| `dataset.source_hf` | `latency-sensitive-bench/benchmark-datasets` |
| `dataset.revision` | Resolve `main` to an immutable SHA and record it |
| `paths.base_model_dir` | Existing copy of the pinned Qwen3-VL-4B-Instruct backbone |
| `paths.dataset_local_dir` | Parent directory for prepared datasets |
| `run_id`, `run_root_dir` | Training identity and checkpoint root |
| `checkpoint.sync.repo_id` | Model upload destination |
| `wandb_entity`, `wandb_project` | `talha1503`, `starvla_tasks` |
| `launch.num_processes` | 1; direct single-GPU training |
| `datasets.vla_data.per_device_batch_size` | 16 |
| `trainer.gradient_accumulation_steps` | 8; GPU count x microbatch x accumulation must be 128 |
| `framework.qwenvl.enable_gradient_checkpointing` | false, matching the original recipe |
| `rl_games.env_eval.post_train.enabled` | true; evaluate after training |
| `rl_games.env_eval.post_train.num_episodes` | 200 per latency |
| `launch.dry_run` | false; true prepares datasets/configs without training |

HF inputs live under
`fixed-latency-lN/mikasa-intercept-grab-fast/teacher-rollouts-h1-success-250train-25val/{lerobot,lerobot__val}`.
The preparer copies the source files into run-specific assets, adds latency
instructions, and computes one shared normalization from training frames only.
Both validation and evaluation use that same normalization. Source data is preserved.

## Exact baseline provenance

The reference is the solid H1 QwenOFT curve in
`overleaf_latest_paper/figures/fig2_vla_degradation`, identified by:

- Model repo `latency-sensitive-bench/benchmark-models`, revision
  `56b2c095d76ea5b10dd304e6e03c2ca81801630b`, path
  `zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z`.
- [Published H1 inference config](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/56b2c095d76ea5b10dd304e6e03c2ca81801630b/zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z/config.yaml).
- [Original training log](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks/runs/orf7ua9j/files):
  `output.log` records 5,000 optimizer steps, per-device batch 16,
  accumulation 1, total batch 128, and the exact checkpoint path above.
- Benchmark code revision `2f18d4620afd1861059d8cecfc2b1410a4a4777d`,
  `scripts/starvla/train_task.py::training_config` and `scripts/starvla/model_config.py`.
- Backbone `Qwen/Qwen3-VL-4B-Instruct` at
  `ebb281ec70b05090aa6165b016eac8ec08e71b17`.

The recipe retains fresh action-head initialization, continuous 7D state
projection, two camera views resized to 224×224, action horizon 1, no past/future
action window, L1 loss, seed 0, 5,000 updates, 100 warmup steps, global batch 128,
AdamW, learning rates 3e-5 for the backbone and 1e-4 for the head, cosine decay
to 1e-6, betas 0.9/0.95, and weight decay 1e-8. The original baseline used eight
GPUs; these scripts use direct single-GPU training without DeepSpeed, as requested,
with accumulation retaining global batch 128. They initialize from
the pinned language/vision backbone with a fresh head, with no task checkpoint load.

After training, native evaluation runs L0–L4 with 200 episodes each, seed start
4242424242, 60 transitions, capacity 1, 10 parallel environments and inference
batch 8. Results, the final checkpoint, and provenance are uploaded to the run's
model repository. These scripts launch training; creating them does not run it.
