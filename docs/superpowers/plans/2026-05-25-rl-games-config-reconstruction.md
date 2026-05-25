# RL Games Config Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the parallel RL-games experiment YAML system with one Hydra configuration tree composed from `model`, `env`, `init`, and `mode`.

**Architecture:** Hydra owns the canonical configuration. `setup_training_assets.py` remains the asset-preparation engine, but a new launcher derives its setup namespace from the composed Hydra config instead of from `examples/rl_games/experiments/*.yaml`. Shell commands become thin launch wrappers that pass Hydra group selections and machine-specific overrides.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, Bash, Pytest, existing StarVLA RL-games training scripts.

---

## File Structure

- Modify `pyproject.toml`: add Hydra to the dev validation environment.
- Modify `examples/rl_games/config/train.yaml`: make the defaults list compose `base/common`, `base/runtime`, `model`, `env`, `init`, `mode`, and `checkpoint`.
- Create `examples/rl_games/config/base/runtime.yaml`: runtime/path/auth/launch defaults.
- Modify `examples/rl_games/config/base/common.yaml`: shared framework, dataset, trainer, and env-eval defaults only.
- Create `examples/rl_games/config/model/base.yaml`: model-shared defaults.
- Modify `examples/rl_games/config/model/{openvla,pi0,pi05,gr00t}.yaml`: add `model/base` inheritance and keep only model-owned values.
- Create `examples/rl_games/config/env/base.yaml`: env-shared defaults.
- Modify `examples/rl_games/config/env/{flappy,demon_attack,deadly_corridor}.yaml`: add `env/base` inheritance and keep only env-owned values.
- Create `examples/rl_games/config/init/{base,scratch,bridge}.yaml`: initialization semantics.
- Create `examples/rl_games/config/mode/base.yaml`: latency-mode shared defaults.
- Modify `examples/rl_games/config/mode/{single,mixed_latency}.yaml`: add `mode/base` inheritance and keep only mode-owned values.
- Modify `examples/rl_games/config/checkpoint/default.yaml`: add resume-source fields used by asset setup.
- Delete `examples/rl_games/config/mode/pre-trained.yaml`: replaced by `init=bridge`.
- Delete `examples/rl_games/config/mode/cross_task.yaml`: out of current supported matrix.
- Create `starVLA/training/rl_games/config_validation.py`: explicit cross-group validation.
- Modify `starVLA/training/train_starvla.py`: call config validation before alias/action-spec normalization.
- Create `examples/rl_games/scripts/launch_train.py`: compose the Hydra config, prepare assets, and launch training.
- Delete `examples/rl_games/scripts/run_experiment.py`: removes the old experiment-YAML translator.
- Delete `examples/rl_games/scripts/run_experiment.sh`: removes the old experiment-YAML shell entrypoint.
- Modify `examples/rl_games/scripts/run_train.sh`: reduce to a thin wrapper around the new launcher or remove if commands no longer need it.
- Delete `examples/rl_games/experiments/`: remove the active second configuration tree.
- Modify `commands/train_*.sh`: call the new launcher with Hydra group selections.
- Modify `tests/rl_games/test_training_commands.py`: assert command wrappers no longer reference `experiments/`.
- Modify `tests/rl_games/test_pi05_flappy_sft_path.py`: replace experiment-YAML assertions with composed-Hydra assertions and launcher tests.
- Create `tests/rl_games/test_config_composition.py`: matrix coverage for supported Hydra compositions.
- Create `tests/rl_games/test_config_validation.py`: cross-group validation tests.
- Modify `examples/rl_games/README.md` and top-level `README.md`: document the canonical Hydra path.

## Task 1: Add Hydra Composition Tests First

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/rl_games/test_config_composition.py`

- [ ] **Step 1: Add Hydra to the dev validation dependencies**

Edit `pyproject.toml` so the `dev` extra includes Hydra:

```toml
dev = [
    "black>=24.2.0",
    "gpustat",
    "hydra-core>=1.3.2",
    "ipython",
    "pre-commit",
    "ruff>=0.2.2",
]
```

- [ ] **Step 2: Write the failing config composition test**

Create `tests/rl_games/test_config_composition.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from starVLA.training.rl_games.action_spec import apply_action_spec
from starVLA.training.rl_games.alias import apply_model_alias
from starVLA.training.rl_games.config_validation import validate_rl_games_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"


@dataclass(frozen=True)
class ExpectedComposition:
    model: str
    env: str
    init: str
    mode: str
    model_alias: str
    framework_name: str
    task: str
    action_carrier: str
    latency_values: tuple[int, ...]
    data_mix: str
    source_hf: str
    action_env_dim: int
    base_model_repo_id: str
    initialization_hf_repo_id: str | None


SUPPORTED_COMPOSITIONS: tuple[ExpectedComposition, ...] = (
    ExpectedComposition(
        model="openvla",
        env="flappy",
        init="scratch",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="flappy",
        action_carrier="native",
        latency_values=(0,),
        data_mix="flappy_train",
        source_hf="talha1503/flappy_bird_zero_latency_parquet",
        action_env_dim=2,
        base_model_repo_id="StarVLA/Qwen3-VL-4B-Instruct-Action",
        initialization_hf_repo_id=None,
    ),
    ExpectedComposition(
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="flappy",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="flappy_train",
        source_hf="talha1503/flappy_bird_zero_latency_parquet",
        action_env_dim=2,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-OFT-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="pi0",
        env="demon_attack",
        init="bridge",
        mode="single",
        model_alias="pi-0",
        framework_name="QwenPI",
        task="demon_attack",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="demon_attack_train",
        source_hf="talha1503/demon_attack_zero_latency_parquet",
        action_env_dim=6,
        base_model_repo_id="StarVLA/Qwen2.5-VL-3B-Instruct-Action",
        initialization_hf_repo_id="StarVLA/Qwen-PI-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="pi05",
        env="deadly_corridor",
        init="bridge",
        mode="mixed_latency",
        model_alias="pi-0.5",
        framework_name="QwenPI_v3",
        task="deadly_corridor",
        action_carrier="bridge",
        latency_values=(0, 1, 2, 3, 4, 5),
        data_mix="deadly_corridor_mixed_latency_train",
        source_hf="latency-sensitive-bench/deadly_corridor_mixed_latency_parquet",
        action_env_dim=7,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
    ),
    ExpectedComposition(
        model="gr00t",
        env="deadly_corridor",
        init="scratch",
        mode="mixed_latency",
        model_alias="gr00t",
        framework_name="QwenGR00T",
        task="deadly_corridor",
        action_carrier="native",
        latency_values=(0, 1, 2, 3, 4, 5),
        data_mix="deadly_corridor_mixed_latency_train",
        source_hf="latency-sensitive-bench/deadly_corridor_mixed_latency_parquet",
        action_env_dim=7,
        base_model_repo_id="StarVLA/Qwen3-VL-4B-Instruct-Action",
        initialization_hf_repo_id=None,
    ),
)


def _compose_cfg(expected: ExpectedComposition) -> DictConfig:
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"model={expected.model}",
                f"env={expected.env}",
                f"init={expected.init}",
                f"mode={expected.mode}",
            ],
        )
    validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    return cfg


@pytest.mark.parametrize("expected", SUPPORTED_COMPOSITIONS)
def test_supported_rl_games_config_composes(expected: ExpectedComposition) -> None:
    cfg = _compose_cfg(expected)

    assert cfg.rl_games.model_alias == expected.model_alias
    assert cfg.framework.name == expected.framework_name
    assert cfg.rl_games.task == expected.task
    assert cfg.rl_games.action_carrier == expected.action_carrier
    assert tuple(OmegaConf.to_container(cfg.rl_games.env_eval.latency.values, resolve=True)) == expected.latency_values
    assert cfg.datasets.vla_data.data_mix == expected.data_mix
    assert cfg.dataset.source_hf == expected.source_hf
    assert cfg.framework.action_model.action_env_dim == expected.action_env_dim
    assert cfg.base_model.repo_id == expected.base_model_repo_id
    assert cfg.initialization.checkpoint_hf_repo_id == expected.initialization_hf_repo_id
```

- [ ] **Step 3: Run the new test and verify it fails for missing config/validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_config_composition.py -q
```

Expected: FAIL because `starVLA.training.rl_games.config_validation` and the new `init`/base config groups do not exist yet.

- [ ] **Step 4: Commit the failing test**

Run:

```bash
git add pyproject.toml tests/rl_games/test_config_composition.py
git commit -m "test: cover rl-games hydra config composition"
```

## Task 2: Rebuild the Hydra Config Tree

**Files:**
- Modify: `examples/rl_games/config/train.yaml`
- Modify: `examples/rl_games/config/base/common.yaml`
- Create: `examples/rl_games/config/base/runtime.yaml`
- Create: `examples/rl_games/config/model/base.yaml`
- Modify: `examples/rl_games/config/model/{openvla,pi0,pi05,gr00t}.yaml`
- Create: `examples/rl_games/config/env/base.yaml`
- Modify: `examples/rl_games/config/env/{flappy,demon_attack,deadly_corridor}.yaml`
- Create: `examples/rl_games/config/init/{base,scratch,bridge}.yaml`
- Create: `examples/rl_games/config/mode/base.yaml`
- Modify: `examples/rl_games/config/mode/{single,mixed_latency}.yaml`

- [ ] **Step 1: Replace the train defaults list**

Update `examples/rl_games/config/train.yaml` to begin with:

```yaml
defaults:
  - base/common
  - base/runtime
  - model: openvla
  - env: flappy
  - init: scratch
  - mode: single
  - checkpoint: default
  - _self_
```

Keep only these top-level fields in `train.yaml`:

```yaml
hydra:
  run:
    dir: ${run_root_dir}/${run_id}/hydra
  job:
    chdir: false

run_id: starvla_rl_games
output_dir: null
config_yaml: null
is_debug: false
version_id: "0.21"
```

- [ ] **Step 2: Move runtime fields into `base/runtime.yaml`**

Create `examples/rl_games/config/base/runtime.yaml`:

```yaml
# @package _global_

workspace_dir: WORKSPACE_DIR
run_root_dir: results/Checkpoints
seed: 42
wandb_entity: your_wandb_entity
wandb_project: starVLA_rl_games

auth:
  env_file: null
  hf_token_env: HF_TOKEN
  wandb_api_key_env: WANDB_API_KEY

paths:
  run_root_dir: results/Checkpoints
  dataset_local_dir: playground/Datasets/rl_games
  dataset_cache_dir: null
  base_model_dir: playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
  accelerate_config: starVLA/config/deepseeds/deepspeed_zero2.yaml

launch:
  use_accelerate: true
  gpus: null
  num_processes: 1
  dry_run: false

conda:
  enabled: true
  env_name: null
```

- [ ] **Step 3: Move shared training defaults into `base/common.yaml`**

Replace `examples/rl_games/config/base/common.yaml` with:

```yaml
# @package _global_

framework:
  qwenvl:
    attn_implementation: flash_attention_2
    enable_gradient_checkpointing: true
  action_model:
    state_dim: 1
    loss_type: l1
    action_horizon: 1
    future_action_window_size: 0
    past_action_window_size: 0

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    include_state: false
    data_root_dir: playground/Datasets/rl_games
    data_mix: ${dataset.converted_name}
    eval_data_mix: null
    action_type: discrete
    sequential_step_sampling: false
    per_device_batch_size: 4
    load_all_data_for_training: true
    obs_image_size: [84, 84]
    video_backend: torchvision_av

dataset:
  source_hf: ""
  converted_name: flappy_train
  single_source_hf: ""
  mixed_source_hf: ""
  single_converted_name: flappy_train
  mixed_converted_name: flappy_mixed_latency_train
  single_latency_filter: null
  mixed_latency_filter: null
  force_download: false
  setup_force: false
  verify_rows: 200
  max_episodes: null
  latency_filter: null
  debug_subset:
    enabled: false
    max_episodes: 5
    suffix: debug

base_model:
  repo_id: StarVLA/Qwen3-VL-4B-Instruct-Action

initialization:
  checkpoint_local_dir: null
  checkpoint_hf_repo_id: null
  checkpoint_filename: null

trainer:
  max_train_steps: 2000
  num_warmup_steps: 100
  save_interval: 100
  eval_interval: 100
  eval_num_batches: 20
  learning_rate:
    base: 2.0e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: ''
  loss_scale:
    vla: 1.0
    vlm: 0.1
  max_grad_norm: 1.0
  weight_decay: 0.0
  logging_frequency: 10
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1
  distributed_backend: deepspeed
  is_resume: false
  pretrained_checkpoint: null
  resume_step: 0
  reload_modules: null
  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08
    fused: false
  save_format: pt
```

- [ ] **Step 4: Add model base inheritance**

Create `examples/rl_games/config/model/base.yaml`:

```yaml
# @package _global_

framework:
  qwenvl:
    attn_implementation: flash_attention_2
    enable_gradient_checkpointing: true
  action_model:
    action_horizon: 1
    future_action_window_size: 0
    past_action_window_size: 0
```

Add this defaults block to the top of each `examples/rl_games/config/model/*.yaml` file:

```yaml
defaults:
  - base
  - _self_
```

Then keep the existing model-specific bodies. Do not move env-specific action dimensions into model files.

Add the top-level selector to each model file:

```yaml
model: openvla
```

Use the matching value in each file: `openvla`, `pi0`, `pi05`, or `gr00t`.

- [ ] **Step 5: Add env base inheritance and dataset ownership**

Create `examples/rl_games/config/env/base.yaml`:

```yaml
# @package _global_

rl_games:
  env_eval:
    image_size: 84
    frameskip: 1
    task_description: ""

datasets:
  vla_data:
    action_type: discrete
    obs_image_size: [84, 84]
```

Update `examples/rl_games/config/env/flappy.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  task: flappy
  env_eval:
    image_size: 84
    frameskip: 1
    task_description: ""

dataset:
  single_source_hf: talha1503/flappy_bird_zero_latency_parquet
  mixed_source_hf: talha1503/flappy_bird_mixed_latency_parquet
  single_converted_name: flappy_train
  mixed_converted_name: flappy_mixed_latency_train
  single_latency_filter: null
  mixed_latency_filter: null

env: flappy
```

Update `examples/rl_games/config/env/demon_attack.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  task: demon_attack
  env_eval:
    image_size: 84
    frameskip: 4
    task_description: "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE."

dataset:
  single_source_hf: talha1503/demon_attack_zero_latency_parquet
  mixed_source_hf: talha1503/demon_attack_mixed_latency_parquet
  single_converted_name: demon_attack_train
  mixed_converted_name: demon_attack_mixed_latency_train
  single_latency_filter: null
  mixed_latency_filter: null

env: demon_attack
```

Update `examples/rl_games/config/env/deadly_corridor.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  task: deadly_corridor
  env_eval:
    image_size: 84
    frameskip: 4
    task_description: "You are playing Deadly Corridor in VizDoom. Choose actions from MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK."
    deadly:
      action_layout: multibinary_7

dataset:
  single_source_hf: latency-sensitive-bench/deadly_corridor_mixed_latency_parquet
  mixed_source_hf: latency-sensitive-bench/deadly_corridor_mixed_latency_parquet
  single_converted_name: deadly_corridor_train
  mixed_converted_name: deadly_corridor_mixed_latency_train
  single_latency_filter: [0]
  mixed_latency_filter: null

env: deadly_corridor
```

- [ ] **Step 6: Add init config group**

Create `examples/rl_games/config/init/base.yaml`:

```yaml
# @package _global_

rl_games:
  initialization_mode: scratch
  action_carrier: native

init: scratch
```

Create `examples/rl_games/config/init/scratch.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  initialization_mode: scratch
  action_carrier: native

datasets:
  vla_data:
    include_state: false

init: scratch
```

Create `examples/rl_games/config/init/bridge.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  initialization_mode: bridge
  action_carrier: bridge

datasets:
  vla_data:
    include_state: true

framework:
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 1
    future_action_window_size: 0
    past_action_window_size: 0

init: bridge
```

- [ ] **Step 7: Add mode base inheritance**

Create `examples/rl_games/config/mode/base.yaml`:

```yaml
# @package _global_

rl_games:
  env_eval:
    enabled: true
    interval_steps: 100
    num_episodes: 5
    max_episode_steps: 3600
    latency:
      prompt_map_path: null
    mid_train:
      enabled: true
      interval_steps: 100
      latencies: null
      num_episodes: 5
      max_steps_per_episode: 3600
    post_train:
      enabled: true
      latencies: null
      num_episodes: 5
      max_steps_per_episode: 3600

mode: single
```

Update `examples/rl_games/config/mode/single.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  env_eval:
    latency:
      mode: single
      values: [0]
    mid_train:
      latencies: [0]
    post_train:
      latencies: [0]

dataset:
  source_hf: ${dataset.single_source_hf}
  converted_name: ${dataset.single_converted_name}
  latency_filter: ${dataset.single_latency_filter}

datasets:
  vla_data:
    data_mix: ${dataset.converted_name}

mode: single
```

Update `examples/rl_games/config/mode/mixed_latency.yaml`:

```yaml
defaults:
  - base
  - _self_

# @package _global_

rl_games:
  env_eval:
    latency:
      mode: mixed
      values: [0, 1, 2, 3, 4, 5]
      prompt_map_path: null
    mid_train:
      latencies: [0, 1, 2, 3, 4, 5]
    post_train:
      latencies: [0, 1, 2, 3, 4, 5]

dataset:
  source_hf: ${dataset.mixed_source_hf}
  converted_name: ${dataset.mixed_converted_name}
  latency_filter: ${dataset.mixed_latency_filter}

datasets:
  vla_data:
    data_mix: ${dataset.converted_name}

mode: mixed_latency
```

- [ ] **Step 8: Add checkpoint resume-source fields**

Update `examples/rl_games/config/checkpoint/default.yaml`:

```yaml
# @package _global_

checkpoint:
  load: auto
  hf_repo_id: null
  local:
    keep_last_n: 3
  sync:
    enabled: false
    repo_id: null
    keep_last_n: 0
    sync_every_n_checkpoints: 1
    resume_policy: local_latest
```

- [ ] **Step 9: Run the composition test and inspect failures**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_config_composition.py -q
```

Expected: FAIL because validation is not implemented.

- [ ] **Step 10: Commit config tree changes**

Run:

```bash
git add examples/rl_games/config pyproject.toml
git commit -m "refactor: split rl-games hydra config groups"
```

## Task 3: Add Explicit Cross-Group Validation

**Files:**
- Create: `starVLA/training/rl_games/config_validation.py`
- Modify: `starVLA/training/rl_games/__init__.py`
- Modify: `starVLA/training/train_starvla.py`
- Create: `tests/rl_games/test_config_validation.py`

- [ ] **Step 1: Write validation tests**

Create `tests/rl_games/test_config_validation.py`:

```python
from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from starVLA.training.rl_games.config_validation import validate_rl_games_config


def test_validate_rejects_pi05_scratch() -> None:
    cfg = OmegaConf.create({
        "rl_games": {
            "model_alias": "pi-0.5",
            "task": "flappy",
            "initialization_mode": "scratch",
            "action_carrier": "native",
            "env_eval": {
                "latency": {"values": [0]},
                "deadly": {"action_layout": "multibinary_7"},
            },
        },
        "dataset": {"source_hf": "source", "converted_name": "flappy_train"},
        "base_model": {"repo_id": "base"},
        "initialization": {
            "checkpoint_local_dir": None,
            "checkpoint_hf_repo_id": None,
            "checkpoint_filename": None,
        },
        "framework": {"action_model": {"action_dim": 32}},
    })

    with pytest.raises(ValueError, match="pi-0.5 scratch is not supported"):
        validate_rl_games_config(cfg)


def test_validate_rejects_bridge_without_checkpoint() -> None:
    cfg = OmegaConf.create({
        "rl_games": {
            "model_alias": "openvla",
            "task": "flappy",
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
            "env_eval": {
                "latency": {"values": [0]},
                "deadly": {"action_layout": "multibinary_7"},
            },
        },
        "dataset": {"source_hf": "source", "converted_name": "flappy_train"},
        "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
        "initialization": {
            "checkpoint_local_dir": None,
            "checkpoint_hf_repo_id": None,
            "checkpoint_filename": None,
        },
        "framework": {"action_model": {"action_dim": 7}},
    })

    with pytest.raises(ValueError, match="bridge initialization requires"):
        validate_rl_games_config(cfg)


def test_validate_accepts_bridge_with_checkpoint() -> None:
    cfg = OmegaConf.create({
        "rl_games": {
            "model_alias": "openvla",
            "task": "flappy",
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
            "env_eval": {
                "latency": {"values": [0]},
                "deadly": {"action_layout": "multibinary_7"},
            },
        },
        "dataset": {"source_hf": "source", "converted_name": "flappy_train"},
        "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
        "initialization": {
            "checkpoint_local_dir": "local",
            "checkpoint_hf_repo_id": None,
            "checkpoint_filename": "checkpoints/steps_5000_pytorch_model.pt",
        },
        "framework": {"action_model": {"action_dim": 7}},
    })

    validate_rl_games_config(cfg)
```

- [ ] **Step 2: Run validation tests and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_config_validation.py -q
```

Expected: FAIL because `config_validation.py` does not exist.

- [ ] **Step 3: Implement validation**

Create `starVLA/training/rl_games/config_validation.py`:

```python
from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


BRIDGE_INIT_MODES: set[str] = {"bridge", "pre-trained", "pretrained"}


def _select_str(cfg: Any, path: str) -> str:
    value = OmegaConf.select(cfg, path)
    return "" if value is None else str(value)


def _select_optional_str(cfg: Any, path: str) -> str | None:
    value = OmegaConf.select(cfg, path)
    return None if value in (None, "") else str(value)


def _select_int_list(cfg: Any, path: str) -> list[int]:
    value = OmegaConf.select(cfg, path)
    if value is None:
        raise ValueError(f"Missing required integer list config: {path}")
    return [int(item) for item in OmegaConf.to_container(value, resolve=True)]


def _is_bridge(cfg: Any) -> bool:
    initialization_mode = _select_str(cfg, "rl_games.initialization_mode").lower()
    action_carrier = _select_str(cfg, "rl_games.action_carrier").lower()
    return initialization_mode in BRIDGE_INIT_MODES or action_carrier == "bridge"


def validate_rl_games_config(cfg: Any) -> None:
    model_alias = _select_str(cfg, "rl_games.model_alias")
    task = _select_str(cfg, "rl_games.task")
    initialization_mode = _select_str(cfg, "rl_games.initialization_mode")
    action_carrier = _select_str(cfg, "rl_games.action_carrier")
    dataset_source = _select_optional_str(cfg, "dataset.source_hf")
    dataset_name = _select_optional_str(cfg, "dataset.converted_name")
    base_model_repo_id = _select_optional_str(cfg, "base_model.repo_id")
    latencies = _select_int_list(cfg, "rl_games.env_eval.latency.values")

    if not model_alias:
        raise ValueError("Missing rl_games.model_alias")
    if not task:
        raise ValueError(f"Missing rl_games.task for model_alias={model_alias}")
    if not dataset_source:
        raise ValueError(f"Missing dataset.source_hf for model_alias={model_alias}, task={task}")
    if not dataset_name:
        raise ValueError(f"Missing dataset.converted_name for model_alias={model_alias}, task={task}")
    if not base_model_repo_id:
        raise ValueError(f"Missing base_model.repo_id for model_alias={model_alias}, task={task}")
    if not latencies:
        raise ValueError(f"Missing latency values for model_alias={model_alias}, task={task}")

    if model_alias == "pi-0.5" and initialization_mode == "scratch":
        raise ValueError("pi-0.5 scratch is not supported; use init=bridge")

    if not _is_bridge(cfg):
        return

    checkpoint_local_dir = _select_optional_str(cfg, "initialization.checkpoint_local_dir")
    checkpoint_hf_repo_id = _select_optional_str(cfg, "initialization.checkpoint_hf_repo_id")
    checkpoint_filename = _select_optional_str(cfg, "initialization.checkpoint_filename")
    if not checkpoint_local_dir and not checkpoint_hf_repo_id:
        raise ValueError(
            "bridge initialization requires initialization.checkpoint_local_dir "
            f"or initialization.checkpoint_hf_repo_id for model_alias={model_alias}, task={task}"
        )
    if not checkpoint_filename:
        raise ValueError(
            f"bridge initialization requires initialization.checkpoint_filename for "
            f"model_alias={model_alias}, task={task}"
        )
    if action_carrier != "bridge":
        raise ValueError(
            f"bridge initialization requires rl_games.action_carrier=bridge for "
            f"model_alias={model_alias}, task={task}"
        )
```

- [ ] **Step 4: Export validation**

Modify `starVLA/training/rl_games/__init__.py`:

```python
from starVLA.training.rl_games.action_spec import apply_action_spec
from starVLA.training.rl_games.alias import apply_model_alias
from starVLA.training.rl_games.checkpoint_sync import CheckpointSyncManager
from starVLA.training.rl_games.config_validation import validate_rl_games_config
from starVLA.training.rl_games.eval_core import RlGamesEvalRunner

__all__ = [
    "CheckpointSyncManager",
    "RlGamesEvalRunner",
    "apply_action_spec",
    "apply_model_alias",
    "validate_rl_games_config",
]
```

- [ ] **Step 5: Call validation from training**

Modify imports in `starVLA/training/train_starvla.py`:

```python
from starVLA.training.rl_games import (
    CheckpointSyncManager,
    RlGamesEvalRunner,
    apply_action_spec,
    apply_model_alias,
    validate_rl_games_config,
)
```

Modify `main(cfg) -> None` before `apply_model_alias(cfg)`:

```python
    if hasattr(cfg, "rl_games"):
        login_training_services(cfg, workspace_dir=getattr(cfg, "workspace_dir", None))
        validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
```

- [ ] **Step 6: Run validation and composition tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_config_validation.py tests/rl_games/test_config_composition.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit validation changes**

Run:

```bash
git add starVLA/training/rl_games/config_validation.py starVLA/training/rl_games/__init__.py starVLA/training/train_starvla.py tests/rl_games/test_config_validation.py
git commit -m "feat: validate rl-games hydra config combinations"
```

## Task 4: Replace Experiment YAML Launcher With Hydra Launcher

**Files:**
- Create: `examples/rl_games/scripts/launch_train.py`
- Delete: `examples/rl_games/scripts/run_experiment.py`
- Delete: `examples/rl_games/scripts/run_experiment.sh`
- Modify: `tests/rl_games/test_pi05_flappy_sft_path.py`

- [ ] **Step 1: Write launcher tests for Hydra config input**

In `tests/rl_games/test_pi05_flappy_sft_path.py`, replace imports:

```python
from examples.rl_games.scripts import launch_train, setup_training_assets
```

Add this test:

```python
def test_launch_train_setup_namespace_uses_composed_hydra_config(tmp_path: Path) -> None:
    cfg = _namespace({
        "run_id": "pi05_flappy_single",
        "model": "pi05",
        "env": "flappy",
        "mode": "single",
        "workspace_dir": str(tmp_path),
        "paths": {
            "dataset_local_dir": "playground/Datasets/rl_games",
            "dataset_cache_dir": None,
            "base_model_dir": "playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        },
        "dataset": {
            "source_hf": "talha1503/flappy_bird_zero_latency_parquet",
            "converted_name": "flappy_train",
            "force_download": False,
            "setup_force": False,
            "verify_rows": 200,
            "max_episodes": None,
            "latency_filter": None,
            "debug_subset": {"enabled": False, "max_episodes": 5, "suffix": "debug"},
        },
        "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
        "checkpoint": {
            "load": "auto",
            "hf_repo_id": None,
            "sync_enabled": False,
            "sync_repo_id": None,
        },
        "initialization": {
            "checkpoint_local_dir": "playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1",
            "checkpoint_hf_repo_id": "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
            "checkpoint_filename": "checkpoints/steps_50000_pytorch_model.pt",
        },
        "rl_games": {
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
            "env_eval": {"latency": {"mode": "single"}},
        },
    })

    setup_args = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.model == "pi05"
    assert setup_args.env == "flappy"
    assert setup_args.mode == "single"
    assert setup_args.initialization_mode == "bridge"
    assert setup_args.action_carrier == "bridge"
    assert setup_args.dataset_local_dir == str(tmp_path / "playground" / "Datasets" / "rl_games")
    assert setup_args.converted_dataset_name == "flappy_train"
    assert setup_args.initialization_checkpoint_filename == "checkpoints/steps_50000_pytorch_model.pt"
```

- [ ] **Step 2: Run launcher test and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_pi05_flappy_sft_path.py::test_launch_train_setup_namespace_uses_composed_hydra_config -q
```

Expected: FAIL because `launch_train.py` does not exist.

- [ ] **Step 3: Implement the new launcher**

Create `examples/rl_games/scripts/launch_train.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rl_games.scripts.setup_training_assets import setup_assets
from starVLA.training.rl_games.auth import login_training_services
from starVLA.training.rl_games.config_validation import validate_rl_games_config


CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"


def _select(cfg: Any, path: str) -> Any:
    return OmegaConf.select(cfg, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: Any, workspace_dir: Path) -> str:
    if value in (None, ""):
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str(workspace_dir / path)


def _optional_int_list(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in OmegaConf.to_container(value, resolve=True)]


def _safe_suffix(value: Any) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "debug")).strip("_")
    return suffix or "debug"


def _dataset_setup_values(cfg: Any) -> tuple[str, int | None]:
    converted_name = str(_select(cfg, "dataset.converted_name"))
    max_episodes_raw = _select(cfg, "dataset.max_episodes")
    max_episodes = None if max_episodes_raw in (None, "") else int(max_episodes_raw)

    if _as_bool(_select(cfg, "dataset.debug_subset.enabled")):
        debug_max_raw = _select(cfg, "dataset.debug_subset.max_episodes")
        max_episodes = int(debug_max_raw)
        suffix = _safe_suffix(_select(cfg, "dataset.debug_subset.suffix"))
        debug_suffix = "debug" if suffix == "debug" else f"debug_{suffix}"
        converted_name = f"{converted_name}__{debug_suffix}_{max_episodes}ep"

    return converted_name, max_episodes


def workspace_dir_from_cfg(cfg: Any) -> Path:
    configured = _select(cfg, "workspace_dir")
    if configured not in (None, "", "WORKSPACE_DIR"):
        return Path(_resolve_path(configured, REPO_ROOT)).resolve()
    env_workspace = os.environ.get("WORKSPACE_DIR")
    if env_workspace:
        return Path(_resolve_path(env_workspace, REPO_ROOT)).resolve()
    return REPO_ROOT


def setup_namespace_from_cfg(cfg: Any, workspace_dir: Path, run_root_dir: str) -> SimpleNamespace:
    run_id = str(_select(cfg, "run_id"))
    checkpoint_dir = str(Path(run_root_dir) / run_id / "checkpoints")
    converted_dataset_name, max_episodes = _dataset_setup_values(cfg)
    return SimpleNamespace(
        model=str(_select(cfg, "model")),
        env=str(_select(cfg, "env")),
        mode=str(_select(cfg, "mode")),
        initialization_mode=str(_select(cfg, "rl_games.initialization_mode") or ""),
        action_carrier=str(_select(cfg, "rl_games.action_carrier") or ""),
        latency_mode=str(_select(cfg, "rl_games.env_eval.latency.mode") or ""),
        source_dataset_hf=str(_select(cfg, "dataset.source_hf") or ""),
        dataset_local_dir=_resolve_path(_select(cfg, "paths.dataset_local_dir"), workspace_dir),
        converted_dataset_name=converted_dataset_name,
        dataset_cache_dir=(
            _resolve_path(_select(cfg, "paths.dataset_cache_dir"), workspace_dir)
            if _select(cfg, "paths.dataset_cache_dir") not in (None, "")
            else None
        ),
        dataset_force_download=str(_as_bool(_select(cfg, "dataset.force_download"))).lower(),
        setup_force=str(_as_bool(_select(cfg, "dataset.setup_force"))).lower(),
        verify_rows=int(_select(cfg, "dataset.verify_rows")),
        max_episodes=max_episodes,
        latency_filter=_optional_int_list(_select(cfg, "dataset.latency_filter")),
        base_model_dir=_resolve_path(_select(cfg, "paths.base_model_dir"), workspace_dir),
        base_model_repo_id=_select(cfg, "base_model.repo_id"),
        checkpoint_local_dir=checkpoint_dir,
        checkpoint_load=str(_select(cfg, "checkpoint.load") or "auto"),
        checkpoint_hf_repo_id=str(_select(cfg, "checkpoint.hf_repo_id") or ""),
        initialization_local_dir=(
            _resolve_path(_select(cfg, "initialization.checkpoint_local_dir"), workspace_dir)
            if _select(cfg, "initialization.checkpoint_local_dir") not in (None, "")
            else ""
        ),
        initialization_hf_repo_id=str(_select(cfg, "initialization.checkpoint_hf_repo_id") or ""),
        initialization_checkpoint_filename=str(_select(cfg, "initialization.checkpoint_filename") or ""),
        checkpoint_sync_enabled=str(_as_bool(_select(cfg, "checkpoint.sync.enabled"))).lower(),
        checkpoint_sync_repo_id=str(_select(cfg, "checkpoint.sync.repo_id") or ""),
        hf_repo_id="",
    )


def compose_training_cfg(overrides: list[str]) -> Any:
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=overrides)
    validate_rl_games_config(cfg)
    return cfg


def train_command_from_cfg(
    cfg: Any,
    setup: dict[str, Any],
    run_root_dir: str,
    original_overrides: list[str],
) -> list[str]:
    cfg_path_overrides = [
        *original_overrides,
        f"run_root_dir={run_root_dir}",
        f"trainer.is_resume={str(bool(setup.get('resume_found'))).lower()}",
    ]
    if setup.get("resume_checkpoint"):
        cfg_path_overrides.append(f"trainer.pretrained_checkpoint={setup['resume_checkpoint']}")
        cfg_path_overrides.append(f"trainer.resume_step={int(setup.get('resume_step') or 0)}")
    elif setup.get("pretrained_checkpoint"):
        cfg_path_overrides.append(f"trainer.pretrained_checkpoint={setup['pretrained_checkpoint']}")
        cfg_path_overrides.append("trainer.resume_step=0")
    if setup.get("dataset_local_dir"):
        cfg_path_overrides.append(f"datasets.vla_data.data_root_dir={setup['dataset_local_dir']}")
    if setup.get("data_mix"):
        cfg_path_overrides.append(f"datasets.vla_data.data_mix={setup['data_mix']}")
    if setup.get("eval_data_mix"):
        cfg_path_overrides.append(f"datasets.vla_data.eval_data_mix={setup['eval_data_mix']}")
    if setup.get("base_model_dir"):
        cfg_path_overrides.append(f"framework.qwenvl.base_vlm={setup['base_model_dir']}")
    if setup.get("latency_prompt_map_path"):
        cfg_path_overrides.append(f"rl_games.env_eval.latency.prompt_map_path={setup['latency_prompt_map_path']}")
    return [
        "starVLA/training/train_starvla_hydra.py",
        "--config-name",
        "train",
        *cfg_path_overrides,
    ]


def launch_command_from_cfg(cfg: Any, trainer_cmd: list[str], workspace_dir: Path) -> list[str]:
    backend = str(_select(cfg, "trainer.distributed_backend") or "deepspeed").lower()
    use_accelerate = _as_bool(_select(cfg, "launch.use_accelerate"))
    if backend == "none" or not use_accelerate:
        return [sys.executable, *trainer_cmd]
    accelerate_config = _resolve_path(_select(cfg, "paths.accelerate_config"), workspace_dir)
    return [
        "accelerate",
        "launch",
        "--config_file",
        accelerate_config,
        "--num_processes",
        str(_select(cfg, "launch.num_processes")),
        *trainer_cmd,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = compose_training_cfg(args.overrides)
    workspace_dir = workspace_dir_from_cfg(cfg)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir = _resolve_path(_select(cfg, "paths.run_root_dir"), workspace_dir)
    login_training_services(cfg, workspace_dir=workspace_dir, repo_root=REPO_ROOT)

    gpus = _select(cfg, "launch.gpus")
    if gpus not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    with contextlib.redirect_stdout(sys.stderr):
        setup = setup_assets(setup_namespace_from_cfg(cfg, workspace_dir, run_root_dir))

    trainer_cmd = train_command_from_cfg(cfg, setup, run_root_dir, list(args.overrides))
    launch_cmd = launch_command_from_cfg(cfg, trainer_cmd, workspace_dir)

    print("Setup summary:")
    for key in sorted(setup):
        print(f"  {key}: {setup[key]}")
    print("Running command:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))

    if _as_bool(_select(cfg, "launch.dry_run")):
        return 0
    subprocess.run(launch_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run launcher tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_pi05_flappy_sft_path.py::test_launch_train_setup_namespace_uses_composed_hydra_config -q
```

Expected: PASS.

- [ ] **Step 5: Delete old experiment launcher files**

Delete:

```text
examples/rl_games/scripts/run_experiment.py
examples/rl_games/scripts/run_experiment.sh
```

- [ ] **Step 6: Commit launcher replacement**

Run:

```bash
git add examples/rl_games/scripts/launch_train.py examples/rl_games/scripts/run_experiment.py examples/rl_games/scripts/run_experiment.sh tests/rl_games/test_pi05_flappy_sft_path.py
git commit -m "refactor: launch rl-games training from hydra config"
```

## Task 5: Update Command Wrappers

**Files:**
- Modify: `commands/train_*.sh`
- Modify: `tests/rl_games/test_training_commands.py`

- [ ] **Step 1: Update command wrapper test**

Replace `tests/rl_games/test_training_commands.py` with:

```python
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ("openvla", "pi0", "pi05", "gr00t")
ENVS = ("flappy", "demon_attack", "deadly_corridor")


def _command_path(model: str, env: str) -> Path:
    return REPO_ROOT / "commands" / f"train_{env}_{model}.sh"


def test_training_command_matrix_uses_hydra_launcher() -> None:
    for model in MODELS:
        for env in ENVS:
            command_path = _command_path(model, env)
            command_text = command_path.read_text(encoding="utf-8")

            assert command_path.exists(), f"Missing command wrapper: {command_path}"
            assert "python examples/rl_games/scripts/launch_train.py" in command_text
            assert "examples/rl_games/experiments/" not in command_text
            assert f"model={model}" in command_text
            assert f"env={env}" in command_text
            assert "init=bridge" in command_text
            assert "mode=single" in command_text
            assert f"conda.env_name=starvla_{model}" in command_text
            assert 'wandb_entity="$WANDB_ENTITY"' in command_text
            assert "rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]" in command_text


def test_training_commands_are_valid_bash() -> None:
    command_paths = [str(_command_path(model, env)) for model in MODELS for env in ENVS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)
```

- [ ] **Step 2: Run command test and verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_training_commands.py -q
```

Expected: FAIL because commands still reference `run_experiment.sh` and `experiments/`.

- [ ] **Step 3: Update all 12 command wrappers**

Use this command shape in each wrapper:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  mode=single \
  conda.env_name=starvla_openvla \
  workspace_dir=/inspire/hdd/project/spatialintelligence/public/lzj/starVLA \
  run_id=openvla_flappy_fix_latency_0 \
  wandb_entity="$WANDB_ENTITY" \
  paths.dataset_local_dir=data/flappy_fix_latency_0_parquet \
  paths.base_model_dir=playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  dataset.source_hf=data/raw/flappy_bird_zero_latency_parquet \
  dataset.converted_name=flappy_train \
  datasets.vla_data.data_mix=flappy_train \
  base_model.repo_id=Qwen/Qwen3-VL-4B-Instruct \
  initialization.checkpoint_local_dir=playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1 \
  initialization.checkpoint_hf_repo_id=StarVLA/Qwen3VL-OFT-Bridge-RT-1 \
  initialization.checkpoint_filename=checkpoints/steps_5000_pytorch_model.pt \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=16 \
  trainer.batch_size=16 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=100 \
  trainer.eval_interval=100 \
  checkpoint.load=local \
  checkpoint.sync.enabled=false \
  rl_games.env_eval.latency.values=[0] \
  rl_games.env_eval.mid_train.interval_steps=100 \
  rl_games.env_eval.mid_train.latencies=[0] \
  rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
```

Apply model/env-specific values from the current command wrappers:

- `flappy`: `rl_games.env_eval.latency.values=[0]`, dataset dir `data/flappy_fix_latency_0_parquet`.
- `demon_attack`: `rl_games.env_eval.latency.values=[1]`, dataset dir `data/demon_attack_fix_latency_1`.
- `deadly_corridor`: `rl_games.env_eval.latency.values=[0]`, dataset dir `data/deadly_corridor_fix_latency_0`, `dataset.latency_filter=[0]`.
- `openvla`: initializer `Qwen3VL-OFT-Bridge-RT-1`, max train steps `5000` for flappy and `2000` otherwise.
- `pi0`: base model and initializer `Qwen2.5-VL-3B-Instruct-Action` / `Qwen-PI-Bridge-RT-1`.
- `pi05`: initializer `Qwen3VL-PI_v3-Bridge-RT_1`.
- `gr00t`: initializer `Qwen3VL-GR00T-Bridge-RT-1`.

- [ ] **Step 4: Run command tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_training_commands.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit command wrapper migration**

Run:

```bash
git add commands tests/rl_games/test_training_commands.py
git commit -m "refactor: point rl-games command wrappers at hydra launcher"
```

## Task 6: Remove Experiment YAML Tests and Directory

**Files:**
- Modify: `tests/rl_games/test_pi05_flappy_sft_path.py`
- Delete: `examples/rl_games/experiments/`

- [ ] **Step 1: Replace experiment-YAML assertions with composition assertions**

In `tests/rl_games/test_pi05_flappy_sft_path.py`, remove `_load_experiment_config`, `EXPECTED_PI05_BRIDGE_EXPERIMENTS`, `_assert_pi05_bridge_experiment`, and tests that directly read `examples/rl_games/experiments`.

Add this replacement test:

```python
@pytest.mark.parametrize(
    ("model", "env", "mode", "action_env_dim"),
    [
        ("pi05", "flappy", "single", 2),
        ("pi05", "flappy", "mixed_latency", 2),
        ("pi05", "demon_attack", "single", 6),
        ("pi05", "demon_attack", "mixed_latency", 6),
        ("pi05", "deadly_corridor", "single", 7),
        ("pi05", "deadly_corridor", "mixed_latency", 7),
    ],
)
def test_pi05_bridge_composed_config_uses_qwenpi_v3(
    model: str,
    env: str,
    mode: str,
    action_env_dim: int,
) -> None:
    cfg = launch_train.compose_training_cfg([
        f"model={model}",
        f"env={env}",
        "init=bridge",
        f"mode={mode}",
    ])
    apply_model_alias(cfg)
    apply_action_spec(cfg)

    assert cfg.framework.name == "QwenPI_v3"
    assert cfg.rl_games.model_alias == "pi-0.5"
    assert cfg.rl_games.initialization_mode == "bridge"
    assert cfg.rl_games.action_carrier == "bridge"
    assert cfg.initialization.checkpoint_hf_repo_id == "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1"
    assert cfg.initialization.checkpoint_filename == "checkpoints/steps_50000_pytorch_model.pt"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.action_env_dim == action_env_dim
```

- [ ] **Step 2: Delete the experiments directory**

Delete:

```text
examples/rl_games/experiments/
```

- [ ] **Step 3: Verify no active code references experiments**

Run:

```bash
rg -n "examples/rl_games/experiments|run_experiment" examples commands tests starVLA
```

Expected: no matches except historical prose that is being updated in Task 7.

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games/test_pi05_flappy_sft_path.py tests/rl_games/test_training_commands.py tests/rl_games/test_config_composition.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit old experiment removal**

Run:

```bash
git add tests/rl_games/test_pi05_flappy_sft_path.py examples/rl_games/experiments
git commit -m "refactor: remove rl-games experiment yaml configs"
```

## Task 7: Update Documentation

**Files:**
- Modify: `examples/rl_games/README.md`
- Modify: `README.md`

- [ ] **Step 1: Update RL-games README examples**

Replace examples using:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla/scratch/mixed_latency/flappy.yaml
```

with:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency
```

For bridge/single examples, use:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=bridge \
  mode=single
```

- [ ] **Step 2: Document group ownership**

Add this section to `examples/rl_games/README.md`:

```markdown
## Configuration

RL-games training uses one Hydra configuration tree under `examples/rl_games/config`.
A run is composed from four primary groups:

- `model`: `openvla`, `pi0`, `pi05`, or `gr00t`.
- `env`: `flappy`, `demon_attack`, or `deadly_corridor`.
- `init`: `scratch` or `bridge`.
- `mode`: `single` or `mixed_latency`.

Do not add new YAML files under `examples/rl_games/experiments`; that directory is intentionally removed from the active training path.
```

- [ ] **Step 3: Update top-level README references**

Replace references to `examples/rl_games/experiments/<model>/<scratch|bridge>/<single|mixed_latency>/<env>.yaml` with `examples/rl_games/config` group composition examples.

- [ ] **Step 4: Verify docs no longer point to the old path**

Run:

```bash
rg -n "examples/rl_games/experiments|run_experiment" README.md examples/rl_games/README.md
```

Expected: no matches.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md examples/rl_games/README.md
git commit -m "docs: document canonical rl-games hydra config"
```

## Task 8: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run config-focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest \
  tests/rl_games/test_config_composition.py \
  tests/rl_games/test_config_validation.py \
  tests/rl_games/test_training_commands.py \
  tests/rl_games/test_pi05_flappy_sft_path.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the full lightweight RL-games test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --group dev pytest tests/rl_games -q
```

Expected: PASS.

- [ ] **Step 3: Run syntax checks for command wrappers**

Run:

```bash
bash -n commands/train_flappy_openvla.sh commands/train_flappy_pi0.sh commands/train_flappy_pi05.sh commands/train_flappy_gr00t.sh commands/train_demon_attack_openvla.sh commands/train_demon_attack_pi0.sh commands/train_demon_attack_pi05.sh commands/train_demon_attack_gr00t.sh commands/train_deadly_corridor_openvla.sh commands/train_deadly_corridor_pi0.sh commands/train_deadly_corridor_pi05.sh commands/train_deadly_corridor_gr00t.sh
```

Expected: PASS with no output.

- [ ] **Step 4: Verify old experiment system is gone**

Run:

```bash
test ! -d examples/rl_games/experiments
rg -n "examples/rl_games/experiments|run_experiment" examples commands tests starVLA README.md
```

Expected: first command exits 0; second command returns no matches.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git --no-pager diff HEAD
```

Expected: no uncommitted changes after the task commits, or only intentionally unstaged review edits.
