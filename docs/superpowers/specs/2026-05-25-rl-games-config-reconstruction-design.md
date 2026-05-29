# RL Games Configuration Reconstruction Design

## Goal

Replace the parallel RL-games configuration systems with one canonical Hydra configuration tree. The final system should make each runnable training job a composition of `model`, `env`, `init`, and `mode`, without keeping `examples/rl_games/experiments/` as a second configuration DSL.

## Current Effective Configuration

RL-games training currently has three configuration layers:

1. `examples/rl_games/config/` is the Hydra configuration consumed by `starVLA/training/train_starvla_hydra.py`.
2. `examples/rl_games/experiments/` contains 42 outer YAML files consumed by `examples/rl_games/scripts/run_experiment.py`, which translates them into Hydra overrides.
3. `commands/train_*.sh` contains 12 commonly used bridge/single remote command wrappers that load experiment YAMLs and then add more overrides.

The currently effective experiment matrix is:

- `openvla`, `pi0`, and `gr00t`: `scratch` and `bridge`, each with `single` and `mixed_latency`, for `flappy`, `demon_attack`, and `deadly_corridor`.
- `pi05`: `bridge` only, with `single` and `mixed_latency`, for the same three environments.
- The 12 top-level `commands/train_*.sh` wrappers target only `bridge/single`, one per model and environment.

Important effective behavior is spread across code and wrappers:

- `run_experiment.py` converts outer fields such as `rl_games.latencies` into Hydra fields such as `rl_games.env_eval.latency.values`.
- `setup_training_assets.py` changes bridge dataset names from `*_train` to `*_train__bridge`.
- `starVLA/training/rl_games/action_spec.py` applies action dimensions at runtime:
  - bridge mode uses a shared 7D action carrier and sets `action_env_dim` to the task dimension.
  - scratch OpenVLA emits the environment action dimension directly.
  - scratch `pi0`, `pi05`, and `gr00t` keep model action dimensions only when they are at least the task action dimension.
- The command wrappers add practical remote-run settings, including fixed training latency and post-train eval latencies `[0, ..., 15]`.

## Target Configuration Structure

Use one Hydra configuration tree:

```text
examples/rl_games/config/
  train.yaml

  base/
    common.yaml
    runtime.yaml

  model/
    base.yaml
    openvla.yaml
    pi0.yaml
    pi05.yaml
    gr00t.yaml

  env/
    base.yaml
    flappy.yaml
    demon_attack.yaml
    deadly_corridor.yaml

  init/
    base.yaml
    scratch.yaml
    bridge.yaml

  mode/
    base.yaml
    single.yaml
    mixed_latency.yaml
```

`examples/rl_games/experiments/` should be removed from the active training path. A single experiment is represented by one Hydra composition, for example:

```bash
python starVLA/training/train_starvla_hydra.py \
  --config-name train \
  model=openvla \
  env=flappy \
  init=bridge \
  mode=single \
  run_id=openvla_flappy_fix_latency_0 \
  rl_games.env_eval.latency.values=[0]
```

The `commands/train_*.sh` wrappers may remain, but only as thin launch helpers. They should pass Hydra group selections and machine-specific path overrides. They must not contain hidden training semantics that are absent from the Hydra config.

## File Responsibilities

### `train.yaml`

Owns the Hydra defaults list and the top-level composition order:

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

It may define only top-level defaults that are not naturally owned by a group.

### `base/common.yaml`

Owns stable training defaults:

- generic `framework` defaults shared by all models.
- generic `datasets.vla_data` defaults.
- generic `trainer` defaults.
- generic `checkpoint` defaults that are not already owned by `checkpoint/default.yaml`.
- generic `rl_games.env_eval` stage defaults.

### `base/runtime.yaml`

Owns the single runtime default:

- `workspace_dir`
- auth env variable names
- W&B defaults
- launch defaults
- common path defaults such as dataset root, run root, base-model root, and accelerate config.

There is no separate local/H100 runtime group.

### `model/base.yaml`

Owns defaults shared by all model aliases:

- common Qwen-VL interface defaults.
- common action model defaults that are truly model-agnostic.
- model-independent learning-rate defaults only if they are not environment or init specific.

### `model/*.yaml`

Each model file owns only model-specific values:

- `rl_games.model_alias`
- `framework.name`
- model-specific `framework.qwenvl.base_vlm`
- model-specific `framework.action_model.*`
- model-specific optimizer or learning-rate values
- model-specific bridge initializer local path, Hugging Face repo, and checkpoint filename

`pi05` remains bridge-supported only unless the project explicitly decides to add scratch support.

### `env/base.yaml`

Owns environment-independent RL-games task defaults:

- image size default
- frameskip default if shared
- default environment eval limits
- default dataset local root shape if not owned by runtime

### `env/*.yaml`

Each environment file owns only environment-specific values:

- `rl_games.task`
- `rl_games.env_eval.frameskip`
- `rl_games.env_eval.image_size`
- `rl_games.env_eval.task_description`
- `rl_games.env_eval.deadly.action_layout` for Deadly Corridor
- default dataset source
- default converted dataset name
- action type if it is environment-specific

Known dataset-source differences that currently appear only in selected experiment files must become explicit environment or command-level overrides. They must not remain hidden in duplicated full experiment YAMLs.

### `init/base.yaml`

Owns initialization defaults:

- default initialization mode
- default action carrier
- default state inclusion policy if it is common

### `init/scratch.yaml`

Owns scratch initialization semantics:

- `rl_games.initialization_mode: scratch`
- `rl_games.action_carrier: native`
- no bridge checkpoint source
- native dataset carrier

### `init/bridge.yaml`

Owns bridge initialization semantics:

- `rl_games.initialization_mode: bridge`
- `rl_games.action_carrier: bridge`
- bridge action carrier settings that are not model-specific
- dataset state inclusion required by bridge models, except OpenVLA if it intentionally stays stateless

The exact model checkpoint source stays in `model/*.yaml`, because it differs by model.

### `mode/base.yaml`

Owns latency-mode defaults:

- global `rl_games.env_eval.enabled`
- default mid-train and post-train eval settings
- default prompt-map behavior

### `mode/single.yaml`

Owns single-latency semantics:

- `rl_games.env_eval.latency.mode: single`
- default training latency `[0]`
- default mid-train eval latencies `[0]`
- default post-train eval latencies unless the command overrides them

### `mode/mixed_latency.yaml`

Owns mixed-latency semantics:

- `rl_games.env_eval.latency.mode: mixed`
- training/eval latencies `[0, 1, 2, 3, 4, 5]`
- prompt-map requirement where applicable
- mixed-latency dataset name override if it is not already derived by setup

## Launcher Design

`run_experiment.py` should stop reading experiment YAML files and stop translating a second schema into Hydra overrides.

The asset preparation logic in `setup_training_assets.py` can remain, but it should receive values from the composed Hydra config or a small typed namespace derived from that config. The training launcher should then execute the Hydra-composed training command directly.

`run_train.sh` should either be removed or reduced to a thin launch wrapper around direct Hydra group selections.

## Migration Rules

- Do not add compatibility fallback behavior for old experiment YAMLs.
- Do not silently preserve duplicated experiment fields.
- Move each currently effective setting into exactly one owning group.
- Keep command wrappers only for practical launch overrides, not model/env/init/mode semantics.
- Any mismatch between current duplicated configs should be resolved explicitly by choosing the intended value.
- Error messages should name the selected model, env, init, mode, dataset source, and relevant checkpoint path when configuration validation fails.

## Validation

Local validation should stay lightweight:

1. Compose every supported `model x env x init x mode` combination that exists today.
2. Assert that the composed config contains the intended:
   - model alias
   - task
   - initialization mode
   - action carrier
   - latency values
   - dataset source and converted name
   - base model path/repo
   - bridge initialization checkpoint path/repo
3. Run existing RL-games tests:

```bash
uv run --group dev pytest tests/rl_games -q
```

4. Run shell syntax checks for retained command wrappers:

```bash
bash -n commands/train_flappy_openvla.sh
```

Full SFT and environment rollout validation should remain remote-only.
