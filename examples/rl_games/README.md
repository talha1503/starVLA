# RL Games Pipeline

Detailed guide: `examples/rl_games/USAGE.md`

## Installation

One-command bootstrap (recommended):

```bash
bash examples/rl_games/install/bootstrap.sh --split-envs
```

This command will:
- create one conda env per model
- install the repo-standard `starVLA` stack plus all RL-games env deps in each model env
- run a validation smoke check

Shared-env bootstrap:

```bash
bash examples/rl_games/install/bootstrap.sh
```

Split bootstrap creates one env per model:
- `starvla_rl_games_openvla`
- `starvla_rl_games_pi0`
- `starvla_rl_games_pi05`
- `starvla_rl_games_gr00t`

Each split env installs the repo-standard `starVLA` stack plus the requested RL-games env dependencies. With the default `--env all`, every model env gets `flappy`, `demon_attack`, and `deadly_corridor` dependencies.

PyTorch is selected automatically during install. Blackwell GPUs (`compute_cap=12.0` / `sm_120`) get the CUDA 12.8 PyTorch stack; other CUDA GPUs use the repo-compatible CUDA 12.4 stack. Override with `STARVLA_TORCH_PROFILE=cu128|cu126|cu124|cpu`.

Layered installer (manual control):

```bash
bash examples/rl_games/install/install_stack.sh openvla flappy
bash examples/rl_games/install/install_stack.sh pi0 demon_attack
bash examples/rl_games/install/install_stack.sh pi05 flappy
bash examples/rl_games/install/install_stack.sh gr00t flappy
bash examples/rl_games/install/install_stack.sh gr00t demon_attack
bash examples/rl_games/install/install_stack.sh gr00t deadly_corridor
```

By default, layered install uses one conda env per model:
- `openvla` -> `starvla_rl_games_openvla`
- `pi0` -> `starvla_rl_games_pi0`
- `pi05` -> `starvla_rl_games_pi05`
- `gr00t` -> `starvla_rl_games_gr00t`

You can override with `--conda-env <name>`, or skip conda handling with `--no-conda`.

Available scripts:
- `examples/rl_games/install/common.sh`
- `examples/rl_games/install/model/{openvla,pi0,pi05,gr00t}.sh`
- `examples/rl_games/install/env/{flappy,demon_attack,deadly_corridor}.sh`
- `examples/rl_games/install/validate/*.sh`

## Train

The current training flow is Hydra-driven. Compose the run from the canonical config tree and pass the active groups on the command line.

`scratch` trains with the native task action width. `bridge` starts from the released StarVLA Bridge/RT-1 checkpoints, uses `Qwen/Qwen3-VL-4B-Instruct` as the base backbone, and trains through a shared 7D action/state carrier. Losses and inference are masked to the active task action dimensions: 2 for Flappy, 6 for Demon Attack, and 7 for Deadly Corridor.

These pi-0.5 configs use `Qwen/Qwen3-VL-4B-Instruct` as the base backbone and initialize from `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`. Setup first checks the local initializer at `playground/Pretrained_models/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt`; if it is missing, it falls back to the Hugging Face repo.

Edit `workspace_dir`, `auth`, `wandb`, `dataset`, `base_model`, `checkpoint`, `launch`, `train_data`, and `trainer` in the YAML. Relative asset paths are resolved under `workspace_dir`.

Authentication tokens are read from `HF_TOKEN` and `WANDB_API_KEY` by default. The RL-games train launchers log into Hugging Face and W&B before asset setup/training when those values are available:

```bash
export HF_TOKEN=HF_TOKEN_VALUE
export WANDB_API_KEY=WANDB_API_KEY_VALUE
```

You can also copy `examples/rl_games/auth.env.example` to a private file such as `WORKSPACE_DIR/auth.env`, then set `auth.env_file: auth.env` in the Hydra config. If `auth.env_file` is null, the launcher auto-detects `WORKSPACE_DIR/auth.env` and `examples/rl_games/auth.env`. Do not commit real tokens.

Mixed-latency Flappy:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency
```

Single-latency Deadly Corridor with bridge initialization:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=bridge \
  mode=single
```

Override config values from the command line:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency \
  workspace_dir=WORKSPACE_DIR \
  wandb.entity=WANDB_ENTITY \
  run_id=smoke_test \
  trainer.max_train_steps=10 \
  trainer.save_interval=5 \
  trainer.eval_interval=5
```

Fast end-to-end preprocessing debug:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency \
  run_id=debug_flappy_mixed_e2e \
  dataset.debug_subset.enabled=true \
  dataset.debug_subset.max_episodes=5 \
  trainer.max_train_steps=2 \
  trainer.batch_size=1 \
  trainer.distributed_backend=none \
  rl_games.mid_train_eval.enabled=false \
  rl_games.post_train_eval.enabled=false
```

## Configuration

RL-games training uses one Hydra configuration tree under `examples/rl_games/config`.

The primary config groups are:
- `model`
- `env`
- `init`
- `mode`

Use `python examples/rl_games/scripts/launch_train.py ...` to compose runs from those groups. Do not add new YAML files under the legacy `experiments` directory; it is intentionally removed from the active training path.

Single-GPU direct backend with a custom run name:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=single \
  run_id=test_qwen3_flappy_gc_none_backend \
  trainer.distributed_backend=none
```

Debug subsets are written to separate converted dataset folders, for example
`flappy_mixed_latency_train__debug_5ep`, so they do not overwrite the full
preprocessed dataset. The same overrides work for the OpenVLA Demon Attack
single and mixed-latency experiment YAMLs.

The converter also writes a held-out validation LeRobot dataset next to the
training dataset, for example `flappy_mixed_latency_train__val` or
`flappy_mixed_latency_train__debug_5ep__val`. The trainer logs `train/loss`,
`eval/loss`, and `train/grad_norm_pre_clip`.

Deadly Corridor uses the mixed-latency HF dataset for both modes. The single
config filters it to `dataset.latency_filter: [0]`, while the mixed config
keeps all latencies and requires `latency_prompt_map.json`.

Checkpoint fields have separate meanings: `checkpoint.hf_repo_id` is the resume/download source, while `checkpoint.sync_repo_id` is the upload destination when `checkpoint.sync_enabled: true`. The trainer saves full Accelerate training-state directories (`steps_<N>_state/`) for exact resume, including optimizer/scheduler state, and also saves lightweight model files for convenience. A missing `sync_repo_id` repo is created during sync if Hugging Face auth is available. `checkpoint.hf_keep_last_n: 0` keeps all uploaded HF checkpoints.

Environment rollout eval is controlled in the `rl_games` block:

```yaml
rl_games:
  env_eval_enabled: true

  mid_train_eval:
    enabled: true
    interval_steps: 100
    latencies: [0, 1, 2, 3, 4, 5]
    num_episodes: 5
    max_steps_per_episode: 2000

  post_train_eval:
    enabled: true
    latencies: [0, 1, 2, 3, 4, 5]
    num_episodes: 5
    max_steps_per_episode: 2000
```

This is separate from `trainer.eval_interval`. `trainer.eval_interval` runs the trainer's action-model eval; `rl_games.mid_train_eval` and `rl_games.post_train_eval` run the current model in the actual environment.

The launcher activates the conda env from the config, then downloads/prepares the dataset, writes dataset statistics, downloads the base model if needed, checks local/Hugging Face checkpoints, and launches training.

## Eval summary

```bash
bash examples/rl_games/scripts/run_eval.sh --run-dir results/Checkpoints/starvla_rl_games --stage post_train
```
