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
- `starvla_rl_games_gr00t`

Each split env installs the repo-standard `starVLA` stack plus the requested RL-games env dependencies. With the default `--env all`, every model env gets `flappy`, `demon_attack`, and `deadly_corridor` dependencies.

PyTorch is selected automatically during install. Blackwell GPUs (`compute_cap=12.0` / `sm_120`) get the CUDA 12.8 PyTorch stack; other CUDA GPUs use the repo-compatible CUDA 12.4 stack. Override with `STARVLA_TORCH_PROFILE=cu128|cu126|cu124|cpu`.

Layered installer (manual control):

```bash
bash examples/rl_games/install/install_stack.sh openvla flappy
bash examples/rl_games/install/install_stack.sh pi0 demon_attack
bash examples/rl_games/install/install_stack.sh gr00t deadly_corridor
```

By default, layered install uses one conda env per model:
- `openvla` -> `starvla_rl_games_openvla`
- `pi0` -> `starvla_rl_games_pi0`
- `gr00t` -> `starvla_rl_games_gr00t`

You can override with `--conda-env <name>`, or skip conda handling with `--no-conda`.

Available scripts:
- `examples/rl_games/install/common.sh`
- `examples/rl_games/install/model/{openvla,pi0,gr00t}.sh`
- `examples/rl_games/install/env/{flappy,demon_attack,deadly_corridor}.sh`
- `examples/rl_games/install/validate/*.sh`

## Train

The current training flow is config-driven. Pick one YAML file from:

```text
examples/rl_games/experiments/
```

Main OpenVLA configs:

```text
examples/rl_games/experiments/openvla_flappy_mixed_latency.yaml
examples/rl_games/experiments/openvla_flappy_single.yaml
examples/rl_games/experiments/openvla_demon_attack_mixed_latency.yaml
examples/rl_games/experiments/openvla_demon_attack_single.yaml
examples/rl_games/experiments/openvla_deadly_corridor_mixed_latency.yaml
examples/rl_games/experiments/openvla_deadly_corridor_single.yaml
```

Edit `workspace_dir`, `auth`, `dataset`, `datasets`, `base_model`, `checkpoint`, `launch`, `framework`, and `trainer` in the YAML. Relative asset paths are resolved under `workspace_dir`.

Authentication tokens are read from `HF_TOKEN` and `WANDB_API_KEY` by default:

```bash
export HF_TOKEN=HF_TOKEN_VALUE
export WANDB_API_KEY=WANDB_API_KEY_VALUE
```

You can also copy `examples/rl_games/auth.env.example` to a private file such as `WORKSPACE_DIR/auth.env`, then set `auth.env_file: auth.env` in the experiment YAML. Do not commit real tokens.

Mixed-latency Flappy:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_mixed_latency.yaml \
  workspace_dir=WORKSPACE_DIR \
  wandb_entity=WANDB_ENTITY
```

Single-latency Flappy:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_single.yaml \
  workspace_dir=WORKSPACE_DIR \
  wandb_entity=WANDB_ENTITY
```

Single-GPU direct backend with a custom run name:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_single.yaml \
  run_id=test_qwen3_flappy_gc_none_backend \
  trainer.distributed_backend=none
```

Override any YAML value from the command line:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_mixed_latency.yaml \
  workspace_dir=WORKSPACE_DIR \
  run_id=smoke_test \
  trainer.max_train_steps=10 \
  trainer.save_interval=5 \
  trainer.eval_interval=5
```

Fast training smoke test:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_mixed_latency.yaml \
  run_id=debug_flappy_mixed_e2e \
  trainer.max_train_steps=2 \
  datasets.vla_data.per_device_batch_size=1 \
  trainer.distributed_backend=none \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false
```

For debug subsets, prepare a separate LeRobot data mix during the conversion
stage and set `datasets.vla_data.data_mix` to that converted folder.

Raw rollout exports and StarVLA training datasets are now separate artifacts.
First export the canonical raw parquet dataset, then prepare the StarVLA
LeRobot dataset as a derived artifact:

```bash
python scripts/rollout_data/export_sf_teacher_dataset.py \
  --checkpoint-root /path/to/sample_factory_checkpoint \
  --output-dir /mnt/data/latency_data/outputs/demon_attack_fixed_2_latency \
  --max-episodes 1000 \
  --max-steps-per-episode 1000 \
  --push-to-hub \
  --hf-repo-id saberrr-zju/demon_attack_fixed_2_latency_raw

python scripts/rollout_data/prepare_starvla_lerobot_dataset.py \
  /mnt/data/latency_data/outputs/demon_attack_fixed_2_latency \
  /mnt/data/latency_data/playground/Datasets/rl_games \
  demon_attack_train \
  --push-to-hub saberrr-zju/demon_attack_train_lerobot
```

Training reads only the prepared LeRobot dataset. If it is not already present
under `datasets.vla_data.data_root_dir`, point `dataset.converted_hf` at the
converted repo:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_demon_attack_single.yaml \
  dataset.converted_hf=saberrr-zju/demon_attack_train_lerobot
```

The prepared dataset includes a held-out validation LeRobot dataset next to the
training dataset, for example `flappy_mixed_latency_train__val` or
`demon_attack_train__val`. The trainer logs `train/loss`, `eval/loss`, and
`train/grad_norm_pre_clip`.

Deadly Corridor latency filtering is part of the LeRobot preparation step. The
training launcher only reads the converted folder and its `latency_prompt_map.json`.

Checkpoint fields have separate meanings: `checkpoint.hf_repo_id` is the resume/download source, while `checkpoint.sync.repo_id` is the upload destination when `checkpoint.sync.enabled: true`. The trainer saves full Accelerate training-state directories (`steps_<N>_state/`) for exact resume, including optimizer/scheduler state, and also saves lightweight model files for convenience. A missing `checkpoint.sync.repo_id` repo is created during sync if Hugging Face auth is available. `checkpoint.sync.keep_last_n: 0` keeps all uploaded HF checkpoints.

Environment rollout eval is controlled in the `rl_games` block:

```yaml
rl_games:
  env_eval:
    enabled: true

    mid_train:
      enabled: true
      interval_steps: 100
      latencies: [0, 1, 2, 3, 4, 5]
      num_episodes: 5
      max_steps_per_episode: 2000

    post_train:
      enabled: true
      latencies: [0, 1, 2, 3, 4, 5]
      num_episodes: 5
      max_steps_per_episode: 2000
```

This is separate from `trainer.eval_interval`. `trainer.eval_interval` runs the trainer's action-model eval; `rl_games.env_eval.mid_train` and `rl_games.env_eval.post_train` run the current model in the actual environment.

The launcher activates the conda env from the config, then downloads/prepares the dataset, writes dataset statistics, downloads the base model if needed, checks local/Hugging Face checkpoints, and launches training.

## Eval summary

```bash
bash examples/rl_games/scripts/run_eval.sh --run-dir results/Checkpoints/starvla_rl_games --stage post_train
```
