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

Main OpenVLA Flappy configs:

```text
examples/rl_games/experiments/openvla_flappy_mixed_latency.yaml
examples/rl_games/experiments/openvla_flappy_single.yaml
```

Edit `workspace_dir`, `auth`, `wandb`, `dataset`, `base_model`, `checkpoint`, `launch`, `train_data`, and `trainer` in the YAML. Relative asset paths are resolved under `workspace_dir`.

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
  wandb.entity=WANDB_ENTITY
```

Single-latency Flappy:

```bash
bash examples/rl_games/scripts/run_experiment.sh \
  examples/rl_games/experiments/openvla_flappy_single.yaml \
  workspace_dir=WORKSPACE_DIR \
  wandb.entity=WANDB_ENTITY
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
