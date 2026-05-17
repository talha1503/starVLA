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

```bash
bash examples/rl_games/scripts/run_train.sh --model openvla --env flappy --mode single
```

## Eval summary

```bash
bash examples/rl_games/scripts/run_eval.sh --run-dir results/Checkpoints/starvla_rl_games --stage post_train
```
