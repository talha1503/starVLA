# RL Games Pipeline

Detailed guide: `examples/rl_games/USAGE.md`

## Installation

Layered installer:

```bash
bash examples/rl_games/install/install_stack.sh openvla flappy
bash examples/rl_games/install/install_stack.sh pi0 demon_attack
bash examples/rl_games/install/install_stack.sh gr00t deadly_corridor
```

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
