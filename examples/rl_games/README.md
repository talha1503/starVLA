# RL Games Pipeline

Detailed guide: `examples/rl_games/USAGE.md`

## Installation

Install one model family at a time. The default is the OpenVLA use tier with
Flappy, Demon Attack, and Deadly Corridor:

```bash
bash examples/rl_games/install/bootstrap.sh --accept-rom-license
```

Choose another family or add its development/training dependencies explicitly:

```bash
bash examples/rl_games/install/bootstrap.sh --tier use --model pi05 --env flappy
bash examples/rl_games/install/bootstrap.sh --tier dev --model gr00t --env all --accept-rom-license
bash examples/rl_games/install/bootstrap.sh --tier dev --model wan_oft --env flappy
```

Each family uses an independent environment:
- `starvla_rl_games_openvla`
- `starvla_rl_games_pi0`
- `starvla_rl_games_pi05`
- `starvla_rl_games_gr00t`
- `starvla_rl_games_wan_oft`

Torch is detected automatically. Override it with
`--torch-profile cpu|cu126|cu128|cu130`. The use tier contains checkpoint
inference dependencies; dev adds training/data packages and installs flash-attn
for CUDA profiles. `install_stack.sh <model> <env>` remains as a dev-tier
compatibility wrapper for existing training scripts.

Available scripts:
- `examples/rl_games/install/common.sh`
- `examples/rl_games/install/model/{openvla,pi0,pi05,gr00t,wan_oft}.sh`
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

The `memory-rollouts` Flappy config stores one image per row instead of an
explicit context-image column. Convert it directly into the existing WanOFT
LeRobot interface by deriving each row's history inside its episode:

```bash
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_history_to_starvla_lerobot.py \
  --image-sequence-length 5

bash commands/wanoft/train_flappy_wan_oft.sh 3
```

The converter writes the final `flappy_train__bridge` and validation datasets
directly. It uses the Hugging Face cache for source parquet shards but does not
materialize an intermediate dataset with duplicated context images.

Single-latency Deadly Corridor with bridge initialization:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=bridge \
  mode=single
```

Deadly Corridor context-5 training with WanOFT uses the released 7D Bridge
carrier. The raw teacher export stores `deadly_corridor_joint_54` action IDs;
the pipeline decodes those IDs into the equivalent 7D semantic multi-hot
buttons and trains the current action with binary cross-entropy:

```bash
bash scripts/run_deadly_corridor_wan_oft_pipeline.sh --latency 2
```

The standalone training boundary is:

```bash
bash commands/wanoft/train_deadly_corridor_wan_oft.sh 2
```

Single-latency Deadly Corridor with the pi-0.5 VLA backbone initialized from
Bridge/RT-1, but with a fresh native factorized 11D action head:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none
```

Use this mode when you want the trained VLA backbone weights from
`StarVLA/Qwen3VL-PI_v3-Bridge-RT_1`, but do not want to reuse the Bridge action
head or projector. The init config sets:

```text
rl_games.action_carrier=native
rl_games.env_eval.deadly.action_layout=factorized_11
framework.action_model.action_dim=11
framework.action_model.action_env_dim=11
framework.action_model.action_horizon=1
trainer.reload_modules=qwen_vl_interface
```

`trainer.reload_modules=qwen_vl_interface` is the important checkpoint-loading
boundary: setup still resolves the Bridge/RT-1 checkpoint, but training only
loads the `qwen_vl_interface` module from it. `project_layers` and
`action_model` are left as the newly constructed model modules.

The factorized 11D Deadly Corridor layout is:

```text
turn[3] + move[3] + strafe[3] + attack[2]

TURN_NONE, TURN_LEFT, TURN_RIGHT,
MOVE_NONE, MOVE_FORWARD, MOVE_BACKWARD,
STRAFE_NONE, STRAFE_LEFT, STRAFE_RIGHT,
ATTACK_OFF, ATTACK_ON
```

The raw Deadly Corridor source dataset for this mode must provide either:

- `action_tuple`: `[turn, move, strafe, attack]`, where the group sizes are
  `[3, 3, 3, 2]`
- or `action` / `actions`: an already one-hot 11D vector in the order above

If the converted StarVLA/LeRobot dataset already exists under
`paths.dataset_local_dir`, setup reuses it when its manifest matches
`action_layout=factorized_11`. If it does not exist yet, pass the raw source
dataset so setup can verify and convert it:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none \
  dataset.source_hf=<deadly_corridor_factorized11_source_dataset>
```

Before launching a long run, dry-run the composition and setup:

```bash
python examples/rl_games/scripts/launch_train.py \
  --dry-run \
  model=pi05 \
  env=deadly_corridor \
  init=backbone_bridge_factorized11 \
  mode=single \
  run_id=pi05_deadly_corridor_factorized11 \
  checkpoint.load=none
```

Do not use `init=bridge` for this experiment. `init=bridge` intentionally uses
the 7D Bridge carrier and is the wrong action surface for factorized 11D.

Override config values from the command line:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=scratch \
  mode=mixed_latency \
  workspace_dir=WORKSPACE_DIR \
  wandb_entity=WANDB_ENTITY \
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
  datasets.vla_data.per_device_batch_size=1 \
  trainer.distributed_backend=none \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false
```

Cross-task OpenVLA bridge runs use the same launcher with a named setup from
`examples/rl_games/config/cross_task_setup`:

```bash
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=cross_task \
  init=bridge \
  mode=cross_task \
  cross_task_setup=flappy_mixed_demon_zero
```

Canonical wrappers are available at:

```bash
bash examples/rl_games/bash_scripts/openvla/bridge/cross_task/flappy_mixed_demon_zero.sh
bash examples/rl_games/bash_scripts/openvla/bridge/cross_task/demon_mixed_flappy_zero.sh
```

## Configuration

RL-games training uses one Hydra configuration tree under `examples/rl_games/config`.

The primary config groups are:
- `model`
- `env`
- `init`
- `mode`
- `cross_task_setup`

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
