#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_eval.sh [options]

Options:
  --run-dir <dir>          Training run directory (required)
  --gpus <csv>             CUDA visible devices, e.g. 0 or 1 (optional)
  --checkpoint <path>      Explicit checkpoint file (optional)
  --step <int>             Evaluate a specific step checkpoint (optional)
  --stage <name>           Output stage folder name (default: post_train)
  --config <path>          Explicit config path (default: <run-dir>/config.full.yaml)
  --wandb-enabled <true|false>  Enable wandb logging from eval script (default: true)
  --wandb-project <name>   Wandb project override (optional)
  --wandb-entity <name>    Wandb entity override (optional)
  --wandb-run-name <name>  Wandb run name override (optional)
  --help                   Show this help
USAGE
}

RUN_DIR=""
GPUS=""
CHECKPOINT=""
STEP=""
STAGE="post_train"
CONFIG=""
WANDB_ENABLED="true"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --wandb-enabled) WANDB_ENABLED="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required"
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

CMD=(
  python examples/rl_games/scripts/eval_checkpoint.py
  --run-dir "$RUN_DIR"
  --stage "$STAGE"
  --wandb-enabled "$WANDB_ENABLED"
)
if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$STEP" ]]; then
  CMD+=(--step "$STEP")
fi
if [[ -n "$CONFIG" ]]; then
  CMD+=(--config "$CONFIG")
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
  CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
