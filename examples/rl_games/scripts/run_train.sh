#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  run_train.sh [options]

Options:
  --config-name <name>         Hydra top-level config name (default: train)
  --model <openvla|pi0|gr00t>  Model config group (default: openvla)
  --env <flappy|demon_attack|deadly_corridor>  Env config group (default: flappy)
  --mode <single|mixed_latency|cross_task>      Mode config group (default: single)
  --run-id <id>                Run id override (default: starvla_rl_games)
  --run-root-dir <dir>         Output root (default: results/Checkpoints)
  --gpus <csv>                 CUDA visible devices, e.g. 0 or 0,1 (optional)
  --num-processes <int>        Processes for accelerate launch (default: 1)
  --use-accelerate <true|false> Use accelerate launch wrapper (default: true)
  --accelerate-config <path>   Accelerate config path (default: starVLA/config/deepseeds/deepspeed_zero2.yaml)
  --seed <int>                 Seed override (default: 42)
  --wandb-entity <name>        Wandb entity (default: your_wandb_entity)
  --wandb-project <name>       Wandb project (default: starVLA_rl_games)
  --max-train-steps <int>      Training steps override (default: 2000)
  --save-interval <int>        Checkpoint save interval (default: 100)
  --eval-interval <int>        In-trainer eval interval (default: 100)
  --gradient-accumulation-steps <int>  Trainer grad accumulation (default: 1)
  --batch-size <int>           Per-device batch size (default: 4)
  --micro-batch-size <int>     Alias of per-device batch size (overrides --batch-size)
  --task <name>                rl_games.task override
  --model-alias <name>         rl_games.model_alias override
  --latency-mode <single|mixed> rl_games.env_eval.latency.mode override
  --latencies <csv>            Comma-separated latency list (e.g. 0,1,2,3)
  --latency-prompt-map-path <path>  Prompt map JSON path override
  --env-eval-enabled <true|false>   Enable/disable rl_games env eval
  --num-episodes <int>         Eval episodes per latency (default: 5)
  --max-episode-steps <int>    Max steps per episode (default: 2000)
  --frameskip <int>            Frameskip override
  --image-size <int>           Eval image size override
  --deadly-action-layout <multibinary_7|factorized_11>  Deadly layout override
  --hf-sync-enabled <true|false>      checkpoint.sync.enabled
  --hf-repo-id <repo>          checkpoint.sync.repo_id
  --hf-keep-last-n <int>       checkpoint.sync.keep_last_n
  --local-keep-last-n <int>    checkpoint.local.keep_last_n
  --dataset-mode <none|local|hf>  Dataset bootstrap mode (default: none)
  --dataset-local-dir <dir>    Dataset local directory (default: playground/Datasets/rl_games)
  --dataset-hf-repo-id <repo>  HF dataset repo id for dataset-mode=hf
  --dataset-allow-patterns <csv>  HF dataset allow patterns (comma-separated)
  --dataset-required-subdirs <csv> Required subdirs to consider dataset ready (default: train)
  --dataset-force-download <true|false> Force re-download for dataset-mode=hf
  --preprocess-cmd <cmd>       Optional preprocessing command run before training
  --checkpoint-load <none|local|hf>  Resume source policy (default: none)
  --checkpoint-hf-repo-id <repo>  HF model repo id for checkpoint-load=hf
  --help                       Show this help
USAGE
}

CONFIG_NAME="train"
MODEL="openvla"
ENV_NAME="flappy"
MODE="single"
RUN_ID="starvla_rl_games"
RUN_ROOT_DIR="results/Checkpoints"
GPUS=""
NUM_PROCESSES="1"
USE_ACCELERATE="true"
ACCELERATE_CONFIG="starVLA/config/deepseeds/deepspeed_zero2.yaml"
SEED="42"
WANDB_ENTITY="your_wandb_entity"
WANDB_PROJECT="starVLA_rl_games"
MAX_TRAIN_STEPS="2000"
SAVE_INTERVAL="100"
EVAL_INTERVAL="100"
GRADIENT_ACCUMULATION_STEPS="1"
BATCH_SIZE="4"
MICRO_BATCH_SIZE=""
TASK_OVERRIDE=""
MODEL_ALIAS_OVERRIDE=""
LATENCY_MODE_OVERRIDE=""
LATENCIES_CSV=""
LATENCY_PROMPT_MAP_PATH=""
ENV_EVAL_ENABLED="true"
NUM_EPISODES="5"
MAX_EPISODE_STEPS="2000"
FRAMESKIP=""
IMAGE_SIZE=""
DEADLY_ACTION_LAYOUT="multibinary_7"
HF_SYNC_ENABLED="false"
HF_REPO_ID=""
HF_KEEP_LAST_N="0"
LOCAL_KEEP_LAST_N="3"
DATASET_MODE="none"
DATASET_LOCAL_DIR="playground/Datasets/rl_games"
DATASET_HF_REPO_ID=""
DATASET_ALLOW_PATTERNS=""
DATASET_REQUIRED_SUBDIRS="train"
DATASET_FORCE_DOWNLOAD="false"
PREPROCESS_CMD=""
CHECKPOINT_LOAD="none"
CHECKPOINT_HF_REPO_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root-dir) RUN_ROOT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --num-processes) NUM_PROCESSES="$2"; shift 2 ;;
    --use-accelerate) USE_ACCELERATE="$2"; shift 2 ;;
    --accelerate-config) ACCELERATE_CONFIG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --gradient-accumulation-steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --task) TASK_OVERRIDE="$2"; shift 2 ;;
    --model-alias) MODEL_ALIAS_OVERRIDE="$2"; shift 2 ;;
    --latency-mode) LATENCY_MODE_OVERRIDE="$2"; shift 2 ;;
    --latencies) LATENCIES_CSV="$2"; shift 2 ;;
    --latency-prompt-map-path) LATENCY_PROMPT_MAP_PATH="$2"; shift 2 ;;
    --env-eval-enabled) ENV_EVAL_ENABLED="$2"; shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --max-episode-steps) MAX_EPISODE_STEPS="$2"; shift 2 ;;
    --frameskip) FRAMESKIP="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --deadly-action-layout) DEADLY_ACTION_LAYOUT="$2"; shift 2 ;;
    --hf-sync-enabled) HF_SYNC_ENABLED="$2"; shift 2 ;;
    --hf-repo-id) HF_REPO_ID="$2"; shift 2 ;;
    --hf-keep-last-n) HF_KEEP_LAST_N="$2"; shift 2 ;;
    --local-keep-last-n) LOCAL_KEEP_LAST_N="$2"; shift 2 ;;
    --dataset-mode) DATASET_MODE="$2"; shift 2 ;;
    --dataset-local-dir) DATASET_LOCAL_DIR="$2"; shift 2 ;;
    --dataset-hf-repo-id) DATASET_HF_REPO_ID="$2"; shift 2 ;;
    --dataset-allow-patterns) DATASET_ALLOW_PATTERNS="$2"; shift 2 ;;
    --dataset-required-subdirs) DATASET_REQUIRED_SUBDIRS="$2"; shift 2 ;;
    --dataset-force-download) DATASET_FORCE_DOWNLOAD="$2"; shift 2 ;;
    --preprocess-cmd) PREPROCESS_CMD="$2"; shift 2 ;;
    --checkpoint-load) CHECKPOINT_LOAD="$2"; shift 2 ;;
    --checkpoint-hf-repo-id) CHECKPOINT_HF_REPO_ID="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$MICRO_BATCH_SIZE" ]]; then
  BATCH_SIZE="$MICRO_BATCH_SIZE"
fi

LATENCIES_EXPR=""
if [[ -n "$LATENCIES_CSV" ]]; then
  IFS=',' read -r -a LAT_ARR <<< "$LATENCIES_CSV"
  LATENCIES_EXPR="["
  for i in "${!LAT_ARR[@]}"; do
    if [[ "$i" -gt 0 ]]; then LATENCIES_EXPR+=", "; fi
    LATENCIES_EXPR+="${LAT_ARR[$i]}"
  done
  LATENCIES_EXPR+="]"
fi

CMD=(
  starVLA/training/train_starvla_hydra.py
  --config-name "$CONFIG_NAME"
  "model=$MODEL"
  "env=$ENV_NAME"
  "mode=$MODE"
  "run_id=$RUN_ID"
  "run_root_dir=$RUN_ROOT_DIR"
  "seed=$SEED"
  "wandb_entity=$WANDB_ENTITY"
  "wandb_project=$WANDB_PROJECT"
  "trainer.max_train_steps=$MAX_TRAIN_STEPS"
  "trainer.save_interval=$SAVE_INTERVAL"
  "trainer.eval_interval=$EVAL_INTERVAL"
  "trainer.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  "datasets.vla_data.per_device_batch_size=$BATCH_SIZE"
  "rl_games.env_eval.enabled=$ENV_EVAL_ENABLED"
  "rl_games.env_eval.num_episodes=$NUM_EPISODES"
  "rl_games.env_eval.max_episode_steps=$MAX_EPISODE_STEPS"
  "checkpoint.sync.enabled=$HF_SYNC_ENABLED"
  "checkpoint.sync.keep_last_n=$HF_KEEP_LAST_N"
  "checkpoint.local.keep_last_n=$LOCAL_KEEP_LAST_N"
)

if [[ -n "$TASK_OVERRIDE" ]]; then
  CMD+=("rl_games.task=$TASK_OVERRIDE")
fi
if [[ -n "$MODEL_ALIAS_OVERRIDE" ]]; then
  CMD+=("rl_games.model_alias=$MODEL_ALIAS_OVERRIDE")
fi
if [[ -n "$LATENCY_MODE_OVERRIDE" ]]; then
  CMD+=("rl_games.env_eval.latency.mode=$LATENCY_MODE_OVERRIDE")
fi
if [[ -n "$LATENCIES_EXPR" ]]; then
  CMD+=("rl_games.env_eval.latency.values=$LATENCIES_EXPR")
fi
if [[ -n "$LATENCY_PROMPT_MAP_PATH" ]]; then
  CMD+=("rl_games.env_eval.latency.prompt_map_path=$LATENCY_PROMPT_MAP_PATH")
fi
if [[ -n "$FRAMESKIP" ]]; then
  CMD+=("rl_games.env_eval.frameskip=$FRAMESKIP")
fi
if [[ -n "$IMAGE_SIZE" ]]; then
  CMD+=("rl_games.env_eval.image_size=$IMAGE_SIZE")
fi
if [[ "$ENV_NAME" == "deadly_corridor" || "${TASK_OVERRIDE:-}" == "deadly_corridor" ]]; then
  CMD+=("rl_games.env_eval.deadly.action_layout=$DEADLY_ACTION_LAYOUT")
fi
if [[ -n "$HF_REPO_ID" ]]; then
  CMD+=("checkpoint.sync.repo_id=$HF_REPO_ID")
fi

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

RUN_OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
CHECKPOINT_LOCAL_DIR="${RUN_OUTPUT_DIR}/checkpoints"
BOOTSTRAP_JSON="$(
  python examples/rl_games/scripts/bootstrap_data_and_checkpoint.py \
    --dataset-mode "$DATASET_MODE" \
    --dataset-local-dir "$DATASET_LOCAL_DIR" \
    --dataset-hf-repo-id "${DATASET_HF_REPO_ID:-}" \
    --dataset-allow-patterns "${DATASET_ALLOW_PATTERNS:-}" \
    --dataset-required-subdirs "${DATASET_REQUIRED_SUBDIRS:-train}" \
    --dataset-force-download "${DATASET_FORCE_DOWNLOAD}" \
    --checkpoint-mode "$CHECKPOINT_LOAD" \
    --checkpoint-local-dir "$CHECKPOINT_LOCAL_DIR" \
    --checkpoint-hf-repo-id "${CHECKPOINT_HF_REPO_ID:-}"
)"
echo "Bootstrap summary: $BOOTSTRAP_JSON"

RESUME_FOUND="$(
  python -c 'import json,sys; print("true" if json.loads(sys.argv[1]).get("resume_found") else "false")' "$BOOTSTRAP_JSON"
)"
if [[ "$RESUME_FOUND" == "true" ]]; then
  CMD+=("trainer.is_resume=true")
else
  CMD+=("trainer.is_resume=false")
fi

CMD+=("datasets.vla_data.data_root_dir=$DATASET_LOCAL_DIR")

if [[ -n "$PREPROCESS_CMD" ]]; then
  echo "Running preprocess command:"
  echo "  $PREPROCESS_CMD"
  eval "$PREPROCESS_CMD"
fi

if [[ "$DATASET_MODE" != "none" ]]; then
  DATASET_READY="$(
    python -c 'import json,sys; print("true" if json.loads(sys.argv[1]).get("dataset_ready") else "false")' "$BOOTSTRAP_JSON"
  )"
  if [[ "$DATASET_READY" != "true" ]]; then
    POST_PREPROCESS_JSON="$(
      python examples/rl_games/scripts/bootstrap_data_and_checkpoint.py \
        --dataset-mode local \
        --dataset-local-dir "$DATASET_LOCAL_DIR" \
        --dataset-required-subdirs "${DATASET_REQUIRED_SUBDIRS:-train}" \
        --checkpoint-mode none \
        --checkpoint-local-dir "$CHECKPOINT_LOCAL_DIR"
    )"
    DATASET_READY="$(
      python -c 'import json,sys; print("true" if json.loads(sys.argv[1]).get("dataset_ready") else "false")' "$POST_PREPROCESS_JSON"
    )"
    if [[ "$DATASET_READY" != "true" ]]; then
      echo "Dataset is not ready at ${DATASET_LOCAL_DIR} (required subdirs: ${DATASET_REQUIRED_SUBDIRS})."
      exit 1
    fi
  fi
fi

USE_ACCELERATE_LOWER="$(echo "$USE_ACCELERATE" | tr '[:upper:]' '[:lower:]')"
if [[ "$USE_ACCELERATE_LOWER" == "true" || "$USE_ACCELERATE_LOWER" == "1" || "$USE_ACCELERATE_LOWER" == "yes" ]]; then
  LAUNCH_CMD=(
    accelerate launch
    --config_file "$ACCELERATE_CONFIG"
    --num_processes "$NUM_PROCESSES"
    "${CMD[@]}"
  )
else
  LAUNCH_CMD=(python "${CMD[@]}")
fi

echo "Running command:"
printf '  %q' "${LAUNCH_CMD[@]}"
echo
"${LAUNCH_CMD[@]}"
