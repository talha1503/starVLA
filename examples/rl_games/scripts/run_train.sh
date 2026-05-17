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
  --workspace-dir <dir>        Workspace root for relative outputs/assets (default: repo root)
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
  --source-dataset-hf <repo>  Raw/source HF dataset repo to verify and convert at setup time
  --converted-dataset-name <name> StarVLA dataset subdir/data_mix name (default: flappy_train)
  --dataset-cache-dir <dir>   HF datasets cache override for verification/conversion
  --setup-force <true|false>  Force setup-time conversion/model checks (default: false)
  --preprocess-cmd <cmd>       Optional preprocessing command run before training
  --base-model-dir <dir>      Local base model directory
  --base-model-repo-id <repo> HF repo for base model download
  --checkpoint-load <auto|none|local|hf>  Resume policy (default: auto; local first, then HF)
  --checkpoint-hf-repo-id <repo>  HF model repo id for checkpoint-load=hf
  --conda-env <name>            Conda env to activate (default: starvla_rl_games_<model>)
  --no-conda                    Use the current python environment
  --help                       Show this help
USAGE
}

CONFIG_NAME="train"
MODEL="openvla"
ENV_NAME="flappy"
MODE="single"
RUN_ID="starvla_rl_games"
WORKSPACE_DIR="$REPO_ROOT"
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
SOURCE_DATASET_HF=""
CONVERTED_DATASET_NAME="flappy_train"
DATASET_CACHE_DIR=""
SETUP_FORCE="false"
PREPROCESS_CMD=""
BASE_MODEL_DIR="playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action"
BASE_MODEL_REPO_ID="StarVLA/Qwen3-VL-4B-Instruct-Action"
CHECKPOINT_LOAD="auto"
CHECKPOINT_HF_REPO_ID=""
CONDA_ENV_NAME=""
USE_CONDA="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --workspace-dir) WORKSPACE_DIR="$2"; shift 2 ;;
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
    --source-dataset-hf) SOURCE_DATASET_HF="$2"; shift 2 ;;
    --converted-dataset-name) CONVERTED_DATASET_NAME="$2"; shift 2 ;;
    --dataset-cache-dir) DATASET_CACHE_DIR="$2"; shift 2 ;;
    --setup-force) SETUP_FORCE="$2"; shift 2 ;;
    --preprocess-cmd) PREPROCESS_CMD="$2"; shift 2 ;;
    --base-model-dir) BASE_MODEL_DIR="$2"; shift 2 ;;
    --base-model-repo-id) BASE_MODEL_REPO_ID="$2"; shift 2 ;;
    --checkpoint-load) CHECKPOINT_LOAD="$2"; shift 2 ;;
    --checkpoint-hf-repo-id) CHECKPOINT_HF_REPO_ID="$2"; shift 2 ;;
    --conda-env) CONDA_ENV_NAME="$2"; shift 2 ;;
    --no-conda) USE_CONDA="false"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

activate_conda_env() {
  if [[ "$USE_CONDA" != "true" ]]; then
    return
  fi

  if [[ -z "$CONDA_ENV_NAME" ]]; then
    CONDA_ENV_NAME="starvla_rl_games_${MODEL}"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate ${CONDA_ENV_NAME}. Use --no-conda only if the current python env is already correct." >&2
    exit 1
  fi

  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Conda env '${CONDA_ENV_NAME}' does not exist." >&2
    echo "Install it first with: bash examples/rl_games/install/install_stack.sh ${MODEL} ${ENV_NAME}" >&2
    exit 1
  fi

  conda activate "$CONDA_ENV_NAME"
  echo "Using conda env: ${CONDA_ENV_NAME}"
  echo "Python: $(python --version)"
}

activate_conda_env

resolve_workspace_path() {
  local path_value="$1"
  if [[ -z "$path_value" ]]; then
    echo ""
  elif [[ "$path_value" = /* || "$path_value" = "~"* ]]; then
    echo "$path_value"
  else
    echo "${WORKSPACE_DIR%/}/$path_value"
  fi
}

WORKSPACE_DIR="$(cd "$WORKSPACE_DIR" && pwd)"
RUN_ROOT_DIR="$(resolve_workspace_path "$RUN_ROOT_DIR")"
DATASET_LOCAL_DIR="$(resolve_workspace_path "$DATASET_LOCAL_DIR")"
BASE_MODEL_DIR="$(resolve_workspace_path "$BASE_MODEL_DIR")"
if [[ -n "$DATASET_CACHE_DIR" ]]; then
  DATASET_CACHE_DIR="$(resolve_workspace_path "$DATASET_CACHE_DIR")"
fi
if [[ -n "$ACCELERATE_CONFIG" && "$ACCELERATE_CONFIG" != /* && "$ACCELERATE_CONFIG" != "~"* ]]; then
  if [[ -f "$REPO_ROOT/$ACCELERATE_CONFIG" ]]; then
    ACCELERATE_CONFIG="$REPO_ROOT/$ACCELERATE_CONFIG"
  else
    ACCELERATE_CONFIG="$(resolve_workspace_path "$ACCELERATE_CONFIG")"
  fi
fi

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
SETUP_JSON="$(
  python examples/rl_games/scripts/setup_training_assets.py \
    --model "$MODEL" \
    --env "$ENV_NAME" \
    --mode "$MODE" \
    --latency-mode "${LATENCY_MODE_OVERRIDE:-}" \
    --source-dataset-hf "${SOURCE_DATASET_HF:-${DATASET_HF_REPO_ID:-}}" \
    --dataset-local-dir "$DATASET_LOCAL_DIR" \
    --converted-dataset-name "$CONVERTED_DATASET_NAME" \
    --dataset-cache-dir "${DATASET_CACHE_DIR:-}" \
    --dataset-force-download "${DATASET_FORCE_DOWNLOAD}" \
    --setup-force "${SETUP_FORCE}" \
    --base-model-dir "$BASE_MODEL_DIR" \
    --base-model-repo-id "${BASE_MODEL_REPO_ID:-}" \
    --checkpoint-local-dir "$CHECKPOINT_LOCAL_DIR" \
    --checkpoint-load "$CHECKPOINT_LOAD" \
    --checkpoint-hf-repo-id "${CHECKPOINT_HF_REPO_ID:-}" \
    --hf-repo-id "${HF_REPO_ID:-}"
)"
echo "Setup summary: $SETUP_JSON"

RESUME_FOUND="$(
  python -c 'import json,sys; print("true" if json.loads(sys.argv[1]).get("resume_found") else "false")' "$SETUP_JSON"
)"
if [[ "$RESUME_FOUND" == "true" ]]; then
  CMD+=("trainer.is_resume=true")
else
  CMD+=("trainer.is_resume=false")
fi

RESOLVED_DATA_ROOT="$(
  python -c 'import json,sys; print(json.loads(sys.argv[1]).get("dataset_local_dir") or "")' "$SETUP_JSON"
)"
RESOLVED_DATA_MIX="$(
  python -c 'import json,sys; print(json.loads(sys.argv[1]).get("data_mix") or "")' "$SETUP_JSON"
)"
RESOLVED_BASE_MODEL="$(
  python -c 'import json,sys; print(json.loads(sys.argv[1]).get("base_model_dir") or "")' "$SETUP_JSON"
)"
RESOLVED_LATENCY_PROMPT_MAP="$(
  python -c 'import json,sys; print(json.loads(sys.argv[1]).get("latency_prompt_map_path") or "")' "$SETUP_JSON"
)"

if [[ -n "$RESOLVED_DATA_ROOT" ]]; then
  CMD+=("datasets.vla_data.data_root_dir=$RESOLVED_DATA_ROOT")
else
  CMD+=("datasets.vla_data.data_root_dir=$DATASET_LOCAL_DIR")
fi
if [[ -n "$RESOLVED_DATA_MIX" ]]; then
  CMD+=("datasets.vla_data.data_mix=$RESOLVED_DATA_MIX")
fi
if [[ -n "$RESOLVED_BASE_MODEL" ]]; then
  CMD+=("framework.qwenvl.base_vlm=$RESOLVED_BASE_MODEL")
fi
if [[ -z "$LATENCY_PROMPT_MAP_PATH" && -n "$RESOLVED_LATENCY_PROMPT_MAP" ]]; then
  CMD+=("rl_games.env_eval.latency.prompt_map_path=$RESOLVED_LATENCY_PROMPT_MAP")
fi

if [[ -n "$PREPROCESS_CMD" ]]; then
  echo "Running preprocess command:"
  echo "  $PREPROCESS_CMD"
  eval "$PREPROCESS_CMD"
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
