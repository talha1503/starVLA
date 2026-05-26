#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  bash examples/rl_games/scripts/download_convert_rl_games_raw.sh --hfd /path/to/hfd.sh --workspace-dir /path/to/workspace [options]

Options:
  --hfd <path>               hfd.sh path
  --workspace-dir <dir>      Workspace root used for raw and converted dataset roots
  --env <csv|all>            Envs to process (default: flappy,demon_attack,deadly_corridor)
  --frame-stacks <csv>       Frame stacks to process (default: 1,2,4,6)
  --repo-namespace <name>    Hugging Face dataset namespace (default: latency-sensitive-bench)
  --raw-root <dir>           Raw parquet root (default: <workspace-dir>/outputs/rl_games_raw)
  --lerobot-root <dir>       Converted LeRobot root (default: <workspace-dir>/playground/Datasets/rl_games)
  --hfd-extra-args <args>    Extra args passed to hfd.sh, e.g. '--tool aria2c -x 8'
  --dry-run                  Print commands without running them
  -h, --help                 Show this help
USAGE
}

DEFAULT_ENVS="flappy,demon_attack,deadly_corridor"
HFD=""
WORKSPACE_DIR=""
ENVS="$DEFAULT_ENVS"
FRAME_STACKS="1,2,4,6"
REPO_NAMESPACE="latency-sensitive-bench"
RAW_ROOT=""
LEROBOT_ROOT=""
HFD_EXTRA_ARGS_TEXT=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hfd) HFD="$2"; shift 2 ;;
    --workspace-dir) WORKSPACE_DIR="$2"; shift 2 ;;
    --env) ENVS="$2"; shift 2 ;;
    --frame-stacks) FRAME_STACKS="$2"; shift 2 ;;
    --repo-namespace) REPO_NAMESPACE="$2"; shift 2 ;;
    --raw-root) RAW_ROOT="$2"; shift 2 ;;
    --lerobot-root) LEROBOT_ROOT="$2"; shift 2 ;;
    --hfd-extra-args) HFD_EXTRA_ARGS_TEXT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[download_convert] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$HFD" || -z "$WORKSPACE_DIR" ]]; then
  usage
  exit 1
fi

if [[ "$ENVS" == "all" ]]; then
  ENVS="$DEFAULT_ENVS"
fi

if [[ -z "$RAW_ROOT" ]]; then
  RAW_ROOT="${WORKSPACE_DIR%/}/outputs/rl_games_raw"
fi
if [[ -z "$LEROBOT_ROOT" ]]; then
  LEROBOT_ROOT="${WORKSPACE_DIR%/}/playground/Datasets/rl_games"
fi

IFS=',' read -r -a ENV_LIST <<< "$ENVS"
IFS=',' read -r -a FRAME_STACK_LIST <<< "$FRAME_STACKS"

HFD_EXTRA_ARGS=()
if [[ -n "$HFD_EXTRA_ARGS_TEXT" ]]; then
  read -r -a HFD_EXTRA_ARGS <<< "$HFD_EXTRA_ARGS_TEXT"
fi

converter_for_env() {
  local env_name="$1"
  case "$env_name" in
    flappy) echo "examples/rl_games/data_conversion/convert_flappy_to_starvla_lerobot.py" ;;
    demon_attack) echo "examples/rl_games/data_conversion/convert_demon_attack_to_starvla_lerobot.py" ;;
    deadly_corridor) echo "examples/rl_games/data_conversion/convert_deadly_corridor_to_starvla_lerobot.py" ;;
    *) echo "[download_convert] Unknown env: ${env_name}" >&2; exit 1 ;;
  esac
}

print_cmd() {
  local item
  for item in "$@"; do
    printf "%q " "$item"
  done
  printf "\n"
}

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    print_cmd "$@"
  else
    "$@"
  fi
}

if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$RAW_ROOT" "$LEROBOT_ROOT"
fi

for env_name in "${ENV_LIST[@]}"; do
  for frame_stack in "${FRAME_STACK_LIST[@]}"; do
    dataset_name="${env_name}_fixed_l2_fs${frame_stack}"
    repo_id="${REPO_NAMESPACE}/${dataset_name}"
    raw_dir="${RAW_ROOT%/}/${dataset_name}"

    hfd_cmd=("$HFD" "$repo_id" --dataset --local-dir "$raw_dir" "${HFD_EXTRA_ARGS[@]}")

    if [[ "$DRY_RUN" != "true" ]]; then
      echo "[download_convert] Downloading ${repo_id} -> ${raw_dir}"
      mkdir -p "$raw_dir"
    fi
    run_cmd "${hfd_cmd[@]}"
  done
done

for env_name in "${ENV_LIST[@]}"; do
  converter="$(converter_for_env "$env_name")"
  for frame_stack in "${FRAME_STACK_LIST[@]}"; do
    dataset_name="${env_name}_fixed_l2_fs${frame_stack}"
    raw_dir="${RAW_ROOT%/}/${dataset_name}"
    output_dir="${LEROBOT_ROOT%/}/${dataset_name}"

    convert_cmd=(python "$converter" --dataset-name "$raw_dir" --output-dir "$output_dir" --force)

    if [[ "$DRY_RUN" != "true" ]]; then
      echo "[download_convert] Converting ${raw_dir} -> ${output_dir}"
    fi
    run_cmd "${convert_cmd[@]}"
  done
done
