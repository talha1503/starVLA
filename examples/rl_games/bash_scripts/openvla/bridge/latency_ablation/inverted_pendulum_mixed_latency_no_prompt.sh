#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=inverted_pendulum_common.sh
source "${SCRIPT_DIR}/inverted_pendulum_common.sh"

RUN_ID="${RUN_ID:-openvla_bridge_inverted_pendulum_mixed_latency_024_no_latency_prompt_exp1}"
ip_run_openvla_training "${RUN_ID}" latency_neutral
