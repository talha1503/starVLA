#!/usr/bin/env bash

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${INSTALL_DIR}/../../.." && pwd)"
LATENCY_BENCH_ROOT="${LATENCY_BENCH_ROOT:-$(git -C "${STARVLA_ROOT}" rev-parse --show-superproject-working-tree)}"

export STARVLA_ROOT LATENCY_BENCH_ROOT
